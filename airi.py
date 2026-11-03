"""Optional conversation CLI around the unchanged VINFREY agent workflow."""

import contextlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from api import MAX_TOOL_ROUNDS, AgentError, generate
from approval import ActionFailed, ActionRejected, ApprovalError, ApprovalGate
from main import MAX_RETRIES, run_workflow
from memory import MemoryStore, content_hash, normalize_memory
from persona import PERSONA, build_persona, load_profile
from suggestions import looks_like_secret, suggest_memories, valid_suggestion
from tools import default_tools
from usage import DEMO_REQUEST_LIMIT, NORMAL_REQUEST_LIMIT, UsageGuard, describe_usage

MAX_HISTORY_TURNS = 6
SQLITE_MAX_INTEGER = 2**63 - 1
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MAX_WORK_TASK_CHARS = 300
MAX_WORK_RESULT_CHARS = 1500
# Worst-case /work requests: Planner, Researcher, then Builder and Reviewer per draft.
WORK_MAX_REQUESTS = 2 + 2 * (MAX_RETRIES + 1)
DEMO_ENV = "VINFREY_DEMO"
DEMO_DIR_PREFIX = "vinfrey-demo-"
DEMO_DB_NAME = "demo.sqlite3"
DEMO_BANNER = (
    "LIVE API DEMO: this session sends real requests to the OpenAI API, which may cost money.\n"
    "Demo memory is temporary and deleted when the session ends; no personal profile is used.\n"
    f"Limit: {DEMO_REQUEST_LIMIT} API requests per session. "
    "This is a safety limit, not a spending limit.")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Home-directory paths name the local account; other paths are kept.
_PRIVATE_PATH = re.compile(r"(?i)\b[a-z]:[\\/]users[\\/][^\s\"'<>|]*"
                           r"|(?<![\w.])/(?:home|users)/[^\s\"'<>|]*")
HELP = """Chat normally, or use:
  /remember <note>  Save an intentional local memory (max 2000 characters)
  /memories        Show saved memories
  /forget <id>     Ask to delete one memory permanently (needs /approve)
  /suggest         Ask AiRi to suggest memories from your recent messages (needs /approve)
  /pending         Show actions waiting for your approval
  /approve <id>    Approve a pending action, for example /approve A1
  /deny <id>       Deny a pending action
  /work <task>     Run Planner -> Researcher -> Builder -> Reviewer
  /usage           Show API requests and tokens used in this session
  /help            Show commands
  /quit            Exit
Chat sends your message, recent chat history, relevant saved notes, the current date
and time, and a short filtered summary of the last /work result to the API.
Chat history stays in RAM.
A successful /forget also clears recent chat context (including that /work summary),
so the note is not sent again; the visible transcript stays, and requests already
sent cannot be recalled.
"""


class AiRi:
    def __init__(self, memory, *, client=None, model=None, persona=PERSONA, tools=None, clock=None,
                 usage=None):
        if client is not None and usage is not None:
            raise ValueError("Pass the client to UsageGuard, not to AiRi.")
        self.memory = memory
        # A UsageGuard stands in for the client, so every API request in this session is counted.
        self.usage = usage
        self.client = usage if usage is not None else client
        self.model = model
        self.persona = persona
        # Pure computation tools for chat only; None (the default) keeps chat tool-free.
        self.tools = tools
        # Read on every chat request, never cached; tests inject a fixed clock.
        self.clock = clock or _local_now
        # Sanitized summary of the last finished /work; chat context only, never history.
        self.last_work = None
        self.history = []
        # One gate per AiRi instance, so each CLI run or browser session has its own queue.
        self.gate = ApprovalGate()
        self.gate.register("memory.delete", validate=self._validate_forget,
                           describe=_describe_forget, execute=self._execute_forget)
        self.gate.register("memory.save", validate=self._validate_save,
                           describe=_describe_save, execute=self._execute_save)

    def _validate_forget(self, params):
        note_id = params.get("note_id")
        # SQLite integers are 64-bit; larger values would raise OverflowError on lookup.
        if type(note_id) is not int or not 0 < note_id <= SQLITE_MAX_INTEGER:
            raise ApprovalError("Use /forget <memory id>, for example /forget 3.")
        if self._pending_deletes(note_id):
            raise ApprovalError(
                f"A deletion of memory #{note_id} is already waiting for approval. Use /pending.")
        note = self.memory.get(note_id)
        if note is None:
            raise ApprovalError(f"Memory #{note_id} does not exist. Use /memories to check.")
        # Keep only identity and safe metadata; the note text never enters the action.
        return {"note_id": note_id, "content_sha256": content_hash(note["content"]),
                "created_at": note["created_at"], "length": len(note["content"])}

    def _execute_forget(self, params):
        if not self.memory.delete_if_unchanged(
                params["note_id"], params["content_sha256"], params["created_at"]):
            raise ActionRejected("the memory changed or no longer exists. Nothing was deleted.")
        self._invalidate_deletes(params["note_id"])
        # The deleted text may still sit in recent turns; stop sending it with later requests.
        self.history = []
        self.last_work = None
        return f"memory #{params['note_id']} deleted."

    def _pending_deletes(self, note_id):
        return [a.id for a in self.gate.pending()
                if a.kind == "memory.delete" and a.params["note_id"] == note_id]

    def _invalidate_deletes(self, note_id):
        """Cancel pending deletions of an id that now names a different (or no) note.

        Partial protection only: another process can still reuse the id unseen.
        """
        return self.gate.invalidate(
            lambda a: a.kind == "memory.delete" and a.params["note_id"] == note_id)

    def _saved(self, note_id):
        cancelled = self._invalidate_deletes(note_id)
        if not cancelled:
            return ""
        return (f"\nCancelled pending deletion {', '.join(cancelled)} because memory "
                f"#{note_id} now refers to a new note.")

    def _validate_save(self, params):
        content, reason = params.get("content"), params.get("reason")
        if isinstance(content, str) and isinstance(reason, str):
            content, reason = content.strip(), reason.strip()
        if not valid_suggestion(content, reason):
            raise ApprovalError("This memory suggestion is not allowed.")
        key = normalize_memory(content)
        pending = (a.params["content"] for a in self.gate.pending() if a.kind == "memory.save")
        if (self.memory.find_duplicate(content) is not None
                or any(normalize_memory(text) == key for text in pending)):
            raise ApprovalError("This memory is already saved or waiting for approval.")
        return {"content": content, "reason": reason}

    def _execute_save(self, params):
        # Checked again under the database write lock: another session may have saved it.
        note_id = self.memory.save_if_absent(params["content"])
        if note_id is None:
            raise ActionRejected("the same memory already exists. Nothing was saved.")
        return f"saved memory #{note_id}.{self._saved(note_id)}"

    def new_conversation(self):
        """Forget chat context and undecided approvals; saved notes stay."""
        self.history = []
        self.last_work = None
        self.gate.clear()

    def chat(self, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please enter a nonempty message.")
        message = message.strip()
        # Worst case: one request, or a full tool loop when tools are on.
        self._preflight(1 if self.tools is None else 1 + MAX_TOOL_ROUNDS, "chat")
        notes = self.memory.relevant(message)
        context = {
            "message": message,
            "memories": notes,
            "conversation": self.history,
            "now": _now_context(self.clock()),
        }
        if self.last_work is not None:
            # Untrusted background data: a bounded, filtered summary, never instructions.
            context["recent_work"] = self.last_work
        prompt = json.dumps(context, ensure_ascii=False)
        # /work and /suggest build their own requests and never receive these tools.
        options = {} if self.tools is None else {"tools": self.tools}
        reply = generate("AiRi", self.persona, prompt, client=self.client, model=self.model,
                         **options)
        # Keep only successful turns in RAM. Never write chat messages to SQLite.
        self.history.append({"user": message, "assistant": reply})
        self.history = self.history[-MAX_HISTORY_TURNS:]
        return reply

    def handle(self, message):
        """Route explicit commands locally; ordinary text makes one chat call."""
        message = message.strip()
        command, _, argument = message.partition(" ")
        argument = argument.strip()
        if command == "/remember":
            note_id = self.memory.save(argument)
            return f"Saved memory #{note_id}.{self._saved(note_id)}"
        if command == "/memories" and not argument:
            notes = self.memory.list_memories()
            return "\n".join(f"#{n['id']}: {n['content']}" for n in notes) or "No memories saved."
        if command == "/forget":
            if not (argument.isascii() and argument.isdigit()):
                raise ValueError("Use /forget <memory id>, for example /forget 3.")
            action = self.gate.request("memory.delete", {"note_id": int(argument)})
            return (f"Approval needed {action.id}: {action.summary}\n"
                    f"Use /approve {action.id} or /deny {action.id}.")
        if command == "/pending" and not argument:
            actions = self.gate.pending()
            return "\n".join(f"{a.id}: {a.summary}" for a in actions) or "No pending actions."
        if command == "/suggest" and not argument:
            return self._suggest()
        if command == "/approve" and argument:
            return self._approve(argument)
        if command == "/deny" and argument:
            action = self.gate.deny(argument)
            return f"Denied {action.id}. Nothing was changed."
        if command == "/work":
            outcome = self.work(argument)
            return f"Workflow: {outcome['review']}\n{outcome['result']}\n{outcome['feedback']}"
        if command == "/usage" and not argument:
            if self.usage is None:
                return "Usage tracking is off for this session."
            return describe_usage(self.usage.snapshot())
        if command == "/help" and not argument:
            return HELP
        if message.startswith("/"):
            raise ValueError("Unknown command or invalid arguments. Use /help.")
        return self.chat(message)

    def _suggest(self):
        """Turn model suggestions into pending actions. Nothing is saved here."""
        self._preflight(1, "/suggest")
        accepted, skipped = suggest_memories(self.history, client=self.client, model=self.model)
        ids = []
        for item in accepted:
            try:
                ids.append(self.gate.request("memory.save", item, requested_by="assistant").id)
            except ApprovalError:
                skipped += 1  # already saved, already pending, or not allowed
        if ids:
            noun = "memory" if len(ids) == 1 else "memories"
            lines = [f"Suggested {len(ids)} {noun} for approval: {', '.join(ids)}. "
                     "Nothing is saved until you approve. Use /pending to review."]
        else:
            lines = ["No suitable memories to suggest."]
        if skipped:
            lines.append(f"Skipped {skipped} suggestion(s) that were unsafe, invalid, or duplicates.")
        return "\n".join(lines)

    def _approve(self, action_id):
        """Only explicit user input reaches here; model replies are never routed to handle()."""
        action_id = action_id.strip().upper()
        try:
            result = self.gate.approve(action_id)
        except ActionRejected as rejected:
            # Executors write these messages themselves; they never contain note text.
            return f"Not executed {action_id}: {rejected}"
        except ActionFailed:
            raise ValueError(f"Failed {action_id}: the action may have partly run. "
                             "Check /memories before retrying.") from None
        return f"Approved {action_id}: {result}"

    def work(self, task, on_event=None):
        """Expose the existing workflow's structured result and progress events."""
        if not isinstance(task, str) or not task.strip():
            # Same message as run_workflow; checked here so it comes before the quota check.
            raise ValueError("Please enter a nonempty task.")
        self._preflight(WORK_MAX_REQUESTS, "/work")
        outcome = run_workflow(task, client=self.client, model=self.model, on_event=on_event)
        # Only a finished run replaces the summary; a failed run raises and keeps the old one.
        self.last_work = _summarize_work(task, outcome)
        return outcome

    def _preflight(self, needed, action):
        """Refuse before the first request if the session could run out midway.

        Nothing is reserved: concurrent requests in the same session can still use the
        quota first, and the guard then stops the next request without exceeding the cap.
        """
        if self.usage is not None:
            self.usage.ensure_available(needed, action)


def _local_now():
    return datetime.now().astimezone()


def _now_context(moment):
    """Date, weekday, time, and UTC offset for chat; the weekday name never depends on locale."""
    if moment.utcoffset() is None:
        moment = moment.astimezone()
    offset = moment.strftime("%z")
    return {"date": moment.date().isoformat(), "weekday": _WEEKDAYS[moment.weekday()],
            "time": moment.strftime("%H:%M"), "utc_offset": f"{offset[:3]}:{offset[3:]}"}


def _omit_code_blocks(text):
    """Replace fenced code with a line count; an unclosed fence runs to the end."""
    lines, inside = [], None
    for line in text.splitlines():
        if _FENCE.match(line):
            if inside is None:
                inside = 0
            else:
                lines.append(_code_placeholder(inside))
                inside = None
        elif inside is None:
            lines.append(line)
        else:
            inside += 1
    if inside is not None:
        lines.append(_code_placeholder(inside))
    return lines


def _code_placeholder(count):
    return f"[code block omitted: {count} line{'' if count == 1 else 's'}]"


def _sanitize(text, limit):
    """Drop code, credential-looking lines, and home-directory paths, then bound the length."""
    lines = ["[line omitted: possible credential]" if looks_like_secret(line)
             else _PRIVATE_PATH.sub("[path]", line) for line in _omit_code_blocks(text)]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) <= limit:
        return cleaned, False
    return cleaned[:limit] + "…", True


def _summarize_work(task, outcome):
    """The only /work data chat may see: task, PASS/FAIL, and a filtered result excerpt.

    Filtering is heuristic and not comprehensive. Feedback and agent traces are never kept.
    """
    if looks_like_secret(task):
        task_text, task_cut = "[task omitted: possible credential]", False
    else:
        task_text, task_cut = _sanitize(task, MAX_WORK_TASK_CHARS)
    result, result_cut = _sanitize(outcome["result"], MAX_WORK_RESULT_CHARS)
    # Fail closed if anything recognizable survived filtering and truncation.
    if looks_like_secret(task_text):
        task_text = "[task omitted: possible credential]"
    if looks_like_secret(result):
        result = "[result omitted: possible credential]"
    return {"task": task_text, "review": outcome["review"], "result": result,
            "truncated": task_cut or result_cut}


def _describe_forget(params):
    return (f"Delete memory #{params['note_id']} permanently (saved {params['created_at']}, "
            f"{params['length']} characters). This cannot be undone. "
            "Check /memories before approving.")


def _describe_save(params):
    # Model-written text: the Web UI must show this as plain text, never Markdown.
    return f"Save AI-suggested memory: {params['content']} | Reason: {params['reason']}"


def demo_mode():
    """Only the exact value "1" turns on the live API demo."""
    return os.environ.get(DEMO_ENV) == "1"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        with contextlib.ExitStack() as cleanup:
            if demo_mode():
                # Temporary memory and no profile: the real DB and personal name are never read.
                directory = tempfile.mkdtemp(prefix=DEMO_DIR_PREFIX)
                cleanup.callback(shutil.rmtree, directory, True)
                print(DEMO_BANNER)
                airi = AiRi(MemoryStore(Path(directory) / DEMO_DB_NAME), persona=build_persona(),
                            tools=default_tools(), usage=UsageGuard(DEMO_REQUEST_LIMIT))
            else:
                profile, warning = load_profile()
                if warning:
                    print(warning)
                airi = AiRi(MemoryStore(), persona=build_persona(profile.get("user_name")),
                            tools=default_tools(), usage=UsageGuard(NORMAL_REQUEST_LIMIT))
            print(f"VINFREY | AiRi\n{HELP}")
            while True:
                message = input("You > ").strip()
                if message == "/quit":
                    return 0
                if not message:
                    continue
                try:
                    print(f"AiRi > {airi.handle(message)}", flush=True)
                except (AgentError, ValueError) as error:
                    print(f"Stopped: {error}", flush=True)
    except (sqlite3.Error, OSError):
        print("Stopped: cannot access the local memory database. Check local_data permissions.")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nStopped: cancelled by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
