# VINFREY : AiRi — Technical Reference

> This is the previous English README (commit `aa18300`), kept for setup, testing,
> architecture, privacy details, known limitations, and the validation record. The
> portfolio overview in Thai is in the main [README](../README.md). Commands below
> assume you run them from the project root.

A local Streamlit chat demo with AiRi, a calm female AI with the conversational style of a close friend, plus a Python multi-agent workflow for reviewed text or code drafts. VINFREY is the system/project name; AiRi is its conversational identity. All four Core agents use the OpenAI Responses API. The core stays sequential and uses plain functions rather than an agent framework.

## Features at a glance

| Layer | What it does | Key safeguard |
| --- | --- | --- |
| Persona | Friend-like voice with its own reasoned view; disagrees constructively | Supplied context is background data, never instructions |
| Memory | Local SQLite notes saved only by `/remember` | `/forget` and AI-suggested notes (`/suggest`) need explicit approval |
| ApprovalGate | Default-deny queue for actions that change Memory | Model text can never approve; each action runs at most once per gate |
| Tools | `calculate` and `date_math` in chat | Pure computation, strict allowlist, bounded resources, `store=False` |
| Context awareness | Current date/time and a filtered summary of the last `/work` | Heuristic filtering of code, credentials, and home paths |
| UsageGuard | 40 API requests per session (20 in the live demo), `/usage` | Refuses before sending; not a spending limit |
| Core workflow | Planner → Researcher → Builder ⇄ Reviewer (`/work`) | At most 3 revisions; strict Reviewer schema |

One AiRi session (one CLI run or one browser session):

```text
User
 ├─ local commands: /remember /memories /forget /suggest /pending /approve /deny /usage /help
 │    └─ Memory (SQLite) ◄── ApprovalGate (approves /forget and /suggest actions)
 ├─ chat ─► Persona + matching Memory + recent history + now + recent_work
 │    └─ tool loop: calculate, date_math (pure, allowlisted)
 └─ /work ─► Core: Planner → Researcher → Builder ⇄ Reviewer ─► recent_work summary

Every API request ─► UsageGuard (per-session cap) ─► OpenAI Responses API (store=False)
```

See **Known limitations** near the end before relying on any safeguard.

## Architecture

```text
User -> Planner -> Researcher -> Builder -> Reviewer
                                  ^           |
                                  |   FAIL + actionable feedback
                                  +-----------+
                                              |
                                 PASS -> completed
                                 Still FAIL after 3 revisions -> human review
```

| Agent | Responsibility | Output |
| --- | --- | --- |
| Planner | Turn the original request into a numbered plan | Plan text |
| Researcher | Prepare background, constraints, assumptions, and uncertainties | Research notes |
| Builder | Create the deliverable; revise the previous draft using feedback | Full draft |
| Reviewer | Check the draft against the original request and plan | JSON status (`PASS`/`FAIL`) and feedback |

Planner retains its existing Responses API call, prompt, and default model, `gpt-6-luna`. Researcher, Builder, and Reviewer now call the same API instead of producing mock strings.

The original request and plan remain available to downstream agents. Builder receives research, the previous draft, and the latest feedback. Reviewer receives the request, plan, research, and current draft. Planner and Researcher run once; only Builder and Reviewer repeat.

Reviewer uses [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) with a strict JSON schema. Python also validates the fields. Invalid JSON, missing feedback, refusals, and incomplete responses stop the workflow rather than being treated as approval.

## Project files

```text
main.py           Console entry point, orchestration, and bounded feedback loop
agents.py         Four agent functions and Reviewer validation
api.py            API client configuration, shared call, stateless tool loop, and safe errors
tools.py          Pure computation tools for chat (calculate, date_math) and their allowlist
usage.py          Per-session API request limit and usage counts (UsageGuard)
test_workflow.py  Offline workflow, response, storage, and CLI tests
test_tools.py     Offline tests for the tools, their limits, and the tool loop
test_usage.py     Offline tests for the request limit, counting, and retries
test_demo_rehearsal.py  Offline rehearsal of the seven-step live demo (CLI and Web UI)
requirements.txt  Pinned OpenAI SDK and Streamlit dependencies
.gitignore        Keeps local secrets and generated Python files out of Git
airi.py           Optional conversation CLI and explicit command routing
persona.py        Stable AiRi conversational persona and local profile loading
profile.example.toml  Example local profile (copy to local_data/profile.toml)
memory.py         SQLite storage and simple keyword retrieval
approval.py       Approval gate for protected actions such as /forget
suggestions.py    /suggest context limits, strict output, and suggestion checks
test_airi.py      Persona, memory, interaction, approval command, and Git privacy tests
test_approval.py  Offline unit tests for the approval gate
test_suggestions.py  Offline tests for memory suggestions
streamlit_app.py  Local Web UI using the same AiRi and Core workflow
test_streamlit_app.py  Offline Streamlit AppTest integration tests
.streamlit/config.toml  Localhost binding, theme, and disabled usage telemetry
```

## Setup and usage

### Check the existing Python first

In the VS Code PowerShell terminal, from this project directory:

```powershell
python --version
py -0p
Get-Command python | Select-Object Source
python -c "import sys, struct, importlib.util; print(sys.executable); print('Bits:', struct.calcsize('P') * 8); print('Streamlit installed:', importlib.util.find_spec('streamlit') is not None)"
```

Use your existing interpreter to run checks before changing the environment:

```powershell
python -W error::ResourceWarning -m unittest -v test_airi test_workflow
python -m compileall -q main.py agents.py api.py airi.py persona.py memory.py streamlit_app.py test_workflow.py test_airi.py test_streamlit_app.py
git diff --check
```

If Streamlit is already installed in that interpreter, also run:

```powershell
python -m unittest -v test_streamlit_app
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

If `python` is unavailable in one terminal but worked previously, check VS Code's
selected interpreter and `py -0p` before installing anything. Use `py` in place of
`python`, or invoke the existing executable by full path. Agent sandbox PATH and
program access can differ from your VS Code terminal; failure there does not mean
Python is missing from the computer.

### Dependency setup when needed

Use **64-bit Python 3.11+** for the Web UI and an OpenAI API account with quota and access to the selected model. The current project `.venv` uses **Python 3.13.16 64-bit** with `openai==3.19.2` and `streamlit==1.64.0`; all tests pass there. Streamlit dependencies such as PyArrow may not install on 32-bit Python. A 32-bit Python 3.14 may still be installed on this computer (`py -0p`); it lacks Streamlit, so always run the UI and tests through `.\.venv\Scripts\python.exe`. Check your interpreter with `python -c "import struct; print(struct.calcsize('P') * 8)"` (it should print `64` for UI setup). If `python` is unavailable but the Windows launcher exists, use `py -0p` to list installed interpreters and run the selected **64-bit** executable by its full path.

From this project directory, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.venv` with that 64-bit interpreter. If `.venv` was previously created
with 32-bit Python, close processes using it and rename it (for example to
`.venv-32bit`) before recreating it; switching the system interpreter alone does
not change an existing virtual environment. Keep renamed environments out of Git.

The current `.venv` was created at an earlier project location
(`Desktop\CODING\Project_VINFREY`). `.venv\Scripts\python.exe` and
`Activate.ps1` still work, but the console launchers (`streamlit.exe`, `pip.exe`,
and similar) and `activate`/`activate.bat`/`activate.fish` still point to the old
path. Use `.\.venv\Scripts\python.exe -m streamlit` and `-m pip` rather than the
`.exe` launchers until the environment is recreated in this folder.

Set `OPENAI_API_KEY` in the environment. To avoid entering a key in shell history, PowerShell can prompt for it:

```powershell
$vinfreyKey = Read-Host "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $vinfreyKey).Password
Remove-Variable vinfreyKey
.\.venv\Scripts\python.exe main.py
```

If your existing Python environment already has the dependency, use `python main.py`. Do not put a real key in source files, documentation, or Git. The application reads environment variables directly; it does not load `.env` files.

## Start the local Web UI

Run these commands **from the project directory** after installing the dependencies
and setting `OPENAI_API_KEY` in the same terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open **http://127.0.0.1:8501**. Stop the server with `Ctrl+C`. The committed
Streamlit config also binds to localhost, disables static file serving and usage
telemetry, and supplies the light sage theme. Do not bind to `0.0.0.0` or publish
this demo with personal Memory: it has no login and all local sessions share the
same SQLite notes. There is no API key entry field or database download button.
To present AiRi without your personal Memory or profile, start the live API demo
instead (see Live API demo below).

Try these messages in order:

```text
วันนี้มีเรื่องอยากคุยด้วย ช่วยคุยแบบตรง ๆ ได้ไหม
ลองแย้งความคิดนี้อย่างมีเหตุผล: เรียน Python ต้องจำทุกคำสั่งก่อนเริ่มเขียน
/remember Python ชอบตัวอย่างสั้น ๆ
/memories
Python ขอตัวอย่าง list แบบสั้น ๆ
/work Write exactly two short beginner bullet points about Python lists.
```

The first messages exercise AiRi's friend-like voice and reasoned disagreement.
`/remember` and `/memories` work offline without a key. Only `/remember` writes
notes (maximum 2000 characters); `/forget` deletes one only after you approve it
with the **อนุมัติ** button or `/approve`. The next chat uses matching Memory. `/work`
calls the existing Planner → Researcher → Builder → Reviewer workflow, including
its three-revision feedback loop. While it runs, a live status box lists each
step (Planner ✓, Builder draft 2…, Reviewer: FAIL → แก้ไขรอบ 1/3) without the
work content, and turns to complete or error at the end. After completion, the UI
shows the final draft, PASS/FAIL, actual revision count, Reviewer feedback, a
one-line step summary, and a collapsed **รายละเอียดขั้นตอนของ Agents** section
with the plan, research, every draft, and every review. FAIL keeps the
last draft visible for human review. Generated code is displayed, never executed.
Model-written replies, drafts, and feedback are shown as plain text or code blocks,
never rendered as Markdown or HTML (see Safe rendering of model text below).

Messages, errors, and the AiRi instance live in `st.session_state`, so ordinary
Streamlit reruns keep the transcript and do not resubmit API calls. The model
receives only the last six successful chat turns. A new tab, browser refresh that
creates a new session, or server restart can clear chat; SQLite notes persist.
**เริ่มบทสนทนาใหม่** clears the current transcript and AiRi's in-memory context,
while preserving saved notes. `/quit` is a CLI command; close the tab to leave UI.

Chat sends the current message, recent chat context, and matching notes to OpenAI;
`/work` sends only the task into Core. No conversation or workflow result is
automatically stored in SQLite. The workflow step trace stays in the browser
session only and is cleared by a new conversation. The Web UI no longer prints
workflow output to the server terminal; the CLI (`main.py`, `airi.py /work`)
still prints it, so do not record/share that terminal with private tasks.
API errors are readable and unexpected exceptions are hidden from the UI.
You can correct the problem and submit again; failed chat turns do not enter
AiRi's model context. Refreshing does not retry a failed submission automatically.

Optionally select a model your API account can access:

```powershell
$env:OPENAI_MODEL = "gpt-6-luna"
```

The selected model must support Responses API and Structured Outputs. Availability depends on your account; the program does not silently switch models. All agents use this setting. Blank model values are rejected.

Example task:

> Write a beginner Python guide explaining lists, with one runnable example and its expected output.

The CLI uses UTF-8 output so Thai and other model text can be printed on Windows. The console shows each agent's progress, plan, research, draft, review feedback, revision count, and final status. A revision appears as:

```text
=== Reviewer ===
FAIL
Feedback: Add a runnable example and expected output.

=== Retry ===
FAIL -> Builder revision 1/3
```

Generated code is returned as text. VINFREY does not execute code or write generated projects to disk.

## Retry and error behavior

- A normal run makes four agent calls.
- A failed review sends the current draft and feedback back to Builder, then runs Reviewer again.
- At most three revisions are allowed, for four drafts and ten agent calls total, excluding network retries.
- A remaining FAIL prints `STOP -> Human review required`; the last draft and feedback are visible in the console.
- API calls have a 60-second timeout. In the AiRi CLI (`airi.py`) and the Web UI, requests go through the usage guard, which turns automatic SDK retries off, so a network error stops the request at once and you can send it again. `python main.py` (Core only) still lets the SDK retry retryable network/API failures up to two times, separately from Builder revisions; total elapsed time can then exceed one timeout.
- Missing keys, authentication errors, rate limits/quota, timeouts, connection errors, API status errors, empty output, incomplete responses, and model refusals stop with a readable message.
- Raw API error bodies and keys are not printed.
- In chat, the tool loop also stops with a readable message when a turn exceeds the
  tool-call limits or the API returns an unexpected tool response.
- CLI exit codes: `0` for PASS, `1` for errors or exhausted revisions, `130` for cancelled input.

## Tests

No API key or network is needed. Run from the project directory with the project `.venv` Python:

```powershell
$env:OPENAI_API_KEY = ""
.\.venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -v
.\.venv\Scripts\python.exe -m compileall -q main.py agents.py api.py tools.py usage.py airi.py persona.py memory.py streamlit_app.py test_workflow.py test_tools.py test_usage.py test_airi.py test_streamlit_app.py test_demo_rehearsal.py
git diff --check
```

`$env:OPENAI_API_KEY = ""` clears the key for that PowerShell window until it is
closed. Use a separate window for tests rather than the one where you set a real key.

Tests run the actual agent functions with scripted Responses API results. They cover immediate PASS, FAIL -> feedback -> revised draft -> PASS, PASS on the last allowed revision, repeated FAIL and stopping at the limit, zero revisions, context propagation, invalid reviews, API failures, missing configuration, refusal/incomplete output, import safety, and CLI exit codes.

Offline tests verify integration and control flow; they do not establish real model quality, account access, or billing. A live smoke test requires a configured key and quota and incurs API usage. Run the example task above to check your account.

Do not substitute plain `python` unless it is the same 64-bit `.venv` interpreter;
the system Python may lack Streamlit. `test_streamlit_app.py` uses Streamlit's [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
with temporary SQLite databases and scripted API replies. It covers session/rerun
behavior, explicit Memory writes, context delivery, Core retry and exhausted FAIL
results, safe errors and recovery, a new conversation, and separate browser
sessions sharing local notes. It never reads personal Memory or makes live calls.
The full suite has 293 tests: 32 in `test_workflow`, 34 in `test_tools`, 13 in
`test_usage`, 120 in `test_airi`, 53 in `test_streamlit_app`, 3 in
`test_demo_rehearsal`, 22 in `test_approval`, and 16 in `test_suggestions`. After the UI tests pass, Streamlit may print
`ResourceWarning: Implicitly cleaning up <TemporaryDirectory ...>` at interpreter
exit. It comes from a module-level temporary directory inside Streamlit's AppTest
package, not from project code, and does not change the test result or exit code.
Without Streamlit installed, run the original regression separately:
`python -m unittest -v test_airi test_workflow`.

## Core scope and remaining work

Researcher prepares notes from model knowledge; it has no web search or source verification tool. Reviewer is another model judgment, so a PASS is not a guarantee of factual correctness or code safety. Review important outputs yourself.

The core agents and loop are unchanged. `run_workflow()` accepts an optional
`on_event` callback that receives plain progress dicts (`agent_started`,
`agent_finished`, `revision`, `finished`); without it, the CLI prints exactly the
same output as before, which golden-output tests check. The return value is
unchanged. AiRi adds the Persona, explicit local Memory, the approval gate, pure
computation tools, context awareness, the usage guard, and a Streamlit Web UI around
it (see Features at a glance). Voice, microphone, TTS, STT, Camera, embeddings, cloud memory,
login, autonomous code execution, and additional frameworks remain outside this version.

## AiRi Persona + Local Memory

Run `python airi.py` from this directory to start a conversation. `python main.py`
still runs the original task workflow directly. The CLI needs only the existing
OpenAI SDK; the Web UI additionally needs Streamlit. Memory uses standard-library `sqlite3`.

AiRi has a calm, candid, friend-like female voice, reasoned opinions, and her own
consistent style rather than merely mirroring the user. Light teasing is welcome
when appropriate, without forced jokes, flattery, or exaggerated enthusiasm. She
uses the partner's name only when it is configured in a local profile, and then
sparingly; without one she does not know or guess a name. She focuses when given work while keeping her voice. She must not invent memories
or claim real human feelings or experiences. `persona.py` supplies the same
instructions on every chat call in both CLI and UI; model behavior still needs
manual evaluation across conversations.

To set a name, copy `profile.example.toml` to `local_data/profile.toml` and edit
`user_name` (one line, at most 40 characters). `local_data/` is ignored by Git, so
the real profile stays local. An invalid profile, including one without
`user_name`, is ignored with a short warning
in the CLI and the Web UI sidebar; AiRi then runs without a name. Restart the CLI
or open a new browser session after changing it. Tests use a temporary profile.

Try this sequence in the AiRi console (the note is an example, not personal data):

```text
/remember เวลาเรียน Python ชอบตัวอย่างสั้น ๆ
/memories
Python ช่วยยกตัวอย่าง list แบบง่าย ๆ
ช่วยอธิบายตัวอย่างเมื่อกี้ต่อ
/work Write exactly two short beginner bullet points about Python lists.
/quit
```

- `/remember <note>` explicitly saves a note of up to 2000 characters.
- `/memories` lists saved note IDs and contents without an API call.
- Ordinary text chats with AiRi using matching memories and up to six previous
  successful chat turns, held only in RAM. Restarting clears the conversation.
- `/work <task>` passes the task directly to the existing `run_workflow()`.
  Persona, memories, and chat history are not added to Core agent prompts.
  There is no model-based intent router; use `/work` explicitly for reviewed work.
- `/forget <id>` asks to delete one note permanently. Nothing is deleted until you
  approve it; see Approval gate below.
- `/suggest` asks AiRi to suggest up to three memories from your own recent chat
  messages. Each suggestion waits for approval; see Memory suggestions below.
- `/pending`, `/approve <id>`, and `/deny <id>` list and decide actions waiting
  for approval (for example `/approve A1`).
- `/help` shows commands; `/quit` exits. Memory commands work without an API key.
  Chat and `/work` use the same environment configuration and safe API errors
  as the existing core. A chat API error leaves the console available to retry.

The database is created at `local_data/memory.sqlite3` beside `memory.py`, even
when launched from another working directory. No chat message, assistant reply,
or workflow result is saved automatically. Notes survive restarts. SQLite is local,
unencrypted storage: avoid saving secrets or unnecessary sensitive information.
To remove one note, use `/forget <id>` and approve it. To reset all notes, close
AiRi and remove this database locally.

### Approval gate

`approval.py` holds protected actions until a person approves them. `/forget` is
the first protected action; `/remember` stays a direct command because typing it
is already explicit consent.

1. `/forget 3` validates the target and records a pending action such as `A1`.
   The summary shows the note id, when it was saved, its length, and that deletion
   is permanent. It never shows the note text; check `/memories` before approving.
2. `/approve A1` or the **อนุมัติ** button runs it once. `/deny A1` or **ปฏิเสธ**
   discards it. Both buttons send the same `/approve` / `/deny` command text that
   the CLI accepts, through the same handler.
3. A decided action cannot run again. Repeated clicks, reruns, or a second
   `/approve` show a safe error.

Safeguards:

- **Default deny:** unknown action kinds, invalid parameters, unknown ids, and
  already-decided actions do nothing.
- **Recommendation is not execution:** creating a request never runs it. Model
  replies are never parsed as commands, so text such as `/approve A1` from the
  model cannot approve anything.
- **Frozen parameters:** parameters are validated, deep-copied, and made read-only
  when requested; the summary is built from them, not from model text.
- **Single execution:** an action is marked `executing` under a lock before it
  runs. Results are `executed`, `rejected` (precondition failed, nothing changed),
  or `failed` (it may have partly run; never retried automatically).
- **Bound handlers:** each action kind can be registered once, and a pending
  action always runs the handler that validated it.
- **Stale-target protection:** deletion checks the note id, a SHA-256 of its
  content, and `created_at` inside one `BEGIN IMMEDIATE` transaction. If the note
  changed, was removed, or its id was reused by a new note, nothing is deleted.
  Memory ids must be between 1 and SQLite's 64-bit limit; anything else is a safe
  error, and the CLI keeps running.
- **Id-reuse mitigation (partial):** a second `/forget` for the same id is refused
  while one is pending. When AiRi saves a note under an id that has a pending
  deletion (`/remember` or an approved suggestion), or deletes that id, the
  pending deletion is cancelled (`invalidated`) and the reply says so. This only
  covers changes made through the **same AiRi instance** (one CLI run or one
  browser session). Any other instance, including another browser session in the
  same Streamlit process, the CLI, another process, or a direct database edit, can
  still delete the note and re-save identical text under the same id within the
  same second. That cannot be told apart, so the pending deletion could then
  remove the new note. A full fix needs a separately approved schema change
  (`AUTOINCREMENT` or a UUID column).
- **Per session:** each CLI run and browser session has its own queue. Pending
  actions survive Streamlit reruns. **เริ่มบทสนทนาใหม่** removes every record of
  that session's queue. Action ids are never reused.
- **Minimal retention:** once an action is decided (approved, denied, rejected,
  failed, or cancelled), the gate keeps only its id, kind, and status; its
  parameters and summary, including any suggested text, are dropped.
- **Safe errors:** messages contain action ids and states only, never note text,
  file paths, or raw exceptions.

After a successful `/forget`, the AiRi instance that ran the deletion (that CLI
run or browser session) clears its own recent chat context and its `/work`
summary, so the deleted note
(or replies that quoted it) is not sent with its later chat or `/suggest`
requests. Only that instance is cleared: other active sessions keep any content
they already saw, including replies that quoted the note, and can still send it
until they start a new conversation. The visible Web UI transcript stays on
screen. This cannot recall anything already sent to the AI provider in earlier
requests. A denied, rejected, or failed deletion keeps the chat context.

This is an application-level control, not an operating-system security boundary:
code running in the same process could call the gate directly. The protection is
that only explicit user commands and approval buttons reach it.

### Memory suggestions

`/suggest` is the only way AiRi proposes memories. It runs only when you type it;
AiRi never suggests or saves memories on its own during normal chat.

1. `/suggest` makes **one** API call (`suggestions.py`). It sends only your own
   recent chat messages: at most the last six, at most 3,000 characters, newest
   first. Messages that look like credentials are left out before sending. AiRi's
   replies, saved memories, the profile name, slash commands, `/work` traces, and
   approval records are never sent. With nothing usable, it stops locally without
   an API call.
2. The model must return strict JSON (Structured Outputs) with at most three
   suggestions; Python checks the shape again. A malformed response is rejected
   as a whole with a safe error and is not retried.
3. Each suggestion must be one line of 1–280 characters with a reason of 1–160
   characters, without control characters or credential-like text. Suggestions
   that fail, repeat each other, match a saved memory, or match a pending
   suggestion are skipped and counted in the reply.
4. Accepted suggestions become pending `memory.save` actions requested by the
   assistant. Nothing is saved until you approve with `/approve <id>` or the
   **อนุมัติ** button; `/deny` or **ปฏิเสธ** discards it. If none are left, the
   reply is `No suitable memories to suggest.`
5. On approval the duplicate check runs again inside one `BEGIN IMMEDIATE`
   transaction, so two sessions approving the same text store it only once.
   A note that already exists is reported as not executed.

The approval card shows the exact suggested text and reason so you can judge
them. Because that text comes from the model, the Web UI shows it, the
`/suggest` reply, and `/pending` as plain text only, never as Markdown or HTML.

Known limitations:

- Credential detection is a pattern heuristic, and avoiding inferred sensitive
  details (health, religion, finances, and so on) or other people's private
  information relies on the model instructions plus your approval.
- Duplicates are compared as normalized text (Unicode NFKC, case, whitespace);
  the same idea in different words is not detected. `/remember` still saves
  duplicates as before.
- Saved notes do not record whether they came from a suggestion; that and other
  metadata need a future, separately approved schema change.
- CLI chat history lives in RAM, so `/suggest` has nothing to use after a restart.
  If the latest message alone is longer than 3,000 characters, nothing is sent.
- Suggestion quality has not been evaluated against the live API.

### Safe rendering of model text

Everything the model writes is shown verbatim in the Web UI, never through a
Markdown or HTML renderer, so a reply cannot load remote images (which could leak
data through the URL), show disguised links, or inject HTML:

- AiRi's chat replies, `/suggest`, `/pending`, and approval cards: plain text.
- `/work` drafts, Reviewer feedback, and the agent step details: code blocks
  (`st.code`, Markdown syntax shown as text, with line wrapping and a copy button).

Labels written by the app itself (PASS/FAIL banners, headings, step names) still
use Markdown. Text you type yourself is still shown as Markdown. The trade-off is
that formatting such as bold or lists in model replies appears as raw symbols.

### Tool calling (P0-5)

In chat, AiRi can use two **pure computation** tools. The CLI (`python airi.py`) and
the Web UI turn them on explicitly; an `AiRi` object created in code has no tools
unless `tools=` is passed. `/work` and `/suggest` never receive tools.

| Tool | What it does | Limits |
| --- | --- | --- |
| `calculate(expression)` | Arithmetic with numbers, `+ - * / // % **`, and parentheses | At most 200 characters of ASCII digits, operators, dots, spaces, and parentheses (no names, functions, or scientific notation). At most 64 expression nodes nested 20 deep. Every number and every intermediate result stays within ±10^15. At most two `**`, never nested (`2**3**4` is rejected); the exponent must be a whole number up to 64, and the result size is estimated before computing. Division by zero is rejected. Decimal results use ordinary floating point and are shown to 15 significant digits. |
| `date_math(operation, start, end, days)` | `days_between` two `YYYY-MM-DD` dates, or `add_days` to a date | Years 1–9999; at most 36,500 days. The tool never reads the clock; in chat, AiRi takes "today" from the chat context (see Context awareness below). |

How a chat turn with tools works:

- Every request uses `store=False` and no `previous_response_id`. Each follow-up
  request resends the turn: the user message, every output item of earlier rounds in
  their original order (including reasoning items and any encrypted reasoning), and
  one `function_call_output` per call with the same `call_id`.
- At most three rounds may contain tool calls and at most four tool calls run per
  message, so a message makes at most four requests. Limits are checked before any
  tool of that round runs. Parallel tool calls are turned off.
- An unknown tool or invalid arguments never run: the model receives a fixed error
  code such as `unknown_tool` or `invalid_arguments` and can answer without the
  tool. A malformed response (missing or repeated `call_id`, an unexpected item
  type, an incomplete call) stops the turn with a safe message.

Safety boundaries:

- The registry is an explicit allowlist (default deny). It accepts only tools with
  `side_effects="none"` and strict schemas, rejects duplicate names, and checks the
  arguments again in Python (exact keys, exact types, lengths, enums).
- Tools never read or write files, use the network, start processes, read the
  clock, or touch the Memory database. `tools.py` imports only standard-library
  computation modules, and a test checks this.
- Results are capped at 1,000 characters. Handler errors become fixed codes; raw
  exception text never reaches the model or the screen.
- Tool results are untrusted data for the model only. They are never shown on
  screen, never parsed as commands, never routed to `/approve` or the approval gate,
  and never saved as memories. Only AiRi's final reply is shown, as plain text.
  Chat history keeps the user message and that final reply, not tool calls, tool
  results, or reasoning. Encrypted reasoning is never printed, logged, or shown.

Privacy: tool arguments are written by the model from the conversation it already
received. They are computed locally, and results go back to the same provider in
the same turn. Because each follow-up request resends the turn, one message can
send the same context up to four times. `store=False` applies to every request,
including `/work` and `/suggest`: responses are not kept for later retrieval, but
the provider's own data-retention policy still applies.

Known limitations:

- A live smoke test on 2026-11-03 passed for both tools (see Validation record).
  Neither live run returned a reasoning item, so sending encrypted reasoning back
  while `store=False` is covered by offline tests only. If a model returns reasoning
  items without encrypted content, a follow-up request could be rejected; the
  prepared fix is one line (`include=["reasoning.encrypted_content"]`).
- Time is bounded by the input limits above, not by a hard timeout.
- The model may skip a tool or call it wrongly. Tool results can influence the
  wording of a reply, but they can never become a command or an approval.

### Tool Safety Contract

Any future tool must follow these rules (P0-5 follows them):

1. **Read-only tools only.** Every tool declares `side_effects="none"`, and the
   registry refuses any tool that does not. Tools may not write files, use the
   network, start processes, or change the database.
2. **No external side-effecting tools** until durable idempotency (idempotency
   keys plus a persistent ledger) and crash recovery are designed and approved.
   The approval gate guarantees only that **one action in one gate** runs at most
   once. Each AiRi instance (a CLI run or a browser session) has its own gate in
   memory, so separate instances are never deduplicated against each other, even
   inside the same process. If a process stops after a side effect but before
   recording it, the outcome is unknown. There is no cross-instance or
   cross-process exactly-once guarantee.
3. Memory changes go only through the existing approval-gated actions. Their
   database conditions only narrow what a stale approval can do: `save_if_absent`
   refuses to store a note equal to an existing one, and `delete_if_unchanged`
   deletes only a row whose id, content hash, and `created_at` still match. This
   is not idempotency for deletion: approvals in two instances are independent, and
   a note re-saved with identical text under the same id within the same second
   can be deleted by an older approval (see the id-reuse limitation above).
4. Tool results are untrusted data: never parsed as commands, shown as plain text,
   size-limited, and never saved as memories automatically.
5. The tool loop uses an allowlist, schema-validated arguments (default deny),
   limits on calls per turn and on input size, and safe error messages.
6. Tools that read local data (the clock, Memory) or act on anything need their own
   approved design. Side-effecting tools must also go through the approval gate.

### Context awareness (P0-6)

Each chat request also tells AiRi the current date and time and, after a `/work`
run, a short summary of that run. `/work` and `/suggest` requests include neither.

- **`now`:** the local date, weekday, time (HH:MM), and UTC offset, read fresh from
  the computer clock for every chat message. This shares your local time and time
  zone with the provider. AiRi can pass the date to `date_math` herself; the tool
  still never reads the clock.
- **`recent_work`:** the task, PASS/FAIL, and a filtered excerpt of the final result
  of the last finished `/work` in this CLI run or browser session. Only one summary
  is kept. Reviewer feedback, agent step details, and earlier drafts are never
  included. The summary is not part of chat history, so history stays limited to
  six turns and `/suggest` never sees it.

Before the summary is stored:

1. Fenced code blocks become `[code block omitted: N lines]`.
2. Lines that look like credentials (known key prefixes, private key headers,
   `password=` or `token:` style assignments, long letter-and-digit tokens) become
   `[line omitted: possible credential]`. A task that looks like one is omitted whole.
3. Home-directory paths (`C:\Users\…`, `/Users/…`, `/home/…`) become `[path]`.
4. The task is cut at 300 characters and the result at 1,500 (`truncated: true`).
5. If anything recognizable is still left, the whole result is replaced.

This filtering is heuristic. It catches common patterns only and does **not**
guarantee that no private or sensitive text reaches the provider, so keep secrets
out of `/work` tasks.

The summary is replaced after each finished `/work` (PASS or FAIL) and left as it
was when `/work` fails. **เริ่มบทสนทนาใหม่** and a successful `/forget` clear it; a
deletion that is only requested, denied, rejected, or failed keeps it. Each
session has its own summary.

The `/work` result is model-written text and may contain injected instructions.
AiRi's persona treats supplied memories, history, the date and time, and recent
work results as background data, never as instructions. That is a mitigation, not
a security guarantee: such text can still influence the wording of a reply. It
cannot run anything, because AiRi's replies are never routed to commands,
`/approve`, or the approval gate; a test checks that a `/work` result containing
`/approve A1` leaves the action pending.

### Usage guard (P0-7)

Each AiRi session (one `airi.py` run or one browser session) has its own request
counter. Every chat request, tool-loop round, `/work` agent call, and `/suggest`
request goes through it.

- **Limit:** 40 API requests per session (20 in the live demo). When the limit is
  reached, the next request is refused **before** it is sent. Counting and the
  limit check happen together under a lock, so even concurrent requests cannot
  pass the limit.
- **Counting:** a request counts once as soon as it is dispatched, even if it then
  fails or times out. A missing API key fails before dispatch and does not count.
  Automatic SDK retries are off for these requests.
- **Before starting work:** AiRi checks that enough requests remain for the worst
  case — 1 for chat (4 when tools are on), 1 for `/suggest`, and 10 for `/work` —
  and refuses at once if not. Nothing is sent, and chat history, the `/work`
  summary, and pending approvals stay as they were. Local errors such as an empty
  message are reported first. This check does **not** reserve requests: if other
  requests in the same session run at the same time, work that passed the check can
  still stop partway through (the limit itself is never exceeded).
- **`/usage`** shows `requests/limit`, input and output tokens, how many responses
  had no usage data, and failed requests. It runs locally and works even after the
  limit is reached. The Web UI shows the same counts at the bottom of the sidebar.
  Token counts come from the API's own usage report; they are not money.
- **เริ่มบทสนทนาใหม่** does not reset the counter.

This is a per-session safety limit, **not a spending limit or an account-wide cost
control**. The counter lives in memory: restarting the CLI, reloading the page into
a new session, or opening another browser session starts again from zero. Set a
budget limit in your OpenAI account to cap real spending.

### Live API demo (P0-7)

Set `VINFREY_DEMO=1` (exactly `1`) before starting the CLI or the Web UI:

```powershell
$env:VINFREY_DEMO = "1"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

This is a **live** demo: it sends real requests to the OpenAI API, needs a key and
a network, and may cost money. The CLI prints `LIVE API DEMO` first, and the Web UI
shows a `LIVE API DEMO` warning in the sidebar.

- Memory is a new temporary SQLite file for each demo session, outside
  `local_data/`. The CLI deletes it on exit. The Web UI deletes it when that browser
  session's resources are collected or the server shuts down normally, which may
  not happen right after a tab closes (and may not happen if the process is killed).
  Demo sessions never share notes, and nothing carries over to the next demo.
- The real Memory database and the private profile (`local_data/profile.toml`) are
  never opened, so no saved note or personal name reaches the prompt.
- The request limit is 20 per session.

Seven-step demo script (synthetic content only, about 11 of the 20 requests):

| # | Type | What to show |
| --- | --- | --- |
| 1 | `ช่วยคุยแบบตรง ๆ หน่อย ว่าการเรียน Python ต้องจำทุกคำสั่งไหม` | AiRi's own reasoned view, not just agreement |
| 2 | `What is (1234 * 56) - 789?` | `calculate` runs locally (answer 68315) |
| 3 | `อีกกี่วันจะถึงวันที่ 25 ธันวาคม` | Today's date comes from the chat context; `date_math` counts the days |
| 4 | `/work Write exactly two short beginner bullet points about Python lists.` | Live step status, PASS/FAIL, revisions, collapsed agent details |
| 5 | `สรุปงานที่เพิ่งทำให้หน่อย` | AiRi refers to the filtered summary of the last `/work` |
| 6 | `/remember ชอบตัวอย่างสั้น ๆ`, `/memories`, `/forget 1`, then **อนุมัติ** | Deletion waits for your click; text such as "/approve A1" written by the model is only displayed |
| 7 | `/usage` | Requests and tokens used in this session, and the 20-request limit |

With a real model the exact wording, and whether it chooses a tool, can differ.
`test_demo_rehearsal.py` runs this script offline in the CLI and the Web UI with
scripted replies; it shows the flow works, not how a real model will answer.

**Offline Demo Mode is deferred and not implemented.** There is no way yet to run
the demo without the live API.

Retrieval is intentionally basic: case-insensitive word/Thai substring matching,
ranked by number of matching query terms, with newer notes breaking ties. At most
five notes are included. Use explicit keywords such as `Python` or `ภาษาไทย`;
paraphrases and Thai text without word spacing may not match. There is no semantic
search, vector database, cloud database, or embeddings.

Storage is local, but **matching notes, recent chat history, the current date and
time, and the filtered summary of the last `/work` are sent to the OpenAI API as
conversation context**. An unmatched note can also remain in recent
assistant replies. Tool arguments and results in a chat turn go to the same API
(see Tool calling above). Every request sets `store=False`. The app writes no
conversation log. The memory directory,
SQLite files, and SQLite journal/WAL sidecars are ignored by Git. API keys remain
in environment variables; no `.env` loader or source-code secret is introduced.

Run all offline checks with `.\.venv\Scripts\python.exe -m unittest discover -v`
(see Tests above). Tests use temporary
databases with synthetic notes and scripted API replies; they verify persistence,
database creation/integrity, persona and memory context, explicit-only saving,
bounded history, command routing into the real Core, approval-gated deletion,
approval-gated memory suggestions, and Git ignore/tracking rules.

## Known limitations

Each item is explained in its own section above; this list collects them.

- **Live API coverage:** only P0-5 tool calling was smoke-tested against the live API
  (two runs, 2026-11-03). Neither run returned a reasoning item, so sending encrypted
  reasoning back with `store=False` is tested offline only. Context awareness (P0-6),
  the usage guard and live demo (P0-7), and the demo rehearsal are tested offline
  only, with scripted replies.
- **Offline Demo Mode** is deferred and not implemented; the demo needs the live API,
  a key, a network, and may cost money.
- **Usage guard:** a per-session safety limit, not a spending limit or account-wide
  cost control. Counters reset with a new CLI run or browser session. The worst-case
  check before work does not reserve requests, so concurrent work in one session can
  still stop partway through. `python main.py` is not guarded.
- **Filtering is heuristic:** the `/work` summary filter and the `/suggest`
  credential check catch common patterns only.
- **Prompt injection:** the persona treats supplied context as data, which is a
  mitigation, not a guarantee. Model text can influence a reply's wording but is
  never routed to commands or approvals.
- **Memory:** basic keyword retrieval; SQLite may reuse an id, and the protection
  against a stale approval deleting a re-saved note is partial (one instance only).
  A full fix needs a schema change.
- **`/forget`** clears only the deleting session's context and cannot recall
  anything already sent to the provider.
- **Model judgment:** Reviewer PASS is not a guarantee of correctness or safety, and
  Researcher has no web search.
- **Approval gate** is an application-level control, not an OS security boundary,
  with no cross-instance or cross-process exactly-once guarantee.
- **Web UI:** no login; bind to localhost only. Demo temporary files are removed when
  a session is collected or the server stops normally, not the instant a tab closes.
- **Tests:** Streamlit's AppTest prints a `ResourceWarning` at interpreter exit; the
  concurrency test checks the outcome but cannot prove the lock under the GIL.

## Validation record

Checked on 2026-11-03 for integration verification and demo readiness:

- All 293 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 290 tests that existed before were also run by
  ID and passed without changes, including the four CLI golden-output tests.
- Three new acceptance tests (`test_demo_rehearsal.py`) ran the seven-step demo
  script in the CLI and the Web UI with `VINFREY_DEMO=1`, scripted replies, and a
  fixed clock. They found no defect. Model text "/approve A1" stayed a displayed
  reply, unknown action ids were refused, usage reached 11/20 requests, a further
  `/work` was refused before sending, sessions stayed separate, and no prompt
  contained data from the private profile.
- A local server check started the Web UI with `VINFREY_DEMO=1` and a blank key on
  127.0.0.1 only: the health endpoint and index page answered, there was no
  traceback, and the process and port were closed afterwards. No browser was
  available, so the visible UI was not inspected in this check.
- A privacy audit of the four commits after `808d339` found no secrets, no new
  author identity, and only documented or synthetic paths. The real Memory database
  was not read for content or changed, and no API call was made.

Checked on 2026-11-03 for P0-7 (usage guard and live API demo):

- All 290 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 250 tests that existed before P0-7 were also run
  by ID and passed without changes, including the four CLI golden-output tests.
  Only the Streamlit exit-time warning described under Tests appeared.
- 40 new tests cover the request limit refusing before dispatch (also with 20
  concurrent callers), counting failed and timed-out requests once, a missing key
  not counting, tokens and missing usage data, SDK retries off for guarded clients,
  injected and patched clients, worst-case checks before chat, `/suggest`, and
  `/work` that leave history, the `/work` summary, and pending approvals unchanged,
  `/usage` after the limit, no reset on a new conversation, separate sessions, every
  CLI and Web UI API path going through the session guard, and the live demo
  (banner, temporary per-session memory removed at the end, limit 20, no profile
  read, and prompts that contain nothing from the private profile), and the sidebar
  Memory caption in both normal and demo modes.
- Every test used scripted API replies and temporary SQLite databases; no demo
  directory was left behind. The real Memory database was not read for content or
  changed, the approval gate and schema are unchanged, and no API call was made.

Checked on 2026-11-03 for P0-6 (context awareness):

- All 250 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 229 tests that existed before P0-6 were also run
  by ID and passed without changes, including the four CLI golden-output tests.
  Only the Streamlit exit-time warning described under Tests appeared.
- 21 new tests cover a fresh, locale-independent date and time on every chat
  request (with an injected clock); the `/work` summary after PASS and FAIL, its
  size limits, and the removal of code blocks, credential-looking lines, and
  home-directory paths (including a credential at the 1,500-character edge); the
  summary staying unchanged after a failed `/work`, separate from chat history, out
  of `/work` and `/suggest`, and per session; clearing on a new conversation and
  only on a successful `/forget`; a `/work` result containing `/approve A1` that
  cannot approve anything; the persona sentence; and the CLI and Web UI privacy
  text.
- Every test used scripted API replies, an injected clock, and temporary SQLite
  databases. The real Memory database was not read for content or changed, the
  approval gate and schema are unchanged, and no API call was made.

Live smoke test on 2026-11-03 for P0-5 (run by the owner in their own terminal):

- Synthetic prompts only, default persona with no profile, an empty temporary
  database, `store=False`, SDK retries off, and at most 2,000 output tokens per
  request. AiRi called no command and no approval path.
- `calculate`: PASS in 2 requests (1,391 input / 32 output tokens), answer 68315.
- `date_math`: PASS in 2 requests (1,424 input / 46 output tokens), answer 50.
- Both runs exited 0, the real Memory database was unchanged, and the Git working
  tree stayed clean. Neither run returned a reasoning item, so sending encrypted
  reasoning back with `store=False` remains tested offline only.

Checked on 2026-11-02 for P0-5 (pure computation tool calling):

- All 229 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 181 tests that existed before P0-5 were also run
  by ID and passed without changes, including the four CLI golden-output tests and
  the tests that mock `airi.generate`. Only the Streamlit exit-time warning
  described under Tests appeared.
- 48 new tests cover `store=False` on every request; the tool allowlist, strict
  argument checks, and safe error codes; `calculate` budgets (nested and large
  powers rejected before computing, node and depth limits, the ±10^15 bound);
  `date_math` ranges; no file, network, or process calls from tools; the stateless
  loop (item order, reasoning items, `call_id` linkage, round and call limits
  checked before tools run, malformed responses); encrypted reasoning never
  appearing in replies, errors, output, history, or the Web UI; tool results that
  cannot approve actions; tools enabled only at the CLI and Web UI entry points;
  and `/work` and `/suggest` never receiving tools.
- An in-memory mutation check on an isolated copy of the source removed each tool
  loop guard in turn (round limit, call limit, item type check, item validation,
  and the reasoning-safe refusal check). The tests caught all five.
- Every test used scripted API replies and temporary SQLite databases. The real
  Memory database was not read for content or changed, the schema and approval
  gate are unchanged, and no API call was made. (The live smoke test above was run
  afterwards.)

Checked on 2026-10-31 for P0-4.1 (security hardening after an independent audit):

- All 181 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 161 tests that existed before P0-4.1 were also
  run by ID and passed, including the CLI golden-output tests. Only the rendering
  checks of three existing UI tests changed (Markdown elements to code blocks),
  with the same expected content.
- 20 new tests cover plain-text rendering of chat replies, `/work` output,
  feedback, and step details against Markdown images, links, and HTML; the
  approval gate's bound handlers, single registration, minimal retention, and
  cancellation; SQLite id bounds including the CLI; the partial id-reuse
  mitigation; and `/forget` clearing outbound chat and `/suggest` context.
- The audit findings were first reproduced with an offline script on temporary
  databases. The same script now shows them fixed, except the documented
  cross-process id-reuse case.
- Every test used scripted API replies and temporary SQLite databases. The real
  Memory database was not read for content or changed, the schema is unchanged,
  and no API call was made.

Checked on 2026-10-30 for P0-4 (approval-gated memory suggestions):

- All 161 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 114 tests that existed before P0-4 were also
  run by ID and passed unchanged, including the CLI golden-output tests.
- 47 new tests cover the outbound context limits (six messages, 3,000
  characters, credential exclusion), strict response checks, SDK format
  compatibility, duplicate checks including two SQLite connections saving at
  once, approval-only saving, and Web UI cards that show Markdown and HTML image
  text as plain text.
- Every test used scripted API replies and temporary SQLite databases. The real
  Memory database was not read for content or changed, the schema is unchanged,
  and no API call was made.

Checked on 2026-10-29 for P0-3 (approval gate and `/forget`):

- All 114 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 73 tests that existed before P0-3 were also
  run by ID and passed unchanged, including the CLI golden-output tests.
- 41 new tests cover the gate (default deny, frozen parameters, single and
  concurrent execution, rejected/failed results), stale-target deletion with a
  reused SQLite id, model text that cannot approve, and the Web UI approve/deny
  buttons, reruns, repeated approval, reset, and separate sessions.
- Every test used temporary SQLite databases; the real Memory database was not
  read for content or changed. `main.py`, `agents.py`, `api.py`, and `persona.py`
  are unchanged. No API call was made.

Checked on 2026-10-29 for P0-1 (workflow events and live UI status):

- All 73 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`; the 56 tests that existed before P0-1 were also
  run by ID and passed unchanged.
- Four golden-output tests, recorded before `main.py` changed, confirm the CLI
  prints exactly the same text when no `on_event` hook is given.
- AppTest checks the live `st.status` state (complete or error), the in-session
  trace, the collapsed details section, safe errors, and that the Web UI prints
  nothing about the task to the server terminal.
- `agents.py`, `api.py`, and `memory.py` are unchanged. No API call was made.

Checked on 2026-10-27 for P0-2 (local profile and UI text fixes):

- All 56 offline tests passed (the original 37 unchanged, plus 19 new) with
  `OPENAI_API_KEY` blank and `-W error::ResourceWarning`.
- The partner's name is no longer in tracked source; it comes from the ignored
  `local_data/profile.toml`. Tests use a temporary profile and temporary SQLite.
- `/help`, `/memories`, and `/remember` replies render as multi-line text in the UI.
- `main.py`, `agents.py`, `api.py`, and `memory.py` are unchanged. No API call was made.

Checked on 2026-10-27 (baseline audit and UI smoke test, no source changes):

- Environment: project `.venv` with Python 3.13.16 64-bit, `openai==3.19.2`,
  `streamlit==1.64.0`. `pip check` reported no broken requirements.
- All 37 offline tests passed with `OPENAI_API_KEY` blank and
  `-W error::ResourceWarning`, including the nine Streamlit AppTest UI tests
  that were pending on 2026-10-11. Only the Streamlit exit-time warning described
  under Tests appeared.
- The real app started with `.\.venv\Scripts\python.exe -m streamlit run
  streamlit_app.py` on 127.0.0.1 with no startup error. A headless Chrome session
  checked the page title, sidebar, missing-key notice, `/help`, the missing-key
  error for chat and `/work`, the unknown-command error, and **เริ่มบทสนทนาใหม่**.
  No browser console errors or Streamlit exceptions appeared.
- Memory UI checks used a throwaway SQLite database, not personal Memory:
  `/remember`, the empty-note error, `/memories`, notes kept after a new
  conversation, and a fresh browser session with empty history but the same notes.
- Known display issue (fixed later by P0-2, see above): multi-line `/help` and
  `/memories` replies were shown on one line in the Web UI, because single
  newlines collapse in Markdown. The CLI showed them correctly.
- Not covered: Persona tone and real chat or `/work` replies need a live API key
  and quota; no external API call was made.

Checked on 2026-10-11 for the friend-like Persona and local Web UI:

- All 28 existing offline tests passed, including Core feedback/retry regression.
- Python compilation passed for all application and test files, including UI.
- UI integration tests have been added; see the earlier environment limitation below.
- Git ignore/tracking and source privacy checks passed; no local database or
  secret file is tracked. `git diff --check` passed.
- `main.py`, `agents.py`, `api.py`, and `memory.py` remain unchanged.
- No live API call was made for this update. No commit or push was performed.

Earlier environment limitation (resolved by the 2026-10-27 check above, which
used a 64-bit Python 3.13 `.venv`): the installed interpreter was Python 3.14.7
**32-bit**, and Streamlit was not initially installed. Installing `requirements.txt`
into a project `.venv` failed while preparing pandas from source (no compatible
wheel was selected and the required Visual Studio build tools were absent).
No new Python runtime was installed. Full UI/AppTest execution and visual browser
verification remain pending a compatible 64-bit environment with dependencies
installed. At that time the nine UI tests had not been run.

Follow-up interpreter check on 2026-10-11: outside the agent sandbox,
`python --version`, `py -0p`, and `Get-Command python` all found the existing
`C:\Users\<user>\AppData\Local\Programs\Python\Python314-32\python.exe`.
Inside the sandbox, `python` was unresolved and `py` reported no interpreters;
this was an environment access/PATH difference, not a missing installation.
The existing interpreter has no Streamlit. No further download or installation
was attempted during this follow-up; the instructions above allow verification
in the user's VS Code terminal first.

Checked on 2026-10-09 for Persona + Local Memory:

- All 28 offline unittest tests passed, with ResourceWarning treated as an error.
- Python compilation and `git diff --check` passed.
- A live AiRi chat correctly used a synthetic saved preference for concise Python
  explanations. The smoke test used a temporary SQLite database, not personal memory.
- A live `/work` request completed all four Core agents with PASS on the first
  draft (zero revisions). The production revision limit remains three.
- Git ignore/tracking checks passed for the database and sidecars. A scan of all
  tracked/unignored files found no configured API key or likely API-key literal,
  and no personal database artifact. Core source and dependencies stayed unchanged.
- Persona and memory-context delivery are tested; broader persona consistency and
  retrieval quality across many conversations have not been evaluated.

Checked on 2026-10-08:

- All 18 offline unittest tests passed, including the forced Reviewer revision path and the three-revision limit.
- Python compilation and Git whitespace checks passed.
- A live smoke test using the configured environment key completed Planner -> Researcher -> Builder -> Reviewer with PASS on the first draft. Task: exactly two short beginner bullet points about Python lists, under 60 words. The smoke test limited revisions to one; the production default remains three.
- The live test confirms API integration for the configured account. Forced FAIL/revision cases are tested offline for reproducibility.
