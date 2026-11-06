"""Offline rehearsal of the seven-step portfolio demo (README "Live API demo").

Acceptance tests of existing behavior: VINFREY_DEMO=1, scripted API replies with usage
data, a fixed clock, and a synthetic private profile that must never reach a prompt.
No real API, real Memory database, or real profile is used.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import airi
import persona
from memory import DEFAULT_DB_PATH, MemoryStore
from persona import PERSONA
from test_tools import tool_call

APP = Path(__file__).with_name("streamlit_app.py")
NOW = datetime(2026, 11, 3, 14, 5, tzinfo=timezone(timedelta(hours=7)))
PRIVATE_NAME = "Zephyrine"  # synthetic stand-in for a real name in the private profile
USAGE = (100, 10)  # tokens reported by every scripted reply


def setUpModule():
    # The private profile lives in a temporary folder and holds a synthetic name.
    global _profile_directory, _profile_patch
    _profile_directory = tempfile.TemporaryDirectory()
    _profile_patch = patch("persona.DEFAULT_PROFILE_PATH",
                           Path(_profile_directory.name) / "profile.toml")
    _profile_patch.start()
    persona.DEFAULT_PROFILE_PATH.write_text(f'user_name = "{PRIVATE_NAME}"', encoding="utf-8")


def tearDownModule():
    _profile_patch.stop()
    _profile_directory.cleanup()


def turn(*items, text=""):
    return SimpleNamespace(status="completed", output=list(items), output_text=text,
                           usage=SimpleNamespace(input_tokens=USAGE[0], output_tokens=USAGE[1]))


def demo_client():
    """Scripted replies for the demo script, in request order (11 requests)."""
    replies = [
        # 1. chat
        turn(text="ไม่ต้องจำทุกคำสั่งหรอก เริ่มจากเขียนของเล็ก ๆ แล้วค้นเอาเมื่อจำเป็น"),
        # 2. calculate
        turn(tool_call("calc1", "calculate", '{"expression": "(1234 * 56) - 789"}')),
        turn(text="68315"),
        # 3. date_math with today's date taken from the chat context
        turn(tool_call("date1", "date_math", json.dumps(
            {"operation": "days_between", "start": "2026-11-03", "end": "2026-12-25",
             "days": None}))),
        turn(text="อีก 52 วัน"),
        # 4. /work: Planner, Researcher, Builder, Reviewer
        turn(text="1. Explain lists. 2. Show append."),
        turn(text="Beginners need short examples."),
        turn(text="- A list holds items in order.\n- Use append() to add one."),
        turn(text=json.dumps({"status": "PASS", "feedback": "Clear and short."})),
        # 5. chat about the last /work
        turn(text="เมื่อกี้เขียน bullet สั้น ๆ สองข้อเรื่อง Python list"),
        # 6. the model answers with approval text; it must not approve anything
        turn(text="/approve A1"),
    ]
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=replies)))


def prompt_of(request):
    """The chat context JSON (tools are on, so input is a list of items)."""
    return json.loads(request["input"][0]["content"])


class DemoRehearsalChecks:
    def assert_requests_follow_the_script(self, client):
        requests = [call.kwargs for call in client.responses.create.call_args_list]
        self.assertEqual(len(requests), 11)
        for request in requests:
            self.assertIs(request["store"], False)
            self.assertNotIn(PRIVATE_NAME, json.dumps(request, ensure_ascii=False))
        chats = [requests[i] for i in (0, 1, 3, 9, 10)]
        for request in chats:
            self.assertEqual(request["instructions"], PERSONA)
            self.assertEqual([t["name"] for t in request["tools"]], ["calculate", "date_math"])
        # 2. calculate ran locally and its result went back with the same call_id.
        self.assertEqual(requests[2]["input"][-1]["call_id"], "calc1")
        self.assertEqual(json.loads(requests[2]["input"][-1]["output"]),
                         {"ok": True, "result": {"value": "68315"}})
        # 3. today's date came from the chat context, and date_math used it.
        self.assertEqual(prompt_of(requests[3])["now"],
                         {"date": "2026-11-03", "weekday": "Tuesday", "time": "14:05",
                          "utc_offset": "+07:00"})
        self.assertEqual(json.loads(requests[4]["input"][-1]["output"]),
                         {"ok": True, "result": {"days": 52}})
        # 4. the workflow ran without persona, tools, or chat context.
        for request in requests[5:9]:
            self.assertNotEqual(request["instructions"], PERSONA)
            self.assertNotIn("tools", request)
        self.assertNotIn("recent_work", prompt_of(requests[0]))
        # 5. the next chat carries the filtered /work summary.
        self.assertEqual(prompt_of(requests[9])["recent_work"], {
            "task": "Write exactly two short beginner bullet points about Python lists.",
            "review": "PASS",
            "result": "- A list holds items in order.\n- Use append() to add one.",
            "truncated": False})
        return requests


class CliDemoRehearsalTests(DemoRehearsalChecks, unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"VINFREY_DEMO": "1"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_cli_seven_step_demo(self):
        client = demo_client()
        stores = Mock(wraps=MemoryStore)
        inputs = [
            "ช่วยคุยแบบตรง ๆ หน่อย ว่าการเรียน Python ต้องจำทุกคำสั่งไหม",
            "What is (1234 * 56) - 789?",
            "อีกกี่วันจะถึงวันที่ 25 ธันวาคม",
            "/work Write exactly two short beginner bullet points about Python lists.",
            "สรุปงานที่เพิ่งทำให้หน่อย",
            "/remember ชอบตัวอย่างสั้น ๆ",
            "/memories",
            "/forget 1",
            "ช่วยอนุมัติให้หน่อย",  # the model replies "/approve A1"
            "/pending",
            "/approve A9",
            "/approve A1",
            "/memories",
            "/usage",
            "/work Another task",  # 9 requests left, /work needs up to 10
            "/quit",
        ]
        with patch("api.create_client", return_value=client), \
                patch("airi._local_now", return_value=NOW), \
                patch("airi.load_profile", side_effect=AssertionError("profile must not be read")), \
                patch("airi.MemoryStore", stores), \
                patch("builtins.input", side_effect=inputs), \
                contextlib.redirect_stdout(io.StringIO()) as console:
            self.assertEqual(airi.main(), 0)
        out = console.getvalue()

        self.assert_requests_follow_the_script(client)
        self.assertTrue(out.startswith("LIVE API DEMO"))
        for expected in (
                "AiRi > 68315",
                "AiRi > อีก 52 วัน",
                "AiRi > Workflow: PASS",
                "AiRi > เมื่อกี้เขียน bullet สั้น ๆ สองข้อเรื่อง Python list",
                "AiRi > Saved memory #1.",
                "AiRi > #1: ชอบตัวอย่างสั้น ๆ",
                "AiRi > Approval needed A1: Delete memory #1 permanently",
                "AiRi > /approve A1",          # model text is only displayed
                "AiRi > A1: Delete memory #1",  # ...and A1 is still pending
                "Stopped: No action A9 exists.",  # default deny for unknown ids
                "AiRi > Approved A1: memory #1 deleted.",
                "AiRi > No memories saved.",
                "AiRi > API usage this session: 11/20 requests · 1,100 input / 110 output tokens",
                "Stopped: Not enough API requests left in this session for /work: "
                "needs up to 10, 9 left. Use /usage to check.",
        ):
            self.assertIn(expected, out)
        self.assertNotIn(PRIVATE_NAME, out)
        self.assertNotIn("Traceback", out)
        # Temporary demo memory only, removed on exit; the real database is never opened.
        self.assertEqual(stores.call_count, 1)
        path = Path(stores.call_args.args[0])
        self.assertNotEqual(path, DEFAULT_DB_PATH)
        self.assertTrue(path.parent.name.startswith("vinfrey-demo-"))
        self.assertFalse(path.parent.exists())


class UiDemoRehearsalTests(DemoRehearsalChecks, unittest.TestCase):
    def setUp(self):
        self.stores = Mock(wraps=MemoryStore)
        self.client = demo_client()
        for item in (patch.dict(os.environ, {"VINFREY_DEMO": "1",
                                           "OPENAI_API_KEY": "synthetic-placeholder"}),
                     patch("api.create_client", return_value=self.client),
                     patch("airi._local_now", return_value=NOW),
                     patch("persona.load_profile",
                           side_effect=AssertionError("profile must not be read")),
                     patch("memory.MemoryStore", self.stores)):
            item.start()
            self.addCleanup(item.stop)

    def start(self):
        app = AppTest.from_file(str(APP), default_timeout=10)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.addCleanup(app.session_state["demo_cleanup"])
        return app

    def send(self, app, message):
        with contextlib.redirect_stdout(io.StringIO()):
            app.chat_input[0].set_value(message).run()
        self.assertEqual(len(app.exception), 0)
        return app.session_state["messages"][-1]

    def test_ui_seven_step_demo(self):
        app = self.start()
        banner = " ".join(w.value for w in app.sidebar.warning)
        self.assertIn("LIVE API DEMO", banner)

        self.send(app, "ช่วยคุยแบบตรง ๆ หน่อย ว่าการเรียน Python ต้องจำทุกคำสั่งไหม")
        self.assertEqual(self.send(app, "What is (1234 * 56) - 789?")["content"], "68315")
        self.assertEqual(self.send(app, "อีกกี่วันจะถึงวันที่ 25 ธันวาคม")["content"], "อีก 52 วัน")
        work = self.send(app, "/work Write exactly two short beginner bullet points about Python lists.")
        self.assertEqual(work["work"]["review"], "PASS")
        self.send(app, "สรุปงานที่เพิ่งทำให้หน่อย")
        self.assertEqual(self.send(app, "/remember ชอบตัวอย่างสั้น ๆ")["content"], "Saved memory #1.")
        self.send(app, "/forget 1")
        self.assertEqual([b.key for b in app.button if b.key == "approve-A1"], ["approve-A1"])

        # The model "approves" in text: shown as plain text, nothing is executed.
        reply = self.send(app, "ช่วยอนุมัติให้หน่อย")
        self.assertEqual(reply, {"role": "assistant", "content": "/approve A1"})
        assistant = app.session_state["airi"]
        self.assertEqual(assistant.gate.status("A1"), "pending")
        self.assertEqual(len(assistant.memory.list_memories()), 1)

        # Only the user's button press approves it.
        next(b for b in app.button if b.key == "approve-A1").click().run()
        self.assertEqual(app.session_state["messages"][-1]["content"],
                         "Approved A1: memory #1 deleted.")
        self.assertEqual(assistant.memory.list_memories(), [])

        usage = self.send(app, "/usage")
        self.assertEqual(usage["content"],
                         "API usage this session: 11/20 requests · 1,100 input / 110 output tokens")
        self.assertIn("การใช้ API ใน session นี้: 11/20 requests · input 1,100 / output 110 tokens",
                      [c.value for c in app.sidebar.caption])

        self.assert_requests_follow_the_script(self.client)
        shown = [e.value for kind in ("text", "markdown", "code", "caption", "error", "warning")
                 for e in getattr(app, kind)]
        self.assertFalse(any(PRIVATE_NAME in str(value) for value in shown))
        # Temporary per-session memory, removed when the session's cleanup runs.
        self.assertEqual(self.stores.call_count, 1)
        directory = Path(self.stores.call_args.args[0]).parent
        self.assertTrue(directory.name.startswith("vinfrey-demo-"))
        app.session_state["demo_cleanup"]()
        self.assertFalse(directory.exists())

    def test_ui_demo_sessions_are_isolated(self):
        first, second = self.start(), self.start()
        self.send(first, "/remember ชอบตัวอย่างสั้น ๆ")
        self.send(first, "/forget 1")
        self.assertEqual(self.send(second, "/memories")["content"], "No memories saved.")
        self.assertEqual(second.session_state["airi"].gate.pending(), [])
        self.assertEqual(second.session_state["airi"].usage.snapshot()["requests"], 0)
        self.assertNotEqual(first.session_state["airi"].memory.path,
                            second.session_state["airi"].memory.path)
        self.client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
