"""Offline UI integration tests using Streamlit's real AppTest runtime."""

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

import persona
import streamlit_app
from airi import AiRi
from api import AgentError
from memory import MemoryStore
from persona import PERSONA
from test_tools import reasoning, reply, tool_call
from test_workflow import response, review, scripted_client

APP = Path(__file__).with_name("streamlit_app.py")
NORMAL_MEMORY_CAPTION = "Memory ใช้ SQLite ในเครื่องร่วมกับ CLI และทุกแท็บของ UI"
DEMO_MEMORY_CAPTION = ("โหมดสาธิต: Memory เป็นไฟล์ชั่วคราวเฉพาะ session นี้ แยกจากแท็บอื่น "
                       "ไม่ใช้ฐานข้อมูล Memory ส่วนตัว · ลบเมื่อระบบเก็บกวาด session นี้หรือเมื่อปิด "
                       "server ตามปกติ อาจไม่ลบทันทีที่ปิดแท็บ")


def setUpModule():
    # Never read the real local profile: point the default at an empty temp folder.
    global _profile_directory, _profile_patch
    _profile_directory = tempfile.TemporaryDirectory()
    _profile_patch = patch("persona.DEFAULT_PROFILE_PATH",
                           Path(_profile_directory.name) / "profile.toml")
    _profile_patch.start()


def tearDownModule():
    _profile_patch.stop()
    _profile_directory.cleanup()


class ToolEntryPointTests(unittest.TestCase):
    """P0-5: the real UI entry point enables only the approved pure tools."""

    SECRET = "ENC-SECRET-VALUE"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(Path(directory.name) / "notes.sqlite3")

    def start(self, client=None):
        app = AppTest.from_file(str(APP), default_timeout=10)
        patches = [patch("memory.MemoryStore", return_value=self.store)]
        if client is not None:
            patches.append(patch("api.create_client", return_value=client))
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_ui_entry_point_enables_tools(self):
        app = self.start()
        self.assertEqual(app.session_state["airi"].tools.names(), ("calculate", "date_math"))

    def test_ui_tool_turn_shows_only_final_plain_text(self):
        client = scripted_client(
            reply(reasoning(self.SECRET), tool_call("c1", arguments='{"expression": "12*7"}')),
            reply(text="84"))
        app = self.start(client)
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            app.chat_input[0].set_value("What is 12*7?").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(client.responses.create.call_count, 2)
        self.assertEqual(app.session_state["messages"][-1], {"role": "assistant", "content": "84"})
        self.assertIn("84", [t.value for t in app.text])
        shown = [e.value for kind in ("text", "markdown", "code", "caption", "error", "warning")
                 for e in getattr(app, kind)]
        for hidden in (self.SECRET, "function_call", '"ok"'):
            self.assertFalse(any(hidden in str(value) for value in shown), hidden)
        self.assertNotIn(self.SECRET, json.dumps(app.session_state["airi"].history))
        self.assertNotIn(self.SECRET, out.getvalue() + err.getvalue())


def usage_reply(text, input_tokens, output_tokens):
    return SimpleNamespace(status="completed", output=[], output_text=text,
                           usage=SimpleNamespace(input_tokens=input_tokens,
                                                 output_tokens=output_tokens))


class UsageEntryPointTests(unittest.TestCase):
    """P0-7: each browser session gets its own request guard; usage is shown in the sidebar."""

    USAGE_CAPTION = "การใช้ API ใน session นี้"

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(Path(directory.name) / "notes.sqlite3")
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("VINFREY_DEMO", None)

    def start(self, client=None, **patches):
        factory = Mock(return_value=client) if client is not None else Mock(
            side_effect=AssertionError("no API client should be created"))
        active = [patch("memory.MemoryStore", return_value=self.store),
                  patch("api.create_client", factory)]
        active += [patch(target, value) for target, value in patches.items()]
        for item in active:
            item.start()
            self.addCleanup(item.stop)
        app = AppTest.from_file(str(APP), default_timeout=10)
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app, factory

    def send(self, app, message):
        with contextlib.redirect_stdout(io.StringIO()):
            app.chat_input[0].set_value(message).run()
        self.assertEqual(len(app.exception), 0)

    def usage_caption(self, app):
        return [c.value for c in app.sidebar.caption if c.value.startswith(self.USAGE_CAPTION)]

    def test_ui_routes_chat_and_work_through_guard_and_shows_usage(self):
        client = scripted_client(usage_reply("Hi", 120, 30), response("Plan"),
                                 response("Research"), response("Draft"), review("PASS", "Ready"))
        app, _ = self.start(client)
        self.assertEqual(app.session_state["airi"].usage.snapshot()["limit"], 40)
        self.assertEqual(self.usage_caption(app),
                         ["การใช้ API ใน session นี้: 0/40 requests · input 0 / output 0 tokens"])
        self.send(app, "Hello")
        self.send(app, "/work Write a guide")
        self.assertEqual(client.responses.create.call_count, 5)
        self.assertEqual(self.usage_caption(app), [
            "การใช้ API ใน session นี้: 5/40 requests · input 120 / output 30 tokens"
            " · ไม่มีข้อมูล usage 4 ครั้ง"])
        captions = [c.value for c in app.sidebar.caption]
        self.assertIn("เพดานนับจำนวน request ต่อ session นี้เท่านั้น ไม่ใช่เพดานค่าใช้จ่าย", captions)
        # Both locations must include this run's requests, without waiting for another rerun.
        self.assertEqual({c.value for c in app.caption if c.value.startswith(self.USAGE_CAPTION)},
                         set(self.usage_caption(app)))

    def test_ui_limit_reached_message(self):
        app, factory = self.start(**{"usage.NORMAL_REQUEST_LIMIT": 0})
        self.send(app, "Hello")
        message = app.session_state["messages"][-1]
        self.assertTrue(message["error"])
        self.assertTrue(message["content"].startswith("ถึงเพดานการใช้ API ของ session นี้ · "))
        self.assertIn("needs up to 4, 0 left", message["content"])
        factory.assert_not_called()

    def test_ui_usage_command_is_local_plain_text(self):
        app, factory = self.start(**{"usage.NORMAL_REQUEST_LIMIT": 0})
        self.send(app, "/usage")
        self.assertEqual(app.session_state["messages"][-1], {
            "role": "assistant", "plain": True,
            "content": "API usage this session: 0/0 requests · 0 input / 0 output tokens"})
        factory.assert_not_called()

    def test_ui_normal_memory_caption_is_unchanged(self):
        app, _ = self.start()
        captions = [c.value for c in app.sidebar.caption]
        self.assertIn(NORMAL_MEMORY_CAPTION, captions)
        self.assertNotIn(DEMO_MEMORY_CAPTION, captions)

    def test_ui_new_conversation_keeps_usage(self):
        app, _ = self.start(scripted_client(usage_reply("Hi", 10, 5)))
        self.send(app, "Hello")
        app.sidebar.button[0].click().run()
        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(app.session_state["airi"].usage.snapshot()["requests"], 1)


class LiveDemoUITests(unittest.TestCase):
    """P0-7: VINFREY_DEMO=1 in the Web UI — labeled live demo, temporary per-session memory."""

    PRIVATE_NAME = "Zephyrine"

    def setUp(self):
        environment = patch.dict(os.environ, {"VINFREY_DEMO": "1",
                                              "OPENAI_API_KEY": "synthetic-placeholder"})
        environment.start()
        self.addCleanup(environment.stop)
        persona.DEFAULT_PROFILE_PATH.write_text(f'user_name = "{self.PRIVATE_NAME}"',
                                                encoding="utf-8")
        self.addCleanup(persona.DEFAULT_PROFILE_PATH.unlink)
        self.stores = Mock(wraps=MemoryStore)
        self.factory = Mock(side_effect=AssertionError("no API client should be created"))
        for item in (patch("memory.MemoryStore", self.stores),
                     patch("persona.load_profile",
                           side_effect=AssertionError("profile must not be read")),
                     patch("api.create_client", self.factory)):
            item.start()
            self.addCleanup(item.stop)

    def start(self):
        app = AppTest.from_file(str(APP), default_timeout=10)
        app.run()
        self.assertEqual(len(app.exception), 0)
        cleanup = app.session_state["demo_cleanup"]
        self.addCleanup(cleanup)  # never leave demo files behind, even if a test fails
        return app

    def send(self, app, message):
        with contextlib.redirect_stdout(io.StringIO()):
            app.chat_input[0].set_value(message).run()
        self.assertEqual(len(app.exception), 0)

    def test_ui_demo_banner_temp_memory_and_limit(self):
        app = self.start()
        banner = " ".join(w.value for w in app.sidebar.warning)
        self.assertIn("LIVE API DEMO", banner)
        self.assertIn("อาจมีค่าใช้จ่าย", banner)
        self.assertIn("ไม่ใช่เพดานค่าใช้จ่าย", banner)
        self.assertIn("อาจไม่ลบทันทีที่ปิดแท็บ", banner)
        sidebar_text = " ".join([banner] + [c.value for c in app.sidebar.caption])
        self.assertNotIn("ถูกลบเมื่อจบ session", sidebar_text)  # would imply deletion on tab close
        assistant = app.session_state["airi"]
        self.assertEqual(assistant.usage.snapshot()["limit"], 20)
        self.assertEqual(assistant.persona, PERSONA)
        path = Path(self.stores.call_args.args[0])
        self.assertEqual(self.stores.call_count, 1)
        self.assertTrue(path.parent.name.startswith("vinfrey-demo-"))
        self.assertNotIn("local_data", path.parts)
        self.assertEqual(assistant.memory.path, path)

    def test_ui_demo_memory_caption_matches_demo_mode(self):
        app = self.start()
        captions = [c.value for c in app.sidebar.caption]
        self.assertIn(DEMO_MEMORY_CAPTION, captions)
        self.assertNotIn(NORMAL_MEMORY_CAPTION, captions)

    def test_ui_demo_prompts_contain_no_private_profile_data(self):
        client = scripted_client(response("Hi"))
        self.factory.side_effect = None
        self.factory.return_value = client
        app = self.start()
        self.send(app, "Hello")
        request = client.responses.create.call_args.kwargs
        self.assertNotIn(self.PRIVATE_NAME, json.dumps(request, ensure_ascii=False))
        self.assertEqual(request["instructions"], PERSONA)

    def test_ui_demo_sessions_do_not_share_memory(self):
        first, second = self.start(), self.start()
        self.send(first, "/remember Demo note")
        self.send(second, "/memories")
        self.assertEqual(second.session_state["messages"][-1]["content"], "No memories saved.")
        self.assertNotEqual(first.session_state["airi"].memory.path,
                            second.session_state["airi"].memory.path)

    def test_ui_demo_cleanup_is_tied_to_the_session(self):
        app = self.start()
        cleanup = app.session_state["demo_cleanup"]
        directory = Path(app.session_state["airi"].memory.path).parent
        self.send(app, "/remember Demo note")
        self.assertTrue(cleanup.alive)
        self.assertIs(cleanup.peek()[0], app.session_state["airi"])
        cleanup()
        self.assertFalse(directory.exists())
        self.assertFalse(cleanup.alive)


class WebUITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(Path(directory.name) / "notes.sqlite3")
        self.client = scripted_client()
        self.assistant = AiRi(self.store, client=self.client, model="test-model")
        self.app = AppTest.from_file(str(APP), default_timeout=10)
        self.app.session_state["airi"] = self.assistant
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)

    def send(self, message):
        with contextlib.redirect_stdout(io.StringIO()):
            self.app.chat_input[0].set_value(message).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_chat_persona_history_and_rerun_without_duplicate_api_calls(self):
        self.client.responses.create.side_effect = [response("ลองเริ่มจากตัวอย่างสั้น ๆ"), response("ได้เลย")]
        self.send("Python เริ่มยังไงดี")
        self.app.run()
        self.assertEqual(self.client.responses.create.call_count, 1)
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertIs(self.app.session_state["airi"], self.assistant)
        self.send("ขอตัวอย่างต่อ")
        call = self.client.responses.create.call_args
        self.assertEqual(call.kwargs["instructions"], PERSONA)
        self.assertEqual(json.loads(call.kwargs["input"])["conversation"][0]["user"], "Python เริ่มยังไงดี")
        self.assertEqual(self.store.list_memories(), [])

    def test_explicit_memory_commands_and_matching_chat_context(self):
        self.send("/remember Python ชอบตัวอย่างสั้น ๆ")
        self.app.run()
        self.assertEqual(len(self.store.list_memories()), 1)
        self.send("/memories")
        self.assertIn("Python ชอบตัวอย่างสั้น ๆ", self.app.session_state["messages"][-1]["content"])
        self.send("/help")
        self.client.responses.create.assert_not_called()
        self.client.responses.create.side_effect = [response("ใช้ list สั้น ๆ กัน")]
        self.send("Python")
        context = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(context["memories"][0]["content"], "Python ชอบตัวอย่างสั้น ๆ")
        self.assertEqual(len(self.store.list_memories()), 1)

    def test_work_runs_core_retry_and_renders_structured_result(self):
        self.store.save("Synthetic private note")
        self.client.responses.create.side_effect = [
            response("Plan"), response("Research"), response("First draft"),
            review("FAIL", "Add example"), response("Revised draft"), review("PASS", "Ready"),
        ]
        self.send("/work Write a guide")
        outcome = self.app.session_state["messages"][-1]["work"]
        self.assertEqual(outcome, {"result": "Revised draft", "review": "PASS", "feedback": "Ready", "retries": 1})
        self.assertEqual(len(self.app.success), 1)
        self.assertTrue(any(m.value == "Revised draft" for m in self.app.code))
        self.assertTrue(any(m.value == "Ready" for m in self.app.code))
        self.app.run()
        self.assertEqual(self.client.responses.create.call_count, 6)
        self.assertEqual(self.assistant.history, [])
        for call in self.client.responses.create.call_args_list:
            self.assertNotIn("Synthetic private note", call.kwargs["input"])
            self.assertNotEqual(call.kwargs["instructions"], PERSONA)

    def test_exhausted_work_keeps_draft_and_feedback_visible(self):
        outputs = [response("Plan"), response("Research")]
        for _ in range(4):
            outputs.extend([response("Needs human review"), review("FAIL", "Missing example")])
        self.client.responses.create.side_effect = outputs
        self.send("/work Write a guide")
        outcome = self.app.session_state["messages"][-1]["work"]
        self.assertEqual(outcome["review"], "FAIL")
        self.assertEqual(outcome["retries"], 3)
        self.assertEqual(len(self.app.warning), 1)
        self.assertTrue(any(m.value == "Missing example" for m in self.app.code))

    def test_validation_api_database_and_unexpected_errors_allow_recovery(self):
        for command in ("/remember", "/work", "/unknown", "/memories extra"):
            self.send(command)
            self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.client.responses.create.assert_not_called()
        with patch("airi.generate", side_effect=AgentError("API unavailable")):
            self.send("Hello")
        self.assertIn("API unavailable", self.app.error[-1].value)
        self.assertEqual(self.assistant.history, [])
        with patch.object(self.store, "save", side_effect=sqlite3.OperationalError("private-db-path")):
            self.send("/remember note")
        self.assertNotIn("private-db-path", self.app.error[-1].value)
        with patch("airi.generate", side_effect=RuntimeError("synthetic-private-secret")):
            self.send("Hello again")
        self.assertNotIn("synthetic-private-secret", self.app.error[-1].value)
        self.client.responses.create.side_effect = [response("Recovered")]
        self.send("Try again")
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "Recovered")

    def test_new_conversation_clears_ui_and_model_history_but_preserves_notes(self):
        self.send("/remember Keep this note")
        self.client.responses.create.side_effect = [response("Hello")]
        self.send("Hello")
        self.app.button(key="new-conversation").click().run()
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.assistant.history, [])
        self.assertEqual(len(self.store.list_memories()), 1)
        self.assertEqual(len(self.app.chat_message), 0)

    def test_new_browser_session_has_separate_history_and_shared_local_notes(self):
        self.send("/remember Shared synthetic note")
        second = AppTest.from_file(str(APP))
        with patch("memory.MemoryStore", return_value=MemoryStore(self.store.path)):
            second.run()
        self.assertEqual(len(second.exception), 0)
        self.assertEqual(second.session_state["messages"], [])
        self.assertEqual(second.session_state["airi"].history, [])
        second.chat_input[0].set_value("/memories").run()
        self.assertIn("Shared synthetic note", second.session_state["messages"][-1]["content"])

    def test_missing_key_is_safe_and_memory_still_works(self):
        self.assistant.client = None
        with patch.dict(os.environ, {}, clear=True):
            self.send("/remember Offline note")
            self.send("Hello")
        self.assertIn("OPENAI_API_KEY", self.app.error[-1].value)
        self.assertEqual(self.assistant.history, [])
        self.assertEqual(len(self.store.list_memories()), 1)

    def test_local_command_replies_render_as_multiline_text(self):
        self.send("/remember A")
        self.send("/help")
        self.assertIs(self.app.session_state["messages"][-1]["plain"], True)
        self.assertTrue(any("\n" in t.value and "/work" in t.value for t in self.app.text))
        self.client.responses.create.assert_not_called()

    def test_chat_reply_is_markdown_not_plain(self):
        self.client.responses.create.side_effect = [response("Hi")]
        self.send("Hello")
        self.assertNotIn("plain", self.app.session_state["messages"][-1])

    def write_profile(self, content):
        persona.DEFAULT_PROFILE_PATH.write_text(content, encoding="utf-8")
        self.addCleanup(persona.DEFAULT_PROFILE_PATH.unlink)
        app = AppTest.from_file(str(APP))
        with patch("memory.MemoryStore", return_value=self.store):
            app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_ui_builds_persona_from_profile(self):
        app = self.write_profile('user_name = "Mali"')
        self.assertIn("partner is Mali.", app.session_state["airi"].persona)
        self.assertEqual(len(app.warning), 0)

    def test_ui_warns_on_invalid_profile(self):
        app = self.write_profile("user_name = ")
        self.assertTrue(any("local profile" in w.value for w in app.warning))
        self.assertEqual(app.session_state["airi"].persona, PERSONA)

    def test_status_line_mapping(self):
        cases = [
            ({"type": "agent_finished", "agent": "Planner", "content": "x"}, "Planner ✓"),
            ({"type": "agent_finished", "agent": "Researcher", "content": "x"}, "Researcher ✓"),
            ({"type": "agent_started", "agent": "Builder", "draft": 2}, "Builder draft 2…"),
            ({"type": "agent_finished", "agent": "Reviewer", "draft": 1, "content": "x",
              "status": "FAIL", "feedback": "x"}, "Reviewer: FAIL"),
            ({"type": "revision", "revision": 1, "max_revisions": 3}, "FAIL → แก้ไขรอบ 1/3"),
            ({"type": "finished", "review": "PASS", "retries": 1, "max_retries": 3},
             "เสร็จสิ้น · PASS · แก้ไข 1/3 รอบ"),
            ({"type": "agent_started", "agent": "Planner"}, None),
            ({"type": "agent_finished", "agent": "Builder", "draft": 1, "content": "x"}, None),
        ]
        for event, expected in cases:
            with self.subTest(event=event):
                self.assertEqual(streamlit_app.status_line(event), expected)

    def work_fail_then_pass(self):
        self.client.responses.create.side_effect = [
            response("Plan"), response("Research"), response("First draft"),
            review("FAIL", "Add example"), response("Revised draft"), review("PASS", "Ready"),
        ]
        self.send("/work Write a guide")
        return self.app.session_state["messages"][-1]

    def test_sidebar_discloses_date_and_recent_work(self):
        captions = [c.value for c in self.app.sidebar.caption]
        self.assertIn("แชตส่งข้อความ ประวัติล่าสุด Memory ที่ตรงคำค้น วันเวลาปัจจุบัน "
                      "และสรุปผล /work ล่าสุดที่กรองแล้วไปยัง OpenAI", captions)
        self.assertIn("/work ส่งโจทย์เข้า Core โดยไม่แนบ Memory หรือประวัติแชต", captions)

    def test_ui_work_then_chat_sends_bounded_recent_work(self):
        self.work_fail_then_pass()
        self.client.responses.create.side_effect = [response("It is a guide.")]
        self.send("What did you make?")
        context = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(context["recent_work"], {"task": "Write a guide", "review": "PASS",
                                                  "result": "Revised draft", "truncated": False})
        self.assertEqual(set(context["now"]), {"date", "weekday", "time", "utc_offset"})
        for raw in ("Add example", "First draft", "Plan", "Research"):
            self.assertNotIn(raw, json.dumps(context["recent_work"]))
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "It is a guide.")

    def test_work_stores_trace_and_rerun_does_not_repeat_calls(self):
        message = self.work_fail_then_pass()
        self.assertEqual(message["work"], {"result": "Revised draft", "review": "PASS",
                                           "feedback": "Ready", "retries": 1})
        types = [event["type"] for event in message["trace"]]
        self.assertIn("revision", types)
        self.assertEqual(types[-1], "finished")
        self.assertEqual(self.app.status[0].state, "complete")
        self.app.run()
        self.assertEqual(self.client.responses.create.call_count, 6)
        self.assertEqual(self.store.list_memories(), [])

    def test_work_in_ui_does_not_print_to_server_terminal(self):
        self.client.responses.create.side_effect = [
            response("Plan"), response("Research"), response("Draft"), review("PASS", "Ready")]
        with contextlib.redirect_stdout(io.StringIO()) as console:
            self.app.chat_input[0].set_value("/work Write a guide").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertNotIn("=== Planner ===", console.getvalue())
        self.assertNotIn("Draft", console.getvalue())

    def test_trace_details_collapsed_after_rerun(self):
        self.work_fail_then_pass()
        self.app.run()
        labels = [expander.label for expander in self.app.expander]
        self.assertIn("รายละเอียดขั้นตอนของ Agents", labels)
        self.assertTrue(any(m.value == "First draft" for m in self.app.code))
        self.assertTrue(any("Reviewer: FAIL" in c.value for c in self.app.caption))

    def test_workflow_failure_marks_status_error_with_safe_message(self):
        self.client.responses.create.side_effect = [
            response("Plan"), RuntimeError("synthetic-private-secret")]
        self.send("/work Write a guide")
        self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.assertNotIn("synthetic-private-secret", self.app.error[-1].value)
        self.assertEqual(self.app.status[0].state, "error")

    def test_empty_work_does_not_leave_status_running(self):
        self.send("/work")
        self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.assertTrue(all(status.state != "running" for status in self.app.status))
        self.client.responses.create.assert_not_called()

    def forget_first_note(self):
        self.store.save("Private synthetic note")
        self.store.save("Other synthetic note")
        self.send("/forget 1")
        return self.app.button(key="approve-A1"), self.app.button(key="deny-A1")

    def notes(self):
        return [note["content"] for note in self.store.list_memories()]

    def click(self, button):
        with contextlib.redirect_stdout(io.StringIO()):
            button.click().run()
        self.assertEqual(len(self.app.exception), 0)

    def test_forget_shows_safe_pending_card(self):
        self.forget_first_note()
        card = " ".join(w.value for w in self.app.warning)
        self.assertIn("A1", card)
        self.assertIn("permanently", card)
        self.assertNotIn("Private synthetic note", card)
        self.assertEqual(len(self.notes()), 2)
        self.client.responses.create.assert_not_called()

    def test_approve_button_deletes_through_command_path(self):
        approve, _ = self.forget_first_note()
        self.click(approve)
        self.assertEqual(self.notes(), ["Other synthetic note"])
        messages = self.app.session_state["messages"]
        self.assertEqual(messages[-2]["content"], "/approve A1")
        self.assertIn("Approved A1", messages[-1]["content"])
        self.assertEqual(len([b for b in self.app.button if b.key == "approve-A1"]), 0)

    def test_deny_button_keeps_memory(self):
        _, deny = self.forget_first_note()
        self.click(deny)
        self.assertEqual(len(self.notes()), 2)
        self.assertIn("Denied A1", self.app.session_state["messages"][-1]["content"])

    def test_pending_survives_rerun_without_executing(self):
        self.forget_first_note()
        self.app.run()
        self.app.run()
        self.assertEqual(self.app.button(key="approve-A1").label, "อนุมัติ")
        self.assertEqual(len(self.notes()), 2)

    def test_repeated_approval_runs_once_with_safe_error(self):
        approve, _ = self.forget_first_note()
        self.click(approve)
        self.app.run()
        self.send("/approve A1")
        reply = self.app.session_state["messages"][-1]
        self.assertTrue(reply["error"])
        self.assertNotIn("Private synthetic note", reply["content"])
        self.assertEqual(self.notes(), ["Other synthetic note"])

    def test_new_conversation_clears_pending(self):
        self.forget_first_note()
        reset = next(b for b in self.app.button if b.label == "เริ่มบทสนทนาใหม่")
        self.click(reset)
        self.assertEqual(self.assistant.gate.pending(), [])
        self.assertEqual(len([b for b in self.app.button if b.key == "approve-A1"]), 0)
        self.assertEqual(len(self.notes()), 2)

    def test_sessions_do_not_share_pending(self):
        self.forget_first_note()
        second = AppTest.from_file(str(APP))
        with patch("memory.MemoryStore", return_value=MemoryStore(self.store.path)):
            second.run()
        self.assertEqual(len(second.exception), 0)
        self.assertEqual(second.session_state["airi"].gate.pending(), [])
        self.assertEqual(len([b for b in second.button if b.key == "approve-A1"]), 0)
        second.chat_input[0].set_value("/approve A1").run()
        self.assertTrue(second.session_state["messages"][-1]["error"])
        self.assertEqual(len(self.notes()), 2)

    def suggest(self, *items):
        self.client.responses.create.side_effect = [
            response("Nice"), response(json.dumps({"suggestions": [
                {"content": content, "reason": reason} for content, reason in items]}))]
        self.send("I like tea")
        self.send("/suggest")
        return self.app.session_state["messages"][-1]

    def rendered_rich_text(self):
        """Values of every element that renders Markdown/HTML-capable text."""
        groups = (self.app.markdown, self.app.warning, self.app.caption, self.app.info,
                  self.app.error, self.app.success)
        return [element.value for group in groups for element in group]

    def assertOnlyPlain(self, *raw):
        plain = " ".join(t.value for t in self.app.text)
        for text in raw:
            self.assertIn(text, plain)
            self.assertFalse(any(text in value for value in self.rendered_rich_text()), text)

    def test_suggest_ui_creates_card_with_plain_reply(self):
        reply = self.suggest(("Likes green tea", "User said so"))
        self.assertIs(reply["plain"], True)
        self.assertIn("Suggested 1 memory for approval: A1.", reply["content"])
        self.assertEqual(self.app.button(key="approve-A1").label, "อนุมัติ")
        self.assertOnlyPlain("Likes green tea", "User said so")
        self.assertEqual(self.store.list_memories(), [])

    def test_suggest_markdown_image_injection_is_plain_text(self):
        content = "![x](http://example.invalid/a.png) Likes tea"
        reason = "see [link](http://example.invalid/b)"
        self.suggest((content, reason))
        self.assertOnlyPlain(content, reason)
        self.send("/pending")
        self.assertOnlyPlain(content)

    def test_suggest_html_image_injection_is_plain_text(self):
        content = '<img src="http://example.invalid/c.png"> Likes tea'
        reason = "<b>bold</b> <script>x</script>"
        self.suggest((content, reason))
        self.assertOnlyPlain(content, reason)
        # After approval the stored text shows up again in /memories, still as plain text.
        self.click(self.app.button(key="approve-A1"))
        self.send("/memories")
        self.assertOnlyPlain(content)
        self.assertFalse(any("Likes tea" in value for value in self.rendered_rich_text()))

    def test_suggest_approve_button_saves_once(self):
        self.suggest(("Likes green tea", "User said so"))
        self.click(self.app.button(key="approve-A1"))
        self.assertEqual(self.notes(), ["Likes green tea"])
        self.assertIn("Approved A1: saved memory #", self.app.session_state["messages"][-1]["content"])
        self.send("/approve A1")
        self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.assertEqual(self.notes(), ["Likes green tea"])

    def test_suggest_deny_button_saves_nothing(self):
        self.suggest(("Likes green tea", "User said so"))
        self.click(self.app.button(key="deny-A1"))
        self.assertEqual(self.notes(), [])
        self.assertIn("Denied A1", self.app.session_state["messages"][-1]["content"])

    def test_suggest_card_survives_reruns_without_saving(self):
        self.suggest(("Likes green tea", "User said so"))
        self.app.run()
        self.app.run()
        self.assertEqual(self.app.button(key="approve-A1").label, "อนุมัติ")
        self.assertOnlyPlain("Likes green tea")
        self.assertEqual(self.notes(), [])
        self.assertEqual(self.client.responses.create.call_count, 2)

    def test_suggest_empty_result_is_safe(self):
        reply = self.suggest()
        self.assertEqual(reply["content"], "No suitable memories to suggest.")
        self.assertIs(reply["plain"], True)
        self.assertNotIn("error", reply)
        self.assertEqual([b for b in self.app.button if (b.key or "").startswith("approve-")], [])

    def test_suggest_without_history_is_safe_error(self):
        self.send("/suggest")
        reply = self.app.session_state["messages"][-1]
        self.assertTrue(reply["error"])
        self.assertIn("Chat with AiRi first", reply["content"])
        self.client.responses.create.assert_not_called()

    # P0-4.1 F1: model-written text never goes through a Markdown/HTML renderer.
    INJECTION = ('![x](http://example.invalid/a.png) <img src="http://example.invalid/b.png"> '
                 '[link](http://example.invalid/c) http://example.invalid/d')

    def assertOnlyCode(self, *raw):
        code = " ".join(c.value for c in self.app.code)
        for text in raw:
            self.assertIn(text, code)
            self.assertFalse(any(text in value for value in self.rendered_rich_text()), text)

    def test_chat_reply_injection_is_plain_text(self):
        self.client.responses.create.side_effect = [response(self.INJECTION)]
        self.send("Hello")
        self.assertOnlyPlain(self.INJECTION)
        self.app.run()
        self.assertOnlyPlain(self.INJECTION)

    def test_work_result_and_feedback_injection_is_plain_code(self):
        draft, feedback = "Draft " + self.INJECTION, "Feedback " + self.INJECTION
        self.client.responses.create.side_effect = [
            response("Plan"), response("Research"), response(draft), review("PASS", feedback)]
        self.send("/work Write a guide")
        self.assertOnlyCode(draft, feedback)
        self.assertEqual(len(self.app.success), 1)  # system label still rendered

    def test_trace_injection_is_plain_code(self):
        plan, research = "Plan " + self.INJECTION, "Research " + self.INJECTION
        self.client.responses.create.side_effect = [
            response(plan), response(research), response("Draft"), review("PASS", "Ready")]
        self.send("/work Write a guide")
        self.app.run()
        self.assertOnlyCode(plan, research)

    def test_user_message_rendering_unchanged(self):
        self.client.responses.create.side_effect = [response("Hi")]
        self.send("**bold request**")
        self.assertTrue(any(m.value == "**bold request**" for m in self.app.markdown))

    def test_forget_in_ui_keeps_transcript_but_clears_outbound_context(self):
        self.store.save("FORGET-ME-MARKER likes jazz")
        self.client.responses.create.side_effect = [
            response("You like jazz (FORGET-ME-MARKER)"), response("ok")]
        self.send("jazz")
        self.send("/forget 1")
        self.click(self.app.button(key="approve-A1"))
        self.assertEqual(self.store.list_memories(), [])
        transcript = " ".join(m.get("content", "") for m in self.app.session_state["messages"])
        self.assertIn("FORGET-ME-MARKER", transcript)  # the visible transcript is kept
        self.send("/suggest")
        self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.send("anything")
        self.assertNotIn("FORGET-ME-MARKER", self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(self.client.responses.create.call_count, 2)

    def test_database_initialization_failure_is_readable_without_traceback(self):
        app = AppTest.from_file(str(APP))
        with patch("memory.MemoryStore", side_effect=OSError("synthetic-private-path")):
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_input), 0)
        self.assertIn("local_data", app.error[0].value)
        self.assertNotIn("synthetic-private-path", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
