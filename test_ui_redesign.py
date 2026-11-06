"""UI behavior checks with synthetic notes and a scripted client; no live API."""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from airi import AiRi
from memory import MemoryStore
from test_workflow import response, scripted_client
from tools import default_tools
from ui_components import system_state


class RedesignTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(Path(directory.name) / "notes.sqlite3")
        self.client = scripted_client()
        self.assistant = AiRi(self.store, client=self.client, tools=default_tools())
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
        self.app.session_state["airi"] = self.assistant
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)

    def click(self, key):
        self.app.button(key=key).click().run()
        self.assertEqual(len(self.app.exception), 0)

    def test_quick_actions_use_local_command_once_and_do_not_call_api(self):
        self.store.save("Synthetic note")
        self.click("show-memories")
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "#1: Synthetic note")
        self.app.run()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.click("show-help")
        self.assertIn("/remember", self.app.session_state["messages"][-1]["content"])
        self.client.responses.create.assert_not_called()

    def test_memory_form_saves_only_on_explicit_submit(self):
        self.app.text_area(key="memory-note").set_value("Synthetic preference").run()
        self.assertEqual(self.store.list_memories(), [])
        next(b for b in self.app.button if b.label == "บันทึก Memory").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual([n["content"] for n in self.store.list_memories()], ["Synthetic preference"])
        self.app.run()
        self.assertEqual(len(self.store.list_memories()), 1)
        self.client.responses.create.assert_not_called()

    def test_starters_disabled_without_key_and_welcome_is_not_model_context(self):
        self.assertTrue(self.app.button(key="starter-plan").disabled)
        self.assertTrue(self.app.button(key="starter-calculate").disabled)
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.assistant.history, [])
        self.client.responses.create.assert_not_called()

    def test_starter_dispatches_one_chat_turn_and_rerun_does_not_repeat(self):
        os.environ["OPENAI_API_KEY"] = "synthetic-placeholder"
        self.client.responses.create.side_effect = [response("Synthetic plan")]
        self.app.run()
        self.click("starter-plan")
        self.assertEqual(self.client.responses.create.call_count, 1)
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "Synthetic plan")
        self.app.run()
        self.assertEqual(self.client.responses.create.call_count, 1)

    def test_dashboard_only_reads_metadata_and_handles_memory_failure(self):
        from streamlit_app import dashboard_snapshot
        self.store.save("Synthetic private note")
        snapshot = dashboard_snapshot(self.assistant)
        self.assertEqual(snapshot["memory_count"], 1)
        self.assertEqual(snapshot["tool_names"], ("calculate", "date_math"))
        self.assertNotIn("Synthetic private note", str(snapshot))
        with patch.object(self.store, "list_memories", side_effect=sqlite3.OperationalError("private-path")):
            self.assertIsNone(dashboard_snapshot(self.assistant)["memory_count"])

    def test_reset_clears_history_and_pending_but_preserves_saved_notes(self):
        self.store.save("Synthetic note")
        self.assistant.handle("/forget 1")
        self.click("show-pending")
        self.click("new-conversation")
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(self.assistant.gate.pending(), [])
        self.assertEqual(len(self.store.list_memories()), 1)

    def test_normal_chat_without_environment_key_still_uses_injected_api_client(self):
        self.client.responses.create.side_effect = [response("Normal API reply")]
        self.app.chat_input[0].set_value("Hello").run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.client.responses.create.call_count, 1)
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "Normal API reply")

    def test_long_messages_are_kept_complete_as_plain_text(self):
        text = "x" * 4000 + "\nsecond line"
        self.client.responses.create.side_effect = [response(text)]
        self.app.chat_input[0].set_value("y" * 4000).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn(text, [t.value for t in self.app.text])


class OfflineDemoTests(unittest.TestCase):
    def setUp(self):
        from streamlit_app import CALCULATION_PROMPT
        self.calculation_prompt = CALCULATION_PROMPT
        self.factory = Mock(side_effect=AssertionError("offline demo must not create an API client"))
        for active in (patch.dict(os.environ, {"VINFREY_DEMO": "1", "OPENAI_API_KEY": ""}),
                       patch("api.create_client", self.factory),
                       patch("persona.load_profile", side_effect=AssertionError("no private profile"))):
            active.start()
            self.addCleanup(active.stop)
        self.app = AppTest.from_file(str(Path(__file__).with_name("streamlit_app.py")))
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assistant = self.app.session_state["airi"]
        self.addCleanup(self.app.session_state["demo_cleanup"])

    def send(self, message):
        self.app.chat_input[0].set_value(message).run()
        self.assertEqual(len(self.app.exception), 0)
        return self.app.session_state["messages"][-1]

    def test_chat_work_and_suggest_stop_before_api_and_keep_model_context_empty(self):
        for key in ("", "   "):
            os.environ["OPENAI_API_KEY"] = key
            for message in ("Hello", "/work Make a plan", "/suggest"):
                with self.subTest(message=message, blank_key=repr(key)):
                    reply = self.send(message)
                    self.assertTrue(reply["demo_notice"])
                    self.assertNotIn("error", reply)
                    self.assertIn("ยังไม่ได้ตั้ง API key", reply["content"])
        self.assertEqual(self.assistant.history, [])
        self.assertIsNone(self.assistant.last_work)
        self.assertEqual(self.assistant.usage.snapshot()["requests"], 0)
        self.assertEqual(len(self.app.status), 0)  # no fake agent execution
        self.factory.assert_not_called()

    def test_calculate_starter_runs_real_local_tool_once_without_api(self):
        with patch.object(self.assistant.tools, "run", wraps=self.assistant.tools.run) as run:
            self.assertFalse(self.app.button(key="starter-calculate").disabled)
            self.app.button(key="starter-calculate").click().run()
            self.assertEqual(len(self.app.exception), 0)
            reply = self.app.session_state["messages"][-1]
            self.assertTrue(reply["local_tool"])
            self.assertIn("12 × 7 = 84", reply["content"])
            self.assertIn("ไม่ได้เรียก AI หรือ API", reply["content"])
            run.assert_called_once_with("calculate", '{"expression": "12 * 7"}')
            self.app.run()
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.assistant.usage.snapshot()["requests"], 0)
        self.assertEqual(self.assistant.history, [])
        self.factory.assert_not_called()

    def test_chat_calculation_uses_same_example_but_arbitrary_text_is_not_inferred(self):
        self.assertTrue(self.send(self.calculation_prompt)["local_tool"])
        self.assertTrue(self.send("ลองคำนวณ 12 × 7")["local_tool"])
        self.assertTrue(self.send("Please calculate a different expression")["demo_notice"])
        self.factory.assert_not_called()

    def test_nonlocal_starter_shows_friendly_limit_without_api(self):
        self.app.button(key="starter-plan").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(self.app.session_state["messages"][-1]["demo_notice"])
        self.factory.assert_not_called()

    def test_missing_or_failed_local_tool_does_not_fabricate_a_result(self):
        self.assistant.tools = None
        self.assertTrue(self.send(self.calculation_prompt)["demo_notice"])
        self.assistant.tools = default_tools()
        with patch.object(self.assistant.tools, "run", return_value='{"ok": false, "error": "tool_failed"}'):
            reply = self.send(self.calculation_prompt)
        self.assertTrue(reply["demo_notice"])
        self.assertNotIn("= 84", reply["content"])
        self.factory.assert_not_called()

    def test_key_added_then_removed_does_not_reuse_cached_client_offline(self):
        client = scripted_client(response("Live demo API reply"))
        self.factory.side_effect = None
        self.factory.return_value = client
        os.environ["OPENAI_API_KEY"] = "synthetic-placeholder"
        self.assertEqual(self.send("Hello")["content"], "Live demo API reply")
        self.assertEqual(client.responses.create.call_count, 1)
        os.environ.pop("OPENAI_API_KEY", None)
        self.assertTrue(self.send("Hello again")["demo_notice"])
        self.assertEqual(client.responses.create.call_count, 1)
        self.assertEqual(self.assistant.usage.snapshot()["requests"], 1)

    def test_queued_prompt_checks_key_at_dispatch_and_memory_commands_still_work(self):
        self.app.session_state["ui_command"] = "ช่วยวางแผนการเรียนสำหรับสัปดาห์นี้"
        self.app.run()
        self.assertTrue(self.app.session_state["messages"][-1]["demo_notice"])
        self.send("/remember Synthetic demo note")
        self.send("/forget 1")
        self.app.button(key="approve-A1").click().run()
        self.assertEqual(self.assistant.memory.list_memories(), [])
        self.assertEqual(self.assistant.usage.snapshot()["requests"], 0)
        self.factory.assert_not_called()

    def test_offline_banner_does_not_claim_live_api_activity(self):
        self.assertTrue(any("OFFLINE DEMO" in i.value for i in self.app.sidebar.info))
        self.assertFalse(any("LIVE API DEMO" in w.value for w in self.app.sidebar.warning))


class ObservedStatusTests(unittest.TestCase):
    def state(self, **changes):
        values = {"api_configured": False, "memory_available": True, "messages": []}
        values.update(changes)
        return system_state(**values)[0]

    def test_key_configuration_is_not_reported_as_verified_ready(self):
        self.assertEqual(self.state(), "Local controls")
        self.assertEqual(self.state(offline_demo=True), "Offline demo")
        self.assertEqual(self.state(api_configured=True), "Configured")
        self.assertEqual(self.state(api_configured=True, messages=[
            {"role": "assistant", "content": "Saved memory #1.", "plain": True}]), "Configured")
        self.assertEqual(self.state(api_configured=True, messages=[
            {"role": "assistant", "content": "Synthetic reply"}]), "Ready")

    def test_error_limit_and_processing_are_observed_states(self):
        self.assertEqual(self.state(memory_available=False), "Memory unavailable")
        self.assertEqual(self.state(messages=[{"role": "assistant", "error": True}]), "Error")
        self.assertEqual(self.state(usage={"remaining": 0}), "Limit reached")
        self.assertEqual(self.state(busy=True), "Processing")


if __name__ == "__main__":
    unittest.main()
