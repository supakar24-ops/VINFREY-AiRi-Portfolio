"""Offline tests: real agent functions, scripted Responses API results, no API key."""

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import (
    APIConnectionError, APIStatusError, APITimeoutError,
    AuthenticationError, OpenAIError, RateLimitError,
)

import agents
import api
import main


def response(text, status="completed", output=None):
    return SimpleNamespace(output_text=text, status=status, output=output or [])


def review(status, feedback):
    return response(json.dumps({"status": status, "feedback": feedback}))


def scripted_client(*outputs):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=outputs)))


class WorkflowTests(unittest.TestCase):
    def run_quietly(self, client, **options):
        with contextlib.redirect_stdout(io.StringIO()) as console:
            outcome = main.run_workflow("Write a beginner Python guide.", client=client, **options)
        return outcome, console.getvalue()

    def test_pass_on_first_draft_and_context(self):
        client = scripted_client(response("Plan"), response("Research"), response("Guide"),
                                 review("PASS", "Meets the request."))
        outcome, console = self.run_quietly(client)
        self.assertEqual(outcome, {"result": "Guide", "review": "PASS",
                                  "feedback": "Meets the request.", "retries": 0})
        self.assertIn("PASS -> Task completed", console)
        self.assertEqual(client.responses.create.call_count, 4)
        calls = client.responses.create.call_args_list
        self.assertIn("Planner Agent", calls[0].kwargs["instructions"])
        self.assertIn("Researcher Agent", calls[1].kwargs["instructions"])
        self.assertIn("Builder Agent", calls[2].kwargs["instructions"])
        self.assertIn("Reviewer Agent", calls[3].kwargs["instructions"])
        build_input = json.loads(calls[2].kwargs["input"])
        self.assertEqual(build_input["plan"], "Plan")
        self.assertEqual(build_input["research"], "Research")
        self.assertEqual(build_input["request"], "Write a beginner Python guide.")
        self.assertEqual(build_input["previous_draft"], "")
        self.assertEqual(build_input["reviewer_feedback"], "")
        self.assertEqual(json.loads(calls[3].kwargs["input"])["draft"], "Guide")
        self.assertTrue(calls[3].kwargs["text"]["format"]["strict"])

    def test_fail_sends_feedback_and_previous_draft_to_builder(self):
        client = scripted_client(response("Plan"), response("Research"), response("Draft one"),
                                 review("FAIL", "Add an example."), response("Draft two with example"),
                                 review("PASS", "Example included."))
        outcome, console = self.run_quietly(client)
        self.assertEqual(outcome["review"], "PASS")
        self.assertEqual(outcome["retries"], 1)
        self.assertEqual(outcome["result"], "Draft two with example")
        self.assertIn("Builder revision 1/3", console)
        calls = client.responses.create.call_args_list
        revised = json.loads(calls[4].kwargs["input"])
        self.assertEqual(revised["reviewer_feedback"], "Add an example.")
        self.assertEqual(revised["previous_draft"], "Draft one")
        self.assertEqual(revised["plan"], "Plan")
        self.assertEqual(revised["research"], "Research")
        self.assertEqual(json.loads(calls[5].kwargs["input"])["draft"], outcome["result"])

    def test_repeated_fail_stops_after_three_revisions(self):
        outputs = [response("Plan"), response("Research")]
        for draft in range(4):
            outputs.extend([response(f"Draft {draft}"), review("FAIL", f"Fix draft {draft}")])
        client = scripted_client(*outputs)
        outcome, console = self.run_quietly(client)
        self.assertEqual(outcome["retries"], 3)
        self.assertEqual(outcome["review"], "FAIL")
        self.assertEqual(outcome["result"], "Draft 3")
        self.assertEqual(client.responses.create.call_count, 10)
        self.assertIn("STOP -> Human review required", console)
        for index in (4, 6, 8):
            data = json.loads(client.responses.create.call_args_list[index].kwargs["input"])
            previous = (index - 4) // 2
            self.assertEqual(data["previous_draft"], f"Draft {previous}")
            self.assertEqual(data["reviewer_feedback"], f"Fix draft {previous}")

    def test_pass_on_last_allowed_revision(self):
        client = scripted_client(
            response("Plan"), response("Research"),
            response("D0"), review("FAIL", "Fix 0"),
            response("D1"), review("FAIL", "Fix 1"),
            response("D2"), review("FAIL", "Fix 2"),
            response("D3"), review("PASS", "Ready"),
        )
        outcome, _ = self.run_quietly(client)
        self.assertEqual((outcome["review"], outcome["retries"]), ("PASS", 3))

    def test_zero_retries(self):
        client = scripted_client(response("Plan"), response("Research"), response("Draft"),
                                 review("FAIL", "Fix it"))
        outcome, _ = self.run_quietly(client, max_retries=0)
        self.assertEqual(outcome["retries"], 0)
        self.assertEqual(client.responses.create.call_count, 4)

    def test_invalid_input_never_calls_api(self):
        client = scripted_client()
        for task in ("", "   ", None):
            with self.subTest(task=task), self.assertRaises(ValueError):
                main.run_workflow(task, client=client)
        for retries in (-1, 1.5, True):
            with self.subTest(retries=retries), self.assertRaises(ValueError):
                main.run_workflow("Task", client=client, max_retries=retries)
        client.responses.create.assert_not_called()

    def test_invalid_review_stops_workflow_without_revision_or_success(self):
        client = scripted_client(response("Plan"), response("Research"), response("Draft"),
                                 response("Not JSON"))
        with contextlib.redirect_stdout(io.StringIO()) as console:
            with self.assertRaises(api.AgentError):
                main.run_workflow("Task", client=client)
        self.assertEqual(client.responses.create.call_count, 4)
        self.assertNotIn("Task completed", console.getvalue())

    def test_api_error_stops_downstream_agents(self):
        client = scripted_client(response("Plan"), APIConnectionError(request=Mock()))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(api.AgentError, "Researcher: cannot connect"):
                main.run_workflow("Task", client=client)
        self.assertEqual(client.responses.create.call_count, 2)


class ResponseTests(unittest.TestCase):
    def test_invalid_reviewer_outputs_are_not_approved(self):
        invalid = ["not json", "[]", "null", '{"status": "PASS"}',
                   '{"status": "MAYBE", "feedback": "OK"}',
                   '{"status": "PASS", "feedback": "  "}',
                   '{"status": "PASS", "feedback": 42}',
                   '{"status": "PASS", "feedback": "OK", "extra": true}']
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(api.AgentError):
                agents.reviewer("Draft", client=scripted_client(response(output)))

    def test_empty_incomplete_failed_and_refused_responses(self):
        refusal = SimpleNamespace(content=[SimpleNamespace(type="refusal")])
        invalid = [response(""), response("   "), response(None),
                   response("partial", status="incomplete"), response("text", status="failed"),
                   response("text", output=[refusal])]
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(api.AgentError):
                api.generate("Builder", "Instructions", "Task", client=scripted_client(output))

    def test_api_errors_are_sanitized(self):
        request = Mock()
        errors = [
            (AuthenticationError("sensitive", response=Mock(status_code=401), body=None), "authentication"),
            (RateLimitError("sensitive", response=Mock(status_code=429), body=None), "quota"),
            (APITimeoutError(request=request), "timed out"),
            (APIConnectionError(message="sensitive", request=request), "cannot connect"),
            (APIStatusError("sensitive", response=Mock(status_code=500), body=None), "HTTP 500"),
            (OpenAIError("sensitive"), "request failed"),
        ]
        for error, expected in errors:
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(api.AgentError) as caught:
                    api.generate("Planner", "Instructions", "Task", client=scripted_client(error))
                self.assertIn(expected, str(caught.exception))
                self.assertNotIn("sensitive", str(caught.exception))

    def test_model_configuration_and_override(self):
        client = scripted_client(response("A"), response("B"), response("C"))
        with patch.dict(os.environ, {}, clear=True):
            api.generate("Planner", "I", "T", client=client)
        with patch.dict(os.environ, {"OPENAI_MODEL": "configured-model"}):
            api.generate("Planner", "I", "T", client=client)
            api.generate("Planner", "I", "T", client=client, model="override-model")
        models = [call.kwargs["model"] for call in client.responses.create.call_args_list]
        self.assertEqual(models, ["gpt-6-luna", "configured-model", "override-model"])

    def test_blank_model_rejected(self):
        client = scripted_client()
        with patch.dict(os.environ, {"OPENAI_MODEL": "   "}):
            with self.assertRaises(api.AgentError):
                api.generate("Planner", "I", "T", client=client)
        client.responses.create.assert_not_called()

    def test_missing_key_has_clear_message(self):
        with patch.dict(os.environ, {}, clear=True), patch("api.OpenAI") as constructor:
            with self.assertRaisesRegex(api.AgentError, "OPENAI_API_KEY"):
                api.create_client()
        constructor.assert_not_called()

    def test_client_has_bounded_network_retries_and_timeout(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}):
            with patch("api.OpenAI") as constructor:
                api.create_client()
        constructor.assert_called_once_with(timeout=60.0, max_retries=2)


class StorageTests(unittest.TestCase):
    """Every request asks the provider not to store the response (store=False)."""

    def assert_stateless(self, calls):
        self.assertTrue(calls)
        for call in calls:
            self.assertIs(call.kwargs["store"], False)
            self.assertNotIn("previous_response_id", call.kwargs)
            self.assertNotIn("tools", call.kwargs)

    def test_workflow_requests_disable_storage(self):
        client = scripted_client(*pass_outputs())
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_workflow("Task", client=client)
        calls = client.responses.create.call_args_list
        self.assertEqual(len(calls), 4)
        self.assert_stateless(calls)
        self.assertTrue(calls[3].kwargs["text"]["format"]["strict"])

    def test_workflow_revision_requests_disable_storage(self):
        client = scripted_client(*fail_then_pass_outputs())
        with contextlib.redirect_stdout(io.StringIO()):
            main.run_workflow("Task", client=client)
        self.assertEqual(client.responses.create.call_count, 6)
        self.assert_stateless(client.responses.create.call_args_list)

    def test_chat_and_suggest_requests_disable_storage(self):
        import airi
        from memory import MemoryStore
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        client = scripted_client(response("Nice"), response(json.dumps({"suggestions": []})))
        assistant = airi.AiRi(MemoryStore(Path(directory.name) / "notes.sqlite3"), client=client)
        assistant.handle("I like Python")
        assistant.handle("/suggest")
        self.assertEqual(client.responses.create.call_count, 2)
        self.assert_stateless(client.responses.create.call_args_list)

    def test_generate_payload_keys_unchanged_except_store(self):
        fmt = {"type": "json_schema", "name": "x", "schema": {}, "strict": True}
        for text_format, expected in ((None, {"model", "instructions", "input", "store"}),
                                      (fmt, {"model", "instructions", "input", "store", "text"})):
            with self.subTest(text_format=text_format):
                client = scripted_client(response("  Done \n"))
                self.assertEqual(api.generate("Builder", "I", "T", client=client, model="m",
                                              text_format=text_format), "Done")
                self.assertEqual(set(client.responses.create.call_args.kwargs), expected)


class CliTests(unittest.TestCase):
    def test_import_does_not_prompt_or_initialize_client(self):
        with patch("builtins.input") as prompt, patch("api.OpenAI") as constructor:
            importlib.reload(main)
        prompt.assert_not_called()
        constructor.assert_not_called()

    def test_cli_error_no_traceback_or_success(self):
        with patch("builtins.input", return_value="Task"):
            with patch("main.run_workflow", side_effect=api.AgentError("Planner: timed out")):
                with contextlib.redirect_stdout(io.StringIO()) as console:
                    code = main.main()
        self.assertEqual(code, 1)
        self.assertIn("Planner: timed out", console.getvalue())
        self.assertNotIn("Traceback", console.getvalue())
        self.assertNotIn("Task completed", console.getvalue())

    def test_cli_exit_codes(self):
        for status, expected in (("PASS", 0), ("FAIL", 1)):
            with self.subTest(status=status), patch("builtins.input", return_value="Task"):
                with patch("main.run_workflow", return_value={"review": status}):
                    self.assertEqual(main.main(), expected)
        for error in (KeyboardInterrupt(), EOFError()):
            with patch("builtins.input", side_effect=error):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main.main(), 130)


# Exact console output recorded from run_workflow before the on_event hook existed.
_START = ("\n[Planner] Creating a plan...\n\n=== Planner ===\nPlan\n"
          "\n[Researcher] Preparing background...\n\n=== Researcher ===\nResearch\n")


def _draft(number, text, status, feedback):
    return (f"\n[Builder] Draft {number}...\n\n=== Builder - draft {number} ===\n{text}\n"
            f"\n[Reviewer] Checking the draft...\n\n=== Reviewer ===\n{status}\n"
            f"Feedback: {feedback}\n")


def _retry(number, limit=3):
    return f"\n=== Retry ===\nFAIL -> Builder revision {number}/{limit}\n"


def _final(message, retries, limit=3):
    return f"\n=== Final Status ===\n{message}\nBuilder revisions: {retries}/{limit}\n"


GOLDEN_PASS = (_START + _draft(1, "Guide", "PASS", "Meets the request.")
               + _final("PASS -> Task completed", 0))
GOLDEN_FAIL_THEN_PASS = (_START + _draft(1, "Draft one", "FAIL", "Add an example.") + _retry(1)
                         + _draft(2, "Draft two with example", "PASS", "Example included.")
                         + _final("PASS -> Task completed", 1))
GOLDEN_EXHAUSTED = (_START
                    + "".join(_draft(n + 1, f"Draft {n}", "FAIL", f"Fix draft {n}") + _retry(n + 1)
                              for n in range(3))
                    + _draft(4, "Draft 3", "FAIL", "Fix draft 3")
                    + _final("STOP -> Human review required", 3))
GOLDEN_ZERO_RETRIES = (_START + _draft(1, "Draft", "FAIL", "Fix it")
                       + _final("STOP -> Human review required", 0, limit=0))


def pass_outputs():
    return [response("Plan"), response("Research"), response("Guide"),
            review("PASS", "Meets the request.")]


def fail_then_pass_outputs():
    return [response("Plan"), response("Research"), response("Draft one"),
            review("FAIL", "Add an example."), response("Draft two with example"),
            review("PASS", "Example included.")]


def exhausted_outputs():
    outputs = [response("Plan"), response("Research")]
    for draft in range(4):
        outputs.extend([response(f"Draft {draft}"), review("FAIL", f"Fix draft {draft}")])
    return outputs


def zero_retry_outputs():
    return [response("Plan"), response("Research"), response("Draft"), review("FAIL", "Fix it")]


class GoldenOutputTests(unittest.TestCase):
    """The CLI must print exactly what it printed before workflow events existed."""

    def console(self, outputs, **options):
        with contextlib.redirect_stdout(io.StringIO()) as console:
            main.run_workflow("Write a beginner Python guide.",
                              client=scripted_client(*outputs), **options)
        return console.getvalue()

    def test_pass_output_unchanged(self):
        self.assertEqual(self.console(pass_outputs()), GOLDEN_PASS)

    def test_fail_then_pass_output_unchanged(self):
        self.assertEqual(self.console(fail_then_pass_outputs()), GOLDEN_FAIL_THEN_PASS)

    def test_exhausted_output_unchanged(self):
        self.assertEqual(self.console(exhausted_outputs()), GOLDEN_EXHAUSTED)

    def test_zero_retries_output_unchanged(self):
        self.assertEqual(self.console(zero_retry_outputs(), max_retries=0), GOLDEN_ZERO_RETRIES)


class EventTests(unittest.TestCase):
    def collect(self, outputs, **options):
        events = []
        client = scripted_client(*outputs)
        with contextlib.redirect_stdout(io.StringIO()) as console:
            outcome = main.run_workflow("Write a beginner Python guide.", client=client,
                                        on_event=events.append, **options)
        return outcome, events, console.getvalue(), client

    def test_pass_event_sequence(self):
        outcome, events, _, _ = self.collect(pass_outputs())
        self.assertEqual(outcome, {"result": "Guide", "review": "PASS",
                                   "feedback": "Meets the request.", "retries": 0})
        self.assertEqual(events, [
            {"type": "agent_started", "agent": "Planner"},
            {"type": "agent_finished", "agent": "Planner", "content": "Plan"},
            {"type": "agent_started", "agent": "Researcher"},
            {"type": "agent_finished", "agent": "Researcher", "content": "Research"},
            {"type": "agent_started", "agent": "Builder", "draft": 1},
            {"type": "agent_finished", "agent": "Builder", "draft": 1, "content": "Guide"},
            {"type": "agent_started", "agent": "Reviewer", "draft": 1},
            {"type": "agent_finished", "agent": "Reviewer", "draft": 1,
             "content": "PASS\nFeedback: Meets the request.",
             "status": "PASS", "feedback": "Meets the request."},
            {"type": "finished", "review": "PASS", "retries": 0, "max_retries": 3},
        ])

    def test_fail_then_pass_includes_revision_event(self):
        _, events, _, _ = self.collect(fail_then_pass_outputs())
        revision = {"type": "revision", "revision": 1, "max_revisions": 3}
        index = events.index(revision)
        self.assertEqual(events[index - 1]["agent"], "Reviewer")
        self.assertEqual(events[index - 1]["draft"], 1)
        self.assertEqual(events[index + 1], {"type": "agent_started", "agent": "Builder", "draft": 2})
        self.assertEqual(events[-1], {"type": "finished", "review": "PASS", "retries": 1,
                                      "max_retries": 3})

    def test_exhausted_and_zero_retries_events(self):
        _, events, _, _ = self.collect(exhausted_outputs())
        self.assertEqual([e["revision"] for e in events if e["type"] == "revision"], [1, 2, 3])
        self.assertEqual(events[-1], {"type": "finished", "review": "FAIL", "retries": 3,
                                      "max_retries": 3})
        _, events, _, _ = self.collect(zero_retry_outputs(), max_retries=0)
        self.assertFalse(any(e["type"] == "revision" for e in events))
        self.assertEqual(events[-1], {"type": "finished", "review": "FAIL", "retries": 0,
                                      "max_retries": 0})

    def test_hook_replaces_printing(self):
        _, events, console, _ = self.collect(fail_then_pass_outputs())
        self.assertTrue(events)
        self.assertEqual(console, "")

    def test_handler_error_propagates_and_stops(self):
        client = scripted_client(*pass_outputs())

        def handler(event):
            if event["type"] == "agent_finished" and event["agent"] == "Planner":
                raise RuntimeError("handler failed")

        with self.assertRaises(RuntimeError):
            main.run_workflow("Task", client=client, on_event=handler)
        self.assertEqual(client.responses.create.call_count, 1)

    def test_invalid_input_emits_no_events(self):
        events = []
        with self.assertRaises(ValueError):
            main.run_workflow("", client=scripted_client(), on_event=events.append)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
