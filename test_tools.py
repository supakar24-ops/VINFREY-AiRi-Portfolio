"""Offline tests for the pure computation tools and the stateless tool loop."""

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import RateLimitError
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseFunctionWebSearch, ResponseOutputMessage,
    ResponseOutputRefusal, ResponseOutputText, ResponseReasoningItem,
)

import api
import tools
from test_workflow import response, scripted_client


def run(name, arguments, registry=None):
    registry = registry or tools.default_tools()
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return json.loads(registry.run(name, raw))


def calc(expression):
    return run("calculate", {"expression": expression})


def dates(operation, start, end=None, days=None):
    return run("date_math", {"operation": operation, "start": start, "end": end, "days": days})


def schema(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def fake_tool(name="fake", handler=None, **options):
    return tools.Tool(name=name, description="Test tool.",
                      parameters=options.pop("parameters", schema(
                          text={"type": "string", "maxLength": 20},
                          count={"type": ["integer", "null"]},
                          mode={"type": "string", "enum": ["a", "b"]})),
                      side_effects=options.pop("side_effects", "none"),
                      handler=handler or Mock(return_value="done"), **options)


class RegistryTests(unittest.TestCase):
    def test_default_registry_is_exact_allowlist(self):
        registry = tools.default_tools()
        self.assertEqual(registry.names(), ("calculate", "date_math"))
        for spec in registry.specs():
            self.assertEqual(spec["parameters"]["additionalProperties"], False)
        self.assertTrue(all(t.side_effects == "none" for t in registry._tools.values()))

    def test_register_rejects_duplicates_side_effects_and_bad_schema(self):
        loose = schema(text={"type": "string"})
        loose["additionalProperties"] = True
        partial = schema(text={"type": "string"}, other={"type": "string"})
        partial["required"] = ["text"]
        cases = {
            "side effect": fake_tool(side_effects="write"),
            "bad name": fake_tool(name="Bad-Name"),
            "open schema": fake_tool(parameters=loose),
            "missing required": fake_tool(parameters=partial),
            "unsupported type": fake_tool(parameters=schema(value={"type": "number"})),
        }
        for label, tool in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                tools.ToolRegistry().register(tool)
        registry = tools.ToolRegistry()
        registry.register(fake_tool())
        with self.assertRaises(ValueError):
            registry.register(fake_tool())
        self.assertEqual(registry.names(), ("fake",))

    def test_specs_are_strict_function_tools_and_copies(self):
        registry = tools.default_tools()
        specs = registry.specs()
        for spec in specs:
            self.assertEqual(spec["type"], "function")
            self.assertIs(spec["strict"], True)
            self.assertEqual(set(spec), {"type", "name", "description", "parameters", "strict"})
        specs[0]["parameters"]["properties"].clear()
        self.assertTrue(registry.specs()[0]["parameters"]["properties"])

    def test_unknown_tool_is_denied_without_running(self):
        self.assertEqual(run("shell", {}), {"ok": False, "error": "unknown_tool"})
        handler = Mock()
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=handler))
        self.assertEqual(run("Fake", {}, registry)["error"], "unknown_tool")
        handler.assert_not_called()

    def test_invalid_arguments_are_denied(self):
        handler = Mock(return_value="done")
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=handler))
        good = {"text": "hi", "count": None, "mode": "a"}
        self.assertEqual(run("fake", good, registry), {"ok": True, "result": "done"})
        handler.assert_called_once_with(**good)
        handler.reset_mock()
        cases = {
            "not a string": None,
            "too long": json.dumps(good) + " " * 1000,
            "bad json": "{",
            "array": "[]",
            "duplicate key": '{"text": "a", "text": "b", "count": 1, "mode": "a"}',
            "NaN": '{"text": "a", "count": NaN, "mode": "a"}',
            "extra key": json.dumps({**good, "extra": 1}),
            "missing key": json.dumps({"text": "hi", "mode": "a"}),
            "wrong type": json.dumps({**good, "text": 5}),
            "bool for integer": json.dumps({**good, "count": True}),
            "float for integer": json.dumps({**good, "count": 1.0}),
            "too long value": json.dumps({**good, "text": "x" * 21}),
            "not in enum": json.dumps({**good, "mode": "c"}),
        }
        for label, raw in cases.items():
            with self.subTest(label):
                result = json.loads(registry.run("fake", raw))
                self.assertEqual(result, {"ok": False, "error": "invalid_arguments"})
        handler.assert_not_called()

    def test_handler_errors_are_safe(self):
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=Mock(side_effect=RuntimeError("secret-detail"))))
        raw = registry.run("fake", json.dumps({"text": "a", "count": 1, "mode": "a"}))
        self.assertEqual(json.loads(raw), {"ok": False, "error": "tool_failed"})
        self.assertNotIn("secret-detail", raw)

    def test_tool_error_codes_are_fixed(self):
        with self.assertRaises(ValueError):
            tools.ToolError("free text from a handler")
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=Mock(side_effect=tools.ToolError("invalid_date"))))
        raw = registry.run("fake", json.dumps({"text": "a", "count": 1, "mode": "a"}))
        self.assertEqual(json.loads(raw), {"ok": False, "error": "invalid_date"})

    def test_oversized_output_is_rejected(self):
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=Mock(return_value="x" * 2000)))
        raw = registry.run("fake", json.dumps({"text": "a", "count": 1, "mode": "a"}))
        self.assertEqual(json.loads(raw), {"ok": False, "error": "output_too_large"})

    def test_tools_module_imports_only_pure_modules(self):
        tree = ast.parse(Path(tools.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertLessEqual(imported, {"ast", "dataclasses", "datetime", "json", "math", "re", "types"})
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {"eval", "exec", "compile", "open", "__import__", "input"})

    def test_tools_make_no_io_or_persistent_calls(self):
        guards = [patch(target, Mock(side_effect=AssertionError(target))) for target in
                  ("builtins.open", "socket.socket", "subprocess.Popen", "os.system")]
        mocks = [guard.start() for guard in guards]
        try:
            self.assertEqual(calc("(1+2)*3"), {"ok": True, "result": {"value": "9"}})
            self.assertEqual(dates("add_days", "2026-10-31", days=5),
                             {"ok": True, "result": {"date": "2026-11-05"}})
            self.assertEqual(calc("1/0")["error"], "division_by_zero")
        finally:
            for guard in guards:
                guard.stop()
        for mock in mocks:
            mock.assert_not_called()


class CalculateTests(unittest.TestCase):
    def test_calculate_basic_arithmetic(self):
        cases = {"1+2*3": "7", "(1+2)*3": "9", "7//2": "3", "7%3": "1", "1/4": "0.25",
                 "-3**2": "-9", "2**10": "1024", "2**-1": "0.5", "4/2": "2", " 10 - 2.5 ": "7.5",
                 "1000000000000000": "1000000000000000", "10**15": "1000000000000000"}
        for expression, expected in cases.items():
            with self.subTest(expression):
                self.assertEqual(calc(expression), {"ok": True, "result": {"value": expected}})

    def test_calculate_rejects_non_arithmetic(self):
        cases = ["__import__('os')", "abs(1)", "x+1", "1 if 1 else 2", "[1]", "1e5", "1_000",
                 "True", "1j", "", "   ", "1+", "1\n+1", "１+1", "(" * 150 + "1" + ")" * 49,
                 "2**0.5", "1..2"]
        for expression in cases:
            with self.subTest(expression=expression):
                self.assertEqual(calc(expression), {"ok": False, "error": "invalid_expression"})
        self.assertEqual(calc("1" * 201), {"ok": False, "error": "invalid_arguments"})

    def test_calculate_rejects_chained_and_large_powers_before_computing(self):
        cases = ["2**3**4", "(2**3)**4", "2**(1+3**2)", "2**2*2**2*2**2", "9**99", "10**16",
                 "0.001**-64", "(-10)**17"]
        for expression in cases:
            with self.subTest(expression=expression), patch("tools._power") as power:
                self.assertEqual(calc(expression), {"ok": False, "error": "result_too_large"})
                power.assert_not_called()

    def test_calculate_bounds_operands_and_intermediates(self):
        for expression in ("1000000000000000+1", "9999999*9999999*99", "1234567890123456",
                           "999999999999999.9*10", "-1000000000000000-1", "0-1000000000000001"):
            with self.subTest(expression=expression):
                self.assertEqual(calc(expression), {"ok": False, "error": "result_too_large"})

    def test_calculate_rejects_division_by_zero(self):
        for expression in ("1/0", "1//0", "1%0", "0**-1", "5/(2-2)", "1/0.0"):
            with self.subTest(expression=expression):
                self.assertEqual(calc(expression), {"ok": False, "error": "division_by_zero"})

    def test_calculate_limits_node_count_and_depth(self):
        def balanced(leaves):  # shallow tree: tests the node count, not the depth
            if leaves == 1:
                return "1"
            return f"({balanced(leaves // 2)}+{balanced(leaves - leaves // 2)})"

        too_deep = "(1+" * 25 + "1" + ")" * 25  # 51 nodes, nested 25 levels
        for expression in (balanced(40), too_deep, "-" * 25 + "1"):
            with self.subTest(expression=expression[:12]):
                self.assertEqual(calc(expression), {"ok": False, "error": "invalid_expression"})
        self.assertEqual(calc(balanced(32)), {"ok": True, "result": {"value": "32"}})
        self.assertEqual(calc("(" * 30 + "1" + ")" * 30), {"ok": True, "result": {"value": "1"}})


class DateMathTests(unittest.TestCase):
    def test_days_between(self):
        self.assertEqual(dates("days_between", "2026-10-31", "2026-11-05"),
                         {"ok": True, "result": {"days": 5}})
        self.assertEqual(dates("days_between", "2026-11-05", "2026-10-31"),
                         {"ok": True, "result": {"days": -5}})

    def test_add_days(self):
        for start, days, expected in (("2026-10-31", 5, "2026-11-05"), ("2026-10-31", -1, "2026-10-30"),
                                      ("2028-02-28", 1, "2028-02-29"), ("2026-01-01", 0, "2026-01-01")):
            with self.subTest(start=start, days=days):
                self.assertEqual(dates("add_days", start, days=days),
                                 {"ok": True, "result": {"date": expected}})

    def test_date_math_rejects_invalid_input(self):
        cases = [
            (("days_between", "2026-02-30", "2026-03-01"), "invalid_date"),
            (("days_between", "26-10-31", "2026-03-01"), "invalid_date"),
            (("add_days", "2026/10/31", None, 1), "invalid_date"),
            (("add_days", "0000-01-01", None, 1), "invalid_date"),
            (("add_days", "２026-10-31", None, 1), "invalid_date"),
            (("days_between", "2026-10-31", None, None), "invalid_arguments"),
            (("days_between", "2026-10-31", "2026-11-01", 3), "invalid_arguments"),
            (("add_days", "2026-10-31", None, None), "invalid_arguments"),
            (("add_days", "2026-10-31", "2026-11-01", 3), "invalid_arguments"),
            (("add_days", "2026-10-31", None, 36501), "out_of_range"),
            (("add_days", "9999-12-31", None, 1), "out_of_range"),
            (("add_days", "0001-01-01", None, -1), "out_of_range"),
        ]
        for arguments, code in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(dates(*arguments), {"ok": False, "error": code})
        self.assertEqual(run("date_math", {"operation": "today", "start": "2026-10-31",
                                           "end": None, "days": None})["error"], "invalid_arguments")


def reply(*items, text="", status="completed"):
    return SimpleNamespace(status=status, output=list(items), output_text=text)


def tool_call(call_id, name="calculate", arguments='{"expression": "1+1"}'):
    return ResponseFunctionToolCall(type="function_call", call_id=call_id, name=name,
                                    arguments=arguments, id=f"fc_{call_id}", status="completed")


def reasoning(secret="ENC"):
    return ResponseReasoningItem(type="reasoning", id="rs_1", summary=[], encrypted_content=secret)


def message(text):
    return ResponseOutputMessage(type="message", id="msg_1", role="assistant", status="completed",
                                 content=[ResponseOutputText(type="output_text", text=text,
                                                             annotations=[])])


USER = {"role": "user", "content": "Prompt"}
CALL_C1 = {"arguments": '{"expression": "1+1"}', "call_id": "c1", "name": "calculate",
           "type": "function_call", "id": "fc_c1", "status": "completed"}
TWO = '{"ok": true, "result": {"value": "2"}}'


class ToolLoopTests(unittest.TestCase):
    def setUp(self):
        self.registry = tools.default_tools()

    def loop(self, *outputs, registry=None):
        self.client = scripted_client(*outputs)
        return api.generate_with_tools("AiRi", "Persona", "Prompt", tools=registry or self.registry,
                                       client=self.client, model="test-model")

    def requests(self):
        return [call.kwargs for call in self.client.responses.create.call_args_list]

    def test_text_only_reply_makes_one_stateless_request(self):
        self.assertEqual(self.loop(reply(message("Hi"), text="  Hi \n")), "Hi")
        [request] = self.requests()
        self.assertEqual(set(request), {"model", "instructions", "input", "tools", "tool_choice",
                                        "parallel_tool_calls", "store"})
        self.assertEqual(request["input"], [USER])
        self.assertEqual(request["tools"], self.registry.specs())
        self.assertEqual(request["tool_choice"], "auto")
        self.assertIs(request["parallel_tool_calls"], False)
        self.assertIs(request["store"], False)

    def test_tool_round_preserves_items_order_and_call_id(self):
        result = self.loop(reply(reasoning(), tool_call("c1")), reply(message("2"), text="2"))
        self.assertEqual(result, "2")
        first, second = self.requests()
        self.assertEqual(first["input"], [USER])  # the first request was not mutated later
        self.assertEqual(second["input"], [
            USER,
            {"id": "rs_1", "summary": [], "type": "reasoning", "encrypted_content": "ENC"},
            CALL_C1,
            {"type": "function_call_output", "call_id": "c1", "output": TWO},
        ])
        self.assertIs(second["store"], False)
        self.assertNotIn("previous_response_id", second)

    def test_two_sequential_calls_keep_linkage(self):
        self.loop(reply(tool_call("c1")),
                  reply(tool_call("c2", arguments='{"expression": "6*7"}')),
                  reply(text="42"))
        items = self.requests()[2]["input"]
        self.assertEqual([item.get("call_id") for item in items], [None, "c1", "c1", "c2", "c2"])
        self.assertEqual([item.get("type") for item in items[1:]],
                         ["function_call", "function_call_output"] * 2)
        self.assertEqual(json.loads(items[4]["output"])["result"], {"value": "42"})

    def test_message_items_between_calls_are_preserved(self):
        self.loop(reply(message("Let me check."), tool_call("c1")), reply(text="2"))
        items = self.requests()[1]["input"]
        self.assertEqual([item.get("type") for item in items],
                         [None, "message", "function_call", "function_call_output"])
        self.assertEqual(items[1]["content"][0]["text"], "Let me check.")

    def test_unknown_tool_returns_error_output_and_continues(self):
        result = self.loop(reply(tool_call("c1", name="shell", arguments="{}")), reply(text="Sorry"))
        self.assertEqual(result, "Sorry")
        self.assertEqual(self.requests()[1]["input"][-1],
                         {"type": "function_call_output", "call_id": "c1",
                          "output": '{"ok": false, "error": "unknown_tool"}'})

    def test_invalid_arguments_are_returned_without_running(self):
        handler = Mock(return_value="done")
        registry = tools.ToolRegistry()
        registry.register(fake_tool(handler=handler))
        self.loop(reply(tool_call("c1", name="fake", arguments='{"text": 1}')), reply(text="ok"),
                  registry=registry)
        handler.assert_not_called()
        self.assertIn("invalid_arguments", self.requests()[1]["input"][-1]["output"])

    def test_failed_tool_and_oversized_output_are_returned_safely(self):
        registry = tools.ToolRegistry()
        registry.register(fake_tool("boom", handler=Mock(side_effect=RuntimeError("secret-detail"))))
        registry.register(fake_tool("big", handler=Mock(return_value="x" * 2000)))
        args = json.dumps({"text": "a", "count": 1, "mode": "a"})
        result = self.loop(reply(tool_call("c1", "boom", args), tool_call("c2", "big", args)),
                           reply(text="Done"), registry=registry)
        self.assertEqual(result, "Done")
        outputs = [json.loads(i["output"]) for i in self.requests()[1]["input"][-2:]]
        self.assertEqual(outputs, [{"ok": False, "error": "tool_failed"},
                                   {"ok": False, "error": "output_too_large"}])
        self.assertNotIn("secret-detail", json.dumps(self.requests()))

    def test_round_limit_fails_closed(self):
        rounds = [reply(tool_call(f"c{n}")) for n in range(api.MAX_TOOL_ROUNDS + 1)]
        with patch.object(self.registry, "run", wraps=self.registry.run) as run:
            with self.assertRaisesRegex(api.AgentError, "tool call limit"):
                self.loop(*rounds)
        self.assertEqual(len(self.requests()), api.MAX_TOOL_ROUNDS + 1)
        self.assertEqual(run.call_count, api.MAX_TOOL_ROUNDS)

    def test_call_limit_fails_closed(self):
        cases = {
            "five in one round": ([reply(*[tool_call(f"c{n}") for n in range(5)])], 0),
            "five across rounds": ([reply(*[tool_call(f"a{n}") for n in range(3)]),
                                    reply(*[tool_call(f"b{n}") for n in range(2)])], 3),
        }
        for label, (rounds, expected_runs) in cases.items():
            with self.subTest(label), patch.object(self.registry, "run",
                                                   wraps=self.registry.run) as run:
                with self.assertRaisesRegex(api.AgentError, "tool call limit"):
                    self.loop(*rounds)
                self.assertEqual(run.call_count, expected_runs)

    def test_malformed_tool_responses_fail_closed(self):
        construct = ResponseFunctionToolCall.model_construct
        cases = {
            "missing call_id": [reply(construct(type="function_call", name="calculate",
                                                arguments="{}"))],
            "empty name": [reply(tool_call("c1", name=""))],
            "arguments not text": [reply(construct(type="function_call", call_id="c1",
                                                   name="calculate", arguments={"x": 1}))],
            "incomplete call": [reply(construct(type="function_call", call_id="c1",
                                                name="calculate", arguments="{}",
                                                status="incomplete"))],
            "duplicate in round": [reply(tool_call("c1"), tool_call("c1"))],
            "duplicate across rounds": [reply(tool_call("c1")), reply(tool_call("c1"))],
            # A real, serializable SDK item: only the type allowlist can reject it.
            "unknown item type": [reply(ResponseFunctionWebSearch.model_construct(
                type="web_search_call", id="ws_1", status="completed"))],
            "not serializable": [reply(SimpleNamespace(type="function_call", call_id="c1",
                                                       name="calculate", arguments="{}"))],
        }
        for label, rounds in cases.items():
            with self.subTest(label), patch.object(self.registry, "run",
                                                   wraps=self.registry.run) as run:
                with self.assertRaisesRegex(api.AgentError, "unexpected tool response") as caught:
                    self.loop(*rounds)
                self.assertNotIn("calculate", str(caught.exception))
                expected_runs = 1 if label == "duplicate across rounds" else 0
                self.assertEqual(run.call_count, expected_runs)

    def test_tool_loop_refusal_incomplete_and_empty_text(self):
        refusal = ResponseOutputMessage(
            type="message", id="msg_1", role="assistant", status="completed",
            content=[ResponseOutputRefusal(type="refusal", refusal="No")])
        cases = {
            "incomplete": (reply(text="partial", status="incomplete"),
                           response("partial", status="incomplete")),
            "refusal": (reply(refusal), response("", output=[refusal])),
            "empty text": (reply(text="  "), response("  ")),
            "empty after tool": (reply(text=""), response("")),
        }
        for label, (loop_reply, plain_reply) in cases.items():
            with self.subTest(label):
                rounds = [reply(tool_call("c1")), loop_reply] if label == "empty after tool" \
                    else [loop_reply]
                with self.assertRaises(api.AgentError) as loop_error:
                    self.loop(*rounds)
                with self.assertRaises(api.AgentError) as plain_error:
                    api.generate("AiRi", "P", "T", client=scripted_client(plain_reply), model="m")
                self.assertEqual(str(loop_error.exception), str(plain_error.exception))

    def test_tool_loop_maps_api_errors_like_generate(self):
        error = RateLimitError("sensitive", response=Mock(status_code=429), body=None)
        with self.assertRaises(api.AgentError) as caught:
            self.loop(reply(tool_call("c1")), error)
        self.assertEqual(str(caught.exception),
                         "AiRi: rate limit or quota reached. Check quota and try later.")

    def test_encrypted_reasoning_is_never_exposed(self):
        secret = "ENC-SECRET-VALUE"
        cases = {
            "success": [reply(reasoning(secret), tool_call("c1")), reply(text="2")],
            "malformed": [reply(reasoning(secret), tool_call("c1", name=""))],
            "limit": [reply(reasoning(secret), tool_call(f"c{n}")) for n in range(4)],
            "failed after tool": [reply(reasoning(secret), tool_call("c1")),
                                   reply(text="", status="failed")],
        }
        for label, rounds in cases.items():
            with self.subTest(label), contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                try:
                    result = self.loop(*rounds)
                except api.AgentError as error:
                    result = str(error)
                self.assertNotIn(secret, result)
                self.assertNotIn(secret, out.getvalue() + err.getvalue())
        # Sent back to the same provider only, never kept anywhere else.
        self.assertIn(secret, json.dumps(self.requests()[1]["input"]))

    def test_generate_delegates_only_when_tools_given(self):
        client = scripted_client(response("Plain"), response("Tooled"))
        self.assertEqual(api.generate("AiRi", "P", "T", client=client, model="m"), "Plain")
        self.assertEqual(api.generate("AiRi", "P", "T", client=client, model="m",
                                      tools=self.registry), "Tooled")
        plain, tooled = [call.kwargs for call in client.responses.create.call_args_list]
        self.assertEqual(set(plain), {"model", "instructions", "input", "store"})
        self.assertEqual(plain["input"], "T")
        self.assertEqual(tooled["tools"], self.registry.specs())
        self.assertEqual(tooled["input"], [{"role": "user", "content": "T"}])
        unused = scripted_client()
        with self.assertRaises(ValueError):
            api.generate("AiRi", "P", "T", client=unused, model="m", tools=self.registry,
                         text_format={"type": "json_schema"})
        unused.responses.create.assert_not_called()

    def test_generate_return_contract_unchanged(self):
        for registry in (None, self.registry):
            with self.subTest(tools=registry is not None):
                client = scripted_client(response("  Hi \n"), response(""))
                self.assertEqual(api.generate("AiRi", "P", "T", client=client, model="m",
                                              tools=registry), "Hi")
                with self.assertRaisesRegex(api.AgentError, "^AiRi: API returned no usable text.$"):
                    api.generate("AiRi", "P", "T", client=client, model="m", tools=registry)


if __name__ == "__main__":
    unittest.main()
