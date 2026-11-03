"""Offline unit tests for the application-level ApprovalGate."""

import contextlib
import threading
import unittest

from approval import ActionFailed, ActionRejected, ApprovalError, ApprovalGate


def echo_gate():
    """A gate with one test action whose executor records every call."""
    calls = []
    gate = ApprovalGate()

    def validate(params):
        if "value" not in params:
            raise ApprovalError("Missing value.")
        return dict(params)

    def describe(params):
        return f"Echo {params['value']}"

    def execute(params):
        calls.append(params)
        return f"ran {params['value']}"

    gate.register("test.echo", validate=validate, describe=describe, execute=execute)
    return gate, calls


class GateRequestTests(unittest.TestCase):
    def setUp(self):
        self.gate, self.calls = echo_gate()

    def test_unknown_kind_is_denied(self):
        with self.assertRaises(ApprovalError):
            self.gate.request("nope", {})
        self.assertEqual(self.gate.pending(), [])

    def test_invalid_params_are_denied(self):
        with self.assertRaises(ApprovalError):
            self.gate.request("test.echo", {})
        self.assertEqual(self.gate.pending(), [])

    def test_request_never_executes(self):
        action = self.gate.request("test.echo", {"value": 1})
        self.assertEqual(action.id, "A1")
        self.assertEqual(action.kind, "test.echo")
        self.assertEqual(action.requested_by, "user")
        self.assertEqual(self.gate.status("A1"), "pending")
        self.assertEqual(self.gate.pending(), [action])
        self.assertEqual(self.calls, [])

    def test_params_are_deep_frozen(self):
        source = {"value": 1, "tags": ["a"], "meta": {"k": "v"}}
        action = self.gate.request("test.echo", source)
        source["tags"].append("b")
        source["meta"]["k"] = "changed"
        self.assertEqual(action.params["tags"], ("a",))
        self.assertEqual(action.params["meta"]["k"], "v")
        with self.assertRaises(TypeError):
            action.params["meta"]["k"] = "x"
        with self.assertRaises(TypeError):
            action.params["value"] = 2

    def test_unsupported_param_type_denied(self):
        with self.assertRaises(ApprovalError):
            self.gate.request("test.echo", {"value": object()})
        self.assertEqual(self.gate.pending(), [])

    def test_summary_comes_from_describe(self):
        action = self.gate.request("test.echo", {"value": "x"})
        self.assertEqual(action.summary, "Echo x")

    def test_deny_never_executes_and_is_final(self):
        self.gate.request("test.echo", {"value": 1})
        denied = self.gate.deny("A1")
        self.assertEqual(denied.id, "A1")
        self.assertEqual(self.gate.status("A1"), "denied")
        self.assertEqual(self.gate.pending(), [])
        for decide in (self.gate.deny, self.gate.approve):
            with self.subTest(decide=decide.__name__), self.assertRaises(ApprovalError):
                decide("A1")
        self.assertEqual(self.calls, [])

    def test_unknown_id_is_denied(self):
        self.gate.request("test.echo", {"value": "private-value"})
        with self.assertRaises(ApprovalError) as raised:
            self.gate.deny("A99")
        self.assertNotIn("private-value", str(raised.exception))
        with self.assertRaises(ApprovalError):
            self.gate.status("A99")

    def test_clear_drops_pending_and_ids_not_reused(self):
        self.gate.request("test.echo", {"value": 1})
        self.gate.clear()
        self.assertEqual(self.gate.pending(), [])
        with self.assertRaises(ApprovalError):
            self.gate.status("A1")
        self.assertEqual(self.gate.request("test.echo", {"value": 2}).id, "A2")

    def test_action_id_case_insensitive(self):
        self.gate.request("test.echo", {"value": 1})
        self.assertEqual(self.gate.deny(" a1 ").id, "A1")


class GateApproveTests(unittest.TestCase):
    def gate_with(self, execute):
        gate = ApprovalGate()
        gate.register("test.run", validate=dict, describe=lambda p: "Run", execute=execute)
        gate.request("test.run", {"value": "private-value"})
        return gate

    def test_approve_executes_once_with_frozen_params(self):
        gate, calls = echo_gate()
        gate.request("test.echo", {"value": 1})
        self.assertEqual(gate.approve("A1"), "ran 1")
        self.assertEqual(gate.status("A1"), "executed")
        with self.assertRaises(TypeError):
            calls[0]["value"] = 2
        with self.assertRaises(ApprovalError):
            gate.approve("A1")
        self.assertEqual(len(calls), 1)

    def test_status_is_executing_during_execution(self):
        seen = []

        def execute(params):
            seen.append(gate.status("A1"))
            with self.assertRaises(ApprovalError):
                gate.approve("A1")
            return "done"

        gate = self.gate_with(execute)
        self.assertEqual(gate.approve("A1"), "done")
        self.assertEqual(seen, ["executing"])

    def test_concurrent_approve_runs_once(self):
        started, release, calls, errors = threading.Event(), threading.Event(), [], []

        def execute(params):
            calls.append(params)
            started.set()
            release.wait(5)

        gate = self.gate_with(execute)
        first = threading.Thread(target=gate.approve, args=("A1",))
        first.start()
        self.assertTrue(started.wait(5))
        try:
            gate.approve("A1")
        except ApprovalError as error:
            errors.append(error)
        release.set()
        first.join(5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(gate.status("A1"), "executed")

    def test_action_rejected_marks_rejected(self):
        def execute(params):
            raise ActionRejected("Precondition failed.")

        gate = self.gate_with(execute)
        with self.assertRaises(ActionRejected):
            gate.approve("A1")
        self.assertEqual(gate.status("A1"), "rejected")
        with self.assertRaises(ApprovalError):
            gate.approve("A1")

    def test_executor_failure_marks_failed_without_retry(self):
        calls = []

        def execute(params):
            calls.append(params)
            raise RuntimeError("secret-path")

        gate = self.gate_with(execute)
        with self.assertRaises(ActionFailed) as raised:
            gate.approve("A1")
        self.assertIsInstance(raised.exception, ApprovalError)
        self.assertNotIn("secret-path", str(raised.exception))
        self.assertNotIn("private-value", str(raised.exception))
        self.assertIn("may have partly run", str(raised.exception))
        self.assertEqual(gate.status("A1"), "failed")
        with self.assertRaises(ApprovalError):
            gate.approve("A1")
        self.assertEqual(len(calls), 1)

    def test_cannot_approve_denied(self):
        gate, calls = echo_gate()
        gate.request("test.echo", {"value": 1})
        gate.deny("A1")
        with self.assertRaises(ApprovalError):
            gate.approve("A1")
        self.assertEqual(calls, [])


class GateHardeningTests(unittest.TestCase):
    """P0-4.1: handler integrity, retention, and invalidation."""

    def gate_with(self, execute=lambda params: "ran"):
        gate = ApprovalGate()
        gate.register("k", validate=dict, describe=lambda p: f"Do {p['value']}", execute=execute)
        return gate

    def test_duplicate_registration_is_rejected(self):
        gate = self.gate_with()
        with self.assertRaises(RuntimeError):
            gate.register("k", validate=dict, describe=str, execute=str)

    def test_pending_action_uses_handler_that_validated_it(self):
        ran = []
        gate = self.gate_with(lambda params: ran.append("original"))
        gate.request("k", {"value": 1})
        # Simulate a later swap of the registry entry; the pending action must not follow it.
        gate._handlers["k"] = type(gate._handlers["k"])(
            dict, str, lambda params: ran.append("replaced"))
        gate.approve("A1")
        self.assertEqual(ran, ["original"])

    def test_decided_records_drop_params_and_summary(self):
        def reject(params):
            raise ActionRejected("no")

        def fail(params):
            raise RuntimeError("boom")

        cases = {"denied": None, "executed": lambda p: "ok", "rejected": reject, "failed": fail}
        for status, execute in cases.items():
            with self.subTest(status=status):
                gate = self.gate_with(execute or (lambda p: "ok"))
                gate.request("k", {"value": "private-value"})
                if status == "denied":
                    gate.deny("A1")
                else:
                    with contextlib.suppress(ActionRejected, ApprovalError):
                        gate.approve("A1")
                record = gate._records["A1"]
                self.assertEqual(gate.status("A1"), status)
                self.assertEqual(dict(record.action.params), {})
                self.assertEqual(record.action.summary, "")
                self.assertNotIn("private-value", repr(record))
                for decide in (gate.approve, gate.deny):
                    with self.assertRaises(ApprovalError):
                        decide("A1")

    def test_executor_receives_params_before_redaction(self):
        seen = []
        gate = self.gate_with(lambda params: seen.append(params["value"]))
        gate.request("k", {"value": "kept"})
        gate.approve("A1")
        self.assertEqual(seen, ["kept"])

    def test_clear_removes_all_records_without_reusing_ids(self):
        gate = self.gate_with()
        gate.request("k", {"value": 1})
        gate.deny("A1")
        gate.request("k", {"value": 2})
        gate.clear()
        for action_id in ("A1", "A2"):
            with self.subTest(action_id=action_id), self.assertRaises(ApprovalError):
                gate.status(action_id)
        self.assertEqual(gate._records, {})
        self.assertEqual(gate.request("k", {"value": 3}).id, "A3")

    def test_invalidate_cancels_matching_pending_only(self):
        ran = []
        gate = self.gate_with(lambda params: ran.append(params))
        for value in (1, 2, 3):
            gate.request("k", {"value": value})
        gate.deny("A3")
        cancelled = gate.invalidate(lambda action: action.id in ("A2", "A3"))
        self.assertEqual(cancelled, ["A2"])
        self.assertEqual(gate.status("A2"), "invalidated")
        self.assertEqual(gate.status("A3"), "denied")
        self.assertEqual(dict(gate._records["A2"].action.params), {})
        with self.assertRaises(ApprovalError):
            gate.approve("A2")
        self.assertEqual([a.id for a in gate.pending()], ["A1"])
        self.assertEqual(ran, [])


if __name__ == "__main__":
    unittest.main()
