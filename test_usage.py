"""Offline tests for the per-session API request guard (P0-7)."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import APITimeoutError, RateLimitError

import api
import usage
from api import AgentError


def reply(input_tokens=100, output_tokens=20, text="ok"):
    return SimpleNamespace(status="completed", output=[], output_text=text,
                           usage=SimpleNamespace(input_tokens=input_tokens,
                                                 output_tokens=output_tokens))


def client(*outputs):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=list(outputs))))


class UsageGuardTests(unittest.TestCase):
    def test_counts_requests_and_tokens(self):
        guard = usage.UsageGuard(40, client=client(reply(100, 20), reply(100, 20)))
        guard.responses.create(model="m", input="a")
        guard.responses.create(model="m", input="b")
        self.assertEqual(guard.snapshot(), {"limit": 40, "requests": 2, "remaining": 38,
                                            "failed": 0, "input_tokens": 200,
                                            "output_tokens": 40, "missing_usage": 0})

    def test_limit_stops_before_dispatch(self):
        inner = client(reply(), reply(), reply())
        guard = usage.UsageGuard(2, client=inner)
        guard.create(input="a")
        guard.create(input="b")
        with self.assertRaises(usage.UsageLimitReached) as caught:
            guard.create(input="c")
        self.assertIsInstance(caught.exception, AgentError)
        self.assertEqual(inner.responses.create.call_count, 2)
        self.assertIn("2/2", str(caught.exception))
        self.assertIn("not a spending limit", str(caught.exception))
        self.assertEqual(guard.snapshot()["requests"], 2)

    def test_zero_limit_never_dispatches(self):
        inner = client(reply())
        with self.assertRaises(usage.UsageLimitReached):
            usage.UsageGuard(0, client=inner).create(input="a")
        inner.responses.create.assert_not_called()

    def test_missing_usage_is_counted_separately(self):
        cases = {
            "no usage attribute": SimpleNamespace(status="completed", output=[], output_text="ok"),
            "usage is None": SimpleNamespace(status="completed", output=[], output_text="ok",
                                             usage=None),
            "tokens not int": SimpleNamespace(status="completed", output=[], output_text="ok",
                                              usage=SimpleNamespace(input_tokens=Mock(),
                                                                    output_tokens=None)),
        }
        for label, response in cases.items():
            with self.subTest(label):
                guard = usage.UsageGuard(5, client=client(response))
                guard.create(input="a")
                snapshot = guard.snapshot()
                self.assertEqual((snapshot["requests"], snapshot["missing_usage"]), (1, 1))
                self.assertEqual((snapshot["input_tokens"], snapshot["output_tokens"]), (0, 0))

    def test_failed_dispatch_counts_once_without_retry(self):
        errors = [APITimeoutError(request=Mock()),
                  RateLimitError("quota", response=Mock(status_code=429), body=None)]
        for error in errors:
            with self.subTest(type(error).__name__):
                inner = client(error)
                guard = usage.UsageGuard(5, client=inner)
                with self.assertRaises(type(error)):
                    guard.create(input="a")
                self.assertEqual(inner.responses.create.call_count, 1)
                snapshot = guard.snapshot()
                self.assertEqual((snapshot["requests"], snapshot["failed"]), (1, 1))

    def test_missing_key_is_not_counted(self):
        message = "Set OPENAI_API_KEY in your environment before running VINFREY."
        with patch("api.create_client", side_effect=AgentError(message)):
            guard = usage.UsageGuard(5)
            with self.assertRaises(AgentError) as caught:
                guard.create(input="a")
        self.assertNotIsInstance(caught.exception, usage.UsageLimitReached)
        self.assertEqual(str(caught.exception), message)
        self.assertEqual(guard.snapshot()["requests"], 0)

    def test_created_client_has_retries_disabled(self):
        inner = client(reply(), reply())
        factory_client = Mock()
        factory_client.with_options.return_value = inner
        with patch("api.create_client", return_value=factory_client) as factory:
            guard = usage.UsageGuard(5)
            guard.create(input="a")
            guard.create(input="b")
        factory.assert_called_once_with()
        factory_client.with_options.assert_called_once_with(max_retries=0)
        self.assertEqual(inner.responses.create.call_count, 2)
        factory_client.responses.create.assert_not_called()

    def test_injected_client_is_used_as_is(self):
        inner = client(reply())
        with patch("api.create_client") as factory:
            usage.UsageGuard(5, client=inner).create(input="a")
        factory.assert_not_called()
        inner.responses.create.assert_called_once_with(input="a")

    def test_concurrent_requests_never_exceed_limit(self):
        release = threading.Event()

        def slow_create(**_):
            release.wait(5)
            return reply()

        inner = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=slow_create)))
        guard = usage.UsageGuard(5, client=inner)
        refused, lock = [], threading.Lock()

        def call():
            try:
                guard.create(input="x")
            except usage.UsageLimitReached:
                with lock:
                    refused.append(1)

        threads = [threading.Thread(target=call) for _ in range(20)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 5
        while len(refused) < 15 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(inner.responses.create.call_count, 5)
        self.assertEqual(len(refused), 15)
        self.assertEqual(guard.snapshot()["requests"], 5)

    def test_ensure_available_checks_without_reserving(self):
        guard = usage.UsageGuard(10, client=client(*(reply() for _ in range(7))))
        for _ in range(7):
            guard.create(input="a")
        guard.ensure_available(3, "chat")
        with self.assertRaises(usage.UsageLimitReached) as caught:
            guard.ensure_available(4, "/work")
        self.assertEqual(str(caught.exception),
                         "Not enough API requests left in this session for /work: "
                         "needs up to 4, 3 left. Use /usage to check.")
        self.assertEqual(guard.snapshot()["requests"], 7)

    def test_describe_usage(self):
        snapshot = {"limit": 40, "requests": 7, "remaining": 33, "failed": 0,
                    "input_tokens": 9120, "output_tokens": 1450, "missing_usage": 0}
        self.assertEqual(usage.describe_usage(snapshot),
                         "API usage this session: 7/40 requests · 9,120 input / 1,450 output tokens")
        snapshot.update(failed=2, missing_usage=1)
        self.assertEqual(usage.describe_usage(snapshot),
                         "API usage this session: 7/40 requests · 9,120 input / 1,450 output "
                         "tokens · 1 without usage data · 2 failed")

    def test_works_as_client_for_generate(self):
        guard = usage.UsageGuard(1, client=client(reply(text="  Hi ")))
        self.assertEqual(api.generate("AiRi", "P", "T", client=guard, model="m"), "Hi")
        with self.assertRaises(usage.UsageLimitReached):
            api.generate("AiRi", "P", "T", client=guard, model="m")

    def test_limits(self):
        self.assertEqual((usage.NORMAL_REQUEST_LIMIT, usage.DEMO_REQUEST_LIMIT), (40, 20))
        for bad in (-1, 1.5, True, "40"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                usage.UsageGuard(bad)


if __name__ == "__main__":
    unittest.main()
