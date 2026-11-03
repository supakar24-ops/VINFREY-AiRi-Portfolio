"""Per-session API request guard for AiRi (P0-7).

UsageGuard stands in for the OpenAI client: every chat, tool-loop, /work, and /suggest
request already goes through AiRi's client, so wrapping it covers them all. It counts
requests in memory for one session only. It is a safety limit, not a spending limit:
a new CLI run or browser session starts again from zero.
"""

import threading

import api
from api import AgentError

NORMAL_REQUEST_LIMIT = 40
DEMO_REQUEST_LIMIT = 20


class UsageLimitReached(AgentError):
    """The session has no API requests left; nothing was sent."""


class UsageGuard:
    def __init__(self, limit, *, client=None):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer.")
        self.limit = limit
        # An injected client is used as given; one created here has SDK retries off.
        self._client = client
        self._lock = threading.Lock()
        self._requests = self._failed = 0
        self._input_tokens = self._output_tokens = self._missing_usage = 0

    @property
    def responses(self):
        return self

    def create(self, **options):
        client = self._real_client()  # a missing key fails here, before anything is counted
        with self._lock:
            if self._requests >= self.limit:
                raise UsageLimitReached(
                    f"Usage limit reached: {self.limit}/{self.limit} API requests used in this "
                    "session. This is a per-session safety limit, not a spending limit.")
            # Counted once dispatch is attempted, even if the request then fails.
            self._requests += 1
        try:
            response = client.responses.create(**options)
        except Exception:
            with self._lock:
                self._failed += 1
            raise
        self._record(response)
        return response

    def ensure_available(self, needed, action):
        """Refuse work that could need more requests than remain. Reserves nothing."""
        with self._lock:
            remaining = self.limit - self._requests
        if remaining < needed:
            raise UsageLimitReached(
                f"Not enough API requests left in this session for {action}: "
                f"needs up to {needed}, {remaining} left. Use /usage to check.")

    def snapshot(self):
        with self._lock:
            return {"limit": self.limit, "requests": self._requests,
                    "remaining": self.limit - self._requests, "failed": self._failed,
                    "input_tokens": self._input_tokens, "output_tokens": self._output_tokens,
                    "missing_usage": self._missing_usage}

    def _real_client(self):
        with self._lock:
            if self._client is None:
                # Looked up at call time so tests can patch api.create_client.
                created = api.create_client()
                with_options = getattr(created, "with_options", None)
                self._client = with_options(max_retries=0) if callable(with_options) else created
            return self._client

    def _record(self, response):
        report = getattr(response, "usage", None)
        tokens = (getattr(report, "input_tokens", None), getattr(report, "output_tokens", None))
        with self._lock:
            if all(type(count) is int for count in tokens):
                self._input_tokens += tokens[0]
                self._output_tokens += tokens[1]
            else:
                self._missing_usage += 1


def describe_usage(snapshot):
    text = (f"API usage this session: {snapshot['requests']}/{snapshot['limit']} requests · "
            f"{snapshot['input_tokens']:,} input / {snapshot['output_tokens']:,} output tokens")
    if snapshot["missing_usage"]:
        text += f" · {snapshot['missing_usage']} without usage data"
    if snapshot["failed"]:
        text += f" · {snapshot['failed']} failed"
    return text
