"""Application-level approval gate: protected actions run only after a human approves.

This is not an operating-system security boundary. Code in the same process could
call approve(); the protection is that only explicit user commands and UI buttons
route here, and model-generated text is never parsed as a command.
"""

import copy
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_LEAF_TYPES = (str, int, float, bool, type(None))


class ApprovalError(ValueError):
    """An invalid request or decision. Messages name only the action id."""


class ActionFailed(ApprovalError):
    """An approved action raised during execution and may have partly run."""


class ActionRejected(Exception):
    """An executor's precondition failed; it guarantees nothing was changed."""


@dataclass(frozen=True)
class PendingAction:
    id: str
    kind: str
    params: Mapping
    summary: str
    requested_by: str


@dataclass(frozen=True)
class _Handler:
    validate: object
    describe: object
    execute: object


@dataclass
class _Record:
    action: PendingAction
    handler: _Handler  # the handler that validated this action; it alone may run it
    status: str = "pending"


_EMPTY = MappingProxyType({})


def _freeze(value):
    """Return a deep read-only copy, or raise ApprovalError for unsupported data."""
    if isinstance(value, _LEAF_TYPES):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ApprovalError("Action parameters must use text keys.")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    raise ApprovalError("Action parameters contain an unsupported value.")


class ApprovalGate:
    """Per-session queue of protected actions. Default deny for anything unclear."""

    def __init__(self):
        self._handlers = {}
        self._records = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def register(self, kind, *, validate, describe, execute):
        if kind in self._handlers:
            raise RuntimeError(f"Action kind {kind!r} is already registered.")
        self._handlers[kind] = _Handler(validate, describe, execute)

    def request(self, kind, params, *, requested_by="user"):
        """Validate and record an action. Never executes it."""
        handler = self._handlers.get(kind)
        if handler is None:
            raise ApprovalError("This action is not allowed.")
        if not isinstance(params, dict):
            raise ApprovalError("Action parameters are invalid.")
        frozen = _freeze(handler.validate(copy.deepcopy(params)))
        summary = handler.describe(frozen)
        with self._lock:
            action = PendingAction(f"A{self._next_id}", kind, frozen, summary, requested_by)
            self._next_id += 1
            self._records[action.id] = _Record(action, handler)
        return action

    def pending(self):
        with self._lock:
            return [r.action for r in self._records.values() if r.status == "pending"]

    def status(self, action_id):
        with self._lock:
            return self._record(action_id).status

    def approve(self, action_id):
        """Run a pending action exactly once; failures are final and never retried."""
        with self._lock:
            record = self._claim(action_id)
        action = record.action
        try:
            result = record.handler.execute(action.params)
        except ActionRejected:
            self._finish(record, "rejected")
            raise
        except Exception:
            # The executor may have changed something before failing; say so.
            self._finish(record, "failed")
            raise ActionFailed(f"Action {action.id} failed and may have partly run.") from None
        self._finish(record, "executed")
        return result

    def deny(self, action_id):
        with self._lock:
            record = self._record(action_id)
            if record.status != "pending":
                raise ApprovalError(f"Action {record.action.id} is already {record.status}.")
            action = record.action
            self._redact(record, "denied")
            return action

    def invalidate(self, predicate):
        """Cancel matching pending actions without running them; return their ids."""
        with self._lock:
            matched = [r for r in self._records.values()
                       if r.status == "pending" and predicate(r.action)]
            for record in matched:
                self._redact(record, "invalidated")
            return [record.action.id for record in matched]

    def clear(self):
        """Drop every record. Ids keep counting so old ids are never reused."""
        with self._lock:
            self._records = {}

    def _finish(self, record, status):
        with self._lock:
            self._redact(record, status)

    @staticmethod
    def _redact(record, status):
        # A decided action keeps only its id, kind and status; params and summary may be private.
        old = record.action
        record.action = PendingAction(old.id, old.kind, _EMPTY, "", old.requested_by)
        record.status = status

    def _record(self, action_id):
        key = str(action_id).strip().upper()
        record = self._records.get(key)
        if record is None:
            raise ApprovalError(f"No action {key or '(empty)'} exists.")
        return record

    def _claim(self, action_id):
        record = self._record(action_id)
        if record.status != "pending":
            raise ApprovalError(f"Action {record.action.id} is already {record.status}.")
        record.status = "executing"
        return record
