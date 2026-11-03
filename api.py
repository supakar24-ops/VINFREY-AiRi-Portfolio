"""Small shared boundary for OpenAI configuration and safe API errors."""

import os

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

DEFAULT_MODEL = "gpt-6-luna"
MAX_TOOL_ROUNDS = 3  # rounds that may contain tool calls; at most 4 requests per turn
MAX_TOOL_CALLS = 4  # tool calls per turn, across all rounds
_TOOL_LOOP_ITEMS = frozenset({"message", "reasoning", "function_call"})


class AgentError(Exception):
    """An API or response problem that the CLI can explain without a traceback."""


def create_client():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise AgentError("Set OPENAI_API_KEY in your environment before running VINFREY.")
    # Network retries are separate from the workflow's three Builder revisions.
    return OpenAI(timeout=60.0, max_retries=2)


def generate(agent, instructions, prompt, *, client=None, model=None, text_format=None, tools=None):
    """Make one Responses API call and return only completed, nonempty text.

    With a tool registry, delegate to generate_with_tools(); the return contract is the same.
    """
    if tools is not None:
        if text_format is not None:
            raise ValueError("tools and text_format cannot be combined.")
        return generate_with_tools(agent, instructions, prompt, tools=tools,
                                   client=client, model=model)
    client, selected_model = _prepare(client, model)
    options = {
        "model": selected_model,
        "instructions": instructions,
        "input": prompt,
        # Stateless: nothing reads a stored response back, so ask the provider not to keep one.
        "store": False,
    }
    if text_format is not None:
        options["text"] = {"format": text_format}
    response = _create(agent, client, options)
    _check_completed(agent, response)
    return _output_text(agent, response)


def generate_with_tools(agent, instructions, prompt, *, tools, client=None, model=None):
    """Stateless tool loop: every request resends the whole turn, so store=False holds.

    Output items go back in their original order, including reasoning items (and any
    encrypted reasoning they carry), followed by one output per call with the same call_id.
    Tool results are untrusted data for the model only; they are never returned or logged.
    """
    client, selected_model = _prepare(client, model)
    options = {
        "model": selected_model,
        "instructions": instructions,
        "tools": tools.specs(),
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
    }
    items = [{"role": "user", "content": prompt}]
    seen_calls, used_calls = set(), 0
    for round_number in range(MAX_TOOL_ROUNDS + 1):
        response = _create(agent, client, {**options, "input": list(items)})
        _check_completed(agent, response)
        replay, calls = _tool_round(agent, response, seen_calls)
        if not calls:
            return _output_text(agent, response)
        used_calls += len(calls)
        # Limits are checked before any tool of this round runs.
        if round_number == MAX_TOOL_ROUNDS or used_calls > MAX_TOOL_CALLS:
            raise AgentError(f"{agent}: tool call limit reached. Try a simpler request.")
        items.extend(replay)
        items.extend({"type": "function_call_output", "call_id": call.call_id,
                      "output": tools.run(call.name, call.arguments)} for call in calls)
    raise AssertionError("unreachable")


def _tool_round(agent, response, seen_calls):
    """Validate every output item before any tool runs; fail closed on anything unexpected."""
    malformed = AgentError(f"{agent}: API returned an unexpected tool response. Try again.")
    replay, calls = [], []
    for item in response.output:
        kind = getattr(item, "type", None)
        if kind not in _TOOL_LOOP_ITEMS:
            raise malformed
        if kind == "function_call":
            call_id, name = getattr(item, "call_id", None), getattr(item, "name", None)
            if (not isinstance(call_id, str) or not call_id or call_id in seen_calls
                    or not isinstance(name, str) or not name
                    or not isinstance(getattr(item, "arguments", None), str)
                    or getattr(item, "status", None) not in (None, "completed")):
                raise malformed
            seen_calls.add(call_id)
            calls.append(item)
        dump = getattr(item, "model_dump", None)
        if not callable(dump):
            raise malformed
        try:
            replay.append(dump(mode="json", by_alias=True, exclude_none=True))
        except Exception:
            raise malformed from None
    return replay, calls


def _prepare(client, model):
    if client is None:
        client = create_client()
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip()
    if not selected_model:
        raise AgentError("OPENAI_MODEL must not be empty.")
    return client, selected_model


def _create(agent, client, options):
    """One request with API exceptions mapped to safe messages."""
    try:
        return client.responses.create(**options)
    except AuthenticationError:
        raise AgentError(f"{agent}: authentication failed. Check OPENAI_API_KEY.") from None
    except RateLimitError:
        raise AgentError(f"{agent}: rate limit or quota reached. Check quota and try later.") from None
    except APITimeoutError:
        raise AgentError(f"{agent}: API request timed out. Try again later.") from None
    except APIConnectionError:
        raise AgentError(f"{agent}: cannot connect to OpenAI. Check your connection.") from None
    except APIStatusError as error:
        raise AgentError(
            f"{agent}: API returned HTTP {error.status_code}. Check model access and request settings."
        ) from None
    except OpenAIError:
        raise AgentError(f"{agent}: OpenAI request failed. Check your API configuration.") from None


def _check_completed(agent, response):
    if response.status != "completed":
        raise AgentError(f"{agent}: API response did not complete. Try again or simplify the task.")
    for item in response.output:
        # Reasoning items carry content=None, so treat a missing or empty value alike.
        for content in getattr(item, "content", None) or []:
            if content.type == "refusal":
                raise AgentError(f"{agent}: the model declined this request. Revise the task.")


def _output_text(agent, response):
    output = response.output_text
    if not isinstance(output, str) or not output.strip():
        raise AgentError(f"{agent}: API returned no usable text.")
    return output.strip()
