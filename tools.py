"""Pure computation tools for AiRi chat: an explicit allowlist with strict arguments.

Tools here only compute. They never read files, the clock, the memory database, or the
network, and never start processes. Results go back to the model as untrusted data.
"""

import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from types import MappingProxyType

MAX_ARGUMENT_CHARS = 1000
MAX_OUTPUT_CHARS = 1000
MAX_EXPRESSION_CHARS = 200
MAX_NODES = 64
MAX_DEPTH = 20
MAX_POWERS = 2
MAX_MAGNITUDE = 10**15
MAX_DIGITS = 15  # log10(MAX_MAGNITUDE)
MAX_EXPONENT = 64
MAX_DAYS = 36500

ERROR_CODES = frozenset({
    "unknown_tool", "invalid_arguments", "tool_failed", "output_too_large",
    "invalid_expression", "division_by_zero", "result_too_large", "invalid_date", "out_of_range",
})
_NAME = re.compile(r"[a-z_]{1,32}\Z")
_TYPES = MappingProxyType({"string": str, "integer": int, "null": type(None)})


class ToolError(Exception):
    """A safe, fixed error code a handler may report back to the model."""

    def __init__(self, code):
        if code not in ERROR_CODES:
            raise ValueError("unknown tool error code")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    side_effects: str
    handler: object
    max_output_chars: int = MAX_OUTPUT_CHARS


class ToolRegistry:
    """Default deny: only registered pure tools run, and only with valid arguments."""

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        if tool.side_effects != "none":
            raise ValueError("only tools without side effects can be registered")
        if not isinstance(tool.name, str) or not _NAME.match(tool.name):
            raise ValueError("tool names use lowercase letters and underscores only")
        if not callable(tool.handler):
            raise ValueError("tool handler must be callable")
        _check_schema(tool.parameters)
        self._tools[tool.name] = tool

    def names(self):
        return tuple(self._tools)

    def specs(self):
        """Strict function definitions for the API, as fresh copies each call."""
        return [json.loads(json.dumps({
            "type": "function", "name": tool.name, "description": tool.description,
            "parameters": tool.parameters, "strict": True,
        })) for tool in self._tools.values()]

    def run(self, name, arguments):
        """Return a JSON string for the model. Never raises and never leaks exception text."""
        tool = self._tools.get(name) if isinstance(name, str) else None
        if tool is None:
            return _error("unknown_tool")
        values = _parse_arguments(arguments, tool.parameters)
        if values is None:
            return _error("invalid_arguments")
        try:
            output = json.dumps({"ok": True, "result": tool.handler(**values)}, ensure_ascii=False)
        except ToolError as error:
            return _error(error.code)
        except Exception:
            return _error("tool_failed")
        if len(output) > tool.max_output_chars:
            return _error("output_too_large")
        return output


def _error(code):
    return json.dumps({"ok": False, "error": code})


def _types(spec):
    kinds = spec.get("type")
    return kinds if isinstance(kinds, list) else [kinds]


def _check_schema(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("tool parameters must be a strict object schema")
    properties = parameters.get("properties")
    if (parameters.get("type") != "object" or not isinstance(properties, dict) or not properties
            or parameters.get("additionalProperties") is not False
            or sorted(parameters.get("required", [])) != sorted(properties)):
        raise ValueError("tool parameters must be a strict object schema")
    for spec in properties.values():
        if not _types(spec) or not all(kind in _TYPES for kind in _types(spec)):
            raise ValueError("unsupported parameter type")


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key")
    return dict(pairs)


def _reject_constant(_):
    raise ValueError("NaN and Infinity are not JSON")


def _parse_arguments(arguments, parameters):
    """Exact keys, exact types (bool is not an integer), bounded strings, enums."""
    if not isinstance(arguments, str) or len(arguments) > MAX_ARGUMENT_CHARS:
        return None
    try:
        values = json.loads(arguments, object_pairs_hook=_reject_duplicates,
                            parse_constant=_reject_constant)
    except ValueError:
        return None
    properties = parameters["properties"]
    if not isinstance(values, dict) or set(values) != set(properties):
        return None
    for key, spec in properties.items():
        value = values[key]
        if not any(type(value) is _TYPES[kind] for kind in _types(spec)):
            return None
        if isinstance(value, str) and len(value) > spec.get("maxLength", MAX_ARGUMENT_CHARS):
            return None
        if "enum" in spec and value not in spec["enum"]:
            return None
    return values


# calculate ---------------------------------------------------------------------------

_EXPRESSION = re.compile(r"[0-9.+\-*/%() ]+\Z")
_BINARY = MappingProxyType({
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
})
_DIVISIONS = (ast.Div, ast.FloorDiv, ast.Mod)


def calculate(expression):
    expression = expression.strip()
    if not expression or len(expression) > MAX_EXPRESSION_CHARS or not _EXPRESSION.match(expression):
        raise ToolError("invalid_expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise ToolError("invalid_expression") from None
    _check_tree(tree.body)
    return {"value": _format(_evaluate(tree.body))}


def _check_tree(root):
    """Static budget, checked before any arithmetic runs."""
    nodes, powers, stack = 0, 0, [(root, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH:
            raise ToolError("invalid_expression")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ToolError("invalid_expression")
            _bounded(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            stack.append((node.operand, depth + 1))
        elif isinstance(node, ast.BinOp) and (type(node.op) in _BINARY
                                              or isinstance(node.op, ast.Pow)):
            if isinstance(node.op, ast.Pow):
                powers += 1
                nested = any(isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow)
                             for side in (node.left, node.right) for inner in ast.walk(side))
                if nested or powers > MAX_POWERS:
                    raise ToolError("result_too_large")
            stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
        else:
            raise ToolError("invalid_expression")


def _evaluate(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)
        return _bounded(-value if isinstance(node.op, ast.USub) else value)
    left, right = _evaluate(node.left), _evaluate(node.right)
    if isinstance(node.op, ast.Pow):
        _check_power(left, right)
        return _bounded(_power(left, right))
    if isinstance(node.op, _DIVISIONS) and right == 0:
        raise ToolError("division_by_zero")
    return _bounded(_BINARY[type(node.op)](left, right))


def _check_power(base, exponent):
    """Estimate the size of base**exponent before computing it."""
    if type(exponent) is not int:
        raise ToolError("invalid_expression")  # a negative base would give a complex number
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError("result_too_large")
    if base == 0:
        if exponent < 0:
            raise ToolError("division_by_zero")
        return
    if exponent * math.log10(abs(base)) > MAX_DIGITS:
        raise ToolError("result_too_large")


def _power(base, exponent):
    try:
        return base ** exponent
    except OverflowError:
        raise ToolError("result_too_large") from None


def _bounded(value):
    if isinstance(value, float) and not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
        raise ToolError("result_too_large")
    return value


def _format(value):
    return str(value) if isinstance(value, int) else format(value, ".15g")


# date_math ---------------------------------------------------------------------------

_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")


def _parse_date(text):
    if not _DATE.match(text):
        raise ToolError("invalid_date")
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ToolError("invalid_date") from None


def date_math(operation, start, end, days):
    first = _parse_date(start)
    if operation == "days_between":
        if end is None or days is not None:
            raise ToolError("invalid_arguments")
        return {"days": (_parse_date(end) - first).days}
    if days is None or end is not None:
        raise ToolError("invalid_arguments")
    if abs(days) > MAX_DAYS:
        raise ToolError("out_of_range")
    try:
        return {"date": (first + timedelta(days=days)).isoformat()}
    except OverflowError:
        raise ToolError("out_of_range") from None


def _object(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def default_tools():
    """A new registry with the approved P0-5 tools: calculate and date_math only."""
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculate",
        description=("Evaluate a basic arithmetic expression: numbers, + - * / // % **, and "
                     "parentheses. No variables, functions, or scientific notation."),
        parameters=_object(expression={"type": "string", "maxLength": MAX_EXPRESSION_CHARS}),
        side_effects="none", handler=calculate))
    registry.register(Tool(
        name="date_math",
        description=("Count days between two dates, or add days to a date. Dates use "
                     "YYYY-MM-DD. This tool does not know today's date."),
        parameters=_object(
            operation={"type": "string", "enum": ["days_between", "add_days"]},
            start={"type": "string", "maxLength": 10},
            end={"type": ["string", "null"], "maxLength": 10},
            days={"type": ["integer", "null"]}),
        side_effects="none", handler=date_math))
    return registry
