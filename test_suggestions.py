"""Offline tests for AI-suggested memories: scripted API replies only, no network."""

import json
import typing
import unittest
from types import SimpleNamespace

from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

import suggestions
from api import AgentError
from test_workflow import response, scripted_client


def history(*messages):
    return [{"user": message, "assistant": f"reply to {message}"} for message in messages]


def reply(*items):
    return response(json.dumps({"suggestions": [
        {"content": content, "reason": reason} for content, reason in items]}))


class ContextTests(unittest.TestCase):
    def test_user_context_keeps_last_six_user_messages_only(self):
        turns = history(*(f"message {n}" for n in range(8)))
        context = suggestions.user_context(turns)
        self.assertEqual(context, [f"message {n}" for n in range(2, 8)])
        self.assertFalse(any("reply to" in message for message in context))

    def test_context_is_capped_at_3000_characters_newest_first(self):
        turns = history("a" * 1500, "b" * 1500, "c" * 1500)
        context = suggestions.user_context(turns)
        self.assertEqual(context, ["b" * 1500, "c" * 1500])
        self.assertLessEqual(sum(map(len, context)), suggestions.MAX_CONTEXT_CHARACTERS)

    def test_context_excludes_messages_with_credentials(self):
        turns = history("I like Python", "my key is sk-" + "a" * 24, "pass" + "word: hunter2")
        self.assertEqual(suggestions.user_context(turns), ["I like Python"])

    def test_empty_or_unusable_history_never_calls_api(self):
        for turns in ([], history("x" * 3001), history("token=" + "a1" * 20)):
            client = scripted_client()
            with self.subTest(size=len(turns)), self.assertRaises(ValueError):
                suggestions.suggest_memories(turns, client=client)
            client.responses.create.assert_not_called()

    def test_payload_contains_only_user_messages(self):
        client = scripted_client(reply(("Likes Python", "User said so")))
        suggestions.suggest_memories(history("I like Python", "Thanks"), client=client)
        self.assertEqual(client.responses.create.call_count, 1)
        call = client.responses.create.call_args.kwargs
        self.assertEqual(json.loads(call["input"]), {"user_messages": ["I like Python", "Thanks"]})
        self.assertEqual(call["instructions"], suggestions.SUGGESTION_INSTRUCTIONS)
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertNotIn("reply to", call["input"])


class FormatTests(unittest.TestCase):
    def test_format_matches_installed_sdk_json_schema_type(self):
        allowed = set(typing.get_type_hints(ResponseFormatTextJSONSchemaConfigParam))
        fmt = suggestions.SUGGESTION_FORMAT
        self.assertLessEqual(set(fmt), allowed)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertIs(fmt["strict"], True)

    def test_schema_is_strict_compatible(self):
        schema = suggestions.SUGGESTION_FORMAT["schema"]
        item = schema["properties"]["suggestions"]["items"]
        for node in (schema, item):
            self.assertIs(node["additionalProperties"], False)
            self.assertEqual(set(node["required"]), set(node["properties"]))


class ParseTests(unittest.TestCase):
    def parse(self, *items):
        return suggestions.parse_suggestions(reply(*items).output_text)

    def test_valid_suggestions_are_returned(self):
        accepted, dropped = self.parse(("  Likes Python ", " Said so "), ("ชอบภาษาไทย", "บอกเอง"))
        self.assertEqual(accepted, [{"content": "Likes Python", "reason": "Said so"},
                                    {"content": "ชอบภาษาไทย", "reason": "บอกเอง"}])
        self.assertEqual(dropped, 0)

    def test_malformed_responses_fail_closed(self):
        too_many = {"suggestions": [{"content": f"c{n}", "reason": "r"} for n in range(4)]}
        for text in ("not json", "[]", "{}", '{"suggestions": {}}', '{"suggestions": [1]}',
                     '{"suggestions": [{"content": "x"}]}',
                     '{"suggestions": [{"content": "x", "reason": "r", "extra": 1}]}',
                     '{"suggestions": [{"content": 1, "reason": "r"}]}',
                     '{"suggestions": [], "extra": 1}', json.dumps(too_many)):
            with self.subTest(text=text[:40]), self.assertRaises(AgentError) as raised:
                suggestions.parse_suggestions(text)
            self.assertNotIn("suggestions", str(raised.exception))  # no raw model output

    def test_invalid_items_are_dropped(self):
        cases = [("", "r"), ("x" * 281, "r"), ("ok", ""), ("ok", "r" * 161),
                 ("line\nbreak", "r"), ("ok", "bell\x07")]
        for item in cases:
            with self.subTest(item=repr(item)[:30]):
                self.assertEqual(self.parse(item), ([], 1))

    def test_exactly_280_and_160_are_allowed(self):
        accepted, dropped = self.parse(("x" * 280, "r" * 160))
        self.assertEqual((len(accepted), dropped), (1, 0))

    def test_credentials_are_dropped(self):
        secrets = ["sk-" + "a" * 20, "ghp_" + "b" * 24, "AKIA" + "C" * 16, "pass" + "word: hunter2",
                   "-----BEGIN RSA PRIVATE KEY-----", "a1" * 20]
        for secret in secrets:
            with self.subTest(secret=secret[:12]):
                self.assertEqual(self.parse((f"Note {secret}", "r")), ([], 1))
                self.assertEqual(self.parse(("Likes tea", f"because {secret}")), ([], 1))
        self.assertEqual(self.parse(("I like Python 3.13", "r"))[1], 0)

    def test_batch_duplicates_are_dropped(self):
        accepted, dropped = self.parse(("Likes Python", "a"), (" likes  python ", "b"))
        self.assertEqual(accepted, [{"content": "Likes Python", "reason": "a"}])
        self.assertEqual(dropped, 1)

    def test_empty_list_is_valid(self):
        self.assertEqual(suggestions.parse_suggestions('{"suggestions": []}'), ([], 0))


class SuggestTests(unittest.TestCase):
    def test_refusal_is_safe_error(self):
        refused = response("", output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])])
        client = scripted_client(refused)
        with self.assertRaises(AgentError):
            suggestions.suggest_memories(history("I like Python"), client=client)
        self.assertEqual(client.responses.create.call_count, 1)

    def test_all_filtered_result_is_normal_and_not_retried(self):
        client = scripted_client(reply(("sk-" + "a" * 20, "r"), ("", "r")))
        self.assertEqual(suggestions.suggest_memories(history("Hi"), client=client), ([], 2))
        self.assertEqual(client.responses.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
