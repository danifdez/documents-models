import unittest

from services.llm_service import (
    _MAX_GRAMMAR_STRING_LENGTH,
    sanitize_tools_for_grammar,
)


def _tool(name, max_length):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "d",
            "parameters": {
                "type": "object",
                "required": ["goal"],
                "properties": {
                    "goal": {"type": "string", "maxLength": max_length},
                },
                "additionalProperties": False,
            },
        },
    }


class SanitizeToolsForGrammarTest(unittest.TestCase):
    def test_drops_max_length_above_the_grammar_limit(self):
        tools = [_tool("oversized", _MAX_GRAMMAR_STRING_LENGTH + 3000)]

        sanitized = sanitize_tools_for_grammar(tools)

        self.assertNotIn("maxLength", sanitized[0]["function"]["parameters"]["properties"]["goal"])

    def test_keeps_max_length_within_the_grammar_limit(self):
        tools = [_tool("bounded", _MAX_GRAMMAR_STRING_LENGTH)]

        sanitized = sanitize_tools_for_grammar(tools)

        self.assertEqual(
            sanitized[0]["function"]["parameters"]["properties"]["goal"]["maxLength"],
            _MAX_GRAMMAR_STRING_LENGTH,
        )

    def test_preserves_other_schema_fields_and_does_not_mutate_input(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "user_tasks.create",
                    "description": "Propose creating a local user task.",
                    "parameters": {
                        "type": "object",
                        "required": ["title"],
                        "properties": {
                            "title": {"type": "string", "minLength": 1, "maxLength": 200},
                            "description": {"type": "string", "maxLength": 4000},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ]

        sanitized = sanitize_tools_for_grammar(tools)
        properties = sanitized[0]["function"]["parameters"]["properties"]

        self.assertEqual(properties["title"], {"type": "string", "minLength": 1, "maxLength": 200})
        self.assertEqual(properties["description"], {"type": "string"})
        # The original definitions are untouched for the caller's own use.
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"]["description"]["maxLength"],
            4000,
        )


if __name__ == "__main__":
    unittest.main()
