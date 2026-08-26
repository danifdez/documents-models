import json
import unittest

from lib.execution.active_context import effective_payload_from_active_context


class ActiveContextTest(unittest.TestCase):
    def test_chat_uses_the_frozen_payload_instead_of_mutable_work(self):
        frozen = {
            "schemaVersion": "active-context/1",
            "effectivePayload": {
                "conversation": [{"role": "user", "content": "frozen"}],
                "folderScope": "/workspace/frozen",
            },
        }

        payload = effective_payload_from_active_context(
            "assistant-chat",
            {"conversation": [{"role": "user", "content": "mutable"}]},
            {"active_context": json.dumps(frozen).encode("utf-8")},
        )

        self.assertEqual(payload, frozen["effectivePayload"])

    def test_chat_rejects_a_missing_active_context(self):
        with self.assertRaisesRegex(ValueError, "Missing active context"):
            effective_payload_from_active_context("agent-chat", {}, {})

    def test_chat_rejects_an_unknown_active_context_version(self):
        body = json.dumps(
            {"schemaVersion": "active-context/2", "effectivePayload": {}}
        ).encode("utf-8")

        with self.assertRaisesRegex(ValueError, "Unsupported active context"):
            effective_payload_from_active_context(
                "assistant-chat", {}, {"active_context": body}
            )

    def test_non_chat_work_keeps_its_assignment_payload(self):
        payload = {"value": 7}

        self.assertEqual(
            effective_payload_from_active_context("detect-language", payload, {}),
            payload,
        )


if __name__ == "__main__":
    unittest.main()
