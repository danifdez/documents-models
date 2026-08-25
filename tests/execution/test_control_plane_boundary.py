import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_PLANE_TABLES = {
    "executions",
    "execution_steps",
    "execution_step_dependencies",
    "execution_step_attempts",
    "execution_result_receipts",
    "execution_events",
    "execution_artifacts",
    "execution_operations",
    "execution_tool_plans",
    "execution_tool_invocations",
    "execution_outbox",
    "workers",
}


class ControlPlaneBoundaryTest(unittest.TestCase):
    def test_models_has_no_execution_database_adapter(self):
        self.assertFalse((PROJECT_ROOT / "database" / "execution.py").exists())

        database_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((PROJECT_ROOT / "database").glob("*.py"))
        )
        referenced = {
            table
            for table in CONTROL_PLANE_TABLES
            if re.search(rf"\b{re.escape(table)}\b", database_source)
        }
        self.assertEqual(referenced, set())

    def test_models_configuration_has_no_queue_settings(self):
        for relative in ("common/config.default.json", "config/config.json"):
            config = json.loads(
                (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            )
            self.assertNotIn("executions_table", config.get("database", {}))
            self.assertNotIn(
                "background_hours_start", config.get("worker", {})
            )
            self.assertNotIn(
                "background_hours_end", config.get("worker", {})
            )


if __name__ == "__main__":
    unittest.main()
