import unittest

from common.execution_registry import TASK_HANDLERS
from utils.task_dispatch import ensure_task_handler


class DataSourceSyncTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ensure_task_handler("data-source-sync"))
        self.handler = TASK_HANDLERS["data-source-sync"]

    def test_validates_and_returns_the_backend_sync_assignment(self):
        self.assertEqual(
            self.handler({"dataSourceId": 4}),
            {"dataSourceId": 4, "ready": True},
        )

    def test_rejects_invalid_data_source_ids(self):
        for value in (None, True, 0, -1, "4"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.handler({"dataSourceId": value})


if __name__ == "__main__":
    unittest.main()
