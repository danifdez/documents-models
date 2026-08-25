import json
import unittest

from lib.execution.step_executor import execute_assignment


class DatasetAnalysisAssignmentTest(unittest.TestCase):
    def assignment(self, task_type, payload):
        return {
            "executionId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca701",
            "stepId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca702",
            "operationId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca703",
            "attemptId": "018f1d8a-54d7-7d63-a1ee-5e9a6adca704",
            "stepKind": "code",
            "work": {"taskType": task_type, "payload": payload},
        }

    def artifact(self, datasets):
        return {
            "datasets": json.dumps(
                {
                    "schemaVersion": "dataset-analysis-input/1",
                    "datasets": datasets,
                }
            ).encode("utf-8")
        }

    def test_distribution_uses_records_and_linked_labels_from_artifact(self):
        result = execute_assignment(
            self.assignment(
                "distribution",
                {"datasetId": 1, "params": {"field": "category_id"}},
            ),
            self.artifact(
                [
                    {
                        "datasetId": 1,
                        "schema": [
                            {
                                "key": "category_id",
                                "name": "Category",
                                "type": "text",
                                "linkedLabels": {"20": "Research"},
                            }
                        ],
                        "records": [
                            {"id": 1, "data": {"category_id": 20}},
                            {"id": 2, "data": {"category_id": 20}},
                            {"id": 3, "data": {"category_id": 30}},
                        ],
                    }
                ]
            ),
        )

        self.assertEqual(result["status"], "succeeded")
        value = result["output"]["value"]
        self.assertEqual(value["chartData"]["labels"], ["Research", "30"])
        self.assertEqual(value["chartData"]["values"], [2, 1])

    def test_query_joins_multiple_datasets_from_one_artifact(self):
        result = execute_assignment(
            self.assignment(
                "query",
                {
                    "datasetIds": [1, 2],
                    "params": {"joinField": "code", "select": ["value_r"]},
                },
            ),
            self.artifact(
                [
                    {
                        "datasetId": 1,
                        "schema": [
                            {"key": "code", "name": "Code", "type": "text"}
                        ],
                        "records": [
                            {
                                "id": 1,
                                "data": {"code": "A", "left": 1, "value": 3},
                            }
                        ],
                    },
                    {
                        "datasetId": 2,
                        "schema": [
                            {"key": "code", "name": "Code", "type": "text"}
                        ],
                        "records": [
                            {"id": 2, "data": {"code": "A", "value": 9}}
                        ],
                    },
                ]
            ),
        )

        self.assertEqual(result["status"], "succeeded")
        value = result["output"]["value"]
        self.assertEqual(value["tableData"]["columns"], ["value_r"])
        self.assertEqual(value["tableData"]["rows"], [[9]])

    def test_rejects_an_assignment_without_its_dataset_artifact(self):
        result = execute_assignment(
            self.assignment(
                "summary",
                {"datasetId": 1, "params": {}},
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "STEP_EXECUTION_FAILED")

    def test_all_dataset_analysis_handlers_accept_the_snapshot_contract(self):
        snapshot = {
            "datasetId": 1,
            "schema": [
                {"key": "category", "name": "Category", "type": "text"},
                {"key": "group", "name": "Group", "type": "text"},
                {"key": "value", "name": "Value", "type": "number"},
                {"key": "other", "name": "Other", "type": "number"},
                {"key": "date", "name": "Date", "type": "date"},
            ],
            "records": [
                {
                    "id": index,
                    "data": {
                        "category": "A" if index < 4 else "B",
                        "group": "X" if index % 2 else "Y",
                        "value": value,
                        "other": value * 2,
                        "date": f"2026-0{index}-01",
                    },
                }
                for index, value in enumerate([1, 2, 3, 4, 20], start=1)
            ],
        }
        cases = {
            "distribution": {"field": "category"},
            "correlation": {"field1": "value", "field2": "other"},
            "correlation-matrix": {"fields": ["value", "other"]},
            "group-by": {
                "valueField": "value",
                "groupField": "category",
                "fn": "mean",
            },
            "time-series": {"dateField": "date", "valueField": "value"},
            "outliers": {"field": "value"},
            "pivot-table": {
                "rowField": "category",
                "colField": "group",
                "valueField": "value",
                "fn": "sum",
            },
            "summary": {},
            "query": {"select": ["value"]},
            "chart": {
                "chartType": "bar",
                "xField": "category",
                "yField": "value",
                "aggregation": "mean",
            },
        }

        for task_type, params in cases.items():
            with self.subTest(task_type=task_type):
                result = execute_assignment(
                    self.assignment(
                        task_type,
                        {"datasetId": 1, "params": params},
                    ),
                    self.artifact([snapshot]),
                )
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(result["output"]["kind"], "code")


if __name__ == "__main__":
    unittest.main()
