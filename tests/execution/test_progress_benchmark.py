import unittest

from tests.execution.benchmark_progress_overhead import (
    build_report,
    summarize_scenario,
)


def sample(mode, elapsed, content="same", *, correct=True, instrumentation=0):
    return {
        "mode": mode,
        "elapsedMs": elapsed,
        "instrumentationMs": instrumentation,
        "content": content,
        "correct": correct,
    }


class ProgressBenchmarkSummaryTest(unittest.TestCase):
    def test_uses_only_the_requested_correct_samples_and_applies_the_threshold(self):
        attempts = [
            sample("baseline", 9999, correct=False),
            sample("baseline", 1000),
            sample("instrumented", 1130, instrumentation=80),
            sample("instrumented", 9999, correct=False),
            sample("baseline", 1020),
            sample("instrumented", 1150, instrumentation=85),
        ]

        summary = summarize_scenario(attempts, 2)

        self.assertEqual(summary["baselineMs"], [1000, 1020])
        self.assertEqual(summary["instrumentedMs"], [1130, 1150])
        self.assertEqual(summary["baselineMedianMs"], 1010)
        self.assertEqual(summary["deltaMs"], 130)
        self.assertEqual(summary["thresholdMs"], 150)
        self.assertTrue(summary["passed"])

    def test_report_fails_when_semantic_outputs_diverge(self):
        report = build_report(
            {
                "divergent": [
                    sample("baseline", 500, "before"),
                    sample("instrumented", 510, "after"),
                ],
            },
            "model.gguf",
            1,
        )

        self.assertFalse(report["scenarios"]["divergent"]["semanticMatch"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
