import unittest
from unittest.mock import ANY, patch

import setup_models


class SetupModelsTest(unittest.TestCase):
    @patch("setup_models.progress")
    @patch("setup_models.check_lora_files")
    @patch("setup_models.download_sequence_classifier")
    @patch("setup_models.load_tasks")
    def test_downloads_the_pinned_relationship_classifier(
        self,
        load_tasks,
        download_sequence_classifier,
        _check_lora_files,
        _progress,
    ):
        load_tasks.return_value = {
            "relationship-extraction-map": {
                "enabled": True,
                "type": "zero-shot-classification",
                "model": "example/xlm-roberta-large-xnli",
                "model_revision": "fixed-revision",
            }
        }

        setup_models.setup()

        download_sequence_classifier.assert_called_once_with(
            "example/xlm-roberta-large-xnli",
            ANY,
            revision="fixed-revision",
        )


if __name__ == "__main__":
    unittest.main()
