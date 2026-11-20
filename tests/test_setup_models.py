import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, Mock, patch

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


class SetupModelDownloadTests(unittest.TestCase):
    def test_uses_overrides_and_falls_back_to_bundled_task_configuration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "common").mkdir()
            (root / "common" / "tasks.default.json").write_text(
                '{"embedding": {"enabled": false}}', encoding="utf-8",
            )
            with patch.object(setup_models, "SCRIPT_DIR", directory):
                self.assertEqual(setup_models.load_tasks(), {"embedding": {"enabled": False}})
                (root / "config").mkdir()
                (root / "config" / "tasks.json").write_text(
                    '{"embedding": {"enabled": true}}', encoding="utf-8",
                )
                self.assertEqual(setup_models.load_tasks(), {"embedding": {"enabled": True}})

    def test_downloader_continues_after_a_failure_and_deduplicates_shared_models(self):
        tasks = {
            "embedding": {"enabled": True, "model": "passage-model"},
            "entity-extraction-map": {
                "enabled": True, "type": "token-classification", "model": "entity-model", "model_revision": "v1",
            },
            "transcribe": {"enabled": True, "model": "tiny"},
            "chat": {"enabled": True, "type": "llm", "model": "shared.gguf"},
            "summary": {"enabled": True, "type": "llm", "model": "shared.gguf"},
            "disabled": {"enabled": False, "type": "llm", "model": "disabled.gguf"},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(setup_models, "load_tasks", return_value=tasks),
            patch.object(setup_models, "download_embedding", side_effect=SystemExit("offline")) as embedding,
            patch.object(setup_models, "download_token_classifier") as entities,
            patch.object(setup_models, "download_whisper") as whisper,
            patch.object(setup_models, "download_gguf", side_effect=lambda name, report: report(75)) as gguf,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            setup_models.setup()

        embedding.assert_called_once_with("passage-model", ANY)
        entities.assert_called_once_with("entity-model", ANY, revision="v1")
        whisper.assert_called_once_with("tiny", ANY)
        gguf.assert_called_once_with("shared.gguf", ANY)
        self.assertIn("Failed to download embeddings (passage-model): offline", stderr.getvalue())
        progress = [int(line.rsplit(":", 1)[1]) for line in stdout.getvalue().splitlines() if line.startswith("PROGRESS:")]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual((progress[0], progress[-1]), (0, 100))

    def test_missing_adapter_warning_only_applies_to_enabled_llm_tasks(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "present.gguf").touch()
            missing_path = str(Path(directory) / "missing.gguf")
            stderr = io.StringIO()
            with patch.dict(os.environ, {"MODELS_MODEL_DIR": directory}), redirect_stderr(stderr):
                setup_models.check_lora_files({
                    "enabled": {"enabled": True, "type": "llm", "lora_model": "missing.gguf"},
                    "present": {"enabled": True, "type": "llm", "lora_model": "present.gguf"},
                    "ignored": {"enabled": False, "type": "llm", "lora_model": "disabled.gguf"},
                })
            self.assertIn("LoRA for task 'enabled' not found", stderr.getvalue())
            self.assertIn(missing_path, stderr.getvalue())
            self.assertNotIn("present.gguf", stderr.getvalue())
            self.assertNotIn("disabled.gguf", stderr.getvalue())

    def test_gguf_download_reuses_a_local_model_without_contacting_the_hub(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "local.gguf").touch()
            report = Mock()
            with (
                patch.dict(os.environ, {"MODELS_MODEL_DIR": directory}),
                patch("huggingface_hub.hf_hub_download") as download,
                redirect_stdout(io.StringIO()),
            ):
                setup_models.download_gguf("local.gguf", report)
            download.assert_not_called()
            report.assert_called_once_with(100)

    def test_gguf_download_keeps_running_when_remote_size_is_unavailable(self):
        with TemporaryDirectory() as directory:
            report = Mock()
            with (
                patch.dict(os.environ, {"MODELS_MODEL_DIR": directory}),
                patch("huggingface_hub.get_hf_file_metadata", side_effect=OSError("size unavailable")),
                patch("huggingface_hub.hf_hub_download") as download,
            ):
                setup_models.download_gguf("model.gguf", report)
            download.assert_called_once_with(
                repo_id="Qwen/Qwen3-8B-GGUF", filename="model.gguf", local_dir=directory,
            )
            report.assert_called_once_with(100)


if __name__ == "__main__":
    unittest.main()
