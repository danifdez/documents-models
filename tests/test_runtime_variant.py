import unittest
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils.runtime_variant import llama_build_metadata, validate_llama_variant


class RuntimeVariantTest(unittest.TestCase):
    def test_cpu_rejects_cuda_build(self):
        compatible, reason = validate_llama_variant(
            "cpu",
            "ggml_cuda_init: CUDA device detected",
            "linux",
        )

        self.assertFalse(compatible)
        self.assertIn("CUDA-enabled", reason)

    def test_cpu_accepts_cpu_build(self):
        compatible, reason = validate_llama_variant(
            "cpu",
            "built with GNU for Linux x86_64",
            "linux",
        )

        self.assertTrue(compatible)
        self.assertEqual(reason, "")

    def test_cpu_ignores_backend_names_in_library_directories(self):
        compatible, reason = validate_llama_variant(
            "cpu",
            "libllama.so => /builds/cuda/metal/libllama.so",
            "linux",
        )

        self.assertTrue(compatible)
        self.assertEqual(reason, "")

    def test_cuda_accepts_backend_loaded_dynamically_next_to_binary(self):
        with TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            binary = runtime / "llama-server"
            binary.touch()
            (runtime / "libggml-cuda.so").touch()
            alias = Path(directory) / "llama-server"
            try:
                alias.symlink_to(binary)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with (
                patch("utils.runtime_variant.sys.platform", "linux"),
                patch("utils.runtime_variant.shutil.which", side_effect=lambda name: name),
                patch("utils.runtime_variant.subprocess.run", side_effect=[
                    CompletedProcess([], 0, "version: 1.1.0", ""),
                    CompletedProcess([], 0, "libllama.so", ""),
                ]) as run,
            ):
                metadata = llama_build_metadata(str(alias))

        self.assertEqual(run.call_args_list[1].args[0], ["ldd", str(binary.resolve())])
        self.assertEqual(validate_llama_variant("cuda", metadata, "linux"), (True, ""))

    def test_cuda_requires_cuda_metadata(self):
        compatible, reason = validate_llama_variant(
            "cuda",
            "built with GNU for Linux x86_64",
            "linux",
        )

        self.assertFalse(compatible)
        self.assertIn("without CUDA", reason)

    def test_cuda_accepts_linked_cuda_backend(self):
        compatible, reason = validate_llama_variant(
            "cuda",
            "version: 10243\nlibggml-cuda.so => /usr/lib/libggml-cuda.so",
            "linux",
        )

        self.assertTrue(compatible)
        self.assertEqual(reason, "")

    def test_metal_requires_macos_and_metal_metadata(self):
        compatible, reason = validate_llama_variant(
            "metal",
            "Metal backend enabled",
            "darwin",
        )

        self.assertTrue(compatible)
        self.assertEqual(reason, "")

    def test_unknown_variant_is_rejected(self):
        compatible, reason = validate_llama_variant(
            "unknown",
            "built with GNU for Linux x86_64",
            "linux",
        )

        self.assertFalse(compatible)
        self.assertIn("Unsupported", reason)


if __name__ == "__main__":
    unittest.main()
