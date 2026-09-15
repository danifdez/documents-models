import unittest

from utils.runtime_variant import validate_llama_variant


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
