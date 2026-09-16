#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.runtime_variant import llama_build_metadata, validate_llama_variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("cpu", "cuda", "metal"), required=True)
    parser.add_argument("--llama-server", required=True)
    args = parser.parse_args()

    try:
        import sentence_transformers
        import torch
        import torchvision
        import transformers
        from torchvision.ops import nms
    except Exception as error:
        raise RuntimeError(
            "ML runtime imports are inconsistent; verify the resolved Torch, "
            "torchvision, Transformers and sentence-transformers versions"
        ) from error

    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1]])
    scores = torch.tensor([0.9, 0.8])
    nms(boxes, scores, 0.5)

    engine_output = llama_build_metadata(args.llama_server)
    torch_cuda = torch.version.cuda

    if args.variant == "cpu" and torch_cuda:
        raise RuntimeError(f"CPU bundle contains CUDA Torch {torch_cuda}")
    if args.variant == "cuda" and not torch_cuda:
        raise RuntimeError("CUDA bundle contains a CPU-only Torch build")
    compatible, reason = validate_llama_variant(args.variant, engine_output)
    if not compatible:
        raise RuntimeError(reason)

    print(json.dumps({
        "variant": args.variant,
        "python": sys.version.split()[0],
        "torch": {"version": torch.__version__, "cuda": torch_cuda},
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "sentenceTransformers": sentence_transformers.__version__,
        "llamaServer": {"path": "llama/llama-server", "build": engine_output},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
