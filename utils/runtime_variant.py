import shutil
import subprocess
import sys
from pathlib import Path


def llama_build_metadata(llama_server: str) -> str:
    completed = subprocess.run(
        [llama_server, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = [completed.stdout, completed.stderr]
    binary = Path(shutil.which(llama_server) or llama_server).resolve()
    linker = None
    if sys.platform.startswith("linux"):
        linker = "ldd"
    elif sys.platform == "darwin":
        linker = "otool"
    if linker and shutil.which(linker):
        arguments = [linker, str(binary)] if linker == "ldd" else [linker, "-L", str(binary)]
        linked = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output.extend((linked.stdout, linked.stderr))
    # llama.cpp can load its GPU backends at runtime instead of linking them.
    for backend in ("cuda", "metal"):
        output.extend(
            f"backend library: {library.name}"
            for library in sorted(binary.parent.glob(f"*ggml-{backend}.*"))
            if library.is_file()
        )
    return "\n".join(part.strip() for part in output if part.strip())


def validate_llama_variant(
    variant: str,
    version_output: str,
    platform: str | None = None,
) -> tuple[bool, str]:
    current_platform = platform or sys.platform
    lowered = "\n".join(
        line.split(" => ", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        for line in version_output.lower().splitlines()
    )
    has_cuda = "cuda" in lowered or "ggml_cuda" in lowered
    has_metal = "metal" in lowered

    if variant == "cpu":
        if has_cuda:
            return False, "CPU bundle contains a CUDA-enabled llama-server"
        if has_metal:
            return False, "CPU bundle contains a Metal-enabled llama-server"
        return True, ""
    if variant == "cuda":
        if not has_cuda:
            return False, "CUDA bundle contains a llama-server without CUDA build metadata"
        return True, ""
    if variant == "metal":
        if current_platform != "darwin":
            return False, "Metal bundles can only run on macOS"
        if not has_metal:
            return False, "Metal bundle contains a llama-server without Metal build metadata"
        return True, ""
    return False, f"Unsupported Models variant: {variant}"
