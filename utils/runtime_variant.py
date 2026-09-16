import shutil
import subprocess
import sys


def llama_build_metadata(llama_server: str) -> str:
    completed = subprocess.run(
        [llama_server, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    linker = "ldd" if sys.platform.startswith("linux") else "otool" if sys.platform == "darwin" else None
    if linker and shutil.which(linker):
        arguments = [linker, llama_server] if linker == "ldd" else [linker, "-L", llama_server]
        linked = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = f"{output}\n{linked.stdout}\n{linked.stderr}".strip()
    return output


def validate_llama_variant(
    variant: str,
    version_output: str,
    platform: str | None = None,
) -> tuple[bool, str]:
    current_platform = platform or sys.platform
    lowered = version_output.lower()
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
