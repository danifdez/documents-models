import hashlib
import io
import logging
import os

from PIL import Image

from config import DOCUMENTS_STORAGE_DIR
from services.diffusion_service import get_diffusion_service
from services.model_config import get_task_config
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


def _storage_path(hash_hex: str, extension: str) -> str:
    return os.path.join(
        DOCUMENTS_STORAGE_DIR,
        hash_hex[:3],
        hash_hex[3:6],
        hash_hex + extension,
    )


@job_handler("image-edit")
def edit_image(payload: dict) -> dict:
    task_config = get_task_config("image-edit")

    source_hash = payload.get("sourceHash")
    source_relative_path = payload.get("sourcePath")
    source_extension = payload.get("sourceExtension", ".png")
    prompt = payload["prompt"]
    negative_prompt = payload.get("negativePrompt")
    strength = payload.get(
        "strength", task_config.get("default_strength", 0.75)
    )
    steps = payload.get("steps", task_config.get("default_steps", 30))
    guidance_scale = payload.get(
        "guidanceScale", task_config.get("default_guidance_scale", 7.5)
    )
    seed = payload.get("seed")
    request_id = payload.get("requestId")

    # Load source image — try hash-based path first, fall back to relative path
    source_path = None
    if source_hash:
        source_path = _storage_path(source_hash, source_extension)
    if (not source_path or not os.path.exists(source_path)) and source_relative_path:
        source_path = os.path.join(DOCUMENTS_STORAGE_DIR, source_relative_path)
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image not found (hash={source_hash}, path={source_relative_path})")

    init_image = Image.open(source_path)
    logger.info(
        "Editing image: source=%s, prompt=%r, strength=%.2f",
        source_hash[:12] if source_hash else source_path,
        prompt[:80],
        strength,
    )

    service = get_diffusion_service()
    result_image = service.image_to_image(
        init_image=init_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # Save to PNG bytes and compute hash
    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    hash_hex = hashlib.sha256(png_bytes).hexdigest()
    extension = ".png"
    filename = hash_hex + extension
    relative_path = os.path.join("temp", filename)
    file_path = os.path.join(DOCUMENTS_STORAGE_DIR, relative_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(png_bytes)

    logger.info("Edited image saved to temp: %s (%d bytes)", file_path, len(png_bytes))

    return {
        "hash": hash_hex,
        "relativePath": relative_path,
        "extension": extension,
        "width": result_image.width,
        "height": result_image.height,
        "prompt": prompt,
        "requestId": request_id,
        "sourceHash": source_hash,
    }
