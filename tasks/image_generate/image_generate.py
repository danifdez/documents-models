import hashlib
import io
import logging
import os

from config import DOCUMENTS_STORAGE_DIR
from services.diffusion_service import get_diffusion_service
from services.model_config import get_task_config
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


@job_handler("image-generate")
def generate_image(payload: dict) -> dict:
    task_config = get_task_config("image-generate")

    prompt = payload["prompt"]
    negative_prompt = payload.get("negativePrompt")
    width = payload.get("width", task_config.get("default_width", 1024))
    height = payload.get("height", task_config.get("default_height", 1024))
    steps = payload.get("steps", task_config.get("default_steps", 30))
    guidance_scale = payload.get(
        "guidanceScale", task_config.get("default_guidance_scale", 7.5)
    )
    seed = payload.get("seed")
    request_id = payload.get("requestId")

    logger.info(
        "Generating image: prompt=%r, size=%dx%d, steps=%d",
        prompt[:80],
        width,
        height,
        steps,
    )

    service = get_diffusion_service()
    image = service.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # Save to PNG bytes and compute hash
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    hash_hex = hashlib.sha256(png_bytes).hexdigest()
    extension = ".png"
    filename = hash_hex + extension
    relative_path = os.path.join("temp", filename)
    file_path = os.path.join(DOCUMENTS_STORAGE_DIR, relative_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(png_bytes)

    logger.info("Image saved to temp: %s (%d bytes)", file_path, len(png_bytes))

    return {
        "hash": hash_hex,
        "relativePath": relative_path,
        "extension": extension,
        "width": image.width,
        "height": image.height,
        "prompt": prompt,
        "requestId": request_id,
    }
