import hashlib
import io
import logging

from PIL import Image

from services.diffusion_service import get_diffusion_service
from services.model_config import get_task_config
from utils.job_registry import job_handler

logger = logging.getLogger(__name__)


@job_handler("image-edit")
def edit_image(payload: dict) -> dict:
    task_config = get_task_config("image-edit")

    source_hash = payload.get("sourceHash")
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

    input_blob = payload.get("_input_blob")
    if input_blob is None:
        raise ValueError("image-edit job is missing input_blob (source image)")

    init_image = Image.open(io.BytesIO(input_blob))
    logger.info(
        "Editing image: source=%s, prompt=%r, strength=%.2f",
        source_hash[:12] if source_hash else "<unknown>",
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

    buf = io.BytesIO()
    result_image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    hash_hex = hashlib.sha256(png_bytes).hexdigest()

    logger.info("Edited image: hash=%s, %d bytes", hash_hex[:12], len(png_bytes))

    return {
        "hash": hash_hex,
        "extension": ".png",
        "width": result_image.width,
        "height": result_image.height,
        "prompt": prompt,
        "requestId": request_id,
        "sourceHash": source_hash,
        "_result_blob": png_bytes,
    }
