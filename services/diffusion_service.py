import logging
from typing import Optional
from PIL import Image
from utils.device import get_device, HAS_CUDA
from services.model_config import get_task_config

logger = logging.getLogger(__name__)


class DiffusionService:
    """Centralized image diffusion service using HuggingFace diffusers."""

    def __init__(self):
        import torch

        task_config = get_task_config("image-generate")
        self.model_name = task_config.get(
            "model", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.device = get_device()
        self.dtype = torch.float16 if HAS_CUDA else torch.float32

        self._txt2img_pipe = None
        self._img2img_pipe = None

        logger.info(
            "DiffusionService initialized (model=%s, device=%s, dtype=%s)",
            self.model_name,
            self.device,
            self.dtype,
        )

    def _get_txt2img_pipe(self):
        if self._txt2img_pipe is not None:
            return self._txt2img_pipe

        import torch
        from diffusers import AutoPipelineForText2Image

        logger.info("Loading text-to-image pipeline: %s", self.model_name)
        self._txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            variant="fp16" if HAS_CUDA else None,
        )

        if HAS_CUDA:
            self._txt2img_pipe.enable_model_cpu_offload()
            self._txt2img_pipe.enable_attention_slicing()
            self._txt2img_pipe.enable_vae_slicing()
        else:
            self._txt2img_pipe = self._txt2img_pipe.to(self.device)

        logger.info("Text-to-image pipeline loaded")
        return self._txt2img_pipe

    def _get_img2img_pipe(self):
        if self._img2img_pipe is not None:
            return self._img2img_pipe

        import torch
        from diffusers import AutoPipelineForImage2Image

        # Try to reuse components from txt2img if already loaded
        if self._txt2img_pipe is not None:
            logger.info("Loading img2img pipeline from existing txt2img components")
            self._img2img_pipe = AutoPipelineForImage2Image.from_pipe(
                self._txt2img_pipe
            )
        else:
            logger.info("Loading image-to-image pipeline: %s", self.model_name)
            self._img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                variant="fp16" if HAS_CUDA else None,
            )

            if HAS_CUDA:
                self._img2img_pipe.enable_model_cpu_offload()
                self._img2img_pipe.enable_attention_slicing()
                self._img2img_pipe.enable_vae_slicing()
            else:
                self._img2img_pipe = self._img2img_pipe.to(self.device)

        logger.info("Image-to-image pipeline loaded")
        return self._img2img_pipe

    def _cleanup_vram(self):
        if HAS_CUDA:
            import torch
            torch.cuda.empty_cache()

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate an image from a text prompt."""
        import torch

        pipe = self._get_txt2img_pipe()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return result.images[0]
        finally:
            self._cleanup_vram()

    def image_to_image(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Edit an image using img2img with a text prompt."""
        import torch

        pipe = self._get_img2img_pipe()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure image is RGB
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            return result.images[0]
        finally:
            self._cleanup_vram()


# Singleton instance
_diffusion_service = None


def get_diffusion_service() -> DiffusionService:
    """Get the singleton diffusion service instance."""
    global _diffusion_service
    if _diffusion_service is None:
        _diffusion_service = DiffusionService()
    return _diffusion_service
