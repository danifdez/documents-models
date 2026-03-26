## Image Generate

The **image-generate** task creates images from text descriptions using a local Stable Diffusion model via the HuggingFace `diffusers` library. Generated images are saved as PNG files in the shared document storage.

### What it does

Given a text prompt (and optional parameters), the task loads a Stable Diffusion pipeline, generates an image, and writes the result to disk using the same content-addressed storage scheme as uploaded files. The diffusion model is loaded lazily on first use and cached as a singleton for subsequent calls.

This task is triggered from the Canvas AI Image panel when the user clicks "Generate".

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | — | Text description of the image to generate |
| `negativePrompt` | string | No | `null` | What to avoid in the generated image |
| `width` | int | No | `1024` | Image width in pixels (512, 768, or 1024) |
| `height` | int | No | `1024` | Image height in pixels (512, 768, or 1024) |
| `steps` | int | No | `30` | Number of diffusion inference steps (10–50) |
| `guidanceScale` | float | No | `7.5` | Classifier-free guidance scale (1–20) |
| `seed` | int | No | Random | Seed for reproducible generation |
| `requestId` | string | No | `null` | Client-generated UUID for correlating the WebSocket response |
| `canvasId` | int | No | `null` | Canvas where the image will be used |
| `projectId` | int | No | `null` | Project to associate the generated resource with |

### Returns

```json
{
  "hash": "a1b2c3d4e5f6...",
  "relativePath": "a1b/2c3/a1b2c3d4e5f6...png",
  "extension": ".png",
  "width": 1024,
  "height": 1024,
  "prompt": "A mountain landscape at sunset",
  "requestId": "uuid"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hash` | string | SHA-256 hash of the generated PNG bytes |
| `relativePath` | string | Path relative to the documents storage directory |
| `extension` | string | Always `.png` |
| `width` | int | Width of the generated image |
| `height` | int | Height of the generated image |
| `prompt` | string | The prompt used for generation |
| `requestId` | string | Passed through from the input for client correlation |

### Configuration

Configured in `config/tasks.json` under the `image-generate` key:

```json
{
  "image-generate": {
    "enabled": true,
    "type": "diffusion",
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "capabilities": ["gpu"],
    "default_steps": 30,
    "default_guidance_scale": 7.5,
    "default_width": 1024,
    "default_height": 1024
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `stabilityai/stable-diffusion-xl-base-1.0` | HuggingFace model ID for the diffusion pipeline |
| `default_steps` | `30` | Default number of inference steps |
| `default_guidance_scale` | `7.5` | Default classifier-free guidance scale |
| `default_width` | `1024` | Default image width |
| `default_height` | `1024` | Default image height |

### Requirements

- **GPU with 7+ GB VRAM** — SDXL requires approximately 7 GB in float16. A lighter model like `runwayml/stable-diffusion-v1-5` (~4 GB) can be configured as an alternative.
- **diffusers**, **accelerate**, **safetensors** Python packages (included in `requirements.txt`)
- **torch** with CUDA support for GPU acceleration

The model weights (~6.5 GB for SDXL) are downloaded automatically from HuggingFace on first use. The first generation will be slower due to this download.

### Backend processing

When the Python worker completes the job, the backend `ImageGenerateProcessor`:

1. Creates a `ResourceEntity` with `type: image` and `mimeType: image/png`, linked to the project
2. Sends an `imageGenerateResponse` WebSocket event with the `resourceId` so the frontend can display the image

### Example

**Input:**

```json
{
  "prompt": "A serene mountain landscape at sunset with a reflective lake",
  "negativePrompt": "blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidanceScale": 7.5,
  "requestId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Output:**

```json
{
  "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "relativePath": "e3b/0c4/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.png",
  "extension": ".png",
  "width": 1024,
  "height": 1024,
  "prompt": "A serene mountain landscape at sunset with a reflective lake",
  "requestId": "550e8400-e29b-41d4-a716-446655440000"
}
```
