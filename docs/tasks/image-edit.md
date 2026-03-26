## Image Edit

The **image-edit** task modifies an existing image based on a text prompt using image-to-image diffusion (img2img). It loads a source image from storage, applies the diffusion model with the given prompt and strength, and saves the result as a new PNG file.

### What it does

Given a source image (identified by its storage hash) and a text prompt, the task uses the Stable Diffusion img2img pipeline to generate a variation of the original image guided by the prompt. The `strength` parameter controls how much the output deviates from the input — lower values produce subtle edits, higher values produce more dramatic changes.

This task is triggered from the Canvas AI Image panel when the user selects an existing image node and clicks "Apply Edit".

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sourceHash` | string | Yes | — | SHA-256 hash of the source image file |
| `sourceExtension` | string | No | `.png` | File extension of the source image |
| `prompt` | string | Yes | — | Text description of the desired modification |
| `negativePrompt` | string | No | `null` | What to avoid in the edited image |
| `strength` | float | No | `0.75` | How much to change the image (0.1 = subtle, 1.0 = complete redraw) |
| `steps` | int | No | `30` | Number of diffusion inference steps |
| `guidanceScale` | float | No | `7.5` | Classifier-free guidance scale |
| `seed` | int | No | Random | Seed for reproducible generation |
| `requestId` | string | No | `null` | Client-generated UUID for correlating the WebSocket response |
| `canvasId` | int | No | `null` | Canvas where the image will be used |
| `projectId` | int | No | `null` | Project to associate the generated resource with |

### Returns

```json
{
  "hash": "f4a3b2c1d0e9...",
  "relativePath": "f4a/3b2/f4a3b2c1d0e9...png",
  "extension": ".png",
  "width": 1024,
  "height": 1024,
  "prompt": "Add a sunset sky",
  "requestId": "uuid",
  "sourceHash": "a1b2c3d4e5f6..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hash` | string | SHA-256 hash of the edited PNG bytes |
| `relativePath` | string | Path relative to the documents storage directory |
| `extension` | string | Always `.png` |
| `width` | int | Width of the edited image |
| `height` | int | Height of the edited image |
| `prompt` | string | The prompt used for editing |
| `requestId` | string | Passed through from the input for client correlation |
| `sourceHash` | string | Hash of the original source image |

### Configuration

Configured in `config/tasks.json` under the `image-edit` key:

```json
{
  "image-edit": {
    "enabled": true,
    "type": "diffusion",
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "capabilities": ["gpu"],
    "default_steps": 30,
    "default_strength": 0.75,
    "default_guidance_scale": 7.5
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `stabilityai/stable-diffusion-xl-base-1.0` | HuggingFace model ID (shares weights with image-generate) |
| `default_steps` | `30` | Default number of inference steps |
| `default_strength` | `0.75` | Default edit strength |
| `default_guidance_scale` | `7.5` | Default guidance scale |

### Requirements

Same as [image-generate](./image-generate.md). Both tasks share the same `DiffusionService` singleton — loading the img2img pipeline reuses weights from the text-to-image pipeline if it was already loaded.

### Backend processing

When the Python worker completes the job, the backend `ImageEditProcessor`:

1. Creates a `ResourceEntity` with `type: image` and `mimeType: image/png`, linked to the project
2. Sends an `imageEditResponse` WebSocket event with the `resourceId`

### Example

**Input:**

```json
{
  "sourceHash": "a1b2c3d4e5f6789abcdef1234567890a",
  "sourceExtension": ".png",
  "prompt": "Transform into a watercolor painting style",
  "strength": 0.6,
  "steps": 30,
  "guidanceScale": 7.5,
  "requestId": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Output:**

```json
{
  "hash": "f4a3b2c1d0e98765abcdef1234567890b",
  "relativePath": "f4a/3b2/f4a3b2c1d0e98765abcdef1234567890b.png",
  "extension": ".png",
  "width": 1024,
  "height": 1024,
  "prompt": "Transform into a watercolor painting style",
  "requestId": "550e8400-e29b-41d4-a716-446655440001",
  "sourceHash": "a1b2c3d4e5f6789abcdef1234567890a"
}
```
