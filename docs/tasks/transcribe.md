## Transcribe

The **transcribe** task converts speech from audio and video files into text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2-based reimplementation of OpenAI's Whisper model.

### What it does

Given a file identified by its hash and extension, the task loads the audio (extracting it from video files via `ffmpeg` if needed) and runs the Whisper model to produce a text transcript. The model is loaded lazily on first use and cached as a singleton for subsequent calls.

This task runs automatically after `document-extraction` completes for media files.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `hash` | string | Yes | The file's storage hash (used to locate the file on disk) |
| `extension` | string | Yes | The file extension including the dot (e.g. `.mp3`, `.mp4`) |

**Supported extensions:**

| Format | Extensions |
|--------|-----------|
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.opus`, `.aiff`, `.aif` |
| Video | `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` |

### Returns

```json
{
  "transcript": "The transcribed text content...",
  "language": "en",
  "language_probability": 0.98,
  "duration": 125.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `transcript` | string | The full transcribed text |
| `language` | string | Detected language code (ISO 639-1) |
| `language_probability` | float | Confidence of the language detection (0–1) |
| `duration` | float | Audio duration in seconds |

### Configuration

Configured in `config/tasks.json` under the `transcribe` key:

```json
{
  "transcribe": {
    "enabled": true,
    "type": "whisper",
    "model": "base",
    "device": "cpu",
    "compute_type": "int8",
    "beam_size": 5
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `device` | `cpu` | `cpu` or `cuda` |
| `compute_type` | `int8` | `int8` (CPU), `float16` (GPU), or `float32` |
| `beam_size` | `5` | Beam search width for decoding |

### Requirements

- **faster-whisper** Python package (included in `requirements.txt`)
- **ffmpeg** system binary (required for video files)

### Example

**Input:**

```json
{
  "hash": "a1b2c3d4e5f6789abcdef1234567890a",
  "extension": ".mp3"
}
```

**Output:**

```json
{
  "transcript": "Welcome to the presentation. Today we will discuss the main findings of our research project.",
  "language": "en",
  "language_probability": 0.97,
  "duration": 342.5
}
```
