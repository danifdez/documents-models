# Transcribe audio and video

Transcription converts speech in a supported audio or video file into text. It can run automatically after media metadata extraction when the capability is enabled.

## Supported formats

| Media | Extensions |
|---|---|
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.wma`, `.opus`, `.aiff`, `.aif` |
| Video | `.mp4`, `.m4v`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wmv` |

For video, Documents first extracts the audio track.

## Result

The result includes:

- the full transcript;
- the detected two-letter language code;
- a language-detection confidence from 0 to 1;
- the audio duration in seconds.

The transcription model is loaded when first needed and reused for later actions. Administrators can choose different model sizes and CPU or compatible GPU processing; larger models may improve results but require more resources.
