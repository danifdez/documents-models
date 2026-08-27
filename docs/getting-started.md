# Getting started with AI features

AI features are used from the Documents desktop application. There is no separate Models application to open.

## Check availability

In a standalone workspace, open **Settings → Local Server** and check whether AI Features are installed. If not, choose the CPU or compatible NVIDIA GPU package. The download is approximately 2–5 GB.

In a remote workspace, the server administrator controls installation and availability. Your account must also have permission for protected actions such as summarization, translation, entity extraction, key points, keywords, and question answering.

## Try the first action

1. Open a project.
2. Import a supported document.
3. Wait for document extraction to complete.
4. Open the resource and start an available action, such as language detection or summarization.
5. Continue working while the action runs.
6. Open the completion notification to review the result.

Semantic search and question answering require content to be indexed first. Audio and video transcription requires the optional transcription capability.

## First-use downloads

Some models are downloaded only when a capability is first used. The first run can therefore take longer and requires sufficient disk space and, for remote installations, any network access allowed by the administrator.

Translation uses a separate model for each language pair. A pair that has not been installed may require an additional download; if no suitable model is available, the translation fails explicitly.

## Understanding queued or failed actions

- **Queued**: a compatible processor is busy or not currently online.
- **Failed immediately**: required content, a supported format, permission, or model may be missing.
- **Failed after processing**: the processor could not produce a valid result.
- **Cancelled**: an active local calculation is stopped and its result is not applied.

The rest of the project remains usable when an optional AI action is unavailable.
