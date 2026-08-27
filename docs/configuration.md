# AI feature options

The administrator of a Documents installation controls which processing capabilities are available and how much local computer capacity they may use. These are installation-level choices; normal users start the corresponding actions from the desktop application.

## Enabled capabilities

Capabilities can be enabled or disabled individually. Disabling language-model processing removes actions that depend on text generation, while disabling embeddings removes semantic indexing, semantic search, and related question-answering work.

The rest of Documents remains usable when optional AI capabilities are disabled.

## Processing capacity

The service runs a limited number of assignments at the same time. The default is two. Each slot has an isolated handler process so a cancelled calculation can be terminated without stopping other work. Increasing the limit can process more work concurrently, but can also keep more model state in processor or GPU memory. Documents will leave additional work queued until a slot becomes available.

## Language-model options

Text-generation actions can use a locally installed model and can have separate limits for input size, output length, and the number of returned items. Administrators can also provide task-specific prompts or approved model adaptations.

Changing these choices can affect speed, memory use, language coverage, and output quality, but does not change how users start the action or where the result appears.

## Semantic search options

Administrators can choose:

- the default number of matching passages;
- the minimum relevance score;
- the maximum answer length;
- the target and maximum size of indexed passages;
- how much neighboring passages overlap.

The current defaults retrieve five passages, require a relevance score of 0.35, use a bounded answer length, target 150 words per passage, cap passages at 250 words, and overlap them by 30 words.

## Model downloads

Most supporting models are downloaded on first use or during installation. Translation uses a separate model for each language pair. If a required model is unavailable, the affected action fails explicitly rather than silently returning a substitute result.

Standalone installations can choose a CPU or compatible NVIDIA GPU package from the desktop application's Local Server settings.
