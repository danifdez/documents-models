# Creating New Tasks

This guide explains how to add a new task to the models worker.

## Task Structure

Each task lives in its own directory under `tasks/`:

```
tasks/
  my_task/
    my_task.py      # Handler implementation
    prompt.md       # Default prompt template (optional)
```

## Step by Step

### 1. Create the task directory and handler

```bash
mkdir -p tasks/my_task
```

Create `tasks/my_task/my_task.py`:

```python
from utils.job_registry import job_handler
from services.model_config import get_task_config


@job_handler("my-task")
def my_task(payload) -> dict:
    task_config = get_task_config("my-task")
    text = payload.get("content", "")

    # Your task logic here
    result = process(text)

    return {"result": result}
```

The `@job_handler("my-task")` decorator registers the function to handle jobs of type `"my-task"`.

### 2. Add configuration to `common/tasks.default.json`

Add an entry under the `"tasks"` section:

```json
{
  "tasks": {
    "my-task": {
      "enabled": true,
      "type": "utility",
      "capabilities": [],
      "my_param": 42
    }
  }
}
```

Fields:
- **enabled**: `true`/`false` - whether this task is active
- **type**: model type category (`"llm"`, `"sentence-transformer"`, `"seq2seq"`, `"spacy"`, `"utility"`, etc.)
- **capabilities**: list of required worker capabilities. Use `["llm"]` if the task needs an LLM, `["embeddings"]` for embedding models, or `[]` for no special requirements
- **model**: model name/path (for tasks that load a model)
- Additional task-specific parameters can be added freely

### 3. Register the task module

Add an import in `utils/process_job.py`:

```python
import tasks.my_task.my_task
```

### 4. Add a prompt (optional)

If your task uses an LLM with a prompt template, create `tasks/my_task/prompt.md`:

```markdown
You are a helpful assistant. Process the following text:

{text}
```

Load it in your handler:

```python
from services.prompts import get_prompt

prompt = get_prompt("my-task").format(text=text)
```

### 5. Update your config

After adding the task to `common/tasks.default.json`, update your local `config/tasks.json` to include the new task entry (or delete `config/tasks.json` and re-run `./install`).

## Configuration Override System

Users can override task behavior without modifying source code:

- **Prompt override**: Create `config/tasks/my-task/prompt.md` with a custom prompt
- **Config override**: Create `config/tasks/my-task/config.json` with parameter overrides:
  ```json
  {
    "my_param": 100,
    "enabled": false
  }
  ```
  These values are merged on top of the task's entry in `config/tasks.json`.

## Capabilities

Workers detect their capabilities at startup:
- `gpu`: CUDA-capable GPU detected
- `llm`: LLM inference enabled (not disabled in config)
- `embeddings`: Embedding models enabled (not disabled in config)

A task is only processed by a worker if:
1. The task is `enabled` in config
2. The worker has all capabilities listed in the task's `capabilities` array

## Example: Sentiment Analysis Task

Full example creating a sentiment analysis task:

**`tasks/sentiment/sentiment.py`**:
```python
from utils.job_registry import job_handler
from services.model_config import get_task_config, get_llm_params
from services.llm_service import get_llm_service
from services.prompts import get_prompt


@job_handler("sentiment")
def sentiment(payload) -> dict:
    text = payload.get("content", "")
    if not text:
        return {"sentiment": "neutral", "confidence": 0.0}

    task_config = get_task_config("sentiment")
    prompt = get_prompt("sentiment").format(text=text)

    try:
        params = get_llm_params("sentiment")
        llm = get_llm_service(**params)
        response = llm.generate(prompt, max_tokens=task_config.get("max_tokens", 50))

        sentiment = response.strip().lower()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        return {"sentiment": sentiment}
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {e}"}
```

**`tasks/sentiment/prompt.md`**:
```markdown
Classify the sentiment of the following text as exactly one of: positive, negative, neutral.

Respond with only the sentiment label, nothing else.

Text: {text}
```

**Entry in `common/tasks.default.json`**:
```json
"sentiment": {
  "enabled": true,
  "type": "llm",
  "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
  "capabilities": ["llm"],
  "max_tokens": 50
}
```

**Import in `utils/process_job.py`**:
```python
import tasks.sentiment.sentiment
```

### Applying a LoRA Adapter

Any `type: "llm"` task can load a LoRA adapter on top of its base GGUF. Drop the adapter `.gguf` in the `models/` directory and add `lora_model` to the task entry:

```json
"sentiment": {
  "enabled": true,
  "type": "llm",
  "model": "Phi-4-mini-instruct-Q4_K_M.gguf",
  "lora_model": "sentiment-finetune.gguf",
  "lora_scale": 1.0,
  "capabilities": ["llm"],
  "max_tokens": 50
}
```

LoRA files are not auto-downloaded — place them manually alongside the base GGUF. A missing adapter raises `FileNotFoundError` on task load.
