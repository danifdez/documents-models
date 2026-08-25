## Ask

The **ask** task answers a natural language question from a bounded candidate
snapshot prepared by Backend. It ranks that snapshot and uses a language model
to compose a grounded response without accessing a vector database.

### What it does

Given a question, this task retrieves relevant text snippets from the project's indexed content and then generates a natural language response based on those snippets. If no relevant information is found, it returns a message indicating so.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question to answer |
| `candidates` | array | direct harness only | Vector candidates; production supplies the equivalent artifact |
| `graphContext` | array | no | Backend-resolved, project-scoped relationship triples |

### Returns

```json
{
  "response": "The answer to your question based on the indexed content."
}
```

If no relevant content is found:

```json
{
  "response": "No relevant information was found to answer this question."
}
```

### Example

**Input:**

```json
{
  "question": "What were the main conclusions of the 2023 annual report?",
  "candidates": [],
  "graphContext": []
}
```

**Output:**

```json
{
  "response": "According to the 2023 annual report, the main conclusions were a 15% revenue growth driven by the new product line and an expansion into three new markets."
}
```
