## Ask

The **ask** task answers a natural language question by searching through the documents and knowledge stored in the project using a Retrieval-Augmented Generation (RAG) pipeline. It finds the most relevant content from the vector database and uses a language model to compose a coherent answer.

### What it does

Given a question, this task retrieves relevant text snippets from the project's indexed content and then generates a natural language response based on those snippets. If no relevant information is found, it returns a message indicating so.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | The question to answer |
| `projectId` | number | No | Limits the search to a specific project's content |

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
  "projectId": 5
}
```

**Output:**

```json
{
  "response": "According to the 2023 annual report, the main conclusions were a 15% revenue growth driven by the new product line and an expansion into three new markets."
}
```
