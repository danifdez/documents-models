## Ingest Content

The **ingest-content** task indexes the content of a document, resource, or knowledge entry into the vector database so it can be found later via semantic search. There is also a companion **delete-vectors** task for removing indexed content.

### What it does

The content is cleaned, split into smaller chunks, and each chunk is converted into a vector embedding. These vectors are stored in the vector database with metadata that links them back to the original source. If the source was previously indexed, all old vectors are deleted first to keep the data consistent.

### Task: `ingest-content`

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The HTML content to index |
| `projectId` | number | No | The project this content belongs to |
| `sourceType` | string | No | One of `resource`, `doc`, or `knowledge`. Defaults to `resource` |
| `resourceId` | number | Conditional | Required when `sourceType` is `resource` |
| `docId` | number | Conditional | Required when `sourceType` is `doc` |
| `knowledgeEntryId` | number | Conditional | Required when `sourceType` is `knowledge` |

#### Returns

```json
{
  "success": true
}
```

### Task: `delete-vectors`

Removes all indexed vectors associated with a given source.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sourceId` | string | Yes | The identifier of the source whose vectors should be deleted |

#### Returns

```json
{
  "success": true
}
```

On error:

```json
{
  "error": "sourceId is required"
}
```

### Example

**Ingesting a resource:**

```json
{
  "content": "<p>This document discusses renewable energy strategies...</p>",
  "sourceType": "resource",
  "resourceId": 42,
  "projectId": 7
}
```

**Output:**

```json
{
  "success": true
}
```

**Deleting vectors:**

```json
{
  "sourceId": "42"
}
```

**Output:**

```json
{
  "success": true
}
```
