## Query

The **query** task runs a custom data query across one or more datasets. It supports filtering, grouping, aggregation, and field selection, returning both tabular data and chart-ready output.

### What it does

Given a dataset (or multiple datasets joined together), the task applies any specified filters, groups the data by a field, and aggregates values using a chosen function. The result can be used to build charts or display tables. When no grouping is specified, the raw rows are returned.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Conditional | ID of the dataset to query. Required if `datasetIds` is not provided |
| `datasetIds` | array | Conditional | List of dataset IDs to join together. The first is the primary dataset |
| `params` | object | No | Query configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `filters` | array | List of filter conditions to apply |
| `groupBy` | string | Field key to group results by |
| `select` | array | List of field keys to return |
| `fn` | string | Aggregation function: `count`, `mean`, `sum`, `min`, `max`, `median`. Defaults to `count` |
| `chartType` | string | Suggested chart type for the result: `bar`, `line`, `pie`. Defaults to `bar` |
| `joinField` | string | Field key used to join multiple datasets |

### Returns

When grouping is applied:

```json
{
  "chartType": "bar",
  "chartData": { "labels": ["A", "B"], "values": [10, 5] },
  "stats": { "groupCount": 2, "totalRecords": 150 },
  "tableData": { "labels": ["A", "B"], "values": [10, 5] },
  "operation": "query",
  "datasetId": 1
}
```

When selecting specific fields without grouping:

```json
{
  "chartType": "none",
  "stats": { "totalRecords": 150, "returnedRecords": 50 },
  "tableData": { "columns": ["name", "value"], "rows": [["Alice", 42]] },
  "operation": "query",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 3,
  "params": {
    "groupBy": "country",
    "select": ["revenue"],
    "fn": "sum",
    "chartType": "bar"
  }
}
```

**Output:**

```json
{
  "chartType": "bar",
  "chartData": { "labels": ["USA", "Germany", "Spain"], "values": [500000, 320000, 180000] },
  "stats": { "groupCount": 3, "totalRecords": 1200 },
  "tableData": { "labels": ["USA", "Germany", "Spain"], "values": [500000, 320000, 180000] },
  "operation": "query",
  "datasetId": 3
}
```
