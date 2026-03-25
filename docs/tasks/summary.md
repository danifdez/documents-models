## Summary

The **summary** task provides a quick statistical overview of all fields in a dataset. It returns key descriptive statistics for every field in a single call, making it easy to understand the shape and content of a dataset at a glance.

### What it does

For each field in the dataset, the task calculates a set of statistics appropriate to the field's type: numeric statistics (mean, median, min, max, standard deviation) for number fields, unique value counts for text and categorical fields, and date ranges for date fields. It also reports null counts for every field.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to summarize |

No additional parameters are needed.

### Returns

```json
{
  "chartType": "none",
  "recordCount": 1000,
  "fieldCount": 5,
  "fields": [
    {
      "field": "age",
      "name": "Age",
      "type": "number",
      "totalCount": 1000,
      "nonNullCount": 998,
      "nullCount": 2,
      "nullPercent": 0.2,
      "mean": 34.2,
      "median": 33.0,
      "std": 10.5,
      "min": 18.0,
      "max": 72.0
    },
    {
      "field": "status",
      "name": "Status",
      "type": "text",
      "totalCount": 1000,
      "nonNullCount": 1000,
      "nullCount": 0,
      "nullPercent": 0.0,
      "uniqueCount": 3,
      "topValue": "Active"
    },
    {
      "field": "created_at",
      "name": "Created At",
      "type": "date",
      "totalCount": 1000,
      "nonNullCount": 995,
      "nullCount": 5,
      "nullPercent": 0.5,
      "min": "2022-01-03",
      "max": "2024-12-28"
    }
  ],
  "tableData": { "fields": [ "..." ] },
  "operation": "summary",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 7
}
```

**Output (abbreviated):**

```json
{
  "chartType": "none",
  "recordCount": 500,
  "fieldCount": 4,
  "fields": [
    { "field": "price", "name": "Price", "type": "number", "mean": 99.5, "min": 5.0, "max": 499.0, "nullCount": 0 },
    { "field": "category", "name": "Category", "type": "select", "uniqueCount": 8, "topValue": "Electronics", "nullCount": 3 },
    { "field": "in_stock", "name": "In Stock", "type": "text", "uniqueCount": 2, "topValue": "true", "nullCount": 0 },
    { "field": "listed_at", "name": "Listed At", "type": "date", "min": "2023-01-10", "max": "2024-11-30", "nullCount": 12 }
  ],
  "operation": "summary",
  "datasetId": 7
}
```
