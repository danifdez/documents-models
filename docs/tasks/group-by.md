## Group By

The **group-by** task aggregates a numeric field grouped by a categorical field. It answers questions like "What is the average salary per department?" or "What is the total revenue per country?".

### What it does

The task divides the dataset into groups based on the values of a categorical field, then calculates a summary statistic (mean, sum, count, etc.) for a numeric field within each group. Results are returned sorted from highest to lowest value, and include data suitable for rendering as a bar chart.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `groupField` | string | Key of the categorical field to group by |
| `valueField` | string | Key of the numeric field to aggregate |
| `fn` | string | Aggregation function: `mean`, `sum`, `count`, `min`, `max`, `median`. Defaults to `mean` |

### Returns

```json
{
  "chartType": "bar",
  "field": "salary",
  "fieldName": "Salary",
  "groupField": "department",
  "groupFieldName": "Department",
  "fn": "mean",
  "chartData": {
    "labels": ["Engineering", "Sales", "Marketing"],
    "values": [95000.0, 72000.0, 68000.0]
  },
  "stats": {
    "groupCount": 3,
    "totalRecords": 800
  },
  "tableData": {
    "labels": ["Engineering", "Sales", "Marketing"],
    "values": [95000.0, 72000.0, 68000.0]
  },
  "operation": "group_by",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 8,
  "params": {
    "groupField": "region",
    "valueField": "sales_amount",
    "fn": "sum"
  }
}
```

**Output:**

```json
{
  "chartType": "bar",
  "field": "sales_amount",
  "fieldName": "Sales Amount",
  "groupField": "region",
  "groupFieldName": "Region",
  "fn": "sum",
  "chartData": {
    "labels": ["North", "South", "East", "West"],
    "values": [1250000.0, 980000.0, 870000.0, 620000.0]
  },
  "stats": {
    "groupCount": 4,
    "totalRecords": 3600
  },
  "operation": "group_by",
  "datasetId": 8
}
```
