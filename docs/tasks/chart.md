## Chart

The **chart** task generates data structured for visualization from a dataset. It supports bar, line, pie, and scatter charts, with options for aggregation, sorting, and filtering.

### What it does

Given a dataset and a set of chart parameters, the task reads the relevant fields, applies any filters, aggregates the data, and returns the result in a format that can be directly rendered as a chart. For scatter charts, raw X/Y coordinate pairs are returned. For other chart types, data is grouped by the X field and aggregated.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to use |
| `params` | object | No | Chart configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `chartType` | string | Chart type: `bar`, `line`, `pie`, `scatter`. Defaults to `bar` |
| `xField` | string | Field key for the X axis (required) |
| `yField` | string | Field key for the Y axis (optional; if omitted, counts occurrences) |
| `aggregation` | string | How to aggregate Y values: `count`, `mean`, `sum`, `min`, `max`, `median`. Defaults to `count` |
| `sortBy` | string | Sort results by `value` or `label`. Defaults to `value` |
| `sortOrder` | string | Sort direction: `asc` or `desc`. Defaults to `desc` |
| `limit` | number | Maximum number of categories to return. Defaults to 20 |
| `filters` | array | List of filter conditions to apply before charting |

### Returns

For bar, line, and pie charts:

```json
{
  "chartType": "bar",
  "chartData": { "labels": ["Category A", "Category B"], "values": [120, 85] },
  "stats": { "totalRecords": 500, "categories": 2, "min": 85, "max": 120, "total": 205 },
  "title": "Sum of Revenue by Region",
  "xLabel": "Region",
  "yLabel": "Revenue",
  "tableData": { "labels": ["Category A", "Category B"], "values": [120, 85] },
  "operation": "chart",
  "datasetId": 1
}
```

For scatter charts:

```json
{
  "chartType": "scatter",
  "chartData": { "points": [{ "x": 1.5, "y": 3.2 }, { "x": 2.1, "y": 4.7 }] },
  "stats": { "totalPoints": 200 },
  "title": "Age vs Income",
  "xLabel": "Age",
  "yLabel": "Income",
  "operation": "chart",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 5,
  "params": {
    "chartType": "bar",
    "xField": "department",
    "yField": "salary",
    "aggregation": "mean",
    "limit": 10,
    "sortOrder": "desc"
  }
}
```

**Output:**

```json
{
  "chartType": "bar",
  "chartData": { "labels": ["Engineering", "Sales", "Marketing"], "values": [95000, 72000, 68000] },
  "stats": { "totalRecords": 800, "categories": 3, "min": 68000, "max": 95000, "total": 235000 },
  "title": "Mean of Salary by Department",
  "xLabel": "Department",
  "yLabel": "Salary",
  "operation": "chart",
  "datasetId": 5
}
```
