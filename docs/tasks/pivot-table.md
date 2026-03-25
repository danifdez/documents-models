## Pivot Table

The **pivot-table** task creates a cross-tabulation of two categorical fields, showing how values are distributed across their combinations. It can either count occurrences or aggregate a numeric value.

### What it does

The task takes two categorical fields — one for the rows and one for the columns — and builds a matrix showing the relationship between them. Each cell in the matrix contains either a count or an aggregated numeric value. Row and column totals are included automatically, along with data formatted for a stacked bar chart.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `rowField` | string | Key of the field used for rows (required) |
| `colField` | string | Key of the field used for columns (required) |
| `valueField` | string | Key of a numeric field to aggregate. If omitted, counts occurrences |
| `fn` | string | Aggregation function: `count`, `mean`, `sum`, `min`, `max`, `median`. Defaults to `count` |

### Returns

```json
{
  "chartType": "stacked_bar",
  "chartData": {
    "rows": ["Row A", "Row B"],
    "cols": ["Col X", "Col Y"],
    "values": [[10, 5], [3, 12]],
    "rowTotals": [15, 15],
    "colTotals": [13, 17],
    "grandTotal": 30,
    "barLabels": ["Row A", "Row B"],
    "barDatasets": [
      { "label": "Col X", "data": [10, 3], "backgroundColor": "hsla(0, 70%, 60%, 0.7)" },
      { "label": "Col Y", "data": [5, 12], "backgroundColor": "hsla(180, 70%, 60%, 0.7)" }
    ]
  },
  "tableData": {
    "rows": ["Row A", "Row B"],
    "cols": ["Col X", "Col Y"],
    "values": [[10, 5], [3, 12]],
    "rowTotals": [15, 15],
    "colTotals": [13, 17],
    "grandTotal": 30
  },
  "stats": { "rowCount": 2, "colCount": 2, "totalRecords": 30, "fn": "count" },
  "rowFieldName": "Category",
  "colFieldName": "Status",
  "valueFieldName": "Count",
  "operation": "pivot_table",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 10,
  "params": {
    "rowField": "department",
    "colField": "employment_type",
    "valueField": "salary",
    "fn": "mean"
  }
}
```

**Output (abbreviated):**

```json
{
  "chartType": "stacked_bar",
  "chartData": {
    "rows": ["Engineering", "Sales", "Marketing"],
    "cols": ["Full-time", "Part-time", "Contract"],
    "values": [[95000, 48000, 72000], [70000, 35000, 55000], [65000, 32000, 50000]],
    "rowTotals": [215000, 160000, 147000],
    "grandTotal": 522000
  },
  "rowFieldName": "Department",
  "colFieldName": "Employment Type",
  "valueFieldName": "Salary",
  "operation": "pivot_table",
  "datasetId": 10
}
```
