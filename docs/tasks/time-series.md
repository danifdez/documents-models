## Time Series

The **time-series** task analyzes how a numeric value changes over time. It groups data by a date field into regular time periods (monthly, quarterly, yearly) and returns the average value per period along with a trend indicator.

### What it does

The task takes a date field and a numeric field, resamples the data into equal-length time periods, and calculates the average for each period. The result is suitable for rendering as a line chart and includes a trend slope that indicates whether the value is generally increasing or decreasing over time.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `dateField` | string | Key of the date field to use as the time axis |
| `valueField` | string | Key of the numeric field to track over time |
| `period` | string | Time period for grouping: `M` (monthly), `Q` (quarterly), `Y` (yearly). Defaults to `M` |

### Returns

```json
{
  "chartType": "line",
  "dateField": "created_at",
  "dateFieldName": "Created At",
  "valueField": "revenue",
  "valueFieldName": "Revenue",
  "period": "ME",
  "chartData": {
    "labels": ["2024-01-31", "2024-02-29", "2024-03-31"],
    "values": [45200.0, 51800.0, 48900.0]
  },
  "stats": {
    "dataPoints": 12,
    "trend": 850.5
  },
  "tableData": {
    "labels": ["2024-01-31", "2024-02-29", "2024-03-31"],
    "values": [45200.0, 51800.0, 48900.0]
  },
  "operation": "time_series",
  "datasetId": 1
}
```

A positive `trend` value means the metric is generally increasing over time; a negative value means it is decreasing.

### Example

**Input:**

```json
{
  "datasetId": 11,
  "params": {
    "dateField": "order_date",
    "valueField": "total_amount",
    "period": "Q"
  }
}
```

**Output:**

```json
{
  "chartType": "line",
  "dateField": "order_date",
  "dateFieldName": "Order Date",
  "valueField": "total_amount",
  "valueFieldName": "Total Amount",
  "period": "QE",
  "chartData": {
    "labels": ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
    "values": [125000.0, 148000.0, 162000.0, 195000.0]
  },
  "stats": {
    "dataPoints": 4,
    "trend": 23333.3
  },
  "operation": "time_series",
  "datasetId": 11
}
```
