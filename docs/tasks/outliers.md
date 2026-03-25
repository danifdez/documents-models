## Outliers

The **outliers** task detects unusual values in a numeric field using the IQR (Interquartile Range) method. It identifies data points that fall significantly outside the typical range and returns a box plot-ready summary.

### What it does

The task calculates the statistical boundaries for normal values in a field. Any value below the lower fence or above the upper fence is considered an outlier. The result includes the position of every data point, whether it is an outlier, and summary statistics about the spread of the data.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `field` | string | Key of the numeric field to check for outliers |

At least 4 data points are required.

### Returns

```json
{
  "chartType": "box",
  "field": "price",
  "fieldName": "Price",
  "chartData": {
    "min": 5.0,
    "q1": 45.0,
    "median": 89.0,
    "q3": 150.0,
    "max": 999.0,
    "lowerFence": -112.5,
    "upperFence": 307.5,
    "outliers": [450.0, 720.0, 999.0],
    "allPoints": [
      { "value": 89.0, "isOutlier": false, "recordId": 1 },
      { "value": 450.0, "isOutlier": true, "recordId": 7 }
    ]
  },
  "stats": {
    "totalCount": 500,
    "outlierCount": 12,
    "outlierPercent": 2.4,
    "zScoreOutliers": 8,
    "lowerBound": -112.5,
    "upperBound": 307.5,
    "iqr": 105.0,
    "mean": 95.3,
    "std": 87.2
  },
  "operation": "outliers",
  "datasetId": 1
}
```

**How outlier boundaries are calculated:**

- Lower fence: Q1 − 1.5 × IQR
- Upper fence: Q3 + 1.5 × IQR

The `zScoreOutliers` stat counts points with a Z-score above 3 (more than 3 standard deviations from the mean).

### Example

**Input:**

```json
{
  "datasetId": 9,
  "params": {
    "field": "response_time_ms"
  }
}
```

**Output (abbreviated):**

```json
{
  "chartType": "box",
  "field": "response_time_ms",
  "fieldName": "Response Time (ms)",
  "chartData": {
    "min": 45.0,
    "q1": 120.0,
    "median": 185.0,
    "q3": 260.0,
    "max": 4200.0,
    "lowerFence": -90.0,
    "upperFence": 470.0,
    "outliers": [890.0, 1450.0, 4200.0]
  },
  "stats": {
    "totalCount": 800,
    "outlierCount": 15,
    "outlierPercent": 1.875
  },
  "operation": "outliers",
  "datasetId": 9
}
```
