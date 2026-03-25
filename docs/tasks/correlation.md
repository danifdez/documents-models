## Correlation

The **correlation** task measures the statistical relationship between two numeric fields in a dataset. It calculates Pearson correlation along with a regression line, and returns data suitable for rendering as a scatter plot.

### What it does

Given two numeric fields, the task computes how strongly they are related. A correlation close to 1 or -1 means the two fields move together closely; a value near 0 means little or no relationship. The result includes the scatter plot data points and a linear regression line.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `field1` | string | Key of the first numeric field |
| `field2` | string | Key of the second numeric field |

### Returns

```json
{
  "chartType": "scatter",
  "field1": "age",
  "field1Name": "Age",
  "field2": "income",
  "field2Name": "Income",
  "chartData": {
    "points": [{ "x": 25, "y": 35000 }, { "x": 40, "y": 62000 }],
    "regression": { "slope": 1800.5, "intercept": 10200.0, "xRange": [20, 65] }
  },
  "stats": {
    "correlation": 0.78,
    "pValue": 0.0001,
    "rSquared": 0.61,
    "n": 350,
    "slope": 1800.5,
    "intercept": 10200.0
  },
  "operation": "correlation",
  "datasetId": 1
}
```

**Key stats:**

- `correlation`: Pearson R coefficient (-1 to 1)
- `rSquared`: How much of the variance in one field is explained by the other
- `pValue`: Statistical significance (lower is more significant)

At minimum 3 data points are required. On error:

```json
{
  "error": "Not enough data points (minimum 3)"
}
```

### Example

**Input:**

```json
{
  "datasetId": 4,
  "params": {
    "field1": "study_hours",
    "field2": "exam_score"
  }
}
```

**Output:**

```json
{
  "chartType": "scatter",
  "field1": "study_hours",
  "field1Name": "Study Hours",
  "field2": "exam_score",
  "field2Name": "Exam Score",
  "chartData": {
    "points": [{ "x": 2, "y": 55 }, { "x": 5, "y": 78 }, { "x": 8, "y": 91 }],
    "regression": { "slope": 5.2, "intercept": 44.6, "xRange": [1, 10] }
  },
  "stats": {
    "correlation": 0.93,
    "pValue": 0.0000021,
    "rSquared": 0.86,
    "n": 120
  },
  "operation": "correlation",
  "datasetId": 4
}
```
