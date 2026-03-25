## Correlation Matrix

The **correlation-matrix** task computes the Pearson correlation between all pairs of numeric fields in a dataset, presenting the results as a heatmap-ready matrix. It also highlights the strongest relationships found.

### What it does

For every combination of numeric fields, the task calculates how strongly the two fields are correlated. The result is a square matrix where each cell shows the correlation coefficient between two fields. Pairs with a correlation stronger than 0.5 (or weaker than -0.5) are highlighted as notable relationships.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | No | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `fields` | array of strings | Optional list of field keys to include. If omitted, all numeric fields are used |

At least 2 numeric fields are required.

### Returns

```json
{
  "chartType": "heatmap",
  "chartData": {
    "fields": [{ "key": "age", "name": "Age" }, { "key": "income", "name": "Income" }],
    "matrix": [[1.0, 0.78], [0.78, 1.0]],
    "pValues": [[0.0, 0.0001], [0.0001, 0.0]]
  },
  "stats": {
    "fieldCount": 2,
    "strongCorrelations": [
      {
        "field1": "age",
        "field1Name": "Age",
        "field2": "income",
        "field2Name": "Income",
        "correlation": 0.78,
        "pValue": 0.0001
      }
    ]
  },
  "operation": "correlation_matrix",
  "datasetId": 1
}
```

The diagonal of the matrix is always 1.0 (a field is perfectly correlated with itself). `strongCorrelations` lists up to 10 pairs with an absolute correlation of 0.5 or higher, sorted strongest first.

### Example

**Input:**

```json
{
  "datasetId": 6,
  "params": {
    "fields": ["temperature", "humidity", "rainfall", "crop_yield"]
  }
}
```

**Output (abbreviated):**

```json
{
  "chartType": "heatmap",
  "chartData": {
    "fields": [
      { "key": "temperature", "name": "Temperature" },
      { "key": "humidity", "name": "Humidity" },
      { "key": "rainfall", "name": "Rainfall" },
      { "key": "crop_yield", "name": "Crop Yield" }
    ],
    "matrix": [
      [1.0, -0.42, -0.31, 0.65],
      [-0.42, 1.0, 0.83, -0.21],
      [-0.31, 0.83, 1.0, 0.11],
      [0.65, -0.21, 0.11, 1.0]
    ]
  },
  "stats": {
    "fieldCount": 4,
    "strongCorrelations": [
      { "field1": "humidity", "field1Name": "Humidity", "field2": "rainfall", "field2Name": "Rainfall", "correlation": 0.83, "pValue": 0.0 },
      { "field1": "temperature", "field1Name": "Temperature", "field2": "crop_yield", "field2Name": "Crop Yield", "correlation": 0.65, "pValue": 0.001 }
    ]
  }
}
```
