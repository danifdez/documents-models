## Distribution

The **distribution** task analyzes how the values of a single field are spread across a dataset. For numeric fields it generates a histogram; for text or categorical fields it shows the most frequent values.

### What it does

The task reads a chosen field and calculates the distribution of its values. For numbers, the data is divided into bins and the count in each bin is returned, along with descriptive statistics like mean, median, and standard deviation. For text or categorical data, the top 30 most frequent values are returned.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `datasetId` | number | Yes | ID of the dataset to analyze |
| `params` | object | Yes | Analysis configuration (see below) |

**`params` object:**

| Field | Type | Description |
|-------|------|-------------|
| `field` | string | Key of the field to analyze |
| `bins` | number or `"auto"` | Number of histogram bins for numeric fields. Defaults to `"auto"` |

### Returns

For numeric fields:

```json
{
  "chartType": "histogram",
  "field": "price",
  "name": "Price",
  "type": "number",
  "totalCount": 500,
  "nonNullCount": 495,
  "nullCount": 5,
  "chartData": {
    "labels": ["0-100", "100-200", "200-300"],
    "values": [120, 250, 125]
  },
  "stats": {
    "mean": 165.4,
    "median": 158.0,
    "std": 62.3,
    "min": 5.0,
    "max": 299.0,
    "q25": 110.0,
    "q75": 220.0,
    "skewness": 0.23,
    "kurtosis": -0.45
  },
  "operation": "distribution",
  "datasetId": 1
}
```

For categorical fields:

```json
{
  "chartType": "bar",
  "field": "status",
  "name": "Status",
  "type": "text",
  "totalCount": 500,
  "nonNullCount": 500,
  "nullCount": 0,
  "chartData": {
    "labels": ["Active", "Pending", "Closed"],
    "values": [310, 120, 70]
  },
  "stats": { "uniqueCount": 3 },
  "operation": "distribution",
  "datasetId": 1
}
```

### Example

**Input:**

```json
{
  "datasetId": 2,
  "params": {
    "field": "age",
    "bins": 10
  }
}
```

**Output:**

```json
{
  "chartType": "histogram",
  "field": "age",
  "name": "Age",
  "type": "number",
  "totalCount": 1000,
  "nonNullCount": 998,
  "nullCount": 2,
  "chartData": {
    "labels": ["18-26", "26-34", "34-42", "42-50", "50-58"],
    "values": [230, 310, 250, 145, 63]
  },
  "stats": {
    "mean": 34.2,
    "median": 33.0,
    "std": 10.5,
    "min": 18.0,
    "max": 72.0
  },
  "operation": "distribution",
  "datasetId": 2
}
```
