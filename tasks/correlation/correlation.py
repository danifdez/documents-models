from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
from scipy import stats as scipy_stats
import pandas as pd
import numpy as np


@job_handler("correlation")
def correlation(payload) -> dict:
    """Scatter plot + correlation between two numeric fields."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        field1 = params.get("field1")
        field2 = params.get("field2")

        f1_def = next((f for f in schema if f["key"] == field1), None)
        f2_def = next((f for f in schema if f["key"] == field2), None)
        if not f1_def or not f2_def:
            return {"error": "Both fields must exist"}

        if field1 not in df.columns or field2 not in df.columns:
            return {"error": "Fields not found in data"}

        valid = df[[field1, field2]].dropna()
        x = pd.to_numeric(valid[field1], errors="coerce")
        y = pd.to_numeric(valid[field2], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]

        if len(x) < 3:
            return {"error": "Not enough data points (minimum 3)"}

        r, p_value = scipy_stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)

        result = {
            "chartType": "scatter",
            "field1": field1,
            "field1Name": f1_def["name"],
            "field2": field2,
            "field2Name": f2_def["name"],
            "chartData": {
                "points": [{"x": safe_float(xi), "y": safe_float(yi)} for xi, yi in zip(x, y)],
                "regression": {
                    "slope": safe_float(slope),
                    "intercept": safe_float(intercept),
                    "xRange": [safe_float(x.min()), safe_float(x.max())],
                },
            },
            "stats": {
                "correlation": safe_float(r),
                "pValue": safe_float(p_value),
                "rSquared": safe_float(r ** 2),
                "n": len(x),
                "slope": safe_float(slope),
                "intercept": safe_float(intercept),
            },
            "tableData": {
                "labels": [f1_def["name"], f2_def["name"]],
                "rows": [[safe_float(xi), safe_float(yi)] for xi, yi in zip(x, y)],
            },
            "operation": "correlation",
            "datasetId": dataset_id,
        }

        return result

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
