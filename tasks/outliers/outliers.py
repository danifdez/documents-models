from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
import pandas as pd
import numpy as np


@job_handler("outliers")
def outliers(payload) -> dict:
    """Detect outliers in a numeric field using IQR method."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        field_key = params.get("field")
        field_def = next((f for f in schema if f["key"] == field_key), None)
        if not field_def or field_key not in df.columns:
            return {"error": f"Field '{field_key}' not found"}

        col = pd.to_numeric(df[field_key], errors="coerce").dropna()
        if len(col) < 4:
            return {"error": "Not enough data points"}

        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (col < lower) | (col > upper)
        outlier_vals = col[outlier_mask]

        all_points = []
        for idx, val in col.items():
            is_outlier = bool(outlier_mask.loc[idx])
            record_id = int(df.loc[idx, "_id"]) if "_id" in df.columns else idx
            all_points.append({
                "value": safe_float(val),
                "isOutlier": is_outlier,
                "recordId": record_id,
            })

        z_scores = np.abs((col - col.mean()) / col.std()) if col.std() > 0 else pd.Series([0] * len(col), index=col.index)
        z_outlier_count = int((z_scores > 3).sum())

        return {
            "chartType": "box",
            "field": field_key,
            "fieldName": field_def["name"],
            "chartData": {
                "min": safe_float(col.min()),
                "q1": safe_float(q1),
                "median": safe_float(col.median()),
                "q3": safe_float(q3),
                "max": safe_float(col.max()),
                "lowerFence": safe_float(lower),
                "upperFence": safe_float(upper),
                "outliers": [safe_float(v) for v in outlier_vals.tolist()],
                "allPoints": all_points,
            },
            "stats": {
                "totalCount": len(col),
                "outlierCount": len(outlier_vals),
                "outlierPercent": safe_float(len(outlier_vals) / len(col) * 100),
                "zScoreOutliers": z_outlier_count,
                "lowerBound": safe_float(lower),
                "upperBound": safe_float(upper),
                "iqr": safe_float(iqr),
                "mean": safe_float(col.mean()),
                "std": safe_float(col.std()),
            },
            "tableData": {
                "columns": ["Value", "Record ID", "Z-Score"],
                "rows": [
                    [safe_float(v), int(df.loc[idx, "_id"]) if "_id" in df.columns else idx,
                     safe_float(z_scores.loc[idx])]
                    for idx, v in outlier_vals.items()
                ],
            },
            "operation": "outliers",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
