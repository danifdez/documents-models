from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
import pandas as pd


@job_handler("group-by")
def group_by(payload) -> dict:
    """Aggregate a numeric field grouped by a categorical field."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        value_field = params.get("valueField")
        group_field = params.get("groupField")
        agg_fn = params.get("fn", "mean")

        v_def = next((f for f in schema if f["key"] == value_field), None)
        g_def = next((f for f in schema if f["key"] == group_field), None)
        if not v_def or not g_def:
            return {"error": "Both fields must exist"}

        if value_field not in df.columns or group_field not in df.columns:
            return {"error": "Fields not found in data"}

        sub = df[[group_field, value_field]].copy()
        sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
        sub = sub.dropna()

        agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
        fn = agg_map.get(agg_fn, "mean")

        grouped = sub.groupby(group_field)[value_field].agg(fn).sort_values(ascending=False)

        return {
            "chartType": "bar",
            "field": value_field,
            "fieldName": v_def["name"],
            "groupField": group_field,
            "groupFieldName": g_def["name"],
            "fn": agg_fn,
            "chartData": {
                "labels": [str(v) for v in grouped.index.tolist()],
                "values": [safe_float(v) for v in grouped.values],
            },
            "stats": {
                "groupCount": len(grouped),
                "totalRecords": len(sub),
            },
            "tableData": {
                "labels": [str(v) for v in grouped.index.tolist()],
                "values": [safe_float(v) for v in grouped.values],
            },
            "operation": "group_by",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
