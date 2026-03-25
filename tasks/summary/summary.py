from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
import pandas as pd


@job_handler("summary")
def summary(payload) -> dict:
    """Quick summary overview of all fields."""
    try:
        dataset_id, df, schema = load_dataset(payload)

        fields_summary = []
        for field in schema:
            key = field["key"]
            if key not in df.columns:
                continue
            col = df[key]
            non_null = col.dropna()
            entry = {
                "field": key,
                "name": field["name"],
                "type": field["type"],
                "totalCount": len(col),
                "nonNullCount": len(non_null),
                "nullCount": int(col.isna().sum()),
                "nullPercent": safe_float(col.isna().sum() / len(col) * 100) if len(col) > 0 else 0,
            }

            if field["type"] == "number" and len(non_null) > 0:
                numeric = non_null.astype(float)
                entry.update({
                    "mean": safe_float(numeric.mean()),
                    "median": safe_float(numeric.median()),
                    "std": safe_float(numeric.std()),
                    "min": safe_float(numeric.min()),
                    "max": safe_float(numeric.max()),
                })
            elif field["type"] in ("text", "select") and len(non_null) > 0:
                entry["uniqueCount"] = int(non_null.nunique())
                entry["topValue"] = str(non_null.value_counts().index[0]) if len(non_null) > 0 else None
            elif field["type"] in ("date", "datetime") and len(non_null) > 0:
                dates = pd.to_datetime(non_null, errors="coerce").dropna()
                if len(dates) > 0:
                    entry["min"] = str(dates.min())
                    entry["max"] = str(dates.max())

            fields_summary.append(entry)

        return {
            "chartType": "none",
            "recordCount": len(df),
            "fieldCount": len(schema),
            "fields": fields_summary,
            "tableData": {"fields": fields_summary},
            "operation": "summary",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
