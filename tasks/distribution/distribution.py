from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float, resolve_fk_labels, _normalize_fk_value
import pandas as pd
import numpy as np


@job_handler("distribution")
def distribution(payload) -> dict:
    """Distribution of a single field: histogram for numeric, frequency for categorical."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        field_key = params.get("field")
        field_def = next((f for f in schema if f["key"] == field_key), None)
        if not field_def or field_key not in df.columns:
            return {"error": f"Field '{field_key}' not found"}

        col = df[field_key].dropna()
        ftype = field_def["type"]

        result = {
            "field": field_key,
            "name": field_def["name"],
            "type": ftype,
            "totalCount": len(df),
            "nonNullCount": len(col),
            "nullCount": int(df[field_key].isna().sum()),
        }

        if ftype == "number" and len(col) > 0:
            numeric = col.astype(float)
            bins = params.get("bins", "auto")
            if isinstance(bins, str) and bins != "auto":
                bins = int(bins)

            counts, bin_edges = np.histogram(numeric, bins=bins)
            result["chartType"] = "histogram"
            result["chartData"] = {
                "labels": [f"{safe_float(bin_edges[i])}-{safe_float(bin_edges[i+1])}" for i in range(len(counts))],
                "values": [int(c) for c in counts],
            }
            result["stats"] = {
                "mean": safe_float(numeric.mean()),
                "median": safe_float(numeric.median()),
                "std": safe_float(numeric.std()),
                "min": safe_float(numeric.min()),
                "max": safe_float(numeric.max()),
                "q25": safe_float(numeric.quantile(0.25)),
                "q75": safe_float(numeric.quantile(0.75)),
                "skewness": safe_float(numeric.skew()),
                "kurtosis": safe_float(numeric.kurtosis()),
            }
        else:
            freq = col.value_counts().head(30)
            raw_labels = freq.index.tolist()
            fk_map = resolve_fk_labels(schema, field_key, raw_labels)
            labels = [fk_map.get(_normalize_fk_value(v), str(v)) for v in raw_labels]
            result["chartType"] = "bar"
            result["chartData"] = {
                "labels": labels,
                "values": [int(c) for c in freq.values],
            }
            result["stats"] = {"uniqueCount": int(col.nunique())}

        result["tableData"] = result["chartData"]
        result["operation"] = "distribution"
        result["datasetId"] = dataset_id

        return result

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
