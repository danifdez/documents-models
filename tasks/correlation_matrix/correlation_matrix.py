from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
from scipy import stats as scipy_stats
import pandas as pd


@job_handler("correlation-matrix")
def correlation_matrix(payload) -> dict:
    """NxN Pearson correlation matrix for all (or selected) numeric fields."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        selected_keys = params.get("fields", [])
        numeric_fields = [f for f in schema if f["type"] == "number" and f["key"] in df.columns]
        if selected_keys:
            numeric_fields = [f for f in numeric_fields if f["key"] in selected_keys]

        if len(numeric_fields) < 2:
            return {"error": "At least 2 numeric fields are required"}

        keys = [f["key"] for f in numeric_fields]
        sub = df[keys].apply(pd.to_numeric, errors="coerce")

        n = len(keys)
        matrix = [[0.0] * n for _ in range(n)]
        p_values = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matrix[i][i] = 1.0
            p_values[i][i] = 0.0
            for j in range(i + 1, n):
                valid = sub[[keys[i], keys[j]]].dropna()
                if len(valid) < 3:
                    matrix[i][j] = matrix[j][i] = None
                    p_values[i][j] = p_values[j][i] = None
                    continue
                r, p = scipy_stats.pearsonr(valid[keys[i]], valid[keys[j]])
                matrix[i][j] = matrix[j][i] = safe_float(r)
                p_values[i][j] = p_values[j][i] = safe_float(p)

        strong = []
        for i in range(n):
            for j in range(i + 1, n):
                r = matrix[i][j]
                if r is not None and abs(r) >= 0.5:
                    strong.append({
                        "field1": keys[i], "field1Name": numeric_fields[i]["name"],
                        "field2": keys[j], "field2Name": numeric_fields[j]["name"],
                        "correlation": r,
                        "pValue": p_values[i][j],
                    })
        strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        fields_info = [{"key": f["key"], "name": f["name"]} for f in numeric_fields]

        return {
            "chartType": "heatmap",
            "chartData": {
                "fields": fields_info,
                "matrix": matrix,
                "pValues": p_values,
            },
            "stats": {
                "fieldCount": n,
                "strongCorrelations": strong[:10],
            },
            "tableData": {
                "fields": fields_info,
                "matrix": matrix,
                "pValues": p_values,
            },
            "operation": "correlation_matrix",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
