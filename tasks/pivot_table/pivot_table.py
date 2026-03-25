from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float
import pandas as pd


@job_handler("pivot-table")
def pivot_table(payload) -> dict:
    """Cross-tabulation / pivot table."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        row_field = params.get("rowField")
        col_field = params.get("colField")
        value_field = params.get("valueField")
        agg_fn = params.get("fn", "count")

        r_def = next((f for f in schema if f["key"] == row_field), None)
        c_def = next((f for f in schema if f["key"] == col_field), None)
        if not r_def or not c_def:
            return {"error": "Row and column fields are required"}
        if row_field not in df.columns or col_field not in df.columns:
            return {"error": "Fields not found in data"}

        if value_field and value_field in df.columns:
            v_def = next((f for f in schema if f["key"] == value_field), None)
            sub = df[[row_field, col_field, value_field]].copy()
            sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")

            agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
            fn = agg_map.get(agg_fn, "mean")

            pivot = pd.pivot_table(sub, values=value_field, index=row_field,
                                   columns=col_field, aggfunc=fn, fill_value=0)
        else:
            v_def = None
            sub = df[[row_field, col_field]].copy()
            sub["_count"] = 1
            pivot = pd.pivot_table(sub, values="_count", index=row_field,
                                   columns=col_field, aggfunc="sum", fill_value=0)
            agg_fn = "count"

        rows = [str(v) for v in pivot.index.tolist()]
        cols = [str(v) for v in pivot.columns.tolist()]
        values = [[safe_float(pivot.iloc[i, j]) for j in range(len(cols))] for i in range(len(rows))]
        row_totals = [safe_float(sum(r)) for r in values]
        col_totals = [safe_float(sum(values[i][j] or 0 for i in range(len(rows)))) for j in range(len(cols))]
        grand_total = safe_float(sum(rt or 0 for rt in row_totals))

        chart_datasets = []
        colors = [f"hsla({int(i * 360 / max(len(cols), 1))}, 70%, 60%, 0.7)" for i in range(len(cols))]
        for j, col_name in enumerate(cols):
            chart_datasets.append({
                "label": col_name,
                "data": [values[i][j] for i in range(len(rows))],
                "backgroundColor": colors[j],
            })

        return {
            "chartType": "stacked_bar",
            "chartData": {
                "rows": rows,
                "cols": cols,
                "values": values,
                "rowTotals": row_totals,
                "colTotals": col_totals,
                "grandTotal": grand_total,
                "barLabels": rows,
                "barDatasets": chart_datasets,
            },
            "stats": {
                "rowCount": len(rows),
                "colCount": len(cols),
                "totalRecords": len(sub),
                "fn": agg_fn,
            },
            "tableData": {
                "rows": rows,
                "cols": cols,
                "values": values,
                "rowTotals": row_totals,
                "colTotals": col_totals,
                "grandTotal": grand_total,
            },
            "rowFieldName": r_def["name"],
            "colFieldName": c_def["name"],
            "valueFieldName": v_def["name"] if v_def else "Count",
            "operation": "pivot_table",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
