from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float, apply_filters
import pandas as pd


@job_handler("chart")
def chart(payload) -> dict:
    """Generate chart data for a custom visualization."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        chart_type = params.get("chartType", "bar")
        x_field = params.get("xField")
        y_field = params.get("yField")
        aggregation = params.get("aggregation", "count")
        sort_by = params.get("sortBy", "value")
        sort_order = params.get("sortOrder", "desc")
        limit = params.get("limit", 20)

        df = apply_filters(df, params.get("filters", []))

        if len(df) == 0:
            return {"error": "No data after applying filters", "chartType": "none"}

        x_def = next((f for f in schema if f["key"] == x_field), None)
        if not x_def or x_field not in df.columns:
            return {"error": f"Field '{x_field}' not found"}

        y_def = next((f for f in schema if f["key"] == y_field), None) if y_field else None

        # Scatter: return raw X/Y points
        if chart_type == "scatter":
            if not y_field or y_field not in df.columns:
                return {"error": "Scatter chart requires both X and Y numeric fields"}

            sub = df[[x_field, y_field]].copy()
            sub[x_field] = pd.to_numeric(sub[x_field], errors="coerce")
            sub[y_field] = pd.to_numeric(sub[y_field], errors="coerce")
            sub = sub.dropna()

            if len(sub) == 0:
                return {"error": "No valid numeric data points"}

            points = [{"x": safe_float(row[x_field]), "y": safe_float(row[y_field])} for _, row in sub.iterrows()]

            return {
                "chartType": "scatter",
                "chartData": {"points": points},
                "stats": {"totalPoints": len(points)},
                "title": f"{x_def['name']} vs {y_def['name']}",
                "xLabel": x_def["name"],
                "yLabel": y_def["name"],
                "tableData": {
                    "columns": [x_def["name"], y_def["name"]],
                    "rows": [[p["x"], p["y"]] for p in points],
                },
                "operation": "chart",
                "datasetId": dataset_id,
            }

        # Bar/Line/Pie: group by X, aggregate Y
        if y_field and y_field in df.columns:
            sub = df[[x_field, y_field]].copy()
            sub[y_field] = pd.to_numeric(sub[y_field], errors="coerce")
            sub = sub.dropna(subset=[y_field])

            agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
            fn = agg_map.get(aggregation, "mean")
            grouped = sub.groupby(x_field)[y_field].agg(fn)
            title = f"{aggregation.capitalize()} of {y_def['name']} by {x_def['name']}"
        else:
            grouped = df[x_field].dropna().value_counts()
            title = f"Count by {x_def['name']}"
            aggregation = "count"

        ascending = sort_order == "asc"
        if sort_by == "label":
            grouped = grouped.sort_index(ascending=ascending)
        else:
            grouped = grouped.sort_values(ascending=ascending)

        if limit and limit > 0:
            grouped = grouped.head(int(limit))

        labels = [str(v) for v in grouped.index.tolist()]
        values = [safe_float(v) for v in grouped.values]

        stats_dict = {"totalRecords": len(df), "categories": len(grouped)}
        if values:
            stats_dict["min"] = safe_float(min(v for v in values if v is not None))
            stats_dict["max"] = safe_float(max(v for v in values if v is not None))
            stats_dict["total"] = safe_float(sum(v for v in values if v is not None))

        return {
            "chartType": chart_type,
            "chartData": {"labels": labels, "values": values},
            "stats": stats_dict,
            "title": title,
            "xLabel": x_def["name"],
            "yLabel": y_def["name"] if y_def else "Count",
            "tableData": {"labels": labels, "values": values},
            "operation": "chart",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
