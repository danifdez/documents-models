from utils.job_registry import job_handler
from common.dataset import load_dataset, safe_float, apply_filters, get_dataset_records, build_dataframe
import pandas as pd


@job_handler("query")
def query(payload) -> dict:
    """Custom query across one or multiple datasets."""
    try:
        dataset_id = payload.get("datasetId")
        dataset_ids = payload.get("datasetIds", [])
        params = payload.get("params", {})

        if not dataset_id and not dataset_ids:
            return {"error": "datasetId or datasetIds is required"}

        if dataset_id:
            _, df, schema = load_dataset(payload)
        else:
            primary_id = dataset_ids[0]
            schema, records = get_dataset_records(primary_id)
            if schema is None:
                return {"error": f"Dataset {primary_id} not found"}
            if not records:
                return {"error": "Primary dataset has no records"}
            df = build_dataframe(schema, records)
            dataset_id = primary_id

        # Merge extra dataframes if provided (multi-dataset query)
        if dataset_ids and len(dataset_ids) > 1:
            join_field = params.get("joinField")
            if join_field:
                for did in dataset_ids[1:]:
                    s, r = get_dataset_records(did)
                    if s and r:
                        extra_df = build_dataframe(s, r)
                        if join_field in df.columns and join_field in extra_df.columns:
                            df = df.merge(extra_df, on=join_field, how="inner", suffixes=("", "_r"))

        df = apply_filters(df, params.get("filters", []))

        if len(df) == 0:
            return {"error": "No data after applying filters", "chartType": "none"}

        select_fields = params.get("select", [])
        group_by_field = params.get("groupBy")
        agg_fn = params.get("fn", "count")
        chart_type = params.get("chartType", "bar")

        if group_by_field and group_by_field in df.columns:
            if select_fields and len(select_fields) > 0:
                value_field = select_fields[0]
                if value_field in df.columns:
                    sub = df[[group_by_field, value_field]].copy()
                    sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
                    agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
                    fn = agg_map.get(agg_fn, "mean")
                    grouped = sub.groupby(group_by_field)[value_field].agg(fn).sort_values(ascending=False)
                else:
                    grouped = df.groupby(group_by_field).size().sort_values(ascending=False)
                    grouped.name = "count"
            else:
                grouped = df.groupby(group_by_field).size().sort_values(ascending=False)
                grouped.name = "count"

            labels = [str(v) for v in grouped.index.tolist()]
            values = [safe_float(v) for v in grouped.values]

            return {
                "chartType": chart_type,
                "chartData": {"labels": labels, "values": values},
                "stats": {"groupCount": len(grouped), "totalRecords": len(df)},
                "tableData": {"labels": labels, "values": values},
                "operation": "query",
                "datasetId": dataset_id,
            }

        if select_fields:
            cols = [c for c in select_fields if c in df.columns]
            if cols:
                sub = df[cols].head(500)
                return {
                    "chartType": "none",
                    "stats": {"totalRecords": len(df), "returnedRecords": len(sub)},
                    "tableData": {
                        "columns": cols,
                        "rows": sub.values.tolist(),
                    },
                    "operation": "query",
                    "datasetId": dataset_id,
                }

        return {
            "chartType": "none",
            "stats": {"totalRecords": len(df)},
            "tableData": {
                "columns": list(df.columns),
                "rows": df.head(500).values.tolist(),
            },
            "operation": "query",
            "datasetId": dataset_id,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
