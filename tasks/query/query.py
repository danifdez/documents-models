from common.execution_registry import execution_handler
from common.dataset import load_dataset, safe_float, apply_filters, get_dataset_records, build_dataframe, dataset_labels
import pandas as pd


def _load_query_dataset(payload):
    dataset_id = payload.get("datasetId")
    dataset_ids = payload.get("datasetIds", [])

    if not dataset_id and not dataset_ids:
        raise ValueError("datasetId or datasetIds is required")

    if dataset_id:
        _, df, schema = load_dataset(payload)
    else:
        primary_id = dataset_ids[0]
        schema, records = get_dataset_records(payload, primary_id)
        if schema is None:
            raise ValueError(f"Dataset {primary_id} not found")
        if not records:
            raise ValueError("Primary dataset has no records")
        df = build_dataframe(schema, records)
        dataset_id = primary_id
    return dataset_id, df, schema


def _join_datasets(df, payload, join_field):
    if not join_field or join_field not in df.columns:
        return df
    dataset_ids = payload.get("datasetIds") or []
    for dataset_id in dataset_ids[1:]:
        schema, records = get_dataset_records(payload, dataset_id)
        if not schema or not records:
            continue
        extra_df = build_dataframe(schema, records)
        if join_field in extra_df.columns:
            df = df.merge(extra_df, on=join_field, how="inner", suffixes=("", "_r"))
    return df


def _grouped_query(df, schema, dataset_id, group_field, value_field, aggregation, chart_type):
    if value_field is not None and value_field in df.columns:
        sub = df[[group_field, value_field]].copy()
        sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
        supported = {"mean", "sum", "count", "min", "max", "median"}
        grouped = sub.groupby(group_field)[value_field].agg(aggregation if aggregation in supported else "mean")
    else:
        grouped = df.groupby(group_field).size()
        grouped.name = "count"
    grouped = grouped.sort_values(ascending=False)
    labels = dataset_labels(schema, group_field, grouped.index.tolist())
    values = [safe_float(value) for value in grouped.values]
    return {
        "chartType": chart_type,
        "chartData": {"labels": labels, "values": values},
        "stats": {"groupCount": len(grouped), "totalRecords": len(df)},
        "tableData": {"labels": labels, "values": values},
        "operation": "query",
        "datasetId": dataset_id,
    }


@execution_handler("query")
def query(payload) -> dict:
    """Custom query across one or multiple datasets."""
    try:
        params = payload.get("params", {})
        dataset_id, df, schema = _load_query_dataset(payload)
        df = _join_datasets(df, payload, params.get("joinField"))
        df = apply_filters(df, params.get("filters", []))

        if len(df) == 0:
            raise ValueError("No data after applying filters")

        select_fields = params.get("select", [])
        group_by_field = params.get("groupBy")
        agg_fn = params.get("fn", "count")
        chart_type = params.get("chartType", "bar")

        if group_by_field and group_by_field in df.columns:
            value_field = select_fields[0] if select_fields else None
            return _grouped_query(
                df, schema, dataset_id, group_by_field, value_field, agg_fn, chart_type,
            )

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

    except Exception as error:
        raise RuntimeError("Dataset query failed") from error
