"""
Shared utilities for dataset tasks.

Provides common functions for loading datasets from PostgreSQL,
building DataFrames, and applying filters.
"""

import json
import pandas as pd
import numpy as np
from database.job import get_job_database


def get_dataset_records(dataset_id: int):
    """Fetch dataset schema and records directly from PostgreSQL."""
    db = get_job_database()
    conn = db.get_connection()

    with conn.cursor() as cur:
        cur.execute("SELECT schema FROM datasets WHERE id = %s", (dataset_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None, []

        raw_schema = row["schema"]
        schema = raw_schema if isinstance(raw_schema, list) else json.loads(raw_schema)

        cur.execute(
            "SELECT id, data FROM dataset_records WHERE dataset_id = %s",
            (dataset_id,),
        )
        rows = cur.fetchall()

    conn.close()

    records = [(r["id"], r["data"]) for r in rows]
    return schema, records


def build_dataframe(schema, records):
    """Build a pandas DataFrame from dataset records."""
    if not records:
        return pd.DataFrame()

    rows = []
    for record_id, data in records:
        if isinstance(data, str):
            data = json.loads(data)
        row = {"_id": record_id}
        row.update(data)
        rows.append(row)

    df = pd.DataFrame(rows)

    for field in schema:
        key = field["key"]
        if key not in df.columns:
            continue
        ftype = field["type"]
        if ftype == "number":
            df[key] = pd.to_numeric(df[key], errors="coerce")
        elif ftype in ("date", "datetime"):
            df[key] = pd.to_datetime(df[key], errors="coerce")
        elif ftype == "boolean":
            df[key] = df[key].astype(bool)

    return df


def safe_float(val):
    """Convert a value to a safe float, returning None for NaN/Inf."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return None
    return round(float(val), 6)


def apply_filters(df, filters):
    """Apply a list of filter conditions to a DataFrame."""
    for f in filters:
        field = f.get("field")
        op = f.get("operator", "eq")
        value = f.get("value")
        if field not in df.columns:
            continue
        col = df[field]
        if op == "eq":
            df = df[col.astype(str) == str(value)]
        elif op == "gt":
            df = df[pd.to_numeric(col, errors="coerce") > float(value)]
        elif op == "gte":
            df = df[pd.to_numeric(col, errors="coerce") >= float(value)]
        elif op == "lt":
            df = df[pd.to_numeric(col, errors="coerce") < float(value)]
        elif op == "lte":
            df = df[pd.to_numeric(col, errors="coerce") <= float(value)]
        elif op == "contains":
            df = df[col.astype(str).str.contains(str(value), case=False, na=False)]
    return df


def load_dataset(payload):
    """Load a dataset from payload. Returns (dataset_id, df, schema).

    Raises ValueError if dataset not found or has no records.
    """
    dataset_id = payload.get("datasetId")
    if not dataset_id:
        raise ValueError("datasetId is required")

    schema, records = get_dataset_records(dataset_id)
    if schema is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    if not records:
        raise ValueError("Dataset has no records")

    df = build_dataframe(schema, records)
    return dataset_id, df, schema


def _normalize_fk_value(val):
    """Normalize a FK value to its canonical string form.

    Handles float-to-int conversion (5.0 -> '5') so that values
    coming from pandas numeric columns match stored string/int values.
    """
    if val is None:
        return None
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def resolve_fk_labels(schema, field_key, raw_values):
    """Resolve FK IDs to display values for a linked field.

    Returns a dict mapping raw value (str) -> display string.
    If the field is not a FK or resolution fails, returns empty dict.
    """
    field_def = next((f for f in schema if f["key"] == field_key), None)
    if not field_def:
        return {}

    linked_dataset_id = field_def.get("linkedDatasetId")
    if not linked_dataset_id:
        return {}

    linked_display_field = field_def.get("linkedDisplayField")
    linked_lookup_field = field_def.get("linkedLookupField")

    norm_values = [_normalize_fk_value(v) for v in raw_values if v is not None]
    norm_values = [v for v in norm_values if v is not None and v != ""]
    if not norm_values:
        return {}

    db = get_job_database()
    conn = db.get_connection()
    result_map = {}

    def _extract_display(data):
        if linked_display_field and linked_display_field in data:
            return str(data[linked_display_field])
        first_str = next(
            (str(v) for v in data.values() if isinstance(v, str) and v), None
        )
        return first_str

    try:
        with conn.cursor() as cur:
            if linked_lookup_field:
                placeholders = ",".join(["%s"] * len(norm_values))
                cur.execute(
                    f"SELECT data FROM dataset_records WHERE dataset_id = %s "
                    f"AND data ->> %s IN ({placeholders})",
                    [linked_dataset_id, linked_lookup_field] + norm_values,
                )
                for row in cur.fetchall():
                    data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
                    key = _normalize_fk_value(data.get(linked_lookup_field))
                    display = _extract_display(data)
                    if key and display:
                        result_map[key] = display
            else:
                int_ids = []
                for v in norm_values:
                    try:
                        int_ids.append(int(v))
                    except (ValueError, TypeError):
                        pass
                if not int_ids:
                    return {}
                placeholders = ",".join(["%s"] * len(int_ids))
                cur.execute(
                    f"SELECT id, data FROM dataset_records WHERE dataset_id = %s "
                    f"AND id IN ({placeholders})",
                    [linked_dataset_id] + int_ids,
                )
                for row in cur.fetchall():
                    data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
                    key = str(row["id"])
                    display = _extract_display(data)
                    if display:
                        result_map[key] = display
    finally:
        conn.close()

    return result_map


def get_multiple_datasets(dataset_ids):
    """Fetch multiple datasets and return list of (schema, dataframe) tuples."""
    results = []
    for did in dataset_ids:
        schema, records = get_dataset_records(did)
        if schema is not None and records:
            df = build_dataframe(schema, records)
            results.append((schema, df))
    return results
