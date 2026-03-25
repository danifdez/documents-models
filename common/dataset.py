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


def get_multiple_datasets(dataset_ids):
    """Fetch multiple datasets and return list of (schema, dataframe) tuples."""
    results = []
    for did in dataset_ids:
        schema, records = get_dataset_records(did)
        if schema is not None and records:
            df = build_dataframe(schema, records)
            results.append((schema, df))
    return results
