"""
Shared utilities for dataset tasks.

Provides common functions for building DataFrames from immutable assignment
artifacts and applying filters.
"""

import json
import pandas as pd
import numpy as np


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
    """Load one dataset from the assignment artifact."""
    dataset_id = payload.get("datasetId")
    if not dataset_id:
        raise ValueError("datasetId is required")

    schema, records = get_dataset_records(payload, dataset_id)
    if schema is None:
        raise ValueError(f"Dataset {dataset_id} is missing from the assignment")
    if not records:
        raise ValueError("Dataset has no records")

    df = build_dataframe(schema, records)
    return dataset_id, df, schema


def normalize_fk_value(val):
    """Normalize a FK value to its canonical string form.

    Handles float-to-int conversion (5.0 -> '5') so that values
    coming from pandas numeric columns match stored string/int values.
    """
    if val is None:
        return None
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def get_dataset_records(payload, dataset_id):
    """Return ``(schema, records)`` from the canonical datasets artifact."""
    bundle = _dataset_bundle(payload)
    for snapshot in bundle.get("datasets", []):
        if snapshot.get("datasetId") != dataset_id:
            continue
        schema = snapshot.get("schema")
        raw_records = snapshot.get("records")
        if not isinstance(schema, list) or not isinstance(raw_records, list):
            raise ValueError(f"Dataset {dataset_id} snapshot is invalid")
        records = []
        for record in raw_records:
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(f"Dataset {dataset_id} record is invalid")
            data = record.get("data")
            if not isinstance(data, dict):
                raise ValueError(f"Dataset {dataset_id} record data is invalid")
            records.append((record["id"], data))
        return schema, records
    return None, []


def resolve_fk_labels(schema, field_key, raw_values):
    """Resolve linked labels frozen into the assignment schema."""
    field = next((item for item in schema if item.get("key") == field_key), None)
    labels = field.get("linkedLabels", {}) if field else {}
    if not isinstance(labels, dict):
        return {}
    wanted = {
        normalize_fk_value(value)
        for value in raw_values
        if normalize_fk_value(value) is not None
    }
    return {
        str(key): str(value)
        for key, value in labels.items()
        if str(key) in wanted
    }


def _dataset_bundle(payload):
    cached = payload.get("_dataset_bundle")
    if isinstance(cached, dict):
        return cached
    artifact = (payload.get("_input_artifacts") or {}).get("datasets")
    if artifact is None:
        raise ValueError("Dataset analysis step is missing its datasets artifact")
    try:
        raw = artifact.decode("utf-8") if isinstance(artifact, bytes) else artifact
        bundle = json.loads(raw) if isinstance(raw, str) else raw
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Dataset analysis artifact is invalid JSON") from error
    if not isinstance(bundle, dict) or bundle.get("schemaVersion") != (
        "dataset-analysis-input/1"
    ):
        raise ValueError("Dataset analysis artifact has an invalid schema")
    if not isinstance(bundle.get("datasets"), list):
        raise ValueError("Dataset analysis artifact requires datasets")
    payload["_dataset_bundle"] = bundle
    return bundle
