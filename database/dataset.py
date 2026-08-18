"""Raw PostgreSQL access for datasets.

Fetches dataset schemas/records and resolves FK display labels.
Pure DataFrame/value transformations live in ``common.dataset``.
"""

import json

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


def resolve_fk_labels(schema, field_key, raw_values):
    """Resolve FK IDs to display values for a linked field.

    Returns a dict mapping raw value (str) -> display string.
    If the field is not a FK or resolution fails, returns empty dict.
    """
    # Deferred import: common.dataset imports this module at load time.
    from common.dataset import normalize_fk_value

    field_def = next((f for f in schema if f["key"] == field_key), None)
    if not field_def:
        return {}

    linked_dataset_id = field_def.get("linkedDatasetId")
    if not linked_dataset_id:
        return {}

    linked_display_field = field_def.get("linkedDisplayField")
    linked_lookup_field = field_def.get("linkedLookupField")

    norm_values = [normalize_fk_value(v) for v in raw_values if v is not None]
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
                    key = normalize_fk_value(data.get(linked_lookup_field))
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
