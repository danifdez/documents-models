from utils.job_registry import job_handler
from database.job import get_job_database
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import json


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


# ── Analysis operations ──────────────────────────────────────────────

def op_distribution(df, schema, params):
    """Distribution of a single field: histogram for numeric, frequency for categorical."""
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
        result["chartType"] = "bar"
        result["chartData"] = {
            "labels": [str(v) for v in freq.index.tolist()],
            "values": [int(c) for c in freq.values],
        }
        result["stats"] = {"uniqueCount": int(col.nunique())}

    # Raw data for table
    result["tableData"] = result["chartData"]

    return result


def op_correlation(df, schema, params):
    """Scatter plot + correlation between two numeric fields."""
    field1 = params.get("field1")
    field2 = params.get("field2")

    f1_def = next((f for f in schema if f["key"] == field1), None)
    f2_def = next((f for f in schema if f["key"] == field2), None)
    if not f1_def or not f2_def:
        return {"error": "Both fields must exist"}

    if field1 not in df.columns or field2 not in df.columns:
        return {"error": "Fields not found in data"}

    valid = df[[field1, field2]].dropna()
    x = pd.to_numeric(valid[field1], errors="coerce")
    y = pd.to_numeric(valid[field2], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"error": "Not enough data points (minimum 3)"}

    r, p_value = scipy_stats.pearsonr(x, y)

    # Regression line
    slope, intercept = np.polyfit(x, y, 1)

    result = {
        "chartType": "scatter",
        "field1": field1,
        "field1Name": f1_def["name"],
        "field2": field2,
        "field2Name": f2_def["name"],
        "chartData": {
            "points": [{"x": safe_float(xi), "y": safe_float(yi)} for xi, yi in zip(x, y)],
            "regression": {
                "slope": safe_float(slope),
                "intercept": safe_float(intercept),
                "xRange": [safe_float(x.min()), safe_float(x.max())],
            },
        },
        "stats": {
            "correlation": safe_float(r),
            "pValue": safe_float(p_value),
            "rSquared": safe_float(r ** 2),
            "n": len(x),
            "slope": safe_float(slope),
            "intercept": safe_float(intercept),
        },
        "tableData": {
            "labels": [f1_def["name"], f2_def["name"]],
            "rows": [[safe_float(xi), safe_float(yi)] for xi, yi in zip(x, y)],
        },
    }

    return result


def op_group_by(df, schema, params):
    """Aggregate a numeric field grouped by a categorical field."""
    value_field = params.get("valueField")
    group_field = params.get("groupField")
    agg_fn = params.get("fn", "mean")

    v_def = next((f for f in schema if f["key"] == value_field), None)
    g_def = next((f for f in schema if f["key"] == group_field), None)
    if not v_def or not g_def:
        return {"error": "Both fields must exist"}

    if value_field not in df.columns or group_field not in df.columns:
        return {"error": "Fields not found in data"}

    sub = df[[group_field, value_field]].copy()
    sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
    sub = sub.dropna()

    agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
    fn = agg_map.get(agg_fn, "mean")

    grouped = sub.groupby(group_field)[value_field].agg(fn).sort_values(ascending=False)

    result = {
        "chartType": "bar",
        "field": value_field,
        "fieldName": v_def["name"],
        "groupField": group_field,
        "groupFieldName": g_def["name"],
        "fn": agg_fn,
        "chartData": {
            "labels": [str(v) for v in grouped.index.tolist()],
            "values": [safe_float(v) for v in grouped.values],
        },
        "stats": {
            "groupCount": len(grouped),
            "totalRecords": len(sub),
        },
        "tableData": {
            "labels": [str(v) for v in grouped.index.tolist()],
            "values": [safe_float(v) for v in grouped.values],
        },
    }

    return result


def op_time_series(df, schema, params):
    """Trend of a numeric field over a date field."""
    date_field = params.get("dateField")
    value_field = params.get("valueField")
    period = params.get("period", "ME")  # D, W, ME, QE, YE

    # Map legacy offset aliases to modern pandas equivalents
    period_map = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    period = period_map.get(period, period)

    d_def = next((f for f in schema if f["key"] == date_field), None)
    v_def = next((f for f in schema if f["key"] == value_field), None)
    if not d_def or not v_def:
        return {"error": "Both fields must exist"}

    sub = df[[date_field, value_field]].copy()
    sub[date_field] = pd.to_datetime(sub[date_field], errors="coerce")
    sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
    sub = sub.dropna()

    if len(sub) == 0:
        return {"error": "No valid data points"}

    grouped = sub.set_index(date_field).resample(period)[value_field].mean().dropna()

    result = {
        "chartType": "line",
        "dateField": date_field,
        "dateFieldName": d_def["name"],
        "valueField": value_field,
        "valueFieldName": v_def["name"],
        "period": period,
        "chartData": {
            "labels": [str(d) for d in grouped.index],
            "values": [safe_float(v) for v in grouped.values],
        },
        "stats": {
            "dataPoints": len(grouped),
            "trend": safe_float(np.polyfit(range(len(grouped)), grouped.values, 1)[0]) if len(grouped) > 1 else None,
        },
        "tableData": {
            "labels": [str(d) for d in grouped.index],
            "values": [safe_float(v) for v in grouped.values],
        },
    }

    return result


def op_outliers(df, schema, params):
    """Detect outliers in a numeric field using IQR method."""
    field_key = params.get("field")
    field_def = next((f for f in schema if f["key"] == field_key), None)
    if not field_def or field_key not in df.columns:
        return {"error": f"Field '{field_key}' not found"}

    col = pd.to_numeric(df[field_key], errors="coerce").dropna()
    if len(col) < 4:
        return {"error": "Not enough data points"}

    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = (col < lower) | (col > upper)
    outliers = col[outlier_mask]
    normal = col[~outlier_mask]

    # Build all points for strip/jitter plot
    all_points = []
    for idx, val in col.items():
        is_outlier = bool(outlier_mask.loc[idx])
        record_id = int(df.loc[idx, "_id"]) if "_id" in df.columns else idx
        all_points.append({
            "value": safe_float(val),
            "isOutlier": is_outlier,
            "recordId": record_id,
        })

    # Z-scores
    z_scores = np.abs((col - col.mean()) / col.std()) if col.std() > 0 else pd.Series([0] * len(col), index=col.index)
    z_outlier_count = int((z_scores > 3).sum())

    result = {
        "chartType": "box",
        "field": field_key,
        "fieldName": field_def["name"],
        "chartData": {
            "min": safe_float(col.min()),
            "q1": safe_float(q1),
            "median": safe_float(col.median()),
            "q3": safe_float(q3),
            "max": safe_float(col.max()),
            "lowerFence": safe_float(lower),
            "upperFence": safe_float(upper),
            "outliers": [safe_float(v) for v in outliers.tolist()],
            "allPoints": all_points,
        },
        "stats": {
            "totalCount": len(col),
            "outlierCount": len(outliers),
            "outlierPercent": safe_float(len(outliers) / len(col) * 100),
            "zScoreOutliers": z_outlier_count,
            "lowerBound": safe_float(lower),
            "upperBound": safe_float(upper),
            "iqr": safe_float(iqr),
            "mean": safe_float(col.mean()),
            "std": safe_float(col.std()),
        },
        "tableData": {
            "columns": ["Value", "Record ID", "Z-Score"],
            "rows": [
                [safe_float(v), int(df.loc[idx, "_id"]) if "_id" in df.columns else idx,
                 safe_float(z_scores.loc[idx])]
                for idx, v in outliers.items()
            ],
        },
    }

    return result


def op_summary(df, schema, params):
    """Quick summary overview of all fields."""
    fields_summary = []
    for field in schema:
        key = field["key"]
        if key not in df.columns:
            continue
        col = df[key]
        non_null = col.dropna()
        entry = {
            "field": key,
            "name": field["name"],
            "type": field["type"],
            "totalCount": len(col),
            "nonNullCount": len(non_null),
            "nullCount": int(col.isna().sum()),
            "nullPercent": safe_float(col.isna().sum() / len(col) * 100) if len(col) > 0 else 0,
        }

        if field["type"] == "number" and len(non_null) > 0:
            numeric = non_null.astype(float)
            entry.update({
                "mean": safe_float(numeric.mean()),
                "median": safe_float(numeric.median()),
                "std": safe_float(numeric.std()),
                "min": safe_float(numeric.min()),
                "max": safe_float(numeric.max()),
            })
        elif field["type"] in ("text", "select") and len(non_null) > 0:
            entry["uniqueCount"] = int(non_null.nunique())
            entry["topValue"] = str(non_null.value_counts().index[0]) if len(non_null) > 0 else None
        elif field["type"] in ("date", "datetime") and len(non_null) > 0:
            dates = pd.to_datetime(non_null, errors="coerce").dropna()
            if len(dates) > 0:
                entry["min"] = str(dates.min())
                entry["max"] = str(dates.max())

        fields_summary.append(entry)

    return {
        "chartType": "none",
        "recordCount": len(df),
        "fieldCount": len(schema),
        "fields": fields_summary,
        "tableData": {"fields": fields_summary},
    }


def op_chart(df, schema, params):
    """Generate chart data for a custom visualization.

    params:
      - chartType: bar, line, pie, scatter
      - xField: field for X axis / labels
      - yField: field for Y axis / values (optional — defaults to count)
      - aggregation: sum, mean, count, min, max, median
      - sortBy: "label" or "value" (optional)
      - sortOrder: "asc" or "desc" (optional)
      - limit: max categories to show (optional, default 20)
    """
    chart_type = params.get("chartType", "bar")
    x_field = params.get("xField")
    y_field = params.get("yField")
    aggregation = params.get("aggregation", "count")
    sort_by = params.get("sortBy", "value")
    sort_order = params.get("sortOrder", "desc")
    limit = params.get("limit", 20)

    # Apply filters if provided
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
            "stats": {
                "totalPoints": len(points),
            },
            "title": f"{x_def['name']} vs {y_def['name']}",
            "xLabel": x_def["name"],
            "yLabel": y_def["name"],
            "tableData": {
                "columns": [x_def["name"], y_def["name"]],
                "rows": [[p["x"], p["y"]] for p in points],
            },
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
    }


def op_query(df, schema, params, extra_dfs=None):
    """Custom query across one or multiple datasets.

    params:
      - select: list of field keys to include
      - groupBy: field key to group by (optional)
      - fn: aggregation function (optional: count, sum, mean, min, max, median)
      - filters: list of {field, operator, value} (optional)
      - chartType: preferred chart type (bar, line, pie) (optional)
      - joinField: field to join datasets on (optional, for multi-dataset)
    """
    # Merge extra dataframes if provided (multi-dataset query)
    if extra_dfs:
        join_field = params.get("joinField")
        if join_field:
            for extra_df in extra_dfs:
                if join_field in df.columns and join_field in extra_df.columns:
                    df = df.merge(extra_df, on=join_field, how="inner", suffixes=("", "_r"))

    # Apply filters
    df = apply_filters(df, params.get("filters", []))

    if len(df) == 0:
        return {"error": "No data after applying filters", "chartType": "none"}

    select_fields = params.get("select", [])
    group_by = params.get("groupBy")
    agg_fn = params.get("fn", "count")
    chart_type = params.get("chartType", "bar")

    # If groupBy is provided, do aggregation
    if group_by and group_by in df.columns:
        if select_fields and len(select_fields) > 0:
            value_field = select_fields[0]
            if value_field in df.columns:
                sub = df[[group_by, value_field]].copy()
                sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
                agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max", "median": "median"}
                fn = agg_map.get(agg_fn, "mean")
                grouped = sub.groupby(group_by)[value_field].agg(fn).sort_values(ascending=False)
            else:
                grouped = df.groupby(group_by).size().sort_values(ascending=False)
                grouped.name = "count"
        else:
            grouped = df.groupby(group_by).size().sort_values(ascending=False)
            grouped.name = "count"

        labels = [str(v) for v in grouped.index.tolist()]
        values = [safe_float(v) for v in grouped.values]

        return {
            "chartType": chart_type,
            "chartData": {"labels": labels, "values": values},
            "stats": {"groupCount": len(grouped), "totalRecords": len(df)},
            "tableData": {"labels": labels, "values": values},
        }

    # No groupBy — return raw data or simple stats
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
            }

    return {
        "chartType": "none",
        "stats": {"totalRecords": len(df)},
        "tableData": {
            "columns": list(df.columns),
            "rows": df.head(500).values.tolist(),
        },
    }


def op_correlation_matrix(df, schema, params):
    """NxN Pearson correlation matrix for all (or selected) numeric fields."""
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

    # Find strong correlations
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
    }


def op_pivot_table(df, schema, params):
    """Cross-tabulation / pivot table."""
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

    # Also provide stacked bar chart data
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
    }


def get_multiple_datasets(dataset_ids):
    """Fetch multiple datasets and return list of (schema, dataframe) tuples."""
    results = []
    for did in dataset_ids:
        schema, records = get_dataset_records(did)
        if schema is not None and records:
            df = build_dataframe(schema, records)
            results.append((schema, df))
    return results


# ── Operation dispatch ────────────────────────────────────────────────

OPERATIONS = {
    "distribution": op_distribution,
    "correlation": op_correlation,
    "correlation_matrix": op_correlation_matrix,
    "group_by": op_group_by,
    "time_series": op_time_series,
    "outliers": op_outliers,
    "pivot_table": op_pivot_table,
    "summary": op_summary,
    "query": op_query,
    "chart": op_chart,
}


@job_handler("dataset-stats")
def dataset_statistics(payload) -> dict:
    """Run a specific analysis operation on a dataset.

    Payload:
      - datasetId: int
      - operation: str (distribution, correlation, group_by, time_series, outliers, summary)
      - params: dict (operation-specific parameters)

    Returns operation result with chartType, chartData, stats, tableData.
    """
    try:
        dataset_id = payload.get("datasetId")
        dataset_ids = payload.get("datasetIds", [])
        operation = payload.get("operation", "summary")
        params = payload.get("params", {})

        if not dataset_id and not dataset_ids:
            return {"error": "datasetId or datasetIds is required"}

        # Single dataset
        if dataset_id:
            schema, records = get_dataset_records(dataset_id)
            if schema is None:
                return {"error": f"Dataset {dataset_id} not found"}
            if not records:
                return {"error": "Dataset has no records"}
            df = build_dataframe(schema, records)
        else:
            # Multi-dataset: use first as primary
            primary_id = dataset_ids[0]
            schema, records = get_dataset_records(primary_id)
            if schema is None:
                return {"error": f"Dataset {primary_id} not found"}
            if not records:
                return {"error": "Primary dataset has no records"}
            df = build_dataframe(schema, records)
            dataset_id = primary_id

        op_fn = OPERATIONS.get(operation)
        if not op_fn:
            return {"error": f"Unknown operation: {operation}. Available: {', '.join(OPERATIONS.keys())}"}

        # For query operation with multiple datasets, pass extra dfs
        if operation == "query" and len(dataset_ids) > 1:
            extra_dfs = []
            for did in dataset_ids[1:]:
                s, r = get_dataset_records(did)
                if s and r:
                    extra_dfs.append(build_dataframe(s, r))
            result = op_fn(df, schema, params, extra_dfs=extra_dfs)
        else:
            result = op_fn(df, schema, params)

        result["operation"] = operation
        result["datasetId"] = dataset_id

        return result

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}
