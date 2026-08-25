from common.execution_registry import execution_handler
from common.dataset import load_dataset, safe_float
import pandas as pd
import numpy as np


@execution_handler("time-series")
def time_series(payload) -> dict:
    """Trend of a numeric field over a date field."""
    try:
        dataset_id, df, schema = load_dataset(payload)
        params = payload.get("params", {})

        date_field = params.get("dateField")
        value_field = params.get("valueField")
        period = params.get("period", "ME")

        period_map = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
        period = period_map.get(period, period)

        d_def = next((f for f in schema if f["key"] == date_field), None)
        v_def = next((f for f in schema if f["key"] == value_field), None)
        if not d_def or not v_def:
            raise ValueError("Both fields must exist")

        sub = df[[date_field, value_field]].copy()
        sub[date_field] = pd.to_datetime(sub[date_field], errors="coerce")
        sub[value_field] = pd.to_numeric(sub[value_field], errors="coerce")
        sub = sub.dropna()

        if len(sub) == 0:
            raise ValueError("No valid data points")

        grouped = sub.set_index(date_field).resample(period)[value_field].mean().dropna()

        return {
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
            "operation": "time_series",
            "datasetId": dataset_id,
        }

    except Exception as error:
        raise RuntimeError("Dataset time series analysis failed") from error
