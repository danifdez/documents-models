from common.execution_registry import execution_handler


@execution_handler("data-source-sync")
def data_source_sync(payload, state=None, ctx=None):
    data_source_id = payload.get("dataSourceId")
    if (
        isinstance(data_source_id, bool)
        or not isinstance(data_source_id, int)
        or data_source_id < 1
    ):
        raise ValueError("data-source-sync dataSourceId must be a positive integer")
    return {"dataSourceId": data_source_id, "ready": True}
