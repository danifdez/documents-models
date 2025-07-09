TASK_HANDLERS = {}

def job_handler(job_type) -> dict:
    def decorator(func):
        TASK_HANDLERS[job_type] = func
        return func
    return decorator