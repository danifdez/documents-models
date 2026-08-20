from typing import Callable, Dict, Any

TASK_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def execution_handler(task_type: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        TASK_HANDLERS[task_type] = func
        return func
    return decorator
