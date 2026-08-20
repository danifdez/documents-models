import http.client
import json
import os
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


class ExecutionIngestClient:
    def __init__(self, backend_url: str, token: str, timeout: float):
        backend = urlsplit(backend_url.rstrip("/"))
        self._scheme = backend.scheme
        self._host = backend.hostname or ""
        self._port = backend.port
        self._path = backend.path.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._connection = None

    @classmethod
    def from_environment(cls, token: str) -> "ExecutionIngestClient":
        try:
            timeout = max(
                0.05,
                float(os.environ.get("EXECUTION_HTTP_TIMEOUT_SECONDS", "2")),
            )
        except ValueError:
            timeout = 2
        return cls(
            os.environ.get("BACKEND_URL", "http://localhost:3000"),
            token,
            timeout,
        )

    def post(
        self,
        root_execution_id: str,
        suffix: str,
        body: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        path = (
            f"{self._path}/executions/internal/"
            f"{root_execution_id}/{suffix}"
        )
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(2):
            try:
                if self._connection is None:
                    connection_type = (
                        http.client.HTTPSConnection
                        if self._scheme == "https"
                        else http.client.HTTPConnection
                    )
                    self._connection = connection_type(
                        self._host,
                        self._port,
                        timeout=self._timeout,
                    )
                self._connection.request(
                    "POST",
                    path,
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Execution-Ingest-Token": self._token,
                    },
                )
                response = self._connection.getresponse()
                raw = response.read().decode("utf-8")
                if response.status >= 400:
                    raise RuntimeError(
                        f"Execution ingestion returned HTTP {response.status}: "
                        f"{raw[:200]}"
                    )
                return json.loads(raw) if raw else {}
            except (http.client.HTTPException, ConnectionError, OSError):
                if self._connection is not None:
                    self._connection.close()
                self._connection = None
                if attempt == 1:
                    raise
        return None
