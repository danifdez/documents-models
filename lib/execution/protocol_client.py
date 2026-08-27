import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from worker.identity import WORKER_ID, WORKER_NAME, worker_data_dir
from lib.execution.private_storage import (
    secure_existing_file,
    write_private_text,
)


class ProtocolTransportError(RuntimeError):
    pass


class WorkerAuthenticationError(ProtocolTransportError):
    pass


class ProtocolRejectionError(ProtocolTransportError):
    def __init__(
        self,
        path: str,
        status_code: int,
        error_code: str | None,
        detail: str,
    ) -> None:
        self.path = path
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(
            f"Backend rejected {path}: HTTP {status_code} {detail}"
        )


class ExecutionProtocolClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "BACKEND_URL", "http://localhost:3000"
        ).rstrip("/")
        self.enrollment_token = os.environ.get("MODELS_ENROLLMENT_TOKEN", "")
        self.credential_path = Path(worker_data_dir()) / ".worker_credential"
        self.credential = self._read_credential()

    def ensure_registered(
        self,
        capabilities: list[str],
        step_kinds: list[str],
        maximum_concurrency: int,
        metadata: Dict[str, Any],
    ) -> None:
        if self.credential:
            return
        if not self.enrollment_token:
            raise RuntimeError("MODELS_ENROLLMENT_TOKEN is required")
        response = self._request(
            "/models-work/register",
            {
                "protocolVersion": "step-protocol/1",
                "workerId": WORKER_ID,
                "name": WORKER_NAME,
                "capabilities": capabilities,
                "stepKinds": step_kinds,
                "maximumConcurrency": maximum_concurrency,
                "metadata": metadata,
            },
            {"x-models-enrollment-token": self.enrollment_token},
        )
        credential = response.get("credential")
        if not isinstance(credential, str) or not credential:
            raise ProtocolTransportError("Registration omitted worker credential")
        write_private_text(self.credential_path, credential)
        self.credential = credential

    def heartbeat(
        self,
        capabilities: list[str],
        step_kinds: list[str],
        maximum_concurrency: int,
        metadata: Dict[str, Any],
    ) -> None:
        self._authenticated_request(
            "/models-work/heartbeat",
            {
                "protocolVersion": "step-protocol/1",
                "capabilities": capabilities,
                "stepKinds": step_kinds,
                "maximumConcurrency": maximum_concurrency,
                "metadata": metadata,
            },
        )

    def claim(
        self,
        capabilities: list[str],
        step_kinds: list[str],
        lease_duration_ms: int,
        wait_timeout_ms: int,
    ) -> Optional[Dict[str, Any]]:
        return self._authenticated_request(
            "/models-work/claim",
            {
                "capabilities": capabilities,
                "stepKinds": step_kinds,
                "leaseDurationMs": lease_duration_ms,
                "waitTimeoutMs": wait_timeout_ms,
            },
            timeout=max(15, wait_timeout_ms / 1000 + 5),
        )

    def start(self, attempt_id: str) -> None:
        self._authenticated_request(
            f"/models-work/attempts/{attempt_id}/start", {}
        )

    def renew_lease(
        self, attempt_id: str, lease_duration_ms: int
    ) -> Dict[str, Any]:
        return self._authenticated_request(
            f"/models-work/attempts/{attempt_id}/lease",
            {"leaseDurationMs": lease_duration_ms},
        )

    def read_control(self, attempt_id: str) -> Dict[str, Any]:
        if not self.credential:
            raise RuntimeError("Worker is not registered")
        return self._authenticated_get(
            f"/models-work/attempts/{attempt_id}/control"
        )

    def submit_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self._authenticated_request("/models-work/results", result)

    def upload_artifact(
        self, attempt_id: str, artifact: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._authenticated_request(
            f"/models-work/attempts/{attempt_id}/artifacts", artifact
        )

    def download_artifact(self, attempt_id: str, artifact_id: str) -> bytes:
        if not self.credential:
            raise RuntimeError("Worker is not registered")
        path = f"/models-work/attempts/{attempt_id}/artifacts/{artifact_id}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={
                "x-worker-id": WORKER_ID,
                "x-worker-credential": self.credential,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 401:
                raise WorkerAuthenticationError(
                    "Backend rejected worker credential for artifact download"
                ) from error
            detail = error.read().decode("utf-8", errors="replace")
            raise _rejection(path, error.code, detail) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ProtocolTransportError(
                f"Backend artifact download failed: {error}"
            ) from error

    def _authenticated_get(self, path: str) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={
                "x-worker-id": WORKER_ID,
                "x-worker-credential": self.credential,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 401:
                raise WorkerAuthenticationError(
                    f"Backend rejected worker credential for {path}"
                ) from error
            detail = error.read().decode("utf-8", errors="replace")
            raise _rejection(path, error.code, detail) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ProtocolTransportError(
                f"Backend request failed for {path}: {error}"
            ) from error
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProtocolTransportError(
                f"Backend returned invalid JSON for {path}"
            ) from error
        if not isinstance(decoded, dict):
            raise ProtocolTransportError(
                f"Backend returned an invalid response for {path}"
            )
        return decoded

    def reset_credential(self) -> None:
        self.credential_path.unlink(missing_ok=True)
        self.credential = ""

    def _authenticated_request(
        self,
        path: str,
        body: Dict[str, Any],
        timeout: float = 15,
    ) -> Dict[str, Any]:
        if not self.credential:
            raise RuntimeError("Worker is not registered")
        return self._request(
            path,
            body,
            {
                "x-worker-id": WORKER_ID,
                "x-worker-credential": self.credential,
            },
            timeout,
        )

    def _request(
        self,
        path: str,
        body: Dict[str, Any],
        headers: Dict[str, str],
        timeout: float = 15,
    ) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 401 and path != "/models-work/register":
                raise WorkerAuthenticationError(
                    f"Backend rejected worker credential for {path}"
                ) from error
            raise _rejection(path, error.code, detail) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ProtocolTransportError(
                f"Backend request failed for {path}: {error}"
            ) from error
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProtocolTransportError(
                f"Backend returned invalid JSON for {path}"
            ) from error
        if decoded is None:
            return None
        if not isinstance(decoded, dict):
            raise ProtocolTransportError(
                f"Backend returned an invalid response for {path}"
            )
        return decoded

    def _read_credential(self) -> str:
        try:
            secure_existing_file(self.credential_path)
            return self.credential_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""


def _rejection(
    path: str, status_code: int, detail: str
) -> ProtocolRejectionError:
    error_code = None
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str):
            error_code = message
        elif isinstance(message, list):
            error_code = next(
                (item for item in message if isinstance(item, str)), None
            )
    return ProtocolRejectionError(
        path,
        status_code,
        error_code,
        detail,
    )
