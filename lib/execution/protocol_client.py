import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from worker.identity import WORKER_ID, WORKER_NAME, worker_data_dir


class ProtocolTransportError(RuntimeError):
    pass


class WorkerAuthenticationError(ProtocolTransportError):
    pass


class ExecutionProtocolClient:
    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "BACKEND_URL", "http://localhost:3000"
        ).rstrip("/")
        self.enrollment_token = os.environ.get("MODELS_ENROLLMENT_TOKEN", "")
        self.credential_path = Path(worker_data_dir()) / ".worker_credential"
        self.credential = self._read_credential()

    def ensure_registered(
        self, capabilities: list[str], metadata: Dict[str, Any]
    ) -> None:
        if self.credential:
            return
        if not self.enrollment_token:
            raise RuntimeError("MODELS_ENROLLMENT_TOKEN is required")
        response = self._request(
            "/models-work/register",
            {
                "workerId": WORKER_ID,
                "name": WORKER_NAME,
                "capabilities": capabilities,
                "metadata": metadata,
            },
            {"x-models-enrollment-token": self.enrollment_token},
        )
        credential = response.get("credential")
        if not isinstance(credential, str) or not credential:
            raise ProtocolTransportError("Registration omitted worker credential")
        self.credential_path.parent.mkdir(parents=True, exist_ok=True)
        self.credential_path.write_text(credential, encoding="utf-8")
        self.credential_path.chmod(0o600)
        self.credential = credential

    def heartbeat(
        self, capabilities: list[str], metadata: Dict[str, Any]
    ) -> None:
        self._authenticated_request(
            "/models-work/heartbeat",
            {"capabilities": capabilities, "metadata": metadata},
        )

    def claim(
        self,
        capabilities: list[str],
        step_kinds: list[str],
        lease_duration_ms: int,
    ) -> Optional[Dict[str, Any]]:
        return self._authenticated_request(
            "/models-work/claim",
            {
                "capabilities": capabilities,
                "stepKinds": step_kinds,
                "leaseDurationMs": lease_duration_ms,
            },
        )

    def start(self, attempt_id: str) -> None:
        self._authenticated_request(
            f"/models-work/attempts/{attempt_id}/start", {}
        )

    def submit_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self._authenticated_request("/models-work/results", result)

    def reset_credential(self) -> None:
        self.credential_path.unlink(missing_ok=True)
        self.credential = ""

    def _authenticated_request(
        self, path: str, body: Dict[str, Any]
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
        )

    def _request(
        self, path: str, body: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code == 401 and path != "/models-work/register":
                raise WorkerAuthenticationError(
                    f"Backend rejected worker credential for {path}"
                ) from error
            raise ProtocolTransportError(
                f"Backend rejected {path}: HTTP {error.code} {detail}"
            ) from error
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
            return self.credential_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
