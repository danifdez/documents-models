import json
import logging
import os
from pathlib import Path
from typing import Iterable
from uuid import UUID

from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
)

ACK_CODES = {
    "received",
    "duplicate",
    "stale_attempt",
    "result_conflict",
    "rejected",
}
logger = logging.getLogger(__name__)


class ResultOutbox:
    def __init__(self, data_dir: str | Path) -> None:
        self.directory = Path(data_dir) / ".pending_step_results"

    def store(
        self, result: dict, artifacts: list[dict] | None = None
    ) -> None:
        attempt_id = self._attempt_id(result)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{attempt_id}.json"
        temporary = self.directory / f"{attempt_id}.tmp"
        encoded = json.dumps(
            {"result": result, "artifacts": artifacts or []},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def deliver_all(
        self,
        client: ExecutionProtocolClient,
        excluded_attempt_ids: Iterable[str] = (),
    ) -> None:
        excluded = set(excluded_attempt_ids)
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.json")):
            pending = self._load(path)
            attempt_id = pending["result"]["attemptId"]
            if attempt_id in excluded:
                continue
            self._deliver(client, path, pending)

    def pending_attempt_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return [path.stem for path in sorted(self.directory.glob("*.json"))]

    def _load(self, path: Path) -> dict:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("result"), dict)
            or not isinstance(value.get("artifacts"), list)
        ):
            raise RuntimeError("Pending step result is invalid")
        attempt_id = self._attempt_id(value["result"])
        if path.stem != attempt_id:
            raise RuntimeError("Pending step result identity does not match")
        return value

    def _deliver(
        self,
        client: ExecutionProtocolClient,
        path: Path,
        pending: dict,
    ) -> None:
        result = pending["result"]
        for artifact in pending["artifacts"]:
            ack = client.upload_artifact(result["attemptId"], artifact)
            if ack.get("code") not in {"received", "duplicate"}:
                if ack.get("code") == "stale_attempt":
                    break
                raise ProtocolTransportError(
                    f"Output artifact rejected: {ack.get('code')}"
                )
        ack = client.submit_result(result)
        code = ack.get("code")
        if code not in ACK_CODES:
            raise ProtocolTransportError(f"Unknown result ACK: {code}")
        path.unlink(missing_ok=True)
        logger.info(
            "Result %s acknowledged as %s", result.get("attemptId"), code
        )

    @staticmethod
    def _attempt_id(result: dict) -> str:
        attempt_id = result.get("attemptId")
        if not isinstance(attempt_id, str):
            raise RuntimeError("Pending step result has no attempt identity")
        try:
            return str(UUID(attempt_id))
        except ValueError as error:
            raise RuntimeError(
                "Pending step result has an invalid attempt identity"
            ) from error
