import json
import logging
from pathlib import Path
from typing import Iterable
from uuid import UUID

from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolTransportError,
)
from lib.execution.private_storage import (
    ensure_private_directory,
    secure_existing_file,
    write_private_text,
)

ACK_CODES = {
    "received",
    "duplicate",
    "stale_attempt",
    "result_conflict",
    "rejected",
}
ACCEPTED_ARTIFACT_ACK_CODES = {"received", "duplicate"}
TERMINAL_ARTIFACT_ACK_CODES = {"stale_attempt", "artifact_conflict"}
logger = logging.getLogger(__name__)


class ResultOutbox:
    def __init__(self, data_dir: str | Path) -> None:
        self.directory = Path(data_dir) / ".pending_step_results"

    def store(
        self, result: dict, artifacts: list[dict] | None = None
    ) -> None:
        attempt_id = self._attempt_id(result)
        ensure_private_directory(self.directory)
        path = self.directory / f"{attempt_id}.json"
        encoded = json.dumps(
            {"result": result, "artifacts": artifacts or []},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        write_private_text(path, encoded)

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
        secure_existing_file(path)
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
            code = ack.get("code")
            if code in TERMINAL_ARTIFACT_ACK_CODES:
                path.unlink(missing_ok=True)
                logger.info(
                    "Result %s closed after output artifact ACK %s",
                    result.get("attemptId"),
                    code,
                )
                return
            if code not in ACCEPTED_ARTIFACT_ACK_CODES:
                raise ProtocolTransportError(
                    f"Output artifact rejected: {code}"
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
