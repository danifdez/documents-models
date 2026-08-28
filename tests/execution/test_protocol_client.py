import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from lib.execution.protocol_client import (
    ExecutionProtocolClient,
    ProtocolRejectionError,
    ProtocolTransportError,
    WorkerAuthenticationError,
)
from worker.identity import WORKER_ID


class _Response:
    def __init__(self, value: bytes):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.value


class ProtocolClientTest(unittest.TestCase):
    def _client(self, directory: str) -> ExecutionProtocolClient:
        credential = Path(directory) / ".worker_credential"
        credential.write_text("worker-secret", encoding="utf-8")
        with patch(
            "lib.execution.protocol_client.worker_data_dir",
            return_value=directory,
        ):
            return ExecutionProtocolClient()

    def test_reads_control_with_worker_credentials(self):
        requests = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            return _Response(json.dumps({"cancelled": False}).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.urlopen", side_effect=urlopen
        ):
            result = self._client(directory).read_control("attempt-1")

        request, timeout = requests[0]
        self.assertEqual(
            request.full_url,
            "http://localhost:3000/models-work/attempts/attempt-1/control",
        )
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.headers["X-worker-id"], WORKER_ID)
        self.assertEqual(
            request.headers["X-worker-credential"], "worker-secret"
        )
        self.assertEqual(timeout, 15)
        self.assertEqual(result, {"cancelled": False})

    def test_downloads_artifact_as_bytes_with_its_longer_timeout(self):
        requests = []

        def urlopen(request, timeout):
            requests.append((request, timeout))
            return _Response(b"\x00artifact\xff")

        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.urlopen", side_effect=urlopen
        ):
            result = self._client(directory).download_artifact(
                "attempt-1", "artifact-1"
            )

        request, timeout = requests[0]
        self.assertEqual(
            request.full_url,
            "http://localhost:3000/models-work/attempts/attempt-1/"
            "artifacts/artifact-1",
        )
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(timeout, 60)
        self.assertEqual(result, b"\x00artifact\xff")

    def test_preserves_authentication_errors_across_transports(self):
        calls = (
            (
                lambda client: client.renew_lease("attempt-1", 30_000),
                "Backend rejected worker credential for "
                "/models-work/attempts/attempt-1/lease",
            ),
            (
                lambda client: client.read_control("attempt-1"),
                "Backend rejected worker credential for "
                "/models-work/attempts/attempt-1/control",
            ),
            (
                lambda client: client.download_artifact(
                    "attempt-1", "artifact-1"
                ),
                "Backend rejected worker credential for artifact download",
            ),
        )
        for call, expected_message in calls:
            with self.subTest(
                expected_message=expected_message
            ), tempfile.TemporaryDirectory() as directory:
                rejection = urllib.error.HTTPError(
                    "http://localhost:3000/models-work",
                    401,
                    "Unauthorized",
                    {},
                    io.BytesIO(b'{"message":"unauthorized"}'),
                )
                with patch("urllib.request.urlopen", side_effect=rejection):
                    with self.assertRaisesRegex(
                        WorkerAuthenticationError,
                        expected_message,
                    ):
                        call(self._client(directory))

    def test_preserves_transport_errors_for_control_and_artifacts(self):
        calls = (
            (
                lambda client: client.read_control("attempt-1"),
                "Backend request failed for "
                "/models-work/attempts/attempt-1/control: <urlopen error offline>",
            ),
            (
                lambda client: client.download_artifact(
                    "attempt-1", "artifact-1"
                ),
                "Backend artifact download failed: <urlopen error offline>",
            ),
        )
        for call, expected_message in calls:
            with self.subTest(
                expected_message=expected_message
            ), tempfile.TemporaryDirectory() as directory, patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("offline"),
            ):
                with self.assertRaises(ProtocolTransportError) as raised:
                    call(self._client(directory))
                self.assertEqual(str(raised.exception), expected_message)

    def test_preserves_rejections_and_response_validation_for_get(self):
        with tempfile.TemporaryDirectory() as directory:
            rejection = urllib.error.HTTPError(
                "http://localhost:3000/models-work",
                409,
                "Conflict",
                {},
                io.BytesIO(b'{"message":"lease_expired"}'),
            )
            with patch("urllib.request.urlopen", side_effect=rejection):
                with self.assertRaises(ProtocolRejectionError) as raised:
                    self._client(directory).read_control("attempt-1")
            self.assertEqual(raised.exception.error_code, "lease_expired")

            with patch(
                "urllib.request.urlopen",
                return_value=_Response(b"null"),
            ):
                with self.assertRaisesRegex(
                    ProtocolTransportError,
                    "Backend returned an invalid response",
                ):
                    self._client(directory).read_control("attempt-1")

    def test_registration_unauthorized_remains_a_protocol_rejection(self):
        rejection = urllib.error.HTTPError(
            "http://localhost:3000/models-work/register",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"message":"invalid_enrollment_token"}'),
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"MODELS_ENROLLMENT_TOKEN": "invalid-token"},
            clear=False,
        ), patch(
            "lib.execution.protocol_client.worker_data_dir",
            return_value=directory,
        ), patch("urllib.request.urlopen", side_effect=rejection):
            client = ExecutionProtocolClient()
            with self.assertRaises(ProtocolRejectionError) as raised:
                client.ensure_registered([], [], 1, {})

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(
            raised.exception.error_code,
            "invalid_enrollment_token",
        )


if __name__ == "__main__":
    unittest.main()
