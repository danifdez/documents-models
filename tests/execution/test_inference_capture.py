import base64
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import threading
import unittest
from unittest.mock import patch

from common.canonical_json import canonical_sha256
from lib.execution.inference_capture import current_capture
from lib.execution.step_executor import execute_assignment
from services.llm_service import reset_cache

IDS = {key: f'{index}1234567-89ab-4def-8123-456789abcdef' for index, key in enumerate(
    ['executionId', 'stepId', 'operationId', 'attemptId'])}
GRANT = {'schemaVersion': 'inference-capture/1', 'purpose': 'evaluation', 'authorizedBy': '7'}


class InferenceCaptureTest(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        self.http_status = 200
        self.override = None
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                outer.requests.append(data)
                payload = json.loads(data)
                section = 'summary' in payload.get('response_format', {}).get('schema', {}).get('properties', {})
                content = '{ "summary": "The  private library opens Monday." }' if section else '{ "paragraphs": ["The private library opens Monday."] }'
                body = outer.override if outer.override is not None else json.dumps({
                    'id': 'engine-reply', 'model': 'served-model',
                    'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20},
                }, indent=2).encode()
                outer.responses.append(body)
                self.send_response(outer.http_status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Set-Cookie', 'PRIVATE_HEADER')
                if outer.http_status == 302:
                    self.send_header('Location', f'http://127.0.0.1:{outer.server.server_port}/redirect-target')
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict('os.environ', {'OTEL_SDK_DISABLED': 'true'}))
        self.stack.enter_context(patch('services.llm_service.llama_server.ensure_server', return_value=f'http://127.0.0.1:{self.server.server_port}'))
        self.stack.enter_context(patch('services.llm_service.llama_server.props', return_value={}))
        reset_cache()

    def tearDown(self):
        reset_cache()
        self.stack.close()
        self.server.shutdown()
        self.server.server_close()

    def assignment(self, task='summarize-map', grant=True):
        payload = {'content': 'PRIVATE_SOURCE: The library opens Monday.', 'targetLanguage': 'en'}
        if task == 'summarize-compose': payload = {'partials': ['The library opens Monday.'], 'targetLanguage': 'en'}
        work = {'taskType': task, 'payload': payload}
        if grant: work['inferenceCapture'] = dict(GRANT)
        return {**IDS, 'stepKind': 'inference', 'work': work}

    def run_step(self, assignment=None):
        uploads = []
        with patch('lib.execution.step_executor.logger'):
            result = execute_assignment(assignment or self.assignment(), output_artifacts=uploads)
        return result, uploads

    def exchange(self, uploads):
        self.assertEqual(len(uploads), 1)
        artifact = uploads[0]
        raw = base64.b64decode(artifact['bodyBase64'])
        self.assertEqual(artifact['contentHash'], canonical_sha256(json.loads(raw)))
        self.assertEqual(len(raw), artifact['size'])
        self.assertNotIn(b'PRIVATE_HEADER', raw)
        return json.loads(raw)

    def test_map_captures_exact_http_bytes_before_parsing_the_summary(self):
        result, uploads = self.run_step()
        value = self.exchange(uploads)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(value['status'], 'completed')
        self.assertEqual({key: value[key] for key in IDS}, IDS)
        self.assertEqual(base64.b64decode(value['request']['bodyBase64']), self.requests[0])
        self.assertEqual(base64.b64decode(value['response']['bodyBase64']), self.responses[0])
        request = json.loads(self.requests[0])
        self.assertIn('messages', request)
        self.assertIn('response_format', request)
        self.assertEqual(request['temperature'], 0.0)
        self.assertIn('seed', request)
        self.assertEqual(result['output']['outcome']['value']['summary'], 'The private library opens Monday.')
        self.assertEqual(result['artifactRefs'][0]['artifactId'], uploads[0]['artifactId'])
        self.assertIsNone(current_capture())

    def test_compose_uses_the_same_capture_path(self):
        result, uploads = self.run_step(self.assignment('summarize-compose'))
        value = self.exchange(uploads)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(value['callIndex'], 1)
        self.assertEqual(base64.b64decode(value['response']['bodyBase64']), self.responses[0])
        self.assertEqual(result['output']['outcome']['value']['response'], 'The private library opens Monday.')

    def test_default_is_no_capture_even_when_payload_or_environment_requests_it(self):
        assignment = self.assignment(grant=False)
        assignment['work']['payload']['inferenceCapture'] = GRANT
        with patch.dict('os.environ', {'CAPTURE_INFERENCE': 'true'}):
            result, uploads = self.run_step(assignment)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(uploads, [])
        self.assertEqual(result['artifactRefs'], [])

    def test_invalid_grant_never_sends_a_request(self):
        for change in [{'purpose': 'execution'}, {'authorizedBy': 'standalone'}, {'authorizedBy': 'system'}, {'schemaVersion': 'unknown'}]:
            assignment = self.assignment()
            assignment['work']['inferenceCapture'].update(change)
            result, uploads = self.run_step(assignment)
            self.assertEqual(result['error']['code'], 'INFERENCE_CAPTURE_DENIED')
            self.assertEqual(result['output']['outcome']['kind'], 'failed')
            self.assertEqual(uploads, [])
        assignment = self.assignment()
        assignment['stepKind'] = 'service'
        result, _ = self.run_step(assignment)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.requests, [])

    def test_missing_artifact_delivery_is_rejected_before_execution(self):
        result = execute_assignment(self.assignment())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.requests, [])

    def test_parser_failure_preserves_the_original_engine_response(self):
        self.override = json.dumps({'choices': [{'message': {'content': 'PRIVATE_INVALID_SUMMARY'}}]}).encode()
        result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        value = self.exchange(uploads)
        self.assertEqual(value['status'], 'completed')
        self.assertEqual(base64.b64decode(value['response']['bodyBase64']), self.override)
        self.assertNotIn('PRIVATE_INVALID_SUMMARY', result['error']['message'])

    def test_http_error_is_captured_without_body_or_headers_in_error(self):
        self.http_status = 500
        self.override = b'{"error":"PRIVATE_ENGINE_ERROR"}'
        result, uploads = self.run_step()
        value = self.exchange(uploads)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(value['status'], 'http_error')
        self.assertEqual(value['response']['status'], 500)
        self.assertEqual(base64.b64decode(value['response']['bodyBase64']), self.override)
        self.assertNotIn('PRIVATE', result['error']['message'])

    def test_invalid_json_remains_in_the_artifact_only(self):
        self.override = b'PRIVATE_INVALID_JSON'
        result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.exchange(uploads)['status'], 'invalid_response')
        self.assertNotIn('PRIVATE', result['error']['message'])

    def test_response_limit_is_explicit_and_does_not_publish_partial_bytes(self):
        self.override = b'x' * 257
        with patch('lib.execution.inference_capture.MAX_BODY_BYTES', 4096), patch('lib.execution.inference_capture.MAX_TOTAL_BYTES', 4096):
            # Request fits; the remaining capture budget is smaller than this reply.
            from lib.execution.inference_capture import Exchange
            original = Exchange.read
            def bounded(exchange, response, status):
                exchange.capture.size = 4096 - 128
                return original(exchange, response, status)
            with patch.object(Exchange, 'read', bounded): result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        value = self.exchange(uploads)
        self.assertEqual(value['status'], 'response_too_large')
        self.assertIsNone(value['response']['bodyBase64'])
        self.assertEqual(value['response']['omissionReason'], 'size_limit')

    def test_request_limit_does_not_send_or_log_content(self):
        with patch('lib.execution.inference_capture.MAX_BODY_BYTES', 16): result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.requests, [])
        self.assertEqual(uploads, [])
        self.assertNotIn('PRIVATE_SOURCE', result['error']['message'])

    def test_transport_failure_preserves_request_only(self):
        import urllib.error
        with patch('services.llm_service._capture_opener.open', side_effect=urllib.error.URLError('PRIVATE_TRANSPORT')):
            result, uploads = self.run_step()
        value = self.exchange(uploads)
        self.assertEqual(value['status'], 'transport_error')
        self.assertIsNone(value['response'])
        self.assertNotIn('PRIVATE', result['error']['message'])
        self.assertIsNone(current_capture())

    def test_capture_does_not_follow_redirects(self):
        self.http_status = 302
        result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.exchange(uploads)['response']['status'], 302)

    def test_existing_outbox_stores_private_artifacts_then_removes_them_after_ack(self):
        import stat, tempfile
        from pathlib import Path
        from unittest.mock import MagicMock
        from lib.execution.result_outbox import ResultOutbox
        result, uploads = self.run_step()
        with tempfile.TemporaryDirectory() as directory:
            outbox = ResultOutbox(directory)
            outbox.store(result, uploads)
            pending = next(outbox.directory.glob('*.json'))
            self.assertEqual(stat.S_IMODE(pending.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(outbox.directory.stat().st_mode), 0o700)
            client = MagicMock()
            client.upload_artifact.return_value = {'code': 'received'}
            client.submit_result.return_value = {'code': 'received'}
            outbox.deliver_all(client)
            client.upload_artifact.assert_called_once_with(IDS['attemptId'], uploads[0])
            client.submit_result.assert_called_once_with(result)
            self.assertEqual(list(outbox.directory.iterdir()), [])

    def test_telemetry_contains_no_captured_content(self):
        from lib import telemetry
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        exporter = InMemorySpanExporter()
        with patch.dict('os.environ', {'OTEL_SDK_DISABLED': 'false'}):
            provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        try:
            with patch.object(telemetry, '_provider', provider), patch.object(telemetry, '_initialized', True):
                result, uploads = self.run_step()
            self.assertEqual(result['status'], 'succeeded')
            self.assertTrue(uploads)
            spans = exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            self.assertNotIn('PRIVATE_SOURCE', spans[0].to_json())
            self.assertNotIn('PRIVATE_HEADER', spans[0].to_json())
        finally:
            provider.shutdown()

    def test_protocol_exception_does_not_leak_engine_diagnostics(self):
        with patch('services.llm_service._capture_opener.open', side_effect=ValueError('PRIVATE_ENGINE_DIAGNOSTIC')):
            result, uploads = self.run_step()
        self.assertEqual(self.exchange(uploads)['status'], 'transport_error')
        self.assertNotIn('PRIVATE', result['error']['message'])

    def test_capture_ignores_environment_proxies(self):
        with patch.dict('os.environ', {'http_proxy': 'http://127.0.0.1:1', 'HTTP_PROXY': 'http://127.0.0.1:1', 'no_proxy': '', 'NO_PROXY': ''}):
            result, uploads = self.run_step()
        self.assertEqual(result['status'], 'succeeded')
        self.assertTrue(uploads)

    def test_each_call_has_its_own_revision_and_the_call_limit_is_enforced(self):
        from services.llm_service import _post
        def calls(*_):
            for _ in range(5):
                _post(f'http://127.0.0.1:{self.server.server_port}/completion', {'prompt': 'PRIVATE_SOURCE'})
            return {'summary': 'Unreachable.'}
        with patch('lib.execution.step_executor.call_handler', side_effect=calls):
            result, uploads = self.run_step()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.requests), 4)
        self.assertEqual([json.loads(base64.b64decode(item['bodyBase64']))['callIndex'] for item in uploads], [1, 2, 3, 4])
        self.assertEqual([ref['revision'] for ref in result['artifactRefs']], [1, 2, 3, 4])

    def test_capture_uses_authorization_and_step_kind_for_unknown_task_names(self):
        from common.execution_registry import TASK_HANDLERS
        from services.llm_service import _post
        for task in ['ask', 'translate-map', 'future-inference']:
            with self.subTest(task=task):
                def handler(_payload):
                    _post(f'http://127.0.0.1:{self.server.server_port}/completion', {'prompt': 'Controlled inference'})
                    return {'response': 'Controlled result'}
                with patch.dict(TASK_HANDLERS, {task: handler}), patch('lib.execution.step_executor.ensure_task_handler', return_value=True):
                    result, uploads = self.run_step(self.assignment(task))
                self.assertEqual(result['status'], 'succeeded')
                self.assertEqual(self.exchange(uploads)['status'], 'completed')
                self.assertIsNone(current_capture())
