"""Engine metadata stays useful without granting access to engine content."""
import io
import json
import unittest
from unittest.mock import patch

from tests.execution import test_inference_capture as fixtures
from lib import telemetry
from lib.execution.inference_capture import (
    capture_assignment, observe_inference, response_metadata,
)
from lib.execution.step_result import step_result_base
from services.llm_service import LLMService


class InferenceMetadataTest(unittest.TestCase):
    setUp = fixtures.InferenceCaptureTest.setUp
    tearDown = fixtures.InferenceCaptureTest.tearDown
    assignment = fixtures.InferenceCaptureTest.assignment
    run_step = fixtures.InferenceCaptureTest.run_step

    def test_real_response_fields_without_content_capture(self):
        result, uploads = self.run_step(self.assignment(grant=False))
        response = json.loads(self.responses[0])
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['usage'], {'promptTokens': 12, 'completionTokens': 8, 'totalTokens': 20})
        self.assertEqual(result['inference']['effectiveModel'], response['model'])
        self.assertEqual(result['inference']['finishReason'], 'stop')
        self.assertGreaterEqual(result['inference']['inferenceMs'], 0)
        self.assertNotIn('private', json.dumps(result['inference']))
        self.assertEqual(uploads, [])
        self.assertEqual(result['artifactRefs'], [])

    def test_engine_finish_is_independent_of_handler_failure(self):
        self.override = json.dumps({'model': 'engine', 'choices': [
            {'message': {'content': 'PRIVATE_INVALID'}, 'finish_reason': 'length'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}}).encode()
        result, uploads = self.run_step(self.assignment(grant=False))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['inference']['finishReason'], 'length')
        self.assertEqual(result['usage']['totalTokens'], 12)
        self.assertEqual(uploads, [])

    def test_failed_transport_keeps_unknown_usage_and_model(self):
        self.http_status = 500
        result, _ = self.run_step(self.assignment(grant=False))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['inference']['effectiveModel'], 'unreported')
        self.assertEqual(result['inference']['finishReason'], 'unknown')
        self.assertIsNone(result['usage']['totalTokens'])
        self.assertEqual(result['inference']['calls'][0]['status'], 'incomplete')


class MetadataAggregationTest(unittest.TestCase):
    def run_observations(self, responses):
        @capture_assignment
        def execute(assignment, artifacts=None, output_artifacts=None):
            for response in responses:
                with observe_inference({'model': '/private/requested.gguf'}) as observation:
                    telemetry.inference_response(None, response, observation)
            return {**step_result_base(assignment), 'status': 'succeeded'}
        return execute({**fixtures.IDS, 'stepKind': 'inference'})

    def test_native_completion_fields_and_unknown_are_distinct_from_zero(self):
        metadata = response_metadata({'model': '/private/models/served.gguf',
            'tokens_evaluated': 0, 'tokens_predicted': 3, 'tokens_cached': 0,
            'stop_type': 'limit', 'timings': {'prompt_ms': 1.5, 'predicted_ms': 2.5}})
        self.assertEqual(metadata, {'effectiveModel': 'served.gguf', 'finishReason': 'length',
            'cacheOutcome': 'miss', 'engineMs': 4,
            'usage': {'promptTokens': 0, 'completionTokens': 3, 'totalTokens': 3}})
        self.assertIsNone(response_metadata({})['usage']['promptTokens'])
        self.assertEqual(response_metadata({'tokens_cached': 5})['cacheOutcome'], 'hit')

    def test_untrusted_shapes_cannot_publish_private_text_or_invalid_counts(self):
        for reason in ('PRIVATE_BODY', {}, [], True):
            metadata = response_metadata({'model': 'https://secret.example/key',
                'choices': [{'finish_reason': reason}], 'stop_type': {},
                'usage': {'prompt_tokens': True, 'completion_tokens': -1, 'total_tokens': 2.5},
                'timings': {'prompt_ms': float('nan'), 'predicted_ms': 2}, 'content': 'PRIVATE_BODY'})
            self.assertEqual(metadata, response_metadata({}))

    def test_multiple_calls_sum_known_usage_without_claiming_one_model(self):
        result = self.run_observations([
            {'model': 'first', 'usage': {'prompt_tokens': 2, 'completion_tokens': 3}, 'stop_type': 'eos'},
            {'model': 'second', 'usage': {'prompt_tokens': 4, 'completion_tokens': 5}, 'stop_type': 'limit'}])
        self.assertEqual(result['usage'], {'promptTokens': 6, 'completionTokens': 8, 'totalTokens': 14})
        self.assertEqual(result['inference']['effectiveModel'], 'unreported')
        self.assertEqual(result['inference']['finishReason'], 'unknown')
        self.assertEqual([call['effectiveModel'] for call in result['inference']['calls']], ['first', 'second'])
        self.assertEqual(result['inference']['requestedModel'], 'requested.gguf')
        missing = self.run_observations([{'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}, {}])
        self.assertIsNone(missing['usage']['promptTokens'])
        self.assertEqual(missing['inference']['effectiveModel'], 'unreported')

    def test_metadata_limit_does_not_interrupt_generation_or_publish_partial_totals(self):
        result = self.run_observations([{'model': 'engine', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}] * 129)
        self.assertEqual(len(result['inference']['calls']), 128)
        self.assertIn('inference_metadata_truncated', result['inference']['warnings'])
        self.assertIsNone(result['usage']['totalTokens'])
        self.assertEqual(result['inference']['effectiveModel'], 'unreported')
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(self.run_observations([])['inference']['calls'], [])

    def test_stream_usage_is_cumulative_and_has_no_content(self):
        chunks = [
            {'model': '/private/served.gguf', 'choices': [{'delta': {'content': 'PRIVATE_TEXT'}}],
             'usage': {'prompt_tokens': 2, 'completion_tokens': 1}},
            {'choices': [{'finish_reason': 'stop', 'delta': {}}]},
            {'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}},
        ]
        response = io.BytesIO(b''.join(b'data: ' + json.dumps(chunk).encode() + b'\n\n' for chunk in chunks) + b'data: [DONE]\n')
        client = object.__new__(LLMService)
        client.url = 'http://127.0.0.1:18082'
        @capture_assignment
        def execute(assignment, artifacts=None, output_artifacts=None):
            with patch.object(client, '_model_field', return_value={'model': 'requested'}), \
                 patch.object(client, '_sampling_kwargs', return_value={}), \
                 patch.object(client, '_prepare_request', return_value={'engine': None, 'adapter': None}), \
                 patch('services.llm_service._post', return_value=response) as post, \
                 patch.dict('os.environ', {'OTEL_SDK_DISABLED': 'true'}):
                self.assertEqual(list(client.chat_stream([])), ['PRIVATE_TEXT'])
                self.assertEqual(post.call_args.args[1]['stream_options'], {'include_usage': True})
            return {**step_result_base(assignment), 'status': 'succeeded'}
        result = execute({**fixtures.IDS, 'stepKind': 'inference'})
        self.assertEqual(result['usage'], {'promptTokens': 2, 'completionTokens': 3, 'totalTokens': 5})
        self.assertEqual(result['inference']['effectiveModel'], 'served.gguf')
        self.assertEqual(result['inference']['finishReason'], 'stop')
        self.assertEqual(len(result['inference']['calls']), 1)
        self.assertNotIn('PRIVATE', json.dumps(result))
