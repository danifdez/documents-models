import base64
import json
import unittest
from unittest.mock import patch

from common.canonical_json import canonical_bytes, canonical_sha256
from lib.execution import inference_capture as capture
from lib.execution.step_executor import execute_assignment
from tests.execution import test_inference_capture as fixtures
from tests.execution import test_telemetry as telemetry_tests

PRIVATE = 'PRIVATE_TEMPLATE_Ω {{ messages[0].content }}'
PROPS = {'chat_template': PRIVATE, 'chat_template_tool_use': PRIVATE, 'model_path': '/PRIVATE/model.gguf'}


class TemplateCaptureTests(unittest.TestCase):
    setUp = fixtures.InferenceCaptureTest.setUp
    tearDown = fixtures.InferenceCaptureTest.tearDown
    assignment = fixtures.InferenceCaptureTest.assignment

    def execute(self, granted=True):
        uploads=[]
        with patch('services.llm_service.llama_server.props',return_value=PROPS),patch('lib.execution.step_executor.logger'):
            result=execute_assignment(self.assignment(grant=granted),output_artifacts=uploads)
        runtime=next(item for item in uploads if item['kind']=='inference_runtime')
        return result,json.loads(base64.b64decode(runtime['bodyBase64'])),uploads

    def test_authorized_capture_binds_template_text_to_existing_hash_and_attempt(self):
        result,runtime,uploads=self.execute()
        self.assertEqual(result['status'],'succeeded')
        self.assertEqual(runtime['templates'],{canonical_sha256(PRIVATE):PRIVATE})
        self.assertEqual({key:runtime[key] for key in fixtures.IDS},fixtures.IDS)
        properties=runtime['calls'][0]['engine']['properties']
        self.assertEqual(properties['chatTemplateHash'],canonical_sha256(PRIVATE))
        self.assertEqual(properties['toolTemplateHash'],canonical_sha256(PRIVATE))
        self.assertEqual(sorted(item['kind'] for item in uploads),['inference_exchange','inference_runtime'])
        self.assertNotIn(PRIVATE,json.dumps(result,ensure_ascii=False))

    def test_no_grant_keeps_only_hashes_and_never_serializes_template_text(self):
        result,runtime,uploads=self.execute(granted=False)
        self.assertEqual(result['status'],'succeeded')
        self.assertNotIn('templates',runtime)
        self.assertEqual([item['kind'] for item in uploads],['inference_runtime'])
        self.assertNotIn('PRIVATE_TEMPLATE',json.dumps(runtime)+json.dumps(result))
        self.assertEqual(runtime['calls'][0]['engine']['properties']['chatTemplateHash'],canonical_sha256(PRIVATE))

    def test_capture_context_does_not_carry_text_into_the_next_ungranted_attempt(self):
        self.execute()
        _,runtime,_=self.execute(granted=False)
        self.assertNotIn('templates',runtime)
        self.assertIsNone(capture.current_capture())

    def test_template_content_never_reaches_logs_or_otel_even_with_capture_enabled(self):
        with patch.dict('os.environ',{'OTEL_SDK_DISABLED':'false'}):
            boundary=telemetry_tests.TelemetryBoundaryTest();boundary.setUp()
            try:
                with self.assertLogs('services.llm_service',level='INFO') as logs:
                    result,runtime,_=self.execute()
                self.assertIn(canonical_sha256(PRIVATE),runtime['templates'])
                spans=boundary.exporter.get_finished_spans()
                self.assertTrue(spans)
                self.assertNotIn('PRIVATE_TEMPLATE',''.join(item.to_json() for item in spans))
                self.assertNotIn('PRIVATE_TEMPLATE',repr(logs.output)+json.dumps(result))
            finally:boundary.tearDown()


class TemplateBudgetTests(unittest.TestCase):
    def test_same_template_is_stored_once_for_both_variants_and_repeated_calls(self):
        state=capture.Capture(fixtures.IDS)
        engine=capture.engine_properties(PROPS)
        token=capture._CAPTURE.set(state)
        try:
            for _ in range(4):
                state.record_templates(PROPS,engine)
                with capture.observe_inference({'messages':[]},{'engine':engine,'adapter':None}):pass
        finally:capture._CAPTURE.reset(token)
        runtime=json.loads(next(item.body for item in state.artifacts() if item.kind=='inference_runtime'))
        self.assertEqual(len(runtime['calls']),4)
        self.assertEqual(runtime['templates'],{canonical_sha256(PRIVATE):PRIVATE})
        self.assertNotIn('templateCaptureOmission',runtime)

    def test_disabled_context_cannot_capture_text_even_if_called_directly(self):
        state=capture.Capture(fixtures.IDS,enabled=False)
        state.record_templates(PROPS,capture.engine_properties(PROPS))
        self.assertEqual(state.templates,{})

    def test_table_budget_omits_whole_templates_without_truncating_or_changing_hashes(self):
        props={'chat_template':'A'*140000,'chat_template_tool_use':'B'*140000}
        engine=capture.engine_properties(props);state=capture.Capture(fixtures.IDS)
        state.record_templates(props,engine)
        self.assertLessEqual(len(canonical_bytes(state.templates)),capture.MAX_TEMPLATE_BYTES)
        self.assertTrue(state.templates_omitted)
        self.assertEqual(state.templates,{canonical_sha256(props['chat_template']):props['chat_template']})

    def test_invalid_unicode_or_oversized_content_is_not_exported(self):
        for template in ('X'*(capture.MAX_TEMPLATE_BYTES+1),'\ud800','Ω'*200000):
            props={'chat_template':template};state=capture.Capture(fixtures.IDS)
            state.record_templates(props,capture.engine_properties(props))
            self.assertEqual(state.templates,{})
            self.assertTrue(state.templates_omitted)

    def test_missing_templates_do_not_create_content_or_omission(self):
        state=capture.Capture(fixtures.IDS)
        state.record_templates({'chat_template':'','chat_template_tool_use':None},{})
        self.assertEqual(state.templates,{})
        self.assertFalse(state.templates_omitted)

    def test_only_templates_referenced_by_recorded_calls_are_serialized(self):
        state=capture.Capture(fixtures.IDS)
        engine=capture.engine_properties(PROPS)
        state.record_templates(PROPS,engine)
        orphan={'chat_template':'PRIVATE_UNOBSERVED_TEMPLATE'}
        state.record_templates(orphan,capture.engine_properties(orphan))
        token=capture._CAPTURE.set(state)
        try:
            with capture.observe_inference({'messages':[]},{'engine':engine,'adapter':None}):pass
        finally:capture._CAPTURE.reset(token)
        runtime=json.loads(next(item.body for item in state.artifacts() if item.kind=='inference_runtime'))
        self.assertEqual(runtime['templates'],{canonical_sha256(PRIVATE):PRIVATE})
        self.assertNotIn('PRIVATE_UNOBSERVED_TEMPLATE',json.dumps(runtime))
