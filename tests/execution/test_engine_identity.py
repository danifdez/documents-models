"""Verification uses configured local bytes; runtime evidence contains no private paths."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from common.canonical_json import canonical_sha256
from common.execution_registry import TASK_HANDLERS
from lib.execution.inference_capture import engine_properties
from lib.execution.step_executor import execute_assignment
from lib.llm.config import active_deployments
from services import llama_server
from services.llm_service import get_llm_service, reset_cache
from tests.execution import test_inference_capture as fixtures

PROPS = {'build_info':'b10243-563dec81c','model_path':'/PRIVATE_PATH/served-model.gguf',
         'default_generation_settings':{'n_ctx':2048,'prompt':'PRIVATE_DEFAULT'},'total_slots':1,
         'model_ftype':17,'chat_template':'PRIVATE_TEMPLATE','ui':{'secret':'PRIVATE_UI'}}


class EngineIdentityTest(unittest.TestCase):
    setUp = fixtures.InferenceCaptureTest.setUp
    tearDown = fixtures.InferenceCaptureTest.tearDown
    assignment = fixtures.InferenceCaptureTest.assignment

    def client(self, checksum=None):
        return get_llm_service('requested.gguf',2048,1,64,0,
                               getattr(self,'adapter',None),0.75,checksum)

    def adapter_file(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.adapter=str(Path(directory.name)/'adapter.gguf')
        Path(self.adapter).write_bytes(b'controlled adapter bytes')
        return hashlib.sha256(Path(self.adapter).read_bytes()).hexdigest()

    def execute(self, client, fail=False):
        task='identity-test';TASK_HANDLERS[task]=lambda _payload: self.run_client(client,fail)
        self.addCleanup(TASK_HANDLERS.pop,task,None)
        uploads=[]
        assignment=self.assignment(task,grant=False)
        with patch('services.llm_service.llama_server.props',return_value=PROPS), \
             patch('lib.execution.step_executor.get_task_config',return_value={}), \
             patch('lib.execution.step_executor.logger'):
            result=execute_assignment(assignment,output_artifacts=uploads)
        return result,uploads

    @staticmethod
    def run_client(client,fail):
        text=client.chat([{'role':'user','content':'PRIVATE_INPUT'}],max_tokens=8)
        if fail:raise ValueError('PRIVATE_POSTPROCESSOR')
        return {'text':text}

    def test_runtime_artifact_is_bound_to_attempt_and_contains_only_projected_properties(self):
        result,uploads=self.execute(self.client())
        self.assertEqual(result['status'],'succeeded')
        self.assertEqual(len(uploads),1)
        artifact=uploads[0];self.assertEqual(artifact['kind'],'inference_runtime')
        raw=base64.b64decode(artifact['bodyBase64']);value=json.loads(raw)
        self.assertEqual('sha256:'+hashlib.sha256(raw).hexdigest(),artifact['contentHash'])
        self.assertEqual({key:value[key] for key in fixtures.IDS},fixtures.IDS)
        self.assertEqual(value['schemaVersion'],'inference-runtime/1')
        self.assertEqual(value['calls'][0]['engine']['properties'],{
            'modelLabel':'served-model.gguf','buildNumber':10243,'buildCommit':'563dec81c',
            'contextSize':2048,'slots':1,'modelFileType':17,
            'chatTemplateHash':canonical_sha256('PRIVATE_TEMPLATE')})
        self.assertNotIn('PRIVATE',raw.decode())
        self.assertNotIn('_identity',json.dumps(result))
        self.assertNotIn('engine',result['inference']['calls'][0])
        self.assertEqual(result['artifactRefs'][0]['artifactId'],artifact['artifactId'])
        self.assertNotIn('lora',json.loads(self.requests[0]))

    def test_actual_adapter_checksum_and_scale_reach_evidence_and_effective_identity(self):
        checksum=self.adapter_file()
        with patch.object(llama_server,'lora_adapters',return_value=[{'path':self.adapter,'id':7}]):
            result,uploads=self.execute(self.client(checksum))
        evidence=json.loads(base64.b64decode(uploads[0]['bodyBase64']))
        adapter=evidence['calls'][0]['adapter']
        self.assertEqual(adapter,{'id':7,'sha256':checksum,'scale':0.75,
            'verification':'local_file_sha256','binding':'server_reported_exact_path','expectedChecksumMatched':True})
        self.assertEqual(result['inference']['effectiveAdapter'],canonical_sha256({'available':True,'sha256':checksum,'scale':0.75}))
        self.assertEqual(json.loads(self.requests[0])['lora'],[{'id':7,'scale':0.75}])
        self.assertNotIn(self.adapter,json.dumps(result)+json.dumps(evidence))

    def test_incorrect_or_malformed_checksum_never_generates_or_claims_an_adapter(self):
        self.adapter_file()
        for checksum in ('0'*64,'PRIVATE_INVALID','',True):
            with self.subTest(checksum=type(checksum).__name__):
                reset_cache()
                with patch.object(llama_server,'lora_adapters',return_value=[{'path':self.adapter,'id':1}]):
                    if checksum is True:
                        with self.assertRaisesRegex(ValueError,'Invalid LoRA checksum'):self.client(checksum)
                    else:
                        result,uploads=self.execute(self.client(checksum))
                        self.assertEqual(result['status'],'failed')
                        self.assertNotIn('effectiveAdapter',result['inference'])
                        self.assertEqual(uploads,[])
        self.assertEqual(self.requests,[])

    def test_cached_client_rechecks_file_and_server_id_on_every_request(self):
        checksum=self.adapter_file();client=self.client(checksum)
        with patch.object(llama_server,'lora_adapters',return_value=[{'path':self.adapter,'id':1}]):
            first,_=self.execute(client)
        with patch.object(llama_server,'lora_adapters',return_value=[{'path':self.adapter,'id':9}]):
            second,_=self.execute(client)
        self.assertEqual(first['status'],second['status'])
        self.assertEqual([json.loads(raw)['lora'][0]['id'] for raw in self.requests],[1,9])
        Path(self.adapter).write_bytes(b'changed adapter bytes')
        third,uploads=self.execute(client)
        self.assertEqual(third['status'],'failed');self.assertEqual(uploads,[])
        self.assertEqual(len(self.requests),2)

    def test_actual_checksum_is_computed_even_without_a_declared_checksum(self):
        checksum=self.adapter_file()
        with patch.object(llama_server,'lora_adapters',return_value=[{'path':self.adapter,'id':0}]):
            result,uploads=self.execute(self.client())
        self.assertEqual(result['status'],'succeeded')
        adapter=json.loads(base64.b64decode(uploads[0]['bodyBase64']))['calls'][0]['adapter']
        self.assertEqual(adapter['sha256'],checksum)
        self.assertIs(adapter['expectedChecksumMatched'],False)

    def test_instance_change_after_http_does_not_export_verified_hashes(self):
        from services import managed_engine
        before={'status':'verified_local','instanceId':'fixture',
                'ggufSha256':'sha256:'+'a'*64,'executableSha256':'sha256:'+'b'*64}
        with patch.object(managed_engine,'inspect',side_effect=[before,{'status':'unknown','reason':'instance_not_bound'}]):
            result,uploads=self.execute(self.client())
        self.assertEqual(result['status'],'failed')
        evidence=json.loads(base64.b64decode(uploads[0]['bodyBase64']))
        local=evidence['calls'][0]['engine']['localInstance']
        self.assertEqual(local, {'status': 'unknown', 'reason': 'instance_not_bound'})
        self.assertEqual(len(self.requests),1)

    def test_ask_falls_back_only_when_chat_is_unavailable(self):
        from urllib.error import HTTPError
        from services import llm_service, managed_engine

        client = self.client()
        unavailable = HTTPError(client.url, 404, "Not Found", {}, None)
        identity = {"engine": None}
        with (
            patch.object(client, "_prepare_request", return_value=(identity, {})),
            patch.object(llm_service.urllib.request, "urlopen", side_effect=unavailable),
            patch.object(client, "generate", return_value="legacy answer") as generate,
        ):
            self.assertEqual(client.ask("Hello"), "legacy answer")
            generate.assert_called_once()

        with (
            patch.object(client, "chat", side_effect=managed_engine.InstanceChanged("Instance changed")),
            patch.object(client, "generate") as generate,
        ):
            with self.assertRaisesRegex(managed_engine.InstanceChanged, "Instance changed"):
                client.ask("Hello")
            generate.assert_not_called()

    def test_postprocessing_failure_preserves_the_observed_runtime(self):
        result,uploads=self.execute(self.client(),fail=True)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(uploads),1)
        self.assertEqual(uploads[0]['kind'],'inference_runtime')
        self.assertNotIn('PRIVATE',base64.b64decode(uploads[0]['bodyBase64']).decode())


class AdapterResolutionTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'adapter.gguf';self.path.write_bytes(b'controlled adapter')
        self.checksum=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def resolve(self, catalog):
        with patch.object(llama_server,'lora_adapters',return_value=catalog):
            return llama_server.verified_lora_adapter('http://127.0.0.1:1',str(self.path),'model',self.checksum)

    def test_exact_path_wins_over_a_different_file_with_the_same_name(self):
        self.assertEqual(self.resolve([{'path':'/other/adapter.gguf','id':1},{'path':str(self.path),'id':2}])['id'],2)

    def test_missing_ambiguous_or_invalid_ids_are_rejected(self):
        for catalog in ([],[{'path':'/other/adapter.gguf','id':1}],
                        [{'path':str(self.path),'id':1},{'path':str(self.path),'id':2}],
                        [{'path':str(self.path),'id':1},{'path':'/other/adapter.gguf','id':1}],
                        [{'path':str(self.path),'id':True}], [{'path':str(self.path),'id':-1}]):
            with self.subTest(catalog_size=len(catalog)),self.assertRaises(ValueError):self.resolve(catalog)

    def test_symlinks_directories_and_special_files_are_not_read(self):
        original=self.path.with_suffix('.real');self.path.rename(original)
        self.path.symlink_to(original)
        with self.assertRaises(ValueError):self.resolve([])
        self.path.unlink();self.path.mkdir()
        with self.assertRaises(ValueError):self.resolve([])
        self.path.rmdir()
        if hasattr(os,'mkfifo'):
            os.mkfifo(self.path)
            with self.assertRaises(ValueError):self.resolve([])

    def test_file_mutation_during_verification_is_rejected(self):
        original=llama_server.os.fstat;count=0
        def mutate(fd):
            nonlocal count
            count+=1
            if count==2:self.path.write_bytes(b'mutated adapter')
            return original(fd)
        with patch.object(llama_server.os,'fstat',side_effect=mutate),self.assertRaises(ValueError):self.resolve([])

    def test_untrusted_server_paths_are_only_compared_and_never_opened(self):
        foreign=Path(self.directory.name)/'unrelated';foreign.write_bytes(b'PRIVATE_FOREIGN_FILE')
        with patch('services.llama_server.os.open',wraps=os.open) as opened:
            with self.assertRaises(ValueError):self.resolve([{'path':str(foreign),'id':0}])
        self.assertEqual(opened.call_count,1)
        self.assertEqual(opened.call_args.args[0],str(self.path))


class PropertyProjectionTest(unittest.TestCase):
    def test_invalid_values_and_private_fields_are_omitted(self):
        self.assertIsNone(engine_properties({'model_path':'https://PRIVATE/key','build_info':'PRIVATE BUILD',
            'default_generation_settings':{'n_ctx':True},'total_slots':-1,'model_ftype':[],
            'chat_template':'x'*(256*1024+1),'chat_template_tool_use':'\ud800','api_key':'PRIVATE'}))
        self.assertIsNone(engine_properties([]))


class DeploymentRegistryTest(unittest.TestCase):
    def test_invalid_existing_registry_never_silently_selects_the_base_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'deployments.json'
            with patch.dict('os.environ',{'MODELS_DEPLOYMENTS_PATH':str(path)}):
                self.assertEqual(active_deployments(),{})
                cases=['PRIVATE_INVALID_JSON',json.dumps({'tasks':[]}),
                       json.dumps({'tasks':{'task':[]}}),
                       json.dumps({'tasks':{'task':{'path':''}}}),
                       json.dumps({'tasks':{'task':{'path':'adapter.gguf','scale':'PRIVATE'}}}),
                       json.dumps({'tasks':{'task':{'path':'adapter.gguf','scale':float('nan')}}})]
                for value in cases:
                    path.write_text(value)
                    with self.subTest(case=cases.index(value)),self.assertRaises(ValueError) as caught:
                        active_deployments()
                    self.assertNotIn('PRIVATE',str(caught.exception))

    def test_metadata_can_be_produced_after_a_registry_failure(self):
        from lib.execution.step_executor import _inference_metadata
        with patch('lib.execution.step_executor.get_task_config',side_effect=ValueError('registry invalid')):
            metadata=_inference_metadata('any-task')
        self.assertEqual(metadata['inference']['requestedModel'],'unreported')
        self.assertNotIn('effectiveAdapter',metadata['inference'])


class EngineMetadataTransportTest(unittest.TestCase):
    def test_properties_and_adapter_queries_reject_large_or_malformed_responses(self):
        for raw in (b'x'*(1024*1024+1),b'PRIVATE_INVALID_JSON',b'\xff',b'null',b'"PRIVATE_STRING"'):
            response=MagicMock();response.__enter__.return_value.read.return_value=raw
            with self.subTest(size=len(raw)),patch.object(llama_server.urllib.request,'urlopen',return_value=response):
                self.assertEqual(llama_server.props('http://127.0.0.1:1'),{})
                self.assertEqual(llama_server.lora_adapters('http://127.0.0.1:1'),[])

    def test_managed_engine_uses_absolute_adapter_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.gguf';path.write_bytes(b'adapter')
            relative=os.path.relpath(path)
            with patch.object(llama_server,'active_deployments',return_value={'a':{'path':relative},'b':{'path':str(path)}}):
                self.assertEqual(llama_server._deployed_adapters(),[str(path)])
