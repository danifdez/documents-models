from copy import deepcopy
import json
import unittest

from common.canonical_json import canonical_sha256
from lib.execution.engine_configuration import startup_configuration, generation_defaults, valid_startup_configuration
from lib.execution.inference_capture import engine_properties


class EngineConfigurationTest(unittest.TestCase):
    def command(self):
        return ['/PRIVATE/engine','--host','127.0.0.1','--port','18082','--ctx-size','8192','--parallel','2',
                '--threads','8','--n-gpu-layers','auto','--cache-type-k','q8_0','--cache-type-v','q8_0',
                '--jinja','--metrics','--model','/PRIVATE/model.gguf','--lora','/PRIVATE/adapter.gguf','--lora-init-without-apply']

    def test_actual_last_argument_values_win_without_exporting_paths(self):
        config=startup_configuration(self.command()+['--ctx-size','3072','--threads','2','--temp','0.17'],{})
        self.assertEqual(config['settings']['--ctx-size'],3072)
        self.assertEqual(config['settings']['--threads'],2)
        self.assertEqual(config['settings']['--temp'],0.17)
        self.assertEqual(config['settings']['loraCount'],1)
        self.assertEqual(config['settings']['--n-gpu-layers'],'auto')
        self.assertTrue(config['complete'])
        self.assertTrue(valid_startup_configuration(config))
        self.assertNotIn('PRIVATE',json.dumps(config))
        self.assertEqual(config['sha256'],canonical_sha256({k:v for k,v in config.items() if k!='sha256'}))

    def test_unknown_arguments_and_environment_are_only_marked_as_omitted(self):
        config=startup_configuration(self.command()+['--api-key','PRIVATE_SECRET','--chat-template-file','/PRIVATE/template'],{'LLAMA_ARG_MODEL':'PRIVATE_MODEL','CUDA_VISIBLE_DEVICES':'PRIVATE_DEVICE','OTEL_EXPORTER_OTLP_HEADERS':'PRIVATE_AUTH'})
        self.assertFalse(config['complete'])
        self.assertEqual(config['omissions'],['environment_overrides','unsupported_arguments'])
        self.assertNotIn('PRIVATE',json.dumps(config))
        self.assertNotIn('LLAMA_ARG_MODEL',json.dumps(config))

    def test_exporter_environment_is_not_an_engine_override(self):
        config=startup_configuration(self.command(),{'OTEL_EXPORTER_OTLP_HEADERS':'PRIVATE_AUTH'})
        self.assertTrue(config['complete'])
        self.assertNotIn('PRIVATE',repr(config))

    def test_malformed_or_sensitive_option_values_are_omitted(self):
        for flag,value in (('--threads','PRIVATE'),('--ctx-size','999999999999999999999'),('--cache-type-k','PRIVATE'),('--temp','nan'),('--temp','inf'),('--n-gpu-layers','PRIVATE')):
            config=startup_configuration(self.command()+[flag,value],{})
            self.assertIn('unsupported_arguments',config['omissions'])
            self.assertNotIn('PRIVATE',repr(config))
        self.assertFalse(startup_configuration(self.command()+['--threads'],{})['complete'])

    def test_private_record_projection_is_validated_before_export(self):
        original=startup_configuration(self.command(),{})
        for mode in ('hash','secret','type','unknown_omission'):
            value=deepcopy(original)
            if mode=='hash':value['sha256']='sha256:'+'0'*64
            elif mode=='secret':value['settings']['api_key']='PRIVATE_SECRET'
            elif mode=='type':value['complete']=1
            else:value['omissions']=['PRIVATE_REASON']
            with self.subTest(mode=mode):self.assertFalse(valid_startup_configuration(value))

    def test_declared_defaults_preserve_numeric_values_and_sampler_order(self):
        params={'seed':4294967295,'temperature':0.17,'top_k':40,'top_p':0.95,'ignore_eos':False,
                'samplers':['penalties','dry','top_k','top_p','min_p','temperature'],
                'generation_prompt':'','lora':[]}
        config=generation_defaults({'default_generation_settings':{'params':params}})
        self.assertEqual(config['settings'],params)
        self.assertEqual(config['source'],'server_props_metrics')
        self.assertFalse(config['complete'])
        self.assertEqual(config['omissions'],['unreported_generation_parameters'])
        self.assertEqual(config['sha256'],canonical_sha256({k:v for k,v in config.items() if k!='sha256'}))

    def test_default_projection_never_exports_prompts_paths_unknown_names_or_nonfinite_values(self):
        params={'temperature':float('nan'),'top_k':True,'samplers':['PRIVATE_SAMPLER'],
                'generation_prompt':'PRIVATE_PROMPT','grammar':'PRIVATE_GRAMMAR','stop':['PRIVATE_STOP'],
                'PRIVATE_KEY':'PRIVATE_SECRET','model':'/PRIVATE/model','frequency_penalty':10**500,
                'lora':[{'id':0,'scale':float('inf')}]}
        config=generation_defaults({'default_generation_settings':{'params':params}})
        self.assertEqual(config['settings'],{})
        self.assertIn('unexported_or_invalid_values',config['omissions'])
        self.assertNotIn('PRIVATE',json.dumps(config))

    def test_missing_or_oversized_defaults_remain_unavailable(self):
        for data in ({},[],{'default_generation_settings':{'params':{}}},{'default_generation_settings':{'params':{str(i):0 for i in range(129)}}}):
            self.assertIsNone(generation_defaults(data))

    def test_declared_defaults_snapshot_is_detached_from_server_response(self):
        params={'temperature':0.17,'samplers':['top_k','temperature']}
        config=generation_defaults({'default_generation_settings':{'params':params}})
        params['samplers'].append('PRIVATE_MUTATION');params['temperature']=0.8
        self.assertEqual(config['settings']['samplers'],['top_k','temperature'])
        self.assertEqual(config['settings']['temperature'],0.17)

    def test_existing_engine_properties_preserve_defaults_without_exporting_private_properties(self):
        data={'model_path':'/PRIVATE/model.gguf','chat_template':'PRIVATE_TEMPLATE','ui':{'secret':'PRIVATE'},
              'default_generation_settings':{'n_ctx':3072,'params':{'temperature':0.17,'generation_prompt':'PRIVATE_PROMPT'}}}
        result=engine_properties(data)
        self.assertEqual(result['properties']['contextSize'],3072)
        self.assertEqual(result['generationDefaults']['settings']['temperature'],0.17)
        self.assertNotIn('PRIVATE',json.dumps(result))

    def test_each_new_server_snapshot_preserves_the_current_declaration(self):
        first=engine_properties({'default_generation_settings':{'params':{'temperature':0.17}}})
        second=engine_properties({'default_generation_settings':{'params':{'temperature':0.8}}})
        self.assertNotEqual(first['generationDefaults']['sha256'],second['generationDefaults']['sha256'])

    def test_opposite_switches_preserve_the_last_effective_value(self):
        first=startup_configuration(self.command()+['--mmap','--no-mmap','--context-shift','--no-context-shift'],{})
        self.assertIs(first['settings']['--mmap'],False)
        self.assertIs(first['settings']['--context-shift'],False)
        self.assertNotIn('--no-mmap',first['settings'])
        second=startup_configuration(self.command()+['--no-mmap','--mmap'],{})
        self.assertIs(second['settings']['--mmap'],True)
        self.assertTrue(valid_startup_configuration(second))
