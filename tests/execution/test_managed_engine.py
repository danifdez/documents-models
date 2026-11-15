"""Kernel binding tests use a local fixture listener and mapped regular file."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mmap
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from services import managed_engine


@unittest.skipUnless(sys.platform.startswith('linux'), 'Local process binding currently uses Linux procfs')
class ManagedEngineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='PRIVATE_MANAGED_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model.gguf'
        self.model.write_bytes(b'G' * 4096)
        handle = self.model.open('rb'); self.addCleanup(handle.close)
        self.mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.addCleanup(self.mapping.close)
        listener = HTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
        self.addCleanup(listener.server_close)
        self.url = f'http://127.0.0.1:{listener.server_port}'
        self.path = self.root / 'private' / 'record.json'
        self.addCleanup(patch.stopall)
        patch.object(managed_engine, '_record_path', return_value=self.path).start()
        self.startup = managed_engine.prepare(self.url, sys.executable, {'model_path': str(self.model)}, [sys.executable, *sys.argv])
        managed_engine.bind(self.startup, os.getpid())

    def inspect(self):
        return managed_engine.inspect(self.url, {'model_path': str(self.model)})

    def test_private_startup_record_binds_actual_listener_executable_and_mapping(self):
        public = self.inspect()
        self.assertEqual(public['status'], 'verified_local')
        self.assertEqual(public['instanceId'], self.startup['instanceId'])
        self.assertEqual(public['ggufSha256'], 'sha256:' + self.startup['files']['gguf']['sha256'])
        self.assertEqual(public['executableSha256'], 'sha256:' + self.startup['files']['executable']['sha256'])
        self.assertNotIn('PRIVATE', json.dumps(public))
        self.assertNotIn('pid', public)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_changed_model_is_rejected_even_if_original_mtime_is_restored(self):
        timestamp = self.model.stat().st_mtime_ns
        self.model.write_bytes(b'C' * 4096)
        os.utime(self.model, ns=(timestamp, timestamp))
        with self.assertRaises(managed_engine.InstanceChanged): self.inspect()

    def test_atomic_model_replacement_is_rejected_without_rehashing_new_bytes(self):
        replacement = self.root / 'replacement.gguf'; replacement.write_bytes(b'C' * 4096)
        replacement.replace(self.model)
        with patch.object(managed_engine, 'fingerprint_file') as hashed:
            with self.assertRaises(managed_engine.InstanceChanged): self.inspect()
        hashed.assert_not_called()

    def test_deleted_model_is_rejected(self):
        self.model.unlink()
        with self.assertRaises(managed_engine.InstanceChanged): self.inspect()

    def test_binary_alias_retargeting_requires_a_new_startup(self):
        alias = self.root / 'engine'; alias.symlink_to(sys.executable)
        record = managed_engine.prepare(self.url, str(alias), {'model_path': str(self.model)}, [sys.executable, *sys.argv])
        managed_engine.bind(record, os.getpid())
        self.assertEqual(self.inspect()['status'], 'verified_local')
        other = self.root / 'other-engine'; shutil.copyfile(sys.executable, other)
        alias.unlink(); alias.symlink_to(other)
        with self.assertRaises(managed_engine.InstanceChanged): self.inspect()
        alias.unlink(); alias.symlink_to(sys.executable)
        with self.assertRaises(managed_engine.InstanceChanged): self.inspect()

    def test_process_birth_and_boot_changes_never_reuse_old_hashes(self):
        for field, value in (('startTicks', json.loads(self.path.read_text())['process']['startTicks'] + 1), ('bootId', '00000000-0000-0000-0000-000000000000')):
            record = json.loads(self.path.read_text())
            original = deepcopy(record)
            record['process'][field] = value
            self.path.write_text(json.dumps(record))
            public = self.inspect()
            self.assertEqual(public['status'], 'unknown')
            self.assertNotIn('ggufSha256', public)
            self.path.write_text(json.dumps(original))

    def test_missing_listener_or_mapping_cannot_be_marked_verified(self):
        for helper in ('_owns_listener', '_maps_gguf'):
            with patch.object(managed_engine, helper, return_value=False):
                self.assertEqual(self.inspect()['status'], 'unknown')

    def test_wrong_executable_cannot_be_marked_verified(self):
        record = json.loads(self.path.read_text()); record['files']['executable']['signature'][1] = 0
        self.path.write_text(json.dumps(record))
        self.assertEqual(self.inspect()['status'], 'unknown')

    def test_permissions_symlinks_and_malformed_records_are_not_trusted(self):
        original = self.path.read_bytes()
        self.path.chmod(0o644)
        self.assertEqual(self.inspect()['status'], 'unknown')
        self.path.chmod(0o600); self.path.write_bytes(b'PRIVATE_INVALID_JSON')
        self.assertEqual(self.inspect()['status'], 'unknown')
        self.path.write_bytes(original)
        target = self.path.with_suffix('.other'); self.path.rename(target); self.path.symlink_to(target)
        self.assertEqual(self.inspect()['status'], 'unknown')

    def test_malformed_record_fields_are_unknown_without_opening_untrusted_paths(self):
        original=json.loads(self.path.read_text())
        for value in (None,True,123,[],{}):
            record=deepcopy(original);record['instanceId']=value
            self.path.write_text(json.dumps(record))
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(self.inspect()['status'],'unknown')

    def test_live_unregistered_server_is_not_adopted(self):
        self.path.unlink()
        self.assertEqual(self.inspect(), {'status': 'unknown', 'reason': 'no_managed_startup'})

    def test_server_reported_paths_are_never_used_to_read_files(self):
        with patch.object(managed_engine, 'file_signature', wraps=managed_engine.file_signature) as files:
            public = managed_engine.inspect(self.url, {'model_path': '/PRIVATE_SERVER_PATH/secret'})
        self.assertEqual(public['status'], 'unknown')
        self.assertNotIn('/PRIVATE_SERVER_PATH/secret', [call.args[0] for call in files.call_args_list])

    def test_new_startup_has_new_instance_even_with_identical_files(self):
        first = self.inspect()
        startup = managed_engine.prepare(self.url, sys.executable, {'model_path': str(self.model)}, [sys.executable, *sys.argv])
        managed_engine.bind(startup, os.getpid())
        second = self.inspect()
        self.assertNotEqual(first['instanceId'], second['instanceId'])
        self.assertEqual(first['ggufSha256'], second['ggufSha256'])
        self.assertEqual(first['executableSha256'], second['executableSha256'])

    def test_change_after_request_invalidates_the_evidence_instead_of_relabeling_it(self):
        identity = {'engine': {'localInstance': self.inspect()}}
        self.model.write_bytes(b'C' * 4096)
        with self.assertRaises(managed_engine.InstanceChanged):
            managed_engine.confirm(self.url, {'model_path': str(self.model)}, identity)
        self.assertEqual(identity['engine']['localInstance'], {'status': 'changed', 'reason': 'binding_invalidated'})

    def test_restart_during_request_invalidates_old_evidence(self):
        identity = {'engine': {'localInstance': self.inspect()}}
        startup = managed_engine.prepare(self.url, sys.executable, {'model_path': str(self.model)}, [sys.executable, *sys.argv])
        managed_engine.bind(startup, os.getpid())
        with self.assertRaises(managed_engine.InstanceChanged):
            managed_engine.confirm(self.url, {'model_path': str(self.model)}, identity)
        self.assertEqual(identity['engine']['localInstance']['status'], 'changed')

    def test_remote_router_and_sharded_modes_do_not_hash_unverified_files(self):
        with patch.object(managed_engine, 'fingerprint_file') as hashed:
            self.assertIsNone(managed_engine.prepare('http://remote.example:8080', '/PRIVATE/engine', {'model_path': '/PRIVATE/model'}, []))
            self.assertIsNone(managed_engine.prepare(self.url, '/PRIVATE/engine', {'model_path': '/PRIVATE/model', 'models_dir': '/PRIVATE/models'}, []))
            self.assertIsNone(managed_engine.prepare(self.url, '/PRIVATE/engine', {'model_path': '/PRIVATE/model-00001-of-00002.gguf'}, []))
        hashed.assert_not_called()

    def test_changed_arguments_invalidate_instance_and_remove_startup_configuration(self):
        record=json.loads(self.path.read_text())
        record['startupConfiguration']=managed_engine.startup_configuration([sys.executable,'--threads','77'],{})
        self.path.write_text(json.dumps(record))
        with self.assertRaises(managed_engine.InstanceChanged):self.inspect()

    def test_invalid_startup_configuration_cannot_export_arbitrary_settings(self):
        record=json.loads(self.path.read_text())
        record['startupConfiguration']['settings']['PRIVATE_KEY']='PRIVATE_SECRET'
        self.path.write_text(json.dumps(record))
        value=self.inspect()
        self.assertEqual(value['status'],'unknown')
        self.assertNotIn('PRIVATE',json.dumps(value))
