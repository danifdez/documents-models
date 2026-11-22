import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import websockets
from voice import live_worker as live
from voice import evidence
from tests.voice_fixtures import make_runtime,wav_bytes


class RuntimeTests(unittest.TestCase):
    def test_model_files_are_bound_to_loaded_instance_and_restart_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            first=make_runtime(directory)
            identity=first.evidence['model']['sha256']
            (Path(directory)/'model.bin').write_bytes(b'changed weights')
            with self.assertRaises(ValueError):first.assert_unchanged()
            second=make_runtime(directory)
            self.assertNotEqual(identity,second.evidence['model']['sha256'])
            self.assertEqual(second.evidence['engine']['device'],'cpu')
            self.assertNotIn(directory,json.dumps(second.evidence))

    def test_private_decoder_includes_generator_time_and_cleans_files(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=make_runtime(directory);runtime.model.delay=.025
            original=subprocess.run;paths=[]
            def checked(args,**kwargs):
                source=Path(args[args.index('-i')+1]);target=Path(args[-1]);paths.extend([source,target])
                self.assertEqual(source.stat().st_mode&0o777,0o600)
                self.assertEqual(target.stat().st_mode&0o777,0o600)
                self.assertEqual(target.parent.stat().st_mode&0o777,0o700)
                self.assertEqual(kwargs['env'],{'PATH':os.defpath,'LANG':'C'})
                self.assertEqual(kwargs['timeout'],15)
                self.assertEqual(args[args.index('-format_whitelist')+1],'wav,matroska,webm,ogg')
                self.assertEqual(args[args.index('-protocol_whitelist')+1],'file,pipe')
                return original(args,**kwargs)
            with patch.object(evidence.subprocess,'run',side_effect=checked):
                text,details=evidence.recognize(wav_bytes(),runtime,True)
            self.assertEqual(text,'Synthetic voice output')
            self.assertGreaterEqual(details['inferenceUs'],25000)
            self.assertEqual(details['decoded']['samples'],320)
            self.assertTrue(all(not path.exists() for path in paths))

    def test_decoder_timeout_cleans_files_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=make_runtime(directory);paths=[]
            def timeout(args,**kwargs):
                paths.append(Path(args[-1]));raise subprocess.TimeoutExpired('ffmpeg',15)
            audio = wav_bytes()
            with patch.object(evidence.subprocess, 'run', side_effect=timeout):
                with self.assertRaises(subprocess.TimeoutExpired):
                    evidence.recognize(audio, runtime, True)
            self.assertFalse(paths[0].parent.exists());self.assertEqual(runtime.model.calls,[])

    def test_playlist_format_is_not_decoded_or_transcribed(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=make_runtime(directory)
            with self.assertRaises(ValueError):
                evidence.recognize(b'#EXTM3U\nhttp://127.0.0.1:1/private\n',runtime,True)
            self.assertEqual(runtime.model.calls,[])


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):  # noqa: V105 (unittest lifecycle hook)
        self.temp=tempfile.TemporaryDirectory();self.runtime=make_runtime(self.temp.name)
        self.config={'window_seconds':5,'vad_filter':True}
        self.patches=[patch.object(live,'_loaded',self.runtime),patch.object(live,'_busy',False),
                      patch.object(live,'_live_config',side_effect=lambda:dict(self.config))]
        for p in self.patches:p.start()
        self.server=await websockets.serve(live.handle_connection,'127.0.0.1',0,
                                          process_request=live._build_process_request())
        self.url='ws://127.0.0.1:'+str(self.server.sockets[0].getsockname()[1])
    async def asyncTearDown(self):  # noqa: V105 (unittest lifecycle hook)
        if self.runtime.model.release:self.runtime.model.release.set()
        self.server.close();await self.server.wait_closed()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    async def connect(self,capture=True):
        ws=await websockets.connect(self.url)
        await ws.send(json.dumps({'type':'start','capture':capture}))
        return ws
    async def test_runtime_and_call_frames_precede_final_and_match_input(self):
        async with await self.connect() as ws:
            await ws.send(wav_bytes());await ws.send(json.dumps({'type':'stop'}))
            values=[json.loads(message) async for message in ws]
        frames=[v for v in values if v['type']=='evidence']
        self.assertEqual([v['sequence'] for v in frames],[1,2,3])
        self.assertEqual([v['value'].get('phase') for v in frames],[None,'started','completed'])
        self.assertEqual(frames[1]['value']['input']['sha256'],evidence.sha256(wav_bytes()))
        self.assertEqual(values[-1],{'type':'partial','text':'Synthetic voice output','isFinal':True})
        self.assertEqual(len(self.runtime.model.calls),1)
    async def test_read_ahead_preserves_partial_and_final_call_boundaries(self):
        self.config['window_seconds']=.001;self.runtime.model.delay=.02
        async with await self.connect() as ws:
            await ws.send(wav_bytes());await ws.send(json.dumps({'type':'stop'}))
            values=[json.loads(message) async for message in ws]
        starts=[v['value'] for v in values if v.get('kind')=='stream_inference' and v['value']['phase']=='started']
        self.assertEqual([v['final'] for v in starts],[False,True])
        self.assertEqual([v['input']['sha256'] for v in starts],[evidence.sha256(wav_bytes())]*2)
        self.assertEqual(len(self.runtime.model.calls),2)
        self.assertEqual([v['isFinal'] for v in values if v['type']=='partial'],[False,True])
    async def test_cancel_with_open_socket_discards_late_evidence_and_output(self):
        self.config['window_seconds']=.001;self.runtime.model.release=threading.Event()
        async with await self.connect() as ws:
            await ws.send(wav_bytes())
            runtime=json.loads(await ws.recv());started=json.loads(await ws.recv())
            self.assertEqual(runtime['kind'],'stream_runtime');self.assertEqual(started['value']['phase'],'started')
            await ws.send(json.dumps({'type':'cancel'}))
            await asyncio.sleep(.03)
            self.runtime.model.release.set()
            self.assertEqual([message async for message in ws],[])
        self.assertEqual(len(self.runtime.model.calls),1)
    async def test_normal_session_does_not_emit_evaluation_evidence(self):
        async with await self.connect(False) as ws:
            await ws.send(wav_bytes());await ws.send(json.dumps({'type':'stop'}))
            values=[json.loads(message) async for message in ws]
        self.assertEqual(len(values),1);self.assertEqual(values[0]['type'],'partial')
    async def test_cancel_drains_active_native_work_before_worker_can_be_reused(self):
        self.config['window_seconds']=.001;self.runtime.model.release=threading.Event()
        first=await self.connect();await first.send(wav_bytes())
        for _ in range(200):
            if self.runtime.model.entered.is_set():break
            await asyncio.sleep(.01)
        self.assertTrue(self.runtime.model.entered.is_set())
        await first.send(json.dumps({'type':'cancel'}));await first.close();await asyncio.sleep(.02)
        self.assertTrue(live._busy)
        response=live._build_process_request()(None,type('Request',(),{'path':'/health'})())
        self.assertEqual(response.status_code,503)
        async with websockets.connect(self.url) as second:
            with self.assertRaises(websockets.ConnectionClosed):await second.recv()
        self.assertEqual(len(self.runtime.model.calls),1)
        self.runtime.model.release.set()
        for _ in range(200):
            if not live._busy:break
            await asyncio.sleep(.01)
        self.assertFalse(live._busy)
    async def test_input_limits_and_invalid_controls_do_not_run_recognition(self):
        for payload in (b'x'*65,'{"type":"start","capture":true,"path":"PRIVATE"}'):
            with patch.object(live,'MAX_AUDIO_BYTES',64):
                async with await self.connect(False) as ws:
                    await ws.send(payload)
                    values=[json.loads(message) async for message in ws]
            self.assertEqual(values[-1],{'type':'error','message':'Voice processing failed'})
        self.assertEqual(self.runtime.model.calls,[])


if __name__=='__main__':unittest.main()
