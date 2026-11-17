"""Speech-engine doubles; no Whisper/CTranslate2 model is imported or loaded."""
import base64
import io
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import wave

from common.canonical_json import canonical_sha256
from voice.evidence import load_runtime, sha256


class FakeCore:
    device='cpu';compute_type='int8';device_index=[0]


class FakeWhisper:
    def __init__(self,*args,**kwargs):
        self.model=FakeCore();self.calls=[];self.entered=threading.Event();self.release=None;self.delay=0
    def transcribe(self,path,**kwargs):
        self.calls.append(kwargs)
        with wave.open(path,'rb') as audio:duration=audio.getnframes()/audio.getframerate()
        def segments():
            self.entered.set()
            if self.release is not None:self.release.wait(5)
            time.sleep(self.delay)
            yield SimpleNamespace(text='Synthetic voice output')
        return segments(),SimpleNamespace(duration=duration,duration_after_vad=duration,
            transcription_options=SimpleNamespace(beam_size=kwargs['beam_size'],
                condition_on_previous_text=kwargs['condition_on_previous_text']))


def make_runtime(directory):
    directory=Path(directory)
    for name,body in [('config.json',b'{}'),('model.bin',b'synthetic weights'),('tokenizer.json',b'{}')]:
        if not (directory/name).exists():(directory/name).write_bytes(body)
    faster=ModuleType('faster_whisper');faster.WhisperModel=FakeWhisper
    utils=ModuleType('faster_whisper.utils');utils.download_model=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No downloads'))
    with patch.dict(sys.modules,{'faster_whisper':faster,'faster_whisper.utils':utils}):
        return load_runtime({'model':str(directory),'device':'cpu','compute_type':'int8'})


def wav_bytes():
    stream=io.BytesIO()
    with wave.open(stream,'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000);audio.writeframes(bytes(640))
    return stream.getvalue()


def protocol_fixture():
    """Deterministic golden frames shared with the two receiving boundaries."""
    ref='sha256:'+'a'*64
    files=[{'name':name,'sha256':sha256(body),'size':len(body)} for name,body in
           [('config.json',b'{}'),('model.bin',b'synthetic weights'),('tokenizer.json',b'{}')]]
    runtime={'schemaVersion':'voice-runtime/1','model':{'sha256':canonical_sha256(files),'files':files},
        'engine':{'name':'faster-whisper','packages':{'faster-whisper':'1.2.1','ctranslate2':'4.8.0','av':'17.1.0'},
            'modules':[{'role':role,'sha256':ref,'size':10} for role in ('wrapper','engine')],
            'device':'cpu','computeType':'int8','deviceIndex':[0]},
        'decoder':{'name':'ffmpeg','sha256':ref,'sampleRate':16000,'channels':1,'sampleWidth':2},
        'requested':{'device':'cpu','computeType':'int8'},'codeFingerprint':ref,'runtimeFingerprint':ref,
        'omissions':['engine.unreported_generation_defaults']}
    audio=wav_bytes();started={'schemaVersion':'voice-call/1','index':1,'phase':'started',
        'runtimeHash':canonical_sha256(runtime),'input':{'sha256':sha256(audio),'bytes':len(audio),'chunks':3},
        'final':True,'startedAt':'2026-11-17T00:00:00.000000+00:00'}
    finished={**started,'phase':'completed','decoded':{'sha256':sha256(audio),'samples':320,
        'sampleRate':16000,'channels':1,'sampleWidth':2},'decodeUs':100,'inferenceUs':200,
        'parameters':{'beamSize':1,'vadFilter':True,'conditionOnPreviousText':False},
        'audioDurationUs':20000,'audioAfterVadUs':20000,'text':'Synthetic voice output',
        'finishedAt':'2026-11-17T00:00:00.000300+00:00'}
    return {'audio_base64':base64.b64encode(audio).decode(),'chunks':[256,256,172],
            'frames':[{'type':'evidence','sequence':index,'kind':kind,'value':value}
                for index,(kind,value) in enumerate([('stream_runtime',runtime),('stream_inference',started),
                                                    ('stream_inference',finished)],1)]}
