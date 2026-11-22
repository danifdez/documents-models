"""Speech-engine doubles; no Whisper/CTranslate2 model is imported or loaded."""
import io
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import wave

from voice.evidence import load_runtime


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
