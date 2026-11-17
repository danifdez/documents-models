"""Bounded evidence for the loaded speech engine, without paths or environment."""
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import wave

from common.canonical_json import canonical_sha256
from lib.execution.code_identity import code_fingerprint
from lib.execution.runtime_identity import runtime_fingerprint
from lib.execution.file_identity import file_signature, fingerprint_file

MAX_AUDIO_BYTES = 32 * 1024 * 1024
MAX_CALLS = 128
MAX_EVIDENCE_BYTES = 1024 * 1024
DECODE_TIMEOUT = 15
MAX_PCM_BYTES = 64 * 1024 * 1024


def sha256(body):
    return 'sha256:' + hashlib.sha256(body).hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    identity = fingerprint_file(path)
    return path, identity


@dataclass
class LoadedRuntime:
    model: object
    evidence: dict
    bindings: list
    decoder: Path

    def assert_unchanged(self):
        if any(file_signature(path) != item['signature'] for path, item in self.bindings):
            raise ValueError('Voice runtime files changed; restart the worker')


def load_runtime(cfg):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    configured = str(cfg['model'])
    # Resolving aliases is explicit and local: execution never downloads weights.
    directory = Path(configured)
    if not directory.is_dir():
        directory = Path(download_model(configured, local_files_only=True))
    directory = directory.resolve(strict=True)
    names = ['config.json', 'model.bin', 'tokenizer.json']
    names += [name for name in ['preprocessor_config.json', 'vocabulary.json', 'vocabulary.txt']
              if (directory/name).is_file()]
    bindings = [binding(directory/name) for name in names]
    files = [{'name':name, 'sha256':'sha256:'+item['sha256'], 'size':item['signature'][2]}
             for name, (_, item) in zip(names, bindings)]
    model = WhisperModel(str(directory), device=cfg['device'], compute_type=cfg['compute_type'], local_files_only=True)
    core = model.model
    modules = []
    for role, obj in [('wrapper', model), ('engine', core)]:
        path, item = binding(sys.modules[type(obj).__module__].__file__)
        bindings.append((path, item))
        modules.append({'role':role, 'sha256':'sha256:'+item['sha256'], 'size':item['signature'][2]})
    executable = shutil.which('ffmpeg')
    if not executable:
        raise ValueError('Voice decoder is unavailable')
    decoder, decoder_id = binding(executable)
    bindings.append((decoder, decoder_id))
    record = {'schemaVersion':'voice-runtime/1',
              'model':{'sha256':canonical_sha256(files), 'files':files},
              'engine':{'name':'faster-whisper', 'packages':{name:version(name) for name in ('faster-whisper','ctranslate2','av')},
                        'modules':modules, 'device':core.device, 'computeType':core.compute_type, 'deviceIndex':list(core.device_index)},
              'decoder':{'name':'ffmpeg', 'sha256':'sha256:'+decoder_id['sha256'],
                         'sampleRate':16000, 'channels':1, 'sampleWidth':2},
              'requested':{'device':str(cfg['device']), 'computeType':str(cfg['compute_type'])},
              'codeFingerprint':code_fingerprint(), 'runtimeFingerprint':runtime_fingerprint(),
              'omissions':['engine.unreported_generation_defaults']}
    result = LoadedRuntime(model, record, bindings, decoder)
    result.assert_unchanged()
    return result


@contextmanager
def decoded_audio(blob, runtime):
    """Private files, local-only protocols, bounded time/output and clean env."""
    if not 0 < len(blob) <= MAX_AUDIO_BYTES:
        raise ValueError('Voice input limit exceeded')
    runtime.assert_unchanged()
    with tempfile.TemporaryDirectory(prefix='documents-voice-') as directory:
        source = Path(directory)/'input.bin'; target = Path(directory)/'decoded.wav'
        for path, body in [(source, blob), (target, b'')]:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as file:
                file.write(body)
        result = subprocess.run([str(runtime.decoder), '-nostdin', '-loglevel', 'error', '-y',
            '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'wav,matroska,webm,ogg', '-i', str(source), '-map_metadata', '-1',
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-fs', str(MAX_PCM_BYTES + 1), str(target)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=DECODE_TIMEOUT, check=False, env={'PATH':os.defpath, 'LANG':'C'})
        runtime.assert_unchanged()
        if result.returncode or not 44 <= target.stat().st_size <= MAX_PCM_BYTES:
            raise ValueError('Voice decoding failed or exceeded its limit')
        with wave.open(str(target), 'rb') as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1,2,16000,'NONE'):
                raise ValueError('Unexpected decoded voice format')
            samples = audio.getnframes()
        # The private WAV is consumed and deleted before returning to the pool.
        yield str(target), {'sha256':sha256(target.read_bytes()), 'samples':samples,
                            'sampleRate':16000, 'channels':1, 'sampleWidth':2}


def recognize(blob, runtime, vad_filter):
    import time
    start = time.perf_counter_ns()
    with decoded_audio(blob, runtime) as (path, decoded):
        decoded_at = time.perf_counter_ns()
        runtime.assert_unchanged()
        segments, info = runtime.model.transcribe(path, beam_size=1, vad_filter=vad_filter,
                                                 condition_on_previous_text=False)
        # Consumption is part of inference: faster-whisper returns a generator.
        text = ' '.join(seg.text.strip() for seg in segments if seg.text).strip()
        finished = time.perf_counter_ns()
        runtime.assert_unchanged()
    return text, {'decoded':decoded, 'decodeUs':(decoded_at-start)//1000,
                  'inferenceUs':(finished-decoded_at)//1000,
                  'parameters':{'beamSize':info.transcription_options.beam_size,
                                'vadFilter':vad_filter,
                                'conditionOnPreviousText':info.transcription_options.condition_on_previous_text},
                  'audioDurationUs':round(info.duration * 1000000),
                  'audioAfterVadUs':round(info.duration_after_vad * 1000000)}
