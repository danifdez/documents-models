"""Bounded live dictation; evidence uses the same private worker connection."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import http
import json
import logging
import math
import os

from common.canonical_json import canonical_bytes, canonical_sha256
from lib.llm.config import get_config, get_task_config
from voice.evidence import load_runtime, recognize, sha256, MAX_AUDIO_BYTES, MAX_CALLS, MAX_EVIDENCE_BYTES

logger = logging.getLogger(__name__)
_loaded = None
_busy = False


def _live_config():
    base = {'model':'base', 'device':'cpu', 'compute_type':'int8', 'window_seconds':5,
            'vad_filter':True, 'host':'127.0.0.1', 'port':9100}
    base.update(get_task_config('transcribe') or {})
    base.update(get_config().get('voice_live', {}))
    seconds = base['window_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0 < seconds <= 60 or type(base['vad_filter']) is not bool:
        raise ValueError('Invalid voice configuration')
    return base


def get_runtime():
    global _loaded
    if _loaded is None:
        _loaded = load_runtime(_live_config())
    _loaded.assert_unchanged()
    return _loaded


@dataclass
class Session:
    buffer: bytearray = field(default_factory=bytearray)
    chunks: int = 0
    bytes_since_last_run: int = 0
    calls: int = 0
    evidence_sequence: int = 0
    evidence_bytes: int = 0
    capture: bool = False
    cancelled: bool = False
    stable_text: str = ''
    last_window_text: str = ''

    @property
    def accumulated_text(self):
        return (self.stable_text + ' ' + self.last_window_text).strip()


async def _send_json(ws, payload):
    await ws.send(json.dumps(payload, ensure_ascii=False, allow_nan=False))


async def _evidence(ws, session, kind, value):
    if not session.capture:
        return
    body = canonical_bytes(value)
    if len(body) > 64*1024 or session.evidence_bytes + len(body) > MAX_EVIDENCE_BYTES:
        raise ValueError('Voice evidence limit exceeded')
    session.evidence_sequence += 1
    session.evidence_bytes += len(body)
    await _send_json(ws, {'type':'evidence','sequence':session.evidence_sequence,'kind':kind,'value':value})


async def _run_inference(ws, session, blob, chunks, *, is_final):
    runtime = get_runtime()
    session.calls += 1
    if session.calls > MAX_CALLS:
        raise ValueError('Voice call limit exceeded')
    call = {'schemaVersion':'voice-call/1','index':session.calls,'phase':'started',
            'runtimeHash':canonical_sha256(runtime.evidence),
            'input':{'sha256':sha256(blob),'bytes':len(blob),'chunks':chunks},
            'final':is_final,'startedAt':datetime.now(timezone.utc).isoformat()}
    await _evidence(ws, session, 'stream_inference', call)
    try:
        text, details = await asyncio.to_thread(recognize, blob, runtime, _live_config()['vad_filter'])
        if session.cancelled:
            return None
        if len(text.encode()) > (16*1024 if session.capture else 1024*1024):
            raise ValueError('Voice output limit exceeded')
        await _evidence(ws, session, 'stream_inference', {**call, 'phase':'completed', **details,
                        'text':text,'finishedAt':datetime.now(timezone.utc).isoformat()})
        if is_final:
            session.stable_text = (session.stable_text + ' ' + text).strip()
            session.last_window_text = ''
        else:
            session.last_window_text = text
        return session.accumulated_text
    except Exception:
        if not session.cancelled:
            await _evidence(ws, session, 'stream_inference', {**call,'phase':'failed',
                'errorCode':'voice_processing_failed','finishedAt':datetime.now(timezone.utc).isoformat()})
        raise


async def handle_connection(ws):
    global _busy
    if _busy:
        await ws.close(code=1013, reason='Voice worker is busy')
        return
    cfg = _live_config()
    _busy = True
    session = Session(); pending = None; receiving = None; stopping = False; started = False
    final_call = False
    waiting = []
    threshold = max(1, int(cfg['window_seconds']*16000*2//4))
    def enqueue(final):
        if len(waiting) + session.calls >= MAX_CALLS:
            raise ValueError('Voice call limit exceeded')
        waiting.append((len(session.buffer), session.chunks, final))
        session.bytes_since_last_run = 0
    def launch(snapshot):
        size, chunks, final = snapshot
        return asyncio.create_task(_run_inference(ws, session, bytes(session.buffer[:size]), chunks, is_final=final))
    try:
        receiving = asyncio.create_task(ws.recv())
        while True:
            done, _ = await asyncio.wait([task for task in (receiving,pending) if task], return_when=asyncio.FIRST_COMPLETED)
            if receiving in done:
                try: message = receiving.result()
                except Exception:
                    session.cancelled = True
                    break
                if isinstance(message, bytes):
                    if not started or stopping or not 0 < len(message) <= 256*1024 or len(session.buffer)+len(message)>MAX_AUDIO_BYTES or session.chunks>=4096:
                        raise ValueError('Voice input limit or state is invalid')
                    session.buffer.extend(message);session.chunks+=1;session.bytes_since_last_run+=len(message)
                    if session.bytes_since_last_run >= threshold:
                        enqueue(False)
                else:
                    value = json.loads(message)
                    if not isinstance(value,dict):raise ValueError('Invalid voice control')
                    kind = value.get('type')
                    if kind == 'start' and not started and set(value)=={'type','capture'} and type(value['capture']) is bool:
                        session.capture=value['capture'];started=True
                        await _evidence(ws,session,'stream_runtime',get_runtime().evidence)
                    elif kind == 'cancel' and started and set(value)=={'type'}:
                        session.cancelled=True
                        break
                    elif kind == 'stop' and started and not stopping and set(value)=={'type'}:
                        stopping=True
                        if session.buffer:enqueue(True)
                    else:raise ValueError('Invalid voice control')
                receiving=asyncio.create_task(ws.recv())
            if pending in done:
                text=pending.result();pending=None
                if text is not None:await _send_json(ws,{'type':'partial','text':text,'isFinal':final_call})
                if final_call:break
            if pending is None and waiting:
                snapshot=waiting.pop(0);final_call=snapshot[2];pending=launch(snapshot)
            elif pending is None and stopping:
                await _send_json(ws,{'type':'partial','text':'','isFinal':True});break
    except Exception:
        session.cancelled=True
        logger.warning('Voice session failed')
        try:await _send_json(ws,{'type':'error','message':'Voice processing failed'})
        except Exception:pass
    finally:
        session.cancelled=True
        if receiving:
            receiving.cancel()
            await asyncio.gather(receiving,return_exceptions=True)
        # Native inference cannot be safely interrupted as a Python thread.
        # Health stays busy until it drains; the pool must not reassign it early.
        if pending:await asyncio.gather(pending,return_exceptions=True)
        session.buffer.clear();_busy=False


def _build_process_request():
    from websockets.datastructures import Headers
    from websockets.http11 import Response
    def process_request(_connection, request):
        if request.path=='/health':
            available=_loaded is not None and not _busy
            body=json.dumps({'status':'ok' if available else 'busy','model_loaded':_loaded is not None}).encode()
            return Response(http.HTTPStatus.OK if available else http.HTTPStatus.SERVICE_UNAVAILABLE,
                            'OK' if available else 'Unavailable', Headers([('Content-Type','application/json'),('Content-Length',str(len(body)))]),body)
        return None
    return process_request


async def main():
    import websockets
    cfg=_live_config()
    if cfg['host'] not in ('127.0.0.1','::1'):
        raise ValueError('Voice workers require numeric loopback')
    get_runtime()
    async with websockets.serve(handle_connection,cfg['host'],int(os.environ.get('VOICE_LIVE_PORT',cfg['port'])),
                               max_size=256*1024,max_queue=16,process_request=_build_process_request()):
        await asyncio.Future()


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
