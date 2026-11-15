"""Bind startup file hashes to a local process; never adopt a live external endpoint."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from lib.execution.engine_configuration import startup_configuration, valid_startup_configuration
from lib.execution.file_identity import fingerprint_file, file_signature, stat_signature
from lib.execution.private_storage import write_private_text


class InstanceChanged(ValueError):
    pass


def _endpoint(url):
    try:
        value = urlsplit(url)
        if (value.scheme == 'http' and value.hostname == '127.0.0.1' and value.port
                and not (value.username or value.password or value.query or value.fragment)
                and value.path in ('', '/') and not any(c.isspace() for c in url)):
            return f'http://127.0.0.1:{value.port}', value.port
    except ValueError:
        pass
    return None


def _record_path(url):
    from worker.identity import worker_data_dir
    key = hashlib.sha256(_endpoint(url)[0].encode()).hexdigest()
    return Path(worker_data_dir()) / '.engine' / (key + '.json')


def _process(pid):
    if type(pid) is not int or not 0 < pid < 2**31:
        return None
    root = Path('/proc') / str(pid)
    try:
        if root.stat().st_uid != os.geteuid():
            return None
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': pid, 'startTicks': int(fields[19]),
                'bootId': str(UUID(Path('/proc/sys/kernel/random/boot_id').read_text().strip()))}
    except (OSError, ValueError, IndexError):
        return None


def _snapshot(path):
    configured = os.path.abspath(path)
    alias = stat_signature(os.lstat(configured))
    resolved = str(Path(configured).resolve(strict=True))
    result = {'configuredPath': configured, 'path': resolved, 'alias': alias, **fingerprint_file(resolved)}
    _check_file(result)
    return result


def _check_file(value):
    try:
        if (stat_signature(os.lstat(value['configuredPath'])) == value['alias']
                and str(Path(value['configuredPath']).resolve(strict=True)) == value['path']
                and file_signature(value['path']) == value['signature']):
            return
    except (OSError, ValueError):
        pass
    raise InstanceChanged('Managed inference files changed; restart the engine')


def prepare(url, binary, engine, command):
    """Hash once at a managed startup, before the binary can load the GGUF."""
    if not sys.platform.startswith('linux') or not _endpoint(url) or engine.get('models_dir'):
        return None
    # A single file is the supported model unit in this first local binding.
    if re.search(r'-\d{5}-of-\d{5}\.gguf$', str(engine['model_path'])):
        return None
    return {'schemaVersion': 'managed-engine/1', 'instanceId': str(uuid4()),
            'endpoint': _endpoint(url)[0], 'startupConfiguration': startup_configuration(command),
            'files': {'executable': _snapshot(binary), 'gguf': _snapshot(engine['model_path'])}}


def bind(record, pid):
    if record is None:
        return
    process = _process(pid)
    if process is None:
        raise InstanceChanged('Managed inference process is unavailable')
    for value in record['files'].values():
        _check_file(value)
    path = _record_path(record['endpoint'])
    if path.parent.exists() and (path.parent.is_symlink() or path.parent.stat().st_uid != os.geteuid()):
        raise InstanceChanged('Managed inference state is unavailable')
    write_private_text(path, json.dumps({**record, 'process': process}, separators=(',', ':')))


def _read_record(url):
    path = _record_path(url)
    try:
        directory = path.parent.lstat()
        if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid() or directory.st_mode & 0o077:
            return None
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        with os.fdopen(os.open(path, flags), 'rb') as file:
            metadata = os.fstat(file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077 or metadata.st_nlink != 1:
                return None
            body = file.read(16385)
        if len(body) > 16384:
            return None
        record = json.loads(body)
        if record['schemaVersion'] != 'managed-engine/1' or record['endpoint'] != _endpoint(url)[0]:
            return None
        if not isinstance(record['instanceId'], str) or str(UUID(record['instanceId'])) != record['instanceId']:
            return None
        if not valid_startup_configuration(record.get('startupConfiguration')):
            return None
        if set(record['files']) != {'executable', 'gguf'}:
            return None
        for value in record['files'].values():
            if not re.fullmatch(r'[0-9a-f]{64}', value['sha256']):
                return None
            for field in ('path', 'configuredPath'):
                if not isinstance(value[field], str) or not os.path.isabs(value[field]) or len(value[field]) > 4096 or '\x00' in value[field]:
                    return None
            for field in ('signature', 'alias'):
                if len(value[field]) != 5 or any(type(item) is not int or item < 0 for item in value[field]):
                    return None
        return record
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _owns_listener(pid, port):
    root = Path('/proc') / str(pid)
    sockets = set()
    for path in (root / 'fd').iterdir():
        try:
            sockets.add(os.readlink(path))
        except OSError:
            continue
    for line in (root / 'net/tcp').read_text().splitlines()[1:]:
        fields = line.split()
        if (fields[3] == '0A' and fields[1] == f'0100007F:{port:04X}'
                and f'socket:[{fields[9]}]' in sockets):
            return True
    return False


def _maps_gguf(pid, signature):
    device, inode, size = signature[:3]
    # llama.cpp unmaps metadata and offloaded tensor ranges after loading.
    for line in (Path('/proc') / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        major, minor = (int(value, 16) for value in fields[3].split(':'))
        if os.makedev(major, minor) == device and int(fields[4]) == inode and fields[1].startswith('r'):
            first, last = (int(value, 16) for value in fields[0].split('-'))
            if size > 0 and last > first and int(fields[2], 16) < size:
                return True
    return False


def _unknown(reason):
    return {'status': 'unknown', 'reason': reason}


def inspect(url, properties):
    """Export only public hashes after checking birth, listener, executable and mapping."""
    if not sys.platform.startswith('linux') or not _endpoint(url):
        return _unknown('local_binding_unsupported')
    record = _read_record(url)
    if record is None:
        return _unknown('no_managed_startup')
    try:
        process = record['process']
        pid = process['pid']
        if _process(pid) != process or not _owns_listener(pid, _endpoint(url)[1]):
            return _unknown('instance_not_bound')
        executable = os.stat(Path('/proc') / str(pid) / 'exe')
        if [executable.st_dev, executable.st_ino] != record['files']['executable']['signature'][:2]:
            return _unknown('executable_not_bound')
        if not _maps_gguf(pid, record['files']['gguf']['signature']):
            return _unknown('gguf_not_bound')
        for value in record['files'].values():
            _check_file(value)
        if properties.get('model_path') not in {record['files']['gguf']['path'], record['files']['gguf']['configuredPath']}:
            return _unknown('model_declaration_not_bound')
        with (Path('/proc') / str(pid) / 'cmdline').open('rb') as file:
            argv = file.read(65537)
        if len(argv) > 65536 or not argv.endswith(b'\x00'):
            return _unknown('arguments_unavailable')
        observed = startup_configuration([os.fsdecode(item) for item in argv[:-1].split(b'\x00')], {})
        startup = record['startupConfiguration']
        if (observed['settings'] != startup['settings']
                or set(observed['omissions']) != set(startup['omissions']) - {'environment_overrides'}):
            raise InstanceChanged('Managed inference arguments changed; restart the engine')
        if _process(pid) != process:
            return _unknown('instance_not_bound')
    except InstanceChanged:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError):
        return _unknown('instance_unavailable')
    return {'status': 'verified_local', 'instanceId': record['instanceId'],
            'binding': 'linux_process_listener_mmap', 'startupConfiguration': startup,
            'executableSha256': 'sha256:' + record['files']['executable']['sha256'],
            'ggufSha256': 'sha256:' + record['files']['gguf']['sha256']}


def confirm(url, properties, identity):
    before = (identity.get('engine') or {}).get('localInstance')
    if not before or before['status'] != 'verified_local':
        return
    try:
        after = inspect(url, properties)
        if after != before:
            raise InstanceChanged('Managed inference instance changed during the request')
    except InstanceChanged:
        identity['engine']['localInstance'] = {'status': 'changed', 'reason': 'binding_invalidated'}
        raise
