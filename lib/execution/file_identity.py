"""Stable identities of configured regular files, shared by inference verification."""
import hashlib
import os
import stat


def stat_signature(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def file_signature(path):
    value = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Inference file must be regular')
    return stat_signature(value)


FILE_CHANGED_MESSAGE = 'Inference file changed during verification'


def fingerprint_file(path):
    try:
        initial = file_signature(path)
        flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        with os.fdopen(os.open(path, flags), 'rb') as file:
            before = stat_signature(os.fstat(file.fileno()))
            if before != initial:
                raise ValueError(FILE_CHANGED_MESSAGE)
            digest = hashlib.sha256()
            remaining = before[2]
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(FILE_CHANGED_MESSAGE)
                digest.update(chunk)
                remaining -= len(chunk)
            if file.read(1) or before != stat_signature(os.fstat(file.fileno())) or before != file_signature(path):
                raise ValueError(FILE_CHANGED_MESSAGE)
    except OSError:
        raise ValueError('Inference file verification failed') from None
    return {'sha256': digest.hexdigest(), 'signature': before}
