# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, get_package_paths


def collect_namespace_modules(root):
    modules = []
    for path in Path(root).rglob('*.py'):
        relative = path.with_suffix('')
        if relative.name == '__init__':
            relative = relative.parent
        if relative.parts:
            modules.append('.'.join(relative.parts))
    return sorted(set(modules))

datas = [
    ('common', 'common'),
    ('tasks', 'tasks'),
    ('services/prompt_templates', 'services/prompt_templates'),
]
binaries = []
hiddenimports = ['tiktoken_ext.openai_public', 'tiktoken_ext']
hiddenimports += collect_namespace_modules('tasks')
hiddenimports += collect_namespace_modules('services')
tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sentence_transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
torchvision_root = Path(get_package_paths('torchvision')[1])
for pattern in ('*.so', '*.dylib', '*.pyd'):
    binaries += [(str(path), 'torchvision') for path in torchvision_root.glob(pattern)]
tmp_ret = collect_all('huggingface_hub')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faster_whisper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('docling')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['executions.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='documents-models',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='documents-models',
)
