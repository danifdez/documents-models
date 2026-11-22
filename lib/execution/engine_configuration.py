"""Bounded, body-free projections of managed arguments and server declarations."""
from copy import deepcopy
import math
import os

from common.canonical_json import canonical_sha256

_INTEGER_ARGS = {'--ctx-size', '--parallel', '--threads', '--threads-batch', '--batch-size', '--ubatch-size', '--top-k', '--seed'}
_NUMBER_ARGS = {'--temp', '--top-p', '--min-p', '--repeat-penalty', '--presence-penalty', '--frequency-penalty'}
_SWITCH_ARGS = {'--jinja', '--metrics', '--mmap', '--no-mmap', '--mlock', '--no-mlock',
                '--context-shift', '--no-context-shift', '--cont-batching', '--no-cont-batching', '--lora-init-without-apply'}
_CACHE_TYPES = {'f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'}
_OMISSIONS = {'unsupported_arguments', 'environment_overrides'}


def _pack(source, settings, omissions):
    value = {'source': source, 'settings': deepcopy(settings), 'omissions': sorted(omissions), 'complete': not omissions}
    return {**value, 'sha256': canonical_sha256(value)}


def _setting(name, value):
    if name in _INTEGER_ARGS or name == 'loraCount':
        return type(value) is int and -2**31 <= value <= 2**32 - 1
    if name in _NUMBER_ARGS:
        return type(value) in (int, float) and abs(value) <= 1e12 and math.isfinite(value)
    if name in _SWITCH_ARGS:
        return type(value) is bool
    if name == '--n-gpu-layers':
        return (type(value) is int and -1 <= value < 2**31) or value in ('auto', 'all')
    if name in ('--cache-type-k', '--cache-type-v'):
        return isinstance(value, str) and value in _CACHE_TYPES
    if name == '--flash-attn':
        return value in ('on', 'off', 'auto')
    return False


_IGNORED_VALUE_ARGS = {'--host', '--port', '--model', '--lora'}
_STARTUP_VALUE_ARGS = _INTEGER_ARGS | _NUMBER_ARGS | {
    '--n-gpu-layers', '--cache-type-k', '--cache-type-v', '--flash-attn'
}


def _apply_startup_switch(flag, settings):
    negated = flag.startswith('--no-')
    key = '--' + flag[5:] if negated else flag
    settings[key] = not negated


def _skip_startup_value(args, index, flag, settings):
    if index >= len(args):
        return index, True
    if flag == '--lora':
        settings['loraCount'] = settings.get('loraCount', 0) + 1
    return index + 1, False


def _read_startup_setting(args, index, flag, settings):
    value = args[index]
    try:
        if flag in _INTEGER_ARGS or (flag == '--n-gpu-layers' and value not in ('auto', 'all')):
            value = int(value)
        elif flag in _NUMBER_ARGS:
            value = float(value)
        if not _setting(flag, value):
            raise ValueError()
        settings[flag] = value
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _consume_startup_argument(args, index, settings):
    flag = args[index]
    index += 1
    if flag in _SWITCH_ARGS:
        _apply_startup_switch(flag, settings)
        return index, False
    if flag in _IGNORED_VALUE_ARGS:
        return _skip_startup_value(args, index, flag, settings)
    if flag not in _STARTUP_VALUE_ARGS or index >= len(args):
        return index, True
    return index + 1, not _read_startup_setting(args, index, flag, settings)


def startup_configuration(command, environment=None):
    """Capture the actual argv projection; never store argv, paths or environment values."""
    settings, omissions = {}, set()
    args = list(command[1:])
    index = 0
    while index < len(args):
        index, omitted = _consume_startup_argument(args, index, settings)
        if omitted:
            omissions.add('unsupported_arguments')
    environment = os.environ if environment is None else environment
    if any(key.startswith(('LLAMA_ARG_', 'CUDA_', 'GGML_', 'HIP_', 'OMP_', 'SYCL_', 'VK_')) for key in environment):
        omissions.add('environment_overrides')
    return _pack('managed_argv', settings, omissions)


def valid_startup_configuration(value):
    try:
        return (isinstance(value, dict) and value.get('source') == 'managed_argv'
                and isinstance(value['settings'], dict) and len(value['settings']) <= 64
                and all(_setting(name, item) for name, item in value['settings'].items())
                and type(value.get('complete')) is bool
                and isinstance(value['omissions'], list) and set(value['omissions']) <= _OMISSIONS
                and value == _pack('managed_argv', value['settings'], set(value['omissions'])))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


_INTEGER_DEFAULTS = {'seed', 'top_k', 'repeat_last_n', 'dry_allowed_length', 'dry_penalty_last_n',
                     'mirostat', 'max_tokens', 'n_predict', 'n_keep', 'n_discard', 'n_probs', 'min_keep'}
_NUMBER_DEFAULTS = {'temperature', 'dynatemp_range', 'dynatemp_exponent', 'top_p', 'min_p', 'top_n_sigma',
                    'xtc_probability', 'xtc_threshold', 'typical_p', 'repeat_penalty', 'presence_penalty',
                    'frequency_penalty', 'dry_multiplier', 'dry_base', 'mirostat_tau', 'mirostat_eta',
                    'adaptive_target', 'adaptive_decay'}
_BOOLEAN_DEFAULTS = {'ignore_eos', 'stream', 'reasoning_in_content', 'timings_per_token', 'post_sampling_probs', 'backend_sampling'}
_SAMPLERS = {'dry', 'top_k', 'top_p', 'top_n_sigma', 'min_p', 'xtc', 'typ_p', 'typical_p', 'temperature',
             'penalties', 'infill', 'dist', 'mirostat', 'adaptive_p'}


def _generation_value(name, value):
    if name in _INTEGER_DEFAULTS:
        return type(value) is int and -2**31 <= value <= 2**32 - 1
    if name in _NUMBER_DEFAULTS:
        return type(value) in (int, float) and abs(value) <= 1e12 and math.isfinite(value)
    if name in _BOOLEAN_DEFAULTS:
        return type(value) is bool
    if name == 'samplers':
        return isinstance(value, list) and len(value) <= 32 and all(isinstance(item, str) and item in _SAMPLERS for item in value)
    if name == 'reasoning_format':
        return isinstance(value, str) and value in {'none', 'deepseek', 'deepseek-legacy', 'auto'}
    if name == 'generation_prompt':
        return value == ''
    if name == 'lora':
        return _lora_value(value)
    return False


def _lora_value(value):
    return isinstance(value, list) and len(value) <= 32 and all(
        isinstance(item, dict) and set(item) == {'id', 'scale'} and type(item['id']) is int
        and 0 <= item['id'] < 2**31 and type(item['scale']) in (int, float)
        and abs(item['scale']) <= 1e12 and math.isfinite(item['scale']) for item in value)


def generation_defaults(data):
    """GET /props reports a metrics subset, not all effective sampling options."""
    defaults = data.get('default_generation_settings') if isinstance(data, dict) else None
    params = defaults.get('params') if isinstance(defaults, dict) else None
    if not isinstance(params, dict) or not params or len(params) > 128:
        return None
    settings, omissions = {}, {'unreported_generation_parameters'}
    for name, value in params.items():
        if _generation_value(name, value):
            settings[name] = value
        else:
            omissions.add('unexported_or_invalid_values')
    return _pack('server_props_metrics', settings, omissions)
