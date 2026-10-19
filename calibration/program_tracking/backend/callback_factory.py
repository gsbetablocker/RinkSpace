"""Select the checked, reference-preserving CustomFactor bridge per graph.

The bridge only borrows factor/Values during a synchronous callback. The
backend copies atVector() results before passing them to observation code.
No global gtsam class replacement or runtime compilation is performed.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import gtsam


DEFAULT_DIRECTORY = Path(
    'D:/CodexData/RinkSpace_program_tracking_tmp/solver_acceleration/'
    'round3/native/reference_callback')
SOURCE = Path(__file__).resolve().parents[1] / 'native/reference_callback.cpp'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _load_reference(directory: str):
    folder = Path(directory).resolve()
    manifest_path = folder / 'build_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if (sys.platform != 'win32'
            or manifest['python_version'] != f'{sys.version_info.major}.{sys.version_info.minor}'):
        raise ValueError('Reference bridge was built for a different platform/Python ABI')
    binary = folder / 'reference_callback.pyd'
    # Check the reviewed project source, built source, binary and runtime DLL.
    # A matching Python version alone does not establish C++ ABI compatibility.
    paths = {'source_sha256': SOURCE, 'module_sha256': binary,
             'gtsam_dll_sha256': Path(sys.prefix) / 'Library/bin/gtsam.dll'}
    for field, path in paths.items():
        if _sha(path) != manifest[field]:
            raise ValueError(f'Reference bridge {field} does not match {path}')
    if _sha(folder / 'reference_callback.cpp') != manifest['source_sha256']:
        raise ValueError('Reference bridge build source differs from reviewed source')
    name = 'reference_callback'  # Extension exports PyInit_reference_callback.
    module = sys.modules.get(name)
    if module is not None:
        if Path(module.__file__).resolve() != binary:
            raise ValueError('A different reference_callback extension is already loaded')
    else:
        spec = importlib.util.spec_from_file_location(name, binary)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load {binary}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    if not callable(module.create):
        raise TypeError('Reference bridge create is not callable')
    return module.create, {
        'implementation': 'borrowed_reference', 'binary': str(binary),
        'build_manifest_sha256': _sha(manifest_path), 'build': manifest,
        'values_lifetime': 'synchronous_callback_only',
        'jacobians': 'gtsam.JacobianVector_copy_back',
    }


def select_factory():
    """auto prefers the verified binary; reference requires it; legacy opts out."""
    mode = os.environ.get('RINKSPACE_GTSAM_CALLBACK', 'auto').lower()
    if mode not in ('auto', 'reference', 'legacy'):
        raise ValueError(f'Unknown RINKSPACE_GTSAM_CALLBACK: {mode}')
    error = None
    if mode != 'legacy':
        directory = os.environ.get('RINKSPACE_GTSAM_CALLBACK_DIRECTORY', str(DEFAULT_DIRECTORY))
        try:
            factory, metadata = _load_reference(str(Path(directory).resolve()))
            return factory, {'mode': mode, **metadata, 'load_error': None}
        except (OSError, ValueError, KeyError, AttributeError, ImportError, TypeError) as exc:
            if mode == 'reference':
                raise RuntimeError('Verified GTSAM reference callback bridge unavailable') from exc
            error = f'{type(exc).__name__}: {exc}'
    return gtsam.CustomFactor, {
        'mode': mode, 'implementation': 'legacy_pybind_callback', 'load_error': error,
    }
