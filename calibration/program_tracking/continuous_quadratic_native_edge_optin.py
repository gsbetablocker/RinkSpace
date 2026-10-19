"""Isolated native source-edge bridge for quadratic-logzoom physical15 rigs.

The round6 production binary is deliberately never loaded by this manager:
it does not know the extra roles and would silently omit their Jacobians.
"""
from __future__ import annotations

from collections import Counter
import importlib
import os
from pathlib import Path
import sys
import threading

from continuous_native_edge_optin import NativeEdgeManager, _sha
from continuous_quadratic_rig_optin import QuadraticRigCameraModel


SOURCE_DIR = Path(__file__).resolve().parent / 'native' / 'quadratic_source_edge'
DEFAULT_BINARY_DIR = SOURCE_DIR / 'bin'
MODULE_NAME = 'native_source_edge_quadratic_prod'
CAPABILITY = 'physical15_quadratic_logzoom_roles_v1'


class QuadraticNativeEdgeManager(NativeEdgeManager):
    """Register source-edge factors only with the audited quadratic binary."""

    def __init__(self, model: QuadraticRigCameraModel, states, active_keys, *,
                 threads: int = 2, cache: bool = False,
                 binary_dir: Path | str | None = None):
        if not isinstance(model, QuadraticRigCameraModel):
            raise TypeError('Quadratic native edges require QuadraticRigCameraModel')
        if int(threads) != threads or threads < 1:
            raise ValueError('Native source-edge threads must be a positive integer')
        self.model = model
        self.states = states
        self.active_keys = set(map(int, active_keys))
        self.threads = int(threads)
        self.cache = bool(cache)
        self.binary_dir = Path(binary_dir or DEFAULT_BINARY_DIR).resolve()
        self.module_name = MODULE_NAME
        self._owner_thread = threading.get_ident()
        self._records = {}
        self._fallbacks = []
        self._counts = Counter()
        self._dll_handle = None
        self._closed = False
        module_file = self.binary_dir / (MODULE_NAME + '.pyd')
        kernel_file = self.binary_dir / 'source_edge_native.dll'
        if not module_file.is_file() or not kernel_file.is_file():
            raise FileNotFoundError('Isolated quadratic native PYD/DLL not compiled')
        self.binary_sha256 = _sha(module_file)
        self.kernel_sha256 = _sha(kernel_file)
        import gtsam  # Register GTSAM Boost types before the pybind factor.
        self.gtsam = gtsam
        if sys.platform == 'win32':
            self._dll_handle = os.add_dll_directory(str(self.binary_dir))
        if str(self.binary_dir) not in sys.path:
            sys.path.insert(0, str(self.binary_dir))
        self.native = importlib.import_module(MODULE_NAME)
        if Path(self.native.__file__).resolve() != module_file.resolve():
            raise RuntimeError('Loaded quadratic PYD differs from hashed binary')
        if (not hasattr(self.native, 'quadratic_rig_capability') or
                self.native.quadratic_rig_capability() != CAPABILITY):
            raise RuntimeError('Native source-edge binary lacks quadratic rig capability')
        self.native.set_threads(self.threads)

    def _spec(self, pts, guard, factor_id):
        spec = super()._spec(pts, guard, factor_id)
        group = self.model.frames[int(pts)].camera_id
        index = {int(key): i for i, key in enumerate(self.model.dependencies(pts))}
        spec['roles']['lens_quadratic'] = index[self.model.lens_quadratic_keys[group]]
        spec['roles']['brown_quadratic'] = index[self.model.brown_quadratic_keys[group]]
        return spec

    def metadata(self) -> dict:
        result = super().metadata()
        result['schema'] = 'rinkspace.native-source-edge-quadratic-optin.v1'
        result['source_sha256'] = _sha(Path(__file__).resolve())
        result['cpp_sha256'] = {name: _sha(SOURCE_DIR / name) for name in (
            'native_source_edge_prod.cpp', 'source_edge_native_prod.cpp',
            'source_edge_native.hpp', 'brown15_projection.hpp', 'build.bat')}
        result['capability'] = self.native.quadratic_rig_capability()
        result['old_round6_binary_compatible'] = False
        return result
