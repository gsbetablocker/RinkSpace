"""Typed, read-only Data arrays. No historical source chain or solver import."""
from __future__ import annotations
from pathlib import Path
import threading
import numpy as np


class ArrayInputError(ValueError):
    def __init__(self, array_id, field, message):
        self.array_id, self.field = array_id, field
        super().__init__(f"array {array_id}.{field}: {message}")


def _value(value):
    return getattr(value, "value", value)


class ArrayStore:
    """Headers and current layout are checked; original evidence stays immutable.

    File metadata invalidates this process-local cache. Delivery hashes remain in
    the current descriptor; opening unchanged arrays never replays old provenance.
    Parameter updates use explicit copies and are saved as a new atomic state.
    """
    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self._cache = {}
        self._lock = threading.RLock()

    def _path(self, ref):
        path = Path(ref.path).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise ArrayInputError(ref.id, "path", "large arrays must stay under the configured Data root") from exc
        if not path.is_file():
            raise ArrayInputError(ref.id, "path", "current array file is missing")
        return path

    def open(self, ref):
        path = self._path(ref)
        try:
            dtype = np.dtype(ref.dtype)
        except (TypeError, ValueError) as exc:
            raise ArrayInputError(ref.id, "dtype", "invalid dtype declaration") from exc
        if dtype.hasobject or dtype.kind not in "biufSU":
            raise ArrayInputError(ref.id, "dtype", "object, pickle and implicit structured arrays are forbidden")
        if any(type(n) is not int or n < 0 for n in ref.shape):
            raise ArrayInputError(ref.id, "shape", "nonnegative integer dimensions required")
        if type(ref.generation) is not int or ref.generation < 0:
            raise ArrayInputError(ref.id, "generation", "nonnegative explicit generation required")
        storage = ref.storage.lower()
        if storage == "npz" and not ref.member:
            raise ArrayInputError(ref.id, "member", "npz requires the exact member name")
        if storage != "npz" and ref.member is not None:
            raise ArrayInputError(ref.id, "member", "member only applies to npz")
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns, storage, ref.member, dtype.str, tuple(ref.shape), ref.generation)
        with self._lock:
            cached = self._cache.get(ref.id)
            if cached and cached[0] == identity:
                return cached[1]
            try:
                if storage == "npy":
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                elif storage == "npz":
                    with np.load(path, allow_pickle=False) as packed:
                        if ref.member not in packed.files:
                            raise ArrayInputError(ref.id, "member", "declared member is absent")
                        array = packed[ref.member]
                elif storage == "raw":
                    expected = int(np.prod(ref.shape, dtype=object)) * dtype.itemsize
                    if stat.st_size != expected:
                        raise ArrayInputError(ref.id, "shape", f"raw bytes {stat.st_size} != declared {expected}")
                    array = np.memmap(path, dtype=dtype, mode="r", shape=ref.shape)
                else:
                    raise ArrayInputError(ref.id, "storage", "only explicit npy/npz/raw formats are supported")
            except (OSError, ValueError, TypeError) as exc:
                if isinstance(exc, ArrayInputError):
                    raise
                raise ArrayInputError(ref.id, "storage", str(exc)) from exc
            if array.dtype != dtype:
                raise ArrayInputError(ref.id, "dtype", f"actual {array.dtype.str} != declared {dtype.str}; no reinterpret cast")
            if tuple(array.shape) != tuple(ref.shape):
                raise ArrayInputError(ref.id, "shape", f"actual {array.shape} != declared {tuple(ref.shape)}")
            array.flags.writeable = False
            self._cache[ref.id] = (identity, array)
            return array

    def working_copy(self, ref):
        role = _value(ref.role)
        if role not in {"parameter_state", "auxiliary_state", "compute_cache"}:
            raise ArrayInputError(ref.id, "role", "raw/derived evidence and semantic foreign keys cannot become mutable optimizer state")
        return np.array(self.open(ref), copy=True, order="C")

    def validate_binding(self, ref, binding):
        if binding.array_id != ref.id:
            raise ArrayInputError(ref.id, "binding", "binding references another array")
        array = self.open(ref)
        if binding.generation != ref.generation:
            raise ArrayInputError(ref.id, "generation", "stale CPU/GPU buffer generation")
        if np.dtype(binding.dtype) != array.dtype or tuple(binding.shape) != array.shape:
            raise ArrayInputError(ref.id, "binding", "CPU/GPU dtype or shape differs from the current array")
        if tuple(binding.byte_strides) != array.strides:
            raise ArrayInputError(ref.id, "byte_strides", "CPU/GPU byte-stride declaration differs from the selected array")
        return array

    def close(self):
        """Release cache references after compute consumers release their array views.

        Do not force-close a memmap still in use. New states use new asset paths
        and generations; mapped current input files are never overwritten.
        """
        with self._lock:
            self._cache.clear()

    def invalidate(self, array_ids):
        with self._lock:
            for identity in array_ids:
                self._cache.pop(identity, None)
