"""Current dataset persistence. No historical producers or numerical models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


class DatasetError(ValueError):
    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code, self.message, self.context = code, message, context or {}


def require(condition, message, **context):
    if not condition:
        raise DatasetError("invalid_input", message, context)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(root, value):
    """Manifest references are bounded; registered external assets use AssetStore."""
    root = Path(root).resolve()
    value = Path(value)
    require(not value.is_absolute(), "Manifest reference must be relative", path=str(value))
    path = (root / value).resolve()
    require(path.is_relative_to(root), "Manifest reference escapes its asset root", path=str(value))
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_array(root, name, values, dtype=None):
    root = Path(root)
    array = np.asarray(values, dtype=dtype)
    require(not array.dtype.hasobject, "Pickled/object arrays are forbidden", name=name)
    if array.dtype.kind in "iufc":
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array = np.ascontiguousarray(array)
    require(array.dtype.kind not in "fc" or np.isfinite(array).all(), "Array contains nonfinite values", name=name)
    path = root / "arrays" / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {"path": path.relative_to(root).as_posix(), "dtype": array.dtype.str,
            "shape": list(array.shape), "sha256": digest(path)}


def load_array(root, desc, *, verify=False):
    path = relative(root, desc["path"])
    if verify:
        require(digest(path) == desc["sha256"], "Array content differs from current revision", path=str(path))
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(not array.dtype.hasobject and array.flags.c_contiguous, "Array must be C contiguous without objects", path=str(path))
    require(array.dtype.str == desc["dtype"] and list(array.shape) == desc["shape"],
            "Array shape/dtype differs from manifest", path=str(path))
    return array


@dataclass(frozen=True)
class PreparedReceipt:
    manifest_path: str
    dataset_id: str
    revision: str
    selected_shot_ids: tuple[str, ...]
    scope_status: str


def iter_current_observations(path, features, feature_indices, *, mask=None):
    """Stream current row semantics and preserve the exact selected JSON lines.

    This checks declarations against current numeric bindings. It does not
    inspect provenance files, source frames, historical receipts or producers.
    """
    from ..domain.measurement import validate_observation_binding
    count = 0
    with Path(path).open(encoding='utf-8-sig') as stream:
        for row, line in enumerate(stream):
            require(row < len(feature_indices), 'Observation metadata has extra rows')
            count += 1
            if mask is not None and not mask[row]:
                continue
            index = int(feature_indices[row])
            require(0 <= index < len(features), 'Row feature index is outside current registry', source_row=row)
            validate_observation_binding(json.loads(line), features[index], row)
            yield line
    require(count == len(feature_indices), 'Observation metadata row count differs from arrays')


def load_prepared(prepared_dir, *, purpose="fit"):
    """Current-format structural load; does not rehash historical assets."""
    root = Path(prepared_dir).resolve()
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == "rinkspace.prepared.v1", "Unknown prepared format")
    arrays = {key: load_array(root, desc) for key, desc in manifest["arrays"].items()}
    require(purpose in {"fit", "review"}, "Unknown prepared consumer purpose")
    if purpose == "fit":
        from .prepare import validate_current_feature_contracts
        scene = read_json(relative(root, manifest["scene_ref"]))
        validate_current_feature_contracts(manifest, scene, manifest["features"], arrays, np.arange(len(arrays["pts"])))
        if manifest.get('observations_ref'):
            for _ in iter_current_observations(relative(root, manifest['observations_ref']),
                                               manifest['features'], arrays['feature_index']):
                pass
    return manifest, arrays
