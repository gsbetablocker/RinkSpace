"""Read safe historical source metadata without constructing a legacy owner.

Only the complete, authenticated JSON/NPZ source snapshot is supported. Frames
are decoded one at a time; camera states, residuals and mutable caches are never
loaded. The NPZ archive is read once into memory to avoid hundreds of thousands
of small filesystem reads when pointwise source rows share one archive.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np

from .data import sha256_file


def read_reference(ref):
    path = Path(ref['path'])
    if sha256_file(path) != ref['sha256']:
        raise ValueError('Frozen source reference changed: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def stream_snapshot_frames(path):
    """Yield encoded objects from the first ``payload.frames`` JSON array.

    The prefix is deliberately checked. This cannot accidentally interpret a
    later diagnostic frame list as the authenticated source inventory.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as handle:
        buffer = handle.read(65536)
        if ('"schema": "rinkspace.complete-authenticated-native-source-snapshot.v1"' not in buffer
                or '"payload"' not in buffer):
            raise ValueError('Complete safe source snapshot required')
        marker = '"frames": ['
        offset = buffer.find(marker)
        if offset < 0:
            raise ValueError('Snapshot payload.frames must precede other payload fields')
        buffer = buffer[offset + len(marker):]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(','):
                buffer = buffer[1:].lstrip()
            if buffer.startswith(']'):
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    more = handle.read(262144)
                    if not more:
                        raise ValueError('Incomplete source snapshot frame array')
                    buffer += more
                    buffer = buffer.lstrip()
                    if buffer.startswith(','):
                        buffer = buffer[1:].lstrip()
            if not isinstance(value, dict) or not {'rows', 'whitening_groups', 'native_pts', 'index'} <= set(value):
                raise ValueError('Invalid complete source frame object')
            yield value
            buffer = buffer[end:]


class SnapshotArrays:
    """A no-pickle, bounded immutable source-array reader."""
    def __init__(self, path, expected_sha256=None):
        path = Path(path)
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise ValueError('Complete source array archive changed')
        self._memory = BytesIO(path.read_bytes())
        self._saved = zipfile.ZipFile(self._memory)

    def decode(self, value):
        if isinstance(value, dict):
            if set(value) == {'_safe_npz_array'}:
                key = value['_safe_npz_array'] + '.npy'
                # NameToInfo is a dictionary. Avoid NpzFile's version-dependent
                # linear scans over the 386011 entry source archive.
                info = self._saved.getinfo(key)
                with self._saved.open(info) as handle:
                    result = np.lib.format.read_array(handle, allow_pickle=False)
                if result.dtype.hasobject:
                    raise ValueError('Object source arrays are forbidden')
                result.setflags(write=False)
                return result
            if set(value) == {'_safe_tuple'}:
                return tuple(self.decode(item) for item in value['_safe_tuple'])
            if set(value) == {'_safe_whitening_operator'}:
                # Return plain portable data rather than importing old classes.
                return self.decode(value['_safe_whitening_operator'])
            return {key: self.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        return value

    def close(self):
        self._saved.close()
        self._memory.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
