"""Lossless shared source evidence in NPZ, including large row inventories.

NumPy's fixed-width Unicode scalar has a size limit and allocates four bytes
per character. Exact repeated evidence is held once, with UTF-8 storage; this
changes storage only, never pixel identities or source admission decisions.
"""
from __future__ import annotations

import hashlib
from itertools import zip_longest
import json
import math

import numpy as np


ENCODING = 'rinkspace.exact-shared-source-json.v1'
REFERENCE_KEY = '__rinkspace_exact_source_json_ref__'


class _ContentCollisionBucket(tuple):
    """Separate a digest collision bucket from a canonical JSON list."""


class ContentExactEvidenceCanonicalizer:
    """Share literal source evidence without an encoded mirror or JSON keys.

    The digest index retains only the actual canonical objects. A digest is
    never an equality decision: canonical JSON streams must also match, so
    numeric types, signed zero and deliberately colliding hashes stay exact.
    As with ExactEvidenceInterner.canonicalize, shared dicts/lists are immutable
    by the caller's contract; ordinary JSON containers remain serializable.
    This class is an in-memory writer detail, not a new disk encoding.
    """

    def __init__(self):
        self.lookup = {}
        self._encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(',', ':'))

    def _digest(self, value):
        digest = hashlib.sha256()
        for chunk in self._encoder.iterencode(value):
            digest.update(chunk.encode('utf8'))
        return digest.digest()

    def _literal_equal(self, first, second):
        # Python equality treats True == 1 == 1.0 and -0.0 == 0.0. The source
        # JSON equality used by the existing interner distinguishes all four.
        sentinel = object()
        return all(left == right for left, right in zip_longest(
            self._encoder.iterencode(first), self._encoder.iterencode(second),
            fillvalue=sentinel))

    def canonicalize(self, value):
        if isinstance(value, dict):
            if REFERENCE_KEY in value:
                raise ValueError('Reserved compact source JSON reference key')
            # _literal() supplies string keys in the writer. Reject other
            # domains explicitly rather than stringify competing key types.
            if any(type(key) is not str for key in value):
                raise TypeError('Source JSON object keys must be strings')
        elif isinstance(value, list):
            if all(type(item) in (str, int, float, bool, type(None)) for item in value):
                # UV pairs and primitive index lists need only their real
                # container, not a second container and one cache entry each.
                return [self.canonicalize(item) for item in value]
        elif isinstance(value, str):
            if len(value) < 128:
                return value
        elif type(value) in (int, bool, type(None)):
            return value
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError('Out of range float values are not JSON compliant')
            return value
        else:
            raise TypeError('Literal JSON source evidence required')

        key = self._digest(value)
        previous = self.lookup.get(key)
        if previous is not None:
            candidates = previous if isinstance(previous, _ContentCollisionBucket) else (previous,)
            for candidate in candidates:
                if self._literal_equal(candidate, value):
                    return candidate
        if isinstance(value, dict):
            actual = {key:self.canonicalize(item) for key,item in value.items()}
        elif isinstance(value, list):
            actual = [self.canonicalize(item) for item in value]
        else:
            actual = value
        if previous is None:
            self.lookup[key] = actual
        elif isinstance(previous, _ContentCollisionBucket):
            self.lookup[key] = _ContentCollisionBucket((*previous, actual))
        else:
            self.lookup[key] = _ContentCollisionBucket((previous, actual))
        return actual


class ExactEvidenceInterner:
    def __init__(self):
        self.values = []
        self.objects = []
        self.lookup = {}

    def _encode(self, value):
        if isinstance(value, dict):
            if REFERENCE_KEY in value:
                raise ValueError('Reserved compact source JSON reference key')
            encoded = {key:self._encode(item) for key,item in value.items()}
            actual = {key:self._resolve(item) for key,item in encoded.items()}
        elif isinstance(value, list):
            encoded = [self._encode(item) for item in value]
            actual = [self._resolve(item) for item in encoded]
        elif isinstance(value, str) and len(value) >= 128:
            encoded = actual = value
        else:
            return value
        # Encoded children are small references, so equality is exact without
        # serializing a repeated large source annotation for every segment.
        key = json.dumps(encoded, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(',', ':'))
        index = self.lookup.get(key)
        if index is None:
            index = len(self.values)
            self.lookup[key] = index
            self.values.append(encoded)
            self.objects.append(actual)
        return {REFERENCE_KEY:index}

    def _resolve(self, encoded):
        if isinstance(encoded, dict) and set(encoded) == {REFERENCE_KEY}:
            return self.objects[encoded[REFERENCE_KEY]]
        return encoded

    def canonicalize(self, value):
        """Share exact nested evidence in memory; callers keep it immutable."""
        return self._resolve(self._encode(value))

    def pack(self, value):
        root = self._encode(value)
        envelope = dict(encoding=ENCODING, values=self.values, root=root)
        payload = json.dumps(envelope, ensure_ascii=False, allow_nan=False,
            separators=(',', ':')).encode('utf8')
        return np.frombuffer(payload, dtype=np.uint8)


def pack_source_json(value):
    return ExactEvidenceInterner().pack(value)


def unpack_source_json(array):
    """Read both immutable earlier scalar JSON and exact shared UTF-8 JSON."""
    array = np.asarray(array)
    if array.ndim == 0 and array.dtype.kind in 'US':
        payload = array.item()
        return json.loads(payload.decode('utf8') if isinstance(payload, bytes) else payload)
    if array.ndim != 1 or array.dtype != np.uint8:
        raise ValueError('Exact UTF-8 source JSON byte array required')
    envelope = json.loads(array.tobytes().decode('utf8'))
    if not isinstance(envelope, dict) or envelope.get('encoding') != ENCODING:
        raise ValueError('Unknown compact source evidence encoding')
    values = envelope['values']
    if not isinstance(values, list):
        raise ValueError('Exact shared source evidence table required')
    resolved = []

    def resolve(value):
        if isinstance(value, dict):
            if REFERENCE_KEY in value:
                index = value[REFERENCE_KEY]
                if (set(value) != {REFERENCE_KEY} or type(index) is not int
                        or not 0 <= index < len(resolved)):
                    raise ValueError('Forward or invalid shared source evidence reference')
                return resolved[index]
            return {key:resolve(item) for key,item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    for value in values:
        resolved.append(resolve(value))
    return resolve(envelope['root'])
