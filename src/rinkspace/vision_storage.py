"""Lossless, source-bound array sidecars for native visual evidence records.

Small coordinates and metadata stay in the ordinary review JSON. Large arrays
retain their dtype, shape and exact bytes, including NaNs, rather than becoming
millions of Python floats. Legacy records with inline lists remain readable.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
import zipfile

import numpy as np

ARCHIVE_FIELD = "_source_array_archive"
ARRAY_FIELD = "_source_array_key"
PYTHON_FLOAT_FIELD = "_source_nonfinite_python_float"
SCHEMA = "rinkspace.source-array-archive.v1"
BYTE_POOL_SCHEMA = "rinkspace.source-array-archive.byte-pool.v1"
BYTE_POOL_MEMBER = "__source_array_bytes"


def _binding_key(binding):
    digest = binding.get("source_sha256", binding.get("source_hash"))
    pts = binding.get("source_pts")
    clock = Fraction(str(binding.get("time_base", "0")))
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdef" for c in digest.lower()) or
            isinstance(pts, bool) or not isinstance(pts, int) or pts < 0 or clock <= 0):
        raise ValueError("Array evidence requires the actual source SHA, native PTS and clock")
    return digest.lower(), pts, str(clock)


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_sha(array):
    header = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
    digest = hashlib.sha256(header)
    if array.nbytes:
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def pack_source_record_arrays(record, sidecar_path, *, minimum_array_bytes=16384,
                              storage_layout="individual_arrays"):
    """Return a new record; write large original arrays to one atomic NPZ.

    Call before JSON conversion, while arrays still have their original dtype.
    The original record and every array remain untouched. Equal arrays share
    a member, including repeated material profiles in different diagnostics.
    """
    if not isinstance(record, Mapping) or ARCHIVE_FIELD in record:
        raise ValueError("Pack an original source record once, before JSON conversion")
    binding = record["source_binding"]
    _binding_key(binding)
    if isinstance(minimum_array_bytes, bool) or minimum_array_bytes < 1:
        raise ValueError("The array storage threshold must be a positive byte count")
    if storage_layout not in ("individual_arrays", "byte_pool"):
        raise ValueError("Unknown lossless source array storage layout")
    path = Path(sidecar_path)
    members, manifests, by_hash = {}, {}, {}
    reference_count = 0

    def visit(value, *, force_array=False):
        nonlocal reference_count
        # Legacy JSON may already contain Python NaN/inf. Its Python type is
        # known; its former tensor dtype is not. Preserve this missing-value
        # sentinel explicitly before the finite-JSON writer replaces it.
        if type(value) is float and not math.isfinite(value):
            reference = visit(np.array(value, dtype=np.float64),force_array=True)
            if isinstance(reference, Mapping) and ARRAY_FIELD in reference:
                return {PYTHON_FLOAT_FIELD: reference[ARRAY_FIELD]}
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            preserve_empty = minimum_array_bytes == 1 and value.size == 0
            if (value.nbytes < minimum_array_bytes and not preserve_empty and not force_array) or value.dtype.hasobject:
                return value
            array = value if value.flags.c_contiguous else np.ascontiguousarray(value)
            digest = _array_sha(array)
            key = by_hash.get(digest)
            if key is None:
                key = f"array_{len(members):05d}"
                members[key], by_hash[digest] = array, key
                manifests[key] = dict(dtype=array.dtype.str, shape=list(array.shape),
                                      raw_array_sha256=digest, bytes=array.nbytes)
            reference_count += 1
            return {ARRAY_FIELD: key}
        if isinstance(value, Mapping):
            if ARRAY_FIELD in value or ARCHIVE_FIELD in value or PYTHON_FLOAT_FIELD in value:
                raise ValueError("Original data cannot impersonate a source-array reference")
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        return value

    packed = visit(record)
    if not members:
        return packed
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("An existing source evidence sidecar must not be overwritten")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".source-arrays.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            if storage_layout == "byte_pool":
                # A record can contain tens of thousands of short measured runs.
                # One streamed uint8 NPY avoids one ZIP/NPY entry per tiny array,
                # without changing any original array's bytes or numeric dtype.
                total_bytes = sum(array.nbytes for array in members.values())
                offset = 0
                with archive.open(f"{BYTE_POOL_MEMBER}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array_header_1_0(member, dict(
                        descr=np.dtype(np.uint8).str, fortran_order=False, shape=(total_bytes,)))
                    for key, array in members.items():
                        manifests[key]["byte_offset"] = offset
                        if array.nbytes:
                            member.write(memoryview(array).cast("B"))
                        offset += array.nbytes
            else:
                for key, array in members.items():
                    with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
        checksum = _file_sha(temporary)
        size = temporary.stat().st_size
        # Publish atomically without replacing another writer's source file.
        # Both names are on the same local Data volume.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    packed[ARCHIVE_FIELD] = dict(schema=BYTE_POOL_SCHEMA if storage_layout == "byte_pool" else SCHEMA,
        storage_layout=storage_layout, filename=path.name, file_sha256=checksum,
        compressed_bytes=size, source_binding=dict(binding), arrays=manifests,
        unique_array_count=len(members), reference_count=reference_count,
        duplicate_array_references=reference_count-len(members),
        original_dtype_and_bytes_preserved=True, source_evidence_discarded=False)
    return packed


def hydrate_source_record_arrays(record, record_directory, *, expected_source_binding=None):
    """Restore validated arrays; old inline-only records pass through unchanged."""
    if ARCHIVE_FIELD not in record:
        if expected_source_binding is not None and _binding_key(record["source_binding"]) != _binding_key(expected_source_binding):
            raise ValueError("Array record differs from the requested source frame")
        return record
    metadata = record[ARCHIVE_FIELD]
    if metadata.get("schema") not in (SCHEMA, BYTE_POOL_SCHEMA):
        raise ValueError("Unknown source array sidecar schema")
    binding = _binding_key(record["source_binding"])
    if _binding_key(metadata["source_binding"]) != binding:
        raise ValueError("Array sidecar and review record belong to different native source frames")
    if expected_source_binding is not None and _binding_key(expected_source_binding) != binding:
        raise ValueError("Array record differs from the requested source frame")
    filename = metadata["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("A source array sidecar must be beside its review record")
    path = Path(record_directory) / filename
    if _file_sha(path) != metadata["file_sha256"]:
        raise ValueError("Source array sidecar bytes changed")
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        pooled = metadata["schema"] == BYTE_POOL_SCHEMA
        expected_members = {BYTE_POOL_MEMBER} if pooled else set(metadata["arrays"])
        if set(archive.files) != expected_members:
            raise ValueError("Source array member inventory differs from its recorded manifest")
        byte_pool = archive[BYTE_POOL_MEMBER] if pooled else None
        if pooled and (byte_pool.dtype != np.dtype(np.uint8) or byte_pool.ndim != 1 or
                       not byte_pool.flags.c_contiguous):
            raise ValueError("Source array byte pool has an invalid dtype or domain")
        expected_offset = 0
        for key, manifest in metadata["arrays"].items():
            if pooled:
                offset, length = manifest.get("byte_offset"), manifest["bytes"]
                if (type(offset) is not int or type(length) is not int or offset != expected_offset or
                        length < 0 or offset + length > byte_pool.nbytes):
                    raise ValueError("Source array byte pool intervals changed")
                dtype = np.dtype(manifest["dtype"])
                if dtype.hasobject or dtype.itemsize == 0 or length % dtype.itemsize:
                    raise ValueError("Source array byte pool dtype is invalid")
                array = np.frombuffer(byte_pool, dtype=dtype, count=length // dtype.itemsize,
                                      offset=offset).reshape(manifest["shape"])
                expected_offset += length
            else:
                array = archive[key]
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            if (array.dtype.str != manifest["dtype"] or list(array.shape) != manifest["shape"] or
                    array.nbytes != manifest["bytes"] or _array_sha(array) != manifest["raw_array_sha256"]):
                raise ValueError("Source array dtype, shape or exact values changed")
            arrays[key] = array
        if pooled and expected_offset != byte_pool.nbytes:
            raise ValueError("Source array byte pool contains unrecorded evidence bytes")

    def restore(value):
        if isinstance(value, Mapping):
            if PYTHON_FLOAT_FIELD in value:
                key = value[PYTHON_FLOAT_FIELD]
                if len(value) != 1 or key not in arrays:
                    raise ValueError("Invalid or absent Python float evidence reference")
                array = arrays[key]
                if array.shape != () or array.dtype != np.dtype(np.float64) or np.isfinite(array):
                    raise ValueError("Python float evidence has a different type or value")
                return float(array)
            if ARRAY_FIELD in value:
                if len(value) != 1 or value[ARRAY_FIELD] not in arrays:
                    raise ValueError("Invalid or absent source array reference")
                return arrays[value[ARRAY_FIELD]]
            return {key: restore(item) for key, item in value.items() if key != ARCHIVE_FIELD}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    return restore(record)
