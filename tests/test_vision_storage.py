"""Array evidence must remain byte exact and bound to its original source."""
from copy import deepcopy
import json
import math
import struct

import numpy as np
import pytest

from rinkspace.vision_storage import pack_source_record_arrays, hydrate_source_record_arrays, ARCHIVE_FIELD


def source_record():
    return dict(source_binding=dict(source_sha256="a"*64, source_pts=256, time_base="1/15360"),
                actual_uv=[[12.25, 34.5]], accuracy_accepted=False)


@pytest.mark.parametrize("layout", ["individual_arrays", "byte_pool"])
def test_native_bytes_dtype_nonfinite_values_and_duplicate_profiles_roundtrip(tmp_path, layout):
    record = source_record()
    values = np.arange(8192, dtype=np.float32).reshape(128, 64)
    values[0, :4] = [np.nan, np.inf, -np.inf, -0.]
    record.update(profile=values, same_profile=values, valid=(values > 10),
                  segments=np.arange(8192, dtype=np.int64))
    before = values.tobytes()
    packed = pack_source_record_arrays(record, tmp_path/"frame.npz", storage_layout=layout)
    metadata = packed[ARCHIVE_FIELD]
    assert metadata["unique_array_count"] == 2  # Boolean mask stays inline.
    assert metadata["reference_count"] == 3
    assert metadata["duplicate_array_references"] == 1
    # The JSON writer still handles the small inline arrays by its old policy.
    packed["valid"] = packed["valid"].tolist()
    serialized = json.loads(json.dumps(packed, allow_nan=False))
    restored = hydrate_source_record_arrays(serialized, tmp_path, expected_source_binding=record["source_binding"])
    assert restored["profile"].dtype == values.dtype
    assert restored["profile"].tobytes() == before
    assert restored["same_profile"] is restored["profile"]
    assert restored["segments"].tobytes() == record["segments"].tobytes()
    assert values.tobytes() == before and "_source_array_archive" not in record
    assert restored["actual_uv"] == record["actual_uv"]


def test_changed_source_and_changed_sidecar_fail_before_evidence_is_used(tmp_path):
    record = source_record();record["profile"] = np.arange(8192, dtype=np.float32)
    packed = pack_source_record_arrays(record, tmp_path/"frame.npz")
    other = {**record["source_binding"], "source_pts":512}
    with pytest.raises(ValueError, match="requested source frame"):
        hydrate_source_record_arrays(packed, tmp_path, expected_source_binding=other)
    changed = deepcopy(packed);changed[ARCHIVE_FIELD]["source_binding"]["source_pts"] = 512
    with pytest.raises(ValueError, match="different native source frames"):
        hydrate_source_record_arrays(changed, tmp_path)
    with (tmp_path/"frame.npz").open("ab") as stream:
        stream.write(b"unexpected-change")
    with pytest.raises(ValueError, match="bytes changed"):
        hydrate_source_record_arrays(packed, tmp_path)


def test_sidecar_cannot_replace_existing_source_and_inline_legacy_passes_through(tmp_path):
    record = source_record();record["profile"] = np.arange(8192, dtype=np.float32)
    pack_source_record_arrays(record, tmp_path/"frame.npz")
    with pytest.raises(FileExistsError):
        pack_source_record_arrays(record, tmp_path/"frame.npz")
    legacy = source_record()
    assert hydrate_source_record_arrays(legacy, tmp_path) is legacy


def test_noncontiguous_native_array_keeps_element_values_and_byte_order(tmp_path):
    record = source_record()
    values = np.arange(8192, dtype=">i4").reshape(64, 128)[:, ::2]
    record["profile"] = values
    packed = pack_source_record_arrays(record, tmp_path/"frame.npz", minimum_array_bytes=1)
    restored = hydrate_source_record_arrays(packed, tmp_path)
    assert restored["profile"].dtype.str == values.dtype.str
    assert restored["profile"].tobytes() == values.tobytes()


def test_scalar_native_shape_and_legacy_source_identity_are_preserved(tmp_path):
    record = source_record();record["scalar"] = np.array(-0., dtype=np.float32)
    packed = pack_source_record_arrays(record, tmp_path/"frame.npz", minimum_array_bytes=1)
    restored = hydrate_source_record_arrays(packed, tmp_path)
    assert restored["scalar"].shape == ()
    assert restored["scalar"].tobytes() == record["scalar"].tobytes()
    legacy = source_record()
    with pytest.raises(ValueError, match="requested source frame"):
        hydrate_source_record_arrays(legacy, tmp_path,
            expected_source_binding={**legacy["source_binding"], "source_pts": 512})


def test_all_array_mode_retains_empty_missing_geometry_and_flags_dtype(tmp_path):
    record = source_record()
    record.update(missing_uv=np.empty((0,2),dtype=np.float32),
                  missing_indices=np.empty(0,dtype=np.int64),valid=np.empty((1,0),dtype=np.bool_))
    packed = pack_source_record_arrays(record,tmp_path/"frame.npz",minimum_array_bytes=1)
    restored = hydrate_source_record_arrays(json.loads(json.dumps(packed)),tmp_path)
    assert packed[ARCHIVE_FIELD]["unique_array_count"] == 3
    for key in ("missing_uv","missing_indices","valid"):
        assert restored[key].dtype == record[key].dtype
        assert restored[key].shape == record[key].shape
        assert restored[key].tobytes() == record[key].tobytes()


@pytest.mark.parametrize("threshold",[1,16384])
def test_decoded_legacy_nonfinite_python_float_sentinels_remain_distinct_from_none(tmp_path,threshold):
    record = source_record()
    nan = struct.unpack("d", bytes.fromhex("010000000000f87f"))[0]
    record["old_inline_candidates"] = [nan, float("inf"), -float("inf"), None, nan]
    packed = pack_source_record_arrays(record, tmp_path/"legacy.npz", minimum_array_bytes=threshold)
    # This is valid finite JSON; no fabricated former tensor dtype is stored.
    serialized = json.loads(json.dumps(packed, allow_nan=False))
    restored = hydrate_source_record_arrays(serialized, tmp_path)
    values = restored["old_inline_candidates"]
    assert [type(value) for value in values] == [float,float,float,type(None),float]
    assert math.isnan(values[0]) and values[1] == float("inf") and values[2] == -float("inf")
    assert values[3] is None
    assert struct.pack("d", values[0]) == struct.pack("d", nan)
    assert struct.pack("d", values[4]) == struct.pack("d", nan)
    assert packed[ARCHIVE_FIELD]["unique_array_count"] == 3
    assert packed[ARCHIVE_FIELD]["reference_count"] == 4
    malformed = deepcopy(packed)
    malformed["old_inline_candidates"][0]["extra"] = True
    with pytest.raises(ValueError, match="Python float evidence reference"):
        hydrate_source_record_arrays(malformed,tmp_path)


def test_byte_pool_many_source_runs_empty_scalars_endianness_and_manifest_bounds(tmp_path):
    record = source_record()
    payload_nan = np.array([0x7fc00017], dtype=np.uint32).view(np.float32)
    record.update(runs=[np.array([i, i+1], dtype=np.int64) for i in range(1200)],
                  original_uv=np.array([[4.25, -0.]], dtype=">f8"),
                  empty=np.empty((0, 2), dtype=np.float32), scalar=np.array(-0., dtype=np.float32),
                  nonfinite=payload_nan, duplicate=payload_nan)
    packed=pack_source_record_arrays(record,tmp_path/"pool.npz",minimum_array_bytes=1,storage_layout="byte_pool")
    with np.load(tmp_path/"pool.npz",allow_pickle=False) as archive:
        assert len(archive.files)==1
    restored=hydrate_source_record_arrays(json.loads(json.dumps(packed,allow_nan=False)),tmp_path)
    for old,new in zip(record['runs'],restored['runs']):
        assert new.dtype==old.dtype and new.shape==old.shape and new.tobytes()==old.tobytes()
    for key in ('original_uv','empty','scalar','nonfinite'):
        assert restored[key].dtype==record[key].dtype
        assert restored[key].shape==record[key].shape
        assert restored[key].tobytes()==record[key].tobytes()
    assert restored['nonfinite'] is restored['duplicate']
    bad=deepcopy(packed);next(iter(bad[ARCHIVE_FIELD]['arrays'].values()))['byte_offset']=1
    with pytest.raises(ValueError,match='intervals changed'):
        hydrate_source_record_arrays(bad,tmp_path)
    bad=deepcopy(packed);bad['source_binding']['source_pts']=512
    with pytest.raises(ValueError,match='different native source frames'):
        hydrate_source_record_arrays(bad,tmp_path)
