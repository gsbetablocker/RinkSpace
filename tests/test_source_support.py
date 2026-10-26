"""Read real packet support and exercise wrong-address/wrong-exposure refusals."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from rinkspace.session.source_support import (
    InvalidSupport, MissingSupport, NativeFrameAlias, SourceSupportResolver,
    UnsupportedSupport,
)


REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "events/ankara_2026/programmes/okada_mei_sp/source_datasets"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf8")
    path.write_bytes(raw)
    return {"file": str(path), "sha256": sha(raw)}


@pytest.fixture
def resolver(tmp_path):
    return SourceSupportResolver(repo_root=REPO, data_root=tmp_path)


@pytest.mark.parametrize("relative,shot,frame_index", [
    ("frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014", "s01", 201),
    ("shot7_20261014", "s07", 4572),
    ("shot8_20261014", "s08", 5808),
])
def test_real_registered_packet_segment_and_raw_support_are_preserved(relative, shot, frame_index):
    manifest = json.loads((DATASETS / relative / "batch_manifest.json").read_text(encoding="utf-8-sig"))
    row, = [row for row in manifest["frames"] if row["index"] == frame_index]
    original = json.loads((DATASETS / relative / "frames" / f"{frame_index:06d}.json").read_text(encoding="utf-8-sig"))
    reference = {**row["packet"], "json_pointer": "/features/0/segments/0"}
    before = deepcopy(reference)
    pts = row.get("native_pts", row.get("source_pts"))
    expected = {
        "native_pts": pts, "time_base": "1/12800",
        "source_sha256": "e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d",
    }
    reader = SourceSupportResolver(repo_root=REPO)
    if shot == "s01":
        # This real legacy packet does not store a shot ID. Do not manufacture
        # one from its directory name; its programme scope supplies that link.
        with pytest.raises(MissingSupport, match="shot_id"):
            reader.resolve(reference, expected_binding={"shot_id": shot})
    else:
        expected["shot_id"] = shot
    result = reader.resolve(reference, expected_binding=expected)
    assert result.value == original["features"][0]["segments"][0]
    assert result.source_context["source"] == original["source"]
    assert result.source_context["user_accepted"] == original["user_accepted"]
    assert result.authentication["sha256"] == row["packet"]["sha256"]
    assert reference == before
    assert "independent_fit_weight_issued" not in result.authentication
    if shot in {"s07", "s08"}:
        assert len(result.value["source_native_readout_uv"]) > len(result.value["uv"])
        assert result.source_context["user_accepted"] is False


def test_leaf_uses_registered_manifest_pin_without_creating_trust_from_current_file(resolver, tmp_path):
    value = {"shot_id": "s02", "native_pts": 100, "rows": [{"uv": [1.5, -0.0], "user_confirmed": False}]}
    pin = write_json(tmp_path / "readout.json", value)
    leaf = {"file": pin["file"], "json_pointer": "/rows/0"}
    with pytest.raises(MissingSupport, match="SHA is missing"):
        resolver.resolve(leaf)
    resolver.register_file_pin(pin)
    result = resolver.resolve(leaf, expected_binding={"shot_id": "s02", "native_pts": 100})
    assert result.value == value["rows"][0]
    result.value["uv"][0] = 999
    assert resolver.resolve(leaf).value["uv"][0] == 1.5
    with pytest.raises(InvalidSupport, match="conflicts"):
        resolver.resolve({**leaf, "sha256": "0" * 64})


def test_pointer_escapes_missing_values_and_no_array_index_guessing(resolver, tmp_path):
    pin = write_json(tmp_path / "escaped.json", {"a/b": {"c~d": [None, {"uv": [0, 2]}]}})
    assert resolver.resolve({**pin, "json_pointer": "/a~1b/c~0d/1/uv"}).value == [0, 2]
    assert resolver.resolve({**pin, "json_pointer": "/a~1b/c~0d/0"}).value is None
    for pointer in ("/missing", "/a~1b/c~0d/2"):
        with pytest.raises(MissingSupport):
            resolver.resolve({**pin, "json_pointer": pointer})
    for pointer in ("/a~2b", "/a~1b/c~0d/01", "/a~1b/c~0d/-", "a/b"):
        with pytest.raises(InvalidSupport):
            resolver.resolve({**pin, "json_pointer": pointer})


def test_file_hash_and_native_exposure_are_verified_before_return(resolver, tmp_path):
    original = {"shot_id": "s04", "native_pts": 256, "time_base": "1/12800", "uv": [8, 9]}
    pin = write_json(tmp_path / "source.json", original)
    with pytest.raises(InvalidSupport, match="PTS|native_pts"):
        resolver.resolve(pin, expected_binding={"native_pts": 512})
    with pytest.raises(InvalidSupport, match="shot_id"):
        resolver.resolve(pin, expected_binding={"shot_id": "s05"})
    with pytest.raises(MissingSupport, match="source_sha256"):
        resolver.resolve(pin, expected_binding={"source_sha256": "e" * 64})
    resolver.resolve(pin)
    write_json(tmp_path / "source.json", {**original, "uv": [9, 9]})
    with pytest.raises(InvalidSupport, match="bytes changed"):
        resolver.resolve(pin)


@pytest.mark.parametrize("compressed", [False, True])
def test_jsonl_zero_based_physical_lines_do_not_skip_blanks(resolver, tmp_path, compressed):
    raw = b'{"native_pts":1,"uv":[1,2]}\n\n{"native_pts":2,"uv":[3,4]}\n'
    path = tmp_path / ("rows.jsonl.gz" if compressed else "rows.jsonl")
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    pin = {"file": str(path), "sha256": sha(path.read_bytes())}
    assert resolver.resolve({**pin, "jsonl_line_0based": 2}, expected_binding={"native_pts": 2}).value["uv"] == [3, 4]
    assert resolver.resolve({**pin, "jsonl_line_0based": 0}).value["uv"] == [1, 2]
    for line in (1, 3):
        with pytest.raises(MissingSupport):
            resolver.resolve({**pin, "jsonl_line_0based": line})
    with pytest.raises(InvalidSupport):
        resolver.resolve({**pin, "jsonl_line_0based": True})


def test_gzip_member_uses_its_real_byte_interval_hash_and_line(resolver, tmp_path):
    first = gzip.compress(b'{"native_pts":1}\n')
    second = gzip.compress(b'{"native_pts":2,"uv":[5,6]}\n\n{"native_pts":3,"uv":[7,8]}\n')
    path = tmp_path / "members.jsonl.gz"
    path.write_bytes(first + second)
    ref = {"file": str(path), "gzip_member_offset": len(first), "gzip_member_bytes": len(second),
           "gzip_member_sha256": sha(second), "member_jsonl_line_0based": 1}
    result = resolver.resolve(ref, expected_binding={"native_pts": 3})
    assert result.value == {"native_pts": 3, "uv": [7, 8]}
    assert result.authentication["kind"] == "gzip_member_sha256"
    with pytest.raises(InvalidSupport, match="bytes changed"):
        resolver.resolve({**ref, "gzip_member_sha256": "0" * 64})
    with pytest.raises(InvalidSupport, match="exactly one"):
        resolver.resolve({**ref, "gzip_member_offset": 0, "gzip_member_bytes": len(first + second),
                          "gzip_member_sha256": sha(first + second)})
    with pytest.raises(MissingSupport):
        resolver.resolve({**ref, "member_jsonl_line_0based": 2})
    with pytest.raises(InvalidSupport, match="Incomplete"):
        resolver.resolve({k: v for k, v in ref.items() if k != "gzip_member_bytes"})


def create_native_store(resolver, tmp_path):
    core, module = resolver._core()
    document = {"schema": "rinkfit.dense_native_board_source_frame.v1", "shot_id": "s05", "native_pts": 512,
        "source_video_sha256": "e" * 64,
        "rows": [{"shot_id": "s05", "native_pts": 512, "source_row_id": "original/s05/512/0",
                  "time_base": "1/12800", "uv": [5.25, -0.0], "user_confirmed": False,
                  "independent_fit_weight_issued": False}]}
    original = tmp_path / "original_native.json"
    original_pin = write_json(original, document)
    store_path = tmp_path / "sources.sqlite"
    logical_stream = r"D:\CodexData\RinkSpace_Data\native\opaque-source-identity"
    with module.SourceRecordStore(store_path) as store:
        info = store.import_frame_jsons(logical_stream, [original])
    reference = {"store": str(store_path), "store_content_addressed": True, "stream": logical_stream,
        "kind": "native_frames", "stream_semantic_sha256": info["semantic_sha256"],
        "frame": ["s05", 512], "frame_semantic_sha256": sha(core.canonical_record(document))}
    return original_pin, reference, document


def test_exact_sealed_frame_uses_existing_reader_and_finishes_hash_check(resolver, tmp_path):
    _, frame, original = create_native_store(resolver, tmp_path)
    db_path = Path(frame["store"])
    before = sha(db_path.read_bytes())
    result = resolver.resolve({**frame, "json_pointer": "/rows/0"}, expected_binding={
        "shot_id": "s05", "native_pts": 512, "time_base": "1/12800", "source_sha256": "e" * 64})
    assert result.value == original["rows"][0]
    assert result.authentication["reference"]["stream"] == frame["stream"]
    assert result.value["independent_fit_weight_issued"] is False
    assert sha(db_path.read_bytes()) == before
    with pytest.raises(InvalidSupport, match="actual frame|frame-semantic"):
        resolver.resolve({**frame, "frame_semantic_sha256": "0" * 64})
    with pytest.raises(MissingSupport):
        resolver.resolve({**frame, "frame": ["s05", 768]})
    with pytest.raises(UnsupportedSupport, match="exact native_frames"):
        resolver.resolve({k: v for k, v in frame.items() if k != "frame"})


def test_registered_native_alias_restores_deleted_file_without_unpacking_store(resolver, tmp_path):
    original_pin, frame, document = create_native_store(resolver, tmp_path)
    original_path = Path(original_pin["file"])
    original_path.unlink()
    leaf = {"file": str(original_path), "json_pointer": "/rows/0"}
    with pytest.raises(MissingSupport, match="no authenticated native frame alias"):
        resolver.resolve(leaf)
    resolver.register_native_frame_alias(NativeFrameAlias(str(original_path), original_pin["sha256"], frame))
    result = resolver.resolve(leaf, expected_binding={"native_pts": 512})
    assert result.value == document["rows"][0]
    assert result.authentication["explicit_native_alias_used"] is True
    assert not original_path.exists()
    fake = tmp_path / "not_an_original.json"
    resolver.register_native_frame_alias(NativeFrameAlias(str(fake), "f" * 64, frame))
    with pytest.raises(InvalidSupport, match="preserve the aliased original"):
        resolver.resolve({"file": str(fake)})


def test_unknown_formats_missing_data_root_and_root_escape_are_explicit_blockers(resolver, tmp_path):
    with pytest.raises(UnsupportedSupport, match="Unknown native alias manifest"):
        resolver.register_native_frame_alias({"invented": "private alias schema"})
    with pytest.raises(UnsupportedSupport):
        resolver.resolve({"id": "some_visual_readout"})
    with pytest.raises(InvalidSupport, match="outside explicit roots"):
        resolver.resolve({"file": str(tmp_path.parent / "elsewhere.json"), "sha256": "a" * 64})
    with pytest.raises(MissingSupport, match="Supply data_root"):
        SourceSupportResolver(repo_root=REPO).resolve({"file": "D:/CodexData/RinkSpace_Data/source.json", "sha256": "a" * 64})
    with pytest.raises((MissingSupport, InvalidSupport)):
        resolver.resolve({"file": "D:/OtherProject/source.json", "sha256": "a" * 64})


def test_native_alias_cannot_swap_two_real_frames_in_the_same_stream(resolver, tmp_path):
    first_pin, _, first = create_native_store(resolver, tmp_path)
    second = deepcopy(first)
    second["native_pts"] = 768
    second["rows"][0].update(native_pts=768, source_row_id="original/s05/768/0")
    second_pin = write_json(tmp_path / "second_native.json", second)
    core, module = resolver._core()
    path = tmp_path / "both.sqlite"
    with module.SourceRecordStore(path) as store:
        info = store.import_frame_jsons("two_native_frames", [first_pin["file"], second_pin["file"]])
    wrong = {"store": str(path), "stream": "two_native_frames", "kind": "native_frames", "store_content_addressed": True,
             "stream_semantic_sha256": info["semantic_sha256"], "frame": ["s05", 768],
             "frame_semantic_sha256": sha(core.canonical_record(second))}
    Path(first_pin["file"]).unlink()
    resolver.register_native_frame_alias(NativeFrameAlias(first_pin["file"], first_pin["sha256"], wrong))
    with pytest.raises(InvalidSupport, match="different sealed exposure"):
        resolver.resolve({"file": first_pin["file"], "json_pointer": "/rows/0"})
