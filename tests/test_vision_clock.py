"""Per-source clock evidence, native neighbours and explicit proof failures."""
from fractions import Fraction
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rinkspace import vision


SOURCE = "a" * 64


def test_cut_first_frame_context_padding_cannot_advance_training_memory():
    role = vision.source_previous_observation(dict(native_pts=[15360, 15360, 15616]))
    assert not role["available"]
    assert role["reason"] == "no_previous_source_observation_context_padding"


def test_actual_native_previous_is_preserved_even_with_identical_RGB():
    # Memory roles use source identity, not pixel similarity or FPS.
    assert vision.source_previous_observation(dict(native_pts=[15104, 15360, 15616]))["available"]


def test_distinct_native_aliases_require_actual_source_clock_evidence():
    row = dict(native_pts=[15104, 15360, 15616], canonical_exposure_pts=[15104, 15104, 15616],
               source_video_sha256=SOURCE, time_base="1/15360")
    with pytest.raises(ValueError, match="explicit source-clock proof"):
        vision.source_previous_observation(row)
    row.update(official_exposure_proof=dict(proof_sha256="c"*64, source_sha256=SOURCE, time_base="1/15360"),
               official_exposure_records=[dict(canonical=15104, shot=7), dict(canonical=15104, shot=7), None])
    assert not vision.source_previous_observation(row)["available"]
    row["official_exposure_proof"]["source_sha256"] = "b"*64
    with pytest.raises(ValueError, match="explicit source-clock proof"):
        vision.source_previous_observation(row)


def test_sequence_previous_cannot_come_after_its_centre():
    with pytest.raises(ValueError, match="previous at or before centre"):
        vision.source_previous_observation(dict(native_pts=[15616, 15360, 15872]))


def proof_registry(tmp_path):
    groups = [dict(full_pts=[0, 256, 512], shot_index=0, confirmed_official_pts=0,
                   confirmed_pair_evidence=[{}, {}], frame_accepted_official_pts=[0, 0, 0],
                   frame_match_status=["confirmed"] * 3),
              dict(full_pts=[768], shot_index=0, confirmed_official_pts=None,
                   confirmed_pair_evidence=[], frame_accepted_official_pts=[256],
                   frame_match_status=["accepted"])]
    proof = dict(schema="rinkspace.official50-confirmed-exposure-groups.v1",
                 source_binding=dict(full60=dict(sha256=SOURCE, native_pts_time_base="1/12800"),
                                     official50=dict(sha256="b" * 64, native_pts_time_base="1/12800")),
                 validation=dict(all_groups_single_shot=True, all_nine_cut_pairs_different_groups=True,
                                 confirmed_pair_count=2), groups=groups)
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")
    return dict(sources=[dict(source_sha256=SOURCE, time_base="1/12800", exposure_proof=dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), schema=proof["schema"]))])


def test_another_source_does_not_open_unrelated_configured_proof(tmp_path):
    registry = dict(sources=[dict(source_sha256=SOURCE, time_base="1/15360",
                                 exposure_proof=dict(path=str(tmp_path / "missing.json"), sha256="c" * 64))])
    mapping, metadata = vision.confirmed_exposure_groups("b" * 64, "1/12800", registry=registry)
    assert mapping == {}
    assert metadata["proof_sha256"] is None
    assert metadata["status"] == "native_frames_without_verified_exposure_aliases"


def test_known_source_clock_cannot_be_replaced_with_sample_clock():
    registry = dict(sources=[dict(source_sha256=SOURCE, time_base="1/12800", exposure_proof=None)])
    with pytest.raises(ValueError, match="registered original video"):
        vision.confirmed_exposure_groups(SOURCE, "1/15360", registry=registry)
    assert vision.confirmed_exposure_groups(SOURCE, "2/25600", registry=registry)[0] == {}


def test_explicit_changed_proof_fails_instead_of_native_fallback(tmp_path):
    registry = proof_registry(tmp_path)
    mapping, metadata = vision.confirmed_exposure_groups(SOURCE, "1/12800", registry=registry)
    assert mapping[512]["canonical"] == 0
    assert metadata["confirmed_repeat_pairs"] == 2
    path = registry["sources"][0]["exposure_proof"]["path"]
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="exposure proof changed"):
        vision.confirmed_exposure_groups(SOURCE, "1/12800", registry=registry)


def fake_native_video(monkeypatch):
    rgb = np.zeros((3, 4, 3), np.uint8)

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 12800))])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, *args, **kwargs):
            pass

        def decode(self, stream):
            for pts in (0, 256, 512, 768):
                yield SimpleNamespace(pts=pts, to_ndarray=lambda **kwargs: rgb.copy())

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *args: Container()))
    monkeypatch.setattr(vision, "_image", lambda *args: rgb.copy())
    return dict(source_hash=SOURCE, source_pts=512, time_base="1/12800", frame_path="source.png")


def test_equal_RGB_preserves_all_unproven_native_frames(monkeypatch):
    sample = fake_native_video(monkeypatch)
    sequence, metadata = vision.frame_neighbours("source.mp4", sample)
    assert sequence.shape == (3, 3, 4, 3)
    assert metadata["native_pts"] == [256, 512, 768]
    assert metadata["display_repeat_groups"] == [[0], [256], [512], [768]]
    assert not metadata["equal_RGB_neighbour_pairs_are_exposure_proof"]


def test_three_frame_proven_group_uses_containing_group_not_nearest_start(monkeypatch, tmp_path):
    sample = fake_native_video(monkeypatch)
    _, metadata = vision.frame_neighbours("source.mp4", sample, exposure_registry=proof_registry(tmp_path))
    assert metadata["display_repeat_groups"] == [[0, 256, 512], [768]]
    assert metadata["native_pts"] == [0, 512, 768]
    assert metadata["canonical_exposure_pts"] == [0, 0, 768]


def test_configured_broken_proof_is_checked_before_warm_sequence_cache(tmp_path, monkeypatch):
    registry = proof_registry(tmp_path)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"actual source identity")
    source_sha = vision.sha256(source_video)
    registry["sources"][0]["source_sha256"] = source_sha
    # The proof now belongs to a different source: an old cache must not make
    # this mismatch disappear before any cached RGB bundle is consulted.
    with pytest.raises(ValueError, match="another original source"):
        vision.prepare_sequences(dict(source_video=str(source_video), source_clock_registry=registry),
                                 tmp_path, dict(samples=[dict(source_hash=source_sha, time_base="1/12800")]))
