"""RinkSpace source review keeps a mistracked extremity within view."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path


def test_collapsed_current_arm_does_not_crop_out_visible_wrist():
    tool = Path(__file__).resolve().parents[1] / (
        'events/ankara_2026/programmes/okada_mei_sp/tools/identity_refit/source_review.py')
    spec = importlib.util.spec_from_file_location('okada_source_review', tool)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)

    pixels = np.full((5, 17, 2), [420., 165.])
    pixels[:, 11:13, 1] = 245.
    pixels[:, 13:17, 1] = 320.
    pixels[1:4:2, 9, 0] = 170.
    pixels[[0, 4], 9, 0] = 20.  # unrelated views across physical cuts
    old = dict(source_pts=256*np.arange(5), source_shot_ids=np.array(['s04', 's05', 's05', 's05', 's06']),
               source_uv_native=pixels, source_observation_use=np.ones((5, 17), bool),
               source_body_model_joint_ids=np.broadcast_to(np.arange(17), (5, 17)))
    assert review.crop_bounds(pixels[2, 5:], (440, 700, 3), 22)[0] > 170
    left, _, right, _ = review.review_bounds(old, 2, (440, 700, 3))
    assert 20 < left < 170 < right

    class Source:
        def read(self, pts):
            assert pts == 512
            rgb = np.zeros((440, 700, 3), dtype=np.uint8)
            rgb[:, :90] = [240, 30, 10]  # outside the crop, in the original view
            return rgb

    panel = review.panel(Source(), old, 2, [9])
    assert np.any(np.all(panel == [10, 30, 240], axis=-1))

    pixels[2, 9] = np.nan
    panel = review.panel(Source(), old, 2, [9])
    assert np.any(np.all(panel == [10, 30, 240], axis=-1))


def test_chained_observation_resolution_verifies_every_bound_parent(tmp_path):
    import hashlib

    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    original = prepared / 'observations.npz'
    previous_path = tmp_path / 'first.npz'
    current_path = tmp_path / 'second.npz'
    for path, chunk in [(original, b'original'), (previous_path, b'first revision'), (current_path, b'second revision')]:
        path.write_bytes(chunk)

    def binding(path, parent):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    parent_observations_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())

    previous = binding(previous_path, original)
    current = dict(binding(current_path, previous_path), previous_revision=previous)
    doc = dict(programme_body_observation_revision=current)
    assert programme_observation_path(prepared, doc) == current_path
    previous_path.write_bytes(b'mismatched')
    with pytest.raises(ValueError, match='bound ledger'):
        programme_observation_path(prepared, doc)
