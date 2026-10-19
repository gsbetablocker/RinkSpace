import importlib.util
from pathlib import Path

import cv2
import numpy as np


def module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('gap_test', scripts / 'extract_short_gap_edges_v09.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gap_candidates_are_bounded_and_never_cross_shot(monkeypatch):
    m = module(monkeypatch)
    rows = []
    for i, (time, shot, count) in enumerate([(0., 'a', 12), (.02, 'a', 0), (.04, 'a', 12),
                                            (.06, 'b', 0), (.08, 'b', 12), (.10, 'b', 0), (.80, 'b', 12)]):
        rows.append({'pts': round(time*1000), 'source_time_s': time, 'shot_id': shot,
                     'channels': {k: {'observed_point_pair_count': count} for k in ('ice_texture', 'board_surface')}})
    pairs, _ = m.gap_candidates(rows)
    assert (0, 2) in pairs
    assert all(rows[a]['shot_id'] == rows[b]['shot_id'] and 0 < rows[b]['source_time_s']-rows[a]['source_time_s'] <= .5 for a,b in pairs)
    assert not any(b == 6 for a,b in pairs)


def test_gap_lk_keeps_measured_endpoints_and_obeys_target_mask(monkeypatch):
    m = module(monkeypatch)
    rng = np.random.default_rng(18)
    a = rng.integers(0, 255, (540, 960), np.uint8)
    b = cv2.warpAffine(a, np.array([[1., 0., 4.], [0., 1., 2.]]), (960, 540))
    mask = np.full_like(a, 255)
    x = {'gray': a, 'masks': {'board_surface': mask}}
    y = {'gray': b, 'masks': {'board_surface': mask}}
    result = m.lk_pairs(x, y, 'board_surface')
    assert len(result['from_points_source_px']) >= 8
    delta = np.asarray(result['to_points_source_px']) - np.asarray(result['from_points_source_px'])
    assert np.median(np.linalg.norm(delta-[8., 4.], axis=1)) < .5
    y['masks']['board_surface'] = np.zeros_like(mask)
    assert not m.lk_pairs(x, y, 'board_surface')['from_points_source_px']
