import importlib.util
from pathlib import Path
import pytest


def load_module(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('component_test', scripts / 'extract_component_edges_v09.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    return [{'pts': i, 'source_time_s': i*.2, 'shot_id': 'one',
             'channels': {k: {'observed_point_pair_count': 10, 'active_track_count': 10+i}
                          for k in ('ice_texture', 'board_surface')}} for i in range(16)]


def test_component_selection_is_bounded_to_three_actual_frames(monkeypatch):
    module = load_module(monkeypatch)
    rows = fixture_rows()
    groups, membership = module.components(rows, {'frames': []}, {'frames': []})
    assert len(groups) == 1 and len(membership) == len(rows)
    assert 1 <= len(groups[0]['candidate_indices']) <= 3
    assert set(groups[0]['candidate_source_pts']) <= {r['pts'] for r in rows}


def test_component_graph_rejects_cross_shot_input_edges(monkeypatch):
    module = load_module(monkeypatch)
    rows = fixture_rows()
    rows[-1]['shot_id'] = 'other'
    returns = {'frames': [{'reference_source_pts': 0, 'source_pts': 15,
                           'channels': {'ice_marking': {'reference_points_source_px': [[0, 0]]*8}}}]}
    with pytest.raises(ValueError, match='crosses a shot'):
        module.components(rows, returns, {'frames': []})
