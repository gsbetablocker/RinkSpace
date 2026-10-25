"""D3 uses the scene's admitted instances, not every identity stored in an asset."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from rinkspace.session import print_installation


def selected_scene(primary):
    path = Path(__file__).resolve().parents[1]/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(path.read_text(encoding='utf-8-sig'))
    reference = deepcopy(next(r for r in scene['advertisement_assets']
                              if r['instance_id'] == primary))
    assets = (path.parent/reference['path']).resolve()
    reference['path'] = str(assets)
    reference['instance_ids'] = [primary]
    scene['advertisement_assets'] = [reference]
    return scene, path, assets


def test_d3_rejects_an_instance_omitted_by_the_scene():
    scene, path, assets = selected_scene('print_instance/Ad1')
    with pytest.raises(ValueError, match='not bound to current scene'):
        print_installation.bound_scene_assets(scene, path, assets, 'print_instance/Ad14')


def test_d3_accepts_disjoint_selections_from_one_asset():
    scene, path, assets = selected_scene('print_instance/Ad1')
    other = deepcopy(scene['advertisement_assets'][0])
    other.update(instance_id='print_instance/Ad14', instance_ids=['print_instance/Ad14'])
    scene['advertisement_assets'].append(other)
    registered = print_installation.bound_scene_assets(scene, path, assets, 'print_instance/Ad1')
    assert len(registered) == 2


@pytest.mark.parametrize('primary', ['print_instance/Ad1', 'print_instance/Ad2'])
def test_d3_render_does_not_expand_localized_or_unlocalized_pairs(
        primary, tmp_path, monkeypatch):
    scene, _, assets = selected_scene(primary)
    scene_path = tmp_path/'scene.json'
    scene_path.write_text(json.dumps(scene), encoding='utf-8')
    camera_path = tmp_path/'camera.json'
    camera_path.write_text(json.dumps({'controls_ref': 'controls.npy',
        'first_pts': 0, 'stop_pts_exclusive': 10}), encoding='utf-8')
    np.save(tmp_path/'controls.npy', np.zeros((4, 3)))
    frame = tmp_path/'image.png'
    Image.new('RGB', (8, 8), 'white').save(frame)
    sampled, previews, contexts = [], [], []

    def sample(_assets, identity, _board, _core, *, parameter_state=None):
        assert parameter_state is None
        sampled.append(identity)
        return {'instance_id': identity, 'parameters': [0., 0., 0., .1, .01],
                'paths': [{'id': 'test.edge', 'role': 'panel_outer_boundary', 'closed': False}],
                'chart_board_max_abs': 0.}

    def source_view(_data, _scene, _core, _camera, _frame, _pts, _out, context, _verification):
        contexts.extend(context)
        return {}

    monkeypatch.setattr(print_installation, 'NativeCore', Mock())
    monkeypatch.setattr(print_installation, 'sample_instance', sample)
    monkeypatch.setattr(print_installation, 'unfolded', lambda data, out: previews.append(deepcopy(data)))
    monkeypatch.setattr(print_installation, 'board_view', Mock())
    monkeypatch.setattr(print_installation, 'course_continuity', lambda *a: {})
    monkeypatch.setattr(print_installation, 'source_view', source_view)
    report = print_installation.render(scene_path, assets, primary, camera_path, frame, 1,
                                       tmp_path/'unused-library', tmp_path/'out')
    assert sampled == [primary]
    assert contexts == []
    assert previews[0]['unlocalized_related_instances'] == []
    assert report['source']['unlocalized_context_print_instances'] == []
    assert report['fit_executed'] is False
