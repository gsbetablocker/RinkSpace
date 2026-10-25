"""Current hole references must resolve without changing print/source geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rinkspace.compat.ad_assets import _bind_template_path_references
from rinkspace.domain.print_assets import PrintBindingError, validate_template_topology
from rinkspace.session.print_installation import bound_scene_assets


def test_explicit_import_resolves_short_keys_without_editing_source_records():
    paths = [
        {'id': 'Ad2.print.A', 'original_id': 'A', 'closed': True,
         'material_uv': [[0, 0], [1, 0], [0, 1]],
         'separate_inner_contour_keys': ['A_inner'], 'role': 'letter_material_boundary'},
        {'id': 'Ad2.print.A_inner', 'original_id': 'A_inner', 'closed': True,
         'material_uv': [[.1, .1], [.2, .1], [.1, .2]],
         'parent_contour_key': 'A', 'role': 'letter_hole_boundary',
         'raw_record': {'parent_contour_key': 'A', 'source_native_uv': [[10, 11], [12, 13]]}},
    ]
    before = deepcopy(paths)
    _bind_template_path_references(paths)
    assert paths[1]['parent_contour_key'] == paths[0]['id']
    assert paths[0]['separate_inner_contour_keys'] == [paths[1]['id']]
    assert paths[1]['raw_record'] == before[1]['raw_record']
    assert [p['material_uv'] for p in paths] == [p['material_uv'] for p in before]
    validate_template_topology({'paths': paths})


def test_current_runtime_rejects_short_missing_or_self_parent():
    outer = {'id': 'Ad3.print.outside', 'role': 'letter_material_boundary'}
    for parent in ('outside', 'missing', 'Ad3.print.hole'):
        hole = {'id': 'Ad3.print.hole', 'role': 'letter_hole_boundary',
                'parent_contour_key': parent}
        with pytest.raises(PrintBindingError, match='invalid_print_parent_reference'):
            validate_template_topology({'paths': [outer, hole]})


def test_all_published_current_advertisements_have_resolvable_topology_and_hashes():
    root = Path(__file__).resolve().parents[1]
    scene_path = root/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    checked = set()
    for reference in scene['advertisement_assets']:
        path = scene_path.parent/reference['path']
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == reference['sha256']
        assets = json.loads(payload)
        for template in assets['templates']:
            validate_template_topology(template)
            checked.add(template['id'])
    assert {'print_template/Ad1', 'print_template/Ad2', 'print_template/Ad3'} <= checked


def test_d3_cannot_mix_an_older_selected_template_into_the_current_scene(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scene_path = root/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    reference = next(ref for ref in scene['advertisement_assets']
                     if 'print_instance/Ad2' in ref['instance_ids'])
    current = scene_path.parent/reference['path']
    registered = bound_scene_assets(scene, scene_path, current, 'print_instance/Ad2')
    assert len(registered) >= 3
    older = json.loads(current.read_text(encoding='utf8'))
    older['templates'][0]['paths'][0]['material_uv'][0][0] += .01
    stale = tmp_path/'older_ad2.json'
    stale.write_text(json.dumps(older), encoding='utf8')
    with pytest.raises(ValueError, match='differs from current scene binding'):
        bound_scene_assets(scene, scene_path, stale, 'print_instance/Ad2')
