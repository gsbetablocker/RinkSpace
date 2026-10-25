"""Generic D1 rendering is a model/source inventory, never a fit acceptance."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont
import pytest

from rinkspace.session import ad_review, curling_preview
from rinkspace.core.native import NativeCore
from rinkspace.domain.print_assets import evaluate_print_path, sample_instance


@pytest.fixture
def preview_font(monkeypatch):
    # Layout/code tests need no workstation font installation. Actual audit
    # previews use an installed CJK font and are inspected separately.
    font = lambda size, **kwargs: ImageFont.load_default(size=size)
    monkeypatch.setattr(curling_preview, 'review_font', font)
    monkeypatch.setattr(ad_review, 'review_font', font)


@pytest.fixture
def library():
    path = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not path:
        pytest.skip('explicit native library required')
    return path


@pytest.mark.parametrize('advertisement_number,count,holes', [
    (1, 26, 6),
    (2, 21, 2),
    (3, 19, 11),
])
def test_all_current_templates_use_same_native_chart_without_requiring_media(
        advertisement_number, count, holes, tmp_path, library, preview_font):
    root = Path(__file__).resolve().parents[1]
    field = root/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(field.read_text(encoding='utf-8'))
    ref, = (item for item in scene['advertisement_assets']
            if item['advertisement_number'] == advertisement_number)
    path = (field.parent/ref['path']).resolve()
    original = path.read_bytes()
    assets = json.loads(original)
    receipt = ad_review.render(path, library, tmp_path/'preview')
    template = receipt['templates'][0]
    geometry = json.loads(Path(template['geometry']).read_text(encoding='utf-8'))
    assert template['path_count'] == count
    assert template['hole_count'] == holes
    assert [p['id'] for p in geometry['shapes']] == [p['id'] for p in assets['templates'][0]['paths']]
    assert geometry['geometry_evaluator'] == 'evaluate_print_path'
    assert template['display_basis']['kind'] == 'shared_model_size'
    assert geometry['shape_status'] == assets['templates'][0]['shape_status']
    assert len(receipt['source_rows']) == len(assets['observations'])
    assert receipt['new_source_observations'] == 0
    assert receipt['fit_executed'] is False
    for shape, source in zip(geometry['shapes'], assets['templates'][0]['paths']):
        assert shape['source_observation_id'] == source.get('source_observation_id')
        assert shape['parent_contour_key'] == source.get('parent_contour_key')
        assert shape['closed'] == source.get('closed')
        assert np.all(np.isfinite(shape['native_template_xy']))
        if (shape.get('geometry') or {}).get('kind') == 'unfolded_square':
            edges = np.diff(shape['native_template_xy'], axis=0)
            lengths = np.linalg.norm(edges, axis=1)
            np.testing.assert_allclose(lengths, lengths[0], rtol=0, atol=1e-14)
            assert abs(np.dot(edges[0], edges[1])) < 1e-14
            assert shape['display_fill'] == 'none'
    assert path.read_bytes() == original
    assert Path(receipt['card']).is_file()
    page = Path(receipt['page']).read_text(encoding='utf-8')
    assert '来源缺失与限制' in page
    assert 'D1 · Ad2' not in page or advertisement_number == 2


def synthetic_assets(tmp_path):
    frame = tmp_path/'synthetic.png'
    Image.new('RGB', (64, 40), 'white').save(frame)
    sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    assets = {'templates': [{'id': 'print_template/synthetic',
        'shape_status': 'synthetic_software_test', 'size_class_id': 'size',
        'paths': [{'id': 'synthetic.edge', 'role': 'panel_outer_boundary',
                   'closed': False, 'material_uv': [[0., 0.], [1., 0.]]}]}],
        'parameter_blocks': [{'id': 'size', 'kind': 'print_size', 'size': 2,
                              'values': [math.log(.08), math.log(.012)]}],
        'sources': {'video': {'sha256': 'a'*64, 'time_base': '1/1000', 'pts': 200,
                             'frame_path': str(frame), 'frame_sha256': sha}},
        'observations': [
            {'id': 'first', 'source_id': 'video', 'pts': 100, 'time_base': '1/1000',
             'feature_id': 'synthetic.edge', 'closed': False, 'source_native_uv': [[4., 4.], [10., 4.]],
             'source_image': {'file': str(frame), 'sha256': sha}},
            {'id': 'no_frame', 'source_id': 'video', 'pts': 101, 'time_base': '1/1000',
             'feature_id': 'synthetic.edge', 'source_native_uv': [5., 5.]},
            {'id': 'no_pixels', 'source_id': 'video', 'pts': 200, 'time_base': '1/1000'},
            {'id': 'point_only', 'source_id': 'video', 'pts': 200, 'time_base': '1/1000',
             'source_native_uv': [7., 9.]},
        ]}
    return assets


def test_missing_source_fields_are_explicit_without_changing_readings(tmp_path, library, preview_font):
    assets = synthetic_assets(tmp_path)
    before = deepcopy(assets)
    path = tmp_path/'assets.json'; path.write_text(json.dumps(assets), encoding='utf-8')
    receipt = ad_review.render(path, library, tmp_path/'preview')
    rows = {r['observation_id']: r for r in receipt['source_rows']}
    assert rows['first']['status'] == 'displayed'
    assert rows['point_only']['status'] == 'displayed'
    assert rows['no_frame']['status'] == 'unavailable'
    assert 'source_frame_path_missing' in rows['no_frame']['missing']
    assert 'source_native_uv_missing_or_invalid' in rows['no_pixels']['missing']
    assert 'feature_id_not_recorded' in rows['point_only']['warnings']
    assert {p['pts'] for p in receipt['panels']} == {100, 200}
    assert all(p['native_source_verified'] is False for p in receipt['panels'])
    assert all(p['declared_frame_hash_verified'] for p in receipt['panels'])
    point = next(p for p in receipt['panels'] if p['pts'] == 200)['identifiers'][0]
    assert point['closed'] is None and point['feature_id'] is None
    geometry = json.loads(Path(receipt['templates'][0]['geometry']).read_text(encoding='utf-8'))
    assert geometry['shapes'][0]['source_observation_id'] is None
    assert json.loads(path.read_text(encoding='utf-8')) == before


def test_changed_source_image_does_not_prevent_model_preview_or_become_evidence(
        tmp_path, library, preview_font):
    assets = synthetic_assets(tmp_path)
    assets['observations'] = [assets['observations'][0]]
    assets['observations'][0]['source_image']['sha256'] = '0'*64
    path = tmp_path/'assets.json'; path.write_text(json.dumps(assets), encoding='utf-8')
    receipt = ad_review.render(path, library, tmp_path/'preview')
    assert receipt['panels'] == []
    assert receipt['source_rows'][0]['status'] == 'rejected'
    assert 'source-frame bytes changed' in receipt['source_rows'][0]['reason']
    assert Path(receipt['card']).is_file()


def test_source_registry_can_only_supply_the_matching_exposure(tmp_path):
    assets = synthetic_assets(tmp_path)
    row = assets['observations'][1]
    _, records = ad_review._source_rows(assets, tmp_path/'assets.json')
    assert 'source_frame_path_missing' in records[1]['missing']
    source = assets['sources']['video']
    source['exposures'] = [{k: source[k] for k in ('frame_path', 'frame_sha256')} | {'pts': row['pts']}]
    available, records = ad_review._source_rows(assets, tmp_path/'assets.json')
    assert records[1]['status'] == 'ready'
    matched = next(value for record, value in available if record['observation_id'] == 'no_frame')
    assert matched['pts'] == 101
    assert matched['source_native_uv'] == row['source_native_uv']


@pytest.mark.parametrize('primary', ['print_instance/Ad1', 'print_instance/Ad5'])
def test_d1_consumes_the_same_current_internal_geometry_as_d3(
        primary, tmp_path, library, preview_font):
    scene_path = Path(__file__).resolve().parents[1]/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8-sig'))
    reference = next(r for r in scene['advertisement_assets'] if r['instance_id'] == primary)
    assets = json.loads((scene_path.parent/reference['path']).read_text(encoding='utf-8-sig'))
    before = deepcopy(assets)
    state = {block['id']: deepcopy(block['values']) for block in assets['parameter_blocks']}
    if primary == 'print_instance/Ad1':
        # Include an internal warp kink, analytic square and shared width in
        # the same candidate. Frozen seed vertices cannot represent this state.
        warp = next(b for b in assets['parameter_blocks'] if b['kind'] == 'print_shape')
        state[warp['id']][0] += .075
        state[warp['id']][4] += .35
        state[warp['id']][5] -= .17
        square = next(b for b in assets['parameter_blocks'] if b['kind'] == 'print_square')
        state[square['id']][0] += .045
        state[square['id']][2] += .1
        size = next(b for b in assets['parameter_blocks'] if b['kind'] == 'print_size')
        state[size['id']][0] += math.log(1.15)
    core = NativeCore(library)
    template = assets['templates'][0]
    receipt = ad_review._template_panel(assets, template, core, tmp_path, 0,
                                       parameter_state=state)
    geometry = json.loads(Path(receipt['geometry']).read_text(encoding='utf-8'))
    dimensions = geometry['display_dimensions']
    for shape in geometry['shapes']:
        installed = evaluate_print_path(assets, primary, shape['id'], scene['board'], core,
                                       shape['stations'], parameter_state=state, jacobian=False)
        np.testing.assert_allclose(shape['material_uv'], installed['material_uv'], rtol=0, atol=1e-14)
        # Native chart at a display origin, with the same current shared size.
        expected = core.sample_print_chart([0., 0., 0., *dimensions], installed['material_uv'],
                                           np.linspace(0., 1., len(shape['stations'])))
        np.testing.assert_allclose(shape['native_template_xy'], expected, rtol=0, atol=1e-14)
        assert 'xyz' not in shape and 'instance_id' not in shape
        assert not any(identity.startswith('installation/') for identity in shape['parameter_block_ids'])
    assert geometry['parameter_state_consumed'] is True
    assert geometry['display_origin_is_not_installation'] is True
    assert geometry['physical_placement_estimated'] is False
    assert geometry['new_source_observations'] == 0
    assert assets == before


def test_d1_can_draw_an_unlocalized_template_without_inventing_placement(
        tmp_path, library, preview_font):
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root/'events/ankara_2026/scene/ad1/assets_r5.json').read_text(encoding='utf-8-sig'))
    for block in assets['parameter_blocks']:
        if block['kind'] == 'print_placement':
            block['values'] = None
    before = deepcopy(assets)
    receipt = ad_review._template_panel(assets, assets['templates'][0], NativeCore(library), tmp_path, 0)
    assert receipt['path_count'] == len(assets['templates'][0]['paths'])
    assert assets == before


def test_rectangle_drawing_budget_uses_the_current_native_rectangle(library):
    scene_path = Path(__file__).resolve().parents[1]/'events/ankara_2026/scene/field/model.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8-sig'))
    reference = next(r for r in scene['advertisement_assets'] if r['instance_id'] == 'print_instance/Ad5')
    assets = json.loads((scene_path.parent/reference['path']).read_text(encoding='utf-8-sig'))
    data = sample_instance(assets, reference['instance_id'], scene['board'], NativeCore(library), max_step=.0004)
    bindings = assets['templates'][0]['internal_geometry']['path_bindings']
    rectangle = [path for path in data['paths'] if bindings[path['id']]['kind'] == 'rectangle_side']
    assert len(rectangle) == 4
    for path in rectangle:
        lengths = np.linalg.norm(np.diff(np.asarray(path['sz']), axis=0), axis=1)
        assert float(lengths.max()) <= .0004 + 1e-14, path['id']
