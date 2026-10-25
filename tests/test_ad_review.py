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


@pytest.mark.parametrize('relative,count,holes', [
    ('ad1/assets_r4.json', 26, 6),
    ('ad2/assets_r5.json', 21, 2),
    ('ad3/assets_r3.json', 19, 11),
])
def test_all_current_templates_use_same_native_chart_without_requiring_media(
        relative, count, holes, tmp_path, library, preview_font):
    root = Path(__file__).resolve().parents[1]
    path = root/'events/ankara_2026/scene'/relative
    original = path.read_bytes()
    assets = json.loads(original)
    receipt = ad_review.render(path, library, tmp_path/'preview')
    template = receipt['templates'][0]
    geometry = json.loads(Path(template['geometry']).read_text(encoding='utf-8'))
    assert template['path_count'] == count
    assert template['hole_count'] == holes
    assert [p['id'] for p in geometry['shapes']] == [p['id'] for p in assets['templates'][0]['paths']]
    assert geometry['native_api'] == 'sample_print_chart'
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
    assert 'D1 · Ad2' not in page or relative.startswith('ad2/')


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
