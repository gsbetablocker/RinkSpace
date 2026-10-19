from copy import deepcopy
import json

import numpy as np
import pytest

from events.ankara_2026.tools.source_observations import qualify_saved_board_transition_locations as m


@pytest.fixture(scope='module')
def parent():
    return json.loads(m.PARENT.read_text(encoding='utf8'))


def test_all_reviewed_formats_preserve_actual_original_values(parent):
    images = {}
    records = [m.qualify_row(parent['rows'][i], i, images) for i in m.ROW_INDICES]
    assert len(records) == 25
    assert sum(len(r['measured_boundary_row']['uv']) for r in records) == 83
    for record in records:
        original = parent['rows'][record['original_row_index']]
        measured = record['measured_boundary_row']
        for key in ('uv', 'normal', 'normal_interval_px', 'sigma_px', 'split'):
            assert measured[key] == original[key]
        assert record['source_row_canonical_json_sha256'] == m.hashlib.sha256(m.canonical(original)).hexdigest()
    kinds = {r['original_row_index']: r['location_estimator_kind'] for r in records}
    assert kinds[6] == 'smoothed_negative_gray_plateau_half_level_q50'
    assert kinds[19] == kinds[20] == 'positive_blue_chroma_gradient_subpixel_peak'
    assert 'midpoint' not in kinds[19]


@pytest.mark.parametrize('index', [0, 4, 6, 19])
@pytest.mark.parametrize('field', ['uv', 'normal_interval_px'])
def test_estimator_replay_rejects_changed_uv_or_support(parent, index, field):
    row = deepcopy(parent['rows'][index])
    row[field][0][0] += .1
    with pytest.raises(ValueError): m.qualify_row(row, index, {})


def test_interval_only_and_unreviewed_rows_are_excluded(parent):
    assert not (set(range(95, 101)) & set(m.ROW_INDICES))
    for index in (5, 95, 100):
        with pytest.raises(ValueError): m.qualify_row(parent['rows'][index], index, {})


def test_holdout_role_cannot_be_promoted(parent):
    row = deepcopy(parent['rows'][4]); row['split'] = 'holdout'
    with pytest.raises(ValueError): m.qualify_row(row, 4, {})


def test_bad_saved_grayscale_quantile_is_rejected(parent):
    row = deepcopy(parent['rows'][0]); row['extraction']['samples'][0]['q10_q50_q90_x'][1] += .2
    with pytest.raises(ValueError): m.qualify_row(row, 0, {})
