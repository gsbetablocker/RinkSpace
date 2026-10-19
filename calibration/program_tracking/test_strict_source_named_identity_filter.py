from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack, normalize_selected_ice_rows,
)
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_named_identity_filter import (
    FILTER_SCHEMA, POINT_FIELDS, FRAME_FIELDS, export_exact_named_identity_pending_pack,
)
from calibration.program_tracking.test_strict_absolute_source import fixture_inputs


def _fixture(tmp_path):
    binding, row, measurement, inventory = fixture_inputs(tmp_path)
    row['original_candidate_indices'] = [0, 1, 2]
    sides = np.concatenate([measurement['candidate_sides']+[10.*i, 0.] for i in range(3)])
    measurement.update(candidate_sides=sides, candidate_uv=sides.mean(1),
        candidate_query=np.arange(3), candidate_flags=np.full(3, 8, np.uint8),
        query_curve=np.zeros(3, np.int16))
    quants = np.repeat(sides.astype(float)[:, :, None], 3, axis=2)
    quants[:, :, 0, 1] -= .2
    quants[:, :, 2, 1] += .3
    measurement['candidate_side_quantiles_uv'] = quants
    normalized = normalize_selected_ice_rows([row], measurement,
        source_binding=binding, curve_inventory=inventory)
    current = normalized['rows'][0]
    current['source_binding'] = deepcopy(binding)
    current.setdefault('source_readout_uncertainty', {})['actual_side_gradient_quantiles_uv'] = quants.tolist()
    writer = StrictAbsolutePackWriter(tmp_path/'source.npz', source=normalized['review'].source,
        expected_actual_pts=[100, 200], legacy_pixel_families=['ice'])
    assert writer.add_row(current, normalized['review']) == 3
    held = deepcopy(current)
    held['split'] = 'held'
    assert writer.add_row(held, normalized['review']) == 3
    writer.mark_exposure_processed(100)
    source = writer.finish()
    source_ref = dict(path=source['path'], sha256=source['sha256'])
    pack = StrictAbsoluteSourcePack(source['path'], expected_sha256=source['sha256'])
    fact = dict(schema=FILTER_SCHEMA,
        retain_pending_source_readouts_full_literal_evidence=True,
        original_packs_or_alias_clock_evidence_must_not_be_changed=True,
        source_pack_groups=[dict(source_pack_reference=source_ref, pending_named_point_count=1,
            source_review_witnesses=[dict(path=str(tmp_path/'source_identity_seed.json'),
                sha256=file_sha(tmp_path/'source_identity_seed.json'))],
            entries=[dict(pack_row_index=0, pack_point_indices=[1],
                physical_material_id=pack.rows[0]['physical_material_id'], source_binding=pack.rows[0]['source_binding'],
                original_candidate_indices=pack.rows[0]['original_candidate_indices'],
                source_query_indices=pack.rows[0].get('source_query_indices'),
                exclude_from_final_named_FIT=True, status='source_material_measured_named_identity_pending',
                reason='Explicit fixture identity pending; not real programme evidence')])])
    return source_ref, pack, fact


def _write_fact(tmp_path, fact):
    path = tmp_path/'pending.json'
    path.write_text(json.dumps(fact), encoding='utf8')
    return dict(path=str(path), sha256=file_sha(path))


def test_exact_partial_quarantine_preserves_numeric_proofs_clock_and_held(tmp_path):
    source_ref, original, fact = _fixture(tmp_path)
    receipt = export_exact_named_identity_pending_pack(source_ref, _write_fact(tmp_path, fact), tmp_path/'filtered.npz')
    result = StrictAbsoluteSourcePack(receipt['path'], expected_sha256=receipt['sha256'])
    keep = [0, 2, 3, 4, 5]
    assert receipt['pending_named_points'] == 1
    assert len(result.fit_rows[0]['uv']) == 2
    assert len(result.held_rows[0]['uv']) == 3
    for name in POINT_FIELDS:
        assert result.values[name].dtype == original.values[name].dtype
        assert result.values[name].tobytes() == original.values[name][keep].tobytes()
    for name in FRAME_FIELDS:
        assert result.values[name].tobytes() == original.values[name].tobytes()
    assert result.proof_templates == original.proof_templates
    assert result.rows[1] == original.rows[1]
    assert np.array_equal(result.rows[0]['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'],
        np.asarray(original.rows[0]['source_readout_uncertainty']['actual_side_gradient_quantiles_uv'])[[0, 2]])
    with np.load(receipt['pending_evidence_bank_reference']['path'], allow_pickle=False) as pending:
        assert pending['original_pack_point_indices'].tolist() == [1]
        for name in POINT_FIELDS:
            assert pending[name].tobytes() == original.values[name][[1]].tobytes()
    assert file_sha(source_ref['path']) == source_ref['sha256']


def test_held_quarantine_is_rejected_without_any_export(tmp_path):
    source_ref, original, fact = _fixture(tmp_path)
    entry = fact['source_pack_groups'][0]['entries'][0]
    entry.update(pack_row_index=1, pack_point_indices=[4])
    with pytest.raises(ValueError, match='HELD'):
        export_exact_named_identity_pending_pack(source_ref, _write_fact(tmp_path, fact), tmp_path/'filtered.npz')
    assert not (tmp_path/'filtered.named_identity_pending.npz').exists()


@pytest.mark.parametrize('change', ['source_binding', 'candidate_locator', 'point_outside_row'])
def test_fact_cannot_retarget_different_source_pixel(tmp_path, change):
    source_ref, original, fact = _fixture(tmp_path)
    entry = fact['source_pack_groups'][0]['entries'][0]
    if change == 'source_binding':
        entry['source_binding'] = dict(entry['source_binding'], actual_pts=200)
    elif change == 'candidate_locator':
        entry['original_candidate_indices'] = [999]
    else:
        entry['pack_point_indices'] = [4]
    with pytest.raises(ValueError):
        export_exact_named_identity_pending_pack(source_ref, _write_fact(tmp_path, fact), tmp_path/'filtered.npz')
    assert not (tmp_path/'filtered.npz').exists()
    assert not (tmp_path/'filtered.named_identity_pending.npz').exists()


def test_changed_authenticated_filter_is_rejected(tmp_path):
    source_ref, original, fact = _fixture(tmp_path)
    reference = _write_fact(tmp_path, fact)
    (tmp_path/'pending.json').write_text('{}', encoding='utf8')
    with pytest.raises(ValueError):
        export_exact_named_identity_pending_pack(source_ref, reference, tmp_path/'filtered.npz')
    assert not (tmp_path/'filtered.npz').exists()
