import json

import numpy as np
import pytest

from calibration.program_tracking.strict_source_fit_dataset import SCHEMA, partition_support, reference
from calibration.program_tracking.strict_source_dataset_revision import revise_named_pending
from calibration.program_tracking.test_strict_source_named_identity_filter_with_exclusions import (
    _fixture, _write_fact,
)


def _base(tmp_path):
    source, pack, fact = _fixture(tmp_path)
    partition = partition_support(pack.rows, pack.values['row_offsets'], pack.values['sigma_px'])
    index = tmp_path / 'base_indices.npz'
    np.savez_compressed(index, row_offsets=pack.values['row_offsets'],
        promoted_held_rows=np.array([], np.int64), pending_named_point_indices=np.array([], np.int64),
        **partition)
    manifest = tmp_path / 'base.json'
    manifest.write_text(json.dumps(dict(schema=SCHEMA, source=pack.metadata['source'],
        expected_actual_pts=pack.metadata['expected_actual_pts'], source_review_authorities=[],
        fit_points=3, held_points=3, pending_support_points=0, pending_named_points=0,
        actual_exposures_with_finite_fit=1, actual_exposures_without_finite_fit=1,
        finite_fit_points_by_material={pack.rows[0]['physical_material_id']:3},
        cuts=[dict(source_pack=source, support_index=reference(index), pending_filter_references=[],
            source_rows=2, source_points=6, fit_points=3, held_points=3,
            pending_support_points=0, pending_named_points=0, held_protected_rows_promoted=0)])), encoding='utf8')
    return reference(manifest), pack, fact


def test_exact_pending_revision_preserves_source_and_complete_held(tmp_path):
    base, pack, fact = _base(tmp_path)
    original_base_bytes = (tmp_path/'base.json').read_bytes()
    result = revise_named_pending(base, [_write_fact(tmp_path, fact)], tmp_path/'revised')
    revised = json.loads(open(result['dataset_reference']['path'], encoding='utf8').read())
    assert revised['fit_points'] == 2
    assert revised['held_points'] == 3
    assert revised['pending_named_points'] == 1
    source = fact['source_pack_groups'][0]['source_pack_reference']
    assert revised['cuts'][0]['source_pack'] == source
    with np.load(revised['cuts'][0]['support_index']['path']) as saved:
        assert saved['fit_point_indices'].tolist() == [0,2]
        assert saved['held_point_indices'].tolist() == [3,4,5]
        assert saved['pending_named_point_indices'].tolist() == [1]
    assert (tmp_path/'base.json').read_bytes() == original_base_bytes
    assert reference(source['path']) == source


def test_new_fact_cannot_quarantine_a_held_row(tmp_path):
    base, pack, fact = _base(tmp_path)
    entry = fact['source_pack_groups'][0]['entries'][0]
    entry.update(pack_row_index=1, pack_point_indices=[4])
    with pytest.raises(ValueError, match='exclusion'):
        revise_named_pending(base, [_write_fact(tmp_path, fact)], tmp_path/'rejected')
    assert not (tmp_path/'rejected/fit_dataset.json').exists()
