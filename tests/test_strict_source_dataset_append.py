import json

import numpy as np
import pytest

from calibration.program_tracking.strict_source_fit_dataset import SCHEMA, reference, partition_support
from calibration.program_tracking.strict_source_dataset_append import append_reviewed_deltas
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.test_strict_source_disjoint_merge import _fixture_pack


def _inputs(tmp_path):
    base_pack = _fixture_pack(tmp_path/'base')
    raw_delta = _fixture_pack(tmp_path/'delta', pts=200)
    with np.load(raw_delta['path']) as saved:
        values = {k:saved[k] for k in saved.files}
    metadata = json.loads(str(values['metadata_json'].item()))
    metadata.update(expected_actual_pts=[200], all_exposures_processed=True,
                    all_exposures_exact_native_BGR_bound=True)
    for name in ('source_frame_actual_pts','source_frame_native_pts','source_frame_BGR_sha256'):
        values[name] = values[name][[1]]
    values['metadata_json'] = json.dumps(metadata)
    delta_path = tmp_path/'delta_subset.npz'
    np.savez_compressed(delta_path, **values)
    delta = reference(delta_path)
    pack = StrictAbsoluteSourcePack(base_pack['path'], expected_sha256=base_pack['sha256'])
    indices = tmp_path/'base_index.npz'
    part = partition_support(pack.rows, pack.values['row_offsets'], pack.values['sigma_px'])
    np.savez_compressed(indices, row_offsets=pack.values['row_offsets'],
        promoted_held_rows=np.array([], np.int64), pending_named_point_indices=np.array([], np.int64), **part)
    coverage = tmp_path/'coverage.npz'
    np.savez_compressed(coverage, representative=np.ones(3,bool), actual_pts=np.array([100,200,300]),
        native_pts=np.array([110,210,310]), BGR_sha256=np.array([b'b'*64]*3))
    base = tmp_path/'base.json'
    base.write_text(json.dumps(dict(schema=SCHEMA, source=pack.metadata['source'], expected_actual_pts=[100,200,300],
        original_native_coverage_reference=reference(coverage), source_review_authorities=[],
        fit_points=2, held_points=0, pending_support_points=1, pending_named_points=0,
        actual_exposures_with_finite_fit=1, actual_exposures_without_finite_fit=2,
        finite_fit_points_by_material={'C.paint':2},
        cuts=[dict(source_pack=reference(base_pack['path']), support_index=reference(indices),
            fit_points=2, held_points=0, pending_support_points=1, pending_named_points=0)])),encoding='utf8')
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(schema='rinkspace.source-input-cross-pack-exact-pixel-operator-audit.v1',
        records=[dict(inputs=[dict(source_reference=reference(base_pack['path'])),dict(source_reference=delta)],
                      operator_conflicts=[])], points=6, unique_exact_pixel_target_count=6,
        counts={'typed_input_points':3, 'duplicates':0, 'operator_conflicts':0})),encoding='utf8')
    return reference(base), delta, reference(audit)


def test_measured_subclock_delta_preserves_full_clock_and_reuses_base_indices(tmp_path):
    base, delta, audit = _inputs(tmp_path)
    before = json.loads(open(base['path'], encoding='utf8').read())
    result = append_reviewed_deltas(base, [delta], audit, tmp_path/'final')
    after = json.loads(open(result['dataset_reference']['path'], encoding='utf8').read())
    assert after['expected_actual_pts'] == [100,200,300]
    assert after['cuts'][0] == before['cuts'][0]
    assert after['cuts'][1]['source_pack'] == delta
    assert after['fit_points'] == 4
    assert after['pending_support_points'] == 2
    assert after['actual_exposures_with_finite_fit'] == 2
    assert after['actual_exposures_without_finite_fit'] == 1
    assert len(list((tmp_path/'final').glob('*indices.npz'))) == 1


def test_source_preflight_conflict_is_rejected_before_any_index_write(tmp_path):
    base, delta, audit = _inputs(tmp_path)
    packet = json.loads(open(audit['path'], encoding='utf8').read())
    packet['counts']['operator_conflicts'] = 1
    with open(audit['path'], 'w', encoding='utf8') as stream:
        json.dump(packet, stream)
    with pytest.raises(ValueError, match='preflight'):
        append_reviewed_deltas(base, [delta], reference(audit['path']), tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
