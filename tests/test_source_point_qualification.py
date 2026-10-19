from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.solver_runtime.source_point_qualification import (
    PACKET_FIELD, SCHEMA, file_sha256, original_uv_sha256,
    load_source_point_qualification,
)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf8')


def materials(tmp_path, *, split='fit', user_annotation=False, excluded=None, held=False):
    raw = tmp_path / 'raw_native_229702144.png'
    raw.write_bytes(b'exact native RGB fixture')
    row = dict(id='original_curve', pts=229702144, split=split,
        uv=[[20., 40.], [30., 70.], [80., 90.]], sigma_px=[2., 3., 4.],
        feature_id='F3_R3_center', curve=dict(kind='curling_centreline', row=3),
        source_raw_frame=str(raw), source_file_sha256=file_sha256(raw))
    if user_annotation:
        row['source_user_annotation'] = dict(path='retained actual user drawing')
    parent = dict(schema='rinkspace.additional-native-ice-curves.v1',
        source_video_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        time_base='1/15360', rows=[row])
    if held:
        held_row = deepcopy(row)
        held_row.update(id='original_HELD_curve', split='holdout')
        parent['rows'].append(held_row)
    parent_path = tmp_path / 'parent.json'
    write_json(parent_path, parent)
    record = dict(row_index=0, row_id=row['id'], source_pts=row['pts'],
        original_uv_float64_sha256=original_uv_sha256(row),
        excluded_point_indices=[1] if excluded is None else excluded,
        review_basis='original_rgb_only', reason='Actual RGB shows another crossing paint stroke',
        native_rgb_evidence=[dict(path=str(raw), sha256=file_sha256(raw), source_pts=row['pts'])])
    manifest = dict(schema=SCHEMA, source_family='additional_native_ice_curves',
        original_packet=dict(path=str(parent_path), sha256=file_sha256(parent_path)),
        records=[record])
    manifest_path = tmp_path / 'review.json'
    write_json(manifest_path, manifest)
    derived = deepcopy(parent)
    derived[PACKET_FIELD] = dict(path=str(manifest_path), sha256=file_sha256(manifest_path))
    derived_path = tmp_path / 'derived.json'
    write_json(derived_path, derived)
    return SimpleNamespace(raw=raw, row=row, parent=parent, parent_path=parent_path,
        manifest=manifest, manifest_path=manifest_path, derived=derived, derived_path=derived_path)


def load(m):
    return load_source_point_qualification(m.derived_path, m.derived,
        source_family='additional_native_ice_curves')


def resign_manifest(m):
    write_json(m.manifest_path, m.manifest)
    m.derived[PACKET_FIELD]['sha256'] = file_sha256(m.manifest_path)
    write_json(m.derived_path, m.derived)


def test_default_has_no_mask_or_objective_revision(tmp_path):
    m = materials(tmp_path, held=True)
    assert load_source_point_qualification(m.parent_path, m.parent,
        source_family='additional_native_ice_curves') is None


def test_scalar_mask_preserves_weights_slots_and_excludes_accuracy(tmp_path):
    m = materials(tmp_path)
    qualified = load(m)
    original_r = np.arange(6.) + 1
    original_j = csr_matrix(np.arange(18.).reshape(6, 3) + 1)
    r, j, inactive = qualified.mask_values_and_jacobian([m.row], original_r, original_j)
    assert inactive.tolist() == [2, 3]
    assert np.array_equal(r, [1., 2., 0., 0., 5., 6.])
    assert np.array_equal(j.toarray()[[0, 1, 4, 5]], original_j.toarray()[[0, 1, 4, 5]])
    assert j[2:4].nnz == 0 and j.shape == original_j.shape
    assert original_j.nnz == 18 and np.array_equal(original_r, np.arange(6.) + 1)
    metrics = qualified.review_metrics(m.row, [5., 1000., 13.])
    assert metrics['source_uv_count'] == 3 and metrics['active_FIT_accuracy_uv_count'] == 2
    assert metrics['mean_native_px'] == 9. and metrics['max_native_px'] == 13.
    assert metrics['inactive_zero_residual_is_accuracy_measurement'] is False
    assert qualified.contract['fresh_source_profiles_and_J_required'] is True
    assert qualified.contract['scalar_slots_removed'] == 0


def test_all_excluded_has_no_fake_zero_accuracy(tmp_path):
    m = materials(tmp_path, excluded=[0, 1, 2])
    metrics = load(m).review_metrics(m.row, [5., 1000., 13.])
    assert metrics['active_FIT_accuracy_uv_count'] == 0
    assert metrics['mean_native_px'] is None and metrics['p90_native_px'] is None
    assert metrics['max_native_px'] is None


@pytest.mark.parametrize('changed', ['parent', 'manifest', 'RGB', 'UV', 'sigma', 'HELD_to_FIT'])
def test_source_identity_changes_fail_closed(tmp_path, changed):
    m = materials(tmp_path)
    if changed == 'parent':
        m.parent_path.write_text(m.parent_path.read_text() + ' ', encoding='utf8')
    elif changed == 'manifest':
        m.manifest_path.write_text(m.manifest_path.read_text() + ' ', encoding='utf8')
    elif changed == 'RGB':
        m.raw.write_bytes(b'changed evidence')
    elif changed == 'UV':
        m.derived['rows'][0]['uv'][1][0] += 1.
    elif changed == 'sigma':
        m.derived['rows'][0]['sigma_px'][1] = 99.
    else:
        m.parent['rows'][0]['split'] = 'holdout'
        write_json(m.parent_path, m.parent)
        m.manifest['original_packet']['sha256'] = file_sha256(m.parent_path)
        resign_manifest(m)
    with pytest.raises(ValueError):
        load(m)


@pytest.mark.parametrize('case', ['held', 'direct_user', 'wrong_UV_hash', 'wrong_PTS', 'duplicate_index', 'bool_index', 'cost_based', 'missing_exact_RGB'])
def test_invalid_review_qualifications_rejected(tmp_path, case):
    m = materials(tmp_path, split='holdout' if case == 'held' else 'fit',
        user_annotation=case == 'direct_user')
    record = m.manifest['records'][0]
    if case == 'wrong_UV_hash':
        record['original_uv_float64_sha256'] = '0' * 64
    elif case == 'wrong_PTS':
        record['source_pts'] += 256
    elif case == 'duplicate_index':
        record['excluded_point_indices'] = [1, 1]
    elif case == 'bool_index':
        record['excluded_point_indices'] = [True]
    elif case == 'cost_based':
        record['review_basis'] = 'model_residual'
    elif case == 'missing_exact_RGB':
        record['native_rgb_evidence'][0]['source_pts'] += 256
    resign_manifest(m)
    with pytest.raises(ValueError):
        load(m)


def test_native_factor_block_and_review_share_point_qualification(tmp_path, monkeypatch):
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import native_ice_curve_source as module
    from calibration.solver_runtime import native_factor_dependency_fd as fd
    m = materials(tmp_path, held=True)
    block = SimpleNamespace(xyz_slots=np.array([50, 51, 52]),
        optics_chart=SimpleNamespace(indices=np.array([53]), mount_indices=np.array([54])),
        start=55, bundle=SimpleNamespace(index={229702144: 0}))
    current = SimpleNamespace(scene_unit_m=1.)
    state = SimpleNamespace(geometry=SimpleNamespace(current=current, parameter_count=50),
        geometry_parameter_indices=np.arange(50),
        structured_programme_coordinate_chart=SimpleNamespace(core_count=60),
        original=SimpleNamespace(aliases={229702144: 229702144}), bypts={229702144: block})
    factor = module.NativeIceCurveFactor(state, m.derived_path, 60)
    original = module.NativeIceCurveFactor(state, m.parent_path, 60)
    monkeypatch.setattr(module.NativeIceCurveFactor, '_cameras', lambda self, q: {229702144: np.zeros(25)})
    monkeypatch.setattr(module.NativeIceCurveFactor, '_world', lambda self, q, curve: np.array([[q[12], 0., 0.], [1., 1., 0.]]))
    def source_error(camera, world, uv, with_jacobian=False):
        return np.asarray(uv).reshape(-1) + world[0, 0]
    monkeypatch.setattr(module, 'curve_vector_error_full_brown', source_error)
    monkeypatch.setattr(module, '_curve_error_on_association',
        lambda camera, world, uv, association=None: (source_error(camera, world, uv), None))
    monkeypatch.setattr(module, 'positive_pivot_height_columns', lambda blocks: np.array([52]))
    def finite_difference(owner, q, rows, active, world_indices, base, evaluate, height_columns):
        assert sum(2 * len(row['uv']) for row in rows) == len(base)
        owner._evaluated_fit_uv_count = sum(len(row['uv']) for row in rows)
        matrix = np.empty((len(base), len(active)))
        groups = np.arange(len(rows))
        for i, column in enumerate(active):
            trial = q.copy(); trial[column] += 1e-5
            matrix[:, i] = (evaluate(trial, groups) - base) / 1e-5
        return matrix
    monkeypatch.setattr(fd, 'native_factor_dependency_jacobian', finite_difference)
    q = np.zeros(60)
    before = original.block(q, True)
    after = factor.block(q, True, original_start=100)
    kept = [0, 1, 4, 5]
    assert after.r.shape == (4,) and after.J.shape == (4, 60)
    assert np.array_equal(after.scales, before.scales[kept])
    assert np.array_equal(after.scales, factor.scales)
    assert np.array_equal(after.quadratic, before.quadratic[kept])
    assert np.array_equal(after.r, before.r[kept])
    assert np.array_equal(after.J.toarray(), before.J.toarray()[kept])
    assert after.row_ledger['removed_scalar_rows'] == 2
    assert after.row_ledger['original_stop'] == 104
    assert factor._qualified_active_fit_view()._evaluated_fit_uv_count == 2
    assert after.row_ledger['active_FIT_original_point_map'][0]['retained_original_point_indices'] == [0, 2]
    # Cached original arrays must never leak the excluded observations back.
    assert np.array_equal(factor.block(q, False).r, after.r)
    report = factor.review(q)['rows'][0]
    expected = np.linalg.norm(np.array(m.row['uv'])[[0, 2]], axis=1)
    assert report['mean_native_px'] == float(expected.mean())
    assert report['excluded_point_indices'] == [1]
    # Guard/raw operator remains original, independent of this FIT eligibility.
    assert np.array_equal(factor._residual(q, factor.rows), original._residual(q, original.rows))
    assert factor.rows[0]['sigma'].tolist() == m.row['sigma_px']
    assert factor.review(q)['rows'][1] == original.review(q)['rows'][1]
    # A separately selected human/suffix factor cannot inherit a full-packet
    # compact cache. Exact row membership decides its new view.
    from copy import copy
    selected = copy(factor)
    selected.rows = [factor.held[0]]
    assert selected._qualified_active_fit_view() is None
    # Existing append-only cache migration must reject this target change;
    # it cannot reuse a source normal/J produced before the qualification.
    from structured_programme_source_candidate.source_block_target_extension_replay import _native_append_plan
    family = 'additional_real_native_ice_curves'
    with pytest.raises(ValueError, match='operator/layout/weight domain changed'):
        _native_append_plan({family: original.contract}, {family: factor.contract}, (family,))
