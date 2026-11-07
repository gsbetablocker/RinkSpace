"""Archived objective identity checks with temporary source and candidate files."""
import json

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.programme_projection import fit_participation
from calibration.program_tracking.body_tracking.rebuild_prepare import digest


WEIGHTS = 'objective_block_weights_sha256'
POLICY = 'objective_policy_sha256'


@pytest.fixture
def objective(tmp_path):
    prepared = tmp_path / 'prepared'
    stage = tmp_path / 'stage'
    parameter = stage / 'final' / 'parameters.npz'
    parameter.parent.mkdir(parents=True)
    prepared.mkdir()
    evidence = prepared / 'source_evidence.json'
    fit_row = dict(id='blade-1', pts=256, foot=0, structure_feature='runner', source_status='reviewed')
    support = dict(pts=512, foot=1, phase='support')
    document = dict(
        full_source_skate_structures=dict(rows=[
            dict(fit_row, optimization_enabled=True, role='fit', uv=[[11, 12], [13, 14]]),
            dict(fit_row, id='held', optimization_enabled=True, role='holdout'),
            dict(fit_row, id='disabled', optimization_enabled=False, role='fit')]),
        source_support_evidence=[dict(support, optimization_enabled=True, role='fit')],
        unassigned_source_skate_sets=[dict(pts=768, branches=[dict(rows=[{}])])],
        additional_unassigned_source_episodes=[
            dict(id='default', groups=[dict(pts=1024, branches=[dict(rows=[dict(optimization_enabled=True, role='fit')])])]),
            dict(id='disabled', groups=[dict(pts=1280, branches=[])])])
    evidence.write_text(json.dumps(document), encoding='utf-8')
    weights = stage / 'objective_block_weights.npz'
    policy = stage / 'objective_policy.json'
    np.savez(weights, full_source_skate_structure=5., native_source_support_contact=1.,
             additional_unassigned_source_disabled=0.)
    policy.write_text('{"schema":"fixture"}', encoding='utf-8')
    saved = dict(source_evidence_sha256=digest(evidence), **{WEIGHTS: digest(weights), POLICY: digest(policy)})
    np.savez(parameter, **saved)
    return prepared, parameter, evidence, weights, policy, saved, fit_row, support


def snapshot(*paths):
    return {path: path.read_bytes() for path in paths if path.is_file()}


def test_bound_objective_keeps_rows_and_default_and_zero_weight_semantics(objective):
    prepared, parameter, evidence, weights, policy, saved, fit_row, support = objective
    before = snapshot(parameter, evidence, weights, policy)
    result = fit_participation(prepared, parameter)
    assert result == dict(
        description_zh='入拟合范围按本候选实际保存的目标登记；局部源描线不代表完整冰刀或已确认接冰。',
        source_evidence_sha256=saved['source_evidence_sha256'], objective_weights_sha256=saved[WEIGHTS],
        feature_rows=[fit_row], support_rows=[support], unassigned_source_sets=[
            dict(pts=768, feature_rows_count=1),
            dict(pts=1024, episode_id='default', feature_rows_count=1,
                 source_anatomical_identity='unknown', support_enabled=False)])
    assert snapshot(*before) == before

    np.savez(weights, full_source_skate_structure=0., native_source_support_contact=0.,
             additional_unassigned_source_disabled=0.)
    np.savez(parameter, **dict(saved, **{WEIGHTS: digest(weights)}))
    before = snapshot(parameter, evidence, weights, policy)
    assert fit_participation(prepared, parameter) == dict(
        result, objective_weights_sha256=digest(weights), feature_rows=[], support_rows=[])
    assert snapshot(*before) == before


@pytest.mark.parametrize('target', ['weights', 'policy'])
@pytest.mark.parametrize('change', ['modified', 'removed'])
def test_unbound_stage_file_fails_without_changing_inputs(objective, target, change):
    prepared, parameter, evidence, weights, policy, *_ = objective
    path = weights if target == 'weights' else policy
    if change == 'removed':
        path.unlink()
    elif target == 'weights':
        np.savez(weights, full_source_skate_structure=0.)
    else:
        policy.write_text('{"schema":"different"}', encoding='utf-8')
    before = snapshot(parameter, evidence, weights, policy)
    error = FileNotFoundError if change == 'removed' else ValueError
    with pytest.raises(error, match=path.name):
        fit_participation(prepared, parameter)
    assert snapshot(parameter, evidence, weights, policy) == before


@pytest.mark.parametrize('field', [WEIGHTS, POLICY])
@pytest.mark.parametrize('bad_value', [None, ['a'*64], 0, b'a'*64, 'z'*64])
def test_missing_or_invalid_objective_identity_fails(objective, field, bad_value):
    prepared, parameter, evidence, weights, policy, saved, *_ = objective
    if bad_value is None:
        del saved[field]
    else:
        saved[field] = bad_value
    np.savez(parameter, **saved)
    before = snapshot(parameter, evidence, weights, policy)
    with pytest.raises(ValueError, match=field):
        fit_participation(prepared, parameter)
    assert snapshot(*before) == before


def test_weight_rows_come_from_the_verified_snapshot(objective, monkeypatch):
    prepared, parameter, evidence, weights, policy, saved, fit_row, _ = objective
    unchanged = snapshot(parameter, evidence, policy)
    read_bytes = type(weights).read_bytes
    reads = []

    def replace_after_read(path):
        data = read_bytes(path)
        if path == weights:
            reads.append(path)
            if len(reads) == 1:
                np.savez(weights, full_source_skate_structure=0.)
        return data

    load_npz = np.load

    def reject_weight_path(file, *args, **kwargs):
        assert file not in (weights, str(weights))
        return load_npz(file, *args, **kwargs)

    monkeypatch.setattr(type(weights), 'read_bytes', replace_after_read)
    monkeypatch.setattr(np, 'load', reject_weight_path)
    result = fit_participation(prepared, parameter)
    assert reads == [weights]
    assert result['objective_weights_sha256'] == saved[WEIGHTS]
    assert result['feature_rows'] == [fit_row]
    assert snapshot(*unchanged) == unchanged
