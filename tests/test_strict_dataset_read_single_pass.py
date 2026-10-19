"""FIT/HELD array and source-locator equivalence for the one-pass consumer."""
from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.compact_source_json import pack_source_json, unpack_source_json
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_fit_dataset import SCHEMA, partition_support
from calibration.program_tracking.strict_source_fit_dataset_stream import iter_dataset_rows
from calibration.program_tracking.test_strict_source_disjoint_merge import _fixture_pack
from calibration.solver_runtime.strict_source_dataset_adapter import compact_dataset_rows
import calibration.solver_runtime.strict_dataset_read_runtime as fast


def _ref(path):
    return dict(path=str(path), sha256=file_sha(path))


def _manifest(tmp_path, *, promote=False, loss_field=None, corrupt_selection=False):
    refs = [_fixture_pack(tmp_path/'fit'),
        _fixture_pack(tmp_path/'held', pts=200, split='held', board=True)]
    cuts = []; counts = dict(fit=0, held=0)
    for number, reference in enumerate(refs):
        if loss_field and number == 0:
            with np.load(reference['path'], allow_pickle=False) as archive:
                values = {k:archive[k].copy() for k in archive.files}
            headers = unpack_source_json(values['rows_json'])
            headers[0][loss_field] = 2.
            values['rows_json'] = pack_source_json(headers)
            path = tmp_path/'explicit_loss.npz'
            np.savez_compressed(path, **values); reference = _ref(path)
        pack = fast.StrictDatasetValidatedPack(reference['path'], expected_sha256=reference['sha256'])
        roles = deepcopy(pack.rows)
        if promote and number == 0:
            roles[0]['split'] = 'held'
        partition = partition_support(roles, pack.values['row_offsets'], pack.values['sigma_px'])
        for role in counts:
            counts[role] += len(partition[role+'_point_indices'])
        if corrupt_selection and number == 0:
            partition['fit_point_indices'] = np.array([0], np.int64)
        path = tmp_path/('index_%s.npz' % number)
        kwargs = dict(row_offsets=pack.values['row_offsets'], **partition)
        if promote and number == 0:
            kwargs['promoted_held_rows'] = np.array([0], np.int64)
        np.savez_compressed(path, **kwargs)
        cuts.append(dict(source_pack=reference, support_index=_ref(path)))
    path = tmp_path/'dataset.json'
    path.write_text(json.dumps(dict(schema=SCHEMA, source=pack.metadata['source'],
        cuts=cuts, fit_points=counts['fit'], held_points=counts['held'])), encoding='utf8')
    return _ref(path)


def _assert_groups_identical(expected, actual):
    assert len(actual) == len(expected)
    for before, after in zip(expected, actual):
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == after[key].dtype, key
                assert value.shape == after[key].shape, key
                assert value.tobytes() == after[key].tobytes(), key
            else:
                assert value == after[key], key


@pytest.mark.parametrize('promote', [False, True])
def test_both_roles_all_UV_sigma_payload_paths_and_locators_equal_original(tmp_path, monkeypatch, promote):
    reference = _manifest(tmp_path, promote=promote)
    old = {role:compact_dataset_rows(iter_dataset_rows(reference, split=role), split=role)
        for role in ('fit', 'held')}
    original = fast.StrictDatasetValidatedPack; visits = []
    def counted(path, **kwargs):
        visits.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(fast, 'StrictDatasetValidatedPack', counted)
    fit, held, receipt = fast.load_strict_dataset_rows_once(reference)
    _assert_groups_identical(old['fit'], fit); _assert_groups_identical(old['held'], held)
    assert len(visits) == 2 and len(set(visits)) == 2
    assert receipt['full_source_pack_passes'] == 1
    assert np.isnan(held[-1]['sigma']).all()
    assert held[-1]['source_locators'][0]['local_point_indices'] == [0, 1, 2]


def test_missing_FIT_selection_cannot_be_trusted_as_flag_or_mask(tmp_path):
    reference = _manifest(tmp_path, corrupt_selection=True)
    for consume in (lambda:list(iter_dataset_rows(reference)),
                    lambda:fast.load_strict_dataset_rows_once(reference)):
        with pytest.raises(ValueError, match='selection changed'):
            consume()


@pytest.mark.parametrize('field', ['quadratic', 'loss_weights', 'huber_delta',
    'source_integration_mass_rule', 'source_integration_weights'])
def test_future_source_loss_fields_survive_thinning_until_explicit_rejection(tmp_path, field):
    reference = _manifest(tmp_path, loss_field=field)
    for consume in (lambda:compact_dataset_rows(iter_dataset_rows(reference), split='fit'),
                    lambda:fast.load_strict_dataset_rows_once(reference)):
        with pytest.raises(ValueError, match='Explicit dataset loss policy'):
            consume()


def test_reused_template_cannot_mask_non_integer_boolean_row_clock(tmp_path):
    from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
    reference = _fixture_pack(tmp_path/'clock')
    with np.load(reference['path'], allow_pickle=False) as archive:
        values = {k:archive[k].copy() for k in archive.files}
    headers = unpack_source_json(values['rows_json'])
    proofs = unpack_source_json(values['source_proof_templates_json'])
    metadata = json.loads(str(values['metadata_json'].item()))
    first = headers[0]
    first['actual_pts'] = 1; first['temporal_evidence']['current_actual_pts'] = 1
    boolean = deepcopy(first); boolean['actual_pts'] = True
    headers = [first, boolean]
    for proof in proofs:
        proof['actual_pts'] = 1
    for key in ('uv', 'raw_candidate_uv', 'sigma_px', 'source_sides_uv',
            'source_side_quantiles_uv', 'source_proof_template_index', 'original_point_indices'):
        values[key] = np.concatenate([values[key], values[key]])
    values['row_offsets'] = np.array([0, 3, 6], np.int64)
    values['source_frame_actual_pts'][0] = 1
    metadata['expected_actual_pts'][0] = 1
    metadata['processed_actual_pts'] = [1]
    values['rows_json'] = pack_source_json(headers)
    values['source_proof_templates_json'] = pack_source_json(proofs)
    values['metadata_json'] = json.dumps(metadata)
    path = tmp_path/'cached_boolean_clock.npz'; np.savez_compressed(path, **values)
    reference = _ref(path)
    for reader in (StrictAbsoluteSourcePack, fast.StrictDatasetValidatedPack):
        with pytest.raises(ValueError, match='integer actual exposure'):
            reader(reference['path'], expected_sha256=reference['sha256'])


@pytest.mark.parametrize('field', ['point_support_bounds_xy_px', 'source_tangent_interval_px'])
def test_explicit_source_support_fields_cannot_disappear_before_operator_gate(tmp_path, field):
    reference = _manifest(tmp_path, loss_field=field)
    for consume in (lambda:compact_dataset_rows(iter_dataset_rows(reference), split='fit'),
                    lambda:fast.load_strict_dataset_rows_once(reference)):
        with pytest.raises(ValueError, match='typed source operator'):
            consume()
