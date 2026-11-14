"""Sparse views authenticate saved rows; passing never grants physical eligibility."""
from copy import deepcopy

import pytest

from rinkspace.session.saved_row_views import row_sha256, validate_sparse_row_view


def sparse_view(indices=(1, 4)):
    rows = [dict(uv=[float(i), float(i + 10)], fit_eligible=False) for i in range(5)]
    parent_frame = dict(pts=256, shot_id='s01', rows=deepcopy(rows))
    parent_registration = dict(
        pts=256,
        patterns=[dict(registration=dict(status='whole_pattern_unresolved'),
                       source_readout=dict(queries=8, accepted=5))],
        all_actual_readouts=deepcopy(rows),
    )
    refs = dict(
        source_record='saved-source-record', registration_record='saved-registration-record',
        original_row_indices=list(indices),
        original_row_sha256=[row_sha256(rows[i]) for i in indices],
        selected_original_row_count=len(indices), original_raw_valid_rows=len(rows),
    )
    frame = deepcopy(parent_frame)
    frame.update(
        rows=[deepcopy(rows[i]) for i in indices],
        key_selection_parent_refs=deepcopy(refs), key_selection_plan='saved-selector',
        selection_derivation='saved_original_row_subset', key_shape_sampler='key_shape',
        source_profiles_resampled=False, new_RGB_measurements=False, new_registration_run=False,
    )
    registration = deepcopy(parent_registration)
    registration.update(
        all_actual_readouts=deepcopy(frame['rows']),
        original_full_unthinned_registration_record=refs['registration_record'],
        key_selection_parent_refs=deepcopy(refs), key_selection_plan='saved-selector',
        selection_derivation='saved_original_row_subset',
        all_actual_readouts_scope='selected_original_row_subset_in_this_sparse_view',
        original_source_readout_queries_and_accepted_preserved=True,
        new_registration_run=False, new_profile_run=False,
    )
    selector = dict(
        source_pts=256, shot_id='s01', source_record=refs['source_record'],
        registration_record=refs['registration_record'],
        selected_original_rows=[dict(row_index=i, row_sha=row_sha256(rows[i])) for i in indices],
    )
    return frame, registration, parent_frame, parent_registration, selector


def test_sparse_subset_keeps_original_full_attempt_accepted_count():
    records = sparse_view()
    before = deepcopy(records)

    result = validate_sparse_row_view(*records)

    assert records == before
    assert len(records[0]['rows']) == 2
    assert records[1]['patterns'][0]['source_readout'] == dict(queries=8, accepted=5)
    assert result['original_selected_readout_count'] == 5
    assert result['sparse_view_row_count'] == 2
    assert result['all_selected_rows_exact'] is True
    assert result['original_registration_diagnostics_exact'] is True
    assert result['full_attempt_accepted_counts_reinterpreted'] is False
    assert result['new_measurements'] is False
    assert result['physical_identity_qualified'] is False


def test_changed_uv_is_rejected_even_when_both_view_records_agree():
    frame, registration, *rest = sparse_view()
    frame['rows'][0]['uv'][0] += 0.25
    registration['all_actual_readouts'][0]['uv'][0] += 0.25

    with pytest.raises(ValueError, match='^sparse_view_original_row_changed$'):
        validate_sparse_row_view(frame, registration, *rest)


def test_reinterpreting_original_accepted_as_sparse_count_is_rejected():
    frame, registration, *rest = sparse_view()
    registration['patterns'][0]['source_readout']['accepted'] = 2

    with pytest.raises(ValueError, match='^sparse_view_changed_original_registration_diagnostics$'):
        validate_sparse_row_view(frame, registration, *rest)


def test_duplicate_original_index_is_rejected_with_consistent_selector_and_refs():
    with pytest.raises(ValueError, match='^sparse_view_duplicate_or_wrong_row_count$'):
        validate_sparse_row_view(*sparse_view((1, 1)))


@pytest.mark.parametrize('field,value', [('row_sha', '0' * 64), ('row_index', 2)])
def test_selector_hash_or_index_change_is_rejected(field, value):
    *records, selector = sparse_view()
    selector['selected_original_rows'][0][field] = value

    with pytest.raises(ValueError, match='^sparse_view_explicit_selection_differs$'):
        validate_sparse_row_view(*records, selector)


@pytest.mark.parametrize('field,ref_field,value', [
    ('row_sha', 'original_row_sha256', '0' * 64),
    ('row_index', 'original_row_indices', 2),
])
def test_coordinated_selector_and_ref_changes_still_fail_parent_row_check(field, ref_field, value):
    frame, registration, parent_frame, parent_registration, selector = sparse_view()
    selector['selected_original_rows'][0][field] = value
    frame['key_selection_parent_refs'][ref_field][0] = value
    registration['key_selection_parent_refs'][ref_field][0] = value

    with pytest.raises(ValueError, match='^sparse_view_original_row_changed$'):
        validate_sparse_row_view(frame, registration, parent_frame, parent_registration, selector)
