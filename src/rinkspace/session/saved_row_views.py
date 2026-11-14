"""Authenticate sparse views without changing original measurement diagnostics.

File identity and committed-record checks belong to the caller.  This module
checks the parsed records: every kept row is exactly one explicitly selected
parent row, while registration attempts and their original accepted counts
retain their original scope.  It never grants source or physical eligibility.
"""
from __future__ import annotations

import hashlib
import json


def row_sha256(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_sparse_row_view(frame, registration, parent_frame,
                             parent_registration, selector):
    """Return lineage counts only after exact, lossless subset authentication."""
    frame_additions = {
        'key_selection_parent_refs', 'key_selection_plan', 'selection_derivation',
        'key_shape_sampler', 'source_profiles_resampled', 'new_RGB_measurements',
        'new_registration_run',
    }
    registration_additions = {
        'original_full_unthinned_registration_record', 'key_selection_parent_refs',
        'key_selection_plan', 'selection_derivation', 'all_actual_readouts_scope',
        'original_source_readout_queries_and_accepted_preserved',
        'new_registration_run', 'new_profile_run',
    }
    _require(set(frame) == set(parent_frame) | frame_additions,
             'sparse_view_frame_fields_differ')
    _require(set(registration) == set(parent_registration) | registration_additions,
             'sparse_view_registration_fields_differ')
    _require(all(frame[key] == value for key, value in parent_frame.items()
                 if key != 'rows'), 'sparse_view_changed_parent_frame_field')
    _require(all(registration[key] == value
                 for key, value in parent_registration.items()
                 if key != 'all_actual_readouts'),
             'sparse_view_changed_original_registration_diagnostics')
    _require(parent_frame['rows'] == parent_registration['all_actual_readouts'],
             'parent_chosen_readouts_differ')
    _require(frame['rows'] == registration['all_actual_readouts'],
             'sparse_view_source_and_registration_rows_differ')
    _require(frame['pts'] == registration['pts'] == selector['source_pts']
             and frame['shot_id'] == selector['shot_id'],
             'sparse_view_selector_source_time_differs')
    _require(frame['source_profiles_resampled'] is False
             and frame['new_RGB_measurements'] is False
             and frame['new_registration_run'] is False
             and registration['new_registration_run'] is False
             and registration['new_profile_run'] is False
             and registration['original_source_readout_queries_and_accepted_preserved'] is True,
             'sparse_view_not_a_saved_measurement_subset')
    _require(registration['all_actual_readouts_scope']
             == 'selected_original_row_subset_in_this_sparse_view',
             'sparse_view_scope_not_explicit')
    refs = frame['key_selection_parent_refs']
    _require(refs == registration['key_selection_parent_refs'],
             'sparse_view_parent_references_differ')
    _require(registration['original_full_unthinned_registration_record']
             == refs['registration_record'], 'sparse_view_parent_registration_differs')
    _require(selector['source_record'] == refs['source_record']
             and selector['registration_record'] == refs['registration_record'],
             'sparse_view_selector_parent_record_differs')
    items = selector['selected_original_rows']
    indices = [item['row_index'] for item in items]
    hashes = [item['row_sha'] for item in items]
    _require(indices == refs['original_row_indices']
             and hashes == refs['original_row_sha256'],
             'sparse_view_explicit_selection_differs')
    _require(len(indices) == len(set(indices)) == len(hashes) == len(set(hashes))
             == len(frame['rows']) == refs['selected_original_row_count']
             and refs['original_raw_valid_rows'] == len(parent_frame['rows']),
             'sparse_view_duplicate_or_wrong_row_count')
    for row, index, expected in zip(frame['rows'], indices, hashes):
        _require(type(index) is int and 0 <= index < len(parent_frame['rows']),
                 'sparse_view_original_row_index_invalid')
        _require(row == parent_frame['rows'][index]
                 and row_sha256(row) == expected, 'sparse_view_original_row_changed')
    return dict(original_selected_readout_count=len(parent_frame['rows']),
                sparse_view_row_count=len(frame['rows']),
                all_selected_rows_exact=True, original_registration_diagnostics_exact=True,
                full_attempt_accepted_counts_reinterpreted=False,
                new_measurements=False, physical_identity_qualified=False)
