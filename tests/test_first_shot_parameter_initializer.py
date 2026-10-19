import hashlib
import json

import numpy as np
import pytest

from calibration.program_tracking.first_shot_parameter_initializer import read_parameter_initializer, SCHEMA
from calibration.program_tracking.first_shot_source_transition import reference


def case(tmp_path, *, corrupt_aux=False, accepted_claim=False, unlisted_edit=False):
    def document(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding='utf8')
        return reference(path)
    base = document('old_binding.json', {'old': True})
    revised = document('new_binding.json', {'new': True})
    source = document('transition.json', {'base_binding': base, 'revised_binding': revised})
    parent_q = np.array([1., 2., 3., 4.])
    q = parent_q.copy(); q[1] = 5.
    if unlisted_edit:
        q[2] = 6.
    token = lambda values: hashlib.sha256(values.tobytes()).hexdigest()
    aux = np.array([12, 34], np.int64)
    parent = tmp_path / 'canonical.npz'
    np.savez(parent, q=parent_q, aux__native_pts=aux, damping=np.array(.125),
        next_line_search_fraction=np.array(.25), scales=np.ones(2), quadratic=np.zeros(2, bool))
    prior = document('current.json', {'qaug_sha256': token(parent_q), 'binding': base})
    candidate = tmp_path / 'candidate.npz'
    np.savez(candidate, q=q, parent_q=parent_q)
    proof = document('active_proof.json', dict(passed=True, candidate_q_sha256=token(q),
        parent_q_sha256=token(parent_q), parent_report=prior, candidate=reference(candidate),
        changed_parameter_indices=[1], all_active_target_world_vertices_byte_exact=True,
        all_source_observed_row_camera25_byte_exact=True,
        source_UV_ID_PTS_labels_order_immutable_inventory_byte_exact=True,
        physical_guards=dict(parent=dict(passed=True), candidate=dict(passed=True))))
    path = tmp_path / 'initializer.npz'
    np.savez(path, q=q, aux__native_pts=aux+int(corrupt_aux), damping=np.array(.125),
        next_line_search_fraction=np.array(.25), identity_json=np.array(json.dumps(dict(
            binding=base, parameter_initializer_only=True, accepted_state_claimed=accepted_claim))))
    recipe = document('initializer.json', dict(schema=SCHEMA, initializer=reference(path),
        source_transition=source, accepted_parent_report=prior, accepted_parent_checkpoint=reference(parent),
        rescue_parameter_candidate=reference(candidate), rescue_input_equivalence=proof,
        initializer_q_sha256=token(q), accepted_parent_q_sha256=token(parent_q), parameter_count=4,
        modified_from_accepted_parent_indices=[1], saved_damping=.125, saved_line_search_fraction=.25,
        initialized_parameter_state_is_not_an_accepted_old_source_checkpoint=True,
        borrow_old_source_J_or_r=False, new_source_initial_r_J_world_and_all_guards_required=True,
        original_source_binding=base, revised_source_binding=revised))
    return recipe['path'], source, q, parent_q


def test_explicit_edit_retains_accepted_aux_but_is_not_an_old_source_resume(tmp_path):
    path, source, q, parent = case(tmp_path)
    _, _, arrays, old_q, receipt = read_parameter_initializer(path, source_transition=source)
    np.testing.assert_array_equal(arrays['q'], q)
    np.testing.assert_array_equal(old_q, parent)
    assert receipt['changed_parameter_indices'] == [1]
    assert receipt['accepted_old_source_resume_claimed'] is False
    assert receipt['initial_source_r_J_and_world_cache_reuse'] is False


@pytest.mark.parametrize('change,message', [
    ({'corrupt_aux': True}, 'changed accepted auxiliary'),
    ({'accepted_claim': True}, 'without claiming acceptance'),
    ({'unlisted_edit': True}, 'active-input proof'),
])
def test_parent_aux_acceptance_and_exact_intervening_edit_are_independently_enforced(tmp_path, change, message):
    path, _, _, _ = case(tmp_path, **change)
    with pytest.raises(ValueError, match=message):
        read_parameter_initializer(path)


def test_initializer_cannot_be_reused_for_another_source_transition(tmp_path):
    path, _, _, _ = case(tmp_path)
    with pytest.raises(ValueError, match='another source transition'):
        read_parameter_initializer(path, source_transition={'path': 'foreign', 'sha256': 'foreign'})
