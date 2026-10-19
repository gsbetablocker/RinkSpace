"""Authenticate an explicit parameter edit without claiming an accepted resume.

The accepted parent is still authenticated by the ordinary complete replay
loader. This module binds the intervening safe NPZ parameter candidate and its
active-input proof; it never imports old residuals, Jacobians or world caches.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from .first_shot_source_transition import reference, read

SCHEMA = 'rinkspace.explicit-rescue-parameter-initializer.v1'


def _q_token(q):
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()


def read_parameter_initializer(path, *, source_transition=None):
    ref = reference(path)
    recipe = read(ref)
    if (recipe.get('schema') != SCHEMA
            or recipe.get('initialized_parameter_state_is_not_an_accepted_old_source_checkpoint') is not True
            or recipe.get('borrow_old_source_J_or_r') is not False
            or recipe.get('new_source_initial_r_J_world_and_all_guards_required') is not True):
        raise ValueError('Explicit fresh-source parameter initializer required')
    names = ('initializer', 'source_transition', 'accepted_parent_report',
        'accepted_parent_checkpoint', 'rescue_parameter_candidate', 'rescue_input_equivalence')
    for name in names:
        value = recipe[name]
        if reference(value['path']) != value:
            raise ValueError('Parameter initializer reference changed: ' + name)
    if source_transition is not None and recipe['source_transition'] != source_transition:
        raise ValueError('Parameter initializer belongs to another source transition')
    with np.load(recipe['initializer']['path'], allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(recipe['accepted_parent_checkpoint']['path'], allow_pickle=False) as saved:
        parent = {key: saved[key].copy() for key in saved.files}
    with np.load(recipe['rescue_parameter_candidate']['path'], allow_pickle=False) as saved:
        candidate = {key: saved[key].copy() for key in ('q', 'parent_q')}
    q = arrays['q']
    if (q.shape != (recipe['parameter_count'],) or q.dtype != np.float64
            or not np.isfinite(q).all() or parent['q'].shape != q.shape
            or parent['q'].dtype != q.dtype or not np.isfinite(parent['q']).all()
            or not np.array_equal(q.view(np.uint64), candidate['q'].view(np.uint64))
            or not np.array_equal(parent['q'].view(np.uint64), candidate['parent_q'].view(np.uint64))):
        raise ValueError('Complete explicit candidate and accepted parent parameters differ')
    token, parent_token = _q_token(q), _q_token(parent['q'])
    changed = np.flatnonzero(q.view(np.uint64) != parent['q'].view(np.uint64)).tolist()
    proof = read(recipe['rescue_input_equivalence'])
    prior = read(recipe['accepted_parent_report'])
    if (token != recipe['initializer_q_sha256'] or parent_token != recipe['accepted_parent_q_sha256']
            or prior['qaug_sha256'] != parent_token
            or proof.get('passed') is not True
            or proof.get('candidate_q_sha256') != token or proof.get('parent_q_sha256') != parent_token
            or proof.get('candidate') != recipe['rescue_parameter_candidate']
            or proof.get('parent_report') != recipe['accepted_parent_report']
            or proof.get('changed_parameter_indices') != changed
            or recipe['modified_from_accepted_parent_indices'] != changed):
        raise ValueError('Parameter edit differs from its admitted active-input proof')
    for key in ('all_active_target_world_vertices_byte_exact',
                'all_source_observed_row_camera25_byte_exact',
                'source_UV_ID_PTS_labels_order_immutable_inventory_byte_exact'):
        if proof.get(key) is not True:
            raise ValueError('Missing complete active-input equivalence: ' + key)
    if not all(proof['physical_guards'][key].get('passed') is True for key in ('parent', 'candidate')):
        raise ValueError('Complete physical guards required for parameter initializer')
    identity = json.loads(str(arrays['identity_json']))
    if (identity.get('parameter_initializer_only') is not True
            or identity.get('accepted_state_claimed') is not False
            or identity.get('binding') != recipe['original_source_binding']
            or prior['binding'] != recipe['original_source_binding']):
        raise ValueError('Initializer must identify its old chart without claiming acceptance')
    aux_names = {key for key in parent if key.startswith('aux__')}
    if {key for key in arrays if key.startswith('aux__')} != aux_names:
        raise ValueError('Initializer auxiliary chart inventory changed')
    for key in (*sorted(aux_names), 'damping', 'next_line_search_fraction'):
        if not np.array_equal(arrays[key], parent[key]):
            raise ValueError('Initializer changed accepted auxiliary state: ' + key)
    if (float(arrays['damping']) != recipe['saved_damping']
            or float(arrays['next_line_search_fraction']) != recipe['saved_line_search_fraction']
            or any(key in arrays for key in ('r', 'J', 'scales', 'quadratic'))):
        raise ValueError('Explicit initializer must carry parameters/auxiliary state only')
    receipt = dict(schema=SCHEMA, manifest=ref, initializer=deepcopy(recipe['initializer']),
        accepted_parent_report=deepcopy(recipe['accepted_parent_report']),
        accepted_parent_checkpoint=deepcopy(recipe['accepted_parent_checkpoint']),
        active_input_equivalence=deepcopy(recipe['rescue_input_equivalence']),
        parameter_q_sha256=token, accepted_parent_q_sha256=parent_token,
        changed_parameter_indices=changed, accepted_old_source_resume_claimed=False,
        parameter_candidate_and_all_accepted_auxiliary_arrays_exact=True,
        initial_source_r_J_and_world_cache_reuse=False, fresh_current_full_source_required=True,
        producer=reference(__file__))
    return ref, recipe, arrays, parent['q'], receipt
