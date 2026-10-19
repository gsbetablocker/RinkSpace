"""Same-state truth transport for authenticated original-source positions."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from . import truth_source_location_revision as common
from .original_measured_transition_locations import CONTRACT_KEY, validate_packet

PROOF_SCHEMA = 'rinkspace.same-state-original-location-revision-truth-proof.v1'
OPTION = '--original-measured-transition-location-source'


def prepare_original_location_revision(truth_state_path, source_options, packet_path, output,
        *, base_operator_revision_proof=None):
    _, _, contract = validate_packet(packet_path)
    state = common._read(truth_state_path)
    proof = dict(schema=PROOF_SCHEMA, original_truth_state=common.reference(truth_state_path),
        measured_location_packet=common.reference(packet_path),
        expected_installed_contract=contract, complete_q_sha256=state['complete_q_sha256'],
        original_source_target=state['source_target'],
        protected_group_count=sum(len(f['group_ids']) for f in state['families'].values()),
        original_group_ids_caps_uncertainties_preserved=True, fresh_normal_and_J_required=True,
        base_operator_revision_proof=None if base_operator_revision_proof is None
            else common.reference(base_operator_revision_proof))
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    ref = common._write(output / 'original_location_revision_truth_proof.json', proof)
    options = deepcopy(source_options); options[OPTION] = str(Path(packet_path).resolve())
    options['--truth-restoration-source-location-revision-proof'] = ref['path']
    return dict(source_options=options, proof_reference=ref)


def authenticate_original_location_revision(proof_path, state_path, q, manifest, factors):
    proof = common._read(proof_path)
    if (proof.get('schema') != PROOF_SCHEMA
            or common._checked(proof['original_truth_state']) != Path(state_path).resolve()
            or proof.get('original_group_ids_caps_uncertainties_preserved') is not True
            or proof.get('fresh_normal_and_J_required') is not True):
        raise ValueError('Explicit unchanged-state original-position proof required')
    _, _, contract = validate_packet(common._checked(proof['measured_location_packet']))
    if (not common._same(contract, proof.get('expected_installed_contract'))
            or not common._same(manifest.get(CONTRACT_KEY), contract)
            or not common._same(manifest['source_contract'].get(CONTRACT_KEY), contract)):
        raise ValueError('Actual measured-position source contract differs from the qualification')
    state = common._read(state_path)
    original = common._read(common._checked(state['source_input_manifest_reference']))
    q = np.asarray(q, np.float64)
    with np.load(common._checked(state['complete_state_reference']), allow_pickle=False) as arrays:
        if not np.array_equal(q, arrays['q']):
            raise ValueError('Original-position revision cannot change complete seed q')
    token = hashlib.sha256(q.tobytes()).hexdigest()
    if token != state['complete_q_sha256'] or token != proof['complete_q_sha256']:
        raise ValueError('Original-position revision q identity differs')
    if state['source_target'] != common._target(original) or proof['original_source_target'] != state['source_target']:
        raise ValueError('Original-position parent truth source target differs')
    if CONTRACT_KEY in original['source_contract'] or original.get(CONTRACT_KEY) is not None:
        raise ValueError('An installed original-position revision needs a separate explicit successor proof')
    # Only this one installed contract may be added. Existing native revisions
    # have their own exact source/authority transport; compose, never weaken it.
    base = deepcopy(manifest); base.pop(CONTRACT_KEY)
    base['source_contract'].pop(CONTRACT_KEY)
    base_proof = proof.get('base_operator_revision_proof')
    if base_proof is not None:
        path = common._checked(base_proof)
        from .truth_source_scanline_revision import PROOF_SCHEMA as scanline_schema
        if common._read(path).get('schema') not in (common.PROOF_SCHEMA, scanline_schema):
            raise ValueError('Only a separately authenticated native operator revision may compose')
        inherited = common.authenticate_location_revision(path, state_path, q, base, factors)
        resumed = inherited['packet']
    else:
        if not common._same(base['source_contract'], original['source_contract']):
            raise ValueError('Original-position revision changed another source operator or loss')
        for key in ('highest_direct_human_hard_truth', 'highest_user_confirmed_native_sources',
                'native_base_curve_integration'):
            if not common._same(manifest.get(key), original.get(key)):
                raise ValueError('Original-position revision changed protected support or integration: ' + key)
        resumed = deepcopy(state)
    if (manifest['full_parameters'] != original['full_parameters']
            or manifest['source_rows'] != original['source_rows']
            or manifest.get('source_jacobian_reuse', {}).get('enabled') is True):
        raise ValueError('Original-position revision requires unchanged layouts and fresh source Jacobians')
    for key in ('original_ice_pixel_correction', 'original_source_role_retirement',
            'original_source_point_qualification', 'original_absolute_input_compaction',
            'original_board_curve_integration', 'original_ice_curve_integration',
            'original_wide_curve_integration', 'original_ring_source_qualification'):
        if not common._same(manifest.get(key), original.get(key)):
            raise ValueError('Original-position revision changed protected original source input: ' + key)
    actual_factors = {k: f for k, f in factors.items() if f is not None}
    if set(actual_factors) != set(resumed['families']):
        raise ValueError('Original-position revision changed protected family inventory')
    count = 0
    for family, factor in actual_factors.items():
        record = resumed['families'][family]
        common._checked(record['authority'])
        if not common._same(record['authority'], factor.contract['authority']):
            raise ValueError('Protected source authority changed outside explicit native transport')
        ids = [r['record']['id'] if family == 'wide' else r['row']['id'] for r in factor.records]
        if ids != record['group_ids'] or len(set(ids)) != len(ids):
            raise ValueError('Original protected IDs or order changed')
        for field in ('limits', 'uncertainties'):
            values = np.asarray(record[field], float)
            if (values.shape != (len(ids),) or not np.isfinite(values).all() or np.any(values < 0)
                    or not np.array_equal(values, state['families'][family][field])):
                raise ValueError('Original complete truth caps/uncertainties changed')
        count += len(ids)
    if count != proof['protected_group_count']:
        raise ValueError('Original complete protected-group count differs')
    migration = dict(proof_reference=common.reference(proof_path),
        original_source_target=state['source_target'], actual_current_source_target=common._target(manifest),
        complete_q_sha256=token, protected_group_count=count,
        original_packet_row_indices=[r['original_packet_row_index'] for r in contract['records']],
        original_group_ids_caps_uncertainties_preserved=True, fresh_normal_and_J_required=True,
        old_truth_packet_not_rewritten=True, base_operator_revision_proof=base_proof)
    resumed['authenticated_source_location_revision'] = migration
    return dict(packet=resumed, migration=migration)
