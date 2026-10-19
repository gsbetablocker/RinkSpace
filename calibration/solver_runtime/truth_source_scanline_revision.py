"""Authenticate an existing fixed-axis RGB scan, preserving its source state.

This changes a measured-axis operator, never the original UV, working sigma,
physical curve, qualification, source role, truth cap or parameter vector.
The shared location-revision engine authenticates the actual startup target.
"""
from copy import deepcopy
from pathlib import Path

from . import truth_source_location_revision as common

MODE_FIELDS = ('curve_observation_kind', 'scanline_fixed_axis',
               'scanline_branch', 'scanline_location_measurement')
REVISION_KEY = 'source_scanline_operator_revision'
RECEIPT_SCHEMA = 'rinkspace.saved-RGB-axis-scanline-location-revision.v1'
PROOF_SCHEMA = 'rinkspace.same-state-source-scanline-revision-truth-proof.v1'
ICE = '--additional-ice-source'
CONTRACT_KEY = 'additional_real_native_ice_curves'
OBSERVATION_OPERATOR = 'Native full-Brown distance with explicitly qualified fixed-axis scanline intersections'
TRACE_OBJECTIVE = 'Original curve-vector distances and qualified measured-axis scanline distances normalized by original declared sigma'


def expected_scanline_contract(rows):
    """Shared immutable contract for reader and continuation authentication."""
    measured = [r for r in rows if r.get('curve_observation_kind') == 'axis_scanline_branch']
    return dict(observation='axis_scanline_branch', fixed_axis='y', branch='max_x',
        ids=[r['id'] for r in measured],
        measurement_receipts={str(common._checked(r['scanline_location_measurement'])):
            r['scanline_location_measurement']['sha256'] for r in measured},
        original_UV_sigma_roles_profiles_clock_unchanged=True,
        horizontal_location_residual_uses_original_sigma=True,
        inactive_second_scalar_preserves_source_row_count=True,
        no_nearest_foot_or_ROI_fallback=True)


def validate_scanline_revision(old_path, new_path, changed_row_indices):
    old, new = common._read(old_path), common._read(new_path)
    indices = list(changed_row_indices)
    if (not indices or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices))
            or any(i < 0 or i >= len(old['rows']) for i in indices)):
        raise ValueError('Distinct ordered original scanline row indices required')
    if len(old['rows']) != len(new['rows']):
        raise ValueError('Scanline revision cannot append or delete source rows')
    if not common._same(old, {k: v for k, v in new.items() if k != REVISION_KEY} | {'rows': old['rows']}):
        raise ValueError('Scanline revision changed original packet metadata or qualification')
    observed = []
    for index, (a, b) in enumerate(zip(old['rows'], new['rows'])):
        if not common._same(common._without_mode(a, MODE_FIELDS), common._without_mode(b, MODE_FIELDS)):
            raise ValueError('Scanline revision changed source UV/sigma/curve/role/profile or row order')
        if not common._same(a, b):
            observed.append(index)
            if (any(key in a for key in MODE_FIELDS)
                    or b.get('curve_observation_kind') != 'axis_scanline_branch'
                    or b.get('scanline_fixed_axis') != 'y' or b.get('scanline_branch') != 'max_x'
                    or a['split'] not in ('fit', 'holdout')):
                raise ValueError('Only an unbound original x scan may acquire its fixed-y right branch')
    if observed != indices:
        raise ValueError('Exact reviewed scanline-row inventory differs')
    meta = new.get(REVISION_KEY, {})
    if (meta.get('parent_packet_reference') != common.reference(old_path)
            or meta.get('changed_row_indices') != indices
            or meta.get('changed_row_ids') != [old['rows'][i]['id'] for i in indices]
            or meta.get('all_original_source_rows_except_four_scanline_fields_unchanged') is not True):
        raise ValueError('Exact original packet scanline-revision declaration required')
    receipt_ref = meta['receipt_reference']
    receipt = common._read(common._checked(receipt_ref))
    if (receipt.get('schema') != RECEIPT_SCHEMA
            or receipt.get('original_packet_reference') != common.reference(old_path)
            or receipt.get('source_only_RGB_replayed') is not True
            or receipt.get('source_images_written') != 0
            or receipt.get('model_or_camera_used_for_measurement') is not False):
        raise ValueError('Authenticated saved RGB scanline receipt required')
    common._checked(receipt['original_producer_reference'])
    common._checked(receipt['original_extraction_reference'])
    common._checked(receipt['qualification_script_reference'])
    records = receipt.get('rows', [])
    if [r['original_row_index'] for r in records] != indices:
        raise ValueError('RGB receipt must cover every revised scanline row exactly')
    for index, rec in zip(indices, records):
        a, b = old['rows'][index], new['rows'][index]
        if (rec.get('id') != a['id'] or rec.get('split') != a['split']
                or rec.get('source_row_canonical_json_sha256') != common._row_hash(a)
                or common._checked(rec['source_row_reference']) != Path(old_path).resolve()
                or b.get('scanline_location_measurement') != receipt_ref
                or rec.get('measured_source_row') != a
                or rec.get('physical_branch') != 'max_x'
                or rec.get('fixed_axis') != 'y'
                or rec.get('actual_RGB_review', {}).get('basis') != 'existing_native_RGB_opened_without_model'
                or a.get('extraction_input') != receipt['original_extraction_reference']
                or a.get('curve') != dict(kind='house_ring', side='F', row=3, ring='outer')):
            raise ValueError('Scanline receipt changed an original measured source row')
        if common._sha(a['source_raw_frame']) != a['source_file_sha256']:
            raise ValueError('Original scanline RGB bytes changed')
        support = a.get('source_search_support', [])
        if (len(support) != len(a['uv']) or len(a['sigma_px']) != len(a['uv'])
                or rec.get('replayed_original_UV_sigma_profiles_bitexact') is not True):
            raise ValueError('All original scanline samples and replayed profiles required')
        for point, sigma, sample in zip(a['uv'], a['sigma_px'], support):
            if (sample.get('source_scan_axis') != 'x' or sample.get('uv') != point
                    or sample.get('sigma_px') != sigma
                    or sample.get('manual_source_bracket', [None])[0] != point[1]):
                raise ValueError('Original source estimator must measure x at its exact fixed y')
    # Authenticate the unchanged point exclusions against their original packet.
    from .source_point_qualification import load_source_point_qualification
    load_source_point_qualification(old_path, old, source_family='additional_native_ice_curves')
    return old, new


def original_qualified_packet(packet_path, packet):
    """Feed the unchanged original packet to its strict qualification loader."""
    if REVISION_KEY not in packet:
        return packet
    meta = packet[REVISION_KEY]
    old_path = common._checked(meta['parent_packet_reference'])
    old, actual = validate_scanline_revision(old_path, packet_path, meta['changed_row_indices'])
    if not common._same(packet, actual):
        raise ValueError('Runtime scanline packet differs from its authenticated file')
    return old


def _revised_contract(contract, packet):
    result = deepcopy(contract)
    ordered = [r for split in ('fit', 'holdout') for r in packet['rows'] if r['split'] == split]
    result['axis_scanline_location_objective'] = expected_scanline_contract(ordered)
    result['observation_operator'] = OBSERVATION_OPERATOR
    result['trace_objective'] = TRACE_OBJECTIVE
    return result


def prepare_scanline_revision(source_options, new_ice_packet, output_dir, *,
                              changed_row_indices, truth_state_path):
    options = deepcopy(source_options)
    validate_scanline_revision(options[ICE], new_ice_packet, changed_row_indices)
    state = common._read(truth_state_path)
    original = common._read(common._checked(state['source_input_manifest_reference']))
    if state['source_target'] != common._target(original):
        raise ValueError('Original cap source target differs')
    ice = original[CONTRACT_KEY]
    if (Path(ice['packet_path']).resolve() != Path(options[ICE]).resolve()
            or ice['packet_sha256'] != common._sha(options[ICE])):
        raise ValueError('Scanline revision must start from the complete cap original ice target')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents, _ = common._rebound_documents(options, new_ice_packet,
                                            source_family='ice', mode_fields=MODE_FIELDS)
    names = {common.NATIVE: 'native_authority.json', common.CONFIRMED: 'confirmed_authority.json',
        common.GUARD: 'full_stroke_guard.json', common.CAP: 'stroke_caps.json',
        common.ENVELOPE: 'location_envelope.json', common.INTEGRATION: 'native_integration.json'}
    refs = {}
    for key in (common.NATIVE, common.CONFIRMED):
        refs[key] = common._write(output / names[key], documents[key])
    documents[common.GUARD]['primary_authority_reference'] = refs[common.NATIVE]
    refs[common.GUARD] = common._write(output / names[common.GUARD], documents[common.GUARD])
    common._bind_document_refs(documents, refs)
    for key in (common.CAP, common.ENVELOPE, common.INTEGRATION):
        refs[key] = common._write(output / names[key], documents[key])
    files = [dict(option_key=k, parent_reference=common.reference(options[k]),
                  derived_reference=refs[k]) for k in common.BOUND_KEYS]
    options.update({k: refs[k]['path'] for k in common.BOUND_KEYS})
    options[ICE] = str(Path(new_ice_packet).resolve())
    proof = dict(schema=PROOF_SCHEMA, original_truth_state=common.reference(truth_state_path),
        original_ice_packet=common.reference(source_options[ICE]),
        revised_ice_packet=common.reference(new_ice_packet), complete_q_sha256=state['complete_q_sha256'],
        original_source_target=state['source_target'], changed_row_indices=list(changed_row_indices),
        rebound_files=files, protected_group_count=sum(len(f['group_ids']) for f in state['families'].values()),
        original_group_ids_caps_uncertainties_preserved=True, fresh_normal_and_J_required=True)
    proof_ref = common._write(output / 'scanline_revision_truth_proof.json', proof)
    return dict(source_options=options, proof_reference=proof_ref)


def authenticate_scanline_revision(proof_path, state_path, q, manifest, factors):
    return common._authenticate_operator_revision(proof_path, state_path, q, manifest, factors,
        proof_schema=PROOF_SCHEMA, source_contract_key=CONTRACT_KEY,
        old_packet_key='original_ice_packet', new_packet_key='revised_ice_packet',
        validator=validate_scanline_revision, contract_revision=_revised_contract,
        source_family='ice', mode_fields=MODE_FIELDS)
