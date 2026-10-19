"""Bind an authenticated location-operator revision to unchanged truth caps.

Only the two existing board location-mode fields may change.  This is a
source-target migration, so a previous normal or source Jacobian is invalid.
The helper never evaluates a model, adds observations, or changes a cap.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

MODE_FIELDS = ('boundary_location_observation', 'boundary_location_measurement')
REVISION_KEY = 'source_location_operator_revision'
PROOF_SCHEMA = 'rinkspace.same-state-source-location-revision-truth-proof.v1'
RECEIPT_SCHEMA = 'rinkspace.saved-RGB-transition-midpoint-location-revision.v1'
LOCATION_RECEIPT_SCHEMA = 'rinkspace.saved-RGB-transition-location-revision.v1'
MEASURED_MODES = ('measured_transition_midpoint', 'measured_transition_location')
BOARD = '--additional-board-source'
NATIVE = '--direct-human-truth-authority'
CONFIRMED = '--user-confirmed-source-truth-authority'
GUARD = '--direct-human-full-stroke-guard'
CAP = '--direct-human-stroke-cap-support'
ENVELOPE = '--practical-human-location-envelope'
INTEGRATION = '--native-base-curve-integration-policy'
BOUND_KEYS = (NATIVE, CONFIRMED, GUARD, CAP, ENVELOPE, INTEGRATION)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf8')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=_sha(path))


def _checked(ref):
    path = Path(ref['path']).resolve()
    if _sha(path) != ref['sha256']:
        raise ValueError('Location revision referenced bytes changed')
    return path


def _same(a, b):
    return canonical(a) == canonical(b)


def _without_mode(row, mode_fields=MODE_FIELDS):
    return {k: v for k, v in row.items() if k not in mode_fields}


def _row_hash(row):
    return hashlib.sha256(canonical(row)).hexdigest()


def _target(manifest):
    # Match the existing driver source_target recipe (default ensure_ascii).
    raw = json.dumps(manifest['source_contract'], sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf8')
    return dict(parameter_count=int(manifest['full_parameters']),
                source_rows=int(manifest['source_rows']),
                source_contract_sha256=hashlib.sha256(raw).hexdigest())


def validate_location_revision(old_path, new_path, changed_row_indices):
    old, new = _read(old_path), _read(new_path)
    indices = list(changed_row_indices)
    if (not indices or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices))
            or any(i < 0 or i >= len(old['rows']) for i in indices)):
        raise ValueError('Distinct ordered original row indices required')
    if len(old['rows']) != len(new['rows']):
        raise ValueError('Location revision cannot append or delete rows')
    if not _same(old, {k: v for k, v in new.items() if k != REVISION_KEY} | {'rows': old['rows']}):
        raise ValueError('Location revision changed original packet metadata')
    observed = []
    for i, (a, b) in enumerate(zip(old['rows'], new['rows'])):
        if not _same(_without_mode(a), _without_mode(b)):
            raise ValueError('Location revision changed UV/sigma/support/role or row order')
        if not _same(a, b):
            observed.append(i)
            if (a['split'] != 'fit' or a.get(MODE_FIELDS[0], 'normal_interval') != 'normal_interval'
                    or b.get(MODE_FIELDS[0]) not in MEASURED_MODES):
                raise ValueError('Only an existing FIT interval may become a measured location')
    if observed != indices:
        raise ValueError('Exact reviewed location-row inventory differs')
    meta = new.get(REVISION_KEY, {})
    if (meta.get('parent_packet_reference') != reference(old_path)
            or meta.get('changed_row_indices') != indices
            or meta.get('changed_row_ids') != [old['rows'][i]['id'] for i in indices]
            or meta.get('all_original_source_rows_except_two_location_fields_unchanged') is not True):
        raise ValueError('Exact original packet location-revision declaration required')
    receipt_ref = meta['receipt_reference']
    receipt = _read(_checked(receipt_ref))
    receipt_modes = {RECEIPT_SCHEMA: 'measured_transition_midpoint',
                     LOCATION_RECEIPT_SCHEMA: 'measured_transition_location'}
    receipt_mode = receipt_modes.get(receipt.get('schema'))
    if receipt_mode is None or any(new['rows'][i].get(MODE_FIELDS[0]) != receipt_mode for i in indices):
        raise ValueError('Authenticated saved RGB location receipt must match its observation mode')
    rows = receipt.get('rows', [])
    if [r['original_row_index'] for r in rows] != indices:
        raise ValueError('RGB receipt must cover exactly the reviewed original rows')
    required = ('id', 'split', 'uv', 'sigma_px', 'normal', 'normal_interval_px',
                'curve', 'source_raw_frame', 'source_file_sha256')
    for index, rec in zip(indices, rows):
        a, b = old['rows'][index], new['rows'][index]
        measured = rec['measured_boundary_row']
        # The original loader uses pts only when actual_pts is absent.
        # Authenticate that exact alias without adding a field to the row.
        source_values = dict(a, actual_pts=a.get('actual_pts', a.get('pts')))
        if (rec.get('id') != a['id'] or rec.get('split') != 'fit'
                or rec.get('source_row_canonical_json_sha256') != _row_hash(a)
                or _checked(rec['source_row_reference']) != Path(old_path).resolve()
                or b.get(MODE_FIELDS[1]) != receipt_ref
                or measured.get('actual_pts') is None
                or not _same(measured['actual_pts'], source_values['actual_pts'])
                or any(k not in measured or not _same(measured[k], a[k]) for k in required)
                or any(k not in source_values or not _same(v, source_values[k]) for k, v in measured.items())):
            raise ValueError('RGB receipt changed an original measured source row')
    return old, new


def _rebound_documents(options, new_path, *, source_family='board', mode_fields=MODE_FIELDS):
    """Replay the existing protected-source hash rebind, including loss policy."""
    if source_family not in ('board', 'ice'):
        raise ValueError('Explicit board or ice protected source family required')
    new = _read(new_path)
    new_ref = reference(new_path)
    documents, hashes = {}, {}

    def rebound_record(record, index_key, hash_key, ref_key):
        result = deepcopy(record)
        old_ref = result[ref_key]
        original_packet = _read(_checked(old_ref))
        index = result[index_key]
        original = original_packet['rows'][index]
        current = new['rows'][index]
        if (result[hash_key] != _row_hash(original)
                or original['id'] != current['id']
                or not _same(_without_mode(original, mode_fields), _without_mode(current, mode_fields))):
            raise ValueError('Protected source observation/support identity changed')
        hashes[result[hash_key]] = _row_hash(current)
        result[ref_key] = dict(old_ref, **new_ref)
        result[hash_key] = _row_hash(current)
        return result

    for key in (NATIVE, CONFIRMED):
        value = deepcopy(_read(options[key]))
        value['records'] = [rebound_record(r, 'original_row_index',
            'source_row_canonical_json_sha256', 'packet_reference')
            if r.get('family') == ('native_board_edge' if source_family == 'board' else 'native_ice_curve')
            else r for r in value['records']]
        documents[key] = value
    # References to the derived authority files are inserted by the caller.
    guard = deepcopy(_read(options[GUARD]))
    if len(guard['records']) != len(documents[NATIVE]['records']):
        raise ValueError('Every original primary guard record must remain')
    for rec, primary in zip(guard['records'], documents[NATIVE]['records']):
        if rec['id'] != primary['id']:
            raise ValueError('Original full-stroke guard order changed')
        rec['authority_source_row_canonical_json_sha256'] = primary['source_row_canonical_json_sha256']
    _checked(guard['guard_arrays_reference'])
    documents[GUARD] = guard
    documents[CAP] = deepcopy(_read(options[CAP]))
    documents[ENVELOPE] = deepcopy(_read(options[ENVELOPE]))
    policy = deepcopy(_read(options[INTEGRATION]))
    policy['packets'][source_family] = new_ref
    for position, rec in enumerate(policy['records']):
        if rec.get('family') != 'native_' + source_family:
            continue
        current = rebound_record(rec, 'packet_row_index',
                                 'original_row_canonical_json_sha256', 'source_row_reference')
        if 'packet_reference' in current:
            current['packet_reference'] = new_ref
        policy['records'][position] = current
    documents[INTEGRATION] = policy
    return documents, hashes


def _bind_document_refs(documents, refs):
    documents[GUARD]['primary_authority_reference'] = refs[NATIVE]
    documents[CAP]['primary_authority_reference'] = refs[NATIVE]
    documents[CAP]['full_trace_reference'] = refs[GUARD]
    documents[ENVELOPE]['authority_references']['native'] = refs[NATIVE]


def _write(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    if path.exists() and path.read_text(encoding='utf8') != data:
        raise ValueError('Preserve an existing different location-rebind file')
    path.write_text(data, encoding='utf8')
    return reference(path)


def prepare_location_revision(source_options, new_board_packet, output_dir, *,
                              changed_row_indices, truth_state_path):
    options = deepcopy(source_options)
    validate_location_revision(options[BOARD], new_board_packet, changed_row_indices)
    state = _read(truth_state_path)
    original = _read(_checked(state['source_input_manifest_reference']))
    if state['source_target'] != _target(original):
        raise ValueError('Original cap source target differs')
    if (Path(original['additional_real_native_board_edges']['packet_path']).resolve()
            != Path(options[BOARD]).resolve()
            or original['additional_real_native_board_edges']['packet_sha256'] != _sha(options[BOARD])):
        raise ValueError('Location revision must start from the cap packet original board target')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents, _ = _rebound_documents(options, new_board_packet)
    names = {NATIVE: 'native_authority.json', CONFIRMED: 'confirmed_authority.json',
             GUARD: 'full_stroke_guard.json', CAP: 'stroke_caps.json',
             ENVELOPE: 'location_envelope.json', INTEGRATION: 'native_integration.json'}
    refs = {}
    for key in (NATIVE, CONFIRMED):
        refs[key] = _write(output / names[key], documents[key])
    documents[GUARD]['primary_authority_reference'] = refs[NATIVE]
    refs[GUARD] = _write(output / names[GUARD], documents[GUARD])
    _bind_document_refs(documents, refs)
    for key in (CAP, ENVELOPE, INTEGRATION):
        refs[key] = _write(output / names[key], documents[key])
    files = [dict(option_key=k, parent_reference=reference(options[k]),
                  derived_reference=refs[k]) for k in BOUND_KEYS]
    options.update({k: refs[k]['path'] for k in BOUND_KEYS})
    old_board = source_options[BOARD]
    options[BOARD] = str(Path(new_board_packet).resolve())
    proof = dict(schema=PROOF_SCHEMA, original_truth_state=reference(truth_state_path),
                 original_board_packet=reference(old_board), revised_board_packet=reference(new_board_packet),
                 complete_q_sha256=state['complete_q_sha256'],
                 original_source_target=state['source_target'],
                 changed_row_indices=list(changed_row_indices), rebound_files=files,
                 protected_group_count=sum(len(f['group_ids']) for f in state['families'].values()),
                 original_group_ids_caps_uncertainties_preserved=True,
                 fresh_normal_and_J_required=True)
    proof_ref = _write(output / 'location_revision_truth_proof.json', proof)
    return dict(source_options=options, proof_reference=proof_ref)


def _replace_bindings(value, refs, hashes):
    if isinstance(value, list):
        return [_replace_bindings(v, refs, hashes) for v in value]
    if not isinstance(value, dict):
        return hashes.get(value, value) if isinstance(value, str) else value
    result = {k: _replace_bindings(v, refs, hashes) for k, v in value.items()}
    for path_key, sha_key in (('path', 'sha256'), ('packet_path', 'packet_sha256')):
        if path_key in value and sha_key in value:
            for old, new in refs:
                if (Path(value[path_key]).resolve() == Path(old['path']).resolve()
                        and value[sha_key] == old['sha256']):
                    result[path_key], result[sha_key] = new['path'], new['sha256']
    return result


def _board_revision_contract(contract, packet):
    result = deepcopy(contract)
    location = deepcopy(result['measured_boundary_location_objective'])
    ordered = [r for split in ('fit', 'holdout') for r in packet['rows'] if r['split'] == split]
    measured = [r for r in ordered if r.get(MODE_FIELDS[0]) in MEASURED_MODES]
    location['observation'] = ('measured_transition_location' if any(
        r.get(MODE_FIELDS[0]) == 'measured_transition_location' for r in measured)
        else 'measured_transition_midpoint')
    location['ids'] = [r['id'] for r in measured]
    location['measurement_receipts'] = {str(_checked(r[MODE_FIELDS[1]])): r[MODE_FIELDS[1]]['sha256'] for r in measured}
    result['measured_boundary_location_objective'] = location
    return result


def authenticate_location_revision(proof_path, state_path, q, manifest, factors):
    """Dispatch the existing same-state CLI by its exact source proof schema."""
    from .truth_original_location_revision import PROOF_SCHEMA as original_schema, authenticate_original_location_revision
    if _read(proof_path).get('schema') == original_schema:
        return authenticate_original_location_revision(proof_path, state_path, q, manifest, factors)
    from .truth_source_scanline_revision import PROOF_SCHEMA as scanline_schema, authenticate_scanline_revision
    if _read(proof_path).get('schema') == scanline_schema:
        return authenticate_scanline_revision(proof_path, state_path, q, manifest, factors)
    return _authenticate_operator_revision(proof_path, state_path, q, manifest, factors,
        proof_schema=PROOF_SCHEMA, source_contract_key='additional_real_native_board_edges',
        old_packet_key='original_board_packet', new_packet_key='revised_board_packet',
        validator=validate_location_revision, contract_revision=_board_revision_contract)


def _authenticate_operator_revision(proof_path, state_path, q, manifest, factors, *,
        proof_schema, source_contract_key, old_packet_key, new_packet_key,
        validator, contract_revision, source_family='board', mode_fields=MODE_FIELDS):
    """Authenticate only an explicit operator callback and unchanged truth state."""
    proof = _read(proof_path)
    if (proof.get('schema') != proof_schema
            or _checked(proof['original_truth_state']) != Path(state_path).resolve()
            or proof.get('original_group_ids_caps_uncertainties_preserved') is not True
            or proof.get('fresh_normal_and_J_required') is not True):
        raise ValueError('Explicit same-state location-revision proof required')
    packet = _read(state_path)
    q = np.asarray(q, dtype=np.float64)
    with np.load(_checked(packet['complete_state_reference']), allow_pickle=False) as arrays:
        if not np.array_equal(q, arrays['q']):
            raise ValueError('Location revision complete seed q differs')
    if (hashlib.sha256(q.tobytes()).hexdigest() != packet['complete_q_sha256']
            or proof['complete_q_sha256'] != packet['complete_q_sha256']):
        raise ValueError('Location revision complete q identity differs')
    original = _read(_checked(packet['source_input_manifest_reference']))
    if packet['source_target'] != _target(original) or proof['original_source_target'] != packet['source_target']:
        raise ValueError('Original truth source target differs')
    old_path, new_path = _checked(proof[old_packet_key]), _checked(proof[new_packet_key])
    original_board = original[source_contract_key]
    if (Path(original_board['packet_path']).resolve() != old_path
            or original_board['packet_sha256'] != _sha(old_path)):
        raise ValueError('Revision parent differs from the original cap source')
    _, new_board = validator(old_path, new_path, proof['changed_row_indices'])
    records = proof['rebound_files']
    if [r['option_key'] for r in records] != list(BOUND_KEYS):
        raise ValueError('Exact protected-source rebind inventory required')
    options = {r['option_key']: str(_checked(r['parent_reference'])) for r in records}
    documents, hashes = _rebound_documents(options, new_path, source_family=source_family, mode_fields=mode_fields)
    refs = {r['option_key']: r['derived_reference'] for r in records}
    _bind_document_refs(documents, refs)
    for key in BOUND_KEYS:
        if not _same(documents[key], _read(_checked(refs[key]))):
            raise ValueError('Location rebind changed original support, integration or authority: ' + key)
    replacements = [(r['parent_reference'], r['derived_reference']) for r in records]
    replacements.append((proof[old_packet_key], proof[new_packet_key]))
    expected = _replace_bindings(original['source_contract'], replacements, hashes)
    old_board_contract = original[source_contract_key]
    expected_board = contract_revision(_replace_bindings(old_board_contract, replacements, hashes), new_board)

    def board_contract(value):
        if isinstance(value, list):
            return [board_contract(v) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get('schema') == old_board_contract['schema']:
            if not _same(value, _replace_bindings(old_board_contract, replacements, hashes)):
                raise ValueError('An original board source contract differs')
            return expected_board
        return {k: board_contract(v) for k, v in value.items()}

    expected = board_contract(expected)
    if (manifest['full_parameters'] != original['full_parameters']
            or manifest['source_rows'] != original['source_rows']
            or not _same(manifest['source_contract'], expected)
            or not _same(manifest[source_contract_key], expected_board)):
        raise ValueError('Actual source target changed outside the exact location revision')
    if _target(manifest) == packet['source_target']:
        raise ValueError('Location revision requires an actual changed source target')
    for key in ('highest_direct_human_hard_truth', 'highest_user_confirmed_native_sources', 'native_base_curve_integration'):
        if not _same(manifest[key], _replace_bindings(original[key], replacements, hashes)):
            raise ValueError('Protected support or original integration changed: ' + key)
    if manifest.get('source_jacobian_reuse', {}).get('enabled') is True:
        raise ValueError('Location source-target revision requires fresh source Jacobians')
    expected_factors = {k: f for k, f in factors.items() if f is not None}
    if set(expected_factors) != set(packet['families']):
        raise ValueError('Original protected family inventory differs')
    resumed = deepcopy(packet)
    count = 0
    for family, factor in expected_factors.items():
        rec = resumed['families'][family]
        authority = refs.get(NATIVE if family == 'native' else CONFIRMED if family == 'confirmed' else '', rec['authority'])
        _checked(rec['authority'])
        if not _same(factor.contract['authority'], authority):
            raise ValueError('Actual protected authority differs: ' + family)
        ids = [r['record']['id'] if family == 'wide' else r['row']['id'] for r in factor.records]
        if ids != rec['group_ids'] or len(ids) != len(set(ids)):
            raise ValueError('Original protected group IDs/order differs: ' + family)
        for key in ('limits', 'uncertainties'):
            values = np.asarray(rec[key], dtype=np.float64)
            if values.shape != (len(ids),) or not np.isfinite(values).all() or np.any(values < 0.):
                raise ValueError('Original complete finite caps/uncertainties required')
        rec['authority'] = authority
        count += len(ids)
    if count != proof['protected_group_count']:
        raise ValueError('Original protected group count differs')
    migration = dict(proof_reference=reference(proof_path),
                     original_source_target=packet['source_target'], actual_current_source_target=_target(manifest),
                     complete_q_sha256=packet['complete_q_sha256'], protected_group_count=count,
                     changed_row_indices=proof['changed_row_indices'],
                     original_group_ids_caps_uncertainties_preserved=True,
                     fresh_normal_and_J_required=True, old_truth_packet_not_rewritten=True)
    resumed['authenticated_source_location_revision'] = migration
    return dict(packet=resumed, migration=migration)
