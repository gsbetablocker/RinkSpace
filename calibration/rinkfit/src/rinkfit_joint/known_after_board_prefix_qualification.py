"""Explicit V5 numeric/source helper delegation after a qualified V6 parent.

The frozen V5 module remains the actual prefix and source-proof reader. This
facade owns issuance and recursive parent qualification, records both versions,
and never represents its issuance as an execution of the old issuer.
"""
from pathlib import Path
from collections import Counter
import json
from . import known_entity_prefix_qualification as frozen
from . import static_board_prefix_qualification as board
from .io_utils import read_json, read_jsonl, write_json, write_jsonl

SCHEMA = frozen.SCHEMA
CONTRACT = 'known_entity_prefix_after_static_board_v1'
EXECUTED_MODULE_REF = frozen._ref(__file__)
PARENT_GUARD_REF = frozen._ref(board.__file__)
DELEGATED_HELPERS = dict(module=frozen.EXECUTED_MODULE_REF,
                         functions=['_prefix', '_source_proof'])


def assert_board_source_qualification_current(packet, packet_path, *,
                                             verified_dependency_cache=None):
    seen = set() if verified_dependency_cache is None else verified_dependency_cache
    path = Path(packet_path).resolve()
    ref = packet.get('board_material_source_qualification')
    if ref:
        cp, cert = frozen._json_ref(ref, path.parent, seen)
        if 'parent_recursion_contract' in cert:
            return assert_known_after_board_prefix_qualification(
                packet, path, verified_dependency_cache=seen, certificate=(cp, cert))
        if cert.get('schema') == board.SCHEMA:
            return board.assert_static_board_prefix_qualification(
                packet, path, verified_dependency_cache=seen)
    return frozen.assert_board_source_qualification_current(
        packet, path, verified_dependency_cache=seen)


def _contract(cert, base, seen):
    if (cert.get('schema') != SCHEMA or
        cert.get('parent_recursion_contract') != CONTRACT or
        cert.get('issuer_module_ref') != EXECUTED_MODULE_REF or
        cert.get('recursive_parent_guard_module_ref') != PARENT_GUARD_REF or
        cert.get('delegated_numeric_and_source_helpers') != DELEGATED_HELPERS or
        cert.get('reader_module_refs') != [frozen.EXECUTED_MODULE_REF]):
        raise ValueError('Known-after-board actual issuer/reader/recursive contract differs')
    for ref in (cert['issuer_module_ref'], cert['recursive_parent_guard_module_ref'],
                cert['delegated_numeric_and_source_helpers']['module']):
        frozen._load_ref(ref, base, seen)


def assert_known_after_board_prefix_qualification(packet, packet_path, *,
        verified_dependency_cache=None, parent_document=None, certificate=None):
    seen = set() if verified_dependency_cache is None else verified_dependency_cache
    path = Path(packet_path).resolve()
    certpath, cert = (frozen._json_ref(packet['board_material_source_qualification'],
                     path.parent, seen) if certificate is None else certificate)
    _contract(cert, Path(certpath).parent, seen)
    parentpath = frozen._load_ref(cert['parent_packet'], Path(certpath).parent, seen)
    parent = parent_document if parent_document is not None else read_json(parentpath)
    pcpath, pc = frozen._json_ref(parent['board_material_source_qualification'],
                                 parentpath.parent, seen)
    if (pc.get('schema') != board.SCHEMA and
        pc.get('parent_recursion_contract') != CONTRACT):
        raise ValueError('Known-after-board requires the actual V6 source parent or its explicit descendant')
    assert_board_source_qualification_current(parent, parentpath,
                                             verified_dependency_cache=seen)
    # reader_module_refs truthfully names the frozen helpers actually called.
    # The new issuer and recursion guard have separate, strictly checked refs.
    arrays = frozen._prefix(packet, parent, path, parentpath, cert, seen)
    bundle = frozen._load_ref(cert['actual_append_source_bundle'], Path(certpath).parent, seen)
    proofs = frozen._source_proof(packet, parent, path, parentpath, bundle, arrays, seen)
    if certificate is None:
        native = list(read_jsonl(frozen._load_ref(cert['qualified_rows_native'],
                                                Path(certpath).parent, seen)))
        if proofs != native:
            raise ValueError('Known-after-board native proof differs from actual full source proof')
    return proofs


def finalize_known_entity_after_board_prefix_certificate(parent_packet_path,
        appended_packet_document, bundle_path, output_certificate_path, *, parent_document=None):
    parentpath = Path(parent_packet_path).resolve()
    out = Path(output_certificate_path).resolve()
    bundlepath = Path(bundle_path).resolve()
    packet = appended_packet_document
    if out.exists() or (out.parent / 'qualified_source_rows_native.jsonl').exists():
        raise FileExistsError('Known-after-board source certificate is immutable')
    if frozen._ref(__file__) != EXECUTED_MODULE_REF:
        raise ValueError('Known-after-board issuer changed during execution')
    parent = parent_document if parent_document is not None else read_json(parentpath)
    cert = dict(schema=SCHEMA, parent_recursion_contract=CONTRACT,
        issuer_module_ref=EXECUTED_MODULE_REF,
        recursive_parent_guard_module_ref=PARENT_GUARD_REF,
        delegated_numeric_and_source_helpers=DELEGATED_HELPERS,
        parent_packet=frozen._ref(parentpath), original_raw_rows=int(parent['row_count']),
        raw_rows=int(packet['row_count']),
        appended_raw_rows=int(packet['row_count'])-int(parent['row_count']),
        actual_append_source_bundle=frozen._ref(bundlepath),
        membership_array_sha256={k:packet['arrays'][k]['sha256'] for k in frozen.BOUND_ARRAYS},
        issued_arrays=packet['arrays'], reader_module_refs=[frozen.EXECUTED_MODULE_REF],
        inherited_qualification_rescanned=True, old_raw_and_old_UV_unchanged=True,
        new_statistical_precision_invented=False)
    proofs = assert_known_after_board_prefix_qualification(packet, out.parent/'packet.json',
                                    parent_document=parent, certificate=(out, cert))
    native = out.parent/'qualified_source_rows_native.jsonl'
    write_jsonl(native, proofs)
    cert['qualified_rows_native'] = frozen._ref(native)
    write_json(out, cert)
    packet['board_material_source_qualification'] = frozen._ref(out)
    packet['all_new_source_nuisance_requires_fresh_physical_capture'] = True
    return packet


def combine_prepared_source_bundles(bundle_paths, output):
    """Combine already executed per-row evidence, without another image cache."""
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Combined source bundle is immutable')
    bundles = [read_json(p) for p in bundle_paths]
    seen = set()
    rows, receipts, checks, pending, episodes = [], [], [], [], []
    reader_refs, parents, counts = [], [], Counter()
    for path, b in zip(bundle_paths, bundles):
        if b['schema'] != frozen.BUNDLE_SCHEMA or not b['source_ledger_preserved']:
            raise ValueError('Actual qualified known-corner bundle required')
        for key in ('model', 'anchor_packet', 'anchor_inventory', 'source_generation', 'roles'):
            if b[key] != bundles[0][key]:
                raise ValueError('Source bundle identity or acquisition differs: '+key)
        base = Path(path).resolve().parent
        frozen._refs(b, base, seen)
        rr = list(read_jsonl(frozen._load_ref(b['source_rows_ref'], base, seen)))
        qq = read_json(frozen._load_ref(b['source_mean_replay_QA'], base, seen))
        if not qq['passed'] or len(rr) != b['source_rows']:
            raise ValueError('Existing source generation has not completed actual replay')
        rows.extend(rr)
        receipts.extend(read_jsonl(frozen._load_ref(b['qualification_receipts'], base, seen)))
        checks.extend(read_jsonl(frozen._load_ref(qq['rows_ref'], base, seen)))
        pending.extend(read_json(frozen._load_ref(qq['pending_ref'], base, seen)))
        episodes.extend(read_json(frozen._load_ref(b['ownership_review'], base, seen))['reviewed_episodes'])
        parents.append(frozen._ref(path))
        counts.update(b['counts'])
        for ref in b['reader_module_refs']:
            if ref not in reader_refs:
                reader_refs.append(ref)
    if (len({r['source_row_id'] for r in rows}) != len(rows) or
        len({(r['joint_material_index'], r['existing_point_path'], r['native_pts']) for r in rows}) != len(rows) or
        len({e['track_id'] for e in episodes}) != len(episodes)):
        raise ValueError('Combined source generation repeats an entity exposure or episode')
    out.mkdir(parents=True)
    write_jsonl(out/'source_rows.jsonl', rows)
    write_jsonl(out/'qualification_receipts.jsonl', receipts)
    write_jsonl(out/'source_replay_rows.jsonl', checks)
    write_json(out/'pending.json', pending)
    ownership = dict(schema='rinkfit.combined_native_corner_ownership_review.v1',
        source_body_material_ownership_supported=True, reviewed_episodes=episodes,
        original_reviews=[b['ownership_review'] for b in bundles])
    write_json(out/'ownership_review.json', ownership)
    mapping = {r['target_id']:dict(material_index=r['joint_material_index'], geometry_ready=True) for r in rows}
    write_json(out/'material_map.json', dict(semantic_to_material=mapping))
    qa = dict(schema='rinkfit.actual_native_corner_source_support_replay.v1', passed=True,
        source_candidates=len(checks), qualified_rows=len(rows), pending_rows=len(pending),
        counts=dict(counts), rows_ref=frozen._ref(out/'source_replay_rows.jsonl'),
        pending_ref=frozen._ref(out/'pending.json'), all_source_candidates_accounted=True,
        peak_replay_max_abs_px=max(q['native_peak_replay_max_abs_px'] for q in checks),
        all_original_raw_UV_and_flags_unchanged=True,
        no_covariance_or_displacement_factor_invented=True,
        actual_profile_support_definition='intersection_of_same_polarity_gradient_lobes_of_original_six_finite_material_profiles',
        source_generation=bundles[0]['source_generation'],
        actual_executed_source_generations=parents, reader_module=EXECUTED_MODULE_REF,
        source_pixel_replay_reexecuted_by_combiner=False)
    write_json(out/'source_mean_replay_QA.json', qa)
    gate = dict(schema='rinkfit.known_corner_source_support_qualification.v1',
        input_cumulative_journal=frozen._ref(out/'source_rows.jsonl'),
        qualification_receipts=frozen._ref(out/'qualification_receipts.jsonl'),
        current_model=bundles[0]['model'],
        source_mean_and_floating_estimator_replay_QA=frozen._ref(out/'source_mean_replay_QA.json'),
        qualified_new_measurement_rows=len(rows), old_raw_source_flags_false_are_not_overwritten=True,
        new_measurement_version_geometry_qualification_only=True,
        source_mean_estimator='source_supported_original_UV_or_native_corner_peak_v1')
    write_json(out/'qualification_gate.json', gate)
    by_shot = dict(Counter(r['shot_id'] for r in rows))
    chunk = dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',
        all_requested_frames_accounted=True, source_dataset_tag='known_board_corner_native_versions_v1',
        source_rows_ref=frozen._ref(out/'source_rows.jsonl'),
        geometry_rows_ref=frozen._ref(out/'source_rows.jsonl'),
        source_gate_manifest=frozen._ref(out/'qualification_gate.json'),
        actual_source_rows=len(rows), by_shot=by_shot)
    write_json(out/'chunk_manifest.json', chunk)
    b = dict(bundles[0], source_rows_ref=frozen._ref(out/'source_rows.jsonl'),
        qualification_receipts=frozen._ref(out/'qualification_receipts.jsonl'),
        qualification_gate=frozen._ref(out/'qualification_gate.json'),
        source_mean_replay_QA=frozen._ref(out/'source_mean_replay_QA.json'),
        ownership_review=frozen._ref(out/'ownership_review.json'),
        reader_module_refs=reader_refs+[EXECUTED_MODULE_REF], source_rows=len(rows),
        candidate_denominator=len(checks), counts=dict(counts), by_shot=by_shot,
        closed_chunks=[frozen._ref(out/'chunk_manifest.json')],
        typed_entity_qualification=[frozen._ref(out/'qualification_gate.json')],
        material_map=frozen._ref(out/'material_map.json'),
        combined_executed_source_generations=parents)
    write_json(out/'manifest.json', b)
    return out/'manifest.json'
