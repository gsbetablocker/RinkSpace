"""Reuse completed independent startup derivative tests for unchanged sources.

This does not reuse a residual, Jacobian, profile, guard or solver decision. The
current runtime registration is linked to existing diagnostic evidence explicitly;
it is not represented as a registry snapshot saved by the historical run.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _producer_files():
    from .config import PACKAGE_ROOT
    from structured_programme_source_candidate.complete_same_state_replay import _runtime_implementation_files
    from .event_factories import verify_sources
    verify_sources()
    paths = (PACKAGE_ROOT / 'event_factories_manifest.json', PACKAGE_ROOT / 'legacy_dependencies.json')
    # This original routine authenticates every registered Python/native/shared
    # producer and the existing source-path resolver; no graph is constructed.
    return _runtime_implementation_files(paths)


def _source_facts(manifest, q):
    import numpy as np
    from structured_programme_source_candidate.complete_same_state_replay import (
        _semantic_manifest, _qp_compare_source_semantics)
    from structured_programme_source_candidate.truth_source_role_migration import compaction_identity
    # Authenticate the ORIGINAL evaluated contract before semantic cleaning.
    # This registered helper checks retained CSR/numerical PASS flags and the
    # finite [0,1] error fraction, then removes only q-dependent diagnostics.
    # Every input, source sigma, CSR identity and physical tolerance remains.
    if isinstance(manifest.get('original_motion_input_compaction'), dict):
        manifest = dict(manifest)
        manifest['original_motion_input_compaction'] = compaction_identity(
            manifest['original_motion_input_compaction'])
    value = _qp_compare_source_semantics(_semantic_manifest(manifest))
    backend = value.get('native_backend_requested', {})
    for key in ('RINKSPACE_COMPLETE_CYCLE_VARIANT', 'RINKSPACE_BASELINE_FD_SOURCE',
                'RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE', 'RINKSPACE_OBSERVE_COMPLETE_FIRST_CYCLE'):
        backend.pop(key, None)
    restoration = value.get('joint_source_truth_restoration')
    if isinstance(restoration, dict) and 'complete_initial_q_sha256' in restoration:
        actual = hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest()
        if restoration['complete_initial_q_sha256'] != actual:
            raise ValueError('Diagnostic source initializer identity differs from actual complete q')
        restoration.pop('complete_initial_q_sha256')
    initializer = value.get('independent_pattern_new_slot_initializer')
    if isinstance(initializer, dict) and 'parameter_values' in initializer:
        indices = np.asarray(manifest['independent_pattern_placement_parameter_indices'], int)
        if (initializer.get('initializer_is_not_observation') is not True
                or not np.array_equal(np.asarray(initializer['parameter_values'], float), np.asarray(q)[indices])):
            raise ValueError('Pattern initializer is not the actual declared numerical q slots')
        initializer.pop('parameter_values')
    return value


def _read_evidence(directory):
    import numpy as np
    directory = Path(directory).resolve()
    files = {}
    def read(name):
        path = directory / name
        value = json.loads(path.read_text(encoding='utf8'))
        files[name] = dict(path=str(path), sha256=_sha(path))
        return value
    manifest = read('input_manifest.json')
    wide = read('actual_wide_directional_diagnostic.json')
    small = read('actual_all_small_source_value_J_check.json')
    if (wide.get('passed') is not True or wide['source_manifest_sha256'] != files['input_manifest.json']['sha256']
            or not wide['checked_max_absolute'] <= wide['allowed']
            or not small['coupled_directional_max_absolute'] <= small['coupled_directional_tolerance']
            or small.get('finite_support_and_original_UV_sigma_preserved') is not True
            or not small.get('original_small_families')):
        raise ValueError('Existing complete small/wide derivative evidence did not pass')
    if manifest.get('additional_real_native_board_edges') is not None:
        edge = read('actual_additional_native_board_source_check.json')
        if edge.get('passed') is not True or not edge['coupled_directional_max_error'] <= edge['allowed']:
            raise ValueError('Existing native board directional test did not pass')
    if manifest.get('additional_real_native_ice_curves') is not None:
        ice = read('actual_additional_native_ice_source_check.json')
        if (ice.get('original_source_UV_sigma_and_splits_changed') is not False
                or not ice['coupled_directional_max_absolute'] <= ice['coupled_directional_tolerance']):
            raise ValueError('Existing native ice directional test did not pass')
        if manifest['additional_real_native_ice_curves'].get('objective_revision') is not None:
            raw = read('actual_native_ice_raw_sigma_revision_check.json')
            if (raw.get('passed') is not True or raw.get('original_UV_sigma_split_aliases_byte_exact') is not True
                    or raw.get('quadratic_mask_exact') is not True or raw.get('raw_sigma_direct_exact') is not True):
                raise ValueError('Existing original raw sigma revision diagnostic did not pass')
    normal_path = directory / 'actual_wide_linearization_diagnostic.npz'
    with np.load(normal_path, allow_pickle=False) as saved:
        q = saved['q'].copy()
        if str(saved['source_manifest_sha256']) != files['input_manifest.json']['sha256']:
            raise ValueError('Existing wide diagnostic complete state manifest differs')
        token = str(saved['frozen_profile_token'])
    files[normal_path.name] = dict(path=str(normal_path), sha256=_sha(normal_path))
    qsha = hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest()
    if qsha != wide['q_sha256'] or token != wide['frozen_profile_token']:
        raise ValueError('Existing wide derivative evidence q/profile differs from its saved actual diagnostic state')
    receipt_path = Path(str(directory) + '.bootstrap') / 'same_target_fresh_J_verification_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    files['original_main_receipt'] = dict(path=str(receipt_path), sha256=_sha(receipt_path))
    existing17 = None
    existing17_unavailable = None
    if manifest.get('original_motion_far_board_role_correction') is not None:
        try:
            report = read('actual_existing17_continuous_perimeter_value_J_crossregion_check.json')
            _validate_existing17_evidence(report)
            existing17 = files['actual_existing17_continuous_perimeter_value_J_crossregion_check.json']
        except Exception as exc:
            # Missing/failed historical 17-point evidence does not invalidate
            # the other independent diagnostics: this check runs originally.
            files.pop('actual_existing17_continuous_perimeter_value_J_crossregion_check.json', None)
            existing17_unavailable = dict(type=type(exc).__name__, reason=str(exc))
    return dict(directory=str(directory), files=files, manifest=manifest, source_facts=_source_facts(manifest, q),
                q_sha256=qsha, frozen_profile_token=token, original_main_sha256=receipt['original_main_sha256'],
                existing17=existing17, existing17_unavailable=existing17_unavailable)


def _validate_existing17_evidence(report):
    """Authenticate the original completed diagnostic, never its current q."""
    import math
    if (report.get('schema') != 'rinkspace.actual-existing-seventeen-continuous-perimeter-check.v1'
            or report.get('original_points') != 17 or report.get('source_scalar_rows_added') != 0
            or report.get('cross_region_check_complete') is not True
            or report.get('complete_motion_J_or_normal_not_produced_by_this_check') is not True):
        raise ValueError('Historical existing17 independent diagnostic is incomplete')
    blocks = report['block_checks']
    if [(block['block'], block['original_points']) for block in blocks] != [('main1', 2), ('low', 15)]:
        raise ValueError('Historical existing17 original source block counts differ')
    for block in blocks:
        for flag in ('exact_original_UV_sigma_actual_clock_and_finite_mask',
                     'dynamic_derivatives_match_actual_programme_chain', 'original_six_columns_bitexact',
                     'sorted_CSR_full_physical_column_mapping_exact', 'actual_native_camera_CSR_preserved'):
            if block.get(flag) is not True:
                raise ValueError('Historical existing17 original PASS flag failed: ' + flag)
        checks = block['FD_checks']
        if [row['logical_column'] for row in checks] != block['complete_logical_columns'] or not checks:
            raise ValueError('Historical existing17 declared derivative checks are incomplete')
        for row in checks:
            error, allowed = float(row['FD_error']), float(row['allowed'])
            if not (math.isfinite(error) and math.isfinite(allowed) and 0 <= error <= allowed):
                raise ValueError('Historical existing17 original FD tolerance failed')
    crossings = report['actual_print_station_cross_region_checks']
    if ([row['fraction_of_full_period'] for row in crossings] != [-.125, .125, .375, .625, .875, 1.125]
            or any(row.get('current_source_planes_bitexact') is not True for row in crossings)):
        raise ValueError('Historical existing17 original cross-region checks are incomplete')


def _changed_fields(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            path = prefix + '.' + key if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_changed_fields(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def attach_certified_startup_diagnostic_reuse(tree, namespace, destination, evidence_directory):
    """Attach before the driver's existing AST compile; return transformed tree.

    The explicit existing-run argument registers an evidence link using CURRENT
    authenticated producer files. A saved link is reused only if all those files
    remain exact. Absent/failed/changed evidence executes original diagnostics,
    and writes the reason. The original post-diagnostic guard stays untouched.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    certificate_path = destination / 'startup_diagnostic_evidence_link.json'
    receipt_path = destination / 'startup_diagnostic_reuse.json'
    evidence = None
    producer_files = None
    unavailable = None
    pending17 = {}
    reused_gate = {}
    try:
        evidence = _read_evidence(evidence_directory)
        producer_files = _producer_files()
        from .runtime import source_path
        source = source_path('run_current_named66_full_programme').read_text(encoding='utf8')
        original_tree = ast.parse(source)
        main_node = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        lines = source.splitlines(keepends=True)
        original_main = ''.join(lines[main_node.lineno - 1:main_node.end_lineno])
        original_main_sha = hashlib.sha256(original_main.encode('utf8')).hexdigest()
        if original_main_sha != evidence['original_main_sha256']:
            raise ValueError('Existing diagnostic original driver main producer changed')
        certificate = dict(schema='rinkspace.existing-startup-diagnostics-current-producer-evidence-link.v1',
            evidence_directory=evidence['directory'], evidence_files=evidence['files'],
            diagnostic_q_sha256=evidence['q_sha256'], diagnostic_frozen_profile_token=evidence['frozen_profile_token'],
            original_main_sha256=original_main_sha, current_authenticated_producer_files=producer_files,
            source_facts_sha256=hashlib.sha256(_canonical(evidence['source_facts']).encode('utf8')).hexdigest(),
            producer_association='Current unchanged registered sources linked to explicitly selected existing PASS evidence',
            historical_run_registry_snapshot_claimed=False, prior_residual_J_profile_or_acceptance_reused=False,
            registered_source_files_changed=False, accuracy_accepted=False)
        if certificate_path.exists():
            old_certificate = json.loads(certificate_path.read_text(encoding='utf8'))
            if old_certificate != certificate:
                raise ValueError('Existing startup diagnostic evidence link changed')
        else:
            _write(certificate_path, certificate)
    except Exception as exc:
        unavailable = dict(type=type(exc).__name__, reason=str(exc))

    def reuse(opt, manifest, q, source):
        import numpy as np
        key = (id(opt), id(manifest), id(source), hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest())
        if reused_gate.get('key') == key:
            return reused_gate['report']['reused']
        report = dict(schema='rinkspace.independent-startup-derivative-reuse.v1', reused=False,
            evidence_directory=str(Path(evidence_directory).resolve()),
            current_q_sha256=hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest(),
            current_source_rows=int(len(source.scales)), all_current_source_qualification_and_guards_retained=True,
            fresh_normal_source_derivatives_required=True, previous_J_or_normal_imported=False,
            initial_and_trial_physical_guard_retained=True, original_true_cost_and_acceptance_retained=True,
            numerical_accuracy_gate_relaxed=False, accuracy_accepted=False)
        try:
            if unavailable is not None:
                raise ValueError('Existing evidence unavailable: ' + unavailable['reason'])
            if getattr(opt, 'camera_seed', None) is None:
                raise ValueError('Only explicit complete-state continuation may reuse startup derivative diagnostics')
            if getattr(opt, 'reuse_wide_linearization', None) is not None:
                raise ValueError('Explicit small-wide replay installation requires original startup branch')
            current = _source_facts(manifest, q)
            compaction = manifest.get('original_motion_input_compaction')
            if isinstance(compaction, dict):
                original_compaction = evidence['manifest'].get('original_motion_input_compaction')
                report['compaction_source_identity'] = dict(
                    registered_identity_function='structured_programme_source_candidate.truth_source_role_migration.compaction_identity',
                    current_original_contract_sha256=hashlib.sha256(_canonical(compaction).encode('utf8')).hexdigest(),
                    evidence_original_contract_sha256=hashlib.sha256(_canonical(original_compaction).encode('utf8')).hexdigest(),
                    current_normalized_contract_sha256=hashlib.sha256(_canonical(current['original_motion_input_compaction']).encode('utf8')).hexdigest(),
                    original_current_PASS_flags_and_finite_error_fraction_validated=True,
                    original_actual_diagnostics_preserved_in_hashed_input_manifests=True,
                    input_CSR_sigma_and_physical_tolerance_identity_retained=True)
            changes = _changed_fields(evidence['source_facts'], current)
            if changes:
                report['changed_source_or_domain_fields'] = changes
                raise ValueError('Source/domain facts differ from the completed derivative diagnostics')
            from structured_programme_source_candidate.complete_same_state_replay import _verify_files
            _verify_files(producer_files)
            _verify_files(list(evidence['files'].values()))
            # Source construction before this gate has already performed every
            # original packet/qualification check. Compare those actual live
            # source contracts, including their original file digests; do not
            # rescan raw videos or observation archives as a diagnostic gate.
            report.update(reused=True, reason='Exact registered producers and same source/domain facts; existing independent diagnostics passed',
                          evidence_link=str(certificate_path), evidence_link_sha256=_sha(certificate_path),
                          existing_diagnostic_q_sha256=evidence['q_sha256'],
                          current_q_and_profiles_not_replaced=True,
                          skipped_independent_tests=['native_board_random_directional_and_small_J',
                              'native_ice_raw_sigma_revision_and_random_directional_and_small_J',
                              'original_tail_small_J_diagnostic', 'wide_random_directional_branch_and_boundary_column_diagnostics'],
                          current_fresh_wide_base_cache_must_be_computed_normally=True)
        except Exception as exc:
            report.update(reason=str(exc), error_type=type(exc).__name__, original_diagnostics_execute=True)
        _write(receipt_path, report)
        print(json.dumps(dict(stage='startup_independent_derivative_diagnostic_reuse', **report)), flush=True)
        reused_gate.update(key=key, report=report)
        return report['reused']

    def defer17(opt, state, q, path, original):
        import numpy as np
        if (unavailable is not None or evidence.get('existing17') is None
                or getattr(opt, 'camera_seed', None) is None):
            return original(state, q, output_path=path)
        if pending17:
            raise RuntimeError('Existing17 independent diagnostic already pending')
        pending17.update(state=state, q=np.asarray(q, float).copy(), path=Path(path), original=original)
        print(json.dumps(dict(stage='existing17_independent_diagnostic_deferred',
            awaiting_complete_source_facts_gate=True, current_source_binding_and_qualification_execute=True)), flush=True)
        return None

    def complete17(opt, manifest, q, source):
        import numpy as np
        if not pending17:
            return True
        permitted = reuse(opt, manifest, q, source)
        state, preflight_q = pending17['state'], pending17['q']
        reason = reused_gate['report']['reason']
        # Current registration already authenticates the actual 17 UV/sigma/
        # clock/roles. Link that live binding to the fully compared manifest.
        try:
            revision = manifest['original_motion_far_board_role_correction']['existing_finite_geometry_revision']
            binding = state.geometry._existing_finite_perimeter_binding
            if (len(binding.records) != 17 or revision['binding_sha256'] != binding.binding_id
                    or revision['existing_finite_pixels'] != 17 or revision['new_source_scalar_rows'] != 0
                    or revision['source_UV_sigma_clock_and_finite_role_unchanged'] is not True
                    or revision['complete_current_world_parameter_chain'] is not True):
                raise ValueError('Current original existing17 source binding differs from its complete manifest')
        except Exception as exc:
            permitted = False
            reason = type(exc).__name__ + ': ' + str(exc)
        report = dict(schema='rinkspace.existing17-independent-diagnostic-reuse.v1', reused=permitted,
            reason=reason, current_complete_q_sha256=hashlib.sha256(np.ascontiguousarray(q, dtype=float).tobytes()).hexdigest(),
            current_original_preflight_q_sha256=hashlib.sha256(preflight_q.tobytes()).hexdigest(),
            current_source_17_qualification_binding_guard_and_fresh_J_retained=True,
            current_numerical_values_or_derivatives_replayed=False, historical_q_values_claimed_as_current=False,
            physical_domain_or_FD_tolerance_relaxed=False, accuracy_accepted=False)
        if permitted:
            report.update(historical_PASS_evidence=evidence['existing17'],
                current_producer_and_complete_source_facts_gate=str(receipt_path),
                evidence_complete_diagnostic_state_q_sha256=evidence['q_sha256'],
                historical_existing17_q_sha256_recorded=False,
                existing17_report_contains_no_q_identity=True,
                current_existing17_FD_and_cross_region_test_executed=False)
            # This is a link, with a different name/schema. Never copy the
            # historical facets/residuals/FD errors into current actual output.
            _write(pending17['path'].with_name('existing17_independent_diagnostic_evidence_link.json'), report)
            _write(Path(opt.output) / 'existing17_independent_diagnostic_evidence_link.json', report)
        else:
            report.update(original_existing17_diagnostic_execute=True)
            _write(destination / 'existing17_diagnostic_reuse.json', report)
            print(json.dumps(dict(stage='existing17_independent_diagnostic_original_fallback', **report)), flush=True)
            # Outside the evidence-gate catch: original mathematics, domain
            # failures and SourceControlStop propagate unchanged.
            actual = pending17['original'](state, preflight_q, output_path=pending17['path'])
            _write(pending17['path'], actual)
            _write(Path(opt.output) / 'actual_existing17_continuous_perimeter_value_J_crossregion_check.json', actual)
            report.update(current_existing17_FD_and_cross_region_test_executed=True,
                          original_existing17_diagnostic_completed=True)
        pending17.clear()
        _write(destination / 'existing17_diagnostic_reuse.json', report)
        print(json.dumps(dict(stage='existing17_independent_diagnostic_reuse', **report)), flush=True)
        return True

    sites = []
    class Replace(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if (isinstance(node.test, ast.Compare)
                    and ast.dump(node.test, include_attributes=False) == ast.dump(
                        ast.parse('opt.reuse_complete_linearization is None', mode='eval').body, include_attributes=False)
                    and any(isinstance(part, ast.Constant) and part.value == 'actual_all_small_source_values_and_jacobians'
                            for statement in node.body for part in ast.walk(statement))):
                sites.append(node.lineno)
                # No opt field, source object or original body is changed.
                node.test = ast.BoolOp(op=ast.And(), values=[node.test, ast.UnaryOp(op=ast.Not(),
                    operand=ast.parse('_rinkspace_reuse_startup_derivatives(opt, manifest, q, source)', mode='eval').body)])
            return node
    tree = Replace().visit(tree)
    if len(sites) != 1:
        raise ValueError('Original independent startup derivative diagnostic boundary changed: ' + str(sites))
    namespace['_rinkspace_reuse_startup_derivatives'] = reuse
    namespace['_rinkspace_defer_existing17_diagnostic'] = defer17
    namespace['_rinkspace_complete_existing17_diagnostic'] = complete17
    ast.fix_missing_locations(tree)
    return tree


def attach_deferred_existing17_startup_reuse(tree, namespace):
    """Attach after certified reuse, before source-auth scope/AST compile.

    The original early check is postponed to the full source-facts gate. Its
    original fallback remains mandatory when historical PASS is ineligible.
    Put completion in the existing diagnostic if-test so source-auth's finish
    inserted before that if still runs before any diagnostic/link decision.
    """
    for name in ('_rinkspace_defer_existing17_diagnostic', '_rinkspace_complete_existing17_diagnostic'):
        if name not in namespace:
            raise ValueError('Certified startup reuse must attach before existing17 defer')
    sites = []
    class Replace(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            assignment = next((item for item in node.body if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'existing17_check' for target in item.targets)
                and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name)
                and item.value.func.id == 'check_existing_finite_perimeter_sources'), None)
            if assignment is not None:
                sites.append('original_existing17_preflight')
                index = node.body.index(assignment)
                if len(node.body[index + 1:]) != 2:
                    raise ValueError('Original existing17 write/PASS announcement boundary changed')
                assignment.value = ast.parse('_rinkspace_defer_existing17_diagnostic(opt, state, q, existing17_path, check_existing_finite_perimeter_sources)', mode='eval').body
                tail = ast.parse('if existing17_check is not None:\n    pass').body[0]
                tail.body = node.body[index + 1:]
                node.body[index + 1:] = [tail]
            if (any(isinstance(part, ast.Name) and part.id == '_rinkspace_reuse_startup_derivatives'
                    for part in ast.walk(node.test))
                    and any(isinstance(part, ast.Constant) and part.value == 'actual_all_small_source_values_and_jacobians'
                            for item in node.body for part in ast.walk(item))):
                sites.append('complete_source_diagnostic_gate')
                node.test = ast.BoolOp(op=ast.And(), values=[ast.parse(
                    '_rinkspace_complete_existing17_diagnostic(opt, manifest, q, source)', mode='eval').body, node.test])
            return node
    tree = Replace().visit(tree)
    if sites != ['original_existing17_preflight', 'complete_source_diagnostic_gate']:
        raise ValueError('Original existing17/full source diagnostic boundaries changed: ' + str(sites))
    ast.fix_missing_locations(tree)
    return tree
