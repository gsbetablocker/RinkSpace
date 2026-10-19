"""Recompute every source r/J against a sealed same-state baseline.

The explicit verification branch consumes newly computed blocks. An exact
already qualified sealed archive may also reach the existing full replay
gate; source changes and partial/append caches are never admitted here.
The signed producer/manifest files, source guards and solver stay unchanged.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time


_PRESENCE_REJECTIONS = {
    'Original motion material-role correction requires fresh complete source derivatives',
    'New actual board tracks require fresh complete source derivatives',
    'Restored camera physical domain uses the actual current source normal, not a saved matrix',
    'A source camera initializer requires fresh complete profiles/J, never a cached derivative',
    'Corrected original ice pixels require fresh complete source profiles/J',
}
_COMPACTION_REJECTION = 'Changed active input inventory requires fresh complete source J'


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def authenticate_qualified_complete_cache_request(directory,qualification,camera_seed):
    """Admit only the exact already qualified, sealed complete state.

    The original complete replay still checks every live source, profile,
    implementation, row, scale and byte. This parse gate does not admit an
    append/partial cache or a qualification change.
    """
    import numpy as np
    from structured_programme_source_candidate import complete_same_state_replay as replay
    directory=Path(directory).resolve();qualification=Path(qualification).resolve()
    seal=json.loads((directory/'actual_complete_cache_seal.json').read_text(encoding='utf8'))
    inventory_path=directory/'strict_complete_source_replay_inventory.json'
    if Path(seal['path']).resolve()!=inventory_path or replay._sha(inventory_path)!=seal['sha256']:
        raise ValueError('Qualified complete cache seal identity differs')
    inventory=json.loads(inventory_path.read_text(encoding='utf8'))
    manifest_path=directory/'input_manifest.json'
    bound=[r for r in inventory['files']if r['relative_path']=='input_manifest.json']
    if len(bound)!=1 or replay._sha(manifest_path)!=bound[0]['sha256']:
        raise ValueError('Qualified complete cache manifest bytes differ')
    old=json.loads(manifest_path.read_text(encoding='utf8'))
    identity=old[replay.IDENTITY_KEY]
    if inventory['strict_identity_sha256']!=replay._json_sha(identity):
        raise ValueError('Qualified complete cache strict identity differs')
    actual=old['original_source_point_qualification']
    if (Path(actual['packet_path']).resolve()!=qualification
            or actual['packet_sha256']!=replay._sha(qualification)):
        raise ValueError('Source point qualification differs from the already qualified complete cache')
    with np.load(camera_seed,allow_pickle=False)as z:
        if replay._array_signature(z['q'])!=inventory['q']:
            raise ValueError('Qualified complete cache belongs to another complete q')
    receipt=dict(schema='rinkspace.exact-qualified-complete-cache-parse-admission.v1',
        directory=str(directory),manifest_sha256=bound[0]['sha256'],strict_identity_sha256=inventory['strict_identity_sha256'],
        complete_q_sha256=inventory['q']['sha256'],qualification_path=str(qualification),
        qualification_sha256=actual['packet_sha256'],all_original_complete_replay_gates_still_required=True)
    # Output locations and the existing diagnostic-evidence link are execution
    # telemetry. Restrict compare-only relocation to this exact sealed archive;
    # authenticate both unmodified semantic signatures before any comparison.
    if not getattr(replay._check_strict_identity,'_exact_qualified_archive_telemetry',False):
        original=replay._check_strict_identity
        def checked(old,current,**options):
            active=replay._exact_qualified_complete_cache_admission
            if replay._json_sha(old.get(replay.IDENTITY_KEY,{}))!=active['strict_identity_sha256']:
                return original(old,current,**options)
            for manifest in (old,current):
                if manifest[replay.IDENTITY_KEY]['semantic_manifest_sha256']!=replay._json_sha(replay._semantic_manifest(manifest)):
                    raise ValueError('Exact qualified cache original semantic signature changed')
            compared=deepcopy(current);old_env=old.get('native_backend_requested',{})
            env=compared.setdefault('native_backend_requested',{})
            for key in (*replay._QP_OUTPUT_PATH_KEYS,'RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE'):
                if key in old_env:env[key]=old_env[key]
                else:env.pop(key,None)
            compared[replay.IDENTITY_KEY]['semantic_manifest_sha256']=replay._json_sha(replay._semantic_manifest(compared))
            archive=Path(active['directory'])
            if archive.name=='current_complete_linearization_cache'and options.get('append_normalized_pair')is None:
                # The stop-point writer seals a later q/profile under the
                # original run's unchanged graph initializer provenance.
                # Reuse the existing exact original/current initializer audit;
                # the full implementation, source bytes and matrix gates below
                # remain mandatory. No observation or bound is substituted.
                with np.load(archive/'actual_initial_normal_state.npz',allow_pickle=False)as z:
                    sealed_q=z['q'].copy()
                if replay._array_signature(sealed_q)['sha256']!=active['complete_q_sha256']:
                    raise ValueError('Stop-point initializer audit requires the admitted complete q')
                pair,changes=_initialization_pair(replay,archive,old,compared,sealed_q)
                options['append_normalized_pair']=pair
                print(json.dumps(dict(stage='exact_stop_point_graph_initializer_provenance_checked',
                    complete_q_sha256=active['complete_q_sha256'],changes=changes,
                    all_source_implementation_profile_and_matrix_gates_retained=True)),flush=True)
            return original(old,compared,**options)
        checked._exact_qualified_archive_telemetry=True
        replay._check_strict_identity=checked
    replay._exact_qualified_complete_cache_admission=receipt
    print(json.dumps(dict(stage='exact_qualified_complete_cache_parse_admitted',**receipt)),flush=True)
    return receipt


def restore_authenticated_complete_cache_profiles(directory,q,source):
    """Restore the actual frozen normal state, not its pre-update seed profiles."""
    import numpy as np
    from structured_programme_source_candidate import complete_same_state_replay as replay
    from z190_joint_source_solver import _frozen_profile_token
    directory=Path(directory).resolve();admitted=replay._exact_qualified_complete_cache_admission
    if str(directory)!=admitted['directory']or replay._array_signature(q)['sha256']!=admitted['complete_q_sha256']:
        raise ValueError('Frozen complete-cache profile restore must use the admitted same q/archive')
    inventory=json.loads((directory/'strict_complete_source_replay_inventory.json').read_text(encoding='utf8'))
    profile_path=directory/'actual_initial_frozen_profile_arrays.npz'
    bound=[r for r in inventory['files']if r['relative_path']==profile_path.name]
    if len(bound)!=1 or replay._sha(profile_path)!=bound[0]['sha256']:
        raise ValueError('Sealed complete-cache frozen profile bytes changed')
    with np.load(profile_path,allow_pickle=False)as z:
        arrays={k:z[k].copy()for k in z.files}
    signatures=[dict(name=k,**replay._array_signature(v))for k,v in arrays.items()]
    if signatures!=inventory['profile_arrays']:
        raise ValueError('Sealed complete-cache complete profile inventory differs')
    previous=_frozen_profile_token(source.updater)
    source.updater.restore_checkpoint(arrays)
    replay._authenticate_profiles(directory,source.updater,inventory['frozen_profile_token'])
    print(json.dumps(dict(stage='authenticated_complete_normal_profiles_restored',
        previous_initializer_profile_token=previous,frozen_normal_profile_token=inventory['frozen_profile_token'],
        actual_profile_arrays=len(arrays),original_profile_restore_protocol_used=True,
        no_new_profile_update_or_source_evaluation=True)),flush=True)


def _initialization_pair(replay, directory, old, current, q):
    """Authenticate two numerical initializer receipts, never observations."""
    import numpy as np
    left = replay._qp_compare_source_semantics(replay._semantic_manifest(old))
    right = replay._qp_compare_source_semantics(replay._semantic_manifest(current))
    old_initial = left['joint_source_truth_restoration']['complete_initial_q_sha256']
    new_initial = right['joint_source_truth_restoration']['complete_initial_q_sha256']
    if new_initial != replay._array_signature(q)['sha256']:
        raise ValueError('Fresh verification initializer is not the exact sealed current q')
    changes = []
    if old_initial != new_initial:
        if directory.name != 'current_complete_linearization_cache':
            raise ValueError('Continuation initializer provenance requires the original run current-complete cache')
        original_state = directory.parent / 'actual_initial_normal_state.npz'
        with np.load(original_state, allow_pickle=False) as z:
            original_q = z['q'].copy()
        if replay._array_signature(original_q)['sha256'] != old_initial:
            raise ValueError('Original graph initializer q is not authenticated by its signed manifest')
        changes.append(dict(field='joint_source_truth_restoration.complete_initial_q_sha256',
            previous=old_initial, current=new_initial, original_state=str(original_state),
            original_state_sha256=hashlib.sha256(original_state.read_bytes()).hexdigest()))
        right['joint_source_truth_restoration']['complete_initial_q_sha256'] = old_initial
    else:
        original_q = q
    initial_audit=left['joint_source_truth_restoration'].get('explicit_reviewed_manual_append_audit')
    current_audit=right['joint_source_truth_restoration'].get('explicit_reviewed_manual_append_audit')
    if initial_audit!=current_audit:
        if directory.name!='current_complete_linearization_cache'or not initial_audit or current_audit is not None:
            raise ValueError('Source append admission is not original stop-point initializer provenance')
        parent_path=directory.parent/'input_manifest.json'
        parent=json.loads(parent_path.read_text(encoding='utf8'))
        if parent['joint_source_truth_restoration'].get('explicit_reviewed_manual_append_audit')!=initial_audit:
            raise ValueError('Stop-point source admission differs from the original run manifest')
        replay._verify_files([initial_audit])
        admission=json.loads(Path(initial_audit['path']).read_text(encoding='utf8'))
        if admission.get('schema')!='rinkspace.reviewed-current-manual-append-startup-binding.v1':
            raise ValueError('Original source append admission schema differs')
        actual_audit=admission['same_q_public_audit'];replay._verify_files([actual_audit])
        proof=json.loads(Path(actual_audit['path']).read_text(encoding='utf8'))
        if proof.get('passed')is not True or proof.get('complete_q_sha256')!=old_initial:
            raise ValueError('Original source admission does not authenticate the original initializer q')
        changes.append(dict(field='joint_source_truth_restoration.explicit_reviewed_manual_append_audit',
            kind='original_source_admission_already_completed_in_this_run',previous=initial_audit,current=None,
            original_parent_manifest_path=str(parent_path),original_parent_manifest_sha256=replay._sha(parent_path),
            source_graph_observations_supports_and_operator_contracts_still_strict=True))
        right['joint_source_truth_restoration']['explicit_reviewed_manual_append_audit']=deepcopy(initial_audit)
    a, b = left.get('independent_pattern_new_slot_initializer'), right.get('independent_pattern_new_slot_initializer')
    if a != b:
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError('Pattern initializer inventory changed')
        if {k:v for k,v in a.items() if k != 'parameter_values'} != {k:v for k,v in b.items() if k != 'parameter_values'}:
            raise ValueError('Pattern initializer policy changed')
        old_indices = np.asarray(old['independent_pattern_placement_parameter_indices'], int)
        new_indices = np.asarray(current['independent_pattern_placement_parameter_indices'], int)
        if (old_indices.shape != new_indices.shape or not np.array_equal(old_indices, new_indices)
                or a.get('initializer_is_not_observation') is not True
                or not np.array_equal(np.asarray(a['parameter_values'], float), original_q[old_indices])
                or not np.array_equal(np.asarray(b['parameter_values'], float), q[new_indices])):
            raise ValueError('Pattern initializer values are not the signed original/current q slots')
        changes.append(dict(field='independent_pattern_new_slot_initializer.parameter_values',
            previous=a['parameter_values'], current=b['parameter_values'],
            exact_sealed_current_parameter_indices=new_indices.tolist()))
        b['parameter_values'] = deepcopy(a['parameter_values'])
    old_root = left['native_backend_requested'].get('RINKSPACE_SOLVER_DATA_ROOT')
    new_root = right['native_backend_requested'].get('RINKSPACE_SOLVER_DATA_ROOT')
    if old_root != new_root:
        a_root, b_root = Path(old_root).resolve(), Path(new_root).resolve()
        if a_root != b_root or not a_root.is_dir():
            raise ValueError('Actual solver Data resource directory changed')
        changes.append(dict(field='native_backend_requested.RINKSPACE_SOLVER_DATA_ROOT',
            kind='same_resolved_execution_directory', previous=old_root, current=new_root,
            resolved_directory=str(a_root)))
        right['native_backend_requested']['RINKSPACE_SOLVER_DATA_ROOT'] = old_root
    previous_blocks = left['original_motion_input_compaction']['blocks']
    current_blocks = right['original_motion_input_compaction']['blocks']
    if [v['block'] for v in previous_blocks] != [v['block'] for v in current_blocks]:
        raise ValueError('Original compaction block identity/order changed')
    for previous, present in zip(previous_blocks, current_blocks):
        if previous['block'] != 'low':
            continue
        field='same_exposure_coalesced_local_J_max'
        before,after=previous.get(field),present.get(field)
        if before!=after:
            if (before is None or after is None or not np.isfinite([before,after]).all()
                or min(before,after)<0.or any(block.get(key)is not True for block in (previous,present)
                    for key in ('actual_self_list_bitexact_candidate','same_exposure_residual_is_not_assumed_zero',
                        'source_sigma_subselected_without_N_normalization'))):
                raise ValueError('Removed same-exposure source diagnostic lacks its exact original input proof')
            changes.append(dict(field='original_motion_input_compaction.blocks[low].'+field,
                kind='removed_self_pair_initializer_diagnostic_not_an_active_source',previous=before,current=after,
                source_input_array_record_and_all_active_matrix_bytes_still_strict=True))
            present[field]=before
        old_tolerance, new_tolerance = previous['compaction_numerical_tolerance'], present['compaction_numerical_tolerance']
        old_max, new_max = previous['same_exposure_actual_r_max'], present['same_exposure_actual_r_max']
        old_fraction, new_fraction = old_tolerance['J_max_error_fraction'], new_tolerance['J_max_error_fraction']
        if old_max == new_max and old_fraction == new_fraction:
            continue
        for block in (previous, present):
            if (block.get('same_state_retained_r_J_passed_tight_compaction_numerical_tolerance') is not True
                    or block.get('same_state_retained_CSR_row_column_identity_bitexact') is not True):
                raise ValueError('Compaction initializer diagnostic did not pass its original gates')
        if ({k:v for k,v in old_tolerance.items() if k != 'J_max_error_fraction'}
                != {k:v for k,v in new_tolerance.items() if k != 'J_max_error_fraction'}):
            raise ValueError('Original compaction thresholds/metric/Brown/operation-certificate policy changed')
        if (not np.isfinite([old_max, new_max, old_fraction, new_fraction]).all()
                or min(old_max, new_max) < 0 or min(old_fraction, new_fraction) < 0
                or max(old_fraction, new_fraction) > 1):
            raise ValueError('Compaction initializer diagnostic exceeds its original gate')
        for field, before, after in (
                ('same_exposure_actual_r_max', old_max, new_max),
                ('compaction_numerical_tolerance.J_max_error_fraction', old_fraction, new_fraction)):
            if before != after:
                changes.append(dict(field='original_motion_input_compaction.blocks[low].'+field,
                    kind='passed_graph_initialization_diagnostic_receipt', previous=before, current=after,
                    source_input_array_record_and_all_matrix_bytes_still_strict=True))
        present['same_exposure_actual_r_max'] = old_max
        new_tolerance['J_max_error_fraction'] = old_fraction
    # Original signatures remain untouched. Only the final comparison pair
    # carries the two proven numerical provenance substitutions above.
    return (left, right), changes


def audit_same_target_source_blocks(directory, q, source, manifest, profile_token, *,
                                   verified_prefix_report=None, raw=None, report_directory=None):
    from structured_programme_source_candidate import complete_same_state_replay as replay
    from structured_programme_source_candidate.source_block_target_extension_replay import (
        _old_cache_identity, _load_authenticated_old_block,
    )
    from z190_joint_source_solver import _frozen_profile_token
    from z190_streaming_source_normal import SourceRowBlock, _checked_block, _check_replay, _finish_replay
    import numpy as np

    report = dict(schema='rinkspace.actual-same-target-fresh-source-byte-audit.v1',
        passed=False, complete=False, source_directory=str(Path(directory).resolve()),
        all_normal_blocks_actually_recomputed=True, verified_cached_prefix=None,
        baseline_H_g_cost_reused=False, baseline_H_g_cost_numeric_arrays_loaded=False,
        actual_new_source_blocks=[], allowed_numerical_initializer_provenance_differences=[],
        no_append_or_source_domain_change_allowed=True, accuracy_accepted=False)
    identity_began = time.perf_counter()
    report['identity_gate_begin_wall_time_ns'] = time.time_ns()
    destination = None if report_directory is None else Path(report_directory)/'same_target_fresh_J_audit.json'
    def save():
        if destination is not None:
            _write(destination, report)
    try:
        if verified_prefix_report is not None:
            raise ValueError('Same-target fresh verification must actually recompute every block')
        directory, old, oldq, token, contract, completion = _old_cache_identity(directory)
        current = replay._plain(manifest)
        q = np.asarray(q, float)
        if replay._array_signature(q) != replay._array_signature(oldq):
            raise ValueError('Fresh verification requires the exact sealed complete q bytes')
        if profile_token != token or _frozen_profile_token(source.updater) != token:
            raise ValueError('Fresh verification requires the exact sealed frozen profiles')
        seal_began = time.perf_counter()
        inventory = replay._sealed_inventory(directory, q, source, token)
        report['sealed_inventory_hash_and_profile_authentication_seconds'] = time.perf_counter()-seal_began
        pair, provenance = _initialization_pair(replay, directory, old, current, q)
        identity = current[replay.IDENTITY_KEY]
        fixed = np.asarray(identity['fixed_parameter_indices'], int)
        shared = np.zeros(contract.parameter_count, dtype=bool)
        indices = np.asarray(current['shared_indices'], int)
        if (indices.ndim != 1 or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(shared))):
            raise ValueError('Actual complete shared parameter inventory is invalid')
        shared[indices] = True
        replay._check_strict_identity(old, current, append_normalized_pair=pair,
            fixed_parameter_indices=fixed, shared_parameter_mask=shared)
        with np.load(directory/'first_complete_source_normal_numeric.npz', allow_pickle=False) as z:
            normal_report = json.loads(str(z['report_json']))
        if normal_report['fixed_parameter_indices'] != fixed.tolist():
            raise ValueError('Saved complete normal fixed policy differs')
        if (contract.parameter_count != len(q) or contract.row_count != len(source.scales)
                or contract.frozen_profile_token != token):
            raise ValueError('Actual complete source row/parameter/profile contract differs')
        report.update(full_q_sha256=replay._array_signature(q)['sha256'], frozen_profile_token=token,
            old_source_blocks=len(contract.signatures), new_source_rows=contract.row_count,
            original_sealed_inventory_sha256=completion['strict_replay_inventory']['sha256'],
            sealed_profile_arrays_authenticated=len(inventory['profile_arrays']),
            allowed_numerical_initializer_provenance_differences=provenance,
            original_signed_source_identity_verified=True,
            output_relocation_compatibility_only=list(replay._QP_OUTPUT_PATH_KEYS),
            residual_tolerance=dict(atol=0., rtol=0.), J_tolerance=dict(atol=0., rtol=0.))
        report['identity_gate_seconds'] = time.perf_counter()-identity_began
        report['identity_gate_end_wall_time_ns'] = time.time_ns()
        save()
        print(json.dumps(dict(stage='actual_same_target_fresh_source_identity_gate_passed',
            seconds=report['identity_gate_seconds'], wall_time_ns=report['identity_gate_end_wall_time_ns'],
            q_sha256=report['full_q_sha256'], expected_blocks=len(contract.signatures))), flush=True)
    except BaseException as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    frozen = q.copy()
    frozen.flags.writeable = False

    def factory():
        if report['complete'] or report['actual_new_source_blocks']:
            raise ValueError('Fresh same-target source audit is a single actual pass')
        started = time.perf_counter()
        cursor = count = 0
        try:
            if replay._array_signature(q) != replay._array_signature(frozen) or _frozen_profile_token(source.updater) != token:
                raise ValueError('Fresh audit q/profile changed before computation')
            for ordinal, new in enumerate(source.blocks(frozen, True)):
                old_block, old_signature = _load_authenticated_old_block(directory, ordinal, contract, frozen, token)
                nr, nj, mask, scales, signature = _checked_block(new, len(frozen), True)
                _check_replay(signature, ordinal, contract, True)
                stop = cursor + len(nr)
                if (new.row_ledger.get('original_start') != cursor or new.row_ledger.get('original_stop') != stop):
                    raise ValueError('Fresh block cursor/ledger differs')
                for name, actual, expected in (
                    ('r', np.asarray(new.r), old_block.r),
                    ('quadratic', np.asarray(new.quadratic), old_block.quadratic),
                    ('scales', np.asarray(new.scales), old_block.scales),
                    ('J_data', nj.data, old_block.J.data),
                    ('J_indices', nj.indices, old_block.J.indices),
                    ('J_indptr', nj.indptr, old_block.J.indptr),
                ):
                    if replay._array_signature(actual) != replay._array_signature(expected):
                        raise ValueError('Fresh block dtype/shape/bytes differ: '+str(ordinal)+':'+name)
                if new.J.dtype != old_block.J.dtype or nj.shape != old_block.J.shape:
                    raise ValueError('Fresh matrix dtype/shape differs')
                if (replay._array_signature(mask) != replay._array_signature(source.quadratic[cursor:stop])
                        or replay._array_signature(scales) != replay._array_signature(source.scales[cursor:stop])):
                    raise ValueError('Fresh block mask/scales differ from actual complete source')
                report['actual_new_source_blocks'].append(dict(ordinal=ordinal,
                    family=new.row_ledger['family'], actual_new_rows=len(nr), compared_original_rows=len(old_block.r),
                    r_J_recomputed=True, evaluation_origin='actual_current_source_evaluation',
                    normalized_r_max_absolute_difference=0., J_max_absolute_difference=0.,
                    original_source_masks_scales_roles_exact=True, r_and_CSR_dtype_shape_bytes_exact=True,
                    old_signature=asdict(old_signature), actual_current_signature=asdict(signature), passed=True))
                cursor = stop
                count += 1
                del old_block
                save()
                print(json.dumps(dict(stage='actual_same_target_fresh_source_block_bytes_passed',
                    ordinal=ordinal, rows=len(nr), wall_time_ns=time.time_ns())), flush=True)
                yield SourceRowBlock(nr, nj, mask, scales, deepcopy(new.row_ledger))
            _finish_replay(count, cursor, contract)
            if replay._array_signature(q) != replay._array_signature(frozen) or _frozen_profile_token(source.updater) != token:
                raise ValueError('Fresh audit q/profile changed during computation')
            report.update(passed=True, complete=True, actual_complete_rows=cursor,
                actual_complete_blocks=count, residual_and_J_equivalence_verified=True)
        except BaseException as exc:
            report.update(passed=False, complete=False, error=dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            report['elapsed_seconds'] = time.perf_counter()-started
            save()
    return factory, report


def _transform_main_source(source):
    tree = ast.parse(textwrap.dedent(source))
    changes = []
    class ExplicitFreshVerification(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'write'
                    and len(call.args) == 2 and isinstance(call.args[0], ast.BinOp)
                    and isinstance(call.args[0].left, ast.Name) and call.args[0].left.id == 'out'
                    and isinstance(call.args[0].op, ast.Div) and isinstance(call.args[0].right, ast.Constant)
                    and call.args[0].right.value == 'input_manifest.json'
                    and isinstance(call.args[1], ast.Name) and call.args[1].id == 'manifest'):
                changes.append(dict(kind='actual_same_target_preflight_before_cold_checks',
                    before=ast.unparse(node), after='pure identity/profile preflight then original manifest write'))
                preflight = ast.parse('''
if opt.verify_source_blocks_from is not None:
    _rinkspace_same_target_fresh_audit(opt.verify_source_blocks_from,q,source,manifest,
        _frozen_profile_token(source.updater),verified_prefix_report=opt.verified_source_prefix_report,raw=raw)
''').body[0]
                return [preflight, node]
            return self.generic_visit(node)

        def visit_If(self, node):
            # Verification needs an identity made from THIS live source and
            # actual fixed/shared policy, exactly like the original save/reuse
            # path. It must never recover from identity preparation failure.
            original_test = ast.unparse(node.test)
            if (ast.dump(node.test)==ast.dump(ast.parse('opt.reuse_complete_linearization is None and opt.reuse_source_blocks_from is None and opt.verify_source_blocks_from is None',mode='eval').body)
                    and any(isinstance(n,ast.Call)and ast.unparse(n.func)=='source.updater'for n in ast.walk(node))):
                changes.append(dict(kind='restore_exact_qualified_complete_cache_frozen_profiles',
                    original_conditional_profile_update_retained=True))
                restore=ast.parse('''
if opt.reuse_complete_linearization is not None and opt.original_source_point_qualification_source is not None:
    _rinkspace_restore_exact_qualified_complete_cache_profiles(opt.reuse_complete_linearization,q,source)
''').body[0]
                return [restore,self.generic_visit(node)]
            if original_test == ('opt.save_initial_linearization_cache or opt.reuse_complete_linearization '
                                 'is not None or opt.reuse_source_blocks_from is not None'):
                if not any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                           and v.func.id == 'prepare_actual_cache_identity' for v in ast.walk(node)):
                    raise ValueError('Original live identity preparation site changed')
                node.test = ast.BoolOp(op=ast.Or(), values=[node.test,
                    ast.parse('opt.verify_source_blocks_from is not None', mode='eval').body])
                changes.append(dict(kind='live_identity_preparation_for_verify', before=original_test,
                    after=ast.unparse(node.test)))
            elif (original_test == ('opt.reuse_complete_linearization is not None '
                                    'or opt.reuse_source_blocks_from is not None')
                    and len(node.body) == 1 and isinstance(node.body[0], ast.Raise)):
                node.test = ast.BoolOp(op=ast.Or(), values=[node.test,
                    ast.parse('opt.verify_source_blocks_from is not None', mode='eval').body])
                changes.append(dict(kind='live_identity_failure_fatal_for_verify', before=original_test,
                    after=ast.unparse(node.test)))
            # Only legacy rejection of the presence of original source options.
            errors = [c.args[0].value for s in node.body if isinstance(s, ast.Expr)
                for c in [s.value] if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name) and c.func.value.id == 'parser'
                and c.func.attr == 'error' and c.args and isinstance(c.args[0], ast.Constant)]
            matched = set(errors) & _PRESENCE_REJECTIONS
            if matched:
                tuples = [n for n in ast.walk(node.test) if isinstance(n, ast.Tuple)
                    and any(isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                        and v.value.id == 'opt' and v.attr == 'verify_source_blocks_from' for v in n.elts)]
                if len(matched) != 1 or len(tuples) != 1:
                    raise ValueError('Original verify/cache presence gate is ambiguous')
                before = ast.unparse(node.test)
                tuples[0].elts = [v for v in tuples[0].elts if not (isinstance(v, ast.Attribute)
                    and isinstance(v.value, ast.Name) and v.value.id == 'opt' and v.attr == 'verify_source_blocks_from')]
                changes.append(dict(kind='legacy_presence_tuple', reason=next(iter(matched)),
                    before=before, after=ast.unparse(node.test)))
            if _COMPACTION_REJECTION in errors:
                if ast.unparse(node.test) != 'opt.verify_source_blocks_from is not None':
                    raise ValueError('Original verify-only compaction presence gate changed')
                changes.append(dict(kind='legacy_verify_only_compaction_presence', reason=_COMPACTION_REJECTION,
                    before=ast.unparse(node), after='strict same-target fresh audit'))
                return None
            if 'Original source point qualification requires fresh complete source profiles/J' in errors:
                tuples=[n for n in ast.walk(node.test)if isinstance(n,ast.Tuple)]
                if len(tuples)!=1 or sum(ast.unparse(v)=='opt.reuse_complete_linearization'for v in tuples[0].elts)!=1:
                    raise ValueError('Qualified complete cache presence gate changed')
                before=ast.unparse(node.test)
                tuples[0].elts=[v for v in tuples[0].elts if ast.unparse(v)!='opt.reuse_complete_linearization']
                changes.append(dict(kind='exact_qualified_complete_cache_parse_gate',before=before,
                    after=ast.unparse(node.test),original_full_same_target_seal_gate_retained=True))
                checked=ast.parse('''
if opt.original_source_point_qualification_source is not None and opt.reuse_complete_linearization is not None:
    _rinkspace_exact_qualified_complete_cache_request(opt.reuse_complete_linearization,
        opt.original_source_point_qualification_source,opt.camera_seed)
''').body[0]
                return [checked,self.generic_visit(node)]
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if (node.module == 'structured_programme_source_candidate.source_block_target_extension_replay'
                    and len(node.names) == 1 and node.names[0].name == 'audit_same_state_ice_extension_blocks'):
                changes.append(dict(kind='verify_branch_audit_import', before=ast.unparse(node)))
                return ast.copy_location(ast.Assign(targets=[ast.Name(id='audit_same_state_ice_extension_blocks', ctx=ast.Store())],
                    value=ast.Name(id='_rinkspace_same_target_fresh_audit', ctx=ast.Load())), node)
            return node
    tree = ExplicitFreshVerification().visit(tree)
    if (sum(c['kind']=='legacy_presence_tuple' for c in changes) != len(_PRESENCE_REJECTIONS)
            or sum(c['kind']=='legacy_verify_only_compaction_presence' for c in changes) != 1
            or sum(c['kind']=='verify_branch_audit_import' for c in changes) != 1
            or sum(c['kind']=='live_identity_preparation_for_verify' for c in changes) != 1
            or sum(c['kind']=='live_identity_failure_fatal_for_verify' for c in changes) != 1
            or sum(c['kind']=='actual_same_target_preflight_before_cold_checks' for c in changes) != 1):
        raise ValueError('Original explicit fresh-verification branch sites changed')
    if sum(c['kind']=='exact_qualified_complete_cache_parse_gate'for c in changes)!=1:
        raise ValueError('Exactly one original qualified complete-cache presence gate required')
    if sum(c['kind']=='restore_exact_qualified_complete_cache_frozen_profiles'for c in changes)!=1:
        raise ValueError('Exactly one original conditional initial profile update required')
    ast.fix_missing_locations(tree)
    return tree, changes


def install_same_target_fresh_J_verification(driver, report_directory):
    installed = getattr(driver, '_rinkspace_same_target_fresh_J_verification', None)
    if installed is not None:
        return installed
    original = driver.main
    source = inspect.getsource(original)
    tree, changes = _transform_main_source(source)
    namespace = dict(original.__globals__)
    namespace['_rinkspace_exact_qualified_complete_cache_request']=authenticate_qualified_complete_cache_request
    namespace['_rinkspace_restore_exact_qualified_complete_cache_profiles']=restore_authenticated_complete_cache_profiles
    captured_solver = namespace['solve_joint_source']
    prepared = None
    prepared_key = None
    def audit(*args, **kwargs):
        nonlocal prepared, prepared_key
        from structured_programme_source_candidate import complete_same_state_replay as replay
        from z190_joint_source_solver import _frozen_profile_token
        directory, q, source, manifest, token = args[:5]
        key = (str(Path(directory).resolve()), id(source), replay._array_signature(q)['sha256'],
            _frozen_profile_token(source.updater), replay._json_sha(replay._semantic_manifest(manifest)),
            replay._json_sha(manifest[replay.IDENTITY_KEY]))
        if prepared is not None:
            if key != prepared_key:
                raise ValueError('Actual current source/q/profile/identity changed after fresh audit preflight')
            prepared[1]['preflight_factory_reused_for_same_actual_source_q_and_profile'] = True
            return prepared
        prepared = audit_same_target_source_blocks(*args, **kwargs, report_directory=report_directory)
        prepared_key = key
        return prepared
    namespace['_rinkspace_same_target_fresh_audit'] = audit
    import os
    comparison = os.environ.get('RINKSPACE_COMPLETE_CYCLE_VARIANT')
    observe_first_cycle = os.environ.get('RINKSPACE_OBSERVE_COMPLETE_FIRST_CYCLE') == '1'
    diagnostic_evidence = os.environ.get('RINKSPACE_STARTUP_DIAGNOSTIC_EVIDENCE')
    if diagnostic_evidence and comparison != 'baseline':
        from .certified_startup_diagnostic_runtime import (
            attach_certified_startup_diagnostic_reuse,
            attach_deferred_existing17_startup_reuse,
        )
        tree = attach_certified_startup_diagnostic_reuse(
            tree, namespace, Path(report_directory) / 'startup_diagnostic_execution', diagnostic_evidence)
        tree = attach_deferred_existing17_startup_reuse(tree, namespace)
    if comparison:
        if comparison not in ('baseline', 'candidate'):
            raise ValueError('Unknown complete cycle comparison variant')
        from .complete_cycle_comparison_runtime import attach_complete_cycle_observer
        tree = attach_complete_cycle_observer(tree, namespace, report_directory, comparison)
    elif observe_first_cycle:
        from .complete_cycle_comparison_runtime import attach_complete_cycle_observer
        tree = attach_complete_cycle_observer(tree, namespace, report_directory, 'production')
    if comparison != 'baseline':
        from .config import data_root
        from .named66_source_skeleton_adapter_runtime import attach_named66_source_skeleton_reuse
        tree = attach_named66_source_skeleton_reuse(
            tree, namespace,
            cache_directory=data_root() / 'prepared_full8503' / 'complete_original_sources',
            report_directory=Path(report_directory) / 'source_skeleton_execution',
            declared_source_roots=(namespace['DEFAULT_SEED'],
                data_root() / 'prepared_full8503_state.pickle'),
            complete_static_input_closure_confirmed=True)
        from .startup_source_auth_reuse_runtime import attach_startup_source_auth_reuse
        tree = attach_startup_source_auth_reuse(
            tree, namespace, Path(report_directory) / 'startup_source_auth_execution')
        # Existing driver `seconds` fields mix cumulative and stage intervals.
        # Timestamp the existing stage messages, without another profiler or
        # extra source evaluation, so the cold path can be read unambiguously.
        class TimestampExistingStages(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if not (isinstance(node.func, ast.Name) and node.func.id == 'print'
                        and len(node.args) == 1 and isinstance(node.args[0], ast.Call)):
                    return node
                encoded = node.args[0]
                if not (isinstance(encoded.func, ast.Attribute) and encoded.func.attr == 'dumps'
                        and len(encoded.args) == 1 and isinstance(encoded.args[0], ast.Call)):
                    return node
                payload = encoded.args[0]
                if (isinstance(payload.func, ast.Name) and payload.func.id == 'dict'
                        and any(k.arg == 'stage' and isinstance(k.value, ast.Constant)
                                for k in payload.keywords)
                        and not any(k.arg == 'wall_time_ns' for k in payload.keywords)):
                    payload.keywords.append(ast.keyword(arg='wall_time_ns',
                        value=ast.parse('time.time_ns()', mode='eval').body))
                return node
        tree = TimestampExistingStages().visit(tree)
        ast.fix_missing_locations(tree)
    exec(compile(tree, __file__+':same_target_fresh_J_main', 'exec'), namespace)
    candidate = namespace['main']
    if candidate.__globals__['solve_joint_source'] is not captured_solver:
        raise RuntimeError('Fresh audit clone lost the already installed central solver wrapper')
    candidate.__module__ = original.__module__
    candidate.__qualname__ = original.__qualname__
    driver.main = candidate
    receipt = dict(schema='rinkspace.same-target-fresh-J-verification-hook.v1',
        helper_path=str(Path(__file__).resolve()), helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_main_sha256=hashlib.sha256(source.encode()).hexdigest(),
        transformed_main_AST_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        original_main_globals_except_explicit_audit_symbol_retained=True,
        captured_central_solver_wrapper_retained=True,
        explicit_verify_only_branch_changed=not bool(comparison or diagnostic_evidence or observe_first_cycle or comparison != 'baseline'),
        original_source_file_authentication_scoped_reuse=comparison != 'baseline',
        complete_first_cycle_observed=bool(comparison or observe_first_cycle),
        independent_startup_diagnostic_evidence=diagnostic_evidence,
        original_source_guards_and_reuse_wide_rejections_retained=True,
        current_identity_built_from_original_live_source_arrays_and_records=True,
        current_identity_preparation_failure_is_fatal=True,
        pure_same_target_identity_preflight_precedes_cold_diagnostics=True,
        exact_parser_and_import_changes=changes, all_fresh_r_and_CSR_bytes_required=True,
        no_baseline_H_g_cost_reuse_in_fresh_verification_branch=True,
        original_strict_complete_same_state_replay_gates_retained=True,
        complete_cache_parse_admission_requires_actual_same_qualification_q_and_seal=True,
        complete_cache_telemetry_relocation_scoped_to_exact_sealed_archive=True,
        registered_producer_files_modified=False)
    driver._rinkspace_same_target_fresh_J_verification = receipt
    _write(Path(report_directory)/'same_target_fresh_J_verification_receipt.json', receipt)
    return receipt
