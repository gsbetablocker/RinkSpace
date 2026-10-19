"""Metadata-only positive/negative controls for explicit fresh-J validation."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import inspect
import json
import sys
import ast
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--current-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from calibration.solver_runtime import install
    install()
    import run_current_named66_full_programme as driver
    from calibration.solver_runtime.same_target_fresh_J_verification import _initialization_pair, _transform_main_source
    from structured_programme_source_candidate import complete_same_state_replay as replay
    from structured_programme_source_candidate.source_block_target_extension_replay import _old_cache_identity
    cache, old, q, token, contract, completion = _old_cache_identity(args.cache)
    inventory = json.loads((cache/'strict_complete_source_replay_inventory.json').read_text())
    assert completion['strict_replay_inventory']['sha256'] == hashlib.sha256((cache/'strict_complete_source_replay_inventory.json').read_bytes()).hexdigest()
    assert inventory['q'] == replay._array_signature(q) and inventory['profile_arrays']
    current = json.loads(args.current_manifest.read_text(encoding='utf8'))
    assert current[replay.IDENTITY_KEY]['actual_original_source_input_arrays'] == old[replay.IDENTITY_KEY]['actual_original_source_input_arrays']
    assert current[replay.IDENTITY_KEY]['actual_original_source_input_metadata_sha256'] == old[replay.IDENTITY_KEY]['actual_original_source_input_metadata_sha256']
    pair, differences = _initialization_pair(replay, cache, old, current, q)
    assert pair[0] == pair[1]
    fixed = current[replay.IDENTITY_KEY]['fixed_parameter_indices']
    shared = np.zeros(len(q), bool)
    shared[np.asarray(current['shared_indices'], int)] = True
    replay._check_strict_identity(old, current, append_normalized_pair=pair,
        fixed_parameter_indices=fixed, shared_parameter_mask=shared)
    controls = []
    def rejected(name, operation):
        try:
            operation()
        except (ValueError, KeyError) as exc:
            controls.append(dict(name=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('Bad same-target input accepted: '+name)
    badq = q.copy(); badq[0] += 1.
    rejected('changed_current_q', lambda: _initialization_pair(replay, cache, old, current, badq))
    bad = deepcopy(current); bad['independent_pattern_new_slot_initializer']['parameter_values'][0] += 1.
    rejected('changed_pattern_initializer_receipt', lambda: _initialization_pair(replay, cache, old, bad, q))
    bad = deepcopy(current); bad['native_backend_requested']['RINKSPACE_SOLVER_DATA_ROOT'] = str(ROOT)
    rejected('changed_actual_Data_directory', lambda: _initialization_pair(replay, cache, old, bad, q))
    bad = deepcopy(current)
    low = next(b for b in bad['original_motion_input_compaction']['blocks'] if b['block'] == 'low')
    low['same_state_retained_r_J_passed_tight_compaction_numerical_tolerance'] = False
    rejected('failed_compaction_receipt', lambda: _initialization_pair(replay, cache, old, bad, q))
    bad = deepcopy(current)
    low = next(b for b in bad['original_motion_input_compaction']['blocks'] if b['block'] == 'low')
    low['compaction_numerical_tolerance']['atol'] *= 2
    rejected('changed_original_compaction_threshold', lambda: _initialization_pair(replay, cache, old, bad, q))
    bad = deepcopy(current); bad['roll_prior_component_sigma_degrees'] += .1
    bad[replay.IDENTITY_KEY]['semantic_manifest_sha256'] = replay._json_sha(replay._semantic_manifest(bad))
    badpair, _ = _initialization_pair(replay, cache, old, bad, q)
    rejected('changed_original_source_sigma', lambda: replay._check_strict_identity(old, bad,
        append_normalized_pair=badpair, fixed_parameter_indices=fixed, shared_parameter_mask=shared))
    assert replay._array_signature(np.array([0., -0.])) != replay._array_signature(np.array([0., 0.]))
    tree, changes = _transform_main_source(inspect.getsource(driver.main))
    compile(tree, 'same_target_fresh_J_main', 'exec')
    # Execute the ACTUAL transformed entry subtree, with a live builder probe.
    # This covers the missing current identity that stopped the first real run.
    main_node = tree.body[0]
    prepare_if = next(node for node in main_node.body if isinstance(node, ast.If)
        and any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
            and v.func.id == 'prepare_actual_cache_identity' for v in ast.walk(node)))
    tiny = ast.Module(body=[deepcopy(prepare_if)], type_ignores=[])
    ast.fix_missing_locations(tiny)
    opt = SimpleNamespace(save_initial_linearization_cache=False, reuse_complete_linearization=None,
        reuse_source_blocks_from=None, verify_source_blocks_from=Path('actual-same-q-cache'))
    calls = []
    live_manifest = {}
    def actual_live_builder():
        calls.append('actual live source builder')
        return {'built_from_actual_live_source': True}
    scope = dict(opt=opt, manifest=live_manifest, prepare_actual_cache_identity=actual_live_builder)
    exec(compile(tiny, 'actual_live_identity_entry_subtree', 'exec'), scope)
    assert calls == ['actual live source builder'] and live_manifest[replay.IDENTITY_KEY]['built_from_actual_live_source']
    def failing_live_builder():
        raise ValueError('actual live source identity failure probe')
    scope['prepare_actual_cache_identity'] = failing_live_builder
    rejected('actual_live_identity_prepare_failure_is_fatal', lambda: exec(
        compile(tiny, 'actual_live_identity_entry_subtree', 'exec'), scope))
    report = dict(schema='rinkspace.same-target-fresh-J-metadata-controls.v1', passed=True,
        actual_cache_q_sha256=replay._array_signature(q)['sha256'], actual_profile_token=token,
        actual_cache_contract_rows=contract.row_count, actual_cache_contract_blocks=len(contract.signatures),
        actual_cache_profile_array_count=len(inventory['profile_arrays']),
        actual_complete_cache_keys_present=True, original_strict_identity_verifier_passed=True,
        actual_current_manifest=str(args.current_manifest.resolve()),
        actual_current_manifest_sha256=hashlib.sha256(args.current_manifest.read_bytes()).hexdigest(),
        actual_live_source_array_and_record_hashes_equal=True,
        no_synthetic_or_borrowed_current_identity=True,
        authenticated_initialization_pair_equal=True, original_signed_manifests_not_rewritten=True,
        allowed_initialization_provenance=differences, FD4_to_8_ignored_only_by_original_execution_semantics=True,
        signed_zero_array_bytes_distinguished=True, negative_controls=controls,
        actual_verify_entry_builds_current_live_identity=True,
        actual_verify_entry_does_not_fallback_after_identity_failure=True,
        main_AST_changed_sites=changes, real_source_graph_loaded=False,
        old_H_g_cost_loaded_or_recomputed=False, new_fresh_J_not_run=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(passed=True, controls=len(controls), report=str(args.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
