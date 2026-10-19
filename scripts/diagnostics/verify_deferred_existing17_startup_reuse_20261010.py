"""Saved manifests and lightweight bindings only; no graph/solver/pool."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
import numpy as np
from calibration.solver_runtime import certified_startup_diagnostic_runtime as helper
from calibration.solver_runtime.startup_source_auth_reuse_runtime import attach_startup_source_auth_reuse
from calibration.solver_runtime.runtime import source_path

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
DIAG = BASE / 'diagnostics/current_solver_takeover_20261009/startup_existing17_small'


def main():
    old_directory = BASE / 'complete_cycle_comparison_20261010/baseline'
    directory = BASE / 'okada_full9_review82_speed_candidate_20261010_115855_545'
    manifest = json.loads((directory / 'input_manifest.json').read_text(encoding='utf8'))
    old = json.loads((old_directory / 'input_manifest.json').read_text(encoding='utf8'))
    with np.load(directory / 'actual_initial_normal_state.npz', allow_pickle=False) as saved:
        q = saved['q'].copy()
    with np.load(old_directory / 'actual_initial_normal_state.npz', allow_pickle=False) as saved:
        old_q = saved['q'].copy()
    check_path = old_directory / 'actual_existing17_continuous_perimeter_value_J_crossregion_check.json'
    check = json.loads(check_path.read_text(encoding='utf8'))
    helper._validate_existing17_evidence(check)
    invalid = deepcopy(check)
    invalid['block_checks'][0]['FD_checks'][0]['FD_error'] = 1.
    try:
        helper._validate_existing17_evidence(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Failed historical original FD accepted')
    text = source_path('run_current_named66_full_programme').read_text(encoding='utf8')
    tree = ast.parse(text)
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    main_sha = hashlib.sha256(''.join(text.splitlines(keepends=True)[main_node.lineno - 1:main_node.end_lineno]).encode()).hexdigest()
    evidence = dict(directory=str(old_directory), files={'existing17': dict(path=str(check_path), sha256=helper._sha(check_path))},
        manifest=old, source_facts=helper._source_facts(old, old_q),
        q_sha256=hashlib.sha256(old_q.tobytes()).hexdigest(), frozen_profile_token='test-only-no-profile-import',
        original_main_sha256=main_sha, existing17=dict(path=str(check_path), sha256=helper._sha(check_path)),
        existing17_unavailable=None)
    calls = []
    revision = manifest['original_motion_far_board_role_correction']['existing_finite_geometry_revision']
    state = SimpleNamespace(geometry=SimpleNamespace(_existing_finite_perimeter_binding=SimpleNamespace(
        records=[None] * 17, binding_id=revision['binding_sha256'])))

    def original(state, actual_q, *, output_path=None):
        calls.append(actual_q.copy())
        return check

    def prepare(name):
        destination = DIAG / name
        output = destination / 'run'
        output.mkdir(parents=True, exist_ok=True)
        namespace = {}
        with patch.object(helper, '_read_evidence', return_value=evidence), patch.object(helper, '_producer_files', return_value=[]):
            changed = helper.attach_certified_startup_diagnostic_reuse(ast.parse(text), namespace, destination, old_directory)
        changed = helper.attach_deferred_existing17_startup_reuse(changed, namespace)
        changed = attach_startup_source_auth_reuse(changed, namespace, destination / 'auth')
        compile(changed, '<original-driver-transformed>', 'exec')
        boundary = next(node for node in ast.walk(changed) if isinstance(node, ast.If)
            and any(isinstance(part, ast.Name) and part.id == '_rinkspace_complete_existing17_diagnostic' for part in ast.walk(node.test)))
        previous = None
        for node in ast.walk(changed):
            for _, value in ast.iter_fields(node):
                if isinstance(value, list) and boundary in value:
                    previous = value[value.index(boundary) - 1]
        assert previous is not None and '_rinkspace_finish_startup_source_auth' in ast.dump(previous)
        opt = SimpleNamespace(camera_seed='explicit-test-checkpoint', reuse_wide_linearization=None, output=output)
        path = destination / 'bootstrap/actual_existing17_continuous_perimeter_value_J_crossregion_check.json'
        return namespace, opt, path, destination

    namespace, opt, path, destination = prepare('reuse')
    assert namespace['_rinkspace_defer_existing17_diagnostic'](opt, state, q, path, original) is None
    source = SimpleNamespace(scales=np.ones(1))
    assert namespace['_rinkspace_complete_existing17_diagnostic'](opt, manifest, q, source) is True
    assert not calls and not path.exists()
    report = json.loads((destination / 'existing17_diagnostic_reuse.json').read_text(encoding='utf8'))
    assert report['reused'] and report['historical_q_values_claimed_as_current'] is False
    assert report['current_existing17_FD_and_cross_region_test_executed'] is False
    assert 'FD_error' not in json.dumps(report) and 'current_facets' not in json.dumps(report)
    namespace, opt, path, destination = prepare('changed_source')
    namespace['_rinkspace_defer_existing17_diagnostic'](opt, state, q, path, original)
    changed_manifest = deepcopy(manifest)
    changed_manifest['roll_prior_component_sigma_degrees'] += .25
    assert namespace['_rinkspace_complete_existing17_diagnostic'](opt, changed_manifest, q, source) is True
    assert len(calls) == 1 and np.array_equal(calls[0], q) and path.exists()
    failure = ValueError('original physical domain failure must propagate')
    def fail_original(*args, **kwargs):
        raise failure
    namespace, opt, path, destination = prepare('original_error')
    namespace['_rinkspace_defer_existing17_diagnostic'](opt, state, q, path, fail_original)
    try:
        namespace['_rinkspace_complete_existing17_diagnostic'](opt, changed_manifest, q, source)
    except ValueError as actual:
        assert actual is failure
    else:
        raise AssertionError('Original mathematical failure swallowed')
    result = dict(passed=True, saved_full_source_facts_match=helper._changed_fields(evidence['source_facts'], helper._source_facts(manifest, q)) == [],
        historical_FD_failure_rejected=True, reuse_contains_only_historical_PASS_link=True,
        source_fact_change_runs_original=True, original_domain_failure_propagates_same_object=True,
        real_driver_AST_compiles_with_source_auth_finish_before_finalizer=True,
        graph_solver_process_pool_started=False, helper_sha256=helper._sha(Path(helper.__file__)))
    helper._write(DIAG / 'actual_existing17_deferred_reuse_small_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
