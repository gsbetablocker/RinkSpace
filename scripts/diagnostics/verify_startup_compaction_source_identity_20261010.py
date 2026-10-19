"""Existing small manifest/q evidence only; no graph, FD or process pool."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
import numpy as np
from calibration.solver_runtime.certified_startup_diagnostic_runtime import (
    _source_facts, _changed_fields, _canonical)

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')


def saved(name):
    directory = BASE / name
    manifest = json.loads((directory / 'input_manifest.json').read_text(encoding='utf8'))
    with np.load(directory / 'actual_initial_normal_state.npz', allow_pickle=False) as archive:
        q = archive['q'].copy()
    return directory, manifest, q


def sigma_fields(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            if 'sigma' in key.lower() and isinstance(item, (float, int)) and not isinstance(item, bool):
                yield path + (key,), item
            yield from sigma_fields(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from sigma_fields(item, path + (index,))


def main():
    old_directory, old, old_q = saved('complete_cycle_comparison_20261010/baseline')
    directory, current, q = saved('okada_full9_review82_speed_candidate_20261010_115855_545')
    old_bytes, current_bytes = _canonical(old), _canonical(current)
    before_facts = _source_facts(old, old_q)
    actual = _source_facts(current, q)
    changes = _changed_fields(before_facts, actual)
    assert changes == [], changes
    assert _canonical(before_facts) == _canonical(actual)
    assert _canonical(old) == old_bytes and _canonical(current) == current_bytes
    cases = []
    blocks = current['original_motion_input_compaction']['blocks']
    index = next(i for i, row in enumerate(blocks) if row['removed_points'])
    for field in ('same_state_retained_CSR_row_column_identity_bitexact',
                  'same_state_retained_r_J_passed_tight_compaction_numerical_tolerance'):
        invalid = deepcopy(current)
        invalid['original_motion_input_compaction']['blocks'][index][field] = False
        try:
            _source_facts(invalid, q)
        except ValueError as error:
            cases.append(dict(mutation=field, rejected=True, reason=str(error)))
        else:
            raise AssertionError('Failed original compaction PASS flag accepted')
    for fraction in (-.1, 1.1, float('nan'), float('inf')):
        invalid = deepcopy(current)
        invalid['original_motion_input_compaction']['blocks'][index]['compaction_numerical_tolerance']['J_max_error_fraction'] = fraction
        try:
            _source_facts(invalid, q)
        except ValueError as error:
            cases.append(dict(mutation='J_max_error_fraction=' + repr(fraction), rejected=True, reason=str(error)))
        else:
            raise AssertionError('Invalid compaction error fraction accepted')
    # Real declared numerical sigma metadata must still change source facts.
    candidates = list(sigma_fields(current))
    assert candidates, 'Existing source manifest has no numerical sigma field'
    path, value = candidates[0]
    invalid = deepcopy(current)
    node = invalid
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = float(value) + .25
    differences = _changed_fields(actual, _source_facts(invalid, q))
    assert differences, 'Changed original declared source sigma identity accepted'
    cases.append(dict(mutation='.'.join(map(str, path)), before=value, after=node[path[-1]],
                      rejected_by_exact_source_fact_comparison=True, changed_fields=differences))
    invalid = deepcopy(current)
    invalid['original_motion_input_compaction']['blocks'][index]['exact_prefix_fields']['sigma'] = False
    differences = _changed_fields(actual, _source_facts(invalid, q))
    assert differences
    cases.append(dict(mutation='original_compaction_exact_prefix_sigma_identity=False',
                      rejected_by_exact_source_fact_comparison=True, changed_fields=differences))
    # Preserve the physical numerical tolerance itself, not just its PASS result.
    invalid = deepcopy(current)
    invalid['original_motion_input_compaction']['blocks'][index]['compaction_numerical_tolerance']['atol'] *= 2.
    assert _changed_fields(actual, _source_facts(invalid, q))
    cases.append(dict(mutation='original_compaction_physical_tolerance_atol', rejected=True))
    report = dict(passed=True, baseline_directory=str(old_directory), current_directory=str(directory),
        complete_normalized_source_facts_exactly_equal=True, changed_fields=changes,
        baseline_q_sha256=hashlib.sha256(old_q.tobytes()).hexdigest(), current_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        normalized_source_facts_sha256=hashlib.sha256(_canonical(actual).encode('utf8')).hexdigest(),
        source_manifests_unchanged=True, rejected_mutations=cases,
        reused_registered_compaction_identity_function=True,
        no_graph_or_derivative_or_solver_or_process_pool_executed=True)
    output = BASE / 'diagnostics/current_solver_takeover_20261009/startup_compaction_source_identity_check.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
