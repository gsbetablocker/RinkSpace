"""Resume a crashed source-identity run without changing its graph or priors.

Reconstruct through the original entry, then independently check the complete
objective at the saved accepted state before restoring the LM trace/damping.
The interrupted directory is read-only; all continuation goes to a fresh output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_saved(directory):
    phase = read_json(directory / 'phase_00.json')
    trace = read_json(directory / 'solver_trace.json')
    accepted = [row for row in trace['steps'] if row['status'] == 'accepted']
    if (not phase.get('source_identity_repair') or not accepted
            or (directory / 'summary.json').exists()):
        raise ValueError('Expected an interrupted source-identity run with accepted steps')
    manifest = Path(phase['source_identity_repair']['manifest'])
    if digest(manifest) != phase['source_identity_repair']['sha256']:
        raise ValueError('Saved source-repair manifest changed')
    dependencies = read_json(directory / 'code_dependencies.json')['sha256']
    checked = 0
    for relative, expected in dependencies.items():
        if relative.replace('\\', '/').startswith('.venv/'):
            continue
        if digest(ROOT / relative) != expected:
            raise ValueError('Saved graph implementation changed: ' + relative)
        checked += 1
    with np.load(directory / 'accepted_states.npz', allow_pickle=False) as z:
        states = {int(key): row.copy() for key, row in zip(z['keys'], z['states'])}
    if not states or any(v.shape != (11,) or not np.isfinite(v).all() for v in states.values()):
        raise ValueError('Saved accepted states are invalid')
    return phase, trace, states, dict(
        directory=str(directory), phase_sha256=digest(directory / 'phase_00.json'),
        state_sha256=digest(directory / 'accepted_states.npz'),
        trace_sha256=digest(directory / 'solver_trace.json'),
        accepted_steps=len(accepted), accepted_iteration=accepted[-1]['iteration'],
        expected_cost=float(accepted[-1]['cost']),
        next_lambda=float(trace['steps'][-1]['lambda_next']),
        unchanged_project_dependencies=checked)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--resume-source', type=Path, required=True)
    options, remaining = extra.parse_known_args()
    source = options.resume_source.resolve(strict=True)
    saved_phase, saved_trace, saved, provenance = load_saved(source)
    import fit_continuous_source_identity_repaired as entry
    from continuous_camera_solver import complete_graph_error, write_json
    original_solve = entry.solve

    def resume_solve(graph, initial, output, **kwargs):
        new_phase = read_json(output / 'phase_00.json')
        for field in ('camera_id', 'source_sha256', 'model_sha256', 'factor_counts',
                      'graph_factors_with_source_calibration', 'graph_nodes_with_references',
                      'frame_clock_virtual_pts', 'shot_knots'):
            if saved_phase[field] != new_phase[field]:
                raise ValueError('Reconstructed source graph differs: ' + field)
        if set(initial) != set(saved) or len(graph._pending) != saved_trace['factors']:
            raise ValueError('Saved state nodes or complete factor inventory differ')
        with np.load(source / 'prior_initializer_states.npz', allow_pickle=False) as a, \
                np.load(output / 'prior_initializer_states.npz', allow_pickle=False) as b:
            if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError('Original prior targets changed during recovery')
        restored = complete_graph_error(graph, saved,
            native_edge_manager=kwargs.get('native_edge_manager'),
            diagnostics_path=output / 'crash_resume_objective_diagnostics.json')
        expected = provenance['expected_cost']
        difference = abs(restored - expected)
        tolerance = max(1.e-7, 1.e-9 * abs(expected))
        report = dict(**provenance, restored_cost=restored,
            absolute_difference=difference, tolerance=tolerance,
            priors_bitwise_equal=True, state_role='accepted initialization, not prior',
            graph_reconstructed_from_original_warm=True,
            continued_with_saved_LM_damping=True,
            trace_iteration_numbering='local to each resume epoch; original steps retained',
            prior_trace_step_count=len(saved_trace['steps']))
        write_json(output / 'crash_resume.json', report)
        if not np.isfinite(restored) or difference > tolerance:
            raise ValueError('Saved state does not reproduce its complete accepted objective')
        if digest(source / 'accepted_states.npz') != provenance['state_sha256']:
            raise ValueError('Source state changed while reconstructing graph')
        shutil.copy2(source / 'accepted_states.npz', output / 'accepted_states.npz')
        continued_trace = dict(saved_trace)
        continued_trace['resume_epochs'] = [*saved_trace.get('resume_epochs', []), report]
        write_json(output / 'solver_trace.json', continued_trace)
        new_phase['crash_resume'] = dict(manifest=str(output / 'crash_resume.json'),
                                       sha256=digest(output / 'crash_resume.json'))
        write_json(output / 'phase_00.json', new_phase)
        print(json.dumps(dict(stage='crash_accepted_objective_restored', **report)), flush=True)
        return original_solve(graph, saved, output, resume=True, **kwargs)

    entry.solve = resume_solve
    sys.argv = [sys.argv[0], *remaining]
    try:
        entry.main()
    finally:
        entry.solve = original_solve


if __name__ == '__main__':
    main()
