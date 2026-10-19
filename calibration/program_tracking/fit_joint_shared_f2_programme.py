"""Continue complete main/low camera graphs with one common movable F2.

This entry keeps all camera/source priors and uses the source-pose clock main
model. A frozen main input makes graph reconstruction and interrupted-run
recovery deterministic. The output is always an unvalidated candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

from continuous_camera_solver import complete_graph_error, save_states, solve, write_json
from export_joint_f2_programme import export as export_programme, sha

TMP = Path('D:/CodexData/RinkSpace_program_tracking_tmp')


def packet(path):
    with np.load(path, allow_pickle=False) as z:
        result = {int(k): v.copy() for k, v in zip(z['keys'], z['states'])}
    if any(v.shape != (11,) or not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Expected finite Vector11 checkpoint')
    return result


def freeze_main(directory, output, state_name):
    """Read the atomic solver packet once, then freeze its immutable metadata."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    output.mkdir()
    names = ('phase_00.json', 'prior_initializer_states.npz', 'initial_cameras.npz',
             'preflight_initial_states.npz')
    payload = (directory/state_name).read_bytes()
    (output/'accepted_states.npz').write_bytes(payload)
    for name in names:
        shutil.copyfile(directory/name, output/name)
    phase = json.loads((output/'phase_00.json').read_text(encoding='utf8'))
    identity = phase['source_identity_repair']
    identity_path = directory/'source_identity_repair.json'
    if not identity_path.is_file():
        identity_path = Path(identity['manifest'])
    if sha(identity_path) != identity['sha256']:
        raise ValueError('Original source identity manifest differs from the main graph')
    shutil.copyfile(identity_path, output/'source_identity_repair.json')
    names = (*names, 'source_identity_repair.json')
    provenance = dict(source_directory=str(directory), original_state_name=state_name,
        frozen_state_sha256=sha(output/'accepted_states.npz'),
        frozen_metadata_sha256={name: sha(output/name) for name in names},
        role='immutable warm state only; original prior targets retained')
    write_json(output/'freeze_manifest.json', provenance)
    return provenance


def run(args, *, prepared_graph=None):
    from fit_continuous_rig_source_calibrated import snapshot_dependencies
    from prepare_joint_shared_f2_native import prepare_joint_f2_graph
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(TMP.resolve()):
        raise ValueError('Use a fresh output under programme TMP')
    output.mkdir(parents=True)
    started = time.perf_counter()
    if args.resume_joint:
        previous_phase = json.loads((args.resume_joint/'phase_00.json').read_text(encoding='utf8'))
        source = Path(previous_phase['frozen_main_directory'])
        frozen_info = freeze_main(source, output/'frozen_main_input', 'accepted_states.npz')
    else:
        frozen_info = freeze_main(args.main_directory, output/'frozen_main_input', args.main_state)
    if prepared_graph is None:
        prepared = prepare_joint_f2_graph(output/'frozen_main_input', args.low_directory,
                                          main_state_name='accepted_states.npz')
    else:
        prepared = prepared_graph
        if (args.resume_joint or prepared.graph is None
                or prepared.provenance['main_state_sha256'] != frozen_info['frozen_state_sha256']
                or prepared.provenance['low_states_sha256'] != sha(args.low_directory/'states.npz')):
            raise ValueError('Prepared gate graph does not match the frozen continuation inputs')
    with prepared:
        phase = dict(prepared.provenance, main_freeze=frozen_info,
            frozen_main_directory=str(output/'frozen_main_input'),
            low_directory=str(args.low_directory.resolve()),
            secondary_directory=str(args.secondary_directory.resolve()),
            secondary_camera_sha256=sha(args.secondary_directory/'shot_cameras.npz'),
            secondary_summary_sha256=sha(args.secondary_directory/'summary.json'),
            added_main_f2=prepared.added_main_f2,
            main_key_mapping=prepared.main.key_mapping,
            low_key_mapping=prepared.low.key_mapping,
            source_pixels_changed=False, original_camera_priors_changed=False,
            optical_centres_fixed=False, first_cameras_fixed=False,
            world_delta_key=2001, world_delta_units='scene units / 100',
            first_order_optimality_verified=False, accepted=False)
        phase['code_dependencies'] = snapshot_dependencies(output)
        write_json(output/'phase_00.json', phase)
        save_states(output/'prior_initializer_states.npz', prepared.prior)
        save_states(output/'preflight_initial_states.npz', prepared.states)
        initial_cost = complete_graph_error(prepared.graph, prepared.states,
            native_edge_manager=prepared.native_manager,
            diagnostics_path=output/'initial_objective_diagnostics.json')
        write_json(output/'initial_objective.json', dict(cost=initial_cost,
            includes_new_main_reference_observations=True,
            directly_comparable_to_separate_old_rig_objectives=False))
        initial, resume = prepared.states, False
        if args.resume_joint:
            previous_prior = packet(args.resume_joint/'prior_initializer_states.npz')
            initial = packet(args.resume_joint/'accepted_states.npz')
            if (set(initial) != set(prepared.states) or set(previous_prior) != set(prepared.prior)
                    or any(not np.array_equal(previous_prior[k], prepared.prior[k])
                           for k in prepared.prior)):
                raise ValueError('Resumed complete graph changed its original prior targets')
            for name in ('total_factor_count', 'total_node_count', 'candidate_scene_sha256',
                         'low_states_sha256', 'main_key_mapping', 'low_key_mapping',
                         'secondary_camera_sha256', 'secondary_summary_sha256'):
                if previous_phase[name] != json.loads(json.dumps(phase[name])):
                    raise ValueError('Joint recovery graph differs: '+name)
            trace = json.loads((args.resume_joint/'solver_trace.json').read_text(encoding='utf8'))
            steps = [r for r in trace['steps'] if r.get('status') == 'accepted']
            expected = steps[-1]['cost'] if steps else trace['initial_cost']
            restored = complete_graph_error(prepared.graph, initial,
                native_edge_manager=prepared.native_manager,
                diagnostics_path=output/'resume_objective_diagnostics.json')
            if abs(restored-expected) > max(1e-7, abs(expected)*1e-9):
                raise ValueError('Joint accepted objective was not exactly restored')
            resume_info = dict(source=str(args.resume_joint.resolve()), expected_cost=expected,
                restored_cost=restored, absolute_difference=abs(restored-expected),
                state_sha256=sha(args.resume_joint/'accepted_states.npz'))
            write_json(output/'resume.json', resume_info)
            trace['resume_epochs'] = [*trace.get('resume_epochs', []), resume_info]
            write_json(output/'solver_trace.json', trace)
            save_states(output/'accepted_states.npz', initial)
            resume = True
        print(json.dumps(dict(stage='complete_joint_shared_f2_graph_ready',
            factors=len(prepared.graph._pending), nodes=len(initial),
            initial_cost=initial_cost, resume=resume)), flush=True)
        if args.preflight_only:
            return
        try:
            with prepared.curve_preparation():
                solved, trace = solve(prepared.graph, initial, output,
                    max_iterations=args.iterations, native_edge_manager=prepared.native_manager,
                    resume=resume)
            result = export_programme(output/'programme_review', solved,
                [prepared.main, prepared.low], main_directory=output/'frozen_main_input',
                low_directory=args.low_directory, secondary_directory=args.secondary_directory,
                joint_state_path=output/'states.npz',
                extra_reference_models=(prepared.main.references, prepared.low.references,
                                         prepared.main.new_source_reference_model))
            write_json(output/'summary.json', dict(result, seconds=time.perf_counter()-started,
                final_cost=trace['final_cost'],
                shared_f2_delta_scene=(100*solved[2001][:3]).tolist(),
                whole_programme_source_accuracy_accepted=False))
            print(json.dumps(dict(stage='joint_export_complete', directory=str(output),
                                 final_cost=trace['final_cost'], accepted=False)), flush=True)
        finally:
            write_json(output/'native_runtime.json', prepared.native_manager.metadata())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-directory', type=Path, required=True)
    parser.add_argument('--main-state', choices=('states.npz', 'accepted_states.npz'), default='states.npz')
    parser.add_argument('--low-directory', type=Path, default=TMP/'continuous_camera_model_v11_time_principal/low_end')
    parser.add_argument('--secondary-directory', type=Path, default=TMP/'continuous_camera_model_v10_quadratic/secondary_long_side')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--resume-joint', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
