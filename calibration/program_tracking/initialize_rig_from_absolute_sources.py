"""Initialize an entire rig from existing absolute source observations.

This preparatory solve has no pairwise board intersections, allowing it to
leave a wrong near/far hit branch. Its result is only a starting estimate: the
complete original source objective and board visibility are restored by the
calling driver before any final camera stream can be produced.
"""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import numpy as np

from continuous_camera_solver import (complete_graph_error, solve, write_json,
                                      save_states)
from continuous_reference_calibration import attach_reference_factors
from historic_development_ads import attach_development_ad_factors
from secondary_nearboard_adaptive_curve import attach_secondary_nearboard_adaptive_factors


# The paused eight-step state is an initializer checkpoint, not a camera result.
# Pin all files that determine its original state, objective and damping.
_PAUSED_CHECKPOINT_SHA256 = {
    'accepted_states.npz': '6be34ac49b7d50bb1f30a2158768a059148607c4895f397a49922e129ac259d8',
    'solver_trace.json': 'a1a6c293b1f1031726fa541098231666b9e9f159975c08b533d0a00fc93123c3',
    'initializer_manifest.json': '4507837d65fbcc0427414624fe7d41ad3c5eac093674bfa00d11ad987fe25698',
    'prior_targets.npz': 'b18935e4a7acce16c73fbcc29a796d59d7efd5b32f38b231c443a0bda9466fbf',
    'starting_states.npz': '91db8ebc146528ea406261f61a6c194295fa2f8cc6868014abbc4d2b37e27ccd',
}
_PAUSED_COST = 8451.735306883058
_PAUSED_INITIAL_COST = 20708.407169622493
_PAUSED_NEXT_LAMBDA = 1e-5
_PAUSED_ACCEPTED_STEPS = 8


def _states_from_bytes(blob: bytes, name: str) -> dict[int, np.ndarray]:
    with np.load(BytesIO(blob), allow_pickle=False) as saved:
        if set(saved.files) != {'keys', 'states'}:
            raise ValueError(f'{name} has unexpected arrays')
        keys = np.asarray(saved['keys'], np.int64)
        values = np.asarray(saved['states'], np.float64)
    if (keys.ndim != 1 or values.shape != (len(keys), 11)
            or len(set(map(int, keys))) != len(keys)
            or not np.array_equal(keys, np.sort(keys))
            or not np.isfinite(values).all()):
        raise ValueError(f'{name} has invalid state keys or vectors')
    return {int(key): value.copy() for key, value in zip(keys, values)}


def _paused_checkpoint(source: Path, prior: dict, initial: dict,
                       metadata: dict) -> dict:
    """Read the one identified paused state and prove objective identities."""
    blobs = {name: (source/name).read_bytes()
             for name in _PAUSED_CHECKPOINT_SHA256}
    hashes = {name: hashlib.sha256(blob).hexdigest()
              for name, blob in blobs.items()}
    if hashes != _PAUSED_CHECKPOINT_SHA256:
        raise ValueError('Paused absolute initializer checkpoint hash changed')
    manifest = json.loads(blobs['initializer_manifest.json'].decode('utf8'))
    if manifest != metadata:
        raise ValueError('Rebuilt absolute source constraints or factor counts changed')
    trace = json.loads(blobs['solver_trace.json'].decode('utf8'))
    steps = trace.get('steps', [])
    if (trace.get('factors') != metadata['factor_count']
            or trace.get('states') != len(initial)
            or trace.get('initial_cost') != _PAUSED_INITIAL_COST
            or len(steps) != _PAUSED_ACCEPTED_STEPS
            or any(step.get('status') != 'accepted' or step.get('iteration') != i
                   for i, step in enumerate(steps))
            or steps[-1].get('cost') != _PAUSED_COST
            or steps[-1].get('lambda_next') != _PAUSED_NEXT_LAMBDA):
        raise ValueError('Paused absolute initializer trace is not the eight-step stop')
    accepted = _states_from_bytes(blobs['accepted_states.npz'], 'accepted_states')
    saved_prior = _states_from_bytes(blobs['prior_targets.npz'], 'prior_targets')
    starting = _states_from_bytes(blobs['starting_states.npz'], 'starting_states')
    if (set(accepted) != set(initial) or set(starting) != set(initial)
            or set(saved_prior) != set(prior)
            or set(accepted) != set(saved_prior)
            or any(not np.array_equal(saved_prior[key], prior[key])
                   for key in saved_prior)):
        raise ValueError('Paused initializer state keys or original prior targets changed')
    return dict(blobs=blobs, hashes=hashes, accepted=accepted,
                starting=starting, initial_cost=float(trace['initial_cost']),
                accepted_cost=float(steps[-1]['cost']),
                lambda_next=float(steps[-1]['lambda_next']),
                completed_steps=len(steps), source=str(source.resolve()))


def initialize_absolute_sources(data,model,prior,initial,clock,valid_pairs,
                                references,ads,nearboard_rows,output:Path,*,
                                iterations=80,native_edge_manager=None,
                                affine_priors=None,graph_owner=None,
                                resume_from: Path | None = None,
                                reference_curve_workers: int = 8):
    from fit_continuous_rig_camera import build_graph
    if data['camera_id']!='secondary_long_side' or not nearboard_rows:
        raise ValueError('This initializer requires the complete reviewed secondary group')
    if isinstance(reference_curve_workers, bool) or not isinstance(reference_curve_workers, int) or reference_curve_workers < 0:
        raise ValueError('reference_curve_workers must be a nonnegative integer')
    output = Path(output)
    if output.exists():
        raise ValueError('Absolute initializer output must be a fresh directory')
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        if output.resolve() == resume_from or output.resolve().is_relative_to(resume_from):
            raise ValueError('Resume output must be separate from the paused checkpoint')
    # Preserve the whole captured absolute families, including every source
    # edge and existing ad landmark. Cross-frame keyframe motion is absent
    # only in this initializer, along with the native pairwise motion list.
    def pairwise_keyframe(s):
        return (s['file']=='joint_tracking_refine.py' and s.get('loop')=='links'
                and len(s['keys'])==2 and set(s['keys'])<=data['pts'])
    temporarily_absent=[s for s in data['retained'] if pairwise_keyframe(s)]
    if len(temporarily_absent)!=32:
        raise ValueError('Reviewed secondary keyframe-pair inventory changed')
    absolute_data={**data,'retained':[s for s in data['retained'] if not pairwise_keyframe(s)]}
    retained_joint=[s for s in absolute_data['retained']
                    if s['file']=='joint_tracking_refine.py' and s.get('loop')=='curve_groups.items()']
    if len(retained_joint)!=3 or any(s['kind']!='observation' or len(s['keys'])!=1
                                    for s in retained_joint):
        raise ValueError('Original absolute B2/C/C curve identity changed')
    if sorted(tuple(map(int,s['keys'])) for s in retained_joint) != [
            (229109760,), (229109760,), (230468864,)]:
        raise ValueError('Original absolute B2/C/C source PTS changed')
    graph,counts=build_graph(absolute_data,[],model,prior,clock,valid_pairs,.25,
        native_edge_manager=native_edge_manager,affine_priors=affine_priors,graph_owner=graph_owner)
    metadata=dict(
        purpose='whole-rig starting estimate only; restore complete original objective afterwards',
        camera_id=data['camera_id'],programme_frames=len(model.frames),
        source_sha256=data['diag']['source_sha256'],
        model_sha256=data['diag']['model_sha256'],
        source_uv_changed=False,prior_targets_changed=False,
        first_camera_fixed=False,optical_center_fixed=False,
        final_source_factors_removed=False,
        initializer_objective='all original absolute source edges/ads/shared markers + references/nearboard + original mechanical priors',
        temporarily_absent_families='native board motion and joint_tracking_refine keyframe pairs only during initialization',
        captured_factor_counts=counts)
    if counts.get('joint_tracking_refine.py:solve')!=3:
        raise ValueError('The three original B2/C/C source curves were not retained')
    metadata['temporarily_absent_keyframe_pair_count']=len(temporarily_absent)
    metadata['references']=attach_reference_factors(graph,data,references,adaptive_curves=True)
    metadata['advertisement_factor_count']=attach_development_ad_factors(graph,data,model,prior,ads)
    metadata['nearboard']=attach_secondary_nearboard_adaptive_factors(
        graph,data,model,prior,nearboard_rows)
    metadata['factor_count']=len(graph._pending)
    graph._adaptive_curve_worker_count=reference_curve_workers
    checkpoint = None
    if resume_from is not None:
        checkpoint = _paused_checkpoint(resume_from,prior,initial,metadata)
        # A matching count and source manifest alone cannot prove callback
        # equivalence. Evaluate every rebuilt factor at both original states.
        rebuilt_initial_cost=complete_graph_error(graph,checkpoint['starting'],
            native_edge_manager=native_edge_manager)
        rebuilt_cost=complete_graph_error(graph,checkpoint['accepted'],
            native_edge_manager=native_edge_manager)
        if (not np.isfinite(rebuilt_initial_cost) or
                abs(rebuilt_initial_cost-checkpoint['initial_cost']) >
                1e-8*max(1.,abs(checkpoint['initial_cost'])) or
                not np.isfinite(rebuilt_cost) or
                abs(rebuilt_cost-checkpoint['accepted_cost']) >
                1e-8*max(1.,abs(checkpoint['accepted_cost']))):
            raise ValueError('Rebuilt absolute source objective differs at paused state')
        checkpoint['rebuilt_initial_cost']=float(rebuilt_initial_cost)
        checkpoint['rebuilt_cost']=float(rebuilt_cost)
    output.mkdir(parents=True,exist_ok=False)
    if checkpoint is None:
        save_states(output/'prior_targets.npz',prior)
        save_states(output/'starting_states.npz',initial)
    else:
        for name in ('prior_targets.npz','starting_states.npz',
                     'accepted_states.npz','solver_trace.json'):
            (output/name).write_bytes(checkpoint['blobs'][name])
        resume_record=dict(resume_run=1,source=checkpoint['source'],
            source_file_sha256=checkpoint['hashes'],
            original_initial_cost=checkpoint['initial_cost'],
            rebuilt_original_initial_cost=checkpoint['rebuilt_initial_cost'],
            accepted_cost=checkpoint['accepted_cost'],
            rebuilt_complete_graph_cost=checkpoint['rebuilt_cost'],
            accepted_steps_before_resume=checkpoint['completed_steps'],
            lambda_initial_for_resume=checkpoint['lambda_next'],
            iteration_labels_restart_per_run=True,
            no_source_observation_or_prior_target_changed=True)
        metadata['resume_run']=resume_record
        write_json(output/'resume_provenance.json',resume_record)
    write_json(output/'initializer_manifest.json',metadata)
    copied={k:v.copy() for k,v in prior.items()}
    tick=time.perf_counter()
    solve_initial=initial if checkpoint is None else checkpoint['accepted']
    initialized,trace=solve(graph,solve_initial,output,max_iterations=iterations,
        resume=checkpoint is not None,
        native_edge_manager=native_edge_manager)
    if any(not np.array_equal(prior[k],v) for k,v in copied.items()):
        raise RuntimeError('Absolute source initialization changed prior targets')
    metadata.update(seconds=time.perf_counter()-tick,
        initial_cost=trace['initial_cost'],final_cost=trace['final_cost'],
        accepted_as_accurate=False,complete_board_geometry_checked=False)
    write_json(output/'initializer_summary.json',metadata)
    return initialized,metadata
