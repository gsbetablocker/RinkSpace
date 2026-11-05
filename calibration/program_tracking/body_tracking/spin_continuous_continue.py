"""Continue one absolute S05 state from frozen prepared inputs, without inference.

The warm-start x is expressed about frozen initial seed controls.  Its saved
final absolute curve is an identity check, never a replacement seed to which
x is applied again.  Existing A/B parameters and the frozen run stay read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from rinkspace.path_policy import guard_project_write
from .absolute_motion_state import (CURVE_SCHEMAS, absolute_reference,
    curve_from_packet, evaluate_curve, with_warm_rotation_chart)
from .body_execution import BodyExecution, BodyExecutionConfig
from .contact_feasibility import ContactAL
from .exmr_integration import PROGRAMME, read_case_identity, validate_case_identity, save_packet
from .fullbody_round import write_json
from .rebuild_prepare import digest, load
from .skate_checkpoint import validate_checkpoint_identity
from .skate_run_policy import RunPolicy
from .spin_continuous_refit import prepare, make_problem, export_physics, serial
from .stage_quality import StageQualityTrace, body_groups
from .support_refinement import measure
from .support_root_placement import restore_parent


COMMON = ('input.npz', 'observations.npz', 'compact_smplx.npz',
          'export_source.npz', 'initial.npz')
RESOURCES = ('exmr_body_hand_nofoot.npz', 'palm_grip_geometry.json',
             'grasp_relations.json')
# RunPolicy currently treats zero as an immediate accepted-budget stop.  This
# compatibility sentinel is deliberately unreachable in the wall budget and
# is not a desired step count, convergence criterion, or project-wide policy.
ACCEPTED_SENTINEL = 10**9


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        method = 'hardlink'
    except OSError:
        shutil.copy2(source, target)
        method = 'copy'
    return dict(source=str(source), destination=str(target), sha256=digest(source),
                storage=method)


def _atomic_npz(path, packet, project, *, new=False):
    path = guard_project_write(path, project_root=project)
    pending = guard_project_write(path.with_name(path.stem+'.pending.npz'),
                                  project_root=project)
    if new and path.exists():
        raise FileExistsError('Existing parameter is preserved: '+str(path))
    np.savez_compressed(pending, **packet)
    if new and path.exists():
        raise FileExistsError('Parameter appeared while saving: '+str(path))
    pending.replace(path)


def _same_array(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _freeze_inputs(frozen, output, evidence, observations):
    """Copy/link only current prepared artifacts; retain both input revisions."""
    old_common = frozen/'common/exmr_hand_nofoot'
    required = [old_common/name for name in COMMON]
    required += [frozen/name for name in RESOURCES]
    required += [frozen/'source_evidence.json', frozen/'initialization.json',
                 frozen/'source_stage/cross_stage_quality.json']
    for source in required + [p for p in (evidence, observations) if p is not None]:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True)
    history = output/'frozen_inputs'
    history.mkdir()
    manifest = []
    for name in COMMON:
        source = old_common/name
        if name == 'observations.npz' and observations is not None:
            manifest.append(_link_or_copy(source, history/name))
            source = observations
        manifest.append(_link_or_copy(source, output/'common/exmr_hand_nofoot'/name))
    for name in RESOURCES + ('initialization.json',):
        manifest.append(_link_or_copy(frozen/name, output/name))
    manifest.append(_link_or_copy(frozen/'source_evidence.json', history/'source_evidence.json'))
    manifest.append(_link_or_copy(evidence or frozen/'source_evidence.json', output/'source_evidence.json'))
    manifest.append(_link_or_copy(frozen/'source_stage/cross_stage_quality.json',
                                  history/'cross_stage_quality.json'))
    document = _json(output/'source_evidence.json')
    old_document = _json(frozen/'source_evidence.json')
    for key in ('schema', 'source_id', 'source_sha256', 'shot_id', 'time_base', 'pixel_domain'):
        if document.get(key) != old_document.get(key):
            raise ValueError('Evidence changed source/coordinate identity: '+key)
    if document.get('shot_id') != 's05' or document.get('time_base') != [1, 12800]:
        raise ValueError('Expected native S05 evidence clock')
    current = load(output/'common/exmr_hand_nofoot/observations.npz')
    old = load(old_common/'observations.npz')
    np.testing.assert_array_equal(current['pts'], old['pts'])
    changed_keys = [key for key in sorted(set(current) | set(old))
                    if key not in current or key not in old or not _same_array(current[key], old[key])]
    revision = dict(schema='rinkspace.continuous-input-revision.v1', frozen_run=str(frozen),
        observations=dict(original=str(old_common/'observations.npz'), original_sha256=digest(old_common/'observations.npz'),
            current=str(output/'common/exmr_hand_nofoot/observations.npz'), current_sha256=digest(output/'common/exmr_hand_nofoot/observations.npz'),
            changed_arrays=changed_keys, old_records_preserved=True),
        evidence=dict(original=str(frozen/'source_evidence.json'), original_sha256=digest(frozen/'source_evidence.json'),
            current=str(output/'source_evidence.json'), current_sha256=digest(output/'source_evidence.json'), old_records_preserved=True),
        files=manifest, historical_source_production_replayed=False, network_inference=False)
    write_json(output/'input_revision.json', revision)
    return revision


def _warm_start(problem, saved, frozen_initial):
    """Restore coordinates and AL state in exactly the original seed basis."""
    if str(saved.get('state_schema_version', '')) not in CURVE_SCHEMAS or str(saved.get('state_kind', '')) != 'absolute_motion_v1':
        raise ValueError('Warm start must be an explicit absolute-motion checkpoint')
    validate_checkpoint_identity(saved, problem.reference.fixed_checkpoint_identity)
    expected = read_case_identity(problem.reference.fixed_checkpoint_identity)
    original = read_case_identity(saved)
    # Evidence/ledger revisions are allowed. Every other input and camera field
    # must still identify the same immutable prepared problem.
    for key in set(expected) | set(original):
        if key not in ('observations_sha256', 'support_document_sha256') and expected.get(key) != original.get(key):
            raise ValueError('Warm-start prepared identity changed: '+key)
    state = problem.body.state
    for key in ('absolute_knots', 'absolute_seed_position_controls', 'absolute_seed_rotation_controls'):
        np.testing.assert_array_equal(saved[key], frozen_initial[key], err_msg='Warm-start seed basis changed: '+key)
    np.testing.assert_array_equal(state.knots, saved['absolute_knots'])
    np.testing.assert_array_equal(state.active, np.arange(69))
    np.testing.assert_array_equal(saved['active'], state.active)
    np.testing.assert_array_equal(saved['knots'], state.knots)
    np.testing.assert_allclose(saved['seconds'], state.seconds, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(saved['physics_times'], problem.physics_times)
    x = np.asarray(saved['x'], dtype=float).copy()
    if x.shape != (problem.size,) or not np.isfinite(x).all():
        raise ValueError('Invalid warm-start control vector')
    np.testing.assert_array_equal(x[:17], problem.binding.x)
    actual = state.current_curve(x[17:])
    errors = {}
    for key, value in (('absolute_position_controls', actual.position_controls),
                       ('absolute_rotation_controls', actual.rotation_controls)):
        np.testing.assert_allclose(value, saved[key], rtol=0, atol=2e-12,
            err_msg='Frozen seed + warm x does not reconstruct the saved final absolute curve')
        errors[key] = float(np.max(np.abs(value-saved[key]), initial=0.))
    old_values = evaluate_curve(curve_from_packet(saved), problem.body.times)
    actual_values = evaluate_curve(actual, problem.body.times)
    for key in ('P', 'Q', 'local_R'):
        np.testing.assert_allclose(actual_values[key], old_values[key], rtol=0, atol=2e-10,
            err_msg='Warm start changed the saved curve at a body exposure node: '+key)
        errors['exposure_'+key] = float(np.max(np.abs(actual_values[key]-old_values[key]), initial=0.))
    al = ContactAL(saved['multiplier'], float(saved['penalty']))
    if al.multiplier.shape != problem.al.multiplier.shape:
        raise ValueError('Warm-start AL row layout changed')
    problem.al = al
    problem.cache = problem.cache_x = None
    return x, errors


def _measure_all(problem, review, x, value):
    result = measure(problem, review, x, value)
    body = problem.body.evaluate(x[17:], False)
    frames = problem.body.frames
    result['whole_source_groups'] = body_groups(body['uv'][:, problem.body.layout.body_landmarks],
        problem.body.obs['uv'][frames], problem.body.obs['use'][frames], problem.body.obs['pts'][frames])
    result['whole_source_frames'] = int(problem.body.n)
    result['joint_cost'] = float(value['residual']@value['residual']/2)
    return result


def _check_new_support_geometry(problem,x,path):
    factor=problem.support_image
    selected=[i for i,(row,o,_) in enumerate(factor.rows)
              if o.kind=='support_contact_region' and row.get('arc_selection')=='current_geometric_lowest_runner']
    if not selected:return
    rows=(np.asarray(selected)[:,None]*2+np.arange(2)).ravel()
    value=factor.evaluate(x[17:],True);J=value['jac'][rows]
    from .absolute_motion_curve import greville_times
    state=problem.body.state;times=greville_times(state.knots)+76.
    rng=np.random.default_rng(110505);direction=np.zeros_like(x)
    control=direction[17:].reshape(state.m,69);mask=(times>=87.5)&(times<=92.5)
    axes=np.r_[np.arange(6),np.arange(9,12),np.arange(18,21),np.arange(27,30),np.arange(36,39)]
    control[np.ix_(mask,axes)]=rng.normal(size=(mask.sum(),len(axes)));direction/=np.linalg.norm(direction)
    h=1e-6
    plus=factor.evaluate((x+h*direction)[17:],False)['residual'][rows]
    minus=factor.evaluate((x-h*direction)[17:],False)['residual'][rows]
    numeric=(plus-minus)/(2*h);exact=J@direction
    error=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12))
    report=dict(new_source_support_regions=len(selected),specific_new_geometry_check=True,
                relative_directional_error=error,max_absolute_error=float(np.max(np.abs(exact-numeric))),
                differentiability_warnings=value['metrics']['differentiability_warnings'])
    write_json(path,report)
    if error>3e-3:raise ValueError('New support-region geometry Jacobian mismatch: '+str(error))


def _check_entry_curvature(problem,x,path):
    factor=problem.action_factors.motion
    value=factor.evaluate(x[17:],True);count=len(factor.times)*3
    J=value['jac'][-count:];rng=np.random.default_rng(110506)
    from .absolute_motion_curve import greville_times
    times=greville_times(problem.body.state.knots)+76.
    selected=(times>=80.8)&(times<=82.4)
    direction=np.zeros_like(x);control=direction[17:].reshape(problem.body.state.m,69)
    control[selected,:2]=rng.normal(size=(selected.sum(),2));direction/=np.linalg.norm(direction)
    h=1e-6
    plus=factor.evaluate((x+h*direction)[17:],False)['residual'][-count:]
    minus=factor.evaluate((x-h*direction)[17:],False)['residual'][-count:]
    numeric=(plus-minus)/(2*h);exact=J@direction
    error=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12))
    write_json(path,dict(new_horizontal_entry_curvature=True,relative_directional_error=error,
        max_absolute_error=float(np.max(abs(exact-numeric))),full_source_frames=len(factor.times),
        unchanged_finite_scale=12.,not_a_COM_or_contact_measurement=True))
    if error>3e-3:raise ValueError('Entry curvature Jacobian mismatch: '+str(error))


def _check_lifted_native_chain(problem,x,value,path):
    if problem.body.state.curve.edge_reference_quaternion is None:return
    body_cache=problem.body.cache
    from .absolute_motion_curve import greville_times
    times=greville_times(problem.body.state.knots)+76.;selected=(times>=93.2)&(times<=93.8)
    rng=np.random.default_rng(110508);direction=np.zeros_like(x)
    controls=direction[17:].reshape(problem.body.state.m,69);axes=np.r_[np.arange(6),np.arange(48,51)]
    controls[np.ix_(selected,axes)]=rng.normal(size=(selected.sum(),len(axes)));direction/=np.linalg.norm(direction)
    h=1e-6;plus=problem.evaluate(x+h*direction,False)['residual'];minus=problem.evaluate(x-h*direction,False)['residual']
    numeric=(plus-minus)/(2*h);exact=value['jac']@direction
    relative=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12))
    write_json(path,dict(full_native_residual_rows=len(numeric),relative_directional_error=relative,
        max_absolute_error=float(np.max(abs(exact-numeric))),root_and_head_controls=True,
        source_control_seconds=[93.2,93.8],first_full_J_reused=True))
    problem.cache_x=x.copy();problem.cache=value;problem.body.cache_x=x[17:].copy();problem.body.cache=body_cache
    if relative>3e-3:raise ValueError('Lifted full native chain mismatch: '+str(relative))


def _configure_local_controls(problem,seconds):
    if seconds is None:return None
    seconds=np.asarray(seconds,float)
    if seconds.shape!=(2,) or not np.isfinite(seconds).all() or seconds[0]>=seconds[1]:
        raise ValueError('Ordered finite local control seconds required')
    state=problem.body.state;origin=problem.reference.p['pts'][0]/12800.
    knots=state.knots;left=knots[:state.m]+origin;right=knots[4:]+origin
    selected=(left>=seconds[0]-1e-10)&(right<=seconds[1]+1e-10)
    controls=np.flatnonzero(selected)
    if not len(controls):raise ValueError('Local interval contains no complete cubic control support')
    indices=(17+controls[:,None]*69+np.arange(69)).ravel()
    problem.active_parameter_indices=indices
    problem.local_control_seconds=seconds.tolist()
    problem.checkpoint_extra.update(local_control_seconds=seconds,
        local_active_parameter_indices=indices,
        local_continuous_support_seconds=np.array([left[selected].min(),right[selected].max()]))
    return dict(requested_seconds=seconds.tolist(),effective_support_seconds=[float(left[selected].min()),float(right[selected].max())],
        control_indices=controls.tolist(),active_variables=len(indices),full_body_variables=state.size,
        one_global_curve=True,no_independent_clip_or_display_smoothing=True)


def _local_packet(packet,problem,local):
    if local is not None:
        seconds=packet['source_pts']/12800.;a,b=local['effective_support_seconds'];halo=float(problem.body.tau)/2
        packet['window_mask']=(seconds>=a-halo)&(seconds<=b+halo)
        packet['body_refinement_window_mask']=packet['window_mask'].copy()
        packet['local_control_seconds']=np.asarray(local['requested_seconds'])
    return packet


def _linear_backends(problem):
    statistics = getattr(problem, 'linear_solver_statistics', [])
    return dict(requested=problem.linear_solver_backend,
        actual=sorted({'lsmr' if row.get('fallback') else row.get('backend', 'unknown') for row in statistics}),
        calls=len(statistics), fallback_calls=sum(bool(row.get('fallback')) for row in statistics),
        statistics=statistics)


def run(project, base_root, frozen_run, warm_start, output, parameter_stem,
        wall_seconds=7200., *, evidence=None, observations=None, flight=False,
        lift_rotation_chart=False,local_control_seconds=None):
    started = time.perf_counter()
    project, base, frozen, warm, output = [Path(p).resolve() for p in
        (project, base_root, frozen_run, warm_start, output)]
    evidence = Path(evidence).resolve() if evidence is not None else None
    observations = Path(observations).resolve() if observations is not None else None
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('wall_seconds must be finite and positive')
    if (not parameter_stem or parameter_stem in ('.', '..') or
            any(c in parameter_stem for c in '<>:"/\\|?*') or parameter_stem.endswith(('.', ' '))):
        raise ValueError('parameter_stem must be one Windows filename stem')
    parameter = guard_project_write(project/PROGRAMME/(parameter_stem+'.npz'), project_root=project)
    guard_project_write(parameter.with_name(parameter.stem+'.pending.npz'), project_root=project)
    if output.exists() or parameter.exists():
        raise FileExistsError('Use a new output and parameter stem; existing runs are preserved')
    if (output.is_relative_to(project) or output.is_relative_to(frozen) or
            frozen.is_relative_to(output) or base.is_relative_to(output)):
        raise ValueError('Independent run output must be outside the project and frozen/base inputs')
    main_log = output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(main_log), output=str(output), warm_start=str(warm),
        wall_budget_seconds=wall_seconds, source_context_seconds=[76, 95], expected_source_frames=951,
        accepted_step_limit=ACCEPTED_SENTINEL, accepted_limit_is_stop_target=False)), flush=True)
    saved = load(warm)
    validate_checkpoint_identity(saved)
    frozen_meta = _json(frozen/'initialization.json')
    native = base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll'
    original_camera = json.loads(str(saved['camera_binding_json']))
    if original_camera['state_sha256'] != frozen_meta['camera']['state_sha256']:
        raise ValueError('Warm start does not identify the frozen gravity camera')
    if digest(native) != original_camera['native_library_sha256']:
        raise ValueError('Frozen body/camera native library SHA256 changed')
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = str(native)
    revision = _freeze_inputs(frozen, output, evidence, observations)
    execution = BodyExecution(BodyExecutionConfig(workers=12, blas_threads=1,
        local_derivatives='analytic', compatibility_backend='cuda_resident'))
    with main_log.open('w', encoding='utf-8', buffering=1) as log:
        def emit(event, **values):
            log.write(json.dumps(serial(dict(event=event, wall_seconds=time.perf_counter()-started, **values)), ensure_ascii=False)+'\n')
        emit('continuation_setup', warm_start=str(warm), warm_start_sha256=digest(warm),
             input_revision=str(output/'input_revision.json'), full_wall_budget_seconds=wall_seconds)
        try:
            with execution.limits():
                # initialization.json is already present, so prepare consumes
                # the frozen common packet; it never calls source_document or
                # seed_curve, and restores the existing gravity parent once.
                parent, camera, info, curve, document, archive = prepare(project, base, output)
                if info['state_sha256'] != frozen_meta['camera']['state_sha256']:
                    raise ValueError('Frozen gravity camera parameter SHA256 changed')
                if info['native_library_sha256'] != frozen_meta['camera']['native_library_sha256']:
                    raise ValueError('Frozen gravity native library SHA256 changed')
                initial = load(output/'common/exmr_hand_nofoot/initial.npz')
                if digest(archive) != digest(output/'common/exmr_hand_nofoot/initial.npz'):
                    raise ValueError('Archived initial differs from frozen absolute seed packet')
                curve = with_warm_rotation_chart(curve, saved, lift=lift_rotation_chart)
                problem, review, supports = make_problem(project, base, output, 'continuous_stage',
                    parent, camera, info, curve, document, execution, flight, log)
                dest = output/'continuous_stage'
                x, errors = _warm_start(problem, saved, initial)
                local_selection=_configure_local_controls(problem,local_control_seconds)
                if local_selection is not None:write_json(dest/'local_parameter_selection.json',local_selection)
                pts = problem.reference.p['pts']
                np.testing.assert_array_equal(pts, np.arange(76*12800, 95*12800+1, 256, dtype=np.int64))
                np.testing.assert_array_equal(problem.body.frames, np.arange(951))
                if problem.optimize_shared or problem.body.size != len(curve.position_controls)*69:
                    raise ValueError('Expected all absolute controls active and shared shoes frozen')
                problem.linear_solver_backend = 'sparse_direct'
                problem.checkpoint_extra.update(continuation_warm_start=np.array(str(warm)),
                    continuation_warm_start_sha256=np.array(digest(warm)),
                    continuation_frozen_run=np.array(str(frozen)))
                baseline_file = frozen/'source_stage/cross_stage_quality.json'
                original_baseline = _json(baseline_file)['original_A']
                identity = dict(warm_start=str(warm), warm_start_sha256=digest(warm),
                    original_baseline_path=str(baseline_file), original_baseline_sha256=digest(baseline_file),
                    observations_sha256=digest(problem.body.ledger_path),
                    source_document_sha256=digest(output/'source_evidence.json'),
                    evidence_changed=revision['evidence']['original_sha256'] != revision['evidence']['current_sha256'],
                    observations_changed=bool(revision['observations']['changed_arrays']),
                    original_A_retained_without_reset=True, full_source_seconds=[76, 95])
                problem.stage_quality = StageQualityTrace(dest/'cross_stage_quality.json', identity,
                    original_baseline, lambda xx, value: _measure_all(problem, review, xx, value))
                if local_selection is None:
                    _check_new_support_geometry(problem,x,dest/'new_support_geometry_check.json')
                    _check_entry_curvature(problem,x,dest/'new_entry_curvature_check.json')
                tick = time.perf_counter()
                value = problem.evaluate(x, True)
                jsec = time.perf_counter()-tick
                if not np.isfinite(value['residual']).all() or not np.isfinite(value['jac'].data).all():
                    raise ValueError('Nonfinite first residual/Jacobian')
                if local_selection is None:
                    _check_lifted_native_chain(problem,x,value,dest/'lifted_native_chain_check.json')
                start_quality = _measure_all(problem, review, x, value)
                # measure() uses the same x and preserves the first complete J.
                if problem.cache is not value or not np.array_equal(problem.cache_x, x):
                    raise RuntimeError('Same-state quality measurement invalidated first Jacobian')
                problem.jacobian_seconds_hint = jsec
                write_json(dest/'continuation_start_quality.json', dict(identity=identity,
                    frozen_seed_reconstruction_max_errors=errors, metrics=start_quality,
                    original_A=original_baseline, stage_reset_is_progress=False))
                save_reserve = max(30., 2*jsec)
                setup_seconds = time.perf_counter()-started
                remaining = max(0., wall_seconds-setup_seconds-save_reserve)
                policy = RunPolicy(wall_seconds=remaining, max_accepted=ACCEPTED_SENTINEL,
                    max_outer=1, stagnation_window=12, allow_penalty_growth=False)
                write_json(dest/'execution.json', dict(first_J_seconds=jsec, variables=problem.body.size,
                    controls=problem.body.state.m, actual_source_frames=problem.body.n,
                    execution=execution.record(), inner_linear_backend=problem.linear_solver_backend,
                    budget_seconds=wall_seconds, setup_seconds=setup_seconds,
                    save_reserve_seconds=save_reserve, solve_budget_seconds=remaining,
                    accepted_limit_is_stop_target=False, legacy_direction_checks_repeated=False,
                    first_J_reused=True, same_absolute_seed=True, shared_shape_and_camera_frozen=True,
                    local_parameter_selection=local_selection))
                emit('continuation_solve', first_J_seconds=jsec, setup_seconds=setup_seconds,
                     save_reserve_seconds=save_reserve, solve_budget_seconds=remaining,
                     initial_joint_cost=start_quality['joint_cost'], policy=vars(policy))
                result, report = problem.solve(x, policy)
                candidate = load(problem.output/'candidate.npz')
                np.testing.assert_array_equal(candidate['x'], result)
                np.testing.assert_array_equal(result[:17], saved['x'][:17])
                if local_selection is not None:
                    frozen_columns=np.ones(problem.size,bool);frozen_columns[problem.active_parameter_indices]=False
                    np.testing.assert_array_equal(result[frozen_columns],saved['x'][frozen_columns])
                backends = _linear_backends(problem)
                metadata = dict(stop=report['stop'], accepted_steps=int(problem.accepted),
                    retained_step=report['selection']['retained_step'],
                    wall_seconds_before_export=time.perf_counter()-started,
                    wall_budget_seconds=wall_seconds, linear_backends=backends)
                candidate['continuation_run_json'] = np.array(json.dumps(serial(metadata), sort_keys=True))
                _atomic_npz(parameter, candidate, project, new=True)
                final = absolute_reference(problem.reference,
                    problem.body.state.current_curve(result[17:]), root=problem.reference.root)
                shell = copy.copy(problem)
                shell.reference = final
                packet = save_packet(shell, info, supports, document, dest/'common/exmr_hand_nofoot',
                                     status='joint_fit', report=report)
                packet['absolute_parameter_path'] = np.array(str(parameter))
                packet['continuation_run_json'] = candidate['continuation_run_json']
                packet = _local_packet(export_physics(problem, final, dest, packet),problem,local_selection)
                result_path = dest/'common/exmr_hand_nofoot/fitted/result.npz'
                _atomic_npz(result_path, packet, project)
                summary = dict(metadata, parameter=str(parameter), result=str(result_path),
                    wall_seconds=time.perf_counter()-started, budget_seconds=wall_seconds,
                    source_frames=len(packet['pts']), source_context_seconds=[76, 95], schema=str(packet['state_schema_version']),
                    warm_start=str(warm), warm_start_sha256=digest(warm),
                    original_baseline_sha256=digest(baseline_file),
                    same_frozen_observations_sha256=digest(problem.body.ledger_path),
                    source_evidence_sha256=digest(output/'source_evidence.json'),
                    finite_flight=flight, inner_linear_backend=problem.linear_solver_backend,
                    linear_solver_statistics=backends['statistics'], whole_shot_adjustable=local_selection is None,
                    local_control_seconds=None if local_selection is None else local_selection['requested_seconds'],
                    local_parameter_selection=local_selection,
                    shared_shape_and_camera_frozen=True, accuracy_accepted=False, numerical_convergence=False,
                    initial_joint_cost=start_quality['joint_cost'], final_quality=problem.stage_quality.record())
                write_json(dest/'run_summary.json', summary)
                write_json(output/'run_summary.json', summary)
                emit('continuation_complete', parameter=str(parameter), result=str(result_path),
                     accepted=problem.accepted, retained_step=metadata['retained_step'],
                     stop=report['stop'], linear_backends=backends['actual'])
                print(json.dumps(dict(parameter=str(parameter), result=str(result_path),
                    accepted=problem.accepted, retained_step=metadata['retained_step'],
                    wall_seconds=summary['wall_seconds'])), flush=True)
                return summary
        except BaseException as exc:
            emit('continuation_failed', exception=type(exc).__name__, message=str(exc))
            write_json(output/'attempt_summary.json', dict(status='attempt_only', error=str(exc),
                wall_seconds=time.perf_counter()-started, warm_start=str(warm),
                existing_inputs_preserved=True, accuracy_accepted=False))
            raise
        finally:
            execution.close()


def finalize_existing(project, base_root, output, parameter):
    """Export a completed saved solve only; preserve its parameter and history.

    Historical counters, stop and linear solves come from the original files.
    This action restores the same frozen basis/AL and evaluates one residual;
    it never calls solve(), creates a new optimization step, or rewrites inputs.
    """
    started = time.perf_counter()
    project, base, output, parameter = [Path(p).resolve() for p in
        (project, base_root, output, parameter)]
    dest = output/'continuous_stage'
    case = dest/'common/exmr_hand_nofoot'
    candidate_path = dest/'solve/candidate.npz'
    report_path = dest/'solve/candidate.json'
    result_path = case/'fitted/result.npz'
    staging = dest/'finalize_staging'
    summary_paths = (dest/'run_summary.json', output/'run_summary.json')
    final_paths = (result_path, dest/'centroidal_physics.npz', dest/'physics_diagnostics.json',
                   *summary_paths, output/'finalize_integrity.json')
    for path in (*final_paths, staging, output/'finalize_progress.jsonl',
                 output/'finalize_failure.json', output/'finalize_geometry'):
        guard_project_write(path, project_root=project)
    for path in (*final_paths, staging, output/'finalize_progress.jsonl', output/'finalize_geometry'):
        if path.exists():raise FileExistsError('Existing recovery/final artifact is preserved: '+str(path))
    for path in (parameter, candidate_path, report_path, output/'input_revision.json',
                 case/'initial.npz', dest/'cross_stage_quality.json'):
        if not path.is_file():raise FileNotFoundError(path)
    before_parameter_sha = digest(parameter)
    saved = load(parameter)
    candidate = load(candidate_path)
    for key, value in candidate.items():
        if key not in saved or not _same_array(value, saved[key]):
            raise ValueError('Saved parameter and completed candidate differ: '+key)
    metadata = json.loads(str(saved['continuation_run_json']))
    report = _json(report_path)
    accepted = metadata['accepted_steps']
    retained = metadata['retained_step']
    if (type(accepted) is not int or type(retained) is not int or
            accepted < 0 or not 0 <= retained <= accepted or
            retained != report['selection']['retained_step'] or
            accepted != report['run_policy']['accepted'] or metadata['stop'] != report['stop']):
        raise ValueError('Historical accepted/retained/stop records conflict')
    if not report.get('candidate_checkpoint') or Path(report['candidate_checkpoint']).resolve()!=candidate_path:
        raise ValueError('Historical final report names a different candidate')
    if Path(str(saved['absolute_case_root'])).resolve()!=dest:
        raise ValueError('Saved parameter belongs to a different prepared case')
    revision = _json(output/'input_revision.json')
    for record in revision['files']:
        if digest(Path(record['destination']))!=record['sha256']:
            raise ValueError('Frozen current input changed: '+record['destination'])
    camera_record = json.loads(str(saved['camera_binding_json']))
    native = base.parent/'body_perf_1103/native_cpu/rinkspace_core.dll'
    if digest(native)!=camera_record['native_library_sha256']:
        raise ValueError('Pinned native DLL differs from the completed solve')
    os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = str(native)
    log_path = output/'finalize_progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path), action='finalize_existing',
        parameter=str(parameter), completed_candidate=str(candidate_path), no_fit=True)), flush=True)
    execution = BodyExecution(BodyExecutionConfig(workers=12, blas_threads=1,
        local_derivatives='analytic', compatibility_backend='cuda_resident'))
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        def emit(event, **values):
            record = dict(event=event, export_recovery_seconds=time.perf_counter()-started,
                optimization_progress=False)
            record.update(values)
            log.write(json.dumps(serial(record), ensure_ascii=False)+'\n')
        emit('finalize_existing_begin', accepted_steps=accepted, retained_step=retained,
            parameter_sha256=before_parameter_sha, historical_stop=report['stop'])
        try:
            with execution.limits():
                parent, camera, info, _ = restore_parent(project, base,
                    Path(str(saved['absolute_parent_case_root'])), output/'finalize_geometry',
                    project/PROGRAMME/'spin_entry_gravity_1104.npz')
                initial = load(output/'common/exmr_hand_nofoot/initial.npz')
                curve = with_warm_rotation_chart(curve_from_packet(initial, seed=True), saved)
                document = _json(output/'source_evidence.json')
                flight = bool(saved['finite_flight_enabled'])
                problem, _, supports = make_problem(project, base, output, 'continuous_stage',
                    parent, camera, info, curve, document, execution, flight, log, existing=True)
                validate_case_identity(saved, read_case_identity(problem.reference.fixed_checkpoint_identity))
                x, reconstruction_errors = _warm_start(problem, saved, initial)
                recovered_local=_configure_local_controls(problem,saved.get('local_control_seconds'))
                np.testing.assert_array_equal(problem.reference.p['pts'],
                    np.arange(76*12800, 95*12800+1, 256, dtype=np.int64))
                np.testing.assert_array_equal(problem.body.frames, np.arange(951))
                value = problem.evaluate(x, False)
                actual_cost = float(value['residual']@value['residual']/2)
                expected_cost = float(report['selection']['retained_metrics']['joint_cost'])
                if not np.isfinite(actual_cost):raise ValueError('Nonfinite restored residual cost')
                np.testing.assert_allclose(actual_cost, expected_cost, rtol=2e-10, atol=1e-8,
                    err_msg='Restored final residual differs from the completed solve')
                if set(value['parts'])!=set(report['parts']):raise ValueError('Restored objective parts changed')
                for key in report['parts']:
                    np.testing.assert_allclose(value['parts'][key], report['parts'][key], rtol=2e-10, atol=1e-8,
                        err_msg='Restored objective part changed: '+key)
                emit('finalize_residual_verified', actual_joint_cost=actual_cost,
                    historical_joint_cost=expected_cost, parts=value['parts'],
                    frozen_seed_reconstruction_max_errors=reconstruction_errors)
                # Export through an isolated staging destination, so a failed
                # recovery cannot masquerade as the completed case result.
                staging.mkdir()
                final = absolute_reference(problem.reference, problem.body.state.current_curve(x[17:]),
                    root=problem.reference.root)
                shell = copy.copy(problem)
                shell.reference = final
                packet = save_packet(shell, info, supports, document, staging, status='joint_fit', report=report)
                packet['absolute_parameter_path'] = np.array(str(parameter))
                packet['continuation_run_json'] = saved['continuation_run_json'].copy()
                packet = _local_packet(export_physics(problem, final, staging, packet),problem,recovered_local)
                _atomic_npz(staging/'fitted/result.npz', packet, project)
                if digest(parameter)!=before_parameter_sha:
                    raise ValueError('Parameter changed during export recovery')
                for path in final_paths:
                    if path.exists():raise FileExistsError('Final artifact appeared during recovery: '+str(path))
                result_path.parent.mkdir(parents=True, exist_ok=True)
                (staging/'fitted/result.npz').replace(result_path)
                for name in ('centroidal_physics.npz', 'physics_diagnostics.json'):
                    (staging/name).replace(dest/name)
                backends = metadata['linear_backends']
                start_quality = _json(dest/'continuation_start_quality.json')
                summary = dict(metadata, accepted=accepted, parameter=str(parameter), result=str(result_path),
                    wall_seconds=metadata['wall_seconds_before_export'],
                    wall_seconds_scope='original setup and solve through parameter save; export recovery separately timed',
                    budget_seconds=metadata['wall_budget_seconds'],
                    source_frames=len(packet['pts']), source_context_seconds=[76, 95], schema=str(packet['state_schema_version']),
                    warm_start=str(saved['continuation_warm_start']),
                    warm_start_sha256=str(saved['continuation_warm_start_sha256']),
                    original_baseline_sha256=start_quality['identity']['original_baseline_sha256'],
                    same_frozen_observations_sha256=digest(problem.body.ledger_path),
                    source_evidence_sha256=digest(output/'source_evidence.json'), finite_flight=flight,
                    inner_linear_backend=backends['requested'], linear_solver_statistics=backends['statistics'],
                    whole_shot_adjustable=recovered_local is None,
                    local_control_seconds=None if recovered_local is None else recovered_local['requested_seconds'],
                    local_parameter_selection=recovered_local, shared_shape_and_camera_frozen=True,
                    accuracy_accepted=False, numerical_convergence=report.get('numerical_convergence', False),
                    initial_joint_cost=start_quality['metrics']['joint_cost'], final_joint_cost=actual_cost,
                    final_quality=report['cross_stage_quality'], finalized_existing_solve=True,
                    export_recovery_seconds=time.perf_counter()-started, new_optimization_steps=0)
                for path in summary_paths:write_json(path, summary)
                integrity = dict(parameter=str(parameter), parameter_sha256=before_parameter_sha,
                    parameter_preserved=digest(parameter)==before_parameter_sha,
                    completed_candidate=str(candidate_path), candidate_sha256=digest(candidate_path),
                    historical_report=str(report_path), historical_report_sha256=digest(report_path),
                    result=str(result_path), result_sha256=digest(result_path), source_frames=len(packet['pts']),
                    actual_joint_cost=actual_cost, historical_joint_cost=expected_cost,
                    accepted_steps=accepted, retained_step=retained, new_optimization_steps=0,
                    original_failure_log_preserved=True, no_fit=True, no_network=True, no_gui=True,
                    export_recovery_seconds=time.perf_counter()-started)
                write_json(output/'finalize_integrity.json', integrity)
                emit('finalize_existing_complete', **integrity)
                print(json.dumps(integrity), flush=True)
                return summary
        except BaseException as exc:
            emit('finalize_existing_failed', exception=type(exc).__name__, message=str(exc))
            write_json(output/'finalize_failure.json', dict(error=str(exc), no_fit=True,
                parameter_preserved=digest(parameter)==before_parameter_sha,
                export_recovery_seconds=time.perf_counter()-started))
            raise
        finally:
            execution.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'base-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('frozen-run', 'warm-start', 'parameter'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--parameter-stem')
    parser.add_argument('--local-control-seconds',type=float,nargs=2)
    parser.add_argument('--finalize-existing', action='store_true')
    parser.add_argument('--wall-seconds', type=float, default=7200.)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--flight', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lift-rotation-chart', action='store_true',
        help='Explicitly upgrade a v1 warm start to its anchored SU(2) rotation chart')
    args = parser.parse_args(argv)
    if args.finalize_existing:
        if args.parameter is None:parser.error('--finalize-existing requires --parameter')
        if args.lift_rotation_chart:parser.error('--lift-rotation-chart is only valid for a continuation solve')
        if args.local_control_seconds is not None:parser.error('--local-control-seconds is only valid for a continuation solve')
        return finalize_existing(args.project, args.base_root, args.output, args.parameter)
    for name in ('frozen_run', 'warm_start', 'parameter_stem'):
        if getattr(args, name) is None:parser.error('--'+name.replace('_', '-')+' is required for a solve')
    if args.parameter is not None:parser.error('--parameter is only valid with --finalize-existing')
    return run(args.project, args.base_root, args.frozen_run, args.warm_start,
        args.output, args.parameter_stem, args.wall_seconds, evidence=args.evidence,
        observations=args.observations, flight=args.flight, lift_rotation_chart=args.lift_rotation_chart,
        local_control_seconds=args.local_control_seconds)


if __name__ == '__main__':
    main()
