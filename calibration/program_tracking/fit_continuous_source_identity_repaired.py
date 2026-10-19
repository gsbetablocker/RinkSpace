"""Continue the complete movable-camera graph after source-identity repair.

Restore and check the saved complete objective first. Replace the old main
board-top class with actual, independently identified blue-cap pixels, add the
user-confirmed near cap and reviewed original motion, and retain all adjustable
camera parameters and original priors. Numerical completion is not acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import fit_continuous_complete_motion_source as continuity
from fit_continuous_rig_source_calibrated import (
    base, reuse_prior_initializer, prepare_source_references,
    attach_reference_factors, snapshot_dependencies, release_native_resources)
from continuous_camera_solver import solve, write_json, save_states, complete_graph_error
from continuous_quadratic_rig_optin import (
    prepare_quadratic_rig, QuadraticRigCameraModel, add_quadratic_priors)
from continuous_full_timeline_regularization import add_complete_motion_priors
import continuous_precise_inverse_optin as precise
import numpy as np


def run(args, resources):
    previous=json.loads((args.warm_group/'phase_00.json').read_text(encoding='utf8'))
    if (args.camera!='main_long_side' or previous['camera_id']!=args.camera
            or previous['schema']!='rinkspace.shared-rig-original-source-calibration.v1'
            or not previous.get('quadratic_logzoom')
            or not previous.get('complete_motion_prior')
            or previous.get('source_identity_repair')
            or args.warm_state!='accepted_states.npz' or not args.prior_state_cache
            or args.initialize_absolute_sources or args.absolute_initializer_resume_from
            or args.nearboard_source_report or args.surface_review
            or not all((args.native_source_edges,args.native_affine_priors,
                        args.native_board_first_hit,args.native_central_domain,
                        args.adaptive_source_curves,args.implicit_adaptive_curve_jacobian))):
        raise ValueError('Requires original complete-motion quadratic main graph and audited backends')
    trace_path=args.warm_group/'solver_trace_at_snapshot.json'
    if not trace_path.is_file():
        trace_path=args.warm_group/'solver_trace.json'
    old_trace=json.loads(trace_path.read_text(encoding='utf8'))
    accepted=[s for s in old_trace['steps'] if s['status']=='accepted']
    if not accepted:
        raise ValueError('Warm state has no accepted trace step')
    expected_cost=float(accepted[-1]['cost'])
    from audit_quadratic_result import reconstruct
    _,warm_model,warm_reference_model,warm,loaded=reconstruct(
        args.warm_group,state_name=args.warm_state,require_quadratic=True)
    warm_meta={**loaded['provenance'],'trace_path':str(trace_path),
        'trace_sha256':base.sha(trace_path),
        'accepted_iteration':int(accepted[-1]['iteration']),
        'previous_complete_cost':expected_cost,
        'resume_role':'exact saved accepted state; never prior targets'}
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(base.TMP.resolve()):
        raise ValueError('Use a fresh programme TMP output directory')
    output.mkdir(parents=True)
    started=time.perf_counter()
    from continuous_adaptive_curve_implicit_optin import (
        install_adaptive_implicit_optin,adaptive_implicit_optin_metadata)
    from continuous_central_brown4_optin import enable_projection_backend,backend_stats
    from continuous_central_edge_optin import (
        install_edge_optin,registered_edge_adapter,edge_optin_metadata)
    install_adaptive_implicit_optin()
    central_acceleration=enable_projection_backend()
    install_edge_optin()
    base.adapt_factor=registered_edge_adapter
    # Capture the original source callbacks before changing inverse backends.
    data,pairs,inputs=base.prepare_inputs(output,args.camera)
    for field in ('camera_id','source_sha256','model_sha256'):
        actual=data['camera_id'] if field=='camera_id' else data['diag'][field]
        if actual!=previous[field]:
            raise ValueError('Warm graph source identity differs: '+field)
    aliases,alias_meta=base.load_exposure_aliases(data)
    clock,clock_meta,valid_pairs=base.build_exposure_clock(data,aliases)
    model,prior,baseline,init_meta=base.make_rig(data,aliases,clock,1.,True)
    adapted=reuse_prior_initializer(args.prior_state_cache,model,prior,data)
    prior_copy={k:v.copy() for k,v in prior.items()}
    pairs=base.extended_native_pairs(pairs)
    from continuous_board_first_hit_optin import upgrade_pairs
    pairs,board_acceleration=upgrade_pairs(pairs)
    uncertain_ad4=[s for s in data['retained'] if s['file']=='ad_landmarks.py'
                   and s['function']=='factor' and 1004 in s['keys']]
    data={**data,'retained':[s for s in data['retained'] if not
        (s['file']=='ad_landmarks.py' and s['function']=='factor' and 1004 in s['keys'])]}
    from historic_development_ads import prepare_development_ads,attach_development_ad_factors
    ads,ad_metadata=prepare_development_ads(data,aliases)
    references=prepare_source_references(data,model,prior)
    model,augmented_prior,quadratic_meta=prepare_quadratic_rig(model,references.initial)
    references.model=QuadraticRigCameraModel.from_rig(references.model,
        lens_quadratic_keys=model.lens_quadratic_keys,
        brown_quadratic_keys=model.brown_quadratic_keys)
    references.initial=augmented_prior
    with np.load(args.warm_group/'prior_initializer_states.npz',allow_pickle=False) as z:
        saved_prior={int(k):v.copy() for k,v in zip(z['keys'],z['states'])}
    if (saved_prior.keys()!=augmented_prior.keys()
            or any(not np.array_equal(saved_prior[k],augmented_prior[k]) for k in saved_prior)
            or any(not np.array_equal(prior_copy[k],augmented_prior[k]) for k in prior_copy)
            or warm.keys()!=augmented_prior.keys()):
        raise ValueError('Original programme/reference/quadratic prior targets or keys differ')
    initial={k:v.copy() for k,v in warm.items()}
    checked=0
    for before,after in ((warm_model,model),(warm_reference_model,references.model)):
        if set(before.frames)!=set(after.frames):
            raise ValueError('Warm frame coverage differs')
        for pts in before.frames:
            if not np.array_equal(before.camera25(pts,warm),after.camera25(pts,initial)):
                raise ValueError('Reconstructed warm camera differs at '+str(pts))
            checked+=1
    for name in ('lens_quadratic_keys','brown_quadratic_keys'):
        if quadratic_meta[name]!=previous['quadratic_logzoom'][name]:
            raise ValueError('Quadratic role allocation differs')
    quadratic_meta={**previous['quadratic_logzoom'],'continued_nonzero_curvature':True,
        'warm_camera_count_bitwise_equal':checked,
        'warm_state_sha256':base.sha(args.warm_group/args.warm_state)}

    def build_complete(source_data):
        manager,affine=continuity.make_native_backend(args,model,augmented_prior,resources)
        graph,counts=base.build_graph(source_data,pairs,model,augmented_prior,
            clock,valid_pairs,.25,native_edge_manager=manager,
            graph_owner=resources,affine_priors=affine)
        graph._adaptive_curve_worker_count=args.reference_curve_workers
        ref_meta=attach_reference_factors(graph,source_data,references,
                                          adaptive_curves=True)
        ad_factors=attach_development_ad_factors(graph,source_data,model,initial,ads)
        original_count=len(graph._pending)
        qpriors=add_quadratic_priors(graph,model,initial,affine_priors=affine)
        time_meta=add_complete_motion_priors(graph,model,clock,clock_meta,valid_pairs,
                                            affine_priors=affine)
        if (len(graph._pending)!=original_count+4+time_meta['added_factor_count']
                or time_meta['added_factor_count']!=previous['complete_motion_prior']['added_factor_count']
                or time_meta['uncovered_canonical_source_pts']):
            raise ValueError('Complete temporal/quadratic prior inventory differs')
        return graph,manager,affine,counts,original_count,ref_meta,ad_factors,qpriors,time_meta

    # Establish an exact continuity gate independently of the changed objective.
    old_graph,old_manager,old_affine,old_counts,old_n,_,_,_,_=build_complete(data)
    if (old_counts!=previous['factor_counts']
            or old_n!=previous['original_complete_factor_count']
            or len(old_graph._pending)!=previous['graph_factors_with_source_calibration']):
        raise ValueError('Original complete objective factor inventory differs')
    precise.install()
    validity=base.preflight(pairs,model,initial)
    write_json(output/'initial_original_source_geometry.json',validity)
    if validity['invalid_pair_count']:
        raise ValueError('Existing source motion has invalid warm geometry')
    restored=complete_graph_error(old_graph,initial,native_edge_manager=old_manager,
        diagnostics_path=output/'restored_objective_diagnostics.json')
    equivalence=dict(previous_accepted_cost=expected_cost,restored_original_objective=restored,
        absolute_difference=abs(restored-expected_cost),
        tolerance=max(1.e-6,1.e-9*abs(expected_cost)),
        original_complete_factors=len(old_graph._pending),
        original_priors_bitwise_equal=True,warm_cameras_bitwise_equal=checked)
    write_json(output/'original_objective_equivalence.json',equivalence)
    if not np.isfinite(restored) or equivalence['absolute_difference']>equivalence['tolerance']:
        raise ValueError('Saved state/trace or restored complete objective differs')
    print(json.dumps(dict(stage='original_complete_objective_restored',**equivalence)),flush=True)
    # Clear all Factors before unloading the old native thread state.
    old_graph=old_manager=old_affine=None
    release_native_resources(resources)

    from far_rail_reviewed_source import load_reviewed_main_rail
    from main_reviewed_rail_inputs import replace_main_top_source_specs
    rail_rows,rail_provenance=load_reviewed_main_rail(data,args.rail_review,
        expected_manifest_sha256=args.rail_review_sha256)
    changed,rail_meta=replace_main_top_source_specs(data,model,initial,rail_rows,rail_provenance)
    graph,manager,affine,counts,original_n,ref_meta,ad_factors,qpriors,time_meta=build_complete(changed)
    expected_counts=dict(old_counts)
    delta=rail_meta['new_top_exposure_factors']-rail_meta['original_top_factors']
    for name in ('track_sequence.py:_add_source_edge_factors','total_with_model_priors'):
        expected_counts[name]+=delta
    if counts!=expected_counts or original_n!=old_n+delta:
        raise ValueError('Source repair changed another original factor class')
    from main_nearboard_reviewed_source import prepare_main_nearboard_source
    from main_nearboard_normal_factors import attach_main_nearboard_normal_factors
    near_rows,near_meta=prepare_main_nearboard_source(changed,aliases)
    before_near=len(graph._pending)
    near_meta['factor_attachment']=attach_main_nearboard_normal_factors(
        graph,changed,model,initial,near_rows)
    near_meta['attached_factor_count']=len(graph._pending)-before_near
    from continuous_reviewed_surface_motion import (
        load_reviewed_main_motion,attach_reviewed_main_motion_factors)
    motion_rows,motion_meta=load_reviewed_main_motion(changed,args.motion_sidecar,
        args.motion_review,expected_sidecar_sha256=args.motion_sidecar_sha256,
        expected_review_sha256=args.motion_review_sha256)
    motion_meta=attach_reviewed_main_motion_factors(graph,changed,model,initial,motion_rows,
        reviewed_metadata=motion_meta)
    expected_total=(previous['graph_factors_with_source_calibration']+delta
                    +near_meta['attached_factor_count']+motion_meta['attached_factor_count'])
    if len(graph._pending)!=expected_total:
        raise ValueError('Repaired graph inventory differs from declared source changes')
    repair_meta=dict(schema='rinkspace.complete-main-source-identity-repair.v1',
        original_objective_equivalence=equivalence,rail=rail_meta,
        near_board=near_meta,reviewed_original_motion=motion_meta,
        old_complete_factor_count=previous['graph_factors_with_source_calibration'],
        new_complete_factor_count=len(graph._pending),
        original_uv_cache_modified=False,source_observation_set_replaced=True,
        original_camera_prior_targets_changed=False,camera_parameters_fixed=False,
        localization_accuracy_accepted=False)
    write_json(output/'source_identity_repair.json',repair_meta)
    phase={**previous,'quadratic_logzoom':quadratic_meta,'quadratic_priors':qpriors,
        'model':'Movable C rig; quadratic K4/Brown4; complete motion; source-identified caps and motion',
        'original_complete_factor_count':original_n,'complete_motion_prior':time_meta,
        'source_identity_repair':dict(manifest=str(output/'source_identity_repair.json'),
            sha256=base.sha(output/'source_identity_repair.json')),
        'source_inputs':inputs,'warm_estimate':warm_meta,'aliases':alias_meta,
        'exposure_clock':clock_meta,'initializer':init_meta,'initializer_adaptation':adapted,
        'factor_counts':counts,'initial_geometry':validity,'reference_calibration':ref_meta,
        'board_first_hit_acceleration':board_acceleration,
        'central_domain_acceleration':central_acceleration,
        'source_edge_acceleration':edge_optin_metadata(),
        'native_source_edge_acceleration':manager.metadata(),
        'native_affine_priors':affine.metadata(),
        'implicit_adaptive_curve_jacobian':adaptive_implicit_optin_metadata(),
        'development_ad_source':ad_metadata,'development_ad_factors':ad_factors,
        'user_confirmed_nearboard_top':near_meta,
        'excluded_unvalidated_ad4_absolute_pixel_factors':len(uncertain_ad4),
        'excluded_ad4_factor_keys':[list(map(int,s['keys'])) for s in uncertain_ad4],
        'graph_factors_with_source_calibration':len(graph._pending),
        'graph_nodes_with_references':len(initial),
        'source_uv_changed':False,'source_factor_set_changed':True,
        'fresh_source_validation_fitted':False,'full_programme_accepted':False}
    phase['code_dependencies']=snapshot_dependencies(output)
    phase['code_sha256']={name:base.sha(Path(__file__).with_name(name))
                          for name in previous['code_sha256']}
    phase['source_repair_entry_sha256']=base.sha(Path(__file__))
    write_json(output/'phase_00.json',phase)
    save_states(output/'prior_initializer_states.npz',augmented_prior)
    base.export(model,initial,changed,aliases,output/'initial_cameras.npz')
    try:
        initial_cost=complete_graph_error(graph,initial,native_edge_manager=manager,
            diagnostics_path=output/'initial_solver_diagnostics.json')
        if not np.isfinite(initial_cost):
            raise ValueError('Repaired complete source objective is nonfinite')
    except Exception as exc:
        write_json(output/'initial_graph_error.json',dict(evaluated_all_factors=False,
            error_type=type(exc).__name__,detail=str(exc),
            factor_family=getattr(exc,'factor_family',None),
            source_keys=getattr(exc,'source_keys',None)))
        raise
    write_json(output/'initial_graph_error.json',dict(evaluated_all_factors=True,
        finite_objective=True,cost=initial_cost,factors=len(graph._pending),
        derivative_domain_verified=False,old_and_new_objectives_comparable=False))
    print(json.dumps(dict(stage='source_identity_repaired_graph_ready',
        programme_frames=len(model.frames),reference_frames=len(references.references),
        factors=len(graph._pending),initial_cost=initial_cost,
        seconds=time.perf_counter()-started)),flush=True)
    if args.preflight_only:
        save_states(output/'preflight_initial_states.npz',initial)
        write_json(output/'native_source_edge_runtime.json',manager.metadata())
        return
    try:
        solved,trace=solve(graph,initial,output,max_iterations=args.iterations,
                            native_edge_manager=manager)
    finally:
        write_json(output/'acceleration_runtime.json',dict(
            central_domain=backend_stats(),source_edge=edge_optin_metadata(),
            native_source_edge=manager.metadata(),
            implicit_adaptive_curve=adaptive_implicit_optin_metadata()))
    report=base.export(model,solved,changed,aliases,output/'shot_cameras.npz')
    stamps=np.asarray(sorted(references.model.frames),np.int64)
    np.savez_compressed(output/'reference_cameras.npz',pts=stamps,
        camera25=np.asarray([references.model.camera25(int(p),solved) for p in stamps]),
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'])
    report.update(camera_id=args.camera,shot_indices=list(data['shot_indices']),
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'],
        phase_sha256=base.sha(output/'phase_00.json'),seconds=time.perf_counter()-started,
        accepted=False,scope='Complete source-identity repaired main rig; real-video validation required')
    write_json(output/'summary.json',report)
    write_json(output/'precise_inverse_backend.json',precise.metadata())
    print(json.dumps(report),flush=True)


def main():
    extra=argparse.ArgumentParser(add_help=False)
    for name in ('rail-review','motion-sidecar','motion-review'):
        extra.add_argument('--'+name,type=Path,required=True)
        extra.add_argument('--'+name+'-sha256',required=True)
    options,remaining=extra.parse_known_args()
    previous_argv=sys.argv
    original_run=continuity.affine_driver.run
    def augmented_run(args,resources):
        for key,value in vars(options).items():
            setattr(args,key,value)
        return run(args,resources)
    continuity.affine_driver.run=augmented_run
    sys.argv=[sys.argv[0],*remaining]
    try:
        continuity.affine_driver.main()
    finally:
        continuity.affine_driver.run=original_run
        sys.argv=previous_argv


if __name__=='__main__':
    main()
