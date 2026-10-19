"""Continue a complete quadratic rig with temporal continuity at every exposure.

This independent entry preserves the complete source objective and original
prior targets. It resumes the exact saved quadratic state, then adds only the
missing within-shot rotation/log-zoom acceleration factors. Exposure identities,
source observations and the recorded exposure clock remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import fit_continuous_rig_source_calibrated as affine_driver
from fit_continuous_rig_source_calibrated import (
    base, warm_states, reuse_prior_initializer, prepare_source_references,
    attach_reference_factors, snapshot_dependencies)
from continuous_camera_solver import solve, write_json, save_states, complete_graph_error, values_for
from continuous_quadratic_rig_optin import (
    prepare_quadratic_rig, QuadraticRigCameraModel, add_quadratic_priors)
import continuous_precise_inverse_optin as precise


def make_native_backend(args, model, prior, resources):
    from continuous_quadratic_native_edge_optin import QuadraticNativeEdgeManager
    from continuous_native_affine_priors import NativeAffinePriors
    import continuous_native_linearization_preflight
    import continuous_native_trial_preflight
    manager = QuadraticNativeEdgeManager(model, prior, set(prior),
                                         threads=args.native_edge_threads, cache=False)
    resources['native_edge_manager'] = manager
    return manager, NativeAffinePriors(manager)


def run(args,resources):
    previous=json.loads((args.warm_group/'phase_00.json').read_text(encoding='utf8'))
    continuing_references=(previous['schema']=='rinkspace.shared-rig-original-source-calibration.v1')
    if (args.camera != 'main_long_side' or previous['camera_id'] != args.camera
            or not continuing_references
            or not previous.get('quadratic_logzoom') or previous.get('complete_motion_prior') or args.initialize_absolute_sources
            or args.absolute_initializer_resume_from or args.warm_state != 'accepted_states.npz'
            or not args.prior_state_cache or args.nearboard_source_report
            or args.surface_review
            or not all((args.native_source_edges, args.native_affine_priors,
                        args.native_board_first_hit, args.native_central_domain,
                        args.adaptive_source_curves, args.implicit_adaptive_curve_jacobian))):
        raise ValueError('This continuation requires a saved quadratic main graph and all audited backends')
    old_trace=json.loads((args.warm_group/'solver_trace.json').read_text(encoding='utf8'))
    accepted_steps=[step for step in old_trace['steps'] if step['status']=='accepted']
    if not accepted_steps:
        raise ValueError('Warm quadratic run has no saved accepted step')
    expected_previous_cost=float(accepted_steps[-1]['cost'])
    from audit_quadratic_result import reconstruct
    warm_phase,warm_model,warm_reference_model,warm,loaded_meta=reconstruct(
        args.warm_group,state_name=args.warm_state,require_quadratic=True)
    warm_meta={**loaded_meta['provenance'],
        'resume_role':'exact accepted quadratic state, never prior targets',
        'accepted_iteration':int(accepted_steps[-1]['iteration']),
        'previous_complete_cost':expected_previous_cost}
    # This independent adapter is explicitly opted into; v2 inputs stay intact.
    from historic_development_ads import (prepare_development_ads,
                                          attach_development_ad_factors)
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(base.TMP.resolve()):
        raise ValueError('Use a fresh programme TMP output directory')
    output.mkdir(parents=True)
    started=time.perf_counter()
    central_acceleration={'enabled':False}
    if args.implicit_adaptive_curve_jacobian:
        from continuous_adaptive_curve_implicit_optin import (
            install_adaptive_implicit_optin,adaptive_implicit_optin_metadata)
        install_adaptive_implicit_optin()
    if args.native_central_domain:
        from continuous_central_brown4_optin import enable_projection_backend
        central_acceleration=enable_projection_backend()
    if args.central_edge_brown15:
        from continuous_central_edge_optin import (install_edge_optin,
            registered_edge_adapter,edge_optin_metadata)
        # Install before the source callback capture. Only callbacks registered
        # by this capture use Brown15; every other family keeps its old adapter.
        install_edge_optin()
        base.adapt_factor=registered_edge_adapter
    data,pairs,inputs=base.prepare_inputs(output,args.camera)
    for field in ('camera_id','source_sha256','model_sha256'):
        actual=data['camera_id'] if field=='camera_id' else data['diag'][field]
        if actual!=previous[field]:
            raise ValueError('Warm graph source identity differs: '+field)
    aliases,alias_meta=base.load_exposure_aliases(data)
    clock,clock_meta,valid_pairs=base.build_exposure_clock(data,aliases)
    model,prior_initial,baseline,init_meta=base.make_rig(data,aliases,clock,1.,True)
    adapted=(reuse_prior_initializer(args.prior_state_cache,model,prior_initial,data)
        if args.prior_state_cache is not None else
        base.adapt_rig_initializer(model,prior_initial,baseline,data['scene'],
                                  base.extended_native_pairs(pairs)))
    prior_copy={k:v.copy() for k,v in prior_initial.items()}
    surface_metadata=None
    if args.surface_review is not None:
        from continuous_source_surface_review import apply_source_surface_review
        data,pairs,surface_metadata=apply_source_surface_review(
            data,pairs,args.surface_review)
    elif args.camera=='secondary_long_side':
        raise ValueError('Secondary source calibration requires its frozen foreground review')
    pairs=base.extended_native_pairs(pairs)
    board_acceleration={'enabled':False}
    if args.native_board_first_hit:
        from continuous_board_first_hit_optin import upgrade_pairs
        pairs,board_acceleration=upgrade_pairs(pairs)
    # The irregular short-end #4 print has unvalidated rectangle geometry.
    # Remove its complete class of absolute pixel factors, not error-selected
    # corners. Keep the historical source packet and non-pixel scene priors.
    uncertain_ad4=[spec for spec in data['retained']
                  if spec['file']=='ad_landmarks.py' and spec['function']=='factor'
                  and 1004 in spec['keys']]
    data={**data,'retained':[spec for spec in data['retained']
                            if not (spec['file']=='ad_landmarks.py'
                                    and spec['function']=='factor' and 1004 in spec['keys'])]}
    ads,ad_metadata=prepare_development_ads(data,aliases)
    nearboard_rows=[]
    nearboard_metadata={'enabled':False}
    if args.nearboard_source_report:
        from secondary_nearboard_top_source import (
            prepare_secondary_nearboard_top,attach_secondary_nearboard_top_factors)
        nearboard_rows,nearboard_metadata=prepare_secondary_nearboard_top(
            data,aliases,report_path=args.nearboard_source_report,
            expected_report_sha256=args.nearboard_source_sha256)
        nearboard_metadata['enabled']=True
    references=prepare_source_references(data,model,prior_initial)
    original_prior=references.initial
    model,augmented_prior,quadratic_meta=prepare_quadratic_rig(model,original_prior)
    references.model=QuadraticRigCameraModel.from_rig(references.model,
        lens_quadratic_keys=model.lens_quadratic_keys,
        brown_quadratic_keys=model.brown_quadratic_keys)
    references.initial=augmented_prior
    with np.load(args.warm_group/'prior_initializer_states.npz',allow_pickle=False) as z:
        saved_prior={int(k):v.copy() for k,v in zip(z['keys'],z['states'])}
    if (saved_prior.keys()!=augmented_prior.keys() or
            any(not np.array_equal(saved_prior[k],augmented_prior[k]) for k in saved_prior)):
        raise ValueError('Original programme/reference/quadratic prior targets differ')
    if warm.keys()!=augmented_prior.keys():
        raise ValueError('Saved accepted state key set differs')
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
            raise ValueError('Warm quadratic role allocation differs')
    quadratic_meta={**previous['quadratic_logzoom'],
        'continued_nonzero_curvature':True,'warm_camera_count_bitwise_equal':checked,
        'warm_state_sha256':base.sha(args.warm_group/args.warm_state)}
    absolute_initialization={'enabled':False}
    # Window allocation includes the extra reference nodes. Programme priors
    # still use the reproduced ORIGINAL v2 initializer, before any warm merge.
    native_edge_manager,affine_priors=make_native_backend(args,model,references.initial,resources)
    graph,counts=base.build_graph(data,pairs,model,references.initial,
        clock,valid_pairs,.25,native_edge_manager=native_edge_manager,graph_owner=resources,
        affine_priors=affine_priors)
    resources['graph']=graph
    graph._adaptive_curve_worker_count=getattr(args,'reference_curve_workers',8)
    if any(not np.array_equal(prior_copy[k],references.initial[k])
           for k in prior_copy):
        raise RuntimeError('Reference preparation changed programme prior targets')
    reference_metadata=(attach_reference_factors(graph,data,references,
        adaptive_curves=args.adaptive_source_curves)
        if references.model is not None else references.metadata)
    ad_factors=attach_development_ad_factors(graph,data,model,initial,ads)
    if args.nearboard_source_report:
        attach_nearboard=attach_secondary_nearboard_top_factors
        if args.adaptive_source_curves:
            from secondary_nearboard_adaptive_curve import attach_secondary_nearboard_adaptive_factors
            attach_nearboard=attach_secondary_nearboard_adaptive_factors
        nearboard_metadata['factor_attachment']=attach_nearboard(
            graph,data,model,initial,nearboard_rows)
    original_factor_count=len(graph._pending)
    if (counts != previous['factor_counts'] or original_factor_count !=
            previous['original_complete_factor_count']):
        raise ValueError('Original complete graph factor inventory changed')
    quadratic_priors=add_quadratic_priors(graph,model,initial,affine_priors=affine_priors)
    if len(graph._pending) != original_factor_count+4:
        raise ValueError('Expected precisely four new curvature regularization factors')
    previous_complete_count=len(graph._pending)
    if previous_complete_count != previous['graph_factors_with_source_calibration']:
        raise ValueError('Reconstructed complete quadratic graph factor count differs')
    from continuous_full_timeline_regularization import add_complete_motion_priors
    complete_motion=add_complete_motion_priors(graph,model,clock,clock_meta,
        valid_pairs,affine_priors=affine_priors)
    complete_motion['old_complete_factor_count']=previous_complete_count
    complete_motion['new_complete_factor_count']=len(graph._pending)
    if len(graph._pending)-previous_complete_count != complete_motion['added_factor_count']:
        raise ValueError('Missing-motion factor inventory differs')
    inverse_record=dict(schema='rinkspace.corrected-inverse-continuation.v1',
        implementation_sha256=base.sha(Path(precise.__file__)),
        source_capture_inverse='original implementation; physical source/plane assignments preserved',
        original_inverse_source_preflight=base.preflight(pairs,model,initial))
    precise.install()
    validity=base.preflight(pairs,model,initial)
    inverse_record.update(backend=precise.metadata(),
        corrected_inverse_source_preflight=validity,status='complete_source_preflight_finished')
    write_json(output/'precise_inverse_backend.json',inverse_record)
    write_json(output/'initial_source_geometry.json',validity)
    if validity['invalid_pair_count']:
        raise ValueError('Preserved source pixels have invalid initial board geometry')
    phase={**previous,'schema':'rinkspace.shared-rig-original-source-calibration.v1',
        'quadratic_logzoom':quadratic_meta,'quadratic_priors':quadratic_priors,
        'original_complete_factor_count':original_factor_count,
        'model':'Movable C rig; quadratic K4/Brown4; complete within-shot R/logzoom continuity',
        'complete_motion_prior':complete_motion,
        'source_inputs':inputs,'source_surface_review':surface_metadata,
        'warm_estimate':warm_meta,'aliases':alias_meta,'exposure_clock':clock_meta,
        'warm_reference_states_preserved':continuing_references,
        'initializer':init_meta,'initializer_adaptation':adapted,
        'initial_geometry':validity,'factor_counts':counts,
        'reference_calibration':reference_metadata,
        'board_first_hit_acceleration':board_acceleration,
        'central_domain_acceleration':central_acceleration,
        'source_edge_acceleration':(edge_optin_metadata()
            if args.central_edge_brown15 else {'enabled':False}),
        'native_source_edge_acceleration':(native_edge_manager.metadata()
            if native_edge_manager is not None else {'enabled':False}),
        'adaptive_source_curves':args.adaptive_source_curves,
        'absolute_source_initialization':absolute_initialization,
        'native_affine_priors':(affine_priors.metadata() if affine_priors is not None else {'enabled':False}),
        'implicit_adaptive_curve_jacobian':(adaptive_implicit_optin_metadata()
            if args.implicit_adaptive_curve_jacobian else {'enabled':False}),
        'development_ad_source':ad_metadata,'development_ad_factors':ad_factors,
        'user_confirmed_nearboard_top':nearboard_metadata,
        'excluded_unvalidated_ad4_absolute_pixel_factors':len(uncertain_ad4),
        'excluded_ad4_factor_keys':[list(map(int,s['keys'])) for s in uncertain_ad4],
        'graph_factors_with_source_calibration':len(graph._pending),
        'graph_nodes_with_references':len(initial),
        'prior_targets':'original v2 programme initializer; warm states supply starts only',
        'source_uv_changed':False,'fresh_source_validation_fitted':False,
        'full_programme_accepted':False}
    if native_edge_manager is not None:
        import continuous_native_linearization_preflight
    phase['code_dependencies']=snapshot_dependencies(output)
    phase['code_sha256']={name:base.sha(Path(__file__).with_name(name))
                          for name in previous['code_sha256']}
    write_json(output/'phase_00.json',phase)
    save_states(output/'prior_initializer_states.npz',references.initial)
    base.export(model,initial,data,aliases,output/'initial_cameras.npz')
    try:
        initial_cost=complete_graph_error(graph,initial,native_edge_manager=native_edge_manager,
            diagnostics_path=output/'initial_solver_diagnostics.json')
        if not np.isfinite(initial_cost):
            raise ValueError('Augmented source graph initial objective is nonfinite')
        values=values_for(graph,initial)
        added_cost=float(sum(f.error(values) for f in graph._pending[previous_complete_count:]))
        recovered_old_cost=float(initial_cost-added_cost)
        objective_equivalence=dict(previous_accepted_cost=expected_previous_cost,
            restored_original_objective=recovered_old_cost,
            added_complete_motion_prior_cost=added_cost,augmented_initial_cost=float(initial_cost),
            absolute_difference=abs(recovered_old_cost-expected_previous_cost),
            tolerance=max(1.e-6,1.e-9*abs(expected_previous_cost)),
            original_complete_factors=previous_complete_count,
            new_regularization_factors=complete_motion['added_factor_count'],
            original_priors_bitwise_equal=True,warm_cameras_bitwise_equal=checked)
        write_json(output/'original_objective_equivalence.json',objective_equivalence)
        if objective_equivalence['absolute_difference'] > objective_equivalence['tolerance']:
            raise ValueError('Restored old complete objective differs from saved accepted state')
    except Exception as exc:
        write_json(output/'initial_graph_error.json',dict(
            evaluated_all_factors=False,error_type=type(exc).__name__,detail=str(exc),
            factor_family=getattr(exc,'factor_family',None),
            source_keys=getattr(exc,'source_keys',None)))
        raise
    write_json(output/'initial_graph_error.json',dict(
        evaluated_all_factors=True,finite_objective=True,cost=initial_cost,
        factors=len(graph._pending),derivative_domain_verified=False))
    print(json.dumps(dict(stage='source_calibrated_model_ready',camera=args.camera,
        programme_frames=len(model.frames),reference_frames=len(references.references),
        factors=len(graph._pending),initial_cost=initial_cost,
        seconds=time.perf_counter()-started)),flush=True)
    if args.preflight_only:
        save_states(output/'preflight_initial_states.npz',initial)
        if native_edge_manager is not None:
            write_json(output/'native_source_edge_runtime.json',native_edge_manager.metadata())
        return
    try:
        solved,trace=solve(graph,initial,output,max_iterations=args.iterations,
            native_edge_manager=native_edge_manager)
    finally:
        acceleration_runtime={}
        if args.native_central_domain:
            from continuous_central_brown4_optin import backend_stats
            acceleration_runtime['central_domain']=backend_stats()
        if args.central_edge_brown15:
            acceleration_runtime['source_edge']=edge_optin_metadata()
        if native_edge_manager is not None:
            acceleration_runtime['native_source_edge']=native_edge_manager.metadata()
        if args.implicit_adaptive_curve_jacobian:
            acceleration_runtime['implicit_adaptive_curve']=adaptive_implicit_optin_metadata()
        write_json(output/'acceleration_runtime.json',acceleration_runtime)
    report=base.export(model,solved,data,aliases,output/'shot_cameras.npz')
    stamps=np.asarray(sorted(references.model.frames) if references.model is not None else [],np.int64)
    np.savez_compressed(output/'reference_cameras.npz',pts=stamps,
        camera25=np.asarray([references.model.camera25(int(p),solved) for p in stamps]).reshape(-1,25),
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'])
    report.update(camera_id=args.camera,shot_indices=list(data['shot_indices']),
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'],
        phase_sha256=base.sha(output/'phase_00.json'),seconds=time.perf_counter()-started,
        accepted=False,scope='Complete programme quadratic rig with preserved sources and continuous motion priors at all exposures; visual acceptance required')
    write_json(output/'summary.json',report)
    inverse_record.update(backend=precise.metadata(),status='driver_completed')
    write_json(output/'precise_inverse_backend.json',inverse_record)
    print(json.dumps(report),flush=True)


def main():
    # Reuse only the established command parser/resource cleanup, not its run.
    original = affine_driver.run
    affine_driver.run = run
    try:
        affine_driver.main()
    finally:
        affine_driver.run = original


if __name__ == '__main__':
    main()
