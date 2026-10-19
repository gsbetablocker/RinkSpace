"""Jointly calibrate a complete movable-centre rig from original source pixels.

Programme motion and v2 mechanical priors are retained. Independent same-rig
reference exposures share the adjustable optics/mechanics; historical real
advertisement observations supply absolute location across the programme.
Old camera vectors initialize states only. This is not a release/acceptance tool.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import fit_continuous_rig_camera as base
import numpy as np
from continuous_camera_solver import solve, write_json, save_states, complete_graph_error
import gtsam
from fit_continuous_rig_surface_review import warm_states
from continuous_reference_calibration import (prepare_reference_calibration,
    seed_reference_initial, attach_reference_factors)


def snapshot_dependencies(output):
    repository=Path(__file__).resolve().parents[2]
    files={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:
            continue
        path=Path(filename).resolve()
        if path.suffix=='.py' and path.is_relative_to(repository) and path.is_file():
            relative=path.relative_to(repository)
            files[str(relative)]=base.sha(path)
            target=output/'code_dependencies'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    write_json(output/'code_dependencies.json',dict(
        scope='loaded project modules at graph construction; source solver files preserved',
        sha256=files))
    return files


def prepare_source_references(data, model, prior_initial):
    if data['camera_id']!='low_end':
        return prepare_reference_calibration(data,model,prior_initial)
    # The reviewed 14-image reference bank contains no low-end rig. Do not
    # relabel another rig's images or invent reference states for this group.
    metadata=dict(camera_id='low_end',reference_frame_count=0,
        reference_pts=[],reference_image_sha256={},reference_sources={},
        new_graph_nodes={},source_video_sha256=data['diag']['source_sha256'],
        current_programme_scene_sha256=data['diag']['model_sha256'],
        source_curve_pixel_count=0,source_measurement_factors=0,
        reason='No low-end camera images in the reviewed 14-reference bank',
        source_uv_modified=False)
    return SimpleNamespace(initial={k:v.copy() for k,v in prior_initial.items()},
        metadata=metadata,references=[],model=None)


def reuse_prior_initializer(directory, model, initial, data):
    """Reuse already verified initializer work, never a fitted/warm estimate."""
    phase=json.loads((directory/'phase_00.json').read_text(encoding='utf8'))
    for key,value in (('camera_id',data['camera_id']),
                      ('source_sha256',data['diag']['source_sha256']),
                      ('model_sha256',data['diag']['model_sha256'])):
        if phase[key]!=value:
            raise ValueError('Cached prior initializer identity differs: '+key)
    saved=directory/'code_dependencies/calibration/program_tracking/fit_continuous_rig_camera.py'
    def initializer_definition(path):
        tree=ast.parse(path.read_text(encoding='utf8'))
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                             and n.name=='adapt_rig_initializer'),include_attributes=False)
    if (base.sha(saved)!=phase['code_dependencies'][
            str(Path('calibration/program_tracking/fit_continuous_rig_camera.py'))]
            or initializer_definition(saved)!=initializer_definition(Path(base.__file__))):
        raise ValueError('Cached original initializer implementation differs')
    payload=directory/'prior_initializer_states.npz'
    with np.load(payload,allow_pickle=False) as z:
        cached={int(k):v.copy() for k,v in zip(z['keys'],z['states'])}
    frame_keys=set(model.frame_keys)
    for key,state in initial.items():
        if key not in cached or cached[key].shape!=(11,) or not np.isfinite(cached[key]).all():
            raise ValueError('Cached original prior state is absent/invalid')
        live=(3,4,5,6) if key in frame_keys else ()
        unchanged=[c for c in range(11) if c not in live]
        if not np.array_equal(state[unchanged],cached[key][unchanged]):
            raise ValueError('Cached initializer changed non-adapted model coordinates')
        initial[key]=cached[key]
    # warm_states independently reconstructs cameras from these targets and
    # checks against the original v2 initial_cameras, including every exposure.
    return dict(phase['initializer_adaptation'],cache_directory=str(directory.resolve()),
                cache_states_sha256=base.sha(payload),
                cache_role='original prior initializer only; no fitted states reused as priors')


def release_native_resources(resources):
    graph=resources.pop('graph',None)
    manager=resources.pop('native_edge_manager',None)
    if graph is not None:
        from adaptive_curve_parallel import close_adaptive_curve_workers
        close_adaptive_curve_workers(graph)
    if manager is not None:
        failure=sys.exc_info()[1]
        if failure is not None:
            traceback.clear_frames(failure.__traceback__)
        if graph is not None:
            graph._smoother=None
            graph._pending.clear()
            graph._callbacks.clear()
        graph=None
        gc.collect()
        manager.close_after_release()


def make_native_backend(args,model,prior,resources):
    manager=affine=None
    if args.native_source_edges:
        from continuous_native_edge_optin import NativeEdgeManager
        import continuous_native_linearization_preflight
        # Trial capture is part of the complete fallback contract, not an
        # optional acceleration: otherwise a new trial can rebuild the whole
        # graph once per unsupported factor.
        import continuous_native_trial_preflight
        plugin_options=dict(module_name='native_source_edge_trial_prod',
            binary_dir=base.TMP/'solver_acceleration/round10/native/full_bin')
        manager=NativeEdgeManager(model,prior,set(prior),threads=args.native_edge_threads,
            cache=False,**plugin_options)
        resources['native_edge_manager']=manager
        if args.native_affine_priors:
            from continuous_native_affine_priors import NativeAffinePriors
            affine=NativeAffinePriors(manager)
    return manager,affine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',required=True,
                        choices=('main_long_side','secondary_long_side','low_end'))
    parser.add_argument('--warm-group',type=Path,required=True)
    parser.add_argument('--warm-state',choices=('states.npz','accepted_states.npz'),
                        default='states.npz')
    parser.add_argument('--prior-state-cache',type=Path,
                        help='Reuse a verified source-calibrated run original initializer payload')
    parser.add_argument('--surface-review',type=Path)
    parser.add_argument('--nearboard-source-report',type=Path,
                        help='User-confirmed lower board-top source intervals for the complete secondary group')
    parser.add_argument('--nearboard-source-sha256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--native-board-first-hit',action='store_true',
                        help='Use the hash-pinned strict C first-hit kernel and scoped hit reuse')
    parser.add_argument('--native-central-domain',action='store_true',
                        help='Use the verified Brown4 central-domain kernel with original boundary fallback')
    parser.add_argument('--central-edge-brown15',action='store_true',
                        help='Use guarded Brown15 source-edge derivatives; changed branches retain original FD')
    parser.add_argument('--native-source-edges',action='store_true',
                        help='Use round6 complete native source-edge factors with original Python fallback')
    parser.add_argument('--native-edge-threads',type=int,default=8,
                        help='TBB workers for native factors (positive integer; default 8)')
    parser.add_argument('--reference-curve-workers',type=int,default=8,
                        help='Persistent processes for complete reference-curve Jacobians; 0 disables')
    parser.add_argument('--native-trial-values',action='store_true',
                        help='Compatibility flag; trial capture is always used with native source edges')
    parser.add_argument('--native-affine-priors',action='store_true',
                        help='Use equivalent constant native factors for original model priors')
    parser.add_argument('--adaptive-source-curves',action='store_true',
                        help='Adaptively project original reference and confirmed near-board world curves')
    parser.add_argument('--implicit-adaptive-curve-jacobian',action='store_true',
                        help='Use guarded implicit curve derivatives, retaining full FD for non-smooth factors')
    parser.add_argument('--initialize-absolute-sources',action='store_true',
                        help='Initialize the complete secondary rig from existing absolute source constraints before restoring all motion')
    parser.add_argument('--absolute-initializer-resume-from',type=Path,
                        help='Resume a verified saved absolute-source initializer into the fresh output')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.native_edge_threads < 1 or args.reference_curve_workers < 0:
        parser.error('Native threads must be positive and reference workers nonnegative')
    if bool(args.nearboard_source_report) != bool(args.nearboard_source_sha256):
        parser.error('Near-board source report and its SHA256 must be supplied together')
    if args.nearboard_source_report and args.camera!='secondary_long_side':
        parser.error('The confirmed lower-board source report belongs to the secondary group')
    if args.native_source_edges:
        args.central_edge_brown15=True
    if args.native_trial_values and not args.native_source_edges:
        parser.error('Native trial capture requires native source-edge factors')
    if args.native_affine_priors and not args.native_source_edges:
        parser.error('Native affine priors require the native source-edge plugin')
    if args.implicit_adaptive_curve_jacobian and not args.adaptive_source_curves:
        parser.error('Implicit curve derivatives require adaptive source curves')
    if args.initialize_absolute_sources and (
            args.camera!='secondary_long_side' or not args.adaptive_source_curves
            or not args.nearboard_source_report):
        parser.error('Absolute initialization requires secondary group, adaptive curves and reviewed near-board source')
    if args.absolute_initializer_resume_from and not args.initialize_absolute_sources:
        parser.error('Initializer resume requires --initialize-absolute-sources')
    resources={}
    try:
        return run(args,resources)
    finally:
        # Release every graph/optimizer owner before ending TBB's scoped thread
        # limit. This is needed for normal Windows shutdown as well as errors.
        release_native_resources(resources)


def run(args,resources):
    previous=json.loads((args.warm_group/'phase_00.json').read_text(encoding='utf8'))
    continuing_references=(previous['schema']=='rinkspace.shared-rig-original-source-calibration.v1')
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
    warm,warm_meta=warm_states(args.warm_group,model,
        references.initial if continuing_references else prior_initial,data,aliases,
        state_filename=args.warm_state,reference_metadata=references.metadata,
        allow_projection_additions=True)
    if continuing_references:
        with np.load(args.warm_group/'prior_initializer_states.npz',allow_pickle=False) as z:
            old_prior={int(k):v for k,v in zip(z['keys'],z['states'])}
        if old_prior.keys()!=references.initial.keys() or any(
                not np.array_equal(old_prior[k],references.initial[k]) for k in old_prior):
            raise ValueError('Source-calibrated warm run prior targets were not reproduced')
    initial={k:v.copy() for k,v in references.initial.items()}
    initial.update(warm)
    if not continuing_references and references.model is not None:
        seed_reference_initial(references.model,initial,references.references)
    absolute_initialization={'enabled':False}
    if args.initialize_absolute_sources:
        from initialize_rig_from_absolute_sources import initialize_absolute_sources
        # Capture the exact initializer implementation even if preparation fails.
        snapshot_dependencies(output)
        init_manager,init_affine=make_native_backend(args,model,references.initial,resources)
        try:
            initial,absolute_initialization=initialize_absolute_sources(
                data,model,references.initial,initial,clock,valid_pairs,references,ads,
                nearboard_rows,output/'absolute_source_initializer',iterations=args.iterations,
                native_edge_manager=init_manager,affine_priors=init_affine,graph_owner=resources,
                resume_from=args.absolute_initializer_resume_from,
                reference_curve_workers=args.reference_curve_workers)
        finally:
            release_native_resources(resources)
            init_manager=init_affine=None
        absolute_initialization['enabled']=True
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
    validity=base.preflight(pairs,model,initial)
    write_json(output/'initial_source_geometry.json',validity)
    if validity['invalid_pair_count']:
        raise ValueError('Preserved source pixels have invalid initial board geometry')
    phase={**previous,'schema':'rinkspace.shared-rig-original-source-calibration.v1',
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
        accepted=False,scope='Complete programme rig with original source observations; reference images only when available')
    write_json(output/'summary.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
