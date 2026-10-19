"""Fit complete camera groups with shared movable-centre mechanics and Brown4.

Same-camera cuts share adjustable P, camera-coordinate optical offsets and
zoom-dependent K/Brown. A softly regularized per-shot residual centre spline
allows departures from that apparatus model. All priors are explicit modeling
assumptions; no first frame or optical centre is fixed. Source UV is unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
from scipy.optimize import least_squares

from backend import IncrementalCameraGraph
from continuous_camera_inputs import prepare_inputs, load_exposure_aliases
from continuous_exposure_clock import (CLOCK_POLICIES,
                                       build_exposure_clock, require_resume_clock_policy)
from continuous_rig_camera import ContinuousRigCameraModel, initialize_rig
from continuous_brown_adapter import adapt_factor
from continuous_camera_solver import solve, write_json
from fit_continuous_camera import (make_model, raw_spec, data_state,
    is_old_zoom_prior, dummy_factor, acceleration_factor, export, sha, TMP)
from projection import from_state, project_fast, attach_visible_surfaces
from robust_observations import pointwise_huber
from correlated_motion import wrap_correlated_pair
from reference_board_pair import BoardIntersectionError


def make_rig(data, aliases, clock, interval, brown_zoom):
    plain, _, baseline, _ = make_model(data, aliases, clock, interval)
    group = data['camera_id']
    key = max(plain.dependencies(p)[-1] for p in plain.frames)+100
    model = ContinuousRigCameraModel(plain.shot_splines, plain.frames,
        plain.lens_base_keys, plain.lens_slope_keys, plain.lens_zoom_reference,
        mount_keys={group:key}, offset_keys={group:key+1},
        offset_slope_keys={group:key+2}, brown_keys={group:key+3},
        brown_slope_keys={group:key+4} if brown_zoom else {})
    initial, report = initialize_rig(model, baseline)
    return model, initial, baseline, report


def selected_prior(graph, key, columns, target, sigma):
    columns = tuple(columns)
    select = np.eye(11)[list(columns)]
    target = np.asarray(target, float).copy()
    graph.add_observation([key], lambda q: select @ q[0]-target,
        jac_fn=lambda q: [select], residual_dim=len(columns), sigma=sigma)


def adapt_rig_initializer(model,initial,baseline,scene,pairs):
    first_pair={}
    for a,b,pair in pairs:
        first_pair.setdefault(a,pair)
    representatives={}
    for stamp,binding in model.frames.items():
        representatives.setdefault(binding.frame_key,stamp)
    samples=np.asarray([(x,y) for y in np.linspace(300,1040,6)
                        for x in np.linspace(60,1860,10)])
    errors=[]
    for key,stamp in representatives.items():
        old=baseline[stamp]
        xyz,_,valid=attach_visible_surfaces(scene,old,samples)
        xyz=xyz[valid]
        if stamp in first_pair:
            hits=first_pair[stamp].source_hits(data_state(old))
            xyz=np.vstack((xyz,hits.xyz[hits.valid]))
        if len(xyz)<4:
            raise ValueError('Insufficient old-view initializer points')
        target=project_fast(old,xyz)[0]
        q=initial[key]
        def residual(v):
            q[3:7]=v
            return (project_fast(model.camera25(stamp,initial),xyz)[0]-target).reshape(-1)
        fitted=least_squares(residual,q[3:7].copy(),jac='3-point',max_nfev=30,
                             ftol=1e-9,xtol=1e-9,gtol=1e-9)
        q[3:7]=fitted.x
        errors.append(float(np.sqrt(np.mean(residual(fitted.x)**2))))
    return dict(exposures=len(errors),old_view_rms_px_p50=float(np.median(errors)),
        old_view_rms_px_max=float(max(errors)),
        role='old_view_initialization_only; no synthetic fitting observations')


def regularize(graph, model, initial, clock, valid_time_pairs, residual_sigma,*,affine_priors=None):
    prior=(selected_prior if affine_priors is None else affine_priors.add_selected_prior)
    dummy=(dummy_factor if affine_priors is None else affine_priors.add_dummy_factor)
    acceleration=(acceleration_factor if affine_priors is None else affine_priors.add_acceleration_factor)
    for key in model.frame_keys:
        dummy(graph, key, (3,4,5,6))
    for shot_id, spline in model.shot_splines.items():
        for key in spline.control_keys:
            dummy(graph, key, (0,1,2))
            prior(graph,key,(0,1,2),np.zeros(3),residual_sigma/100.)
        times = np.asarray([np.mean(spline.knot_pts[i+1:i+4])/15360.
                            for i in range(len(spline.control_keys))])
        for i in range(1,len(times)-1):
            acceleration(graph,spline.control_keys[i-1:i+2],times[i-1:i+2],
                                [0,1,2],np.full(3,.3/100.))
        stamps = sorted({b.frame_key for b in model.frames.values() if b.shot_id==shot_id})
        for i in range(1,len(stamps)-1):
            a,b,c = stamps[i-1:i+2]
            if (a,b) in valid_time_pairs and (b,c) in valid_time_pairs:
                acceleration(graph,(a,b,c),np.asarray([clock[p] for p in (a,b,c)])/15360.,
                                    [3,4,5,6],[.5]*4)
        first=stamps[0]
        prior(graph,first,(3,4,5,6),initial[first][3:7],[.3,.3,.3,.5])
    for key in model.mount_keys.values():
        dummy(graph,key,(0,1,2))
        prior(graph,key,(0,1,2),initial[key][:3],np.asarray([100.,100.,50.])/100.)
    for mapping in (model.offset_keys,model.offset_slope_keys):
        for key in mapping.values():
            dummy(graph,key,(0,1,2))
            prior(graph,key,(0,1,2),np.zeros(3),np.full(3,3./100.))
    for key in model.lens_base_keys.values():
        cols=(6,8,9,10)
        dummy(graph,key,cols)
        prior(graph,key,cols,initial[key][list(cols)],[.2,.3,.3,.1])
    for key in model.lens_slope_keys.values():
        cols=(6,8,9,10)
        dummy(graph,key,cols)
        prior(graph,key,cols,np.zeros(4),[.1,.2,.2,.1])
    for mapping in (model.brown_keys,model.brown_slope_keys):
        for key in mapping.values():
            dummy(graph,key,(0,1,2,3))
            prior(graph,key,(0,1,2,3),np.zeros(4),[.5,.5,.05,.05])


def extended_native_pairs(pairs):
    from continuous_brown_board_pair import ContinuousBrownBoardPair
    extended=[]
    for a,b,pair in pairs:
        # A new instance of the subclass uses the same immutable geometry and
        # raw UV as the original. Its residual and visibility rules are inherited.
        replacement=ContinuousBrownBoardPair.__new__(ContinuousBrownBoardPair)
        replacement.__dict__=pair.__dict__.copy()
        extended.append((a,b,replacement))
    return extended


def build_graph(data,pairs,model,initial,clock,valid_pairs,residual_sigma,*,
                native_edge_manager=None,graph_owner=None,affine_priors=None):
    graph=IncrementalCameraGraph(11,window_size=len(initial)+2,
        relinearize_threshold=1e-5,refinement_updates=1,optimization='dogleg',
        dogleg_initial_delta=.001,diagnose_cost=True)
    if graph_owner is not None:
        graph_owner['graph']=graph
    counts=Counter()
    active=set(initial)
    def add(keys,raw,jac,options,family,captured_spec=None):
        source_keys=tuple(map(int,keys))
        adapted=adapt_factor(source_keys,raw,jac,options,active,data['states'],model)
        if adapted is None:
            counts['constant_'+family]+=1
            return
        keys,raw,jac,options=adapted
        def residual(q):
            try:
                return raw(q)
            except BoardIntersectionError as exc:
                exc.evaluation_stage='residual'
                exc.factor_family=family
                exc.source_keys=source_keys
                raise
        if jac is None:
            derivative=None
        else:
            def derivative(q):
                try:
                    return jac(q)
                except BoardIntersectionError as exc:
                    exc.evaluation_stage='jacobian'
                    exc.factor_family=family
                    exc.source_keys=source_keys
                    raise
        installed=(native_edge_manager is not None and captured_spec is not None
            and native_edge_manager.maybe_add(graph,captured_spec,
                (keys,residual,derivative,options),source_keys))
        if not installed:
            graph.add_observation(keys,residual,jac_fn=derivative,**options)
        counts[family]+=1
    for spec in data['retained']:
        keys=set(spec['keys'])
        if not keys & data['pts']:
            counts['external_constants']+=1
            continue
        if spec['function'] in ('_mount_equality','add_rig_equality','add_frame_lens',
                                 '_lens_factor','_aspect_factor'):
            counts['removed_old_'+spec['function']]+=1
            continue
        if spec['kind'] in ('prior','between') and keys<=data['pts']:
            counts['removed_old_pose_'+spec['kind']]+=1
            continue
        if is_old_zoom_prior(spec,data['pts']):
            counts['removed_old_zoom_prior']+=1
            continue
        raw,jac,options=raw_spec(spec)
        add(spec['keys'],raw,jac,options,spec['file']+':'+spec['function'],spec)
    quality=data['captured']['source_quality']
    for a,b,pair in pairs:
        sigma=float(getattr(pair,'rinkspace_sigma_px',data['config'].get('native_pair_sigma_px',2.)))
        raw,jac=quality.wrap(pair._raw_residual,pair._raw_jacobians,
            quality.pair_scales(a,b,pair.xy0,pair.xy1,
                               getattr(pair,'rinkspace_quality_kind','native_motion')))
        raw,jac=pointwise_huber(raw,jac,group_size=2,delta=2*sigma)
        similarity=getattr(pair,'rinkspace_similarity_noise_px',
                           data['config'].get('native_pair_similarity_noise_px'))
        if similarity is not None:
            raw,jac,_=wrap_correlated_pair(raw,jac,pair.xy1,sigma_point_px=sigma,
                                           sigma_modes_px=similarity)
            sigma=1.
        add((a,b),raw,jac,dict(residual_dim=2*len(pair.xy0),sigma=sigma),
            'actual_uv_dynamic_brown_board_motion')
    regularize(graph,model,initial,clock,valid_pairs,residual_sigma,affine_priors=affine_priors)
    counts['total_with_model_priors']=len(graph._pending)
    return graph,dict(counts)


def preflight(pairs,model,initial):
    failures=[]
    for a,b,pair in pairs:
        valid,reasons=pair.initial_validity([model.physical_state(a,initial),
                                           model.physical_state(b,initial)])
        if not np.all(valid):
            failures.append(dict(source_pts=a,target_pts=b,indices=np.flatnonzero(~valid).tolist(),
                reason_counts=dict(Counter(reasons[i] for i in np.flatnonzero(~valid)))))
    return dict(pair_count=len(pairs),invalid_pair_count=len(failures),invalid_pairs=failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--knot-seconds',type=float,default=1.)
    parser.add_argument('--residual-centre-sigma',type=float,default=.25)
    parser.add_argument('--brown-zoom',action='store_true')
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--exposure-clock-policy',choices=CLOCK_POLICIES,
                        default=None,required=True,
                        help='Explicit source clock policy; a changed policy requires a fresh graph')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.resume:
        require_resume_clock_policy(output,args.exposure_clock_policy)
    if not output.is_relative_to(TMP.resolve()) or args.knot_seconds<=0 or args.residual_centre_sigma<=0:
        raise ValueError('Invalid output or model prior')
    if output.exists() and not args.resume:
        raise ValueError('Use a fresh output directory; never overwrite prior diagnostics')
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    data,pairs,inputs=prepare_inputs(output,args.camera)
    aliases,alias_meta=load_exposure_aliases(data)
    clock,clock_meta,valid_pairs=build_exposure_clock(
        data, aliases, policy=args.exposure_clock_policy)
    model,initial,baseline,init_meta=make_rig(data,aliases,clock,args.knot_seconds,args.brown_zoom)
    pairs=extended_native_pairs(pairs)
    # Refit only the starting R/zoom to old views under shared optics. These
    # synthetic projections do not enter the source-observation objective.
    adapted=adapt_rig_initializer(model,initial,baseline,data['scene'],pairs)
    validity=preflight(pairs,model,initial)
    write_json(output/'initial_source_geometry.json',validity)
    if validity['invalid_pair_count']:
        raise ValueError('Shared camera initializer has invalid finite-board hits; preserved report')
    graph,counts=build_graph(data,pairs,model,initial,clock,valid_pairs,args.residual_centre_sigma)
    code_names=['fit_continuous_rig_camera.py','continuous_rig_camera.py',
        'continuous_camera_model.py','continuous_brown_adapter.py',
        'continuous_brown_board_pair.py','projection.py','native_board_pair.py',
        'surface_pair.py','reference_board_pair.py','continuous_camera_solver.py',
        'continuous_camera_inputs.py','continuous_camera_auxiliary.py',
        'continuous_exposure_clock.py','fit_continuous_camera.py']
    phase=dict(schema='rinkspace.shared-mechanical-centre-brown-camera.v1',
        camera_id=args.camera,source_sha256=data['diag']['source_sha256'],
        model_sha256=data['diag']['model_sha256'],source_inputs=inputs,
        aliases=alias_meta,exposure_clock=clock_meta,
        exposure_clock_policy=args.exposure_clock_policy,
        frame_clock_virtual_pts={str(p):clock[p] for p in sorted(clock)},
        valid_temporal_acceleration_pairs=sorted(map(list,valid_pairs)),
        shot_knots={str(k):dict(keys=list(s.control_keys),pts=list(s.knot_pts)) for k,s in model.shot_splines.items()},
        lens_base_keys=model.lens_base_keys,lens_slope_keys=model.lens_slope_keys,
        zoom_reference=model.lens_zoom_reference,mount_keys=model.mount_keys,
        offset_keys=model.offset_keys,offset_slope_keys=model.offset_slope_keys,
        brown_keys=model.brown_keys,brown_slope_keys=model.brown_slope_keys,
        initializer=init_meta,initializer_adaptation=adapted,initial_geometry=validity,
        factor_counts=counts,code_sha256={name:sha(Path(__file__).with_name(name)) for name in code_names},
        model='C=P+R.T(d0+d1*(logf-zref))+residual_spline; K and Brown4 shared across same-camera cuts',
        prior_assumptions=dict(mount_sigma_scene=[100,100,50],offset_sigma_scene=3.,
            residual_centre_sigma_scene=args.residual_centre_sigma,
            residual_C_acceleration_spectral_intensity_scene_s1_5=.3,
            R_logzoom_acceleration_spectral_intensity_state_s1_5=.5,
            brown4_sigma=[.5,.5,.05,.05],brown_slope_sigma=[.5,.5,.05,.05]),
        first_camera_fixed=False,optical_center_fixed=False,source_uv_changed=False,
        scene_conditioned_on_v13=True,other_cameras_conditioned=True,
        higher_brown10_fixed_zero=True,fresh_source_validation_fitted=False,
        full_programme_accepted=False)
    if args.resume:
        old=json.loads((output/'phase_00.json').read_text(encoding='utf8'))
        if old!=phase:
            raise ValueError('Resume phase differs from exact saved model/code/input')
    else:
        write_json(output/'phase_00.json',phase)
        snapshot=output/'code_at_start'
        snapshot.mkdir()
        for name in code_names:
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    export(model,initial,data,aliases,output/'initial_cameras.npz')
    print(json.dumps(dict(stage='model_ready',camera=args.camera,frames=len(model.frames),
        exposures=len(model.frame_keys),states=len(initial),factors=len(graph._pending),
        seconds=time.perf_counter()-started)),flush=True)
    if args.preflight_only:
        return
    solved,trace=solve(graph,initial,output,max_iterations=args.iterations,resume=args.resume)
    exported=export(model,solved,data,aliases,output/'shot_cameras.npz')
    summary=dict(camera_id=args.camera,phase_sha256=sha(output/'phase_00.json'),
        shot_indices=list(data['shot_indices']),**exported,
        source_sha256=data['diag']['source_sha256'],model_sha256=data['diag']['model_sha256'],
        solver=trace,seconds=time.perf_counter()-started,accepted=False,
        scope='Complete same-camera group with shared movable rig and Brown4; source review required')
    write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='solver'}),flush=True)


if __name__=='__main__':
    main()
