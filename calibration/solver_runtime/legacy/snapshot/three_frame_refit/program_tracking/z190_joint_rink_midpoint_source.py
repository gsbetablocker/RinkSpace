"""Joint fixed-mount recovery with continuous source identities and one rink.

This opt-in successor includes the user's major low F1/Lane1 evidence and
the confirmed secondary near-board sequence. Five numeric rink dimensions
are shared by every source block; three heights are shared per installation.
Preparation and physical derivative checks are the default. A solve remains
an unaccepted candidate until real complete-shot review passes.
"""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
sys.path[:0]=[str(ROOT/'three_frame_refit/program_tracking'),str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2, numpy as np
from scipy.sparse import csr_matrix, vstack
from z190_joint_rink_dimensions_source import RinkBlock, main_held
from z190_joint_shared_F_height_source import N, SCENE, module_at, main_input, with_main1_clipped_support
from z190_main_cut_depth_profile_recovery import CURRENT
from z190_main_zoom_k1_source_information import read, write, ref
from z190_secondary_confirmed_nearboard_source import (
    prepare_full_secondary_confirmed_nearboard, typed_nearboard_absolute_value_jac15)
from z190_secondary_qualified_core_source import review_held_groups
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_physical_camera import PhysicalCameraModel, CanonicalView
from z190_continuous_source_bundle import SCENE_UNIT_M, SOURCE_SHA
from shared_rink_numeric_geometry_midpoint import SharedRinkNumericGeometryMidpoint
from continuous_central_brown4_optin import enable_projection_backend
from z190_joint_source_solver import solve_joint_source, make_profile_update, source_cost

WARM=N/'joint_full_source_shared_rink_dimensions_v1/latest_accepted_state.npz'
BLOCK_NAMES=['main1','main7','secondary','low','main9']


def prepare_joint(output,warm=WARM,low_temporal_source=None):
    output=Path(output).resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Bulk source diagnostics must stay in CodexData')
    output.mkdir(parents=True,exist_ok=False)
    enable_projection_backend();cv2.setNumThreads(1)
    model,native,aliases,views=PhysicalCameraModel.load_npz(CURRENT)
    cache_root=N/'joint_full_source_shared_rink_dimensions_preparation_v1'
    def cache(path):return path if path.exists() else None
    main1=main_input(output/'main1_source',1,cache(cache_root/'main1_source/original_motion_source_admission.npz'))
    with_main1_clipped_support(main1)
    main7=main_input(output/'main7_source',7,cache(cache_root/'main7_source/original_motion_source_admission.npz'))
    sec=prepare_full_secondary_confirmed_nearboard(output/'secondary_source',SCENE)
    secondary=SimpleNamespace(**sec)
    # The low factory retains base canonical world geometry. Its local-only
    # midpoint/height block is not used in this joint five-geometry objective.
    from z190_low_major_ice_continuous_source_recovery import prepare as prepare_major_low
    low_options={}if low_temporal_source is None else dict(temporal_source=Path(low_temporal_source))
    low_state=prepare_major_low(output/'low_source',warm=WARM,**low_options)
    low=SimpleNamespace(**vars(low_state.prepared))
    low.bundle=low_state.block.bundle;low.q=low_state.block.local(low_state.q)
    low.manifest=low_state.manifest
    main9mod=module_at(N/'ice_alignment_review/main9_full_source_factory.py')
    main9=main9mod.prepare_full_main9_core_source(output/'main9_source',SCENE,
        source_admission_cache=cache(cache_root/'main9_source/original_source_adapter/original_motion_source_admission.npz'))
    geometry=SharedRinkNumericGeometryMidpoint(read(SCENE))
    ids=sorted(model.calibrations);slots={name:5+i for i,name in enumerate(ids)}
    items=[('main1',main1,absolute_value_jac15),('main7',main7,absolute_value_jac15),
        ('secondary',secondary,typed_nearboard_absolute_value_jac15),
        ('low',low,absolute_value_jac15),('main9',main9,main9mod.typed_absolute_value_jac15)]
    blocks=[];start=8
    parts=[np.r_[np.zeros(5),[model.calibrations[name].pivot_xyz[2]*SCENE_UNIT_M for name in ids]]]
    for name,prepared,absolute in items:
        block=RinkBlock(name,prepared,geometry,slots[prepared.bundle.calibration.installation_id],start,
            absolute,geometry_parameter_count=5)
        blocks.append(block);parts.append(prepared.q[4:]);start+=len(prepared.q)-4
    q=np.concatenate(parts);production_q=q.copy()
    production_q[0]=-main1.manifest['shared_F_scene_binding']['F_column_dx_m']
    frozen=Path(warm).read_bytes();(output/'warm_joint_state_frozen.npz').write_bytes(frozen)
    with np.load(io.BytesIO(frozen),allow_pickle=False)as z:
        old=z['q'].copy();starts=z['aux__block_start'];lengths=z['aux__block_lengths']
        names=z['aux__block_names'].tolist();pts=z['aux__canonical_pts']
        ng=int(z['aux__geometry_parameter_count'])if 'aux__geometry_parameter_count'in z.files else 4
    actual_pts=np.concatenate([b.bundle.stamps for b in blocks])
    if names!=BLOCK_NAMES or not np.array_equal(actual_pts,pts) or ng not in (4,5):
        raise ValueError('Warm actual source identities/block order changed')
    if len(old)!=ng+3+sum(map(int,lengths)) or len(starts)!=len(blocks):
        raise ValueError('Warm parameter layout changed')
    q[:ng]=old[:ng];q[5:8]=old[ng:ng+3]
    for block,oldstart,length in zip(blocks,starts,lengths):
        if int(length)!=len(block.q0)-4 or int(oldstart)<ng+3:
            raise ValueError('Warm complete exposure state count changed')
        q[block.start:block.start+int(length)]=old[int(oldstart):int(oldstart)+int(length)]
    geometry.validate_parameters(q[:5])
    held={b.name:main_held(b.prepared)for b in blocks if b.name.startswith('main')}
    secondary_held={int(p):list(rows)for p,rows in review_held_groups(sec)}
    for row in sec['confirmed_nearboard_held_rows']:
        secondary_held.setdefault(int(row['pts']),[]).append(row)
    held['secondary']=sorted(secondary_held.items())
    low_held={}
    for p,rows in low_state.held+low_state.pending:low_held.setdefault(int(p),[]).extend(rows)
    for p,rows in low_state.independent:low_held.setdefault(int(p),[]).extend(rows)
    held['low']=sorted(low_held.items())
    original_counts=[246156,400539,39679,434991,799549]
    if sum(len(b.bundle.stamps)for b in blocks)!=7703:
        raise ValueError('Complete actual exposure inventory changed')
    for b,count in zip(blocks,original_counts):
        if b.name!='low'and len(b.bundle.a)!=count:
            raise ValueError('Original admitted source graph inventory changed:'+b.name)
    if len(low.bundle.a)<434991 or (low_temporal_source is None and len(low.bundle.a)!=434991):
        raise ValueError('Low original source prefix or explicit temporal scope changed')
    # A temporal family adds real measurements; its source pixels do not
    # replace or silently reclassify the old pairwise graph.
    original_low=low_state.prepared.bundle
    for key in ('a','b','ia','ib'):
        if not np.array_equal(getattr(low.bundle,key)[:434991],getattr(original_low,key)):
            raise ValueError('Persistent source changed original pixel/endpoint prefix:'+key)
    if not np.array_equal(low.bundle.motion_sigma[:869982],original_low.motion_sigma):
        raise ValueError('Persistent source changed original motion sigma prefix')
    legacy_finite=original_low.finite_mask
    if not np.array_equal(low.bundle.finite_planes[:434991][legacy_finite],original_low.finite_planes[legacy_finite]):
        raise ValueError('Persistent source changed already confirmed finite materials')
    return SimpleNamespace(output=output,model=model,native=native,aliases=aliases,views=views,
        geometry=geometry,blocks=blocks,q=q,production_q=production_q,ids=ids,slots=slots,held=held,
        low_temporal_source=None if low_temporal_source is None else ref(low_temporal_source),
        added_motion_rows=len(low.bundle.a)-434991,
        warm_binding=dict(checkpoint=ref(warm),starting_state_only=True,old_geometry_count=ng,
            same_actual_exposure_and_block_order=True,old_main4_rejected_candidate_not_read=True))


def run(args):
    state=prepare_joint(args.output,args.warm,args.low_temporal_source);blocks=state.blocks
    refs={name:next(b for b in blocks if b.bundle.calibration.installation_id==name)for name in state.ids}
    # Per-block height prior was removed by RinkBlock; roll remains once per
    # actual exposure. Three reference heights and one midpoint are global.
    quadratic=np.r_[np.concatenate([b.quadratic for b in blocks]),np.ones(4,bool)]
    scales=np.r_[np.concatenate([np.r_[b.bundle.native_residual_sigma,np.ones(len(b.bundle.stamps))]for b in blocks]),np.ones(4)]
    def evaluate(q,jacobian=True):
        result=[b.evaluate(q,jacobian)for b in blocks]
        values=[r[0]if jacobian else r for r in result]
        matrices=[r[1]for r in result]if jacobian else []
        for name in state.ids:
            b=refs[name];local=b.local(q);bundle=b.bundle;p=bundle.reference_pts
            cal=bundle.current_calibration(local);cam,_,vj=cal.camera25_and_jacobians(bundle.view(local,p))
            values.append(np.array([(cam[2]*SCENE_UNIT_M-1.6)/.18]))
            if jacobian:
                row=np.zeros((1,len(q)));row[0,state.slots[name]]=1/.18
                pos=b.start+4*bundle.index[p];row[0,pos:pos+4]=vj[2]*SCENE_UNIT_M/.18
                matrices.append(csr_matrix(row))
        mid,mj=state.geometry.midpoint_prior(q[:5]);values.append(mid)
        if jacobian:
            row=np.zeros((1,len(q)));row[:,:5]=mj;matrices.append(csr_matrix(row))
        value=np.concatenate(values)
        return(value,vstack(matrices,format='csr'))if jacobian else value
    def guard(q):
        state.geometry.validate_parameters(q[:5])
        if not np.isfinite(q).all()or np.any(q[5:8]<=0):raise ValueError('Physical installation height domain')
        for b in blocks:
            local=b.local(q);focal=local[7::4]
            if np.any((focal<np.log(100.))|(focal>np.log(50000.))):raise ValueError('Physical focal domain')
            b.bundle.validate_finite_extents(b.bundle.cache(local,False))
    def diagnostics(q):
        return dict(shared_geometry_metres=q[:5].tolist(),
            common_total_row_midpoint_m=state.geometry.row_midpoint*SCENE_UNIT_M+float(q[4]),
            pivot_heights_m={name:float(q[state.slots[name]])for name in state.ids})
    def source_review(q,event=None):
        rows=[]
        for b in blocks:
            rows.extend(b.absolute_records(q));rows.extend(b.absolute_records(q,state.held[b.name]))
        persistent=None
        if state.low_temporal_source is not None:
            from z190_persistent_ice_source import persistent_ice_source_report
            low=next(b for b in blocks if b.name=='low')
            persistent=persistent_ice_source_report(low.bundle,low.local(q),
                Path(state.low_temporal_source['path']),low.prepared.aliases)
        write(state.output/'latest_actual_source_review.json',dict(event=event,
            diagnostics=diagnostics(q),rows=rows,persistent_ice_native_fit_and_held=persistent,
            accuracy_accepted=False,
            full_cut_and_independent_held_review_required=True))
        write(state.output/'latest_candidate_scene_UNACCEPTED.json',state.geometry.scene(q[:5]))
    manifest=dict(schema='rinkspace.full-source.shared-rink-midpoint-continuous-identities.v2',
        current=ref(CURRENT),base_scene=ref(SCENE),warm_start=state.warm_binding,
        blocks=[dict(name=b.name,actual=len(b.bundle.stamps),motion=len(b.bundle.a),
            input_manifest=b.prepared.manifest if hasattr(b.prepared,'manifest')else b.prepared.input_manifest,
            source_absolute_points=sum(len(r['uv'])for _,rows in b.groups for r in rows))for b in blocks],
        actual_exposures=7703,original_source_motion_rows=1920914,
        added_persistent_ice_source_rows=state.added_motion_rows,
        source_motion_rows=1920914+state.added_motion_rows,low_temporal_source=state.low_temporal_source,
        original_motion_prefix_UV_endpoints_sigma_and_confirmed_materials_exact=True,
        geometry_numeric_schema=state.geometry.manifest(),
        global_parameter_names=['F_column_dx_m','R_column_dx_m','row_pitch_delta_m','outer_radius_delta_m',
            'common_row_midpoint_delta_m',*['pivot_height_m:'+name for name in state.ids]],
        source_identities_fixed_across_entire_cuts=True,source_motion_native_UV_sigma_alias_materials_retained=True,
        original_unknown_depth_is_pairwise_nuisance_not_persistent_world_track=True,
        priors_once=dict(actual_roll=7703,installation_reference_optical_height=3,common_midpoint=1),
        fixed=['pivotXY','nonzero lever per installation','PP/k1','metric','boards/ads','remaining markings'],
        main4_unrecovered=True,body_rebound=False,full_programme_exported=False,accuracy_accepted=False)
    write(state.output/'input_manifest.json',manifest)
    guard(state.q)
    source_review(state.production_q,dict(stage='unchanged_production_source_baseline'))
    (state.output/'latest_actual_source_review.json').replace(state.output/'current_production_actual_source_review.json')
    source_review(state.q,dict(stage='joint_unaccepted_initial'))
    residual,jac=evaluate(state.q,True)
    if residual.shape!=scales.shape or jac.shape!=(len(scales),len(state.q)):
        raise ValueError('Complete source/prior CSR inventory changed')
    weights=np.minimum(1.,2./np.maximum(abs(residual),1e-12));weights[quadratic]=1.
    gradient=np.asarray(jac.T@(weights*residual)).ravel();checks=[]
    for col in range(8):
        width=1e-5;plus=state.q.copy();minus=state.q.copy();plus[col]+=width;minus[col]-=width
        fd=(source_cost(evaluate(plus,False),quadratic)-source_cost(evaluate(minus,False),quadratic))/(2*width)
        error=abs(fd-gradient[col])/max(1.,abs(fd),abs(gradient[col]))
        checks.append(dict(column=col,analytic=float(gradient[col]),source_fd=float(fd),relative_error=float(error)))
    write(state.output/'actual_global_source_cost_direction_checks.json',checks)
    print(json.dumps(dict(stage='joint_midpoint_complete_source_checked',actual=7703,
        original_motion=1920914,added_persistent_ice_source_rows=state.added_motion_rows,checks=checks)),flush=True)
    if max(r['relative_error']for r in checks)>.005:raise ValueError('Complete source global derivative failed')
    del residual,jac
    if not args.solve:return state
    result=solve_joint_source(evaluate,state.q,quadratic,scales,state.output/'solver',guard,
        update_profiles=make_profile_update(blocks),checkpoint_callback=source_review,
        checkpoint_identity=dict(input_manifest=ref(state.output/'input_manifest.json')),
        checkpoint_arrays=dict(block_start=np.asarray([b.start for b in blocks]),
            block_lengths=np.asarray([len(b.q0)-4 for b in blocks]),block_names=np.asarray(BLOCK_NAMES),
            canonical_pts=np.concatenate([b.bundle.stamps for b in blocks]),geometry_parameter_count=np.array(5)),
        diagnostic_values=diagnostics)
    source_review(result.q,dict(stage='candidate_final',status=result.status,stop_reason=result.stop_reason))
    calibrations=dict(state.model.calibrations);views=dict(state.views)
    for b in blocks:
        local=b.local(result.q);cal=b.bundle.current_calibration(local);calibrations[cal.installation_id]=cal
        for p in b.bundle.stamps:views[p]=CanonicalView(cal.installation_id,b.bundle.view(local,p))
    camera=state.output/'full_camera_WITH_UNRECOVERED_MAIN4_UNACCEPTED.npz'
    PhysicalCameraModel(calibrations).save_npz(camera,state.native,state.aliases,views,
        source_sha256=SOURCE_SHA,time_base='1/15360')
    write(state.output/'report.json',dict(status=result.status,stop_reason=result.stop_reason,
        seconds=result.seconds,camera=ref(camera),scene=ref(state.output/'latest_candidate_scene_UNACCEPTED.json'),
        accuracy_accepted=False,main4_shared_geometry_height_but_current_RF=True))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--warm',type=Path,default=WARM)
    parser.add_argument('--low-temporal-source',type=Path,
        help='Explicit frozen real persistent ice source family; original motion remains an exact prefix')
    parser.add_argument('--solve',action='store_true');run(parser.parse_args())
