"""Full-main moving optical centre from current source and bounded rigid mount."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
from dataclasses import replace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from z190_main_zoom_k1_source_information import prepare,read,write,CURRENT_PHYSICAL,AUTHORITY,HELD,CURRENT,IDENTIFIER,ref
from z190_main_zoom_k1_full_source_diagnostic import source_changes,source_summary
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle,solve_moving_bundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_continuous_source_bundle import SOURCE_SHA,SCENE_UNIT_M,radial_inverse
from z190_source_camera_diagnostic import absolute_value_jac15
from main7_source_bundle_candidate import Bundle
from camera_hardware_metadata import calibration_context
from continuous_central_brown4_optin import enable_projection_backend

FACTS=Path('D:/OneDrive/codex_projs/RinkSpace/evidence/camera_hardware_20261001/user_camera_mount_and_height_20261002_v3.json')
FACTS_SHA='e42424eb9f0830336cc44cd9b5d24fa0808eba2186eea00662df25b886fee3e4'
ICE=CURRENT/'z190_main1_incident_source_material_v1/main1_incident_8_ice_source_admission.json'
WALL=CURRENT/'z190_main_early_finite_board_lever_source_v1/source_qualified_finite_diagnostic.json'


def moving_motion_wrapper(**k):
    from z190_moving_center_motion import moving_center_motion_value_jacobians as kernel
    planes=k.pop('planes');jac=k.pop('jacobian')
    return kernel(k.pop('a'),k.pop('b'),k.pop('fa'),k.pop('fb'),k.pop('ra'),k.pop('rb'),
        k.pop('da'),k.pop('db'),k.pop('ca'),k.pop('cb'),k.pop('dca_shared'),k.pop('dcb_shared'),
        k.pop('dca_view'),k.pop('dcb_view'),principal_px=k.pop('pp'),k1=k.pop('k1'),
        sigma=k.pop('sigma'),profile_weight=k.pop('profile_weight'),
        finite_normals=planes[:,:3],finite_d=-planes[:,3],finite_mask=np.linalg.norm(planes[:,:3],axis=1)>0.,jac=jac)


def bind_finite(g,data,cal):
    planes=np.zeros((len(data.a),4));records=[];inputs=[];offsets={};start=0
    for cut in(1,4,7,9):
        for i,(pa,pb,a,b)in enumerate(g.rays[cut]):
            offsets[(cut,i)]=(start,len(a),pa,pb);start+=len(a)
    if start!=len(data.a):raise ValueError('Full source motion order changed')
    expected=next(r['sha256']for r in g.raybindings if r['cut']==1)
    def exact(r,wall=False):
        pair=int(r['original_pair_index']);loc=int(r['original_local_index'])
        lo,n,pa,pb=offsets[(1,pair)];j=lo+loc
        pts=r['canonical_pair_pts']if wall else r['source_native_pair_pts']
        if [pa,pb]!=pts or not np.array_equal(data.a[j],r['xy0'])or not np.array_equal(data.b[j],r['xy1']):
            raise ValueError('Finite original row/UV/direction changed')
        return j,pair,loc,pa,pb
    ice=read(ICE);inputs.append(ref(ICE))
    if ice['original_motion_graph']['sha256']!=expected:raise ValueError('Ice review graph changed')
    for r in ice['rows']:
        j,pair,loc,pa,pb=exact(r)
        if not r['both_end_original_source_reviewed']or r['finite_surface']['kind']!='ice_plane':
            raise ValueError('No both-end finite ice source support')
        planes[j]=[0.,0.,1.,0.]
        records.append(dict(global_bundle_index=j,cut=1,original_pair_index=pair,original_local_index=loc,
            original_global_index=r['original_global_index'],source_native_pair_pts=[pa,pb],xy0=r['xy0'],xy1=r['xy1'],
            surface_kind='ice_plane',material_identity=r['static_texture_identity'],review=ref(ICE),
            source_role=r['split'],circle_identity='unassigned_not_needed_for_material_plane',
            admission='latest_user_authorized_source_material_motion',match_precision_not_measured=True))
    wall=read(WALL);inputs.append(ref(WALL))
    if wall['source_graph']['sha256']!=expected:raise ValueError('Wall source graph changed')
    from native_board_pair import NativeBoardPair
    from z190_static_pair import BrownFixedCenterRayPair
    scene=read(g.scene);bottom=np.asarray(scene['board_surface']['bottom_edge_xyz'],float)
    successor=np.roll(bottom,-1,axis=0)
    for r in wall['finite_source_diagnostic_rows']:
        j,pair,loc,pa,pb=exact(r,True);ca,cb=cal.camera25(g.views[pa].view),cal.camera25(g.views[pb].view)
        bp=NativeBoardPair(data.a[j:j+1],data.b[j:j+1],scene,ca,cb,allowed_panels=np.array(r['allowed_existing_panels']))
        hits=bp._hits_from_rays(ca[:3],BrownFixedCenterRayPair._world_directions(ca,data.a[j:j+1]))
        if not hits.valid.all():raise ValueError('Finite wall initial source facet invalid')
        panel=int(hits.panel[0]);edge=successor[panel,:2]-bottom[panel,:2]
        n=np.r_[edge[1],-edge[0],0.];n/=np.linalg.norm(n);d=-n@bottom[panel]
        allowed=np.array(r['allowed_existing_panels']);same=allowed[np.abs(bottom[allowed]@n+d)<1e-8]
        same=same[np.abs(successor[same]@n+d)<1e-8]
        ends=np.vstack((bottom[same,:2],successor[same,:2]));stations=ends@(edge/np.linalg.norm(edge))
        planes[j]=np.r_[n,d]
        records.append(dict(global_bundle_index=j,cut=1,original_pair_index=pair,original_local_index=loc,
            original_global_index=r['original_global_index'],source_native_pair_pts=[pa,pb],xy0=r['xy0'],xy1=r['xy1'],
            surface_kind='finite_board_face',material_identity=r['stationary_texture'],review=ref(WALL),
            physical_surface_identity=r['finite_surface'],extent_start_xy=ends[np.argmin(stations)],extent_end_xy=ends[np.argmax(stations)],
            computational_initial_facet=panel,source_role=r['split'],seal_preserved_not_renamed_to_print=True,
            admission='latest_user_authorized_source_material_motion',match_precision_not_measured=True))
    return planes,records,inputs


def complete_bundle(g,reference_pts):
    stamps=sorted(g.views);pairs=[p for cut in(1,4,7,9)for p in g.rays[cut]]
    base=Bundle(stamps,g.observed,pairs)
    if len(base.a)!=1720425 or len(stamps)!=5591 or sum(len(rr)for _,rr in base.abs_groups)!=11500:
        raise ValueError('Complete current main source changed')
    if ref(FACTS)['sha256']!=FACTS_SHA:raise ValueError('Latest user mount facts changed')
    facts=read(FACTS);l=np.asarray(facts['rigid_mount_offset']['initial_camera_xyz_m'])/SCENE_UNIT_M
    reference_pts=g.aliases[int(reference_pts)];r=cv2.Rodrigues(np.asarray(g.views[reference_pts].view.rvec_world_to_camera))[0]
    old_cz=g.model.camera25(g.views[reference_pts])[2];pz=old_cz-(r.T@l)[2]
    settings=dict(g.calibration.field_settings)
    settings['hardware_context']=calibration_context('main_long_side',source_sha256=SOURCE_SHA)
    settings['source_authority_admission']=dict(**g.packet['source_authority_admission'],unconfirmed_ice_proposals_excluded_from_fit=True)
    settings['active_motion_model']='fixed_tripod_pivot_shared_constant_nonzero_camera_local_lever'
    settings['active_mount_parameters']=dict(source=ref(FACTS),initial_camera_xyz_m=[0.,-.20,.10],reference_exposure_pts=reference_pts,
        fixed_pivot_xy_scene_units=g.calibration.pivot_xyz[:2],pivot_initialization='old reference optical Cz-(R_reference.T@lever)_z',
        per_frame_unrestricted_C=False,offset_zoom_dependent=False,independent_installation=True)
    settings['reference_optical_height_prior']=dict(reference_exposure_pts=reference_pts,target_above_ice_working_m=1.6,
        sigma_m=.18,operator_ground_minus_ice_unknown=True,pivot_height_is_not_lens_height=True,counted_once_per_installation=True)
    cal=replace(g.calibration,pivot_xyz=(*g.calibration.pivot_xyz[:2],float(pz)),lever_xyz=tuple(l),field_settings=settings)
    planes,records,inputs=bind_finite(g,base,cal)
    bundle=MovingPivotSourceBundle(base,cal,moving_motion_wrapper,reference_pts=reference_pts,
        initial_reference_optical_height_scene=old_cz,finite_planes=planes,finite_records=records,scene=read(g.scene))
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[g.views[p].view.rvec_world_to_camera,
        g.views[p].view.log_f_ratio+np.log(1920.)]for p in stamps])]
    if abs(bundle.camera(q,reference_pts)[2]-old_cz)>1e-12:raise ValueError('Reference lens height double-counts lever')
    del base;gc.collect()
    return bundle,q,dict(reference_pts=reference_pts,old_reference_optical_height_m=old_cz*SCENE_UNIT_M,
        initial_pivot_height_m=pz*SCENE_UNIT_M,initial_lever_xyz_m=[0.,-.20,.10],
        source_motion_points=len(bundle.a),source_pairs=len(pairs),fit_absolute_rows=11500,
        canonical_exposures=len(stamps),finite_material_motion_rows=len(records),
        unknown_depth_conditional_rows=len(bundle.a)-len(records),finite_source_inputs=inputs)


def derivative_checks(bundle,q):
    # Both actual motion endpoints, all shared components, and actual finite
    # absolute source. The kernel owns further profile/KKT precision checks.
    selected=np.array([0]+[r['global_bundle_index']for r in bundle.finite_records],int)
    ia,ib=bundle.ia[selected],bundle.ib[selected]
    def call(x,jac):
        v,r,d,c,s,w=bundle.cache(x,True)
        return moving_motion_wrapper(a=bundle.a[selected],b=bundle.b[selected],fa=np.exp(v[ia,3]),fb=np.exp(v[ib,3]),
            ra=r[ia],rb=r[ib],da=d[ia],db=d[ib],ca=c[ia],cb=c[ib],dca_shared=s[ia],dcb_shared=s[ib],
            dca_view=w[ia],dcb_view=w[ib],pp=np.array([1920*bundle.calibration.principal_x_fraction[0],
            1080*bundle.calibration.principal_y_fraction[0]]),k1=bundle.calibration.k1[0],
            planes=bundle.finite_planes[selected],jacobian=jac,sigma=np.repeat(bundle.motion_sigma[2*selected,None],2,axis=1),profile_weight=None)
    residual,ja,jb,js=call(q,True)[:4];checks=[];columns=list(range(4))
    for i in sorted(set(ia.tolist()+ib.tolist())):columns.extend((4+4*i+np.arange(4)).tolist())
    for col in columns:
        h=1e-7;plus,minus=q.copy(),q.copy();plus[col]+=h;minus[col]-=h
        fd=(call(plus,False)-call(minus,False))/(2*h);a=js[:,col].copy()if col<4 else np.zeros(len(fd))
        if col>=4:
            i,axis=(col-4)//4,(col-4)%4
            a+=ja[:,axis]*np.repeat(ia==i,2)+jb[:,axis]*np.repeat(ib==i,2)
        rel=float(np.linalg.norm(fd-a)/max(1.,np.linalg.norm(fd)))
        checks.append(dict(kind='actual_motion',column=col,relative_error=rel,max_native_error=float(np.max(np.abs(fd-a)))))
    scale=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    for p,rows in bundle.abs_groups[::max(1,len(bundle.abs_groups)//4)]:
        cam,sj,vj=bundle.current_calibration(q).camera25_and_jacobians(bundle.view(q,p));res,raw=absolute_value_jac15(cam,rows,True)
        a=raw@(np.c_[sj[:15,bundle.mapping_names],vj[:15]]*scale[:,None])
        cols=list(range(4))+list(4+4*bundle.index[p]+np.arange(4))
        for local,col in enumerate(cols):
            h=1e-7;plus,minus=q.copy(),q.copy();plus[col]+=h;minus[col]-=h
            fd=(absolute_value_jac15(bundle.camera(plus,p),rows,False)-absolute_value_jac15(bundle.camera(minus,p),rows,False))/(2*h)
            rel=float(np.linalg.norm(fd-a[:,local])/max(1.,np.linalg.norm(fd)))
            checks.append(dict(kind='actual_absolute',pts=p,column=col,relative_error=rel,max_sigma_error=float(np.max(np.abs(fd-a[:,local])))))
    if max(x['relative_error']for x in checks)>5e-4:raise ValueError('Actual moving-C derivative mismatch:'+json.dumps(checks))
    return checks


def save_depth_trace(bundle,q,output):
    from z190_moving_center_motion import moving_center_motion_value_jacobians as kernel
    v,r,d,c,s,w=bundle.cache(q,False);pp=np.array([1920*bundle.calibration.principal_x_fraction[0],1080*bundle.calibration.principal_y_fraction[0]])
    pieces={}
    for lo in range(0,len(bundle.a),65536):
        hi=min(lo+65536,len(bundle.a));ia,ib=bundle.ia[lo:hi],bundle.ib[lo:hi];p=bundle.finite_planes[lo:hi]
        result=kernel(bundle.a[lo:hi],bundle.b[lo:hi],np.exp(v[ia,3]),np.exp(v[ib,3]),r[ia],r[ib],None,None,
            c[ia],c[ib],None,None,None,None,principal_px=pp,k1=bundle.calibration.k1[0],
            sigma=bundle.motion_sigma[2*lo:2*hi].reshape(-1,2),
            profile_weight=None if bundle.profile_weights is None else bundle.profile_weights[lo:hi],
            finite_normals=p[:,:3],finite_d=-p[:,3],finite_mask=bundle.finite_mask[lo:hi],jac=False,return_state=True)
        for key,value in result[-1].items():
            a=np.asarray(value)
            if a.ndim and len(a)==hi-lo:pieces.setdefault(key,[]).append(a)
    arrays={key:np.concatenate(value)for key,value in pieces.items()}
    arrays['original_motion_global_index']=np.arange(len(bundle.a));arrays['finite_material_qualified']=bundle.finite_mask
    path=output/'latent_depth_conditional_trace.npz';np.savez_compressed(path,**arrays)
    return dict(file=ref(path),rows=len(bundle.a),arrays=list(arrays),accepted_material_or_depth_truth=False)


def run(a):
    continuing=a.resume_huber or a.export_accepted
    if continuing:
        if not a.solve or not(a.output/'input_manifest.json').exists():raise ValueError('Resume requires existing source run and --solve')
    else:a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);enable_projection_backend()
    print(json.dumps(dict(stage='prepare_current_main_moving_C')),flush=True);g=prepare(a)
    bundle,q,checks=complete_bundle(g,a.reference_pts)
    manifest=dict(schema='rinkspace.current-main-nonzero-shared-rigid-mount-full-source.v1',
        current=ref(a.current),current_binding=ref(g.binding_path),user_mount_height_facts=ref(FACTS),
        source_authority=ref(a.authority),source_authority_admission=g.packet['source_authority_admission'],
        scene=ref(g.scene),source_sha256=SOURCE_SHA,time_base='1/15360',complete_source=checks,original_rays=g.raybindings,
        pending68_not_fit=True,original_held_future_not_fit=True,new41300held=ref(a.held),
        unknown_motion='positive-depth/infinity latent nuisance, source material untyped; conditional static hypothesis',
        finite_motion='exact original source UV and independently reviewed material/physical surface',
        source_pixel_loss='original sigma and scalar Huber2; inner depth frozen IRLS weights',
        fixed=['pivot XY','PP','k1 intercept','hard scene','zoom dependence zero','k2/tangential/skew'],
        active=['independent pivot Pz','shared constant lever3','all5591 real-exposure R3/effectivef'],
        lever_bounds_m=bundle.lever_bounds_m,reference_optical_height_prior=bundle.calibration.field_settings['reference_optical_height_prior'],
        entrypoint=ref(__file__),adapter=ref(Path(__file__).with_name('z190_moving_pivot_source_bundle.py')),
        motion_kernel=ref(Path(__file__).with_name('z190_moving_center_motion.py')),
        accuracy_accepted=False,production_pointer_changed=False,video_generated=False)
    if continuing:
        old_manifest=read(a.output/'input_manifest.json')
        for key in('current','source_authority','scene','motion_kernel','user_mount_height_facts'):
            if old_manifest[key]!=manifest[key]:raise ValueError('Resume source/geometry/kernel/facts changed:'+key)
        with np.load(continuing,allow_pickle=False)as checkpoint:
            q=checkpoint['q'].copy()
            if not np.array_equal(checkpoint['canonical_pts'],bundle.stamps):raise ValueError('Resume exposure graph changed')
        previous=read(a.output/'bundle_iterations.json')
        if not previous:raise ValueError('Continuation requires original solver history')
        if a.resume_huber and any(r['phase']!='linear_source'for r in previous):raise ValueError('Resume is only linear-to-Huber continuation')
        if a.resume_huber:write(a.output/'linear_source_iterations_preserved.json',previous)
        if a.export_accepted:
            if not(a.output/'huber_precision_check.json').exists():raise ValueError('Precision export requires distinct complete q replay evidence')
            precision=read(a.output/'huber_precision_check.json')
            if precision['checkpoints'][0]['sha256']==precision['checkpoints'][1]['sha256']:raise ValueError('Invalid identical-state precision evidence')
            write(a.output/'huber_source_iterations_preserved.json',previous)
            write(a.output/'source_precision_solver_stop.json',dict(
                schema='rinkspace.main-moving-C-source-precision-stop.v1',checkpoint=ref(a.export_accepted),
                precision_evidence=ref(a.output/'huber_precision_check.json'),
                reason='All fit absolute and held/future actual changes below original native sampling support; motion tail conditional profile remains explicitly unverified. No source row removed.',
                original_motion_points=len(bundle.a),original_absolute_fit_rows=11500,
                last_iteration=previous[-1],all_history_preserved=ref(a.output/'huber_source_iterations_preserved.json'),
                trace_metric='Native L2 depth reprofile at frozen final camera; original optimization retained scalar Huber2 source weights',
                accuracy_accepted=False))
        else:write(a.output/'huber_phase_continuation.json',dict(
            schema='rinkspace.accepted-main-moving-C-linear-to-huber-continuation.v1',
            checkpoint=ref(a.resume_huber),precision_evidence=ref(a.output/'phase_precision_check.json'),
            reason='Actual complete adjacent accepted-state residual updates below native source sampling support; max motion 0.108662 px / absolute 0.025928 px, no rows over 0.5 px',
            inherited_initial_manifest=ref(a.output/'input_manifest.json'),
            inherited_linear_history=ref(a.output/'linear_source_iterations_preserved.json'),
            continuation_entrypoint=ref(__file__),continuation_adapter=ref(Path(__file__).with_name('z190_moving_pivot_source_bundle.py')),
            complete_source_inventory_unchanged=checks,held_pending_future_not_fit=True,
            source_or_accuracy_acceptance=False))
        print(json.dumps(dict(stage='export_complete_accepted_main_q'if a.export_accepted else'resume_complete_accepted_main_q_huber',checkpoint=str(continuing),inherited_iterations=len(previous))),flush=True)
    else:
        write(a.output/'input_manifest.json',manifest);write(a.output/'qualified_finite_source_rows.json',bundle.finite_records)
        np.savez_compressed(a.output/'initial_state4rf.npz',q=q,canonical_pts=bundle.stamps)
        write(a.output/'all_source_current_baseline.json',source_changes(g,g.model,g.views))
        d=derivative_checks(bundle,q);write(a.output/'actual_source_derivative_checks.json',d)
        print(json.dumps(dict(stage='full_main_moving_C_prepared',**checks,max_derivative_relative=max(x['relative_error']for x in d))),flush=True)
    if not a.solve:return
    print(json.dumps(dict(stage='full_main_nonzero_mount_export_started'if a.export_accepted else'full_main_nonzero_mount_solver_started',exposures=len(bundle.stamps))),flush=True)
    if a.export_accepted:
        fitted,history,seconds=q,previous,previous[-1]['elapsed_seconds']
    elif a.resume_huber:
        fitted,history,seconds=solve_moving_bundle(bundle,q,a.output,phases=(True,),initial_history=previous,initial_damping=previous[-1]['damping'])
    else:fitted,history,seconds=solve_moving_bundle(bundle,q,a.output)
    np.savez_compressed(a.output/'optimized_state4rf.npz',q=fitted,canonical_pts=bundle.stamps)
    cal=bundle.current_calibration(fitted);settings=dict(cal.field_settings)
    settings['active_mount_parameters']=dict(settings['active_mount_parameters'],
        fitted_camera_xyz_m=(np.asarray(cal.lever_xyz)*SCENE_UNIT_M).tolist(),fitted_pivot_xyz_scene_units=cal.pivot_xyz)
    cal=replace(cal,field_settings=settings);model=PhysicalCameraModel({IDENTIFIER:cal})
    views={p:CanonicalView(IDENTIFIER,bundle.view(fitted,p))for p in bundle.stamps}
    path=a.output/'physical_main_nonzero_mount_candidate_unaccepted.npz'
    model.save_npz(path,g.native,g.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,native,aliases,loaded=PhysicalCameraModel.load_npz(path)
    if aliases!=g.aliases or not np.array_equal(native,g.native)or not all(np.array_equal(saved.camera25(loaded[p]),bundle.camera(fitted,p))for p in loaded):
        raise ValueError('Moving physical/alias replay failed')
    saved.save_calibration_json(a.output/'calibration_nonzero_mount_candidate.json')
    changes=source_changes(g,saved,loaded);write(a.output/'all_absolute_source_changes.json',changes)
    write(a.output/'source_summary.json',source_summary(changes))
    trace=save_depth_trace(bundle,fitted,a.output);centres=np.array([saved.camera25(loaded[p])[:3]for p in bundle.stamps])*SCENE_UNIT_M
    report=dict(manifest,physical_candidate=ref(path),solve_seconds=seconds,history=history,latent_depth_trace=trace,
        fitted_pivot_xyz_scene_units=cal.pivot_xyz,fitted_pivot_height_m=cal.pivot_xyz[2]*SCENE_UNIT_M,
        fitted_lever_xyz_m=(np.asarray(cal.lever_xyz)*SCENE_UNIT_M).tolist(),
        optical_height_range_m=[float(centres[:,2].min()),float(centres[:,2].max())],
        fitted_reference_optical_height_m=bundle.camera(fitted,bundle.reference_pts)[2]*SCENE_UNIT_M,
        source_aliases_bitexact=True,all_source_changes=ref(a.output/'all_absolute_source_changes.json'),
        source_summary=ref(a.output/'source_summary.json'),disposition='source_review_pending_unaccepted',body_not_updated=True)
    if continuing:report['inherited_linear_to_huber_continuation']=ref(a.output/'huber_phase_continuation.json')
    if a.export_accepted:
        report['source_precision_solver_stop']=ref(a.output/'source_precision_solver_stop.json')
        report['latent_depth_trace']['profile_metric']='Native L2 reprofile at final frozen camera; no further camera optimization'
    write(a.output/'report.json',report)
    print(json.dumps(dict(stage='moving_C_full_source_export_complete',physical=str(path),seconds=seconds,
        pivot_height_m=report['fitted_pivot_height_m'],lever_m=report['fitted_lever_xyz_m'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=CURRENT_PHYSICAL)
    p.add_argument('--authority',type=Path,default=AUTHORITY);p.add_argument('--held',type=Path,default=HELD)
    p.add_argument('--reference-pts',type=int,default=229114112)
    p.add_argument('--output',type=Path,default=CURRENT/'z190_main_nonzero_mount_full_source_v1')
    resume=p.add_mutually_exclusive_group()
    resume.add_argument('--resume-huber',type=Path,help='Continue existing accepted full state in Huber, preserving original source/linear history')
    resume.add_argument('--export-accepted',type=Path,help='Export complete accepted camera after actual source precision stop; no further optimization')
    p.add_argument('--solve',action='store_true');run(p.parse_args())
