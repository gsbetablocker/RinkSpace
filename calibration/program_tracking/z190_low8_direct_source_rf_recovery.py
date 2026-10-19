"""Repair complete low cut8 R/f with a new directly read Ad4 source key.

Current complete Z190 installation, scene and every original source role and
motion sigma stay fixed.  Cut3/5 states are copied exactly; original cut8 graph
is retained in full.  The new 0.5s source is development fit, 1s stays held.
"""
from __future__ import annotations
import argparse,gc,json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from z190_low_nonzero_mount_full_source import prepare as original_prepare,source_changes,IDENTIFIER
from z190_low_opening_rf_refine import opening_rows,convert_row
from z190_main_nonzero_mount_full_source import moving_motion_wrapper,derivative_checks
from z190_low_static_foreground_rf_recovery import fixed_mount_rf_solve,CURRENT_FULL,CURRENT_SHA
from z190_main_zoom_k1_source_information import read,write,ref,CURRENT
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_continuous_source_bundle import SOURCE_SHA,SCENE_UNIT_M,HARD_SCENE_SHA
from continuous_central_brown4_optin import enable_projection_backend


def bind_source_finite_board(packet_path,source_motion,base,scene,aliases):
    """Bind explicit native-reviewed Ad4 ink rows, never a projected ROI."""
    planes=np.zeros((len(base.a),4));records=[]
    if packet_path is None:return planes,records,None
    packet=read(packet_path)
    if packet['source_motion_graph']!=ref(Path(source_motion['rawpath'])):raise ValueError('Finite source raw graph changed')
    if packet['scene_sha256']!=HARD_SCENE_SHA or packet['source_sha256']!=SOURCE_SHA:raise ValueError('Finite source scene/event changed')
    import hashlib
    for frame in packet['source_frames'].values():
        png=Path(frame['source_png']['path'])
        if ref(png)!=frame['source_png']:raise ValueError('Finite native source PNG changed')
        pixels=cv2.imread(str(png))
        if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest()!=frame['native_bgr_sha256']:raise ValueError('Finite native source pixels changed')
    from ad_landmarks import AdGeometry
    geometry=AdGeometry(scene,4)
    left=geometry.corner(geometry.initial,'left_bottom');right=geometry.corner(geometry.initial,'right_bottom')
    edge=right-left;n=np.array([edge[1],-edge[0],0.]);n/=np.linalg.norm(n);d=-n@left
    for role in geometry.roles:
        if abs(n@geometry.corner(geometry.initial,role)+d)>1e-9:raise ValueError('Ad4 must be the existing straight finite board face')
    with np.load(Path(source_motion['rawpath']),allow_pickle=False)as raw,np.load(Path(source_motion['selected_rows']),allow_pickle=False)as selected:
        ids=selected['original_global_row_indices'];lookup={int(j):i for i,j in enumerate(ids)}
        if len(ids)!=len(base.a):raise ValueError('Finite source retained cut8 row order changed')
        for row in packet['qualified_rows']:
            if not row['both_end_source_material_reviewed']or row['physical_surface_identity']!='ad4_existing_straight_far_board_face':raise ValueError('Finite row lacks source physical material support')
            pair=int(row['original_pair_index']);local=int(row['original_local_index']);glob=int(row['original_global_index'])
            if glob!=int(raw['offsets'][pair])+local:raise ValueError('Finite original row index changed')
            pts=raw['pair_pts'][pair].astype(int).tolist()
            if pts!=row['native_pair_pts']:raise ValueError('Finite native pair/direction changed')
            j=lookup[glob];ca,cb=aliases[pts[0]],aliases[pts[1]]
            if base.stamps[base.ia[j]]!=ca or base.stamps[base.ib[j]]!=cb:raise ValueError('Finite actual exposure mapping changed')
            if not np.array_equal(raw['xy0'][glob],row['xy0'])or not np.array_equal(raw['xy1'][glob],row['xy1']):raise ValueError('Finite raw native UV changed')
            if not np.array_equal(base.a[j],row['xy0'])or not np.array_equal(base.b[j],row['xy1']):raise ValueError('Finite solver original UV order changed')
            if not np.isclose(raw['fb_error_px'][glob],row['fb_error_px'],rtol=0,atol=0):raise ValueError('Finite raw FB trace changed')
            planes[j]=np.r_[n,d]
            records.append(dict(global_bundle_index=j,cut=8,original_pair_index=pair,original_local_index=local,original_global_index=glob,
                source_native_pair_pts=pts,canonical_pair_pts=[ca,cb],xy0=row['xy0'],xy1=row['xy1'],fb_error_px=row['fb_error_px'],
                surface_kind='finite_board_face',physical_surface_identity=row['physical_surface_identity'],material_identity=row['material_identity'],
                extent_start_xy=left[:2],extent_end_xy=right[:2],source_role='original_motion_retained',review=ref(packet_path),
                admission='both_end_native_Ad4_static_print_source_review',match_precision_not_measured=True))
    if len({r['global_bundle_index']for r in records})!=len(records):raise ValueError('Finite rows duplicate')
    return planes,records,ref(packet_path)


class FixedCutBundle(MovingPivotSourceBundle):
    def priors(self,q,cache):
        # The true installation reference R belongs to fixed, untouched cut3.
        # Its lens-height prior is constant for this cut8-only solve, not a
        # new height measurement at cut8. Original per-exposure roll remains.
        values,jac=super().priors(q,cache)
        return values[1:],jac[1:]


def new_direct_rows(path,scene,aliases):
    packet=read(path)
    if packet.get('schema')!='rinkspace.low8.native-Ad4-external-corners-development.v1':return opening_rows(path,scene,aliases)
    if packet['scene_sha256']!=HARD_SCENE_SHA or packet['source_sha256']!=SOURCE_SHA:raise ValueError('New Ad4 corner source changed scene/event')
    from ad_landmarks import AdGeometry
    geometry=AdGeometry(scene,4);rows=[]
    for source in packet['rows']:
        if source['physical_ids']!=['ad4:left_top','ad4:left_bottom']or not source['point_correspondence']or source['split']!='fit':raise ValueError('Only directly visible Ad4 left external corners supported')
        if int(source['source_native_pts'])!=230907648 or source.get('model_projection_used_to_select_or_generate_uv'):raise ValueError('Corner observations must stay original half-second source')
        row=dict(source);row['world']=np.asarray([geometry.corner(geometry.initial,role)for role in('left_top','left_bottom')])
        row['pts']=aliases[int(row['source_native_pts'])];row['origin']=str(path)
        row['world_scene_binding']=dict(scene_sha256=HARD_SCENE_SHA,geometry_locator='AdGeometry(scene,4).corner(initial,left_top/left_bottom)',source_uv_changed=False,historical_fitted_boundary_curves_used=False)
        rows.append(convert_row(row))
    return rows,packet


def objective_direction_check(bundle,q,output):
    """Replay the source-cost derivative without taking a camera step."""
    bundle.profile_weights=None
    residual,jac=bundle.evaluate(q)
    free=jac[:,4:];gradient=np.asarray(free.T@residual).ravel()
    diagonal=np.maximum(np.asarray(free.multiply(free).sum(axis=0)).ravel(),1.)
    direction=-gradient/diagonal
    native_step=free@direction
    maximum=float(np.max(np.abs(native_step[:bundle.source_residual_rows]*bundle.native_residual_sigma)))
    direction*=.01/max(maximum,1e-30)
    predicted=free@direction
    slices={'motion':slice(0,bundle.motion_rows),'absolute':slice(bundle.motion_rows,bundle.source_residual_rows),'roll':slice(bundle.source_residual_rows,None)}
    records=[]
    for amount in (1.,.1,.01):
        positive=q.copy();negative=q.copy();positive[4:]+=amount*direction;negative[4:]-=amount*direction
        rp=bundle.evaluate(positive,False);rm=bundle.evaluate(negative,False)
        finite_difference=(rp-rm)/(2.*amount)
        parts={name:dict(analytic=float(residual[s]@predicted[s]),central_difference=float((rp[s]@rp[s]-rm[s]@rm[s])/(4.*amount)))for name,s in slices.items()}
        row_delta=residual[:bundle.motion_rows]*(finite_difference[:bundle.motion_rows]-predicted[:bundle.motion_rows])
        pairs=row_delta.reshape(-1,2).sum(1)
        ids=np.argsort(np.abs(pairs))[-12:][::-1]
        records.append(dict(relative_step=amount,parts=parts,largest_motion_gradient_discrepancy=[dict(bundle_row=int(i),source_pts=int(bundle.stamps[bundle.ia[i]]),target_pts=int(bundle.stamps[bundle.ib[i]]),value=float(pairs[i]))for i in ids]))
    write(output/'full_source_objective_direction_check.json',dict(camera_step_taken=False,shared_parameters_fixed=True,predicted_maximum_native_step_px=.01,records=records))
    np.savez_compressed(output/'source_objective_direction_replay.npz',q=q,direction_free_rf=direction,canonical_pts=bundle.stamps)
    print(json.dumps(dict(stage='full_source_cost_derivative_replayed',parts=records[-1]['parts'])),flush=True)


def bounded_motion_derivative_checks(bundle,q,minimum_depth_scene,output):
    """Check the actual near-zero edge and typed rows with exact physical J."""
    from z190_profiled_motion_schur import SchurMotionKernel
    kernel=SchurMotionKernel(minimum_source_depth_scene_units=minimum_depth_scene,schur=False)
    near=np.flatnonzero((np.asarray(bundle.stamps)[bundle.ia]==230914304)&(np.asarray(bundle.stamps)[bundle.ib]==230914816))
    selected=np.unique(np.r_[0,near[:8],near[-8:],[r['global_bundle_index']for r in bundle.finite_records]])
    ia,ib=bundle.ia[selected],bundle.ib[selected]
    def call(x,jac):
        v,r,d,c,s,w=bundle.cache(x,True)
        return kernel(a=bundle.a[selected],b=bundle.b[selected],fa=np.exp(v[ia,3]),fb=np.exp(v[ib,3]),ra=r[ia],rb=r[ib],da=d[ia],db=d[ib],
            ca=c[ia],cb=c[ib],dca_shared=s[ia],dcb_shared=s[ib],dca_view=w[ia],dcb_view=w[ib],
            pp=np.array([1920*bundle.calibration.principal_x_fraction[0],1080*bundle.calibration.principal_y_fraction[0]]),k1=bundle.calibration.k1[0],
            planes=bundle.finite_planes[selected],jacobian=jac,sigma=bundle.motion_sigma.reshape(-1,2)[selected],profile_weight=None)
    fixtures=[]
    for name in ('current_near_zero_baseline_KKT_crossing','finite_baseline_derivative_fixture'):
        x=q.copy()
        if name=='finite_baseline_derivative_fixture':x[4+4*bundle.index[230914304]]+=1e-5
        residual,ja,jb,js=call(x,True);checks=[]
        for col in np.r_[np.arange(4),np.concatenate([4+4*i+np.arange(4)for i in sorted(set(ia)|set(ib))])]:
            h=1e-8;plus=x.copy();minus=x.copy();plus[col]+=h;minus[col]-=h
            fd=(call(plus,False)-call(minus,False))/(2*h)
            if col<4:analytic=js[:,col]
            else:
                i,axis=divmod(col-4,4);analytic=ja[:,axis]*np.repeat(ia==i,2)+jb[:,axis]*np.repeat(ib==i,2)
            checks.append(dict(column=int(col),relative_error=float(np.linalg.norm(fd-analytic)/max(1.,np.linalg.norm(fd))),maximum_native_error=float(np.max(np.abs(fd-analytic)))))
        fixtures.append(dict(name=name,checks=checks,source_camera_state_exported=False))
    write(output/'actual_bounded_depth_derivative_checks.json',dict(original_bundle_rows=selected,minimum_source_camera_Z_depth_scene_units=minimum_depth_scene,
        fixtures=fixtures,near_zero_note='At near-zero baseline plus/minus perturbations cross finite-depth/infinity KKT branches; central J is not a unique derivative. Continuous lower-envelope objective replay is recorded separately.'))
    if max(r['relative_error']for r in fixtures[-1]['checks'])>5e-4:raise ValueError('Bounded actual source motion derivative mismatch away from KKT crossing')


def save_conditional_depth_trace(bundle,q,active_original,minimum_depth_scene,output):
    from z190_moving_center_motion import moving_center_motion_value_jacobians as kernel
    v,r,d,c,s,w=bundle.cache(q,False);pieces={}
    pp=np.array([1920*bundle.calibration.principal_x_fraction[0],1080*bundle.calibration.principal_y_fraction[0]])
    for lo in range(0,len(bundle.a),65536):
        hi=min(lo+65536,len(bundle.a));ia,ib=bundle.ia[lo:hi],bundle.ib[lo:hi];planes=bundle.finite_planes[lo:hi]
        result=kernel(bundle.a[lo:hi],bundle.b[lo:hi],np.exp(v[ia,3]),np.exp(v[ib,3]),r[ia],r[ib],None,None,c[ia],c[ib],None,None,None,None,
            principal_px=pp,k1=bundle.calibration.k1[0],sigma=bundle.motion_sigma.reshape(-1,2)[lo:hi],
            profile_weight=None if bundle.profile_weights is None else bundle.profile_weights[lo:hi],finite_normals=planes[:,:3],finite_d=-planes[:,3],finite_mask=bundle.finite_mask[lo:hi],
            jac=False,return_state=True,minimum_source_depth_scene_units=minimum_depth_scene)
        for key,value in result[-1].items():
            item=np.asarray(value)
            if item.ndim and len(item)==hi-lo:pieces.setdefault(key,[]).append(item)
    arrays={key:np.concatenate(item)for key,item in pieces.items()}
    arrays.update(original_full_low_bundle_row=active_original,source_canonical_pts=np.asarray(bundle.stamps)[bundle.ia],target_canonical_pts=np.asarray(bundle.stamps)[bundle.ib],
        original_source_uv=bundle.a,original_target_uv=bundle.b,original_sigma_xy=bundle.motion_sigma.reshape(-1,2))
    target=output/'conditional_original_row_depth_trace.npz';np.savez_compressed(target,**arrays)
    return dict(file=ref(target),original_rows=len(bundle.a),conditional_depth_floor_hits=int(arrays['conditional_depth_floor_boundary_rows'].sum()),
        depths_not_measured_or_accepted=True,minimum_source_camera_Z_depth_scene_units=minimum_depth_scene)


def run(a):
    out=a.output.resolve()
    if not out.is_relative_to(Path('D:/CodexData').resolve()):raise ValueError('Bulk output must stay in CodexData')
    out.mkdir(parents=True,exist_ok=False);sourceout=out/'original_source_adapter_inputs';sourceout.mkdir()
    cv2.setNumThreads(1);enable_projection_backend()
    current_path=getattr(a,'current',CURRENT_FULL).resolve()
    current_input=ref(current_path)
    if current_path==CURRENT_FULL.resolve() and current_input['sha256']!=CURRENT_SHA:raise ValueError('Current complete camera changed')
    original,_,_,_,native,aliases,_,allrows,historical=original_prepare(sourceout)
    current,_,allaliases,allviews=PhysicalCameraModel.load_npz(current_path)
    cal=current.calibrations[IDENTIFIER];views={p:v for p,v in allviews.items()if v.installation_id==IDENTIFIER}
    if {p:allaliases[int(p)]for p in native}!=aliases:raise ValueError('Source aliases changed')
    scene=read(Path(historical['scene']['path']))
    source_paths=[a.new_source,*getattr(a,'additional_source',[])]
    newrows=[];new_source_inputs=[]
    for source_path in source_paths:
        rr,packet=new_direct_rows(source_path,scene,aliases)
        newrows.extend(rr)
        manifest_source=read(Path(packet['source_manifest']));sourceimage=Path(manifest_source['source_png'])
        bgr=cv2.imread(str(sourceimage))
        import hashlib
        if bgr is None or hashlib.sha256(bgr.tobytes()).hexdigest()!=manifest_source['native_bgr_sha256']:raise ValueError('New source frame changed')
        new_source_inputs.append(dict(packet=ref(source_path),raw=ref(sourceimage),raw_manifest=ref(Path(packet['source_manifest']))))
    if any(int(r['source_native_pts'])!=230907648 or r['split']!='fit'for r in newrows):raise ValueError('Only new directly read 0.5second source admitted')
    admitted_samples=set()
    for row in newrows:
        for uv in row['uv']:
            identity=(int(row['source_native_pts']),tuple(np.asarray(uv,float)))
            if identity in admitted_samples:raise ValueError('Additional source repeats the same original native UV sample')
            admitted_samples.add(identity)
    stamps=[p for p in sorted(views)if 230898688<=p<231013376];index={p:i for i,p in enumerate(stamps)}
    old_index=np.asarray([original.index[p]for p in stamps],int)
    inside=np.zeros(len(original.stamps),bool);inside[old_index]=True
    active=np.flatnonzero(inside[original.ia]&inside[original.ib]);cross=inside[original.ia]^inside[original.ib]
    if cross.any():raise ValueError('Cut decomposition has source edges crossing cuts')
    if len(active)!=83453:raise ValueError('Entire original cut8 motion graph must remain')
    old_to_new=np.full(len(original.stamps),-1,int);old_to_new[old_index]=np.arange(len(stamps))
    observed={p:list(rr)for p,rr in original.abs_groups if p in index}
    for r in newrows:observed.setdefault(int(r['pts']),[]).append(r)
    groups=[(p,observed[p])for p in stamps if p in observed]
    ms=original.motion_sigma.reshape(-1,2)[active].ravel()
    abs_sig=np.concatenate([np.concatenate([np.repeat(r['sigma'],2)for r in rr])for p,rr in groups])
    base=SimpleNamespace(stamps=stamps,index=index,a=original.a[active],b=original.b[active],ia=old_to_new[original.ia[active]],ib=old_to_new[original.ib[active]],
        motion_sigma=ms,abs_groups=groups,native_residual_sigma=np.r_[ms,abs_sig])
    rawbinding=next(r for r in historical['source_motion']if r['cut']==8)
    planes,finite_records,finite_input=bind_source_finite_board(a.finite_source,rawbinding,base,scene,aliases)
    if a.schur_depth:
        from z190_profiled_motion_schur import SchurMotionKernel
        motion_kernel=SchurMotionKernel(minimum_source_depth_scene_units=a.minimum_unknown_depth_m/SCENE_UNIT_M)
    else:motion_kernel=moving_motion_wrapper
    bundle=FixedCutBundle(base,cal,motion_kernel,reference_pts=stamps[0],initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],
        finite_planes=planes,finite_records=finite_records,scene=scene)
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[views[p].view.rvec_world_to_camera,views[p].view.log_f_ratio+np.log(1920.)]for p in stamps])]
    continuation=a.resume_state or getattr(a,'export_state',None)
    if continuation:
        if not a.schur_depth:raise ValueError('Explicit checkpoint continuation is only for the same Schur-depth source recovery')
        with np.load(continuation,allow_pickle=False)as saved:
            if not np.array_equal(saved['canonical_pts'],stamps):raise ValueError('Resume cut exposure graph changed')
            savedq=saved['q'].copy()
        if savedq.shape!=q.shape or not np.array_equal(savedq[:4],q[:4]):raise ValueError('Resume fixed current installation changed')
        previous=read(continuation.parent/'input_manifest.json')
        for key,value in(('current_complete',current_input),('new_source',ref(a.new_source)),('finite_source_input',finite_input)):
            if previous[key]!=value:raise ValueError('Resume source changed:'+key)
        if previous.get('additional_source_inputs',[])!=new_source_inputs[1:]:raise ValueError('Resume additional source changed')
        if previous.get('unknown_motion_depth_closure',{}).get('minimum_source_depth_m',0.)!=a.minimum_unknown_depth_m:raise ValueError('Resume conditional depth closure changed')
        np.savez_compressed(out/'reference_current_v3_state4rf.npz',q=q,canonical_pts=stamps)
        q=savedq
    np.savez_compressed(out/'initial_current_state4rf.npz',q=q,canonical_pts=stamps)
    np.savez_compressed(out/'retained_cut8_source_indices.npz',original_full_low_motion_global_index=active,original_sigma_xy=ms.reshape(-1,2))
    manifest=dict(schema='rinkspace.current-low8-new-direct-Ad4-source-fixed-mount-Rf-recovery.v1',current_complete=current_input,new_source=ref(a.new_source),
        new_source_raw=new_source_inputs[0]['raw'],new_source_raw_manifest=new_source_inputs[0]['raw_manifest'],additional_source_inputs=new_source_inputs[1:],scene=historical['scene'],original_source_absolute=historical['source_authority_rows'],
        original_source_adapter=ref(sourceout/'input_manifest.json'),source_motion_original_cut8_rows=len(active),new_source_fit_rows=len(newrows),new_source_samples=sum(len(r['uv'])for r in newrows),
        new_source_actual_pts=230907648,new_source_native_pts=230907648,existing_plus1s_held230915328_not_fit=True,all_existing_fit_held_pending_roles_retained=True,
        only_free='all cut8 actual R3/effective f',fixed=['current pivotXYZ','current leverXYZ','PP','k1','hard93c','cut3/5 actual R/f'],actual_cut8_exposures=len(stamps),
        unknown_motion_positive_depth_rows=len(active)-len(finite_records),finite_source_motion_rows=len(finite_records),finite_source_input=finite_input,
        no_unreviewed_motion_plane_assignment=True,no_residual_or_confidence_row_selection=True,original_sigma_unchanged=True,
        reference_lens_height_prior='original reference belongs to unchangedcut3; constant during cut8solve; no newcut8heightprior',
        original_per_actual_roll_prior_unchanged=True,entrypoint=ref(__file__),accuracy_accepted=False,video_generated=False)
    manifest['motion_kernel_implementation']=ref(Path(__file__).with_name('z190_moving_center_motion.py'))
    manifest['source_motion_GN_adapter']=ref(Path(__file__).with_name('z190_profiled_motion_schur.py'))
    manifest['source_motion_GN']='raw GN one-depth Schur for unknown interiors; finite planes/positive-depth endpoints unchanged'if a.schur_depth else'exact-profile-residual Jacobian GN'
    if a.minimum_unknown_depth_m:
        if not a.schur_depth:raise ValueError('Conditional depth closure requires tested Schur branch')
        manifest['unknown_motion_depth_closure']=dict(minimum_source_depth_m=a.minimum_unknown_depth_m,minimum_source_depth_scene_units=a.minimum_unknown_depth_m/SCENE_UNIT_M,
            depth_semantics='source camera Z along undistorted [x,y,1] ray; not Euclidean distance',
            range_is_engineering_not_measurement=True,finite_material_rows_unchanged=True,no_plane_or_world_identity_inferred=True,
            reason='Exclude arbitrarily near optical-centre conditional points; finite inverse depth makes near-zero-baseline profile continuous. This bound is a disclosed weak engineering condition, not a measured background depth.',
            source_weights_or_rows_changed=False)
    if continuation:manifest['continuation']=dict(saved_accepted_state=ref(continuation),original_input_manifest=ref(continuation.parent/'input_manifest.json'),
        original_iteration_history=ref(continuation.parent/'bundle_iterations.json'),reason=getattr(a,'continuation_reason',None)or'linear stage reached numerical profile/LM stagnation with no pixel update; continue Huber from preserved accepted q, not rewarm',phases=['export_accepted_without_further_fit']if a.export_state else['huber_source'])
    write(out/'input_manifest.json',manifest)
    write(out/'finite_motion_source_binding.json',finite_records)
    checks=derivative_checks(bundle,q);write(out/'actual_source_derivative_checks.json',checks)
    if a.schur_depth:
        # Objective/Jphysical replay remains the tested exact kernel; this
        # validates the equivalent local joint camera/depth GN information.
        value,_=bundle.evaluate(q)
        exact_kernel=SchurMotionKernel(minimum_source_depth_scene_units=a.minimum_unknown_depth_m/SCENE_UNIT_M,schur=False)if a.minimum_unknown_depth_m else moving_motion_wrapper
        exact=FixedCutBundle(base,cal,exact_kernel,reference_pts=stamps[0],initial_reference_optical_height_scene=current.camera25(views[stamps[0]])[2],
            finite_planes=planes,finite_records=finite_records,scene=scene)
        exact_value=exact.evaluate(q,False)
        if not np.array_equal(value,exact_value):raise ValueError('Schur must preserve exact original source objective')
        write(out/'source_depth_schur_replay_checks.json',dict(objective_native_bitexact=True,chunks=motion_kernel.latest_checks,source_sigma_unchanged=True,exact_physical_derivative_checks=ref(out/'actual_source_derivative_checks.json')))
        del exact
    if a.minimum_unknown_depth_m:bounded_motion_derivative_checks(bundle,q,a.minimum_unknown_depth_m/SCENE_UNIT_M,out)
    print(json.dumps(dict(stage='current_low8_direct_source_rf_prepared',actual=len(stamps),motion=len(active),fit_rows=sum(len(rr)for p,rr in groups),derivative_relative=max(r['relative_error']for r in checks))),flush=True)
    del original,base;gc.collect()
    if getattr(a,'check_objective',False):objective_direction_check(bundle,q,out)
    if not a.solve:return
    if a.export_state:
        fitted=q;history=read(a.export_state.parent/'bundle_iterations.json');seconds=history[-1]['elapsed_seconds']
        write(out/'bundle_iterations.json',history)
    else:fitted,history,seconds=fixed_mount_rf_solve(bundle,q,out,require_undamped_precision=a.schur_depth,phases=(True,)if a.resume_state else(False,True))
    newviews=dict(views)
    for p in stamps:newviews[p]=CanonicalView(IDENTIFIER,bundle.view(fitted,p))
    settings=dict(cal.field_settings);settings['direct_source_rf_recovery']=dict(input_manifest=ref(out/'input_manifest.json'),new_source=ref(a.new_source),additional_source_inputs=new_source_inputs[1:],
        fixed_current_installation=True,original_other_cuts_unchanged=True,actual_low8_only=True,accuracy_accepted=False)
    if a.minimum_unknown_depth_m:settings['direct_source_rf_recovery']['unknown_motion_depth_closure']=manifest['unknown_motion_depth_closure']
    model=PhysicalCameraModel({IDENTIFIER:replace(cal,field_settings=settings)})
    path=out/'physical_low8_direct_source_rf_candidate_unaccepted.npz'
    model.save_npz(path,native,aliases,newviews,source_sha256=SOURCE_SHA,time_base='1/15360')
    loaded,pts2,aliases2,views2=PhysicalCameraModel.load_npz(path)
    if not np.array_equal(pts2,native)or aliases2!=aliases:raise ValueError('Low native inventory changed')
    unchanged=[p for p in views if p not in index]
    if not all(np.array_equal(current.camera25(views[p]),loaded.camera25(views2[p]))for p in unchanged):raise ValueError('Other low cuts must remain bitexact')
    changes=source_changes(allrows+[('new_direct_source_fit',r)for r in newrows],current,allviews,loaded,views2,aliases)
    write(out/'all_absolute_source_changes.json',changes);np.savez_compressed(out/'optimized_state4rf.npz',q=fitted,canonical_pts=stamps)
    depth_trace=save_conditional_depth_trace(bundle,fitted,active,a.minimum_unknown_depth_m/SCENE_UNIT_M,out)if a.minimum_unknown_depth_m else None
    write(out/'report.json',dict(manifest,candidate=ref(path),history=history,seconds=seconds,source_changes=ref(out/'all_absolute_source_changes.json'),other_low_cuts_numeric_bitexact=True,
        conditional_original_motion_depth_trace=depth_trace,disposition='actual_source_review_pending_not_adopted',accuracy_accepted=False))
    print(json.dumps(dict(stage='current_low8_direct_source_rf_exported',candidate=str(path),seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--current',type=Path,default=CURRENT_FULL)
    p.add_argument('--new-source',type=Path,default=CURRENT/'low8_mid_source_v1/source_observations.json')
    p.add_argument('--additional-source',type=Path,action='append',default=[])
    p.add_argument('--finite-source',type=Path)
    p.add_argument('--schur-depth',action='store_true')
    p.add_argument('--check-objective',action='store_true')
    p.add_argument('--minimum-unknown-depth-m',type=float,default=0.)
    continuation=p.add_mutually_exclusive_group()
    continuation.add_argument('--resume-state',type=Path)
    continuation.add_argument('--export-state',type=Path)
    p.add_argument('--continuation-reason')
    p.add_argument('--output',type=Path,default=CURRENT/'z190_low8_direct_source_rf_v1');p.add_argument('--solve',action='store_true')
    run(p.parse_args())
