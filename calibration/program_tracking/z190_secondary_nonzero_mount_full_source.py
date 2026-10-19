"""Current secondary installation: shared rigid mount and complete motion.

The user-fixed pivot XY, current constant optics, source UV, geometry and
source-authority roles remain unchanged. Every original motion correspondence
uses moving-centre positive-depth profiling; two source-reviewed ice rows use
the hard z=0 surface. Held/pending world identities never enter the fit.
"""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from collections import Counter
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from z190_physical_camera import PhysicalCameraModel,CanonicalView
from z190_continuous_source_bundle import SCENE_UNIT_M,HARD_SCENE_SHA,SOURCE_SHA
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle,solve_moving_bundle
from z190_main_nonzero_mount_full_source import moving_motion_wrapper,derivative_checks,save_depth_trace,FACTS,FACTS_SHA
from z190_source_camera_diagnostic import source_errors,jsonable,absolute_value_jac15
from continuous_central_brown4_optin import enable_projection_backend
from z190_source_factor_admission import active_fit_rows

CURRENT=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00009/RinkSpace_tmp')
OLD=Path('D:/CodexData/.codex/visualizations/2026/10/01/private-thread-00010/RinkSpace_tmp')
PHYSICAL=CURRENT/'z190_full_programme_pelvis_source_v6/z190_full_programme_camera_candidate.npz'
AUTHORITY=CURRENT/'z190_secondary_C_line_source_admission_v1/source_authority_rows.json'
RAYS=OLD/'secondary/absolute_ray_graph_v6/source_ray_matches.npz'
FINITE=CURRENT/'z190_finite_motion_source_inventory_v1/secondary_fullray_2_ice_source_admission.json'
SCENE=OLD/'render_markings/main7_geometry/source_seen_row3_carrier_v1/candidate_scene.json'
IDENTIFIER='okada_sp_installation_1_conditional_z190'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def write(p,r):Path(p).write_text(json.dumps(jsonable(r),indent=2,ensure_ascii=False),encoding='utf-8')
def convert(r):
    r=dict(r)
    for k in ('world','uv','sigma','normal','normal_interval'):
        if k in r:r[k]=np.asarray(r[k],float)
    if sha_uv(r)!=r['source_uv_float64_sha256']:raise ValueError('Original source UV changed')
    return r
def sha_uv(r):return hashlib.sha256(np.asarray(r['uv'],np.float64).tobytes()).hexdigest()

def bounded_derivative_checks(bundle,q):
    """Actual-source FD at the saved state, without crossing user lever bounds."""
    selected=np.array([0]+[r['global_bundle_index']for r in bundle.finite_records],int)
    ia,ib=bundle.ia[selected],bundle.ib[selected]
    def call(x,jac):
        v,r,d,c,s,w=bundle.cache(x,True)
        return moving_motion_wrapper(a=bundle.a[selected],b=bundle.b[selected],fa=np.exp(v[ia,3]),fb=np.exp(v[ib,3]),
            ra=r[ia],rb=r[ib],da=d[ia],db=d[ib],ca=c[ia],cb=c[ib],dca_shared=s[ia],dcb_shared=s[ib],
            dca_view=w[ia],dcb_view=w[ib],pp=np.array([1920*bundle.calibration.principal_x_fraction[0],
            1080*bundle.calibration.principal_y_fraction[0]]),k1=bundle.calibration.k1[0],planes=bundle.finite_planes[selected],
            jacobian=jac,sigma=np.repeat(bundle.motion_sigma[2*selected,None],2,axis=1),profile_weight=None)
    def fd(function,col):
        h=1e-7
        if col<4 and q[col]+h>bundle.shared_upper[col]:
            x,y=q.copy(),q.copy();x[col]-=h;y[col]-=2*h
            return (3*function(q)-4*function(x)+function(y))/(2*h),'backward_second_order_at_user_upper_bound'
        if col<4 and q[col]-h<bundle.shared_lower[col]:
            x,y=q.copy(),q.copy();x[col]+=h;y[col]+=2*h
            return (-3*function(q)+4*function(x)-function(y))/(2*h),'forward_second_order_at_user_lower_bound'
        x,y=q.copy(),q.copy();x[col]+=h;y[col]-=h
        return (function(x)-function(y))/(2*h),'central'
    _,ja,jb,js=call(q,True)[:4];checks=[];columns=list(range(4))
    for i in sorted(set(ia.tolist()+ib.tolist())):columns.extend((4+4*i+np.arange(4)).tolist())
    for col in columns:
        numeric,mode=fd(lambda x:call(x,False),col)
        analytic=js[:,col].copy()if col<4 else np.zeros(len(numeric))
        if col>=4:
            i,axis=(col-4)//4,(col-4)%4
            analytic+=ja[:,axis]*np.repeat(ia==i,2)+jb[:,axis]*np.repeat(ib==i,2)
        checks.append(dict(kind='actual_motion',column=col,finite_difference=mode,
            relative_error=float(np.linalg.norm(numeric-analytic)/max(1.,np.linalg.norm(numeric))),
            max_native_error=float(np.max(np.abs(numeric-analytic)))))
    scale=np.array([.01]*3+[1.]*5+[.001]*3+[1.]*4)
    for p,rows in bundle.abs_groups[::max(1,len(bundle.abs_groups)//4)]:
        cam,sj,vj=bundle.current_calibration(q).camera25_and_jacobians(bundle.view(q,p));_,raw=absolute_value_jac15(cam,rows,True)
        analytic=raw@(np.c_[sj[:15,bundle.mapping_names],vj[:15]]*scale[:,None])
        cols=list(range(4))+list(4+4*bundle.index[p]+np.arange(4))
        for local,col in enumerate(cols):
            numeric,mode=fd(lambda x:absolute_value_jac15(bundle.camera(x,p),rows,False),col)
            checks.append(dict(kind='actual_absolute',pts=p,column=col,finite_difference=mode,
                relative_error=float(np.linalg.norm(numeric-analytic[:,local])/max(1.,np.linalg.norm(numeric))),
                max_sigma_error=float(np.max(np.abs(numeric-analytic[:,local])))))
    if max(x['relative_error']for x in checks)>5e-4:raise ValueError('Actual moving-C derivative mismatch:'+json.dumps(checks))
    return checks

def prepare(output, physical_path=PHYSICAL, authority_path=AUTHORITY, *, warm_current_mount=False):
    if sha(SCENE)!=HARD_SCENE_SHA or sha(FACTS)!=FACTS_SHA:raise ValueError('Frozen scene/latest user hardware changed')
    model,native,all_aliases,all_views=PhysicalCameraModel.load_npz(physical_path)
    views={p:v for p,v in all_views.items()if v.installation_id==IDENTIFIER}
    native=np.array([p for p in native if all_views[all_aliases[int(p)]].installation_id==IDENTIFIER],np.int64)
    aliases={int(p):all_aliases[int(p)]for p in native};stamps=sorted(views);index={p:i for i,p in enumerate(stamps)}
    if len(stamps)!=1061:raise ValueError('Secondary actual exposure inventory changed')
    original_cal=model.calibrations[IDENTIFIER];packet=read(authority_path)
    admission=packet['source_authority_admission']
    if sha(admission['path'])!=admission['sha256']:raise ValueError('Authority admission changed')
    record=read(admission['path']);original=read(record['original_packet']['path'])
    if sha(record['original_packet']['path'])!=record['original_packet']['sha256']:raise ValueError('Original rows changed')
    if record['scene_sha256']!=HARD_SCENE_SHA or record['source_sha256']!=SOURCE_SHA:raise ValueError('Authority source/scene changed')
    if [len(packet[k])for k in ('fit_rows','held_rows','pending_rows')]!=[268,35,4]:raise ValueError('Authoritative split inventory changed')
    if original['held_rows']!=packet['held_rows']:raise ValueError('Original held changed')
    numeric=lambda r:json.dumps({k:r.get(k)for k in ('pts','id','world','uv','sigma','point_correspondence','normal','normal_interval')},sort_keys=True)
    if Counter(map(numeric,original['fit_rows']))!=Counter(map(numeric,packet['fit_rows']+packet['pending_rows'])):raise ValueError('Raw original geometry/UV/multiplicity changed')
    rows={k:[convert(r)for r in packet[k]]for k in ('fit_rows','held_rows','pending_rows')}
    canonical_fit,factor_admission=active_fit_rows(packet,expected_scene_sha=HARD_SCENE_SHA,expected_source_sha=SOURCE_SHA)
    groups={};active_lookup={}
    active_raw_indices=[r['raw_fit_row_index'] for r in factor_admission.get('original_fit_row_to_factor',[]) if r['factor_role']=='active_canonical']
    if not active_raw_indices:active_raw_indices=list(range(len(canonical_fit)))
    for factor_index,raw in enumerate(canonical_fit):
        r=convert(raw)
        if r['split']!='fit':raise ValueError('Pending/held may not enter fit')
        p=aliases[int(r['pts'])]
        groups.setdefault(p,[]).append(r)
        active_lookup[id(r)]=(factor_index,active_raw_indices[factor_index])
    abs_groups=[(p,groups[p])for p in stamps if p in groups]
    inventory=[];absolute_point_offset=0
    for p,rr in abs_groups:
        for r in rr:
            factor_index,raw_index=active_lookup[id(r)];points=len(r['uv'])
            inventory.append(dict(objective_factor_index=len(inventory),active_factor_index=factor_index,
                canonical_raw_fit_row_index=raw_index,native_pts=int(r['pts']),canonical_pts=int(p),
                id=r['id'],points=points,absolute_point_offset=absolute_point_offset,
                source_uv_float64_sha256=sha_uv(r),world_float64_sha256=hashlib.sha256(np.asarray(r['world'],np.float64).tobytes()).hexdigest(),
                original_sigma=r['sigma'],origin=r.get('origin')))
            absolute_point_offset+=points
    if len(inventory)!=len(canonical_fit) or len({r['active_factor_index'] for r in inventory})!=len(canonical_fit):
        raise ValueError('Actual absolute objective changed canonical factor multiplicity')
    write(output/'actual_active_absolute_factors.json',dict(schema='rinkspace.secondary-actual-absolute-factor-inventory.v1',
        source_authority_binding=ref(authority_path),raw_fit_rows=len(rows['fit_rows']),active_fit_factors=len(inventory),
        held_rows=len(rows['held_rows']),pending_rows=len(rows['pending_rows']),absolute_source_exposures=len(abs_groups),
        absolute_points=absolute_point_offset,source_factor_admission=packet.get('source_factor_admission'),factors=inventory))
    with np.load(RAYS)as z:
        pairs=z['pair_pts'];off=z['offset'];a=z['source_uv'].astype(float);b=z['target_uv'].astype(float)
    counts=np.diff(off)
    if off[0]!=0 or off[-1]!=len(a) or len(a)!=39679 or len(pairs)!=1105:raise ValueError('Original full secondary motion inventory changed')
    if any(aliases[int(p)]!=int(p)for p in pairs.ravel()):raise ValueError('Original motion source canonical direction changed')
    ia=np.repeat([index[int(p)]for p in pairs[:,0]],counts);ib=np.repeat([index[int(p)]for p in pairs[:,1]],counts)
    sigma=np.full(2*len(a),2.);abs_sigma=np.concatenate([np.repeat(r['sigma'],2)for _,rr in abs_groups for r in rr])
    base=SimpleNamespace(stamps=stamps,index=index,a=a,b=b,ia=ia,ib=ib,motion_sigma=sigma,abs_groups=abs_groups,native_residual_sigma=np.r_[sigma,abs_sigma])
    reference=aliases[229497600]
    lever=np.asarray(original_cal.lever_xyz) if warm_current_mount else np.array([0.,-.20,.10])/SCENE_UNIT_M
    rotation=cv2.Rodrigues(np.array(views[reference].view.rvec_world_to_camera))[0]
    old_cz=model.camera25(views[reference])[2]
    pz=original_cal.pivot_xyz[2] if warm_current_mount else old_cz-(rotation.T@lever)[2]
    settings=dict(original_cal.field_settings)
    settings['source_authority_admission']=dict(admission,unconfirmed_ice_proposals_excluded_from_fit=True)
    settings['source_authority_binding']=ref(authority_path)
    settings['active_source_factor_policy']=dict(raw_fit_rows=len(rows['fit_rows']),active_fit_factors=len(canonical_fit),
        held_rows=len(rows['held_rows']),pending_rows=len(rows['pending_rows']),absolute_source_exposures=len(abs_groups),
        fit_selection_field='active_fit_rows' if packet.get('source_factor_admission') else 'fit_rows',
        actual_objective_factor_inventory=ref(output/'actual_active_absolute_factors.json'),held_or_pending_fitted=False)
    settings['active_motion_model']='fixed_tripod_pivot_shared_constant_nonzero_camera_local_lever'
    settings['active_mount_parameters']=dict(source=ref(FACTS),initial_camera_xyz_m=(lever*SCENE_UNIT_M).tolist(),reference_exposure_pts=reference,fixed_pivot_xy_scene_units=original_cal.pivot_xyz[:2],pivot_initialization='saved current nonzero pivot/lever' if warm_current_mount else 'old reference optical Cz-(R_reference.T@lever)_z',per_frame_unrestricted_C=False,offset_zoom_dependent=False,independent_installation=True)
    if packet.get('source_factor_admission'):
        settings['source_factor_admission']=packet['source_factor_admission']
    settings['reference_optical_height_prior']=dict(reference_exposure_pts=reference,target_above_ice_working_m=1.6,sigma_m=.18,operator_ground_minus_ice_unknown=True,pivot_height_is_not_lens_height=True,counted_once_per_installation=True)
    cal=replace(original_cal,pivot_xyz=(*original_cal.pivot_xyz[:2],float(pz)),lever_xyz=tuple(lever),axial=(0.,),field_settings=settings)
    finite=read(FINITE);planes=np.zeros((len(a),4));records=[]
    if finite['original_motion_graph']['sha256']!=sha(RAYS):raise ValueError('Typed exact source graph changed')
    for r in finite['rows']:
        pair=r['original_pair_index'];loc=r['original_local_index'];j=int(off[pair])+loc
        if j!=r['original_global_index']or pairs[pair].tolist()!=r['source_native_pair_pts']or not np.array_equal(a[j],r['xy0'])or not np.array_equal(b[j],r['xy1']):raise ValueError('Typed original row/direction/UV changed')
        if not r['both_end_original_source_reviewed']or r['finite_surface']['kind']!='ice_plane':raise ValueError('Finite material not both-end source qualified')
        planes[j]=[0.,0.,1.,0.]
        records.append(dict(global_bundle_index=j,original_pair_index=pair,original_local_index=loc,original_global_index=j,source_native_pair_pts=r['source_native_pair_pts'],xy0=r['xy0'],xy1=r['xy1'],surface_kind='ice_plane',material_identity=r['static_texture_identity'],original_source_role=r['split'],review=ref(FINITE),admission='latest user authorized real material finite motion',world_circle_identity='unassigned_not_needed_for_ice_plane'))
    bundle=MovingPivotSourceBundle(base,cal,moving_motion_wrapper,reference_pts=reference,initial_reference_optical_height_scene=old_cz,finite_planes=planes,finite_records=records,scene=read(SCENE))
    q=np.r_[bundle.initial_shared(),np.concatenate([np.r_[views[p].view.rvec_world_to_camera,views[p].view.log_f_ratio+np.log(1920.)]for p in stamps])]
    if abs(bundle.camera(q,reference)[2]-old_cz)>1e-12:raise ValueError('Lever initial optical height double-counted')
    warm_delta=max(float(np.max(np.abs(bundle.camera(q,p)-model.camera25(views[p])))) for p in stamps) if warm_current_mount else None
    if warm_current_mount and warm_delta>1e-12:raise ValueError('Warm current nonzero camera changed at initialization')
    manifest=dict(schema='rinkspace.current-secondary-nonzero-shared-rigid-mount-full-source.v1',current=ref(physical_path),source_authority=ref(authority_path),source_authority_admission=admission,source_factor_admission=factor_admission,original_rays=ref(RAYS),finite_source=ref(FINITE),scene=ref(SCENE),source_sha256=SOURCE_SHA,time_base='1/15360',user_mount_height_facts=ref(FACTS),fit_rows=268,active_canonical_fit_factors=len(canonical_fit),active_absolute_source_exposures=len(abs_groups),held_rows=35,pending_rows=4,held_source_exposures=len({r['pts']for r in rows['held_rows']}),canonical_exposures=len(stamps),native_presentations=len(native),motion_pairs=len(pairs),motion_rows=len(a),finite_motion_rows=len(records),unknown_motion_rows=len(a)-len(records),initial_pivot_height_m=float(pz*SCENE_UNIT_M),initial_reference_optical_height_m=float(old_cz*SCENE_UNIT_M),initial_lever_xyz_m=(lever*SCENE_UNIT_M).tolist(),warm_current_nonzero_mount=warm_current_mount,warm_camera_max_abs_delta=warm_delta,lever_bounds_m=bundle.lever_bounds_m,original_source_uv_and_splits_preserved=True,held_or_pending_not_fit=True,all_original_motion_rows_retained=True,fixed=['pivot XY','constant PP/k1','hard93c scene','zoom dependence zero','skew/k2/tangential'],active=['installation Pz','shared constant lever3','all1061 actual RF'],source_pixel_loss='original sigma scalar Huber2, inner depth uses frozen per-component IRLS',unknown_motion='positive-depth profile with explicit one-sided limits, conditional static material unknown',accuracy_accepted=False,production_pointer_changed=False,video_generated=False,entrypoint=ref(__file__),motion_kernel=ref(Path(__file__).with_name('z190_moving_center_motion.py')))
    write(output/'input_manifest.json',manifest);write(output/'qualified_finite_source_rows.json',records)
    np.savez_compressed(output/'initial_state4rf.npz',q=q,canonical_pts=stamps)
    return bundle,q,model,views,native,aliases,rows,manifest

def changes(model,views,aliases,rows):
    out=[]
    for role,rr in rows.items():
        for r in rr:
            p=aliases[int(r['pts'])]
            out.append(dict(native_pts=r['pts'],canonical_pts=p,id=r['id'],origin=r.get('origin'),source_method=r.get('source_method'),source_identity_status=r.get('source_identity_status',r.get('identity_status')),role=role,split=r['split'],errors=source_errors(model.camera25(views[p]),[r]),independent_physical_truth=False if role=='pending_rows' or r.get('conditional_identity')else None))
    return out

def run(args):
    args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);enable_projection_backend()
    bundle,q,old,views,native,aliases,rows,manifest=prepare(args.output,args.physical,args.authority,warm_current_mount=args.warm_current_mount)
    write(args.output/'all_absolute_source_baseline.json',changes(old,views,aliases,rows))
    checks=bounded_derivative_checks(bundle,q);write(args.output/'actual_source_derivative_checks.json',checks)
    print(json.dumps({'stage':'secondary_nonzero_mount_prepared','max_derivative_relative':max(r['relative_error']for r in checks),'motion':len(bundle.a),'actual_exposures':len(bundle.stamps)}),flush=True)
    if not args.solve:return
    fitted,history,seconds=solve_moving_bundle(bundle,q,args.output)
    np.savez_compressed(args.output/'optimized_state4rf.npz',q=fitted,canonical_pts=bundle.stamps)
    cal=bundle.current_calibration(fitted);settings=dict(cal.field_settings)
    settings['active_mount_parameters']=dict(settings['active_mount_parameters'],fitted_camera_xyz_m=(np.array(cal.lever_xyz)*SCENE_UNIT_M).tolist(),fitted_pivot_xyz_scene_units=cal.pivot_xyz)
    cal=replace(cal,field_settings=settings);model=PhysicalCameraModel({IDENTIFIER:cal});final_views={p:CanonicalView(IDENTIFIER,bundle.view(fitted,p))for p in bundle.stamps}
    physical=args.output/'physical_secondary_nonzero_mount_candidate_unaccepted.npz'
    model.save_npz(physical,native,aliases,final_views,source_sha256=SOURCE_SHA,time_base='1/15360')
    saved,loaded_native,loaded_aliases,loaded_views=PhysicalCameraModel.load_npz(physical)
    if not np.array_equal(native,loaded_native)or aliases!=loaded_aliases or not all(np.array_equal(saved.camera25(loaded_views[p]),bundle.camera(fitted,p))for p in bundle.stamps):raise ValueError('Physical value/alias replay failed')
    np.savez_compressed(args.output/'candidate_cameras.npz',**saved.exposure_arrays(native,aliases,loaded_views),model_sha256=HARD_SCENE_SHA,source_sha256=SOURCE_SHA)
    write(args.output/'all_absolute_source_changes.json',changes(saved,loaded_views,aliases,rows))
    trace=save_depth_trace(bundle,fitted,args.output)
    with np.load(trace['file']['path'])as z:
        statuses,counts=np.unique(z['profile_status'],return_counts=True);trace['statuses']={str(s):int(n)for s,n in zip(statuses,counts)}
        trace['unconverged_original_motion_indices']=np.flatnonzero(~z['profile_converged_rows']).tolist()
    centres=np.array([saved.camera25(loaded_views[p])[:3]for p in bundle.stamps])*SCENE_UNIT_M
    report=dict(manifest,solve_seconds=seconds,history=history,physical_candidate=ref(physical),latent_depth_trace=trace,fitted_pivot_xyz_scene_units=cal.pivot_xyz,fitted_pivot_height_m=cal.pivot_xyz[2]*SCENE_UNIT_M,fitted_lever_xyz_m=(np.array(cal.lever_xyz)*SCENE_UNIT_M).tolist(),optical_height_range_m=[float(centres[:,2].min()),float(centres[:,2].max())],fitted_reference_optical_height_m=bundle.camera(fitted,bundle.reference_pts)[2]*SCENE_UNIT_M,all_absolute_source_changes=ref(args.output/'all_absolute_source_changes.json'),source_aliases_bitexact=True,disposition='source review pending, unaccepted')
    write(args.output/'report.json',report)
    print(json.dumps({'stage':'secondary_nonzero_mount_complete','physical':str(physical),'seconds':seconds,'lever_m':report['fitted_lever_xyz_m']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=CURRENT/'z190_secondary_nonzero_mount_full_source_v1');parser.add_argument('--solve',action='store_true');parser.add_argument('--authority',type=Path,default=AUTHORITY);parser.add_argument('--physical',type=Path,default=PHYSICAL);parser.add_argument('--warm-current-mount',action='store_true');run(parser.parse_args())
