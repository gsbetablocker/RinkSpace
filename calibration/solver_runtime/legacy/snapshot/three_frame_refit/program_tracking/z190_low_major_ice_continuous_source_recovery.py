"""Restore the complete low camera from the user's F1/Lane1 source sequence.

The previous joint state's four dimensions stay fixed. One shared row
midpoint, one low Pz and all 1851 actual R/f exposures are free. Original
motion, finite-material admission, source roles, sigmas and optical mechanics
are retained. This is an unaccepted component, never a default programme.
"""
from __future__ import annotations
import argparse,io,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from z190_joint_shared_F_height_source import N,SCENE
from z190_low_shared_F_scene_rf_recovery import prepare as prepare_low,native_check,fingerprint
from z190_low_opening_rf_refine import convert_row
from z190_moving_pivot_source_bundle import MovingPivotSourceBundle
from z190_main_zoom_k1_source_information import read,write,ref
from z190_source_camera_diagnostic import absolute_value_jac15
from z190_continuous_source_bundle import SCENE_UNIT_M,SOURCE_SHA,HARD_SCENE_SHA
from shared_F_scene_candidate import rebind
from shared_rink_numeric_geometry_midpoint import SharedRinkNumericGeometryMidpoint
from z190_joint_source_solver import solve_joint_source,make_profile_update
from continuous_central_brown4_optin import enable_projection_backend

WARM=N/'joint_full_source_shared_rink_dimensions_v1/latest_accepted_state.npz'
MAJOR=N/'low5_repair/user_major_F1_Lane1_source_v3/source_rows.json'
ADS=N/'low5_repair/ad20_source_intervals_v1/source_rows.json'
INDEPENDENT=N/'ice_alignment_review/low5_user_held_support_v1/independent_user_held_source_support_QUALIFIED_REPORT_ONLY.json'


class ContinuousLowMidpointBlock:
    def __init__(self,bundle,local_q,geometry,fixed_geometry):
        self.bundle,self.local_q,self.geometry=bundle,local_q,geometry
        self.fixed_geometry=np.asarray(fixed_geometry,float).copy()
        self.groups=bundle.abs_groups

    def local(self,q):
        local=self.local_q.copy();local[0]=q[1]/(100.*SCENE_UNIT_M);local[4:]=q[2:]
        return local

    def geometry_q(self,q):return np.r_[self.fixed_geometry,q[0]]

    def evaluate(self,q,jacobian=True):
        local=self.local(q);geo=self.geometry_q(q)
        self.bundle.abs_groups=[(p,[self.geometry.rebind(r,geo)for r in rows])for p,rows in self.groups]
        try:
            if jacobian:values,oldj=self.bundle.evaluate(local,True)
            else:values=self.bundle.evaluate(local,False)
            midpoint,mj=self.geometry.midpoint_prior(geo)
            if not jacobian:return np.r_[values,midpoint]
            # Source branch operators stay exactly as before; only the one
            # absolute-world Y column is differentiated here. Motion is not
            # silently converted into a geometry or ice-depth observation.
            count=sum(2*len(r['uv'])for _,rows in self.groups for r in rows)
            derivative=np.zeros(count);cursor=0;h=1e-5
            for p,rows in self.groups:
                camera=self.bundle.camera(local,p)
                for row in rows:
                    size=2*len(row['uv'])
                    if self.geometry.feature_id(row)is not None:
                        plus,minus=geo.copy(),geo.copy();plus[4]+=h;minus[4]-=h
                        derivative[cursor:cursor+size]=(absolute_value_jac15(camera,[self.geometry.rebind(row,plus)],False)-absolute_value_jac15(camera,[self.geometry.rebind(row,minus)],False))/(2*h)
                    cursor+=size
            column=np.r_[np.zeros(self.bundle.motion_rows),derivative,np.zeros(len(values)-self.bundle.motion_rows-count)]
            jac=hstack((csr_matrix(column[:,None]),oldj[:,0]/(100.*SCENE_UNIT_M),oldj[:,4:]),format='csr')
            last=csr_matrix(([mj[0,4]],([0],[0])),shape=(1,len(q)))
            return np.r_[values,midpoint],vstack((jac,last),format='csr')
        finally:self.bundle.abs_groups=self.groups

    def records(self,q,groups):
        local=self.local(q);geo=self.geometry_q(q);out=[]
        for p,rows in groups:
            camera=self.bundle.camera(local,p)
            for row in rows:
                error=absolute_value_jac15(camera,[self.geometry.rebind(row,geo)],False).reshape(-1,2)*row['sigma'][:,None]
                out.append(dict(actual_pts=int(p),source_native_pts=int(row.get('source_native_pts',row['pts'])),id=row['id'],split=row['split'],source_role=row.get('source_role'),outside_support_xy_px=error.tolist(),median_outside_support_px=float(np.median(np.linalg.norm(error,axis=1))),maximum_outside_support_px=float(np.max(np.linalg.norm(error,axis=1)))))
        return out


def independent_rows(prepared,path):
    """Attach strict base world to frozen report-only native held intervals."""
    packet=read(path);features={r['id']:r for r in prepared.scene['ice_features']};groups={}
    for source in packet['records']:
        if source['fit_admitted']is not False or source['split']!='held':raise ValueError('Independent source must never be fit')
        name=source['physical_curve_id']
        if name=='F1_outer':
            feature=features['F1'];angles=np.linspace(0,2*np.pi,181)
            world=np.asarray(feature['center_xyz'])+float(feature['radius'])*np.c_[np.cos(angles),np.sin(angles),np.zeros(len(angles))]
        elif name=='lane1_edge0':
            polygon=np.asarray(features['lane1']['polygon_xyz']);world=np.linspace(polygon[0],polygon[1],65)
        else:raise ValueError('Independent held source has unsupported physical identity')
        y=float(source['source_y_q05_q50_q95'][1]);lo,hi=source['source_broad_y_support_px'];p=prepared.aliases[int(source['source_native_pts'])]
        row=convert_row(dict(source,pts=p,world=world,uv=[[source['source_x'],y]],sigma=[1.],normal=[[0.,1.]],normal_interval=[[lo-y,hi-y]],canonical_physical_geometry_id=name,physical_curve_id_authority=ref(path),curve_observation_kind='axis_scanline_branch',scanline_fixed_axis='x',scanline_branch='min_y',source_branch_operator_authority=ref(path)))
        groups.setdefault(p,[]).append(row)
    return sorted(groups.items())


def prepare(output,warm=WARM,major=MAJOR,ads=ADS,independent=INDEPENDENT,*,temporal_source=None):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    if not out.is_relative_to(Path('D:/CodexData').resolve()):raise ValueError('Bulk diagnostics must stay in CodexData')
    cv2.setNumThreads(1);enable_projection_backend()
    frozen=Path(warm).read_bytes();(out/'warm_joint_state_frozen.npz').write_bytes(frozen)
    with np.load(io.BytesIO(frozen),allow_pickle=False)as z:
        warmq=z['q'].copy();starts=z['aux__block_start'].copy();lengths=z['aux__block_lengths'].copy();names=z['aux__block_names'].copy();allpts=z['aux__canonical_pts'].copy()
        lowweights=z['profile__weights_3'].copy();lowpresent=bool(z['profile__present_3']);lowrobust=bool(z['profile__robust_3'])
    if list(names)!=['main1','main7','secondary','low','main9']:raise ValueError('Full original joint exposure inventory required')
    prepared=prepare_low(out/'low_source_factory');original=prepared.bundle
    offset=int(np.sum(lengths[:3]//4));lowpts=allpts[offset:offset+lengths[3]//4]
    if lengths[3]!=4*1851 or not np.array_equal(lowpts,original.stamps):raise ValueError('Complete actual low exposure scope changed')
    if len(original.a)!=434991 or original.motion_rows!=869982 or int(original.finite_mask.sum())!=274:raise ValueError('Complete admitted motion/material inventory changed')
    localq=prepared.q0.copy();localq[0]=warmq[6]/(100.*SCENE_UNIT_M);localq[4:]=warmq[starts[3]:starts[3]+lengths[3]]
    geometry=SharedRinkNumericGeometryMidpoint(prepared.scene);fixed=warmq[:4].copy();geometry.validate_parameters(np.r_[fixed,0.])
    observed={p:list(rows)for p,rows in original.abs_groups};allrows=[(role,row,p)for role,_,row,p in prepared.source_rows];newaudit=[]
    from render_result import advertisement_boundary_paths
    inventory={(r['number'],r['role']):np.asarray(r['xyz'])for r in advertisement_boundary_paths(prepared.baseline_scene)}
    for kind,path in [('user_major_F1_Lane1',Path(major)),('same_frame_Ad20',Path(ads))]:
        packet=read(path)
        if packet['source_sha256']!=SOURCE_SHA or packet['scene_sha256']!=HARD_SCENE_SHA or packet.get('frozen_before_fit')is not True:raise ValueError('Immutable native source authority required')
        for split in ('fit_rows','held_rows'):
            for raw in packet[split]:
                row=convert_row(raw);native_check(row)
                if (split=='fit_rows')!=(row['split']=='fit'):raise ValueError('Native source role changed')
                if split=='fit_rows' and prepared.aliases[int(row['source_native_pts'])]==prepared.aliases[230011136]:
                    raise ValueError('Every native alias of the exact user exposure is explicitly held')
                if kind=='user_major_F1_Lane1':
                    if row['physical_curve_id']not in('F1_outer','lane1_edge0'):raise ValueError('Only continuously confirmed F1/Lane1 paint permitted')
                    if row['source_native_pts']==230011136 and split=='fit_rows':raise ValueError('User exact source is explicitly held')
                else:
                    identity=row['physical_identity'];key=(identity['ad_id'],identity['geometry_role'])
                    if identity['ad_id']!=20 or not np.array_equal(row['world'],inventory[key]):raise ValueError('Real complete Ad20 external geometry changed')
                before=fingerprint(row);row=rebind(row,prepared.baseline_scene,prepared.scene)
                if fingerprint(row)!=before:raise ValueError('Scene binding changed source pixels/sigma/normal')
                p=prepared.aliases[int(row['source_native_pts'])];row['pts']=p
                if p not in original.index:raise ValueError('Source lies outside complete physical low camera')
                geometry.rebind(row,np.r_[fixed,0.]);role=kind+'_'+split
                allrows.append((role,row,p));newaudit.append(dict(role=role,id=row['id'],native_pts=row['source_native_pts'],actual_pts=p,source_fingerprint=before))
                if split=='fit_rows':observed.setdefault(p,[]).append(row)
    groups=sorted(observed.items());absolute_sigma=np.concatenate([np.concatenate([np.repeat(r['sigma'],2)for r in rows])for p,rows in groups])
    base=SimpleNamespace(stamps=original.stamps,index=original.index,a=original.a,b=original.b,ia=original.ia,ib=original.ib,motion_sigma=original.motion_sigma,abs_groups=groups,native_residual_sigma=np.r_[original.motion_sigma,absolute_sigma])
    bundle=MovingPivotSourceBundle(base,original.calibration,original.motion_kernel,reference_pts=original.reference_pts,initial_reference_optical_height_scene=original.initial_reference_optical_height_scene,finite_planes=original.finite_planes,finite_records=original.finite_records,scene=prepared.scene)
    if lowweights.shape!=(len(bundle.a),2):raise ValueError('Warm profile row order/shape changed')
    bundle.profile_weights=lowweights if lowpresent else None;bundle.robust_phase=lowrobust
    # Rebuilding the CSR only appends native absolute factors. The original
    # admitted UV, endpoint order, sigma, material planes and person mask are exact.
    for name in ('a','b','ia','ib','motion_sigma','finite_planes'):
        if not np.array_equal(getattr(bundle,name),getattr(original,name)):raise ValueError('Original source graph changed:'+name)
    temporal_manifest=None
    if temporal_source is not None:
        from z190_persistent_ice_source import append_persistent_ice_source
        bundle,temporal_manifest=append_persistent_ice_source(bundle,temporal_source,prepared.aliases)
        write(out/'persistent_ice_source_admission.json',temporal_manifest)
    block=ContinuousLowMidpointBlock(bundle,localq,geometry,fixed);q=np.r_[0.,warmq[6],localq[4:]]
    held={};pending={}
    for role,row,p in allrows:
        target=held if row['split']=='held'else pending if row['split']not in('fit',)else None
        if target is not None:target.setdefault(p,[]).append(row)
        geometry.rebind(row,block.geometry_q(q))
    independent_groups=independent_rows(prepared,Path(independent))
    for p,rows in independent_groups:
        for row in rows:geometry.rebind(row,block.geometry_q(q))
    prior_count=len(bundle.stamps)+1
    quadratic=np.r_[np.zeros(bundle.motion_rows,bool),np.concatenate([np.full(2*len(r['uv']),r.get('curve_observation_kind')in('horizontal_extent','clipped_horizontal_extent'),bool)for p,rows in groups for r in rows]),np.ones(prior_count+1,bool)]
    scales=np.r_[bundle.native_residual_sigma,np.ones(prior_count+1)]
    manifest=dict(schema='rinkspace.complete-low-user-major-continuous-source-midpoint.v1',source=ref(__file__),source_sha256=SOURCE_SHA,warm=ref(warm),warm_frozen_sha256=hashlib.sha256(frozen).hexdigest(),base_scene=ref(SCENE),full_low_factory=ref(out/'low_source_factory/input_manifest.json'),fixed_four_geometry_m=fixed.tolist(),free=['one shared row midpoint metre delta','one low shared pivot height metres','all1851 native actual R3/logf'],fixed=['pivotXY','nonzero camera-local lever','PP','k1','metric','outer/inner radii ratios','four geometry from previous joint17','Lane widths/tilt/offsets','boards/ads'],full_low_actual=1851,complete_cut_counts=prepared.manifest['actual_cut_counts'],original_native_2219_aliases_exact=True,admitted_original_motion=len(bundle.a),finite_source_rows=int(bundle.finite_mask.sum()),old_source_fit_rows=prepared.manifest['fit_rows'],total_source_fit_rows=sum(len(rows)for p,rows in groups),new_qualified_source=ref(major),same_native_Ad20=ref(ads),independent_user_held=ref(independent),new_source_binding_audit=newaudit,source_roles_UV_sigma_materials_retained=True,user_native_230011136_never_fit=True,source_identity_continuous_same_cut_not_per_frame_projection=True,priors_once=dict(low_reference_optical_height=1,actual_roll=len(bundle.stamps),roll_sigma_degrees=.5,midpoint=1,midpoint_sigma_m=geometry.midpoint_prior_sigma_m),midpoint_source=geometry.midpoint_source,unknown_depth_floor_m=.01,unknown_depth_floor_is_engineering_not_measured=True,other_installations_not_recovered_after_new_midpoint=True,accuracy_accepted=False,scene_adopted=False,video_generated=False)
    manifest['original_motion_rows_exact_prefix']=len(original.a)
    manifest['admitted_original_motion']=len(original.a)
    manifest['total_motion_factors_with_explicit_additions']=len(bundle.a)
    manifest['original_finite_source_rows']=int(original.finite_mask.sum())
    manifest['persistent_ice_source_opt_in']=temporal_manifest
    manifest['added_persistent_finite_rows']=0 if temporal_manifest is None else temporal_manifest['added_persistent_finite_rows']
    write(out/'input_manifest.json',manifest);write(out/'all_original_and_new_source_rows.json',[dict(role=role,source=row,actual_pts=p)for role,row,p in allrows])
    return SimpleNamespace(output=out,prepared=prepared,block=block,q=q,scales=scales,quadratic=quadratic,held=sorted(held.items()),pending=sorted(pending.items()),independent=independent_groups,warmq=warmq,starts=starts,lengths=lengths,names=names,allpts=allpts,manifest=manifest,temporal_source=None if temporal_source is None else Path(temporal_source))


def check_prepared(state):
    block,q=state.block,state.q
    values,jac=block.evaluate(q,True);checks=[]
    for col in range(2):
        h=1e-6;plus=q.copy();minus=q.copy();plus[col]+=h;minus[col]-=h
        fd=(block.evaluate(plus,False)-block.evaluate(minus,False))/(2*h);analytic=jac[:,col].toarray().ravel()
        error=float(np.linalg.norm(fd-analytic)/max(1.,np.linalg.norm(fd)))
        if error>2e-5:raise ValueError('Complete actual source global derivative failed')
        checks.append(dict(column=col,relative_error=error,max_abs_error=float(np.max(abs(fd-analytic)))))
    if len(values)!=len(state.scales)or jac.shape!=(len(values),len(q)):raise ValueError('Complete CSR source/prior inventory changed')
    write(state.output/'full_source_prepare_FD.json',dict(actual_rows=len(values),parameters=len(q),global_FD=checks,all_source_geometry_ID_and_original_UV_guards_passed=True,complete_1851_original_motion_and_finite_materials_retained=True,no_solve_yet=True))
    print(dict(stage='full_low_major_source_prepared_and_checked',parameters=len(q),residual_rows=len(values),FD=checks),flush=True)


def combined_checkpoint(state,q,path):
    combined=np.r_[state.warmq[:4],q[0],state.warmq[4:]];combined[7]=q[1]
    starts=state.starts+1;combined[starts[3]:starts[3]+state.lengths[3]]=q[2:]
    np.savez_compressed(path,q=combined,aux__block_start=starts,aux__block_lengths=state.lengths,aux__block_names=state.names,aux__canonical_pts=state.allpts,aux__geometry_parameter_count=np.array(5))


def run(args):
    state=prepare(args.output,args.warm,args.major,args.ads,args.independent,temporal_source=getattr(args,'temporal_source',None));check_prepared(state)
    if not args.solve:return state
    callback_counter=[0]
    def source_review(q,event=None):
        number=callback_counter[0];callback_counter[0]+=1
        folder=state.output/'actual_source_reviews'/f'{number:04d}';folder.mkdir(parents=True,exist_ok=False)
        checkpoint=folder/'complete_low_in_joint_state_UNACCEPTED.npz';combined_checkpoint(state,q,checkpoint)
        record=dict(event=event,common_midpoint_delta_m=float(q[0]),low_Pz_m=float(q[1]),all1851_Rf_from_original_dense_source=True,development=state.block.records(q,state.block.groups),held=state.block.records(q,state.held),pending_not_promoted=state.block.records(q,state.pending),independent_exact_user_held=state.block.records(q,state.independent),checkpoint=ref(checkpoint),other_installations_Rf_not_recovered_after_new_midpoint=True,accuracy_accepted=False)
        if state.temporal_source is not None:
            from z190_persistent_ice_source import persistent_ice_source_report
            record['persistent_ice_native_fit_and_held']=persistent_ice_source_report(state.block.bundle,state.block.local(q),state.temporal_source,state.prepared.aliases)
            record['explicit_added_motion_family']='persistent_ice_z0_source'
        write(folder/'source_review.json',record);write(state.output/'latest_actual_source_review.json',record)
        from render_joint_rink_dimensions_source import render
        render(SimpleNamespace(checkpoint=checkpoint,output=folder/'whole_source_all_model',pts=[230008064,230011136,230014976,230018816,230362112,230368768]))
        return None
    def guard(q):
        state.block.geometry.validate_parameters(state.block.geometry_q(q));local=state.block.local(q)
        state.block.bundle.validate_finite_extents(state.block.bundle.cache(local,False))
    result=solve_joint_source(state.block.evaluate,state.q,state.quadratic,state.scales,state.output/'solver',guard,update_profiles=make_profile_update([state.block]),checkpoint_callback=source_review,checkpoint_identity=state.manifest,checkpoint_arrays=dict(low_actual_pts=np.asarray(state.block.bundle.stamps,np.int64),fixed_four_geometry_m=state.block.fixed_geometry,all_joint_block_names=state.names))
    combined_checkpoint(state,result.q,state.output/'complete_low_in_joint_state_UNACCEPTED.npz')
    write(state.output/'result.json',dict(status=result.status,stop_reason=result.stop_reason,seconds=result.seconds,source_reviews=callback_counter[0],full_low_source_1851_solved=True,accuracy_accepted=False,other_installations_not_recovered=True))
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--warm',type=Path,default=WARM);parser.add_argument('--major',type=Path,default=MAJOR);parser.add_argument('--ads',type=Path,default=ADS);parser.add_argument('--independent',type=Path,default=INDEPENDENT);parser.add_argument('--temporal-source',type=Path);parser.add_argument('--solve',action='store_true');run(parser.parse_args())
