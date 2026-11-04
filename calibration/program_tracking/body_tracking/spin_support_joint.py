"""Continuous source-support initialization and full body/shoe/grasp refinement.

The source blade middle, exact runner minimum and anatomical ankle remain
separate quantities. A flat support boot is an initialization hypothesis only;
all body/root/toe columns are subsequently released in the one joint state.
"""
from pathlib import Path
import argparse,copy,json,time
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline,PchipInterpolator,make_interp_spline
from scipy.spatial.transform import Rotation
from rinkspace.storage import AssetStore
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .source_support_track import build_track
from .support_root_placement import restore_parent,runner,smoothstep
from .action_integration import ClipReference
from .skate_integration import JointContactProblem,source_evidence,restore_checkpoint_layers
from .skate_physics import fk_state
from .skate_checkpoint import checkpoint_identity
from .exmr_integration import PROGRAMME,CASE_FIELD,case_identity,save_packet,replay
from .support_image import SupportImageFactors
from .spin_grasp_factors import SpinGraspFactors,reassociate_support
from .skate_run_policy import RunPolicy
from .action_clip_input import camera25_from_snapshot
from .camera_bridge import rays
from .skate_template import profile_height
from .rebuild_exposure_body import moving_projection


def inverse_pixels(camera,info,frames,uv):
    poses=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],float(camera.p['first_control_time']),float(camera.p['control_dt']),camera.p['motion_times'][frames]+float(camera.p['rig'][37])*(uv[:,1]/1079-.5))
    cams=camera25_from_snapshot(poses);result=[]
    for pixel,cam in zip(uv,cams):
        ray,valid=rays(pixel[None],cam);C=cam[:3]*info['metres_per_scene_unit'];distance=-C[2]/ray[0,2]
        if not valid[0]or distance<=0:raise ValueError('Source blade has no forward nominal-ice ray')
        result.append(C+distance*ray[0])
    return np.array(result)


def prepare(project,base,old,output):
    track=build_track(project,base,old,output);case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True,exist_ok=True)
    active=json.loads((old/'active_candidate.json').read_text(encoding='utf8'));parent,camera,info,parent_saved=restore_parent(project,base,old,output/'parent_geometry',Path(active['parameter']))
    initial_path=case/'initial.npz';oldcase=old/'common/exmr_hand_nofoot'
    for name in ['input.npz','compact_smplx.npz','export_source.npz','observations.npz']:
        target=case/name
        if not target.exists():target.write_bytes((oldcase/name).read_bytes())
    source_doc=json.loads((old/'support_and_free_shoe_regions.json').read_text(encoding='utf8'))
    relations=json.loads((old/'grasp_relations.json').read_text(encoding='utf8'))
    origin=parent.p['pts'][0]/12800.;times=parent.times;seconds=parent.p['pts']/12800.;before=parent.at(times)
    if not initial_path.exists():
        local=before['local_R'].copy();Q=before['Q'];P=before['P'].copy();s=fk_state(parent,before,np.zeros((len(times),69)));yaw=np.unwrap(np.arctan2(s['R'][:,1,1,0],s['R'][:,1,0,0]))
        review={r['pts']for r in source_doc['observations']if r['role']=='review'}
        pairs={}
        for r in source_doc['observations']:
            if r['role']!='fit'or r['foot']!=1 or r['pts']in review:continue
            for part in ('toe','heel'):
                if r['id'].endswith('_'+part):pairs.setdefault(r['pts'],{})[part]=r['centre_uv']
        source_check=json.loads((old/'source_review/support_direct_check_v2.json').read_text(encoding='utf8'))
        for r in source_check['frames']:
            if r['native_pts']in review or r['toe_region_xyxy_px']is None:continue
            pairs[r['native_pts']]={part:np.array(r[key]).reshape(2,2).mean(0).tolist()for part,key in [('toe','toe_region_xyxy_px'),('heel','tail_region_xyxy_px')]}
        yaw_pts=[];corrections=[];yaw_checks=[]
        # A boot/toe pixel is above the plane. Do not infer its 3D direction
        # by putting both endpoints at Z=0. Select the initialization yaw by
        # projecting the actual fixed-length steel through the native camera.
        pars=parent.binding.packet()['parameters'];xm=.5*(-pars.runner_rear_extension_m+pars.sole_length_m-pars.runner_front_inset_m)
        qmid=np.array([xm,0.,float(profile_height(xm,pars))]);xx=np.array([pars.sole_length_m-pars.runner_front_inset_m,-pars.runner_rear_extension_m]);qends=np.c_[xx,np.zeros(2),profile_height(xx,pars)]
        for row in source_check['frames']:
            pt=row['native_pts']
            if pt in review or row['blade_midpoint_px']is None:continue
            i=np.searchsorted(parent.p['pts'],pt);uv=np.array([np.array(row[key]).reshape(2,2).mean(0)for key in ('toe_region_xyxy_px','tail_region_xyxy_px')])
            A=inverse_pixels(camera,info,np.array([i]),np.array([row['blade_midpoint_px']]))[0]/parent.binding.ratio
            theta=np.linspace(-np.pi,np.pi,721);R=Rotation.from_rotvec(np.c_[np.zeros(len(theta)),np.zeros(len(theta)),theta]).as_matrix()
            V=np.einsum('nij,kj->nki',R,qends-qmid)+A;V[:,:,2]+=qmid[2]
            pixels=camera.project(V.reshape(1,-1,3),np.array([i])).reshape(-1,2,2);cost=np.sum((pixels-uv)**2,axis=(1,2));best=int(np.argmin(cost));target=theta[best]
            correction=np.arctan2(np.sin(target-yaw[i]),np.cos(target-yaw[i]));yaw_pts.append(pt);corrections.append(correction)
            yaw_checks.append(dict(pts=pt,native_yaw_seed_rad=float(target),native_endpoint_rms_px=float(np.sqrt(cost[best]/2)),not_contact_or_orientation_truth=True))
        write_json(output/'native_yaw_initialization.json',dict(seeds=yaw_checks,method='fixed metric steel projected at native source camera; root midpoint only uses nominal plane; roll/pitch/yaw subsequently free'))
        if len(yaw_pts)<4:raise ValueError('Insufficient source toe/heel orientation seeds')
        correction=np.interp(parent.p['pts'],yaw_pts,np.unwrap(corrections));desired_yaw=yaw+correction
        flat=Rotation.from_rotvec(np.c_[np.zeros(len(times)),np.zeros(len(times)),desired_yaw]).as_matrix()
        a,b,c,d=(82.,82.4,92.46,92.8);weight=smoothstep((seconds-a)/(b-a))*smoothstep((d-seconds)/(d-c))
        Rbinding=parent.binding.parameters()[0][1];a_joint=8;parent_joint=int(parent.body.parents[a_joint])
        target_local=s['frames'][:,parent_joint].transpose(0,2,1)@Q.transpose(0,2,1)@flat@Rbinding.T
        rel=local[:,7].transpose(0,2,1)@target_local;rv=Rotation.from_matrix(rel).as_rotvec();local[:,7]=local[:,7]@Rotation.from_rotvec(weight[:,None]*rv).as_matrix()
        # A neutral support toe is a boot-fit initialization, not a measured toe
        # angle. Both toe joints remain adjustable in the final 63-column state.
        rv=Rotation.from_matrix(local[:,10].transpose(0,2,1)).as_rotvec();local[:,10]=local[:,10]@Rotation.from_rotvec(weight[:,None]*rv).as_matrix()
        valid=track['valid'];t=(track['pts'][valid]-parent.p['pts'][0])/12800.;uv=track['uv'][valid]
        knots=np.r_[[t[0]]*4,np.arange(t[0]+.18,t[-1]-1e-8,.18),[t[-1]]*4];B=BSpline.design_matrix(t,knots,3).toarray();w=np.sqrt(np.maximum(track['confidence'][valid],.05))
        pixel_curve=PchipInterpolator(t,uv,axis=0,extrapolate=True)
        all_uv=pixel_curve(np.clip(times,t[0],t[-1]));anchors=inverse_pixels(camera,info,np.arange(len(times)),all_uv)/parent.binding.ratio
        pose=dict(P=P,Q=Q,local_R=local,**({'hand_local_R':before['hand_local_R']}if 'hand_local_R'in before else{}));ss=fk_state(parent,pose,np.zeros((len(times),69)))
        pars=parent.binding.packet()['parameters'];xx=.5*(-pars.runner_rear_extension_m+pars.sole_length_m-pars.runner_front_inset_m);q=np.array([xx,0,float(profile_height(xx,pars))])
        middle=ss['O'][:,1]+np.einsum('nij,j->ni',ss['R'][:,1],q);minimum,_=runner(parent,ss);X=middle.copy();X[:,2]=minimum[:,2]
        P+=weight[:,None]*(anchors-X)
        original=load(oldcase/'initial.npz');original.update(P=P,Q=Q,local_R=local,pts=parent.p['pts'])
        curve=make_interp_spline(times,P,k=3);original.update(root_curve_schema=np.array('rinkspace.source-clock-root-bspline.v1'),root_curve_knots=curve.t,root_curve_controls=curve.c,root_curve_degree=np.array(curve.k),root_curve_origin=np.array('source-supported native exposure curve; no 0.2s reapproximation'))
        np.savez_compressed(initial_path,**original)
        np.savez_compressed(output/'source_root_initialization.npz',pts=parent.p['pts'],support_pixel_curve=all_uv,conditional_support_anchor_world_m=anchors*parent.binding.ratio,attachment_weight=weight,source_yaw_pts=yaw_pts,source_yaw_corrections_rad=corrections)
        write_json(output/'initialization.json',dict(parent_parameter=active['parameter'],parent_sha256=digest(Path(active['parameter'])),pose_origin='actual retained four-step HandMR/noFootMR pose; support ankle/toe initialized from source blade direction and nominal plane only',root_origin='source-image support middle, per-exposure native Brown inverse; no original network root LS inside support interval',not_new_inference=True,unverified_flat_boot_initialization=True,support_interval=[82.4,92.46],unknown_entry_and_release=True,source_track=str(output/'source_support_track.json'),yaw_seed_pts=yaw_pts))
    # Replace only the semantically incorrect geometric-minimum pixel rows.
    # They remain present for reporting, and all historical review roles persist.
    for row in source_doc['observations']:
        if row['kind']=='support_contact_region':row.update(optimization_enabled=False,replaced_in_optimization_by='source-image blade-midpoint rows; minimum height is separate finite ice geometry')
    known={r['id']for r in source_doc['observations']};selected=np.flatnonzero(track['optimization_use']);pt=track['pts'][selected];dt=np.r_[np.diff(pt/12800.)[0]/2,(pt[2:]-pt[:-2])/25600.,np.diff(pt/12800.)[-1]/2]
    for k,i in enumerate(selected):
        key='derived_support_middle_'+str(int(track['pts'][i]));quality=float(track['confidence'][i]);half=np.array([12.,10.])+(1-quality)*np.array([20.,15.])
        if key in known:continue
        source_doc['observations'].append(dict(id=key,source_id=source_doc['source_id'],source_sha256=source_doc['source_sha256'],shot_id='s05',pts=int(track['pts'][i]),time_base=[1,12800],foot=1,foot_track='support_track',kind='blade_midpoint_region',centre_uv=track['uv'][i].tolist(),region_shape_px2=np.diag(half**2).tolist(),visibility='visible',contact_state='supported',role='fit',reliability=quality,objective_weight=float(dt[k]),provenance='source_support_track.json: derived source pixels from admitted native blade-middle seeds; 10Hz sampling with time quadrature, not repeated manual truth',source_visibility='derived source-image flow; confidence is not calibrated',source_uncertainty='finite native-pixel region, not a universal metric support band'))
    direct=json.loads((old/'source_review/support_direct_check_v2.json').read_text(encoding='utf8'))
    protected={r['pts']for r in source_doc['observations']if r['role']=='review'}
    known={r['id']for r in source_doc['observations']}
    for row in direct['frames']:
        if row['native_pts']in protected or row['blade_midpoint_px']is None:continue
        for endpoint,key in [('front','toe_region_xyxy_px'),('rear','tail_region_xyxy_px')]:
            rid='direct_blade_'+str(row['native_pts'])+'_'+endpoint
            if rid in known:continue
            box=np.array(row[key],float).reshape(2,2);half=np.maximum((box[1]-box[0])/2,6.)
            source_doc['observations'].append(dict(id=rid,source_id=source_doc['source_id'],source_sha256=source_doc['source_sha256'],shot_id='s05',pts=row['native_pts'],time_base=[1,12800],foot=1,foot_track='support_track',kind='blade_endpoint_region',blade_endpoint=endpoint,centre_uv=box.mean(0).tolist(),region_shape_px2=np.diag(half**2).tolist(),visibility='visible',contact_state='supported',role='fit',reliability=.65,objective_weight=1.,provenance='source_review/support_direct_check_v2.json: directly viewed native steel endpoint tolerance region, not anatomical toe or boot centroid; finite geometric end proxy, material location blurred'))
    write_json(output/'support_and_free_shoe_regions.json',source_doc)
    relations=copy.deepcopy(relations);relations['factor_parameters']['grip']['objective_weight']=80.
    relations['factor_parameter_change_reason']='Retained held-phase palm/rear-shoe gap 0.239m and native 89.56s visible unclosed grip. One explicit finite positional-grip reweight; no ice reweight or threshold continuation.'
    relations['support_intervals'][0]['start_pts']=round(82.4*12800);relations['support_intervals']=relations['support_intervals'][:1]
    write_json(output/'grasp_relations.json',relations)
    for a,b in [(old/'palm_grip_geometry.json',output/'palm_grip_geometry.json'),(old/'exmr_body_hand_nofoot.npz',output/'exmr_body_hand_nofoot.npz')]:
        if not b.exists():b.write_bytes(a.read_bytes())
    ref=ClipReference(parent,case,camera.p,load(initial_path));ref.grasp_geometry=json.loads((output/'palm_grip_geometry.json').read_text(encoding='utf8'));ref.grasp_relations=relations
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance={'status':'explicit native shoe/blade source observations supplied by SupportImageFactors; no legacy opening rows'})
    ref.hashes=parent.hashes.copy();ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    identity=case_identity(project,base,output,'exmr_hand_nofoot',info);ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True));write_json(case/'comparison_case_identity.json',identity)
    return ref,camera,info,source_doc,relations


def make_problem(project,base,old,output,log):
    ref,camera,info,doc,relations=prepare(project,base,old,output);case=output/'common/exmr_hand_nofoot'
    supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=doc['supports'])
    problem=JointContactProblem(ref,camera,case/'observations.npz',(82.,94.8),[],case/'fit',log,'spin_support_v2',optimize_shared=False,record_acceptance_before_jacobian=True)
    runtime=copy.deepcopy(doc)
    for row in runtime['observations']:row['objective_weight']*=relations['factor_parameters']['shoe_region_weight_multiplier']
    problem.support_image=SupportImageFactors(problem,runtime);problem.action_factors=SpinGraspFactors(problem,relations)
    problem.checkpoint_extra=dict(hand_pose_schema_version=np.array('rinkspace.hand-local-R.v1'),grasp_relations_sha256=np.array(digest(output/'grasp_relations.json')),grasp_geometry_sha256=np.array(digest(output/'palm_grip_geometry.json')),source_support_track_sha256=np.array(digest(output/'source_support_track.npz')),initialization_provenance_json=np.array(json.dumps(json.loads((output/'initialization.json').read_text(encoding='utf8')),sort_keys=True)))
    return problem,np.r_[ref.binding.x,np.zeros(problem.body.size)],info,supports,doc


def run(project,base,old,output,stage='all',budget=2100.,steps=8):
    output.mkdir(parents=True,exist_ok=True);case=output/'common/exmr_hand_nofoot';start=time.perf_counter()
    with (output/'progress.jsonl').open('a',encoding='utf8',buffering=1)as log:
        problem,x,info,supports,doc=make_problem(project,base,old,output,log)
        initial=save_packet(problem,info,supports,doc,case/'initial',status='common_initial')
        if stage=='prepare':
            v=problem.evaluate(x,False);write_json(output/'prepared_domain.json',dict(gate=v['gate'],parts=v['raw_data_parts']));print(json.dumps(dict(event='support_seed_ready',gate=v['gate'],parts=v['raw_data_parts'])),flush=True);return problem,info
        tick=time.perf_counter();value=problem.evaluate(x,True);jsecs=time.perf_counter()-tick
        d=np.zeros_like(x);d[17:]=np.random.default_rng(110406).normal(size=problem.body.size);d/=np.linalg.norm(d);eps=1e-5
        a=problem.evaluate(x+eps*d,False)['residual'];b=problem.evaluate(x-eps*d,False)['residual'];fd=(a-b)/(2*eps);jac=value['jac']@d;error=float(np.linalg.norm(fd-jac)/max(1.,np.linalg.norm(fd)))
        profile=dict(jacobian_seconds=jsecs,mixed_derivative_relative_l2=error,gate=value['gate'],source_frames=len(initial['pts']),active_columns=problem.body.state.active.tolist(),variables=problem.body.size,source_rows=len(problem.support_image.rows),coverage=problem.action_factors.last['metrics'],no_ice_reweight=True,profile_and_prepare_seconds=time.perf_counter()-start)
        write_json(case/'profile.json',profile);print(json.dumps(dict(event='support_joint_profile',jacobian_seconds=jsecs,mixed_error=error,gate=value['gate'],source_rows=len(problem.support_image.rows))),flush=True)
        if error>2e-3:raise ValueError('Current full mixed derivative mismatch')
        if stage=='profile':return
        # Restore the already-computed initial J into the exact cache. No
        # second seed Jacobian, no stage reset for unchanged runner identity.
        problem.cache_x=x.copy();problem.cache=value;problem.jacobian_seconds_hint=jsecs
        remain=budget-(time.perf_counter()-start)
        if remain<2*jsecs:raise RuntimeError('Measured run budget cannot fund meaningful joint progress')
        x,report=problem.solve(x,RunPolicy(wall_seconds=remain,max_accepted=steps,max_outer=1))
        checkpoint=load(case/'fit/candidate.npz');init_archive=project/PROGRAMME/'spin_support_joint_v2_initial_1104.npz';archive=project/PROGRAMME/'spin_support_joint_v2_1104.npz'
        for p in (init_archive,archive):
            if len(str(p).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive output path too long')
        if init_archive.exists():assert init_archive.read_bytes()==(case/'initial.npz').read_bytes()
        else:init_archive.write_bytes((case/'initial.npz').read_bytes())
        checkpoint.update(initial_reference=np.array(init_archive.relative_to(project).as_posix()),initial_reference_sha256=np.array(digest(init_archive)),camera_prior_sha256=np.array(digest(output/'exmr_body_hand_nofoot.npz')),original_gvhmr_pose_anchor=np.array(False))
        if archive.exists():raise FileExistsError('Preserve existing full joint candidate')
        np.savez_compressed(archive,**checkpoint);restore_checkpoint_layers(problem.reference,checkpoint,include_primary=True)
        final=save_packet(problem,info,supports,doc,case,status='joint_fit',report=report)
        summary=dict(accepted=problem.accepted,retained_step=report['selection']['retained_step'],seconds=time.perf_counter()-start,budget_seconds=budget,parameter=str(archive),initial_parameter=str(init_archive),result=str(case/'fitted/result.npz'),result_sha256=digest(case/'fitted/result.npz'),source_frames=len(final['pts']),source_image_native=True,accuracy_accepted=False,stop=report['stop'])
        write_json(output/'run_summary.json',summary);print(json.dumps(dict(event='support_joint_candidate',**summary)),flush=True)
    replay(project,base,output,'exmr_hand_nofoot',archive)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--previous-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['prepare','profile','all'],default='all');p.add_argument('--budget',type=float,default=2100.);p.add_argument('--max-steps',type=int,default=8);a=p.parse_args(argv)
    store=AssetStore(a.project);paths=[store.resolve(v,base=store.data,must_exist=i<2)for i,v in enumerate([a.base_root,a.previous_output,a.output])]
    return run(a.project,*paths,stage=a.stage,budget=a.budget,steps=a.max_steps)

if __name__=='__main__':main()
