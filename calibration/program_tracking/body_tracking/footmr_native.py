"""Evaluate a FootMR local candidate through the unchanged native camera."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation,Slerp
from .rebuild_prepare import load,digest
from .footwear_fit import FootwearProblem
from .footwear_geometry import extra_rotations,equipment_world,lowest_blade_world
from .rebuild_exposure_body import moving_projection
from .footmr_contract import from6


def evaluate(root,review,dataset,base_body,joint_result,candidate,input_contract,prediction,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False);joint=load(joint_result);net=load(candidate);contract=load(input_contract);pred=load(prediction)
    if str(net['source_base_leg_result_sha256'].item())!=digest(joint_result):raise ValueError('FootMR candidate does not bind the selected effective leg state')
    p=FootwearProblem(root,review,'articulated',dataset=dataset,exposure_nodes=5,base_body=base_body);np.testing.assert_array_equal(net['pts'],p.p['pts']);frames=np.arange(len(net['pts']));g=joint['shared_geometry'].copy();raw=p.state(np.r_[g,joint['leg_controls'].ravel()],frames,root_controls=joint['root_delta_controls'],project=False);base=raw['base'];extra=extra_rotations(raw['correction']);n=len(frames)
    source_t=p.p['fit_time_seconds'];query=(source_t[:,None]+p.exposure_offsets[None]*float(p.p['rig'][37])).ravel();apply=np.asarray(net['candidate_supported_time'])&~p.p['lower_body_inferred']
    for foot,j in enumerate([7,8]):
        R=Slerp(source_t,Rotation.from_rotvec(net['additional_right_rotation_axis_angle'][:,foot]))(np.clip(query,source_t[0],source_t[-1])).as_matrix();R[~np.repeat(apply,5)]=np.eye(3);old=Rotation.from_rotvec(extra[j]).as_matrix();extra[j]=Rotation.from_matrix(old@R).as_rotvec()
    coco,joints,_,rotations=p.body.forward(base['inputs']['pose'],base['delta'],additional_rotations=extra,with_frames=True);body=base['P'][:,None]+np.einsum('nij,nkj->nki',base['Q'],coco);P=base['P'].reshape(n,5,3)[:,2];G=P.copy();G[:,2]=0
    np.testing.assert_array_equal(P,joint['P_world_mid']);np.testing.assert_array_equal(G,joint['G_world_mid'])
    # On common 50/30Hz instants, compare the actual native body's effective
    # camera-space ankle with the official network output. This exercises the
    # J1 leg conditioning and right composition, not just a synthetic SO(3) test.
    model_t=contract['model_relative_seconds'];hi=np.searchsorted(model_t,source_t).clip(0,len(model_t)-1);common=np.isclose(model_t[hi],source_t,atol=1e-10,rtol=0)&apply
    Rcam=p.p['camera_rotation'][:,None]@base['Q'].reshape(n,5,3,3)[:,2,None]@rotations.reshape(n,5,22,3,3)[:,2,[7,8]];expected=from6(pred['global_ankle_rot6d'][hi]);error=float(np.max(abs(Rcam[common]-expected[common])))
    if error>2e-5:raise ValueError('FootMR world/camera/local composition mismatch '+str(error))
    variants=[]
    for label,angles in [('conditional_installation',g[10:13].copy()),('zero_installation',np.zeros(3))]:
        parameters=g.copy();parameters[10:13]=angles;gear,parts,bootR,bootO=equipment_world(base['P'],base['Q'],joints,rotations,parameters);world=np.concatenate((gear.reshape(n*5,96,3),body),1).reshape(n,5,113,3);projection=moving_projection(p.projector,world,frames);minimum=lowest_blade_world(bootR,bootO,parameters).reshape(n,5,2,3)[:,2]
        packet=dict(schema=np.array('rinkspace.footmr-native-candidate.v1'),pts=net['pts'],source_time_base=net['source_time_base'],P_world_mid=P,G_world_mid=G,Q_world=joint['Q_world'],COCO_world_mid=world[:,2,96:],COCO_uv=projection['uv'][:,96:],equipment_uv=projection['uv'][:,:96].reshape(n,2,48,2),blade_world_at_row=projection['world'][:,:96].reshape(n,2,48,3)[:,:,14:],nominal_blade_minimum_world_m=minimum,shared_geometry=parameters,parent_joint_result_sha256=np.array(digest(joint_result)),candidate_sha256=np.array(digest(candidate)),candidate_applied_time=apply,source_foot_observation_supported=net['source_foot_observation_supported'],source_lower_body_inferred=p.p['lower_body_inferred'],contact_allowed_for_root=np.zeros(n,bool))
        path=output/(label+'.npz');np.savez_compressed(path,**packet);variants.append(dict(label=label,path=str(path.resolve()),sha256=digest(path),installation_angles_rad=angles.tolist()))
    result=dict(schema='rinkspace.footmr-native-projection-audit.v1',model_executed=True,parent_joint_sha256=digest(joint_result),candidate_sha256=digest(candidate),input_contract_sha256=digest(input_contract),camera_updated=False,root_updated=False,contact_feedback=False,applied_frames=int(apply.sum()),unsupported_or_lower_body_inferred_frames=int((~apply).sum()),common_native_model_rotation_checks=int(common.sum()),maximum_camera_global_ankle_rotation_error=error,variants=variants,physical_foot_and_anatomical_heel_toe_verified=False,output_is_candidate_only=True)
    (output/'native_projection_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','review','dataset','base-body','joint-result','candidate','input-contract','prediction','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();evaluate(a.root,a.review,a.dataset,a.base_body,a.joint_result,a.candidate,a.input_contract,a.prediction,a.output)
