"""Place the single body from source-supported blade/ice observations.

This is an explicitly conditional kinematic candidate, not another solver run.
The anchor track comes only from admitted fit pixels and the existing native
camera/nominal ice. Articulation, hands, camera and shared shoes are unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, json, time
import numpy as np
from scipy.interpolate import CubicSpline
from rinkspace.storage import AssetStore
from .rebuild_prepare import load, digest
from .fullbody_round import write_json
from .action_integration import ClipReference
from .skate_integration import build, restore_checkpoint_layers, SkateReference
from .fullbody_state import camera_binding
from .skate_checkpoint import checkpoint_identity, validate_checkpoint_identity
from .exmr_integration import (PROGRAMME, CASE_FIELD, read_case_identity,
    validate_case_identity, resolve_initial_reference, case_identity, replay_errors)
from .skate_physics import fk_state, batch_features
from .skate_delivery import export_candidate
from .action_clip_input import camera25_from_snapshot
from .camera_bridge import rays
from .rebuild_exposure_body import moving_projection

SCHEMA='rinkspace.source-supported-root-placement.v1'


def restore_parent(project,base,output,target,parent_path):
    case=output/'common/exmr_hand_nofoot';saved=load(parent_path)
    initial_path=resolve_initial_reference(project,saved);identity=read_case_identity(saved)
    p=load(case/'input.npz');camera,info,_,pose=camera_binding(project,SimpleNamespace(p=p),shot_id='s05')
    expected=case_identity(project,base,output,'exmr_hand_nofoot',info,initial=initial_path,
                           prior_sha256=identity['camera_prior_sha256'])
    validate_case_identity(saved,expected)
    shared,_,_,_=build(project,base,target/'shared')
    seed=load(project/PROGRAMME/'skate_s01_1103.npz')
    validate_checkpoint_identity(seed,shared.fixed_checkpoint_identity)
    shared.binding.x=seed['x'][:17].copy();shared.binding.enable_collar_constraints()
    np.testing.assert_array_equal(saved['x'][:17],shared.binding.x)
    p=camera.p;p.update(camera_rotation=pose['rotation'],camera_centre=pose['centre'],camera_values=pose['values'])
    ref=ClipReference(shared,case,p,load(initial_path))
    ref.grasp_geometry=json.loads((output/'palm_grip_geometry.json').read_text(encoding='utf8'))
    ref.grasp_relations=json.loads((output/'grasp_relations.json').read_text(encoding='utf8'))
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(expected,sort_keys=True))
    validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    ref.hashes={k:expected[k] for k in ('native_input_sha256','camera_prior_sha256','shared_s01_parameters_sha256')}
    restore_checkpoint_layers(ref,saved,include_primary=True)
    return ref,camera,info,saved


def verify_document(document):
    if document.get('time_base')!=[1,12800] or document.get('shot_id')!='s05':raise ValueError('Wrong source clock or shot')
    rows=document['anchors'];pts=np.array([r['pts'] for r in rows],np.int64)
    if len(rows)<4 or np.any(np.diff(pts)<=0):raise ValueError('Ordered sparse fit anchors required')
    for r in rows:
        if r['role']!='fit' or r['foot']!=1:raise ValueError('Review observations cannot locate candidate')
        if r['source_sha256']!=document['source_sha256']:raise ValueError('Source identity mismatch')
        if digest(Path(r['source_frame']))!=r['source_frame_sha256']:raise ValueError('Native source image SHA mismatch')
    return rows


def inverse_anchors(document,camera,info):
    """Existing Brown inverse at each original observed rolling-shutter row."""
    rows=verify_document(document);lookup={int(p):i for i,p in enumerate(camera.p['pts'])};world=[]
    for row in rows:
        uv=np.asarray(row['centre_uv'],float);i=lookup[row['pts']]
        t=camera.p['motion_times'][i]+float(camera.p['rig'][37])*(uv[1]/1079.-.5)
        pose=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],
          float(camera.p['first_control_time']),float(camera.p['control_dt']),np.array([t]))
        cam25=camera25_from_snapshot(pose)[0];direction,valid=rays(uv[None],cam25)
        C=cam25[:3]*float(info['metres_per_scene_unit']);distance=-C[2]/direction[0,2]
        if not bool(valid[0]) or distance<=0:raise ValueError('Invalid native ice intersection')
        world.append(C+distance*direction[0])
    return np.array([r['pts'] for r in rows],np.int64),np.array(world)


def runner(reference,state,foot=1):
    f=batch_features(reference.binding,state['R'],state['O']);ids=np.flatnonzero(np.asarray(f['semantics'])=='runner')
    selected=ids[np.argmin(f['height'][:,foot,ids],axis=1)];i=np.arange(len(selected))
    q=f['local_point'][i,foot,selected]
    X=state['O'][:,foot]+np.einsum('nij,nj->ni',state['R'][:,foot],q)
    return X,selected


def location_feature(reference,state,document):
    X,selected=runner(reference,state)
    if document.get('root_anchor_feature','runner_lowest_point')=='blade_midpoint_xy_runner_min_z':
        from .skate_template import profile_height
        p=reference.binding.packet()['parameters'];x=.5*p.sole_length_m
        q=np.array([x,0.,float(profile_height(x,p))])
        middle=state['O'][:,1]+np.einsum('nij,j->ni',state['R'][:,1],q)
        X[:,:2]=middle[:,:2]
    return X,selected


def smoothstep(x):
    x=np.clip(np.asarray(x,float),0,1)
    return x*x*x*(10+x*(-15+6*x))


class SupportLocatedReference(SkateReference):
    """Root follows a moving support track; feet remain attached through FK."""
    def __init__(self,parent,pts,world,document):
        self.__dict__.update(parent.__dict__);self.parent=parent
        self.anchor_pts=np.asarray(pts,np.int64);self.anchor_world=np.asarray(world,float)
        self.support_location_document=document
        clock=(self.anchor_pts-self.p['pts'][0])/12800.
        self.anchor_curve=CubicSpline(clock,self.anchor_world/self.binding.ratio,bc_type='natural',extrapolate=True)
        self.taper=np.asarray(document['attachment_seconds'],float)-self.p['pts'][0]/12800.

    def weight(self,t):
        a,b,c,d=self.taper
        return smoothstep((np.asarray(t)-a)/(b-a))*smoothstep((d-np.asarray(t))/(d-c))

    def at(self,t):
        t=np.atleast_1d(t).astype(float);original=self.parent.at(t)
        result={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
        w=self.weight(t);use=w>0
        if use.any():
            sub={k:v[use] for k,v in original.items()}
            s=fk_state(self.parent,sub,np.zeros((use.sum(),69)));X,_=location_feature(self.parent,s,self.support_location_document)
            delta=self.anchor_curve(t[use])-X
            result['P'][use]+=w[use,None]*delta
        return result


def _packet(ref,camera,info,supports,target,document,parent_sha):
    p=export_candidate(ref.root,target,ref,camera,info,supports)
    mask=ref.weight(ref.times)>0
    p.update(body_refinement_window_mask=p['body_refinement_window_mask']|mask,
      window_mask=p['window_mask']|mask,support_location_applied_mask=mask,
      support_location_weight_mid=ref.weight(ref.times),
      support_location_document_json=np.array(json.dumps(document,sort_keys=True)),
      support_location_parent_parameter_sha256=np.array(parent_sha),
      support_location_mode=np.array('conditional_native_support_blade_root_placement'),
      support_location_not_independent_truth=np.array(True),
      support_image_in_optimization=np.zeros((len(mask),2),bool),
      comparison_stage=np.array('kinematic_support_seed_after_four_joint_steps'),
      original_gvhmr_pose_anchor=np.array(False),ice_mode=np.array('finite_soft_objective'),
      ice_is_delivery_gate=np.array(False),**{CASE_FIELD:ref.fixed_checkpoint_identity[CASE_FIELD]})
    # Sparse source evidence remains sparse. The continuous placement mask is
    # explicitly an interpolation/kinematic hypothesis, never contact truth.
    np.savez_compressed(target/'fitted/result.npz',**p)
    return p


def run(project,base,output,*,parameter=None,anchor_document=None,replay=False,candidate_name='support_located'):
    started=time.perf_counter();project,base,output=map(lambda p:Path(p).resolve(),(project,base,output))
    if candidate_name not in ('support_located','support_midpoint'):raise ValueError('Unknown support placement candidate name')
    parameter=Path(parameter or project/PROGRAMME/('spin_'+candidate_name+'_1104.npz'))
    active=json.loads((output/'active_candidate.json').read_text(encoding='utf8'))
    if replay:
        saved=load(parameter)
        if str(saved['schema'])!=SCHEMA:raise ValueError('Unsupported support-location parameter')
        document=json.loads(str(saved['document_json']));parent_path=(project/str(saved['parent_parameter'])).resolve()
        if not parent_path.is_relative_to(project):raise ValueError('Parent reference escapes project')
        if digest(parent_path)!=str(saved['parent_sha256']):raise ValueError('Parent parameter SHA mismatch')
        pts,world=saved['anchor_pts'],saved['anchor_world_m'];target=output/(candidate_name+'_replay_'+digest(parameter)[:12])
    else:
        if parameter.exists():raise FileExistsError('Existing support-location candidate is preserved')
        document=json.loads(Path(anchor_document).read_text(encoding='utf8'));parent_path=Path(active['parameter'])
        target=output/candidate_name
    if target.exists():raise FileExistsError('Existing root-placement output is preserved')
    target.mkdir(parents=True)
    ref,camera,info,parent_saved=restore_parent(project,base,output,target,parent_path)
    if not replay:pts,world=inverse_anchors(document,camera,info)
    else:
        actual_pts,actual_world=inverse_anchors(document,camera,info)
        np.testing.assert_array_equal(actual_pts,pts);np.testing.assert_allclose(actual_world,world,atol=1e-12,rtol=0)
    parent_result=load(active['result'])
    if replay and digest(Path(active['result']))!=str(saved['parent_result_sha256']):raise ValueError('Parent result SHA mismatch')
    located=SupportLocatedReference(ref,pts,world,document)
    supports=json.loads(str(parent_result['footwear_contact_evidence_json']))
    packet=_packet(located,camera,info,supports,target,document,digest(parent_path))
    for key in ('Q_world_mid','local_R_mid','hand_local_R_mid'):
        np.testing.assert_allclose(packet[key],parent_result[key],atol=2e-10,rtol=0)
    outside=~packet['support_location_applied_mask']
    np.testing.assert_allclose(packet['P_world_mid'][outside],parent_result['P_world_mid'][outside],atol=2e-10,rtol=0)
    np.testing.assert_allclose(packet['palm_to_segment_m_mid'],parent_result['palm_to_segment_m_mid'],atol=2e-10,rtol=0)
    times=located.times;s=fk_state(located,located.at(times),np.zeros((len(times),69)));X,arcs=runner(located,s);X*=located.binding.ratio
    np.savez_compressed(target/'support_trajectory.npz',pts=packet['pts'],anchor_pts=pts,anchor_world_m=world,
      runner_world_m=X,runner_arc=arcs,root_world_m=packet['P_world_mid'])
    # Audit through the existing five-node native projection, not the inverse
    # itself or a simplified pinhole closure.
    frames=np.searchsorted(packet['pts'],pts);shutter=float(camera.p['rig'][37]);qt=(times[frames,None]+np.linspace(-.5,.5,5)*shutter).ravel()
    state=fk_state(located,located.at(qt),np.zeros((len(qt),69)));node,_=location_feature(located,state,document)
    projected=moving_projection(camera,node.reshape(len(pts),5,1,3),frames)['uv'][:,0]
    pixels=np.array([r['centre_uv'] for r in document['anchors']]);error=np.linalg.norm(projected-pixels,axis=1)
    report=dict(schema=SCHEMA,status='conditional kinematic candidate; visual accuracy not accepted',no_new_optimization=True,
      no_new_network=True,source_frames=len(times),sparse_fit_anchors=len(pts),review_anchors_used=0,
      native_five_node_anchor_pixel_errors=error.tolist(),native_five_node_anchor_max_px=float(error.max()),
      anchor_attachment_weights=located.weight(times[frames]).tolist(),
      full_attachment_anchor_max_px=float(error[located.weight(times[frames])>.999].max()),
      boundary_anchor_note='81.2s is the unchanged attachment endpoint, not a fully placed anchor; its large original pixel error is preserved',
      root_anchor_feature=document.get('root_anchor_feature','runner_lowest_point'),
      source_anchor_world_span_m=np.ptp(world,axis=0).tolist(),phases={},
      relative_body_hands_grip_unchanged=True,collar_unknown='same verified parent domain; local_R/hand and rigid invariant geometry unchanged',
      limitations=['conditional on nominal ice, native camera and source region interpretation','between sparse anchors the moving track is C2 interpolation, not per-frame measurement','palm gap/articulation failures of parent are preserved','geometric runner witness may switch; not a fixed tracked steel material point'],
      parent_parameter=str(parent_path),parent_parameter_sha256=digest(parent_path),parent_result=str(active['result']),
      seconds=time.perf_counter()-started)
    for name,(a,b) in {'entry_84_86':(84,86),'held_86_8_92_46':(86.8,92.46),'checked_support_87_64_92_46':(87.64,92.46),'release_92_46_92_8':(92.46,92.8)}.items():
        m=(packet['pts']>=round(a*12800))&(packet['pts']<=round(b*12800));P=packet['P_world_mid'];old=parent_result['P_world_mid']
        report['phases'][name]=dict(before_pelvis_xy_span_m=np.ptp(old[m,:2],axis=0).tolist(),
          after_pelvis_xy_span_m=np.ptp(P[m,:2],axis=0).tolist(),after_runner_xy_span_m=np.ptp(X[m,:2],axis=0).tolist(),
          root_change_median_m=float(np.median(np.linalg.norm(P[m]-old[m],axis=1))),raw_runner_z_range_m=[float(X[m,2].min()),float(X[m,2].max())])
    if replay:
        expected=load(output/candidate_name/'fitted/result.npz');result=replay_errors(packet,expected)
        result.update(parameter_sha256=digest(parameter),no_fit_performed=True,no_network_performed=True)
        if not result['passed']:raise ValueError('Support placement replay mismatch')
        write_json(target/'replay.json',result);report['replay_passed']=result['passed']
    else:
        parameter.parent.mkdir(parents=True,exist_ok=True)
        if len(str(parameter).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive parameter path too long')
        np.savez_compressed(parameter,schema=np.array(SCHEMA),document_json=np.array(json.dumps(document,sort_keys=True)),
          parent_parameter=np.array(parent_path.relative_to(project).as_posix()),parent_sha256=np.array(digest(parent_path)),
          parent_result_sha256=np.array(digest(Path(active['result']))),anchor_pts=pts,anchor_world_m=world,
          result_sha256=np.array(digest(target/'fitted/result.npz')),implementation_sha256=np.array(digest(__file__)))
        write_json(output/(candidate_name+'_candidate.json'),dict(parameter=str(parameter),result=str(target/'fitted/result.npz'),report=str(target/'report.json'),
          joint_accepted_steps=4,additional_optimization_steps=0,conditional_kinematic_placement=True,accuracy_accepted=False))
    write_json(target/'report.json',report)
    print(json.dumps(dict(event='source_support_root_ready',replay=replay,target=str(target),phases=report['phases']),ensure_ascii=False),flush=True)
    return packet,report


def audit_source_regions(project,base,output,candidate_name='support_midpoint'):
    from .skate_integration import SpinGraspWindow
    from .support_image import SupportImageFactors
    target=output/candidate_name;parameter=project/PROGRAMME/('spin_'+candidate_name+'_1104.npz')
    saved=load(parameter);parent_path=(project/str(saved['parent_parameter'])).resolve()
    ref,camera,info,_=restore_parent(project,base,output,target/'source_audit',parent_path)
    located=SupportLocatedReference(ref,saved['anchor_pts'],saved['anchor_world_m'],json.loads(str(saved['document_json'])))
    doc=json.loads((output/'support_and_free_shoe_regions.json').read_text(encoding='utf8'));report={}
    for name,r in [('four_joint_steps',ref),(candidate_name,located)]:
        state=SpinGraspWindow(r,(82.,94.8));body=SimpleNamespace(state=state,tau=float(camera.p['rig'][37]))
        stub=SimpleNamespace(reference=r,binding=r.binding,camera=camera,body=body,optimize_shared=False)
        rows=SupportImageFactors(stub,doc,include_review=True).evaluate(np.zeros(state.size),False)['metrics']['observations'];groups={}
        for phase,(a,b) in {'entry_84_86':(84.,86.),'held_86_8_92_46':(86.8,92.46),'release_92_46_93_2':(92.46,93.2)}.items():
            by={}
            for role in ('fit','review'):
                for foot in (0,1):
                    for kind in ('boot_marker_region','support_contact_region'):
                        selected=[v for v in rows if v['role']==role and v['foot']==foot and v['kind']==kind and round(a*12800)<=v['pts']<=round(b*12800)]
                        values=np.asarray([v['outside_region_px'] for v in selected],float)
                        by[role+'_'+kind+'_foot'+str(foot)]=dict(rows=len(values),exposures=len({v['pts'] for v in selected}),median_outside_native_px=float(np.median(values)) if len(values) else None,max_outside_native_px=float(values.max()) if len(values) else None)
            groups[phase]=by
        report[name]=groups
    write_json(target/'native_source_region_audit.json',dict(cases=report,source_region_roles_preserved=True,review_used_for_fit_or_selection=False,metric='unweighted distance outside admitted native uncertainty region; not material-point pixel error',conditional_anchor_is_not_independent_source=True))
    return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchor-document',type=Path);p.add_argument('--parameter',type=Path);p.add_argument('--replay',action='store_true');p.add_argument('--candidate-name',choices=['support_located','support_midpoint'],default='support_located');a=p.parse_args(argv)
    store=AssetStore(a.project)
    return run(a.project,store.resolve(a.base_root,base=store.data,must_exist=True),store.resolve(a.output,base=store.data,must_exist=True),
      parameter=a.parameter,anchor_document=a.anchor_document,replay=a.replay,candidate_name=a.candidate_name)

if __name__=='__main__':main()
