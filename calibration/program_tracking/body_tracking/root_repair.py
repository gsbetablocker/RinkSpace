"""Repair root depth on complete saved A/B clips without re-running inference."""
from pathlib import Path
import argparse, json
import numpy as np
from .action_integration import initialize_clip, ClipReference, summarize
from .skate_integration import JointContactProblem, restore_checkpoint_layers, source_evidence
from .skate_run_policy import RunPolicy
from .skate_physics import fk_state, batch_features
from .root_trajectory import proxy_root_samples, smooth_root_samples, image_height_given_xy
from .rebuild_prepare import load, digest
from .fullbody_round import write_json
from .skate_delivery import export_candidate

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')


def conditional_seed(ref, native, use, clip, base_root, info, recipe):
    poses=ref.at(ref.times);n=len(ref.times);poses['P']=np.zeros((n,3))
    state=fk_state(ref,poses,np.zeros((n,69)))
    features=batch_features(ref.binding,state['R'],state['O'],state)
    arcs=len(ref.binding.packet()['template']['contact_arcs'])
    runner=features['height'][:,:,:arcs].min(2)/ref.binding.ratio
    coco=ref.world(poses)['coco'];sec=ref.p['pts']/12800.
    contact=np.ones(n,bool);foot=np.full(n,-1,int)
    for start,end in recipe['exclude_support_proxy_seconds']:
        contact[(sec>=start)&(sec<=end)]=False
    for start,end,side in recipe['known_proxy_foot_seconds']:
        foot[(sec>=start)&(sec<=end)]=int(side)
    samples,selected=proxy_root_samples(native['camera25'],ref.p['source_uv'],use,coco,
        runner,contact,foot,float(ref.p['metres_per_scene_unit']))
    if recipe['root_origin']=='accepted_s01':
        saved=load(Path(base_root)/recipe['accepted_initial_root_result'])
        lookup={int(p):i for i,p in enumerate(saved['pts'])}
        samples=np.stack([saved['P_world_mid'][lookup[int(p)]] for p in ref.p['pts']])/info['length_export_ratio']
        selected[:]=-1
    smooth=smooth_root_samples(ref.times,samples)
    z=image_height_given_xy(native['camera25'],ref.p['source_uv'],use,coco,
        smooth[:,:2],float(ref.p['metres_per_scene_unit']))
    free=~contact&np.isfinite(z);smooth[free,2]=z[free]
    smooth=smooth_root_samples(ref.times,smooth,sigma_m=.03,acceleration_mps2=25.)
    return dict(pts=ref.p['pts'],P=smooth,conditional_samples=samples,
        initializer_selected_foot=selected,contact_candidate=contact,
        known_initializer_foot=foot,observation_contact_labels_generated=np.array(False))


def run_arm(project,base_root,run,clip,arm,budget=120.):
    project,base_root,run=map(Path,(project,base_root,run))
    root=run/clip;out=root/'root_repair'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'fitted/result.npz').exists():raise FileExistsError(out/'fitted/result.npz')
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'),clip=clip,arm=arm,
        wall_seconds=budget,max_accepted=8,scope='one complete clip, translation of common body/skate state')),flush=True)
    evidence=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf-8'))['clips'][clip]
    native_path=run/'source_cache'/f'{clip}.native50.npz';native=load(native_path)
    ref,camera,info=initialize_clip(project,base_root,root,native_path,root/'prior/camera_prior.npz',
        evidence['shot_id'],heldout_pts=evidence.get('heldout_pts',[]))
    recipe_path=project/PROGRAMME/'root_trajectory_1103.json'
    recipe_document=json.loads(recipe_path.read_text(encoding='utf-8'))
    contract=json.loads((root/'contract.json').read_text(encoding='utf-8'))
    if recipe_document['source_sha256']!=contract['source']['sha256']:raise ValueError('Root recipe source differs from prepared clip')
    recipe=recipe_document['clips'][clip]
    shared=ref
    # Seed A and B from one common trajectory, retaining the same GVHMR origin.
    restore_checkpoint_layers(ref,load(root/'A/fit/candidate.npz'),include_primary=True)
    seed_path=root/'root_repair/initial.npz'
    if seed_path.exists():seed=load(seed_path)
    else:
        seed=conditional_seed(ref,native,load(root/'observations.npz')['use'],clip,base_root,info,recipe)
        np.savez_compressed(seed_path,**seed)
    ref.new_refinements=[]
    restore_checkpoint_layers(ref,load(root/arm/'fit/candidate.npz'),include_primary=True)
    pose=ref.at(ref.times);pose['P']=seed['P'].copy()
    reference=ClipReference(shared,root,ref.p,pose)
    reference.hashes=dict(ref.hashes,previous_arm_parameters_sha256=digest(root/arm/'fit/candidate.npz'),
        common_root_seed_sha256=digest(seed_path))
    target=project/PROGRAMME/f'action_{clip}_{arm}_root_1103.npz'
    initial_target=project/PROGRAMME/f'action_{clip}_{arm}_root_reference_1103.npz'
    for path in (target,initial_target):
        if len(str(path).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    np.savez_compressed(initial_target,pts=ref.p['pts'],**pose,
        root_origin=np.array('conditional proxy/runner initializer; accepted S01 root on step clip'),
        previous_arm_parameters_sha256=np.array(digest(root/arm/'fit/candidate.npz')),
        **reference.fixed_checkpoint_identity)
    supports=source_evidence(root,shot_id=evidence['shot_id'],source_pts=ref.p['pts'],records=evidence.get('supports',[]))
    first,last=ref.p['pts'][[0,-1]]/12800.;padding=max(abs(float(camera.p['rig'][37]))/2,.01)
    with (out/'fit_progress.jsonl').open('a',encoding='utf-8',buffering=1)as log:
        problem=JointContactProblem(reference,camera,root/'observations.npz',
            (float(first-padding),float(last+padding)),supports,out/'fit',log,
            'continuous_root',optimize_shared=False)
        from .action_live import GlideActionFactors
        problem.action_factors=GlideActionFactors(problem,evidence,sigma_mps=evidence['glide_sigma_mps'],enabled=arm=='B')
        initial=np.r_[reference.binding.x,np.zeros(problem.body.size)]
        # A directional check on the actual live residual, before spending the fit budget.
        rng=np.random.default_rng(110304);direction=np.r_[np.zeros(17),rng.normal(size=problem.body.size)]
        direction/=np.linalg.norm(direction);h=1e-5
        v=problem.evaluate(initial,True);numeric=(problem.evaluate(initial+h*direction)['residual']-problem.evaluate(initial-h*direction)['residual'])/(2*h)
        analytic=v['jac']@direction
        error=float(np.linalg.norm(analytic-numeric)/max(np.linalg.norm(numeric),1e-12))
        write_json(out/'derivative_check.json',dict(relative_norm=error,maximum_absolute=float(np.max(abs(analytic-numeric))),nonzero_root_direction=True))
        if error>2e-4:raise ValueError('Root live derivative mismatch: '+str(error))
        problem.cache=problem.cache_x=None
        x,report=problem.solve(initial,RunPolicy(wall_seconds=budget,max_accepted=8,max_outer=1,allow_penalty_growth=False))
    reference.binding.x=x[:17].copy();reference.new_refinements.append((problem.body.state,x[17:].copy()))
    np.testing.assert_array_equal(x[:17],initial[:17])
    np.testing.assert_allclose(reference.at(ref.times)['Q'],pose['Q'],rtol=0,atol=2e-14)
    np.testing.assert_allclose(reference.at(ref.times)['local_R'],pose['local_R'],rtol=0,atol=2e-14)
    packet=export_candidate(root,out,reference,camera,info,supports)
    old=load(root/arm/'fitted/result.npz')
    for name in ('source_action_phase_mid','action_evidence_json'):
        if name in old:packet[name]=old[name].copy()
    if 'source_action_phase_mid' in packet:
        sec=packet['pts']/12800.;phase=packet['source_action_phase_mid']
        for row in recipe['phase_overrides']:
            mask=(sec>=row['seconds'][0])&(sec<=row['seconds'][1])
            phase[np.ix_(mask,row['feet'])]=row['phase']
    packet['root_phase_override_json']=np.array(json.dumps(recipe,sort_keys=True))
    packet.update(ice_soft_policy_json=np.array(json.dumps(vars(problem.ice_policy),sort_keys=True)),
        ice_is_delivery_gate=np.array(False),root_repair=np.array(True),
        root_initialization_sha256=np.array(digest(seed_path)),
        pose_refinement_in_this_repair=np.array(False),costs_json=np.array(json.dumps(report['parts'])))
    np.savez_compressed(out/'fitted/result.npz',**packet)
    params=dict(load(out/'fit/candidate.npz'),initial_state_file=np.array(initial_target.relative_to(project).as_posix()),
        initial_state_sha256=np.array(digest(initial_target)),root_seed_sha256=np.array(digest(seed_path)),
        prior_identity_json=np.array(json.dumps(reference.hashes,sort_keys=True)))
    np.savez_compressed(target,**params)
    write_json(out/'metrics.json',dict(clip=clip,arm=arm,metrics=summarize(packet,evidence),run=report,
        parameters=str(target),reference=str(initial_target),pose_unchanged_at_source_midpoints=True,
        initializer_is_observation=False,additional_support_rows=0))
    print(json.dumps(dict(event='root_repair_complete',clip=clip,arm=arm,result=str(out/'fitted/result.npz'))),flush=True)


def run_pose(project,base_root,run,clip,arm,total_budget=120.):
    project,base_root,run=map(Path,(project,base_root,run));root=run/clip
    previous=root/'root_repair'/arm;out=root/'root_repair'/(arm+'_pose');out.mkdir(parents=True,exist_ok=True)
    if (out/'fitted/result.npz').exists():raise FileExistsError(out/'fitted/result.npz')
    previous_report=json.loads((previous/'metrics.json').read_text(encoding='utf-8'))
    used=previous_report['run']['run_policy']['latest']['elapsed_seconds']
    accepted=previous_report['run']['run_policy']['accepted']
    budget=max(0.,total_budget-used);steps=max(0,8-accepted)
    print(json.dumps(dict(main_log=str(out/'fit_progress.jsonl'),clip=clip,arm=arm,remaining_wall_seconds=budget,
        remaining_accepted_steps=steps,scope='complete-clip pose; repaired root fixed; original ice policy')),flush=True)
    evidence=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf-8'))['clips'][clip]
    shared,camera,info=initialize_clip(project,base_root,root,run/'source_cache'/f'{clip}.native50.npz',
        root/'prior/camera_prior.npz',evidence['shot_id'],heldout_pts=evidence.get('heldout_pts',[]))
    target=project/PROGRAMME/f'action_{clip}_{arm}_root_pose_1103.npz'
    initial_target=project/PROGRAMME/f'action_{clip}_{arm}_root_reference_1103.npz'
    if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    initial_state=load(initial_target)
    reference=ClipReference(shared,root,shared.p,initial_state)
    reference.hashes=dict(shared.hashes,initial_state_sha256=digest(initial_target),root_checkpoint_sha256=digest(previous/'fit/candidate.npz'))
    restore_checkpoint_layers(reference,load(previous/'fit/candidate.npz'),include_primary=True)
    fixed_root=reference.at(reference.times)['P'].copy()
    supports=source_evidence(root,shot_id=evidence['shot_id'],source_pts=reference.p['pts'],records=evidence.get('supports',[]))
    first,last=reference.p['pts'][[0,-1]]/12800.;padding=max(abs(float(camera.p['rig'][37]))/2,.01)
    with (out/'fit_progress.jsonl').open('a',encoding='utf-8',buffering=1)as log:
        problem=JointContactProblem(reference,camera,root/'observations.npz',
            (float(first-padding),float(last+padding)),supports,out/'fit',log,'continuous_pose',optimize_shared=False)
        from .action_live import GlideActionFactors
        problem.action_factors=GlideActionFactors(problem,evidence,sigma_mps=evidence['glide_sigma_mps'],enabled=arm=='B')
        # Use only completed real evaluations to avoid starting a Jacobian
        # that demonstrably cannot fit the remaining aggregate budget.
        rates=[]
        for profile in run.glob('*/root_repair/*_pose/fit_progress.jsonl'):
            if not (profile.parent/'metrics.json').exists():continue
            count=len(load(profile.parents[2]/'input.npz')['pts'])
            for line in profile.read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                if row.get('event')=='joint_evaluation' and row.get('with_jacobian'):
                    rates.append(row['evaluation_seconds']/count)
        if rates:problem.jacobian_seconds_hint=max(rates)*len(reference.times)
        initial=np.r_[reference.binding.x,np.zeros(problem.body.size)]
        x,report=problem.solve(initial,RunPolicy(wall_seconds=budget,max_accepted=steps,max_outer=1,allow_penalty_growth=False))
    reference.binding.x=x[:17].copy();reference.new_refinements.append((problem.body.state,x[17:].copy()))
    np.testing.assert_array_equal(x[:17],initial[:17]);np.testing.assert_array_equal(reference.at(reference.times)['P'],fixed_root)
    packet=export_candidate(root,out,reference,camera,info,supports)
    old=load(previous/'fitted/result.npz')
    for name in ('source_action_phase_mid','action_evidence_json','root_initialization_sha256','root_phase_override_json'):
        if name in old:packet[name]=old[name].copy()
    np.testing.assert_allclose(packet['P_world_mid'],old['P_world_mid'],rtol=0,atol=1e-13)
    packet.update(ice_soft_policy_json=np.array(json.dumps(vars(problem.ice_policy),sort_keys=True)),
        ice_is_delivery_gate=np.array(False),root_repair=np.array(True),
        pose_refinement_in_this_repair=np.array(True),costs_json=np.array(json.dumps(report['parts'])))
    np.savez_compressed(out/'fitted/result.npz',**packet)
    params=dict(load(out/'fit/candidate.npz'),initial_state_file=np.array(initial_target.relative_to(project).as_posix()),
        initial_state_sha256=np.array(digest(initial_target)),prior_identity_json=np.array(json.dumps(reference.hashes,sort_keys=True)))
    np.savez_compressed(target,**params)
    write_json(out/'metrics.json',dict(clip=clip,arm=arm,metrics=summarize(packet,evidence),run=report,
        parameters=str(target),reference=str(initial_target),root_unchanged_at_source_midpoints=True,
        previous_root_solver_seconds=used,total_solver_budget_seconds=total_budget,total_solver_accepted_budget=8,
        initializer_is_observation=False,additional_support_rows=0))
    print(json.dumps(dict(event='pose_refinement_complete',clip=clip,arm=arm,result=str(out/'fitted/result.npz'))),flush=True)


def reuse_inactive_spin(project,run):
    """Record equivalent A/B input and a zero B factor; do not pretend a second fit ran."""
    project,run=map(Path,(project,run));root=run/'spin_s05'
    e=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf-8'))['clips']['spin_s05']
    if any(r['phase']=='stable_glide' for r in e['intervals']):return False
    a=load(root/'A/fitted/result.npz');b=load(root/'B/fitted/result.npz')
    limits={'P_world_mid':1e-12,'Q_world_mid':1e-14,'local_R_mid':1e-14,'COCO_uv':1e-10,'equipment_uv':1e-10}
    maximum={name:float(np.max(np.abs(a[name]-b[name]))) for name in limits}
    if any(maximum[name]>limit for name,limit in limits.items()):return False
    np.testing.assert_array_equal(a['pts'],b['pts'])
    record=dict(status='reused_A_with_numerical_equivalence',actual_B_root_fits=0,actual_B_pose_fits=0,
        reason='B has no enabled stable-glide rows in this spin clip; same source, camera, shared geometry and equivalent saved pose',
        source_result='root_repair/A/fitted/result.npz',maximum_saved_AB_difference=maximum,
        equivalence_limits=limits,original_A_sha256=digest(root/'A/fitted/result.npz'),original_B_sha256=digest(root/'B/fitted/result.npz'),
        reused_result_sha256=digest(root/'root_repair/A/fitted/result.npz'),
        pose_budget_exit='A whole-pose Jacobian exhausted remaining budget without an accepted step; no duplicate pose solve')
    out=root/'root_repair/B';out.mkdir(parents=True,exist_ok=True);write_json(out/'reuse.json',record)
    print(json.dumps(record),flush=True);return True


def register(project,base_root,run,clip):
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    project,base_root,run=map(Path,(project,base_root,run))
    path=project/PROGRAMME/f'viewer3d_action_{clip}_1103.json'
    config=json.loads(path.read_text(encoding='utf-8'))
    config['variants']=[v for v in config['variants'] if v['id'] not in ('A_root','B_root')]
    for arm in ('A','B'):
        folder='root_repair/'+arm
        if clip=='spin_s05' and arm=='B' and (run/clip/'root_repair/B/reuse.json').exists():folder='root_repair/A'
        pose_folder=folder+'_pose'
        pose_report=run/clip/pose_folder/'metrics.json'
        if pose_report.exists():
            review=json.loads(pose_report.read_text(encoding='utf-8'))
            if review['run']['run_policy']['accepted']>0:folder=pose_folder
        config['variants'].append(dict(id=arm+'_root',label=clip+' '+arm+' 根轨迹修正（图像待复核）',kind='fullbody',
            result=folder+'/fitted/result.npz',shell_topology='saved',
            body_reference_label='same_uniform25_gvhmr_then_saved_'+arm+'_pose_root_repair',body_refinement_intervals_seconds=[]))
    config['default_variant']='A_root';config['comparison_variants']=['A','A_root']
    config['root_repair_status']='unaccepted candidate; reduced root drift; inspect image regression and remaining pose/contact failures'
    write_json(path,config)
    catalog=ResultCatalog(project=project,data_root=base_root.parents[1],config=path)
    for arm in ('A_root','B_root'):prepare_variant(catalog,catalog.variant(arm))
    return path


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--base-root',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--clip',choices=('jump_s02','step_s01','spin_s05'),required=True)
    parser.add_argument('--arm',choices=('A','B'));parser.add_argument('--budget',type=float,default=120.)
    parser.add_argument('--register',action='store_true');parser.add_argument('--pose',action='store_true');args=parser.parse_args(argv)
    if args.arm and args.clip=='spin_s05' and args.arm=='B' and reuse_inactive_spin(args.project,args.run):return
    if args.arm:
        if args.pose:run_pose(args.project,args.base_root,args.run,args.clip,args.arm,args.budget)
        else:run_arm(args.project,args.base_root,args.run,args.clip,args.arm,args.budget)
    if args.register:print(register(args.project,args.base_root,args.run,args.clip),flush=True)


if __name__=='__main__':main()
