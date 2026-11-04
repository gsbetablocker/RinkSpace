"""Small source-region refinement on the existing GVHMR/FK/native chain."""
from pathlib import Path
from collections import Counter
import argparse,copy,json,time,hashlib
import numpy as np
from .rebuild_prepare import load,digest
from .fullbody_state import camera_binding
from .action_integration import ClipReference,summarize
from .skate_integration import build,restore_checkpoint_layers,JointContactProblem,source_evidence
from .skate_checkpoint import validate_checkpoint_identity,checkpoint_identity
from .skate_run_policy import RunPolicy
from .skate_delivery import export_candidate
from .source_support_export import apply_source_support
from .support_image import SupportImageFactors
from .stage_quality import StageQualityTrace,body_groups,error_summary
from .fullbody_round import write_json

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')


def prepare_regions(project,base,output):
    """Source review stays immutable; explicit fixed associations add no tracks."""
    path=output/'source_review/source_region_review.json'
    review=json.loads(path.read_text(encoding='utf-8'))
    evidence=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf-8'))['clips']['spin_s05']
    observations=[];supports=[]
    for row in review['rows']:
        for name,region in [('shoe_body',row['support'].get('shoe_body')),
                            ('toe',row['support'].get('toe')),('heel',row['support'].get('heel')),
                            ('contact_candidate',row['support'].get('contact_candidate'))]:
            if not region:continue
            contact=name=='contact_candidate';half=np.asarray(region['half_extent_uv'],float)
            # The source reviewer marked rectangles. Circumscribe them rather
            # than silently treating an inscribed ellipse as the measured region.
            record=dict(id=f"spin_{row['pts']}_{name}",source_id=evidence['source_id'],
                source_sha256=review['source_sha256'],shot_id='s05',pts=int(row['pts']),time_base=[1,12800],
                foot=1,foot_track='support_track',kind='support_contact_region' if contact else 'boot_marker_region',
                centre_uv=region['center_uv'],region_shape_px2=np.diag(2*half**2).tolist(),
                visibility='visible',contact_state='supported' if contact else 'unknown',role=row['role'],
                reliability=.35 if contact else .65,objective_weight=1.,
                provenance='assistant direct source-image development review; '+str(path),
                image_footprint=dict(source_frame=row['source_frame'],source_frame_sha256=digest(row['source_frame']),
                    region_rectangle_half_extent_uv=region['half_extent_uv'],native_dimensions=[1920,1080]),
                feature=name,source_uncertainty=region.get('reason','finite shoe region; no exact material-point claim'),
                association='fixed model foot 1 / support_track, compared against foot 0 using fit frames only',
                human_confirmed=False)
            if not contact:record['adapter_point_ids']={'toe':[1,8],'heel':[0,7],'shoe_body':list(range(14))}[name]
            observations.append(record)
            if contact and row['role']=='fit':
                supports.append(dict(id=record['id'],pts=int(row['pts']),source_frame=row['source_frame'],
                    source=record['provenance'],source_id=evidence['source_id'],source_sha256=review['source_sha256'],
                    shot_id='s05',source_time_base=[1,12800],confidence=.35,mode='glide',feet=[1],
                    physical_side='source support_track; anatomical right is a medium-confidence inference',
                    edge='unknown',source_support_weight_s=.02,
                    source_support_weight_basis='one discrete native exposure; not duration/propagated contact',
                    human_confirmed=False,note=record['source_uncertainty']))
    document=dict(schema='rinkspace.source-support-regions.v1',source_id=evidence['source_id'],source_sha256=review['source_sha256'],
        shot_id='s05',time_base=[1,12800],pixel_domain='native Brown 1920x1080',
        context_seconds=[87.5,92.5],core_seconds=[88.,92.],robust_delta=2.,support_foot=1,
        objective_weight_note='Separate finite image-region weight 1 per sparse admitted observation; existing body sqrt(.02), ice policy and common normalization unchanged. Engineering weighting, not accuracy calibration.',
        association_fit_median_px={'original_A':{'foot_0':694.0693977407329,'foot_1':40.287471341905274},
            'old_root':{'foot_0':634.2972941097128,'foot_1':46.583163062836675}},
        observations=observations,supports=supports,blade_curves=[],
        blade_unknown_reason='All selected steel edges blend with shadow/reflection or are end-on; no precise blade curve admitted.',
        review_exclusion='Review shoe/contact labels are absent from fit factors, seed selection, and height evidence; prior networks saw the images.')
    target=project/PROGRAMME/'support_spin_1104.json'
    if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    write_json(target,document);write_json(output/'support_regions.json',document)
    return document


def reference_for(project,base,output,seed):
    old=base/'action_integration_1103/spin_s05'
    shared,_,_,_=build(project,base,output/'shared')
    s01=load(project/PROGRAMME/'skate_s01_1103.npz')
    validate_checkpoint_identity(s01,shared.fixed_checkpoint_identity)
    shared.binding.x=s01['x'][:17].copy();shared.binding.enable_collar_constraints()
    p=load(old/'input.npz')
    shell=type('Input',(),{'p':p})()
    camera,info,_,pose=camera_binding(project,shell,shot_id='s05')
    p=camera.p;p['camera_rotation']=pose['rotation'];p['camera_centre']=pose['centre'];p['camera_values']=pose['values']
    if seed=='original_A':
        initial=load(old/'initial.npz');checkpoint=old/'A/fit/candidate.npz';saved=old/'A/fitted/result.npz'
    elif seed=='old_root':
        initial=load(project/PROGRAMME/'action_spin_s05_A_root_reference_1103.npz')
        checkpoint=old/'root_repair/A_pose/fit/candidate.npz';saved=old/'root_repair/A_pose/fitted/result.npz'
    elif seed=='stage1_retained':
        initial=load(old/'initial.npz');checkpoint=output/'original_A/fit/candidate.npz';saved=output/'original_A/fitted/result.npz'
    else:raise ValueError(seed)
    ref=ClipReference(shared,old,p,initial)
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    params=load(checkpoint);validate_checkpoint_identity(params,ref.fixed_checkpoint_identity)
    restore_checkpoint_layers(ref,params,include_primary=True)
    ref.export_source_path=old/'export_source.npz'
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],
        provenance={'status':'source shoe and support regions enter the named support_image block; no invented blade curves'})
    ref.hashes=dict(seed=seed,seed_checkpoint_sha256=digest(checkpoint),seed_result_sha256=digest(saved),
        shared_s01_parameters_sha256=digest(project/PROGRAMME/'skate_s01_1103.npz'))
    original=load(saved);at=ref.at(ref.times);ratio=info['length_export_ratio']
    checks=dict(P=float(np.max(abs(at['P']*ratio-original['P_world_mid']))),
        Q=float(np.max(abs(at['Q']-original['Q_world_mid']))),
        local_R=float(np.max(abs(at['local_R']-original['local_R_mid']))))
    if max(checks.values())>2e-10:raise ValueError('Seed replay mismatch: '+str(checks))
    return ref,camera,info,checks


def make_problem(project,base,output,seed,document,log,*,arc_ids=None):
    ref,camera,info,replay=reference_for(project,base,output,seed)
    old_evidence=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf-8'))['clips']['spin_s05']
    supports=source_evidence(ref.root,shot_id='s05',source_pts=ref.p['pts'],
        records=old_evidence.get('supports',[])+document['supports'])
    problem=JointContactProblem(ref,camera,ref.root/'observations.npz',tuple(document['context_seconds']),
        supports,output/seed/'fit',log,'support_joint_right_v1',optimize_shared=False)
    ref.footwear_observations['support_regions']=document['observations']
    problem.support_image=SupportImageFactors(problem,document,arc_ids=arc_ids)
    report_factor=SupportImageFactors(problem,document,include_review=True,arc_ids=problem.support_image.arc_ids)
    initial=np.r_[ref.binding.x,np.zeros(problem.body.size)]
    return problem,report_factor,initial,info,replay,supports


def measure(problem,report_factor,x,value):
    base=problem.body.evaluate(x[17:],False);obs=problem.body.obs;frames=problem.body.frames
    clock=obs['pts'][frames];core=(clock>=88*12800)&(clock<=92*12800)
    result=dict(groups=body_groups(base['uv'][core][:,problem.body.layout.body_landmarks],
        obs['uv'][frames][core],obs['use'][frames][core],clock[core]),objective_image_cost=value['image_cost'])
    details=report_factor.evaluate(x[17:],False)['metrics']['observations']
    for role in ['fit','review']:
        for kind,label in [('boot_marker_region','boot'),('support_contact_region','support_region')]:
            rows=[r for r in details if r['role']==role and r['kind']==kind and 88*12800<=r['pts']<=92*12800]
            for foot in [0,1]:
                part=[r for r in rows if r['foot']==foot];pts=np.array([r['pts']for r in part])
                summary=error_summary(np.array([r['outside_region_px']for r in part]),pts)
                summary['centre_distance_px']=error_summary(np.array([r['centre_error_px']for r in part]),pts)
                summary['measurement']='outside finite source region; centre is not an exact observed material point'
                result['groups'][f'{role}_{label}_foot_{foot}']=summary
    result['groups']['blade']={'count':0,'reason':problem.support_image.document['blade_unknown_reason']}
    result['regions']=details
    result['ice_soft']=dict(value['ice']['parts'])
    result['structural_diagnostics']=dict(problem.evaluate(x,False)['gate'])
    return result


def profile(project,base,output,document):
    path=output/'profile.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    started=time.perf_counter();profile_log=output/'profile.jsonl'
    result=dict(schema='rinkspace.spin-support-profile.v1',candidates={},source_sha256=document['source_sha256'],
        observations_sha256=digest(output/'support_regions.json'),new_network_inference=False,new_fit=False)
    arc_ids=None
    with profile_log.open('a',encoding='utf-8',buffering=1)as log:
        for seed in ['original_A','old_root']:
            problem,report_factor,x,info,replay,supports=make_problem(project,base,output,seed,document,log,arc_ids=arc_ids)
            if arc_ids is None:arc_ids=problem.support_image.arc_ids
            t=time.perf_counter();v=problem.evaluate(x,seed=='original_A');elapsed=time.perf_counter()-t
            measurement=measure(problem,report_factor,x,v)
            row=dict(evaluation_seconds=elapsed,with_jacobian=seed=='original_A',seed_replay=replay,
                initial_objective=float(v['residual']@v['residual']/2),measurement=measurement,
                active_body_columns=problem.body.state.active.tolist(),variables=problem.body.size,
                source_frames=problem.body.n,physics_times=len(problem.physics_times),camera=info)
            if seed=='original_A':
                J=v['jac'];support=v['block_slices']['support_image'];rng=np.random.default_rng(11048892)
                direction=np.r_[np.zeros(17),rng.normal(size=problem.body.size)];direction/=np.linalg.norm(direction)
                h=1e-5;numeric=(problem.evaluate(x+h*direction)['residual']-problem.evaluate(x-h*direction)['residual'])/(2*h)
                exact=J@direction
                dd=dict(relative_norm=float(np.linalg.norm(exact-numeric)/max(np.linalg.norm(numeric),1e-12)),
                    support_relative_norm=float(np.linalg.norm(exact[support]-numeric[support])/max(np.linalg.norm(numeric[support]),1e-12)),
                    support_max_abs=float(np.max(abs(exact[support]-numeric[support]))),
                    mixed_root_XYZ_SO3_support_hip_knee_ankle=True,support_nonzero_jacobian_nnz=int(J[support].nnz))
                row['directional_check']=dd
                if dd['support_relative_norm']>2e-4 or dd['support_nonzero_jacobian_nnz']==0:raise ValueError('Live support chain derivative failed: '+str(dd))
                body_count=len(problem.body.of)*2
                neutral=len(problem.binding.compatibility_rows(problem.binding.neutral_native,problem.binding.neutral_frames,problem.binding.neutral_samples).ravel())
                ice_start=len(problem.body.cache['residual'])+neutral+17
                slices={'body_pixels':slice(0,body_count),'support_image':support,
                    'ice_soft':slice(ice_start,ice_start+len(v['ice']['residual']))}
                directions={'world_X':[1,0,0],'world_Y':[0,1,0],'world_normal':[0,0,1]}
                frame=int(np.argmin(abs(problem.reference.p['pts']/12800-90)))
                cam=load(base/'action_integration_1103/source_cache/spin_s05.native50.npz')['camera_centre_m'][frame]
                P=problem.reference.at([90-problem.origin])['P'][0]*info['length_export_ratio'];ray=P-cam;ray/=np.linalg.norm(ray)
                directions['camera_line_of_sight']=ray.tolist();effects={}
                for name,axis in directions.items():
                    d=np.zeros((problem.body.state.m,problem.body.state.d));d[:,:3]=axis
                    vector=np.r_[np.zeros(17),d.ravel()];change=J@vector
                    effects[name]={key:dict(jacobian_norm_per_solver_m=float(np.linalg.norm(change[s])),
                        cost_directional_derivative_per_solver_m=float(v['residual'][s]@change[s]))for key,s in slices.items()}
                row['directional_block_effects']=effects
            result['candidates'][seed]=row
    result['fixed_arc_ids']=arc_ids
    result['fit_coverage']={f'{kind}_foot_{foot}':len({r['pts']for r in document['observations']if r['role']=='fit'and r['kind']==kind and r['foot']==foot})for kind in ['support_contact_region','boot_marker_region']for foot in [0,1]}
    result['review_exposures']=sorted({r['pts']for r in document['observations']if r['role']=='review'})
    result['seed_selection']='lowest complete fit objective with same fixed source rows; review-region metrics excluded'
    result['preferred_seed']=min(result['candidates'],key=lambda name:result['candidates'][name]['initial_objective'])
    result['total_profile_seconds']=time.perf_counter()-started
    write_json(path,result)
    print(json.dumps(dict(profile=str(path),preferred_seed=result['preferred_seed'],fit_coverage=result['fit_coverage'],
        jacobian_seconds=result['candidates']['original_A']['evaluation_seconds'],directional_check=result['candidates']['original_A'].get('directional_check')),ensure_ascii=False),flush=True)
    return result


def run(project,base,output,document,profile_record,budget=600.,*,second=False):
    clock_path=output/'experiment_clock.json'
    start=time.perf_counter()
    if second:
        clock=json.loads(clock_path.read_text(encoding='utf-8'))
        if int(clock.get('initializations_run',1))>=2:raise RuntimeError('Two initialization budget already used')
        first=json.loads((output/'run_summary.json').read_text(encoding='utf-8'))
        remaining_wall=budget-(start-clock['start_monotonic'])-profile_record['total_profile_seconds']-30.
        if remaining_wall<2*profile_record['candidates']['original_A']['evaluation_seconds']:
            raise RuntimeError('Aggregate experiment budget exhausted')
        clock.update(initializations_run=2,second_initializer='stage1_retained',
            diagnosis='Stage1 rejected a mixed actual-root-acceleration cost as pose quality; objective unchanged, source groups remain fixed.')
        write_json(clock_path,clock)
        seed='stage1_retained';remaining_steps=max(0,24-first['actual_numerically_accepted_steps'])
    else:
        if clock_path.exists():raise FileExistsError('This bounded experiment has already started; preserve its candidates.')
        write_json(clock_path,dict(start_monotonic=start,wall_budget_seconds=budget,max_initializations=2,initializations_run=1))
        seed=profile_record['preferred_seed'];remaining_wall=budget-profile_record['total_profile_seconds']-30.;remaining_steps=24
    out=output/seed;out.mkdir(exist_ok=True)
    original_A=profile_record['candidates']['original_A']['measurement']
    main_log=out/'fit_progress.jsonl'
    print(json.dumps(dict(main_log=str(main_log),seed=seed,budget_seconds=budget,
        scope='87.5–92.5s context; joint root XYZ/SO3 and support hip/knee/ankle; no whole-program run')),flush=True)
    with main_log.open('a',encoding='utf-8',buffering=1)as log:
        problem,report_factor,x,info,replay,supports=make_problem(project,base,output,seed,document,log,arc_ids=profile_record['fixed_arc_ids'])
        problem.jacobian_seconds_hint=profile_record['candidates']['original_A']['evaluation_seconds']
        identity=dict(source_id=document['source_id'],source_sha256=document['source_sha256'],shot='s05',time_base=[1,12800],
            original_A_result_sha256=digest(base/'action_integration_1103/spin_s05/A/fitted/result.npz'),
            source_region_sha256=digest(output/'support_regions.json'),camera=info,
            fit_pts=sorted({r['pts']for r in document['observations']if r['role']=='fit'}),
            review_pts=profile_record['review_exposures'],body_pts=problem.reference.p['pts'][problem.body.frames].tolist())
        problem.stage_quality=StageQualityTrace(out/'cross_stage_quality.json',identity,original_A,
            lambda xx,v:measure(problem,report_factor,xx,v))
        remaining=max(0.,remaining_wall-(time.perf_counter()-start))
        if remaining<=2*problem.jacobian_seconds_hint:
            raise RuntimeError('Profile leaves insufficient budget for an actual joint update; reduce unrelated work first.')
        x,report=problem.solve(x,RunPolicy(wall_seconds=remaining,max_accepted=remaining_steps,max_outer=1,
            allow_penalty_growth=False))
        problem.reference.binding.x=x[:17].copy()
        problem.reference.new_refinements.append((problem.body.state,x[17:].copy()))
        before=time.perf_counter();packet=export_candidate(problem.reference.root,out,problem.reference,problem.camera,info,supports)
        apply_source_support(packet,supports,shot_id='s05')
        parent_path=(output/'original_A/fitted/result.npz')if second else base/('action_integration_1103/spin_s05/A/fitted/result.npz'if seed=='original_A'else'action_integration_1103/spin_s05/root_repair/A_pose/fitted/result.npz')
        parent=load(parent_path)
        for name in ['source_action_phase_mid','action_evidence_json']:
            if name in parent:packet[name]=parent[name].copy()
        observed=np.zeros((len(packet['pts']),2),bool)
        for row in document['observations']:
            if row['role']=='fit':observed[packet['pts']==row['pts'],row['foot']]=True
        packet.update(support_image_observations_json=np.array(json.dumps(document,sort_keys=True)),
            support_image_in_optimization=observed,ice_mode=np.array('finite_soft_objective'),
            ice_soft_policy_json=np.array(json.dumps(vars(problem.ice_policy),sort_keys=True)),
            ice_is_delivery_gate=np.array(False),costs_json=np.array(json.dumps(report['parts'])),
            stopping_reason_json=np.array(json.dumps(report['run_policy'],sort_keys=True)),
            support_refinement_parent=np.array(seed))
        np.savez_compressed(out/'fitted/result.npz',**packet)
        params=dict(load(out/'fit/candidate.npz'),source_region_sha256=np.array(digest(output/'support_regions.json')),
            initial_reference_kind=np.array(seed),source_region_file=np.array((PROGRAMME/'support_spin_1104.json').as_posix()))
        target=project/PROGRAMME/('spin_support_joint_1104.npz' if second else 'spin_support_1104.npz')
        np.savez_compressed(target,**params)
        outside=(packet['pts']<87.5*12800)|(packet['pts']>92.5*12800)
        unchanged={name:float(np.max(abs(packet[name][outside]-parent[name][outside])))for name in ['P_world_mid','Q_world_mid','local_R_mid']}
        actual=problem.accepted
        final=dict(schema='rinkspace.spin-support-bounded-run.v1',seed=seed,actual_numerically_accepted_steps=actual,
            retained_accepted_step=report.get('selection',{}).get('retained_step'),stage_update='updated'if actual else'no_update',
            solve_and_export_seconds=time.perf_counter()-start,export_seconds=time.perf_counter()-before,wall_budget_seconds=budget,
            initial_profile_seconds=profile_record['total_profile_seconds'],export_reservation_seconds=30.,
            max_initializations=2,initializations_run=2 if second else 1,run=report,quality=problem.stage_quality.record(),
            core_seconds=document['core_seconds'],context_seconds=document['context_seconds'],
            outside_context_replay_max_error=unchanged,metrics=summarize(packet,document),
            parameters=str(target),result=str(out/'fitted/result.npz'),accuracy_accepted=False,
            remaining='Visual/source and complete-target tradeoffs must be reviewed before extension.')
        summary_path=output/('run_stage2_summary.json'if second else'run_summary.json')
        write_json(summary_path,final)
        print(json.dumps(dict(event='bounded_support_complete',accepted_steps=actual,summary=str(summary_path),
            result=final['result'],parameters=str(target))),flush=True)
    return final


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--base-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['prepare','profile','run','second'],required=True);p.add_argument('--budget',type=float,default=600.)
    a=p.parse_args(argv);project,base,output=map(Path,[a.project,a.base_root,a.output]);output.mkdir(exist_ok=True)
    doc=prepare_regions(project,base,output)
    if a.stage=='prepare':return
    record=profile(project,base,output,doc)
    if a.stage=='run':run(project,base,output,doc,record,a.budget)
    if a.stage=='second':run(project,base,output,doc,record,a.budget,second=True)


if __name__=='__main__':main()
