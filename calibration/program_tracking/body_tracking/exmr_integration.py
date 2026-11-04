"""Common continuous-shot native body/skate evaluation for camera-frame priors.

All cases get the same source pixels, accepted S01 shape and fixed shoe geometry.
No case imports another network's articulation or world-root correction.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from .exmr_prior import load,write_json,digest
from .action_integration import ClipReference,initialize_clip,summarize
from .fullbody_state import camera_binding
from .skate_checkpoint import CAMERA_LOCATION_FIELDS,checkpoint_identity,validate_checkpoint_identity
from .skate_integration import JointContactProblem,build,restore_checkpoint_layers,source_evidence
from .skate_run_policy import RunPolicy
from .support_image import SupportImageFactors
from .support_refinement import measure
from .stage_quality import StageQualityTrace
from .skate_delivery import export_candidate
from .source_support_export import apply_source_support

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')
LABELS={'gvhmr':'fixed_original_GVHMR','exmr_prefoot':'ExMR_camera_before_FootMR','exmr_final':'ExMR_camera_after_FootMR_HandMR','exmr_hand_nofoot':'ExMR_BodyNet_HandMR_FootMR_off'}


CASE_FIELD='comparison_case_identity_json'
CASE_SCHEMA='rinkspace.network-comparison-case.v1'
CASE_FIELDS={'schema','label','prior_label','native_input_sha256','camera_prior_sha256',
    'initial_reference_sha256','case_input_sha256','compact_smplx_sha256',
    'shared_s01_parameters_sha256','support_document_sha256','action_evidence_sha256',
    'source_pts_sha256','source_frames','source_time_base','camera',
    'export_source_sha256','observations_sha256'}


def case_paths(project,base,output,label):
    return dict(case=output/'common'/label,
        native=base/'action_integration_1103/source_cache/spin_s05.native50.npz',
        prior=base/'action_integration_1103/spin_s05/prior/camera_prior.npz' if label=='gvhmr' else output/('exmr_body_hand_nofoot.npz' if label=='exmr_hand_nofoot' else label+'.npz'),
        shared=project/PROGRAMME/'skate_s01_1103.npz',
        support=output/'support_and_free_shoe_regions.json' if label=='exmr_hand_nofoot' else project/PROGRAMME/'support_spin_1104.json',
        evidence=project/PROGRAMME/'action_evidence_1103.json')


def case_identity(project,base,output,label,info,*,initial=None,prior_sha256=None):
    paths=case_paths(project,base,output,label);case=paths['case']
    pts=np.asarray(load(case/'input.npz')['pts'],dtype='<i8')
    identity=dict(schema=CASE_SCHEMA,label=label,prior_label=LABELS[label],
        native_input_sha256=digest(paths['native']),
        camera_prior_sha256=prior_sha256 or digest(paths['prior']),
        initial_reference_sha256=digest(initial or case/'initial.npz'),
        case_input_sha256=digest(case/'input.npz'),compact_smplx_sha256=digest(case/'compact_smplx.npz'),
        export_source_sha256=digest(case/'export_source.npz'),observations_sha256=digest(case/'observations.npz'),
        shared_s01_parameters_sha256=digest(paths['shared']),support_document_sha256=digest(paths['support']),
        action_evidence_sha256=digest(paths['evidence']),source_pts_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),
        source_frames=len(pts),source_time_base=[1,12800],
        camera={k:v for k,v in info.items() if k not in CAMERA_LOCATION_FIELDS})
    if label=='exmr_hand_nofoot':
        from .hand_state import SCHEMA,MEAN
        identity.update(schema='rinkspace.network-comparison-case.v2',hand_pose_schema_version=SCHEMA,
            hand_mean_convention=MEAN,relation_document_sha256=digest(output/'grasp_relations.json'),
            geometry_semantics_sha256=digest(output/'palm_grip_geometry.json'))
    return identity


def read_case_identity(saved):
    if CASE_FIELD not in saved:raise ValueError('Missing comparison case identity')
    try:
        value=np.asarray(saved[CASE_FIELD])
        if value.shape!=():raise ValueError('Expected scalar case identity')
        result=json.loads(str(value))
    except (TypeError,ValueError) as exc:raise ValueError('Invalid comparison case identity') from exc
    extra={'hand_pose_schema_version','hand_mean_convention','relation_document_sha256','geometry_semantics_sha256'}
    schema=result.get('schema') if isinstance(result,dict) else None
    expected_fields=CASE_FIELDS|extra if schema=='rinkspace.network-comparison-case.v2' else CASE_FIELDS
    if not isinstance(result,dict) or set(result)!=expected_fields or schema not in (CASE_SCHEMA,'rinkspace.network-comparison-case.v2'):
        raise ValueError('Incomplete comparison case identity')
    return result


def validate_case_identity(saved,expected):
    actual=read_case_identity(saved)
    if actual!=expected:raise ValueError('Comparison case identity mismatch; wrong case or changed inputs')
    return actual


def resolve_initial_reference(project,saved):
    identity=read_case_identity(saved)
    if 'initial_reference' not in saved or 'initial_reference_sha256' not in saved:
        raise ValueError('Missing archived initial reference path/SHA256')
    relative=Path(str(saved['initial_reference']))
    project=Path(project).resolve();path=(project/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(project):
        raise ValueError('Initial reference must be a project-relative archive')
    expected=str(saved['initial_reference_sha256'])
    if expected!=identity['initial_reference_sha256'] or not path.is_file() or digest(path)!=expected:
        raise ValueError('Initial reference SHA256 mismatch')
    return path


def replay_errors(actual,saved):
    if not np.array_equal(actual['pts'],saved['pts']):raise ValueError('Replay source PTS mismatch')
    for packet in (actual,saved):
        if 'source_pts' in packet and not np.array_equal(packet['source_pts'],packet['pts']):
            raise ValueError('Replay source PTS aliases differ')
    errors={};passed=True
    for field in ('P_world_mid','Q_world_mid','local_R_mid','native22_world_mid',
                  'COCO_world_mid','COCO_uv','equipment_uv'):
        a,b=np.asarray(actual[field]),np.asarray(saved[field])
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid replay comparison field: '+field)
        errors[field]=float(np.max(np.abs(a-b),initial=0.))
        passed &= errors[field] <= (1e-6 if field.endswith('_uv') else 2e-10)
    if 'hand_local_R_mid' in actual or 'hand_local_R_mid' in saved:
        if 'hand_local_R_mid' not in actual or 'hand_local_R_mid' not in saved:raise ValueError('Replay hand channel missing')
        errors['hand_local_R_mid']=float(np.max(np.abs(actual['hand_local_R_mid']-saved['hand_local_R_mid'])))
        passed &= errors['hand_local_R_mid']<=2e-10
    for field in ('grasp_palm_world_mid','grasp_segment_world_mid','palm_to_segment_m_mid'):
        if field in actual or field in saved:
            if field not in actual or field not in saved:raise ValueError('Replay grasp channel missing: '+field)
            errors[field]=float(np.max(np.abs(actual[field]-saved[field])));passed &= errors[field]<=2e-10
    for name,part in [('shoe',slice(0,14)),('blade',slice(14,None))]:
        errors[name+'_native_uv']=float(np.max(np.abs(actual['equipment_uv'][:,:,part]-saved['equipment_uv'][:,:,part]),initial=0.))
    return dict(passed=bool(passed),source_pts_equal=True,source_frames=len(actual['pts']),
        maximum_absolute_errors=errors,world_rotation_tolerance=2e-10,native_pixel_tolerance=1e-6,
        projection='same five-node native Brown/rolling-exposure projection; no pose interpolation after export')


def make_case(project,base,output,label,log=None):
    case=output/'common'/label
    paths=case_paths(project,base,output,label);prior=paths['prior']
    document=json.loads(paths['support'].read_text(encoding='utf8'))
    evidence=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf8'))['clips']['spin_s05']
    ref,camera,info=initialize_clip(project,base,case,
        base/'action_integration_1103/source_cache/spin_s05.native50.npz',prior,'s05',
        heldout_pts=evidence.get('heldout_pts',[]),prior_label=LABELS[label])
    identity=case_identity(project,base,output,label,info)
    identity_path=case/'comparison_case_identity.json'
    if identity_path.exists():
        previous=json.loads(identity_path.read_text(encoding='utf8'))
        validate_case_identity({CASE_FIELD:np.array(json.dumps(previous,sort_keys=True))},identity)
    else:write_json(identity_path,identity)
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    review={r['pts'] for r in document['observations'] if r['role']=='review'}
    supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],
        records=[r for r in evidence.get('supports',[])+document['supports'] if r['pts'] not in review])
    # One continuous state over the complete original clip. No internal windows,
    # fixed-body boundaries or original-GVHMR articulation anchors are introduced.
    half=abs(float(camera.p['rig'][37]))/2
    seconds=tuple(ref.p['pts'][[0,-1]]/12800.+np.array([-half,half]))
    problem=JointContactProblem(ref,camera,case/'observations.npz',seconds,supports,case/'fit',log,
        'continuous_shot',optimize_shared=False)
    ref.footwear_observations['support_regions']=document['observations']
    problem.support_image=SupportImageFactors(problem,document)
    report_factor=SupportImageFactors(problem,document,include_review=True,arc_ids=problem.support_image.arc_ids)
    x=np.r_[ref.binding.x,np.zeros(problem.body.size)]
    return problem,report_factor,x,info,supports,document


def save_packet(problem,info,supports,document,out,*,status,report=None):
    packet=export_candidate(problem.reference.root,out,problem.reference,problem.camera,info,supports)
    apply_source_support(packet,supports,shot_id='s05')
    observed=np.zeros((len(packet['pts']),2),bool)
    if status in ('joint_fit','parameter_replay'):
        for row in document['observations']:
            if row['role']=='fit' and row.get('optimization_enabled',True):observed[packet['pts']==row['pts'],row['foot']]=True
    packet.update(support_image_observations_json=np.array(json.dumps(document,sort_keys=True)),
        support_image_in_optimization=observed,ice_mode=np.array('finite_soft_objective'),
        ice_soft_policy_json=np.array(json.dumps(vars(problem.ice_policy),sort_keys=True)),
        ice_is_delivery_gate=np.array(False),comparison_stage=np.array(status),
        network_pose_origin=np.array(problem.reference.hashes['camera_prior_sha256']),
        original_gvhmr_pose_anchor=np.array(False),
        **{CASE_FIELD:problem.reference.fixed_checkpoint_identity[CASE_FIELD]})
    if report:packet['costs_json']=np.array(json.dumps(report['parts']))
    np.savez_compressed(out/'fitted/result.npz',**packet)
    return packet


def profile(project,base,output,label,*,return_case=False):
    case=output/'common'/label
    if (case/'profile.json').exists():raise FileExistsError('Existing profile is preserved')
    case.mkdir(parents=True,exist_ok=True)
    with (case/'profile_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem,review,x,info,supports,document=make_case(project,base,output,label,log)
        start=time.perf_counter();value=problem.evaluate(x,True);elapsed=time.perf_counter()-start
        metrics=measure(problem,review,x,value)
        zero=save_packet(problem,info,supports,document,case/'initial',status='common_initial')
        record=dict(label=label,prior_label=LABELS[label],jacobian_seconds=elapsed,
            **{CASE_FIELD:str(problem.reference.fixed_checkpoint_identity[CASE_FIELD])},
            source_seconds=list(problem.body.state.seconds),source_frames=problem.body.n,
            variables=problem.body.size,shared_parameters_optimized=False,
            shared_s01_geometry_preserved=True,original_gvhmr_pose_anchor=False,
            own_network_correction_prior='same finite engineering correction priors for each explicit network input',
            initial_objective=float(value['residual']@value['residual']/2),metrics=metrics,
            initial_physical=summarize(zero,document),ice_policy=vars(problem.ice_policy),
            support_foot_association='same source anatomical-right association; source inference, not human confirmed',
            support_image_fit_pts=sorted({r['pts'] for r in document['observations'] if r['role']=='fit'}),
            review_pts=sorted({r['pts'] for r in document['observations'] if r['role']=='review'}),
            native_camera=info,accuracy_accepted=False)
        write_json(case/'profile.json',record)
        print(json.dumps(dict(event='profile_complete',label=label,seconds=elapsed,frames=problem.body.n,
            variables=problem.body.size,original_gvhmr_pose_anchor=False)),flush=True)
    if return_case:return record,(problem,review,x,info,supports,document)
    return record


def fit(project,base,output,label,budget=240.,max_steps=6,*,prepared_case=None):
    case=output/'common'/label
    if (case/'fitted/result.npz').exists():raise FileExistsError('Existing result is preserved')
    prof=json.loads((case/'profile.json').read_text(encoding='utf8'))
    baseline=json.loads((output/'common/gvhmr/profile.json').read_text(encoding='utf8'))['metrics']
    main_log=case/'fit_progress.jsonl'
    print(json.dumps(dict(main_log=str(main_log),label=label,wall_budget_seconds=budget,max_accepted=max_steps,
        full_continuous_clip_seconds=[76,95],no_ice_AL=True,shared_shape_fixed=True)),flush=True)
    with main_log.open('a',encoding='utf8',buffering=1) as log:
        if prepared_case is None:
            problem,review,x,info,supports,document=make_case(project,base,output,label,log)
        else:
            problem,review,x,info,supports,document=prepared_case
            problem.log=log;problem.body.log=log
        validate_case_identity(prof,read_case_identity(problem.reference.fixed_checkpoint_identity))
        problem.jacobian_seconds_hint=prof['jacobian_seconds']
        identity=dict(label=label,prior_sha256=problem.reference.hashes['camera_prior_sha256'],
            fixed_gvhmr_profile_sha256=digest(output/'common/gvhmr/profile.json'),
            source_region_sha256=digest(project/PROGRAMME/'support_spin_1104.json'),
            continuous_seconds=[76,95],network_static_contact_truth=False)
        problem.stage_quality=StageQualityTrace(case/'cross_stage_quality.json',identity,baseline,
            lambda xx,v:measure(problem,review,xx,v))
        start=time.perf_counter()
        problem.output.mkdir(parents=True,exist_ok=True)
        attempt=problem.output/'initial_attempt.npz'
        if not attempt.exists():np.savez_compressed(attempt,**problem.checkpoint_fields(x))
        try:
            x,report=problem.solve(x,RunPolicy(wall_seconds=budget,max_accepted=max_steps,max_outer=1,
                allow_penalty_growth=False))
        except Exception as exc:
            write_json(case/'attempt_summary.json',dict(status='attempt_only',final_candidate=False,
                replay_passed=False,error=str(exc),initial_attempt=str(attempt),
                last_attempt=str(problem.output/'last_attempt.npz')))
            raise
        candidate=case/'fit/candidate.npz'
        if not candidate.is_file() or not report.get('candidate_checkpoint'):
            raise ValueError('No final candidate; attempts preserved without replay claim')
        retained=load(candidate)
        validate_case_identity(retained,read_case_identity(problem.reference.fixed_checkpoint_identity))
        if not np.array_equal(retained['x'],x):raise ValueError('Returned controls differ from final candidate')
        problem.reference.binding.x=x[:17].copy()
        problem.reference.new_refinements.append((problem.body.state,x[17:].copy()))
        packet=save_packet(problem,info,supports,document,case,status='joint_fit',report=report)
        record=dict(label=label,wall_budget_seconds=budget,max_accepted=max_steps,
            accepted_steps=problem.accepted,retained_step=report.get('selection',{}).get('retained_step'),
            solve_and_export_seconds=time.perf_counter()-start,run=report,
            quality=problem.stage_quality.record(),physical=summarize(packet,document),
            continuous_source_seconds=[76,95],source_exposures=len(packet['pts']),
            source_time_base=[1,12800],ice_is_delivery_gate=False,accuracy_accepted=False,
            result=str(case/'fitted/result.npz'))
        write_json(case/'run_summary.json',record)
        # Parameters and their exact network-derived reference, not a borrowed GVHMR snapshot.
        params=load(case/'fit/candidate.npz')
        params.update(initial_reference=np.array((PROGRAMME/f'spin_models_{label}_initial_1104.npz').as_posix()),
            initial_reference_sha256=np.array(digest(case/'initial.npz')),
            camera_prior_sha256=np.array(problem.reference.hashes['camera_prior_sha256']),
            original_gvhmr_pose_anchor=np.array(False))
        for target in [project/PROGRAMME/f'spin_models_{label}_1104.npz',project/PROGRAMME/f'spin_models_{label}_initial_1104.npz']:
            if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
            if target.exists():raise FileExistsError('Existing final model parameters are preserved')
        np.savez_compressed(project/PROGRAMME/f'spin_models_{label}_1104.npz',**params)
        (project/PROGRAMME/f'spin_models_{label}_initial_1104.npz').write_bytes((case/'initial.npz').read_bytes())
        print(json.dumps(dict(event='comparison_joint_fit_complete',label=label,accepted=problem.accepted,
            retained=record['retained_step'],result=record['result'])),flush=True)
    return record



def replay(project,base,output,label,checkpoint=None,replay_output=None):
    """Replay this case's archived initial and controls; never rebuild a network prior."""
    from types import SimpleNamespace
    project,base,output=map(lambda p:Path(p).resolve(),(project,base,output))
    paths=case_paths(project,base,output,label);case=paths['case']
    checkpoint=Path(checkpoint or project/PROGRAMME/f'spin_models_{label}_1104.npz').resolve()
    saved=load(checkpoint);identity=read_case_identity(saved)
    if identity['label']!=label:raise ValueError('Comparison case identity mismatch: wrong label')
    initial_path=resolve_initial_reference(project,saved)
    recorded=json.loads((case/'comparison_case_identity.json').read_text(encoding='utf8'))
    validate_case_identity(saved,recorded)
    p=load(case/'input.npz');initial=load(initial_path);native=load(paths['native'])
    native_pts=native['pts'] if 'pts' in native else native['source_pts']
    if not np.array_equal(initial['pts'],p['pts']) or not np.array_equal(native_pts,p['pts']):
        raise ValueError('Archived initial/native source PTS mismatch')
    camera,info,_,pose=camera_binding(project,SimpleNamespace(p=p),shot_id='s05')
    expected=case_identity(project,base,output,label,info,initial=initial_path,
        prior_sha256=recorded['camera_prior_sha256'])
    validate_case_identity(saved,expected)
    target=Path(replay_output or case/('replay_'+digest(checkpoint)[:12])).resolve()
    if target==case or target==case/'fitted' or target.is_relative_to(case/'fitted') or case.is_relative_to(target):
        raise ValueError('Replay output must be separate from the original result')
    if target.exists():raise FileExistsError('Existing replay is preserved; choose a new replay output')
    target.mkdir(parents=True)
    # build supplies only the identical fixed generic geometry/attachment. Its
    # historical at() is never used as the new case body reference.
    shared,_,_,_=build(project,base,target/'shared')
    seed=load(paths['shared']);validate_checkpoint_identity(seed,shared.fixed_checkpoint_identity)
    shared.binding.x=seed['x'][:17].copy();shared.binding.enable_collar_constraints()
    if not np.array_equal(saved['x'][:17],shared.binding.x):
        raise ValueError('Replay shared S01 parameters changed')
    p=camera.p;p['camera_rotation']=pose['rotation'];p['camera_centre']=pose['centre'];p['camera_values']=pose['values']
    ref=ClipReference(shared,case,p,initial)
    if label=='exmr_hand_nofoot':
        ref.grasp_geometry=json.loads((output/'palm_grip_geometry.json').read_text(encoding='utf8'))
        ref.grasp_relations=json.loads((output/'grasp_relations.json').read_text(encoding='utf8'))
    ref.fixed_checkpoint_identity=checkpoint_identity(ref.binding,project,info)
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(expected,sort_keys=True))
    validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    validate_case_identity(saved,expected)
    ref.hashes=dict(native_input_sha256=expected['native_input_sha256'],
        camera_prior_sha256=expected['camera_prior_sha256'],
        shared_s01_parameters_sha256=expected['shared_s01_parameters_sha256'])
    restore_checkpoint_layers(ref,saved,include_primary=True)
    document=json.loads(paths['support'].read_text(encoding='utf8'))
    evidence=json.loads(paths['evidence'].read_text(encoding='utf8'))['clips']['spin_s05']
    review={r['pts'] for r in document['observations'] if r['role']=='review'}
    supports=source_evidence(case,shot_id='s05',source_pts=p['pts'],
        records=[r for r in evidence.get('supports',[])+document['supports'] if r['pts'] not in review])
    ice=json.loads(str(saved['ice_soft_policy_json']))
    problem=SimpleNamespace(reference=ref,camera=camera,ice_policy=SimpleNamespace(**ice))
    actual=save_packet(problem,info,supports,document,target,status='parameter_replay')
    original=case/'fitted/result.npz'
    if 'result_reference' in saved:
        relative=Path(str(saved['result_reference']));original=(case/relative).resolve()
        if relative.is_absolute() or not original.is_relative_to(case.resolve()):raise ValueError('Result reference escapes the identified case')
        if 'result_reference_sha256' not in saved or not original.is_file() or digest(original)!=str(saved['result_reference_sha256']):raise ValueError('Result reference SHA256 mismatch')
    result=load(original);validate_case_identity(result,expected)
    report=replay_errors(actual,result)
    report.update(checkpoint_sha256=digest(checkpoint),initial_reference_sha256=digest(initial_path),
        original_result_sha256=digest(original),replayed_result=str(target/'fitted/result.npz'),
        no_fit_performed=True,no_network_performed=True,comparison_case_identity=expected)
    write_json(target/'replay.json',report)
    if not report['passed']:raise ValueError('Parameter replay differs; see replay.json')
    print(json.dumps(dict(event='comparison_replay_verified',label=label,report=str(target/'replay.json'))),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--base-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--stage',required=True,choices=['profile','fit','compare','replay']);p.add_argument('--label',required=True,choices=list(LABELS))
    p.add_argument('--budget',type=float,default=240);p.add_argument('--max-steps',type=int,default=6)
    p.add_argument('--checkpoint');p.add_argument('--replay-output')
    a=p.parse_args(argv);project,base,output=map(Path,[a.project,a.base_root,a.output])
    if a.stage=='profile':profile(project,base,output,a.label)
    elif a.stage=='fit':fit(project,base,output,a.label,a.budget,a.max_steps)
    elif a.stage=='compare':
        _,prepared_case=profile(project,base,output,a.label,return_case=True)
        fit(project,base,output,a.label,a.budget,a.max_steps,prepared_case=prepared_case)
    else:replay(project,base,output,a.label,a.checkpoint,a.replay_output)

if __name__=='__main__':main()