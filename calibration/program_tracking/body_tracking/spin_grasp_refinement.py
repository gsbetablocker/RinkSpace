"""One source-clock S05 closed-chain refinement; reuse the joint body/skate solver."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,time,hashlib,re
import numpy as np
from rinkspace.storage import AssetStore
from rinkspace.path_policy import guard_project_write
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .action_integration import initialize_clip,ClipReference
from .exmr_integration import LABELS,CASE_FIELD,case_identity,save_packet,replay_errors
from .skate_integration import JointContactProblem,restore_checkpoint_layers,source_evidence
from .skate_run_policy import RunPolicy
from .support_image import SupportImageFactors
from .skate_physics import fk_state,NORMALIZATION
from scipy.sparse import diags
from .spin_grasp_factors import SpinGraspFactors,prepare_geometry,reassociate_support,export_grasp
from .stage_quality import StageQualityTrace

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')


def archive_paths(project,output,parameter_stem=None,*,persist=False):
    """Keep each output's parameter archives distinct and record their names."""
    project=Path(project).resolve();output=Path(output).resolve();manifest=output/'archive_names.json'
    recorded=json.loads(manifest.read_text(encoding='utf8')) if manifest.exists() else None
    if recorded is not None:
        if recorded.get('schema')!='rinkspace.spin-grasp-archive-names.v1':raise ValueError('Unsupported grasp archive names')
        if parameter_stem is not None and parameter_stem!=recorded['parameter_stem']:raise ValueError('Parameter stem differs from this output archive record')
        parameter_stem=recorded['parameter_stem']
    if parameter_stem is None:
        parameter_stem=re.sub(r'[^a-z0-9_-]+','_',output.name.lower()).strip('_-')
        if not parameter_stem:parameter_stem='spin_grasp_'+hashlib.sha256(output.name.encode('utf8')).hexdigest()[:12]
    reserved={'con','prn','aux','nul'}|{f'{name}{i}' for name in ('com','lpt') for i in range(1,10)}
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',parameter_stem) or parameter_stem.lower() in reserved:
        raise ValueError('Parameter stem must be a safe filename stem using letters, digits, underscores or hyphens')
    # The retained 1104 output predates the per-output naming record.
    initial='spin_grasp_initial_1104' if parameter_stem=='spin_grasp_1104' else parameter_stem+'_initial'
    finished='spin_grasp_finished_1104' if parameter_stem=='spin_grasp_1104' else parameter_stem+'_finished'
    paths={key:guard_project_write(project/PROGRAMME/(name+'.npz'),project_root=project)
           for key,name in [('parameter',parameter_stem),('initial_parameter',initial),('finished_parameter',finished)]}
    record=dict(schema='rinkspace.spin-grasp-archive-names.v1',parameter_stem=parameter_stem,
                **{key:path.relative_to(project).as_posix() for key,path in paths.items()})
    if recorded is not None and recorded!=record:raise ValueError('Archive names differ from the recorded project-relative paths')
    if persist and recorded is None:
        guard_project_write(manifest,project_root=project)
        with manifest.open('x',encoding='utf8') as stream:stream.write(json.dumps(record,ensure_ascii=False,indent=2))
    return dict(parameter_stem=parameter_stem,manifest=manifest,**paths)


def check_run_destination(project,output,prior_label,parameter_stem=None):
    paths=archive_paths(project,output,parameter_stem)
    for path in (paths['parameter'],paths['finished_parameter'],output/'common'/prior_label/'run_summary.json',output/'active_candidate.json'):
        if path.exists():raise FileExistsError(f'Existing grasp candidate is preserved; choose a new output and parameter stem: {path}')
    return paths


def prepare_sources(project,output):
    if (output/'grasp_relations.json').exists() and not (output/'source_review/source_findings.json').exists():
        document=json.loads((output/'support_and_free_shoe_regions.json').read_text(encoding='utf8'))
        for row in document['observations']:
            footprint=row.get('image_footprint',{})
            if footprint.get('source_frame_sha256'):
                frame=AssetStore(project).resolve(footprint['source_frame'],must_exist=True)
                if digest(frame)!=footprint['source_frame_sha256']:raise ValueError('Source image SHA mismatch')
        return document,json.loads((output/'grasp_relations.json').read_text(encoding='utf8'))
    findings=json.loads((output/'source_review/source_findings.json').read_text(encoding='utf8'))
    doc_path=output/'support_and_free_shoe_regions.json';doc=json.loads(doc_path.read_text(encoding='utf8'))
    original=output/'admission_original_60.json'
    if not original.exists():original.write_bytes(doc_path.read_bytes())
    frames=json.loads((output/'source_review/native_frames.json').read_text(encoding='utf8'))
    mapping={int(r['pts']):r for r in frames['rows']};body=[];visibility=[]
    if 'early_and_release_regions' not in doc:
        for row in findings['new_regions']:
            pt=int(row['pts']);frame=mapping[pt];assert digest(Path(frame['frame']))==frame['sha256']
            role='review' if row['role']=='review_candidate' else 'fit'
            for key,foot in [('support_right_foot1',1),('free_left_foot0',0)]:
                source=row.get(key,{});region=source.get('shoe_body');vis=source.get('visibility','uncertain')
                if region is None or vis in ('uncertain','self_occluded','externally_occluded','out_of_frame'):continue
                half=np.asarray(region['half_extent_uv'],float);reliability=.4 if 'blurred' in vis else .65
                obs=copy.deepcopy(doc['observations'][0]);obs.update(id=f'grasp_{pt}_{foot}_shoe_body',pts=pt,foot=foot,foot_track='support_track' if foot==1 else 'free_track',centre_uv=region['center_uv'],region_shape_px2=np.diag(2*half**2).tolist(),role=role,reliability=reliability,visibility='visible',contact_state='unknown',
                    provenance='native source review source_findings.json; fixed episode anatomical hypothesis, medium confidence',
                    source_visibility=vis,source_uncertainty='wide finite shoe-body region, not blade/contact/ankle point',human_confirmed=False,
                    image_footprint=dict(source_frame=frame['frame'],source_frame_sha256=frame['sha256'],native_dimensions=[1920,1080],region_rectangle_half_extent_uv=half.tolist()))
                doc['observations'].append(obs)
        doc['early_and_release_regions']=dict(source_findings_sha256=digest(output/'source_review/source_findings.json'),old_observations_unchanged=True,no_new_discrete_support=True)
        write_json(doc_path,doc)
    for row in findings['new_regions']:
        pt=int(row['pts']);role='review' if row['role']=='review_candidate' else 'fit';frame=mapping[pt]
        for name,joint,coco in [('left_wrist',20,9),('left_elbow',18,7),('left_shoulder',16,5)]:
            source=row.get(name,{});vis=source.get('visibility','uncertain');reg=source.get('region')
            normalized='partially_visible' if 'visible' in vis and (vis!='visible') else vis
            visibility.append(dict(pts=pt,part=name,coco_joint=coco,visibility=normalized,role=role,source_frame_sha256=frame['sha256'],source_reason=source.get('reason','')))
            if reg is not None and normalized in ('visible','partially_visible'):
                half=np.asarray(reg['half_extent_uv'],float)
                body.append(dict(id=f'grasp_{pt}_{name}',source_id=doc['source_id'],source_sha256=doc['source_sha256'],shot_id='s05',time_base=[1,12800],pts=pt,native_joint=joint,coco_joint=coco,anatomical_part=name,role=role,visibility=normalized,centre_uv=reg['center_uv'],region_shape_px2=np.diag(2*half**2).tolist(),reliability=.4 if 'blurred' in vis else .65,objective_weight=4.,source_frame_sha256=frame['sha256'],provenance='direct native frame development review; wide joint/cuff region, not precise skin/material point'))
    relation=dict(schema='rinkspace.spin-grasp-relations.v1',source_id=doc['source_id'],source_sha256=doc['source_sha256'],shot_id='s05',time_base=[1,12800],
        episode_identity=dict(support_foot=1,free_foot=0,grip_hand='left',confidence='medium, fixed whole-episode anatomical hypothesis'),
        grip_interval=dict(start_pts=1111040,end_pts=1183488,hand='left',foot=0,state='held',role='fit',evidence_pts=[1111040,1121792,1146368,1177344,1183488],grip_region_id='left_rear_sole_blade_web_finite_segment',provenance='source-backed sustained relation between sparse frames, not per-frame manual truth'),
        grip_unknown_intervals=[[1080320,1111040],[1183488,1187840]],released_from_pts=1187840,
        support_intervals=[dict(start_pts=1080320,end_pts=1183488,foot=1,weight=1.,source_evidence_pts=[1080320,1090560,1111040,1121792,1146368,1177344,1183488],provenance='sparse source-backed continuous right-foot support hypothesis; no fixed world point'),dict(start_pts=1183488,end_pts=1187840,foot=1,weight=.25,provenance='release/downward transition, weaker inferred support; not extended into kneeling')],
        phase_boundaries_seconds=[82.,84.4,86.,86.8,92.46,92.8,94.8],body_regions=body,visibility=visibility,
        source_findings_sha256=digest(output/'source_review/source_findings.json'),
        factor_parameters=dict(shoe_region_weight_multiplier=16.,grip=dict(contact_radius_m=.035,residual_scale_m=.06,objective_weight=8.,robust_delta=2.),local_motion=dict(scale_m_s2=20.,weight=.8,robust_delta=3.),continuous_support=dict(deadzone_m=.025,scale_m=.08,weight=20.)),
        parameter_meaning='finite first-integration engineering scales/weights, not calibrated probabilities or acceptance thresholds; 35mm tube covers finite central palm patch and uncertain rear equipment material',
        not_grip_truth=True,not_per_frame_manual_contact=True)
    target=output/'grasp_relations.json'
    if not target.exists():write_json(target,relation)
    else:relation=json.loads(target.read_text(encoding='utf8'))
    return doc,relation


def visibility_ledger(case,doc,relations):
    path=case/'observations.npz';obs=load(path);original=case/'observations_original_detector.npz'
    if not original.exists():original.write_bytes(path.read_bytes())
    changes=[];pts=obs['pts'];review={int(r['pts']) for r in doc['observations'] if r['role']=='review'}|{r['pts'] for r in relations['body_regions'] if r['role']=='review'}
    for pt in review:
        i=np.flatnonzero(pts==pt)
        if len(i):obs['use'][max(0,i[0]-1):i[0]+2]=False
    measured={(r['pts'],r['coco_joint']) for r in relations['body_regions'] if r['role']=='fit'}
    for row in relations['visibility']:
        key=(row['pts'],row['coco_joint'])
        if row['role']=='fit' and (row['visibility'] in ('self_occluded','externally_occluded','out_of_frame') or key in measured):
            for i in np.flatnonzero(pts==row['pts']):
                changes.append(dict(pts=int(pts[i]),joint=row['coco_joint'],previous_use=bool(obs['use'][i,row['coco_joint']]),reason='source region replaces detector point' if key in measured else row['visibility']))
                obs['use'][i,row['coco_joint']]=False
    np.savez_compressed(path,**obs);write_json(case/'visibility_ledger.json',dict(changes=changes,review_pts=sorted(review),original_detector_sha256=digest(original),raw_detector_values_unchanged=True,scope='only source-evidenced exact parts/exposures and explicit review neighbourhoods'))


def make_problem(project,base,output,prior_label,context,log):
    doc,relations=prepare_sources(project,output);case=output/'common'/prior_label
    prior=output/'exmr_body_hand_nofoot.npz' if prior_label=='exmr_hand_nofoot' else output/(prior_label+'.npz')
    ref,camera,info=initialize_clip(project,base,case,base/'action_integration_1103/source_cache/spin_s05.native50.npz',prior,'s05',prior_label=LABELS[prior_label])
    visibility_ledger(case,doc,relations)
    geometry_path=output/'palm_grip_geometry.json'
    if not geometry_path.exists():
        store=AssetStore(project,data_root=base.parents[1]);model=store.resolve(project/'environments/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz',must_exist=True)
        write_json(geometry_path,prepare_geometry(ref,digest(model)))
    ref.grasp_geometry=json.loads(geometry_path.read_text(encoding='utf8'));ref.grasp_relations=relations
    identity=case_identity(project,base,output,prior_label,info);write_json(case/'comparison_case_identity.json',identity)
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    old=json.loads((project/PROGRAMME/'action_evidence_1103.json').read_text(encoding='utf8'))['clips']['spin_s05']
    review={r['pts'] for r in doc['observations'] if r['role']=='review'}
    supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=[r for r in old.get('supports',[])+doc['supports'] if r['pts'] not in review])
    # Continuous inferred support replaces its discrete height duplicates; source labels stay sparse in export.
    evidence=[r for r in supports if not 1080320<=r['pts']<=1187840]
    problem=JointContactProblem(ref,camera,case/'observations.npz',context,evidence,case/'fit',log,'spin_grasp_v1',optimize_shared=False)
    runtime=copy.deepcopy(doc)
    for r in runtime['observations']:r['objective_weight']*=relations['factor_parameters']['shoe_region_weight_multiplier']
    problem.support_image=SupportImageFactors(problem,runtime);problem.action_factors=SpinGraspFactors(problem,relations)
    review_factor=SupportImageFactors(problem,runtime,include_review=True,arc_ids=problem.support_image.arc_ids)
    problem.checkpoint_extra=dict(hand_pose_schema_version=np.array('rinkspace.hand-local-R.v1'),grasp_relations_sha256=np.array(digest(output/'grasp_relations.json')),grasp_geometry_sha256=np.array(digest(geometry_path)))
    initial=np.r_[ref.binding.x,np.zeros(problem.body.size)]
    return problem,review_factor,initial,info,supports,doc


def direction_profile(problem,x,value):
    rng=np.random.default_rng(110405);direction=np.r_[np.zeros(17),rng.normal(size=problem.body.size)];direction/=np.linalg.norm(direction)
    eps=1e-5;a=problem.evaluate(x+eps*direction)['residual'];b=problem.evaluate(x-eps*direction)['residual'];fd=(a-b)/(2*eps);analytic=value['jac']@direction
    error=float(np.max(np.abs(fd-analytic)));rel=float(np.linalg.norm(fd-analytic)/max(1.,np.linalg.norm(fd)))
    result=dict(mixed_direction=dict(maximum_absolute_error=error,relative_l2_error=rel,includes_root_support_free_leg_and_grasp_arm=True),groups={})
    action=problem.action_factors.evaluate(x[17:],True);support=problem.support_image.evaluate(x[17:],True);base=problem.body.evaluate(x[17:],True)
    for name,axis in [('normal',np.array([0.,0.,1.])),('line_of_sight',None)]:
        if axis is None:
            i=int(np.argmin(abs(problem.reference.times-14.)));P=problem.reference.at(problem.reference.times[[i]])['P'][0]
            C=problem.camera.p['camera_centre'][i]*float(problem.camera.p['metres_per_scene_unit']);axis=P-C;axis/=np.linalg.norm(axis)
        v=np.zeros_like(x);controls=v[17:].reshape(problem.body.state.m,problem.body.state.d)
        for j in range(3):controls[:,np.flatnonzero(problem.body.state.active==j)[0]]=axis[j]
        entry=dict(body=float(.02*base['residual']@(base['jac']@v[17:])),support_image=float(support['residual']@(support['jac']@v)))
        for group,sl in action['block_slices'].items():entry[group]=float(action['residual'][sl]@(action['jac'][sl]@v))
        # Existing finite ice rows have an exact analytic body constraint chain.
        raw=problem.contact(x,True);from .ice_soft import ice_soft_rows
        ice=ice_soft_rows(raw,problem.ice_policy);J=diags(ice['derivative_per_margin_m']*NORMALIZATION)@raw['body_jac'][ice['indices']]
        entry['ice']=float(ice['residual']@(J@v[17:]));result['groups'][name]=entry
    problem.cache=problem.cache_x=None
    return result


def profile(project,base,output,prior_label,context):
    from scipy.sparse import diags
    started=time.perf_counter();case=output/'common'/prior_label;case.mkdir(parents=True,exist_ok=True)
    with (case/'profile_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem,review,x,info,supports,doc=make_problem(project,base,output,prior_label,context,log)
        tick=time.perf_counter();initial=problem.evaluate(x,False)
        write_json(case/'initial_domain.json',dict(gate=initial['gate'],parts=initial['raw_data_parts'],action_metrics=initial['action_metrics']))
        if initial['gate']['collar_unknown_sections']:raise RuntimeError('Initial hand-only collar unknown; preserve and diagnose before selecting a defined prior')
        initial_packet=save_packet(problem,info,supports,doc,case/'initial',status='common_initial')
        t=time.perf_counter();value=problem.evaluate(x,True);jseconds=time.perf_counter()-t
        directions=direction_profile(problem,x,value)
        record=dict(jacobian_seconds=jseconds,total_profile_seconds=time.perf_counter()-started,variables=problem.body.size,active_columns=problem.body.state.active.tolist(),body_frames=problem.body.n,source_frames=len(problem.reference.times),source_seconds=[76,95],action_context=list(context),initial_gate=value['gate'],initial_cost=float(value['residual']@value['residual']/2),parts=value['raw_data_parts'],directional_check=directions,coverage=problem.action_factors.last['metrics'],source_regions=len(problem.support_image.rows),five_node_projection=True,no_new_network=True,execution=problem.body.execution.record())
        write_json(case/'profile.json',record);print(json.dumps(dict(event='grasp_profile_complete',jacobian_seconds=jseconds,relative_derivative_error=directions['mixed_direction']['relative_l2_error'],gate=value['gate'],variables=problem.body.size)),flush=True)
        if directions['mixed_direction']['relative_l2_error']>2e-3:raise ValueError('Mixed current-state derivative mismatch')
    return record


def run(project,base,output,prior_label,context,budget,steps,parameter_stem=None):
    paths=check_run_destination(project,output,prior_label,parameter_stem)
    paths=archive_paths(project,output,paths['parameter_stem'],persist=True)
    case=output/'common'/prior_label;profile_path=case/'profile.json'
    prof=json.loads(profile_path.read_text(encoding='utf8')) if profile_path.exists() else profile(project,base,output,prior_label,context)
    available=budget-prof['total_profile_seconds']-90.
    if available<4*prof['jacobian_seconds']:raise RuntimeError('Measured J budget would allow only 0/1 effective updates: improve batching before starting fit')
    start=time.perf_counter();phases=[];x=None;total_accepted=0
    with (case/'fit_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem,review,initial,info,supports,doc=make_problem(project,base,output,prior_label,context,log);x=initial.copy()
        save_packet(problem,info,supports,doc,case/'initial_current_objective',status='common_initial')
        for revision in range(2):
            association=reassociate_support(problem.support_image,x[17:]);problem.cache=problem.cache_x=None
            baseline=problem.evaluate(initial,False);seed=problem.evaluate(x,False)
            record=dict(revision=revision,associations=association,initial_cost=float(baseline['residual']@baseline['residual']/2),seed_cost=float(seed['residual']@seed['residual']/2),comparison='both states re-evaluated under this same new association; monitor window resets')
            write_json(case/f'association_{revision}.json',record)
            remain=available-(time.perf_counter()-start)
            if remain<3*prof['jacobian_seconds']:break
            problem.output=case/('fit' if revision==0 else 'fit_reassociated');problem.accepted=0;problem.start=time.perf_counter();problem.jacobian_seconds_hint=prof['jacobian_seconds']
            limit=min(3,steps) if revision==0 else max(1,steps-total_accepted)
            policy=RunPolicy(wall_seconds=min(remain,prof['jacobian_seconds']*(limit+3)+60.),max_accepted=limit,max_outer=1)
            problem.checkpoint_extra['grasp_arc_ids_json']=np.array(json.dumps(problem.support_image.arc_ids,sort_keys=True))
            x,report=problem.solve(initial=x,run_policy=policy);total_accepted+=problem.accepted
            phases.append(dict(revision=revision,numerically_accepted_steps=problem.accepted,retained_step=report['selection']['retained_step'],report=str(problem.output/'candidate.json'),stop=report['stop']))
            if total_accepted>=steps or report['selection']['quality_failed']:break
        if not phases:raise RuntimeError('No inner solve was run')
        write_json(case/'final_factor_metrics.json',dict(action=problem.action_factors.evaluate(x[17:],False)['metrics'],source_regions=review.evaluate(x[17:],False)['metrics'],objective_parts=problem.evaluate(x,False)['raw_data_parts']))
        checkpoint=load(problem.output/'candidate.npz');archive=paths['parameter'];init_archive=paths['initial_parameter']
        for target,data in [(init_archive,(case/'initial.npz').read_bytes())]:
            if target.exists():assert target.read_bytes()==data
            else:
                guard_project_write(target,project_root=project);target.write_bytes(data)
        checkpoint.update(initial_reference=np.array(init_archive.relative_to(project).as_posix()),initial_reference_sha256=np.array(digest(init_archive)),camera_prior_sha256=np.array(digest(output/'exmr_body_hand_nofoot.npz')),original_gvhmr_pose_anchor=np.array(False))
        if archive.exists():raise FileExistsError('Existing public parameter candidate is preserved')
        guard_project_write(archive,project_root=project)
        np.savez_compressed(archive,**checkpoint)
        restore_checkpoint_layers(problem.reference,checkpoint,include_primary=True)
        packet=save_packet(problem,info,supports,doc,case,status='joint_fit',report=report)
        summary=dict(status='new numerical candidate; requires source and 3D review',accepted_steps=total_accepted,phases=phases,retained_phase_step=phases[-1]['retained_step'],source_seconds=[76,95],action_context=list(context),seconds=time.perf_counter()-start,parameter=str(archive),initial_parameter=str(init_archive),finished_parameter=str(paths['finished_parameter']),parameter_stem=paths['parameter_stem'],archive_names=str(paths['manifest']),result_sha256=digest(case/'fitted/result.npz'),source_frames=len(packet['pts']),accuracy_accepted=False,no_new_network=True)
        write_json(case/'run_summary.json',summary);print(json.dumps(dict(event='grasp_candidate_exported',**summary)),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--base-root',required=True,type=Path,help='Existing body run root; relative paths use the registered Data root');p.add_argument('--output',required=True,type=Path,help='New run directory; relative paths use the registered Data root')
    p.add_argument('--support-document',type=Path,help='Source support regions; relative paths use the project root');p.add_argument('--relation-document',type=Path,help='Grasp relations; relative paths use the project root')
    p.add_argument('--parameter-stem',help='OneDrive parameter filename stem; defaults to a safe output directory name')
    p.add_argument('--prior-label',default='exmr_hand_nofoot',choices=list(LABELS));p.add_argument('--source-context',nargs=2,type=float,default=[76.,95.]);p.add_argument('--action-context',nargs=2,type=float,default=[82.,94.8]);p.add_argument('--budget',type=float,default=1200.);p.add_argument('--max-steps',type=int,default=12);p.add_argument('--stage',choices=['profile','run','all'],default='all');a=p.parse_args(argv)
    store=AssetStore(a.project);a.project=store.project;base=store.resolve(a.base_root,base=store.data,must_exist=True);output=store.resolve(a.output,base=store.data)
    archive_paths(a.project,output,a.parameter_stem)
    if a.stage in ('run','all'):check_run_destination(a.project,output,a.prior_label,a.parameter_stem)
    guard_project_write(output,project_root=a.project);output.mkdir(parents=True,exist_ok=True)
    for source,name in [(a.support_document,'support_and_free_shoe_regions.json'),(a.relation_document,'grasp_relations.json')]:
        if source is not None:
            source=store.resolve(source,base=store.project,must_exist=True);target=guard_project_write(output/name,project_root=a.project)
            if target.exists():
                if digest(target)!=digest(source):raise ValueError('Output input document already differs; preserve it')
            else:target.write_bytes(source.read_bytes())
    prior=output/'exmr_body_hand_nofoot.npz'
    if a.prior_label=='exmr_hand_nofoot' and not prior.exists():
        from .exmr_hand_only import main as build_hand_prior
        native=base/'exmr_spin_compare_1104/native_protocol'
        build_hand_prior(['--project',str(a.project),'--prefoot',str(native/'exmr_prefoot.npz'),'--postfoot',str(native/'exmr_postfoot.npz'),'--final',str(native/'exmr_final.npz'),'--inference-json',str(native/'inference.json'),'--output',str(prior)])
    if tuple(a.source_context)!=(76.,95.):raise ValueError('This source-bound episode is 76-95s')
    if a.stage in ('profile','all'):profile(a.project,base,output,a.prior_label,tuple(a.action_context))
    if a.stage in ('run','all'):
        run(a.project,base,output,a.prior_label,tuple(a.action_context),a.budget,a.max_steps,a.parameter_stem)
        from .spin_grasp_finish import finish
        finish(a.project,base,output,a.prior_label,a.budget,a.max_steps,a.parameter_stem)
        from .spin_grasp_review import deliver
        deliver(a.project,base,output,a.prior_label)

if __name__=='__main__':main()
