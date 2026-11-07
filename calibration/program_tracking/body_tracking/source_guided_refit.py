"""Current-only whole-sequence evidence diagnostics and staged common-state refit.

No inference, historical reconstruction, camera/shape optimization or independent
shoe translations. The audit probe is reversible and cannot publish a candidate.
"""
from pathlib import Path
import argparse,copy,json,time,shutil
import numpy as np
from scipy.sparse import csr_matrix
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .blade_current_reference import restore_current_reference
from .blade_boot_refit import _link,_mass,_prior
from .body_execution import BodyExecution,BodyExecutionConfig
from .sequence_refit import SequenceJoint,SequenceActions
from .skate_integration import source_evidence
from .support_image import SupportImageFactors
from .exmr_integration import PROGRAMME,CASE_FIELD,case_identity
from .weighted_blocks import verify


def prepare(project,stage,parameter,output,log,document=None,*,reuse=False):
    project,stage,parameter,output=map(Path,(project,stage,parameter,output))
    case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True,exist_ok=reuse)
    def link(src,dst):
        if reuse and dst.exists():
            if digest(src)!=digest(dst):raise ValueError('Changed current input '+str(dst))
        else:_link(src,dst)
    def evidence(path,value):
        if reuse:
            if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Changed current evidence '+str(path))
        else:write_json(path,value)
    for name in ('input.npz','initial.npz','observations.npz','compact_smplx.npz','export_source.npz'):
        link(stage/'common/exmr_hand_nofoot'/name,case/name)
    for name in ('palm_grip_geometry.json','exmr_body_hand_nofoot.npz'):link(stage/name,output/name)
    ref,camera,info,saved=restore_current_reference(project,stage,parameter,seed=False)
    source=copy.deepcopy(ref.source_document if document is None else document)
    ledger=source.get('source_episode_review')
    if ledger:
        relations=copy.deepcopy(ref.grasp_relations)
        supports=[e for e in ledger['episodes']if e['kind']=='support_candidate'and e.get('foot')==1]
        relations['support_intervals']=[dict(start_pts=e['start_pts'],end_pts=e['end_pts_inclusive'],foot=1,weight=1.,source_evidence_pts=e['evidence_pts'],provenance='explicit reviewed source episode '+e['episode_id'])for e in supports]
        held=next(e for e in ledger['episodes']if e['kind']=='hand_to_raised_shoe_grasp_candidate')
        previous=relations['grip_interval'].copy()
        start_pts=max(previous['start_pts'],held['start_pts']);end_pts=min(previous['end_pts'],held['end_pts_inclusive'])
        relations['grip_interval'].update(start_pts=start_pts,end_pts=end_pts,evidence_pts=sorted({pt for pt in held['evidence_pts']+previous['evidence_pts']if start_pts<=pt<=end_pts}),provenance='intersection of inherited admitted grip and new source candidate; earlier candidate onset stays unknown; finite material proxy')
        relations['source_episode_provenance']=source['source_episode_provenance']
        relations['phase_boundaries_seconds']=sorted({v/12800. for e in ledger['episodes']for v in (e['start_pts'],e['end_pts_inclusive'])})
        ref.grasp_relations=relations
    if source.get('support_expectation_policy',{}).get('response')=='tolerant_signed_expectation':
        relations=copy.deepcopy(ref.grasp_relations);relations['factor_parameters']['continuous_support']['response']='tolerant_signed_expectation';ref.grasp_relations=relations
    evidence(output/'grasp_relations.json',ref.grasp_relations)
    evidence(output/'support_and_free_shoe_regions.json',source);evidence(output/'source_evidence.json',source)
    ref.root=case;ref.export_source_path=case/'export_source.npz';ref.source_document=source
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance=dict(source_document_sha256=digest(output/'source_evidence.json')))
    base=next((path for path in (stage,*stage.parents)if (path/'action_integration_1103/source_cache/spin_s05.native50.npz').is_file()),None)
    if base is None:raise FileNotFoundError('Current shared native input base is missing')
    identity=case_identity(project,base,output,'exmr_hand_nofoot',info,initial=case/'initial.npz')
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'));execution.attach(ref)
    mass,mass_path=_mass(ref,stage);prior=_prior(load(case/'initial.npz'),ref)
    link(mass_path,output/'mass_geometry.npz')
    supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=source['supports'])
    problem=SequenceJoint(ref,camera,case/'observations.npz',(76.,95.),supports,output/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=ref.ice_policy,record_acceptance_before_jacobian=True)
    class NoOpening:
        def evaluate(self,x,jacobian=False):return dict(residual=np.empty(0),jac=csr_matrix((0,problem.body.size))if jacobian else None,parts={},metrics={})
    problem.source_motion=NoOpening();problem.image_safe_trials=True
    runtime=copy.deepcopy(source);mult=ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
    for row in runtime['observations']:row['objective_weight']*=mult
    for row in runtime['boot_axis_observations']:row['objective_weight']*=mult
    problem.support_image=SupportImageFactors(problem,runtime,arc_ids={})
    x=np.r_[ref.binding.x,np.zeros(problem.body.size)]
    problem.action_factors=SequenceActions(problem,prior,mass,runtime,x[17:])
    problem.checkpoint_extra=dict(parent_parameter_path=np.array(str(parameter.resolve())),parent_parameter_sha256=np.array(digest(parameter)),
        absolute_parent_case_root=np.array(str(stage.resolve())),absolute_case_root=np.array(str(output.resolve())),
        initial_reference=saved['initial_reference'].copy(),initial_reference_sha256=saved['initial_reference_sha256'].copy(),
        evidence_document_sha256=np.array(digest(output/'source_evidence.json')),actual_observations_sha256=np.array(digest(case/'observations.npz')),
        finite_flight_enabled=np.array(True),no_independent_window_stitching=np.array(True))
    return problem,x,execution,info,mass,saved


def source_quality(value):
    """Per-exposure source distances; independent from cost normalization."""
    groups={}
    for row in value['action_metrics'].get('visible_runner_contours',[]):
        if row['structure_feature']!='runner_lower_edge':continue
        if row.get('source_structure_feature',row['structure_feature'])!='runner_lower_edge':continue
        key=f"{row['pts']}_{row['foot']}";outside=np.asarray(row['outside_visible_band_px'])
        groups[key]=dict(pts=row['pts'],foot=row['foot'],radius=row['pixel_uncertainty_radius_px'],forward=float(np.mean(outside)),reverse=0.,source_terminal_distances=[float(row['distance_px'][0]),float(row['distance_px'][-1])],model_terminal_distances=[])
    for row in value['action_metrics'].get('complete_runner_extents',[]):
        key=f"{row['pts']}_{row['foot']}"
        if key not in groups:continue
        outside=np.asarray(row['outside_complete_source_band_px']);mask=np.asarray(row.get('model_visible_mask',np.ones_like(outside)),bool)
        groups[key]['reverse']=float(np.mean(outside[mask]))
        distances=np.asarray(row['model_to_source_distance_px']);visible_sides=np.flatnonzero(mask[:,0]&mask[:,-1])
        groups[key]['model_terminal_distances']=[float(v)for side in visible_sides for v in distances[side,[0,-1]]]
    for v in groups.values():v['distance']=max(v['forward'],v['reverse'])
    return groups


def set_objective_policy(p,phase):
    # Curriculum changes confidence, never source UV, calibration, shared units,
    # cavity constraints or legal attachment. The inherited detector ankle is a
    # boot-occluded proxy, not the observed runner or bony ankle centre.
    guarded=phase in ('joint_source_guarded','support_joint');joint=p.body.oj;weight={'source_guided':.2,'joint_evidence':.65,'joint_physics':1.,'joint_source_guarded':1.,'support_joint':.65}[phase]
    row=np.full(len(joint),weight);row[np.isin(joint,[13,14])]*=.35;row[np.isin(joint,[15,16])]*=.1
    if phase=='support_joint'and p.reference.source_document.get('body_proxy_policy',{}).get('clothed_hip_confidence')is not None:
        row[np.isin(joint,[11,12])]*=float(p.reference.source_document['body_proxy_policy']['clothed_hip_confidence'])
    confidence_review={}
    if guarded:
        pts=p.reference.p['pts'][p.body.frames[p.body.of]];ledger=p.reference.source_document['source_episode_review']['episodes']
        spin=next(e for e in ledger if e['episode_id']=='spin_support');grasp=next(e for e in ledger if e['episode_id']=='grasp_core');release=next(e for e in ledger if e['episode_id']=='grasp_release_transition')
        face=(pts>=spin['start_pts'])&(pts<=spin['end_pts_inclusive'])&np.isin(joint,[0,1,2,3,4])
        arm=(pts>=grasp['start_pts'])&(pts<=release['end_pts_inclusive'])&np.isin(joint,[7,8,9,10])
        row[face]*=.2;row[arm]*=.25
        confidence_review=dict(spin_facial_proxy_confidence=.2,grasp_elbow_wrist_proxy_confidence=.25,facial_rows=int(face.sum()),arm_rows=int(arm.sum()),source_episode_ids=[spin['episode_id'],grasp['episode_id'],release['episode_id']],
            basis='actual final-to-source-guided weighted receipt: body +198.71 versus source -9.60; 2px detector/COCO surface proxies under rotation, hair/clothing and grasp are not material-point measurements; finite engineering confidence, not newly calibrated covariance',raw_uv_sigma_use_unchanged=True,direct_body_source_regions_unchanged=True)
    # Dynamic idealized toe/heel anchors are softer than actual skin/cavity rows.
    reviewed=[]
    if phase=='support_joint':
        lookup={int(pt):i for i,pt in enumerate(p.reference.p['pts'])};frames=p.body.frames[p.body.of]
        policy=p.reference.source_document.get('body_proxy_review',{})
        for frame in policy.get('frames',[]):
            for item in frame['joints']:
                category=item['visibility_category'];confidence={'inside_boot':.1,'inside_boot_blurred':.1,'ear_not_resolved':.15,'ear_indistinct':.15,'covered_limb':.2,'grasp_occlusion':.2,'hand_body_overlap':.2,'limb_overlap':.25,'severe_blur':.15,'back_occlusion':.15}.get(category)
                if confidence is None:continue
                at=(frames==lookup[frame['source_pts']])&(joint==item['coco_index']);row[at]*=confidence
                reviewed.append(dict(pts=frame['source_pts'],coco_joint=item['coco_index'],source_visibility_category=category,finite_proxy_confidence=confidence,changed_rows=int(at.sum()),not_a_new_measured_covariance=True))
        confidence_review['source_reviewed_rows']=reviewed
    dynamic=np.ones((p.body.n,p.compatibility_width));dynamic[:,:12]={'source_guided':.1,'joint_evidence':.35,'joint_physics':.5,'joint_source_guarded':.35,'support_joint':.35}[phase]
    p.objective_block_weights=dict(body_pixels=np.repeat(row,2),dynamic_foot_compatibility=dynamic.ravel(),
        finite_network_pose_prior={'source_guided':.2,'joint_evidence':.65,'joint_physics':1.,'joint_source_guarded':.65,'support_joint':.35}[phase],
        finite_parent_root_prior={'source_guided':.2,'joint_evidence':.65,'joint_physics':1.,'joint_source_guarded':.65,'support_joint':.35}[phase],
        finite_relative_skate_wear_prior={'source_guided':.25,'joint_evidence':.6,'joint_physics':1.,'joint_source_guarded':.6,'support_joint':.6}[phase])
    p.linear_solver_backend='sparse_direct';return dict(phase=phase,body_detector_confidence=weight,knee_relative_confidence=.35,boot_occluded_ankle_relative_confidence=.1,
        dynamic_toe_heel_anchor_confidence=float(dynamic[0,0]),all_physical_cavity_and_collar_weights_unchanged=True,ice_policy_unchanged=True,
        reason='actual weighted directional conflict; source-guided initialization then gradual old evidence re-fusion; raw detector preserved',source_phase_confidence_review=confidence_review)


def build_document(stage,source_root,output):
    from .source_structure_integration import integration
    source_root=Path(source_root);inputs=source_root/'full_structure_1106_inputs';contracts=source_root/'source_contract_review_1106/known_foot_contracts.json';episodes=source_root/'source_contract_review_1106/source_episode_review.json'
    integrated=integration(inputs/'structures_fit_v1.json',inputs/'contacts_fit_v1.json',None,contracts=contracts)
    for src in (contracts,episodes):
        target=Path(output)/src.name
        if target.exists():
            if digest(src)!=digest(target):raise ValueError('Frozen source contract changed: '+str(src))
        else:shutil.copy2(src,target)
    source=json.loads((Path(stage)/'source_evidence.json').read_text(encoding='utf-8'))
    for name in ('full_source_skate_structures','source_support_locations','source_support_evidence','source_structure_budget','complete_runner_extent','unassigned_source_skate_sets'):source[name]=integrated[name]
    source['source_episode_review']=json.loads(episodes.read_text(encoding='utf-8'))
    source['source_episode_provenance']=dict(path=str(episodes),sha256=digest(episodes))
    source['source_contour_contract_provenance']=dict(path=str(contracts),sha256=digest(contracts))
    write_json(Path(output)/'source_integration.json',integrated);return source


def fit_phase(project,stage,parameter,output,document,phase,wall_seconds,max_accepted,*,initial_probe=None,parameter_stem=None,compare_parameters=()):
    from .skate_run_policy import RunPolicy
    from .absolute_motion_state import absolute_reference
    from .exmr_integration import save_packet
    from .skate_physics import fk_state
    from rinkspace.path_policy import guard_project_write
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.perf_counter();log_path=output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path.resolve()),phase=phase,whole_exposures=951,wall_seconds=wall_seconds,max_accepted=max_accepted)),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1)as log:
        p,x,execution,info,mass,saved=prepare(project,stage,parameter,output,log,document)
        p.body.compatibility_difference_step=1e-8
        p.record_compatibility_association=True
        if initial_probe is not None:
            seed=load(initial_probe)
            if str(seed['parent_parameter_sha256'])!=digest(parameter):raise ValueError('Source initializer belongs to another parent')
            np.testing.assert_array_equal(seed['source_pts'],p.reference.p['pts'])
            if seed['x'].shape!=x.shape or not np.array_equal(seed['x'][:17],x[:17]):raise ValueError('Changed shared shape or initializer layout')
            x=seed['x'].copy();p.checkpoint_extra.update(source_initializer_path=np.array(str(initial_probe)),source_initializer_sha256=np.array(digest(initial_probe)))
            p.action_factors.modes.associate(x[17:])
        with execution.limits():
            snapshot=output/'implementation';snapshot.mkdir();implementation={}
            for code in Path(__file__).parent.glob('*.py'):
                implementation[code.name]=digest(code);shutil.copy2(code,snapshot/code.name)
            write_json(output/'implementation.json',implementation)
            policy_record=set_objective_policy(p,phase);write_json(output/'objective_policy.json',policy_record)
            np.savez_compressed(output/'objective_block_weights.npz',**p.objective_block_weights)
            p.checkpoint_extra.update(objective_phase=np.array(phase),objective_policy_sha256=np.array(digest(output/'objective_policy.json')),objective_block_weights_sha256=np.array(digest(output/'objective_block_weights.npz')))
            tick=time.perf_counter();first=p.evaluate(x,True);jsec=time.perf_counter()-tick
            write_json(output/'initial_registry.json',verify(first['weighted_blocks'],first['residual'],first['jac']))
            write_json(output/'initial_metrics.json',native_metrics(p,first));start_quality=source_quality(first)
            # Directional FD checks exercise the new frozen association and actual
            # weighted common-state Jacobian without 20k column-wise probes.
            rng=np.random.default_rng(9511106);direction=np.zeros(len(x));direction[17:]=rng.normal(size=p.body.size);direction/=np.linalg.norm(direction)
            checks=[]
            for h in ((1e-4,1e-5,1e-6)if phase=='source_guided'else(1e-6,)):
                a=p.evaluate(x+h*direction);b=p.evaluate(x-h*direction)
                for original,ap,bm in zip(first['weighted_blocks'],a['weighted_blocks'],b['weighted_blocks']):
                    if original.name not in ('body_pixels','full_source_skate_structure','source_pixel_edge_profile','complete_runner_extent','source_support_location_xy','native_source_support_contact','complete_visible_structure_extent','continuous_support','finite_parent_root_prior','dynamic_foot_compatibility','cavity_AL'):continue
                    fd=(ap.residual-bm.residual)/(2*h);jd=original.jacobian@direction;den=max(np.linalg.norm(fd),np.linalg.norm(jd),1e-12)
                    checks.append(dict(name=original.name,h=h,relative_error=float(np.linalg.norm(fd-jd)/den),max_absolute=float(np.max(abs(fd-jd),initial=0.)),branch_changed=ap.association!=original.association or bm.association!=original.association))
            write_json(output/'actual_weighted_directional_derivatives.json',checks)
            for name in ('body_pixels','full_source_skate_structure','source_pixel_edge_profile','complete_runner_extent','source_support_location_xy','native_source_support_contact','complete_visible_structure_extent','continuous_support','finite_parent_root_prior'):
                smooth=[v for v in checks if v['name']==name and not v['branch_changed']]
                if smooth and min(v['relative_error']for v in smooth)>3e-3:raise ValueError('New source Jacobian mismatch '+name)
            for name in ('dynamic_foot_compatibility','cavity_AL'):
                smooth=[v for v in checks if v['name']==name and not v['branch_changed']]
                if smooth and min(v['relative_error']for v in smooth)>3e-3:raise ValueError('Same-branch footwear Jacobian mismatch '+name)
            p.evaluate(x,False);p.cache=first;p.cache_x=x.copy()
            # Quality is checked per clear exposure. A lower aggregate objective
            # cannot conceal worsening of a clear source shoe.
            complete={(str(row['pts'])+'_'+str(row['foot']))for row in document['complete_runner_extent']['rows']}
            start_locations={f"{r['pts']}_{r['foot']}":float(r['outside_range_normalized_source_pixels'])for r in first['action_metrics']['source_support_locations']}
            def all_structure_quality(value):
                result={}
                for row in value['action_metrics'].get('visible_runner_contours',[]):
                    radius=float(row['pixel_uncertainty_radius_px']);dist=np.asarray(row['outside_visible_band_px'])
                    result[row['id']]=dict(mean=float(np.mean(dist)),p95=float(np.percentile(dist,95)),radius=radius)
                return result
            initial_full=all_structure_quality(first)
            def quality_guard(value,relative_limit):
                current=source_quality(value);groups={}
                for key in complete:
                    base=start_quality[key];after=current[key];allowance=max(.25*base['radius'],.05*base['distance'])
                    groups['clear_runner_'+key]=dict(rejected=after['distance']>base['distance']+allowance,before=base['distance'],after=after['distance'],pixel_allowance=allowance,scope='source-specific finite image quality guard, not physical accuracy')
                    for family in ('source_terminal_distances','model_terminal_distances'):
                        for terminal,(old,new)in enumerate(zip(base[family],after[family])):
                            groups[f'clear_range_{key}_{family}_{terminal}']=dict(rejected=new>old+.25*base['radius'],before=old,after=new,source_relative_pixel_allowance=.25*base['radius'],scope='visible path/profile range preservation; no source material endpoint identity and no attraction to range centre')
                for key,after in all_structure_quality(value).items():
                    if phase!='support_joint':break
                    base=initial_full[key];allowance=.25*base['radius']
                    for statistic in ('mean','p95'):
                        groups['full_source_'+key+'_'+statistic]=dict(rejected=after[statistic]>base[statistic]+allowance,before=base[statistic],after=after[statistic],pixel_allowance=allowance,scope='same original finite source feature, not a material contact or physical threshold')
                for row in value['action_metrics']['source_support_locations']:
                    if phase=='support_joint':break # old whole-line Z=0 mapping is diagnostic only
                    key=f"{row['pts']}_{row['foot']}";base=start_locations[key];after=float(row['outside_range_normalized_source_pixels']);allowance=max(.25,.05*base)
                    groups['conditional_source_xy_'+key]=dict(rejected=after>base+allowance,before=base,after=after,allowed_increment_in_original_source_range_units=allowance,scope='same-source camera-conditional image evidence quality; no metre/ice feasibility threshold')
                return groups
            p.additional_trial_quality=quality_guard;p.source_group_baseline={}
            write_json(output/'association_before.json',dict(contours=p.action_factors.contour.visibility_record,extent=p.action_factors.extent.visibility_record,modes=p.action_factors.modes.record,
                unassigned=p.action_factors.last['metrics'].get('source_unassigned_episode')))
            # Save exact before-state provenance and real row opposition for the
            # leg/foot blocks, in addition to the baseline movement receipt.
            from .directional_receipt import Block,source_guided_direction,receipt
            actual=[Block(b.name,b.residual,b.jacobian,b.association)for b in first['weighted_blocks']]
            source_names=[b.name for b in first['weighted_blocks']if b.name in ('full_source_skate_structure','source_pixel_edge_profile','complete_runner_extent','source_support_location_xy','native_source_support_contact','complete_visible_structure_extent')]
            d,di=source_guided_direction(actual,source_names,parameter_scales=np.full(len(x),.1),damping=.1,active_columns=np.arange(17,len(x)),maxiter=160)
            row_conflicts={}
            for block in first['weighted_blocks']:
                if block.name not in ('dynamic_foot_compatibility','cavity_AL'):continue
                g=block.residual*(block.jacobian@d);ids=np.argsort(g)[-20:][::-1];rows=[]
                for i in ids:
                    if block.name=='dynamic_foot_compatibility':row=dict(pts=int(p.reference.p['pts'][p.body.frames[i//p.compatibility_width]]),row_in_footwear=int(i%p.compatibility_width))
                    else:row=dict(label=p.foot_row_labels[i])
                    row.update(weighted_residual=float(block.residual[i]),slope=float(g[i]));rows.append(row)
                row_conflicts[block.name]=rows
            write_json(output/'weighted_source_movement.json',dict(receipt=receipt(actual,d),footwear_opposing_rows=row_conflicts,probe_only=True,direction=di))
            p.cache=first;p.cache_x=x.copy();p.jacobian_seconds_hint=jsec
            remaining=max(30.,wall_seconds-(time.perf_counter()-started)-30.)
            print(json.dumps(dict(event='fit_start',phase=phase,initial_cost=float(first['residual']@first['residual']/2),first_jacobian_seconds=jsec,remaining_seconds=remaining)),flush=True)
            result,report=p.solve(x,RunPolicy(wall_seconds=remaining,max_accepted=max_accepted,max_outer=1,allow_penalty_growth=False,stagnation_window=12))
            value=p.evaluate(result,False);write_json(output/'final_metrics.json',native_metrics(p,value))
            write_json(output/'final_source_quality.json',dict(before=start_quality,after=source_quality(value)))
            final=absolute_reference(p.reference,p.body.state.current_curve(result[17:]),root=p.reference.root);final.relative_skate_curve=p.body.state.shoe_current_curve(result[17:])
            archive=guard_project_write(Path(project)/PROGRAMME/((parameter_stem or 'spin_'+phase+'_1106')+'.npz'),project_root=Path(project))
            if archive.exists():raise FileExistsError(archive)
            shutil.copy2(p.output/'candidate.npz',archive)
            shell=copy.copy(p);shell.reference=final
            packet=save_packet(shell,info,p.evidence,document,p.reference.root,status='joint_fit',report=report)
            s=fk_state(final,final.at(final.times),np.zeros((len(final.times),69)));C,R=mass.evaluate(final.at(final.times),np.zeros((len(final.times),69)))
            packet.update(com_world_mid=mass.com(C),absolute_parameter_path=np.array(str(archive)),**p.body.state.packet(result[17:]))
            template=final.binding.packet();edge=template['edges'];dimensions=template['parameters']
            from .skate_template import profile_height
            rear=-dimensions.runner_rear_extension_m;front=dimensions.sole_length_m-dimensions.runner_front_inset_m;xx=np.array([rear,(rear+front)/2,front])
            stations=np.c_[xx,np.zeros(3),profile_height(xx,dimensions)]
            packet['blade_material_world_mid']=(s['O'][:,:,None]+np.einsum('nfij,kj->nfki',s['R'],stations))*final.binding.ratio
            packet['runner_edges_world_mid']=(s['O'][:,:,None,None,:]+np.einsum('nfij,ekj->nfeki',s['R'],edge))*final.binding.ratio
            modes=p.action_factors.modes;packet.update(sequence_contact_modes=modes.modes,sequence_contact_status=modes.status,sequence_contact_mode_names=np.asarray(modes.record['mode_names']),sequence_contact_association_json=np.array(json.dumps(modes.record,sort_keys=True)))
            counts=np.zeros((len(final.times),2),np.int32);location=np.zeros_like(counts,bool);lookup={int(pt):i for i,pt in enumerate(final.p['pts'])}
            for row in document['full_source_skate_structures']['rows']:counts[lookup[row['pts']],row['foot']]+=1
            for row in document['source_support_evidence']:location[lookup[row['pts']],row['foot']]=True
            packet.update(source_complete_structure_observation_count=counts,source_support_location_in_optimization=location,runner_trajectory_evidence_status=np.where(counts>0,'direct_source_structure_fit','inferred_between_source_exposures'),source_unassigned_episode_json=np.array(json.dumps(p.action_factors.last['metrics'].get('source_unassigned_episode'),sort_keys=True)))
            np.savez_compressed(p.reference.root/'fitted/result.npz',**packet)
            if compare_parameters:
                from .absolute_motion_state import curve_from_packet
                from .sequence_skate_state import shoe_curve_from_packet
                from .absolute_motion_curve import _log
                reevaluated=[]
                try:
                    for candidate_path in list(compare_parameters)+[archive]:
                        candidate=load(candidate_path);curve=curve_from_packet(candidate);shoe=shoe_curve_from_packet(candidate)
                        np.testing.assert_array_equal(curve.knots,p.body.state.curve.knots);np.testing.assert_array_equal(shoe.knots,p.body.state.shoe_curve.knots)
                        np.testing.assert_array_equal(candidate['x'][:17],result[:17]);z=result.copy();controls=z[17:17+p.body.state.body_size].reshape(p.body.state.m,69)
                        controls[:,:3]=curve.position_controls-p.body.state.curve.position_controls
                        controls[:,3:]=_log(p.body.state.curve.rotation_controls.swapaxes(-1,-2)@curve.rotation_controls).reshape(p.body.state.m,66)
                        z[17+p.body.state.body_size:]=_log(p.body.state.shoe_curve.rotation_controls.swapaxes(-1,-2)@shoe.rotation_controls).ravel()
                        other=p.evaluate(z,False)
                        reevaluated.append(dict(parameter=str(candidate_path),parameter_sha256=digest(candidate_path),weighted_cost=float(other['residual']@other['residual']/2),weighted_blocks={b.name:float(b.residual@b.residual/2)for b in other['weighted_blocks']},source_quality=source_quality(other),physical=other['gate']))
                finally:p.evaluate(result,False)
                write_json(output/'same_final_objective_re_evaluation.json',dict(candidates=reevaluated,association='same frozen final-stage source visibility/segments/whole-episode identity/contact-mode plan',AL='same zero multipliers and finite cavity penalty; no ice AL',no_candidate_written=True))
            summary=dict(phase=phase,parameter=str(archive),result=str(p.reference.root/'fitted/result.npz'),accepted_steps=p.accepted,retained_step=report['selection']['retained_step'],stop=report['stop'],wall_seconds=time.perf_counter()-started,initial_cost=float(first['residual']@first['residual']/2),final_cost=float(value['residual']@value['residual']/2),first_jacobian_seconds=jsec,
                whole_seconds=[76.,95.],source_frames=len(final.times),body_controls=p.body.state.m,shoe_controls=p.body.state.shoe_m,objective_policy=policy_record,source_exposures=len({r['pts']for r in document['full_source_skate_structures']['rows']}),source_feature_rows=len(document['full_source_skate_structures']['rows']),complete_source_rows=len(document['complete_runner_extent']['rows']),source_quality_guard=quality_guard(value,.05),accuracy_accepted=False,user_accepted=False,not_default=True)
            write_json(output/'run_summary.json',summary);print(json.dumps(dict(event='phase_complete',phase=phase,accepted=p.accepted,result=summary['result'])),flush=True)
    execution.close();return summary


def native_metrics(problem,value):
    body=problem.body.cache;out=dict(cost=float(value['residual']@value['residual']/2),image_cost=float(value['image_cost']))
    fi,j=problem.body.of,problem.body.oj;uv=body['uv'][fi,problem.body.layout.body_landmarks[j]]
    error=np.linalg.norm(uv-problem.body.obs['uv'][problem.body.frames[fi],j],axis=-1)
    out['body_pixel_distance']=dict(median=float(np.median(error)),p95=float(np.percentile(error,95)))
    for key in ('visible_runner_contours','complete_runner_extents','source_support_locations','native_source_support_contacts','complete_visible_structure_extents','source_pixel_edge_profiles'):
        rows=value['action_metrics'].get(key,[])
        out[key]=rows
    out['physical']=value['gate'];return out


def diagnose(project,stage,parameter,output):
    from .directional_receipt import Block,source_guided_direction,probe
    from .audit_probe_contract import choose_source_names,restore_saved_objective
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);log_path=output/'progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path.resolve()),diagnostic_only=True)),flush=True)
    with log_path.open('a',encoding='utf-8',buffering=1)as log:
        p,x,execution,info,mass,saved=prepare(project,stage,parameter,output,log)
        restored=restore_saved_objective(p,stage,saved)
        write_json(output/'objective_replay.json',restored)
        with execution.limits():
            start=time.perf_counter();first=p.evaluate(x,True)
            write_json(output/'registry.json',verify(first['weighted_blocks'],first['residual'],first['jac']))
            write_json(output/'baseline_metrics.json',native_metrics(p,first))
            blocks=[Block(b.name,b.residual,b.jacobian,b.association)for b in first['weighted_blocks']]
            np.savez_compressed(output/'probe_baseline.npz',x=x,source_pts=p.reference.p['pts'],actual_weighted_residual=first['residual'])
            source_names=choose_source_names(blocks)
            scales=np.full(len(x),.1);m=p.body.state.m;d=p.body.state.d
            lower=np.concatenate([np.arange(6+(j-1)*3,9+(j-1)*3)for j in (1,2,4,5,7,8,10,11)])
            columns=lambda local:np.concatenate([17+i*d+np.asarray(local)for i in range(m)])
            shoe=np.arange(17+p.body.state.body_size,len(x))
            masks=dict(root=columns(np.arange(6)),leg_and_shoe=np.r_[columns(lower),shoe],all=np.arange(17,len(x)))
            def evaluate(at):
                v=p.evaluate(at,False)
                return {b.name:dict(residual=b.residual,association=b.association)for b in v['weighted_blocks']}
            reports={};directions={}
            for name,ids in masks.items():
                direction,record=source_guided_direction(blocks,source_names,parameter_scales=scales,damping=.1,active_columns=ids,maxiter=300)
                # A declared trust scale bounds this diagnostic, not model accuracy.
                maximum=float(np.max(abs(direction[17:])))
                if maximum>.15:direction*=.15/maximum
                directions[name]=direction
                report=probe(blocks,x,direction,evaluate=evaluate,alphas=(-.1,.1,.5,1.),derivative_steps=(1e-4,1e-5,1e-6))
                report['source_direction']=record;report['declared_scales']=dict(translation_m=.1,rotation_rad=.1,damping=.1,maximum_parameter_move=.15)
                body=next(b for b in first['weighted_blocks']if b.name=='body_pixels');effects=(body.residual*(body.jacobian@direction)).reshape(-1,2).sum(1)
                worst=np.argsort(effects)[-24:][::-1];bf=p.body.frames[p.body.of]
                report['opposing_body_observations']=[dict(pts=int(p.reference.p['pts'][bf[i]]),coco_joint=int(p.body.oj[i]),slope=float(effects[i]),source_uv=p.body.obs['uv'][bf[i],p.body.oj[i]].tolist(),sigma_px=float(p.body.obs['sigma_px'][bf[i],p.body.oj[i]]))for i in worst]
                report['finite_step_native_metrics']={}
                for alpha in (.5,1.):report['finite_step_native_metrics'][str(alpha)]=native_metrics(p,p.evaluate(x+alpha*direction,False))
                p.evaluate(x,False);reports[name]=report
                write_json(output/(name+'_direction.json'),report)
                print(json.dumps(dict(direction=name,seconds=time.perf_counter()-start,slope=report['directional_slope'],opposing=[(b['name'],b['directional_slope'])for b in report['blocks']if b['directional_slope']>0])),flush=True)
            np.savez_compressed(output/'probe_directions.npz',**directions)
            p.evaluate(x,False)
            write_json(output/'diagnosis_summary.json',dict(current_parameter=str(Path(parameter).resolve()),parameter_sha256=digest(parameter),whole_exposures=len(p.reference.p['pts']),variables=len(x),elapsed_seconds=time.perf_counter()-start,source_names=source_names,probe_not_fit=True,baseline_restored=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--parameter',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--action',choices=['diagnose','fit'],default='diagnose');ap.add_argument('--source-root',type=Path)
    ap.add_argument('--phase',choices=['source_guided','joint_evidence','joint_physics','joint_source_guarded','support_joint'],default='source_guided');ap.add_argument('--wall-seconds',type=float,default=900.);ap.add_argument('--max-accepted',type=int,default=60)
    ap.add_argument('--initial-probe',type=Path);ap.add_argument('--parameter-stem')
    ap.add_argument('--compare-parameter',type=Path,action='append',default=[])
    ap.add_argument('--document',type=Path)
    a=ap.parse_args()
    if a.action=='diagnose':diagnose(a.project,a.stage,a.parameter,a.output)
    else:
        if a.source_root is None and a.document is None:ap.error('--source-root or --document required for source contracts')
        inputs=a.output.parent/'inputs'
        if not inputs.exists():inputs.mkdir(parents=True)
        document=json.loads(a.document.read_text(encoding='utf-8'))if a.document is not None else build_document(a.stage,a.source_root,inputs)
        fit_phase(a.project,a.stage,a.parameter,a.output,document,a.phase,a.wall_seconds,a.max_accepted,initial_probe=a.initial_probe,parameter_stem=a.parameter_stem,compare_parameters=a.compare_parameter)

if __name__=='__main__':main()
