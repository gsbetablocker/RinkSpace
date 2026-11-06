"""Joint whole-S05 solve with one body/relative-skate state and native source clock."""
from pathlib import Path
import argparse,copy,json,time,shutil
import numpy as np
from scipy.sparse import csr_matrix,vstack
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .blade_current_reference import restore_current_reference
from .blade_boot_refit import _link,_mass,_prior,ExistingActionsWithBootAxes
from .spin_continuous_refit import ContinuousJoint
from .absolute_motion_state import absolute_reference
from .body_execution import BodyExecution,BodyExecutionConfig
from .skate_integration import source_evidence
from .support_image import SupportImageFactors
from .boot_axis_factors import BootAxisFactors
from .sequence_dynamics import SequenceShoeDynamics,TemporalContactFactors
from .skate_run_policy import RunPolicy
from .exmr_integration import PROGRAMME,CASE_FIELD,case_identity,save_packet


class SequenceActions:
    def __init__(self,problem,prior,mass,document,x):
        self.existing=ExistingActionsWithBootAxes(problem,prior,mass,document)
        self.shoe=SequenceShoeDynamics(problem);self.modes=TemporalContactFactors(problem,x);self.last=None
        self.contour=None
        if 'visible_runner_contours' in document:
            from .visible_runner_contours import VisibleRunnerContourFactors
            self.contour=VisibleRunnerContourFactors(problem,document['visible_runner_contours'])
        if 'full_source_skate_structures' in document:
            from .source_skate_structure_factors import FullSkateStructureFactors
            self.contour=FullSkateStructureFactors(problem,document['full_source_skate_structures'])
        self.extent=None
        if document.get('complete_runner_extent',{}).get('rows'):
            from .complete_runner_extent import CompleteRunnerExtentFactors
            self.extent=CompleteRunnerExtentFactors(problem,document['complete_runner_extent'])
        self.source_locations=None
        if 'source_support_locations' in document:
            from .source_support_locations import SourceSupportLocationFactors
            cls=SourceSupportLocationFactors
            if document.get('support_location_policy',{}).get('mode')=='native_image_candidate':
                from .source_contact_image import NativeSupportContactFactors
                cls=NativeSupportContactFactors
            self.source_locations=cls(problem,document['source_support_locations'],support_evidence=document['source_support_evidence'],**document['source_structure_budget'])
        self.structure_extent=None
        if document.get('complete_visible_structure_extent',{}).get('rows'):
            from .source_structure_extent import CompleteStructureExtentFactors
            self.structure_extent=CompleteStructureExtentFactors(problem,document['complete_visible_structure_extent'])
        self.unassigned=None
        if document.get('unassigned_source_skate_sets'):
            from .source_skate_structure_factors import UnassignedSkateSetFactors
            self.unassigned=UnassignedSkateSetFactors(problem,document['unassigned_source_skate_sets'],episode=document.get('source_episode_review',{}).get('late_identity_episode'))
    def evaluate(self,x,jacobian=False):
        outputs=[self.existing.evaluate(x,jacobian),self.shoe.evaluate(x,jacobian),self.modes.evaluate(x,jacobian)]
        if self.contour is not None:outputs.append(self.contour.evaluate(x,jacobian))
        if self.extent is not None:outputs.append(self.extent.evaluate(x,jacobian))
        if self.source_locations is not None:outputs.append(self.source_locations.evaluate(x,jacobian))
        if self.structure_extent is not None:outputs.append(self.structure_extent.evaluate(x,jacobian))
        if self.unassigned is not None:outputs.append(self.unassigned.evaluate(x,jacobian))
        out=dict(residual=np.concatenate([v['residual'] for v in outputs]),jac=vstack([v['jac'] for v in outputs],format='csr') if jacobian else None,parts={},metrics={})
        for v in outputs:out['parts'].update(v['parts']);out['metrics'].update(v['metrics'])
        out['image_cost']=self.existing.grasp.last.get('image_cost',0.)+sum(v.get('image_cost',0.) for v in outputs[1:])
        factors=[self.shoe,self.modes]+([] if self.contour is None else [self.contour])+([] if self.extent is None else [self.extent])+([] if self.source_locations is None else [self.source_locations])+([] if self.structure_extent is None else [self.structure_extent])+([] if self.unassigned is None else [self.unassigned])
        out['factor_row_counts']=outputs[0]['factor_row_counts']+[(type(f).__name__,len(v['residual'])) for f,v in zip(factors,outputs[1:])]
        from .weighted_blocks import block,signature
        out['weighted_blocks']=[]
        for factor,value in zip([self.existing]+factors,outputs):
            name=next(iter(value['parts']),type(factor).__name__)
            association=getattr(factor,'association_signature','')
            out['weighted_blocks'].extend(value.get('weighted_blocks',[block(name,value['residual'],value['jac'],dict(factor=type(factor).__name__),association)]))
        self.last=out;return out


class SequenceJoint(ContinuousJoint):
    def evaluate(self,x,jacobian=False):
        out=super().evaluate(x,jacobian)
        out['source_group_costs']=dict(body_pixels=.02*out['raw_data_parts'].get('body_pixels',0.),existing_shoe_regions=out['parts'].get('support_image',0.))
        if not hasattr(self,'source_group_baseline'):self.source_group_baseline={'body_pixels':out['source_group_costs']['body_pixels']}
        return out
    def additional_trial_quality(self,value,relative_limit):
        from .skate_run_policy import image_trial_quality
        return {key:image_trial_quality(base,value['source_group_costs'][key],relative_limit) for key,base in self.source_group_baseline.items()}
    def checkpoint_fields(self,x):
        out=super().checkpoint_fields(x)
        out.update(sequence_contact_modes=self.action_factors.modes.modes,
            sequence_contact_status=self.action_factors.modes.status,
            sequence_contact_mode_names=np.asarray(self.action_factors.modes.record['mode_names']),
            sequence_contact_association_json=np.array(json.dumps(self.action_factors.modes.record,sort_keys=True)))
        return out


def run(project,stage,parameter,source_sequence,output,*,wall_seconds=1200.,max_accepted=160,parameter_stem='spin_sequence_global_1105',visible_contours=None,structure_integration=None):
    project,stage,parameter,source_sequence,output=map(lambda p:Path(p).resolve(),(project,stage,parameter,source_sequence,output))
    if output.exists():raise FileExistsError('Existing candidates are immutable: '+str(output))
    output.mkdir(parents=True);started=time.perf_counter();log_path=output/'fit_progress.jsonl'
    print(json.dumps(dict(main_log=str(log_path),whole_source_seconds=[76.,95.],wall_seconds=wall_seconds,max_accepted=max_accepted)),flush=True)
    case=output/'common/exmr_hand_nofoot';case.mkdir(parents=True);oldcase=stage/'common/exmr_hand_nofoot'
    for name in ['input.npz','compact_smplx.npz','initial.npz','observations.npz','export_source.npz']:_link(oldcase/name,case/name)
    for name in ['grasp_relations.json','palm_grip_geometry.json','exmr_body_hand_nofoot.npz']:_link(stage/name,output/name)
    source=json.loads((stage/'support_and_free_shoe_regions.json').read_text(encoding='utf-8'))
    temporal=json.loads((source_sequence/'temporal_regions.json').read_text(encoding='utf-8'))
    known={r['id'] for r in source['observations']}
    source['observations'].extend(r for r in temporal['observations'] if r['id'] not in known)
    if visible_contours is not None:
        visible_contours=Path(visible_contours).resolve()
        source['visible_runner_contours']=json.loads(visible_contours.read_text(encoding='utf-8'))
        source['visible_runner_contours_provenance']=dict(path=str(visible_contours),sha256=digest(visible_contours))
    if structure_integration is not None:
        integration=json.loads(Path(structure_integration).read_text(encoding='utf-8'))
        for key in ['full_source_skate_structures','source_support_locations','source_support_evidence','source_structure_budget']:
            source[key]=integration[key]
        source['unassigned_source_skate_sets']=integration.get('unassigned_source_skate_sets',[])
        source['complete_runner_extent']=integration.get('complete_runner_extent',{'rows':[]})
        source['source_structure_integration_provenance']=dict(path=str(structure_integration),sha256=digest(structure_integration))
        known_support={(int(r['pts']),tuple(r['feet']))for r in source['supports']}
        for row in source['source_support_evidence']:
            if(int(row['pts']),(row['foot'],))in known_support:continue
            source['supports'].append(dict(id=f"complete_source_support_{row['pts']}_foot{row['foot']}",pts=row['pts'],feet=[row['foot']],
                source='current complete source structure sequence review',source_id=integration['source_sha256'],source_sha256=integration['source_sha256'],
                shot_id='s05',source_time_base=[1,12800],confidence=.35,mode='glide',edge='unknown',physical_side='source sequence association; not human-confirmed anatomy',
                source_support_weight_s=.02,source_support_weight_basis='one observed native50Hz exposure, not propagated contact duration',
                note=row['support_identity_evidence'],context_source_pts=row['context_source_pts'],human_confirmed=False))
        # The complete boundaries replace their old short subsegments. Keep the
        # old source records archived, without counting those pixels twice.
        source.pop('visible_runner_contours',None)
    source['temporal_sequence_integration']=dict(source_ledger=str(source_sequence/'source_sequence.json'),
        source_ledger_sha256=digest(source_sequence/'source_sequence.json'),temporal_review=str(source_sequence/'sequence_temporal_review.json'),
        temporal_review_sha256=digest(source_sequence/'sequence_temporal_review.json'),region_tracks=str(source_sequence/'temporal_regions.json'),
        region_tracks_sha256=digest(source_sequence/'temporal_regions.json'),inferred_is_not_direct_observation=True)
    write_json(output/'support_and_free_shoe_regions.json',source);write_json(output/'source_evidence.json',source)
    ref,camera,info,saved=restore_current_reference(project,stage,parameter,seed=False)
    ref.root=case;ref.export_source_path=case/'export_source.npz';ref.source_document=source
    ref.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[],provenance=dict(source_document_sha256=digest(output/'support_and_free_shoe_regions.json')))
    identity=case_identity(project,output.parents[1],output,'exmr_hand_nofoot',info,initial=case/'initial.npz')
    ref.fixed_checkpoint_identity[CASE_FIELD]=np.array(json.dumps(identity,sort_keys=True))
    execution=BodyExecution(BodyExecutionConfig(workers=12,blas_threads=1,local_derivatives='analytic',compatibility_backend='cuda_resident'));execution.attach(ref)
    mass,mass_path=_mass(ref,stage);prior=_prior(load(case/'initial.npz'),ref)
    np.savez_compressed(output/'mass_geometry.npz',model_identity_json=np.array(json.dumps(mass.proxy,sort_keys=True)),source_model_file_sha256=np.array(digest(mass_path)),rest_geometry_only=np.array(True))
    snapshot=output/'implementation';snapshot.mkdir()
    implementation={}
    for path in Path(__file__).parent.glob('*.py'):
        implementation[path.name]=digest(path);shutil.copy2(path,snapshot/path.name)
    write_json(output/'implementation.json',implementation)
    archive=project/PROGRAMME/(parameter_stem+'.npz')
    from rinkspace.path_policy import guard_project_write
    guard_project_write(archive,project_root=project)
    if archive.exists():raise FileExistsError(archive)
    with log_path.open('a',encoding='utf-8',buffering=1) as log:
        def emit(event,**kw):log.write(json.dumps(dict(event=event,elapsed=time.perf_counter()-started,**kw))+'\n');print(json.dumps(dict(event=event,**kw)),flush=True)
        with execution.limits():
            supports=source_evidence(case,shot_id='s05',source_pts=ref.p['pts'],records=source['supports'])
            problem=SequenceJoint(ref,camera,case/'observations.npz',(76.,95.),supports,output/'solve',log,'sequence_skates_v1',optimize_shared=False,ice_policy=ref.ice_policy,record_acceptance_before_jacobian=True)
            problem.image_safe_trials=True
            if structure_integration is not None and integration.get('cross_stage_quality_reference'):
                problem.source_group_baseline={'body_pixels':float(integration['cross_stage_quality_reference']['body_pixel_scaled_cost'])}
            class NoOpening:
                def evaluate(self,x,jacobian=False):return dict(residual=np.empty(0),jac=csr_matrix((0,problem.body.size)) if jacobian else None,parts={},metrics={})
            problem.source_motion=NoOpening()
            runtime=copy.deepcopy(source);mult=ref.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
            for row in runtime['observations']:row['objective_weight']*=mult
            for row in runtime['boot_axis_observations']:row['objective_weight']*=mult
            problem.support_image=SupportImageFactors(problem,runtime,arc_ids={});review=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
            x=np.r_[ref.binding.x,np.zeros(problem.body.size)]
            problem.action_factors=SequenceActions(problem,prior,mass,runtime,x[17:])
            problem.checkpoint_extra=dict(parent_parameter_path=np.array(str(parameter)),parent_parameter_sha256=np.array(digest(parameter)),
                absolute_parent_case_root=np.array(str(stage)),absolute_case_root=np.array(str(output)),
                initial_reference=saved['initial_reference'].copy(),initial_reference_sha256=saved['initial_reference_sha256'].copy(),
                evidence_document_sha256=np.array(digest(output/'support_and_free_shoe_regions.json')),
                actual_observations_sha256=np.array(digest(case/'observations.npz')),finite_flight_enabled=np.array(True),
                sequence_source_ledger_sha256=np.array(digest(source_sequence/'source_sequence.json')),no_independent_window_stitching=np.array(True))
            if structure_integration is not None:
                problem.checkpoint_extra.update(source_structure_integration_path=np.array(str(structure_integration)),source_structure_integration_sha256=np.array(digest(structure_integration)))
            emit('whole_sequence_setup',body_frames=problem.body.n,source_frames=len(ref.p['pts']),body_controls=problem.body.state.m,
                shoe_controls=problem.body.state.shoe_m,variables=problem.size,source_image_rows=len(problem.support_image.rows),
                original_body_and_camera_unchanged=True,whole_sequence_active=True)
            t=time.perf_counter();first=problem.evaluate(x,True);jsec=time.perf_counter()-t
            if structure_integration is not None:write_json(output/'start_source_structure_metrics.json',problem.action_factors.last['metrics'])
            emit('first_full_J',seconds=jsec,rows=len(first['residual']),nnz=first['jac'].nnz,cost=float(first['residual']@first['residual']/2),parts=first['raw_data_parts'])
            # A full-objective derivative check exercises new shoe columns in
            # the real case; it is an implementation check, not visual acceptance.
            rng=np.random.default_rng(11057695);direction=np.zeros_like(x)
            direction[17:]=rng.normal(size=problem.body.size);direction/=np.linalg.norm(direction)
            h=2e-7;fd=(problem.evaluate(x+h*direction)['residual']-problem.evaluate(x-h*direction)['residual'])/(2*h)
            analytic=first['jac']@direction;err=fd-analytic
            check=dict(relative_error=float(np.linalg.norm(err)/max(np.linalg.norm(fd),np.linalg.norm(analytic),1e-12)),max_absolute_error=float(np.max(abs(err))),finite=bool(np.isfinite(analytic).all()),body_and_relative_shoe_columns=True)
            sizes=[('body_pixels',problem.body.cache['data_rows']),('dynamic_compatibility',len(problem.body.cache['compatibility_residual'])),('neutral_and_shared',len(problem.binding.compatibility_rows(problem.binding.neutral_native,problem.binding.neutral_frames,problem.binding.neutral_samples).ravel())+17),('ice_soft',len(first['ice']['residual']))]
            sizes+=problem.action_factors.last['factor_row_counts']+[('support_image',len(problem.support_image.last['residual']))]
            sizes+=[('cavity_AL',len(fd)-sum(size for _,size in sizes))]
            offset=0;blocks={}
            for name,size in sizes:
                a=fd[offset:offset+size];b=analytic[offset:offset+size];e=err[offset:offset+size]
                blocks[name]=dict(rows=int(size),relative_error=float(np.linalg.norm(e)/max(np.linalg.norm(a),np.linalg.norm(b),1e-12)),max_absolute_error=float(np.max(abs(e),initial=0.)),fd_norm=float(np.linalg.norm(a)))
                offset+=size
            check['blocks']=blocks
            write_json(output/'shoe_full_objective_derivative.json',check);emit('shoe_derivative',**check)
            if not check['finite'] or check['relative_error']>3e-3:raise ValueError('Whole-objective body/skate Jacobian mismatch')
            for name,block in blocks.items():
                if name in ('FullSkateStructureFactors','CompleteRunnerExtentFactors','SourceSupportLocationFactors','UnassignedSkateSetFactors')and block['fd_norm']>1e-6 and block['relative_error']>3e-3:raise ValueError('Source structure/location Jacobian mismatch: '+name)
            start_review=review.evaluate(x[17:])['metrics'];write_json(output/'start_quality.json',dict(cost=float(first['residual']@first['residual']/2),parts=first['raw_data_parts'],source_regions=start_review,image_cost=first['image_cost'],mode_association=problem.action_factors.modes.record))
            problem.cache_x=x.copy();problem.cache=first
            remaining=max(0.,wall_seconds-(time.perf_counter()-started)-60.)
            policy=RunPolicy(wall_seconds=remaining,max_accepted=max_accepted,max_outer=1,allow_penalty_growth=False,stagnation_window=12)
            problem.jacobian_seconds_hint=jsec;emit('whole_sequence_solve_start',policy=vars(policy),new_shoe_jacobian=check)
            result,report=problem.solve(x,policy)
            final_value=problem.evaluate(result,False);end_review=review.evaluate(result[17:])['metrics']
            if structure_integration is not None:write_json(output/'final_source_structure_metrics.json',problem.action_factors.last['metrics'])
            final=absolute_reference(ref,problem.body.state.current_curve(result[17:]),root=case)
            final.relative_skate_curve=problem.body.state.shoe_current_curve(result[17:])
            candidate=load(problem.output/'candidate.npz')
            with archive.open('xb') as stream:np.savez_compressed(stream,**candidate)
            shell=copy.copy(problem);shell.reference=final
            packet=save_packet(shell,info,supports,source,case,status='joint_fit',report=report)
            from .skate_physics import fk_state
            s=fk_state(final,final.at(final.times),np.zeros((len(final.times),69)))
            C,R=mass.evaluate(final.at(final.times),np.zeros((len(final.times),69)))
            packet.update(com_world_mid=mass.com(C),absolute_parameter_path=np.array(str(archive)),**problem.body.state.packet(result[17:]))
            if structure_integration is not None:
                # Export the same continuous physical state, with exact native
                # exposure identities and direct/inferred evidence separated.
                edge=final.binding.packet()['edges'];ratio=final.binding.ratio
                packet['runner_edges_world_mid']=(s['O'][:,:,None,None,:]+np.einsum('nfij,ekj->nfeki',s['R'],edge))*ratio
                lookup={int(v):i for i,v in enumerate(ref.p['pts'])};counts=np.zeros((len(ref.p['pts']),2),dtype=np.int32);locations=np.zeros_like(counts,dtype=bool)
                for row in source['full_source_skate_structures']['rows']:counts[lookup[int(row['pts'])],int(row['foot'])]+=1
                for row in source['source_support_evidence']:locations[lookup[int(row['pts'])],int(row['foot'])]=True
                packet.update(source_complete_structure_observation_count=counts,source_support_location_in_optimization=locations,
                    runner_trajectory_evidence_status=np.where(counts>0,'direct_source_structure_fit','inferred_between_source_exposures'),
                    source_structure_integration_sha256=np.array(digest(structure_integration)),runner_trajectory_coordinate_domain=np.array('conditional_metres_same_bound_body_skate_state'))
                packet['source_unassigned_association_json']=np.array(json.dumps(problem.action_factors.last['metrics'].get('source_unassigned_associations',[]),sort_keys=True))
            modes=problem.action_factors.modes
            packet.update(sequence_contact_modes=modes.modes,sequence_contact_status=modes.status,sequence_contact_mode_names=np.asarray(modes.record['mode_names']),sequence_contact_association_json=np.array(json.dumps(modes.record,sort_keys=True)))
            np.savez_compressed(case/'fitted/result.npz',**packet)
            summary=dict(accepted_steps=problem.accepted,retained_step=report['selection']['retained_step'],stop=report['stop'],
                wall_seconds=time.perf_counter()-started,first_full_J_seconds=jsec,parameter=str(archive),result=str(case/'fitted/result.npz'),
                whole_sequence_seconds=[76.,95.],source_frames=len(ref.p['pts']),fit_frames=problem.body.n,
                body_controls=problem.body.state.m,relative_shoe_controls=problem.body.state.shoe_m,
                initial_cost=float(first['residual']@first['residual']/2),final_cost=float(final_value['residual']@final_value['residual']/2),
                image_cost_before=first['image_cost'],image_cost_after=final_value['image_cost'],
                initial_parts=first['raw_data_parts'],final_parts=final_value['raw_data_parts'],source_regions_before=start_review,source_regions_after=end_review,
                derivative_check=check,physical_constraints_in_objective=True,ice_policy=vars(ref.ice_policy),
                all_951_trajectory_estimated=True,direct_image_observations_not_invented=True,no_independent_window_stitching=True,
                accuracy_accepted=False,user_accepted=False,not_default=True)
            if structure_integration is not None:
                summary.update(full_structure_source_exposures=len(set(r['pts']for r in source['full_source_skate_structures']['rows'])),
                    full_structure_source_foot_exposures=len(set((r['pts'],r['foot'])for r in source['full_source_skate_structures']['rows'])),
                    full_structure_feature_rows=len(source['full_source_skate_structures']['rows']),source_support_location_rows=len(source['source_support_evidence']),
                    source_structure_budget=source['source_structure_budget'],whole_runner_trajectory_world_shape=list(packet['runner_edges_world_mid'].shape))
            write_json(output/'run_summary.json',summary);emit('whole_sequence_complete',accepted_steps=problem.accepted,retained_step=summary['retained_step'],stop=summary['stop'],result=summary['result'])
    execution.close();return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['project','stage','parameter','source-sequence','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parameter-stem',default='spin_sequence_global_1105');p.add_argument('--visible-contours',type=Path)
    p.add_argument('--structure-integration',type=Path)
    p.add_argument('--wall-seconds',type=float,default=1200.);p.add_argument('--max-accepted',type=int,default=160)
    a=p.parse_args();run(a.project,a.stage,a.parameter,a.source_sequence,a.output,wall_seconds=a.wall_seconds,max_accepted=a.max_accepted,parameter_stem=a.parameter_stem,visible_contours=a.visible_contours,structure_integration=a.structure_integration)
if __name__=='__main__':main()
