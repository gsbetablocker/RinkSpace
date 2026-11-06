"""Bake a saved unified sequence checkpoint, without another solve or inference."""
from pathlib import Path
import argparse,copy,io,json,time
import numpy as np
from .rebuild_prepare import load,digest
from .blade_current_reference import restore_current_reference
from .absolute_motion_state import absolute_reference,curve_from_packet
from .sequence_skate_state import shoe_curve_from_packet
from .body_execution import BodyExecution,BodyExecutionConfig
from .spin_continuous_refit import ContinuousJoint
from .blade_boot_refit import _mass
from .exmr_integration import CASE_FIELD,case_identity,validate_case_identity,save_packet
from .skate_checkpoint import validate_checkpoint_identity
from .skate_integration import source_evidence
from .support_image import SupportImageFactors
from .fullbody_round import write_json


def export(project,geometry_stage,geometry_parameter,stage,checkpoint,parameter,reason='diagnosed_missing_visible_steel_image_residual'):
    project,geometry_stage,geometry_parameter,stage,checkpoint,parameter=map(Path,(project,geometry_stage,geometry_parameter,stage,checkpoint,parameter))
    from rinkspace.path_policy import guard_project_write
    guard_project_write(parameter,project_root=project)
    if parameter.exists() or (stage/'common/exmr_hand_nofoot/fitted/result.npz').exists():raise FileExistsError('Preserve existing sequence exports')
    saved=load(checkpoint);parent=load(geometry_parameter);case=stage/'common/exmr_hand_nofoot'
    ref,camera,info,_=restore_current_reference(project,geometry_stage,geometry_parameter,seed=False)
    validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    expected=case_identity(project,stage.parents[1],stage,'exmr_hand_nofoot',info,initial=case/'initial.npz')
    validate_case_identity(saved,expected)
    # This is the *same* exact current initial byte stream. Repair only this
    # new export's legacy archive pointer; no old checkpoint is migrated.
    archive=project/str(parent['initial_reference'])
    if not archive.is_file() or digest(archive)!=str(parent['initial_reference_sha256']) or digest(case/'initial.npz')!=digest(archive):raise ValueError('Initial archive identity missing or different')
    saved['initial_reference']=parent['initial_reference'].copy();saved['initial_reference_sha256']=parent['initial_reference_sha256'].copy()
    saved['sequence_raw_solver_checkpoint_sha256']=np.array(digest(checkpoint))
    final=absolute_reference(ref,curve_from_packet(saved),root=case);final.relative_skate_curve=shoe_curve_from_packet(saved)
    final.fixed_checkpoint_identity=dict(ref.fixed_checkpoint_identity);final.fixed_checkpoint_identity[CASE_FIELD]=saved[CASE_FIELD].copy()
    final.export_source_path=case/'export_source.npz';final.observations_path=case/'observations.npz'
    source=json.loads((stage/'support_and_free_shoe_regions.json').read_text(encoding='utf-8'));final.source_document=source
    final.footwear_observations=dict(visible_rows=[],blade_rows=[],landmark_rows=[])
    execution=BodyExecution(BodyExecutionConfig(workers=1,blas_threads=1));execution.attach(final)
    supports=source_evidence(case,shot_id='s05',source_pts=final.p['pts'],records=source['supports'])
    with execution.limits():
        problem=ContinuousJoint(final,camera,case/'observations.npz',(76.,95.),supports,stage/'export_not_a_solve',io.StringIO(),'sequence_skates_v1',optimize_shared=False,ice_policy=final.ice_policy)
        runtime=copy.deepcopy(source);mult=final.grasp_relations['factor_parameters']['shoe_region_weight_multiplier']
        for row in runtime['observations']:row['objective_weight']*=mult
        problem.support_image=SupportImageFactors(problem,runtime,arc_ids={})
        review=SupportImageFactors(problem,runtime,include_review=True,arc_ids={})
        source_after=review.evaluate(np.zeros(problem.body.size))['metrics']
        packet=save_packet(problem,info,supports,source,case,status='joint_fit')
        metadata={k:v for k,v in saved.items() if k.startswith(('absolute_','relative_skate_','sequence_contact_')) or k in ('state_schema_version','state_kind','no_inherited_high_frequency_curve','body_parameter_count')}
        packet.update(metadata)
        mass,mass_path=_mass(final,stage);mid=final.at(final.times);C,R=mass.evaluate(mid,np.zeros((len(final.times),69)))
        packet.update(com_world_mid=mass.com(C),absolute_parameter_path=np.array(str(parameter)),sequence_pose_estimate_status=np.full((len(final.times),2),'inferred_joint_sequence'))
        status=np.full((len(final.times),2),'unknown',dtype='<U16');lookup={int(v):i for i,v in enumerate(final.p['pts'])}
        for row in source['observations']:
            if row.get('role')!='fit' or not row.get('optimization_enabled',True) or row.get('foot') not in (0,1):continue
            key=(lookup[int(row['pts'])],row['foot']);value='inferred' if row.get('evidence_status')=='inferred' or row.get('source_evidence_origin')=='temporal_source_region_inference' else 'observed'
            if status[key]!='observed':status[key]=value
        packet['sequence_image_evidence_status']=status
        np.savez_compressed(case/'fitted/result.npz',**packet)
    with parameter.open('xb') as stream:np.savez_compressed(stream,**saved)
    policy=json.loads((stage/'solve/run_policy.json').read_text(encoding='utf-8'));start=json.loads((stage/'start_quality.json').read_text(encoding='utf-8'));latest=policy['latest']
    summary=dict(accepted_steps=latest['accepted'],retained_step=latest['accepted'],stop=dict(reason=reason,not_convergence=True),
        parameter=str(parameter),result=str(case/'fitted/result.npz'),whole_sequence_seconds=[76.,95.],source_frames=951,fit_frames=951,
        initial_cost=start['cost'],final_cost=latest['joint_cost'],image_cost_before=start['image_cost'],image_cost_after=latest['image_cost'],
        initial_parts=start['parts'],source_regions_before=start['source_regions'],source_regions_after=source_after,
        raw_solver_checkpoint=str(checkpoint),raw_solver_checkpoint_sha256=digest(checkpoint),
        exact_initial_archive_pointer_repaired=True,initial_archive=str(archive),initial_archive_sha256=digest(archive),
        no_new_solve=True,all_951_trajectory_estimated=True,accuracy_accepted=False,user_accepted=False,not_default=True)
    write_json(stage/'run_summary.json',summary);execution.close();return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','geometry-stage','geometry-parameter','stage','checkpoint','parameter']:p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--reason',default='diagnosed_missing_visible_steel_image_residual')
    a=p.parse_args();r=export(a.project,a.geometry_stage,a.geometry_parameter,a.stage,a.checkpoint,a.parameter,a.reason)
    print(json.dumps({k:r[k]for k in ['accepted_steps','parameter','result','stop']}),flush=True)
if __name__=='__main__':main()
