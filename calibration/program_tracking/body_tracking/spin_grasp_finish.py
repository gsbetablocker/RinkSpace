"""Finish at most one retained-state update inside the same measured budget."""
from pathlib import Path
import json,time
import numpy as np
from rinkspace.storage import AssetStore
from rinkspace.path_policy import guard_project_write
from .spin_grasp_refinement import make_problem,archive_paths
from .spin_grasp_factors import reassociate_support
from .skate_run_policy import RunPolicy
from .skate_integration import restore_checkpoint_layers
from .exmr_integration import save_packet
from .rebuild_prepare import load,digest
from .fullbody_round import write_json

def finish(project,base,output,label,budget=1500.,max_steps=6,parameter_stem=None):
    case=output/'common'/label;old=json.loads((case/'run_summary.json').read_text(encoding='utf8'));prof=json.loads((case/'profile.json').read_text(encoding='utf8'))
    if (output/'active_candidate.json').exists():return
    remaining=budget-prof['total_profile_seconds']-old['seconds']
    if len(old['phases'])!=1 or old['accepted_steps']>=max_steps or remaining<2*prof['jacobian_seconds']+25:return
    paths=archive_paths(project,output,parameter_stem);store=AssetStore(project)
    source_archive=store.resolve(old['parameter'],base=store.project,must_exist=True)
    archive=store.resolve(old.get('finished_parameter',paths['finished_parameter']),base=store.project)
    if source_archive!=paths['parameter'] or archive!=paths['finished_parameter']:raise ValueError('Run summary differs from the recorded archive names')
    if archive.exists():raise FileExistsError(f'Preserve existing finished parameter archive: {archive}')
    # Same explicit initializer and objective. This is the planned second bounded
    # association phase, using the retained full-state controls, not a new model.
    tick=time.perf_counter()
    with (case/'finish_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem,review,x0,info,supports,doc=make_problem(project,base,output,label,tuple(old['action_context']),log)
        saved=load(source_archive);x=saved['x'].copy()
        old_arcs=json.loads(str(saved.get('grasp_arc_ids_json',np.array('{}'))))
        if old_arcs:problem.support_image.arc_ids.update(old_arcs)
        changes=reassociate_support(problem.support_image,x[17:]);problem.cache=problem.cache_x=None
        initial=problem.evaluate(x0,False);seed=problem.evaluate(x,False)
        write_json(case/'finish_association.json',dict(associations=changes,changed=sum(v['new_arc']!=v['old_arc'] for v in changes),original_initial_cost=float(initial['residual']@initial['residual']/2),retained_seed_cost=float(seed['residual']@seed['residual']/2),same_new_target_reassessment=True,reason='remaining budget supports one full accepted update; first solve stopped at its planned three-step allocation, not convergence'))
        problem.output=case/'fit_finish';problem.jacobian_seconds_hint=prof['jacobian_seconds'];problem.checkpoint_extra['grasp_arc_ids_json']=np.array(json.dumps(problem.support_image.arc_ids,sort_keys=True))
        remaining-=time.perf_counter()-tick
        x,report=problem.solve(initial=x,run_policy=RunPolicy(wall_seconds=max(0.,remaining-10.),max_accepted=1,max_outer=1))
        checkpoint=load(problem.output/'candidate.npz');retained=int(report['selection']['retained_step'])
        summary=dict(previous_accepted=old['accepted_steps'],numerically_accepted=problem.accepted,retained_additional=retained,total_accepted=old['accepted_steps']+problem.accepted,total_retained=old['accepted_steps']+retained,seconds=time.perf_counter()-tick,profile_and_solver_seconds=prof['total_profile_seconds']+old['seconds']+time.perf_counter()-tick,budget_seconds=budget,accuracy_accepted=False)
        write_json(case/'finish_summary.json',summary)
        if retained==0:
            write_json(output/'finish_not_adopted.json',dict(summary=summary,reason='No further quality-valid retained update; original three-step state preserved',report=report));return
        checkpoint.update({k:saved[k] for k in ('initial_reference','initial_reference_sha256','camera_prior_sha256','original_gvhmr_pose_anchor')})
        final_metrics=dict(action=problem.action_factors.evaluate(x[17:],False)['metrics'],source_regions=review.evaluate(x[17:],False)['metrics'])
        restore_checkpoint_layers(problem.reference,checkpoint,include_primary=True)
        target=case/'finished';packet=save_packet(problem,info,supports,doc,target,status='joint_fit',report=report)
        checkpoint.update(result_reference=np.array('finished/fitted/result.npz'),result_reference_sha256=np.array(digest(target/'fitted/result.npz')))
        if archive.exists():raise FileExistsError('Preserve existing finished parameter archive')
        guard_project_write(archive,project_root=project)
        np.savez_compressed(archive,**checkpoint)
        write_json(case/'finished_factor_metrics.json',final_metrics)
        for name in ('phase_comparison.json','grasp_world_comparison.png','delivery.json','GRASP_RESULT_ZH.txt'):
            path=output/name
            if path.exists():
                keep=path.with_name(path.stem+'_stage1'+path.suffix)
                if not keep.exists():keep.write_bytes(path.read_bytes())
        write_json(output/'active_candidate.json',dict(parameter=str(archive),result=str(target/'fitted/result.npz'),summary=summary,first_stage_result=str(case/'fitted/result.npz')))
        print(json.dumps(dict(event='grasp_finish_retained',**summary)),flush=True)
