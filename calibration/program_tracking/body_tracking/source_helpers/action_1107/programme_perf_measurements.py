import json,statistics,sys
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def stats(root,run,device):
    e=records(run/'progress.jsonl')
    run_summary=read(run/'run_summary.json')
    mem_folder=run.parent/(run.name+'_resources')
    mem=records(mem_folder/'memory.jsonl')
    total=read(mem_folder/'memory_summary.json')
    metrics=read(run/'final_metrics.json')
    j=[x for x in e if x['event']=='joint_evaluation' and x['with_jacobian']]
    r=[x for x in e if x['event']=='joint_evaluation' and not x['with_jacobian']]
    def median(rows,key):
        values=[x[key] for x in rows if key in x]
        return statistics.median(values) if values else None
    solved=[x for x in e if x['event']=='linear_solve']
    distinct={}
    for x in solved:
        key=x.get('linearization_id')
        if key is not None: distinct.setdefault(key,x)
    sample=None
    if run.name.startswith('A') and (root/'A_peak_os_sample.json').exists():
        sample=read(root/'A_peak_os_sample.json')['peak_ram']
    t=0.
    steps=[]
    for v in e:
        if v['event']=='joint_evaluation': t=v['seconds']
        elif v['event']=='accepted': steps.append(dict(step=v['iteration'],seconds=t,cost=v['cost']))
    peak_gpu=max(x['device_used_bytes'] for x in records(device))
    return dict(stage=str(run),frames=run_summary['source_frames'],
        stage_wall_s=run_summary['wall_seconds'],first_J_s=run_summary['first_full_J_seconds'],
        warm_J_median_s=median(j[1:],'evaluation_seconds'),residual_median_s=median(r,'evaluation_seconds'),
        initial_J_phases=j[0]['evaluation_phases'],
        warm_J_phases={k:median([v['evaluation_phases']for v in j[1:]],k) for k in j[0]['evaluation_phases']},
        initial_foot_profile=j[0].get('fused_compatibility_timing'),
        warm_foot_profile=next((v.get('fused_compatibility_timing') for v in j[1:] if v.get('fused_compatibility_timing')),None),
        source_guard_rejections=sum(v['event']=='trial_source_group_quality_rejected' for v in e),
        recorded_trf_non_improving=sum(v['event']=='trf_trial_decision' and not v['improves'] for v in e) if distinct else None,
        first_J_start_s=next((v['setup_seconds'] for v in e if v['event']=='initial_linearization_start'),
                             j[0]['seconds']-j[0]['evaluation_seconds']),
        first_J_snapshot_s=next((v['seconds'] for v in e if v['event']=='first_J_snapshot'),None),
        output_export_s=next((v['seconds'] for v in e if v['event']=='result_export'),None),
        checkpoint_median_s=median([v for v in e if v['event']=='checkpoint_written'],'seconds'),
        total_cpu_core_s=total['cpu_core_seconds'],
        average_cpu_cores=total['cpu_core_seconds']/total['seconds'],
        peak_working_set_bytes=max(sample or 0,*[max(x.get('peak_working_set_bytes',0),x['working_set_bytes'])for x in mem]),
        peak_process_commit_bytes=total['peak_commit_bytes'],device_NVML_peak_bytes=peak_gpu,
        system_minimum_free_RAM_bytes=min(x['system_free_physical_bytes']for x in mem),
        linear_solve_calls=len(solved),largest_normal_relative_error=max((v['relative_normal_residual']for v in solved),default=None),
        linear_actuals=solved,
        analysis_count=solved[-1].get('session_analysis_count',len(solved)) if solved else 0,
        analysis_reuse_count=sum(bool(x.get('analysis_reused'))for x in solved),
        factor_reuse_count=sum(bool(x.get('numeric_factor_reused'))for x in solved),
        jacobian_structure_reuse_count=sum(bool(x.get('jacobian_structure_reused'))for x in distinct.values()),
        H2D_unique_bytes=sum(v['jacobian_H2D_bytes'] for v in distinct.values()) if distinct else None,
        Gram_unique_seconds=sum(v['gram_GPU_wall_seconds'] for v in distinct.values()) if distinct else None,
        guarded_source_valid=run_summary['original_primary_guards_pass'],
        steps=steps,retained_step=read(run/'solve/candidate_selection.json')['retained_step'],
        groups=metrics['groups'],final_weighted_cost=sum(metrics['weighted_parts'].values()),
        parameter=run_summary['parameter'],result=run_summary['result'],stop=run_summary['stop'])

root,a,b=map(Path,sys.argv[1:])
v=dict(A=stats(root,a,root/'A_device_v2.jsonl'),B=stats(root,b,root/'B_device.jsonl'))
v['same_steps']=[dict(step=x['step'],A_seconds=x['seconds'],B_seconds=y['seconds'],cost_difference=y['cost']-x['cost'])
                for x,y in zip(v['A']['steps'],v['B']['steps'])]
with (root/'programme_ab_measurements.json').open('x',encoding='utf8') as stream:
    json.dump(v,stream,indent=2)
print(json.dumps(v['same_steps'],indent=2))
