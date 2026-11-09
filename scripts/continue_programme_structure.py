"""Continue the existing global solver, refreshing true surface witnesses per step.

A one-step boundary versions the nonlinear active set; it is not a run budget
or an accuracy claim. No network inference, local trajectory or default change.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
import numpy as np
from calibration.program_tracking.body_tracking.rebuild_prepare import digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('project','prepared','source-evidence','warm-start','quality-baseline','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--review-after-refresh',action='store_true',
        help='Return at the next complete refreshed step for source review')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'inputs').mkdir();clock=time.monotonic()
    log=(a.output/'progress.jsonl').open('a',encoding='utf8',buffering=1)
    def emit(event,**record):
        row=dict(event=event,elapsed_s=time.monotonic()-clock,**record)
        log.write(json.dumps(row,ensure_ascii=False)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
    def call(script,args,logfile,*,solver=False):
        started=time.monotonic()
        emit('phase_start',round=round_number,script=script,log=str(logfile))
        command=([sys.executable,'-B','-X','utf8','scripts/run_body_performance.py','--backend','hybrid',script,*map(str,args)]
                 if solver else [sys.executable,'-B','-X','utf8',script,*map(str,args)])
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(a.project/'src'),str(a.project)))
        env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with logfile.open('w',encoding='utf8')as stream:
            run=subprocess.run(command,cwd=a.project,env=env,stdout=stream,stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if run.returncode:
            emit('stage_error',script=script,returncode=run.returncode,log=str(logfile))
            raise RuntimeError('Stage failed; no automatic retry: '+str(logfile))
        emit('phase_complete',round=round_number,script=script,log=str(logfile),wall_s=time.monotonic()-started)
    doc=json.loads(a.source_evidence.read_text('utf8'));parameter=a.warm_start;round_number=0
    quality_baseline=a.quality_baseline
    while True:
        round_number+=1;stage=a.output/f'r{round_number:03d}'
        source=a.output/'inputs'/f'source_{round_number:03d}.json'
        source.write_text(json.dumps(doc,separators=(',',':'),ensure_ascii=False),encoding='utf8')
        emit('solver_start',round=round_number,main_log=str(stage/'progress.jsonl'),warm_start=str(parameter),
             source_document=str(source),source_document_sha256=digest(source))
        call('scripts/fit_programme_structure.py',['--project',a.project,'--prepared',a.prepared,
            '--source-evidence',source,'--output',stage,'--warm-start',parameter,
            '--quality-baseline',quality_baseline,'--max-accepted','1',
            '--gpu-linear','--memory-gib','0','--source-conditioned-step'],a.output/f'r{round_number:03d}_solver.log',solver=True)
        summary=json.loads((stage/'run_summary.json').read_text('utf8'))
        if summary['additional_accepted_steps']<1:
            emit('needs_diagnosis',round=round_number,reason='No accepted whole-state step; not accuracy acceptance',
                 summary=str(stage/'run_summary.json'))
            break
        after=stage/'solve/candidate.npz'
        final_metrics=json.loads((stage/'final_metrics.json').read_text('utf8'))
        diagnostics=final_metrics.get('physical_diagnostics',{})
        opening=diagnostics.get('opening_source_standing') or {}
        with np.load(stage/'fitted/result.npz',allow_pickle=False)as packet:
            acceleration=packet['com_acceleration_world_mps2'];pts=packet['source_pts'] if 'source_pts'in packet else packet['pts']
            magnitude=np.linalg.norm(acceleration,axis=1);lsp=(pts>=int(156.2*12800))&(pts<=int(165.2*12800))
            physical=dict(com_acceleration_max_mps2=float(magnitude.max()),lsp_com_acceleration_max_mps2=float(magnitude[lsp].max()))
        if opening:
            height=np.asarray(opening['minimum_steel_height_m'])
            physical.update(opening_steel_height_range_m=[float(height.min()),float(height.max())],opening_quiet_P_range_m=opening['quiet_P_range_m'],opening_hand_waist_max_m=opening['hand_waist_max_distance_m'])
        physical['opening_contact_mode']=opening.get('standing_contact_mode','legacy_unspecified')
        physical['opening_height_interpretation']=dict(
            reference='conditional model ice plane',calibrated_accuracy_m=None,
            establishes_real_contact_or_penetration=False,engineering_scales_are_not_accuracy_tolerances=True)
        if 'steel_height_by_foot_m'in opening:
            each=np.asarray(opening['steel_height_by_foot_m'],float)
            if each.shape!=(len(opening['source_pts']),2) or not np.isfinite(each).all():raise ValueError('Incomplete per-foot opening heights')
            physical['opening_per_foot_steel_height_range_m']=np.c_[each.min(axis=0),each.max(axis=0)].tolist()
            physical['opening_both_feet_max_abs_height_m']=(float(np.abs(each).max())if opening.get('standing_contact_mode')=='both_feet'else None)
        physical['opening_held_hand_waist_max_m']=opening.get('held_hand_waist_max_distance_m')
        physical['opening_release_hand_waist_max_m']=opening.get('release_hand_waist_max_distance_m')
        physical['joint_domains']=diagnostics.get('anatomical_joint_domain',{})
        physical['com_quadrature']=diagnostics.get('actual_model_com_acceleration',{})
        if diagnostics.get('confirmed_flight_dense_com'):
            physical['confirmed_flight_dense_com']=diagnostics['confirmed_flight_dense_com']
        physical['grasp_distances']=[{key:row[key]for key in ('hand','foot','source_seconds','median_distance_m','p95_distance_m','max_distance_m')}
            for row in (diagnostics.get('source_supported_grasp')or{}).get('episodes',[])]
        emit('saved_motion_diagnostics',round=round_number,**physical)
        call('scripts/scan_programme_surface.py',['--project',a.project,'--prepared',a.prepared,
            '--source-evidence',source,'--parameter',after,'--output',stage/'surface'],a.output/f'r{round_number:03d}_surface.log')
        call('scripts/refine_programme_surface.py',['--project',a.project,'--prepared',a.prepared,
            '--source-evidence',source,'--parameter',after,'--base-scan',stage/'surface/surface_scan.npz',
            '--output',stage/'adaptive_surface'],a.output/f'r{round_number:03d}_adaptive.log')
        witnesses=stage/'adaptive_surface/surface_scan.npz'
        call('scripts/compare_surface_refresh.py',['--project',a.project,'--prepared',a.prepared,
            '--source-evidence',source,'--before',parameter,'--after',after,'--witnesses',witnesses,
            '--stage',stage,'--output',stage/'refreshed_objective.json'],a.output/f'r{round_number:03d}_refresh.log')
        refresh=json.loads((stage/'refreshed_objective.json').read_text('utf8'))
        surface=json.loads(witnesses.with_suffix('.json').read_text('utf8'))
        record=dict(round=round_number,parameter=str(after),parameter_sha256=digest(after),
            source_document=str(source),source_document_sha256=digest(source),
            witnesses=str(witnesses),witnesses_sha256=digest(witnesses),
            same_refreshed_cost_before=refresh['full_objective_before_on_new_association'],
            same_refreshed_cost_after=refresh['full_objective_after_on_new_association'],
            refreshed_nonincrease=refresh['nonincrease_on_new_association'],
            sampled_crossing_pairs=surface['total_crossing_pairs'],
            sample_count=surface['total_samples'],physical_diagnostics=physical,accuracy_accepted=False)
        (a.output/'latest.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        emit('surface_feedback_complete',**record)
        if not record['refreshed_nonincrease']:
            emit('needs_diagnosis',round=round_number,reason='Refreshed actual surface objective increased; retained as failed candidate, no silent continuation')
            break
        parameter=after
        quality_baseline=stage/'primary_quality_baseline.json'
        doc['programme_structure']['surface_witnesses']=str(witnesses)
        doc['programme_structure']['surface_witnesses_sha256']=digest(witnesses)
        if a.review_after_refresh or (a.output/'REVIEW_AFTER_REFRESH').exists():
            emit('review_checkpoint',round=round_number,reason='Explicit review of actual state after surface refresh; not convergence')
            break
    log.close()

if __name__=='__main__':main()
