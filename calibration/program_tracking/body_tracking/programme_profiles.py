"""Serial, bounded scale/depth experiments on one saved programme state.

Each initializer/solve has its own process. GPU allocations cannot accumulate
between branches; failed experiments are retained without automatic retries.
This driver does not choose candidates, train, infer networks or open a GUI.
"""
from pathlib import Path
import argparse,json,os,re,subprocess,sys,time
from .rebuild_prepare import digest
from .fullbody_round import write_json


def run(project,parent_prepared,parent_parameter,spec,output):
    project,parent_prepared,parent_parameter,spec,output=map(lambda p:Path(p).resolve(),
        (project,parent_prepared,parent_parameter,spec,output))
    if output.exists():raise FileExistsError(output)
    cases=json.loads(spec.read_text(encoding='utf8'))['cases'];names=[v['name']for v in cases]
    if not cases or len(set(names))!=len(names)or any(re.fullmatch(r'[A-Za-z0-9_]+',v)is None for v in names):
        raise ValueError('Distinct stable experiment names required')
    parent_sha=digest(parent_parameter)
    output.mkdir();report=dict(parent_parameter=str(parent_parameter),parent_parameter_sha256=parent_sha,
        parent_prepared=str(parent_prepared),spec=str(spec),spec_sha256=digest(spec),cases=[],
        serial_GPU_processes=True,no_automatic_retry=True,candidate_selection_performed=False)
    write_json(output/'execution.json',report)
    prefix=[sys.executable,'-u','-B','-m','calibration.program_tracking.body_tracking.']
    for case in cases:
        if(output/'STOP').is_file():report['stopped_before_case']=case['name'];break
        if digest(parent_parameter)!=parent_sha:raise ValueError('Common parent state changed')
        prepared,start,stage=map(lambda k:Path(case[k]).resolve(),('prepared','start','stage'))
        for path in(start,stage):
            if path.exists():raise FileExistsError('Preserve existing experiment '+str(path))
        record=dict(case,started_unix=time.time(),status='initializing');report['cases'].append(record)
        write_json(output/'execution.json',report)
        init=prefix[:-1]+[prefix[-1]+'programme_continue','--project',str(project),
            '--parent-prepared',str(parent_prepared),'--parent-parameter',str(parent_parameter),
            '--prepared',str(prepared),'--output',str(start),'--scale-factor',str(case.get('scale_factor',1.)),
            '--depth-m',str(case.get('depth_m',0.))]
        solve=prefix[:-1]+[prefix[-1]+'trajectory_backend','--project',str(project),'--prepared',str(prepared),
            '--output',str(stage),'--resume',str(start/'solve/candidate.npz'),
            '--wall-seconds',str(case['wall_seconds']),'--max-accepted',str(case['max_accepted']),
            '--memory-gib','0','--gpu-linear','--device-batch','1024','--diagnostic-profile']
        if case.get('freeze_root',False):solve.append('--freeze-root')
        for phase,command in(('initialize',init),('solve',solve)):
            log=output/(case['name']+'_'+phase+'.log');env=os.environ.copy()
            if phase=='initialize':env['CUDA_VISIBLE_DEVICES']=''
            record.update(status=phase,command=command,log=str(log));write_json(output/'execution.json',report)
            with log.open('x',encoding='utf8')as stream:
                process=subprocess.Popen(command,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record['active_pid']=process.pid;write_json(output/'execution.json',report)
                code=process.wait()
            record.pop('active_pid',None);record[phase+'_exit_code']=code
            if code:record.update(status='failed',failed_phase=phase);break
        else:
            summary=json.loads((stage/'run_summary.json').read_text(encoding='utf8'))
            record.update(status='completed',summary=str(stage/'run_summary.json'),
                actual_wall_seconds=summary['wall_seconds'],accepted_steps=summary['additional_accepted_steps'],
                original_primary_guards_pass=summary['original_primary_guards_pass'])
        record['finished_unix']=time.time();record.pop('command',None)
        write_json(output/'execution.json',report);print(json.dumps(record),flush=True)
    report['finished_unix']=time.time();write_json(output/'execution.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('project','parent-prepared','parent-parameter','spec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.project,a.parent_prepared,a.parent_parameter,a.spec,a.output)


if __name__=='__main__':main()
