"""Run explicit shared-camera hypotheses sequentially without changing the baseline."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

HERE=Path(__file__).resolve().parent
RUN=Path('D:/CodexData/RinkSpace_Data/runs/native_joint_s01_s04_20261027')

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='*');args=p.parse_args()
    directory=RUN/'camera_models';baseline=json.loads((directory/'baseline.json').read_text(encoding='utf-8'))
    seed=Path(baseline['snapshot'])
    if hashlib.sha256(seed.read_bytes()).hexdigest()!=baseline['sha256']:raise ValueError('Accepted baseline changed')
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    started=time.monotonic()
    with (directory/'experiments.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        def emit(event,**data):
            rec=dict(event=event,elapsed_s=time.monotonic()-started,**data);log.write(json.dumps(rec)+'\n');print(json.dumps(rec),flush=True)
        profiles=baseline['profiles']
        # Keep consecutive ablations together; each starts from the same frozen seed.
        profiles=sorted(profiles,key=lambda x:(x['scene_mode']!='frozen',next(i for i,n in enumerate(['full','no_wide','constant_pp','low_order','square','square_no_readout','square_no_pupil']) if n==x['model'])))
        for profile in profiles:
            key=Path(profile['path']).stem
            if args.models and key not in args.models:continue
            out=directory/key;out.mkdir(exist_ok=True)
            if (out/'result.json').exists():
                emit('existing_result',model=key,path=str(out/'result.json'));continue
            if (out/'activity.json').exists():
                raise ValueError('An incomplete previous run needs explicit diagnosis/resume: '+str(out))
            command=[sys.executable,'-B',str(HERE/'fit_native_joint_shots.py'),'--phase','joint','--resume',str(seed),'--output',str(out),'--camera-experiment',profile['path']]
            emit('start_model',model=key,main_log=str(out/'solve.jsonl'),progress_log=str(out/'fit_progress.log'),command=command)
            with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
                result=subprocess.run(command,env=env,stdout=stdout,stderr=stderr)
            emit('finish_model',model=key,returncode=result.returncode)
            if result.returncode:
                emit('model_failed',model=key,error=(out/'stderr.log').read_text(encoding='utf-8')[-8000:])
        if hashlib.sha256(seed.read_bytes()).hexdigest()!=baseline['sha256']:raise ValueError('Accepted baseline changed')
        emit('batch_finished',baseline_unchanged=True)

if __name__=='__main__':main()
