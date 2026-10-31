"""Run disjoint original-PTS ice rereads and verify complete shot coverage."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time


def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def run(args):
    project=Path(__file__).resolve().parents[4]
    entry=Path(__file__).with_name('capture_legacy_ice_bulk.py')
    launcher=project/'scripts/run_native_entry.py'
    manifest=read(args.prepared/'joint_manifest.json')
    library=Path(manifest['native_library'])
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    logs=output/'worker_logs';logs.mkdir(exist_ok=True)
    workers=[];counts={}
    for value in args.shot_shards:
        sid,count=value.split(':');count=int(count)
        if sid not in ('s07','s08','s09') or sid in counts or count<1:raise ValueError('Invalid shot shard declaration')
        counts[sid]=count
    if set(counts)!={'s07','s08','s09'}:raise ValueError('All three complete shots are required')
    index=read(args.root/'index.json');source={s['shot_id']:s for s in index['shots']}
    # Every shard is a contiguous slice of registered exposures, never a
    # sampled subset. Keep the actual source domain fixed in every process.
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=str(project/'src')+os.pathsep+str(entry.parent)
    base=[sys.executable,'-B',str(launcher),'--library',str(library),str(entry),
        '--root',str(args.root),'--prepared',str(args.prepared),'--state',str(args.state),
        '--legacy',str(args.legacy),'--output',str(output),'--profile-backend',args.profile_backend,
        '--shots','s07','s08','s09']
    start=time.monotonic()
    for sid,count in counts.items():
        for shard in range(count):
            path=logs/f'{sid}_{shard:02d}_of_{count:02d}.log'
            handle=path.open('wb')
            command=base+['--worker-shot',sid,'--shard-index',str(shard),'--shard-count',str(count)]
            process=subprocess.Popen(command,cwd=project,env=env,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            workers.append(dict(shot=sid,shard=shard,count=count,process=process,handle=handle,log=str(path)))
    identity=dict(schema='rinkspace.disjoint_source_reread_workers.v1',
        entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        profile_backend=args.profile_backend,
        workers=[dict(shot=w['shot'],shard=w['shard'],count=w['count'],pid=w['process'].pid,log=w['log']) for w in workers],
        no_fit_started=True,prepared_modified=False)
    write(output/'workers.json',identity);print(json.dumps(identity),flush=True)
    remaining=list(workers);finished=[]
    while remaining:
        if (output/'stop_requested').exists():
            for worker in remaining:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(worker['process'].pid),'/T','/F'],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                else:worker['process'].terminate()
            for worker in remaining:worker['process'].wait();worker['handle'].close()
            write(output/'worker_status.json',dict(stopped=True,complete=False,finished=finished))
            return 2
        for worker in list(remaining):
            code=worker['process'].poll()
            if code is None:continue
            worker['handle'].close();remaining.remove(worker)
            item=dict(shot=worker['shot'],shard=worker['shard'],exit_code=code,log=worker['log'],elapsed_s=time.monotonic()-start)
            finished.append(item);print(json.dumps(item),flush=True)
            write(output/'worker_status.json',dict(remaining=len(remaining),finished=finished,complete=False))
        if remaining:time.sleep(1.)
    if any(w['exit_code'] for w in finished):return 1
    coverage={}
    for sid in counts:
        scope=source[sid]['processing']
        expected=list(range(scope['first_pts'],scope['stop_pts_exclusive'],256))
        actual=sorted(int(p.name.split('.')[0]) for p in (output/sid).glob('[0-9]*.json.gz'))
        if actual!=expected:raise ValueError(f'{sid}: source exposure coverage differs: {len(actual)}/{len(expected)}')
        paths={};total=0
        for pts in expected:
            record=read(output/sid/f'{pts}.json.gz')
            if record['pts']!=pts or record['source_video_sha256']!=index['source']['actual_sha256']:
                raise ValueError('Source identity changed')
            raw=record['raw_candidates'];file=Path(raw['path'])
            if hashlib.sha256(file.read_bytes()).hexdigest()!=raw['sha256']:raise ValueError('Candidate array digest differs')
            total+=record['summary']['actual_RGB_candidates']
            for name,item in record['summary']['by_path'].items():
                aggregate=paths.setdefault(name,dict(queried_frames=0,candidate_frames=0,fragment_frames=0,rows=0,source_pts=[]))
                aggregate['queried_frames']+=int(item['queries']>0)
                aggregate['candidate_frames']+=int(item['actual_RGB_candidates']>0)
                aggregate['fragment_frames']+=int(item['partial_or_complete_fragments']>0)
                aggregate['rows']+=item['rows']
                if item['partial_or_complete_fragments']:aggregate['source_pts'].append(pts)
        coverage[sid]=dict(frames=len(expected),first_pts=expected[0],last_pts=expected[-1],
            actual_RGB_candidates=total,paths=paths,all_source_frames_processed=True,
            all_visible_features_captured=False,fit_eligible=False)
        write(output/sid/'coverage.json',coverage[sid])
    write(output/'coverage.json',dict(shots=coverage,elapsed_s=time.monotonic()-start,
        profile_backend=args.profile_backend,
        candidate_recovery_complete=True,physical_identity_and_source_accuracy_review_complete=False,no_fit_started=True))
    write(output/'worker_status.json',dict(complete=True,finished=finished,elapsed_s=time.monotonic()-start))
    print(json.dumps(dict(complete=True,elapsed_s=time.monotonic()-start)),flush=True)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','prepared','state','legacy','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shot-shards',nargs='+',default=['s07:3','s08:1','s09:5'])
    p.add_argument('--profile-backend',choices=['cpu','cuda'],default='cuda')
    raise SystemExit(run(p.parse_args()))
