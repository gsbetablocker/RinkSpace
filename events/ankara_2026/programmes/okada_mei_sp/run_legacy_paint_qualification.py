"""Run unchanged known-shape organization over disjoint complete-shot ranges."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np
from rinkspace.path_policy import guard_project_write
from rinkspace.session import source_shape_consensus, material_rings
from run_legacy_ice_links import read, write, digest, terminate_owned


def validate(args,jobs,index,capture_binding,frozen,started):
    expected={};source={s['shot_id']:s for s in index['shots']}
    for sid in ('s07','s08','s09'):
        scope=source[sid]['processing']
        expected[sid]=list(range(scope['first_pts'],scope['stop_pts_exclusive'],256))
    inventory=capture_binding['native_inventory']
    curves={ci:p for ci,p in enumerate(inventory) if p['feature_id'] in args.features and p['closed']}
    family_curves={f:{ci for ci,p in curves.items() if p['feature_id']==f} for f in args.features}
    entries={};ranges=[];worker_counts={sid:Counter()for sid in expected}
    for job in jobs:
        stem=f'{job["shot"]}_{job["first_pts"]}_{job["stop_pts"]}'
        binding=read(job['output']/(stem+'_binding.json'))
        if (binding['code_sha256']!=frozen[str(job['entry'])]
                or binding['known_shape_algorithm_sha256']!=frozen[str(Path(source_shape_consensus.__file__).resolve())]
                or binding['algorithm_sha256']!=frozen[str(Path(material_rings.__file__).resolve())]
                or binding['capture_binding_sha256']!=digest(args.capture/'binding.json')
                or binding['source_video_sha256']!=index['source']['actual_sha256']
                or binding['first_pts']!=job['first_pts'] or binding['stop_pts_exclusive']!=job['stop_pts']
                or binding['features']!=args.features or binding['shape_model']!='native-similarity'
                or binding['fit_eligible'] or binding['physical_identity_confirmed'] or not binding['source_UV_unchanged']):
            raise ValueError('Worker source/algorithm/range binding changed')
        coverage=read(job['output']/(stem+'_coverage.json'))
        wanted=list(range(job['first_pts'],job['stop_pts'],256))
        if coverage['frames']!=len(wanted) or coverage['binding']!=binding:raise ValueError('Incomplete worker range')
        folder=job['output']/job['shot']
        actual=sorted(int(p.name.split('.')[0])for p in folder.glob('*.json.gz')if p.name.split('.')[0].isdigit())
        if actual!=wanted:raise ValueError('Worker native frame result coverage differs')
        for pts in wanted:
            key=(job['shot'],pts)
            if key in entries:raise ValueError('Duplicate shape result source exposure')
            entries[key]=folder/f'{pts}.json.gz'
        for label,count in coverage['by_curve'].items():
            for k,v in count.items():worker_counts[job['shot']][label+'|'+k]+=v
        ranges.append(dict(shot=job['shot'],first_pts=job['first_pts'],stop_pts_exclusive=job['stop_pts'],
            binding=str(job['output']/(stem+'_binding.json')),coverage=str(job['output']/(stem+'_coverage.json')),
            elapsed_seconds=coverage['elapsed_s']))
    output_index=args.output/'results_index.jsonl.gz';shots={};total=0
    with gzip.open(output_index,'wt',encoding='utf8',compresslevel=1)as stream:
        for sid,pts_list in expected.items():
            if sorted(pts for shot,pts in entries if shot==sid)!=pts_list:raise ValueError('Missing/extra complete-shot shape result')
            counts={p['curve_id']:dict(frames=0,with_hypotheses=0,consensus_frames=0,consensus_readouts=0,
                hypothesis_pts=[],consensus_pts=[])for p in curves.values()}
            families={f:dict(frames=0,with_hypotheses=0,consensus_frames=0,consensus_readouts=0,ambiguous_alternative_frames=0)
                for f in args.features}
            for pts in pts_list:
                path=entries[sid,pts];record=read(path)
                original_json=args.capture/sid/f'{pts}.json.gz';original=read(original_json)
                measurement=args.capture/sid/f'{pts}.npz';sha=digest(measurement)
                if (record['pts']!=pts or record['shot']!=sid
                        or record['source_video_sha256']!=index['source']['actual_sha256']
                        or record['source_BGR_sha256']!=original['source_BGR_sha256']
                        or Path(record['measurement_path']).resolve()!=measurement.resolve()
                        or record['measurement_sha256']!=sha or sha!=original['raw_candidates']['sha256']
                        or not record['source_UV_unchanged'] or record['fit_eligible'] or record['physical_identity_confirmed']
                        or set(record['shape_families'])!=set(args.features)):
                    raise ValueError('Result source/PTS/UV/eligibility binding differs')
                with np.load(measurement,allow_pickle=False)as z:
                    cq=z['candidate_query'];qc=z['query_curve']
                family_index={}
                for feature,result in record['shape_families'].items():
                    if result['identity_from_query_name']:raise ValueError('Query name became physical identity')
                    hypotheses=result['hypotheses'];selected=np.asarray(result['consensus_candidate_ids'],np.int64)
                    if (len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=len(cq))
                            or any(int(ci)not in family_curves[feature]for ci in qc[cq[selected]])):
                        raise ValueError('Consensus IDs do not reference original candidate family')
                    for h in hypotheses:
                        ids=np.asarray(h['candidate_ids'],np.int64)
                        if (np.any(ids<0)or np.any(ids>=len(cq))or any(int(ci)not in family_curves[feature]for ci in qc[cq[ids]])):
                            raise ValueError('Hypothesis candidate IDs leave their captured family')
                        if h.get('fit_eligible')or h.get('physical_identity_confirmed'):raise ValueError('Hypothesis was admitted')
                    f=families[feature];f['frames']+=1;f['with_hypotheses']+=bool(hypotheses)
                    f['consensus_frames']+=bool(len(selected));f['consensus_readouts']+=len(selected)
                    f['ambiguous_alternative_frames']+=result['plausible_alternatives']>1
                    family_index[feature]=dict(hypotheses=len(hypotheses),plausible_alternatives=result['plausible_alternatives'],
                        consensus_readouts=len(selected))
                    for ci in family_curves[feature]:
                        c=counts[curves[ci]['curve_id']];c['frames']+=1
                        has=any(np.any(qc[cq[np.asarray(h['candidate_ids'],np.int64)]]==ci)for h in hypotheses)
                        n=int(np.sum(qc[cq[selected]]==ci));c['with_hypotheses']+=has
                        c['consensus_frames']+=bool(n);c['consensus_readouts']+=n
                        if has:c['hypothesis_pts'].append(pts)
                        if n:c['consensus_pts'].append(pts)
                stream.write(json.dumps(dict(shot=sid,pts=pts,result_path=str(path),result_sha256=digest(path),
                    original_candidates=dict(path=str(measurement),sha256=sha),source_BGR_sha256=record['source_BGR_sha256'],
                    families=family_index,source_UV_unchanged=True,fit_eligible=False),separators=(',',':'))+'\n')
            for label,c in counts.items():
                for k in ('frames','with_hypotheses','consensus_frames','consensus_readouts'):
                    if c[k]!=worker_counts[sid][label+'|'+k]:raise ValueError('Per-curve aggregate differs from worker results')
            shots[sid]=dict(frames=len(pts_list),first_pts=pts_list[0],last_pts=pts_list[-1],
                families=families,by_curve=counts,no_missing_or_duplicate_PTS=True)
            total+=len(pts_list)
    if total!=3866:raise ValueError('Complete S07-S09 source qualification must cover 3866 frames')
    for path,sha in frozen.items():
        if digest(path)!=sha:raise ValueError('Algorithm/script changed while workers ran')
    report=dict(schema='rinkspace.full_legacy_paint_known_shape_organization.v1',capture=str(args.capture.resolve()),
        capture_binding_sha256=digest(args.capture/'binding.json'),source_video_sha256=index['source']['actual_sha256'],
        features=args.features,shape_model='native-similarity',frozen_implementations=frozen,
        frames=total,shots=shots,worker_ranges=ranges,unique_results_index=dict(path=str(output_index),sha256=digest(output_index)),
        all_original_candidate_SHA_and_source_exposure_bindings_verified=True,no_missing_or_duplicate_results=True,
        complete=True,source_UV_unchanged=True,conics_are_observations=False,fit_eligible=False,
        physical_identity_confirmed=False,prepared_modified=False,new_fit_started=False,
        elapsed_seconds=time.monotonic()-started)
    write(args.output/'coverage.json',report)
    return report


def run(args):
    started=time.monotonic();args.output=guard_project_write(args.output.resolve())
    if(args.output/'workers.json').exists():raise FileExistsError('Use an independent immutable shape output')
    args.output.mkdir(parents=True,exist_ok=True);logs=args.output/'worker_logs';logs.mkdir(exist_ok=True)
    entry=Path(__file__).with_name('qualify_legacy_paint_curves.py').resolve();project=entry.parents[4]
    index=read(args.root/'index.json');capture_binding=read(args.capture/'binding.json');sources={s['shot_id']:s for s in index['shots']}
    implementations={str(entry),str(Path(__file__).resolve()),str(Path(source_shape_consensus.__file__).resolve()),
        str(Path(material_rings.__file__).resolve()),str(Path(read.__code__.co_filename).resolve())}
    frozen={p:digest(p)for p in sorted(implementations)}
    counts={}
    for value in args.shot_shards:
        sid,count=value.split(':');count=int(count)
        if sid not in ('s07','s08','s09')or sid in counts or count<1:raise ValueError('Invalid shot shard declaration')
        counts[sid]=count
    if set(counts)!={'s07','s08','s09'}or sum(counts.values())>9:raise ValueError('S07-S09 require at most nine owned workers')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=str(project/'src')+os.pathsep+str(entry.parent)
    jobs=[]
    try:
        for sid,count in counts.items():
            s=sources[sid]['processing'];pts=np.arange(s['first_pts'],s['stop_pts_exclusive'],256,dtype=np.int64)
            for shard,chunk in enumerate(np.array_split(pts,count)):
                if not len(chunk):raise ValueError('Empty qualification source shard')
                first,stop=int(chunk[0]),int(chunk[-1])+256
                output=args.output/'worker_parts'/f'{sid}_{shard:02d}_of_{count:02d}'
                log=logs/f'{sid}_{shard:02d}_of_{count:02d}.log';handle=log.open('wb')
                command=[sys.executable,'-B',str(entry),'--root',str(args.root),'--capture',str(args.capture),
                    '--output',str(output),'--shot',sid,'--first-pts',str(first),'--stop-pts',str(stop),
                    '--features',*args.features,'--shape-model','native-similarity']
                try:p=subprocess.Popen(command,cwd=project,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                except BaseException:handle.close();raise
                jobs.append(dict(shot=sid,shard=shard,first_pts=first,stop_pts=stop,output=output,entry=entry,
                    process=p,handle=handle,log=log))
        write(args.output/'workers.json',dict(schema='rinkspace.disjoint_known_shape_source_workers.v1',
            capture=str(args.capture.resolve()),features=args.features,shape_model='native-similarity',
            frozen_implementations=frozen,overlap_source_frames=0,
            workers=[dict(shot=j['shot'],shard=j['shard'],first_pts=j['first_pts'],stop_pts=j['stop_pts'],
                output=str(j['output']),pid=j['process'].pid,log=str(j['log']))for j in jobs],
            fit_eligible=False,new_fit_started=False))
        print(json.dumps(dict(stage='workers_started',capture=str(args.capture),workers=len(jobs),features=args.features)),flush=True)
        pending=list(jobs);finished=[]
        while pending:
            if(args.output/'stop_requested').exists():raise RuntimeError('Owned shape qualification stop requested')
            for job in list(pending):
                code=job['process'].poll()
                if code is None:continue
                job['handle'].close();pending.remove(job)
                item=dict(shot=job['shot'],shard=job['shard'],exit_code=code,log=str(job['log']),elapsed_seconds=time.monotonic()-started)
                finished.append(item);print(json.dumps(dict(stage='worker_finished',**item)),flush=True)
                if code:raise RuntimeError(f'Shape qualification worker failed: {item}')
            write(args.output/'worker_status.json',dict(complete=False,remaining=len(pending),finished=finished,
                live=[dict(pid=j['process'].pid,log=str(j['log']),log_bytes=j['log'].stat().st_size)for j in pending],
                elapsed_seconds=time.monotonic()-started))
            if pending:time.sleep(1.)
        report=validate(args,jobs,index,capture_binding,frozen,started)
        write(args.output/'worker_status.json',dict(complete=True,remaining=0,finished=finished,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(stage='validated',capture=str(args.capture),frames=report['frames'],
            elapsed_seconds=report['elapsed_seconds'])),flush=True)
    except BaseException as error:
        terminate_owned(jobs);write(args.output/'worker_status.json',dict(complete=False,remaining=0,error=repr(error),
            owned_process_trees_stopped=True,elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--features',nargs='+',required=True)
    p.add_argument('--shot-shards',nargs='+',default=['s07:3','s08:1','s09:5'])
    run(p.parse_args())
