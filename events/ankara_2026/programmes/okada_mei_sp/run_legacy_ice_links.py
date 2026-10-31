"""Run complete-shot adjacent RGB links in disjoint, one-frame-overlap ranges."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from rinkspace.path_policy import guard_project_write


def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    Path(path).write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if str(path).endswith('.gz') else data)


def terminate_owned(workers):
    for w in workers:
        p=w['process']
        if p.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            else:p.terminate()
    for w in workers:
        w['process'].wait()
        if not w['handle'].closed:w['handle'].close()


def load_nodes(record):
    p=Path(record['nodes']['path'])
    if digest(p)!=record['nodes']['sha256']:raise ValueError('Source node NPZ digest changed')
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def validate(args,jobs,index,started):
    frame_entries={};pair_entries={};aliases=[];expected={};ranges=[]
    source={s['shot_id']:s for s in index['shots']}
    for sid in ('s07','s08','s09'):
        s=source[sid]['processing'];expected[sid]=list(range(s['first_pts'],s['stop_pts_exclusive'],256))
    for job in jobs:
        summary_path=job['output']/f'{job["shot"]}_{job["first_pts"]}_{job["stop_pts"]}.summary.json.gz'
        summary=read(summary_path)
        if not(summary['full_requested_range_processed'] and summary['source_hashes_and_exact_node_correspondence_verified']):
            raise ValueError('Worker did not verify its full source range')
        if summary['fit_eligible'] or summary['physical_identity_confirmed'] or summary['new_fit_started']:
            raise ValueError('Source-link worker changed eligibility or identity')
        ranges.append(dict(shot=job['shot'],first_pts=job['first_pts'],owned_first_pts=job['owned_first_pts'],
            stop_pts=job['stop_pts'],summary=str(summary_path),elapsed_seconds=summary['elapsed_seconds']))
        folder=job['output']/job['shot']
        actual=sorted(int(p.name.split('.')[0])for p in folder.glob('*.json.gz')if p.name.split('.')[0].isdigit())
        if actual!=list(range(job['first_pts'],job['stop_pts'],256)):raise ValueError('Worker frame file coverage differs')
        for pts in actual:
            p=folder/f'{pts}.json.gz';entry=(p,job);key=(job['shot'],pts)
            if key not in frame_entries:frame_entries[key]=entry
            elif pts>=job['owned_first_pts']:
                aliases.append((key,frame_entries[key][0],p));frame_entries[key]=entry
            else:aliases.append((key,p,frame_entries[key][0]))
        for p in folder.glob('*.links.json.gz'):
            a,b=map(int,p.name.split('.')[0].split('_'));key=(job['shot'],a,b)
            if key in pair_entries:raise ValueError('Duplicate adjacent source pair would create repeated evidence')
            pair_entries[key]=p
    overlap=sum(1 for j in jobs if j['first_pts']<j['owned_first_pts'])
    if len(aliases)!=overlap:raise ValueError('Only declared one-frame boundary aliases are permitted')
    alias_records=[]
    for (sid,pts),alias,owner in aliases:
        a,b=read(alias),read(owner);an,bn=load_nodes(a),load_nodes(b)
        if set(an)!=set(bn) or any(not np.array_equal(an[k],bn[k],equal_nan=True)for k in an):
            raise ValueError('Overlapping source nodes differ')
        for k in ('source_candidates','source_RGB_sha256','source_exclusion_sha256','same_frame_conflicts'):
            if a[k]!=b[k]:raise ValueError('Overlapping frame provenance or conflicts differ')
        alias_records.append(dict(shot=sid,pts=pts,canonical_frame=str(owner),alias_frame=str(alias),
            node_arrays_exactly_equal=True,additional_evidence=False))
    frame_index=args.output/'frames_index.jsonl.gz';pair_index=args.output/'pairs_index.jsonl.gz'
    coverage={};total_frames=total_pairs=total_edges=total_ambiguous=total_conflicts=total_node_pairs=0
    with gzip.open(frame_index,'wt',encoding='utf8',compresslevel=1) as fi,gzip.open(pair_index,'wt',encoding='utf8',compresslevel=1) as pi:
        for sid,pts_list in expected.items():
            found=sorted(p for s,p in frame_entries if s==sid)
            found_pairs=sorted((a,b)for s,a,b in pair_entries if s==sid)
            wanted_pairs=list(zip(pts_list[:-1],pts_list[1:]))
            if found!=pts_list or found_pairs!=wanted_pairs:raise ValueError(f'{sid}: full native frame/pair coverage differs')
            previous=None;edges=ambiguous=conflicts=node_pairs=0
            for pts in pts_list:
                path,job=frame_entries[sid,pts];record=read(path)
                if record['pts']!=pts or record['shot']!=sid or record['source_video_sha256']!=index['source']['actual_sha256']:
                    raise ValueError('Frame source identity changed')
                if record['fit_eligible'] or record['physical_identity_confirmed']:raise ValueError('Unexpected source admission')
                nodes=load_nodes(record);raw_path=Path(record['source_candidates']['path'])
                if digest(raw_path)!=record['source_candidates']['sha256']:raise ValueError('Original source candidate NPZ changed')
                with np.load(raw_path,allow_pickle=False) as z:
                    uv=z['candidate_uv'];sides=z['candidate_sides'];query=z['candidate_query'];curves=z['query_curve']
                ids=nodes['ids']
                if (len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(uv))
                        or not np.array_equal(nodes['uv'],uv[ids]) or not np.array_equal(nodes['sides'],sides[ids])
                        or not np.array_equal(nodes['curve'],curves[query[ids]])):
                    raise ValueError('Node rows do not retain exact original source candidate IDs/UV')
                if digest(Path(record['source_frame']['path']))!=record['source_frame']['sha256']:
                    raise ValueError('Original source candidate JSON changed')
                conflicts+=len(record['same_frame_conflicts'])
                item=dict(shot=sid,pts=pts,metadata_path=str(path),nodes=record['nodes'],
                    source_candidates=record['source_candidates'],source_RGB_sha256=record['source_RGB_sha256'],
                    source_exclusion_sha256=record['source_exclusion_sha256'],nodes_count=len(ids),
                    same_frame_conflicts=len(record['same_frame_conflicts']),fit_eligible=False)
                fi.write(json.dumps(item,separators=(',',':'))+'\n')
                if previous is not None:
                    pp=pair_entries[sid,previous['pts'],pts];pair=read(pp)
                    if (pair['previous_pts']!=previous['pts'] or pair['current_pts']!=pts
                            or pair['source_video_sha256']!=index['source']['actual_sha256']
                            or pair['fit_eligible'] or pair['physical_identity_confirmed']):
                        raise ValueError('Pair source/eligibility differs')
                    seen=set()
                    for edge in pair['links']:
                        ix=np.asarray(edge['node_pairs'],np.int64).reshape(-1,2)
                        actual=np.asarray(edge['pairs'],np.int64).reshape(-1,2)
                        if (np.any(ix<0) or np.any(ix[:,0]>=len(previous['nodes']['ids']))
                                or np.any(ix[:,1]>=len(ids))):raise ValueError('Node pair index outside source exposure')
                        mapped=np.c_[previous['nodes']['ids'][ix[:,0]],ids[ix[:,1]]]
                        if not np.array_equal(actual,mapped):raise ValueError('Pairs are not original candidate IDs')
                        if edge['identity_assumed_from_curve_name'] or edge['fit_eligible']:raise ValueError('Curve label became identity')
                        for ab in map(tuple,actual.tolist()):
                            if ab in seen:raise ValueError('Duplicate exact candidate correspondence in source pair')
                            seen.add(ab)
                    n_amb=sum(not e['unambiguous']for e in pair['links'])
                    if n_amb!=pair['ambiguous_links']:raise ValueError('Ambiguous alternatives were changed')
                    edges+=len(pair['links']);ambiguous+=n_amb;node_pairs+=len(seen)
                    pi.write(json.dumps(dict(shot=sid,previous_pts=previous['pts'],current_pts=pts,
                        metadata_path=str(pp),links=len(pair['links']),ambiguous_links=n_amb,
                        original_candidate_pairs=len(seen),fit_eligible=False),separators=(',',':'))+'\n')
                previous=dict(pts=pts,nodes=nodes)
            coverage[sid]=dict(frames=len(pts_list),adjacent_pairs=len(wanted_pairs),first_pts=pts_list[0],last_pts=pts_list[-1],
                links=edges,ambiguous_links=ambiguous,same_frame_conflicts=conflicts,original_candidate_pairs=node_pairs,
                no_missing_or_duplicate_source_frames_or_pairs=True)
            total_frames+=len(pts_list);total_pairs+=len(wanted_pairs);total_edges+=edges
            total_ambiguous+=ambiguous;total_conflicts+=conflicts;total_node_pairs+=node_pairs
    if total_frames!=3866 or total_pairs!=3863:raise ValueError('Expected complete S07-S09 source scope is 3866/3863')
    report=dict(schema='rinkspace.full_legacy_ice_source_links.v1',capture=str(args.capture.resolve()),
        candidate_id_space_capture_binding_sha256=digest(args.capture/'binding.json'),
        shots=coverage,frames=total_frames,adjacent_pairs=total_pairs,links=total_edges,
        ambiguous_links=total_ambiguous,same_frame_conflicts=total_conflicts,original_candidate_pairs=total_node_pairs,
        overlap_frame_aliases=alias_records,overlap_frames_are_additional_evidence=False,
        frames_index=dict(path=str(frame_index),sha256=digest(frame_index)),
        pairs_index=dict(path=str(pair_index),sha256=digest(pair_index)),worker_ranges=ranges,
        all_original_candidate_SHA_and_node_UV_verified=True,no_missing_or_duplicate_evidence=True,
        complete=True,fit_eligible=False,physical_identity_confirmed=False,prepared_modified=False,new_fit_started=False,
        elapsed_seconds=time.monotonic()-started)
    write(args.output/'coverage.json',report)
    return report


def run(args):
    started=time.monotonic();args.output=guard_project_write(args.output.resolve())
    if (args.output/'workers.json').exists():raise FileExistsError('Use an independent immutable link output')
    args.output.mkdir(parents=True,exist_ok=True);logs=args.output/'worker_logs';logs.mkdir(exist_ok=True)
    entry=Path(__file__).with_name('link_legacy_ice_sources.py');project=entry.resolve().parents[4]
    index=read(args.root/'index.json');sources={s['shot_id']:s for s in index['shots']};counts={}
    for value in args.shot_shards:
        sid,count=value.split(':');count=int(count)
        if sid not in ('s07','s08','s09') or sid in counts or count<1:raise ValueError('Invalid source shot shards')
        counts[sid]=count
    if set(counts)!={'s07','s08','s09'} or sum(counts.values())>9:raise ValueError('S07-S09 require at most nine owned workers')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=str(project/'src')+os.pathsep+str(entry.parent)
    jobs=[]
    try:
        for sid,count in counts.items():
            scope=sources[sid]['processing'];pts=np.arange(scope['first_pts'],scope['stop_pts_exclusive'],256,dtype=np.int64)
            for shard,chunk in enumerate(np.array_split(pts,count)):
                if not len(chunk):raise ValueError('Empty source shard')
                owned=int(chunk[0]);first=owned-256 if shard else owned;stop=int(chunk[-1])+256
                output=args.output/'worker_parts'/f'{sid}_{shard:02d}_of_{count:02d}'
                log=logs/f'{sid}_{shard:02d}_of_{count:02d}.log';handle=log.open('wb')
                cmd=[sys.executable,'-B',str(entry),'--root',str(args.root),'--capture',str(args.capture),
                    '--output',str(output),'--shot',sid,'--first-pts',str(first),'--stop-pts',str(stop)]
                try:p=subprocess.Popen(cmd,cwd=project,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                except BaseException:handle.close();raise
                jobs.append(dict(shot=sid,shard=shard,count=count,first_pts=first,owned_first_pts=owned,stop_pts=stop,
                    output=output,log=log,process=p,handle=handle))
        identity=dict(schema='rinkspace.disjoint_source_link_workers.v1',capture=str(args.capture.resolve()),
            capture_binding_sha256=digest(args.capture/'binding.json'),entry=str(entry),entry_sha256=digest(entry),
            workers=[dict(shot=j['shot'],shard=j['shard'],first_pts=j['first_pts'],owned_first_pts=j['owned_first_pts'],
                stop_pts=j['stop_pts'],output=str(j['output']),pid=j['process'].pid,log=str(j['log']))for j in jobs],
            overlap_exposures_per_boundary=1,fit_eligible=False,new_fit_started=False)
        write(args.output/'workers.json',identity);print(json.dumps(dict(stage='workers_started',capture=str(args.capture),workers=len(jobs))),flush=True)
        pending=list(jobs);finished=[]
        while pending:
            if (args.output/'stop_requested').exists():raise RuntimeError('Owned source link stop requested')
            for job in list(pending):
                code=job['process'].poll()
                if code is None:continue
                job['handle'].close();pending.remove(job)
                item=dict(shot=job['shot'],shard=job['shard'],exit_code=code,elapsed_seconds=time.monotonic()-started,log=str(job['log']))
                finished.append(item);print(json.dumps(dict(stage='worker_finished',**item)),flush=True)
                if code:raise RuntimeError(f'Source link worker failed: {item}')
            write(args.output/'worker_status.json',dict(complete=False,remaining=len(pending),finished=finished,
                live=[dict(pid=j['process'].pid,log=str(j['log']),log_bytes=j['log'].stat().st_size)for j in pending],
                elapsed_seconds=time.monotonic()-started))
            if pending:time.sleep(1.)
        report=validate(args,jobs,index,started)
        write(args.output/'worker_status.json',dict(complete=True,remaining=0,finished=finished,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(stage='validated',capture=str(args.capture),frames=report['frames'],adjacent_pairs=report['adjacent_pairs'],
            links=report['links'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    except BaseException as error:
        terminate_owned(jobs);write(args.output/'worker_status.json',dict(complete=False,remaining=0,error=repr(error),
            owned_process_trees_stopped=True,elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shot-shards',nargs='+',default=['s07:3','s08:1','s09:5'])
    run(p.parse_args())
