"""Refresh affected source-only shape families using existing guard exclusions.

Original candidate arrays and fragments are read only. The existing clean-bit
filter sees an in-memory flags copy; source IDs and measured UV never change.
"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
import time

import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.session.source_shape_consensus import known_shape_profile_hypotheses


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    data=Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as stream:
        yield from (json.loads(line)for line in stream)


def payload_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()


def write(path,value):
    data=(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8')
    path=guard_project_write(path)
    path.write_bytes(gzip.compress(data,compresslevel=1,mtime=0) if str(path).endswith('.gz') else data)


def refresh_families(measurement,fragments,family_curves,previous,affected):
    """Use the original algorithm and its original 85% consensus rule."""
    affected=np.asarray(affected,np.int64)
    count=len(measurement['candidate_query'])
    if (affected.ndim!=1 or len(np.unique(affected))!=len(affected)
            or (affected<0).any() or (affected>=count).any()):
        raise ValueError('Guard IDs must be unique original candidate indices')
    candidate_curve=measurement['query_curve'][measurement['candidate_query']]
    affected_curves=set(map(int,candidate_curve[affected]))
    changed=[f for f,indices in family_curves.items() if affected_curves.intersection(indices)]
    result=dict(previous)
    flags=np.asarray(measurement['candidate_flags']).copy()
    flags[affected]|=1
    guarded=dict(measurement,candidate_flags=flags)
    details={}
    for feature in changed:
        hypotheses=known_shape_profile_hypotheses(guarded,family_curves[feature],fragments)
        plausible=[h for h in hypotheses if h['score']>=.85*hypotheses[0]['score']] if hypotheses else []
        selected=sorted(set.intersection(*(set(h['candidate_ids'])for h in plausible))) if plausible else []
        result[feature]=dict(hypotheses=hypotheses,plausible_alternatives=len(plausible),
            consensus_candidate_ids=selected,identity_from_query_name=False)
        before=set(previous[feature]['consensus_candidate_ids']);after=set(selected)
        details[feature]=dict(consensus_before=len(before),consensus_after=len(after),
            original_consensus_excluded=len(before.intersection(affected)),
            consensus_removed=len(before-after),consensus_added=len(after-before))
    unchanged={f:payload_digest(v)for f,v in previous.items()if f not in changed}
    if any(payload_digest(result[f])!=sha for f,sha in unchanged.items()):
        raise ValueError('Unaffected family payload changed')
    for feature,value in result.items():
        selected=np.asarray(value['consensus_candidate_ids'],np.int64)
        if (set(map(int,selected)).intersection(affected) or
                not np.isin(candidate_curve[selected],family_curves[feature]).all()):
            raise ValueError('Guarded consensus leaves its original clean candidate family')
        if any(set(h['candidate_ids']).intersection(affected)for h in value['hypotheses']):
            raise ValueError('Excluded candidate supplied a shape hypothesis')
    return result,changed,unchanged,details


_CONTEXT=None


def worker_context(context):
    global _CONTEXT
    _CONTEXT=context
    cv2.setNumThreads(1)


def refresh_frame(job):
    entry,exclusion=job;context=_CONTEXT
    sid,pts=entry['shot'],entry['pts'];raw=entry['original_candidates']
    if (exclusion['capture']!=context['capture_name'] or exclusion['shot']!=sid or exclusion['pts']!=pts
            or Path(exclusion['raw_measurement']['path']).resolve()!=Path(raw['path']).resolve()
            or exclusion['raw_measurement']['sha256']!=raw['sha256']
            or exclusion['source_BGR_sha256']!=entry['source_BGR_sha256']
            or exclusion['original_UV_and_flags_unchanged'] is not True
            or exclusion['applied_to_all_original_candidate_ids'] is not True
            or exclusion.get('automatic_admission')):
        raise ValueError('Guard exclusion does not bind the same original candidate space')
    if digest(raw['path'])!=raw['sha256'] or digest(entry['result_path'])!=entry['result_sha256']:
        raise ValueError('Original measurement or shape result changed')
    array=exclusion['arrays']
    if digest(array['path'])!=array['sha256']:raise ValueError('Guard exclusion array changed')
    with np.load(array['path'],allow_pickle=False)as z:
        affected=z['affected_candidate_ids'].copy();bits=z['trigger_bits']
        if affected.ndim!=1 or bits.shape!=affected.shape or len(affected)!=exclusion['affected_candidate_count']:
            raise ValueError('Invalid guard exclusion array dimensions')
        if len(affected) and (not np.all(np.diff(affected)>0) or not np.all(bits)):
            raise ValueError('Guard candidate IDs must be sorted unique with recorded triggers')
    evidence=dict(coverage=context['exclusions'],index=context['exclusion_index'],
        capture=context['capture_name'],shot=sid,pts=pts,arrays=array,
        guard_candidate=exclusion['guard_candidate'],affected_candidate_count=len(affected))
    result_entry=dict(entry,guard_exclusion=evidence)
    receipt=dict(affected_candidates=len(affected),refreshed_frames=0,reused_frames=0,
        recomputed_family_count=0,original_consensus_excluded=0,consensus_removed=0,consensus_added=0,
        unchanged_family_payloads_exact=True,guarded_consensus_exclusion_intersection=0,
        raw_candidate_SHA_unchanged=True)
    if not len(affected):
        receipt['reused_frames']=1
        result_entry['guard_refresh']=dict(reused_original_result=True,recomputed_families=[],**receipt)
        return result_entry,receipt
    with np.load(raw['path'],allow_pickle=False)as z:measurement=dict(z)
    original_flags=measurement['candidate_flags'].copy()
    uv_sha=hashlib.sha256(measurement['candidate_uv'].tobytes()).hexdigest()
    previous=read(entry['result_path']);metadata=read(Path(context['capture'])/sid/f'{pts}.json.gz')
    if (metadata['raw_candidates']['sha256']!=raw['sha256']
            or previous['measurement_sha256']!=raw['sha256']
            or previous['source_BGR_sha256']!=entry['source_BGR_sha256']
            or metadata['source_BGR_sha256']!=entry['source_BGR_sha256']
            or previous['source_video_sha256']!=context['source_video_sha256']
            or previous['shot']!=sid or previous['pts']!=pts
            or set(previous['shape_families'])!=set(context['family_curves'])
            or previous['source_UV_unchanged'] is not True or previous['fit_eligible']):
        raise ValueError('Previous shape/source exposure binding differs')
    families,changed,unchanged,details=refresh_families(measurement,metadata['fragments'],
        context['family_curves'],previous['shape_families'],affected)
    if (not np.array_equal(measurement['candidate_flags'],original_flags)
            or hashlib.sha256(measurement['candidate_uv'].tobytes()).hexdigest()!=uv_sha
            or digest(raw['path'])!=raw['sha256']):
        raise ValueError('Original measurement flags or UV changed')
    if not changed:
        receipt['reused_frames']=1
        result_entry['guard_refresh']=dict(reused_original_result=True,recomputed_families=[],**receipt)
        return result_entry,receipt
    receipt['refreshed_frames']=1;receipt['recomputed_family_count']=len(changed)
    for value in details.values():
        for key in ('original_consensus_excluded','consensus_removed','consensus_added'):receipt[key]+=value[key]
    provenance=dict(exclusions=evidence,original_shape_result=dict(path=entry['result_path'],sha256=entry['result_sha256']),
        recomputed_families=changed,unchanged_family_payload_sha256=unchanged,family_changes=details,
        original_fragments_and_native_queries_reused=True,in_memory_flags_clean_bit_added_only=True,
        affected_consensus_intersection=0,source_UV_and_original_flags_unchanged=True,
        plausibility_score_fraction=.85,identity_from_query_name=False,fit_eligible=False)
    output=Path(context['output'])/sid/f'{pts}.json.gz'
    refreshed=dict(previous,shape_families=families,guard_refresh=provenance,
        source_UV_unchanged=True,fit_eligible=False,physical_identity_confirmed=False)
    write(output,refreshed)
    result_entry.update(result_path=str(output.resolve()),result_sha256=digest(output),
        families={f:dict(hypotheses=len(v['hypotheses']),plausible_alternatives=v['plausible_alternatives'],
                        consensus_readouts=len(v['consensus_candidate_ids']))for f,v in families.items()},
        guard_refresh=dict(reused_original_result=False,recomputed_families=changed,**receipt))
    return result_entry,receipt


def run(args):
    started=time.perf_counter();previous=read(args.shape_coverage);guard=read(args.exclusions)
    if (previous.get('complete') is not True or previous.get('source_UV_unchanged') is not True
            or previous.get('shape_model')!='native-similarity' or previous.get('fit_eligible')):
        raise ValueError('Complete source-only native-similarity results required')
    if (guard.get('all_prior_shape_and_formal_impacts_contained') is not True
            or guard.get('source_candidate_id_spaces_separate') is not True or guard.get('complete',True) is not True):
        raise ValueError('Completed all-candidate exclusion coverage required')
    capture=Path(previous['capture']);capture_name=capture.name;binding_path=capture/'binding.json'
    binding=read(binding_path)
    if (digest(binding_path)!=previous['capture_binding_sha256']
            or guard['capture_bindings'][capture_name]['sha256']!=previous['capture_binding_sha256']
            or Path(guard['capture_bindings'][capture_name]['path']).resolve()!=binding_path.resolve()):
        raise ValueError('Guard and shape results use different captures')
    algorithm=Path(known_shape_profile_hypotheses.__code__.co_filename).resolve()
    algorithm_sha=digest(algorithm)
    if previous['frozen_implementations'].get(str(algorithm))!=algorithm_sha:
        raise ValueError('Known-shape algorithm differs from original results')
    original_index=previous['unique_results_index'];exclusion_index=guard['index']
    if digest(original_index['path'])!=original_index['sha256'] or digest(exclusion_index['path'])!=exclusion_index['sha256']:
        raise ValueError('Unique shape/exclusion index changed')
    entries=sorted(rows(original_index['path']),key=lambda r:(r['shot'],r['pts']))
    all_exclusions=list(rows(exclusion_index['path']));exclusions={}
    for row in all_exclusions:
        key=row['capture'],row['shot'],row['pts']
        if key in exclusions:raise ValueError('Duplicate capture/shot/PTS guard evidence')
        exclusions[key]=row
    if len(entries)!=3866 or len(exclusions)!=7732:raise ValueError('Complete3866-frame captures required')
    expected={(e['shot'],e['pts'])for e in entries}
    if len(expected)!=3866 or {(s,p)for c,s,p in exclusions if c==capture_name}!=expected:
        raise ValueError('Guard and shape exposure coverage differs')
    for sid,shot in previous['shots'].items():
        if sorted(p for s,p in expected if s==sid)!=list(range(shot['first_pts'],shot['last_pts']+256,256)):
            raise ValueError('Missing original contiguous native exposure')
    output=guard_project_write(args.output.resolve())
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    for sid in previous['shots']:(output/sid).mkdir()
    features=previous['features'];family_curves={f:[i for i,p in enumerate(binding['native_inventory'])
        if p['feature_id']==f and p['closed']]for f in features}
    if any(not c for c in family_curves.values()):raise ValueError('Missing captured closed family')
    context=dict(capture=str(capture),capture_name=capture_name,output=str(output),family_curves=family_curves,
        source_video_sha256=previous['source_video_sha256'],exclusions=ref(args.exclusions),
        exclusion_index=exclusion_index)
    frozen={str(Path(__file__).resolve()):digest(__file__),str(algorithm):algorithm_sha}
    receipt=dict(schema='rinkspace.guarded_known_shape_refresh_binding.v1',original_shape_coverage=ref(args.shape_coverage),
        exclusions=ref(args.exclusions),capture=str(capture),capture_binding_sha256=previous['capture_binding_sha256'],
        frozen_implementations=frozen,features=features,shape_model='native-similarity',workers=args.workers,
        original_fragments_and_native_queries_reused=True,plausibility_score_fraction=.85,
        only_in_memory_candidate_flags_clean_bit_changed=True,source_UV_unchanged=True,
        fit_eligible=False,physical_identity_confirmed=False,new_fit_started=False)
    write(output/'binding.json',receipt)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    jobs=((e,exclusions[capture_name,e['shot'],e['pts']])for e in entries)
    totals=Counter();shots={};result_rows=[];pool=None
    try:
        if args.workers==1:
            worker_context(context);iterator=map(refresh_frame,jobs)
        else:
            pool=ProcessPoolExecutor(max_workers=args.workers,initializer=worker_context,initargs=(context,))
            iterator=pool.map(refresh_frame,jobs,chunksize=1)
        for count,(entry,detail)in enumerate(iterator,1):
            result_rows.append(entry)
            for key,value in detail.items():
                if type(value)is int:totals[key]+=value
            sid=entry['shot'];shot=shots.setdefault(sid,dict(frames=0,first_pts=entry['pts'],last_pts=entry['pts'],
                families={f:dict(frames=0,with_hypotheses=0,consensus_frames=0,consensus_readouts=0,
                    ambiguous_alternative_frames=0)for f in features},no_missing_or_duplicate_PTS=True))
            shot['frames']+=1;shot['last_pts']=entry['pts']
            for family,value in entry['families'].items():
                c=shot['families'][family];c['frames']+=1;c['with_hypotheses']+=bool(value['hypotheses'])
                c['consensus_frames']+=bool(value['consensus_readouts']);c['consensus_readouts']+=value['consensus_readouts']
                c['ambiguous_alternative_frames']+=value['plausible_alternatives']>1
            if count%250==0:print(json.dumps(dict(stage='refresh',capture=capture_name,frames=count,
                refreshed_frames=totals['refreshed_frames'],elapsed_seconds=time.perf_counter()-started)),flush=True)
    except BaseException:
        if pool:
            for process in list((pool._processes or {}).values()):process.terminate()
            pool.shutdown(wait=False,cancel_futures=True)
        raise
    else:
        if pool:pool.shutdown()
    if len(result_rows)!=3866:raise ValueError('Incomplete guarded shape output')
    for path,sha in frozen.items():
        if digest(path)!=sha:raise ValueError('Adapter or known-shape algorithm changed during refresh')
    if digest(exclusion_index['path'])!=exclusion_index['sha256'] or digest(args.exclusions)!=context['exclusions']['sha256']:
        raise ValueError('Exclusion evidence changed during refresh')
    index_path=output/'results_index.jsonl.gz'
    with gzip.open(index_path,'wt',encoding='utf8',compresslevel=1)as stream:
        for entry in result_rows:stream.write(json.dumps(entry,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(schema='rinkspace.full_legacy_paint_known_shape_organization.v1',capture=str(capture),
        capture_binding_sha256=previous['capture_binding_sha256'],source_video_sha256=previous['source_video_sha256'],
        features=features,shape_model='native-similarity',frozen_implementations=frozen,frames=3866,shots=shots,
        unique_results_index=ref(index_path),guard_refresh_binding=ref(output/'binding.json'),
        original_shape_coverage=ref(args.shape_coverage),exclusions=ref(args.exclusions),refresh_counts=dict(totals),
        unaffected_results_reused_by_original_path_and_SHA=True,unaffected_family_payloads_exact=True,
        all_consensus_exclusion_intersections_zero=True,all_original_candidate_SHA_and_source_exposure_bindings_verified=True,
        no_missing_or_duplicate_results=True,complete=True,source_UV_unchanged=True,original_flags_unchanged=True,
        conics_are_observations=False,fit_eligible=False,physical_identity_confirmed=False,
        prepared_modified=False,new_fit_started=False,elapsed_seconds=time.perf_counter()-started)
    write(output/'coverage.json',report)
    print(json.dumps(dict(capture=capture_name,output=str(output/'coverage.json'),counts=dict(totals),
        consensus_readouts=sum(f['consensus_readouts']for s in shots.values()for f in s['families'].values()),
        elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('shape-coverage','exclusions','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=9)
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    run(args)
