"""Independent source cuts, each with one shared ordered RGB replay.

Workers may overlap independent temporal states. They never share or
advance another cut's graph, export RGB caches, fit, or train a model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.strict_model_scope import file_sha, authenticate_source_reference


def freeze_source_registry(registry_path, output_dir, *, mutable_authority_paths=()):
    """Snapshot mutable source records; reuse immutable user image/packet refs."""
    registry_path=Path(registry_path).resolve();output_dir=Path(output_dir).resolve()
    original_sha=file_sha(registry_path)
    data=json.loads(registry_path.read_text(encoding='utf8'))
    mutable={str(Path(p).resolve()).casefold() for p in mutable_authority_paths}
    copied={}
    output_dir.mkdir(parents=True,exist_ok=True)
    def visit(value):
        if isinstance(value,list):return [visit(v) for v in value]
        if not isinstance(value,dict):return value
        value={k:visit(v) for k,v in value.items()}
        if value.get('path') and value.get('sha256'):
            path=Path(value['path']).resolve()
            if str(path).casefold() in mutable:
                if file_sha(path)!=value['sha256']:raise ValueError('Registry authority changed before freezing')
                target=output_dir/(path.stem+'_FROZEN_'+value['sha256'][:16]+path.suffix)
                if target.exists():
                    if file_sha(target)!=value['sha256']:raise ValueError('Earlier frozen authority differs')
                else:shutil.copyfile(path,target)
                copied[str(path)]=dict(path=str(target),sha256=value['sha256'])
                value['path']=str(target)
        return value
    data=visit(data)
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf8')
    target=output_dir/(registry_path.stem+'_FROZEN_'+hashlib.sha256(encoded).hexdigest()[:16]+'.json')
    if target.exists():
        if target.read_bytes()!=encoded:raise ValueError('Earlier source registry snapshot differs')
    else:
        with target.open('xb') as saved:saved.write(encoded)
    if file_sha(registry_path)!=original_sha:raise ValueError('Source registry changed while freezing')
    return dict(path=str(target),sha256=file_sha(target),mutable_authority_snapshots=list(copied.values()),
        original_registry_snapshot_sha256=original_sha)


def source_cut_inventory(identity_path, shot_manifest_path):
    identity=json.loads(Path(identity_path).read_text(encoding='utf8'))
    if file_sha(identity['camera_path'])!=identity['camera_sha256']:
        raise ValueError('Source native clock/camera changed')
    with np.load(identity['camera_path'],allow_pickle=False) as camera:
        native=camera['pts'];actual=camera['exposure_representative_pts']
    shots=json.loads(Path(shot_manifest_path).read_text(encoding='utf8'))['shots']
    cuts=[];seen=set()
    for shot in shots:
        selected=(native>=int(shot['start_pts']))&(native<int(shot['end_pts']))
        if not selected.any():continue
        clock=native[selected];canon=clock==actual[selected]
        if not canon[0] or not set(map(int,actual[selected])).issubset(set(map(int,clock))):
            raise ValueError('Independent source cut needs its own canonical first exposure')
        points=set(map(int,clock))
        if seen&points:raise ValueError('Parallel cuts cannot repeat a native exposure')
        seen.update(points)
        cuts.append(dict(shot_id=shot['id'],start_pts=int(clock[0]),stop_pts=int(clock[-1])+1,
            native_count=len(clock),canonical_count=int(canon.sum())))
    if seen!=set(map(int,native)):raise ValueError('Cut inventory omitted native source images')
    return cuts


def combine_board_source_configs(references, output):
    """Preserve exact source seed/authority semantics without broader approval."""
    result={key:[] for key in ('packet_paths','corner_annotation_paths',
        'confirmed_identity_references','source_annotation_seeds')}
    seen={key:set() for key in result};numbers={}
    for reference in references:
        path,_=authenticate_source_reference(reference)
        original=json.loads(path.read_text(encoding='utf8'))
        for key in result:
            for row in original.get(key,()):
                literal=json.dumps(row,sort_keys=True,separators=(',',':'))
                if literal in seen[key]:continue
                if key=='source_annotation_seeds':
                    name=row['anchor_id']
                    if name in numbers:
                        prior=result[key][numbers[name]]
                        if any(prior[k]!=row[k] for k in set(prior)&set(row)):
                            raise ValueError('One board anchor has competing source facts')
                        # A later authenticated record may add current-shot
                        # restrictions/template navigation while preserving
                        # every original pixel, role and authority verbatim.
                        result[key][numbers[name]]={**prior,**row}
                        seen[key].add(literal)
                        continue
                    numbers[name]=len(result[key])
                seen[key].add(literal);result[key].append(row)
    result.update(schema='rinkspace.combined-exact-board-source-config.v1',
        original_config_references=references,authority_scopes_unchanged=True,
        whole_programme_review_complete=False)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8') as saved:
        json.dump(result,saved,ensure_ascii=False,separators=(',',':'))
    return dict(path=str(output.resolve()),sha256=file_sha(output))


def run_source_cut_manifest(manifest_path, output_dir, *, workers=1):
    """Execute only frozen argv; each child's durable source pack is independent."""
    manifest_path=Path(manifest_path).resolve();output_dir=Path(output_dir).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema')!='rinkspace.exact-source-cut-batch.v1':
        raise ValueError('Frozen source-cut manifest required')
    references=manifest['references']
    for reference in references:authenticate_source_reference(reference)
    common=manifest['common_arguments'];cuts=manifest['cuts']
    if any(option in common for option in ('--output','--start-pts','--stop-pts')):
        raise ValueError('Only the cut scheduler owns output and source ranges')
    if workers<1:raise ValueError('Positive source worker count required')
    if any(a['start_pts']<b['stop_pts'] and b['start_pts']<a['stop_pts']
            for i,a in enumerate(cuts) for b in cuts[i+1:]):
        raise ValueError('Source cuts overlap')
    output_dir.mkdir(parents=True,exist_ok=True)
    def notice(value):print(json.dumps(value,ensure_ascii=False,separators=(',',':')),flush=True)
    def run(cut):
        identifier=re.sub(r'[^A-Za-z0-9_.-]','_',cut['shot_id'])
        output=output_dir/(identifier+'.npz');log=output_dir/(identifier+'.log')
        if output.exists() or log.exists():raise ValueError('Preserve an earlier source cut attempt')
        if manifest.get('ice_graph_class'):
            argv=[sys.executable,str(ROOT/'scripts/admit_programme_source_cut_variant.py'),
                '--manifest',str(manifest_path),'--shot-id',cut['shot_id'],'--output',str(output)]
        else:
            argv=[sys.executable,str(ROOT/'scripts/admit_programme_source_inputs.py'),*common,
                '--start-pts',str(cut['start_pts']),'--stop-pts',str(cut['stop_pts']),'--output',str(output)]
        began=time.monotonic()
        with log.open('x',encoding='utf8') as stream:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            notice(dict(stage='source_cut_started',shot_id=cut['shot_id'],pid=child.pid,
                source_start_pts=cut['start_pts'],source_stop_pts=cut['stop_pts']))
            status=child.wait()
        receipt=dict(cut,exit_code=status,elapsed_seconds=time.monotonic()-began,
            output_path=str(output),log_path=str(log))
        if status==0:
            if not output.exists():raise ValueError('Successful source worker omitted its pack')
            receipt['sha256']=file_sha(output)
        notice(dict(stage='source_cut_terminal',**receipt))
        return receipt
    began=time.monotonic();receipts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(run,cut) for cut in cuts]
        for future in as_completed(pending):receipts.append(future.result())
    for reference in references:
        if file_sha(reference['path'])!=reference['sha256']:
            raise ValueError('Frozen source inputs/code changed during the batch')
    result=dict(schema='rinkspace.exact-source-cut-batch-receipt.v1',
        manifest_reference=dict(path=str(manifest_path),sha256=file_sha(manifest_path)),
        worker_count=workers,elapsed_seconds=time.monotonic()-began,
        cuts=sorted(receipts,key=lambda r:r['start_pts']),
        all_source_workers_succeeded=all(r['exit_code']==0 for r in receipts),
        source_processing_is_not_manual_audit_completion=True,solver_started=False)
    with (output_dir/'batch_receipt.json').open('x',encoding='utf8') as saved:
        json.dump(result,saved,ensure_ascii=False,separators=(',',':'))
    notice(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    result=run_source_cut_manifest(args.manifest,args.output_dir,workers=args.workers)
    if not result['all_source_workers_succeeded']:raise SystemExit(1)


if __name__=='__main__':main()
