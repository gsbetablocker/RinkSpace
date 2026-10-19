"""Check actual correction coverage in the existing configured crop schedule."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from rinkspace.strict_source_training_schedule import source_crop_schedule
from rinkspace.strict_source_review_schedule import reviewed_source_crop_schedule
from rinkspace.vision import configure,load_training_manifest


def audit(config_path,preparation_path,output_path,current_review_priority=False):
    config,data=configure(config_path);manifest,_=load_training_manifest(config,data)
    preparation=json.loads(preparation_path.read_text(encoding='utf8'))
    by_native={};new_by_native={}
    new_sources={r['source_pack_reference']['sha256'] for ref in preparation['exact_review_training_admission_references']
        for r in [json.loads(Path(ref['path']).read_text(encoding='utf8'))]}
    for packet in preparation['packet_records']:
        receipt=packet['supervision_cache_receipt'];authenticate_source_reference(receipt)
        cache_ref=json.loads(Path(receipt['path']).read_text(encoding='utf8'))['cache_reference']
        authenticate_source_reference(cache_ref)
        with gzip.open(cache_ref['path'],'rt',encoding='utf8') as f:cached=json.load(f)
        for native,annotations in cached['annotations_by_native'].items():
            native=int(native)
            selected=[{k:a.get(k) for k in ('state','uv','source_curve_id','physical_material_id','feature_id','kind','boundary_role','authority')}
                for a in annotations]
            by_native.setdefault(native,[]).extend(selected)
            if packet['source_reference']['sha256'] in new_sources:
                new_by_native.setdefault(native,[]).extend([p for a in selected for p in a['uv']])
    humans={int(s['source_pts']):s for s in manifest['samples']
        if s['source_hash']==preparation['packet_records'][0]['source_reference'].get('video_sha256',
            json.loads(Path(preparation['fit_dataset']['path']).read_text(encoding='utf8'))['source']['video_sha256'])}
    catalog=[(n,(humans[n]['annotations'] if n in humans else [])+a,[]) for n,a in sorted(by_native.items())]
    if current_review_priority:
        schedule,record=reviewed_source_crop_schedule(catalog,crop_size=config['crop_size'],native_hw=(1080,1920),
            optimizer_steps=config['training_steps'],reviewed_source_sha256s=new_sources)
    else:
        schedule,record=source_crop_schedule(catalog,crop_size=config['crop_size'],native_hw=(1080,1920),optimizer_steps=config['training_steps'])
    consumed=0;frames=0;size=config['crop_size']
    for native,points in new_by_native.items():
        points=np.asarray(points,float);active=np.zeros(len(points),bool)
        for x,y in schedule.get(native,[]):
            active|=(points[:,0]>=x)&(points[:,0]<x+size)&(points[:,1]>=y)&(points[:,1]<y+size)
        consumed+=int(active.sum());frames+=int(active.any())
    result=dict(schema='rinkspace.prepared-source-review-crop-coverage.v1',
        preparation_reference=dict(path=str(preparation_path.resolve()),sha256=file_sha(preparation_path)),
        configured_optimizer_steps=config['training_steps'],crop_schedule=record,
        newly_admitted_source_points=sum(map(len,new_by_native.values())),
        newly_admitted_points_inside_scheduled_crops=consumed,newly_admitted_native_frames_with_scheduled_crops=frames,
        scheduled_original_frames=len(schedule),optimizer_updates=0)
    if output_path.exists():raise FileExistsError('Preserve earlier crop schedule coverage audit')
    output_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--current-review-priority',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.config,a.preparation,a.output,a.current_review_priority)))
