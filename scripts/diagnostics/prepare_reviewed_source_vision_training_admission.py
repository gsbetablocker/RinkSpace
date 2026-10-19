"""Freeze an explicitly proposed exact source-point training role receipt."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import authenticate_source_reference,file_sha
from rinkspace.vision import configure,load_training_manifest


def prepare(config_path,dataset_path,proposal_path,output_path):
    proposal=json.loads(proposal_path.read_text(encoding='utf8'))
    if proposal.get('schema')!='rinkspace.reviewed-source-exact-vision-training-admission-proposal.v1':
        raise ValueError('Explicit exact current source training admission proposal required')
    source=proposal['source_pack_reference'];witnesses=proposal['source_review_witnesses']
    for ref in (source,*witnesses):authenticate_source_reference(ref)
    ids=proposal['original_pack_point_indices']
    if not ids or any(type(i) is not int for i in ids) or ids!=sorted(set(ids)):
        raise ValueError('Proposed exact original source-point indices changed')
    dataset_ref=dict(path=str(dataset_path.resolve()),sha256=file_sha(dataset_path))
    dataset=json.loads(dataset_path.read_text(encoding='utf8'))
    packets=[p for p in dataset['cuts'] if p['source_pack']['sha256']==source['sha256']
        and Path(p['source_pack']['path']).resolve()==Path(source['path']).resolve()]
    if len(packets)!=1:raise ValueError('Final dataset must contain exactly the reviewed source packet')
    packet=packets[0];authenticate_source_reference(packet['support_index'])
    with np.load(packet['support_index']['path'],allow_pickle=False) as index:
        if len(np.setdiff1d(ids,index['fit_point_indices'])):
            raise ValueError('Pending, missing support or HELD points cannot receive training admission')
    config,data=configure(config_path);_,plan=load_training_manifest(config,data)
    pack=StrictAbsoluteSourcePack(source['path'],expected_sha256=source['sha256'])
    if not all(0<=i<len(pack.values['uv']) for i in ids):raise ValueError('Proposed source pixels lie outside original packet')
    entries=[]
    for ri,row in enumerate(pack.rows):
        lo,hi=map(int,pack.values['row_offsets'][ri:ri+2]);a,b=np.searchsorted(ids,[lo,hi])
        points=ids[a:b]
        if not points:continue
        shots=[s for s in plan['shots'] if s['id']==row['shot_id'] and s.get('whole_shot') is True
            and s['split']=='train' and s['source_sha256']==pack.metadata['source']['video_sha256']
            and s['time_base']==pack.metadata['source']['time_base'] and s['start_pts']<=row['native_pts']<s['end_pts']]
        if len(shots)!=1 or row.get('vision_training_role') is not None or row['split']!='fit':
            raise ValueError('Current receipt must retain the existing whole-shot split and original None source role')
        entries.append(dict(pack_row_index=ri,pack_point_indices=points,
            **{k:row[k] for k in ('physical_material_id','feature_id','actual_pts','native_pts','decoded_BGR_sha256')},
            original_candidate_indices=pack.values['original_point_indices'][points].tolist(),existing_whole_train_shot=shots[0]))
    if sum(len(e['pack_point_indices']) for e in entries)!=len(ids):raise ValueError('Training admission lost exact source pixels')
    receipt=dict(schema='rinkspace.current-reviewed-exact-source-vision-training-admission.v1',
        source_pack_reference=source,source_review_witnesses=witnesses,fit_dataset_reference=dataset_ref,
        proposal_reference=dict(path=str(proposal_path.resolve()),sha256=file_sha(proposal_path)),entries=entries,
        points=len(ids),vision_training_role='train',source_pack_roles_unchanged=True,admission_scope=proposal['admission_scope'],
        source_labels_from_original_RGB_q50_and_measured_support=True,model_projection_used_as_labels=False,
        unknown_pending_used_as_positive_or_negative=False,other_unlisted_source_roles_promoted=False,
        original_source_UV_sigma_clock_and_review_bytes_unchanged=True)
    if output_path.exists():raise FileExistsError('Preserve earlier exact source training admission')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return dict(path=str(output_path.resolve()),sha256=file_sha(output_path),points=len(ids))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--fit-dataset',type=Path,required=True);p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.config,a.fit_dataset,a.proposal,a.output)))
