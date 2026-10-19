"""Preserve the rejected shot-two release and prepare independently reviewed sources."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from calibration.program_tracking.prepare_first_shot_review import decoded
from calibration.program_tracking.prepare_continuation_frame_sources import timestamp,array_sha
from calibration.program_tracking.finalize_shot2_review import read,write,reference,sha

PARENT=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot2_review_20261014')
DATA=PARENT/'r2'
OLD_OUT=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot2_review_20261014'
OUT=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/shot2_r2_20261014'
OLD_MODEL=ROOT/'events/ankara_2026/models/shot2_8a28_20261014'
MODEL=ROOT/'events/ankara_2026/models/shot2_8a28_r2_20261014'
INDICES=[1249,1274,1314,1345,1394,1437,1469,1518,1545,1575,1624,1660,1707,1746,1783,1811]

def check_path(path):
    if len(str(path.resolve()).encode('utf-16-le'))//2>=260 or any(len(p.encode('utf-16-le'))//2>255 for p in path.parts):
        raise ValueError('OneDrive path too long: '+str(path))

def run():
    DATA.mkdir(parents=True,exist_ok=True)
    for p in (OUT/'batch_manifest.json',MODEL/'scene.json',MODEL/'ad2_material_template.json'):
        check_path(p)
    receipt=DATA/'baseline_receipt.json'
    if not receipt.exists():
        paths=list(OLD_OUT.rglob('*.json'))+list(OLD_MODEL.glob('*.json'))
        write(receipt,dict(reason='user rejected release; preserve actual source and manual edits',
            parent_dataset=reference(OLD_OUT/'batch_manifest.json'),files=[reference(p) for p in paths],
            original_files_changed=False))
    MODEL.mkdir(parents=True,exist_ok=True)
    for name in ('scene.json','feature_registry.json','ad2_material_template.json','seed_parameters.json','model.json'):
        if not (MODEL/name).exists():shutil.copy2(OLD_MODEL/name,MODEL/name)
    registry=read(MODEL/'feature_registry.json');registry['model_id']='ankara_2026.shot2_8a28_r2_20261014'
    write(MODEL/'feature_registry.json',registry)
    model=read(MODEL/'model.json');model.update(model_id=registry['model_id'],derived_model_role='second_shot_source_repair_awaiting_review',
        parent_rejected_source_model=reference(OLD_MODEL/'model.json'),user_accepted=False)
    for key,name in [('scene','scene.json'),('feature_registry','feature_registry.json'),('material_templates','ad2_material_template.json'),('seed_parameters','seed_parameters.json')]:
        model[key]=reference(MODEL/name)
    write(MODEL/'model.json',model)
    original=read(PARENT/'frame_manifest.json');oldrows={r['index']:r for r in original['frames']}
    mapping=read(PARENT/'navigation_mapping.json')
    opening=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/opening_100_50fps/navigation_mapping.json')
    ledger=[json.loads(line) for line in Path(opening['mapping_evidence']['path']).read_text(encoding='utf8').splitlines()]
    accepted={}
    for row in ledger:
        if row.get('shot_index')==2 and row.get('match_status')=='unique_pixel_match':
            accepted.setdefault(row['accepted_official_pts'],[]).append(row)
    archive=np.load(opening['model_seed']['inherited_camera_archive']['path'])
    lookup={int(p):i for i,p in enumerate(archive['pts'])}
    template=deepcopy(next(iter(oldrows.values())))
    source_path=Path(template['source']['source'])
    representation=Path(template['source']['ledger_official_representation']['path'])
    rows=[];navrows=[];ai=[];ptslist=[]
    for index in INDICES:
        pts=256256+(index-1)*256;ptslist.append(pts)
        if index in oldrows:
            row=deepcopy(oldrows[index])
        else:
            raw=next(decoded(source_path,pts,pts+256))[1]
            other=next(decoded(representation,pts,pts+256))[1]
            if not np.array_equal(raw,other):raise ValueError('Current and ledger original source differ')
            image=PARENT/'frames'/f'{index:06d}.png'
            if not image.exists():
                cv2.imwrite(str(image),raw,[cv2.IMWRITE_PNG_COMPRESSION,1])
            if not np.array_equal(raw,cv2.imread(str(image))):raise ValueError('PNG/source differ')
            row=deepcopy(template)
            source=deepcopy(template['source'])
            for key in ('mask_path','mask_sha256','navigation_overlay_path','mask_revision_provenance'):source.pop(key,None)
            source.update(programme_frame_index_1based=index,native_frame_index_1based=pts//256+1,
                native_frame_index_0based=pts//256,native_pts=pts,actual_pts=pts,source_time=timestamp(pts),source_time_seconds=pts/12800,
                image_path=str(image),png_sha256=sha(image),decoded_BGR_sha256=array_sha(raw),
                decoded_RGB_sha256=array_sha(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)))
            row.update(index=index,source=source,image_path=str(image),path=str(image),native_pts=pts,source_pts=pts,
                source_time=timestamp(pts),source_time_seconds=pts/12800,native_frame_index_0based=pts//256,
                png_sha256=source['png_sha256'],decoded_native_BGR_sha256=source['decoded_BGR_sha256'],
                decoded_native_RGB_sha256=source['decoded_RGB_sha256'],mask_pending=True)
            for key in ('mask_path','mask_sha256','current_person_mask_provenance','selection_metric'):row.pop(key,None)
            binding=DATA/'source_bindings'/f'{index:06d}.json';write(binding,source);row['source_binding']=reference(binding)
        matched=min(accepted[pts],key=lambda r:r['full_pts']);camera_index=lookup[matched['full_pts']];ai.append(camera_index)
        navrows.append(dict(programme_frame_index_1based=index,official_pts=pts,legacy60_pts=matched['full_pts'],
            mapping_status='confirmed_pixel_match',navigation_only=True,ledger_evidence=matched,
            inherited_camera_archive_row_0based=camera_index,camera25_float64_sha256=array_sha(archive['camera25'][camera_index])))
        row['navigation_path']=str(DATA/'navigation'/f'{index:06d}.png')
        row['source']['navigation_overlay_path']=row['navigation_path'];rows.append(row)
    ai=np.asarray(ai,np.int64)
    np.savez(DATA/'camera_seeds.npz',programme_indices=np.asarray(INDICES),pts=np.asarray(ptslist),
        camera25=archive['camera25'][ai],legacy60_pts=archive['pts'][ai],inherited_camera_archive_indices=ai)
    gaps=np.diff(ptslist)/12800
    if (gaps<.5).any() or (gaps>1).any():raise ValueError('Source cadence violation')
    mapping['frames']=navrows;mapping['model_seed']['scene']=reference(MODEL/'scene.json')
    write(DATA/'navigation_mapping.json',mapping)
    manifest=deepcopy(original);manifest.update(programme_indices=INDICES,frames=rows,frame_count=len(rows),
        repair_revision=2,parent_rejected_manifest=reference(PARENT/'frame_manifest.json'),
        source_images_reused_without_copy=True,navigation_mapping=reference(DATA/'navigation_mapping.json'))
    write(DATA/'frame_manifest.json',manifest)
    selection=read(PARENT/'selection.json');selection.update(selected_pts=ptslist,programme_indices=INDICES,
        selected_interval_seconds=gaps.tolist(),clear_evidence_gaps=[],
        revision_reason='current source recheck considers both visible ice and board materials, preserves0.5–1sec cadence',
        substitutions={1358:1345,1552:1545,1589:1575,1670:1660},parent_selection=reference(PARENT/'selection.json'))
    write(DATA/'selection.json',selection)
    print(json.dumps(dict(frames=len(rows),indices=INDICES,new_masks=[i for i in INDICES if i not in oldrows],data=str(DATA)),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
