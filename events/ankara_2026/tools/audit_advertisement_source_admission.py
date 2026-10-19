"""Compare preserved native advertisement evidence with active source packets."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def numbers(value):return [int(n) for n in re.findall(r'(?:advertisement|ad)(\d+)',str(value),re.I)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    manifest_path=DATA/'calibration/runs/solver_acceleration_20261003/okada_lane23_Ad7_fast_complete_20261007/input_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    catalog_path=ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'
    catalog=json.loads(catalog_path.read_text(encoding='utf8'))
    scene_path=DATA/'calibration/runs/solver_acceleration_20261003/okada_lane23_Ad7_fast_complete_20261007/accepted_states/478afaf776150c398c48e7492da911e1219036cfad3295629a3df309b7d1089c/scene_UNACCEPTED.json'
    scene=json.loads(scene_path.read_text(encoding='utf8'))
    rendered=json.loads((DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/render.json').read_text(encoding='utf8'))['advertisement_ids']
    inventory={i:dict(number=i,catalog_label=next((s['canonical_label']for s in catalog['same_content_pairs']if i in s['numbers']),None),
                      shared_model_rendered=i in rendered,active_programme_source_counts={},source_candidates=[])for i in range(1,27)}
    points=collections.defaultdict(collections.Counter);groups=collections.defaultdict(collections.Counter);stamps=collections.defaultdict(set)
    original=DATA/'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
    for record in pickle.loads(original.read_bytes())['programme_absolute_rows']:
        row=record['source_row'];ids=row.get('physical_ids',[]);uv=row.get('uv',[])
        if len(ids)==len(uv) and len(ids):
            local=collections.Counter(n for item in ids for n in numbers(item))
        else:local=collections.Counter({n:len(uv)for n in set(numbers(row.get('id','')))})
        for n,count in local.items():
            if n in inventory:
                role=record['collection_role'];points[n]['original_'+role]+=count;groups[n]['original_'+role]+=1;stamps[n].add(int(record['group_actual_pts']))
    packet_receipts=[]
    for name in ('additional_real_native_board_edges','additional_real_native_board_material_points','additional_real_native_board_circles'):
        path=Path(manifest[name]['packet_path']);d=json.loads(path.read_text(encoding='utf8'));packet_receipts.append(dict(kind=name,path=str(path),sha256=sha(path)))
        for collection in ('rows','evaluation_rows','holdout_rows'):
            for row in d.get(collection,[]):
                curve=row.get('curve',{});n=curve.get('advertisement_number',curve.get('advertisement_id'))
                if n not in inventory:continue
                role=row.get('split','FIT' if collection=='rows' else 'HELD').upper()
                key=name+'_'+role;points[n][key]+=len(row.get('uv',[]));groups[n][key]+=1;stamps[n].add(int(row.get('actual_pts',row.get('pts'))))
    for n,item in inventory.items():
        item['active_programme_source_counts']=dict(points[n]);item['active_source_group_counts']=dict(groups[n]);item['active_source_pts']=sorted(stamps[n])
    source_files=[DATA/'observations/vision_training/source_ads_complete_frame.json',DATA/'observations/vision_training/missing_ads_native_contour_candidates.json']
    sources=[]
    for path in source_files:
        d=json.loads(path.read_text(encoding='utf8'));sources.append(dict(path=str(path),sha256=sha(path)))
        rows=d.get('sources',[])
        if not rows:
            binding=d['source'];rows=[dict(ad,source_pts=binding['source_pts'],source_video_sha256=binding['source_sha256'],actual_exposure_pts=binding.get('actual_exposure_pts'),
                source_specific_identity_authority=ad.get('physical_identity_authority'),source_instance_identity_human_confirmed=ad.get('physical_identity_human_confirmed'),
                contours=[c for component in ad['components']for c in component.get('contours',[])])for ad in d.get('advertisements',[])]
        for index,row in enumerate(rows):
            n=row.get('advertisement_number',row.get('requested_advertisement_number'));pts=int(row['source_pts']);contours=row.get('contours',[])
            if n not in inventory:continue
            candidate=dict(file=str(path),source_record_index=index,source_pts=pts,actual_exposure_pts=row.get('actual_exposure_pts'),
                source_video_sha256=row.get('source_video_sha256'),programme_interval=229114112<=pts<231705600 and row.get('source_video_sha256')=='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
                physical_instance_candidates=row.get('physical_instance_candidates',[n]),
                identity_authority=row.get('source_specific_identity_authority'),identity_human_confirmed=row.get('source_instance_identity_human_confirmed',False),
                full_print_extent_verified=row.get('full_print_extent_verified',False),
                source_navigation_roi=row.get('navigation_roi_native_xyxy'),contour_count=len(contours),raw_contour_points=sum(len(c.get('uv',[]))for c in contours),
                solver_admitted=row.get('solver_admitted',d.get('solver_admitted',False)),
                contours_have_typed_physical_outer_role=bool(contours) and all(c.get('physical_role') for c in contours),
                direct_ready_FIT_rows=False,
                admission_gaps=['Preserved contours mix internal ink, isophotes and external print strokes; no typed physical edge/point correspondence.',
                                'No active world primitive for this missing advertisement; source ROI/text bounding box is not its geometry.']if n not in rendered else ['Source frame not bound to this active programme/camera, or already represented by a differently scoped source packet.'])
            if candidate['programme_interval'] and len(candidate['physical_instance_candidates'])>1 and not candidate['identity_human_confirmed']:
                candidate['admission_gaps'].append('Candidate repeated print requires source-specific physical installation assignment from original visible order; NCC content alone is insufficient.')
            inventory[n]['source_candidates'].append(candidate)
    report=dict(schema='rinkspace.all26-advertisement-source-admission-inventory.v1',
        active_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)),active_additional_source_packets=packet_receipts,
        original_programme_absolute_packet=dict(path=str(original),sha256=sha(original)),user_identity_catalog=dict(path=str(catalog_path),sha256=sha(catalog_path)),
        source_candidate_packets=sources,all_user_ids_accounted=list(range(1,27)),actual_drawn_model_ids=rendered,
        missing_model_and_active_source_ids=[n for n in inventory if n not in rendered],
        direct_ready_additional_FIT_rows=[],candidate_contours_promoted_to_external_borders=False,
        geometry_or_active_target_modified=False,inventory=list(inventory.values()),
        parallel_owner=dict(Ad24_Ad25_Ad26='visual_correction multitime native source qualification'),accuracy_accepted=False)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(path=str(a.output),missing=report['missing_model_and_active_source_ids'],ready_rows=0),ensure_ascii=False))

if __name__=='__main__':main()
