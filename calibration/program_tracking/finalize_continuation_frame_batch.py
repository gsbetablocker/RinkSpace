"""Merge the authorized 101–200 native observations and check model entry.

The selected frame200 coordinates and approved opening100 remain unchanged.
Entry diagnostics use explicit unit pixel scales and never optimize or accept.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'src', ROOT/'calibration/program_tracking'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from .extract_opening_frame_batch import read, write, sha, MODEL
from .extract_selected_frame_review import make_binding
from .validate_opening_frame_batch import assemble, require, source_vertices, reference
from .framewise_model import evaluate_rows
from .render_opening_frame_batch import run as render

PARENT = ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
OUT = PARENT/'continuation_101_200_50fps'
DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/continuation_101_200_50fps')
REFERENCE = PARENT/'frame_000200_50fps'


from rinkspace.storage import AssetStore

def approved_integrity():
    old = PARENT/'opening_100_50fps'
    receipt = read(old/'user_acceptance.json')
    require(receipt['accepted'] and len(receipt['frames']) == 100, 'Opening100 approval receipt missing')
    for row in receipt['frames']:
        for folder,key in [('frames','packet_sha256'),('bindings','binding_sha256')]:
            require(sha(old/f'{folder}/{row["index"]:06d}.json') == row[key], 'Approved opening100 changed')
    return dict(status='passed',frame_count=100,point_count=receipt['point_count'],receipt=reference(old/'user_acceptance.json'))


def merge():
    integrity = approved_integrity()
    manifest = read(DATA/'frame_manifest.json')
    registry = read(MODEL/'feature_registry.json')
    canonical_path = REFERENCE/'frame_000200_points.json'
    canonical = read(canonical_path)
    require(canonical['point_count'] == 252 and not canonical['user_accepted'], 'Expected corrected frame200 candidate')
    require(len(manifest['frames']) == 100, 'Expected100 exact native frames')
    for record in manifest['frames']:
        index = record['index']; pts = 256256+256*(index-1)
        require(101 <= index <= 200 and record['source']['native_pts'] == pts, 'Unexpected source frame')
        saved_path = OUT/f'frames/{index:06d}.json'
        if saved_path.exists() and read(saved_path).get('manual_editor'):
            saved = read(saved_path)
            require(saved['source']['native_pts'] == pts, 'Manually edited frame clock changed')
            require(saved['source']['decoded_BGR_sha256'] == record['source']['decoded_BGR_sha256'], 'Manually edited frame source changed')
            # Explicit user edits are authoritative; extraction must never restore deleted points.
            write(OUT/f'bindings/{index:06d}.json',make_binding(saved,saved_path,registry))
            continue
        if index == 200:
            packet = deepcopy(canonical)
            packet['template_coordinates_exactly_preserved'] = True
            packet['canonical_selected_frame'] = reference(canonical_path)
            ice = read(REFERENCE/'ice_features.json')
        else:
            board_path = DATA/f'board_candidates/{index:06d}.json'
            ice_path = DATA/f'ice_candidates/{index:06d}.json'
            packet,ice = read(board_path),read(ice_path)
            require(packet['source']['native_pts'] == ice['source_pts'] == pts, 'Board/ice clock mismatch')
            require(packet['source']['mask_sha256'] == ice['source']['mask_sha256'] == record['source']['mask_sha256'], 'Extraction mask versions differ')
            require(not any(f['id'].startswith('ice.') for f in packet['features']), 'Board candidate contains ice')
            packet['features'] += ice['features']
            packet['inferred_segments'] = ice.get('inferred_segments',[])
            packet['source_correlated_groups'] = ice.get('source_correlated_groups',[])
            packet['unresolved_source_regions'] = packet.get('unresolved_source_regions',[]) + ice.get('unresolved_source_regions',[])
            packet['ice_source_readout_packet'] = reference(ice_path)
            if ice.get('user_review_deletions'):
                packet['latest_user_review_correction'] = reference(ice['user_review_deletions']['receipt_path'])
            packet['board_source_readout_packet'] = reference(board_path)
            packet['ice_physical_aliases'] = ice.get('physical_aliases', {'B4':'R4','B3':'R3'})
            packet['source_shape_reasoning'] = dict(circles_checked_as_projected_physical_circles=True,
                lines_checked_as_projected_physical_straight_lines=True,
                inferred_search_guides_are_not_source_observations=True,
                ice_sampling_policy=ice.get('source_sampling_policy'))
        source = deepcopy(record['source'])
        source.update(source_points_extracted=True,optimization_performed=False)
        if ice['source'].get('broadcast_timer_exclusion'):
            source['broadcast_timer_exclusion'] = deepcopy(ice['source']['broadcast_timer_exclusion'])
        source.pop('only_requested_frame_saved',None)
        source.pop('programme_frames_101_to_199_processed',None)
        require(source['mask_sha256'] == sha(Path(source['mask_path'])), 'Mask differs from source record')
        packet.update(source=source,frame_count=1,batch_sequence_index_1based=index,
            programme_frame_index_1based=index,user_accepted=False,
            status='awaiting_user_review_of_frames_101_to_200',user_hard_point_ceiling=500,
            optimization_performed=False,observation_coordinates_from_projection=False,
            sample_frame_reference=reference(canonical_path),
            user_correction=reference(REFERENCE/'user_correction_01.json'))
        for feature in packet['features']:
            for key in ('label_zh','note_zh'):
                if isinstance(feature.get(key),str):
                    feature[key]=feature[key].replace('B3','R3').replace('B4','R4')
        for region in packet.get('unresolved_source_regions',[]):
            for key in ('note_zh','note'):
                if isinstance(region.get(key),str):
                    region[key]=region[key].replace('B3','R3').replace('B4','R4')
        packet.pop('ice_physical_aliases',None)
        packet['ice_physical_names']={'R3':'R3','R4':'R4'}
        packet['user_naming_correction']=reference(OUT/'user_naming_correction.json')
        packet.pop('programme_frames_101_to_199_processed',None)
        packet.pop('next_frame_processed',None)
        packet['point_count'] = sum(len(s['uv']) for f in packet['features'] for s in f['segments'])
        require(packet['point_count'] <= 500, 'Point ceiling exceeded')
        path = OUT/f'frames/{index:06d}.json'
        write(path,packet)
        write(OUT/f'bindings/{index:06d}.json',make_binding(packet,path,registry))
    write(DATA/'accepted100_integrity_validation.json',integrity)
    return manifest


def validate():
    scene,registry,templates = [read(MODEL/name) for name in ('scene.json','feature_registry.json','ad2_material_template.json')]
    manifest = read(DATA/'frame_manifest.json')
    cameras = np.load(DATA/'camera_seeds.npz',allow_pickle=False)
    navigation = read(OUT/'navigation_mapping.json')
    require(len(navigation['frames'])==100, 'Navigation must cover100 exact frames')
    require(np.array_equal(cameras['programme_indices'],np.arange(101,201)), 'Camera index range mismatch')
    require(np.array_equal(cameras['pts'],np.arange(281856,307201,256)), 'Camera PTS mismatch')
    result = dict(schema='rinkspace.continuation-model-entry-validation.v1',status='failed',
        optimization_performed=False,model_accuracy_accepted=False,
        diagnostic_weighting='explicit_unit_pixel_scale_not_formal_measurement_weight',
        shared_line_covariance_preserved_in_packet_not_replaced_by_independent_observations=True,
        frames=[],errors=[])
    totals=Counter()
    for record,camera in zip(manifest['frames'],cameras['camera25']):
        index=record['index'];path=OUT/f'frames/{index:06d}.json';binding_path=OUT/f'bindings/{index:06d}.json'
        packet,binding=read(path),read(binding_path)
        try:
            source=packet['source'];pts=256256+256*(index-1)
            nav=navigation['frames'][index-101]
            require(nav['official_pts']==pts and nav['legacy60_pts']==int(cameras['legacy60_pts'][index-101]),'Navigation clock mismatch')
            require(nav['camera25_float64_sha256']==hashlib.sha256(camera.tobytes()).hexdigest(),'Camera seed bytes differ from navigation')
            require(nav['navigation_overlay']['current_mask_sha256']==source['mask_sha256'],'Navigation uses a stale mask')
            require(sha(nav['navigation_overlay']['path'])==nav['navigation_overlay']['sha256'],'Navigation overlay changed')
            require(source['native_pts']==source['actual_pts']==binding['source_pts']==pts, 'Incorrect native PTS')
            require(source['time_base']==binding['source_time_base']=='1/12800','Incorrect timebase')
            require(source['source_video_sha256']==manifest['source']['sha256'],'Source video mismatch')
            require(binding['source_packet']['sha256']==sha(path),'Binding packet hash mismatch')
            for key in ('board_source_readout_packet','ice_source_readout_packet'):
                if key in packet:
                    require(sha(packet[key]['path'])==packet[key]['sha256'],'Candidate changed after merge: '+key)
            raw=cv2.imread(source['image_path']);mask=cv2.imread(source['mask_path'],0)>0
            require(raw.shape==(1080,1920,3),'Source resolution changed')
            require(hashlib.sha256(raw.tobytes()).hexdigest()==source['decoded_BGR_sha256'],'Source pixels changed')
            require(sha(source['mask_path'])==source['mask_sha256'],'Source mask changed')
            features,vertices=source_vertices(packet);uv=np.asarray(list(vertices.values()))
            xy=np.rint(uv).astype(int)
            require(not mask[xy[:,1],xy[:,0]].any(),'A source point overlaps foreground')
            timer = source.get('broadcast_timer_exclusion',{})
            box = timer.get('actual_native_rectangle_xyxy_inclusive')
            if box:
                x0,y0,x1,y1=box;halo=timer.get('edge_halo_px',0)
                require(not ((xy[:,0]>=x0-halo)&(xy[:,0]<=x1+halo)&(xy[:,1]>=y0-halo)&(xy[:,1]<=y1+halo)).any(), 'A source point overlaps broadcast timer')
            require('board_top' in features and 'board_contact' in features,'Required physical board edges absent')
            for guide in packet.get('inferred_segments',[]):
                require(guide.get('inference_only') is True,'Unlabelled hidden inference')
            rows,targets,corners=assemble(packet,binding,registry,scene,templates)
            residual,jacobian=evaluate_rows(camera,rows,jacobian=True)
            n=packet['point_count']
            require(residual.shape==(2*n,) and jacobian.shape==(2*n,15),'Model entry dimensions wrong')
            require(np.isfinite(residual).all() and np.isfinite(jacobian).all(),'Nonfinite model entry')
            if index==200 and not packet.get('manual_editor'):
                _,expected=source_vertices(read(REFERENCE/'frame_000200_points.json'))
                require(list(vertices)==list(expected) and np.array_equal(uv,list(expected.values())),'Frame200 coordinates changed')
            totals.update(targets)
            result['frames'].append(dict(index=index,status='passed',point_count=n,packet=reference(path),
                model_binding=reference(binding_path),actual_residual_shape=list(residual.shape),
                actual_jacobian_shape=list(jacobian.shape),target_point_counts=dict(targets),
                source_coordinate_addresses_covered_exactly_once=True,corner_bindings=corners,
                foreground_point_count=0))
        except Exception as exc:
            result['errors'].append(dict(index=index,error=f'{type(exc).__name__}: {exc}'))
        if index%10==0: print(f'Model entries through frame {index}: {len(result["errors"])} errors',flush=True)
    result['approved100_integrity']=approved_integrity()
    result['aggregate_target_counts']=dict(totals)
    result['status']='passed' if len(result['frames'])==100 and not result['errors'] else 'failed'
    write(DATA/'model_entry_validation.json',result)
    require(result['status']=='passed',str(result['errors']))
    return result


def publish(validation):
    counts=[r['point_count'] for r in validation['frames']]
    frames=[]
    for item in validation['frames']:
        index=item['index'];p=read(OUT/f'frames/{index:06d}.json')
        frames.append(dict(index=index,native_pts=p['source']['native_pts'],source_time=p['source']['source_time'],
            point_count=p['point_count'],packet=item['packet'],model_binding=item['model_binding'],
            target_point_counts=item['target_point_counts'],user_accepted=False,
            image_path=p['source']['image_path'],overlay_path=str(DATA/f'overlays/{index:06d}.png')))
    manifest=dict(schema='rinkspace.framewise-continuation-batch.v1',event_id='ankara_2026',programme_id='okada_mei_sp',
        status='100_frames_extracted_awaiting_user_review',programme_frame_range=[101,200],frame_count=100,
        source_fps='50/1',source_time_base='1/12800',first_pts=281856,last_pts=307200,end_pts_exclusive=307456,
        source_interval='00:22.020–00:24.000',duration_seconds=2.,user_accepted=False,frame_201_processed=False,
        density=dict(minimum=min(counts),maximum=max(counts),mean=float(np.mean(counts)),total=sum(counts),user_hard_ceiling=500),
        source_clock_manifest=reference(DATA/'frame_manifest.json'),navigation_mapping=reference(OUT/'navigation_mapping.json'),
        sample_frame=reference(REFERENCE/'frame_000200_points.json'),user_correction=reference(REFERENCE/'user_correction_01.json'),
        user_authorization=reference(OUT/'user_authorization.json'),model_entry_validation=reference(DATA/'model_entry_validation.json'),
        accepted100_integrity=validation['approved100_integrity'],model_id=read(MODEL/'model.json')['model_id'],
        optimization_performed=False,model_accuracy_accepted=False,strict_v6_used=False,
        inferred_segments_not_point_observations=True,review_html=str(AssetStore(ROOT).review_page(OUT)),frames=frames)
    correction=OUT/'user_review_correction_02.json'
    if correction.exists():
        manifest['latest_user_review_correction']=reference(correction)
    write(OUT/'batch_manifest.json',manifest)
    render(SimpleNamespace(dataset=OUT,media=DATA,html=None,expected_frames=100,skip_overlays=False))
    make_contact_sheets()
    selection_path=PARENT.parents[1]/'primary_source_selection.json';selection=read(selection_path)
    selection['continuation'].update(source_frame_count_processed=200,
        batch_manifest=str(OUT/'batch_manifest.json'),last_pts=307200,
        last_contiguous_processed_programme_frame=200,last_contiguous_processed_pts=307200,
        frame_101_processed=True,frame_201_processed=False,
        selected_preview_frame_processed=200,selected_preview_user_accepted=False,
        next_step='await_user_review_of_frames_101_to_200')
    write(selection_path,selection)
    model=read(MODEL/'model.json')
    model['candidate_frame_datasets']=[r for r in model.get('candidate_frame_datasets',[]) if r.get('frame_index')!=200]
    batches=[r for r in model.get('candidate_batch_datasets',[]) if Path(r['path']).resolve()!=(OUT/'batch_manifest.json').resolve()]
    batches.append(dict(**reference(OUT/'batch_manifest.json'),frame_count=100,point_count=sum(counts),
        programme_frame_range=[101,200],status='awaiting_user_review',source_fps='50/1',optimization_performed=False))
    model.update(candidate_batch_datasets=batches,candidate_user_acceptance_pending=True,
        candidate_source_frame_count=sum(r['frame_count'] for r in batches+model['candidate_frame_datasets']),
        candidate_source_point_count=sum(r['point_count'] for r in batches+model['candidate_frame_datasets']))
    require(model['accepted_source_frame_count']==100 and model['accepted_source_point_count']==11134,'Approved totals changed')
    write(MODEL/'model.json',model)
    entry=AssetStore(ROOT).project_entry();text=entry.read_text(encoding='utf8')
    begin='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_BEGIN -->';end='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_END -->'
    relative=OUT.relative_to(ROOT).as_posix()
    block=f'''{begin}<section class="surface"><h3>冈田芽衣：50fps 第101–200帧取点审阅</h3>
<p>官方单节目源时间00:22.020–00:24.000，共100帧，每帧{min(counts)}–{max(counts)}点。第200帧已撤去圈内偏点、恢复R3外圈左弧、补全两条长边的可见范围，101–199帧依此样板逐帧提取。冰圈统一采用模型名称R3、R4。</p>
<p><a href="{AssetStore(ROOT).review_page(OUT).as_uri()}">逐帧查看原图、取点与8a28网格导航</a> · <a href="{relative}/batch_manifest.json">批次记录</a> · <a href="events/ankara_2026/models/framewise_8a28_20261013/model.json">对应模型入口</a></p>
<p class="small">101–200帧等待一起验收。已通过的1–100帧保持原样。停在第200帧，尚未求解。</p></section>{end}'''
    require(text.count(begin)==1 and text.count(end)==1,'Review entry not unique')
    entry.write_text(re.sub(re.escape(begin)+r'[\s\S]*?'+re.escape(end),lambda _:block,text,count=1),encoding='utf8')
    approved_integrity()
    return manifest


def make_contact_sheets():
    """Review all frames in order; native overlays remain the detail source."""
    directory=DATA/'contact_sheets';directory.mkdir(exist_ok=True)
    paths=[]
    for first in range(101,201,20):
        sheet=np.full((5*294,4*480,3),22,np.uint8)
        for offset,index in enumerate(range(first,first+20)):
            image=cv2.imread(str(DATA/f'overlays/{index:06d}.png'))
            packet=read(OUT/f'frames/{index:06d}.json')
            row,column=divmod(offset,4);x,y=column*480,row*294
            sheet[y+24:y+294,x:x+480]=cv2.resize(image,(480,270),interpolation=cv2.INTER_AREA)
            cv2.putText(sheet,f'{index} | {packet["source"]["source_time"]} | {packet["point_count"]} pts',
                (x+8,y+17),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,235),1,cv2.LINE_AA)
        path=directory/f'{first:06d}_{first+19:06d}.jpg'
        cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,94]);paths.append(str(path))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validate-only',action='store_true');p.add_argument('--merge-only',action='store_true')
    args=p.parse_args()
    if not args.validate_only: merge()
    if args.merge_only: return
    result=validate()
    if not args.validate_only:
        manifest=publish(result);print(json.dumps(manifest['density'],ensure_ascii=False),flush=True)


if __name__=='__main__': main()
