"""Publish the authorized 100-frame candidate for review, never user-approve it."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import cv2
import numpy as np
from .extract_opening_frame_batch import OUT, DATA, PARENT, MODEL, ROOT, read, write, sha

from rinkspace.storage import AssetStore

def finalize():
    frames=[];counts=[];groups=Counter();violations=[]
    for index in range(1,101):
        path=OUT/f'frames/{index:06d}.json';p=read(path)
        assert p['source']['native_pts']==256256+256*(index-1)
        assert p['source']['time_base']=='1/12800'
        actual_native_index=p['source']['native_pts']//256+1
        if p['source'].get('native_frame_index_1based')!=actual_native_index or 'programme_frame_index_1based'not in p['source']:
            p['source'].update(native_frame_index_1based=actual_native_index,programme_frame_index_1based=index)
            write(path,p)
            binding_path=OUT/f'bindings/{index:06d}.json';binding=read(binding_path)
            binding['source_packet']['sha256']=sha(path);write(binding_path,binding)
        mask_path=DATA/f'masks_current/{index:06d}.png'
        assert p['source']['mask_sha256']==sha(mask_path),'Mask changed since point extraction'
        mask=cv2.imread(str(mask_path),0)>0
        tally=Counter()
        for f in p['features']:
            group=f['id'].split('.fragment')[0]
            for s in f['segments']:
                uv=np.array(s['uv']);tally[group]+=len(uv)
                xy=np.rint(uv).astype(int)
                if index>1 and mask[xy[:,1],xy[:,0]].any():
                    violations.append(dict(index=index,feature=f['id'],vertices=np.flatnonzero(mask[xy[:,1],xy[:,0]]).tolist()))
        assert tally.total()==p['point_count']<=500
        counts.append(p['point_count']);groups.update(tally)
        frames.append(dict(index=index,native_pts=p['source']['native_pts'],source_time=p['source']['source_time'],
            point_count=p['point_count'],feature_counts=dict(tally),user_accepted=p['user_accepted'],
            packet=dict(path=str(path),sha256=sha(path)),
            model_binding=dict(path=str(OUT/f'bindings/{index:06d}.json'),sha256=sha(OUT/f'bindings/{index:06d}.json')),
            image_path=str(DATA/f'frames/{index:06d}.png'),overlay_path=str(DATA/f'overlays/{index:06d}.png')))
    if violations:raise ValueError('Foreground exclusion violation: '+str(violations))
    selection_path=PARENT.parents[1]/'primary_source_selection.json';selection=read(selection_path)
    selection_snapshot=DATA/'source_selection_at_decode.json'
    if not selection_snapshot.exists():
        selection_snapshot.write_bytes(selection_path.read_bytes())
    clock_path=DATA/'frame_manifest.json';clock=read(clock_path)
    assert sha(selection_snapshot)==clock['source_selection_reference']['sha256']
    selection['continuation']=dict(source_choice_recorded=True,frame_2_processed=True,point_extraction_continued=True,
        source_frame_count_processed=100,batch_manifest=str(OUT/'batch_manifest.json'),last_pts=281600,
        old_source_points_or_camera_timestamps_rewritten=False,frame_101_processed=False,
        next_step='await_user_review_of_authorized_opening_100_frames')
    write(selection_path,selection)
    manifest=dict(schema='rinkspace.framewise-opening-batch.v1',event_id='ankara_2026',programme_id='okada_mei_sp',
        status='100_frames_extracted_awaiting_user_review',source_fps='50/1',source_time_base='1/12800',
        first_pts=256256,last_pts=281600,end_pts_exclusive=281856,source_interval='00:20.020–00:22.000',
        duration_seconds=2.,frame_count=100,frame_101_processed=False,
        approved_reference=dict(path=str(PARENT/'frame_000001_points.json'),sha256=sha(PARENT/'frame_000001_points.json')),
        density=dict(reference_points=99,user_hard_ceiling=500,minimum=min(counts),maximum=max(counts),
            mean=float(np.mean(counts)),total=sum(counts),reason='similar_density_plus_newly_visible_K_and_A',feature_totals=dict(groups)),
        source_clock_manifest=dict(path=str(DATA/'frame_manifest.json'),sha256=sha(DATA/'frame_manifest.json')),
        source_selection_snapshot_at_decode=dict(path=str(selection_snapshot),sha256=sha(selection_snapshot)),
        navigation_mapping=dict(path=str(OUT/'navigation_mapping.json'),sha256=sha(OUT/'navigation_mapping.json')),
        model_id=read(MODEL/'model.json')['model_id'],strict_v6_used=False,optimization_performed=False,
        model_accuracy_accepted=False,review_html=str(AssetStore(ROOT).review_page(OUT)),
        foreground_point_exclusion_check=dict(frames=99,first_frame_historical_approval_preserved=True,violations=[]),
        temporal_method='same_shot_static_board_navigation_plus_current_native_colour_readouts',
        first_frame_coordinates_preserved=True,frames=frames)
    write(OUT/'batch_manifest.json',manifest)
    model=read(MODEL/'model.json')
    model.update(next_frame_processed=True,accepted_source_point_count=99,
                 candidate_source_frame_count=100,candidate_source_point_count=sum(counts),
                 candidate_user_acceptance_pending=True)
    model['candidate_batch_datasets']=[dict(path=str(OUT/'batch_manifest.json'),sha256=sha(OUT/'batch_manifest.json'),
        frame_count=100,status='awaiting_user_review',source_fps='50/1',optimization_performed=False)]
    write(MODEL/'model.json',model)
    entry=AssetStore(ROOT).project_entry();text=entry.read_text(encoding='utf8')
    begin='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_BEGIN -->';end='<!-- CURRENT_SINGLE_FRAME_REVIEW_20261013_END -->'
    block=f'''{begin}<section class="surface"><h3>冈田芽衣：50fps 开头100帧取点</h3>
<p>官方单节目原视频 00:20.020–00:22.000，共100帧。每帧 {min(counts)}–{max(counts)} 点，平均 {np.mean(counts):.1f} 点。第一帧历史验收点保持原样，其余等待本批验收。</p>
<p><a href="{AssetStore(ROOT).review_page(OUT).as_uri()}">逐帧查看原图、取点与8a28网格导航</a> · <a href="events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/opening_100_50fps/batch_manifest.json">批次记录</a> · <a href="events/ankara_2026/models/framewise_8a28_20261013/model.json">对应模型入口</a></p>
<p class="small">旧60fps映射只用于导航；当前坐标来自官方50fps原图。已停在第100帧，未求解、未处理第101帧。</p></section>{end}'''
    if text.count(begin)!=1 or text.count(end)!=1:raise ValueError('Expected existing review entry block')
    text=re.sub(re.escape(begin)+r'[\s\S]*?'+re.escape(end),lambda _:block,text,count=1)
    entry.write_text(text,encoding='utf8')
    print(json.dumps(manifest['density'],ensure_ascii=False))
    return manifest

if __name__=='__main__':finalize()
