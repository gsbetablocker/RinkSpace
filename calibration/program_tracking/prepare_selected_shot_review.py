"""Prepare exact native selected-shot sources from a programme-owned request.

Ranking is an aid for choosing exposures. It does not prove static-material
ownership, feature identity or source readout accuracy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking import prepare_shot2_review as common
from calibration.program_tracking.prepare_first_shot_review import decoded, write
from calibration.program_tracking.prepare_continuation_frame_sources import array_sha, timestamp
from rinkspace.programme_source_stream import file_sha
import av
import cv2
import numpy as np


def ranked_selection(metrics, count, radius=6):
    """Constrain every adjacent selected pair to the required native cadence."""
    anchors = np.linspace(0, len(metrics)-1, count)
    windows = []
    for ordinal, anchor in enumerate(anchors):
        lo, hi = max(0, round(anchor)-radius), min(len(metrics), round(anchor)+radius+1)
        candidates = metrics[lo:hi]
        laps = np.log1p([r['laplacian_variance'] for r in candidates])
        ratio = np.array([r['high_frequency_to_gradient_ratio'] for r in candidates])
        score = (laps-laps.min())/max(float(np.ptp(laps)),1e-6)
        score += .8*(ratio-ratio.min())/max(float(np.ptp(ratio)),1e-6)
        score -= .07*np.abs(np.arange(lo,hi)-anchor)/radius
        windows.append(dict(selection_ordinal_1based=ordinal+1,
                            nominal_native_pts=metrics[round(anchor)]['native_pts'],
                            candidates=[dict(native_pts=r['native_pts'], metric_index=i,
                                             score=float(s), laplacian_variance=r['laplacian_variance'],
                                             high_frequency_to_gradient_ratio=r['high_frequency_to_gradient_ratio'])
                                        for i,r,s in zip(range(lo,hi),candidates,score)]))
    states = {r['metric_index']:(r['score'],[r['metric_index']]) for r in windows[0]['candidates']}
    for window in windows[1:]:
        updated = {}
        for r in window['candidates']:
            i = r['metric_index']
            possible = [(score+r['score'],history+[i]) for j,(score,history) in states.items()
                        if 25 <= i-j <= 50]
            if possible:
                updated[i] = max(possible, key=lambda x:x[0])
        states = updated
    if not states:
        raise ValueError('No clear-candidate path within required 0.5–1 second cadence')
    selected = max(states.values(),key=lambda x:x[0])[1]
    for window,i in zip(windows,selected):
        window['selected_native_pts'] = metrics[i]['native_pts']
    return selected,windows


def navigation_evidence(pts, shot_rows, accepted):
    """Choose an existing same-shot camera row without changing source selection.

    A missing unique legacy-camera correspondence is explicitly an identity
    navigation inference. It is never an accepted pixel match or a read point.
    """
    if pts in accepted:
        return min(accepted[pts], key=lambda r:r['full_pts']), dict(
            mapping_status='confirmed_pixel_match', pixel_correspondence_confirmed=True,
            accepted_neighbor_anchors=None, neighbor_interpolated_legacy60_pts_for_candidate_selection=None)
    anchors=sorted(accepted)
    left=[p for p in anchors if p<pts]
    right=[p for p in anchors if p>pts]
    if not left and not right:
        raise ValueError('This shot has no independent accepted navigation anchors')
    neighbors=[]
    for p in ([left[-1]] if left else [])+([right[0]] if right else []):
        r=min(accepted[p],key=lambda r:r['full_pts'])
        neighbors.append(dict(official_pts=p,legacy60_pts=r['full_pts']))
    if len(neighbors)==2:
        l,r=neighbors
        predicted=l['legacy60_pts']+(pts-l['official_pts'])/(r['official_pts']-l['official_pts'])*(r['legacy60_pts']-l['legacy60_pts'])
        bracket=[r for r in shot_rows if l['legacy60_pts']<=r['full_pts']<=neighbors[1]['legacy60_pts']]
        candidates=[r for r in bracket if r.get('best_official_pts_candidate')==pts]
        status='inferred_navigation_from_unaccepted_pixel_candidate_with_accepted_neighbors'
        if not candidates:
            candidates=bracket
            status='inferred_navigation_from_accepted_neighbor_bracket_missing_exact_pixel_candidate'
    else:
        # At a shot boundary there may be no second accepted anchor. Keep the
        # current exposure and flag the reused closest same-shot identity view.
        predicted=float(neighbors[0]['legacy60_pts'])
        candidates=[r for r in shot_rows if r.get('best_official_pts_candidate')==pts]
        status='inferred_navigation_from_unaccepted_pixel_candidate_with_one_sided_accepted_anchor'
        if not candidates:
            candidates=accepted[neighbors[0]['official_pts']]
            status='inferred_navigation_from_closest_same_shot_accepted_identity_view'
    if not candidates:
        raise ValueError('No existing same-shot camera correspondence candidate')
    selected=min(candidates,key=lambda r:(abs(r['full_pts']-predicted),r['full_pts']))
    return selected,dict(mapping_status=status,pixel_correspondence_confirmed=False,
                         accepted_neighbor_anchors=neighbors,
                         neighbor_interpolated_legacy60_pts_for_candidate_selection=float(predicted))


def prepare(request_path, model_path):
    started=time.perf_counter()
    request=common.read(request_path)
    data=Path(request['data_directory'])
    if (data/'frame_manifest.json').exists():
        raise FileExistsError('Prepared source manifest already exists; preserve existing work')
    shot=int(request['shot_index'])
    start,last,stop=(int(request[k]) for k in ('first_pts','last_pts','stop_pts_exclusive'))
    full_physical_scope=deepcopy(request.get('full_physical_shot_scope'))
    excluded_tail=deepcopy(request.get('explicit_user_excluded_tail'))
    if full_physical_scope:
        if (full_physical_scope['time_base']!=request['time_base'] or
                not int(full_physical_scope['first_pts'])<=start<=last<stop<=int(full_physical_scope['stop_pts_exclusive'])):
            raise ValueError('Declared processing interval is outside the actual native physical shot')
        full_physical_scope['native_frame_count']=(int(full_physical_scope['stop_pts_exclusive'])-int(full_physical_scope['first_pts']))//int(request['native_pts_step'])
    trimmed_scope=bool(full_physical_scope and (start!=int(full_physical_scope['first_pts']) or stop!=int(full_physical_scope['stop_pts_exclusive'])))
    first=int(request['programme_first_pts'])
    count=int(request['selected_frame_count'])
    opening=common.read(Path(request['navigation_reference']))
    representation=common.read(Path(request['representation_binding']))['ledger_official_representation']
    source_record=common.read(Path(request['primary_source_selection']))['primary_source']
    source=Path(source_record['path'])
    model=common.read(model_path)
    ledger_ref=opening['mapping_evidence']
    archive_ref=opening['model_seed']['inherited_camera_archive']
    for record in (source_record,ledger_ref,archive_ref,representation,model['scene']):
        if file_sha(record['path']) != record['sha256']:
            raise ValueError('Authenticated input changed: '+record['path'])
    ledger=[dict(json.loads(line),line_1based=i+1) for i,line in enumerate(Path(ledger_ref['path']).read_text('utf8').splitlines())]
    shot_rows=[r for r in ledger if r['shot_index']==shot]
    accepted={}
    for row in shot_rows:
        if row['match_status']=='unique_pixel_match' and row.get('accepted_official_pts') is not None:
            accepted.setdefault(int(row['accepted_official_pts']),[]).append(row)
    metrics=[]
    previous=None
    for pts,bgr in decoded(source,start,stop):
        values,previous=common.measure(bgr,previous)
        values.update(native_pts=pts,source_time=timestamp(pts),
                      programme_frame_index_1based=(pts-first)//256+1,
                      exact_camera_pixel_match_available=pts in accepted)
        metrics.append(values)
    if [r['native_pts'] for r in metrics] != list(range(start,stop,256)):
        raise ValueError('Native source does not contain the declared complete shot range')
    selected,windows=ranked_selection(metrics,count)
    quality_review_ref=request.get('source_selection_quality_review')
    preselected=request.get('preselected_native_pts')
    if preselected is not None:
        if not quality_review_ref or file_sha(quality_review_ref['path'])!=quality_review_ref['sha256']:
            raise ValueError('Source selection override needs its unchanged actual native quality review')
        review=common.read(Path(quality_review_ref['path']))
        pts_override=[int(p) for p in preselected]
        if (len(pts_override)!=count or pts_override!=sorted(set(pts_override)) or
                any(p not in range(start,stop,256) for p in pts_override) or
                any(not 6400<=b-a<=12800 for a,b in zip(pts_override,pts_override[1:]))):
            raise ValueError('Preselected native exposures violate scope, unique count or 0.5–1 second cadence')
        if (review.get('shot_index')!=shot or review.get('first_pts')!=start or
                review.get('stop_pts_exclusive')!=stop or review.get('time_base')!='1/12800' or
                review.get('selected_native_pts')!=pts_override or
                review.get('primary_source',{}).get('sha256')!=source_record['sha256'] or
                review.get('all_selected_raw_native_visually_reviewed') is not True or
                review.get('observations_created') is not False):
            raise ValueError('Native quality review does not authenticate this source selection')
        reviewed_rows={int(r['native_pts']):r for r in review['selected_frames']}
        for pts in pts_override:
            row=reviewed_rows[pts]
            if file_sha(row['image_path'])!=row['png_sha256']:
                raise ValueError('Reviewed native source PNG changed before selected-shot preparation')
        ranked_pts=[metrics[i]['native_pts'] for i in selected]
        lookup={r['native_pts']:i for i,r in enumerate(metrics)}
        selected=[lookup[p] for p in pts_override]
        for window,pts,previous_pts in zip(windows,pts_override,ranked_pts):
            window['ranking_selected_native_pts']=previous_pts
            window['selected_native_pts']=pts
            window['selection_origin']='actual_native_static_material_quality_review'
            window['outside_initial_ranking_window']=pts not in [r['native_pts']for r in window['candidates']]
    pts_list=[metrics[i]['native_pts'] for i in selected]
    for folder in ('frames','source_bindings','navigation'):
        (data/folder).mkdir(parents=True,exist_ok=True)
    rows=[]
    for pts,bgr in decoded(source,start,stop):
        if pts not in pts_list:
            continue
        index=(pts-first)//256+1
        image=data/'frames'/f'{index:06d}.png'
        if not cv2.imwrite(str(image),bgr,[cv2.IMWRITE_PNG_COMPRESSION,1]):
            raise OSError(image)
        if not np.array_equal(bgr,cv2.imread(str(image))):
            raise ValueError('Native PNG roundtrip changed pixels')
        binding=dict(schema='rinkspace.exact-single-native-source-frame.v1',
                     event_id=request['event_id'],programme_id=request['programme_id'],shot_index=shot,shot_id=shot,
                     programme_frame_index_1based=index,native_frame_index_1based=pts//256+1,
                     native_frame_index_0based=pts//256,native_pts=pts,actual_pts=pts,time_base='1/12800',
                     source_time=timestamp(pts),source_time_seconds=pts/12800,source=str(source),
                     source_video_sha256=source_record['sha256'],image_path=str(image),png_sha256=file_sha(image),
                     decoded_BGR_sha256=array_sha(bgr),decoded_RGB_sha256=array_sha(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),
                     native_hw=[1080,1920],original_PTS_verified=True,PNG_roundtrip_lossless_verified=True,
                     optimization_performed=False,ledger_official_representation=representation)
        rows.append(dict(index=index,shot_id=shot,source=binding,image_path=str(image),path=str(image),
                         native_pts=pts,source_pts=pts,source_time_base='1/12800',source_time=timestamp(pts),
                         source_time_seconds=pts/12800,native_frame_index_0based=pts//256,
                         png_sha256=binding['png_sha256'],decoded_native_BGR_sha256=binding['decoded_BGR_sha256'],
                         decoded_native_RGB_sha256=binding['decoded_RGB_sha256'],native_hw=[1080,1920],
                         exact_native_PTS_and_lossless_PNG_verified=True,selection_metric=metrics[(pts-start)//256]))
    by_pts={r['native_pts']:r for r in rows}
    verified=[]
    for pts,bgr in decoded(Path(representation['path']),start,stop):
        if pts not in by_pts:
            continue
        row=by_pts[pts]
        if array_sha(bgr)!=row['decoded_native_BGR_sha256']:
            raise ValueError('Current official versus ledger official source differs')
        row['ledger_official_representation_same_frame_BGR_identical']=True
        row['source']['ledger_official_representation_same_frame_BGR_identical']=True
        path=data/'source_bindings'/f"{row['index']:06d}.json"
        write(path,row['source'])
        row['source_binding']=dict(path=str(path),sha256=file_sha(path))
        verified.append(dict(native_pts=pts,decoded_BGR_sha256=array_sha(bgr),decoded_pixels_identical=True))
    if len(verified)!=count:
        raise ValueError('Incomplete selected source verification')
    with np.load(archive_ref['path'],allow_pickle=False) as saved:
        cameras={k:saved[k].copy() for k in saved.files}
    lookup={int(p):i for i,p in enumerate(cameras['pts'])}
    map_rows=[]
    for ordinal,row in enumerate(rows):
        evidence,mapping_status=navigation_evidence(row['native_pts'],shot_rows,accepted)
        ai=lookup[int(evidence['full_pts'])]
        map_rows.append(dict(sequence_index_1based=row['index'],programme_frame_index_1based=row['index'],
                             official_pts=row['native_pts'],official_time_base='1/12800',
                             official_time_seconds=row['native_pts']/12800,legacy60_pts=evidence['full_pts'],
                             legacy60_time_base='1/15360',legacy60_shot_index=shot,
                             **mapping_status,navigation_only=True,ledger_evidence=evidence,
                             camera_seed_index_0based=ordinal,inherited_camera_archive_row_0based=ai,
                             primary_official50_vs_ledger_official50_BGR_identical=True,
                             camera25_float64_sha256=array_sha(cameras['camera25'][ai])))
    ai=np.array([r['inherited_camera_archive_row_0based'] for r in map_rows],np.int64)
    np.savez(data/'camera_seeds.npz',programme_indices=np.array([r['index'] for r in rows],np.int64),
             pts=np.array(pts_list,np.int64),legacy60_pts=np.array([r['legacy60_pts'] for r in map_rows],np.int64),
             inherited_camera_archive_indices=ai,camera25=cameras['camera25'][ai],
             installation_id=cameras['installation_id'][ai],view_state4=cameras['view_state4'][ai],
             exposure_representative_pts=cameras['exposure_representative_pts'][ai])
    from calibration.program_tracking.render_result import SceneRenderer
    renderer=SceneRenderer(common.read(model['scene']['path']),draw_grid=True,draw_advertisements=True)
    for row,mapped,camera in zip(rows,map_rows,cameras['camera25'][ai]):
        raw=cv2.imread(row['image_path'])
        overlay,_=renderer.draw(raw,camera,person_mask=np.zeros(raw.shape[:2],bool),title=None)
        path=data/'navigation'/f"{row['index']:06d}.png"
        cv2.imwrite(str(path),overlay,[cv2.IMWRITE_PNG_COMPRESSION,1])
        row['navigation_path']=str(path)
        mapped['navigation_overlay']=dict(path=str(path),sha256=file_sha(path),navigation_only=True,
                                          current_source_background=True,current_person_mask_applied=False,
                                          camera_seed_refitted=False,all_original_feature_paths_and_advertisement_ids_enabled=True,
                                          body_or_trajectory_drawn=False,path_counts=renderer.path_counts)
    boundary=dict(shot_index=shot,first_pts=start,last_pts=last,stop_pts_exclusive=stop,time_base='1/12800',
                  first_source_time=timestamp(start),last_source_time=timestamp(last),native_frame_count=len(metrics),
                  first_programme_index=(start-first)//256+1,last_programme_index=(last-first)//256+1,
                  before_cut_pts=start-256,
                  next_shot_pts=int(full_physical_scope['stop_pts_exclusive']) if full_physical_scope else stop,
                  boundary_status=request['boundary_status'],
                  scope_type='user_selected_processing_scope' if trimmed_scope else 'full_physical_shot',
                  processing_stop_is_actual_camera_cut=not bool(full_physical_scope) or stop==int(full_physical_scope['stop_pts_exclusive']))
    processing_scope=deepcopy(boundary)
    if full_physical_scope:
        boundary['full_physical_shot_scope']=deepcopy(full_physical_scope)
        boundary['explicit_user_excluded_tail']=deepcopy(excluded_tail)
    selection=dict(schema='rinkspace.native-shot-clear-exposure-selection.v1',shot_scope=boundary,
                   requested_interval_seconds=[.5,1.],selected_frame_count=count,selected_pts=pts_list,
                   programme_indices=[r['index'] for r in rows],selected_interval_seconds=(np.diff(pts_list)/12800).tolist(),
                   uncovered_start_seconds=(pts_list[0]-start)/12800,uncovered_end_seconds=(last-pts_list[-1])/12800,
                   method=('Actual native static-material quality review after local ranking; 0.5–1 second cadence enforced'
                           if preselected is not None else 'local source board-material ranking constrained to 0.5–1 second cadence; source visual review required'),
                   selection_metrics_are_pixel_visibility_or_accuracy_proof=False,
                   all_native_shot_exposures_measured=len(metrics),metrics=metrics,windows=windows,
                   source_points_extracted=False,optimizer_run=False)
    if quality_review_ref:
        selection['source_selection_quality_review']=deepcopy(quality_review_ref)
    selection['processing_scope']=deepcopy(processing_scope)
    if full_physical_scope:
        selection['full_physical_shot_scope']=deepcopy(full_physical_scope)
        selection['explicit_user_excluded_tail']=deepcopy(excluded_tail)
    write(data/'selection.json',selection)
    verification=dict(schema='rinkspace.exact-current-vs-ledger-official-representation-verification.v1',
                      scope=f'Selected official50 shot{shot} source exposures',selected_frame_count=count,
                      current_primary_source=source_record,ledger_official_representation=representation,
                      all_selected_decoded_BGR_identical=True,whole_shot_pixel_equality_claimed=False,records=verified)
    write(data/'official_representation_verification.json',verification)
    mapping=dict(schema='rinkspace.official50-shot-navigation-mapping.v1',event_id=request['event_id'],
                 programme_id=request['programme_id'],purpose=f'shot{shot} selected sources; 8a28 identity navigation only',
                 official_source=source_record,mapping_evidence=ledger_ref,
                 model_reference=dict(path=str(model_path),sha256=file_sha(model_path)),
                 model_seed=deepcopy(opening['model_seed']),frames=map_rows,
                 official_representation_verification=dict(path=str(data/'official_representation_verification.json'),sha256=file_sha(data/'official_representation_verification.json')),
                 summary={s:sum(r['mapping_status']==s for r in map_rows) for s in sorted({r['mapping_status'] for r in map_rows})},
                 source_clock_substituted=False,official50_camera_fit_performed=False,optimizer_run=False)
    mapping['model_seed'].update(scene=model['scene'],camera_seeds=dict(path=str(data/'camera_seeds.npz'),sha256=file_sha(data/'camera_seeds.npz')),camera25_shape=[count,25])
    mapping['processing_scope']=deepcopy(processing_scope)
    if full_physical_scope:
        mapping['full_physical_shot_scope']=deepcopy(full_physical_scope)
        mapping['explicit_user_excluded_tail']=deepcopy(excluded_tail)
    write(data/'navigation_mapping.json',mapping)
    manifest=dict(schema='rinkspace.exact-native-selected-source-frames.v1',event_id=request['event_id'],
                  programme_id=request['programme_id'],source=source_record,shot_scope=boundary,
                  programme_indices=[r['index'] for r in rows],start_pts=pts_list[0],last_pts=pts_list[-1],
                  stop_pts_exclusive=pts_list[-1]+256,native_pts_step=256,frame_count=count,
                  original_PTS_order_verified=True,native_exposures_all_decoded=True,selected_frames_only_saved=True,
                  PNG_roundtrip_lossless_verified=True,source_points_extracted=False,optimization_performed=False,
                  navigation_mapping=dict(path=str(data/'navigation_mapping.json'),sha256=file_sha(data/'navigation_mapping.json')),
                  selection=dict(path=str(data/'selection.json'),sha256=file_sha(data/'selection.json')),
                  decoder=dict(library='PyAV',version=av.__version__,cpu_threads=2,temporal_resampling=False,spatial_resizing_of_source_PNGs=False),
                  frames=rows,source_prepare_elapsed_seconds=time.perf_counter()-started)
    manifest['processing_scope']=deepcopy(processing_scope)
    if full_physical_scope:
        manifest['full_physical_shot_scope']=deepcopy(full_physical_scope)
        manifest['explicit_user_excluded_tail']=deepcopy(excluded_tail)
    write(data/'frame_manifest.json',manifest)
    common.contact_sheet(rows,'image_path',data/'source_contact_sheet.png')
    common.contact_sheet(rows,'navigation_path',data/'navigation_contact_sheet.png')
    print(json.dumps(dict(shot=shot,indices=manifest['programme_indices'],pts=pts_list,intervals=selection['selected_interval_seconds'],data=str(data))),flush=True)
    return manifest


def refresh_navigation(data, model_path):
    """Reuse the common renderer with this shot's model and current masks."""
    data=Path(data)
    model_path=Path(model_path)
    model=common.read(model_path)
    if file_sha(model['scene']['path'])!=model['scene']['sha256']:
        raise ValueError('Current model scene reference is stale')
    mapping=common.read(data/'navigation_mapping.json')
    mapping['model_reference']=dict(path=str(model_path),sha256=file_sha(model_path))
    mapping['model_seed']['scene']=deepcopy(model['scene'])
    write(data/'navigation_mapping.json',mapping)
    previous_data=common.DATA
    try:
        common.DATA=data
        common.refresh_navigation()
    finally:
        common.DATA=previous_data
    manifest=common.read(data/'frame_manifest.json')
    manifest['selected_frames_only_saved']=True
    manifest['historical_source_caches_retained']=True
    write(data/'frame_manifest.json',manifest)
    for row in manifest['frames']:
        write(data/'navigation'/f"{row['index']:06d}.json",
              dict(scene_sha256=model['scene']['sha256'],source_pts=row['native_pts'],
                   current_mask_sha256=row['mask_sha256'],navigation_only=True))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--refresh-navigation',action='store_true')
    args=parser.parse_args()
    if args.refresh_navigation:
        refresh_navigation(common.read(args.request)['data_directory'],args.model)
    else:
        prepare(args.request,args.model)
