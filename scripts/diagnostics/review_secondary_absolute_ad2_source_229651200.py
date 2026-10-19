"""Extract original RGB and annotate only four immutable source pixels."""
from pathlib import Path
import sys
import os
import json
import pickle
import hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'src')]
import numpy as np
import cv2
import av
from calibration.solver_runtime import install
install()
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import CorrectedSharedAdvertisementPairsNumericGeometry

BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_full9_review82_speed_candidate_20261010_101450_850')
PACKET=RUN.parent/'shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
OUT=BASE/'secondary_229651200_absolute_source_RGB_review'
VIDEO=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def main():
    OUT.mkdir(exist_ok=True)
    packet=pickle.load(PACKET.open('rb'))
    indexed=[(i,r) for i,r in enumerate(packet['programme_absolute_rows']) if r['block']=='secondary' and r['group_actual_pts']==229651200]
    selected=[(i,r) for i,r in indexed if r['row_order'] in (2,3,4,5)]
    assert len(selected)==4 and all(r['collection_role']=='FIT' for _,r in selected)
    origin_file=Path(selected[0][1]['source_row']['origin'].split('#')[0])
    origin=json.loads(origin_file.read_text(encoding='utf8'))
    upstream=origin['observations']['229651200']['ads']
    wanted={229650432,229650688,229650944,229651200,229651456}
    raws={}
    with av.open(str(VIDEO)) as container:
        stream=container.streams.video[0];stream.thread_type='NONE';stream.codec_context.thread_count=1
        assert str(stream.time_base)=='1/15360'
        container.seek(min(wanted),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts<min(wanted):continue
            if frame.pts>max(wanted):break
            if frame.pts in wanted:
                image=frame.to_ndarray(format='bgr24')
                assert image.shape==(1080,1920,3)
                raws[int(frame.pts)]=image
                cv2.imwrite(str(OUT/f'raw_native_{frame.pts}.png'),image)
    assert set(raws)==wanted
    overlay=raws[229651200].copy()
    rows=[]
    binder=CorrectedSharedAdvertisementPairsNumericGeometry.__new__(CorrectedSharedAdvertisementPairsNumericGeometry)
    colors=[(30,240,240),(30,255,80),(255,160,30),(200,60,255)]
    for number,((index,r),item,color) in enumerate(zip(selected,upstream,colors),1):
        source=r['source_row']; current=r['current_existing_geometry_row']
        xy=np.asarray(source['uv'])[0]
        np.testing.assert_array_equal(xy,np.asarray(item['uv']))
        ids=binder.identities(current)
        assert ids==[(2,item['role'])]
        x,y=np.rint(xy).astype(int)
        cv2.drawMarker(overlay,(x,y),color,cv2.MARKER_CROSS,20,2)
        cv2.circle(overlay,(x,y),8,color,1)
        label=f'{number} Ad2:{item["role"]}'
        dx=14 if x<960 else -270
        cv2.putText(overlay,label,(x+dx,y-14),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(overlay,label,(x+dx,y-14),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1,cv2.LINE_AA)
        rows.append(dict(number=number,source_pts=229651200,source_time='04:09:11.250',
            raw_frame_path=str(OUT/'raw_native_229651200.png'),packet_path=str(PACKET),packet_row_index=index,
            packet_row_reference=f'programme_absolute_rows/{index}/source_row',
            block='secondary',group_order=r['group_order'],row_order=r['row_order'],
            collection_role=r['collection_role'],original_source_row=plain(source),
            immutable_upstream_origin_reference=source['origin'],upstream_original_record=plain(item),
            current_existing_geometry_row=plain(current),current_original_identity_parser_result=plain(ids),
            user_confirmed_topology_label_is_not_confirmation_of_this_pixel_or_identity=True,
            current_30_confirmed_authority_contains_this_absolute_row=False,
            absence_from_current_30_table_is_not_absence_of_user_confirmation=True,
            active_FIT_no_retirement_record_for_this_locator=True))
    cv2.imwrite(str(OUT/'source_4_original_UV_labels_229651200.png'),overlay)
    # Larger corner crops preserve exact raw pixels for close inspection.
    corner_panels=[]
    for n,(_,r) in enumerate(selected,1):
        x,y=np.rint(np.asarray(r['source_row']['uv'])[0]).astype(int)
        x0=max(0,x-130);y0=max(0,y-100);x1=min(1920,x+130);y1=min(1080,y+100)
        crop=overlay[y0:y1,x0:x1].copy()
        canvas=np.zeros((230,300,3),np.uint8);canvas[25:25+crop.shape[0],:crop.shape[1]]=crop
        cv2.putText(canvas,f'{n} {r["source_row"]["id"]}',(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        corner_panels.append(canvas)
    cv2.imwrite(str(OUT/'source_4_original_UV_corner_crops.png'),np.vstack([np.hstack(corner_panels[:2]),np.hstack(corner_panels[2:])]))
    original_retirement=json.loads((RUN/'input_manifest.json').read_text(encoding='utf8'))['original_source_role_retirement']
    cfg=json.loads((RUN.parent/(RUN.name+'.bootstrap')/'input_config.json').read_text(encoding='utf8'))
    confirmed_path=Path(cfg['source_options']['--user-confirmed-source-truth-authority'])
    confirmed=json.loads(confirmed_path.read_text(encoding='utf8'))
    confirmed_near=[r for r in confirmed['records'] if abs(int(r['pts'])-229651200)<=15360]
    report=dict(schema='rinkspace.original-secondary-Ad2-four-source-corners-readonly-RGB.v1',
        source_video_path=str(VIDEO),source_video_sha256_from_registered_timeline='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        source_time_base='1/15360',exact_decode_no_resampling=True,source_pts=229651200,
        packet_path=str(PACKET),packet_sha256=sha(PACKET),origin_file=str(origin_file),origin_sha256=sha(origin_file),
        annotated_image=str(OUT/'source_4_original_UV_labels_229651200.png'),
        source_raw_neighbors=[dict(pts=p,path=str(OUT/f'raw_native_{p}.png'),sha256=sha(OUT/f'raw_native_{p}.png')) for p in sorted(raws)],
        original_role_retirement_packet=original_retirement['packet_path'],
        original_retirement_locator=original_retirement['source_row_locator'],
        confirmed_authority_path=str(confirmed_path),confirmed_authority_nearby_records=confirmed_near,
        original_binding_method='Explicit physical_ids ad2:role (or literal row id) parsed by original pair_panels.identities; current named66 retains parsed physical number and material corner role.',
        no_RGB_or_user_confirmation_check_in_literal_identity_parser=True,
        no_later_confirmation_is_inferred_from_global_topology_wording=True,
        latest_direct_user_confirmation=dict(authority='Direct human user confirmation in the parent chat, 2026-10-10',
            screenshot_path='C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00096.png',
            exact_source_pts=229651200,confirmed_physical_identity='Ad2',
            confirmed_visible_material='purple ISUSKATING.COM printed panel',
            prior_cross_shot_numbering_conflict_resolved_by_latest_confirmation=True,
            original_four_UV_not_a_new_precise_user_pixel_annotation=True,
            original_UV_sigma3_and_source_packet_untouched=True,
            further_Ad2_identity_investigation_closed_per_user=True),
        direct_RGB_review=dict(raw_229651200_and_229650688_and_all_four_corner_crops_actually_viewed=True,
            visible_material='purple ISUSKATING.COM printed outer panel',
            four_source_UV_are_near_visible_actual_outer_corners=True,
            four_corners_themselves_not_occluded_by_skater=True,
            panel_four_corners_inside_native_frame=True,
            no_ice_arc_substituted_for_rectangle_observed=True,
            substantial_horizontal_motion_blur=True,
            sigma3_is_original_working_weight_not_measured_pixel_precision=True,
            RGB_review_does_not_confirm_physical_Ad_number_by_model_projection=True),
        no_projection_residual_model_or_numerical_solve_used=True,no_source_packet_or_qualification_modified=True,
        rows=rows)
    (OUT/'source_provenance_and_binding_readonly.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(OUT),rows=[dict(id=r['original_source_row']['id'],uv=r['original_source_row']['uv'],upstream_identity=r['upstream_original_record']['identity']) for r in rows],confirmed_nearby_count=len(confirmed_near)),ensure_ascii=False))

if __name__=='__main__':main()
