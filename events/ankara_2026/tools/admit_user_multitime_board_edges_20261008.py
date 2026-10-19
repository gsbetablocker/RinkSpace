"""Review-authenticated append to the existing finite board material source."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles
from measure_user_multitime_board_edges_20261008 import OUT,PARENT,SOURCE_SHA,serial,ref,read,write_new


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True);a=ap.parse_args()
    assert a.authority.strip()
    cp=OUT/'Ad4_Ad5_multitime_actual_native_edges_source_candidate_v1.json'
    candidate=read(cp);parent=read(PARENT);old=deepcopy(parent['rows']);rows=deepcopy(candidate['rows'])
    assert len(old)==49 and len(rows)==20
    frame=next(r for r in candidate['source_receipts']if r['native_pts']==230320640)
    assert ref(Path(frame['file']['path']))==frame['file']
    bgr=cv2.imread(frame['file']['path']);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    binding=dict(source_sha256=SOURCE_SHA,source_pts=230320640,time_base='1/15360',
        shot_id='223403520:R0402_038',native_hw=[1080,1920],image_domain='native_brown_pixels')
    # The first search was demonstrably truncated on its right plateau. Read
    # the larger real white-to-blue/green strip; no detector threshold changes.
    proof=read_source_material_transition_profiles(rgb,binding=binding,
        search_points_uv=[[548,y]for y in(406,426,520,540)],normals=[[1,0]]*4,
        search_intervals_px=[[-36,36]]*4,transition='light_to_dark',keep_profiles=True)
    good=[m for m in proof['readings']if m['disposition']=='measured_material_transition']
    if good:
        row=deepcopy(next(r for r in rows if r['pts']==230330624 and r['curve']['edge']=='left_vertical'))
        row.update(id='user_fullvideo_multitime_native230320640_Ad5_left_vertical_actual_RGB_full_blur',
            pts=230320640,actual_pts=frame['actual_pts'],physical_installation_id=frame['installation_id'],
            uv=[m['actual_source_uv'].tolist()for m in good],normal=[[1.,0.]]*len(good),
            normal_interval_px=[m['observed_normal_interval_px'].tolist()for m in good],
            sigma_px=[max(4.,float(np.ptp(m['observed_normal_interval_px']))/2.)for m in good],
            source_raw_frame=frame['file']['path'],source_file_sha256=frame['file']['sha256'],
            source_native_RGB_sha256=proof['RGB_sha256'],measured_RGB_sources=serial(good),
            actual_source_navigation_refinement_reason='Original source scan recorded search_interval_truncated=true; wider real RGB strip reaches both plateaus',
            source_profile_reading_thresholds_or_smoothing_changed=False)
        rows.append(row)
    review=deepcopy(bgr)
    for m in good:
        uv=m['actual_source_uv'];n=m['source_search_normal'];lo,hi=m['observed_normal_interval_px']
        cv2.line(review,tuple(np.rint(uv+n*lo).astype(int)),tuple(np.rint(uv+n*hi).astype(int)),(0,0,255),1)
        cv2.circle(review,tuple(np.rint(uv).astype(int)),3,(0,255,255),1)
    marker=OUT/'native_230320640_Ad5_left_full_blur_actual_SOURCE_review.png';assert not marker.exists();assert cv2.imwrite(str(marker),review)
    proofpath=OUT/'Ad5_bad_left_actual_full_blur_source_profile_reading_v1.json'
    write_new(proofpath,dict(original_candidate=ref(cp),original_raw_frame=frame['file'],
        actual_RGB_scan_reading=serial(proof),source_marker_review_image=ref(marker),
        no_model_world_camera_projection_loaded=True,actual_native_marker_review_pending=True))
    qa=dict(schema='rinkspace.user-fullvideo-multitime-native-board-edge-append-source-QA.v1',
        authority=a.authority,source_candidate=ref(cp),additional_blur_reading=ref(proofpath),
        parent=ref(PARENT),original_parent49_rows_exact_preserved=True,
        source_native_RGB_marker_review=dict(all_six_original_full_native_marker_frames_viewed=True,
            blue_white_green_internal_Ad5_graphic_only=True,Ad4_full_event_external_group_only=True,
            no_selected_person_UI_or_ice=True,missing_Ad4_right_edges_in_main1_main7_not_invented=True,
            full_blur_left_marker_review_pending=True),
        optical_profile_q10_q50_q90_preserved=True,source_sigma_engineering_floor_or_half_blur_support=True,
        original_FIT_HELD_UV_sigma_support_not_rewritten=True,original_HELD_not_promoted=True,
        newFIT_points=sum(len(r['uv'])for r in rows),new_scalar_rows=2*sum(len(r['uv'])for r in rows),
        new_HELD_points=0,new_geometry_or_camera_parameters=0,source_only_reading_not_model_error_selected=True,
        source_qualification_not_complete_until_full_blur_marker_view=True,accuracy_accepted=False)
    qpath=OUT/'Ad4_Ad5_multitime_actual_native_edges_source_QA_v1.json';write_new(qpath,qa)
    packet=deepcopy(parent);packet['rows']+=rows
    packet.update(parent_packet=ref(PARENT),source_append_user_multitime=dict(candidate=ref(cp),
        additional_source_profile=ref(proofpath),source_QA=ref(qpath),authority=a.authority,
        new_FIT_points=qa['newFIT_points'],new_HELD_points=0,parent49_exact_prefix_preserved=True),
        current_target_version='parent49_plus_user_multitime_Ad4_Ad5_actual_RGB_v2',
        fit_observation_count=sum(r['split']=='fit'for r in packet['rows']),
        held_observation_count=sum(r['split']=='holdout'for r in packet['rows']),
        current_fit_scalar_count=2*sum(len(r['uv'])for r in packet['rows']if r['split']=='fit'),
        current_primary_held_scalar_count=2*sum(len(r['uv'])for r in packet['rows']if r['split']=='holdout'),
        source_marker_review_pending_full_blur_left=True,solver_target_updated_here=False,accuracy_accepted=False)
    assert packet['rows'][:49]==old
    output=OUT/'board_parent49_plus_Ad4_Ad5_multitime_actual_RGB_v2_SOURCE_REVIEW_PENDING.json';write_new(output,packet)
    print(json.dumps(dict(packet=ref(output),QA=ref(qpath),proof=ref(proofpath),marker=ref(marker),
        newFITpoints=qa['newFIT_points'],full_left_uv=[m['actual_source_uv'].tolist()for m in good],
        full_left_sigma=[max(4.,float(np.ptp(m['observed_normal_interval_px']))/2.)for m in good],
        rejected_full_left=[serial(m['source_flags'])for m in proof['readings']if m['disposition']!='measured_material_transition'])),flush=True)


if __name__=='__main__':main()
