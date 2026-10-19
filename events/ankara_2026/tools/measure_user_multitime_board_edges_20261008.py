"""Append independently read source print transitions across pan/zoom states.

Reuses the existing source profile reader and NativeBoardEdge packet schema.
No camera/world/model projection, optimizer or complete graph is loaded.
"""
from copy import deepcopy
from pathlib import Path
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles
from measure_Ad4_17_19_multicut_source import serial

DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
EVIDENCE=DATA/'evidence/user_video_review_20261008/c134_full_programme'
OUT=DATA/'observations/vision_training/continuous_review/user_fullvideo_systematic_source_append_20261008'
MOTION=DATA/'programmes/okada_mei_sp/results/aligned_complete_20261007_478afaf77615/nine_shot_motion_regression_v2/native_board_motion_source_frames'
PARENT=DATA/'observations/vision_training/continuous_review/Ad4_17_18_19_source_absolute_20261007/board_parent_plus_Ad4_17_19_actual_RGB_multicut_v1.json'
SOURCE_SHA='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
FRAME_FILES={229256192:'01_ad4_and_ice_arc_source_native_229256192.png',
    229476864:'03_ad4_and_multiple_ice_lines_source_native_229476864.png',
    230823424:'10_Ad4_ice_same_view_recurs_source_native_230823424.png',
    230320640:'08_Ad5_shift_source_native_230320640.png',
    230330624:'09_Ad5_recovers_after_650ms_source_native_230330624.png'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write_new(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True);a=ap.parse_args()
    assert a.authority.strip();OUT.mkdir(parents=True,exist_ok=True)
    parent_bytes=PARENT.read_bytes();parent=read(PARENT);assert len(parent['rows'])==49
    camera=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_unlocked_geometry_af8cb_board739_canonical_gate_20261008/complete_physical_camera_model_UNACCEPTED_a90e01e5d788434d.npz')
    with np.load(camera,allow_pickle=False) as z:
        aliases=dict(zip(map(int,z['pts']),map(int,z['exposure_representative_pts'])))
        installations=dict(zip(map(int,z['pts']),map(str,z['installation_id'])))
    shots=read(ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
    # Only opened actual native sources establish these navigation windows.
    cases=[
        (230320640,5,'left_vertical',[[548,y]for y in(406,426,520,540)],[-20,20]),
        (230320640,5,'right_vertical',[[827,y]for y in(406,426,520,540)],[-20,20]),
        (230320640,5,'top_edge',[[x,393]for x in(605,695,785)],[-16,16]),
        (230320640,5,'bottom_edge',[[x,554]for x in(605,695,785)],[-16,16]),
        (230330624,5,'left_vertical',[[1016,y]for y in(361,381,464,485)],[-14,14]),
        (230330624,5,'right_vertical',[[1267,y]for y in(361,381,464,485)],[-14,14]),
        (230330624,5,'top_edge',[[x,350]for x in(1070,1140,1210)],[-13,13]),
        (230330624,5,'bottom_edge',[[x,497]for x in(1070,1140,1210)],[-13,13]),
        (229256192,4,'left_vertical',[[769,y]for y in(450,480,530)],[-14,14]),
        (229256192,4,'top_edge',[[x,426]for x in(1200,1450,1680)],[-10,10]),
        (229256192,4,'bottom_edge',[[1200,565],[1450,567],[1680,570]],[-10,10]),
        (229476864,4,'left_vertical',[[1440,y]for y in(410,450,500)],[-14,14]),
        (229476864,4,'top_edge',[[1530,383],[1770,384]],[-12,12]),
        (229476864,4,'bottom_edge',[[1530,545],[1770,547]],[-12,12]),
        (230823424,4,'left_vertical',[[722,y]for y in(486,520,551)],[-13,13]),
        (230823424,4,'top_edge',[[1140,466],[1300,466],[1430,466]],[-10,10]),
        (230823424,4,'bottom_edge',[[1140,570],[1300,571],[1430,572]],[-9,9]),
        (231534080,4,'left_vertical',[[72,y]for y in(339,365,395)],[-11,11]),
        (231534080,4,'right_vertical',[[1905,y]for y in(339,365,395)],[-11,11]),
        (231534080,4,'top_edge',[[150,324],[500,324],[1500,322]],[-8,8]),
        (231534080,4,'bottom_edge',[[150,413],[500,413],[1500,412]],[-8,8]),
    ]
    rows=[];unknown=[];images={};markers={};source_receipts={}
    for pts,number,edge,seeds,interval in cases:
        path=EVIDENCE/FRAME_FILES[pts] if pts in FRAME_FILES else MOTION/f'native_{pts}.png'
        if pts not in images:
            bgr=cv2.imread(str(path));assert bgr.shape==(1080,1920,3)
            images[pts]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);markers[pts]=bgr.copy()
            source_receipts[pts]=dict(native_pts=pts,actual_pts=aliases[pts],file=ref(path),
                decoded_native_RGB_sha256=hashlib.sha256(images[pts].tobytes()).hexdigest(),
                installation_id=installations[pts],source_exposure_alias_authority=ref(camera))
        shot=next(s for s in shots if s['start_pts']<=pts<s['end_pts'])
        normal=[1,0]if edge.endswith('vertical')else[0,1]
        transition='light_to_dark'if edge in('left_vertical','top_edge')else'dark_to_light'
        binding=dict(source_sha256=SOURCE_SHA,source_pts=pts,time_base='1/15360',shot_id=shot['id'],
            native_hw=[1080,1920],image_domain='native_brown_pixels')
        measurement=read_source_material_transition_profiles(images[pts],binding=binding,search_points_uv=seeds,
            normals=[normal]*len(seeds),search_intervals_px=[interval]*len(seeds),transition=transition,keep_profiles=True)
        good=[m for m in measurement['readings']if m['disposition']=='measured_material_transition']
        unknown.extend(dict(pts=pts,advertisement_number=number,edge=edge,**serial(m))for m in measurement['readings']if m not in good)
        if not good:continue
        for m in good:
            uv=m['actual_source_uv'];n=m['source_search_normal'];support=m['observed_normal_interval_px']
            cv2.line(markers[pts],tuple(np.rint(uv+n*support[0]).astype(int)),tuple(np.rint(uv+n*support[1]).astype(int)),(0,0,255),1)
            cv2.circle(markers[pts],tuple(np.rint(uv).astype(int)),3,(0,255,255),1)
        row=dict(id=f'user_fullvideo_multitime_native{pts}_Ad{number}_{edge}_actual_RGB',pts=pts,actual_pts=aliases[pts],
            feature_id=f'Ad{number}',split='fit',curve=dict(kind='advertisement_boundary',advertisement_number=number,
                edge=edge,extent_scope='external_border'),
            geometry_role='complete_event_name_group_external_border'if number==4 else'retained_internal_blue_white_green_MEIWA_graphic_boundary',
            uv=[m['actual_source_uv'].tolist()for m in good],normal=[m['source_search_normal'].tolist()for m in good],
            normal_interval_px=[m['observed_normal_interval_px'].tolist()for m in good],
            sigma_px=[max(4.,float(np.ptp(m['observed_normal_interval_px']))/2.)for m in good],
            sigma_semantics='Existing4px working floor or half actual q10/q90 optical support width including inherited2px margin; not calibrated covariance',
            point_correspondence=False,curve_observation_kind='normal_interval',source_coordinate_is_scanline_context_only=True,
            source_video_sha256=SOURCE_SHA,time_base='1/15360',source_raw_frame=str(path),source_file_sha256=sha(path),
            source_native_RGB_sha256=measurement['RGB_sha256'],measured_RGB_sources=serial(good),
            source_shot_id=shot['id'],physical_installation_id=installations[pts],
            source_identity_authority=dict(user_catalog=ref(ROOT/'configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json'),
                native_RGB_viewed=True,same_spectator_wall_and_neighbour_print_order=True,physical_number=number,
                Ad4_external_full_group_internal_component_joins_excluded=number==4,
                Ad5_old_internal_graphic_only_whole_40th_and_Japanese_print_not_rectangled=number==5,
                occluded_skater_or_missing_border_not_read=True),
            source_exposure_alias_authority=ref(camera),frozen_before_fit=True,
            model_projection_used_to_generate_or_select_uv=False,source_optical_support_not_model_error_selected=True,
            user_failure_frame_not_an_untouched_validation_frame=True,existing_HELD_pixels_not_promoted=True)
        rows.append(row)
    marker_refs=[]
    for pts,bgr in markers.items():
        target=OUT/f'native_{pts}_actual_SOURCE_edges_review.png'
        assert not target.exists();assert cv2.imwrite(str(target),bgr)
        marker_refs.append(dict(native_pts=pts,actual_pts=aliases[pts],**ref(target)))
    candidate=dict(schema='rinkspace.user-fullvideo-native-multitime-board-edge-source-candidate.v1',
        source_video_sha256=SOURCE_SHA,time_base='1/15360',rows=rows,rejected_or_unknown_source_queries=unknown,
        source_receipts=list(source_receipts.values()),source_marker_review_images=marker_refs,
        source_appearance_not_model_projection_selected=True,all_parent49_UV_sigma_normal_support_split_unchanged=True,
        original_HELD_not_promoted_to_FIT=True,source_frame_selection='User reported bad/good times and independently opened full external-group main9 source',
        model_world_parameters_or_runtime_not_loaded=True,candidate_only=True,authority=a.authority,measurement_script=ref(Path(__file__)))
    candidate_path=OUT/'Ad4_Ad5_multitime_actual_native_edges_source_candidate_v1.json';write_new(candidate_path,candidate)
    packet=deepcopy(parent);packet['rows']+=deepcopy(rows)
    packet.update(parent_packet=ref(PARENT),source_append_user_multitime=dict(candidate=ref(candidate_path),authority=a.authority,
        actual_source_before_geometry_read=True,fit_points=sum(len(r['uv'])for r in rows),held_points_added=0,
        parent49_exact_prefix_preserved=True,original_HELD_not_promoted=True,existing_material_roles_only=True,
        source_marker_review_status='actual_native_marker_review_pending'),
        current_target_version='parent49_plus_user_confirmed_multitime_Ad4_Ad5_native_edges_v1',
        current_fit_scalar_count=2*sum(len(r['uv'])for r in packet['rows']if r['split']=='fit'),
        solver_target_updated_here=False,accuracy_accepted=False)
    assert packet['rows'][:49]==parent['rows'] and PARENT.read_bytes()==parent_bytes
    output=OUT/'board_parent49_plus_Ad4_Ad5_multitime_actual_RGB_v1_CANDIDATE.json';write_new(output,packet)
    print(json.dumps(dict(candidate=ref(candidate_path),packet=ref(output),rows=len(rows),newFITpoints=sum(len(r['uv'])for r in rows),
        unknown=len(unknown),measurements=[dict(id=r['id'],uv=r['uv'],sigma=r['sigma_px'])for r in rows],markers=marker_refs)),flush=True)


if __name__=='__main__':main()
