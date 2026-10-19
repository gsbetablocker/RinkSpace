"""Freeze actually viewed new same-ink anchors; preserve the historical parent."""
from copy import deepcopy
from pathlib import Path
import argparse, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'src')];sys.dont_write_bytecode=True
from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2,numpy as np
from rinkspace.vision_source_shapes import read_source_material_transition_profiles
from measure_Ad4_17_19_multicut_source import serial
from measure_user_multitime_board_edges_20261008 import OUT,SOURCE_SHA,ref,read,write_new
from measure_user_bad_Ad22_23_frozen_material_20261008 import PARENT


def profile_context(rgb,uv,number):
    """Read actual local blur without replacing a two-dimensional ink point.

    A glyph junction is not a single signed edge: preserve reader flags and
    unknowns, rather than calling its q50 an alternative point measurement.
    """
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    records=[]
    for i,p in enumerate(np.asarray(uv,float)):
        x,y=np.rint(p).astype(int);direction=np.array([gx[y-1:y+2,x-1:x+2].mean(),gy[y-1:y+2,x-1:x+2].mean()])
        if np.linalg.norm(direction)<1e-6:
            records.append(dict(point_index=i,disposition='source_material_role_unknown',reason='local_gradient_direction_unresolved',point_measurement_not_replaced=True));continue
        result=read_source_material_transition_profiles(rgb,binding=dict(source_sha256=SOURCE_SHA,source_pts=230036992,
            time_base='1/15360',shot_id='user_bad_same_actual_native',native_hw=[1080,1920],image_domain='native_brown_pixels'),
            search_points_uv=[p],normals=[direction],search_intervals_px=[[-12.,12.]],transition='dark_to_light',keep_profiles=True)
        records.append(dict(point_index=i,advertisement_number=number,point_measurement_not_replaced=True,
            two_dimensional_glyph_identity_not_a_single_edge=bool(any(result['readings'][0]['source_flags'].values())),
            actual_profile=serial(result['readings'][0])))
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority',required=True);a=ap.parse_args();assert a.authority.strip()
    pending=OUT/'native_six_parent_plus_Ad22_Ad23_bad_time_actual_frozen_ink_SOURCE_REVIEW_PENDING_v1.json'
    cpath=OUT/'Ad22_Ad23_bad_time_same_frozen_ink_actual_native_anchor_CANDIDATE_v1.json'
    parent_bytes=PARENT.read_bytes();parent=read(PARENT);source=read(pending);candidate=read(cpath)
    assert candidate['parent']==ref(PARENT)
    n=len(parent['source_anchor_observations']);rows=source['source_anchor_observations'][n:]
    assert len(parent['rows'])==72 and n==12 and len(rows)==2
    assert source['rows']==parent['rows'] and source['source_anchor_observations'][:n]==parent['source_anchor_observations']
    assert source['frozen_source_print_templates']==parent['frozen_source_print_templates']
    assert rows==candidate['new_source_anchor_rows']
    identities=[];profiles=[]
    for row,review in zip(rows,candidate['source_only_pair_reviews']):
        assert row['split']=='fit' and row['actual_pts']==row['native_pts']==row['pts']==230036992
        old=next(r for r in parent['source_anchor_observations']if r['id']==row['old_FIT_material_reference']['id'])
        ix=row['actual_old_FIT_same_ink_indices'];assert old['split']=='fit' and len(ix)==len(set(ix))
        assert row['initialization']['material_coordinates_normalized']==[old['initialization']['material_coordinates_normalized'][i]for i in ix]
        assert row['initialization']['template_id']==old['initialization']['template_id']
        assert row['initialization']['q0_sha256']==old['initialization']['q0_sha256']
        assert row['sigma_px']==old['sigma_px']==[6.,6.]
        for field in('source_raw_frame',):assert ref(Path(row[field]['path']))==row[field]
        assert ref(Path(old['source_raw_frame']['path']))==old['source_raw_frame']
        for i,r in enumerate(review['points']):
            assert r['old_FIT_anchor_point_index']==ix[i] and r['source_UV']==old['uv'][ix[i]]
            assert r['target_UV']==row['uv'][i] and r['original_sigma']==row['sigma_px']
            assert r['normalized_material_coordinates']==row['initialization']['material_coordinates_normalized'][i]
            assert r['FB_native_px']<=.75 and r['LK_vs_source_H_native_px']<=3.
        rgb=cv2.cvtColor(cv2.imread(row['source_raw_frame']['path']),cv2.COLOR_BGR2RGB);assert rgb.shape==(1080,1920,3)
        old_rgb=cv2.cvtColor(cv2.imread(old['source_raw_frame']['path']),cv2.COLOR_BGR2RGB)
        identities.append(dict(advertisement_number=row['advertisement_number'],native_pts=row['native_pts'],points=len(ix),
            new_frame=deepcopy(row['source_raw_frame']),new_decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            reference_frame=deepcopy(old['source_raw_frame']),reference_decoded_native_RGB_sha256=hashlib.sha256(old_rgb.tobytes()).hexdigest(),
            original_FIT_indices=ix,old_sigma_and_frozen_material_coordinates_exact=True,
            original_HELD_not_promoted=True,actual_target_UV_not_model_selected=True))
        profiles.append(dict(advertisement_number=row['advertisement_number'],profile_role='Actual native optical support QA only; not replacing same-ink UV, uncertainty or correspondence',
            original_sigma_px=row['sigma_px'],sigma_semantics='Original NativeBoardTrack6px engineering convention; not a calibrated subpixel covariance',
            profiles=profile_context(rgb,row['uv'],row['advertisement_number'])))
    for page in candidate['source_marker_review_pages']:assert ref(Path(page['path']))=={k:page[k]for k in('path','sha256')}
    qa=dict(schema='rinkspace.user-fullvideo-native-same-frozen-ink-source-qualification.v1',
        authority=a.authority,parent=ref(PARENT),candidate=ref(cpath),pending_packet=ref(pending),
        identities=identities,actual_native_source_profile_context=profiles,
        actual_both_native_RGB_and_marker_pages_viewed_by=['ad_evidence','source_review'],
        source_marker_review_pages=deepcopy(candidate['source_marker_review_pages']),
        material_qualification=dict(Ad22_all9_same_ozcan_shield_top_lower_outline_or_internal_glyph=True,
            Ad23_all5_same_SPORTOTO_left_S_or_red_top_stripe=True,person_UI_ice_equipment_or_wrong_installation_points=0,
            blur_allows_component_identity_and_coarse_correspondence=True,subpixel_precision_not_certified=True),
        independent_actual_review_text={
            'Ad22':'source_review viewed0..8: same özcan red shield, top arch4/5, lower borders0..3/7, internal glyph6/8; coarse identity supported, target blur cannot certify subpixel precision.',
            'Ad23':'source_review viewed0..4: same SPORTOTO left S/red top stripe, no people/equipment/UI/board seam or different instance; coarse correspondence supported, original FB/H and6px engineering sigma retained.'},
        source_only_matching_failed_history_preserved=dict(Ad23_direct_full_native_LK_kept=0,
            Ad23_registered_source_maxLevel5_LK_kept=0,Ad23_registered_source_maxLevel2_final_kept=5,
            FB_gate_original_native_px=.75,LK_vs_source_H_gate_target_native_px=3.,
            source_error_gates_not_relaxed=True,self_invented_minimum6_frame_point_capacity_removed=True),
        old72_tracks_old12_anchors_and_frozen_templates_exact=True,newFITpoints=14,newFITscalar_rows=28,
        newHELDpoints=0,new_world_parameters=0,new_camera_parameters=0,
        solver_or_current_geometry_not_loaded=True,accuracy_accepted=False)
    qpath=OUT/'Ad22_Ad23_bad_time_same_frozen_ink_actual_native_source_QA_FINAL_v1.json';write_new(qpath,qa)
    packet=deepcopy(source)
    for row in packet['source_anchor_observations'][n:]:
        row['source_only_material_review_pending']=False
        row['actual_both_native_RGB_same_ink_components_qualified']=True
        row['source_subpixel_precision_not_certified']=True
        row['source_admission_QA']=ref(qpath)
    packet['source_append_user_bad_Ad22_23'].update(actual_material_review_pending=False,source_QA=ref(qpath),
        old72_track_rows_old12_anchor_rows_exact=True,newFITscalar_rows=28)
    packet.update(source_review_pending_before_production_admission=False,
        actual_native_material_and_same_ink_qualified=True,source_admission_QA=ref(qpath),
        new_source_measurement_authority=a.authority,solver_target_updated_here=False)
    assert packet['rows']==parent['rows'] and packet['source_anchor_observations'][:n]==parent['source_anchor_observations']
    assert packet['frozen_source_print_templates']==parent['frozen_source_print_templates'] and PARENT.read_bytes()==parent_bytes
    out=OUT/'native_six_parent_plus_Ad22_Ad23_bad_time_actual_frozen_ink_ADMITTED_v1.json';write_new(out,packet)
    print(json.dumps(dict(packet=ref(out),source_QA=ref(qpath),newFITrows=2,newFITpoints=14,newFITscalar=28,
        old_track_rows_exact=72,old_source_anchors_exact=12,newHELDpoints=0,new_parameters=0)),flush=True)


if __name__=='__main__':main()
