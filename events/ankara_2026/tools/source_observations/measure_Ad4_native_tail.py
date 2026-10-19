# Project-owned event tool; source assets and all outputs stay in the original Data directory.
"""Measure only native visible print borders; no model/solver import."""
from pathlib import Path
import hashlib, json
import cv2
import numpy as np

cv2.setNumThreads(1)
D = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/ads_actual_identity_correction_20261005/Ad4_end_programme_singleton_source_candidate_20261005')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref = lambda p: {'path':str(p), 'sha256':sha(p)}
R = D/'231705088_SOURCE_RAW_IDENTITY.json'
source = json.loads(R.read_text(encoding='utf8'))
P = Path(source['raw_path'])
assert sha(P)==source['raw_file_sha256']
raw = cv2.imread(str(P))
assert raw.shape==(1080,1920,3)
assert hashlib.sha256(raw.tobytes()).hexdigest()==source['raw_bgr_sha256']
assert source['native_pts']==source['actual_pts']==231705088
assert source['official_pts'] is None
assert source['source_clock_row']['accepted_official_pts'] is None
rows=[]; marked=raw.copy()
for edge,ys,lo,hi,color,sign in (
        ('left_vertical',(310,322,334),6,36,'magenta',1),
        ('right_vertical',(306,316),888,916,'cyan',-1)):
    uv=[]; supports=[]; profiles=[]
    for y in ys:
        rgb = raw[y-3:y+4,lo:hi+1].mean(axis=0)
        score = (rgb[:,0]+rgb[:,2])/2-rgb[:,1] if color=='magenta' else rgb[:,0]-rgb[:,2]
        mass = np.maximum(sign*np.diff(score),0.)
        assert mass.sum()>8., 'No isolated source chroma border support'
        q = np.interp(np.array([.1,.5,.9])*mass.sum(),np.cumsum(mass),np.arange(lo+.5,hi+.5))
        assert lo+2<q[1]<hi-2
        uv.append([float(q[1]),float(y)])
        supports.append((q[[0,2]]-q[1]).tolist())
        profiles.append({'source_y':y,'source_roi_xyxy':[lo,y-3,hi,y+3],
            'source_chroma_definition':'(B+R)/2-G' if color=='magenta' else 'B-R',
            'source_chroma_profile':score.tolist(),
            'signed_positive_chroma_gradient_mass':mass.tolist(),'sign':sign,
            'q10_q50_q90_x':q.tolist(),'q10_q90_width_px':float(q[2]-q[0]),
            'smoothing_sigma_px':0.,'cross_axis_native_pixel_count':7,
            'source_only_window_selected_from_opened_native_raw':True,
            'selection_uses_model_camera_or_residual':False})
        cv2.line(marked,(round(q[0]),y),(round(q[2]),y),(0,0,230),1)
        cv2.circle(marked,(round(q[1]),y),1,(0,230,230),1,cv2.LINE_AA)
    crop_roi=[0,285,75,365] if edge=='left_vertical' else [850,282,945,345]
    x0,y0,x1,y1=crop_roi
    crop=D/f'231705088_Ad4_{edge}_SOURCE_ONLY_MEASURED_CROP.png'
    assert cv2.imwrite(str(crop),cv2.resize(marked[y0:y1,x0:x1],None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST))
    rows.append({'id':f'Ad4_external_{edge}_unused_tail_native231705088',
        'pts':231705088,'actual_pts':231705088,'official_pts':None,'feature_id':'Ad4',
        'curve':{'kind':'advertisement_boundary','advertisement_number':4,'edge':edge,'extent_scope':'external_border'},
        'split':'fit','proposed_split':'fit','solver_admitted':False,
        'uv':uv,'sigma_px':[4.]*len(uv),'normal':[[1.,0.]]*len(uv),
        'normal_interval_px':supports,'curve_observation_kind':'normal_interval',
        'point_correspondence':False,'endpoint_observation':False,
        'source_y_role':'Original native scanline context only; not fixed world Z or endpoint correspondence',
        'source_raw_frame':str(P),'source_file_sha256':source['raw_file_sha256'],
        'source_native_bgr_pixels_sha256':source['raw_bgr_sha256'],
        'source_video_sha256':source['source_video_sha256'],'time_base':'1/15360',
        'source_time':'04:11:24.966667','source_clock':source['source_clock_row'],
        'source_clock_authority':source['source_clock_authority'],
        'actual_alias_authority':source['actual_alias_authority'],'source_exposure_proof':ref(R),
        'source_identity_authority':{'physical_number':4,
            'role':'Same upper spectator-side ANKARA / figure-skating mark / JUNIOR GRAND PRIX event-name group; external magenta left boundary and exposed cyan right boundary',
            'left_vertical_is_material_u0':True,'right_vertical_is_material_u1':True,
            'source_visible_extent':'Left border fully in native frame; only two unobstructed upper scanlines of right border sampled, head-overlapped lower right excluded',
            'source_only_raw_manually_opened_and_identity_confirmed':True,
            'identity_not_from_template_similarity_alone':True},
        'source_scan_profiles':profiles,'source_sigma_semantics':'Same existing 4 native px supplemental board engineering sigma; not covariance, no model-error adjustment',
        'source_only_review_crop':dict(**ref(crop),native_crop_xyxy=crop_roi,scale=6),
        'model_pixels_used_in_source_measurement':False,'source_only':True,
        'frozen_before_fit':True,'nearby_exposure_and_same_edge_pixels_correlated':True,
        'HELD_UV_copied':False})
candidate={'schema':'rinkspace.native-Ad4-end-programme-edge-source-candidates.v1',
    'status':'SOURCE_ONLY_MEASURED_CANDIDATE_NOT_ADMITTED_TO_CURRENT_TARGET',
    'source_video_sha256':source['source_video_sha256'],'time_base':'1/15360',
    'raw_source_identity':ref(R),'rows':rows,'native_pts':231705088,'actual_pts':231705088,
    'official_pts':None,'official_clock_match_not_certified':True,
    'actual_exposure_from_existing_explicit_direct_alias_authority':True,
    'protected_HELD_native_pts':[229943552,231704064,231705344],
    'all_existing_source_UV_sigma_support_split_unchanged':True,
    'new_candidate_point_count':sum(len(r['uv'])for r in rows),
    'new_candidate_scalar_count':2*sum(len(r['uv'])for r in rows),
    'new_geometry_parameters_or_shared_sizes_added':False,
    'no_model_camera_world_solver_decode_video_or_config_change':True,
    'source_image_operation':'Original native BGR strips; no image resampling or sharpening for measurement; display crops nearest-neighbour only',
    'measurement_script':ref(Path(__file__)),'accuracy_accepted':False}
O=D/'Ad4_unused_tail_231705088_SOURCE_ONLY_edge_candidates_v1.json'
assert not O.exists()
O.write_text(json.dumps(candidate,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({'candidate':ref(O),'rows':[{'id':r['id'],'uv':r['uv'],'sigma_px':r['sigma_px'],'normal_interval_px':r['normal_interval_px'],'source_widths_px':[x['q10_q90_width_px']for x in r['source_scan_profiles']],'crop':r['source_only_review_crop']}for r in rows]},indent=2))
