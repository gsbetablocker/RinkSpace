# Project-owned event tool; source assets and all outputs stay in the original Data directory.
"""Source-only Ad11 wall identity and native dark-rim measurement.

No scene, camera, model, residual, solver or video is loaded. The new exposure
was independently cached before this script. Old v4 source rows stay exact.
"""
from pathlib import Path
import copy, hashlib, json
import cv2
import numpy as np

cv2.setNumThreads(1)
D = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/cut6_unused_actual_source_append_v9_20261005')
RUNS = D.parent
OLD = RUNS / 'spin_source_long_baseline_20261005/Ad24_spin_source_append_20261005/v4_mid_exposures'
PREFIX = OLD / 'additional_native_board_circles_Ad11_Ad24_shared_radius_FIT_v4_middle_inst2.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref = lambda p: {'path': str(p), 'sha256': sha(p)}

bindings_path = D / 'CUT6_UNUSED_ACTUAL_RAW_CANDIDATES.json'
bindings = json.loads(bindings_path.read_text(encoding='utf8'))
frame = next(r for r in bindings['records'] if r['native_pts'] == 230379776)
raw_path = Path(frame['raw_path'])
assert sha(raw_path) == frame['raw_file_sha256']
raw = cv2.imread(str(raw_path))
assert hashlib.sha256(raw.tobytes()).hexdigest() == frame['raw_bgr_sha256']
prefix = json.loads(PREFIX.read_text(encoding='utf8'))
assert not any(r['pts'] == 230379776 for r in prefix['rows'])
assert frame['actual_pts'] == 230379776 and frame['official_pts'] == 1310464
assert frame['source_clock_row']['match_status'] == 'unique_pixel_match'
old_ad11 = next(r for r in prefix['rows'] if r['feature_id'] == 'Ad11')
old_path = Path(old_ad11['source_raw_frame'])
assert sha(old_path) == old_ad11['source_file_sha256']
old_raw = cv2.imread(str(old_path))

# These non-print wall marks were found and reviewed in raw images alone.
# Their coordinates are identity evidence, not new fitted observations.
old_wall = [[1013.6819458007812,354.51214599609375],
            [1015.5565795898438,350.3881530761719],
            [1029.8411865234375,372.2512512207031],
            [1029.844970703125,359.2349548339844],
            [1030.76318359375,406.5361022949219],
            [1045.06591796875,364.27032470703125]]
new_wall = [[588.8574829101562,175.41221618652344],
            [591.6611328125,167.94039916992188],
            [618.322998046875,206.67550659179688],
            [615.7697143554688,181.26197814941406],
            [625.1870727539062,267.112548828125],
            [649.0125122070312,188.64254760742188]]
pieces = []
for image, points, roi, scale, title in (
        (old_raw, old_wall, [980,325,1080,430], 4, 'existing Ad11 229376000'),
        (raw, new_wall, [550,130,690,300], 3, 'unused cut6 230379776')):
    x0,y0,x1,y1 = roi
    piece = cv2.resize(image[y0:y1,x0:x1], None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_NEAREST)
    for i,(x,y) in enumerate(points):
        u,v = int(round((x-x0)*scale)), int(round((y-y0)*scale))
        cv2.circle(piece,(u,v),4,(0,220,255),1,cv2.LINE_AA)
        cv2.putText(piece,str(i),(u+5,v-3),0,.45,(0,220,255),1,cv2.LINE_AA)
    header = np.full((40,piece.shape[1],3),245,np.uint8)
    cv2.putText(header,title,(8,25),0,.48,(20,20,20),1,cv2.LINE_AA)
    pieces.append(np.vstack((header,piece)))
wall_canvas = np.full((max(p.shape[0] for p in pieces),sum(p.shape[1] for p in pieces)+12,3),245,np.uint8)
x = 0
for p in pieces:
    wall_canvas[:p.shape[0],x:x+p.shape[1]] = p
    x += p.shape[1]+12
wall_crop = D / '230379776_EXISTING_Ad11_FIXED_WALL_SOURCE_IDENTITY.jpg'
assert cv2.imwrite(str(wall_crop),wall_canvas)
identity = {
    'schema':'rinkspace.source-only-physical-board-circle-identity.v1',
    'source_video_sha256':bindings['source_video_sha256'],
    'native_pts':230379776,'actual_pts':230379776,'official_pts':1310464,
    'physical_advertisement_number':11,'installation_identity':1,
    'existing_identity_authority':ref(PREFIX),
    'existing_Ad11_raw':ref(old_path),'new_cut6_raw':ref(raw_path),
    'existing_Ad24_raw':ref(Path(next(r for r in prefix['rows'] if r['pts']==229582592)['source_raw_frame'])),
    'existing_cut5_Ad24_raw':ref(Path(next(r for r in prefix['rows'] if r['pts']==230112512)['source_raw_frame'])),
    'raw_wall_identity_crop':ref(wall_crop),
    'fixed_wall_evidence':{'old_native_uv':old_wall,'new_native_uv':new_wall,
        'surface':'fixed grey wall above the board, outside all printed logos',
        'interpretation':'Six matching holes/marks in the same fixed wall constellation; same concrete-wall physical placement as existing Ad11, rather than the glass doors/sign/staircase behind existing Ad24.',
        'raw_only_matching_method':'SIFT native raw grayscale wall ROI, ratio 0.75, source-pixel affine consensus; no print pixels or model projection used',
        'not_solver_observations':True},
    'neighbor_logo_order_alone_is_insufficient':True,
    'duplicate_neighbors':'Both physical copies have Federation circle, SPORTOTO and red food-brand badge; no identification from same print alone.',
    'camera_classification_alone_not_used_for_material_identity':True,
    'model_camera_world_or_residual_used':False,
    'source_only_manual_review_pending':True}
identity_path = D / '230379776_PHYSICAL_Ad11_SOURCE_ONLY_IDENTITY.json'
identity_path.write_text(json.dumps(identity,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

# Clear isolated OUTER black stroke windows, fixed from original raw crop.
# The same v3/v4 measurement rule is retained; no ellipse/center is inferred.
specs = [('left_outer','x',480,(327,345)),
         ('left_outer','x',535,(307,325)),
         ('left_outer','x',590,(316,337)),
         ('right_outer','x',480,(523,544)),
         ('right_outer','x',535,(538,556)),
         ('right_outer','x',605,(522,543)),
         ('top_outer','y',410,(424,439)),
         ('top_outer','y',450,(425,442)),
         ('bottom_outer','y',420,(657,677)),
         ('bottom_outer','y',460,(656,677))]
grey = cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(float)
uv, sigma, supports, missing = [], [], [], []
for side,axis,c,(lo,hi) in specs:
    profile = grey[c-1:c+2,lo:hi+1].mean(0) if axis=='x' else grey[lo:hi+1,c-1:c+2].mean(1)
    smooth = cv2.GaussianBlur(profile[None,:].astype('float32'),(3,1),.65)[0].astype(float)
    j = int(np.argmin(smooth))
    baseline = float(np.mean(np.r_[smooth[:2],smooth[-2:]]))
    contrast = baseline-smooth[j]
    if j in (0,len(profile)-1) or contrast<3:
        missing.append({'side':side,'axis':axis,'fixed':c,'native_search_support':[lo,hi],
                        'reason':'no isolated interior dark ridge with raw contrast >=3'})
        continue
    den = smooth[j-1]-2*smooth[j]+smooth[j+1]
    delta = 0. if den<=0 else float(np.clip(.5*(smooth[j-1]-smooth[j+1])/den,-.5,.5))
    loc = lo+j+delta
    p = [loc,float(c)] if axis=='x' else [float(c),loc]
    threshold = .5*(baseline+smooth[j]); left=right=j
    while left>0 and smooth[left-1]<=threshold:left-=1
    while right<len(profile)-1 and smooth[right+1]<=threshold:right+=1
    width=float(right-left+1); s=max(4.,width/2.)
    uv.append(p); sigma.append(s)
    supports.append({'side':side,'axis':axis,'fixed':c,'native_search_support':[lo,hi],
        'uv':p,'sigma_px':s,'source_raw_gray_profile':profile.tolist(),
        'source_smoothed_gray_profile':smooth.tolist(),
        'smoothing':'3 native samples Gaussian sigma0.65 only for dark ridge localization',
        'dark_rim_contrast_native_gray':float(contrast),
        'subpixel_method':'three-sample parabola around isolated source intensity minimum',
        'half_depth_native_width_px':width,
        'source_array_operation':'native gray original pixels; 3 cross-axis pixels; no image resampling'})
marked = raw.copy()
for i,p in enumerate(uv):
    x,y=np.rint(p).astype(int)
    cv2.circle(marked,(x,y),2,(20,210,250),1,cv2.LINE_AA)
    cv2.putText(marked,str(i),(x+4,y-3),0,.35,(20,20,20),1,cv2.LINE_AA)
roi=[285,410,580,690];x0,y0,x1,y1=roi
crop = D / '230379776_Ad11_OUTER_RIM_SOURCE_ONLY_CANDIDATE_CROP.png'
assert cv2.imwrite(str(crop),cv2.resize(marked[y0:y1,x0:x1],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
row = {
    'id':'cut6_Ad11_outer_black_rim_native230379776','pts':230379776,
    'actual_pts':230379776,'feature_id':'Ad11',
    'curve':{'kind':'board_circle','advertisement_id':11,'edge':'outermost_visible_circular_print_stroke'},
    'role':'Only directly visible outermost black Federation printed rim; no lettering, inner ring, ellipse center or hidden outline',
    'uv':uv,'sigma_px':sigma,'source_support':supports,'missing_source_support':missing,
    'split':'fit','candidate_split':'new_fit_after_source_review','solver_admitted':False,
    'source_raw_frame':str(raw_path),'source_file_sha256':frame['raw_file_sha256'],
    'source_native_bgr_pixels_sha256':frame['raw_bgr_sha256'],
    'official_exposure_match':frame['source_clock_row'],
    'source_exposure_binding':ref(bindings_path),'source_time':'04:09:58.683333',
    'physical_identity_evidence':ref(identity_path),'installation_identity':1,
    'source_only_manual_roi_xyxy':roi,
    'source_only_review_crop':dict(**ref(crop),native_crop_xyxy=roi,scale=3),
    'source_sigma_semantics':'max(4px existing board engineering floor, half native dark stroke half-depth width); not calibrated covariance',
    'source_measurement_method':'Same v3/v4 native isolated dark-rim minimum method in raw-only manually selected narrow windows; no ellipse/center/world/model fit',
    'source_geometry_not_observed':True,'point_correspondence':False,'endpoint_observation':False,
    'source_surface_identity':'board_surface','raw_native_size_wh':[1920,1080],
    'model_projection_used_to_generate_or_select_uv':False,
    'source_review_authority':'Parent explicit source-only cut6 authorization, physical Ad11 wall identity and isolated outer-rim candidate review pending final crop qualification'}
candidate = {'schema':'rinkspace.cut6-native-board-circle-source-only-candidate.v1',
    'status':'SOURCE_ONLY_CANDIDATE_PENDING_FINAL_RAW_CROP_REVIEW',
    'source_video_sha256':bindings['source_video_sha256'],'time_base':'1/15360',
    'image_size_wh':[1920,1080],'parent_v4':ref(PREFIX),'row':row,
    'measurement_script':ref(Path(__file__)),'protected_old_HELD_unchanged':True,
    'new_parameter_count':0,'shared_radius_and_independent_centers_unchanged':True,
    'model_camera_world_solver_decode_or_video_executed':False,
    'accuracy_accepted':False}
out = D / 'cut6_230379776_Ad11_OUTER_RIM_SOURCE_ONLY_CANDIDATE_v1.json'
out.write_text(json.dumps(candidate,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({'candidate':ref(out),'identity':ref(identity_path),'wall_crop':ref(wall_crop),
                  'rim_crop':ref(crop),'UV':uv,'sigma':sigma,'missing':missing},indent=2))
