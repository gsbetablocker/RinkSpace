"""Bind the user's exact ad7 reflection example to frozen original FIT pixels."""
from pathlib import Path
import hashlib
import json
import sys
import argparse

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import av
import cv2
import numpy as np

cv2.setNumThreads(1)
parser=argparse.ArgumentParser()
parser.add_argument('--write-reviewed-negatives',action='store_true')
args=parser.parse_args()
BASE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
WORK=BASE/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012'
PTS=229319936
VIDEO=BASE/'media/originals/oxXu1avhrvs.mp4'
source_sha='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
target=WORK/'source_feedback/ad7_original_source_229319936.png'
with av.open(str(VIDEO)) as container:
    stream=container.streams.video[0];stream.thread_count=1
    container.seek(PTS,stream=stream,backward=True)
    for frame in container.decode(stream):
        if frame.pts<PTS:continue
        if frame.pts!=PTS:raise ValueError('User source timestamp differs')
        raw=frame.to_ndarray(format='bgr24');break
    else:raise ValueError('Original frame missing')
digest=hashlib.sha256(raw.tobytes()).hexdigest()
saved=None
for chunk in sorted((WORK/'source_pixels').glob('source_pixels_*.npz')):
    with np.load(chunk,allow_pickle=False) as z:
        at=np.flatnonzero(z['actual_pts']==PTS)
        if len(at):saved=str(z['BGR_sha256'][at[0]]);break
if saved!=digest:raise ValueError('Decoded user example differs from the frozen original source stream')
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
    if not np.array_equal(cv2.imread(str(target)),raw):raise ValueError('Do not replace original feedback source pixels')
else:cv2.imwrite(str(target),raw)
print(json.dumps(dict(original_source=str(target),actual_pts=PTS,decoded_BGR_sha256=digest),ensure_ascii=False))
if args.write_reviewed_negatives:
    from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, file_sha
    from rinkspace.strict_source_supervision import write_strict_negative_source_pack
    user_input=Path('C:/Users/REDACTED/AppData/Local/Temp/codex-clipboard-private-attachment-00054.png')
    preserved=target.with_name('user_ad7_reflection_example.png')
    if preserved.exists():
        if file_sha(preserved)!=file_sha(user_input):raise ValueError('Do not overwrite human source evidence')
    else:preserved.write_bytes(user_input.read_bytes())
    frozen=BASE/'calibration/runs/solver_acceleration_20261003/okada_q32_reviewed_source_pixels_current_points_ready_20261012/current_active_source_pixels_FROZEN.npz'
    with np.load(frozen,allow_pickle=False) as z:
        index=int(np.flatnonzero(z['actual_pts']==PTS)[0]);begin,end=map(int,z['offsets'][index:index+2])
        uv=z['uv'][begin:end]
        # This finite region is the user's marked reflection, manually checked
        # in the exact native RGB above. It is not an automatic ice classifier.
        indices=np.flatnonzero((uv[:,0]>174)&(uv[:,0]<224)&(uv[:,1]>742)&(uv[:,1]<1010))
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots']
    shot=next(s['id'] for s in shots if int(s['start_pts'])<=PTS<int(s['end_pts']))
    original=dict(path=str(VIDEO),sha256=source_sha,kind='original_video_exact_native_frame',
        native_pts=PTS,decoded_BGR_sha256=digest)
    authority=dict(path=str(preserved),sha256=file_sha(preserved),
        statement='这框里的这些点是ad7在冰面上的倒影')
    source=dict(video_sha256=source_sha,time_base='1/15360',pixel_domain='native1920x1080')
    frozen_reference=dict(path=str(frozen),sha256=file_sha(frozen))
    records=[]
    for i in indices:
        records.append(dict(actual_pts=PTS,native_pts=PTS,uv=uv[i].tolist(),shot_id=shot,
            decoded_BGR_sha256=digest,decision='exclude',scope='outside_model',
            physical_material_id='Ad7.ice_reflection',static_material_confirmed=False,
            pixel_location_confirmed=True,review_authority='human_original_source_review',
            reviewer='user; Codex exact native source verification',
            original_source_image_actually_reviewed=True,
            model_projection_or_fit_residual_used_for_identity=False,
            review_note='用户圈定的Ad7蓝广告冰面倒影竖边；已对照精确原片和原FIT像素，非真实冰线或板上广告边界。',
            source_image_evidence=original,human_region_authority=authority,
            reject_only_authority='actual_visual_role_absence',
            actual_source_RGB_role_absence_confirmed=True,
            confidence_used_as_rejection_evidence=False,model_probability_used_as_rejection_evidence=False,
            excluded_kinds=['ice_paint','ad_boundary'],
            original_frozen_pixel_index=begin+int(i),
            original_pixel_inventory_reference=frozen_reference,
            coordinate_origin='existing_frozen_active_input_uv_inside_user_marked_reflection_region',
            no_new_human_point_coordinate_created=True))
    review=ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,source=source,reviewed_endpoints=records))
    output=target.with_name('user_ad7_actual_role_absence_negative_points.npz')
    if output.exists():
        with np.load(output,allow_pickle=False) as saved:
            if not np.array_equal(saved['uv'],uv[indices]):raise ValueError('Existing source negative locations changed')
            if json.loads(str(saved['metadata_json']))['source']!=source:raise ValueError('Existing source negative identity changed')
        reference=dict(path=str(output),sha256=file_sha(output),points=len(records))
    else:reference=write_strict_negative_source_pack(output,review)
    print(json.dumps(dict(reviewed_negative_reference=reference),ensure_ascii=False))
