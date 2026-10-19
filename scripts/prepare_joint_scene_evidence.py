"""Reuse confirmed image observations and add measured complementary contours.

No scene projection is used to choose pixels. Search guides are image-space
regions inspected by the operator; every new point is an image gradient sample.
"""
from rinkspace.storage import resolve_asset as _resolve_asset
from pathlib import Path
import copy
import hashlib
import json
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_scene3d_calibration_observations import extract_ring, sample
from extract_scene3d_board_edges import measure_edge

OUT = _resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v015_joint')
SURVEY = _resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/survey')
OLD = _resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v014_main')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_record(image, pts, installation, shot):
    reference = load(OLD/'real_observations.json')['frames'][0]
    out = {k: copy.deepcopy(reference[k]) for k in ['source_id','media_track_id','source_path','source_sha256','time_base','stream_index']}
    out.update(id=f'{installation}_{pts}', installation_id=installation, source_pts=pts,
               pixel_frame_id=f"{reference['media_track_id']}:pts:{pts}:1920x1080",
               source_seconds=pts/15360, image_path=str(image), image_sha256=digest(image),
               image_size_wh=[1920,1080], principal_point_px=[960.,540.], fy_over_fx=1.,
               distortion_opencv=[0.,0.,0.,0.,0.], focal_bounds_px=[450.,30000.], shot_id=shot,
               intrinsic_assumptions='Square pixels, centred principal point, zero skew; conditional optical model, not independently measured')
    return out


def guided_rail(image, guide, radius=10, step=12):
    """Upper dark/blue rail edge, with downward light-to-dark transition."""
    guide=np.asarray(guide,float)
    x=np.arange(max(8,guide[0,0]), min(image.shape[1]-8,guide[-1,0]),step)
    y=np.interp(x,guide[:,0],guide[:,1])
    gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.)
    offsets=np.arange(-radius,radius+.1,.5)
    q=np.stack(np.broadcast_arrays(x[:,None],y[:,None]+offsets),axis=-1)
    grad=sample(gray,q+[0,-2])-sample(gray,q+[0,2])
    k=grad.argmax(axis=1); strength=grad[np.arange(len(k)),k]
    keep=(k>1)&(k<len(offsets)-2)&(strength>8)
    pts=np.c_[x,y+offsets[k]]
    return {'pixels_uv':pts[keep].tolist(),'sigma_px':3.,
            'method':'Image-only signed luminance-gradient maxima around visually selected blue rail guide',
            'search_guide_image_px':guide.tolist(),'search_radius_px':radius,
            'edge_definition':'visible upper edge of blue/dark board rail; finite rail width/systematic physical interface uncertainty retained'}


def add_observation(rows, frame, fid, target, measurement, sigma=None, provenance=None):
    data=copy.deepcopy(measurement)
    data.update(id=f"{frame['id']}_{fid}",frame_id=frame['id'],physical_group_id=fid,
                target=target,type='curve',use='fit',correlation_group_id=f"{frame['shot_id']}_{fid}",
                identity_evidence=provenance or 'assistant visual association across confirmed reference imagery',
                source_frame={k:frame[k] for k in ['source_id','source_pts','time_base','image_path','image_sha256']},
                measurement_origin='direct_image_measurement', synthetic=False)
    if sigma is not None:data['sigma_px']=sigma
    rows.append(data)


def preview(frame, obs, path):
    im=cv2.imread(frame['image_path'])
    for o in obs:
        color=(30,240,250) if o['target']['kind']!='board_side' else (230,80,250)
        pts=np.asarray(o['pixels_uv'])
        for p in pts:cv2.circle(im,tuple(np.rint(p).astype(int)),2,color,-1,cv2.LINE_AA)
        p=np.median(pts,axis=0).astype(int)
        pos=(max(10,min(1750,p[0])),max(25,p[1]-15))
        cv2.putText(im,o['physical_group_id'],pos,cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(im,o['physical_group_id'],pos,cv2.FONT_HERSHEY_SIMPLEX,.65,color,1,cv2.LINE_AA)
    cv2.imwrite(str(path),im)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    high=frame_record(SURVEY/'survey_3660.png',56218880,'women_sp_high','high_3240_3709')
    camera=load(_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v014_high_preview/projection.json'))['camera']
    high.update(initial_rvec_world_to_camera=camera['rvec_world_to_camera'],initial_focal_px=camera['K_px'][0][0])
    obs=[]
    identity='outputs/rink_model/ankara_2026_shared/v0.13-women-sp/STAGE1_REVIEW.md: user confirmed objects and displayed contours; not pixel-level precision'
    old=load(SURVEY/'observations.json')
    for o in old['features']:
        fid=o['physical_id']
        target=None
        if fid.startswith(('R1_','R2_','R3_','R4_','F2_','F3_')):
            target={'kind':'circle_outline','feature_id':fid[:2]};group=fid[:2]
        elif fid in ['B1','B2','C','goal_line_right']:
            target={'kind':'line_segment','feature_id':fid};group=fid
        elif fid in ['A1_left_arc','A1_lower_arc']:
            target={'kind':'circle_outline','feature_id':'A1'};group=fid
        elif fid in ['boundary_left_visible','boundary_top_visible']:
            target={'kind':'board_side','side':'left_end' if fid=='boundary_left_visible' else 'upper_long_side','edge':'contact'};group=fid
        elif fid in ['boundary_top_left_curve','boundary_top_right_curve']:
            target={'kind':'board_corner','corner':'upper_left' if fid=='boundary_top_left_curve' else 'upper_right','edge':'contact'};group=fid
        if target:
            add_observation(obs,high,group,target,{'pixels_uv':o['points_px'],'method':'reused source image contour; original measurement method/provenance retained in source JSON','original_observation':o},sigma=o['uncertainty_px'],provenance=identity)
    image=cv2.imread(high['image_path'])
    guides=[('left_end',[[15,273],[120,248],[250,218],[345,200]]),
            ('upper_long_side',[[560,198],[800,223],[1050,249],[1250,282],[1450,319],[1600,350]])]
    for side,guide in guides:
        add_observation(obs,high,side+'_rail_top',{'kind':'board_side','side':side,'edge':'top'},guided_rail(image,guide))
    save(OUT/'high_seed_observations.json',{'frames':[high],'observations':obs})
    preview(high,obs,OUT/'high_measured_correspondences.png')

    image_path=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.13-women-sp/evidence/competition/main_left_S032_71116800.png')
    frame=frame_record(image_path,71116800,'women_sp_main','S032')
    im=cv2.imread(str(image_path));rows=[]
    ring=extract_ring(im,{'feature_id':'F1','seed_ellipse_cx_cy_a_b_angle':[920,800,375,42,0],
                         'exclude_boxes':[[752,439,1033,838]],'min_gradient_strength':3.})
    add_observation(rows,frame,'F1',{'kind':'circle_outline','feature_id':'F1'},ring)
    for edge,label,roi in [('blue_rail_upper','top',[350,455,1700,505]),('yellow_kickplate_lower','contact',[340,620,1700,690])]:
        measured=measure_edge(im,roi=roi,edge=edge,exclude_rectangles=[[760,430,1030,840]])
        measured.update(pixels_uv=measured['pixel_points'],sigma_px=measured['clarity']['median_uncertainty_px'])
        add_observation(rows,frame,'left_end_'+label,{'kind':'board_side','side':'left_end','edge':label},measured)
    save(OUT/'additional_seeds.json',{'frames':[frame],'observations':rows})
    preview(frame,rows,OUT/'S032_measured_correspondences.png')
    print(json.dumps({'high_objects':len(obs),'new_S032_objects':len(rows),'output':str(OUT)}))


if __name__=='__main__':main()
