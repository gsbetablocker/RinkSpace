"""Fit one real confirmed camera identity, with the companion's shared-C solver.

Only explicitly designated FIT objects choose cameras. Whole-object checks are
loaded for reporting after selection. Current v0.14 geometry is never changed.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
from fractions import Fraction
import cv2
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rinkspace.calibration_companion import Scene,Camera,KeyframeProblem,look_at_rvec
from rinkspace.calibration_companion.scene_camera import nearest_polyline_vectors
OUT=_resolve_asset(ROOT, ROOT/'outputs/calibration/ankara_2026_v014_main')
SCENE=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def board_srange(scene,side):
    found=[]
    for p in scene.panels:
        a,b=scene.board_vertices[p['bottom_vertex_indices']]
        if (side=='left_end' and abs(a[0]+100)<1e-7 and abs(b[0]+100)<1e-7) or (side=='upper_long_side' and abs(a[1]-50)<1e-7 and abs(b[1]-50)<1e-7):found.append(p['s_range'])
    return [min(x[0] for x in found),max(x[1] for x in found)]

def make_observations(scene):
    obs=load(OUT/'ice_evidence/observations.json')
    for d in load(OUT/'board_evidence/selected_board_observations.json'):
        pts=d['frame_metadata']['source_pts'];fid=f'main_{pts}'
        for o in d['objects']:
            height=scene.data['parameters']['board_height'] if o['edge_type']=='blue_rail_upper' else 0.
            obs.append({'id':f"{fid}_{o['object_id']}",'frame_id':fid,'physical_group_id':o['object_id'],
                'type':'curve','target':{'kind':'board_curve','s_range':board_srange(scene,o['qualitative_side']),'height':height},
                'use':'check','surface_class':'board_elevated' if height else 'ice_board_contact',
                'pixels_uv':o['pixel_points'],'sigma_px':o['clarity']['median_uncertainty_px'],
                'identity_status':o['identity_status'],'identity_evidence':o['identity_evidence'],
                'xyz_status':'fixed scene board height/edge schematic assumptions; no independently measured XYZ',
                'physical_interface':o['physical_interface'],'physical_geometry_caveat':o['physical_geometry_caveat'],
                'measurement_source':str((OUT/'board_evidence/selected_board_observations.json').relative_to(ROOT)),
                'source_frame':d['frame_metadata']})
    return obs

def align_z_to(v):
    v=np.asarray(v,float);v/=np.linalg.norm(v);axis=np.cross([0.,0.,1.],v);norm=np.linalg.norm(axis)
    return np.eye(3) if norm<1e-12 else cv2.Rodrigues(axis/norm*np.arctan2(norm,v[2]))[0]

def initialize_frame(scene,f,obs,c,zoom_factor):
    ring=next(o for o in obs if o['frame_id']==f['id'] and o['use']=='fit' and o['target']['kind']=='circle_outline')
    feature=scene.features[ring['target']['feature_id']];target=np.array(feature['center_xyz']);ellipse=ring['ellipse_cv2_image_only']
    # Image ellipse centre only aims the initial view. No ellipse centre appears
    # as a point observation or in the final residual function.
    uv=np.asarray(ellipse[0]);a=max(ellipse[1])/2
    focal=a*np.linalg.norm(target-c)/feature['radius']*zoom_factor
    focal=np.clip(focal,*f['focal_bounds_px'])
    base=cv2.Rodrigues(look_at_rvec(c,target))[0]
    ray=np.r_[(uv-np.array(f['principal_point_px']))/focal,1.]
    R=align_z_to(ray)@base
    return cv2.Rodrigues(R)[0].ravel().tolist(),float(focal)

def make_problem(scene,frames,obs,c,zoom_factor=1.):
    frames=copy.deepcopy(frames)
    for f in frames:
        r,focal=initialize_frame(scene,f,obs,np.array(c,float),zoom_factor)
        f['initial_rvec_world_to_camera']=r;f['initial_focal_px']=focal
    return {'schema_version':'rinkspace.real-keyframe-problem.v1','synthetic':False,'scene_sha256':scene.sha256,
        'geometry_mode':'fixed_scene_conditional','curve_samples':256,
        'installations':[{'id':'oxXu1avhrvs_camera_main_long_side_observed_keyframes','physical_camera_id':'camera_main_long_side',
            'initial_center_xyz':c,'fit_center':True,'center_bounds_xyz':[[-100.,-500.,.25],[100.,-50.0001,200.]],
            'bounds_rationale':'User-confirmed lower long side outside y=-50; x covers whole rink length, y extends five rink widths from origin, height twice rink width. Broad computational guardrails, not measured venue coordinates; bound hits invalidate finite position claims.',
            'user_identity_evidence':'configs/camera_classification/ankara_2026_user_confirmed.json',
            'no_center_prior':True}],
        'frames':frames,'observations':obs,
        'focal_initialization_rationale':'f0 = image-only observed outer ellipse major semiaxis / schematic circle radius * hypothesized circle-centre distance. Multiplied by independent zoom factors. This uses fit curve only; centre used for initial aiming only.',
        'initial_C_rationale':'Hypotheses on the confirmed lower long side with different offsets/heights; not copied from synthetic companion cameras or a measured camera location.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--starts',type=int,default=None);ap.add_argument('--max-nfev',type=int,default=500);ap.add_argument('--single',type=int);ap.add_argument('--tag',default='pixel_reviewed');args=ap.parse_args()
    scene=Scene(SCENE);obs=make_observations(scene);frames=load(OUT/'source_selection/recommended_frames.json')
    for f in frames:
        assert isinstance(f['source_pts'],int) and f['time_base']=='1/15360'
        assert abs(float(Fraction(f['source_pts'])*Fraction(f['time_base']))-f['source_seconds'])<1e-8
        assert sha(f['image_path'])==f['image_sha256']
        assert cv2.imread(f['image_path']).shape[:2]==(1080,1920)
        f['id']=f"main_{f['source_pts']}";f['installation_id']='oxXu1avhrvs_camera_main_long_side_observed_keyframes'
        f['image_size_wh']=[1920,1080];f['principal_point_px']=[959.5,539.5];f['fy_over_fx']=1.
        f['distortion_opencv']=[0.,0.,0.,0.,0.]
        f['focal_bounds_px']=[float(960/np.tan(np.deg2rad(65))),float(960/np.tan(np.deg2rad(1)))]
        f['intrinsic_assumptions']='native 1920x1080 square pixels, principal point at geometric image centre (959.5,539.5); skew zero; lens distortion fixed zero, all unverified optical assumptions'
        f['focal_bounds_rationale']='horizontal field-of-view range 2 to 130 degrees is a numerical guardrail, not a measured lens specification'
    # These alternatives test plausible side/height/distance and independent
    # focal hypotheses. Their number is a numerical experiment, not task scope.
    starts=[([0.,-65.,6.],1.),([-25.,-85.,10.],.8),([25.,-85.,10.],1.2),([0.,-130.,16.],1.),([-40.,-65.,6.],1.25),([40.,-130.,16.],.75)]
    if args.starts is not None:starts=starts[:args.starts]
    indexed=list(enumerate(starts)) if args.single is None else [(args.single,starts[args.single])]
    dest=OUT/'fits'/args.tag;dest.mkdir(exist_ok=True,parents=True)
    save(OUT/'real_observations.json',{'scene_path':str(SCENE),'scene_sha256':scene.sha256,'frames':frames,'observations':obs,'split_path':'observation_split_pre_fit.json'})
    save(dest/'initialization_protocol.json',{'starts':starts,'selection':'lowest FIT robust cost among geometrically valid starts; no check metric chooses solution','scene_sha256':scene.sha256,'script_sha256':sha(__file__)})
    for i,(c,zf) in indexed:
        p=make_problem(scene,frames,obs,c,zf);save(dest/f'problem_start_{i}.json',p)
        start=time.monotonic();print(f'start {i}: C={c}, f={[round(f["initial_focal_px"]) for f in p["frames"]]}',flush=True)
        try:
            problem=KeyframeProblem(scene,p);result=problem.solve(args.max_nfev)
            result['start_id']=i;result['initial_center_xyz']=c;result['initial_zoom_factor']=zf;result['elapsed_seconds']=time.monotonic()-start
            result['real_observation_extractor']='scripts/prepare_scene3d_calibration_observations.py and scripts/extract_scene3d_board_edges.py'
            result['limitations']=[x for x in result['limitations'] if 'No automatic image detection' not in x]+['Measured image edges were computed inside visually selected search regions; no continuous tracking was run.']
            save(dest/f'fit_start_{i}.json',result)
            print(json.dumps({'start':i,'C':result['installations'][0]['center_xyz'],'cost':result['optimizer']['cost'],'nfev':result['optimizer']['nfev'],'success':result['optimizer']['success'],'f':[f['camera']['K_px'][0][0] for f in result['frames']],'elapsed':result['elapsed_seconds']}),flush=True)
        except Exception as e:
            save(dest/f'fit_start_{i}_failure.json',{'error':str(e),'start_id':i,'initial_C':c});raise

if __name__=='__main__':main()
