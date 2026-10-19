"""Compare full source-fitted and reexpressed bodies against their bound rink."""
import argparse,json,hashlib
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULT=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--body',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'events/ankara_2026/diagnostics/body_source_refit_boundary_20261007.json')
    args=parser.parse_args()
    if (args.scene is None)!=(args.body is None):
        parser.error('--scene and --body must be given together')
    if args.scene is None:
        render=json.loads((RESULT/'aligned_complete_20261007_478afaf77615/render.json').read_text(encoding='utf8'))
        scene_path=Path(render['inputs'][2]['path'])
    else:scene_path=args.scene
    scene=json.loads(scene_path.read_text(encoding='utf8'))
    report=dict(schema='rinkspace.full-body-source-refit-boundary-review.v1',
        scene=str(scene_path),scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        origin='signed distance to actual current scene ice_boundary; each body uses its own saved working scale',
        boundary_was_not_used_to_fit_clip_or_hide_trajectory=True,
        G_is_pelvis_vertical_projection_not_blade_contact=True,
        all_exposures_evaluated=True,whole_video_accuracy_accepted=False,results={})
    bodies=[('current_full_source_refit',args.body)] if args.body is not None else [('reexpressed','aligned_complete_20261007_478afaf77615/actual_body_root_camera_reexpressed.npz'),
                          ('full_source_refit','pelvis_source_refit_478af_20261007/actual_body_root_fit.npz')]
    for name,relative in bodies:
        path=RESULT/relative;z=np.load(path,allow_pickle=False)
        poly=np.array(scene['ice_boundary']['points_xyz'],np.float32)[:,:2]*float(z['metres_per_scene_unit'])
        signed=np.array([cv2.pointPolygonTest(poly,tuple(map(float,p[:2])),True) for p in z['P_world_m']])
        result=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actual_exposures=len(signed),working_metres_per_scene_unit=float(z['metres_per_scene_unit']),
            outside_exposures=int((signed<0).sum()),minimum_signed_margin_working_m=float(signed.min()),shots=[])
        for cut in np.unique(z['shot_index']):
            ix=np.flatnonzero(z['shot_index']==cut);worst=ix[np.argmin(signed[ix])]
            errors=np.linalg.norm(z['COCO17_native_uv'][ix][:,[11,12]]-z['source_COCO17_native_uv'][ix][:,[11,12]],axis=2)
            valid=z['source_COCO17_qualified'][ix][:,[11,12]]
            result['shots'].append(dict(shot_index=int(cut),actual_exposures=len(ix),
                outside_exposures=int((signed[ix]<0).sum()),minimum_signed_margin_working_m=float(signed[worst]),
                worst_pts=int(z['actual_pts'][worst]),source_qualified_hip_pixels_mean_error=float(errors[valid].mean()) if valid.any() else None))
        report['results'][name]=result
    path=args.output
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(path)

if __name__=='__main__':main()
