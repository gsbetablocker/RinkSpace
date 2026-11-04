"""Camera-frame model diagnostics and source overlays, with no pose repair."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .exmr_prior import runtime,load,write_json,digest,Progress
from .stage_quality import body_groups

COCO17=('nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle')
EDGES=((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def camera_to_native(xyz,camera25):
    import cv2
    result=[]
    for points,camera in zip(xyz,camera25):
        K=np.array([[np.exp(camera[6]),0,camera[8]],[0,np.exp(camera[7]),camera[9]],[0,0,1.]])
        uv,_=cv2.projectPoints(np.asarray(points,float),np.zeros(3),np.zeros(3),K,camera[11:25])
        result.append(uv.reshape(-1,2))
    return np.stack(result)


def camera_models(data,output,base,labels):
    torch=runtime(data)
    from exmr.body_models.smplx_model import SmplxModel
    output,base=Path(output),Path(base);d=load(output/'prepared.npz');log=Progress(output,'camera_review')
    model=SmplxModel(Path(data)/'runtime_tools/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz').cuda().eval()
    ids=[model.joint_names.index(x) for x in COCO17]
    native=load(base/'action_integration_1103/source_cache/spin_s05.native50.npz')
    clock=native.get('pts',native.get('source_pts'));lookup={int(p):i for i,p in enumerate(clock)}
    rows=np.array([lookup[int(p)] for p in d['source_pts']]);observed=native['body_uv'][rows];use=native['body_use'][rows]
    report=dict(scope='raw camera-coordinate SMPL-X at the original 25Hz exposures, before any world solve',
        projection='native Brown at midpoint; no rolling/exposure compensation in this diagnostic',
        source_time_base=[1,12800],comparators={},raw_network_is_truth=False)
    for label in labels:
        path=base/'action_integration_1103/spin_s05/prior/camera_prior.npz' if label=='gvhmr' else output/(label+'.npz')
        a=load(path);np.testing.assert_array_equal(a['source_pts'],d['source_pts'])
        joints=[]
        for first in range(0,len(a['source_pts']),32):
            stop=min(first+32,len(a['source_pts']));N=stop-first
            params={k:torch.from_numpy(a[k][first:stop]).float().cuda() for k in ('body_pose','betas','global_orient','transl')}
            for k in ('left_hand_pose','right_hand_pose'):
                params[k]=torch.from_numpy(a[k][first:stop]).float().cuda() if k in a else torch.zeros((N,45),device='cuda')
            with torch.inference_mode():j=model.joints(**params).cpu().numpy()
            joints.append(j)
        joints=np.concatenate(joints);uv=camera_to_native(joints,d['camera25'])
        angular=np.rad2deg(Rotation.from_rotvec(a['global_orient'][:-1]).inv().__mul__(Rotation.from_rotvec(a['global_orient'][1:])).magnitude())
        coco=uv[:,ids];sec=a['source_pts']/12800.;core=(sec>=88)&(sec<=92)
        pelvis_camera=joints[:,0]
        centres=load(base/'action_integration_1103/spin_s05/prepared.npz')['camera_centre_m']
        world_pelvis=centres+np.einsum('nji,nj->ni',d['R_w2c'],pelvis_camera)
        np.savez_compressed(output/(label+'_camera_diagnostics.npz'),source_pts=a['source_pts'],
            coco17_native_uv=coco,joints76_native_uv=uv,joints76_camera=joints,
            raw_pelvis_world=world_pelvis,world_projection_status=np.array('diagnostic raw SMPL-X scale, not measured trajectory'))
        angles=angular[(sec[:-1]>=88)&(sec[1:]<=92)]
        report['comparators'][label]=dict(prior=str(path),prior_sha256=digest(path),
            core_body_groups=body_groups(coco[core],observed[core],use[core],a['source_pts'][core]),
            full_root_angle_step_max_deg=float(angular.max()),core_root_angle_step_max_deg=float(angles.max()),
            root_jump_source_seconds=sec[:-1][angular>90].tolist(),
            raw_core_pelvis_xy_span_m=np.ptp(world_pelvis[core,:2],axis=0).tolist(),
            raw_core_pelvis_xy_path_m=float(np.linalg.norm(np.diff(world_pelvis[core,:2],axis=0),axis=1).sum()),
            model_native_shape=True,shared_shape_replacement=False,
            foot_or_shoe_contact_truth=False)
        log('raw_camera_candidate',label=label,core_root_angle_step_max_deg=float(angles.max()))
    write_json(output/'camera_comparison.json',report)
    return report


def source_overlays(output,base,labels,seconds):
    from PIL import Image,ImageDraw,ImageFont
    from .action_clip_input import _official_frames
    output,base=Path(output),Path(base)
    d=load(output/'prepared.npz');contract=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))
    pts=d['source_pts'];selected=[int(pts[np.argmin(abs(pts-s*12800))]) for s in seconds]
    indices={int(p):i for i,p in enumerate(pts)}
    arrays={label:load(output/(label+'_camera_diagnostics.npz')) for label in labels}
    prepared=load(base/'action_integration_1103/spin_s05/prepared.npz');frames=output/('camera_review' if labels==['gvhmr'] else 'camera_review_models');frames.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    record=[]
    colors=['#ff6666','#66deff','#b1f06b','#f9ca74']
    for p,bgr in _official_frames(contract['source']['path'],sorted(set(selected))):
        i=indices[p];box=prepared['roi_native_xyxy'][i];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        height=max(540.,(box[3]-box[1])*1.20);width=height*.7
        crop=(int(cx-width/2),int(cy-height/2),int(cx+width/2),int(cy+height/2))
        image=Image.fromarray(bgr[:,:,::-1]);panels=[]
        for j,label in enumerate(['source',*labels]):
            panel=image.crop(crop).resize((420,600));draw=ImageDraw.Draw(panel)
            if label!='source':
                uv=arrays[label]['coco17_native_uv'][i]
                xy=(uv-np.array(crop[:2]))*np.array([420/(crop[2]-crop[0]),600/(crop[3]-crop[1])])
                for a,b in EDGES:
                    if np.isfinite(xy[[a,b]]).all():draw.line([tuple(xy[a]),tuple(xy[b])],fill=colors[(j-1)%len(colors)],width=3)
                for x,y in xy:
                    if np.isfinite([x,y]).all():draw.ellipse((x-3,y-3,x+3,y+3),fill=colors[(j-1)%len(colors)])
            draw.rectangle((0,0,420,28),fill='#181818');draw.text((6,4),f'{label} | {p/12800:.2f}s',font=font,fill='white');panels.append(panel)
        canvas=Image.new('RGB',(420*len(panels),600))
        for j,panel in enumerate(panels):canvas.paste(panel,(420*j,0))
        path=frames/f'{p}.png';canvas.save(path)
        record.append(dict(source_pts=p,source_seconds=p/12800,path=str(path),crop_native_xyxy=crop))
    write_json(frames/'source_frames.json',dict(projection='camera model direct native Brown at midpoint; visualization only',frames=record))
    return record


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--base-root',required=True);p.add_argument('--labels',nargs='+',default=['gvhmr','exmr_prefoot','exmr_final'])
    a=p.parse_args(argv);camera_models(a.data_root,a.output,a.base_root,a.labels)
    source_overlays(a.output,a.base_root,a.labels,[87.52,88.08,88.12,88.56,89.28,89.88,90.56,91.48,92.48])

if __name__=='__main__':main()