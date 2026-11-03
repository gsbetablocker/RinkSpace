"""Source-linked preparation view using the frozen secondary native camera.

This adds a single preparation exposure, never changes a programme camera.
Only pan, tilt and log focal length are estimated from visible print contours.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .rebuild_prepare import load,digest
from .fullbody_round import write_json


class PreparationCamera:
    def __init__(self,core,rig,ptz):
        self.core=core;self.rig=np.asarray(rig).copy();self.ptz=np.asarray(ptz).copy()

    def project(self,xyz,ptz=None,*,allow_outside=False):
        controls=np.tile(self.ptz if ptz is None else ptz,(8,1))
        p=self.core.project_motion(self.rig,controls,-3.,1.,1080,0.,np.asarray(xyz).reshape(-1,3))
        if np.any(p['status']) and not allow_outside:raise ValueError('Preparation native projection failed')
        return p['uv']


def geometry(root):
    base=Path(root)/'fullbody_1102'
    packet=load(base/'review_scene.npz')['xyz']
    info=json.loads((base/'review_scene.json').read_text(encoding='utf8'))
    return packet,info['cuts']


def intersection(root):
    xyz,cuts=geometry(root)
    lines=[]
    for entity in ('entity/ice.B2','entity/ice.house_connection_F2_R2'):
        cut=next(c for c in cuts if c['entity_id']==entity)
        line=xyz[cut['begin']:cut['end']];lines.append(line[[0,-1]])
    a,b=lines;v=a[1,:2]-a[0,:2];w=b[1,:2]-b[0,:2]
    uv=np.linalg.solve(np.c_[v,-w],b[0,:2]-a[0,:2])
    if np.any((uv<0)|(uv>1)):raise ValueError('Intersection outside actual finite supports')
    return np.r_[a[0,:2]+uv[0]*v,0.]


def fit_camera(project,root,out,core):
    import cv2
    from PIL import Image,ImageDraw
    project,root,out=map(Path,(project,root,out))
    state_path=project/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.npz'
    state=load(state_path);xyz,cuts=geometry(root)
    camera=PreparationCamera(core,state['rig'][1],[.3,0.,np.log(6000.)])
    image=np.asarray(Image.open(out/'preparation_matched_source.png').convert('RGB'))
    # Source colour boundaries: masks restricted to the two visible MEIWA bars.
    # Their fitted role is explicit; ice and other advertisement contours check it.
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);groups=[]
    definitions=[('blue',98484,98990,(400,337,541,382),(85,70,45),(120,255,255)),
                 ('green',98990,99490,(397,400,543,442),(35,55,35),(95,255,255))]
    for name,begin,end,box,low,high in definitions:
        x0,y0,x1,y1=box
        mask=cv2.inRange(hsv[y0:y1,x0:x1],np.array(low),np.array(high))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea).reshape(-1,2)+[x0,y0]
        pts=xyz[begin:end]
        if not any(c['begin']==begin and c['end']==end and c['entity_id']=='entity/Ad5' for c in cuts):
            raise ValueError('Frozen MEIWA contour identity changed')
        # Native material corners seed the fit; then all source contour samples
        # retain two pixel components with a local station on the native contour.
        corners=[]
        for xx,zz in ((pts[:,0].min(),pts[:,2].max()),(pts[:,0].max(),pts[:,2].max()),
                      (pts[:,0].max(),pts[:,2].min()),(pts[:,0].min(),pts[:,2].min())):
            corners.append(pts[np.argmin((pts[:,0]-xx)**2+(pts[:,2]-zz)**2)])
        rect=cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
        rect=rect[np.argsort(rect[:,1])];top=rect[:2][np.argsort(rect[:2,0])];bottom=rect[2:][np.argsort(rect[2:,0])]
        observed=np.array([top[0],top[1],bottom[1],bottom[0]])
        groups.append(dict(name=name,xyz=pts,source_uv=contour,corners_xyz=np.array(corners),corners_uv=observed))
    seedxyz=np.concatenate([g['corners_xyz'] for g in groups]);seeduv=np.concatenate([g['corners_uv'] for g in groups])
    seed=least_squares(lambda x:(camera.project(seedxyz,x)-seeduv).ravel(),camera.ptz,x_scale=[.1,.1,1.],ftol=1e-12,xtol=1e-12,gtol=1e-9)
    def residual(x):
        rows=[]
        for g in groups:
            uv=camera.project(g['xyz'],x);_,i=cKDTree(uv).query(g['source_uv'])
            rows.append((uv[i]-g['source_uv']).ravel())
        return np.concatenate(rows)
    fitted=least_squares(residual,seed.x,x_scale=[.1,.1,1.],ftol=1e-11,xtol=1e-11,gtol=1e-8)
    camera.ptz=fitted.x
    metrics={}
    for g in groups:
        uv=camera.project(g['xyz']);dist,_=cKDTree(uv).query(g['source_uv'])
        metrics[g['name']]=dict(source_points=len(dist),median_px=float(np.median(dist)),p95_px=float(np.quantile(dist,.95)),max_px=float(dist.max()))
    point=intersection(root);point_uv=camera.project(point[None])[0]
    pose=core.evaluate_motion_camera(camera.rig,np.tile(camera.ptz,(8,1)),-3.,1.,[0.])
    np.savez_compressed(out/'preparation_camera.npz',rig=camera.rig,ptz=camera.ptz,intersection_scene=point,
        rotation=pose['rotation'][0],centre_scene=pose['centre'][0],source_pts=np.array(255488),time_base=np.array([1,12800]))
    record=dict(schema='rinkspace.preparation-camera.v1',source_pts=255488,source_seconds=19.96,
        source_image=dict(path=str(out/'preparation_matched_source.png'),sha256=digest(out/'preparation_matched_source.png')),
        frozen_state=dict(path=str(state_path),sha256=digest(state_path)),rig_index=1,rig_name='secondary_long_side',
        geometry_sha256=digest(root/'fullbody_1102/review_scene.npz'),variables=['pan','tilt','log_f'],
        ptz=camera.ptz.tolist(),fitted_source_groups=metrics,
        source_contours={g['name']:g['source_uv'].tolist() for g in groups},
        intersection_scene=point.tolist(),intersection_source_uv=point_uv.tolist(),
        native_instantaneous_values=pose['values'][0].tolist(),
        assumptions='Existing fixed secondary rig, optics, actual scene; constant PTZ during this exposure; new shot does not modify the nine programme shots.',
        stop=dict(message=fitted.message,cost=float(fitted.cost),optimality=float(fitted.optimality)))
    write_json(out/'preparation_camera.json',record)
    overlay=Image.fromarray(image);draw=ImageDraw.Draw(overlay)
    alluv=camera.project(xyz,allow_outside=True)
    for cut in cuts:
        u=alluv[cut['begin']:cut['end']]
        color=(0,220,250) if cut['layer']=='ice' else (255,185,0)
        width=1
        if cut['entity_id']=='entity/ice.B2':color=(255,70,70);width=3
        if cut['entity_id']=='entity/ice.house_connection_F2_R2':color=(70,255,90);width=3
        for a,b in zip(u[:-1],u[1:]):
            if np.isfinite([a,b]).all() and np.all(np.abs([a,b])<100000):draw.line([tuple(a),tuple(b)],fill=color,width=width)
    xx,yy=point_uv;draw.ellipse((xx-9,yy-9,xx+9,yy+9),outline=(255,255,0),width=3)
    draw.text((30,30),'SOURCE 19.960s | frozen secondary rig | fit: MEIWA bars | ice: check only',fill='white',stroke_width=2,stroke_fill='black')
    draw.text((xx+13,yy+10),'B2 x F2-R2',fill='yellow',stroke_width=2,stroke_fill='black')
    overlay.save(out/'preparation_geometry_overlay.png')
    return camera,record


def fit_pose(reference,camera,out):
    """Fit lower-body candidates to visible shoe proxies, retaining side ambiguity."""
    from scipy.spatial.transform import Rotation
    from .fullbody_state import matrices
    from PIL import Image,ImageDraw
    out=Path(out);anchor=intersection(reference.root)*60.
    original=reference.at([0.]);donor=reference.donor_R.copy()
    # Local anatomy, not independently movable joints or boots.
    dofs=[(j,a) for j in (1,2,7,8) for a in range(3)]+[(4,0),(5,0)]
    base=donor.copy()
    for j in (1,2,4,5,7,8,10,11):base[j-1]=np.eye(3)
    heading=float(reference.heading[0]);q0=original['Q'][0]
    def state(x):
        R=base.copy();v=np.zeros((21,3))
        for value,(j,a) in zip(x[6:],dofs):v[j-1,a]=value
        R=R@matrices(v)
        return reference.world(dict(P=(np.r_[anchor[:2],.91]+x[:3])[None],
            Q=(q0@matrices(x[3:6]))[None],local_R=R[None],times=np.array([0.])))
    candidates=[]
    for front in (0,1):
        rear=1-front
        def residual(x):
            s=state(x);g=s['gear'][0]
            # Shell cuff centre and visible toe/heel proxies are not anatomical
            # joint points. Their coarse image scale records that distinction.
            proxies=np.array([g[front,[4,5,11,12]].mean(0),g[rear,[4,5,11,12]].mean(0),
                g[front,[1,2,8,9]].mean(0),g[rear,[1,2,8,9]].mean(0),g[front,[0,6,7,13]].mean(0)])
            uv=camera.project(proxies/60.)
            source=np.array([[830,815],[775,792],[800,884],[750,874],[865,877]])
            pixels=(uv-source)/12.
            blade=g[front,14:];behind=g[rear,14:]
            # A is nearly flat in the photo; no loading/physical contact label.
            physical=np.r_[blade[[0,16,17,33],2]/.025,np.minimum(behind[:,2],0)/.025]
            # The visible A knee contour is only a broad proxy. B is occluded.
            knee=camera.project(s['native'][0,4+front][None]/60.)[0]
            weak_knee=np.maximum(abs(knee-[810,730])-[25,25],0)/20.
            # Preserve an upright root and regularize unresolved pose/depth.
            up=s['Q'][0,:,1]
            reg=np.r_[x[:2]/.4,x[2]/.2,up[:2]/.25,x[6:]/1.2]
            knees=np.array([x[-2],x[-1]])
            hinge=np.r_[np.minimum(knees,0),np.maximum(knees-1.6,0)]/.1
            return np.r_[pixels.ravel(),physical,weak_knee,reg,hinge]
        x0=np.zeros(6+len(dofs));x0[-2:]=.1
        fit=least_squares(residual,x0,x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-7)
        s=state(fit.x)
        candidates.append(dict(front_native_foot=front,cost=float(fit.cost),x=fit.x,state=s,optimality=float(fit.optimality),message=fit.message))
    # A single image cannot resolve anatomical sides. Both candidates remain;
    # select the smaller pose-deviation solution for a conditional initialization.
    selected=min(candidates,key=lambda c:c['cost'])
    s=selected['state'];a=selected['front_native_foot']
    np.savez_compressed(out/'preparation_pose.npz',local_R=s['local_R'][0],P=s['P'][0],Q=s['Q'][0],
        selected_front_native_foot=np.array(a),candidate_local_R=np.stack([c['state']['local_R'][0] for c in candidates]),
        candidate_P=np.stack([c['state']['P'][0] for c in candidates]),candidate_Q=np.stack([c['state']['Q'][0] for c in candidates]))
    report=dict(schema='rinkspace.preparation-lower-pose.v1',source_pts=255488,source_seconds=19.96,
        camera_sha256=digest(out/'preparation_camera.npz'),source_review_sha256=digest(out/'preparation_observation_review.json'),
        same_body_and_footwear=True,visible_proxies='Cuff/outer toe and A heel; 12px engineering scale, not anatomical joints; only A weak knee contour box',
        selected_front_native_foot=a,side_status='conditional numerical hypothesis; image alone does not identify anatomical left/right',
        candidates=[{k:v for k,v in c.items() if k not in ('state','x')} for c in candidates],
        selected_P_solver_m=s['P'][0].tolist(),intersection_solver_m=anchor.tolist(),
        transfer='Lower local rotations only, smoothly relaxed during cropped opening; upper body remains driven by S01 observations.')
    write_json(out/'preparation_pose.json',report)
    image=Image.open(out/'preparation_matched_source.png').convert('RGB');draw=ImageDraw.Draw(image)
    uv=camera.project(s['native'][0]/60.);gear=camera.project(s['gear'][0].reshape(-1,3)/60.).reshape(2,48,2)
    for foot,(hip,knee,ankle) in enumerate(((1,4,7),(2,5,8))):
        color=(255,100,50) if foot==a else (20,230,255)
        draw.line([tuple(uv[j]) for j in (hip,knee,ankle)],fill=color,width=3)
        for ids in ([0,1,2,3,4,5,6,0],[7,8,9,10,11,12,13,7],list(range(14,31))):
            draw.line([tuple(gear[foot,j]) for j in ids],fill=color,width=2)
    draw.text((30,30),'Preparation lower-pose hypothesis | orange=A | cyan=B | same registered boots',fill='white',stroke_width=2,stroke_fill='black')
    image.save(out/'preparation_pose_overlay.png')
    return report
