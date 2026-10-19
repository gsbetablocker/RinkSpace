"""Cross-view concentric-ring diagnostics from direct native material readouts.
A per-frame projective map is a diagnostic nuisance, not a calibrated physical camera.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import cv2


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_frame(path, target='ice.R3:'):
    path=Path(path);d=json.loads(path.read_text(encoding='utf-8'))
    readings=[]
    for feature in d['features']:
        if not feature['id'].startswith(target):continue
        for j,r in enumerate(feature.get('source_circle_fit',{}).get('direct_readouts',[])):
            ev=r['source_evidence']
            if not ev.get('actual_source_support_outside_person_and_broadcast_masks',False):
                raise ValueError('Unverified foreground support')
            readings.append(dict(uv=r['uv'],sigma=r['sigma_px'],ring=feature['id'].split(':')[-1],
                                 seed_theta=r['theta'],source_index=j,source_feature=feature['id'],
                                 source_evidence=ev))
    return d,readings


def matrix(v):
    return np.array([[v[0],v[1],v[2]],[0,v[3],v[4]],[v[5],v[6],1.]])


def init_map(readings, ratios):
    pts=np.array([r['uv'] for r in readings])/1000
    xy=np.array([[ratios[r['ring']]*np.cos(r['seed_theta']),ratios[r['ring']]*np.sin(r['seed_theta'])] for r in readings])
    H,_=cv2.findHomography(xy,pts,0)
    if H is None:raise ValueError('Native support cannot initialize projective map')
    a,b=H[1,:2];R=np.array([[b,a],[-a,b]])/np.hypot(a,b)
    T=np.eye(3);T[:2,:2]=R;H=H@T;H/=H[2,2]
    return H[[0,0,0,1,1,2,2],[0,1,2,1,2,0,1]]


def distances(v,readings,ratios):
    H=matrix(v)
    if abs(np.linalg.det(H))<1e-12:return np.full(len(readings),1e6)
    inv=np.linalg.inv(H)
    z=np.column_stack((np.array([r['uv'] for r in readings])/1000,np.ones(len(readings))))
    a=z@inv.T;r=np.array([ratios[x['ring']] for x in readings])
    F=a[:,0]**2+a[:,1]**2-r*r*a[:,2]**2
    gx=2*(a[:,0]*inv[0,0]+a[:,1]*inv[1,0]-r*r*a[:,2]*inv[2,0])
    gy=2*(a[:,0]*inv[0,1]+a[:,1]*inv[1,1]-r*r*a[:,2]*inv[2,1])
    return 1000*F/np.maximum(np.hypot(gx,gy),1e-14)


def fit(views,ratios,free=False):
    initial=np.concatenate([init_map(rr,ratios) for _,rr in views]+([np.array([ratios['inner'],ratios['middle']])] if free else []))
    def unpack(x):return dict(inner=x[-2],middle=x[-1],outer=1.) if free else ratios
    def fun(x):
        rs=unpack(x)
        e=np.concatenate([distances(x[7*i:7*i+7],rr,rs)/np.array([r['sigma'] for r in rr]) for i,(_,rr) in enumerate(views)])
        if free and not 0<rs['inner']<rs['middle']<1:return e+1e4
        return e
    lo=np.full(len(initial),-np.inf);hi=np.full(len(initial),np.inf)
    if free:lo[-2:]=[.1,.4];hi[-2:]=[.6,.95]
    result=least_squares(fun,initial,bounds=(lo,hi),x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=3000)
    rs=unpack(result.x);reports=[]
    for i,(d,rr) in enumerate(views):
        e=distances(result.x[7*i:7*i+7],rr,rs)
        reports.append(dict(frame=d['programme_frame_index_1based'],pts=d['source_pts'],
            by_ring={ring:dict(n=len(ee),rms=float(np.sqrt(np.mean(ee**2))),mean=float(np.mean(ee)),max=float(np.max(abs(ee))))
                     for ring in rs for ee in [e[np.array([r['ring']==ring for r in rr])]]}))
    singular=np.linalg.svd(result.jac,compute_uv=False)
    return dict(success=bool(result.success),message=result.message,nfev=result.nfev,
                ratios=rs,weighted_half_cost=float(result.cost),per_view=reports,
                jacobian_singular_values=singular.tolist(),parameters=result.x.tolist(),
                small_normal_distance_approximation=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frames',nargs='+',required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    m=json.loads(Path(a.model).read_text(encoding='utf-8'))
    b=next(x for x in m['blocks'] if x['name']=='ice/circle/R3')['values']
    ratios=dict(inner=b[2]/b[4],middle=b[3]/b[4],outer=1.)
    views=[load_frame(f) for f in a.frames]
    report=dict(schema='rinkfit.cross_view_ring_diagnostic.v1',
        source_frames=[dict(path=str(f),sha256=sha(f),image=d['source']['image_path'],pts=d['source_pts'],
                            readings=len(rr)) for f,(d,rr) in zip(a.frames,views)],
        physical_material_id='ankara_2026/ice/R3',new_camera_id='secondary_long_side',
        baseline_model=dict(path=a.model,sha256=sha(a.model)),
        seed_station_angles_used_only_for_initialization=True,
        source_representation='direct_readouts uv, never representative fitted circle samples',
        inherited_source_selection_depends_on_old_circle_navigation=True,
        independent_held_out_evidence=False,
        fixed_current_shared_ratios=fit(views,ratios),
        diagnostic_shared_ratios=fit(views,ratios,True),
        production_model_changed=False,
        limitations=['Per-frame projective maps do not include full true-RS/Brown optics.',
                    'Sigma-weighted native contour diagnostic is not original full W or a physical camera fit.',
                    'No world center, absolute radius or metric scale is identified by these maps.',
                    'Legacy radius-dependent candidate selection requires fresh source evidence before adoption.'])
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'native_readouts.json').write_text(json.dumps([dict(frame=d['programme_frame_index_1based'],pts=d['source_pts'],readings=rr) for d,rr in views],ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['fixed_current_shared_ratios','diagnostic_shared_ratios']},ensure_ascii=False))

if __name__=='__main__':main()
