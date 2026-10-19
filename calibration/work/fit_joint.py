"""Joint three-source fit with explicit full evidence and central-ray projection.

The old implementation is read only and supplies semantic geometry/advertisement
parameterization. Its observations, weighting, lens guards and rendering are not
used by this solver. All outputs stay in this redo's directory.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'calibration'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import fit_v024_three_view_rink_ads as legacy
from rinkspace.calibration_companion.wide_angle_camera import _brown_map_jacobian, rational_minimum_denominator
from rinkspace.storage import resolve_asset
from shared_corner import SharedCornerGeometry

PTS=legacy.PTS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def fallback_project(v,xyz):
    r=cv2.Rodrigues(v[3:6])[0];q=(np.asarray(xyz)-v[:3])@r.T
    depth=q[:,2]; xy=q[:,:2]/np.maximum(depth[:,None],1e-8)
    d=v[11:23];valid=(depth>.05)&np.isfinite(xy).all(1)&(np.abs(xy).max(1)<8)
    idx=np.flatnonzero(valid);uv=np.full((len(q),2),np.nan)
    if len(idx):
        z=xy[idx]; mapped,*rest=_brown_map_jacobian(z,d)
        good=(rest[-1]>1e-7)&(rational_minimum_denominator(z,d)>1e-7)
        for t in [.25,.5,.75]:good &= _brown_map_jacobian(z*t,d)[-1]>1e-7
        u=np.column_stack([np.exp(v[6])*mapped[:,0]+v[10]*mapped[:,1]+v[8],np.exp(v[7])*mapped[:,1]+v[9]])
        good &= np.isfinite(u).all(1)
        valid[idx[~good]]=False;uv[idx[good]]=u[good]
    return uv,depth,valid

from physical_projection import safe_project, sensor_coverage_residual

def closest_delta(uv,curve,valid):
    good=valid[:-1]&valid[1:]&np.isfinite(curve[:-1]).all(1)&np.isfinite(curve[1:]).all(1)&(np.linalg.norm(np.diff(curve,axis=0),axis=1)<1200.)
    starts=np.flatnonzero(good)
    if not len(starts):return np.full_like(uv,1000.),np.full(len(uv),-1)
    a=curve[starts];b=curve[starts+1];mid=(a+b)*.5
    # Curves are sampled densely; query several adjacent segments, then use
    # exact orthogonal feet. Never reconnect across an invalid projection.
    tree=cKDTree(mid);_,ix=tree.query(uv,k=min(6,len(starts)))
    if ix.ndim==1:ix=ix[:,None]
    aa=a[ix];dd=(b-a)[ix];delta=uv[:,None,:]-aa
    f=np.sum(delta*dd,axis=2)/np.maximum(np.sum(dd*dd,axis=2),1e-20)
    delta-=np.clip(f,0,1)[:,:,None]*dd
    j=np.argmin(np.sum(delta*delta,axis=2),axis=1)
    return delta[np.arange(len(uv)),j],starts[ix[np.arange(len(uv)),j]]

class Problem(legacy.ThreeViewProblem):
    def __init__(self):
        self.input=read(BASE/'inputs/step02_observations.json')
        self.input_sha256=hashlib.sha256((BASE/'inputs/step02_observations.json').read_bytes()).hexdigest()
        self.current_observations=[o for o in self.input['observations'] if o['use']=='fit']
        fpts={f['id']:int(f['pts']) for f in self.input['frames']}
        def normalize(_):
            result=[]
            for o in self.current_observations:
                if o['category']=='ads':continue
                q=copy.deepcopy(o);q.update(type='curve',frame_pts=fpts[o['frame_id']],uv=np.asarray(o['pixels_uv'],float),sigma_px=1.)
                q['sqrt_weights']=np.sqrt(np.asarray(o['weight']['point_integration_weights']))
                result.append(q)
            return result
        old=legacy.normalized_observations;legacy.normalized_observations=normalize
        try:super().__init__(legacy.DEFAULT_INPUT)
        finally:legacy.normalized_observations=old
        self.corner_index=len(self.x0)
        self.corner_aspect_index=self.corner_index+1
        self.x0=np.r_[self.x0,2.,1.]
        self.bounds=(np.r_[self.bounds[0],1.15,.4],np.r_[self.bounds[1],5.,2.5])
        self.names.extend(['scene:shared_corner_exponent','scene:shared_corner_y_over_x'])
        # A labelled advertisement may straddle a straight/corner join. The
        # old segment identity seeds its position; it cannot clamp its corners.
        for sl in self.slices_ad().values():
            self.bounds[0][sl.start:sl.stop-2]=-.3
            self.bounds[1][sl.start:sl.stop-2]=1.3
        for pts in PTS[1:]:
            p=resolve_asset(ROOT, f'outputs/calibration/ankara_2026_low_end_joint_20260923/per_frame_frozen_v020_20260924/{pts}/fit.json')
            d=read(p);cam=next(f['camera'] for f in d['frames'] if f['source_pts']==pts)
            self.x0[self.slices[('camera',pts)]]=legacy.camera_vector(cam)
        # Preserve the successful high-view Brown12, start lows with a regular
        # physical pinhole. No rational denominator or tilted sensor initially.
        for pts in PTS:
            sl=self.slices[('camera',pts)];self.x0[sl][-2:]=0.
        self.curve_samples=420
        self.enabled_frames=set(PTS);self.include_ads=True
        self.data_scale=1.
        self.full_names=self.names
        self.ncurve=sum(len(o['uv'])*2 for o in self.fit_obs)
        self.counts={pts:len({o['correlation_group_id'] for o in self.current_observations if fpts[o['frame_id']]==pts}) for pts in PTS}
        self.frame_weights={pts:1. for pts in PTS}
        self.tail_scale=None
        self.n_guard=3*(63+45)+10+len(self.slices_ad())+2
        self.n_reg=3*18
        self.residual_size=self.ncurve+2*len(self.ad_obs)+self.n_guard+self.n_reg
        self.boundary_curve_targets={}
        self.by_id={o['id']:o for o in self.fit_obs}

    def _initialize_low2_from_shared_ads(self):pass
    def _initialize_low3_candidate_from_ads(self):pass

    def geometry(self,x):
        base=super().geometry(x)
        exponent=x[self.corner_index] if hasattr(self,'corner_index') and len(x)>self.corner_index else 2.
        aspect=x[self.corner_aspect_index] if hasattr(self,'corner_aspect_index') and len(x)>self.corner_aspect_index else 1.
        return SharedCornerGeometry(base,exponent,aspect)

    def ad_world(self,x,geo,number,role):
        values=x[self.slices[('ad',number)]];right=len(values)==4
        along=values[1] if role.startswith('right_') and right else values[0]
        z=float(values[-2] if role.endswith('top') else values[-1])*geo.parameters['board_height']
        side=legacy.AD_SIDES[number]
        return geo.board_position(side,float(along),z)

    @lru_cache(maxsize=160)
    def cached_geometry(self,key):
        return SharedCornerGeometry(super().geometry(np.asarray(key)),key[-2],key[-1])

    def geometry_key(self,x):
        return tuple(x[:self.geometry_count])+(float(x[self.corner_index]),float(x[self.corner_aspect_index]) if len(x)>self.corner_aspect_index else 1.)

    @lru_cache(maxsize=2048)
    def cached_delta(self,gkey,ckey,oid,samples):
        o=self.by_id[oid];geo=self.cached_geometry(gkey)
        world=self.curve_world(geo,o)
        projected,depth,valid=safe_project(np.asarray(ckey),world)
        return closest_delta(o['uv'],projected,valid)

    def curve_world(self,geo,row):
        t=row['target'];n=self.curve_samples
        if t['kind']=='continuous_board_edge':
            pieces=[geo.board_side({'side':s,'edge':t['edge']},max(80,n//3)) for s in t['ordered_regions']]
            # Each domain is connected through actual common physical endpoints.
            out=pieces[0]
            for piece in pieces[1:]:
                if len(out)==len(pieces[0]) and np.linalg.norm(out[0]-piece[0])<1e-6:out=out[::-1]
                if len(out)==len(pieces[0]) and np.linalg.norm(out[0]-piece[-1])<1e-6:out=out[::-1]
                if np.linalg.norm(out[-1]-piece[-1])<1e-6:piece=piece[::-1]
                if np.linalg.norm(out[-1]-piece[0])>1e-5:raise ValueError('disconnected board domain')
                out=np.vstack([out,piece[1:]])
            return out
        return np.asarray(geo.points(t,'curve',n),float)

    def parts(self,x,details=False):
        gkey=self.geometry_key(x);geo=self.cached_geometry(gkey);parts=[];rows=[]
        for o in self.fit_obs:
            pts=o['frame_pts'];cam=x[self.slices[('camera',pts)]]
            d,seg=self.cached_delta(gkey,tuple(cam),o['id'],self.curve_samples)
            scale=o['sqrt_weights']*self.frame_weights[pts] if pts in self.enabled_frames else np.zeros(len(d))
            tail=np.sqrt(1+np.sum(d*d,axis=1)/self.tail_scale**2) if self.tail_scale else np.ones(len(d))
            parts.append((d*(scale*tail)[:,None]).ravel())
            if details:
                mag=np.linalg.norm(d,axis=1)
                rows.append({'id':o['id'],'frame_pts':pts,'category':o['category'],'target':o['target'],'count':len(mag),'rms_px':float(np.sqrt(np.mean(mag**2))),'p95_px':float(np.percentile(mag,95)),'max_px':float(mag.max()),'median_px':float(np.median(mag)),'deltas_uv':d.tolist(),'matched_segments':seg.tolist()})
        for row in self.ad_obs:
            pts=row['pts'];cam=x[self.slices[('camera',pts)]];world=self.ad_world(x,geo,row['number'],row['role']).reshape(1,3)
            projected,depth,valid=safe_project(cam,world)
            delta=projected[0]-row['uv'] if valid[0] else np.full(2,1000.)
            w=self.frame_weights[pts]/math.sqrt(self.ad_frame_sizes[(pts,row['number'])]) if self.include_ads and pts in self.enabled_frames else 0.
            tail=math.sqrt(1+float(delta@delta)/self.tail_scale**2) if self.tail_scale else 1.
            parts.append(delta*w*tail)
            if details:rows.append({'id':f"ad:{row['number']}:{row['letter']}:{pts}",'frame_pts':pts,'category':'ads','target':{'kind':'advertisement_corner','user_ad_number':row['number'],'role':row['role']},'count':1,'rms_px':float(np.linalg.norm(delta)),'p95_px':float(np.linalg.norm(delta)),'max_px':float(np.linalg.norm(delta)),'deltas_uv':[delta.tolist()]})
        return np.concatenate(parts),rows,geo

    def sensor_guard(self,v):
        # Fast positive-Jacobian regularity guard over a conservative sensor ray
        # grid; every target additionally uses the actual central-ray gate.
        fx,fy=np.exp(v[6:8]);u,y=np.meshgrid(np.linspace(0,1920,9),np.linspace(0,1080,7));yn=(y.ravel()-v[9])/fy;xn=(u.ravel()-v[8]-v[10]*yn)/fx
        xy=np.column_stack([xn,yn]);d=v[11:23]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            eig=_brown_map_jacobian(xy,d)[-1];den=rational_minimum_denominator(xy,d)
        eig=np.nan_to_num(eig,nan=-1e3,neginf=-1e3,posinf=1e3);den=np.nan_to_num(den,nan=-1e3,neginf=-1e3,posinf=1e3)
        return np.r_[200*(np.maximum(.08-eig,0)+np.maximum(.1-den,0)),sensor_coverage_residual(v,sensor_grid=(5,3),pixel_scale=1.)]

    def residual(self,x,reg=.01):
        data,_,geo=self.parts(x)
        guards=[]
        for pts in PTS:guards.append(self.sensor_guard(x[self.slices[('camera',pts)]]))
        guards.append(50*np.minimum(np.asarray(list(geo.boundary_validity_margins().values()))-1e-5,0))
        for (_,number),sl in self.slices_ad().items():
            v=x[sl];sense=1. if legacy.AD_SIDES[number]=='upper_long_side' else -1.
            guards.append(np.array([100*max(.001-(v[-2]-v[-1]),0)+ (100*max(.001-sense*(v[1]-v[0]),0) if len(v)==4 else 0)]))
        # A real camera must be outside the rounded ice outline. Requiring x
        # beyond the bounding rectangle incorrectly forbids outside-corner
        # positions that are physically outside this rink.
        guards.append(np.array([100*max(geo.boundary_clearance(x[self.slices[('camera',pts)]][:2])+.05,0) for pts in PTS[1:]]))
        for pts in PTS:
            v=x[self.slices[('camera',pts)]]
            guards.append(reg*np.r_[v[11:],v[10]/1000,(v[8]-960)/960,(v[9]-540)/540,v[6]-v[7]])
        out=np.r_[data,*guards]
        assert len(out)==self.residual_size
        if not np.isfinite(out).all():raise ValueError('nonfinite residual')
        return out

    def save_result(self,x,path,stage_record):
        data,rows,geo=self.parts(x,True)
        save(path,{'schema_version':'rinkspace.refit.fit.v1','complete':False,'parameter_names':self.names,'parameter_values':x.tolist(),'cameras':{str(p):self.camera_dict(x,p) for p in PTS},'rink_parameters':geo.parameters,'ice_features':list(geo.features.values()),'advertisement_estimates':self.advertisement_estimates(x),'data_norm':float(np.linalg.norm(data)),'residuals':rows,'stage':stage_record,'loss_policy':{'physical_group_budget':1,'frame_multiplier':1,'tail_scale_px':self.tail_scale,'purpose':'prevent a short badly displaced corner from disappearing in a long-curve average'},'observation_file':'calibration/inputs/step02_observations.json','observation_sha256':self.input_sha256,'scene_source':str(legacy.SCENE.relative_to(ROOT)),'hard_geometry_checks':geo.hard_geometry_checks(),'boundary_margins':geo.boundary_validity_margins(),'units':'conditional scene units; no verified metre scale'})

def fit_stage(p,x,active,name,out,max_nfev=160,workers=6,reg=.01):
    active=np.asarray(sorted(set(active)),int);frozen=x.copy();lo,hi=(b[active] for b in p.bounds)
    def fun(z):
        xx=frozen.copy();xx[active]=z;return p.residual(xx,reg)
    before=float(np.linalg.norm(fun(x[active])));started=time.time()
    lastsave=[started]
    def callback(intermediate_result):
        if time.time()-lastsave[0]>45:
            xx=frozen.copy();xx[active]=intermediate_result.x
            p.save_result(xx,out/'checkpoint.json',{'name':name,'interim':True,'seconds':time.time()-started})
            print(json.dumps({'stage':name,'cost':float(intermediate_result.cost),'nfev':int(intermediate_result.nfev),'elapsed':round(time.time()-started)}),flush=True);lastsave[0]=time.time()
    options=dict(bounds=(lo,hi),x_scale='jac',loss='linear',jac='3-point',max_nfev=max_nfev,ftol=1e-9,xtol=1e-11,gtol=1e-7,diff_step=1e-6,callback=callback)
    with ThreadPoolExecutor(max_workers=workers) as pool:fit=least_squares(fun,x[active],workers=pool.map,**options)
    x=x.copy();x[active]=fit.x
    record={'name':name,'evaluations':int(fit.nfev),'success':bool(fit.success),'message':str(fit.message),'before':before,'after':float(np.linalg.norm(fun(fit.x))),'seconds':time.time()-started,'active_parameters':[p.names[i] for i in active]}
    p.save_result(x,out/(name+'.json'),record)
    p.save_result(x,out/'current.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='active_parameters'}),flush=True)
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='prepare');ap.add_argument('--resume');ap.add_argument('--nfev',type=int,default=150);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--out');ap.add_argument('--tail-scale',type=float,default=6.)
    args=ap.parse_args();p=Problem();x=p.x0.copy();out=Path(args.out) if args.out else BASE/'work/fit';out.mkdir(exist_ok=True)
    if args.resume:
        d=read(args.resume)
        if d['parameter_names']==p.names:x=np.array(d['parameter_values'])
        elif d['parameter_names']==p.names[:len(d['parameter_names'])]:x=np.r_[d['parameter_values'],p.x0[len(d['parameter_names']):]]
        else:raise ValueError('parameter names mismatch')
        if any(np.max(np.abs(x[p.slices[('camera',pts)]][23:25]))>1e-12 for pts in PTS):raise ValueError('resume camera includes unsupported tilted sensor')
    if args.stage=='prepare':p.save_result(x,out/'initial.json',{'name':'initial'});print('prepared');return
    ads=[i for sl in p.slices_ad().values() for i in range(sl.start,sl.stop)]
    geom=list(range(p.geometry_count))+[p.corner_index,p.corner_aspect_index]
    def cam(pts,indices):return [p.slices[('camera',pts)].start+i for i in indices]
    if args.stage=='high':
        p.enabled_frames={PTS[0]};p.include_ads=False
        active=geom+cam(PTS[0],list(range(10))+list(range(11,16))+list(range(19,23)))
    elif args.stage in ('low2','low3'):
        pts=PTS[1] if args.stage=='low2' else PTS[2]
        p.enabled_frames={pts};p.include_ads=False
        active=cam(pts,list(range(10))+list(range(11,16)))
    elif args.stage=='ads':active=ads
    elif args.stage=='joint':active=geom+ads+[i for pts in PTS for i in cam(pts,list(range(10))+list(range(11,16))+list(range(19,23)))]
    elif args.stage in ('refine','polish'):
        active=geom+ads+[i for pts in PTS for i in cam(pts,list(range(23)))]
        if args.stage=='polish':p.tail_scale=args.tail_scale
    else:raise ValueError(args.stage)
    fit_stage(p,x,active,args.stage,out,max_nfev=args.nfev,workers=args.workers)

def problem_for_fit(fit):
    """Select the saved model, keeping earlier 139-parameter fits reproducible."""
    if any(name.startswith('ring:') for name in fit.get('parameter_names',[])):
        from fit_ring_refinement import RingProblem
        return RingProblem()
    if any(name.startswith('marker:') for name in fit.get('parameter_names',[])):
        from fit_marker_refinement import MarkerProblem
        return MarkerProblem()
    return Problem()

if __name__=='__main__':main()
