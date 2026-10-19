# Adapted from user-supplied RinkSpace_v014_calibration_companion/code/keyframe_solver.py.
# Original SHA-256: 71bd1c53d6af29defad1c80e5ec6dbdc69b191d2419aaaf4d8e07472b92d3fe3. See provenance.json for exact changes.
"""Conditional multi-keyframe calibration, adapted to v0.14 scene.json.

Implemented:
  * one shared (approximate optical) center per installation;
  * independent rotation and focal length per keyframe;
  * fixed, explicitly supplied principal point/aspect/distortion;
  * semantic point and point-to-projected-curve residuals;
  * disjoint fit/check observations and numerical conditioning diagnostics.

NOT implemented: automatic image recognition, optimization of scene geometry,
tripod offset model, continuous video tracking, or total physical uncertainty.
Fitting the schematic scene yields CONDITIONAL camera estimates, not real truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.optimize import least_squares
from .scene_camera import Camera, Scene, nearest_polyline_vectors


def _stats(values: list[float]) -> dict[str, Any]:
    a=np.asarray(values,float)
    return {'n':len(a),'median_px':float(np.median(a)) if len(a) else None,
            'p95_px':float(np.percentile(a,95)) if len(a) else None,
            'max_px':float(a.max()) if len(a) else None,
            'rmse_px':float(np.sqrt(np.mean(a*a))) if len(a) else None}


def _finite(x: Any, n: int, name: str) -> np.ndarray:
    a=np.asarray(x,float)
    if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError(f'{name} must have {n} finite values')
    return a


def _object_ids(observation: dict[str, Any]) -> set[tuple[str, str]]:
    """Whole semantic objects stay on one side of the fit/check split.

    The explicit physical group may join several scene features. A feature ID
    also joins every ring/edge and every frame observing that same object.
    Observations without either identity cannot be matched by this guard; their
    identity must be supplied by the evidence-preparation layer.
    """
    identities=set()
    for key in ('physical_group', 'physical_group_id'):
        value=observation.get(key)
        if value is not None:
            if not isinstance(value,str) or not value:
                raise ValueError(f'{key} must be a nonempty string')
            identities.add(('physical_group',value))
    for value in (observation.get('feature_id'),observation['target'].get('feature_id')):
        if value is not None:
            if not isinstance(value,str) or not value:
                raise ValueError('feature_id must be a nonempty string')
            identities.add(('feature_id',value))
    return identities


class KeyframeProblem:
    def __init__(self, scene: Scene, problem: dict[str,Any]):
        self.scene=scene;self.problem=problem
        if problem.get('scene_sha256')!=scene.sha256:
            raise ValueError('scene_sha256 mismatch: observations must identify their scene version')
        if problem.get('geometry_mode')!='fixed_scene_conditional':
            raise ValueError('This solver only supports fixed_scene_conditional')
        self.installations={i['id']:i for i in problem['installations']}
        self.frames={f['id']:f for f in problem['frames']}
        if not self.frames or len(self.frames)!=len(problem['frames']): raise ValueError('no frames or duplicate frame IDs')
        if len(self.installations)!=len(problem['installations']): raise ValueError('duplicate installation IDs')
        self.center_slices={};self.frame_slices={}
        values=[];lower=[];upper=[]
        for iid,i in self.installations.items():
            c=_finite(i['initial_center_xyz'],3,'initial_center_xyz')
            if i.get('fit_center',True):
                if 'center_bounds_xyz' not in i: raise ValueError(f'{iid}: supply explicit center bounds and their rationale')
                lo=_finite(i['center_bounds_xyz'][0],3,'center lower bounds')
                hi=_finite(i['center_bounds_xyz'][1],3,'center upper bounds')
                if np.any(hi<=lo) or np.any(c<lo) or np.any(c>hi): raise ValueError('invalid center bounds')
                self.center_slices[iid]=slice(len(values),len(values)+3)
                values.extend(c);lower.extend(lo);upper.extend(hi)
            if 'center_prior' in i:
                _finite(i['center_prior']['mean'],3,'center prior mean')
                if np.any(_finite(i['center_prior']['sigma'],3,'center prior sigma')<=0):raise ValueError('prior sigma must be positive')
        for fid,f in self.frames.items():
            if f['installation_id'] not in self.installations: raise ValueError('unknown installation')
            wh=_finite(f['image_size_wh'],2,'image_size_wh')
            if np.any(wh<=0):raise ValueError('invalid image size')
            _finite(f['principal_point_px'],2,'principal_point_px')
            r=_finite(f['initial_rvec_world_to_camera'],3,'initial_rvec')
            focal=float(f['initial_focal_px']);lo,hi=map(float,f['focal_bounds_px'])
            if not 0<lo<=focal<=hi or lo==hi:raise ValueError('invalid focal bounds')
            if float(f.get('fy_over_fx',1.0))<=0:raise ValueError('invalid aspect')
            if not problem.get('synthetic',False):
                for key in ['source_id','media_track_id','source_pts','time_base','image_path','image_sha256','pixel_frame_id']:
                    if key not in f or f[key] is None:raise ValueError(f'{fid}: missing real-frame provenance {key}')
                from fractions import Fraction
                Fraction(f['time_base'])  # actual PTS remains in the original track units.
            self.frame_slices[fid]=slice(len(values),len(values)+4)
            values.extend(r);values.append(np.log(focal))
            lower.extend([-np.inf]*3+[np.log(lo)]);upper.extend([np.inf]*3+[np.log(hi)])
        self.x0=np.asarray(values,float);self.bounds=(np.asarray(lower),np.asarray(upper))
        self.observations=[];seen=set();object_uses={}
        samples=int(problem.get('curve_samples',192))
        for o in problem['observations']:
            if o['id'] in seen: raise ValueError('duplicate observation ID')
            seen.add(o['id'])
            if o['frame_id'] not in self.frames or o['use'] not in ('fit','check'):raise ValueError('invalid observation frame/use')
            for identity in _object_ids(o):
                previous=object_uses.setdefault(identity,o['use'])
                if previous!=o['use']:
                    raise ValueError(f'whole-object split violation: {identity[0]} {identity[1]!r} appears in fit and check')
            uv=np.asarray(o['pixels_uv'],float).reshape(-1,2)
            if not len(uv) or not np.isfinite(uv).all():raise ValueError('empty/nonfinite observation')
            sigma=float(o['sigma_px']);weight=float(o.get('weight',1.0))
            if sigma<=0 or weight<=0:raise ValueError('positive sigma/weight required')
            if o['type']=='point':
                if len(uv)!=1:raise ValueError('point observation must have exactly one pixel')
                xyz=scene.point(o['target']).reshape(1,3)
            elif o['type']=='curve':xyz=scene.curve(o['target'],samples)
            else:raise ValueError('type must be point or curve')
            if not np.isfinite(xyz).all():raise ValueError('unknown target cannot be projected')
            self.observations.append((o,uv,xyz))
        for fid in self.frames:
            if not any(o['frame_id']==fid and o['use']=='fit' for o,_,_ in self.observations):
                raise ValueError('Each optimized frame needs fit evidence. For a wholly unseen frame, solve its pose on a designated subset and check other features.')
        if len(self.residual(self.x0))<len(self.x0):raise ValueError('fewer residuals than unknowns; add evidence or reduce free parameters')
        # Fit-only initial cheirality check avoids optimizing from a flipped view.
        for o,_,xyz in self.observations:
            # Held-out geometry must not choose or reject an initial fit branch.
            if o['use']!='fit':continue
            _,z=self.camera(self.x0,o['frame_id']).project(xyz)
            if np.any(z<=0):raise ValueError('initial view puts observed geometry behind camera; choose another initializer')

    def camera(self,x: np.ndarray,fid: str)->Camera:
        f=self.frames[fid];iid=f['installation_id'];i=self.installations[iid]
        center=x[self.center_slices[iid]] if iid in self.center_slices else np.asarray(i['initial_center_xyz'],float)
        pars=x[self.frame_slices[fid]];focal=np.exp(pars[3]);cx,cy=f['principal_point_px']
        K=np.array([[focal,0,cx],[0,focal*float(f.get('fy_over_fx',1.0)),cy],[0,0,1]])
        return Camera(center,pars[:3],K,np.asarray(f.get('distortion_opencv',[]),float))

    def raw_error(self,x: np.ndarray,o:dict[str,Any],uv:np.ndarray,xyz:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        pred,z=self.camera(x,o['frame_id']).project(xyz)
        if not np.isfinite(pred).all():raise ValueError('nonfinite projection during optimization')
        delta=pred-uv if o['type']=='point' else nearest_polyline_vectors(uv,pred)
        return delta,z

    def residual(self,x:np.ndarray)->np.ndarray:
        parts=[]
        for o,uv,xyz in self.observations:
            if o['use']!='fit':continue
            delta,z=self.raw_error(x,o,uv,xyz)
            # Normalize within each semantic object. 192 curve samples are not 192
            # independent landmarks, nor are they allowed to dominate a point.
            w=np.sqrt(float(o.get('weight',1.0))/len(uv))/float(o['sigma_px'])
            parts.append((delta*w).ravel())
            # Continuous cheirality penalty; final result separately rejects behind-camera points.
            parts.append(np.minimum(z-1e-6,0.0)*1e3/np.sqrt(len(z)))
        for iid,i in self.installations.items():
            if iid in self.center_slices and 'center_prior' in i:
                p=i['center_prior'];parts.append((x[self.center_slices[iid]]-np.asarray(p['mean']))/np.asarray(p['sigma']))
        return np.concatenate(parts)

    def summarize(self,x:np.ndarray)->dict[str,Any]:
        rows=[];all_errors={'fit':[],'check':[]};groups={}
        behind=False
        for o,uv,xyz in self.observations:
            delta,z=self.raw_error(x,o,uv,xyz);err=np.linalg.norm(delta,axis=1)
            behind=behind or bool(np.any(z<=0))
            row={'id':o['id'],'frame_id':o['frame_id'],'use':o['use'],'target':o['target'],
                 'min_depth_scene_units':float(z.min()),**_stats(err.tolist())}
            rows.append(row);all_errors[o['use']].extend(err.tolist())
            surface=o.get('surface_class','unspecified');groups.setdefault((o['use'],surface),[]).extend(err.tolist())
        return {'observations':rows,'overall':{k:_stats(v) for k,v in all_errors.items()},
                'by_use_and_surface':{u+':'+s:_stats(v) for (u,s),v in groups.items()},
                'any_observed_geometry_behind_camera':behind,
                'note':'Overall point counts do not represent independent sample counts; inspect semantic-object and surface errors.'}

    def solve(self,max_nfev:int|None=None)->dict[str,Any]:
        options={} if max_nfev is None else {'max_nfev':int(max_nfev)}
        result=least_squares(self.residual,self.x0,bounds=self.bounds,method='trf',
                             loss='soft_l1',f_scale=1.0,x_scale='jac',**options)
        J=np.asarray(result.jac);norm=np.linalg.norm(J,axis=0)
        normalized=J/np.maximum(norm,1e-15)
        singular=np.linalg.svd(normalized,compute_uv=False)
        threshold=float(singular[0]*1e-6) if len(singular) else 0
        rank=int(np.sum(singular>threshold))
        condition=float(singular[0]/singular[-1]) if len(singular) and singular[-1]>1e-15 else None
        frames=[]
        for fid,f in self.frames.items():
            metadata={k:v for k,v in f.items() if not k.startswith('initial_')}
            frames.append({**metadata,'camera':self.camera(result.x,fid).to_dict()})
        installations=[]
        for iid,i in self.installations.items():
            related=next(fid for fid,f in self.frames.items() if f['installation_id']==iid)
            installations.append({'id':iid,'physical_camera_id':i.get('physical_camera_id'),
                'center_xyz':self.camera(result.x,related).center_xyz.tolist(),
                'center_interpretation':'fixed_optical_center_approximation_NOT_measured_tripod',
                'position_status':'synthetic_estimate' if self.problem.get('synthetic') else 'conditional_on_fixed_schematic_scene',
                'prior':i.get('center_prior'), 'bounds_xyz':i.get('center_bounds_xyz')})
        return {'schema_version':'rinkspace.conditional-keyframe-fit.v1','synthetic':bool(self.problem.get('synthetic',False)),
            'scene_sha256':self.scene.sha256,'scene_revision':self.scene.data['revision'],
            'scene_units':self.scene.data['coordinate_system']['unit'],
            'scene_geometry_held_fixed':True,'physical_accuracy_verified':False,
            'optimizer':{'success':bool(result.success),'message':str(result.message),'nfev':result.nfev,'cost':float(result.cost),
                         'near_bounds_parameter_indices':np.flatnonzero(result.active_mask).tolist()},
            'local_numerical_identifiability':{'column_normalized_singular_values':singular.tolist(),
                'rank_at_relative_1e-6':rank,'parameter_count':len(result.x),'condition_estimate':condition,
                'note':'Local diagnostic only; not physical uncertainty and not proof of global uniqueness.'},
            'before':self.summarize(self.x0),'after':self.summarize(result.x),
            'installations':installations,'frames':frames,
            'limitations':['No scene geometry or lens distortion was optimized.',
                           'No automatic image detection or temporal propagation was run.',
                           'Check observations are excluded from residuals but are not necessarily whole unseen clips.',
                           'A small image error does not prove the schematic world coordinates or camera positions are correct.']}


def pose_candidates_from_points(scene:Scene,observations:list[dict],frame:dict,
                                focal_candidates_px:list[float])->list[dict]:
    """Optional PnP initialization at supplied focal hypotheses.

    Uses FIT point observations only. Returns multiple conditional candidates;
    filtering by the user's camera side/cheirality is still required.
    No claim that OpenCV UPNP solves the unknown-focal problem.
    """
    obs=[o for o in observations if o['frame_id']==frame['id'] and o['use']=='fit' and o['type']=='point']
    if len(obs)<4:raise ValueError('PnP initializer needs at least four actual, non-collinear point correspondences; curve observations are not invented point correspondences.')
    X=np.array([scene.point(o['target']) for o in obs],float)
    uv=np.array([o['pixels_uv'][0] for o in obs],float)
    rank=np.linalg.matrix_rank(X-X.mean(axis=0))
    if rank<2:raise ValueError('collinear/degenerate world points')
    flags=cv2.SOLVEPNP_IPPE if rank==2 else cv2.SOLVEPNP_EPNP
    if rank==3 and len(obs)<6:raise ValueError('Use at least six distributed points for this nonplanar initializer; the count alone does not establish good conditioning.')
    out=[];cx,cy=frame['principal_point_px'];dist=np.asarray(frame.get('distortion_opencv',[]),float)
    for focal in focal_candidates_px:
        if focal<=0:raise ValueError('positive focal hypotheses required')
        K=np.array([[focal,0,cx],[0,focal*frame.get('fy_over_fx',1),cy],[0,0,1]],float)
        try: ans=cv2.solvePnPGeneric(X,uv,K,dist,flags=flags)
        except cv2.error:continue
        if not ans[0]:continue
        for r,t in zip(ans[1],ans[2]):
            r=r.reshape(3);t=t.reshape(3);center=-cv2.Rodrigues(r)[0].T@t
            cam=Camera(center,r,K,dist);pred,z=cam.project(X)
            if np.all(z>0):out.append({'center_xyz':center.tolist(),'rvec_world_to_camera':r.tolist(),
                'focal_px':float(focal),'fit_rmse_px':float(np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))),
                'status':'conditional_initializer_not_confirmed'})
    return sorted(out,key=lambda r:r['fit_rmse_px'])


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--problem',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=None,help='Optional numerical solver setting, not a project execution budget.')
    a=p.parse_args()
    if a.output.resolve() in {a.scene.resolve(),a.problem.resolve()}:raise ValueError('refusing to overwrite an input')
    model=Scene(a.scene);data=json.loads(a.problem.read_text(encoding='utf-8-sig'))
    fit=KeyframeProblem(model,data).solve(a.max_nfev)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(fit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'optimizer':fit['optimizer'],'fit':fit['after']['overall'],
                      'conditional_only':True,'output':str(a.output)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
