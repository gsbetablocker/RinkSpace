"""Explicit similarity gauge changes; never independently rescale camera and scene.

The normalized demo supplement is initialization inventory. It does not assert
physical equality with another historical scene or a verified metre scale.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SimilarityGauge:
    scale: float
    rotation: np.ndarray
    origin: np.ndarray
    source_coordinate_system: str
    target_coordinate_system: str

    def __post_init__(self):
        q = np.array(self.rotation, dtype=np.float64, copy=True)
        o = np.array(self.origin, dtype=np.float64, copy=True)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('gauge scale must be finite and positive')
        if q.shape != (3,3) or o.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(o).all():
            raise ValueError('invalid gauge rotation/origin')
        if not np.allclose(q @ q.T, np.eye(3), atol=1e-12, rtol=0) or abs(np.linalg.det(q)-1)>1e-12:
            raise ValueError('gauge rotation must be right handed orthonormal')
        if not self.source_coordinate_system or not self.target_coordinate_system:
            raise ValueError('gauge coordinate IDs are required')
        q.flags.writeable = False
        o.flags.writeable = False
        object.__setattr__(self,'rotation',q)
        object.__setattr__(self,'origin',o)

    def points(self, xyz):
        p = np.asarray(xyz, dtype=np.float64)
        if p.shape[-1:] != (3,) or not np.isfinite(p).all():
            raise ValueError('points must have final dimension 3 and finite values')
        return self.scale * ((p-self.origin) @ self.rotation.T)

    def directions(self, xyz):
        v = np.asarray(xyz, dtype=np.float64)
        if v.shape[-1:] != (3,) or not np.isfinite(v).all():
            raise ValueError('directions must have final dimension 3 and finite values')
        return v @ self.rotation.T

    def camera(self, center, world_to_camera):
        r = np.asarray(world_to_camera,dtype=np.float64)
        if r.shape[-2:] != (3,3) or not np.isfinite(r).all():
            raise ValueError('rotation must have final dimensions 3x3')
        return self.points(center), r @ self.rotation.T

    def lengths(self, values):
        a = np.asarray(values,dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError('lengths must be finite')
        return a*self.scale


def write_demo_supplement(baseline_dir, output):
    """Make a self-contained normalized geometry/reference supplement.

    Only delivered new baseline data is consumed. No legacy solver, packet,
    source requalification, or historical archive is involved.
    """
    from .baseline import load_initialization, project_world_pinhole
    directory, target = Path(baseline_dir), Path(output)
    package = load_initialization(directory)
    rigs = json.loads((directory/'rig_reference.json').read_text(encoding='utf-8-sig'))
    boundary = np.asarray(package.scene['ice_boundary']['points_xyz'],dtype=np.float64)
    span = float(np.ptp(boundary[:,0]))
    if span <= 0:
        raise ValueError('demo boundary has no long-axis extent')
    source_id = package.scene['coordinate_system']['id']
    gauge = SimilarityGauge(1/span,np.eye(3),np.zeros(3),source_id,'ankara_user_ppt_z_up_L1')
    paths = [dict(p,xyz=gauge.points(p['xyz']).tolist(),
                  initialization_source='demo8a28_normalized',
                  coordinate_system_id=gauge.target_coordinate_system) for p in package.paths]
    references=[]
    for rig in rigs['rigs']:
        record=dict(rig)
        record.update(position_xyz=gauge.points(rig['position_xyz']).tolist(),
                      direction_xyz=gauge.directions(rig['direction_xyz']).tolist(),
                      height=float(gauge.lengths(rig['height'])),
                      unit='normalized_scene_units_L1',pivot_xyz=None,
                      coordinate_system_id=gauge.target_coordinate_system)
        bounds=rig.get('historical_shot_optical_center_range')
        if bounds:
            # Q=I here, so componentwise extrema remain extrema.
            record['historical_shot_optical_center_range']={k:gauge.points(v).tolist() for k,v in bounds.items()}
        record['reference_only']=True
        record['new_shot_fit_started']=False
        references.append(record)
    source_ads={int(a['user_number']):a for a in package.scene.get('advertisements',())}
    path_ads={int(p['id'].split(':',1)[0][2:]) for p in paths if p['id'].startswith('Ad') and p['id'].split(':',1)[0][2:].isdigit()}
    advertisement_registry=[]
    for number in range(1,27):
        ad=source_ads.get(number)
        matching=[p['id'] for p in paths if p['id'].split(':',1)[0]=='Ad'+str(number)]
        scope=ad.get('extent_scope') if ad else ('board_print_pattern' if matching else 'landmark_only_or_missing_geometry')
        advertisement_registry.append({'id':'Ad'+str(number),'user_number':number,
            'extent_scope':scope,'path_ids':matching,'source':'scene.advertisements' if ad else 'scene.board_patterns/source_conditioned_prints',
            'geometry_present':bool(matching),'geometry_status':ad.get('geometry_status') if ad else 'initial_geometry_only',
            'source_status':ad.get('status') if ad else 'not_an_external_banner_boundary',
            'source_fitted_roles':ad.get('fitted_roles') if ad else None})
    for path in paths:
        if path['id'].startswith('Ad'):
            prefix=path['id'].split(':',1)[0][2:]
            if prefix.isdigit() and int(prefix) in source_ads:
                path['extent_scope']=source_ads[int(prefix)].get('extent_scope')
    # Compare the actual full native exposure sequence, not a few camera poses.
    sample=np.concatenate([np.asarray(p['xyz'],dtype=np.float64)[::max(1,len(p['xyz'])//24)] for p in package.paths])
    normalized=gauge.points(sample)
    max_error=0.0; sum_sq=0.0; count=0; missing=0
    for pts in package.arrays['native_pts']:
        center,rotation,intrinsic=package.camera(int(pts))
        uv,valid=project_world_pinhole(center,rotation,intrinsic,sample)
        nc,nr=gauge.camera(center,rotation)
        nv,nvalid=project_world_pinhole(nc,nr,intrinsic,normalized)
        viewport=valid & (uv[:,0]>=0) & (uv[:,0]<=1919) & (uv[:,1]>=0) & (uv[:,1]<=1079)
        missing+=int(np.count_nonzero(viewport & ~nvalid))
        good=viewport & nvalid
        error=np.linalg.norm(nv[good]-uv[good],axis=1)
        if len(error):
            max_error=max(max_error,float(error.max()))
            sum_sq+=float(error @ error); count+=len(error)
    proof={'native_exposures':len(package.arrays['native_pts']),'visible_point_comparisons':count,
           'missing':missing,'maximum_pixel_difference':max_error,
           'rms_pixel_difference':float(np.sqrt(sum_sq/count)) if count else None,
           'scope':'demo gauge invariance only; no equality with d78 or visual acceptance'}
    if not count or missing or max_error>1e-7:
        raise ValueError('uniform gauge changed native projection: '+repr(proof))
    value={'schema':'rinksolve.initialization_supplement.v1','revision':1,
           'source_baseline':str(directory.resolve()),'source_model_identity':rigs['model_source_identity'],
           'source_coordinate_system':source_id,'coordinate_system_id':gauge.target_coordinate_system,
           'unit':'normalized_scene_units_L1','verified_metre_scale':False,
           'transform':{'scale':gauge.scale,'rotation':gauge.rotation.tolist(),'origin':gauge.origin.tolist(),
                        'definition':'Xnew=s*Q*(Xold-origin); Cnew=s*Q*(Cold-origin); Rnew=Rold*Q.T',
                        'normalization':'original demo full long-axis boundary extent equals 1',
                        'original_long_extent':span},
           'status':'initialization_only_not_measurement_or_accepted_model',
           'can_supply_unobserved_inventory':True,'can_override_observed_d78_geometry':False,
           'semantic_binding_required_before_factors':True,
           'advertisement_registry':advertisement_registry,'paths':paths,'rig_references':references,'gauge_projection_check':proof,
           'limitations':['No assertion that demo and d78 paint or board geometries are equal.',
                          'Three camera positions are historical optical-center references, never pivots.',
                          'Sample vertices are drawing samples, never independently confirmed landmarks.',
                          'Subsequent shots remain locked.']}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return value
