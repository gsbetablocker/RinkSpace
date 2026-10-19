"""Shared numeric rink v2: the confirmed approximate centring stays soft.

q5=[Fdx,Rdx,pitch_delta,radius_delta,common_row_midpoint_delta], in metres.
The v1 adapter remains unchanged. Every FR circle, Lane row and the row-3
carrier receives one common Y translation; fixed C/C0, boards and ads do not.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
from shared_rink_numeric_geometry import (SharedRinkNumericGeometry as _V1,
    GeometryIdentityError,SCENE_UNIT_M,PARAMETER_NAMES as _V1_PARAMETERS)

VERSION='shared_rink_numeric_geometry_midpoint_v2'
PARAMETER_NAMES=(*_V1_PARAMETERS,'common_row_midpoint_delta_m')
DEFAULT_LAYOUT_SOURCE=Path('D:/OneDrive/codex_projs/RinkSpace/configs/rink_model/confirmed_layout_v3.json')


class SharedRinkNumericGeometryMidpoint:
    """One shared midpoint variable with the same strict source-world guards."""
    def __init__(self,scene,scene_unit_m=SCENE_UNIT_M,layout_source=DEFAULT_LAYOUT_SOURCE):
        self.v1=_V1(scene,scene_unit_m)
        # Expose the unchanged base geometry fields used by joint adapters.
        for name in ('base','features','names','centres','outer_radius','column_x',
                     'row_midpoint','row_pitch','radius_ratios','scene_unit_m','base_scene_sha256'):
            setattr(self,name,getattr(self.v1,name))
        path=Path(layout_source).resolve();data=path.read_bytes();layout=json.loads(data.decode('utf8'))
        constraint=next(r for r in layout['confirmed_constraints']if r['id']=='approximately_equal_outer_ring_side_margins')
        if (layout['evidence_kind']!='user_statement_not_image_measurement' or
                constraint['kind']!='approximate_relationship' or
                constraint['user_supplied_numeric_tolerance']is not False or
                constraint['measurement_tolerance']is not None):
            raise GeometryIdentityError('Approximate centring source must retain its unmeasured soft relationship')
        weight=layout['implementation_weight']
        self.midpoint_prior_mean_m=float(weight['row_midpoint_prior_mean_model_units'])*self.scene_unit_m
        self.midpoint_prior_sigma_m=float(weight['row_midpoint_prior_sigma_model_units'])*self.scene_unit_m
        if not np.isfinite([self.midpoint_prior_mean_m,self.midpoint_prior_sigma_m]).all()or self.midpoint_prior_sigma_m<=0:
            raise ValueError('Finite positive declared midpoint engineering weight required')
        self.midpoint_source=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),statement_zh=layout['statement_zh'],
            evidence_kind=layout['evidence_kind'],constraint=constraint,implementation_weight=weight,
            user_supplied_numeric_tolerance=False,not_asserted=layout['not_asserted'])

    def _q(self,q5):
        q=np.asarray(q5,float)
        if q.shape!=(5,)or not np.isfinite(q).all():raise ValueError('q5 must contain exactly five finite metre deltas')
        # v1._q checks positive pitch/radius and column order only. Do not
        # check q4's board domain before adding the common midpoint shift.
        self.v1._q(q[:4]);return q

    def validate_parameters(self,q5):
        """Bound all eight complete circles using the final combined q5."""
        import cv2
        q=self._q(q5)
        polygon=np.ascontiguousarray(np.asarray(self.base['ice_boundary']['points_xyz'],np.float32)[:,:2])
        radius=self.outer_radius+q[3]/self.scene_unit_m
        for name in self.names:
            centre=self.centres[name][:2]+np.array([q[0 if name[0]=='F'else 1],
                (int(name[1])-2.5)*q[2]+q[4]])/self.scene_unit_m
            margin=cv2.pointPolygonTest(polygon,tuple(map(float,centre)),True)
            if margin+1e-5<radius:raise ValueError(f'Complete physical {name} outer circle leaves modeled ice footprint')
        return q.copy()

    def midpoint_prior(self,q5):
        """One engineering residual/J row on TOTAL midpoint, not delta alone.

        The caller appends this row once globally as quadratic, native scale1.
        sigma=0.5 scene unit converted using the fixed current metre scale.
        This is not measured error, a confidence interval or a hard y=0 rule.
        """
        q=self._q(q5);midpoint=self.row_midpoint*self.scene_unit_m+q[4]
        jac=np.zeros((1,5));jac[0,4]=1/self.midpoint_prior_sigma_m
        return np.array([(midpoint-self.midpoint_prior_mean_m)/self.midpoint_prior_sigma_m]),jac

    def manifest(self,q5=None):
        q=self._q(np.zeros(5)if q5 is None else q5);manifest=self.v1.manifest(q[:4])
        manifest.update(version=VERSION,parameter_names=list(PARAMETER_NAMES),parameters_m=q.tolist(),
            confirmed_relations='two co-linear columns; eight equal concentric outer circles; four paired equally spaced rows; approximate side-margin centring is soft',
            current_row_midpoint_m=self.row_midpoint*self.scene_unit_m+q[4],
            midpoint_prior=dict(mean_m=self.midpoint_prior_mean_m,sigma_m=self.midpoint_prior_sigma_m,
                applies_to='total common row midpoint in world metres',source=self.midpoint_source,
                role='declared engineering soft optimization weight; not measured uncertainty or confidence interval',
                globally_count_once=True,hard_constraint=False),
            lane_policy='paired-row rigid pitch Y translation plus one common midpoint Y translation; original widths, tilt, offsets and unmeasured left proxy retained',
            carrier_policy='row3 pitch Y plus same common midpoint Y; no inferred endpoint-X constraint',
            circle_domain_policy='all eight complete outer circles checked after combined q5; no intermediate q4 footprint rejection')
        return manifest

    def scene(self,q5):
        q=self.validate_parameters(q5);out=deepcopy(self.base)
        features={f['id']:f for f in out['ice_features']};factor=1+q[3]/(self.outer_radius*self.scene_unit_m)
        for name in self.names:
            feature=features[name]
            delta=np.array([q[0 if name[0]=='F'else 1],(int(name[1])-2.5)*q[2]+q[4],0.])/self.scene_unit_m
            centre=(self.centres[name]+delta).tolist();feature['center_xyz']=centre
            feature['radius']=self.features[name]['radius']*factor
            old_radii={r['role']:r['radius']for r in self.features[name]['rings']}
            for ring in feature['rings']:ring['center_xyz']=list(centre);ring['radius']=old_radii[ring['role']]*factor
        for i in range(1,5):
            name=f'lane{i}'
            if name not in features:continue
            shift=((i-2.5)*q[2]+q[4])/self.scene_unit_m
            for field in ('polygon_xyz','centerline_xyz'):
                if field in features[name]:
                    xyz=np.asarray(self.features[name][field],float).copy();xyz[...,1]+=shift;features[name][field]=xyz.tolist()
        name='curling_centerline_row3'
        if name in features:
            shift=(.5*q[2]+q[4])/self.scene_unit_m
            xyz=np.asarray(self.features[name]['segments_xyz'],float).copy();xyz[...,1]+=shift;features[name]['segments_xyz']=xyz.tolist()
            if 'relation'in features[name]:features[name]['relation']['common_y']=self.centres['F3'][1]+shift
        out[VERSION]=self.manifest(q);return out

    def feature_id(self,row):return self.v1.feature_id(row)

    def _world_and_jacobian(self,row,q5):
        q=self._q(q5)
        # v1 validates the ORIGINAL world against the ORIGINAL base physical
        # circle/finite Lane/carrier and source canonical authority first.
        world,j4=self.v1._world_and_jacobian(row,q[:4]);identity=self.v1._identity(row)
        jac=np.concatenate([j4,np.zeros((len(world),3,1))],axis=2)
        if identity is not None and identity[0]in('circle','lane','carrier'):
            jac[:,1,4]=1/self.scene_unit_m;world[:,1]+=q[4]/self.scene_unit_m
        return world,jac

    def rebind(self,row,q5):
        result=dict(row);result['world']=self._world_and_jacobian(row,q5)[0];return result

    def world_jacobian(self,row,q5):return self._world_and_jacobian(row,q5)[1]

    def project(self,camera,row,q5,*,with_jacobian=True):
        try:from .projection import project_central_valid
        except ImportError:from projection import project_central_valid
        world,dw=self._world_and_jacobian(row,q5)
        if not with_jacobian:return project_central_valid(camera,world)
        uv,depth,camera_jac,central=project_central_valid(camera,world,with_jacobian=True)
        jc=camera_jac.reshape(len(world),2,-1)
        return uv,depth,np.einsum('nij,njk->nik',-jc[:,:,:3]/100.,dw),central


# Drop-in import name for a caller that previously used the four-column class.
SharedRinkNumericGeometry=SharedRinkNumericGeometryMidpoint
