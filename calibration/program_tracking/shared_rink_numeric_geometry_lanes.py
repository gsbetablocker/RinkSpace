"""Opt-in common Lane width/right edge with the existing five rink variables.

Original source worlds are validated against the untouched Midpoint base.
The three painted sides are then mapped by their original finite-segment
parameter to the same physical side of four parallel, equal-width Lanes.
The left vertices remain unmeasured display proxies, not painted endpoints.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import cv2
from shared_rink_numeric_geometry_midpoint import SharedRinkNumericGeometryMidpoint
from z190_continuous_source_bundle import SCENE_UNIT_M

VERSION='shared_rink_numeric_geometry_lanes_v1'
PARAMETER_NAMES=('F_column_dx_m','R_column_dx_m','row_pitch_delta_m',
                 'outer_radius_delta_m','common_row_midpoint_delta_m',
                 'shared_lane_halfwidth_delta_m','shared_lane_rightX_delta_m')
ROOT=Path(__file__).resolve().parents[2]


def reference(path):
    p=Path(path)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


class SharedRinkNumericGeometryLanes:
    """q7 preserves FR geometry and restores declared common Lane relations."""
    parameter_names=PARAMETER_NAMES
    parameter_count=7

    def __init__(self,base,scene_unit_m=SCENE_UNIT_M):
        self.midpoint=SharedRinkNumericGeometryMidpoint(base,scene_unit_m)
        self.base=self.midpoint.base
        self.scene_unit_m=float(self.midpoint.scene_unit_m)
        self.lane_names=tuple(f'lane{i}' for i in range(1,5))
        self.old_polygons={}
        widths,right=[],[]
        for name in self.lane_names:
            f=self.midpoint.features[name]
            p=np.asarray(f['polygon_xyz'],float)
            if len(p)==5 and np.allclose(p[0],p[-1],atol=1e-8):p=p[:-1]
            if p.shape!=(4,3) or not np.isfinite(p).all() or np.max(abs(p[:,2]))>1e-8:
                raise ValueError('Each original Lane requires four finite z0 vertices')
            if f.get('lane_edge_order')!=['upper_long','right_short','lower_long','left_open'] or f.get('open_edges')!=[3]:
                raise ValueError('Explicit three painted sides and unpainted left proxy required')
            if f.get('polygon_left_vertices_role') not in ('unmeasured_display_proxy_not_physical_endpoints','guaranteed_core_display_proxy_not_physical_endpoints'):
                raise ValueError('Left vertices must retain explicit unmeasured proxy authority')
            self.old_polygons[name]=p.copy()
            widths.append((p[0,1]+p[1,1]-p[2,1]-p[3,1])/4.)
            right.extend(p[1:3,0])
        self.base_halfwidth_scene=float(np.mean(widths))
        self.base_rightX_scene=float(np.mean(right))
        if self.base_halfwidth_scene<=0:raise ValueError('Original mean Lane half width must be positive')
        self.authority=dict(
            evidence_kind='existing user-declared relations; not a numerical dimension measurement',
            existing_user_hard_profile='user_20260923_alignment_symmetry',
            implementation=reference(ROOT/'src/rinkspace/calibration_companion/structural_constraints.py'),
            declaration=reference(ROOT/'scripts/fit_wide_joint.py'),
            declaration_source='User follow-ups in wide joint tasks, 2026-09-23',
            provenance_kind='Legacy user-confirmation record in the repository structural profile; original wording is not reconstructed here',
            confirmed_layout_v2_v3_do_not_supply_Lane_authority=True,
            confirmed_relations=['centerline through paired FR row','four equal widths','long edges parallel to world X','one common right X','left side open'],
            numeric_mean_width_and_right_are_old_estimates_not_user_measurements=True,
            independent_source_validation=False)
        self._boundary=np.asarray(self.base['ice_boundary']['points_xyz'],float)[:,:2].astype(np.float32)

    def __getattr__(self,name):
        return getattr(self.midpoint,name)

    def _q(self,q7):
        q=np.asarray(q7,float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Seven finite shared metric parameters required')
        self.midpoint._q(q[:5])
        if self.base_halfwidth_scene+q[5]/self.scene_unit_m<=0:raise ValueError('Nonpositive shared Lane half width')
        return q

    def _polygon_and_jacobian(self,name,q):
        i=int(name[-1]);p=self.old_polygons[name].copy()
        cy=self.midpoint.centres[f'F{i}'][1]+((i-2.5)*q[2]+q[4])/self.scene_unit_m
        halfwidth=self.base_halfwidth_scene+q[5]/self.scene_unit_m
        right=self.base_rightX_scene+q[6]/self.scene_unit_m
        if right<=max(p[0,0],p[3,0]):raise ValueError('Shared Lane right edge must follow both unmeasured left proxies')
        p[0,1]=p[1,1]=cy+halfwidth
        p[2,1]=p[3,1]=cy-halfwidth
        p[1,0]=p[2,0]=right
        j=np.zeros((4,3,7))
        j[:,1,2]=(i-2.5)/self.scene_unit_m
        j[:,1,4]=1/self.scene_unit_m
        j[:,1,5]=np.array([1.,1.,-1.,-1.])/self.scene_unit_m
        j[1:3,0,6]=1/self.scene_unit_m
        return p,j

    def validate_parameters(self,q7):
        q=self._q(q7)
        self.midpoint.validate_parameters(q[:5])
        for name in self.lane_names:
            p,_=self._polygon_and_jacobian(name,q)
            for x,y in p[:,:2]:
                if cv2.pointPolygonTest(self._boundary,(float(x),float(y)),True)<-2e-5:
                    raise ValueError('Actual Lane paint lies outside the shared ice footprint: '+name)
        return q

    def feature_id(self,row):return self.midpoint.feature_id(row)

    def _world_and_jacobian(self,row,q7):
        q=self._q(q7)
        identity=self.midpoint.v1._identity(row)
        if identity is None or identity[0]!='lane':
            world,j5=self.midpoint._world_and_jacobian(row,q[:5])
            return world,np.concatenate((j5,np.zeros((len(world),3,2))),axis=2)
        # This is the existing strict authority/world check. New geometry must
        # never make an invalid original world qualify as the claimed side.
        original,_=self.midpoint._world_and_jacobian(row,np.zeros(5))
        _,name,edge=identity
        if edge==3:raise ValueError('Unmeasured left Lane display proxy is not a source painted side')
        old=self.old_polygons[name]
        a,b=old[edge],old[(edge+1)%4]
        d=b-a;length2=float(d@d)
        if length2<=0:raise ValueError('Degenerate original finite Lane side')
        t=(original-a)@d/length2
        if np.any(t<-1e-7) or np.any(t>1+1e-7) or np.max(np.linalg.norm(original-(a+t[:,None]*d),axis=1))>1e-5:
            raise ValueError('Original source vertices do not belong to their finite Lane side')
        polygon,pj=self._polygon_and_jacobian(name,q)
        first,last=polygon[edge],polygon[(edge+1)%4]
        world=first+t[:,None]*(last-first)
        jac=pj[edge][None,:,:]+t[:,None,None]*(pj[(edge+1)%4]-pj[edge])[None,:,:]
        return world,jac

    def rebind(self,row,q7):
        out=dict(row);out['world']=self._world_and_jacobian(row,q7)[0];return out

    def world_jacobian(self,row,q7):return self._world_and_jacobian(row,q7)[1]

    def project(self,camera,row,q7,*,with_jacobian=True):
        from projection import project_central_valid
        world,dw=self._world_and_jacobian(row,q7)
        if not with_jacobian:return project_central_valid(camera,world)
        uv,depth,camera_jac,central=project_central_valid(camera,world,with_jacobian=True)
        jc=camera_jac.reshape(len(world),2,-1)
        return uv,depth,np.einsum('nij,njk->nik',-jc[:,:,:3]/100.,dw),central

    def midpoint_prior(self,q7):
        q=self._q(q7);r,j=self.midpoint.midpoint_prior(q[:5])
        return r,np.c_[j,np.zeros((len(r),2))]

    def manifest(self,q7=None):
        q=self._q(np.zeros(7)if q7 is None else q7)
        # Keep the same base scene, circle/carrier guards and exactly-once
        # engineering midpoint prior metadata used by the q5 joint driver.
        manifest=self.midpoint.manifest(q[:5])
        manifest.update(version=VERSION,parameter_count=7,
            parameter_names=list(PARAMETER_NAMES),parameters_m=q.tolist(),authority=self.authority,
            base_halfwidth_scene=self.base_halfwidth_scene,base_halfwidth_m=self.base_halfwidth_scene*self.scene_unit_m,
            base_rightX_scene=self.base_rightX_scene,base_rightX_m=self.base_rightX_scene*self.scene_unit_m,
            base_numeric_initialization='Mean of actual old polygon half widths and right-side vertex X coordinates, not stale metadata dimensions or source measurements',
            halfwidth_prior=None,rightX_prior=None,
            rightX_observability='Must be supplied by actual finite-side/end evidence; interior long-side source alone may leave this variable unobservable',
            source_binding='Validate untouched original finite side with old Midpoint rules, then map original segment parameter t to same-role shared Lane side',
            source_UV_normal_sigma_roles_unchanged=True,other_ice_geometry='Exactly the old Midpoint q5; no Lane parameter moves any circle or row3 carrier',
            lane_policy='Four equal widths and world-X-parallel long edges; paired FR row centre Y and shared right X; untouched unmeasured left X display proxies',
            source_painted_edges=[0,1,2],source_forbidden_proxy_edges=[3],
            left_vertices='Retain original X display proxies; side3 is open and must never be a paint source',
            accuracy_accepted=False)
        return manifest

    def scene(self,q7):
        q=self.validate_parameters(q7)
        out=self.midpoint.scene(q[:5]);features={f['id']:f for f in out['ice_features']}
        for name in self.lane_names:
            p,_=self._polygon_and_jacobian(name,q);f=features[name]
            f['polygon_xyz']=p.tolist()
            # The left vertices are separate unknown-end display proxies.
            # Start the displayed centreline only where both paint sides
            # exist, as the original structural profile does; do not infer
            # a physical end from their average X.
            cy=(p[0,1]+p[3,1])/2
            f['centerline_xyz']=[[float(max(p[0,0],p[3,0])),float(cy),0.],
                                 [float(p[1,0]),float(cy),0.]]
            f['lane_half_width_scene_units']=self.base_halfwidth_scene+q[5]/self.scene_unit_m
            f['lane_shared_right_x_scene_units']=self.base_rightX_scene+q[6]/self.scene_unit_m
            f['lane_centerline_through_FR_row']=int(name[-1])
            f['centerline_source_features']=[f'F{name[-1]}',f'R{name[-1]}']
            f['association_fields_status']='Actual reconstructed shared hard relations; dimensions remain source-estimated unknowns'
            f['paint_topology']='three_sides_open_at_xmin';f['closed']=False;f['open_edges']=[3]
            f['lane_edge_order']=['upper_long','right_short','lower_long','left_open']
            if 'paint_segments_xyz'in f:
                f['paint_segments_xyz']=[p[[0,1]].tolist(),p[[1,2]].tolist(),p[[2,3]].tolist()]
        out[VERSION]=self.manifest(q)
        return out
