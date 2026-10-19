"""One shared hockey-frame transverse displacement/yaw on owned world47.

All original hockey marks move together, including both goal/crease carriers.
This is a minimal missing-frame chart, not a claim that the older K20 local
deltas have already been replaced by a complete structured rink template.
"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from wide_anchor_candidate.native_wide_source import world_curve
from .complete_ice_numeric_geometry import _frame,_boundary_hit
from .current_shared_geometry_with_free_boundary import CurrentSharedGeometryWithFreeBoundary

SCHEMA='rinkspace.current-world47-common-hockey-transverse-yaw-frame2.v1'
EXTRA_NAMES=('common_hockey_frame_transverse_delta_m','common_hockey_frame_yaw_delta_rad')
HOCKEY_IDS=('C0','A1','A2','C','B1','B2','goal_line_left','goal_line_right',
            'goal_crease_left','goal_crease_right')


class SharedHockeyFrameFreeBoundaryGeometry:
    parameter_count=49

    def __init__(self,original):
        if type(original)is not CurrentSharedGeometryWithFreeBoundary:
            raise TypeError('Explicit owned current corrected world47 boundary/height provider required')
        self.original=deepcopy(original);self.base=self.original.base
        self.scene_unit_m=float(self.original.scene_unit_m)
        self.initial=np.r_[self.original.initial,np.zeros(2)]
        self.parameter_names=tuple(self.original.parameter_names)+EXTRA_NAMES
        self.parameter_units=tuple(self.original.parameter_units)+('m','rad')
        self._scene_cache=OrderedDict()

    def __getattr__(self,name):
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(49,)or not np.isfinite(q).all():
            raise ValueError('Complete world47 plus declared common frame2 world49 required')
        return q

    def validate_parameters(self,q):
        q=self._q(q);self.original.validate_parameters(q[:47]);return q.copy()

    def current_course6(self,q):return self.original.current_course6(self._q(q)[:47])

    def _transform_scene(self,scene,q):
        if np.all(q[47:]==0.):return scene
        features={f['id']:f for f in scene['ice_features']}
        pivot,_,across=_frame(features,'goal_line_right')
        theta=q[48];c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s],[s,c]])
        translation=across[:2]*(q[47]/self.scene_unit_m)
        def move(points):
            p=np.asarray(points,float).copy();p[...,:2]=(p[...,:2]-pivot[:2])@R.T+pivot[:2]+translation
            return p
        frames={}
        for side in ('left','right'):
            a,inside,v=_frame(features,'goal_line_'+side)
            frames[side]=(move(a),np.r_[R@inside[:2],0.],np.r_[R@v[:2],0.])
        current=self.current_course6(q)
        for f in scene['ice_features']:
            if f['id']not in HOCKEY_IDS:continue
            if f['kind']=='circle':
                f['center_xyz']=move(f['center_xyz']).tolist()
                # A circle has no distinguished angular material origin.
                # Its radius/rings remain original circles under rigid motion.
            elif f['kind']=='polyline':
                old=np.asarray(f['segments_xyz'],float)
                transformed=move(old)
                # Current finite wall response, preserving declared historical
                # unmeasured endpoint offsets instead of unlimited support.
                oldroots=self.original._line_roots(old[0],current)
                newroots=self.original._line_roots(transformed[0],current)
                transformed[0]+=newroots-move(oldroots)
                f['segments_xyz']=transformed.tolist()
                f['current_shared_hockey_frame_finite_domain']=dict(
                    current_wall_intersections_used=True,original_unmeasured_finite_offsets_rotated_and_retained=True,
                    no_visible_USER_endpoint_assigned_physical_t0_or_t1=True)
            elif f['kind']=='polygon':f['polygon_xyz']=move(f['polygon_xyz']).tolist()
            else:raise ValueError('Unexpected actual hockey primitive schema: '+f['id'])
            f['shared_hockey_frame_rigid_transform']=dict(transverse_delta_m=float(q[47]),yaw_delta_rad=float(q[48]),
                no_independent_side_endpoint_or_circle_transform=True)
        for side,(centre,inside,v)in frames.items():
            f=features['goal_line_'+side]
            f['shared_physical_goal_frame']=dict(centre_xyz=centre.tolist(),inward_xyz=inside.tolist(),across_xyz=v.tolist(),
                finite_line_endpoint_midpoint_not_silently_used_as_anchor_after_clipping=True,
                chart_anchor_not_new_surveyed_world_point=True)
        scene['shared_hockey_rigid_frame']=dict(schema=SCHEMA,pivot_original_right_goal_xyz=pivot.tolist(),
            transverse_direction_original_right_goal_xyz=across.tolist(),transverse_delta_m=float(q[47]),yaw_delta_rad=float(q[48]),
            hockey_only_not_curling_or_advertisements=True)
        return scene

    def _scene(self,q,rebuild_grid):
        q=self.validate_parameters(q);key=(q.tobytes(),bool(rebuild_grid))
        cached=self._scene_cache.get(key)
        if cached is not None:self._scene_cache.move_to_end(key);return deepcopy(cached)
        source=self.original.scene(q[:47])if rebuild_grid else self.original.source_scene(q[:47])
        scene=self._transform_scene(source,q);scene[SCHEMA]=self.manifest(q)
        self._scene_cache[key]=deepcopy(scene)
        while len(self._scene_cache)>4:self._scene_cache.popitem(last=False)
        return scene

    def source_scene(self,q):return self._scene(q,False)
    def scene(self,q):return self._scene(q,True)

    def feature_state(self,q):
        s=self.source_scene(q)
        return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in s['ice_features']},
            ice_boundary=s['ice_boundary'],parameters=s['parameters'],world_geometry_manifest=self.manifest(q))

    def semantic_world(self,target,q,*,samples=256):
        return world_curve(self.source_scene(q),target,samples=samples,scene_unit_m=self.scene_unit_m)

    def rebind(self,row,q):return self.original.rebind(row,self._q(q)[:47])

    def world_jacobian(self,row,q):
        value=self.original.world_jacobian(row,self._q(q)[:47]);return np.concatenate((value,np.zeros((*value.shape[:-1],2))),axis=-1)

    def bind_world_jacobian(self,row,q,*,steps=None,samples=256):
        q=self._q(q);target=row.get('target')
        if not isinstance(target,dict):raise ValueError('Original literal admitted wide target required')
        hs=np.asarray([1e-6]*49 if steps is None else steps,float)
        if hs.shape!=(49,)or not np.isfinite(hs).all()or np.any(hs<=0.):raise ValueError('49 finite positive declared world FD steps required')
        value=self.semantic_world(target,q,samples=samples);j=np.empty((*value.shape,49))
        for k,h in enumerate(hs):
            p=q.copy();m=q.copy();p[k]+=h;m[k]-=h
            j[:,:,k]=(self.semantic_world(target,p,samples=samples)-self.semantic_world(target,m,samples=samples))/(2*h)
        return value,j

    def trapezoid_side_segments(self,q,*,goal_carrier_span_m,board_carrier_span_m):
        widths=np.asarray([goal_carrier_span_m,board_carrier_span_m],float)
        if not np.isfinite(widths).all()or widths[0]<=0. or widths[1]<=widths[0]:raise ValueError('Two positive shared outward trapezoid spans required')
        scene=self.source_scene(q);features={f['id']:f for f in scene['ice_features']};out={}
        for side in ('left','right'):
            f=features['goal_line_'+side];frame=f.get('shared_physical_goal_frame')
            if frame is None:centre,inside,across=_frame(features,'goal_line_'+side)
            else:centre,inside,across=(np.asarray(frame[k],float)for k in ('centre_xyz','inward_xyz','across_xyz'))
            for sign,label in ((-1,'minus'),(1,'plus')):
                a=centre+sign*across*(widths[0]/(2*self.scene_unit_m))
                seed=centre+sign*across*(widths[1]/(2*self.scene_unit_m))
                b=_boundary_hit(seed,-inside,scene['ice_boundary']['points_xyz'])
                out['goal_trapezoid_'+side+'_side_'+label]=dict(world_scene=np.array([a,b]),scene_unit_m=self.scene_unit_m,
                    physical_t0='current shared physical goal carrier',physical_t1='actual current finite endboard ice boundary',
                    goal_frame_transverse_sign=sign,shared_widths_both_ends=True,original_visible_endpoints_not_physical_endpoint_observations=True,
                    image_upper_lower_identity_not_inferred=True,original_wall_not_infinite_line_extension=True)
        return out

    def finite_board_records(self,records,q,*,original_planes=None):
        return self.original.finite_board_records(records,self._q(q)[:47],original_planes=original_planes)

    def manifest(self,q):
        q=self._q(q)
        return dict(schema=SCHEMA,world_parameter_count=49,original_world_parameter_count=47,
            extra_names=list(EXTRA_NAMES),extra_delta=q[47:].tolist(),
            all_C0_A_C_blue_goal_crease_carriers_share_rigid_frame=True,
            old_K20_local_deltas_retained_not_yet_replaced_by_full_structured_template=True,
            current_boundary4_and_advertisement_materials_delegate_same_owned_world47=True,
            zero_frame_all45_source_worlds_and26_ad_world_oldJ_exact=True,
            finite_transverse_line_offsets_preserved_and_current_wall_response=True,
            trapezoid_physical_support_ends_at_current_boundary_not_infinite_line=True,
            no_new_pixel_sigma_role_reference_60x30_hard_lock_or_source_admission=True,
            full_programme_graph_finiteguards_pattern_external_rebinding_not_installed=True)
