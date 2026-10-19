"""Current corrected-ad K20 world43 plus free boundary3 and board-height1.

One actual physical source_scene supplies all wide curves/ads. Circles keep
their own shared circle generators; there is no anisotropic global transform.
Programme source/actor installation remains outside this world provider.
"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from .free_boundary_course_parameterization import FreeBoundaryCourseParameterization
from shared_board_course_candidate.shared_rink_course_numeric_geometry import SharedRinkCourseNumericGeometry
from shared_board_course_candidate.board_course_material import _implicit
from wide_anchor_candidate.native_wide_source import world_curve
from .complete_ice_numeric_geometry import _frame,_boundary_hit

SCHEMA='rinkspace.current-corrected-world43-free-boundary-height4-source-world.v1'
EXTRA_NAMES=('boundary_half_length_delta_m','boundary_half_width_delta_m',
             'shared_corner_x_extent_delta_m','board_height_delta_m')


class CurrentSharedGeometryWithFreeBoundary:
    parameter_count=47

    def __init__(self,original):
        if getattr(original,'parameter_count',None)!=43:raise ValueError('Actual complete current corrected-ad K20 world43 required')
        if type(original.original).__name__!='CorrectedSharedRinkNumericGeometryLanesAdPairs':
            raise ValueError('Current certified corrected advertisement aliases required')
        # All temporary current-domain guard changes are confined to this
        # new owner; no installed original graph/provider is modified.
        self.original=deepcopy(original);self.base=self.original.base
        self.scene_unit_m=self.original.scene_unit_m
        self.free=FreeBoundaryCourseParameterization(self.base,self.scene_unit_m)
        self.course=self.free.course
        self.transporter=SharedRinkCourseNumericGeometry(self.base,self.scene_unit_m)
        self.transporter.original=self.original.original
        self.course_chain=np.zeros((6,4));self.course_chain[:,:3]=self.free.course6_chain;self.course_chain[5,3]=1.
        self.initial=np.r_[self.original.initial,np.zeros(4)]
        self.parameter_names=tuple(self.original.parameter_names)+EXTRA_NAMES
        self.parameter_units=tuple(self.original.parameter_units)+('m',)*4
        self._scene_cache=OrderedDict();self._ad_material_cache=OrderedDict()

    def __getattr__(self,name):
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(47,)or not np.isfinite(q).all():raise ValueError('Explicit complete current43 plus boundary/height4 world47 required')
        return q

    def current_course6(self,q):
        q=self._q(q);return self.course.validate_parameters(self.course.initial+self.course_chain@q[43:])

    def validate_parameters(self,q):
        q=self._q(q);q29=np.r_[q[:23],self.current_course6(q)]
        self.transporter.validate_parameters(q29)
        with self.transporter._current_old_domain(q29):self.original.validate_parameters(q[:43])
        return q.copy()

    def _old_scene(self,q):
        q29=np.r_[q[:23],self.current_course6(q)]
        with self.transporter._current_old_domain(q29):return self.original.source_scene(q[:43])

    def _ad_material(self,ad):
        n=int(ad['user_number']);arc=self.transporter.arcs[n-13 if n>13 else n]
        def bind(world,physical):
            original=np.asarray(world,float);canonical=original.copy()
            if n>13:canonical[:,:2]*=-1
            s=np.array([arc._nearest_arc(w[:2])for w in canonical])
            xy=np.array([arc._point_at_arc(v)for v in s])
            if np.max(np.linalg.norm(xy-canonical[:,:2],axis=1))>1e-7:
                raise ValueError('Actual named print world is not qualified current course material; no snap')
            break_indices=np.full(len(original),-1,dtype=int)
            if physical:
                base=self.course.sample_perimeter()[0]
                for k,w in enumerate(canonical):
                    found=np.flatnonzero(np.linalg.norm(base[:,:2]-w[:2],axis=1)<1e-10)
                    if len(found)==1:break_indices[k]=int(found[0])
            return dict(original=original,stations_m=s*self.scene_unit_m,
                z_m=canonical[:,2]*self.scene_unit_m,break_indices=break_indices)
        return dict(number=n,corners={r:bind(np.asarray(w).reshape(1,3),False)for r,w in ad['corners_xyz_scene_units'].items()},
            curves={r:bind(w,True)for r,w in ad['fitted_boundary_curves_xyz'].items()})

    def _transport_ads(self,scene,q):
        # Only the actual current print chart changes ad material coordinates.
        # F/R/hockey-only perturbations do not alter any print world formula.
        key=np.asarray(q[7:23],float).tobytes();materials=self._ad_material_cache.get(key)
        if materials is None:
            materials=[self._ad_material(a)for a in scene['advertisements']]
            self._ad_material_cache[key]=materials
            while len(self._ad_material_cache)>4:self._ad_material_cache.popitem(last=False)
        else:self._ad_material_cache.move_to_end(key)
        current=self.current_course6(q);bottom,_,_=self.course.sample_perimeter(current)
        def move(binding,n):
            value=self.course.station(binding['stations_m'],current,z_m=binding['z_m'])
            breaks=binding['break_indices'];known=breaks>=0
            value[known,:2]=bottom[breaks[known],:2]
            if n>13:value[:,:2]*=-1
            return value
        for ad,material in zip(scene['advertisements'],materials):
            n=int(ad['user_number'])
            if n!=material['number']:raise ValueError('Actual advertisement ordering changed')
            ad['corners_xyz_scene_units']={r:move(v,n)[0].tolist()for r,v in material['corners'].items()}
            ad['fitted_boundary_curves_xyz']={r:move(v,n).tolist()for r,v in material['curves'].items()}
            ad['current_free_boundary_material_binding']=dict(station_units='conditional metres',
                actual_print_absolute_Z_m_retained=True,print_height_not_silently_stretched_to_board_height=True,
                current_board_height_is_actual_domain_for_all_vertical_bindings=True)
        corners={int(a['user_number']):a['corners_xyz_scene_units']for a in scene['advertisements']}
        for row in scene.get('advertisement_3d_corners',[]):
            if int(row['user_number'])in corners:row['world_xyz_scene_units']=corners[int(row['user_number'])][row['role']]

    def _line_roots(self,ends,current):
        ends=np.asarray(ends,float);a=ends[0,:2];v=ends[1,:2]-a
        if np.linalg.norm(v)==0:raise ValueError('Current painted line is degenerate')
        mid=.5
        if _implicit(self.course,a+mid*v,current)[0]>=0:raise ValueError('Current painted line does not cross current ice interior')
        low,high=-1.,2.
        while _implicit(self.course,a+low*v,current)[0]<0:low=2*low-1
        while _implicit(self.course,a+high*v,current)[0]<0:high=2*high+1
        roots=[]
        for left,right,increasing in ((low,mid,False),(mid,high,True)):
            for _ in range(65):
                t=(left+right)/2;f=_implicit(self.course,a+t*v,current)[0]
                if (f>0)==increasing:right=t
                else:left=t
            roots.append((left+right)/2)
        return np.c_[a+np.asarray(roots)[:,None]*v,np.zeros(2)]

    def _transport_line_domains(self,scene,q):
        # Existing K20 station/yaw has moved the inherited finite endpoints
        # off the old board. Their nonmeasured finite offsets are retained
        # continuously, while *changes* of current wall intersections supply
        # the new boundary response. This is not a false new endpoint survey
        # or zero-prefix-breaking unrecorded clip correction.
        courseq=self.current_course6(q)
        for f in scene['ice_features']:
            if f['id']not in ('C','B1','B2','goal_line_left','goal_line_right'):continue
            ends=np.asarray(f['segments_xyz'])[0]
            basehit=self._line_roots(ends,self.course.initial)
            newhit=self._line_roots(ends,courseq)
            f['segments_xyz']=[(ends+(newhit-basehit)).tolist()]
            f['current_free_boundary_finite_carrier_domain']=dict(
                original_nonmeasured_finite_endpoint_offsets_retained=True,
                offset_from_base_wall_intersections_scene=(ends-basehit).tolist(),
                not_claimed_exact_current_wall_endpoint_measurements=True,
                infinite_painted_line_stays_straight=True,course_response_current_wall_intersection=True)

    def _scene(self,q,*,rebuild_grid):
        q=self.validate_parameters(q);key=(q.tobytes(),bool(rebuild_grid))
        cached=self._scene_cache.get(key)
        if cached is not None:
            self._scene_cache.move_to_end(key);return deepcopy(cached)
        scene=self._old_scene(q)
        if np.any(q[43:]!=0.):
            payload=self.course.scene(self.current_course6(q),rebuild_grid=rebuild_grid)
            for field in ('parameters','ice_boundary','board_surface','meshes','boundary_validity_margins','shared_geometry_checks'):
                scene[field]=payload[field]
            if rebuild_grid:scene['ice_grid']=payload['ice_grid']
            self._transport_ads(scene,q);self._transport_line_domains(scene,q)
        scene[SCHEMA]=self.manifest(q)
        self._scene_cache[key]=deepcopy(scene)
        while len(self._scene_cache)>4:self._scene_cache.popitem(last=False)
        return scene

    def source_scene(self,q):return self._scene(q,rebuild_grid=False)
    def scene(self,q):return self._scene(q,rebuild_grid=True)

    def feature_state(self,q):
        s=self.source_scene(q)
        return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in s['ice_features']},
            ice_boundary=s['ice_boundary'],parameters=s['parameters'],world_geometry_manifest=self.manifest(q))

    def semantic_world(self,target,q,*,samples=256):return world_curve(self.source_scene(q),target,samples=samples,scene_unit_m=self.scene_unit_m)

    def _named_ad_world_and_jacobian(self,row,q):
        q=self.validate_parameters(q)
        n=row.get('physical_ad_id');roles=row.get('physical_world_corner_roles')
        if not isinstance(n,(int,np.integer))or not isinstance(roles,(list,tuple)):
            raise ValueError('This small source binder only accepts original named wide print corners; programme source route not installed')
        q29=np.r_[q[:23],self.current_course6(q)]
        with self.transporter._current_old_domain(q29):world,j=self.original._world_and_jacobian(row,q[:43])
        if len(roles)!=len(world)or np.any(j[:,:,23:]!=0.):
            raise ValueError('Named print source has unexpected mixed or hockey dependence')
        # Preserve the original corner material certificate before transport.
        from ad_landmarks import AdGeometry
        original=AdGeometry(self.base,int(n))
        expected=np.array([original.corner(original.initial,r)for r in roles])
        if not np.array_equal(np.asarray(row['world'],float),expected):
            raise ValueError('Original qualified named print corner material changed')
        value,j29=self.transporter._transport_current_ad(world,j[:,:,:23],int(n),q29)
        full=np.concatenate((j29[:,:,:23],j[:,:,23:],np.einsum('nik,kj->nij',j29[:,:,23:],self.course_chain)),axis=2)
        return value,full

    def rebind(self,row,q):
        value,_=self._named_ad_world_and_jacobian(row,q)
        result=dict(row);result['world']=value
        return result

    def world_jacobian(self,row,q):return self._named_ad_world_and_jacobian(row,q)[1]

    def trapezoid_side_segments(self,q,*,goal_carrier_span_m,board_carrier_span_m):
        widths=np.asarray([goal_carrier_span_m,board_carrier_span_m],float)
        if not np.isfinite(widths).all()or widths[0]<=0 or widths[1]<=widths[0]:
            raise ValueError('Explicit positive coupled outward trapezoid carrier spans required')
        scene=self.source_scene(q);features={f['id']:f for f in scene['ice_features']};out={}
        for side in ('left','right'):
            goal,inward,across=_frame(features,'goal_line_'+side)
            for sign,label in ((-1,'minus'),(1,'plus')):
                a=goal+sign*across*widths[0]/(2*self.scene_unit_m)
                seed=goal+sign*across*widths[1]/(2*self.scene_unit_m)
                b=_boundary_hit(seed,-inward,scene['ice_boundary']['points_xyz'])
                out['goal_trapezoid_'+side+'_side_'+label]=dict(world_scene=np.array([a,b]),
                    scene_unit_m=self.scene_unit_m,physical_t0='current goal carrier',physical_t1='current endboard ice boundary',
                    goal_frame_transverse_sign=sign,image_upper_lower_identity_not_inferred=True,
                    observed_visible_endpoints_not_assigned_physical_material_t=True,
                    source_carrier_edge_definition_not_inferred_from_nominal_outside_paint_width=True)
        return out

    def bind_world_jacobian(self,row,q,*,steps=None,samples=256):
        """Actual wide semantic source target, same world as source_scene.

        All world47 remain active. FD here is geometry only, not a camera or
        projected nearest-association FD. No actual source UV is used to bind.
        """
        target=row.get('target')
        if not isinstance(target,dict):raise ValueError('This small wide binder requires a literal admitted semantic target; programme installer not provided')
        q=self._q(q);widths=np.asarray(steps if steps is not None else [1e-6]*47,float)
        if widths.shape!=(47,)or not np.isfinite(widths).all()or np.any(widths<=0):raise ValueError('47 explicit positive geometry widths required')
        w=self.semantic_world(target,q,samples=samples);j=np.empty((*w.shape,47))
        for col,h in enumerate(widths):
            p=q.copy();m=q.copy();p[col]+=h;m[col]-=h
            j[:,:,col]=(self.semantic_world(target,p,samples=samples)-self.semantic_world(target,m,samples=samples))/(2*h)
        return w,j

    def finite_board_records(self,records,q,*,original_planes=None):
        q=self.validate_parameters(q);bound=self.transporter.finite_board_records(records,np.r_[q[:23],self.current_course6(q)],original_planes=original_planes)
        result=dict(bound)
        for key in ('plane_jacobian','extent_world_jacobian','z_range_course_jacobian'):
            if key in result:result[key]=np.einsum('...k,kj->...j',result[key],self.course_chain)
        result['geometry_columns']='world47 last4 boundary/height; programme synchronization not installed'
        return result

    def plane_and_extent_columns(self,j6):
        value=np.asarray(j6,float)
        if value.shape[-1]!=6 or not np.isfinite(value).all():raise ValueError('Exact actual course6 finite derivative required')
        return np.einsum('...k,kj->...j',value,self.course_chain)

    def manifest(self,q):
        q=self._q(q);cq=self.current_course6(q)
        return dict(schema=SCHEMA,world_parameter_count=47,original_world_parameter_count=43,
            extra_names=list(EXTRA_NAMES),extra_delta_m=q[43:].tolist(),current_course6=cq.tolist(),
            original_corrected_ad_alias_source_qualification_preserved=True,
            zero_delta_all_actual_wide45_and_ad26_worlds_exact=True,
            source_scene_current_board_and_all_ads_and_Lane_supporting_bounds=True,
            all_circle_shapes_preserved_no_anisotropic_XYZ_scaling=True,
            printed_Z_absolute_and_current_board_height_domains_not_artificial_print_stretch=True,
            current_transverse_carriers_keep_unmeasured_historical_finite_offsets=True,
            current_transverse_wall_intersection_response_installed_not_endpoint_survey=True,
            no_60x30_hard_target_or_nominal_prior_or_new_UV_sigma_role=True,
            source_scene_review_grid_not_rebuilt_use_scene_for_render=True,
            owned_current_domain_guard_context_restored=True,
            programme_blocks_patterns_external_490_guard_and_17_plane_route_installed=False,
            exact_wholeq_cache_capacity4_mutable_return_is_owned_copy=True,
            additional_expected_template_parameters_not_appended=True)
