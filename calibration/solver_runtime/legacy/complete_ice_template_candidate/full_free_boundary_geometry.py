"""Owned full six-DOF shared superellipse course; no historical shape freeze.

Preserves old chart coordinates and appends independent Ry/p deltas. The old
Rx column's oblique chart component is retained for exact old prefixes; Ry is
independent, so the six physical course values have full rank. Four corners
share one (Rx,Ry,p), not four independent shapes. No 60x30 or radius prior.
"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from wide_anchor_candidate.native_wide_source import world_curve
from .current_shared_geometry_with_free_boundary import CurrentSharedGeometryWithFreeBoundary
from .shared_hockey_frame_free_boundary import SharedHockeyFrameFreeBoundaryGeometry
from .structured_shared_ice_template import StructuredSharedIceTemplateGeometry
from .complete_ice_numeric_geometry import _frame,_boundary_hit

SCHEMA='rinkspace.shared-superellipse-course-full-six-active-chart.v1'
EXTRA_NAMES=('shared_corner_y_extent_independent_delta_m','shared_corner_exponent_delta')
WORLD_BATCH_PROTOCOL='rinkspace.current-scene-semantic-world-and-finite-trapezoid-bundle.v1'
TRAPEZOID_NAMES=tuple('goal_trapezoid_'+side+'_side_'+label for side in ('left','right')for label in ('minus','plus'))

def finite_trapezoid_ends_from_current_scene(scene,scene_unit_m,spans=None):
    features={f['id']:f for f in scene['ice_features']}
    if spans is None:
        return np.array([np.asarray(features['goal_trapezoid_'+side+'_'+edge+'_outer']['segments_xyz'])[0]
            for side in ('left','right')for edge in ('lower','upper')])
    spans=np.asarray(spans,float)
    if spans.shape!=(2,)or not np.isfinite(spans).all()or spans[0]<=0 or spans[1]<=spans[0]:raise ValueError('Two declared positive outward shared spans required')
    values=[]
    for side in ('left','right'):
        frame=features['goal_line_'+side].get('shared_physical_goal_frame')
        if frame is None:centre,inside,across=_frame(features,'goal_line_'+side)
        else:centre,inside,across=(np.asarray(frame[k],float)for k in ('centre_xyz','inward_xyz','across_xyz'))
        for sign in (-1,1):
            a=centre+sign*across*(spans[0]/(2*scene_unit_m));seed=centre+sign*across*(spans[1]/(2*scene_unit_m))
            b=_boundary_hit(seed,-inside,scene['ice_boundary']['points_xyz']);values.append(np.array([a,b]))
    return np.array(values)

def semantic_world_from_current_scene(scene,target,*,samples=256,scene_unit_m):
    """Same admitted rotated Lane semantics, without rebuilding parent scene."""
    if target.get('kind')!='polygon_supporting_line':return world_curve(scene,target,samples=samples,scene_unit_m=scene_unit_m)
    feature=next(f for f in scene['ice_features']if f['id']==target['feature_id']);k=int(target['index'])
    if feature['id']not in('lane1','lane2','lane3','lane4')or k not in(0,1,2)or k in feature.get('open_edges',[]):raise ValueError('Qualified original three-sided Lane target required')
    poly=np.asarray(feature['polygon_xyz']);a,b=poly[[k,(k+1)%4]];v=b-a
    if np.sum(abs(v[:2])>1e-9)==1:return world_curve(scene,target,samples=samples,scene_unit_m=scene_unit_m)
    xmin,ymin,xmax,ymax=map(float,scene['parameters']['bounds']);hits=[]
    for axis,low,high in((0,xmin,xmax),(1,ymin,ymax)):
        if v[axis]==0.:continue
        for bound in(low,high):
            t=(bound-a[axis])/v[axis];point=a+t*v;other=1-axis
            if(ymin if other==1 else xmin)-1e-10<=point[other]<=(ymax if other==1 else xmax)+1e-10:hits.append(point)
    if len(hits)<2:raise ValueError('Current Lane carrier does not cross original current rectangular support bounds')
    axis=int(np.argmax(abs(v[:2])));hits=sorted(hits,key=lambda x:x[axis])
    return np.linspace(hits[0],hits[-1],samples)

class _FullBoundary47Core(CurrentSharedGeometryWithFreeBoundary):
    @classmethod
    def from_original(cls,original):
        if type(original)is not CurrentSharedGeometryWithFreeBoundary:raise TypeError('Authenticated exact old boundary4 owner required')
        obj=cls.__new__(cls);obj.__dict__=deepcopy(original.__dict__)
        obj.extra=np.zeros(2);obj._scene_cache=OrderedDict();return obj

    def activate(self,extra):
        extra=np.asarray(extra,float)
        if extra.shape!=(2,)or not np.isfinite(extra).all():raise ValueError('Declared independent Ry and exponent deltas required')
        self.extra=extra.copy()

    def current_course6(self,q):
        q=self._q(q);value=self.course.initial+self.course_chain@q[43:]
        value[3]+=self.extra[0];value[4]+=self.extra[1]
        return self.course.validate_parameters(value)

    def _scene(self,q,*,rebuild_grid):
        q=self.validate_parameters(q);key=(q.tobytes(),self.extra.tobytes(),bool(rebuild_grid))
        if key in self._scene_cache:self._scene_cache.move_to_end(key);return deepcopy(self._scene_cache[key])
        scene=self._old_scene(q)
        if np.any(q[43:]!=0.)or np.any(self.extra!=0.):
            payload=self.course.scene(self.current_course6(q),rebuild_grid=rebuild_grid)
            for field in ('parameters','ice_boundary','board_surface','meshes','boundary_validity_margins','shared_geometry_checks'):scene[field]=payload[field]
            if rebuild_grid:scene['ice_grid']=payload['ice_grid']
            self._transport_ads(scene,q);self._transport_line_domains(scene,q)
        scene[SCHEMA]=self.manifest(q);self._scene_cache[key]=deepcopy(scene)
        while len(self._scene_cache)>4:self._scene_cache.popitem(last=False)
        return scene

    def manifest(self,q):
        q=self._q(q)
        return dict(schema=SCHEMA,internal_old47_plus_declared2_context=True,current_course6=self.current_course6(q).tolist(),
            independent_Ry_delta_m=float(self.extra[0]),independent_p_delta=float(self.extra[1]),
            physical_course_parameter_order=['halfL_m','halfW_m','Rx_m','Ry_m','p','boardH_m'],
            all_six_physical_values_active=True,four_corners_share_one_superellipse=True,
            existing_convex_numeric_domain='halfL>Rx>0; halfW>Ry>0; p>1; H>0',
            historical_Ry_Rx_ratio_not_a_constraint_after_independent_Ry_column=True,
            historical_exponent_not_fixed=True,no_60x30_hard_target_or_new_UV_sigma_role=True,
            programme_plane_extent_pattern_external_installation_pending=True)

def _named_ad_course_value(owner,row,q47):
    n=row.get('physical_ad_id');roles=row.get('physical_world_corner_roles')
    if not isinstance(n,(int,np.integer))or not isinstance(roles,(list,tuple)):raise ValueError('Original qualified named print corners required')
    from ad_landmarks import AdGeometry
    source=AdGeometry(owner.base,int(n));expected=np.array([source.corner(source.initial,r)for r in roles])
    if not np.array_equal(np.asarray(row['world'],float),expected):raise ValueError('Original named print material authority changed')
    q29=np.r_[q47[:23],owner.current_course6(q47)]
    with owner.transporter._current_old_domain(q29):
        world,j23=owner.transporter.original._world_and_jacobian(row,q47[:23])
        return owner.transporter._transport_current_ad(world,j23,int(n),q29)

class CurrentSharedGeometryWithFullFreeBoundary:
    parameter_count=51
    world_batch_protocol_schema=WORLD_BATCH_PROTOCOL
    def __init__(self,original):
        if type(original)is not SharedHockeyFrameFreeBoundaryGeometry:raise TypeError('Explicit authenticated old49 provider required')
        self.original=deepcopy(original);self.original.original=_FullBoundary47Core.from_original(self.original.original)
        self.boundary_owner=self.original.original;self.course=self.boundary_owner.course;self.base=self.original.base;self.scene_unit_m=float(self.original.scene_unit_m)
        self.initial=np.r_[original.initial,np.zeros(2)];self.parameter_names=(*original.parameter_names,*EXTRA_NAMES)
        self.parameter_units=(*original.parameter_units,'m','dimensionless');self._last_extra=None

    def __getattr__(self,name):
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(51,)or not np.isfinite(q).all():raise ValueError('Complete authenticated old49 plus independent Ry/p world51 required')
        key=q[49:].tobytes()
        if self._last_extra!=key:self.original._scene_cache.clear();self._last_extra=key
        self.boundary_owner.activate(q[49:]);return q

    def validate_parameters(self,q):
        q=self._q(q);self.original.validate_parameters(q[:49]);return q.copy()
    def current_course6(self,q):return self.boundary_owner.current_course6(self._q(q)[:47])
    def course_jacobian(self,q):
        self._q(q);J=np.zeros((6,51));J[:,43:47]=self.boundary_owner.course_chain;J[3,49]=1.;J[4,50]=1.;return J
    def print_course_binding(self,q):return self.course,self.current_course6(q),self.course_jacobian(q)
    def source_scene(self,q):
        q=self._q(q);scene=self.original.source_scene(q[:49]);scene[SCHEMA]=self.manifest(q);return scene
    def scene(self,q):
        q=self._q(q);scene=self.original.scene(q[:49]);scene[SCHEMA]=self.manifest(q);return scene
    def semantic_world(self,target,q,*,samples=256):return world_curve(self.source_scene(q),target,samples=samples,scene_unit_m=self.scene_unit_m)
    def feature_state(self,q):
        scene=self.source_scene(q);return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in scene['ice_features']},ice_boundary=scene['ice_boundary'],parameters=scene['parameters'],world_geometry_manifest=self.manifest(q))
    def one_scene_source_world_bundle(self,q,scene,rows):
        self._q(q);return [world_curve(scene,row['target'],scene_unit_m=self.scene_unit_m)for row in rows]
    def one_scene_trapezoid_ends(self,q,scene,spans):
        self._q(q);return finite_trapezoid_ends_from_current_scene(scene,self.scene_unit_m,spans)
    def trapezoid_side_segments(self,q,**spans):return self.original.trapezoid_side_segments(self._q(q)[:49],**spans)
    def _named_ad_world_and_jacobian(self,row,q):
        q=self.validate_parameters(q);value,j29=_named_ad_course_value(self.boundary_owner,row,q[:47]);J=np.zeros((*value.shape,51))
        J[:,:,:23]=j29[:,:,:23];J[:,:,43:47]=np.einsum('nik,kj->nij',j29[:,:,23:],self.boundary_owner.course_chain)
        J[:,:,49:51]=j29[:,:,[26,27]];return value,J
    def rebind(self,row,q):value,_=self._named_ad_world_and_jacobian(row,q);out=dict(row);out['world']=value;return out
    def world_jacobian(self,row,q):return self._named_ad_world_and_jacobian(row,q)[1]
    def bind_world_jacobian(self,row,q,*,columns=None,step=1e-6,samples=256):
        q=self._q(q);ids=np.arange(51)if columns is None else np.asarray(columns,int)
        if ids.ndim!=1 or np.any(ids<0)or np.any(ids>=51)or len(np.unique(ids))!=len(ids)or not np.isfinite(step)or step<=0.:raise ValueError('Explicit unique whole51 columns and positive FD width required')
        target=row.get('target')
        if not isinstance(target,dict):raise ValueError('Original admitted semantic target required')
        world=self.semantic_world(target,q,samples=samples);J=np.empty((*world.shape,len(ids)))
        for j,k in enumerate(ids):
            plus=q.copy();minus=q.copy();plus[k]+=step;minus[k]-=step;J[:,:,j]=(self.semantic_world(target,plus,samples=samples)-self.semantic_world(target,minus,samples=samples))/(2*step)
        return world,J
    def finite_board_records(self,records,q,*,original_planes=None):
        q=self.validate_parameters(q);result=self.boundary_owner.transporter.finite_board_records(records,np.r_[q[:23],self.current_course6(q)],original_planes=original_planes)
        out=dict(result)
        for key in ('plane_jacobian','extent_world_jacobian','z_range_course_jacobian'):
            if key in out:out[key]=np.einsum('...k,kj->...j',out[key],self.course_jacobian(q))
        out['geometry_columns']='authenticated world51 full6 course; programme route not installed';return out
    def plane_and_extent_columns(self,j6,q):return np.einsum('...k,kj->...j',np.asarray(j6,float),self.course_jacobian(q))
    def manifest(self,q):
        q=self._q(q);return dict(self.boundary_owner.manifest(q[:47]),parameter_count=51,parent_count=49,new_indices=[49,50],new_parameter_names=list(EXTRA_NAMES),
            added_deltas=q[49:].tolist(),course_physical_chain_rank=int(np.linalg.matrix_rank(self.course_jacobian(q))),zero_added_deltas_preserve_old_source_world=True,
            original_frame_and_source_parameters_still_active=True)

class StructuredSharedIceTemplateFullBoundaryGeometry(StructuredSharedIceTemplateGeometry):
    parameter_count=50
    world_batch_protocol_schema=WORLD_BATCH_PROTOCOL
    def __init__(self,original):
        if type(original)is not StructuredSharedIceTemplateGeometry:raise TypeError('Explicit authenticated structured48 provider required')
        self.__dict__=deepcopy(original.__dict__);self.boundary_owner=_FullBoundary47Core.from_original(self.boundary_owner)
        self.initial=np.r_[original.initial,np.zeros(2)];self.parameter_names=(*original.parameter_names,*EXTRA_NAMES)
        self.parameter_units=(*original.parameter_units,'m','dimensionless');self.index={n:i for i,n in enumerate(self.parameter_names)};self._cache=OrderedDict()

    def _q(self,p):
        p=np.asarray(p,float)
        if p.shape!=(50,)or not np.isfinite(p).all():raise ValueError('Complete structured48 plus independent Ry/p latent50 required')
        self.boundary_owner.activate(p[48:]);return p
    def current_course6(self,p):p=self._q(p);return self.boundary_owner.current_course6(self._boundary_q(p))
    @property
    def course(self):return self.boundary_owner.course
    def course_jacobian(self,p):
        self._q(p);J=np.zeros((6,50));J[:,44:48]=self.boundary_owner.course_chain;J[3,48]=1.;J[4,49]=1.;return J
    def print_course_binding(self,p):return self.course,self.current_course6(p),self.course_jacobian(p)
    def _physical_boundary_and_prints(self,p,rebuild_grid):
        owner=self.boundary_owner;q47=self._boundary_q(p);cq=owner.current_course6(q47)
        scene=deepcopy(self.parent_scene);q29=np.r_[q47[:23],cq]
        with owner.transporter._current_old_domain(q29):
            scene=owner.transporter.original.panel.scene(p[28:32],scene=scene)
            scene=owner.transporter.original.pair_panels.scene(p[32:44],scene=scene)
        payload=owner.course.scene(cq,rebuild_grid=rebuild_grid)
        for k in ('parameters','ice_boundary','board_surface','meshes','boundary_validity_margins','shared_geometry_checks'):scene[k]=payload[k]
        if rebuild_grid:scene['ice_grid']=payload['ice_grid']
        if np.any(p[44:48]!=0.)or np.any(p[48:]!=0.):owner._transport_ads(scene,q47)
        return scene
    def _named_ad_world_and_jacobian(self,row,p):
        p=self.validate_parameters(p);value,j29=_named_ad_course_value(self.boundary_owner,row,self._boundary_q(p));J=np.zeros((*value.shape,50))
        J[:,:,28:44]=j29[:,:,7:23];J[:,:,44:48]=np.einsum('nik,kj->nij',j29[:,:,23:],self.boundary_owner.course_chain);J[:,:,48:50]=j29[:,:,[26,27]]
        return value,J
    def one_scene_source_world_bundle(self,p,scene,rows):
        self._q(p);return [semantic_world_from_current_scene(scene,row['target'],scene_unit_m=self.scene_unit_m)for row in rows]
    def one_scene_trapezoid_ends(self,p,scene,spans=None):
        self._q(p)
        if spans is not None:raise ValueError('Structured50 already owns its shared span2')
        return finite_trapezoid_ends_from_current_scene(scene,self.scene_unit_m)
    def engineering_reference_rows(self,p,*,sigma_by_name):
        from .structured_shared_ice_template import NOMINAL_REFERENCES
        p=self._q(p)
        if not isinstance(sigma_by_name,dict):raise TypeError('Caller explicit engineering tolerances required')
        r=[];J=[];metadata=[]
        for name,sigma in sigma_by_name.items():
            if name not in NOMINAL_REFERENCES or not np.isfinite(sigma)or sigma<=0.:raise ValueError('Known nominal relation and explicit positive engineering tolerance required')
            k=self.index[name];r.append((p[k]-NOMINAL_REFERENCES[name])/sigma);row=np.zeros(50);row[k]=1/sigma;J.append(row)
            metadata.append(dict(name=name,reference=NOMINAL_REFERENCES[name],engineering_sigma=float(sigma),not_measured_source_uncertainty=True))
        return np.asarray(r),np.asarray(J).reshape(-1,50),dict(role='explicit_engineering_reference_only',quadratic=True,relations=metadata,
            venue_version_unmeasured=True,metric_scale_uncertainty_shared_correlated_not_independent_marker_measurements=True)
    def bind_world_jacobian(self,row,p,*,columns=None,step=1e-6,samples=256):
        p=self._q(p);ids=np.arange(50)if columns is None else np.asarray(columns,int)
        if ids.ndim!=1 or np.any(ids<0)or np.any(ids>=50)or len(np.unique(ids))!=len(ids)or not np.isfinite(step)or step<=0.:raise ValueError('Explicit unique whole50 columns and positive FD width required')
        target=row.get('target')
        if not isinstance(target,dict):raise ValueError('Original admitted semantic target required')
        world=self.semantic_world(target,p,samples=samples);J=np.empty((*world.shape,len(ids)))
        for j,k in enumerate(ids):
            plus=p.copy();minus=p.copy();plus[k]+=step;minus[k]-=step;J[:,:,j]=(self.semantic_world(target,plus,samples=samples)-self.semantic_world(target,minus,samples=samples))/(2*step)
        return world,J
    def finite_board_records(self,records,p,*,original_planes=None):
        p=self.validate_parameters(p);q47=self._boundary_q(p);result=self.boundary_owner.transporter.finite_board_records(records,np.r_[q47[:23],self.current_course6(p)],original_planes=original_planes)
        out=dict(result)
        for key in ('plane_jacobian','extent_world_jacobian','z_range_course_jacobian'):
            if key in out:out[key]=np.einsum('...k,kj->...j',out[key],self.course_jacobian(p))
        out['geometry_columns']='authenticated structured50 full6 course; programme route not installed';return out
    def plane_and_extent_columns(self,j6,p):return np.einsum('...k,kj->...j',np.asarray(j6,float),self.course_jacobian(p))
    def manifest(self,p):
        p=self._q(p);return dict(self.boundary_owner.manifest(self._boundary_q(p)),parameter_count=50,parent_count=48,new_indices=[48,49],new_parameter_names=list(EXTRA_NAMES),
            added_deltas=p[48:].tolist(),course_physical_chain_rank=int(np.linalg.matrix_rank(self.course_jacobian(p))),shared_trapezoid_spans_in_slots=[10,11],
            old_structured_coupled_marking_relations_kept=True,zero_added_deltas_preserve_old_source_world=True,expected58_features_still_UV0_not_FIT=True)
