"""Uninstalled world29: original q23 plus an actual shared course6.

All pixels, factor roles and source confidence remain outside this world
layer. Explicit original material identity is required for course binding.
"""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import numpy as np
from ad_landmarks import AdGeometry,ROLES
from shared_advertisement_pairs_candidate.shared_rink_numeric_geometry_lanes_ad_pairs import SharedRinkNumericGeometryLanesAdPairs
from shared_advertisement_pairs_candidate.shared_advertisement_pairs_numeric_geometry import DEFAULT_PAIRS,CURVE_ROLES
from .shared_board_course_numeric_geometry import SharedBoardCourseNumericGeometry
from .board_course_material import transverse_intersections,vertical_plane_value_jacobian

VERSION='shared_rink_course_numeric_geometry_v1'
TRANSVERSE_IDS=('C','B1','B2','goal_line_left','goal_line_right')


def _restore(cls,base,unit,pairs,registered):
    instance=cls(base,unit,pairs=pairs)
    instance.registered=registered
    # Existing world-only certificates contain their own immutable course.
    return instance


class SharedRinkCourseNumericGeometry:
    """Seven rink deltas, four Ad1/14, twelve paired prints, six course.

    The original prefix is delegated exactly when the course is at its
    initial values. Its existing priors gain six zeros, never new prior rows.
    All existing modeled panels use the same current finite course function.
    """
    parameter_count=29

    def __init__(self,base,scene_unit_m=None,*,pairs=DEFAULT_PAIRS):
        self.original=SharedRinkNumericGeometryLanesAdPairs(base,scene_unit_m,pairs=pairs)
        self.base=self.original.base
        self.scene_unit_m=self.original.scene_unit_m
        self.course=SharedBoardCourseNumericGeometry(self.base,self.scene_unit_m)
        self.initial=np.r_[self.original.initial,self.course.initial]
        self.parameter_names=(*self.original.parameter_names,*self.course.parameter_names)
        self.course_local_indices=np.arange(23,29)
        self.registered={}
        self.ads={int(a['user_number']):a for a in self.base['advertisements']}
        self.arcs={n:AdGeometry(self.base,n)for n in self.ads if n<=13}
        self.features={f['id']:f for f in self.base['ice_features']}
        self._transverse={}
        for name in TRANSVERSE_IDS:
            if name not in self.features:continue
            endpoints=np.asarray(self.features[name]['segments_xyz'],float)
            if endpoints.shape!=(1,2,3):raise ValueError('One original finite transverse segment required')
            transverse_intersections(self.course,endpoints[0])
            self._transverse[name]=endpoints[0].copy()
        for n,a in self.ads.items():
            lo=n-13 if n>13 else n
            if lo not in self.arcs:raise ValueError('Existing print lacks original low-number course identity')
            for role,world in a['fitted_boundary_curves_xyz'].items():
                if role in CURVE_ROLES:self._certify_ad(np.asarray(world,float),n,role)

    def __getattr__(self,name):
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def __reduce__(self):
        return _restore,(type(self),self.base,self.scene_unit_m,self.original.pair_panels.pairs,self.registered)

    @property
    def panel(self):
        # External source binding gets a separately bound course view. The
        # old prefix generator always uses original.panel, never this view.
        return self.__dict__.get('_current_external_panel',self.original.panel)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(29,)or not np.isfinite(q).all():raise ValueError('Complete original23 + absolute course6 required')
        # Basic structural checks, while every physical domain is evaluated on
        # the current course in validate_parameters, rather than stale bounds.
        self.original.rink._q(q[:7])
        self.course.validate_parameters(q[23:])
        for n in self.arcs:
            panel=self._panel_q(n,q)
            if panel[1]<=0 or panel[3]<=0:raise ValueError('Original positive panel width/height required')
        return q

    @contextmanager
    def _current_old_domain(self,q):
        """Private old geometry gets current guards, restored even on failure.

        No original module/object shared with the active graph is changed.
        Generation still uses the original course; only guard domains update.
        """
        import cv2
        b,t,_=self.course.sample_perimeter(q[23:]);polygon=b[:,:2].astype(np.float32)
        top=q[28]/self.scene_unit_m
        old_rink_boundary=self.original.rink._boundary
        old_mid_base=self.original.rink.midpoint.base
        old_panel_top=self.original.panel.board_top_m
        pair_tops={p:arc._board_top for p,arc in self.original.pair_panels.arc.items()}
        # Midpoint validation reads only its base's ice boundary; keep every
        # other original metadata/world property at the untouched source base.
        midbase=dict(old_mid_base);midbase['ice_boundary']=dict(old_mid_base['ice_boundary'],points_xyz=b.tolist())
        self.original.rink._boundary=polygon
        self.original.rink.midpoint.base=midbase
        self.original.panel.board_top_m=q[28]
        for arc in self.original.pair_panels.arc.values():arc._board_top=top
        try:yield
        finally:
            self.original.rink._boundary=old_rink_boundary
            self.original.rink.midpoint.base=old_mid_base
            self.original.panel.board_top_m=old_panel_top
            for p,arc in self.original.pair_panels.arc.items():arc._board_top=pair_tops[p]

    def validate_parameters(self,q):
        q=self._q(q)
        b,_,_=self.course.sample_perimeter(q[23:])
        perimeter=float(np.sum(np.linalg.norm(np.roll(b,-1,axis=0)-b,axis=1)))*self.scene_unit_m
        for n in self.arcs:
            panel=self._panel_q(n,q)
            if panel[1]>=perimeter/2:raise ValueError('Panel must retain positive minor physical-course width')
            if panel[2]-panel[3]/2< -1e-8 or panel[2]+panel[3]/2>q[28]+1e-8:
                raise ValueError('Original absolute print Z exceeds current conditional board height')
        with self._current_old_domain(q):
            self.original.validate_parameters(q[:23])
        return q.copy()

    def _panel_q(self,low,q):
        if low==1:return q[7:11]
        if low in self.original.pair_panels.slot:
            p=self.original.pair_panels.slot[low];return q[11+4*p:15+4*p]
        arc=self.arcs[low]
        return np.array([(arc._base_s['left']+arc._width_sign*arc._nominal_width/2)*self.scene_unit_m,
            arc._nominal_width*self.scene_unit_m,
            (arc._base_height['bottom']+arc._base_height['top'])*self.scene_unit_m/2,
            (arc._base_height['top']-arc._base_height['bottom'])*self.scene_unit_m])

    def _certify_ad(self,world,number,role):
        low=number-13 if number>13 else number
        arc=self.arcs[low]
        original=np.asarray(world,float).copy()
        if number>13:original[:,:2]*=-1
        if original.ndim!=2 or original.shape[1]!=3 or not np.isfinite(original).all():raise ValueError('Original finite print material required')
        stations=np.array([arc._nearest_arc(w[:2])for w in original])
        xy=np.array([arc._point_at_arc(s)for s in stations])
        if np.max(np.linalg.norm(xy-original[:,:2],axis=1))>1e-7:
            raise ValueError('Original named ad material is an off-course chord; no automatic correction')
        left=arc._base_s['left'];width=arc._width_sign*arc._nominal_width
        u=((stations-left+arc._perimeter/2)%arc._perimeter-arc._perimeter/2)/width
        v=(original[:,2]-arc._base_height['bottom'])/(arc._base_height['top']-arc._base_height['bottom'])
        if np.any(u< -1e-7)or np.any(u>1+1e-7)or np.any(v< -1e-7)or np.any(v>1+1e-7):
            raise ValueError('Original named material exceeds its existing finite print domain')
        if role in ROLES:
            side,level=role.split('_')
            if np.max(abs(u-float(side=='right')))>1e-7 or np.max(abs(v-float(level=='top')))>1e-7:
                raise ValueError('Original world is not its declared existing print corner')
        elif role.startswith(('left_','right_')):
            if np.max(abs(u-float(role.startswith('right_'))))>1e-7:raise ValueError('Wrong existing vertical print material')
        elif role.startswith(('top_','bottom_')):
            if np.max(abs(v-float(role.startswith('top_'))))>1e-7:raise ValueError('Wrong existing horizontal print material')
        else:raise ValueError('Explicit original print role required')
        return stations*self.scene_unit_m,u,v

    def _transport_current_ad(self,world,jold,number,q,*,physical_curve=False):
        """One same function for source and render current finite world knots."""
        lo=number-13 if number>13 else number;arc=self.arcs[lo]
        canonical=np.asarray(world,float).copy();jcanonical=np.asarray(jold,float).copy()
        if number>13:canonical[:,:2]*=-1;jcanonical[:,:2]*=-1
        s=np.array([arc._nearest_arc(w[:2])for w in canonical])
        old_xy=np.array([arc._point_at_arc(v)for v in s])
        if np.max(np.linalg.norm(old_xy-canonical[:,:2],axis=1))>1e-7:
            raise ValueError('Generated old q23 world is not finite course material')
        i=np.minimum(np.searchsorted(arc._cum,s%arc._perimeter,side='right')-1,len(arc._starts)-1)
        old_tangent=arc._edges[i]/arc._lengths[i,None]
        js=np.einsum('ni,nik->nk',old_tangent,jcanonical[:,:2])*self.scene_unit_m
        current,jc,tangent,_=self.course.station(s*self.scene_unit_m,q[23:],z_m=canonical[:,2]*self.scene_unit_m,with_jacobian=True)
        jac=np.zeros((len(current),3,29))
        jac[:,:,:23]=tangent[:,:,None]*js[:,None,:]
        jac[:,2,:23]=jcanonical[:,2]
        jac[:,:,23:]=jc
        if physical_curve:
            # Existing explicitly inserted boundary vertices are physical
            # finite break knots, not unknown print-corner measurements.
            base=self.course.sample_perimeter()[0]
            now,_,jnow,_,_=self.course.sample_perimeter(q[23:],with_jacobian=True)
            for k,w in enumerate(canonical):
                exact=np.flatnonzero(np.linalg.norm(base[:,:2]-w[:2],axis=1)<1e-10)
                if len(exact)==1 and np.all(jcanonical[k,:2]==0):
                    v=int(exact[0]);current[k,:2]=now[v,:2];jac[k,:2,23:]=jnow[v,:2]
        if number>13:current[:,:2]*=-1;jac[:,:2]*=-1
        if np.array_equal(q[23:],self.course.initial):
            current=np.asarray(world,float).copy();jac[:,:,:23]=jold
        return current,jac

    def _row_key(self,row):
        world=np.ascontiguousarray(np.asarray(row['world'],float))
        companions=tuple((key,np.shape(row[key]),hashlib.sha256(np.ascontiguousarray(np.asarray(row[key],float)).tobytes()).hexdigest())
            for key in ('upper_world',)if key in row)
        return (str(row.get('id')),str(row.get('physical_curve_id')),world.shape,hashlib.sha256(world.tobytes()).hexdigest(),companions)

    def register_course_row(self,row,binding):
        """Explicitly install a prior authenticated world certificate only."""
        if not np.array_equal(binding.old_world,np.asarray(row['world'],float)):
            raise ValueError('Certificate is not the same original finite source world')
        if not np.array_equal(binding.geometry.initial,self.course.initial):
            raise ValueError('Certificate uses a different physical base course')
        self.registered[self._row_key(row)]=binding

    def _transverse_identity(self,row):
        candidates=[row.get('physical_curve_id'),row.get('feature_id'),row.get('physical_feature_id'),row.get('id')]
        target=row.get('target',{})
        if isinstance(target,dict):candidates.append(target.get('feature_id'))
        names=[n for n in candidates if n in self._transverse]
        return names[0]if names else None

    def feature_id(self,row):
        ids=self.original.pair_panels.identities(row)
        if ids and any(n in self.ads for n,_ in ids):return 'shared_course_ad_material'
        name=self._transverse_identity(row)
        if name:return 'shared_course_transverse_'+name
        if self._row_key(row)in self.registered:return 'shared_course_explicit_finite_material'
        return self.original.feature_id(row)

    def _world_and_jacobian(self,row,q):
        q=self._q(q)
        old,jold=self.original._world_and_jacobian(row,q[:23])
        ids=self.original.pair_panels.identities(row)
        if ids and any(n in self.ads for n,_ in ids):
            original=np.asarray(row['world'],float)
            if all(role in ROLES for _,role in ids):
                if len(ids)!=len(original):raise ValueError('One physical print identity per original corner required')
                world=old.copy();jac=np.concatenate((jold,np.zeros((len(old),3,6))),axis=2)
                for k,(n,role)in enumerate(ids):
                    if n not in self.ads:continue
                    self._certify_ad(original[k:k+1],n,role)
                    w,j=self._transport_current_ad(old[k:k+1],jold[k:k+1],n,q)
                    world[k],jac[k]=w[0],j[0]
                return world,jac
            if len(ids)!=1 or ids[0][1]not in CURVE_ROLES:raise ValueError('One original physical finite print edge required')
            n,role=ids[0];self._certify_ad(original,n,role)
            return self._transport_current_ad(old,jold,n,q,physical_curve=True)
        name=self._transverse_identity(row)
        if name:
            original=np.asarray(row['world'],float);ends=self._transverse[name];d=ends[1]-ends[0]
            t=(original-ends[0])@d/(d@d)
            if np.any(t< -1e-7)or np.any(t>1+1e-7)or np.max(np.linalg.norm(original-(ends[0]+t[:,None]*d),axis=1))>1e-7:
                raise ValueError('Declared transverse source is outside its original finite painted material')
            clipped,jends,_=transverse_intersections(self.course,ends,q[23:])
            world=(1-t[:,None])*clipped[0]+t[:,None]*clipped[1]
            jac=np.zeros((len(world),3,29));jac[:,:,23:]=(1-t[:,None,None])*jends[0]+t[:,None,None]*jends[1]
            if np.array_equal(q[23:],self.course.initial):world=old.copy();jac[:,:,:23]=jold
            return world,jac
        binding=self.registered.get(self._row_key(row))
        if binding is not None:
            world,j=binding.evaluate(q[23:],with_jacobian=True)
            if not np.array_equal(old,np.asarray(row['world']))or np.any(jold):
                raise ValueError('Course material overlaps an old moved physical entity; needs explicit combined certificate')
            return world,np.concatenate((jold,j),axis=2)
        return old,np.concatenate((jold,np.zeros((len(old),3,6))),axis=2)

    def rebind(self,row,q):
        world,_=self._world_and_jacobian(row,q)
        out=dict(row);out['world']=world if isinstance(row['world'],np.ndarray)else world.tolist()
        for key in ('world_xyz','world_xyz_scene_units'):
            if key not in row:continue
            shape=np.asarray(row[key]).shape
            if shape==(3,)and len(world)==1:out[key]=world[0].tolist()
            elif shape==world.shape:out[key]=world.tolist()
            else:raise ValueError('Source companion world requires explicit same-shape binding')
        binding=self.registered.get(self._row_key(row))
        if binding is not None and hasattr(binding,'companion_values'):
            for key,current in binding.companion_values(self._q(q)[23:]).items():
                if key not in row or np.shape(row[key])!=np.shape(current):
                    raise ValueError('Declared course companion finite sample count changed')
                out[key]=current if isinstance(row[key],np.ndarray)else current.tolist()
        return out

    def world_jacobian(self,row,q):return self._world_and_jacobian(row,q)[1]

    def source_parameter_fields(self,row):
        binding=self.registered.get(self._row_key(row))
        return ('world',)+tuple(getattr(binding,'companions',{}))

    def source_parameter_activity(self,row,q):
        """Real activity union for typed factors reading companion geometry.

        world_jacobian remains the exact PRIMARY world derivative. This
        independent activity API never fabricates a nonzero primary column.
        """
        q=self._q(q)
        active=np.any(self.world_jacobian(row,q)!=0.,axis=(0,1))
        binding=self.registered.get(self._row_key(row))
        if binding is not None and hasattr(binding,'course_parameter_activity'):
            active[23:]|=binding.course_parameter_activity(q[23:])
        return active

    geometry_parameter_activity=source_parameter_activity

    def finite_board_records(self,records,q=None,*,original_planes=None):
        """Actual admitted main1 named straight / low Ad4 finite material.

        Preserve original support endpoints and orientation at initial course.
        This world/plane helper installs no motion row. Unknown identities fail.
        It cannot turn a curved supported arc into one invented flat plane.
        """
        q=self._q(self.initial if q is None else q)
        planes=[];jplanes=[];updated=[];extent_j=[]
        originals=None if original_planes is None else np.asarray(original_planes,float)
        if originals is not None and originals.shape!=(len(records),4):raise ValueError('Original finite planes row count changed')
        for row_index,record in enumerate(records):
            if record.get('surface_kind')!='finite_board_face':raise ValueError('Only declared board finite rows enter this helper')
            ends=np.c_[np.asarray([record['extent_start_xy'],record['extent_end_xy']],float),np.zeros(2)]
            identity=record.get('physical_surface_identity')
            if isinstance(identity,dict)and 'upper_long_side'in identity.get('scene_boundary_parts',[]):
                expected=self.course.primitive('upper_long_side',[1.,0.])
                if not np.array_equal(ends,expected):raise ValueError('Main named-side tangent extent differs from actual source record')
                world,j=self.course.primitive('upper_long_side',[1.,0.],q[23:],with_jacobian=True)
            elif identity=='ad4_existing_straight_far_board_face':
                expected=np.asarray([self.ads[4]['corners_xyz_scene_units'][r]for r in('left_bottom','right_bottom')],float)
                if np.max(abs(ends[:,:2]-expected[:,:2]))>1e-9:raise ValueError('Low source extent is not original named Ad4 material')
                world,jall=self._transport_current_ad(ends,np.zeros((2,3,23)),4,q)
                j=jall[:,:,23:]
                # It was qualified as a straight material plane. A moved
                # station crossing a current corner must be rejected by this
                # original admission, not silently assigned a chord plane.
                a,b,rx,_,_,_=self.course._scene_q(q[23:])
                if np.any(abs(world[:,1]-b)>1e-8)or np.any(abs(world[:,0])>a-rx+1e-8):
                    raise ValueError('Qualified finite Ad4 material has left its actual upper-long straight domain')
            else:raise ValueError('Unimplemented finite board identity requires source-specific material certificate')
            plane,jplane=vertical_plane_value_jacobian(world[0],world[1],j[0],j[1])
            oldplane=vertical_plane_value_jacobian(ends[0],ends[1],np.zeros((3,6)),np.zeros((3,6)))[0]
            if originals is not None:
                if not np.allclose(originals[row_index],oldplane,rtol=0,atol=1e-9):raise ValueError('Actual original nX+d=0 plane does not match its physical finite extent')
                if np.array_equal(q[23:],self.course.initial):plane=originals[row_index].copy()
            if np.array_equal(q[23:],self.course.initial):world=ends.copy()
            out=deepcopy(record);out['extent_start_xy']=world[0,:2].tolist();out['extent_end_xy']=world[1,:2].tolist()
            updated.append(out);planes.append(plane);jplanes.append(jplane);extent_j.append(j)
        return dict(planes=np.asarray(planes).reshape(-1,4),plane_jacobian=np.asarray(jplanes).reshape(-1,4,6),
            records=updated,extent_world_jacobian=np.asarray(extent_j).reshape(-1,2,3,6),
            z_range_scene=np.array([0.,self.course._scene_q(q[23:])[5]]),
            z_range_course_jacobian=np.array([[0.]*6,[0.,0.,0.,0.,0.,1/self.scene_unit_m]]),
            source_role_row_index_native_UV_and_static_identity_unchanged=True,
            unknown_depth_relabelled=False,motion_kernel_installed=False)

    def midpoint_prior(self,q):
        q=self._q(q);r,j=self.original.midpoint_prior(q[:23])
        return r,np.c_[j,np.zeros((len(r),6))]

    def project(self,camera,row,q,*,with_jacobian=True):
        from projection import project_central_valid
        world,j=self._world_and_jacobian(row,q)
        if not with_jacobian:return project_central_valid(camera,world)
        uv,depth,jc,valid=project_central_valid(camera,world,with_jacobian=True)
        return uv,depth,np.einsum('nij,njk->nik',-jc.reshape(len(world),2,-1)[:,:,:3]/100.,j),valid

    def manifest(self,q=None):
        q=self._q(self.initial if q is None else q)
        value=self.original.manifest(q[:23])
        value.update(version=VERSION,parameter_count=29,parameter_names=list(self.parameter_names),
            parameters=q.tolist(),course_local_indices=self.course_local_indices.tolist(),
            shared_board_course=dict(self.course.contract,current_parameters=q[23:].tolist()),
            original23_parameter_order_unchanged=True,additional_prior_rows=0,
            all_existing_modeled_ad_numbers=sorted(self.ads),unmodeled_ads_added=False,
            original_course_initial_prefix_world_and_J_exact=True,
            source_pixels_UV_sigma_role_actual_clock_and_aliases_unchanged=True,
            finite_world_route='explicit print role/station, certified course t, or authenticated transverse wall endpoints',
            generic_off_course_chord_corrected=False,
            Lane_left_proxy='Unmeasured display proxy remains unchanged; current footprint guard rejects outside paint/proxy. No wall endpoint identity invented.',
            finite_board_motion_and_guard_installed=False,full_source_graph_installed=False,accuracy_accepted=False)
        return value

    def source_scene(self,q):
        """Identical physical source world, without review-grid regeneration.

        The historical grid is not a source factor; its inherited cells must
        not be used for a review of a noninitial course. Use scene() for that.
        """
        return self.scene(q,rebuild_grid=False)

    def scene(self,q,*,rebuild_grid=True):
        q=self.validate_parameters(q)
        with self._current_old_domain(q):scene=self.original.scene(q[:23])
        if np.array_equal(q[23:],self.course.initial):
            # Preserve original prefix exporter geometry, including its old
            # finite sample knots, exactly. Metadata is appended separately.
            scene[VERSION]=self.manifest(q);return scene
        cscene=self.course.scene(q[23:],rebuild_grid=rebuild_grid)
        for key in ('parameters','ice_boundary','board_surface','meshes','ice_grid','boundary_validity_margins'):
            scene[key]=cscene[key]
        scene['shared_geometry_checks']=cscene['shared_geometry_checks']
        features={f['id']:f for f in scene['ice_features']}
        for name,ends in self._transverse.items():
            features[name]['segments_xyz']=[transverse_intersections(self.course,ends,q[23:])[0].tolist()]
        for ad in scene['advertisements']:
            n=int(ad['user_number'])
            ad['corners_xyz_scene_units']={role:self._transport_current_ad(np.asarray(w,float)[None],np.zeros((1,3,23)),n,q)[0][0].tolist()
                for role,w in ad['corners_xyz_scene_units'].items()}
            ad['fitted_boundary_curves_xyz']={role:self._transport_current_ad(np.asarray(w,float),np.zeros((len(w),3,23)),n,q,physical_curve=True)[0].tolist()
                for role,w in ad['fitted_boundary_curves_xyz'].items()}
            ad['current_course_material_binding']=dict(physical_source_role_retained=True,station_units='conditional metres',
                original_finite_sample_count_retained=True,print_z_m_retained=True,unmodeled_edge_created=False)
        corners={int(a['user_number']):a['corners_xyz_scene_units']for a in scene['advertisements']}
        for row in scene.get('advertisement_3d_corners',[]):
            if int(row['user_number'])in corners:row['world_xyz_scene_units']=corners[int(row['user_number'])][row['role']]
        scene[VERSION]=self.manifest(q)
        scene[VERSION]['review_grid_rebuilt']=bool(rebuild_grid)
        return scene
