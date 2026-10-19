"""Current-course prints: same-brand sizes shared; observed faces placed separately.

The released print12 V2 stays intact.  Parent named Ad1/2/6/7 four-parameter
charts supply common width/height.  Ad3/4/5 keep their independent print12.
The legacy Ad14/20 columns are retained. Additional unverified placements
and admitted frozen print templates extend that chart without moving it.
Unobserved positions remain disclosed provisional initializers, not facts.
"""
from copy import deepcopy,copy
import numpy as np
from ad_landmarks import AdGeometry, ROLES
from .shared_unmeasured_advertisement_geometry_v2 import SharedUnmeasuredAdvertisementGeometryV2
from .shared_unmeasured_advertisement_geometry import current_course_binding, CURVE_ROLES

SCHEMA='rinkspace.current-course-paired-print-common-size-independent-locations.v1'
PAIRS=((1,14),(2,15),(3,16),(4,17),(5,18),(6,19),(7,20))
LOCATIONS=(14,20)

def _restore_paired_geometry(cls,state):
    # Preserve actual constructor materials/initial locations exactly.  A
    # reconstructed parent can intentionally discard transient scene caches;
    # it must not reinitialize our physical locations from a changed cache.
    obj=cls.__new__(cls);obj.__dict__.update(state);return obj

def _current_station(course,cq,point):
 bottom,top,*_=course.sample_perimeter(cq,with_jacobian=True)
 xy=np.asarray(bottom)[:,:2];edges=np.roll(xy,-1,axis=0)-xy
 length=np.linalg.norm(edges,axis=1);l2=length*length
 if np.any(l2<=0):raise ValueError('Nondegenerate current finite course required')
 t=np.clip(np.einsum('ij,ij->i',np.asarray(point)[:2]-xy,edges)/l2,0.,1.)
 hits=xy+t[:,None]*edges;d2=np.sum((hits-np.asarray(point)[:2])**2,axis=1);i=int(np.argmin(d2))
 if d2[i]>1e-14:raise ValueError('Current print is off actual finite course; no image-derived snap')
 cum=np.r_[0.,np.cumsum(length)]
 return float((cum[i]+t[i]*length[i])%cum[-1]),float(cum[-1])

class PairedLocationsSharedSizesGeometry(SharedUnmeasuredAdvertisementGeometryV2):
    paired_material_protocol_schema='rinkspace.same-size-paired-original-materials-independent-station-zmin.v1'

    def __init__(self,parent,parent_initial_q,*,additional_independent_locations=(),source_print_templates=(),independent_lane_right_extensions=False,strict_half_circle_crease=False,derive_hockey_center_from_course=False,independent_goal_crease_transverse_placements=False,shared_hockey_curling_axis=False):
        super().__init__(parent,parent_initial_q)
        self.print12_parameter_count=self.parameter_count
        self.old.update({n:AdGeometry(self.base,n)for pair in PAIRS for n in pair if n not in self.old})
        for pair in PAIRS:
            for n in pair:
                for role,world in self._ad(n).get('fitted_boundary_curves_xyz',{}).items():
                    if role in CURVE_ROLES:
                        self._materials[n,role]=self._material_coordinates(np.asarray(world,float),n,role)
        self._primary_indices={}
        for n,other in ((1,14),(2,15),(6,19),(7,20)):
            fields=('station_center_m','width_m'if n==1 else'arc_width_m','z_center_m','height_m')
            expected=tuple(f'Ad{n}_{other}_{field}'for field in fields)
            try:indices=np.array([self.parameter_names.index(name)for name in expected],int)
            except ValueError as exc:raise ValueError('Explicit current named shared size/primary location columns required:'+str(expected))from exc
            if np.any(indices>=self.parent_parameter_count):raise ValueError('Common parent size columns must be in physical parent prefix')
            self._primary_indices[n]=indices
        initial_scene=self.source_scene(self.initial)
        ads={int(ad['user_number']):ad for ad in initial_scene['advertisements']}
        course,cq,_=current_course_binding(self.parent,self.parent_initial_q)
        self._provisional_locations={};self.partner_extent_scope={};self._materials.update({})
        for n,other in PAIRS:
            lhs=self._ad(n).get('extent_scope');rhs=self._ad(other).get('extent_scope')
            if lhs!=rhs or lhs not in('external_border','internal_graphic_only'):
                raise ValueError(f'Explicit same material extent scope required for {n}/{other}; not brand alone')
            self.partner_extent_scope[n,other]=dict(primary_scope=lhs,partner_scope=rhs,
                common_size_is_latest_user_model_instruction_not_survey=True,
                counterpart_same_extent_qualification_not_inferred_from_brand=True,
                existing_model_scope_is_not_new_source_qualification=True)
            self.initial_materials[other]=dict(width_sign=int(self.old[other]._width_sign),original_extent_scope=rhs)
            ad=ads[other];corners=ad['corners_xyz_scene_units']
            left,L=_current_station(course,cq,corners['left_top']);right,_=_current_station(course,cq,corners['right_top'])
            signed=(right-left+L/2)%L-L/2;station=((left+signed/2)%L)*self.scene_unit_m
            self._provisional_locations[other]=np.array([station,float(corners['left_bottom'][2])*self.scene_unit_m])
            for role,world in self._ad(other).get('fitted_boundary_curves_xyz',{}).items():
                if role in CURVE_ROLES:self._materials[other,role]=self._material_coordinates(np.asarray(world,float),other,role)
        self.location_indices={n:np.arange(self.print12_parameter_count+2*i,self.print12_parameter_count+2*i+2)for i,n in enumerate(LOCATIONS)}
        self.parameter_names=(*self.parameter_names,*(f'Ad{n}_independent_{field}'for n in LOCATIONS for field in('station_center_m','zmin_m')))
        self.parameter_units=(*self.parameter_units,'m','m','m','m')
        self.parameter_count=self.print12_parameter_count+4
        self.initial=np.r_[self.initial,*(self._provisional_locations[n]for n in LOCATIONS)]
        self.validate_parameters(self.initial)
        self.location_initialization_contract=dict(schema='rinkspace.current-parent-counterpart-location-material-initialization.v1',
            station_and_zmin_from_current_parent_finite_course=True,new_location_parameters=list(LOCATIONS),
            unobserved_provisional_locations={str(n):v.tolist()for n,v in self._provisional_locations.items()if n not in LOCATIONS},
            no_exact_halfturn_location_or_equal_bottom_height=True,
            common_width_height_not_fixed_to_historical_values=True,
            source_UV_sigma_role_or_qualification_modified=False,
            initial_old_parent_values_or_J_not_claimed_exact_where_parent_material_mapping_was_wrong=True)
        self.original_world66_parameter_count=self.parameter_count
        self.additional_independent_locations=tuple(int(n)for n in additional_independent_locations)
        if (len(set(self.additional_independent_locations))!=len(self.additional_independent_locations)
                or any(n not in(15,16,17,18,19)for n in self.additional_independent_locations)):
            raise ValueError('Additional counterpart locations must name distinct existing provisional materials')
        self.location_offset_indices={}
        self.source_print_templates={};self.source_template_instance={}
        names=list(self.parameter_names);units=list(self.parameter_units);initial=list(self.initial)
        for n in self.additional_independent_locations:
            indices=np.arange(len(initial),len(initial)+2)
            self.location_indices[n]=indices;self.location_offset_indices[n]=indices
            names.extend(f'Ad{n}_independent_{axis}_offset_from_provisional_initializer_m'for axis in('station','zmin'))
            units.extend(('m','m'));initial.extend((0.,0.))
        for literal in source_print_templates:
            record=deepcopy(literal);identifier=str(record['id'])
            a,b=int(record['canonical_advertisement_number']),int(record['counterpart_advertisement_number'])
            primary,partner=sorted((a,b))
            if (primary,partner)not in((8,21),(9,22),(10,23))or identifier in self.source_print_templates:
                raise ValueError('One qualified normalized template for each admitted irregular pair required')
            init=record['initialization'];width,height=map(float,init['shared_width_height_m'])
            if not np.isfinite([width,height]).all()or min(width,height)<=0:
                raise ValueError('Positive finite inferred q0 template dimensions required')
            indices=np.arange(len(initial),len(initial)+6)
            record['width_height_offset_indices']=indices[:2]
            record['instance_location_offset_indices']={primary:indices[2:4],partner:indices[4:6]}
            names.extend((f'Ad{primary}_{partner}_shared_width_offset_from_inferred_q0_m',f'Ad{primary}_{partner}_shared_height_offset_from_inferred_q0_m',
                f'Ad{primary}_station_offset_from_inferred_q0_m',f'Ad{primary}_zmin_offset_from_inferred_q0_m',
                f'Ad{partner}_station_offset_from_inferred_q0_m',f'Ad{partner}_zmin_offset_from_inferred_q0_m'))
            units.extend(('m',)*6);initial.extend((0.,)*6)
            self.source_print_templates[identifier]=record
            for n in(a,b):
                datum=np.asarray(init['instance_datum_station_zmin_m'][str(n)],float)
                if datum.shape!=(2,)or not np.isfinite(datum).all():raise ValueError('Finite inferred instance station/Z datum required')
                self.source_template_instance[n]=identifier
        self.parameter_names=tuple(names);self.parameter_units=tuple(units)
        self.parameter_count=len(initial);self.initial=np.asarray(initial,float)
        self.derive_hockey_center_from_course=bool(derive_hockey_center_from_course)
        self.shared_hockey_curling_axis=bool(shared_hockey_curling_axis)
        self.strict_half_circle_crease=bool(strict_half_circle_crease)
        if self.strict_half_circle_crease or self.derive_hockey_center_from_course:
            self.parent=copy(self.parent)
            self.parent._strict_half_circle_crease=self.strict_half_circle_crease
            self.parent._derive_hockey_center_from_course=self.derive_hockey_center_from_course
        self.lane_right_extension_indices={}
        if independent_lane_right_extensions:
            self.lane_right_extension_indices={f'house_connection_F{i}_R{i}':self.parameter_count+i-1 for i in range(1,5)}
            self.parameter_names=(*self.parameter_names,*(f'Lane{i}_right_extension_fraction_offset_from_user_initializer'for i in range(1,5)))
            self.parameter_units=(*self.parameter_units,*(('fraction',)*4))
            self.parameter_count+=4;self.initial=np.r_[self.initial,np.zeros(4)]
        self.goal_crease_transverse_indices={}
        if independent_goal_crease_transverse_placements:
            self.goal_crease_transverse_indices={side:self.parameter_count+i for i,side in enumerate(('left','right'))}
            self.parameter_names=(*self.parameter_names,*(f'goal_crease_{side}_transverse_offset_from_unsurveyed_goal_axis_m'for side in ('left','right')))
            self.parameter_units=(*self.parameter_units,'m','m')
            self.parameter_count+=2;self.initial=np.r_[self.initial,np.zeros(2)]
        self.location_initialization_contract.update(
            additional_free_counterpart_locations=list(self.additional_independent_locations),
            additional_locations_are_offsets_not_location_observations=True,
            inferred_source_template_ids=list(self.source_print_templates),
            independent_lane_right_extension_parameter_indices=dict(self.lane_right_extension_indices),
            strict_half_circle_crease=self.strict_half_circle_crease,
            derive_hockey_center_from_course=self.derive_hockey_center_from_course,
            shared_hockey_curling_axis=self.shared_hockey_curling_axis,
            shared_axis_binding=(dict(active_named_column=14,active_name='curling_common_yaw_rad',
                derived_hockey_axis_column=2,derived_name='hockey_yaw_rad',
                legacy_column_is_saved_initializer_history_not_an_independent_yaw=True,
                Lane_long_edges_perpendicular_to_C_B1_B2_in_world=True,
                structural_user_fact_not_image_angle_constraint=True)
                if self.shared_hockey_curling_axis else None),
            independent_goal_crease_transverse_placements=bool(self.goal_crease_transverse_indices),
            goal_crease_transverse_parameter_indices=dict(self.goal_crease_transverse_indices),
            goal_crease_placement_policy='paint centre may move along its current goal-line carrier; common radius and goal/trapezoid/faceoff frames retained',
            hockey_center_structure=(dict(authority="user_confirmed_C0_is_current_course_center",
                formula="midpoint of current finite-course XY bounds in working metres",
                archived_legacy_initializer_columns=[0,1],
                legacy_initializer_columns_are_not_effective_physical_degrees_of_freedom=True)
                if self.derive_hockey_center_from_course else None),
            actual_physical_parameter_count=self.parameter_count,
            original_world66_logical_prefix_preserved=True)
        self.validate_parameters(self.initial)

    def __reduce__(self):return(_restore_paired_geometry,(type(self),self.__dict__))

    def _center_data(self,q):
        course,cq,chain=current_course_binding(self.parent,q[:self.parent_parameter_count])
        key=cq.tobytes()
        cache=self.__dict__.setdefault('_derived_course_center_cache',{})
        if key not in cache:
            bottom,_,jb,*_=course.sample_perimeter(cq,with_jacobian=True)
            xy=np.asarray(bottom,float)[:,:2]
            lower=np.argmin(xy,axis=0);upper=np.argmax(xy,axis=0)
            axes=np.arange(2)
            center=.5*(xy[lower,axes]+xy[upper,axes])*self.scene_unit_m
            derivative=.5*(jb[lower,axes,:]+jb[upper,axes,:])*self.scene_unit_m
            if len(cache)>=128:cache.clear()
            cache[key]=(center,derivative)
        center,derivative=cache[key]
        return center,derivative@chain

    def effective_world(self,q):
        """Apply confirmed center and common-axis relations to material values.

        The archived two initializer slots remain in the canonical state. They
        are not measurements or independent physical coordinates in this mode.
        All source and export values consume this same structural mapping.
        """
        q=super()._q(q)
        center=getattr(self,'derive_hockey_center_from_course',False)
        axis=getattr(self,'shared_hockey_curling_axis',False)
        if not center and not axis:return q
        value=q.copy()
        if center:value[:2]=self._center_data(q)[0]
        if axis:value[2]=value[14]
        return value

    def effective_world_chain(self,q):
        q=super()._q(q);chain=np.eye(self.parameter_count)
        if getattr(self,'derive_hockey_center_from_course',False):
            chain[:2]=0.;chain[:2,:self.parent_parameter_count]=self._center_data(q)[1]
        if getattr(self,'shared_hockey_curling_axis',False):
            chain[2]=0.;chain[2,14]=1.
        return chain

    def _q(self,q):return self.effective_world(q)

    def _size(self,n,q):
        q=self._q(q);primary=n if n<14 else n-13
        jw=np.zeros(self.parameter_count);jh=np.zeros(self.parameter_count)
        if n in getattr(self,'source_template_instance',{}):
            record=self.source_print_templates[self.source_template_instance[n]]
            indices=record['width_height_offset_indices']
            width,height=np.asarray(record['initialization']['shared_width_height_m'],float)+q[indices]
            jw[indices[0]]=1.;jh[indices[1]]=1.
            return float(width),float(height),jw,jh
        if primary in(3,4,5):
            k=self.parent_parameter_count+4*self.slot[primary]
            width=float(q[k+1]);height=float(q[k+3]-q[k+2]);jw[k+1]=1.;jh[k+3]=1.;jh[k+2]=-1.
        else:
            indices=self._primary_indices[primary]
            width=float(q[indices[1]]);height=float(q[indices[3]]);jw[indices[1]]=1.;jh[indices[3]]=1.
        return width,height,jw,jh

    def _material(self,n,u,v,q):
        if n in getattr(self,'source_template_instance',{}):
            return self.source_template_material(n,u,v,q)
        if not hasattr(self,'location_indices')or n in(3,4,5):
            return super()._material(n,u,v,q)
        q=self._q(q)
        if n not in self.old or n not in [x for pair in PAIRS for x in pair]:raise ValueError('Explicit original paired print required')
        u,v=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float));u,v=u.reshape(-1),v.reshape(-1)
        width,height,jw,jh=self._size(n,q);ds=np.zeros(self.parameter_count);dz=np.zeros(self.parameter_count)
        if n<14:
            indices=self._primary_indices[n];station=float(q[indices[0]]);zmin=float(q[indices[2]]-height/2)
            ds[indices[0]]=1.;dz[indices[2]]=1.;dz-=jh/2
        elif n in self.location_indices:
            indices=self.location_indices[n];station,zmin=q[indices]
            if n in getattr(self,'location_offset_indices',{}):station,zmin=np.asarray(self._provisional_locations[n])+q[indices]
            ds[indices[0]]=1.;dz[indices[1]]=1.
        else:station,zmin=self._provisional_locations[n]
        sign=self.old[n]._width_sign
        stations=station+sign*(u-.5)*width;z=zmin+v*height
        course,cq,chain=current_course_binding(self.parent,q[:self.parent_parameter_count])
        world,J6,Js,report=course.station(stations,cq,z_m=z,with_jacobian=True)
        J=np.zeros((*world.shape,self.parameter_count));J[:,:,:self.parent_parameter_count]=np.einsum('nik,kj->nij',J6,chain)
        dstation=ds[None,:]+sign*(u-.5)[:,None]*jw[None,:]
        J+=Js[:,:,None]*dstation[:,None,:]
        J[:,2,:]+=(dz[None,:]+v[:,None]*jh[None,:])/self.scene_unit_m
        return world,J,report

    def material(self,n,u,v,q):return self._material(int(n),u,v,q)

    def source_template_material(self,n,u,v,q):
        """One inferred printed template, current course and estimable placement."""
        n=int(n);q=self._q(q);record=self.source_print_templates[self.source_template_instance[n]]
        u,v=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float));u,v=u.reshape(-1),v.reshape(-1)
        if not np.isfinite(u).all()or not np.isfinite(v).all():raise ValueError('Finite normalized material coordinates required')
        width,height,jw,jh=self._size(n,q)
        datum=np.asarray(record['initialization']['instance_datum_station_zmin_m'][str(n)],float)
        indices=record['instance_location_offset_indices'][n];station,zmin=datum+q[indices]
        # The source packet's datum is the developed bbox minimum. Descriptor
        # centres outside the ink bbox are retained; no normalized-coordinate
        # clamping or fictitious rectangular print boundary is introduced.
        stations=station+u*width;z=zmin+v*height
        course,cq,chain=current_course_binding(self.parent,q[:self.parent_parameter_count])
        world,J6,Js,report=course.station(stations,cq,z_m=z,with_jacobian=True)
        J=np.zeros((*world.shape,self.parameter_count));J[:,:,:self.parent_parameter_count]=np.einsum('nik,kj->nij',J6,chain)
        ds=u[:,None]*jw[None,:];ds[:,indices[0]]+=1.
        J+=Js[:,:,None]*ds[:,None,:]
        dz=v[:,None]*jh[None,:];dz[:,indices[1]]+=1.
        J[:,2,:]+=dz/self.scene_unit_m
        return world,J,dict(report,material_template_id=record['id'],source_world_shape_inferred_only=True)

    def source_template_contours(self,n,q):
        record=self.source_print_templates[self.source_template_instance[int(n)]]
        return [self.source_template_material(n,*np.asarray(c['initialization']['material_coordinates_normalized'],float).T,q)[0]
                for c in record['external_contours']]
    def material_coordinates(self,original_source_world,n,role):
        return self._material_coordinates(original_source_world,int(n),role)

    def material_parameter_indices(self,n):
        n=int(n);_,_,chain=current_course_binding(self.parent,self.parent_initial_q)
        cols=set(np.flatnonzero(np.any(chain!=0,axis=0)).tolist())
        if n in getattr(self,'source_template_instance',{}):
            record=self.source_print_templates[self.source_template_instance[n]]
            cols.update(record['width_height_offset_indices'].tolist());cols.update(record['instance_location_offset_indices'][n].tolist())
        elif n in(3,4,5):cols.update(range(self.parent_parameter_count+4*self.slot[n],self.parent_parameter_count+4*self.slot[n]+4))
        else:
            _,_,jw,jh=self._size(n,self.initial);cols.update(np.flatnonzero((jw!=0)|(jh!=0)).tolist())
            if n<14:cols.update(self._primary_indices[n].tolist())
            elif n in self.location_indices:cols.update(self.location_indices[n].tolist())
        return np.array(sorted(cols),int)

    def validate_parameters(self,q):
        if not hasattr(self,'location_indices'):return super().validate_parameters(q)
        q=self._q(q);self.parent.validate_parameters(q[:self.parent_parameter_count])
        course,cq,_=current_course_binding(self.parent,q[:self.parent_parameter_count]);bottom,top,*_=course.sample_perimeter(cq,with_jacobian=True)
        perimeter=float(np.linalg.norm(np.roll(bottom,-1,axis=0)[:,:2]-bottom[:,:2],axis=1).sum())*self.scene_unit_m
        low=float(np.min(bottom[:,2]))*self.scene_unit_m;high=float(np.max(top[:,2]))*self.scene_unit_m
        for identity,k in getattr(self,'lane_right_extension_indices',{}).items():
            if 108./945.+q[k]<0:raise ValueError(f'{identity}: nonnegative independent rightward paint extent required')
        for n in self._qualified_domain_instances():
            width,height,_,_=self._size(n,q)
            if width<=0 or width>=perimeter/2 or height<=0:raise ValueError(f'Ad{n}: positive common width/height required')
            zmin,_=self._instance_zmin(n,q)
            vlow,vhigh=self._material_vertical_extent(n)
            if zmin+vlow*height<low-1e-8 or zmin+vhigh*height>high+1e-8:raise ValueError(f'Ad{n}: current qualified print exceeds current board Z domain')
        return q.copy()

    def _qualified_domain_instances(self):
        return (1,2,3,4,5,6,7,14,20,*getattr(self,'additional_independent_locations',()),*getattr(self,'source_template_instance',{}))

    def ice_extent_state(self,q):
        q=self._q(q)
        return dict(independent_lane_right_extent_fractions={identity:float(108./945.+q[k])for identity,k in getattr(self,'lane_right_extension_indices',{}).items()})

    def apply_goal_crease_transverse_placements(self,scene,q):
        """One paint-placement producer for full export and source ice views.

        The axis-centred position was an unsurveyed template initializer.
        Translation along the goal carrier preserves the semicircle, its
        baseline attachment and the independently supported goal/trap frame.
        """
        indices=getattr(self,'goal_crease_transverse_indices',{})
        if not indices:return scene
        q=self._q(q);across=np.array([-np.sin(q[2]),np.cos(q[2]),0.])
        features={f['id']:f for f in scene['ice_features']}
        import cv2
        boundary=np.ascontiguousarray(np.asarray(scene['ice_boundary']['points_xyz'])[:,:2],np.float32)
        for side,column in indices.items():
            feature=features['goal_crease_'+side]
            points=np.asarray(feature['polygon_xyz'],float)
            if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
                raise ValueError('Finite existing goal-crease paint polygon required')
            offset=float(q[column]);shift=offset*across/self.scene_unit_m
            for point in points+shift:
                # Same float32 footprint roundoff used by the parent for
                # actual Lane paint. This is containment, not a placement
                # prior or an arbitrary numeric search interval.
                if cv2.pointPolygonTest(boundary,tuple(map(float,point[:2])),True)<-2e-5:
                    raise ValueError('Goal-crease paint leaves actual ice footprint: '+side)
            # Zero initialization preserves every existing material value.
            if offset!=0.:feature['polygon_xyz']=(points+shift).tolist()
            centre=.5*(points[0]+points[-1])+shift
            feature['current_crease_placement']=dict(
                transverse_offset_working_m=offset,logical_world_parameter_index=int(column),
                centre_xyz_scene_units=centre.tolist(),along_current_goal_carrier=True,
                original_axis_position_is_unsurveyed_initializer=True,
                common_radius_estimated=True,goal_trapezoid_and_faceoff_frames_unchanged=True)
        return scene

    def _instance_zmin(self,n,q):
        dz=np.zeros(self.parameter_count)
        if n in getattr(self,'source_template_instance',{}):
            rec=self.source_print_templates[self.source_template_instance[n]]
            k=rec['instance_location_offset_indices'][n][1]
            z=float(rec['initialization']['instance_datum_station_zmin_m'][str(n)][1]+q[k]);dz[k]=1.
        elif n in(3,4,5):
            k=self.parent_parameter_count+4*self.slot[n]+2;z=float(q[k]);dz[k]=1.
        elif n<14:
            _,height,_,jh=self._size(n,q);idx=self._primary_indices[n]
            z=float(q[idx[2]]-height/2);dz[idx[2]]=1.;dz-=jh/2
        else:
            k=self.location_indices[n][1];z=float(q[k]);dz[k]=1.
            if n in getattr(self,'location_offset_indices',{}):z+=float(self._provisional_locations[n][1])
        return z,dz

    def _material_vertical_extent(self,n):
        if n not in getattr(self,'source_template_instance',{}):return 0.,1.
        rec=self.source_print_templates[self.source_template_instance[n]]
        points=np.concatenate([np.asarray(c['initialization']['material_coordinates_normalized'],float)for c in rec['external_contours']])
        return float(points[:,1].min()),float(points[:,1].max())

    def print_vertical_domain_step_constraints(self,q):
        """Linearize the existing qualified-print Z guards in world columns.

        Return A @ delta >= lower.  These are feasible-step constraints, not
        source observations or priors.  The nonlinear validator remains the
        authority for every trial, with exactly its existing 1e-8 tolerance.
        Placements without observations remain free; finite board material
        bounds are physical feasibility checks, not placement observations.
        """
        q=self._q(q)
        course,cq,chain=self.print_course_binding(q)
        bottom,top,jb,jt,*_=course.sample_perimeter(cq,with_jacobian=True)
        ilow=int(np.argmin(bottom[:,2]));ihigh=int(np.argmax(top[:,2]))
        low=float(bottom[ilow,2])*self.scene_unit_m
        high=float(top[ihigh,2])*self.scene_unit_m
        dlow=np.asarray(jb[ilow,2],float)@chain*self.scene_unit_m
        dhigh=np.asarray(jt[ihigh,2],float)@chain*self.scene_unit_m
        rows=[];margins=[];bounds=[]
        for n in self._qualified_domain_instances():
            _,height,_,jh=self._size(n,q)
            zmin,dz=self._instance_zmin(n,q);vlow,vhigh=self._material_vertical_extent(n)
            for bound,margin,jac in(
                ('bottom',zmin+vlow*height-low+1e-8,dz+vlow*jh-dlow),
                ('top',high-zmin-vhigh*height+1e-8,dhigh-dz-vhigh*jh)):
                rows.append(jac);margins.append(margin)
                bounds.append(dict(ad_number=n,bound=bound,margin_m=float(margin)))
        # The already-enforced extension >= 0 is a one-sided physical domain,
        # not a finite working interval or an invented upper endpoint.
        for identity,column in getattr(self,'lane_right_extension_indices',{}).items():
            row=np.zeros(self.parameter_count);row[column]=1.
            margin=108./945.+q[column]
            rows.append(row);margins.append(margin)
            bounds.append(dict(paint_identity=identity,bound='nonnegative_rightward_extension',
                margin_fraction=float(margin),upper_extent_not_assumed=True))
        return dict(matrix=np.asarray(rows,float),lower=-np.asarray(margins,float),
            report=dict(kind='existing_qualified_print_vertical_and_nonnegative_paint_extent_domains',
                bounds=bounds,world_parameter_count=self.parameter_count,
                nonlinear_trial_guard_retained=True,new_source_or_prior_rows=False))

    def corner(self,n,role,q,*,with_jacobian=False):
        if not hasattr(self,'location_indices'):return super().corner(n,role,q,with_jacobian=with_jacobian)
        if role not in ROLES:raise ValueError('Original named material corner role required')
        side,level=role.split('_');w,J,_=self._material(int(n),float(side=='right'),float(level=='top'),q)
        return(w[0],J[0])if with_jacobian else w[0]

    def _world_and_jacobian(self,row,q):
        if not hasattr(self,'location_indices'):return super()._world_and_jacobian(row,q)
        q=self._q(q);ids=self.identities(row);original=np.asarray(row['world'],float)
        active={n for pair in PAIRS for n in pair}
        if not ids or all(n not in active for n,role in ids):
            value=np.asarray(self.parent.rebind(row,q[:self.parent_parameter_count])['world'],float)
            oldJ=self.parent.world_jacobian(row,q[:self.parent_parameter_count])
            J=np.concatenate((oldJ,np.zeros((*oldJ.shape[:-1],self.parameter_count-self.parent_parameter_count))),axis=-1)
            if getattr(self,'derive_hockey_center_from_course',False)or getattr(self,'shared_hockey_curling_axis',False):
                J=np.einsum('nik,kj->nij',J,self.effective_world_chain(q))
            return value,J
        if all(role in ROLES for n,role in ids):
            if len(ids)!=len(original):raise ValueError('Original named role count differs')
            value=original.copy();J=np.zeros((*original.shape,self.parameter_count))
            for i,(n,role)in enumerate(ids):
                if n not in active:raise ValueError('Mixed unrelated corner source needs explicit original authority dispatch')
                expected=np.asarray(self._ad(n)['corners_xyz_scene_units'][role])
                if not np.array_equal(original[i],expected):raise ValueError('Named corners require original role certificate binding; historical world is not a measured corner')
                value[i],J[i]=self.corner(n,role,q,with_jacobian=True)
            return value,J
        if len(ids)==1 and ids[0][0]in active and ids[0][1]in CURVE_ROLES:
            n,role=ids[0];u,v=self.material_coordinates(original,n,role)
            w,J,_=self.material(n,u,v,q);return w,J
        raise ValueError('Explicit qualified finite paired material required')

    def print_course_binding(self,q):
        course,cq,J=current_course_binding(self.parent,self._q(q)[:self.parent_parameter_count])
        return course,cq,np.column_stack((J,np.zeros((6,self.parameter_count-self.parent_parameter_count))))

    def engineering_reference_rows(self,q,**kwargs):
        r,J,meta=self.parent.engineering_reference_rows(self._q(q)[:self.parent_parameter_count],**kwargs)
        J=np.column_stack((J,np.zeros((len(J),self.parameter_count-self.parent_parameter_count))))
        if getattr(self,'derive_hockey_center_from_course',False)or getattr(self,'shared_hockey_curling_axis',False):J=J@self.effective_world_chain(q)
        return r,J,meta

    def _scene(self,q,rebuild_grid):
        if not hasattr(self,'location_indices'):return super()._scene(q,rebuild_grid)
        q=self.validate_parameters(q);out=self.parent.scene(q[:self.parent_parameter_count])if rebuild_grid else self.parent.source_scene(q[:self.parent_parameter_count])
        out=self.apply_goal_crease_transverse_placements(out,q)
        if getattr(self,'lane_right_extension_indices',{}):out.update(self.ice_extent_state(q))
        ads={int(ad['user_number']):ad for ad in out['advertisements']}
        for pair in PAIRS:
            for n in pair:
                ad=ads[n];width,height,_,_=self._size(n,q)
                ad['corners_xyz_scene_units']={role:self.corner(n,role,q).tolist()for role in ROLES}
                ad['fitted_boundary_curves_xyz']={role:self.material(n,*uv,q)[0].tolist()for(nn,role),uv in self._materials.items()if nn==n}
                zbottom=float(ad['corners_xyz_scene_units']['left_bottom'][2]);ztop=float(ad['corners_xyz_scene_units']['left_top'][2])
                ad.update(width_m=width,height_m=height,bottom_z_scene_units=zbottom,top_z_scene_units=ztop,
                    geometry_status='same_size_current_course_prints_independent_locations_candidate_not_measured',
                    position_provisional_no_independent_source_claim=n in(15,16,17,18,19))
                ad[SCHEMA]=dict(pair=pair,common_width_height_estimated=True,material_extent_scope=self._ad(n).get('extent_scope'),
                    station_or_bottom_height_not_hard_halfturn=True)
        for row in out.get('advertisement_3d_corners',[]):
            n=int(row['user_number'])
            if n in self.old:row['world_xyz_scene_units']=self.corner(n,row['role'],q).tolist()
        for n,template_id in getattr(self,'source_template_instance',{}).items():
            rec=self.source_print_templates[template_id];width,height,_,_=self._size(n,q)
            ad=ads.get(n)
            if ad is None:
                ad=dict(user_number=n,label=f'Ad{n}',source_geometry_role='actual admitted source print contours')
                out['advertisements'].append(ad);ads[n]=ad
            contours=self.source_template_contours(n,q)
            # These are actual admitted source contours. Bbox corners are a
            # material datum and must not become a fictitious exterior edge.
            ad.pop('corners_xyz_scene_units',None);ad.pop('fitted_boundary_curves_xyz',None)
            ad['outline_polylines_xyz_scene_units']=[w.tolist()for w in contours]
            ad['external_contours']=[dict(id=c.get('id',str(i)),world_xyz_scene_units=w.tolist())for i,(c,w)in enumerate(zip(rec['external_contours'],contours))]
            ad.update(width_m=width,height_m=height,extent_scope='source_print_external_contours',
                geometry_status='frozen_source_inferred_template_current_estimated_dimensions_and_location',
                source_template_id=template_id,position_provisional_no_independent_source_claim=False)
            ad[SCHEMA]=dict(pair=[rec['canonical_advertisement_number'],rec['counterpart_advertisement_number']],
                common_width_height_estimated=True,independent_station_zmin_estimated=True,
                normalized_source_shape_inferred_from_explicit_q0=True,initializer_q_sha256=rec['initialization']['q0_sha256'],
                same_material_producer_as_source_residuals=True)
        if getattr(self,'source_template_instance',{}):
            out['advertisement_3d_corners']=[r for r in out.get('advertisement_3d_corners',[])if int(r['user_number'])not in self.source_template_instance]
        out[SCHEMA]=self.manifest(q);return out

    def manifest(self,q):
        if not hasattr(self,'location_indices'):return super().manifest(q)
        q=self._q(q)
        return dict(schema=SCHEMA,parent_world_parameter_count=self.parent_parameter_count,world_parameter_count=self.parameter_count,
            shared_hockey_curling_axis=getattr(self,'shared_hockey_curling_axis',False),
            effective_hockey_yaw_rad=float(q[2]),effective_curling_yaw_rad=float(q[14]),
            historical_hockey_yaw_slot_is_inactive=getattr(self,'shared_hockey_curling_axis',False),
            print12_parameters_conditional_metres=q[self.parent_parameter_count:self.print12_parameter_count].tolist(),
            independent_partner_location_coordinates={str(n):dict(values=q[idx].tolist(),indices=idx.tolist(),
                parameterization='offset_from_provisional_initializer'if n in getattr(self,'location_offset_indices',{})else'absolute')for n,idx in self.location_indices.items()},
            active_independent_station_zmin_parameters=2*len(self.location_indices)+2*len(getattr(self,'source_template_instance',{})),shared_sizes={str(pair):dict(width_m=self._size(pair[0],q)[0],height_m=self._size(pair[0],q)[1])for pair in PAIRS},
            source_print_templates={identifier:dict(width_height=self._size(rec['canonical_advertisement_number'],q)[:2],
                width_height_offset_indices=rec['width_height_offset_indices'].tolist(),
                instance_location_offset_indices={str(n):idx.tolist()for n,idx in rec['instance_location_offset_indices'].items()},
                source_initializer_q_sha256=rec['initialization']['q0_sha256'])for identifier,rec in getattr(self,'source_print_templates',{}).items()},
            sizes_shared_exactly_per_latest_user_instruction_not_hardware_measurement=True,
            all_counterpart_positions_not_forced_by_primary_halfturn=True,
            unobserved_counterpart_positions_provisional_not_source_or_hidden_H_constraint=True,
            Ad5_18_scope='internal_graphic_only; not whole-banner external border',
            counterpart_extent_scope_ledger={str(pair):v for pair,v in self.partner_extent_scope.items()},
            original_pixel_UV_sigma_roles_and_native_clocks_unchanged=True,no_new_source_qualification_or_prior=True,
            original_named_corner_roles_must_be_certified_separately_from_historical_world_arrays=True,
            current_actual_course_and_one_sided_continuous_material_mapping=True,
            initial_print12_contract=deepcopy(self.initialization_contract),location_initialization=deepcopy(self.location_initialization_contract),
            source_graph_installed=False,solver_run=False,accuracy_accepted=False)
