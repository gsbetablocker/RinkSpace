"""Qualified original programme sources on named66, common sizes/dual locations.

This new adapter intentionally does not modify or claim equivalence to the
62 prototype. Every paired material uses the same current producer as scene.
"""
from copy import deepcopy
from collections import OrderedDict
import hashlib,numpy as np
from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry,PAIRS
from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry_v2 import SharedUnmeasuredAdvertisementGeometryV2
from .structured_programme_geometry import StructuredProgrammeGeometry,_as_world
from .ad14_source_role_certificate import qualify_ad14_assigned_roles

SCHEMA='rinkspace.qualified-original-programme-materials-paired-location-named66.v1'
class PairedLocationProgrammeGeometry(StructuredProgrammeGeometry):
    parameter_count=66
    @staticmethod
    def qualified_scene(scene):
        from rinkspace.user_contours import apply_user_visible_markings
        return apply_user_visible_markings(scene,pairs=[('F1','R1'),('F2','R2'),('F3','R3'),('F4','R4')],
            authority='D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json')
    def scene(self,q):return self.qualified_scene(super().scene(q))
    def source_scene(self,q):return self.qualified_scene(super().source_scene(q))
    def _scene(self,q):
        view=self.__dict__.get('_source_ice_material_view')
        if view is None:return super()._scene(q)
        q=self._q(q)
        result=dict(ice_features=tuple(view.features(q).values()))
        if getattr(self.current,'lane_right_extension_indices',{}):result.update(self.current.ice_extent_state(q))
        return result
    def effective_world(self,q):return self._q(q)
    def validate_parameters(self,q):return self._q(q)
    def _q(self,q):
        """Validate each exact physical world once, never a stale nearby world.

        One linearization reuses the same world and declared FD perturbations
        across thousands of exposure rows.  Geometry/domain validation is a
        pure function of those bytes and does not depend on the image row.
        """
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():raise ValueError('Finite complete declared physical world required')
        key=q.tobytes();cache=self.__dict__.setdefault('_successful_world_validation',OrderedDict())
        if key in cache:return cache[key]
        value=self.current.validate_parameters(q);value.flags.writeable=False;cache[key]=value
        while len(cache)>256:cache.popitem(last=False)
        return value
    def __init__(self,current_geometry66,original_geometry43,*,world_fd_step=1e-6):
        g=current_geometry66
        if type(g)is not PairedLocationsSharedSizesGeometry or g.parameter_count<66 or g.paired_material_protocol_schema!='rinkspace.same-size-paired-original-materials-independent-station-zmin.v1':raise TypeError('Explicit original world66 prefix and declared independent material coordinates required')
        if g.location_indices[14].tolist()!=[62,63] or g.location_indices[20].tolist()!=[64,65] or g.location_initialization_contract['common_width_height_not_fixed_to_historical_values']is not True or g.location_initialization_contract['no_exact_halfturn_location_or_equal_bottom_height']is not True:raise ValueError('Current common width/height and distinct observed partner placements required')
        prefix=SharedUnmeasuredAdvertisementGeometryV2(g.parent,g.parent_initial_q)
        super().__init__(prefix,original_geometry43,world_fd_step=world_fd_step)
        self._qualification_prefix=StructuredProgrammeGeometry(prefix,original_geometry43,world_fd_step=world_fd_step)
        self.current=g;self.parameter_count=g.parameter_count;self.initial=g.initial.copy();self.parameter_names=tuple(g.parameter_names);self.parameter_units=tuple(g.parameter_units)
        self._value_cache=OrderedDict();self._scene_cache=OrderedDict()
        self._companion_alias_certificates={};self.companion_alias_ledger=[]

    def _register_companions(self,row,b,key,actual_pts):
        """Certify report aliases per source literal, apart from material identity.

        The same physical material occurs in fit, holdout and pending sources.
        Its historical report aliases need not be identical across those rows.
        The solver consumes only the explicitly declared material fields.
        """
        for field in ('world_xyz','world_xyz_scene_units'):
            if field not in row:continue
            old=np.asarray(row[field],float);world=np.asarray(row['world'],float)
            ref=world[0]if old.shape==(3,)and len(world)==1 else world
            if old.shape!=ref.shape:raise ValueError('Original companion support shape differs')
            original_sha=hashlib.sha256(old.tobytes()).hexdigest();token=(key,field,original_sha)
            if token in self._companion_alias_certificates:continue
            certificate=None
            for factor in (1.,self.scene_unit_m):
                target=ref*factor;floor=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(old),abs(target)))
                if np.all(abs(old-target)<=floor):
                    certificate=dict(scale=factor,maximum_roundoff=float(abs(old-target).max()),
                        support='Same registered source WORLD samples; float64 reconstruction roundoff only')
                    break
            if certificate is None:
                relation=row.get('world_scene_binding',{})
                if (field=='world_xyz'and field not in b['fields']and
                    row.get('split')in('holdout','unfitted_identity_pending')and
                    b['kind']=='circle'and b['name']=='F2'and relation.get('feature')=='F2'and
                    relation.get('shared_complete_F_column_translation')is True and
                    relation.get('source_UV_role_and_sigma_changed')is False and
                    original_sha==row.get('original_world_sha256')):
                    certificate=dict(scale=1.,maximum_original_alias_difference=float(abs(old-ref).max()),
                        support='Authenticated historical report alias before explicit shared F-column translation; current physical source fields are WORLD only')
                else:raise ValueError('Unqualified original companion coordinate relationship: '+str(row.get('id')))
            certificate.update(original_sha256=original_sha,original_source_id=row.get('id'),
                actual_pts=actual_pts,source_split=row.get('split'),field=field,
                source_WORLD_sha256=hashlib.sha256(world.tobytes()).hexdigest(),
                active_source_fields=list(b['fields']),source_UV_sigma_support_and_split_changed=False)
            self._companion_alias_certificates[token]=certificate
            self.companion_alias_ledger.append(certificate)

    def register_row(self,row,*,actual_pts=None,allow_previously_qualified_held_ad2_representation=False):
        key=self._key(row)
        if key in self.bindings:
            self._register_companions(row,self.bindings[key],key,actual_pts)
            return self.bindings[key]['contract']
        if row.get('physical_ad_id')==14:
            u,v,authority=qualify_ad14_assigned_roles(row,actual_pts=actual_pts);world=_as_world(row)
            b=dict(kind='ad',ids=[(14,r)for r in row['physical_world_corner_roles']],material_coordinates=(u,v),fields=('world',),normalized=deepcopy(row),
                contract=dict(original_source_id=row['id'],actual_pts=actual_pts,split=row['split'],source_world_sha256=hashlib.sha256(world.tobytes()).hexdigest(),source_UV_sigma_role_support_order_changed=False,physical_identities=[(14,r)for r in row['physical_world_corner_roles']],original_assigned_corner_role_certificate=authority,world_literal_is_conditional_model_not_identity_measurement=True))
        else:
            self._qualification_prefix.register_row(row,actual_pts=actual_pts,allow_previously_qualified_held_ad2_representation=allow_previously_qualified_held_ad2_representation)
            b=deepcopy(self._qualification_prefix._binding(row)[1])
            if b['kind']=='ad':
                ids=b['ids']
                if not all(n in{n for pair in PAIRS for n in pair}for n,role in ids):raise ValueError('Explicit current producer for every original physical print required')
                if not all(role in('left_top','right_top','left_bottom','right_bottom')for n,role in ids):
                    if len(ids)!=1:raise ValueError('One qualified finite material edge required')
                    n,role=ids[0];b['material_coordinates']=self.current.material_coordinates(np.asarray(b['normalized']['world']),n,role)
        if b['kind']=='ad':
            columns=set()
            for n,_ in b['ids']:columns.update(self.current.material_parameter_indices(n).tolist())
            b['columns']=tuple(sorted(columns));b['contract'].update(sizes_shared_per_latest_user_rule=True,current_scene_and_source_use_identical_material_producer=True,partner_station_zmin_not_hard_halfturn=True)
        elif b.get('name')in getattr(self.current,'lane_right_extension_indices',{}):
            b['columns']=tuple(sorted(set(b['columns'])|{self.current.lane_right_extension_indices[b['name']]}))
        self._register_companions(row,b,key,actual_pts)
        self.bindings[key]=b;self.registration_ledger.append(b['contract']);return b['contract']

    def rebind(self,row,q):
        key,b=self._binding(row);normalized=dict(row)
        for field in ('world_xyz','world_xyz_scene_units'):
            if field not in row:continue
            old=np.asarray(row[field],float)
            token=(key,field,hashlib.sha256(old.tobytes()).hexdigest())
            if token not in self._companion_alias_certificates:raise ValueError('Original qualified companion field changed or unregistered')
            certificate=self._companion_alias_certificates[token]
            world=np.asarray(row['world'],float);ref=world[0]if old.shape==(3,)and len(world)==1 else world
            normalized[field]=(ref*certificate['scale']).tolist()
        return super().rebind(normalized,q)

    def _ad_value(self,b,q,jacobian):
        q=self._q(q);ids=b['ids']
        if all(role in('left_top','right_top','left_bottom','right_bottom')for _,role in ids):
            value=np.empty((len(ids),3));J=np.empty((len(ids),3,self.parameter_count))
            for i,(n,role)in enumerate(ids):
                side,level=role.split('_');w,j,_=self.current.material(n,float(side=='right'),float(level=='top'),q);value[i],J[i]=w[0],j[0]
            return value,J
        n,role=ids[0];u,v=b['material_coordinates'];w,j,_=self.current.material(n,u,v,q);return w,j

    def _current_material_columns(self,b):
        columns=b['columns']
        if getattr(self.current,'shared_hockey_curling_axis',False)and b['kind']not in('ad','course'):
            columns=tuple(sorted({14 if k==2 else k for k in columns}))
        return columns

    def world_jacobian(self,row,q):
        q=self._q(q);key,b=self._binding(row);token=(key,q.tobytes())
        if token in self._value_cache:return self._value_cache[token]
        if b['kind']=='course':
            _,J6=b['material'].evaluate(self.current.current_course6(q),with_jacobian=True);J=np.einsum('nik,kj->nij',J6,self.current.print_course_binding(q)[2])
        elif b['kind']=='ad':J=self._ad_value(b,q,True)[1]
        else:
            J=np.zeros((len(row['world']),3,self.parameter_count));h=self.fd_step
            for k in self._current_material_columns(b):
                plus,minus=q.copy(),q.copy();plus[k]+=h;minus[k]-=h;J[:,:,k]=(self._material_value(b,plus)-self._material_value(b,minus))/(2*h)
        J.flags.writeable=False;self._value_cache[token]=J
        while len(self._value_cache)>128:self._value_cache.popitem(last=False)
        return J
    def source_parameter_activity(self,row,q):
        q=self._q(q);_,b=self._binding(row);active=np.zeros(self.parameter_count,bool)
        if b['kind']=='course':
            a6=b['material'].course_parameter_activity(self.current.current_course6(q));active=np.any((self.current.print_course_binding(q)[2]!=0)&a6[:,None],axis=0)
        else:active[list(self._current_material_columns(b))]=True
        return active
    geometry_parameter_activity=source_parameter_activity
    def finite_board_records(self,records,q,*,original_planes):
        # Extend the released exact17 protocol: plane/extent fields are current
        #66, including sourceAd4. No new parameter can hide outside validation.
        q=self._q(q);prefix=self._qualification_prefix
        # The shared finite-plane geometry is unaffected by Ad14/20 placements.
        old=prefix.finite_board_records(records,q[:62],original_planes=original_planes)
        for field in('plane_jacobian','extent_world_jacobian','z_range_world_jacobian'):old[field]=np.concatenate((old[field],np.zeros((*old[field].shape[:-1],self.parameter_count-62))),axis=-1)
        return old
    def manifest(self,q):
        q=self._q(q);return dict(schema=SCHEMA,parameter_count=self.parameter_count,original_world66_prefix_count=66,parameters=q.tolist(),parameter_names=self.parameter_names,current_geometry=self.current.manifest(q),
            registered_unique_material_count=len(self.bindings),original_native_UV_sigma_roles_operators_and_counts_preserved=True,same_pair_width_height_estimable_shared=True,Ad14_20_only_add_independent_station_zmin4=True,
            qualified_original_companion_aliases=self.companion_alias_ledger,
            no_new_source_from_missing_15_19_or_template_features=True,original14_role_certificate_separate_from_old_model_WORLD=True,whole_programme_blocks_patterns_external_gate_not_installed=True,accuracy_accepted=False)
