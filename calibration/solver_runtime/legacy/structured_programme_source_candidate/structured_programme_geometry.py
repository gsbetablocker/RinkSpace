"""Qualified original programme materials on the owned structured62 world.

This is a geometry/source binding protocol, not a complete-source installation.
Every row must first pass an original WORLD/material identity certificate.
No image inversion, source resampling, point admission or tolerance change.
"""
from __future__ import annotations
from copy import deepcopy
from collections import OrderedDict
import hashlib,re
import numpy as np
from ad_landmarks import ROLES
from shared_board_course_candidate.shared_rink_course_numeric_geometry import SharedRinkCourseNumericGeometry
from shared_board_course_candidate.course_absolute_material import declared_absolute_course_identity,certify_absolute_course_material
from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
from shared_ice_hockey_geometry_candidate.shared_ice_hockey_numeric_geometry import SharedIceHockeyNumericGeometry
from shared_unmeasured_advertisement_geometry_candidate.shared_unmeasured_advertisement_geometry_v2 import SharedUnmeasuredAdvertisementGeometryV2
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import _authority,_same_origin,_same_array,AUTHORITIES

SCHEMA='rinkspace.qualified-structured62-programme-material-geometry.v1'
COLUMNS_COURSE=tuple(range(44,50))

def _as_world(row):
    w=np.asarray(row['world'],float)
    if w.ndim!=2 or w.shape[1]!=3 or not len(w)or not np.isfinite(w).all():raise ValueError('Finite nonempty original source WORLD required')
    return w

def _finite_t(world,ends):
    a,b=np.asarray(ends,float);d=b-a;den=d@d
    if den<=0.:raise ValueError('Nondegenerate original finite material required')
    t=(world-a)@d/den
    if np.any(t< -1e-8)or np.any(t>1+1e-8)or np.max(np.linalg.norm(world-a-t[:,None]*d,axis=1))>1e-7:
        raise ValueError('Original WORLD is not on its declared finite material; no snap or widening')
    return t

def _qualified_main9_ad3(row):
    """The actual one original admitted Ad3 right-edge source, no new UV."""
    if row.get('id')!='ad3_right_source_boundary':return None
    packet=_authority('main9_ad2');pts=int(row.get('source_native_pts',row.get('pts',-1)))
    matches=[r for r in packet['groups']if r['id']==row['id']and int(r['pts'])==pts]
    if len(matches)!=1 or row.get('pts')!=pts or row.get('point_correspondence')is not False:raise ValueError('Original Ad3 source exposure/correspondence changed')
    original=matches[0];_same_origin(row,'main9_ad2')
    if row.get('split')!=original['split']:raise ValueError('Original Ad3 FIT/HELD role changed')
    for key in('world','uv','normal','normal_interval'):_same_array(row,key,original.get(key,original.get({'normal':'source_normal_uv','normal_interval':'normal_interval_px'}.get(key,''))))
    sigma=original.get('sigma')
    if sigma is None:sigma=np.full(len(original['uv']),float(original['sigma_px']))
    _same_array(row,'sigma',sigma)
    out=dict(row);out['physical_ids']=['ad3:right_vertical_xyz']
    return out,dict(path=str(AUTHORITIES['main9_ad2'][0]),sha256=AUTHORITIES['main9_ad2'][1],
        original_id=row['id'],physical_id='ad3:right_vertical_xyz',original_world_and_UV_sigma_role_exact=True)

class StructuredProgrammeGeometry:
    parameter_count=62
    def __init__(self,geometry,original_geometry43,*,world_fd_step=1e-6):
        if type(geometry)is not SharedUnmeasuredAdvertisementGeometryV2 or geometry.parameter_count!=62 or type(original_geometry43)is not SharedIceHockeyNumericGeometry or original_geometry43.parameter_count!=43:
            raise TypeError('Explicit owned structured50+print12 and original hockey43 geometry required')
        constructor=geometry.initialization_contract
        if constructor.get('schema')!='rinkspace.current-parent-print12-material-initialization.v2' or constructor.get('no_equality_branch_in_value_or_J')is not True or constructor.get('original_source_UV_sigma_role_or_pixel_qualification_modified')is not False:
            raise ValueError('Actual current-parent continuous print12 material initialization certificate required')
        if not np.array_equal(np.asarray(constructor['actual_current_course_parameters']),geometry.current_course6(geometry.initial)):
            raise ValueError('Print12 constructor course must match the current physical parameter namespace')
        if not np.array_equal(np.asarray(geometry.base['ice_boundary']['points_xyz']),np.asarray(original_geometry43.base['ice_boundary']['points_xyz'])):
            raise ValueError('Same-event original physical boundary source required')
        self.current=geometry;self.original=original_geometry43;self.base=geometry.base;self.scene_unit_m=float(geometry.scene_unit_m)
        self.initial=geometry.initial.copy();self.parameter_names=tuple(geometry.parameter_names);self.parameter_units=tuple(geometry.parameter_units)
        self.fd_step=float(world_fd_step)
        if not np.isfinite(self.fd_step)or self.fd_step<=0.:raise ValueError('Positive explicit geometry FD width required')
        self.source_course=SharedRinkCourseNumericGeometry(self.base,self.scene_unit_m)
        self.source_course.original=deepcopy(original_geometry43.original)
        self.source_course.initial[:23]=self.source_course.original.initial
        self.features={f['id']:f for f in self.base['ice_features']};self.bindings={};self._scene_cache=OrderedDict();self._value_cache=OrderedDict()
        self.registration_ledger=[];self.unrecognized=[]

    def __getattr__(self,name):
        current=self.__dict__.get('current')
        if current is None:raise AttributeError(name)
        return getattr(current,name)

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items()if k not in('_scene_cache','_value_cache')}
    def __setstate__(self,state):self.__dict__.update(state);self._scene_cache=OrderedDict();self._value_cache=OrderedDict()
    def _key(self,row):return self.source_course._row_key(row)
    def _q(self,q):return self.current.validate_parameters(q)
    def source_scene(self,q):
        q=self._q(q);key=q.tobytes()
        if key not in self._scene_cache:
            self._scene_cache[key]=self.current.source_scene(q)
            while len(self._scene_cache)>128:self._scene_cache.popitem(last=False)
        return deepcopy(self._scene_cache[key])
    def _scene(self,q):
        q=self._q(q);key=q.tobytes()
        if key not in self._scene_cache:self.source_scene(q)
        return self._scene_cache[key]
    def scene(self,q):return self.current.scene(self._q(q))

    def register_row(self,row,*,actual_pts=None,allow_previously_qualified_held_ad2_representation=False):
        key=self._key(row)
        if key in self.bindings:return self.bindings[key]['contract']
        world=_as_world(row);certificate=dict(original_source_id=row.get('id'),actual_pts=actual_pts,source_pts=row.get('pts'),
            split=row.get('split'),source_role=row.get('source_role'),source_world_sha256=hashlib.sha256(world.tobytes()).hexdigest(),
            original_finite_world_count=len(world),source_UV_sigma_role_support_order_changed=False,geometry_membership_tolerance_scene=1e-7)
        declaration=declared_absolute_course_identity(row)
        if declaration is not None:
            k=self.source_course._row_key(row)
            material=self.source_course.registered.get(k)
            if material is None:
                material=certify_absolute_course_material(self.source_course,row);self.source_course.register_course_row(row,material)
            binding=dict(kind='course',material=material,fields=('world',*material.companions),columns=COLUMNS_COURSE,
                contract=dict(certificate,physical_identity=declaration[0],qualified_original_material=material.contract))
        else:
            ids=self.source_course.original.pair_panels.identities(row)
            extra_alias=None
            if not ids:
                extra_alias=_qualified_main9_ad3(row)
                if extra_alias is not None:ids=[(3,'right_vertical_xyz')]
            if ids:
                normalized=row if extra_alias is None else extra_alias[0];representation=None
                try:self.source_course._world_and_jacobian(normalized,self.source_course.initial)
                except ValueError:
                    if not allow_previously_qualified_held_ad2_representation or actual_pts is None:raise
                    normalized,representation=self.source_course.original.pair_panels.correct_certified_legacy_held_ad2_chord(row,int(actual_pts),{229535232:229534976})
                    self.source_course._world_and_jacobian(normalized,self.source_course.initial)
                # Additional print3/4/5 coordinates inherit actual original material u/v.
                selected={}
                if all(role in ROLES for _,role in ids):
                    if len(ids)!=len(world):raise ValueError('One original physical print role per WORLD point required')
                    for i,(n,role)in enumerate(ids):
                        if n in self.current.slot:
                            side,level=role.split('_');selected[i]=(n,np.array([float(side=='right')]),np.array([float(level=='top')]))
                elif len(ids)==1 and ids[0][0]in self.current.slot:
                    n,role=ids[0];u,v=self.current._material_coordinates(np.asarray(normalized['world']),n,role);selected['curve']=(n,u,v)
                elif not(len(ids)==1):raise ValueError('Original print source needs corner roles or one finite named edge')
                cols=set(COLUMNS_COURSE)
                for n,_ in ids:
                    if n in self.current.slot:cols.update(range(50+4*self.current.slot[n],54+4*self.current.slot[n]))
                    elif n in(1,14):cols.update(range(28,32))
                    elif n in self.source_course.original.pair_panels.slot:
                        p=self.source_course.original.pair_panels.slot[n];cols.update(range(32+4*p,36+4*p))
                binding=dict(kind='ad',ids=ids,normalized=deepcopy(normalized),selected=selected,fields=('world',),columns=tuple(sorted(cols)),
                    contract=dict(certificate,physical_identities=ids,original_Ad3_source_alias_authority=None if extra_alias is None else extra_alias[1],previously_qualified_HELD_representation=representation,
                        zero_original_HELD_chord_world_claim=False if representation is not None else None))
            else:
                hockey=self.original._identity(row);fr=self.original.original.rink.midpoint.v1._identity(row)
                identity=hockey if hockey is not None else fr
                if identity is None:raise ValueError('Unrecognized original programme physical source identity: '+str(row.get('id')))
                kind,name,*rest=identity;role=rest[0]if rest else'outer';feature=self.features[name]
                if kind in('circle','fixed_circle'):
                    ring=next((r for r in feature.get('rings',[])if r['role']==role),feature if role=='outer'else None)
                    if ring is None:raise ValueError('Original declared ring absent')
                    c=np.asarray(ring['center_xyz']);radius=float(ring['radius']);radial=world-c
                    if np.max(abs(radial[:,2]))>1e-8 or np.max(abs(np.linalg.norm(radial[:,:2],axis=1)-radius))>1e-7:raise ValueError('Original source WORLD is not the named finite ring')
                    if name.startswith(('F','R')):columns=tuple(range(12,18))+(({'middle':18,'inner':19,'center':20}[role],)if role!='outer'else())
                    elif name=='C0':columns=(0,1,3)
                    else:columns=(0,1,2,3,4,5,7,8,*COLUMNS_COURSE)
                    binding=dict(kind='circle',name=name,role=role,direction=radial/radius,fields=('world',),columns=columns,contract=dict(certificate,physical_identity=(kind,name,role)))
                elif kind in('line','carrier','lane'):
                    if kind=='lane':
                        edge=int(role);ends=np.asarray(feature['polygon_xyz'])[[edge,(edge+1)%4]]
                        if edge==3 or edge in feature.get('open_edges',[]):raise ValueError('Unmeasured Lane left proxy is not source material')
                        columns=tuple(range(12,17))+(21,22)
                    else:
                        ends=np.asarray(feature['segments_xyz'])[0];edge=None
                        columns=tuple(range(12,17))if kind=='carrier'else(0,1,2,6,7,8,*COLUMNS_COURSE)
                    t=_finite_t(world,ends)
                    binding=dict(kind=kind,name=name,edge=edge,t=t,fields=('world',),columns=columns,contract=dict(certificate,physical_identity=identity))
                elif kind=='crease':
                    chain=np.asarray(feature['polygon_xyz']);edge=np.roll(chain,-1,axis=0)-chain;l2=np.sum(edge**2,axis=1)
                    vv=world[:,None]-chain;t=np.clip(np.divide(np.sum(vv*edge,axis=2),l2,out=np.zeros(vv.shape[:2]),where=l2!=0),0,1)
                    error=np.linalg.norm(chain[None]+t[:,:,None]*edge[None]-world[:,None],axis=2);indices=np.argmin(error,axis=1);ii=np.arange(len(world))
                    if np.max(error[ii,indices])>1e-7:raise ValueError('Original source is not the original finite crease material')
                    binding=dict(kind=kind,name=name,segments=indices,t=t[ii,indices],fields=('world',),columns=(0,1,2,7,8,9,*COLUMNS_COURSE),contract=dict(certificate,physical_identity=identity))
                else:raise ValueError('Unsupported original semantic source kind: '+str(identity))
        self.bindings[key]=binding;self.registration_ledger.append(binding['contract']);return binding['contract']

    def _binding(self,row):
        key=self._key(row)
        if key not in self.bindings:raise ValueError('Source row must be explicitly registered against original world authority before evaluation')
        return key,self.bindings[key]
    def _ad_value(self,b,q,jacobian):
        q23=self.source_course.original.initial.copy();q23[7:23]=q[28:44];cq=self.current.current_course6(q);q29=np.r_[q23,cq]
        value,j29=self.source_course._world_and_jacobian(b['normalized'],q29)
        if np.any(j29[:,:,:7]):raise ValueError('Programme print unexpectedly depends on removed historical FR coordinates')
        J=np.zeros((*value.shape,62));J[:,:,28:44]=j29[:,:,7:23];J+=np.einsum('nik,kj->nij',j29[:,:,23:],self.current.print_course_binding(q)[2])
        for pos,(n,u,v)in b['selected'].items():
            w,j,_=self.current._material(n,u,v,q)
            if pos=='curve':value,J=w,j
            else:value[pos],J[pos]=w[0],j[0]
        return value,J
    def _material_value(self,b,q):
        if b['kind']=='course':return b['material'].evaluate(self.current.current_course6(q))
        if b['kind']=='ad':return self._ad_value(b,q,False)[0]
        fs={f['id']:f for f in self._scene(q)['ice_features']};f=fs[b['name']]
        if b['kind']=='circle':
            ring=next((r for r in f.get('rings',[])if r['role']==b['role']),f if b['role']=='outer'else None)
            if ring is None:raise ValueError('Current same physical ring disappeared')
            return np.asarray(ring['center_xyz'])+b['direction']*ring['radius']
        if b['kind'] in('line','carrier','lane'):
            ends=np.asarray(f['polygon_xyz'])[[b['edge'],(b['edge']+1)%4]]if b['kind']=='lane'else np.asarray(f['segments_xyz'])[0]
            return ends[0]+b['t'][:,None]*(ends[1]-ends[0])
        chain=np.asarray(f['polygon_xyz']);i=b['segments'];return chain[i]+b['t'][:,None]*(chain[(i+1)%len(chain)]-chain[i])

    def rebind(self,row,q):
        q=self._q(q);_,b=self._binding(row);value=self._material_value(b,q);out=dict(row)
        out['world']=value if isinstance(row['world'],np.ndarray)else value.tolist()
        if b['kind']=='course':
            for field,current in b['material'].companion_values(self.current.current_course6(q)).items():out[field]=current if isinstance(row[field],np.ndarray)else current.tolist()
        for field in('world_xyz','world_xyz_scene_units'):
            if field not in row:continue
            old=np.asarray(row[field]);original=np.asarray(row['world']);shape=old.shape
            original=original[0]if shape==(3,)and len(original)==1 else original
            replacement=value[0]if shape==(3,)and len(value)==1 else value
            if shape!=original.shape:raise ValueError('Source world companion needs the same registered physical material')
            if np.array_equal(old,original):out[field]=replacement.tolist()
            elif np.array_equal(old,original*self.scene_unit_m):out[field]=(replacement*self.scene_unit_m).tolist()
            else:raise ValueError('Source companion unit convention not explicitly represented by original material')
        return out
    def world_jacobian(self,row,q):
        q=self._q(q);key,b=self._binding(row);token=(key,q.tobytes())
        if token in self._value_cache:return self._value_cache[token]
        if b['kind']=='course':
            _,J6=b['material'].evaluate(self.current.current_course6(q),with_jacobian=True);J=np.einsum('nik,kj->nij',J6,self.current.print_course_binding(q)[2])
        elif b['kind']=='ad':J=self._ad_value(b,q,True)[1]
        else:
            J=np.zeros((len(row['world']),3,62));h=self.fd_step
            for k in b['columns']:
                plus=q.copy();minus=q.copy();plus[k]+=h;minus[k]-=h
                J[:,:,k]=(self._material_value(b,plus)-self._material_value(b,minus))/(2*h)
        J.flags.writeable=False;self._value_cache[token]=J
        while len(self._value_cache)>128:self._value_cache.popitem(last=False)
        return J
    def source_parameter_fields(self,row):return self._binding(row)[1]['fields']
    def source_parameter_activity(self,row,q):
        q=self._q(q);_,b=self._binding(row);active=np.zeros(62,bool)
        if b['kind']=='course':
            cq=self.current.current_course6(q);a6=b['material'].course_parameter_activity(cq);active=np.any((self.current.print_course_binding(q)[2]!=0)&a6[:,None],axis=0)
        else:active[list(b['columns'])]=True
        return active
    def feature_id(self,row):return self._binding(row)[1]['contract'].get('physical_identity',self._binding(row)[1]['kind'])
    geometry_parameter_activity=source_parameter_activity

    def finite_board_records(self,records,q,*,original_planes):
        """17 original finite planes: source guard extents follow course AND print4."""
        q=self._q(q);course,cq,chain=self.current.print_course_binding(q);original_planes=np.asarray(original_planes,float)
        if original_planes.shape!=(len(records),4):raise ValueError('One original oriented plane per same declared finite source required')
        planes=[];jplanes=[];updated=[];extents=[];extentJ=[]
        for i,r in enumerate(records):
            if r.get('surface_kind')!='finite_board_face':raise ValueError('Only the declared original finite board rows required')
            old=np.c_[np.asarray([r['extent_start_xy'],r['extent_end_xy']]),np.zeros(2)];identity=r.get('physical_surface_identity')
            if isinstance(identity,dict)and'upper_long_side'in identity.get('scene_boundary_parts',[]):
                expected=course.primitive('upper_long_side',[1.,0.])
                if not np.array_equal(old,expected):raise ValueError('Original main1 named tangent extent changed')
                w,j6=course.primitive('upper_long_side',[1.,0.],cq,with_jacobian=True);j=np.einsum('nik,kj->nij',j6,chain)
            elif identity=='ad4_existing_straight_far_board_face':
                expected=np.asarray([self.source_course.ads[4]['corners_xyz_scene_units'][role]for role in('left_bottom','right_bottom')])
                if np.max(abs(old[:,:2]-expected[:,:2]))>1e-9:raise ValueError('Original low finite support is not the actual Ad4 material')
                w,j,_=self.current._material(4,np.array([0.,1.]),np.zeros(2),q);w[:,2]=0.;j[:,2]=0.
                a,b,rx,_,_,_=course._scene_q(cq)
                if np.any(abs(w[:,1]-b)>1e-8)or np.any(abs(w[:,0])>a-rx+1e-8):raise ValueError('Qualified finite Ad4 plane leaves its original straight upper-long physical domain')
            else:raise ValueError('Unrecognized original finite board physical material')
            oldplane=vertical_plane_value_jacobian(old[0],old[1],np.zeros((3,62)),np.zeros((3,62)))[0]
            if not np.allclose(oldplane,original_planes[i],rtol=0,atol=1e-9):raise ValueError('Original finite oriented plane disagrees with source extent authority')
            p,jp=vertical_plane_value_jacobian(w[0],w[1],j[0],j[1])
            if np.array_equal(w,old):p=original_planes[i].copy()
            changed=deepcopy(r);changed['extent_start_xy']=w[0,:2].tolist();changed['extent_end_xy']=w[1,:2].tolist()
            planes.append(p);jplanes.append(jp);updated.append(changed);extents.append(w);extentJ.append(j)
        z=np.array([0.,course._scene_q(cq)[5]]);zJ=np.zeros((2,62));zJ[1]=chain[5]/self.scene_unit_m
        return dict(planes=np.asarray(planes).reshape(-1,4),plane_jacobian=np.asarray(jplanes).reshape(-1,4,62),records=updated,
            extent_world=np.asarray(extents).reshape(-1,2,3),extent_world_jacobian=np.asarray(extentJ).reshape(-1,2,3,62),z_range_scene=z,z_range_world_jacobian=zJ,
            unknown_depth_rows_or_sigma_or_profile_weights_changed=False,current_Ad4_print_and_boundary_plane_extent_chain=True,complete_source_route_installed=False)

    def manifest(self,q):
        q=self._q(q);return dict(schema=SCHEMA,parameter_count=62,parameters=q.tolist(),parameter_names=self.parameter_names,
            same_physical_current_geometry=self.current.manifest(q),registered_unique_material_count=len(self.bindings),unrecognized_sources=self.unrecognized,
            source_world_replaced_by_structural_family_not_zero_equivalence=True,original_native_pixels_sigmas_roles_and_counts_preserved=True,
            source_source_operator_nearest_association_not_frozen=True,all17_board_plane_extent_course_and_Ad4_print_chain=True,
            no_new_source_pixel_or_prior_admission=True,programme_blocks_patterns_external_actor_GN_gate_not_installed_by_geometry_protocol=True,accuracy_accepted=False)
