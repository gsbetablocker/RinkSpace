"""Original declared finite board-course absolute material certificates.

The membership check uses only original source WORLD and its already declared
physical semantics. It never assigns UV along a board or snaps an off-course
chord. The original finite knots and their segment material fractions remain.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import numpy as np
from z190_secondary_confirmed_nearboard_source import NEAR_KIND
from .board_course_material import certified_finite_course_binding


@dataclass
class AbsoluteCourseMaterial:
    primary:object
    companions:dict
    contract:dict

    @property
    def geometry(self):return self.primary.geometry

    @property
    def old_world(self):return self.primary.old_world

    def _value(self,key,binding,q):
        value=binding.geometry.initial if q is None else np.asarray(q,float)
        token=(key,value.dtype.str,value.tobytes())
        cache=self.__dict__.setdefault('_material_cache',OrderedDict())
        if token in cache:
            result=cache.pop(token);cache[token]=result;return result
        world,jac=binding.evaluate(value,with_jacobian=True)
        # The immutable material world can be shared across many exposures
        # with the same exact course, without approximate q comparison.
        world.setflags(write=False);jac.setflags(write=False)
        cache[token]=(world,jac)
        if len(cache)>48:cache.popitem(last=False)
        return world,jac

    def __getstate__(self):return {k:v for k,v in self.__dict__.items()if k!='_material_cache'}
    def __setstate__(self,value):self.__dict__.update(value)

    def evaluate(self,q=None,*,with_jacobian=False):
        world,jac=self._value('world',self.primary,q)
        return (world,jac)if with_jacobian else world

    def companion_values(self,q=None):
        return {key:self._value(key,binding,q)[0]for key,binding in self.companions.items()}

    def course_parameter_activity(self,q):
        fields={'world':self.primary,**self.companions}
        return np.logical_or.reduce([np.any(self._value(key,binding,q)[1]!=0.,axis=(0,1))
            for key,binding in fields.items()])


def declared_absolute_course_identity(row):
    """Return a supported old source declaration, never infer it from shape."""
    typed=row.get('world_scene_binding',{})
    kind=typed.get('kind')if isinstance(typed,dict)else None
    if kind=='finite_full_board_top_perimeter':return 'full_board_top','top'
    if kind=='finite_full_ice_board_contact_perimeter':return 'full_ice_board_contact','contact'
    if (row.get('curve_id')in('board_top','board_top_near_visible')and
        row.get('surface')=='board_surface_top_edge'and row.get('finite_curve')is True):
        return 'full_board_top','top'
    if (row.get('curve_id')=='ice_board_contact'and row.get('surface')=='ice_board_contact'and
        row.get('finite_curve')is True):return 'full_ice_board_contact','contact'
    # These exact canonical identities are bound by the original source
    # factories and scene_source_geometry_binding.py, even in older thin
    # source rows without the redundant later surface metadata. Preserve
    # the actual original finite row; naming never widens it to a full loop.
    if row.get('origin')and row.get('id')in('far_contact','ice_board_contact'):
        return 'original_named_ice_board_contact','contact'
    if row.get('origin')and row.get('id')in('far_top','board_top','board_top_near_visible'):
        return 'original_named_board_top','top'
    material=row.get('geometry',{})
    if (row.get('id')=='reviewed_lower_board_top'and isinstance(material,dict)and
        material.get('side')=='lower_long_side'and material.get('corner_transition_excluded')is True and
        np.array_equal(material.get('source_scene_vertex_indices'),[336,360])):
        return 'original_reviewed_lower_long_side_top','top'
    if (row.get('curve_observation_kind')==NEAR_KIND and
        row.get('id')=='lower_long_side_board_top_confirmed_source_interval'and
        row.get('physical_identity','lower_long_side_board_top')=='lower_long_side_board_top'and
        row.get('point_correspondence')is False and len(row.get('source_native_members',()))>0):
        # This original typed operator itself declares the confirmed lower
        # straight top course. Historical thin HELD rows keep its exact type,
        # ID and native members while omitting redundant FIT packet fields.
        return 'qualified_lower_long_side_top','top'
    return None


def certify_absolute_course_material(geometry,row):
    declaration=declared_absolute_course_identity(row)
    if declaration is None:return None
    identity,z_policy=declaration
    original=np.asarray(row['world'],float)
    authority=(f"unchanged source row={row['id']}; pts={row.get('pts')}; "
        f"split={row.get('split')}; original semantic declaration={identity}")
    primary=certified_finite_course_binding(geometry.course,original,identity=identity,
        authority=authority,z_policy=z_policy)
    companions={}
    if 'upper_world'in row:
        if identity!='qualified_lower_long_side_top':
            raise ValueError('Unrecognized course companion requires its own physical identity')
        support=row.get('source_horizontal_intervals',())
        if not support or not any('upper_source_packet'in interval for interval in support):
            raise ValueError('Connected upper course has no original qualified source packet')
        # Certify the original connected cap explicitly: the old source
        # constructor fixes these finite facets, not an arbitrary near curve.
        world=np.asarray(row['upper_world'],float)
        a,b,rx,ry,_,h=geometry.course._scene_q(geometry.course.initial)
        tolerance=1e-7
        if (not np.allclose(world[0],[-a,b-ry,h],rtol=0,atol=tolerance)or
            not np.allclose(world[-1],[a-rx,-b,h],rtol=0,atol=tolerance)):
            raise ValueError('Original upper finite cap endpoints do not have the declared left/corner/lower identity')
        domain=(abs(world[:,0]+a)<=tolerance)|((world[:,0]<=-a+rx+tolerance)&(world[:,1]<=-b+ry+tolerance))|(abs(world[:,1]+b)<=tolerance)
        if not np.all(domain):raise ValueError('Original connected cap includes an opposite or unqualified physical facet')
        companions['upper_world']=certified_finite_course_binding(geometry.course,world,
            identity='original_connected_left_short_lower_left_lower_long_top_cap',
            authority=authority+'; original connected-cap operator and qualified upper source packet',z_policy='top')
    if identity in('qualified_lower_long_side_top','original_reviewed_lower_long_side_top'):
        a,b,rx,_,_,h=geometry.course._scene_q(geometry.course.initial)
        if (np.any(abs(original[:,1]+b)>1e-7)or np.any(abs(original[:,2]-h)>1e-7)or
            np.any(abs(original[:,0])>a-rx+1e-7)):
            raise ValueError('Original near lower-long source contains a different physical facet')
    contract=dict(schema='rinkspace.original-declared-finite-absolute-course-material.v1',
        source_id=row['id'],actual_pts=row.get('pts'),original_split=row.get('split'),identity=identity,
        source_primary_world_sha256=hashlib.sha256(np.ascontiguousarray(original).view(np.uint8)).hexdigest(),
        source_companion_world_sha256={key:hashlib.sha256(np.ascontiguousarray(binding.old_world).view(np.uint8)).hexdigest()
            for key,binding in companions.items()},
        original_finite_sample_counts={'world':len(original),**{key:len(binding.old_world)for key,binding in companions.items()}},
        world_to_course_certificate='original finite-world membership only; preserve original segment t; no UV inversion',
        source_parameter_activity='union of real primary and companion world derivatives; no fabricated nonzero primary J',
        z_policy=z_policy,source_UV_support_sigma_roles_and_split_changed=False,
        generic_off_course_chord_snapped=False,additional_source_or_prior_rows=0,accuracy_accepted=False)
    return AbsoluteCourseMaterial(primary,companions,contract)


def register_declared_absolute_course_rows(geometry,rows):
    """Qualify explicit old semantics; caller handles unrecognized dependencies."""
    certificates=[];unrecognized=[]
    for row in rows:
        declaration=declared_absolute_course_identity(row)
        if declaration is None:
            if any(token in str(row.get('id','')).lower()for token in('board','contact','cap')):
                unrecognized.append(dict(id=row.get('id'),pts=row.get('pts'),split=row.get('split'),
                    curve_observation_kind=row.get('curve_observation_kind'),
                    physical_identity=row.get('physical_identity'),surface=row.get('surface'),
                    world_sample_count=len(row.get('world',()))))
            continue
        key=geometry._row_key(row);material=geometry.registered.get(key)
        if material is None:
            material=certify_absolute_course_material(geometry,row)
            geometry.register_course_row(row,material)
        else:
            # The original factories repeat the same finite material in thin
            # and later annotated rows. Only these explicit semantic aliases
            # may share its certificate; the key already includes the exact
            # primary and companion WORLD bytes and physical curve identity.
            aliases=(frozenset(('full_ice_board_contact','original_named_ice_board_contact')),
                frozenset(('full_board_top','original_named_board_top')))
            same_identity=(isinstance(material,AbsoluteCourseMaterial)and
                material.contract['z_policy']==declaration[1]and
                (material.contract['identity']==declaration[0]or any(
                    {material.contract['identity'],declaration[0]}<=group for group in aliases)))
            if not same_identity:
                raise ValueError('Conflicting physical source authority for one original finite course material')
        certificates.append(dict(material.contract,identity=declaration[0],
            material_certificate_identity=material.contract['identity'],actual_pts=row.get('pts'),original_split=row.get('split'),
            original_source_origin=row.get('origin'),original_source_role=row.get('source_role'),
            source_packet_sha256=row.get('source_packet_sha256'),world_scene_binding=row.get('world_scene_binding'),
            physical_identity_authority=row.get('physical_identity_authority')))
    return dict(registered=certificates,unrecognized_possible_course_rows=unrecognized,
        all_course_dependencies_qualified=not unrecognized,full_source_graph_installed=False)
