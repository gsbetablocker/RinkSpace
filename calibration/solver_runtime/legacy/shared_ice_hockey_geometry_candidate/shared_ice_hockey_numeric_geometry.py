"""Current q23 world plus K20 shared ice-hockey deltas; no graph installation.

Initial new deltas preserve the actual inherited scene and original finite
material exactly. No standard dimension, precision, observation or prior is
inserted. Parallel/symmetric/topological reference factors remain explicit
caller responsibilities, so inherited imperfections are not silently erased.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
import pickle
import numpy as np
from shared_rink_numeric_geometry import GeometryIdentityError


VERSION = 'shared_ice_hockey_delta_geometry_v1'
LINE_NAMES = ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right')
CIRCLE_NAMES = ('C0', 'A1', 'A2')
CREASE_NAMES = ('goal_crease_left', 'goal_crease_right')
NEW_NAMES = (
    'C0_center_dx_m', 'C0_center_dy_m', 'C0_radius_delta_m',
    'A_pair_station_dx_m', 'A_pair_midpoint_dy_m',
    'A_pair_half_separation_delta_m', 'A_pair_half_stagger_dx_m', 'A_pair_radius_delta_m',
    *tuple(v for name in LINE_NAMES for v in (name+'_station_dx_m', name+'_yaw_delta_rad')),
    'goal_crease_left_template_radius_delta_m', 'goal_crease_right_template_radius_delta_m',
)
FR_RATIO_NAMES = ('FR_middle_outer_ratio_delta', 'FR_inner_outer_ratio_delta',
                  'FR_center_outer_ratio_delta')


def _restore(cls, original, include_fr_ratio_deltas):
    return cls(original, include_fr_ratio_deltas=include_fr_ratio_deltas)


class SharedIceHockeyNumericGeometry:
    """One physical event-wide geometry provider, keeping original q23 intact.

    New coordinates are deltas, initially zero. Five finite transverse lines
    retain their original span and centre-Y; each gets station-X/yaw. These
    are conditional finite templates, not certified whole boundary clipping.
    Goal-crease templates keep all 61 original knots and their historical
    half-circle shape; they follow their goal line's translation/rotation and
    get only a template-radius delta. Their source feature schema is unchanged.

    Optional FR ratio deltas expose a separate uninstalled interface. They
    never turn the WorldCurling nominal radii into measured constants.
    """
    def __init__(self, original, *, include_fr_ratio_deltas=False):
        if getattr(original, 'parameter_count', None) != 23:
            raise ValueError('Actual complete original q23 world provider required')
        self.original = original
        self.base = original.base
        self.scene_unit_m = float(original.scene_unit_m)
        self.include_fr_ratio_deltas = bool(include_fr_ratio_deltas)
        self.new_parameter_count = 20 + (3 if self.include_fr_ratio_deltas else 0)
        self.parameter_count = 23 + self.new_parameter_count
        self.parameter_names = (*original.parameter_names, *NEW_NAMES,
                                *(FR_RATIO_NAMES if self.include_fr_ratio_deltas else ()))
        self.parameter_units = ('m',)*23 + tuple('rad' if n.endswith('_rad') else 'm' for n in NEW_NAMES) + (
                              ('dimensionless',)*3 if self.include_fr_ratio_deltas else ())
        self.initial = np.r_[original.initial, np.zeros(self.new_parameter_count)]
        self.features = {f['id']: f for f in self.base['ice_features']}
        self._original_row_signature=None
        self._original_row_world_cache=OrderedDict()
        self.centres = {}
        self.radii = {}
        for name in CIRCLE_NAMES:
            f = self.features[name]
            if f['kind'] != 'circle' or f.get('rings') or not np.array_equal(f['plane_normal_xyz'], [0.,0.,1.]):
                raise GeometryIdentityError('Actual single planar ice circle required: '+name)
            self.centres[name] = np.asarray(f['center_xyz'], dtype=float)
            self.radii[name] = float(f['radius'])
        if self.radii['A1'] != self.radii['A2']:
            raise GeometryIdentityError('Inherited A pair must have its actual equal common radius')
        self.line_centres = {}
        self.line_endpoints = {}
        for name in LINE_NAMES:
            f = self.features[name]
            world = np.asarray(f['segments_xyz'], dtype=float)
            if f['kind'] != 'polyline' or world.shape != (1,2,3) or not np.isfinite(world).all():
                raise GeometryIdentityError('Actual single finite line segment required: '+name)
            if np.any(world[:,:,2] != 0.) or np.linalg.norm(world[0,1]-world[0,0]) == 0.:
                raise GeometryIdentityError('Nondegenerate inherited ice segment required: '+name)
            self.line_endpoints[name] = world[0].copy()
            self.line_centres[name] = world[0].mean(axis=0)
        self.crease_centres = {}
        self.crease_radii = {}
        for name in CREASE_NAMES:
            f = self.features[name]
            points = np.asarray(f['polygon_xyz'], dtype=float)
            if f['kind'] != 'polygon' or points.shape != (61,3) or np.any(points[:,2] != 0.):
                raise GeometryIdentityError('Actual inherited 61-knot crease template required')
            # Original first/last are the two endpoints of the half-circle,
            # not UV-derived points and not a modern rectangular crease.
            centre = (points[0]+points[-1])/2
            radius = float(np.linalg.norm(points[0]-points[-1])/2)
            if radius <= 0.:
                raise GeometryIdentityError('Degenerate original crease template')
            self.crease_centres[name] = centre
            self.crease_radii[name] = radius

    def __getattr__(self, name):
        old = self.__dict__.get('original')
        if old is None:
            raise AttributeError(name)
        return getattr(old, name)

    def __reduce__(self):
        return _restore, (type(self), self.original, self.include_fr_ratio_deltas)

    def _original_rebind(self,row,q23):
        """Exact-content one-row cache for unchanged original q23 fields.

        Sparse source FD still checks both actual worlds for every declared
        inactive column. New hockey-only perturbations often request exactly
        the same original q23. Keep at most16 such values for one exact row;
        no graph-wide world/J cache or source-array mutation is introduced.
        Every output world is an owned copy, so a consumer cannot poison the
        cached original value. Original opaque metadata stays the source's.
        """
        try:signature=pickle.dumps(row,protocol=5)
        except (TypeError,pickle.PicklingError):
            return self.original.rebind(row,q23)
        if signature!=self._original_row_signature:
            self._original_row_signature=signature
            self._original_row_world_cache.clear()
        key=np.asarray(q23,float).tobytes()
        cached=self._original_row_world_cache.get(key)
        if cached is None:
            cached=self.original.rebind(row,q23)
            cached=dict(cached)
            for field in ('world','world_xyz','world_xyz_scene_units'):
                if field in cached:cached[field]=deepcopy(cached[field])
            self._original_row_world_cache[key]=cached
            while len(self._original_row_world_cache)>16:self._original_row_world_cache.popitem(last=False)
        else:self._original_row_world_cache.move_to_end(key)
        result=dict(cached)
        for field in ('world','world_xyz','world_xyz_scene_units'):
            if field in result:result[field]=deepcopy(result[field])
        return result

    def _q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.parameter_count,) or not np.isfinite(q).all():
            raise ValueError('Complete original q23 and declared current shared ice geometry deltas required')
        # Structural checking is cheap on every material row. Physical chart
        # guards are performed explicitly by validate_parameters/scene and
        # the parent's original trial guard, not thousands of times per FD.
        self.original._q(q[:23])
        return q

    def validate_parameters(self, q):
        q = self._q(q)
        self.original.validate_parameters(q[:23])
        d = q[23:]
        for name, column in (('C0',2), ('A1',7), ('A2',7)):
            if self.radii[name] + d[column]/self.scene_unit_m <= 0.:
                raise ValueError('Current physical circle radius must be positive: '+name)
        for name, column in zip(CREASE_NAMES, (18,19)):
            if self.crease_radii[name] + d[column]/self.scene_unit_m <= 0.:
                raise ValueError('Current inherited crease template radius must be positive')
        if self.include_fr_ratio_deltas:
            f1 = self.features['F1']
            ratios = {r['role']:float(r['radius'])/float(f1['radius']) for r in f1['rings']}
            current = [1., *(ratios[role]+d[20+i] for i,role in enumerate(('middle','inner','center')))]
            if not all(a > b > 0. for a,b in zip(current, current[1:])):
                raise ValueError('Current common FR radial roles must retain strict concentric order')
        return q.copy()

    def _circle_delta(self, name, d):
        delta = np.zeros(3)
        derivatives = np.zeros((3,self.new_parameter_count))
        s = self.scene_unit_m
        if name == 'C0':
            delta[:2] = d[:2]/s
            derivatives[0,0] = derivatives[1,1] = 1/s
            radius_column = 2
        else:
            sign = -1. if name == 'A1' else 1.
            delta[0] = (d[3]+sign*d[6])/s
            delta[1] = (d[4]+sign*d[5])/s
            derivatives[0,3] = derivatives[1,4] = 1/s
            derivatives[0,6] = derivatives[1,5] = sign/s
            radius_column = 7
        return delta, derivatives, radius_column

    def _line_columns(self, name):
        ordinal = LINE_NAMES.index(name)
        return 8+2*ordinal, 9+2*ordinal

    def _line_transform(self, world, name, d, *, centre=None, radius_column=None):
        dxcol, yawcol = self._line_columns(name)
        theta = d[yawcol]
        c,s = np.cos(theta),np.sin(theta)
        rotation = np.array([[c,-s],[s,c]])
        centre = self.line_centres[name] if centre is None else centre
        radial = world[:,:2]-centre[:2]
        scale = 1.
        if radius_column is not None:
            crease = 'goal_crease_left' if name == 'goal_line_left' else 'goal_crease_right'
            scale += d[radius_column]/(self.crease_radii[crease]*self.scene_unit_m)
        # A crease scales around its original semicircle centre, then follows
        # the same whole goal-line rotation around that line's own centre.
        anchor = centre[:2]-self.line_centres[name][:2]
        current_vector = anchor[None,:]+scale*radial
        transformed_vector = current_vector @ rotation.T
        transformed_radial = radial @ rotation.T
        # Add only deltas, so a zero new chart preserves inherited raw bits.
        out = world.copy()
        out[:,:2] += transformed_vector-(anchor[None,:]+radial)
        out[:,0] += d[dxcol]/self.scene_unit_m
        jac = np.zeros((len(world),3,self.new_parameter_count))
        jac[:,0,dxcol] = 1/self.scene_unit_m
        jac[:,0,yawcol] = -transformed_vector[:,1]
        jac[:,1,yawcol] = transformed_vector[:,0]
        if radius_column is not None:
            jac[:,:2,radius_column] = transformed_radial/(self.crease_radii[crease]*self.scene_unit_m)
        if not np.any(d[[dxcol,yawcol]]) and (radius_column is None or d[radius_column] == 0.):
            out = world.copy()
        return out,jac

    def _identity(self, row):
        original_id = str(row.get('id',''))
        name = str(row.get('physical_curve_id',original_id))
        canonical = row.get('canonical_physical_geometry_id')
        if canonical is not None and str(canonical) != name:
            raise GeometryIdentityError('Canonical shared ice feature identity disagrees')
        if name != original_id and not any(row.get(k) for k in ('origin','physical_identity_authority',
                    'source_qualification','physical_curve_id_authority','source_branch_operator_authority')):
            raise GeometryIdentityError('Canonical shared ice feature needs original physical source authority')
        if name.endswith('_outer') and name[:-6] in CIRCLE_NAMES:
            return 'circle',name[:-6]
        if name in CIRCLE_NAMES:
            return 'circle',name
        if name in LINE_NAMES:
            return 'line',name
        if name in CREASE_NAMES:
            return 'crease',name
        return None

    def feature_id(self, row):
        identity = self._identity(row)
        return identity[1] if identity is not None else self.original.feature_id(row)

    def _world_and_jacobian(self, row, q):
        q = self._q(q)
        d = q[23:]
        before = np.asarray(row['world'], dtype=float)
        identity = self._identity(row)
        world, oldj = self.original._world_and_jacobian(row,q[:23])
        newj = np.zeros((len(world),3,self.new_parameter_count))
        if identity is not None:
            kind,name = identity
            if kind == 'circle':
                radial = before-self.centres[name]
                if (np.max(np.abs(radial[:,2]),initial=0.) > 1e-8 or
                    np.max(np.abs(np.linalg.norm(radial[:,:2],axis=1)-self.radii[name]),initial=0.) > 1e-7):
                    raise GeometryIdentityError('Original source is not this base finite circle: '+name)
                delta, derivatives, rcol = self._circle_delta(name,d)
                world = before+delta+radial*(d[rcol]/(self.radii[name]*self.scene_unit_m))
                newj[:] = derivatives[None,:,:]
                newj[:,:,rcol] = radial/(self.radii[name]*self.scene_unit_m)
            elif kind == 'line':
                from shared_rink_numeric_geometry import SharedRinkNumericGeometry
                SharedRinkNumericGeometry._finite_line_check(before,self.line_endpoints[name],name)
                world,newj = self._line_transform(before,name,d)
            else:
                # Only exactly original knots or points on their original
                # finite material segments may use this template transport.
                chain = np.asarray(self.features[name]['polygon_xyz'],float)
                a,b = chain,np.roll(chain,-1,axis=0)
                v=b-a;length2=np.sum(v*v,axis=1)
                ok=np.zeros(len(before),bool)
                for start,vec,l2 in zip(a,v,length2):
                    if l2 == 0.:continue
                    t=(before-start)@vec/l2
                    error=np.linalg.norm(before-(start+t[:,None]*vec),axis=1)
                    ok |= (t>=-1e-8)&(t<=1+1e-8)&(error<=1e-7)
                if not ok.all():
                    raise GeometryIdentityError('Source is not the original finite crease template')
                goal='goal_line_left' if name.endswith('left') else 'goal_line_right'
                rcol=18 if name.endswith('left') else 19
                world,newj=self._line_transform(before,goal,d,centre=self.crease_centres[name],radius_column=rcol)
        elif self.include_fr_ratio_deltas:
            old_identity = self.original.rink.midpoint.v1._identity(row)
            if old_identity is not None and old_identity[0] == 'circle' and old_identity[2] in ('middle','inner','center'):
                _,name,role=old_identity
                col=20+('middle','inner','center').index(role)
                base_center=np.asarray(self.features[name]['center_xyz'],float)
                ring=next(r for r in self.features[name]['rings'] if r['role']==role)
                unit_direction=(before-base_center)/float(ring['radius'])
                base_outer=float(self.features[name]['radius'])
                current_outer=base_outer+q[3]/self.scene_unit_m
                world=world+unit_direction*(current_outer*d[col])
                newj[:,:,col]=unit_direction*current_outer
                oldj[:,:,3]+=unit_direction*(d[col]/self.scene_unit_m)
        return world,np.concatenate((oldj,newj),axis=2)

    def rebind(self, row, q):
        # The original provider keeps all opaque source fields intact.
        q=self._q(q)
        result = self._original_rebind(row,q[:23])
        # Most programme rows do not depend on a new hockey coordinate.
        # Their original rebind already supplies the complete current world;
        # constructing another original world/J would only repeat work.
        identity=self._identity(row)
        ratio=False
        if self.include_fr_ratio_deltas and identity is None:
            old_identity=self.original.rink.midpoint.v1._identity(row)
            ratio=(old_identity is not None and old_identity[0]=='circle' and
                   old_identity[2] in ('middle','inner','center'))
        if identity is not None or ratio:
            old_world=np.asarray(result['world'],float)
            result['world'] = self._world_and_jacobian(row,q)[0]
            # A companion world field is a representation of the same
            # certified material, never a second source. Preserve its units.
            for key in ('world_xyz','world_xyz_scene_units'):
                if key not in result:continue
                companion=np.asarray(result[key],float)
                if companion.shape!=old_world.shape:continue
                if np.array_equal(companion,old_world):
                    result[key]=result['world'].copy()
                elif np.array_equal(companion,old_world*self.scene_unit_m):
                    result[key]=result['world']*self.scene_unit_m
        return result

    def world_jacobian(self, row, q):
        return self._world_and_jacobian(row,q)[1]

    def source_parameter_fields(self, row, q):
        self._q(q)
        return ('world',)

    def source_parameter_activity(self, row, q):
        return np.any(self.world_jacobian(row,q) != 0.,axis=(0,1))

    def midpoint_prior(self, q):
        q = self._q(q)
        r,j = self.original.midpoint_prior(q[:23])
        return r,np.c_[j,np.zeros((len(r),self.new_parameter_count))]

    def manifest(self, q=None):
        q = self._q(self.initial if q is None else q)
        return dict(version=VERSION, parameter_count=self.parameter_count,
                    original_world_parameter_count=23, new_shared_parameter_count=self.new_parameter_count,
                    parameter_names=list(self.parameter_names), parameter_units=list(self.parameter_units),
                    parameters=q.tolist(), original_q23_geometry=self.original.manifest(q[:23]),
                    new_initial_deltas=np.zeros(self.new_parameter_count).tolist(),
                    new_numeric_basis='actual inherited finite scene templates; engineering metric, not survey or nominal standard',
                    nominal_standards_installed_as_prior=False, nominal_dimension_forced=False,
                    rink_engineering_design_target_metres=[60.,30.],
                    rink_design_target_is_not_survey=True,
                    rink_boundary_60x30_geometry_or_soft_reference_installed=False,
                    actual_original_board_boundary_and_ads_not_rescaled=True,
                    source_pixels_roles_support_sigma_sampling_order_unchanged=True,
                    all_original_FR_Lane_ads_and_board_preserved=True,
                    initial_new_deltas_preserve_original_world_bits=True,
                    transverse_lines_parallel_symmetry_reference_not_hardcoded=True,
                    conditional_line_domain='retain original finite span, centerY and original end-vertex material order; stationX/yaw are common event fields',
                    whole_course_endpoint_reclipping_installed=False,
                    goal_crease_shape='original 61-knot half-circle templates; no modern crease replacement',
                    optional_FR_ratio_deltas=self.include_fr_ratio_deltas,
                    WorldCurling_nominals_not_measurements=True,
                    no_new_source_admission_or_source_rows=True, additional_prior_rows=0,
                    exact_content_original_world_cache='one complete source row and at most16 q23 byte keys; owned output worlds; all sparse inactive FD checks still run',
                    complete_source_graph_installed=False, accuracy_accepted=False)

    def _update_feature(self, feature, d):
        f=deepcopy(feature);name=f['id']
        if name in CIRCLE_NAMES:
            delta,_,rcol=self._circle_delta(name,d)
            f['center_xyz']=(self.centres[name]+delta).tolist()
            f['radius']=self.radii[name]+d[rcol]/self.scene_unit_m
        elif name in LINE_NAMES:
            points=np.asarray(f['segments_xyz'],float).reshape(-1,3)
            f['segments_xyz']=self._line_transform(points,name,d)[0].reshape(1,2,3).tolist()
        elif name in CREASE_NAMES:
            goal='goal_line_left' if name.endswith('left') else 'goal_line_right'
            f['polygon_xyz']=self._line_transform(np.asarray(f['polygon_xyz'],float),goal,d,
                centre=self.crease_centres[name],radius_column=18 if name.endswith('left') else 19)[0].tolist()
        return f

    def scene(self, q):
        q=self.validate_parameters(q)
        out=self.original.scene(q[:23]);d=q[23:]
        out['ice_features']=[self._update_feature(f,d) if f['id'] in
                            (*CIRCLE_NAMES,*LINE_NAMES,*CREASE_NAMES) else f for f in out['ice_features']]
        if self.include_fr_ratio_deltas:
            for f in out['ice_features']:
                if f['id'] in {col+str(i) for col in 'FR' for i in range(1,5)}:
                    base= self.features[f['id']]
                    ratios={r['role']:float(r['radius'])/float(base['radius']) for r in base['rings']}
                    for r in f['rings']:
                        if r['role'] in ('middle','inner','center'):
                            col=20+('middle','inner','center').index(r['role'])
                            if d[col] != 0.:
                                r['radius']=float(f['radius'])*(ratios[r['role']]+d[col])
        out[VERSION]=self.manifest(q)
        return out

    def source_scene(self, q):
        return self.scene(q)

    def feature_state(self, q):
        """Current same-world dimensional relations input, in explicit units."""
        scene=self.scene(q)
        return dict(scene_unit_m=self.scene_unit_m,
                    ice_features={f['id']:f for f in scene['ice_features']},
                    ice_boundary=scene['ice_boundary'], parameters=scene['parameters'],
                    geometric_standard_version_measured=False,
                    world_geometry_manifest=self.manifest(q))
