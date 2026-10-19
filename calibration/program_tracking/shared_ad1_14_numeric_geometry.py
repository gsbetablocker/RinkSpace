"""Opt-in one shared Ad1/14 panel on the existing zero-thickness board.

q4 is absolute metres: [closed-boundary station centre, arc width,
Z centre, height].  Old numeric dimensions initialize unknowns; no new
measurement or prior is introduced.  Ad14 is the exact XY half-turn of Ad1
with the same physical print roles, Z and dimensions.  Source pixels and
roles are never selected or changed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import numpy as np

try:
    from .ad_landmarks import AdGeometry, ROLES
except ImportError:
    from ad_landmarks import AdGeometry, ROLES

VERSION = 'shared_ad1_14_numeric_geometry_v1'
PARAMETER_NAMES = ('Ad1_14_station_center_m', 'Ad1_14_width_m',
                   'Ad1_14_z_center_m', 'Ad1_14_height_m')
CURVE_ROLES = ('left_vertical_xyz', 'right_vertical_xyz',
               'top_edge_xyz', 'bottom_edge_xyz')
_SOURCE_ID = re.compile(r'ad([1-9]|1[0-9]|2[0-6]):(left_top|left_bottom|right_top|right_bottom|left_vertical_xyz|right_vertical_xyz|top_edge_xyz|bottom_edge_xyz)\Z', re.I)


class SharedAd1Ad14NumericGeometry:
    """A single station/Z rectangle, folded onto the actual board course."""
    parameter_names = PARAMETER_NAMES
    parameter_count = 4

    def __init__(self, base, scene_unit_m=None):
        if scene_unit_m is None:
            try:
                from .z190_continuous_source_bundle import SCENE_UNIT_M
            except ImportError:
                from z190_continuous_source_bundle import SCENE_UNIT_M
            scene_unit_m = SCENE_UNIT_M
        self.base = deepcopy(base)
        self.scene_unit_m = float(scene_unit_m)
        if not np.isfinite(self.scene_unit_m) or self.scene_unit_m <= 0:
            raise ValueError('Positive fixed metric scale required')
        if base.get('coordinate_system', {}).get('origin') != 'user_PPT_ice_outline_center':
            raise ValueError('Explicit rink-centre half-turn authority required')
        boundary = np.asarray(base['ice_boundary']['points_xyz'], float)
        centre = (boundary[:, :2].min(0) + boundary[:, :2].max(0))/2
        if np.max(abs(centre)) > 1e-7:
            raise ValueError('Boundary centre differs from the declared origin')
        self.old = {n: AdGeometry(base, n) for n in (1, 14)}
        if any(set(g.roles) != set(ROLES) for g in self.old.values()):
            raise ValueError('All four existing physical Ad1/14 corners required')
        self.arc = self.old[1]
        self.perimeter_m = self.arc._perimeter*self.scene_unit_m
        self.width_sign = self.arc._width_sign
        self.initial = np.array([
            (self.arc._base_s['left'] + self.width_sign*self.arc._nominal_width/2)
            * self.scene_unit_m,
            self.arc._nominal_width*self.scene_unit_m,
            (self.arc._base_height['bottom']+self.arc._base_height['top'])
            * self.scene_unit_m/2,
            (self.arc._base_height['top']-self.arc._base_height['bottom'])
            * self.scene_unit_m])
        for role in ROLES:
            one = self.old[1].corner(self.old[1].initial, role)
            other = self.old[14].corner(self.old[14].initial, role)
            if np.max(abs(other - one*np.array([-1., -1., 1.]))) > 1e-7:
                raise ValueError('Original Ad1/14 must already have the same hard print-role half-turn binding')
        self.board_bottom_m = self.arc._ice_z*self.scene_unit_m
        self.board_top_m = self.arc._board_top*self.scene_unit_m
        self.base_sha256 = hashlib.sha256(json.dumps(
            base, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.authority = deepcopy(base.get('shared_centrosymmetric_advertisement_authority', {}))
        self.validate_parameters(self.initial)

    def _q(self, q4):
        q = np.asarray(q4, float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError('Four finite absolute panel metre parameters required')
        if q[1] <= 0 or q[1] >= self.perimeter_m/2 or q[3] <= 0:
            raise ValueError('Positive panel height and positive minor-arc width required')
        return q

    def validate_parameters(self, q4):
        q = self._q(q4)
        if q[2]-q[3]/2 < self.board_bottom_m-1e-8 or q[2]+q[3]/2 > self.board_top_m+1e-8:
            raise ValueError('Printed panel exceeds the current engineering board Z domain')
        return q.copy()

    def _arc_point_jacobian(self, station_m):
        station = np.asarray(station_m, float).reshape(-1)/self.scene_unit_m
        s = station % self.arc._perimeter
        index = np.minimum(np.searchsorted(self.arc._cum, s, side='right')-1,
                           len(self.arc._starts)-1)
        tangent = self.arc._edges[index]/self.arc._lengths[index, None]
        xy = self.arc._starts[index]+(s-self.arc._cum[index])[:, None]*tangent
        return xy, tangent/self.scene_unit_m

    def _material(self, q4, u, v, number):
        q = self._q(q4)
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        u, v = u.reshape(-1), v.reshape(-1)
        station = q[0]+self.width_sign*(u-.5)*q[1]
        xy, tangent = self._arc_point_jacobian(station)
        world = np.column_stack((xy, (q[2]+(v-.5)*q[3])/self.scene_unit_m))
        jac = np.zeros((len(world), 3, 4))
        jac[:, :2, 0] = tangent
        jac[:, :2, 1] = tangent*(self.width_sign*(u-.5))[:, None]
        jac[:, 2, 2] = 1/self.scene_unit_m
        jac[:, 2, 3] = (v-.5)/self.scene_unit_m
        if int(number) == 14:
            world[:, :2] *= -1
            jac[:, :2] *= -1
        elif int(number) != 1:
            raise ValueError('This shared panel only defines Ad1/14')
        return world, jac

    def corner(self, number, role, q4=None, *, with_jacobian=False):
        if role not in ROLES:
            raise ValueError('Existing physical print corner role required')
        side, level = role.split('_')
        world, jac = self._material(self.initial if q4 is None else q4,
                                   float(side == 'right'), float(level == 'top'), number)
        return (world[0], jac[0]) if with_jacobian else world[0]

    def _boundary_knots(self, q, low_u, high_u):
        first = q[0]+self.width_sign*(low_u-.5)*q[1]
        last = q[0]+self.width_sign*(high_u-.5)*q[1]
        low, high = sorted((first, last))
        values = []
        for turn in range(int(np.floor(low/self.perimeter_m))-1,
                          int(np.ceil(high/self.perimeter_m))+1):
            values.extend(float(s*self.scene_unit_m+turn*self.perimeter_m)
                          for s in self.arc._cum[:-1]
                          if low < s*self.scene_unit_m+turn*self.perimeter_m < high)
        return np.unique(values)

    @staticmethod
    def _identities(row):
        roles = row.get('physical_world_corner_roles')
        if roles is not None:
            number = row.get('physical_ad_id', row.get('ad_id'))
            if number is None or any(role not in ROLES for role in roles):
                raise ValueError('Explicit advertisement number and physical corner roles required')
            return [(int(number), role) for role in roles]
        identities = None
        ids = row.get('physical_ids')
        if ids:
            parsed = [_SOURCE_ID.fullmatch(str(value)) for value in ids]
            if any(value is None for value in parsed):
                if any(str(value).lower().startswith(('ad1:', 'ad14:')) for value in ids):
                    raise ValueError('Mixed or unsupported explicit Ad1/14 physical identities')
            else:
                identities = [(int(value[1]), value[2].lower()) for value in parsed]
        else:
            parsed = _SOURCE_ID.fullmatch(str(row.get('id', '')))
            if parsed is not None:
                identities = [(int(parsed[1]), parsed[2].lower())]
            elif row.get('ad_id') is not None and row.get('role') in (*ROLES, *CURVE_ROLES):
                identities = [(int(row['ad_id']), row['role'])]
        if identities is None and int(row.get('physical_ad_id', 0)) in (1, 14):
            role = row.get('role')
            if role not in (*ROLES, *CURVE_ROLES):
                raise ValueError('Ad1/14 source lacks its explicit physical role')
            return [(int(row['physical_ad_id']), role)]
        return identities

    def _old_curve_coordinates(self, world, number, role):
        # Unfold Ad14 into Ad1's exact chart before validating the ORIGINAL
        # finite panel. A new camera or changed panel cannot qualify old UV.
        original = world*np.array([-1., -1., 1.]) if number == 14 else world
        arcs = np.array([self.arc._nearest_arc(p[:2]) for p in original])
        snapped = np.array([self.arc._point_at_arc(s) for s in arcs])
        if np.max(np.linalg.norm(snapped-original[:, :2], axis=1)) > 1e-7:
            raise ValueError('Source world is off the original actual boundary course')
        delta = (arcs-self.arc._base_s['left']+self.arc._perimeter/2)%self.arc._perimeter-self.arc._perimeter/2
        u = delta/(self.width_sign*self.arc._nominal_width)
        v = (original[:, 2]-self.arc._base_height['bottom'])/(self.arc._base_height['top']-self.arc._base_height['bottom'])
        if np.any((u < -1e-7)|(u > 1+1e-7)|(v < -1e-7)|(v > 1+1e-7)):
            raise ValueError('Source world exceeds its original finite print panel')
        if role.startswith(('left_', 'right_')):
            expected = float(role.startswith('right_'))
            if np.max(abs(u-expected)) > 1e-7:
                raise ValueError('Source world does not lie on its named vertical print edge')
        else:
            expected = float(role.startswith('top_'))
            if np.max(abs(v-expected)) > 1e-7:
                raise ValueError('Source world does not lie on its named horizontal print edge')
            if len(u) > 1 and not (np.all(np.diff(u) >= -1e-8) or np.all(np.diff(u) <= 1e-8)):
                raise ValueError('Finite print-edge source support order must be monotonic')
        return u, v

    def _world_and_jacobian(self, row, q4):
        q = self._q(q4)
        identities = self._identities(row)
        original = np.asarray(row['world'], float)
        if original.ndim != 2 or original.shape[1] != 3 or not len(original) or not np.isfinite(original).all():
            raise ValueError('Finite original source world N by 3 required')
        if identities is None or all(n not in (1, 14) for n, _ in identities):
            return original.copy(), np.zeros((len(original), 3, 4))
        if all(role in ROLES for _, role in identities):
            if len(identities) != len(original):
                raise ValueError('Physical corner identities/source worlds count differs')
            world = original.copy(); jac = np.zeros((len(original), 3, 4))
            for i, (number, role) in enumerate(identities):
                if number not in (1, 14):
                    continue
                expected = self.old[number].corner(self.old[number].initial, role)
                if np.max(abs(original[i]-expected)) > 1e-7:
                    raise ValueError('Original world is not its named physical print corner')
                world[i], jac[i] = self.corner(number, role, q, with_jacobian=True)
            return world, jac
        if len(identities) != 1 or identities[0][0] not in (1, 14) or identities[0][1] not in CURVE_ROLES:
            raise ValueError('Finite print curve requires one explicit physical panel edge')
        number, role = identities[0]
        u, v = self._old_curve_coordinates(original, number, role)
        world, jac = self._material(q, u, v, number)
        if role in ('top_edge_xyz', 'bottom_edge_xyz') and len(u) > 1:
            # A folded perimeter edge must contain actual board vertices.
            # These are geometry samples, not extra source observations.
            knots = self._boundary_knots(q, float(u.min()), float(u.max()))
            if len(knots):
                xy, _ = self._arc_point_jacobian(knots)
                added = np.column_stack((xy, np.full(len(knots), world[0, 2])))
                added_j = np.zeros((len(knots), 3, 4))
                added_j[:, 2, 2:] = np.array([1., v[0]-.5])/self.scene_unit_m
                if number == 14: added[:, :2] *= -1
                ku = .5+(knots-q[0])/(self.width_sign*q[1])
                order = np.argsort(np.r_[u, ku], kind='stable')
                if u[-1] < u[0]: order = order[::-1]
                world = np.vstack((world, added))[order]
                jac = np.concatenate((jac, added_j))[order]
        return world, jac

    def rebind(self, row, q4):
        identities = self._identities(row)
        if identities is None or all(n not in (1, 14) for n, _ in identities):
            return row
        world, _ = self._world_and_jacobian(row, q4)
        result = dict(row)
        result['world'] = world if isinstance(row['world'], np.ndarray) else world.tolist()
        for key in ('world_xyz', 'world_xyz_scene_units'):
            if key not in row: continue
            value = np.asarray(row[key])
            if value.shape == (3,) and len(world) == 1: result[key] = world[0].tolist()
            elif value.shape == np.asarray(row['world']).shape: result[key] = world.tolist()
            else: raise ValueError('Source companion world geometry has an incompatible shape')
        return result

    def world_jacobian(self, row, q4):
        return self._world_and_jacobian(row, q4)[1]

    def boundary(self, number, role, q4):
        q = self._q(q4)
        if role not in CURVE_ROLES: raise ValueError('One physical panel edge required')
        if role.startswith(('left_', 'right_')):
            u = np.full(17, float(role.startswith('right_'))); v = np.linspace(0., 1., 17)
        else:
            u = np.linspace(0., 1., max(2, int(np.ceil(q[1]/(.25*self.scene_unit_m)))+1))
            knots = self._boundary_knots(q, 0., 1.)
            u = np.unique(np.r_[u, .5+(knots-q[0])/(self.width_sign*q[1])])
            v = np.full(len(u), float(role.startswith('top_')))
        return self._material(q, u, v, number)[0]

    def manifest(self, q4=None):
        q = self._q(self.initial if q4 is None else q4)
        return dict(version=VERSION, parameter_names=list(PARAMETER_NAMES), parameters_m=q.tolist(),
            parameterization='Four absolute metres, one shared station-Z panel; no frame-specific print geometry',
            fixed_metric_m_per_scene_unit=self.scene_unit_m, base_scene_sha256=self.base_sha256,
            initial_m=self.initial.tolist(), initial_source='Actual old AdGeometry corners and minor boundary arc; estimates, not new measurements',
            print_role_width_sign=self.width_sign, actual_closed_boundary_perimeter_m=self.perimeter_m,
            station_wrap='Modulo the unchanged actual closed boundary, with original physical print left/right direction',
            pair_rule='Ad14 equals Ad1 XY half-turn about declared rink origin; same Z, arc width and height',
            pair_authority=self.authority, panel_numeric_priors=None,
            board_z_domain_m=[self.board_bottom_m, self.board_top_m],
            board_z_domain_authority='Existing coarse engineering physical domain, not a newly measured board height or panel height',
            source_binding='Strict untouched original corner/finite side authority; same physical role and normalized old arc/height parameter',
            source_pixels_sigma_roles_actual_aliases_indices_unchanged=True,
            non_pair_geometry_unchanged=True, thickness_parameter_added=False,
            curve_samples_are_engineering_geometry_not_extra_source=True, accuracy_accepted=False)

    def scene(self, q4, *, scene=None):
        q = self.validate_parameters(q4)
        result = deepcopy(self.base if scene is None else scene)
        ads = {int(a['user_number']): a for a in result['advertisements']}
        for number in (1, 14):
            ad = ads[number]
            corners = {role: self.corner(number, role, q).tolist() for role in ROLES}
            ad['corners_xyz_scene_units'] = corners
            ad['fitted_boundary_curves_xyz'] = {role: self.boundary(number, role, q).tolist() for role in CURVE_ROLES}
            ad['top_z_scene_units'] = (q[2]+q[3]/2)/self.scene_unit_m
            ad['bottom_z_scene_units'] = (q[2]-q[3]/2)/self.scene_unit_m
            ad['width_m'] = float(q[1]); ad['height_m'] = float(q[3])
            ad['along_fractions_status'] = 'Historical only; actual shared station and corners define geometry'
            ad['boundary_construction'] = 'Actual shared folded boundary-station/Z panel; all physical board vertices retained'
            ad['geometry_status'] = 'shared_four_unknown_panel_dimensions_candidate_unaccepted'
            ad[VERSION] = self.manifest(q)
        for row in result.get('advertisement_3d_corners', []):
            number, role = int(row['user_number']), row['role']
            if number in (1, 14) and role in ROLES:
                row['world_xyz_scene_units'] = self.corner(number, role, q).tolist()
        result[VERSION] = self.manifest(q)
        return result
