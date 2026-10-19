"""User-confirmed, scene-wide 180-degree advertisement pairs.

Ad x+13 is the horizontal half-turn of Ad x, at the same Z. Print-corner
roles are physical roles, never world-X ordering or per-frame image sorting.
Only pairs with at least one existing model are materialized. The current
AdGeometry/render boundary is authoritative; obsolete fitted-curve sidecars
are reconciled, rather than used to move the current lower-numbered model.
This opt-in module never changes source pixels or solver admission roles.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

import numpy as np

try:
    from .ad_landmarks import AdGeometry, ROLES
    from .render_result import advertisement_boundary_paths
except ImportError:
    from ad_landmarks import AdGeometry, ROLES
    from render_result import advertisement_boundary_paths


PRINT_CLOCKWISE_ROLES = ("left_top", "right_top", "right_bottom", "left_bottom")
_ID = re.compile(r"ad([1-9]|1[0-9]|2[0-6]):(left_top|left_bottom|right_top|right_bottom|left_vertical_xyz|right_vertical_xyz|top_edge_xyz|bottom_edge_xyz)\Z", re.I)
_SEGMENTS = {"upper_right": "lower_left", "lower_left": "upper_right",
             "upper_left": "lower_right", "lower_right": "upper_left",
             "upper_long_side": "lower_long_side", "lower_long_side": "upper_long_side",
             "left_short_side": "right_short_side", "right_short_side": "left_short_side"}
_XYZ_KEYS = {"xyz", "xyz_scene_units", "world_xyz", "world_xyz_scene_units",
             "center_xyz", "points_xyz", "contour_xyz", "corners_xyz_scene_units"}
_XYZ_VECTOR_KEYS = {"normal_xyz", "plane_normal_xyz", "basis_u_xyz", "basis_v_xyz", "tangent_xyz"}
_SOURCE_ID_ALIASES = {"advertisement7.left_outer_boundary": "ad7:left_vertical_xyz"}
_CURVE_ENDPOINT_ROLES = {"left_vertical_xyz": ("left_bottom", "left_top"),
                         "right_vertical_xyz": ("right_bottom", "right_top"),
                         "top_edge_xyz": ("left_top", "right_top"),
                         "bottom_edge_xyz": ("left_bottom", "right_bottom")}


def _digest(value):
    return hashlib.sha256(np.asarray(value, np.float64).tobytes()).hexdigest()


def _paths(scene):
    return {(int(r["number"]), r["role"]): np.asarray(r["xyz"], np.float64).copy()
            for r in advertisement_boundary_paths(scene)}


def _geometry(scene):
    numbers = [int(r["user_number"]) for r in scene.get("advertisements", [])]
    if len(numbers) != len(set(numbers)) or any(n < 1 or n > 26 for n in numbers):
        raise ValueError("Advertisement user numbers must be unique in 1..26")
    return {n: AdGeometry(scene, n) for n in numbers}


def _centre(scene):
    # The released coordinate definition names the user's outline centre as
    # origin. Independently verify the boundary is centred there; an ad pair
    # must not silently introduce a different translational gauge.
    if scene.get("coordinate_system", {}).get("origin") != "user_PPT_ice_outline_center":
        raise ValueError("Explicit user ice-outline centre coordinate authority required")
    xy = np.asarray(scene["ice_boundary"]["points_xyz"], np.float64)[:, :2]
    centre = (xy.min(axis=0)+xy.max(axis=0))/2.
    if np.max(np.abs(centre)) > 1e-7:
        raise ValueError("User-defined outline centre and world origin disagree")
    return np.zeros(2, np.float64)


def _rotate(value, centre):
    result = np.asarray(value, np.float64).copy()
    if result.shape[-1:] != (3,) or not np.isfinite(result).all():
        raise ValueError("Finite world XYZ required for advertisement half-turn")
    result[..., :2] = 2*centre-result[..., :2]
    return result


def _mirror_descriptors(value, centre, key=None):
    """Retain local pattern metadata; transform only explicit world XYZ fields."""
    if key in _XYZ_VECTOR_KEYS:
        return None if value is None else _rotate(value, np.zeros(2)).tolist()
    if key in _XYZ_KEYS:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return {k: _rotate(v, centre).tolist() for k, v in value.items()}
        return _rotate(value, centre).tolist()
    if isinstance(value, Mapping):
        return {k: _mirror_descriptors(v, centre, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_mirror_descriptors(v, centre) if isinstance(v, Mapping) else copy.deepcopy(v) for v in value]
    return copy.deepcopy(value)


class SharedCentrosymmetricAdvertisementGeometry:
    def __init__(self, scene, *, authority=None):
        self.original = copy.deepcopy(scene)
        self.centre_xy = _centre(scene)
        self.authority = copy.deepcopy(authority or {
            "kind": "direct_user_instruction_20261003",
            "rule": "Ad(x) and Ad(x+13), x=1..13: horizontal rink-centre 180-degree rotation; identical Z, four-corner dimensions",
            "source_uv_or_world_inferred_from_camera": False})
        old = {int(r["user_number"]): r for r in scene.get("advertisements", [])}
        geometries = _geometry(scene)
        paths = _paths(scene)
        result = copy.deepcopy(scene)
        new = {}
        pairs = []
        reconciled = []
        for low in range(1, 14):
            high = low+13
            if low not in old and high not in old:
                continue
            origin = low if low in old else high
            base = copy.deepcopy(old[origin])
            # Materialize actual current boundaries. Field names in an old
            # candidate do not establish which geometry the solver used.
            actual_corners = {r: geometries[origin].corner(geometries[origin].initial, r).tolist()
                              for r in geometries[origin].roles}
            actual_paths = {role: xyz.tolist() for (number, role), xyz in paths.items() if number == origin}
            prior_paths = base.get("fitted_boundary_curves_xyz", {})
            if prior_paths != actual_paths:
                reconciled.append(dict(user_number=origin, old_fitted_curve_sha256=hashlib.sha256(
                    json.dumps(prior_paths, sort_keys=True).encode()).hexdigest(),
                    current_geometry_source="AdGeometry.corner(initial), advertisement_boundary_paths(scene)"))
            base["corners_xyz_scene_units"] = actual_corners
            base["fitted_boundary_curves_xyz"] = actual_paths
            base["boundary_construction"] = "Actual AdGeometry finite boundary; unified source/render geometry"
            if origin == high:
                base = _mirror_descriptors(base, self.centre_xy)
                base["corners_xyz_scene_units"] = {r: _rotate(xyz, self.centre_xy).tolist() for r, xyz in actual_corners.items()}
                base["fitted_boundary_curves_xyz"] = {r: _rotate(xyz, self.centre_xy).tolist() for r, xyz in actual_paths.items()}
                base["board_segment"] = _SEGMENTS.get(base.get("board_segment"), base.get("board_segment"))
                base["inverse_pair_origin_user_number"] = high
            base["user_number"] = low
            base["centrosymmetric_pair_authority"] = copy.deepcopy(self.authority)
            new[low] = base
            derived = _mirror_descriptors(base, self.centre_xy)
            derived["user_number"] = high
            derived["board_segment"] = _SEGMENTS.get(base.get("board_segment"), base.get("board_segment"))
            derived["corners_xyz_scene_units"] = {r: _rotate(xyz, self.centre_xy).tolist() for r, xyz in base["corners_xyz_scene_units"].items()}
            derived["fitted_boundary_curves_xyz"] = {r: _rotate(xyz, self.centre_xy).tolist() for r, xyz in base["fitted_boundary_curves_xyz"].items()}
            derived["derived_from_user_number"] = low
            derived["geometry_status"] = "user_hard_centrosymmetric_pair; shared conditional base geometry, not independent measurement"
            # Fraction fields belong to an older fit parameterization. The
            # finite actual boundary positions are now the common authority.
            derived["along_fractions_status"] = "historical base descriptor only; actual world corners define derived stations"
            derived["boundary_construction"] = "Exact horizontal half-turn of paired actual finite boundary; identical Z"
            new[high] = derived
            pairs.append(dict(base_user_number=low, derived_user_number=high,
                              base_origin_user_number=origin,
                              previous_high_geometry=copy.deepcopy(old.get(high)),
                              base_actual_corners=copy.deepcopy(base["corners_xyz_scene_units"])))
        result["advertisements"] = [new[n] for n in sorted(new)]
        self.actual = result
        self.geometries = _geometry(result)
        self.actual_paths = _paths(result)
        for r in result.get("advertisement_3d_corners", []):
            n, role = int(r["user_number"]), r["role"]
            if n in self.geometries and role in self.geometries[n].roles:
                old_world = copy.deepcopy(r.get("world_xyz_scene_units"))
                r["world_xyz_scene_units"] = self.corner(n, role).tolist()
                if old_world != r["world_xyz_scene_units"]:
                    r["original_world_xyz_scene_units_before_centrosymmetric_binding"] = old_world
                    r["centrosymmetric_world_binding_authority"] = copy.deepcopy(self.authority)
        self.audit = dict(schema="rinkspace.shared-centrosymmetric-advertisement-geometry.v1",
                          authority=copy.deepcopy(self.authority), centre_xy_scene_units=self.centre_xy.tolist(),
                          original_numbers=sorted(old), actual_numbers=sorted(new), pairs=pairs,
                          historical_fitted_curves_reconciled=reconciled,
                          missing_both_pairs=[n for n in range(1, 14) if n not in new],
                          source_pixels_roles_sigmas_not_modified=True,
                          print_corner_role_order=list(PRINT_CLOCKWISE_ROLES),
                          per_frame_projected_sorting_used=False,
                          no_missing_ad_measurement_or_corner_source_created=True)
        result["shared_centrosymmetric_advertisement_authority"] = dict(
            schema=self.audit["schema"], authority=copy.deepcopy(self.authority),
            materialized_pairs=[[r["base_user_number"], r["derived_user_number"]] for r in pairs],
            current_actual_boundary_used=True, source_and_render_geometry_same=True)

    def scene(self):
        return copy.deepcopy(self.actual)

    def manifest(self):
        return copy.deepcopy(self.audit)

    def corner(self, number, role):
        geometry = self.geometries[int(number)]
        return geometry.corner(geometry.initial, role)

    def quad(self, number, roles=PRINT_CLOCKWISE_ROLES):
        """Physical print TL/TR/BR/BL; facing-side identity must be source-qualified."""
        return np.stack([self.corner(number, role) for role in roles])

    def boundary(self, number, role):
        return self.actual_paths[int(number), role].copy()

    def rebind(self, row):
        return AdvertisementSourceBinding(self.original, self.actual).rebind(row)


class AdvertisementSourceBinding:
    """Rebind existing advertisement source world; leave every observation intact."""
    def __init__(self, old_scene, new_scene):
        self.old_scene, self.new_scene = old_scene, new_scene
        self.old, self.new = _geometry(old_scene), _geometry(new_scene)
        self.paths = _paths(new_scene)

    @staticmethod
    def _identity(row):
        ids = row.get("physical_ids")
        if ids:
            parsed = [_ID.fullmatch(_SOURCE_ID_ALIASES.get(str(x), str(x))) for x in ids]
            if any(x is None for x in parsed):
                if any(str(x).lower().startswith("ad") for x in ids):
                    raise ValueError("Mixed/unsupported explicit advertisement physical_ids")
                return None
            return [(int(x[1]), x[2].lower()) for x in parsed]
        match = _ID.fullmatch(str(row.get("id", "")))
        if match:
            return [(int(match[1]), match[2].lower())]
        # Historical interval source has a separate explicit ad_id and a
        # fully specified physical boundary role, with a retained source ID.
        if row.get("ad_id") is not None and row.get("role") in (*ROLES, "left_vertical_xyz", "right_vertical_xyz", "top_edge_xyz", "bottom_edge_xyz"):
            return [(int(row["ad_id"]), row["role"])]
        if row.get("id") == "ad7_left_vertical":
            return [(7, "left_vertical_xyz")]
        return None

    def _map_local(self, world, number):
        old, new = self.old[number], self.new[number]
        world = np.asarray(world, np.float64).reshape(-1, 3)
        if not np.isfinite(world).all():
            raise ValueError("Finite advertisement source world required")
        lo, hi = old._base_height["bottom"], old._base_height["top"]
        t = (world[:, 2]-lo)/(hi-lo)
        if np.any((t < -1e-5) | (t > 1+1e-5)):
            raise ValueError("Advertisement source height outside its original finite panel")
        arcs = np.array([old._nearest_arc(w[:2]) for w in world])
        snapped = np.array([old._point_at_arc(s) for s in arcs])
        if np.max(np.linalg.norm(snapped-world[:, :2], axis=1), initial=0.) > 1e-5:
            raise ValueError("Advertisement source world is not on its original actual finite board")
        if old._nominal_width is None:
            if np.max(np.linalg.norm(snapped-old._point_at_arc(old._base_s["left"]), axis=1), initial=0.) > 1e-5:
                raise ValueError("Partial advertisement has no known along-panel coordinate")
            xy = np.repeat(new._point_at_arc(new._base_s["left"])[None, :], len(world), axis=0)
        else:
            delta = (arcs-old._base_s["left"]+old._perimeter/2)%old._perimeter-old._perimeter/2
            u = delta/(old._width_sign*old._nominal_width)
            if np.any((u < -1e-5) | (u > 1+1e-5)):
                raise ValueError("Advertisement source world outside its original finite along-panel support")
            xy = np.array([new._point_at_arc(new._base_s["left"]+v*new._width_sign*new._nominal_width) for v in u])
        z = new._base_height["bottom"]+t*(new._base_height["top"]-new._base_height["bottom"])
        return np.column_stack((xy, z))

    def rebind(self, row):
        identities = self._identity(row)
        if identities is None:
            return row
        if "world" not in row:
            raise ValueError("Advertisement source must contain its actual world array")
        original = np.asarray(row["world"], np.float64).reshape(-1, 3)
        for n, _ in identities:
            if n not in self.old or n not in self.new:
                raise ValueError(f"Existing source ad{n} missing in old/new common model")
        point = bool(row.get("point_correspondence")) or all(role in ROLES for _, role in identities)
        if point:
            if len(identities) != len(original):
                raise ValueError("Explicit advertisement corner identities and world count differ")
            rebound = np.stack([self.new[n].corner(self.new[n].initial, role) for n, role in identities])
        else:
            if len(identities) != 1:
                raise ValueError("Advertisement curve requires one explicit physical boundary")
            number, role = identities[0]
            rebound = self._map_local(original, number)
            # Curve samples are model support, not point correspondences.
            # Rebuild complete curved boundaries on the exact new board
            # vertices so a mapped chord cannot cut a corner that rendering
            # retains. Observation UV, residual count and sigma stay intact.
            if role in _CURVE_ENDPOINT_ROLES:
                old = self.old[number]
                first_role, last_role = _CURVE_ENDPOINT_ROLES[role]
                left = old.corner(old.initial, first_role)
                right = old.corner(old.initial, last_role)
                changed = any(not np.allclose(old.corner(old.initial, r), self.new[number].corner(
                    self.new[number].initial, r), rtol=0., atol=1e-10) for r in old.roles)
                if changed and np.allclose(original[[0, -1]], np.stack((left, right)), rtol=0., atol=1e-8):
                    rebound = self.paths[number, role].copy()
                elif changed and np.allclose(original[[0, -1]], np.stack((right, left)), rtol=0., atol=1e-8):
                    rebound = self.paths[number, role][::-1].copy()
                elif changed and role in ("top_edge_xyz", "bottom_edge_xyz"):
                    # Explicit partial supports retain the same normalized
                    # endpoints, plus every intervening physical board vertex.
                    g = self.new[number]
                    def coordinate(xyz):
                        arcs = np.array([g._nearest_arc(x[:2]) for x in xyz])
                        return ((arcs-g._base_s["left"]+g._perimeter/2)%g._perimeter-g._perimeter/2)/(g._width_sign*g._nominal_width)
                    u = coordinate(rebound)
                    whole = self.paths[number, role]
                    whole_u = coordinate(whole)
                    selected = whole[(whole_u > u.min()+1e-10)&(whole_u < u.max()-1e-10)]
                    mixed = np.vstack((rebound, selected))
                    order = np.argsort(coordinate(mixed))
                    if u[-1] < u[0]:
                        order = order[::-1]
                    rebound = mixed[order]
        if original.shape == rebound.shape and np.allclose(original, rebound, rtol=0., atol=1e-10):
            # Existing low-numbered source is bit-exact when the current
            # actual base has not changed; do not manufacture roundoff edits.
            return row
        result = dict(row)
        result["world"] = rebound if isinstance(row["world"], np.ndarray) else rebound.tolist()
        for key in ("world_xyz", "world_xyz_scene_units"):
            if key in row:
                value = np.asarray(row[key])
                if value.shape == (3,) and len(rebound) == 1:
                    result[key] = rebound[0].tolist()
                elif value.shape == original.shape:
                    result[key] = rebound.tolist()
                else:
                    raise ValueError("Advertisement world companion shape differs")
        result["centrosymmetric_world_rebinding"] = dict(
            original_world_float64_sha256=_digest(original), new_world_float64_sha256=_digest(rebound),
            method="same explicit print corner or same normalized actual finite-panel arc/height coordinate",
            physical_ids=[f"ad{n}:{role}" for n, role in identities],
            source_uv_sigma_id_split_role_unchanged=True,
            authority=self.new_scene.get("shared_centrosymmetric_advertisement_authority"))
        return result


def rebind_ad_source_rows(old_scene, new_scene, rows):
    binding = AdvertisementSourceBinding(old_scene, new_scene)
    return [binding.rebind(row) for row in rows]


def rebind_ad_source_groups(old_scene, new_scene, groups):
    binding = AdvertisementSourceBinding(old_scene, new_scene)
    return [(pts, [binding.rebind(row) for row in rows]) for pts, rows in groups]


def build_centrosymmetric_ad_scene(base, *, authority=None):
    helper = SharedCentrosymmetricAdvertisementGeometry(base, authority=authority)
    return helper.scene(), helper.manifest()


def corner_xyz(scene, adnumber, role):
    geometry = AdGeometry(scene, int(adnumber))
    return geometry.corner(geometry.initial, role)


def mirror_pattern_descriptor(descriptor, source_number, target_number, *, centre_xy=(0., 0.)):
    """Mirror explicit world pattern coordinates; keep print-local metadata.

    Circle/point patterns are separate from a rectangular AdGeometry panel.
    This function does not create a rectangle, a missing measurement, or a
    guessed world position for an observed pattern with unknown coordinates.
    """
    source_number, target_number = int(source_number), int(target_number)
    if not (1 <= source_number <= 26 and 1 <= target_number <= 26 and abs(source_number-target_number) == 13):
        raise ValueError("A pattern must use its explicitly numbered 13-separated pair")
    centre = np.asarray(centre_xy, np.float64)
    if centre.shape != (2,) or not np.isfinite(centre).all():
        raise ValueError("Finite shared ice-outline centre required")
    result = _mirror_descriptors(descriptor, centre)
    result["user_number"] = target_number
    result["derived_from_user_number"] = source_number
    result["horizontal_centrosymmetry"] = dict(centre_xy_scene_units=centre.tolist(),
        print_local_descriptor_unchanged=True, source_pixels_not_generated=True,
        explicit_world_XYZ_half_turn=True, Z_unchanged=True)
    return result


def rebind_ad_row(row, oldscene, newscene):
    return AdvertisementSourceBinding(oldscene, newscene).rebind(row)


def audit_finite_advertisement_domains(old_scene, new_scene, finite_records):
    """Named board facets are unchanged. Explicit ad domains need a new adapter."""
    if (old_scene["ice_boundary"] != new_scene["ice_boundary"] or
            old_scene["board_surface"] != new_scene["board_surface"]):
        raise ValueError("Advertisement pairing must not change physical board facets")
    explicit, requires_rebind = [], []
    old, new = _geometry(old_scene), _geometry(new_scene)
    for i, row in enumerate(finite_records):
        identifiers = {k: row.get(k) for k in ("ad_id", "advertisement_id", "physical_advertisement_id", "surface_id", "physical_id", "physical_surface_identity") if row.get(k) is not None}
        numbers = set()
        for key, value in identifiers.items():
            if key in ("ad_id", "advertisement_id", "physical_advertisement_id"):
                numbers.add(int(value))
            else:
                numbers.update(int(m[1]) for m in re.finditer(r"(?:^|[^a-z0-9])ad(\d+)(?:$|[^0-9])", str(value), re.I))
        if numbers:
            entry = dict(index=i, identifiers=identifiers, ad_numbers=sorted(numbers))
            explicit.append(entry)
            for n in numbers:
                if n not in old or n not in new or old[n].roles != new[n].roles or any(not np.allclose(
                        old[n].corner(old[n].initial, role), new[n].corner(new[n].initial, role), rtol=0., atol=1e-10) for role in old[n].roles):
                    requires_rebind.append(entry)
    if requires_rebind:
        raise ValueError(f"Changed advertisement finite domains require source-specific rebind, not unchanged board plane: {requires_rebind[:3]}")
    return dict(count=len(finite_records), explicit_advertisement_domains=explicit,
                unchanged_ad_domains=[r for r in explicit if r not in requires_rebind],
                changed_advertisement_finite_domains=requires_rebind,
                board_xy_z_unchanged=True, finite_planes_and_original_material_support_unchanged=True,
                rationale="Original named physical board facets do not move when their printed advertisement extents change")
