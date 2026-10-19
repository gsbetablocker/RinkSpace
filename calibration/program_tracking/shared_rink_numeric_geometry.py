"""Four shared metric rink dimensions, with fixed measured source identities.

This is a versioned diagnostic adapter; importing it neither solves nor saves a
scene. q = [F column dx, R column dx, row pitch delta, outer radius delta], in
metres relative to the supplied scene. In particular a 427806 base is already
translated from hard93c: there is no implicit additional F-column translation.

User-confirmed relations constrain all eight circles, not their numeric sizes.
Current inner/outer radius ratios are inherited estimates, not measured user
dimensions. Lane polygons retain their existing widths, tilt and imperfect row
alignment: pitch changes translate each whole Lane in Y. Their unmeasured left
proxies do not follow column X or radius. The row-3 carrier follows row pitch.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

import numpy as np


# Same fixed latest metric as z190_continuous_source_bundle.SCENE_UNIT_M.
# This historical scene's coordinate_system.metres_per_unit is not its runtime
# calibration, so the adapter takes the metric explicitly rather than guessing.
SCENE_UNIT_M = 0.31353729994266055
VERSION = "shared_rink_numeric_geometry_v1"
PARAMETER_NAMES = ("F_column_dx_m", "R_column_dx_m", "row_pitch_delta_m",
                   "outer_radius_delta_m")
RING_ROLES = ("outer", "middle", "inner", "center")


class GeometryIdentityError(ValueError):
    """Source identity or base-world geometry cannot be safely rebound."""


class SharedRinkNumericGeometry:
    """One shared numeric rink shape, without frame/ring-specific freedom.

    world_jacobian is d(world scene units)/d(q metres), shape (N, 3, 4).
    Source UV, normal support intervals, sigma, roles and admission never change.
    The caller supplies already-rebound source rows from this exact base scene.
    """

    def __init__(self, scene, scene_unit_m=SCENE_UNIT_M):
        self.base = deepcopy(scene)
        self.scene_unit_m = float(scene_unit_m)
        if not np.isfinite(self.scene_unit_m) or self.scene_unit_m <= 0:
            raise ValueError("scene_unit_m must be a positive fixed metric")
        self.features = {f["id"]: f for f in self.base["ice_features"]}
        if len(self.features) != len(self.base["ice_features"]):
            raise GeometryIdentityError("Duplicate physical feature IDs")
        self.names = [f"{col}{row}" for col in "FR" for row in range(1, 5)]
        for name in self.names:
            f = self.features[name]
            if f["kind"] != "circle":
                raise GeometryIdentityError(f"{name} must be a circle")
        self.centres = {n: np.asarray(self.features[n]["center_xyz"], float)
                        for n in self.names}
        self.outer_radius = float(self.features["F1"]["radius"])
        self.column_x = {c: float(self.centres[c+"1"][0]) for c in "FR"}
        ys = np.array([self.centres[f"F{i}"][1] for i in range(1, 5)])
        self.row_midpoint = float((ys[0]+ys[-1])/2)
        self.row_pitch = float((ys[-1]-ys[0])/3)
        rings = {r["role"]: r for r in self.features["F1"].get("rings", [])}
        if set(rings) != set(RING_ROLES):
            raise GeometryIdentityError("Four explicit common FR ring roles required")
        self.radius_ratios = {r: float(rings[r]["radius"])/self.outer_radius
                              for r in RING_ROLES}
        self._validate_base()
        self.base_scene_sha256 = hashlib.sha256(json.dumps(
            self.base, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False).encode("utf8")).hexdigest()

    def _validate_base(self):
        if self.outer_radius <= 0 or self.row_pitch <= 0:
            raise GeometryIdentityError("Base radius/pitch must be positive")
        tol = 1e-8
        for name in self.names:
            f, c = self.features[name], self.centres[name]
            expected = [self.column_x[name[0]],
                        self.row_midpoint+(int(name[1])-2.5)*self.row_pitch, 0]
            if not np.allclose(c, expected, atol=tol, rtol=0):
                raise GeometryIdentityError(f"Base {name} violates paired/equal rows")
            if abs(float(f["radius"])-self.outer_radius) > tol:
                raise GeometryIdentityError("Base FR outer radii differ")
            rings = {r["role"]: r for r in f.get("rings", [])}
            if len(rings) != len(f.get("rings", [])) or set(rings) != set(RING_ROLES):
                raise GeometryIdentityError(f"Base {name} ring roles differ")
            for role, ring in rings.items():
                if not np.allclose(ring["center_xyz"], c, atol=tol, rtol=0):
                    raise GeometryIdentityError(f"Base {name}/{role} not concentric")
                if abs(float(ring["radius"])-self.outer_radius*self.radius_ratios[role]) > tol:
                    raise GeometryIdentityError(f"Base {name}/{role} ratio differs")
        if not (self.column_x["F"] < self.column_x["R"]):
            raise GeometryIdentityError("Base physical F/R column order reversed")
        ratios = [self.radius_ratios[r] for r in RING_ROLES]
        if not (ratios[0] == 1 and all(a > b > 0 for a, b in zip(ratios, ratios[1:]))):
            raise GeometryIdentityError("Invalid inherited concentric ring order")
        carrier = self.features.get("curling_centerline_row3")
        if carrier is not None:
            path = np.asarray(carrier["segments_xyz"], float)
            if not np.allclose(path[..., 1], self.centres["F3"][1], atol=tol, rtol=0):
                raise GeometryIdentityError("Base row-3 carrier not through F3/R3 Y")

    def _q(self, q4):
        q = np.asarray(q4, float)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("q4 must contain exactly four finite metre deltas")
        if self.row_pitch+q[2]/self.scene_unit_m <= 0:
            raise ValueError("Nonpositive common row pitch")
        if self.outer_radius+q[3]/self.scene_unit_m <= 0:
            raise ValueError("Nonpositive common outer radius")
        if self.column_x["F"]+q[0]/self.scene_unit_m >= self.column_x["R"]+q[1]/self.scene_unit_m:
            raise ValueError("Physical F/R column order reversed")
        return q

    def validate_parameters(self, q4):
        """Check positivity/order and all eight circles inside modeled ice.

        Signed distance to the existing finite ice polygon bounds the complete
        Euclidean circle, rather than checking a few candidate vertices. This
        is the existing board footprint, not an assumed generic rink size.
        """
        import cv2
        q = self._q(q4)
        polygon = np.ascontiguousarray(np.asarray(
            self.base["ice_boundary"]["points_xyz"], np.float32)[:, :2])
        radius = self.outer_radius+q[3]/self.scene_unit_m
        for name in self.names:
            centre = self.centres[name][:2]+np.array([
                q[0 if name[0] == "F" else 1], (int(name[1])-2.5)*q[2]])/self.scene_unit_m
            margin = cv2.pointPolygonTest(polygon, tuple(map(float, centre)), True)
            if margin+1e-5 < radius:
                raise ValueError(f"Complete physical {name} outer circle leaves modeled ice footprint")
        return q.copy()

    def manifest(self, q4=None):
        q = self._q(np.zeros(4) if q4 is None else q4)
        return dict(version=VERSION, parameter_names=list(PARAMETER_NAMES),
                    parameters_m=q.tolist(), scene_unit_m=self.scene_unit_m,
                    base_scene_canonical_sha256=self.base_scene_sha256,
                    base_column_x_m={c: x*self.scene_unit_m for c, x in self.column_x.items()},
                    base_row_midpoint_m=self.row_midpoint*self.scene_unit_m,
                    base_row_pitch_m=self.row_pitch*self.scene_unit_m,
                    base_outer_radius_m=self.outer_radius*self.scene_unit_m,
                    inherited_inner_outer_radius_ratios=self.radius_ratios,
                    numeric_provenance="old source-model mean columns with retained later source-diagnostic column translations, least-squares equal-row pitch and median ring radii; not user-measured dimensions",
                    relation_provenance=["configs/rink_model/confirmed_ring_layout.json",
                                         "configs/rink_model/confirmed_layout_v2.json",
                                         "configs/rink_model/confirmed_layout_v3.json"],
                    numeric_initialization_provenance="program_tracking/shared_f_column_source_diagnostic.py:hard_fr_base; supplied scene retains its full historical source provenance",
                    radius_ratio_status="fixed inherited numerical assumption; no independent measured ratio claim",
                    confirmed_relations="two co-linear columns; eight equal concentric outer circles; four paired equally spaced rows; current row midpoint fixed",
                    lane_policy="paired-row rigid Y translation only; retain existing numeric width, tilt, row offset and unmeasured left proxy",
                    lane_limitation="does not restore exact equal width, right alignment or exact F/R row incidence",
                    carrier_policy="curling_centerline_row3 translates with row3 Y; no endpoint-X proxy constraint",
                    other_world_geometry="unchanged; including C/C0, boundaries, board and advertisements",
                    source_world_sampling="preserve every supplied polyline vertex and order; no smoothing, densification or new source samples",
                    state="diagnostic numeric adapter; not a source acceptance or adopted scene")

    def scene(self, q4):
        q = self.validate_parameters(q4)
        out = deepcopy(self.base)
        features = {f["id"]: f for f in out["ice_features"]}
        factor = 1+q[3]/(self.outer_radius*self.scene_unit_m)
        for name in self.names:
            f = features[name]
            delta = np.array([q[0 if name[0] == "F" else 1],
                              (int(name[1])-2.5)*q[2], 0])/self.scene_unit_m
            centre = (self.centres[name]+delta).tolist()
            f["center_xyz"] = centre
            f["radius"] = self.features[name]["radius"]*factor
            original_radii = {r["role"]: r["radius"] for r in self.features[name]["rings"]}
            for ring in f["rings"]:
                ring["center_xyz"] = list(centre)
                ring["radius"] = original_radii[ring["role"]]*factor
        for i in range(1, 5):
            name = f"lane{i}"
            if name not in features:
                continue
            delta_y = (i-2.5)*q[2]/self.scene_unit_m
            for field in ("polygon_xyz", "centerline_xyz"):
                if field in features[name]:
                    xyz = np.asarray(self.features[name][field], float).copy()
                    xyz[..., 1] += delta_y
                    features[name][field] = xyz.tolist()
        name = "curling_centerline_row3"
        if name in features:
            xyz = np.asarray(self.features[name]["segments_xyz"], float).copy()
            xyz[..., 1] += .5*q[2]/self.scene_unit_m
            features[name]["segments_xyz"] = xyz.tolist()
            if "relation" in features[name]:
                features[name]["relation"]["common_y"] = self.centres["F3"][1]+.5*q[2]/self.scene_unit_m
        out[VERSION] = self.manifest(q)
        return out

    def _identity(self, row):
        original_name = str(row["id"])
        name = str(row.get("physical_curve_id", original_name))
        canonical = row.get("canonical_physical_geometry_id")
        if canonical is not None and str(canonical) != name:
            raise GeometryIdentityError("Canonical physical curve metadata disagree")
        if name != original_name:
            # A qualified adapter may retain a detailed original source ID and
            # add its independently established canonical physical curve ID.
            # The full world must still pass the exact base feature test below.
            authority = any(row.get(k) for k in (
                "origin", "physical_identity_authority", "source_qualification",
                "source_branch_operator_authority", "physical_curve_id_authority"))
            if not authority:
                raise GeometryIdentityError("Canonical source identity requires retained source authority")
        fr = re.fullmatch(r"([FR][1-4])(?:_(outer|middle|inner|center))?", name)
        if fr:
            return ("circle", fr[1], fr[2] or "outer")
        if re.match(r"[FR]\d", name):
            raise GeometryIdentityError(f"Unsupported or ambiguous physical FR identity {name}")
        lane = re.fullmatch(r"(lane[1-4])_(upper|lower|(?:edge)?[0-3])", name)
        if lane:
            f = self.features[lane[1]]
            role = lane[2]
            if role in ("upper", "lower"):
                edge = f["lane_edge_order"].index(role+"_long")
            else:
                edge = int(role.removeprefix("edge"))
            if edge in f.get("open_edges", []):
                raise GeometryIdentityError(f"Unmeasured Lane proxy edge {name} is not source paint")
            return ("lane", lane[1], edge)
        if name.startswith("lane"):
            raise GeometryIdentityError(f"Explicit measured finite Lane edge required: {name}")
        if name == "curling_centerline_row3":
            return ("carrier", name, None)
        # Known fixed circles can be audited too, but never get an FR derivative.
        fixed = re.fullmatch(r"(.+)_(outer|middle|inner|center)", name)
        if fixed and fixed[1] in self.features and self.features[fixed[1]]["kind"] == "circle":
            return ("fixed_circle", fixed[1], fixed[2])
        return None

    def feature_id(self, row):
        identity = self._identity(row)
        return None if identity is None else identity[1]

    @staticmethod
    def _finite_line_check(world, endpoints, name):
        a, b = np.asarray(endpoints, float)
        d = b-a
        denom = float(d@d)
        if denom <= 0:
            raise GeometryIdentityError(f"Degenerate finite source line {name}")
        t = (world-a)@d/denom
        if (np.max(np.linalg.norm(world-(a+t[:, None]*d), axis=1)) > 1e-7 or
                np.min(t) < -1e-8 or np.max(t) > 1+1e-8):
            raise GeometryIdentityError(f"Source world does not lie on this base finite line: {name}")

    def _world_and_jacobian(self, row, q4):
        q = self._q(q4)
        before = np.asarray(row["world"], float)
        if before.ndim != 2 or before.shape[1] != 3 or not len(before) or not np.isfinite(before).all():
            raise GeometryIdentityError("Source world must be a nonempty finite N by 3 array")
        dw = np.zeros((len(before), 3, 4))
        identity = self._identity(row)
        if identity is None:
            return before.copy(), dw
        kind, name, role = identity
        f = self.features[name]
        if kind in ("circle", "fixed_circle"):
            rings = {r["role"]: r for r in f.get("rings", [])}
            ring = rings.get(role)
            if ring is None:
                if role != "outer":
                    raise GeometryIdentityError(f"No physical {role} ring in {name}")
                ring = f
            centre = np.asarray(ring["center_xyz"], float)
            radial = before-centre
            if (np.max(np.abs(radial[:, 2])) > 1e-8 or
                    np.max(np.abs(np.linalg.norm(radial[:, :2], axis=1)-float(ring["radius"]))) > 1e-7):
                raise GeometryIdentityError(f"Source {row['id']} world is stale, wrong ring, or not this base circle")
            if kind == "circle":
                dw[:, 0, 0 if name[0] == "F" else 1] = 1/self.scene_unit_m
                dw[:, 1, 2] = (int(name[1])-2.5)/self.scene_unit_m
                dw[:, :2, 3] = radial[:, :2]/(self.outer_radius*self.scene_unit_m)
        elif kind == "lane":
            polygon = np.asarray(f["polygon_xyz"], float)
            self._finite_line_check(before, [polygon[role], polygon[(role+1)%len(polygon)]], row["id"])
            dw[:, 1, 2] = (int(name[-1])-2.5)/self.scene_unit_m
        elif kind == "carrier":
            segments = f["segments_xyz"]
            if len(segments) != 1:
                raise GeometryIdentityError("Explicit segment identity required for multiple carrier paths")
            self._finite_line_check(before, segments[0], row["id"])
            dw[:, 1, 2] = .5/self.scene_unit_m
        return before+np.einsum("nij,j->ni", dw, q), dw

    def rebind(self, row, q4):
        result = dict(row)
        result["world"] = self._world_and_jacobian(row, q4)[0]
        # All other original source keys are deliberately unchanged, including
        # original immutable role/UV/admission/provenance fields.
        return result

    def world_jacobian(self, row, q4):
        return self._world_and_jacobian(row, q4)[1]

    def project(self, camera, row, q4, *, with_jacobian=True):
        """Actual distorted fixed-camera vertices and their numeric-shape J.

        This returns full vertices, never selected source branches/nearest
        points. A source helper must still apply its retained branch, interval
        and full central-forward domain. Camera C Jacobian uses C/100 units.
        """
        try:
            from .projection import project_central_valid
        except ImportError:
            from projection import project_central_valid
        world, dw = self._world_and_jacobian(row, q4)
        if not with_jacobian:
            return project_central_valid(camera, world)
        uv, depth, camera_jac, central = project_central_valid(camera, world, with_jacobian=True)
        jc = camera_jac.reshape(len(world), 2, -1)
        geometry_jac = np.einsum("nij,njk->nik", -jc[:, :, :3]/100., dw)
        return uv, depth, geometry_jac, central
