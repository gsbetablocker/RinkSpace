"""Explicit disjoint-print successor on the existing row-time/W execution.

Import has no effect. Call install_successor only after historical source/parent
authentication and the existing value/raw/fused/camera installations, before
either execution pool or the new PrintLayoutDomain is constructed. No historical
producer, source pixel, source ID, clock, W, parameter or auxiliary is rewritten.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType, ModuleType
import numpy as np

KIND = "ad_print_curve"
SCHEMA = "rinkspace.explicit-disjoint-print-successor.v1"
TOPOLOGY = "rinkspace.explicit-print-path-union.v1"
_CONFIG = None
_ORIGINAL = {}


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _read(ref):
    if reference(ref["path"]) != {k:ref[k] for k in ("path", "sha256")}:
        raise ValueError("Frozen disjoint-print reference changed: "+ref["path"])
    return json.loads(Path(ref["path"]).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PathUnion:
    ids: tuple[str, ...]
    offsets: tuple[int, ...]
    closed: tuple[bool, ...]

    @classmethod
    def from_target(cls, target):
        value = target.get("curve_topology", {})
        ids, offsets, closed = (value.get(k) for k in ("path_ids", "offsets", "closed"))
        if (target.get("kind") != KIND or value.get("schema") != TOPOLOGY
                or not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or any(not isinstance(i, str) or not i for i in ids)
                or not isinstance(offsets, list) or len(offsets) != len(ids)+1
                or any(type(i) is not int for i in offsets) or offsets[0] != 0
                or any(b-a < 2 for a, b in zip(offsets[:-1], offsets[1:]))
                or not isinstance(closed, list) or len(closed) != len(ids)
                or any(type(i) is not bool for i in closed)):
            raise ValueError("Complete explicit true-path union required")
        if _CONFIG is not None and value.get("material_reference") != _CONFIG["manifest"]["material_override"]:
            raise ValueError("Target topology belongs to another material successor")
        return cls(tuple(ids), tuple(offsets), tuple(closed))

    def split(self, array):
        if len(array) != self.offsets[-1]:
            raise ValueError("Disjoint world/J/topology vertex inventory differs")
        return tuple(array[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:]))

    def edges(self):
        valid = np.ones(self.offsets[-1]-1, bool)
        valid[np.asarray(self.offsets[1:-1], int)-1] = False
        return valid


def _material(templates, number):
    from calibration.program_tracking.first_shot_print_layout import print_layout_template
    return print_layout_template(templates, number)


def _record_uv(record):
    uv = np.asarray(record.get("material_uv"), float)
    if (uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 2
            or not np.isfinite(uv).all() or type(record.get("closed")) is not bool):
        raise ValueError("Finite bank path with explicit closedness required")
    return uv


def build_material(official_prototype, coordinate_chart="parent_bbox", advertisement_id=7):
    """Build ONLY this advertisement's complete shape bank, not other ads.

    The fixed chart A transports official art into the old prototype bbox, so
    the complete inherited q/H8 can remain byte exact. It is an art placement
    initializer, not a measured physical dimension or old-shape equivalence.
    """
    if isinstance(official_prototype, (str, Path)):
        source = reference(official_prototype)
        official = _read(source)
    elif isinstance(official_prototype, dict) and set(official_prototype) == {"path", "sha256"}:
        source = deepcopy(official_prototype)
        official = _read(source)
    else:
        source, official = None, deepcopy(official_prototype)
    if coordinate_chart not in ("parent_bbox", "parent_hull"):
        raise ValueError("Explicit inherited parent_bbox or parent_hull chart required")
    bounds = np.asarray(official["original_template"]["original_prototype_material_bbox"], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Finite nondegenerate inherited prototype bbox required")
    lower, width = bounds[0], bounds[1]-bounds[0]
    hull_receipt = None
    if coordinate_chart == "parent_hull":
        from fractions import Fraction as F
        from calibration.program_tracking.first_shot_print_halfspaces import exact_hull_support
        parent = _read(official["original_template"])
        old_uv = np.concatenate([np.asarray(c.get("prototype_material_uv", c["material_uv"]), float)
            for c in _material(parent, advertisement_id)["contours"].values()])
        hull = old_uv[exact_hull_support(old_uv)]
        center = bounds.mean(axis=0)
        corners = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        directions = (corners-.5)*width
        ratios = [1.]
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            normal = np.array([b[1]-a[1], a[0]-b[0]])
            support = float(np.max(directions@normal))
            slack = float((a-center)@normal)
            if slack <= 0:
                raise ValueError("Inherited bbox center is not strictly inside original print hull")
            if support > 0:
                ratios.append(slack/support)
        scale = min(ratios)
        canonical_all = np.concatenate([np.asarray(c["canonical_print_uv"], float) for c in official["contours"].values()])
        if ((canonical_all < 0) | (canonical_all > 1)).any():
            raise ValueError("Official art leaves its explicitly declared unit canvas")
        def inside(points):
            # Dyadic float coordinates, exact signs, no geometric tolerance.
            hp = [[F(float(x)) for x in p] for p in hull]
            pp = [[F(float(x)) for x in p] for p in points]
            for a, b in zip(hp, hp[1:]+hp[:1]):
                for c in pp:
                    if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]) < 0:
                        return False
            return True
        scale = float(np.nextafter(scale, 0.))
        while True:
            mapped_width = width*scale
            mapped_lower = center-.5*mapped_width
            actual = mapped_lower+canonical_all*mapped_width
            rectangle = mapped_lower+corners*mapped_width
            if inside(np.vstack((rectangle, actual))):
                break
            scale = float(np.nextafter(scale, 0.))
            if scale <= 0:
                raise ValueError("No positive representable inherited-hull rectangle")
        lower, width = mapped_lower, mapped_width
        hull_receipt = dict(original_template=deepcopy(official["original_template"]),
            old_prototype_hull_vertices=hull.tolist(), uniform_scale=scale,
            old_bbox_center=center.tolist(), exact_dyadic_containment_verified=True,
            rectangle_and_every_actual_new_vertex_checked=True,
            scale_selection="largest common bbox-axis rectangle scale from original hull supports, nextafter inward for exact float containment",
            initializer_only_not_a_parameter_bound=True)
    affine = np.array([[width[0], 0., lower[0]+.5*width[0]-.5],
        [0., width[1], lower[1]+.5*width[1]-.5], [0., 0., 1.]])
    contours = {}
    for key, source_contour in official["contours"].items():
        canonical = np.asarray(source_contour["canonical_print_uv"], float)
        mapped = lower+canonical*width
        contours[key] = dict(advertisement_id=int(advertisement_id), closed=source_contour["closed"],
            role=source_contour["role"], material_uv=mapped.tolist(),
            canonical_print_uv=canonical.tolist(), official_contour_id=key,
            svg_path_index=source_contour.get("svg_path_index"),
            svg_subpath_index=source_contour.get("svg_subpath_index"))
        _record_uv(contours[key])
    aliases = {key:list(value["paths"]) for key,value in official["legacy_aliases"].items()}
    if any(p not in contours for ids in aliases.values() for p in ids):
        raise ValueError("Official alias refers outside the complete shape bank")
    return dict(schema="rinkspace.official-disjoint-print-material.v1", advertisement_number=int(advertisement_id),
        contours=contours, curve_aliases=aliases,
        alias_qualification=deepcopy(official["legacy_aliases"]), groups=deepcopy(official["groups"]),
        coordinate_chart=dict(name=coordinate_chart, parent_bbox=bounds.tolist(), centered_affine_A=affine.tolist(),
            mapped_lower=lower.tolist(), mapped_width=width.tolist(), parent_hull=hull_receipt,
            canonical_to_parent_material="uv_parent=mapped_lower+uv_canonical*mapped_width",
            full_parent_q_H8_retained=True, inherited_H8_applied_once=True,
            art_initialization_not_physical_dimensions_or_old_shape_equivalence=True),
        official_prototype=source, source_svg=deepcopy(official["source_svg"]),
        user_authority=official["user_authority"], accuracy_accepted=False,
        source_points_added_or_removed=False, per_glyph_or_per_frame_parameters_added=False)


def successor_target(old_target, templates, template_reference):
    """Retain the old label; aliases list whole confirmed paths, never an argmin."""
    number = old_target["advertisement_id"]
    material = _material(templates, number)
    ids = list(material.get("curve_aliases", {})[old_target["contour_key"]])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Nonempty unique explicit source path alias required")
    counts, closed = [], []
    for key in ids:
        rec = material["contours"][key]
        uv = _record_uv(rec)
        c = rec["closed"]
        counts.append(len(uv)+int(c and not np.array_equal(uv[0], uv[-1])))
        closed.append(c)
    result = deepcopy(old_target)
    result["kind"] = KIND
    result["curve_topology"] = dict(schema=TOPOLOGY, path_ids=ids,
        offsets=np.r_[0, np.cumsum(counts)].tolist(), closed=closed,
        material_reference=deepcopy(template_reference))
    PathUnion.from_target(result)
    return result


def feature_world(scene, target_descriptor, templates=None):
    if target_descriptor.get("kind") != KIND:
        return _ORIGINAL["feature_world"](scene, target_descriptor, templates)
    target = target_descriptor
    if target.get("geometry_source") is not None:
        raise ValueError("Disjoint template target cannot change geometry source")
    topology = PathUnion.from_target(target)
    paths = [_ORIGINAL["feature_world"](scene, dict(kind="ad_print",
        advertisement_id=target["advertisement_id"], contour_key=path_id), templates)
        for path_id in topology.ids]
    world = np.concatenate(paths)
    topology.split(world)
    for path, closed in zip(paths, topology.closed):
        if closed and not np.array_equal(path[0], path[-1]):
            raise ValueError("Bank path lost its own physical closure")
    return world


def _evaluate_disjoint(camera, rows, worlds, derivatives, *, jacobian):
    from calibration.program_tracking import first_shot_source_kernel as kernel
    key = rows[0]["_target_key"]
    target = rows[0]["target"]
    topology = PathUnion.from_target(target)
    if any(row["_target_key"] != key or row["target"] != target for row in rows):
        raise ValueError("One explicit source target union per query required")
    camera = np.asarray(camera, float)
    world = np.asarray(worlds[key], float)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("Finite Brown camera25 required")
    if (world.ndim != 2 or world.shape[1:] != (3,) or not np.isfinite(world).all()
            or len(world) != topology.offsets[-1]):
        raise ValueError("Finite same-inventory physical union required")
    if not isinstance(jacobian, (bool, np.bool_)) or (not jacobian and derivatives is not None):
        raise ValueError("Explicit consistent Jacobian mode required")
    d = np.empty(world.shape+(0,)) if derivatives is None else np.asarray(derivatives[key], float)
    if d.ndim != 3 or d.shape[:2] != world.shape or not np.isfinite(d).all():
        raise ValueError("Finite original-shape world Jacobian required")
    checked = [kernel._checked_row(row) for row in rows]
    if any(point for point, _ in checked) or any(row.get("_continuous_world_line")
            or "native_line_queries" in row for row in rows):
        raise ValueError("Print union requires ordinary curve correspondence")
    observed = np.concatenate([uv for _, uv in checked])
    api = kernel._interfaces()
    projected = kernel._project_target(camera, world, d, True, api, jacobian)
    real = np.flatnonzero(topology.edges())
    _, _, midpoint_valid = api[0](camera, .5*(world[real]+world[real+1]), with_jacobian=False)
    direction = np.diff(projected.uv, axis=0)
    length2 = np.sum(direction*direction, axis=1)
    valid = np.zeros(len(world)-1, bool)
    valid[real] = (projected.valid[real] & projected.valid[real+1] & midpoint_valid
        & (projected.depth[real] > 0.) & (projected.depth[real+1] > 0.)
        & np.isfinite(projected.uv[real]).all(axis=1)
        & np.isfinite(projected.uv[real+1]).all(axis=1) & (length2[real] > 1e-4))
    if not valid.any():
        raise api[3]("ice_curve_no_central_segment", np.arange(len(observed)))
    index, t, clipped, closest = api[2](observed, projected.uv[:-1], direction, length2, valid)
    r = (closest-observed).reshape(-1)
    if not jacobian:
        return r, np.empty((len(r), 0)), np.empty((len(r), 0))
    joint = np.concatenate((projected.camera_jac, projected.geometry_jac), axis=2)
    j = kernel._nearest_jacobian(observed, projected.uv, joint, index, t, clipped)
    return r, j[:, :, :15].reshape(-1, 15), j[:, :, 15:].reshape(len(r), d.shape[2])


def evaluate_projected_targets(camera25, rows, world_by_target, world_jac_by_target=None, *, jacobian=True):
    """Consumed by unchanged row_timed_raw/evaluate_row_timed_frame (RS + W)."""
    if not any(row.get("target", {}).get("kind") == KIND for row in rows):
        return _ORIGINAL["evaluate_projected_targets"](camera25, rows, world_by_target,
            world_jac_by_target, jacobian=jacobian)
    # Row-time source calls already contain one original observed point. This
    # mixed fallback retains the original source row/order and all old semantics.
    result = []
    for row in rows:
        fn = _evaluate_disjoint if row.get("target", {}).get("kind") == KIND else _ORIGINAL["evaluate_projected_targets"]
        result.append(fn(camera25, [row], world_by_target, world_jac_by_target, jacobian=jacobian))
    return tuple(np.concatenate([r[i] for r in result], axis=0) for i in range(3))


def evaluate_rows(camera25, rows, jacobian=True, *, scene=None, templates=None):
    """Keep the public framewise evaluator from reconnecting a new union."""
    if not any(r.get("target", {}).get("kind") == KIND for r in rows):
        return _ORIGINAL["evaluate_rows"](camera25, rows, jacobian, scene=scene, templates=templates)
    from calibration.program_tracking.framewise_model import build_row
    from calibration.program_tracking.first_shot_fit import _target_key
    output = []
    for row in rows:
        if row.get("target", {}).get("kind") != KIND:
            output.append(_ORIGINAL["evaluate_rows"](camera25, [row], jacobian, scene=scene, templates=templates))
            continue
        if scene is not None:
            current = build_row(scene, row["target"], row["uv"], row["sigma"], templates,
                row_id=row.get("id"), point_correspondence=row["point_correspondence"],
                diagnostic_unit_weights=row.get("diagnostic_unit_weights", False))
        else:
            current = dict(row)
        if current.get("point_correspondence") is not False:
            raise ValueError("New print union requires explicit curve correspondence")
        sigma = np.broadcast_to(np.asarray(current["sigma"], float), (len(current["uv"]),))
        if not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("Original positive source sigma required")
        if current.get("diagnostic_unit_weights") and not np.array_equal(sigma, np.ones(len(sigma))):
            raise ValueError("Diagnostic unit weights require original sigma=1")
        current["_target_key"] = _target_key(current["target"])
        support_row = dict(current)
        support_row.pop("normal_interval", None)
        raw, jcam, _ = evaluate_projected_targets(camera25, [support_row],
            {current["_target_key"]:current["world"]}, jacobian=jacobian)
        if "normal_interval" in row:
            from source_edge_intervals import SourceNormalIntervals
            support = SourceNormalIntervals(row["normal"], row["normal_interval"], np.ones(len(sigma), bool))
            if jacobian:
                raw, matrices = support.apply(raw, [jcam])
                jcam = matrices[0]
            else:
                raw = support.apply(raw)
        scale = np.repeat(sigma, 2)
        output.append((raw/scale, jcam/scale[:, None]) if jacobian else raw/scale)
    return ((np.concatenate([r[0] for r in output]), np.vstack([r[1] for r in output]))
        if jacobian else np.concatenate(output))


def print_layout_world_jacobians(self, shared_geometry, descriptors):
    """Reuse the existing material/H8 derivative, with independent path closures."""
    # New union keys are aliases rather than bank keys. Call the original only
    # on old descriptors; otherwise its direct key lookup would invent a path.
    old_indices = [i for i, d in enumerate(descriptors) if d.get("kind") != KIND]
    q = self._geometry(shared_geometry)
    paths = self.world_paths(q, descriptors)
    result = [np.zeros(path.shape+(len(q)-self.geometry_count,)) for path in paths]
    if old_indices:
        old = _ORIGINAL["print_layout_world_jacobians"](self, q, [descriptors[i] for i in old_indices])
        for i, value in zip(old_indices, old):
            result[i] = value
    from calibration.program_tracking.first_shot_print_layout import material_world_layout_jacobian
    for i, target in enumerate(descriptors):
        if target.get("kind") != KIND:
            continue
        number = target["advertisement_id"]
        if number not in self.print_layout_world_slices:
            raise ValueError("Disjoint print requires an existing shared H8 coordinate slice")
        topology = PathUnion.from_target(target)
        material = _material(self.templates, number)
        values = [material_world_layout_jacobian(self.provider.current, number,
            _record_uv(material["contours"][key]), q[self.print_layout_world_slices[number]],
            q[:100], fd_step=float(self.provider.fd_step), closed=closed)
            for key, closed in zip(topology.ids, topology.closed)]
        value = np.concatenate(values)
        topology.split(value)
        if value.shape != paths[i].shape+(8,):
            raise ValueError("Disjoint shared layout Jacobian changed world inventory")
        result[i][..., self.print_layout_tail_slices[number]] = value
    return tuple(result)


def _target_label(target):
    if target.get("kind") == KIND:
        return "Ad"+str(target["advertisement_id"])+".print."+target["contour_key"]
    return _ORIGINAL["target_label"](target)


def verify_physical_target(feature, segment, vi, descriptor):
    # The representation successor retains the same physical AdN.print.oldkey.
    # Run every original source-label/letter/point binding check on that label.
    if descriptor.get("kind") == KIND:
        PathUnion.from_target(descriptor)
        descriptor = dict(descriptor, kind="ad_print")
    return _ORIGINAL["verify_physical_target"](feature, segment, vi, descriptor)


def install_execution(configuration_reference):
    """Explicit process-local install AFTER original constructor authentication."""
    global _CONFIG
    if _CONFIG is not None:
        if _CONFIG["reference"] != configuration_reference:
            raise ValueError("Another disjoint-print configuration is installed")
        return deepcopy(_CONFIG["receipt"])
    manifest = _read(configuration_reference)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Explicit independently referenced material successor required")
    if manifest.get("runtime_producer") is not None and manifest["runtime_producer"] != reference(__file__):
        raise ValueError("Configured disjoint runtime producer changed")
    material_override = _read(manifest["material_override"])
    ids = tuple(manifest["changed_advertisement_ids"])
    if not ids or any(type(n) is not int for n in ids) or len(set(ids)) != len(ids):
        raise ValueError("Explicit distinct changed advertisements required")
    for ref in manifest.get("base_producers", []):
        if reference(ref["path"]) != ref:
            raise ValueError("Original authenticated print producer changed")
    if ids != (material_override.get("advertisement_number"),):
        raise ValueError("One explicitly numbered advertisement override required")
    templates = dict(advertisements={str(ids[0]):material_override})
    for number in ids:
        material = _material(templates, number)
        if not material.get("curve_aliases"):
            raise ValueError("Full unique shape bank and separate source aliases required")
        for rec in material["contours"].values():
            _record_uv(rec)
        for path_ids in material["curve_aliases"].values():
            if not path_ids or any(p not in material["contours"] for p in path_ids):
                raise ValueError("A source alias refers outside the complete shape bank")
    from calibration.program_tracking import framewise_model, first_shot_physical_state
    from calibration.program_tracking import first_shot_source_kernel, first_shot_row_source, first_shot_fit
    from calibration.program_tracking import first_shot_alignment_monitor, official_static_surface_dataset, validate_opening_frame_batch
    _ORIGINAL.update(feature_world=framewise_model.feature_world,
        evaluate_projected_targets=first_shot_source_kernel.evaluate_projected_targets,
        print_layout_world_jacobians=first_shot_physical_state.FirstShotPhysicalState.print_layout_world_jacobians,
        target_label=first_shot_alignment_monitor._target_label, evaluate_rows=framewise_model.evaluate_rows,
        verify_physical_target=validate_opening_frame_batch.verify_physical_target)
    framewise_model.feature_world = feature_world
    framewise_model.evaluate_rows = evaluate_rows
    # Source kernel itself stays original: its ordinary interface is the exact
    # fallback. Only declared consumer aliases opt in to the new target type.
    first_shot_row_source.evaluate_projected_targets = evaluate_projected_targets
    first_shot_fit.evaluate_projected_targets = evaluate_projected_targets
    official_static_surface_dataset.feature_world = feature_world
    official_static_surface_dataset.verify_physical_target = verify_physical_target
    validate_opening_frame_batch.verify_physical_target = verify_physical_target
    first_shot_physical_state.FirstShotPhysicalState.print_layout_world_jacobians = print_layout_world_jacobians
    first_shot_alignment_monitor._target_label = _target_label
    # Entry modules can have imported this exact public label function before
    # the opt-in installation (fit_resume is one such consumer). Rebind only
    # that known function object, never an unrelated similarly named helper.
    label_aliases = []
    for name, module in tuple(sys.modules.items()):
        if isinstance(module, ModuleType) and vars(module).get("_target_label") is _ORIGINAL["target_label"]:
            module._target_label = _target_label
            label_aliases.append(name)
    official_static_surface_dataset.KINDS[KIND] = "board"
    receipt = dict(schema=SCHEMA, producer=reference(__file__), configuration=deepcopy(configuration_reference),
        material_override=deepcopy(manifest["material_override"]), changed_advertisement_ids=list(ids),
        source_entry="original first_shot_row_source._raw_and_local / evaluate_row_timed_frame",
        row_time_camera_chain_and_complete_W_original=True,
        full_shape_bank_separate_from_source_alias_lookup=True,
        projected_edge_numerics="original nearest/J backend and original length2>1e-4 predicate, masked only at true path boundaries",
        actual_front_central_material_and_H8_guards_unchanged=True,
        loaded_exact_original_label_aliases_rebound=label_aliases,
        old_source_constructor_producers_unchanged=True, old_world_or_J_cache_compatible=False)
    _CONFIG = dict(reference=deepcopy(configuration_reference), manifest=manifest,
        templates=templates, ids=ids, receipt=receipt)
    return deepcopy(receipt)


def _physical_states(*roots):
    from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
    stack, seen = list(roots), set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        values = vars(obj) if hasattr(obj, "__dict__") else {}
        for name in ("original", "_unmodified_original", "actual_transverse_geometry", "fixed_head"):
            if name in values:
                stack.append(values[name])
        if not isinstance(obj, FirstShotPhysicalState):
            continue
        yield obj


def _template_sha(templates):
    encoded = json.dumps(templates, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def install_material_state(state):
    """Replace material after construction in every actual/unshifted state copy."""
    if _CONFIG is None:
        raise ValueError("Explicit execution installation required first")
    changed = []
    for obj in _physical_states(state):
        if getattr(obj, "disjoint_print_successor", None) == _CONFIG["reference"]:
            continue
        # Merge the one override into this state's actual current template;
        # never replace/copy another advertisement from a stale base template.
        updated = deepcopy(obj.templates)
        for number in _CONFIG["ids"]:
            if str(number) not in updated.get("advertisements", {}):
                raise ValueError("Material successor cannot create a missing advertisement")
            updated["advertisements"][str(number)] = deepcopy(_material(_CONFIG["templates"], number))
        obj._disjoint_historical_templates = obj.templates
        obj._disjoint_historical_template_sha256 = _template_sha(obj.templates)
        obj.templates = updated
        obj.disjoint_print_successor = deepcopy(_CONFIG["reference"])
        for name in ("_world_cache",):
            if name in vars(obj):
                getattr(obj, name).clear()
        changed.append(type(obj).__name__)
    return changed


@contextmanager
def historical_source_templates(programme):
    """Run ORIGINAL source-delta validation against its true historical shape.

    New print halfspaces remain compiled from the successor bank. This scoped
    swap is solely for the original source metadata/target authentication, with
    pools closed. Old ad_print queries still invoke the captured original world
    producer. Exit restores every new bank and drops all scoped world caches,
    including on an exception; it does not assert old/new shape equivalence.
    """
    if _CONFIG is None or not hasattr(programme, "disjoint_print_receipt"):
        raise ValueError("Install explicit material successor before historical authentication")
    if programme.disjoint_print_receipt.get("source_target_phase") != "pending_original_source_delta_authentication":
        raise ValueError("Historical template authentication must precede source target refresh")
    if programme.pool is not None or programme.bundle._geometry_pool is not None or programme.bundle._source_pool is not None:
        raise ValueError("Historical template authentication requires unopened pools")
    states = list(_physical_states(programme.state, programme.bundle.state))
    saved = []
    for state in states:
        old = vars(state).get("_disjoint_historical_templates")
        digest = vars(state).get("_disjoint_historical_template_sha256")
        if old is None or digest is None or _template_sha(old) != digest:
            raise ValueError("Authenticated historical material template was not retained intact")
        saved.append((state, state.templates, old, digest))
    if not saved:
        raise ValueError("No installed actual physical state for historical authentication")
    _clear_caches(programme)
    programme._disjoint_historical_template_depth = getattr(programme, "_disjoint_historical_template_depth", 0)+1
    for state, _, old, _ in saved:
        state.templates = old
        state._world_cache.clear()
    try:
        yield dict(original_material_states=len(saved), original_ad_print_world_producer_retained=True,
            historical_template_sha256=[digest for _, _, _, digest in saved],
            successor_shape_equivalence_claimed=False)
    finally:
        for state, current, _, _ in saved:
            state.templates = current
            state._world_cache.clear()
        _clear_caches(programme)
        programme._disjoint_historical_template_depth -= 1
        if any(_template_sha(old) != digest for _, _, old, digest in saved):
            raise ValueError("Original source authentication changed a historical template")


def transition_frames(frames):
    """Only model target/key/history change. Original row arrays/W stay intact."""
    from calibration.program_tracking.first_shot_fit import _target_key
    if _CONFIG is None:
        raise ValueError("Explicit execution installation required first")
    output, changed, points = [], 0, 0
    for old_frame in frames:
        frame = dict(old_frame)
        rows = []
        for old_row in old_frame["rows"]:
            target = old_row["target"]
            if target.get("kind") == "ad_print" and target.get("advertisement_id") in _CONFIG["ids"]:
                row = dict(old_row)
                row["target"] = successor_target(target, _CONFIG["templates"], _CONFIG["manifest"]["material_override"])
                row["_target_key"] = _target_key(row["target"])
                row["print_curve_target_history"] = deepcopy(old_row.get("print_curve_target_history", []))+[
                    dict(target=deepcopy(target), target_key=old_row["_target_key"], configuration=_CONFIG["reference"])]
                changed += 1
                points += len(old_row["uv"])
            else:
                row = old_row
            rows.append(row)
        frame["rows"] = rows
        output.append(frame)
    return output, dict(changed_source_rows=changed, preserved_source_points_on_changed_rows=points)


def source_worker_init(*args, configuration_reference, original_initializer):
    original_initializer(*args)
    install_execution(configuration_reference)


def geometry_worker_init(*args, configuration_reference, original_initializer):
    original_initializer(*args)
    install_execution(configuration_reference)
    from calibration.program_tracking import first_shot_fit as fit
    changed = install_material_state(fit._geometry_worker[0])
    if not changed:
        raise ValueError("Geometry worker has no actual physical state for material successor")


def source_chunk(task, *, original_sampler):
    result = original_sampler(task)
    if _CONFIG is None:
        raise ValueError("Source worker did not install declared disjoint execution")
    values, jacobian, receipt = result
    return values, jacobian, dict(receipt, disjoint_print_execution=deepcopy(_CONFIG["receipt"]))


def raw_chunk(task, *, original_sampler):
    values, receipt = original_sampler(task)
    if _CONFIG is None:
        raise ValueError("Raw worker did not install declared disjoint execution")
    return values, dict(receipt, disjoint_print_execution=deepcopy(_CONFIG["receipt"]))


def geometry_sample(task, *, original_sampler):
    result, receipt = original_sampler(task)
    if _CONFIG is None:
        raise ValueError("Geometry worker did not install declared disjoint material")
    return result, dict(receipt, disjoint_print_execution=deepcopy(_CONFIG["receipt"]))


def _clear_caches(programme):
    for obj, names in ((programme, ("transverse_world_cache", "guard_cache")),
            (programme.bundle, ("_world_cache", "_finite_coverage_cache"))):
        for name in names:
            if hasattr(obj, name):
                getattr(obj, name).clear()


def refresh_source_targets(programme):
    """Phase two AFTER all original source deltas authenticate old targets."""
    if _CONFIG is None or not hasattr(programme, "disjoint_print_receipt"):
        raise ValueError("Install explicit material/runtime before source refresh")
    if getattr(programme, "_disjoint_historical_template_depth", 0):
        raise ValueError("Restore successor templates before refreshing source targets")
    if programme.pool is not None or programme.bundle._geometry_pool is not None or programme.bundle._source_pool is not None:
        raise ValueError("Refresh successor targets before workers/evaluation")
    from calibration.program_tracking.first_shot_source_snapshot import _frame_contract
    before = _frame_contract(programme.bundle.frames)
    frames, transition = transition_frames(programme.bundle.frames)
    after = _frame_contract(frames)
    if before != after:
        raise ValueError("Disjoint successor changed native source UV/ID/clock/sigma/W contract")
    bundle = programme.bundle
    bundle.frames = frames
    bundle.descriptors = OrderedDict()
    for frame in frames:
        for row in frame["rows"]:
            bundle.descriptors.setdefault(row["_target_key"], deepcopy(row["target"]))
    bundle.keys = tuple(bundle.descriptors)
    bundle.target_descriptors = tuple(bundle.descriptors.values())
    _clear_caches(programme)
    programme.disjoint_print_receipt.update(source_target_transition=transition, source_contract=after,
        source_target_phase="refreshed_after_original_source_delta_authentication")
    return deepcopy(programme.disjoint_print_receipt)


def install_successor(programme, *, configuration_reference, entry_reference, transform_source_targets=True):
    """New-entry hook after existing fused/camera installation, before QP/J."""
    if (programme.pool is not None or programme.bundle._geometry_pool is not None
            or programme.bundle._source_pool is not None):
        raise ValueError("Install material successor before all pools open")
    if getattr(programme, "initial_J_manifest", None) is not None or getattr(programme, "_initial_complete_J_reuse", False):
        raise ValueError("Old source/J replay cannot represent a material successor")
    q_before = np.asarray(programme.start).copy()
    aux_before = {key:np.asarray(value).copy() for key,value in programme.checkpoint_arrays().items()}
    installation = install_execution(configuration_reference)
    states = install_material_state(programme.state)+install_material_state(programme.bundle.state)
    if not states:
        raise ValueError("New parent owner has no physical material state")
    _clear_caches(programme)
    import native_transverse_bundle as runtime
    from calibration.program_tracking import first_shot_transverse_execution as context
    runtime.init_worker = partial(source_worker_init, configuration_reference=deepcopy(configuration_reference),
        original_initializer=runtime.init_worker)
    runtime.source_chunk = partial(source_chunk, original_sampler=runtime.source_chunk)
    runtime.raw_chunk = partial(raw_chunk, original_sampler=runtime.raw_chunk)
    context._init_context_worker = partial(geometry_worker_init,
        configuration_reference=deepcopy(configuration_reference), original_initializer=context._init_context_worker)
    context._context_sample = partial(geometry_sample, original_sampler=context._context_sample)
    refs = [reference(__file__), deepcopy(configuration_reference), deepcopy(_CONFIG["manifest"]["material_override"]),
        deepcopy(entry_reference)]
    for ref in refs:
        if reference(ref["path"]) != ref:
            raise ValueError("Disjoint entry/configuration/producer changed")
        if ref not in programme.source_producers:
            programme.source_producers.append(deepcopy(ref))
    receipt = dict(installation, entry=deepcopy(entry_reference), changed_physical_state_copies=states,
        source_target_phase="pending_original_source_delta_authentication",
        historical_source_template_context_required_before_refresh=True,
        q_and_existing_aux_retained_byte_exact=True,
        source_target_labels_preserved=True, source_target_history_retained=True,
        all_material_world_source_J_caches_cleared=True, fresh_J_required=True,
        print_domain_rebuild="Call existing first_shot_print_halfspaces.install on NEW outer owner after this hook",
        parent_and_workers_original_constructor_authentication_precedes_successor=True)
    previous_identity, previous_arrays = programme.checkpoint_identity, programme.checkpoint_arrays
    def identity(adapter):
        return dict(previous_identity(), disjoint_print_material_successor=deepcopy(receipt))
    def arrays(adapter):
        return dict(previous_arrays(), disjoint_print_material_successor_json=np.asarray(json.dumps(receipt, sort_keys=True)))
    programme.checkpoint_identity = MethodType(identity, programme)
    programme.checkpoint_arrays = MethodType(arrays, programme)
    programme.disjoint_print_receipt = receipt
    if not np.array_equal(np.asarray(programme.start).view(np.uint8), q_before.view(np.uint8)):
        raise ValueError("Material successor changed complete parent q")
    for key, expected in aux_before.items():
        actual = programme.checkpoint_arrays()[key]
        if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("Material successor changed inherited auxiliary: "+key)
    if transform_source_targets:
        refresh_source_targets(programme)
    return deepcopy(receipt)
