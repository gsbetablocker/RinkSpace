"""Source-qualified material-side association on the original print curves.

A source segment retains its original labels and UVs, not persistent material points.
Its index order is not proof of material orientation. The current white-hole
policy uses authenticated original RGB light-side gradients; ordered_nearest
is retained as an explicit historical mathematical operator.
The discrete association is recomputed at every value/J evaluation. Source UV,
IDs, row exposure cameras, target paths and complete W are never changed.
Only independently qualified target labels opt in; no import-time installation.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

_SCHEMA = "rinkspace.source-material-side-curve-association.v2"
_ORIGINAL = {}
_ALLOWED = None
_INSTALL_RECEIPT = None
_PACKET_CACHE = OrderedDict()
_ENABLED = ContextVar("ordered_native_print_arcs_enabled", default=True)
_RECEIPT_SINK = ContextVar("ordered_native_print_arc_receipt_sink", default=None)


def source_tangents(observed, *, closed=False):
    """Forward tangent bisectors from the saved visible segment only.

    Open/occluded runs retain their own endpoints. No chord crosses a missing
    interval or joins distinct source segments. A reversal/duplicate is explicit
    degeneracy; inventing its direction would be a material association claim.
    """
    uv = np.asarray(observed, float)
    if uv.ndim != 2 or uv.shape[1:] != (2,) or len(uv) < 2 or not np.isfinite(uv).all():
        raise ValueError("An ordered finite source arc with at least two UVs is required")
    edges = np.diff(uv, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0.):
        raise ValueError("A duplicate source vertex has no directed tangent")
    unit = edges / lengths[:, None]
    tangent = np.empty_like(uv)
    tangent[0], tangent[-1] = unit[0], unit[-1]
    tangent[1:-1] = unit[:-1] + unit[1:]
    if closed:
        closing = uv[0]-uv[-1]
        norm = np.linalg.norm(closing)
        if norm == 0.:
            raise ValueError("Closed source arcs must not duplicate the first vertex")
        closing /= norm
        tangent[0], tangent[-1] = closing+unit[0], unit[-1]+closing
    size = np.linalg.norm(tangent, axis=1)
    if np.any(size == 0.):
        raise ValueError("A source reversal has no unique directed tangent")
    return tangent/size[:, None]


def _ordered_path(stations, costs, valid, *, first=None, maximum=None):
    """Exact monotone finite-choice DP; rows may use different RS cameras."""
    n, e = costs.shape
    previous = np.where(valid[0], costs[0], np.inf)
    if first is not None:
        previous[:] = np.inf
        previous[first] = costs[0, first]
    back = np.full((n, e), -1, int)
    for i in range(1, n):
        # Accumulated argmin, retaining deterministic earlier-station ties.
        prefix = np.minimum.accumulate(previous)
        improves = previous < np.r_[np.inf, prefix[:-1]]
        best = np.maximum.accumulate(np.where(improves, np.arange(e), 0))
        ix = np.searchsorted(stations[i-1], stations[i], side="right")-1
        admissible = valid[i] & (ix >= 0)
        if maximum is not None:
            admissible &= stations[i] <= maximum
        source = best[np.maximum(ix, 0)]
        current = np.full(e, np.inf)
        current[admissible] = costs[i, admissible]+previous[source[admissible]]
        back[i, admissible] = source[admissible]
        previous = current
    last = int(np.argmin(previous))
    value = float(previous[last])
    if not np.isfinite(value):
        return None, value
    choice = np.empty(n, int); choice[-1] = last
    for i in range(n-1, 0, -1):
        choice[i-1] = back[i, choice[i]]
    return choice, value


def ordered_nearest(observed, projected, segment_valid, *, world_closed, source_closed=False):
    """Return current edge IDs and feet for one qualified directed source arc.

    Every row has its own projected path (the native observed-row camera).
    The admissible feet retain curve direction and monotone material station;
    a closed path can cross its seam once, with total span at most one circuit.
    The sum of the original squared pixel distances selects the association.
    This does not assign or freeze any material coordinate across evaluations.
    """
    observed = np.asarray(observed, float)
    p = np.asarray(projected, float)
    valid = np.asarray(segment_valid, bool)
    tangent = source_tangents(observed, closed=source_closed)
    if p.ndim != 3 or p.shape[0] != len(observed) or p.shape[2] != 2 or p.shape[1] < 2:
        raise ValueError("Each observed row requires the same ordered projected path")
    if valid.shape != (len(observed), p.shape[1]-1):
        raise ValueError("Finite segment validity must retain each native row camera")
    if source_closed and not world_closed:
        raise ValueError("A closed source arc requires a closed material path")
    direction = np.diff(p, axis=1)
    length2 = np.sum(direction*direction, axis=2)
    valid = valid & np.isfinite(p[:, :-1]).all(axis=2) & np.isfinite(p[:, 1:]).all(axis=2)
    valid &= length2 > 1e-4
    valid &= np.einsum("nej,nj->ne", direction, tangent) > 0.
    if np.any(~valid.any(axis=1)):
        raise ValueError("No finite forward-oriented material edge for source arc")
    safe_length = np.where(length2 > 0., length2, 1.)
    t = np.clip(np.sum((observed[:, None]-p[:, :-1])*direction, axis=2)/safe_length, 0., 1.)
    closest = p[:, :-1]+t[:, :, None]*direction
    costs = np.sum((closest-observed[:, None])**2, axis=2)
    costs[~valid] = np.inf
    edges = costs.shape[1]
    station = np.arange(edges)[None]+t
    independent = np.argmin(costs, axis=1)
    selected_stations = station[np.arange(len(observed)), independent].copy()
    if world_closed:
        for i in range(1, len(observed)):
            if selected_stations[i] < selected_stations[i-1]:
                selected_stations[i:] += edges
        feasible = selected_stations[-1]-selected_stations[0] <= edges
    else:
        feasible = np.all(np.diff(selected_stations) >= 0.)
    if feasible:
        choice = independent
    elif not world_closed:
        choice, _ = _ordered_path(station, costs, valid)
        if choice is None:
            raise ValueError("No directed monotone association on open material path")
    else:
        # The independent lower bound makes normal operation cheap. Exact
        # anchor enumeration is needed only when its feet violate arc order.
        double_station = np.concatenate((station, station+edges), axis=1)
        double_cost = np.tile(costs, (1, 2))
        double_valid = np.tile(valid, (1, 2))
        lower_other = np.min(costs[1:], axis=1).sum()
        best_value = np.inf; choice = None
        anchors = np.argsort(costs[0], kind="stable")
        for anchor in anchors:
            if costs[0, anchor]+lower_other >= best_value:
                break
            permitted = double_valid.copy()
            permitted &= double_station >= station[0, anchor]
            trial, value = _ordered_path(double_station, double_cost, permitted,
                first=int(anchor), maximum=float(station[0, anchor]+edges))
            if value < best_value:
                best_value, choice = value, trial
        if choice is None:
            raise ValueError("No directed monotone association within one material circuit")
        choice %= edges
    return choice, closest[np.arange(len(observed)), choice], dict(
        schema=_SCHEMA, source_points=len(observed), world_edges=edges,
        independent_directed_feet_already_ordered=bool(feasible),
        source_closed=bool(source_closed), world_closed=bool(world_closed),
        original_squared_pixel_distance_sum=float(costs[np.arange(len(observed)), choice].sum()))


def _packet(reference):
    key = str(reference["path"]), str(reference["sha256"])
    if key not in _PACKET_CACHE:
        p = Path(key[0]); raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != key[1]:
            raise ValueError("Ordered arc source packet changed")
        _PACKET_CACHE[key] = json.loads(raw)
        while len(_PACKET_CACHE) > 32:
            _PACKET_CACHE.popitem(last=False)
    return _PACKET_CACHE[key]


def bind_alias_lineage(frames, binding_reference, *, allowed_targets):
    """Bind each retained alias to its actual latest point revision.

    A later whole-frame revision may retain an earlier corrected row while
    replacing frame.source_revision_reference. The binding, rather than that
    last frame label, declares the revision ancestry actually used by the loader.
    Only exact existing point corrections gain provenance; no alias is inferred.
    """
    binding = _packet(binding_reference)
    pending = [binding[k] for k in ('post_native_ordinary_revision', 'source_revision')
               if binding.get(k) is not None]
    latest = {}; seen = set()
    while pending:
        ref = pending.pop(0); key = ref['path'], ref['sha256']
        if key in seen: continue
        seen.add(key); revision = _packet(ref)
        local = set()
        for item in revision.get('identity_corrections', ()):
            address = int(item['native_pts']), item['source_point_id']
            if address in local: raise ValueError('Duplicate source identity revision address')
            local.add(address); latest.setdefault(address, (ref, item))
        for field in ('post_native_ordinary_revision', 'source_revision', 'parent_source_revision_ref', 'base_binding'):
            ancestor = revision.get(field)
            if isinstance(ancestor, dict) and {'path','sha256'} <= set(ancestor): pending.append(ancestor)
    allowed = set(map(tuple, allowed_targets)); records = []
    for frame in frames:
        for row in frame['rows']:
            target = row['target']
            if target.get('kind') != 'ad_print' or (target.get('advertisement_id'), target.get('contour_key')) not in allowed: continue
            label = 'Ad'+str(target['advertisement_id'])+'.print.'+target['contour_key']
            if row.get('source_feature_id') == label: continue
            if len(row['source_point_ids']) != 1: raise ValueError('Pointwise alias provenance requires original split rows')
            address = int(frame['native_pts']), row['source_point_ids'][0]
            found = latest.get(address)
            if found is None or found[1] != row.get('source_identity_revision'):
                raise ValueError('Current alias differs from the latest binding-declared point revision')
            if row.get('source_video_sha256', binding['source']['sha256']) != binding['source']['sha256']:
                raise ValueError('Alias revision belongs to another source video')
            row['source_identity_revision_reference'] = deepcopy(found[0])
            _alias_witness(row, frame, target, row['source_feature_id'])
            records.append(dict(native_pts=address[0], source_point_id=address[1], revision=deepcopy(found[0])))
    return dict(binding=deepcopy(binding_reference), points=len(records), records=records,
                original_UV_ID_targets_and_W_unchanged=True, whole_frame_revision_not_used_as_point_provenance=True)


def _alias_witness(row, frame, target, original_label):
    """Use an already RGB-reviewed point revision, never a name-only alias."""
    ref = row.get('source_identity_revision_reference') or (frame or {}).get('source_revision_reference')
    if frame is None or ref is None:
        raise ValueError('A renamed source arc requires its authenticated revision lineage')
    item = row.get('source_identity_revision')
    if item is None:
        raise ValueError('Source alias lacks a pointwise RGB-reviewed identity correction')
    address = int(frame['native_pts']), row['source_point_ids'][0]
    pending = [ref]; seen = set(); witnessed = False
    while pending:
        ref = pending.pop(0); key = (ref['path'], ref['sha256'])
        if key in seen: continue
        seen.add(key); revision = _packet(ref)
        matches = [v for v in revision.get('identity_corrections', ())
            if (int(v['native_pts']), v['source_point_id']) == address]
        if matches:
            if len(matches) != 1 or matches[0] != item:
                raise ValueError('Source alias contradicts its latest pointwise revision')
            witnessed = True; break
        for field in ('post_native_ordinary_revision','source_revision','parent_source_revision_ref','base_binding'):
            ancestor = revision.get(field)
            if isinstance(ancestor, dict) and 'path' in ancestor and 'sha256' in ancestor:
                pending.append(ancestor)
    if not witnessed:
        raise ValueError('Source alias is absent from its authenticated revision ancestry')
    old = item['original_target']
    old_label = 'Ad'+str(old.get('advertisement_id'))+'.print.'+str(old.get('contour_key'))
    if (item['target'] != target or old.get('kind') != 'ad_print'
            or old_label != original_label or row.get('original_model_target') != old
            or not np.array_equal(np.asarray(item['original_UV']), row['uv'][0])):
        raise ValueError('Source alias changed its physical identity or original pixel')
    from .first_shot_revision import _evidence
    # This is the original revision validator, including the reviewer, reason
    # and every source-image hash. Its result is independent of the model q.
    _evidence(item)


def _qualified_segment(rows, *, frame=None):
    """Authenticate explicit source arc without inferring material point IDs."""
    first = rows[0]
    target = first["target"]
    label = "Ad"+str(target["advertisement_id"])+".print."+target["contour_key"]
    original_label = first.get('source_feature_id')
    if any(r.get('source_feature_id') != original_label or r['target'] != target
           or len(r['uv']) != 1 for r in rows):
        raise ValueError('Qualified print arc must retain one original named source component')
    if original_label != label:
        for row in rows:
            _alias_witness(row, frame, target, original_label)
    label = original_label
    ref = first["source_packet"]
    if any(r["source_packet"] != ref or r["source_segment_index"] != first["source_segment_index"] for r in rows):
        raise ValueError("One original source packet/visible segment required")
    packet = _packet(ref)
    features = [f for f in packet["features"] if f["id"] == label]
    if len(features) != 1:
        raise ValueError("Original ordered print component is not unique")
    feature = features[0]
    if (feature.get("world_correspondence_status") != "named_same_physical_print_component"
            or feature.get("current_source_only_readout") is not True
            or feature.get("model_or_template_generated_observation") is not False):
        raise ValueError("Explicit measured same-component source arc evidence required")
    segment = feature["segments"][first["source_segment_index"]]
    if (segment.get("measurement_origin") != "current_native_material_normal_half_transition"
            or segment.get("pixels_measured_in_this_frame") is not True):
        raise ValueError("Qualified directed native material extraction required")
    indices = [int(r["original_vertex_indices"][0]) for r in rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Source arc order/identity must be retained without duplicates")
    original = np.asarray(segment["uv"], float)
    uv = np.concatenate([np.asarray(r["uv"], float) for r in rows])
    if not np.array_equal(uv, original[indices]):
        raise ValueError("Ordered source UV differs from the authenticated native segment")
    # Missing vertices do not authorize joining an occlusion gap. The original
    # extraction already splits visible runs, so use only a complete source run.
    if indices != list(range(indices[0], indices[-1]+1)):
        raise ValueError('Ordered association cannot join a missing original vertex')
    complete = indices == list(range(len(original)))
    return uv, bool(segment['closed']) and complete


def prepare_frame(frame, worlds, trajectory, *, allowed_targets):
    """Attach evaluation-local selected edges, preserving original row order."""
    from . import first_shot_source_kernel as kernel
    allowed = set(tuple(x) for x in allowed_targets)
    groups = OrderedDict()
    for i, row in enumerate(frame["rows"]):
        target = row["target"]
        identity = target.get("advertisement_id"), target.get("contour_key")
        if target.get("kind") == "ad_print" and identity in allowed and isinstance(row.get("point_correspondence"), (bool, np.bool_)) and not row["point_correspondence"]:
            key = row["_target_key"], row.get("source_feature_id"), row.get("source_segment_index")
            groups.setdefault(key, []).append((i, row))
    changed = {}; records = []
    api = kernel._interfaces()
    runs = []
    for group in groups.values():
        run = []
        for entry in group:
            previous = run[-1][1].get("original_vertex_indices") if run else None
            current = entry[1].get("original_vertex_indices")
            if run and previous is not None and current is not None and current[0] != previous[0]+1:
                runs.append(run); run = []
            run.append(entry)
        if run: runs.append(run)
    for group in runs:
        indices, rows = zip(*group)
        identity = dict(rows=list(indices), source_points=len(rows),
            target=[rows[0]["target"]["advertisement_id"], rows[0]["target"]["contour_key"]],
            source_feature_id=rows[0].get("source_feature_id"),
            source_segment_index=rows[0].get("source_segment_index"),
            source_packet=deepcopy(rows[0].get("source_packet")),
            original_vertex_indices=[r.get("original_vertex_indices") for r in rows],
            identity_revision_reference=deepcopy(frame.get("source_revision_reference")),
            alias_lineage_required=rows[0].get("source_feature_id") != "Ad"+str(rows[0]["target"]["advertisement_id"])+".print."+rows[0]["target"]["contour_key"])
        if len(rows) < 2:
            records.append(dict(identity, status="unqualified_single_source_vertex"))
            continue
        try:
            uv, source_closed = _qualified_segment(rows, frame=frame)
            sides=[r.get('_source_material_side') for r in rows]
            if any(s is None or s.get('material')!='white_hole_inside_dark_print' for s in sides):
                raise ValueError('Authenticated original RGB white-hole material-side evidence required')
            if any(r['target'].get('contour_role')!='printed_internal_hole_boundary' for r in rows):
                raise ValueError('A verified white internal hole is required by this association policy')
        except ValueError as error:
            records.append(dict(identity, status="unqualified_source_arc", reason=str(error)))
            continue
        world = np.asarray(worlds[rows[0]["_target_key"]], float)
        world_closed = np.array_equal(world[0], world[-1])
        projected = []; validity = []
        for point in uv:
            camera = trajectory.camera15_mapping(float(point[1]), jacobian=False)[0]
            p = kernel._project_target(camera, world, np.empty(world.shape+(0,)), False, api, False)
            projected.append(p.uv); validity.append(p.segment_valid)
        try:
            if not world_closed:raise ValueError('The physical white-hole interior requires its original closed boundary')
            from .first_shot_arc_polarity import material_side_nearest
            selected, _, receipt = material_side_nearest(uv,np.asarray(projected),np.asarray(validity),
                np.asarray([s['gradient'] for s in sides]),np.asarray([s['component_bound'] for s in sides]))
        except ValueError as error:
            domain = api[3]("print_arc_no_original_material_side_path", np.arange(len(uv)))
            domain.add_note("Qualified source arc has no original-material-side model path: "+str(identity))
            raise domain from error
        for i, row, edge in zip(indices, rows, selected, strict=True):
            changed[i] = dict(row, _ordered_curve_edge_index=int(edge))
        records.append(dict(identity, **receipt, status="associated", edge_indices=selected.tolist()))
    return dict(frame, rows=[changed.get(i, row) for i, row in enumerate(frame["rows"])]), records


def evaluate_projected_targets(camera25, rows, world_by_target, world_jac_by_target=None, *, jacobian=True):
    from . import first_shot_source_kernel as kernel
    if not any("_ordered_curve_edge_index" in row for row in rows):
        return _ORIGINAL["evaluate_projected_targets"](camera25, rows, world_by_target,
            world_jac_by_target, jacobian=jacobian)
    if not isinstance(jacobian, (bool, np.bool_)) or (not jacobian and world_jac_by_target is not None):
        raise ValueError("Explicit consistent ordered-arc Jacobian mode required")
    camera = np.asarray(camera25, float)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("A finite original Brown camera25 is required")
    output = []
    api = kernel._interfaces()
    for row in rows:
        if "_ordered_curve_edge_index" not in row:
            output.append(_ORIGINAL["evaluate_projected_targets"](camera25, [row], world_by_target,
                world_jac_by_target, jacobian=jacobian))
            continue
        point, observed = kernel._checked_row(row)
        if point or len(observed) != 1 or row["target"]["kind"] != "ad_print":
            raise ValueError("Ordered arc edge selection requires one ordinary print curve query")
        world = np.asarray(world_by_target[row["_target_key"]], float)
        d = (np.empty(world.shape+(0,)) if world_jac_by_target is None else
             np.asarray(world_jac_by_target[row["_target_key"]], float))
        if (world.ndim != 2 or world.shape[1:] != (3,) or len(world) < 2
                or not np.isfinite(world).all() or d.ndim != 3
                or d.shape[:2] != world.shape or not np.isfinite(d).all()):
            raise ValueError("Finite original-shape material path and geometry Jacobian required")
        p = kernel._project_target(camera, world, d, False, api, jacobian)
        edge = row["_ordered_curve_edge_index"]
        if type(edge) is not int or not 0 <= edge < len(world)-1 or not p.segment_valid[edge]:
            raise ValueError("Evaluation-local ordered edge is outside its finite optical domain")
        valid = np.zeros(len(world)-1, bool); valid[edge] = True
        direction = np.diff(p.uv, axis=0); length2 = np.sum(direction*direction, axis=1)
        index, t, clipped, closest = api[2](observed, p.uv[:-1], direction, length2, valid)
        raw = (closest-observed).reshape(-1)
        if jacobian:
            joint = np.concatenate((p.camera_jac, p.geometry_jac), axis=2)
            j = kernel._nearest_jacobian(observed, p.uv, joint, index, t, clipped)
            output.append((raw, j[:, :, :15].reshape(-1, 15), j[:, :, 15:].reshape(len(raw), d.shape[2])))
        else:
            output.append((raw, np.empty((len(raw), 0)), np.empty((len(raw), 0))))
    return tuple(np.concatenate([item[i] for item in output], axis=0) for i in range(3))


def _raw_and_local(state, q, frame, worlds, derivatives=None):
    if not _ENABLED.get():
        return _ORIGINAL["raw_and_local"](state, q, frame, worlds, derivatives)
    from .first_shot_row_camera import prepare_frame_row_camera
    trajectory = prepare_frame_row_camera(state, q, frame["index"])
    current, records = prepare_frame(frame, worlds, trajectory, allowed_targets=_ALLOWED)
    sink = _RECEIPT_SINK.get()
    if sink is not None:
        sink.append(dict(frame=int(frame["index"]), native_pts=int(frame["native_pts"]),
            q_sha256=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
            groups=records, ordered_source_policy=_SCHEMA))
    return _ORIGINAL["raw_and_local"](state, q, current, worlds, derivatives)


def install(*, allowed_targets):
    """Install only explicitly qualified labels after historical authentication.

    Parent and source workers must use exactly the same allowed inventory. The
    original source/J cache cannot be reused after this objective successor.
    """
    global _ALLOWED, _INSTALL_RECEIPT
    selected = tuple((int(n), str(key)) for n, key in allowed_targets)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Explicit unique qualified source print targets required")
    if _ALLOWED is not None:
        if _ALLOWED != selected:
            raise ValueError("Another ordered source arc policy is already installed")
        return deepcopy(_INSTALL_RECEIPT)
    from . import first_shot_row_source as source
    _ORIGINAL.update(evaluate_projected_targets=source.evaluate_projected_targets, raw_and_local=source._raw_and_local)
    _ALLOWED = selected
    source.evaluate_projected_targets = evaluate_projected_targets
    source._raw_and_local = _raw_and_local
    producer_path = Path(__file__).resolve()
    _INSTALL_RECEIPT = dict(schema=_SCHEMA, allowed_targets=[list(x) for x in selected],
        producer=dict(path=str(producer_path), sha256=hashlib.sha256(producer_path.read_bytes()).hexdigest()),
        source_UV_ID_sigma_W_and_native_PTS_unchanged=True, material_point_correspondences_added=False,
        association_method="original_RGB_white_inside_side_qualified_nearest",
        source_vertex_order_is_not_material_orientation=True,
        historical_ordered_transport_label_retained_for_original_worker_adapters=True,
        current_per_observed_row_camera_used=True, association_recomputed_each_evaluation=True,
        old_source_world_or_J_cache_reusable=False, unresolved_arcs_keep_original_semantics=True)
    return deepcopy(_INSTALL_RECEIPT)


def source_worker_init(*args, original_initializer, allowed_targets):
    """Install the same explicit association after original worker authentication."""
    original_initializer(*args)
    install(allowed_targets=allowed_targets)


@contextmanager
def original_semantics():
    """Explicit same-process old-nearest diagnostic, restored even on failure.

    A source worker task must enter this context itself; changing a parent
    context does not claim to change already running workers. Never use its
    residual/J with the ordered objective's linearization or source cache.
    """
    token = _ENABLED.set(False)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def source_chunk_original(task, *, original_sampler):
    """Worker-side dual scoring of the unchanged historical nearest operator."""
    with original_semantics():
        return original_sampler(task)


@contextmanager
def capture_associations():
    """Collect every queried frame's fixed source eligibility/current association."""
    records = []
    token = _RECEIPT_SINK.set(records)
    try:
        yield records
    finally:
        _RECEIPT_SINK.reset(token)


def source_chunk_with_associations(task, *, original_sampler):
    """Attach audit records to the worker's existing final receipt dictionary."""
    with capture_associations() as records:
        result = original_sampler(task)
    if not isinstance(result, tuple) or not isinstance(result[-1], dict):
        raise ValueError("Original source sampler receipt contract changed")
    return result[:-1]+(dict(result[-1], ordered_contour_associations=records,
        ordered_contour_policy=policy_receipt(),
        ordered_contour_evaluation_mode="ordered" if _ENABLED.get() else "original_nearest_diagnostic"),)


def policy_receipt():
    """Frozen producer/policy stamp emitted by actual source worker tasks."""
    if _INSTALL_RECEIPT is None:
        raise ValueError("Ordered contour policy is not installed in this worker")
    producer = _INSTALL_RECEIPT["producer"]
    if hashlib.sha256(Path(producer["path"]).read_bytes()).hexdigest() != producer["sha256"]:
        raise ValueError("Installed ordered contour producer changed during execution")
    return deepcopy(_INSTALL_RECEIPT)
