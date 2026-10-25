"""Organize measured source fragments into board-strip search hypotheses.

Every emitted point comes from an existing native RGB reading. Hough/PCA and
missing-gap interpolation only navigate among fragments, never supply pixels.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def board_foot_material_authority(contract_path):
    path = Path(contract_path).resolve()
    packet = json.loads(path.read_text(encoding="utf-8-sig"))
    definition = packet["definition"]
    if (packet["schema"] != "rinkspace.ice-contact-material-role-confirmation.v1"
            or packet["authority"]["kind"] not in ("direct_user_confirmation", "direct_user_drawing")
            or not packet["authority_scope"]["semantic_role_authoritative"]
            or definition["bottom_strip_region_belongs_to"] != "board"
            or definition["bottom_strip_is_ice"] is not False
            or definition["bottom_strip_material_in_reference"] != "very_narrow_pale_yellow_board_bottom_strip"):
        raise ValueError("Current original-source material contract does not authorize this reader")
    return dict(kind=packet["authority"]["kind"], reference=str(path),
        contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=packet["reference_source"]["source_sha256"], event_id=packet["event_id"],
        user_reference_source_pts=packet["reference_source"]["source_pts"],
        board_foot_material="warm_yellow_strip_belongs_to_board",
        authority=packet["authority"], definition=definition,
        user_annotation_pixel_coordinates_loaded=False,
        scope="confirmed reference event material; source pixels independently RGB-read")


def _run_points(runs):
    return (np.concatenate([np.column_stack((np.arange(x, x+n), np.full(n, y)))
                            for y, x, n in runs]).astype(np.int64)
            if runs else np.empty((0, 2), np.int64))


def _runs(points):
    points = np.unique(np.asarray(points, np.int64), axis=0)
    if not len(points):
        return []
    points = points[np.lexsort((points[:, 0], points[:, 1]))]
    starts = np.r_[0, np.flatnonzero((np.diff(points[:, 1]) != 0) | (np.diff(points[:, 0]) != 1))+1]
    stops = np.r_[starts[1:], len(points)]
    return [[int(points[start, 1]), int(points[start, 0]), int(stop-start)]
            for start, stop in zip(starts, stops)]


def _components(packet):
    rows = packet.get("components", packet.get("source_components", ()))
    result = []
    for index, row in enumerate(rows):
        kind = row.get("typed_class_candidate", row.get("kind"))
        points = (_numpy(row["source_support_uv"]) if "source_support_uv" in row else
                  _run_points(row["source_pixel_runs_y_x_length"]))
        result.append(dict(index=index, kind=kind, points=points.astype(np.int64),
                           candidate_kind=row["candidate_kind"]))
    return result


def _print_parents(instances):
    parents = []
    for instance in instances:
        if instance.get("kind") != "advertisement_panel":
            continue
        arrays = [_numpy(reading["actual_source_uv"]) for reading in instance.get("boundary_readings", ())
                  if reading.get("boundary_role", "").startswith("printed_") and len(reading["actual_source_uv"])]
        if not arrays:
            continue
        points = np.concatenate(arrays)
        if len(points) < 20:
            continue
        bounds = np.r_[points.min(0), points.max(0)]
        parents.append(dict(object_id=instance["object_id"], points=points, bounds=bounds,
            source_print_role_verified=False, physical_id=instance.get("physical_id"),
            expected_id=instance.get("expected_id")))
    return parents


def _material_parents(image, components):
    """Colored source material plus actual print-type evidence; search only."""
    import cv2
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    material = ((hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 55)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(material, connectivity=8)
    print_points = [component["points"] for component in components
                    if component["kind"] in ("ad_boundary", "ice_paint")]
    if not print_points:
        return []
    print_points = np.unique(np.concatenate(print_points), axis=0)
    parents = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < 1200 or w < 48 or h < 25:
            continue
        selected = (print_points[:, 0] >= x) & (print_points[:, 0] < x+w) & (
            print_points[:, 1] >= y) & (print_points[:, 1] < y+h)
        actual = print_points[selected]
        if len(actual) < 24:
            continue
        related = labels[actual[:, 1], actual[:, 0]] == label
        actual = actual[related]
        if len(actual) < 24:
            continue
        parents.append(dict(object_id=f"source_material_{label}", points=actual,
            bounds=np.array([x, y, x+w, y+h], float), expected_id=None, physical_id=None,
            source_print_role_verified=False, parent_origin="original_RGB_colored_material_and_typed_print_evidence",
            material_is_advertisement_verified=False))
    return parents


def _line_fragments(points, kinds, native_hw):
    import cv2
    h, w = native_hw
    mask = np.zeros((h, w), np.uint8)
    mask[points[:, 1], points[:, 0]] = 255
    # The minimum is local native support, not a required global image span.
    segments = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=20,
        minLineLength=30, maxLineGap=8)
    nodes = []
    for segment in (() if segments is None else np.asarray(segments).reshape(-1, 4)):
        first, last = segment[:2].astype(float), segment[2:].astype(float)
        if first[0] > last[0]:
            first, last = last, first
        vector = last-first
        length = float(np.linalg.norm(vector))
        if abs(vector[0]) < 20 or abs(vector[1]) > abs(vector[0])*.8:
            continue
        tangent = vector/length
        normal = np.array([-tangent[1], tangent[0]])
        along = (points-first)@tangent
        distance = np.abs((points-first)@normal)
        selected = (along >= -2) & (along <= length+2) & (distance <= 1.6)
        indices = np.flatnonzero(selected)
        if len(indices) < 20:
            continue
        actual = points[indices]
        order = np.lexsort((actual[:, 1], actual[:, 0]))
        indices, actual = indices[order], actual[order]
        # Only navigate with the fit. Actual arrays remain exactly native UV.
        center = actual.mean(0)
        _, _, directions = np.linalg.svd(actual-center, full_matrices=False)
        tangent = directions[0]
        if tangent[0] < 0:
            tangent = -tangent
        slope = tangent[1]/tangent[0]
        line = [float(slope), float(center[1]-slope*center[0])]
        x0, x1 = int(actual[:, 0].min()), int(actual[:, 0].max())
        duplicate = next((node for node in nodes if abs(node["line"][0]-line[0]) < .015
            and abs(np.polyval(node["line"], center[0])-center[1]) < 1.2
            and min(node["x1"], x1)-max(node["x0"], x0) > .65*min(node["x1"]-node["x0"], x1-x0)), None)
        if duplicate:
            if len(indices) > len(duplicate["indices"]):
                duplicate.update(indices=indices, line=line, x0=x0, x1=x1, tangent=tangent)
            continue
        nodes.append(dict(indices=indices, line=line, x0=x0, x1=x1, tangent=tangent))
    return nodes


def _fragment_chains(nodes, points, kinds, native_hw):
    h, w = native_hw
    parent = list(range(len(nodes)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    links = []
    for i, left in enumerate(nodes):
        for j in range(i+1, len(nodes)):
            right = nodes[j]
            a, b = (left, right) if left["x0"] <= right["x0"] else (right, left)
            gap = b["x0"]-a["x1"]
            if gap > min(360, w*.2):
                continue
            angle = abs(np.arctan(a["line"][0])-np.arctan(b["line"][0]))
            if angle > np.deg2rad(20):
                continue
            x = (max(a["x0"], b["x0"])+min(a["x1"], b["x1"]))/2 if gap <= 0 else (a["x1"]+b["x0"])/2
            error = abs(np.polyval(a["line"], x)-np.polyval(b["line"], x))
            allowance = 2.5 if gap <= 8 else min(18., 3.+gap*.05)
            if error > allowance:
                continue
            ai, bi = find(i), find(j)
            if ai != bi:
                parent[bi] = ai
            links.append(dict(first_fragment=i, second_fragment=j, source_x_gap_px=max(0, gap),
                tangent_angle_radians=float(angle), navigation_continuity_error_px=float(error),
                gap_pixels_are_missing_and_not_output=True))
    groups = {}
    for index in range(len(nodes)):
        groups.setdefault(find(index), []).append(index)
    chains = []
    for members in groups.values():
        indices = np.unique(np.concatenate([nodes[index]["indices"] for index in members]))
        actual = points[indices]
        x0, x1 = int(actual[:, 0].min()), int(actual[:, 0].max())
        chains.append(dict(indices=indices, members=members, x0=x0, x1=x1,
            actual=actual, kind_counts=dict(Counter(kinds[indices])),
            links=[link for link in links if link["first_fragment"] in members and link["second_fragment"] in members]))
    return sorted(chains, key=lambda chain: (chain["x1"]-chain["x0"], len(chain["actual"])), reverse=True)


def _chain_y(chain, x, nodes, *, role=None, source_parent_width=0):
    distance = lambda node: max(node["x0"]-x, x-node["x1"], 0)
    ordered = sorted((nodes[index] for index in chain["members"]), key=distance)
    local = [node for node in ordered if distance(node) <= max(160, source_parent_width*.65, distance(ordered[0]))]
    values = [np.polyval(node["line"], x) for node in local]
    return float(min(values) if role == "top" else max(values) if role == "contact" else np.median(values))


def _outer_role_points(chain, role):
    actual = chain["actual"]
    output = []
    for x in np.unique(actual[:, 0]):
        near = actual[np.abs(actual[:, 0]-x) <= 8]
        boundary = np.quantile(near[:, 1], .1 if role == "top" else .9)
        column = actual[actual[:, 0] == x]
        output.extend(column[np.abs(column[:, 1]-boundary) <= 2])
    return np.asarray(output, np.int64).reshape(-1, 2)


def source_board_fragment_topology(rgb, source_packet, *, advertisement_instances=()):
    """Return exact observed fragment curves and separate structural candidates."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected unmodified native uint8 RGB")
    binding = source_packet.get("source_binding")
    if (not isinstance(binding, dict) or not isinstance(binding.get("source_sha256"), str)
            or not isinstance(binding.get("source_pts"), int)
            or Fraction(binding["time_base"]) <= 0):
        raise ValueError("Measured source fragments require their exact original source and clock")
    if tuple(binding.get("native_hw", image.shape[:2])) != image.shape[:2]:
        raise ValueError("Source fragment binding differs from native RGB dimensions")
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if source_packet.get("RGB_sha256", digest) != digest:
        raise ValueError("Source fragment packet belongs to different original RGB pixels")
    components = _components(source_packet)
    eligible = [component for component in components if component["kind"] in ("board_top", "ice_contact")]
    if not eligible:
        return dict(status="missing_source_boundary_fragments", chains=[], pairs=[], selected_pair=None)
    points = np.concatenate([component["points"] for component in eligible])
    kinds = np.concatenate([np.full(len(component["points"]), component["kind"], object) for component in eligible])
    # Duplicate semantic admission of one actual pixel must not double support.
    _, first = np.unique(points, axis=0, return_index=True)
    first.sort()
    points, kinds = points[first], kinds[first]
    nodes = _line_fragments(points, kinds, image.shape[:2])
    chains = _fragment_chains(nodes, points, kinds, image.shape[:2])
    parents = _print_parents(advertisement_instances)
    parents += _material_parents(image, components)
    pairs = []
    for parent in parents:
        bx0, by0, bx1, by1 = parent["bounds"]
        x = (bx0+bx1)/2
        parent_points = parent["points"]
        if parent.get("parent_origin"):
            local = parent_points[np.abs(parent_points[:, 0]-x) <= max(32, min(80, (bx1-bx0)/8))]
            if len(local) >= 20:
                parent_points = local
                by0, by1 = np.quantile(local[:, 1], [.01, .99])
        top_candidates, contact_candidates = [], []
        for index, chain in enumerate(chains):
            overlap = max(0, min(chain["x1"], bx1)-max(chain["x0"], bx0))
            if overlap < min(40, (bx1-bx0)*.2):
                continue
            top_y = _chain_y(chain, x, nodes, role="top", source_parent_width=bx1-bx0)
            contact_y = _chain_y(chain, x, nodes, role="contact", source_parent_width=bx1-bx0)
            if by0-max(180, (by1-by0)*4) <= top_y <= by0-3:
                top_candidates.append((index, top_y))
            if by1+3 <= contact_y <= by1+140:
                contact_candidates.append((index, contact_y))
        for top_index, top_y in top_candidates:
            for contact_index, contact_y in contact_candidates:
                top, contact = chains[top_index], chains[contact_index]
                common_span = max(0, min(top["x1"], contact["x1"])-max(top["x0"], contact["x0"]))
                if common_span < 60 or contact_y-top_y < 24:
                    continue
                inside = (parent_points[:, 1] > top_y) & (parent_points[:, 1] < contact_y)
                print_fraction = float(inside.mean())
                if print_fraction < .8:
                    continue
                top_type = top["kind_counts"].get("board_top", 0)/len(top["actual"])
                contact_type = contact["kind_counts"].get("ice_contact", 0)/len(contact["actual"])
                proximity = np.exp(-((by0-top_y)+(contact_y-by1))/100)
                correspondence = .5 if not parent.get("parent_origin") else 0.
                score = common_span/image.shape[1]+print_fraction+proximity+.25*top_type+contact_type+correspondence
                pairs.append(dict(top_chain=top_index, contact_chain=contact_index,
                    source_parent_object_id=parent["object_id"], expected_id=parent["expected_id"],
                    parent_origin=parent.get("parent_origin", "actual_source_print_correspondence_reading"),
                    source_parent_x=float(x), navigation_top_y=top_y, navigation_contact_y=contact_y,
                    actual_print_pixels_between_fraction=print_fraction,
                    common_source_x_span=common_span, score=float(score),
                    identity_verified=False, same_board_structure_accuracy_verified=False))
    pairs.sort(key=lambda pair: pair["score"], reverse=True)
    inventory = []
    for index, chain in enumerate(chains):
        actual = chain["actual"]
        inventory.append(dict(chain_id=index, source_point_count=len(actual),
            source_pixel_runs_y_x_length=_runs(actual),
            source_x_range=[chain["x0"], chain["x1"]], semantic_hypothesis_counts=chain["kind_counts"],
            actual_source_fragments=[dict(source_pixel_runs_y_x_length=_runs(points[nodes[node]["indices"]]),
                navigation_line=nodes[node]["line"], line_pixels_are_not_output=True) for node in chain["members"]],
            navigation_links=chain["links"], primary_points_are_existing_actual_source_readings=True,
            outer_top_source_pixel_runs_y_x_length=_runs(_outer_role_points(chain, "top")),
            outer_contact_source_pixel_runs_y_x_length=_runs(_outer_role_points(chain, "contact")),
            physical_role_unresolved=True, physical_identity_verified=False))
    return dict(schema="rinkspace.source-fragment-board-topology.v1",
        status="source_print_parent_board_pair_candidates" if pairs else "no_source_print_parent_pair",
        source_binding=source_packet.get("source_binding"),
        RGB_sha256=digest,
        chains=inventory, pairs=pairs, selected_pair=pairs[0] if pairs else None,
        print_parent_evidence=[dict(object_id=parent["object_id"], expected_id=parent["expected_id"],
            actual_source_point_count=len(parent["points"]),
            navigation_extent_xyxy=parent["bounds"].tolist(), extent_is_boundary=False) for parent in parents],
        all_original_source_fragments=[dict(source_component_index=component["index"],
            typed_class_candidate=component["kind"], candidate_kind=component["candidate_kind"],
            source_pixel_runs_y_x_length=_runs(component["points"]),
            physical_role_unresolved=True) for component in eligible],
        fresh_RGB_boundary_measurement_performed_by_topology=False,
        source_short_fragments_retained=True, global_horizontal_span_threshold_used=False,
        fitted_or_gap_pixels_used_as_output=False, source_roles_accuracy_accepted=False)


def source_board_top_foot_role_pairs(rgb, source_packet, foot_source_packet, *, advertisement_instances=()):
    """Pair separate actual top/foot roles through current measured print.

    A top fragment is never a contact chain. Only current printed RGB readings
    supply parents; a huge colored wall/body component cannot supply the board.
    """
    image = np.asarray(rgb)
    binding = source_packet.get('source_binding')
    if foot_source_packet.get('source_binding') != binding:
        raise ValueError('Source board top and foot observations have different source bindings')
    rows = source_packet.get('components', source_packet.get('source_components', ()))
    components = [c for c in rows if c.get('typed_class_candidate', c.get('kind')) == 'board_top']
    nav = source_board_fragment_topology(image, dict(source_packet, components=components),
        advertisement_instances=())
    readings = foot_source_packet.get('readings', ())
    foot = _numpy(readings[0]['primary_source_uv'] if readings else np.empty((0, 2), np.float32))
    foot = foot[np.isfinite(foot).all(1)]
    parents = _print_parents(advertisement_instances)
    for instance in advertisement_instances:
        if instance.get('source_binding') is not None and instance['source_binding'] != binding:
            raise ValueError('Source board print parent has a different original source binding')

    def local_y(points, x, upper=False):
        selected = np.abs(points[:, 0, None]-x[None]) <= 8
        count = selected.sum(0)
        output = np.full(len(x), np.nan)
        available = count > 0
        if available.any():
            values = np.where(selected[:, available], points[:, 1, None], np.nan)
            output[available] = np.nanmin(values, axis=0) if upper else np.nanmedian(values, axis=0)
        return output, count

    hypotheses = []
    for parent in parents:
        printed, bounds = parent['points'], parent['bounds']
        lower = foot[(foot[:, 0] >= bounds[0]) & (foot[:, 0] <= bounds[2])]
        if len(lower) < 8:
            continue
        for chain in nav.get('chains', ()):
            upper = _run_points(chain['outer_top_source_pixel_runs_y_x_length'])
            if not len(upper):
                continue
            first = max(bounds[0], upper[:, 0].min(), lower[:, 0].min())
            last = min(bounds[2], upper[:, 0].max(), lower[:, 0].max())
            p = printed[(printed[:, 0] >= first) & (printed[:, 0] <= last)]
            if last <= first or not len(p):
                continue
            top_y, top_count = local_y(upper, p[:, 0], True)
            foot_y, foot_count = local_y(lower, p[:, 0])
            supported = (top_count > 0) & (foot_count > 0)
            between = supported & (p[:, 1] > top_y+3) & (p[:, 1] < foot_y-3)
            supported_fraction = float(between.sum()/max(1, supported.sum()))
            fraction = float(between.mean())
            distance = float(np.nanmedian(p[supported, 1]-top_y[supported])) if supported.any() else float('inf')
            accepted = between.sum() >= 20 and supported_fraction >= .8 and fraction >= .3
            hypotheses.append(dict(top_chain_id=chain['chain_id'], source_print_object_id=parent['object_id'],
                expected_id=parent.get('expected_id'), accepted_same_source_role_hypothesis=bool(accepted),
                actual_source_top_uv=upper, actual_source_foot_uv=lower, current_actual_print_uv=p,
                local_top_y_search_relation=top_y, local_foot_y_search_relation=foot_y,
                actual_print_supported_by_both_source_roles=supported,
                actual_print_between_observed_source_roles=between,
                actual_print_between_supported_fraction=supported_fraction,
                whole_print_between_with_missing_source_support_fraction=fraction,
                actual_print_between_count=int(between.sum()), common_actual_source_x_range=[float(first), float(last)],
                score=float(fraction+(last-first)/image.shape[1]+np.exp(-distance/100)),
                top_and_contact_roles_are_separate=True, role_accuracy_accepted=False,
                relation_uses_observed_neighbors_only_no_gap_pixels=True))
    accepted = sorted((h for h in hypotheses if h['accepted_same_source_role_hypothesis']),
        key=lambda h: h['score'], reverse=True)
    selected = accepted[0] if accepted else None
    return dict(source_binding=binding, RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        status='actual_source_print_between_top_and_foot_role_hypothesis' if selected else
            'missing_supported_separate_source_role_pair',
        hypotheses=hypotheses, selected_hypothesis=selected,
        actual_source_top_uv=selected['actual_source_top_uv'] if selected else np.empty((0, 2), np.int64),
        actual_original_source_foot_uv=foot, current_actual_print_parents_only=True,
        top_fragment_allowed_as_contact=False, RGB_colored_global_components_used_as_parents=False,
        source_raw_measurements_unmodified=True, human_or_world_pixels_used=False,
        physical_structure_accuracy_accepted=False, fresh_RGB_measurement_performed=False)


def observed_signed_material_transition(values, signed_gradient, valid, index, sign):
    """Group observed nonreversing material samples, keeping native zero steps.

    This changes interval organization only. A flat/reversing alternative is
    retained outside the interval, and no gradient or selected UV is invented.
    """
    values, signed, valid = _numpy(values), _numpy(signed_gradient), _numpy(valid).astype(bool)
    if len(values) != len(signed)+1 or valid.shape != signed.shape or sign not in (-1, 1):
        raise ValueError('Signed material interval requires its complete actual source samples')
    index = int(index)
    if not 0 <= index < len(signed) or not valid[index] or signed[index]*sign <= 0:
        return dict(observed=False, source_indices=np.empty(0, np.int64),
            original_signed_gradient_samples=np.empty(0, signed.dtype))
    # The native source material values must themselves be nonreversing.
    # Zero differences are measured quantized/flat samples, not filled slopes.
    support = valid & (signed*sign >= 0)
    first = last = index
    while first > 0 and support[first-1]:
        first -= 1
    while last+1 < len(signed) and support[last+1]:
        last += 1
    nonzero = np.flatnonzero(signed[first:last+1]*sign > 0)+first
    first, last = int(nonzero[0]), int(nonzero[-1])
    indices = np.arange(first, last+1, dtype=np.int64)
    return dict(observed=True, source_indices=indices,
        original_signed_gradient_samples=signed[indices].copy(),
        original_material_value_samples=values[first:last+2].copy(),
        zero_gradient_source_indices=indices[signed[indices] == 0],
        source_material_values_are_nonreversing=True,
        original_gradient_samples_interpolated=False,
        selected_source_optical_peak_index_unchanged=index,
        interval_is_observed_optical_support_not_exact_physical_boundary=True)


def read_source_board_rail_material_profiles(rgb, pair_packet, *, source_instances=(),
        search_radius_px=36, sample_spacing_px=4):
    """Independently sample source blue-rail entry and internal blue/white exit.

    The blue material is a current-source candidate, not a universal rink color
    contract. The source chain supplies navigation only; UV are RGB signed peaks.
    """
    import cv2
    image = np.asarray(rgb)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if pair_packet.get('RGB_sha256') != digest:
        raise ValueError('Board rail source navigation belongs to different original RGB')
    binding = pair_packet['source_binding']
    hint = _numpy(pair_packet['actual_source_top_uv'])
    anchors, normals = [], []
    for x in np.unique(hint[:, 0])[::max(1, int(sample_spacing_px))]:
        column = hint[hint[:, 0] == x]
        seed = column[np.argmin(column[:, 1])].astype(np.float32)
        nearby = hint[np.abs(hint[:, 0]-x) <= 16]
        if len(nearby) >= 8:
            _, _, directions = np.linalg.svd(nearby-nearby.mean(0), full_matrices=False)
            tangent = directions[0] * (1 if directions[0, 0] >= 0 else -1)
            normal = np.array([-tangent[1], tangent[0]], np.float32)
        else:
            normal = np.array([0, 1], np.float32)
        anchors.append(seed); normals.append(normal)
    anchors = np.asarray(anchors, np.float32).reshape(-1, 2)
    normals = np.asarray(normals, np.float32).reshape(-1, 2)
    offsets = np.arange(-search_radius_px, search_radius_px+.01, .5, dtype=np.float32)
    uv = anchors[:, None]+normals[:, None]*offsets[None, :, None]
    h, w = image.shape[:2]
    valid = (uv[..., 0] >= 0) & (uv[..., 0] <= w-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
    profile = (cv2.remap(image.astype(np.float32)/255, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0) if len(anchors) else np.empty((0, len(offsets), 3), np.float32))
    blue = profile[..., 2]-(profile[..., 0]+profile[..., 1])/2
    signed, drgb = np.diff(blue, axis=1), np.diff(profile, axis=1)
    signed_valid = valid[:, :-1] & valid[:, 1:]
    midpoint = (offsets[:-1]+offsets[1:])/2
    mask_rows = []
    for instance in source_instances:
        if instance.get('kind') not in ('occluder', 'broadcast_overlay'):
            continue
        if instance.get('source_binding') is not None and instance['source_binding'] != binding:
            raise ValueError('Board rail role context belongs to a different original source')
        mask = _numpy(instance['region_mask']).astype(np.float32)
        if mask.shape != (h, w):
            raise ValueError('Board rail role mask has a different native domain')
        mask_rows.append((instance['object_id'], instance['kind'], mask))
    entries, exits, evidence, missing, pool = [], [], [], [], []
    for i, (seed, normal) in enumerate(zip(anchors, normals)):
        qualified = (profile[i, :, 2]-profile[i, :, 0] > 4/255) & (
            profile[i, :, 2]-profile[i, :, 1] > 4/255) & valid[i]
        bands = [a for a in _split_native_material_runs(qualified) if len(a) >= 3]
        peaks = np.flatnonzero(signed_valid[i, 1:-1] & (np.abs(signed[i, 1:-1]) >= np.abs(signed[i, :-2])) &
            (np.abs(signed[i, 1:-1]) >= np.abs(signed[i, 2:])))+1
        pool.append(dict(source_query_index=i, actual_source_peak_uv=seed+normal*midpoint[peaks, None],
            actual_signed_blue_material_gradient=signed[i, peaks], actual_RGB_gradient_vector=drgb[i, peaks],
            source_indices=peaks, physical_role_accepted=False))
        if not bands:
            missing.append(dict(query_index=i, reason='no_current_source_blue_material_band_near_top_navigation'))
            continue
        band = min(bands, key=lambda a: float(np.min(abs(offsets[a]))))
        peak = int(band[np.argmax(blue[i, band])])
        previous = max([a[-1] for a in bands if a[-1] < band[0]], default=0)
        following = min([a[0] for a in bands if a[0] > band[-1]], default=len(offsets)-1)
        entering, leaving = np.arange(previous, peak), np.arange(peak, following)
        entering, leaving = entering[signed_valid[i, entering]], leaving[signed_valid[i, leaving]]
        if not len(entering) or not len(leaving):
            missing.append(dict(query_index=i, reason='material_band_profile_clipped_or_entry_exit_missing'))
            continue
        a, b = int(entering[np.argmax(signed[i, entering])]), int(leaving[np.argmin(signed[i, leaving])])
        if signed[i, a] <= 0 or signed[i, b] >= 0 or b <= a:
            missing.append(dict(query_index=i, reason='current_source_material_entry_exit_sign_or_order_conflict'))
            continue
        first, last = seed+normal*midpoint[a], seed+normal*midpoint[b]
        support, body = uv[i, a:b+2], []
        for object_id, kind, mask in mask_rows:
            inside = cv2.remap(mask, support[:, 0][None], support[:, 1][None], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)[0]
            if (inside >= .5).any():
                body.append(dict(object_id=object_id, kind=kind, support_fraction=float((inside >= .5).mean()),
                    region_is_model_source_candidate_not_pixel_GT=True))
        ein = entering[signed[i, entering] >= signed[i, a]*.75]
        eout = leaving[-signed[i, leaving] >= -signed[i, b]*.75]
        eruns = _split_native_material_runs(np.isin(np.arange(len(signed[i])), ein))
        xruns = _split_native_material_runs(np.isin(np.arange(len(signed[i])), eout))
        intervals = lambda groups: [[float(midpoint[g[0]]-.25), float(midpoint[g[-1]]+.25)] for g in groups]
        entry_support = observed_signed_material_transition(blue[i], signed[i], signed_valid[i], a, 1)
        exit_support = observed_signed_material_transition(blue[i], signed[i], signed_valid[i], b, -1)
        def inside_transition(groups, support):
            indices = support['source_indices']
            return bool(len(indices)) and all(len(g) and g[0] >= indices[0] and g[-1] <= indices[-1] for g in groups)
        competing_entry = not inside_transition(eruns, entry_support)
        competing_exit = not inside_transition(xruns, exit_support)
        at_search_boundary = a in (0, len(signed[i])-1) or b in (0, len(signed[i])-1)
        conflict = competing_entry or competing_exit or bool(body) or at_search_boundary
        for support in (entry_support, exit_support):
            indices = support['source_indices']
            support['actual_native_normal_interval_px'] = np.asarray(
                [midpoint[indices[0]]-.25, midpoint[indices[-1]]+.25], np.float32)
            support['actual_source_optical_interval_UV'] = seed+normal*support['actual_native_normal_interval_px'][:, None]
        entries.append(first); exits.append(last)
        evidence.append(dict(source_query_index=i, original_source_navigation_seed_uv=seed,
            actual_source_material_entry_uv=first, actual_source_material_exit_uv=last,
            entry_role_hypothesis='board_upper_rim_blue_material_entry',
            exit_role_hypothesis='board_internal_blue_to_white_material_interface',
            actual_entry_signed_gradient=float(signed[i, a]), actual_exit_signed_gradient=float(signed[i, b]),
            actual_source_material_normal_width_px=float(midpoint[b]-midpoint[a]),
            entry_transition_support_native_normal_intervals_px=intervals(eruns),
            exit_transition_support_native_normal_intervals_px=intervals(xruns),
            entry_observed_same_signed_material_transition=entry_support,
            exit_observed_same_signed_material_transition=exit_support,
            multiple_75pct_peaks_within_same_material_interval_not_multiple_borders=(
                (len(eruns) > 1 or len(xruns) > 1) and not competing_entry and not competing_exit),
            actual_distinct_entry_material_transition_competing=competing_entry,
            actual_distinct_exit_material_transition_competing=competing_exit,
            selected_optical_peak_at_profile_search_boundary=at_search_boundary,
            sampled_native_profile_partially_outside_image=not bool(valid[i].all()),
            competing_current_source_person_support=body,
            source_role_status='source_role_conflict_or_multiple_optical_transitions' if conflict else
                'current_source_dual_material_interface_hypothesis', physical_pixel_accuracy_accepted=False,
            material_qualification_mask_boundary_used_as_actual_UV=False,
            sampled_original_RGB_signed_peak_used_as_UV=True))
    entry = np.asarray(entries, np.float32).reshape(-1, 2)
    exit = np.asarray(exits, np.float32).reshape(-1, 2)
    primary = np.asarray([e['source_role_status'] == 'current_source_dual_material_interface_hypothesis'
        for e in evidence], bool)
    return dict(source_binding=binding, RGB_sha256=digest,
        actual_original_RGB_profiles=dict(anchors_native_uv=anchors, normals_unit=normals,
            offsets_native_px=offsets, sampled_native_uv=uv, valid_mask=valid, RGB_profile=profile,
            blue_channel_difference=blue, all_signed_blue_material_derivatives=signed, all_RGB_derivative_vectors=drgb),
        all_observed_signed_optical_peak_pool=pool, actual_source_blue_rail_entry_uv=entry,
        actual_source_blue_to_white_exit_uv=exit, primary_role_samples=primary,
        source_material_role_evidence=evidence, missing_or_competing_source_roles=missing,
        original_source_topology_candidates_unmodified=True, source_top_chain_is_navigation_only=True,
        fitted_navigation_UV_used_as_measurement=False, human_UV_or_world_projection_used=False,
        current_RGB_blue_material_region_is_candidate_only=True,
        current_source_person_regions_are_competing_roles_not_raw_RGB_veto=True,
        source_pixel_accuracy_accepted=False)


def _split_native_material_runs(mask):
    active = np.flatnonzero(mask)
    return [a for a in np.split(active, np.flatnonzero(np.diff(active) > 1)+1) if len(a)]


def source_board_material_search_navigation(top_source_packet, foot_source_packet, *,
        native_hw, neighbor_radius_px=8, boundary_uncertainty_margin_px=12):
    """Bound a wide board search by same-source observations, without creating UV.

    This mask is navigation only. Local missing support stays explicit; material,
    person and optical uncertainty widens search and never establishes identity.
    """
    h, w = map(int, native_hw)
    top_binding = top_source_packet.get('source_binding')
    foot_binding = foot_source_packet.get('source_binding')
    required = ('source_sha256', 'source_pts', 'time_base')
    if not isinstance(top_binding, dict) or not isinstance(foot_binding, dict):
        raise ValueError('Board material navigation requires two original-source bindings')
    if any(top_binding.get(k) is None or top_binding.get(k) != foot_binding.get(k) for k in required):
        raise ValueError('Board material navigation cannot mix source, PTS or clock')
    if (not isinstance(top_binding['source_sha256'], str) or len(top_binding['source_sha256']) != 64
            or not isinstance(top_binding['source_pts'], int) or isinstance(top_binding['source_pts'], bool)
            or h <= 0 or w <= 0):
        raise ValueError('Board material navigation requires exact native source identity')
    if Fraction(top_binding['time_base']) <= 0:
        raise ValueError('Board material navigation requires a positive original clock')
    for binding in (top_binding, foot_binding):
        if tuple(binding.get('native_hw', (h, w))) != (h, w):
            raise ValueError('Board material navigation has a different native image domain')
    top_rgb, foot_rgb = top_source_packet.get('RGB_sha256'), foot_source_packet.get('RGB_sha256')
    if top_rgb and foot_rgb and top_rgb != foot_rgb:
        raise ValueError('Board material navigation has different original RGB pixels')
    top = _numpy(top_source_packet.get('actual_source_blue_rail_entry_uv',
        top_source_packet.get('actual_source_top_uv', np.empty((0, 2), np.float32)))).reshape(-1, 2)
    readings = foot_source_packet.get('readings', ())
    foot = _numpy(readings[0]['primary_source_uv'] if readings else
        foot_source_packet.get('actual_original_source_foot_uv', np.empty((0, 2), np.float32))).reshape(-1, 2)
    mask = np.zeros((h, w), bool)
    missing, uncertain, evidence = [], [], []
    top_finite, foot_finite = np.isfinite(top).all(1), np.isfinite(foot).all(1)
    top_valid, foot_valid = top[top_finite], foot[foot_finite]
    material_evidence = top_source_packet.get('source_material_role_evidence', ())
    role_conflict_points = np.asarray([e['actual_source_material_entry_uv'] for e in material_evidence
        if e.get('source_role_status') != 'current_source_dual_material_interface_hypothesis'], float).reshape(-1, 2)
    radius = float(neighbor_radius_px)
    margin = float(boundary_uncertainty_margin_px)
    if radius < 0 or margin < 0:
        raise ValueError('Source board search margins cannot be negative')
    for x in range(w):
        upper = top_valid[np.abs(top_valid[:, 0]-x) <= radius]
        lower = foot_valid[np.abs(foot_valid[:, 0]-x) <= radius]
        if not len(upper) or not len(lower):
            missing.append(x)
            evidence.append(dict(source_column=x, status='missing_local_actual_source_boundary_support',
                actual_top_support_count=len(upper), actual_foot_support_count=len(lower)))
            continue
        # Neighborhood aggregation fills a SEARCH region, never an observation.
        a, b = float(upper[:, 1].min()), float(lower[:, 1].max())
        if b <= a:
            missing.append(x)
            evidence.append(dict(source_column=x, status='actual_source_top_foot_order_conflict',
                actual_top_support_count=len(upper), actual_foot_support_count=len(lower)))
            continue
        conflict = bool(len(role_conflict_points) and (np.abs(role_conflict_points[:, 0]-x) <= radius).any())
        if conflict:
            uncertain.append(x)
        first, last = max(0, int(np.floor(a-margin))), min(h, int(np.ceil(b+margin))+1)
        mask[first:last, x] = True
        evidence.append(dict(source_column=x, status='uncertain_source_role_wide_search' if conflict else
            'actual_source_boundary_supported_search', actual_top_support_count=len(upper),
            actual_foot_support_count=len(lower), search_rows_half_open=[first, last],
            source_role_accuracy_accepted=False, aggregated_boundaries_are_navigation_only=True))
    return dict(source_binding=dict(top_binding), RGB_sha256=top_rgb or foot_rgb,
        search_mask=mask, origin='current_original_RGB_top_and_foot_material_search_navigation',
        source_role_evidence_per_column=evidence, missing_source_support_columns=missing,
        uncertain_search_columns=uncertain, top_actual_source_uv=top.copy(), foot_actual_source_uv=foot.copy(),
        mask_is_navigation_only=True, pixels_are_RGB_measurements=False,
        RGB_hard_veto_allowed=False, navigation_region_independently_verifies_board_role=False,
        missing_gaps_filled_as_measurement=False, world_or_human_UV_used=False,
        neighbor_search_radius_native_px=radius, boundary_uncertainty_margin_native_px=margin)


def source_fragment_region_prompts(topology, *, native_hw, rgb=None):
    """Source fragment/print relations propose SAM search points, never labels."""
    pair = topology.get("selected_pair")
    if pair is None:
        return []
    h, w = native_hw
    same_top = [candidate for candidate in topology["pairs"] if candidate["top_chain"] == pair["top_chain"]]
    contact_ids = {candidate["contact_chain"] for candidate in same_top
        if topology["chains"][candidate["contact_chain"]]["semantic_hypothesis_counts"].get("ice_contact", 0) > 0}
    ice_points = []
    for chain_id in contact_ids:
        actual = _run_points(topology["chains"][chain_id]["outer_contact_source_pixel_runs_y_x_length"])
        if not len(actual):
            continue
        samples = actual[np.linspace(0, len(actual)-1, 5).astype(int)].astype(float)
        samples[:, 1] += max(40, h*.06)
        ice_points.extend(samples[(samples[:, 1] < h-3) & (samples[:, 0] > 3) & (samples[:, 0] < w-3)].tolist())
    parent_ids = {candidate["source_parent_object_id"] for candidate in same_top
                  if candidate["contact_chain"] in contact_ids}
    board_points = []
    for parent in topology["print_parent_evidence"]:
        if parent["object_id"] in parent_ids:
            x0, y0, x1, y1 = parent["navigation_extent_xyxy"]
            board_points.append([(x0+x1)/2, (y0+y1)/2])
    # Print-centre points alone make SAM segment glyphs/panels rather than their
    # containing board. Inspect native pixels in the already selected rail's
    # inner search strip too. Bright neutral material only qualifies a prompt;
    # it never ranks a boundary or becomes an output/label.
    top = _run_points(topology["chains"][pair["top_chain"]]["outer_top_source_pixel_runs_y_x_length"])
    face_points = []
    if rgb is not None:
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.shape[:2] != tuple(native_hw):
            raise ValueError("Board navigation prompt pixels must share native RGB")
        for x in np.linspace(top[:, 0].min(), top[:, 0].max(), 18):
            near = top[np.abs(top[:, 0]-x) <= 12]
            if not len(near):
                continue
            point = [int(round(x)), int(round(np.median(near[:, 1])+max(12, h*.015)))]
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                continue
            pixel = image[point[1], point[0]].astype(float)
            if pixel.mean() >= 115 and pixel.max()-pixel.min() <= 40:
                face_points.append(point)
        board_points = face_points+board_points
    if not ice_points or not board_points:
        return []
    return [dict(object_id=1000+index, kind=kind, origin="original_rgb_candidate",
        source_binding=topology["source_binding"], points_uv=positive+negative,
        point_labels=[1]*len(positive)+[0]*len(negative),
        same_board_structure_verified=False,
        evidence_origin="native_RGB_fragment_tangents_and_current_source_print_strip_parent",
        source_blank_face_prompt_count=len(face_points),
        pale_or_white_pixels_used_for_boundary_ranking=False,
        navigation_pixels_are_source_measurements=False, is_training_label=False)
        for index, (kind, positive, negative) in enumerate((("ice_region", ice_points, board_points),
                                                          ("board_region", board_points, ice_points)))]


def _ordered_foot_material_pairs(profiles, ice, board_height_px, semantic_search_profiles=None):
    """Pair entry/platform/exit in the SAME original material band on CUDA."""
    import torch
    import torch.nn.functional as F
    rgb = profiles["rgb_profile"]
    valid = profiles["valid_mask"]
    step = profiles["metadata"]["sample_step_native_px"]
    gamma = (rgb[..., 0]+rgb[..., 1])*.5-rgb[..., 2]
    gamma = torch.nan_to_num(gamma)
    luminance = torch.nan_to_num(rgb.mean(-1))
    derivative = (gamma[:, 1:]-gamma[:, :-1])/step
    light_derivative = (luminance[:, 1:]-luminance[:, :-1])/step
    n, length = derivative.shape
    previous = F.pad(derivative[:, :-1], (1, 0), value=-float("inf"))
    following = F.pad(derivative[:, 1:], (0, 1), value=-float("inf"))
    entering = (derivative >= previous) & (derivative > following) & (derivative >= .001)
    leaving = (-derivative >= F.pad(-derivative[:, :-1], (1, 0), value=-float("inf"))) & (
        -derivative > F.pad(-derivative[:, 1:], (0, 1), value=-float("inf"))) & (-derivative >= .001)
    entering &= profiles["rgb_gradient_valid_mask"]
    leaving &= profiles["rgb_gradient_valid_mask"]
    # A warm reflection can continue below the physical strip. The native
    # RGB brightening out of its contact-shadow valley is a separate optical
    # exit candidate, rather than the final decline of yellow chroma in ice.
    brightening = (light_derivative >= F.pad(light_derivative[:, :-1], (1, 0), value=-float("inf"))) & (
        light_derivative > F.pad(light_derivative[:, 1:], (0, 1), value=-float("inf"))) & (light_derivative >= .001)
    brightening &= profiles["rgb_gradient_valid_mask"]
    # Retain a native shallow RGB valley too. The strongest later rise can be
    # the ice reflection's far side; its dark-to-light turning point is the
    # material-transition hypothesis, not that later gradient maximum.
    light_before_sample = F.pad(luminance[:, :-1], (1, 0), value=float("inf"))
    light_after_sample = F.pad(luminance[:, 1:], (0, 1), value=float("inf"))
    valleys = (luminance <= light_before_sample) & (luminance < light_after_sample) & valid
    valleys = valleys[:, :-1] & profiles["rgb_gradient_valid_mask"]
    entering[:, :4] = False
    entering[:, -4:] = False
    leaving[:, :4] = False
    leaving[:, -4:] = False
    brightening[:, :4] = False
    brightening[:, -4:] = False
    valleys[:, :4] = False
    valleys[:, -4:] = False
    warm_material = (gamma >= .012) & (rgb.mean(-1) >= .28) & valid
    # Source white balance and blur can make the confirmed yellow strip
    # warmer than a fixed colour reference. Record that value, but do not
    # erase its original RGB entry/exit candidates with an upper chroma cap.
    fixed_colour_reference = warm_material & (gamma <= .18)
    coordinates = profiles["sampled_native_uv"]
    height, width = ice.shape
    grid = torch.stack((2*(coordinates[..., 0]+.5)/width-1,
                        2*(coordinates[..., 1]+.5)/height-1), -1)[None]
    ice_support = F.grid_sample(ice[None, None].float(), grid, align_corners=False)[0, 0]
    axis = torch.arange(length, device=rgb.device)
    first, last = axis[:, None], axis[None, :]
    count = (last-first).clamp_min(1)
    span_px = (last-first)*step

    def prefix(values):
        return F.pad(values.cumsum(1), (1, 0))
    def interval_mean(values):
        total = prefix(values)
        return (total[:, last+1]-total[:, first+1])/count[None]
    def before_after_mean(values, samples=6):
        total = prefix(values)
        before_count = (axis+1).clamp_max(samples)
        after_end = (axis+samples+1).clamp_max(length+1)
        after_count = (after_end-axis-1).clamp_min(1)
        before = (total[:, axis+1]-total[:, (axis+1-samples).clamp_min(0)])/before_count
        after = (total[:, after_end]-total[:, axis+1])/after_count
        return before, after
    inside = interval_mean(gamma)
    variance = (interval_mean(gamma.square())-inside.square()).clamp_min(0)
    warm_fraction = interval_mean(warm_material.float())
    fixed_colour_fraction = interval_mean(fixed_colour_reference.float())
    before, after = before_after_mean(gamma)
    light_before, light_after = before_after_mean(luminance)
    _, ice_after = before_after_mean(ice_support)
    contrast_enter = inside-before[:, :, None]
    contrast_exit = inside-after[:, None, :]
    max_width = torch.as_tensor(board_height_px, device=rgb.device).clamp_min(1)*.18+4
    max_width = max_width.clamp(12, 96)
    inside_complete = interval_mean(valid.float()) >= .999
    bright_exit = (brightening[:, None, :] &
        (after[:, None, :] <= inside*.65) &
        (light_after[:, None, :]-light_before[:, None, :] >= .006))
    valley_value = luminance[:, :-1]
    valley_depth = torch.minimum(light_before-valley_value, light_after-valley_value)
    valley_exit = valleys[:, None, :] & (valley_depth[:, None, :] >= .001) & (
        gamma[:, None, :-1] <= inside*.85) & (contrast_exit >= .006)
    exit_candidates = leaving[:, None, :] | bright_exit | valley_exit
    admissible = (entering[:, :, None] & exit_candidates &
        (span_px[None] >= 2) & (span_px[None] <= max_width[:, None, None]) &
        (warm_fraction >= .7) & (contrast_enter >= .006) & (contrast_exit >= .006) & inside_complete)
    contrast = torch.minimum(contrast_enter, contrast_exit)
    consistency = torch.exp(-variance/(contrast.square()+1e-5))
    exit_strength = torch.maximum((-derivative).clamp_min(0)[:, None, :],
        torch.where(bright_exit, light_derivative.clamp_min(0)[:, None, :], 0.))
    strength = torch.minimum(derivative[:, :, None], exit_strength)
    # Actual iceward brightening is a stronger source-layer cue than a
    # diffuse warm-colour tail, while all alternatives remain in the ledger.
    rebound = (light_after-light_before).clamp_min(0)[:, None, :]
    scores = (contrast_enter+contrast_exit)*warm_fraction*consistency+.5*strength+torch.where(bright_exit, rebound, 0.)
    # A supported transition valley preceding a diffuse colour/reflection
    # tail has explicit source-layer priority. It is still a hypothesis;
    # multiple comparable valleys remain missing rather than being joined.
    scores = scores+torch.where(valley_exit, 5*valley_depth[:, None, :], 0.)
    semantic_contact = None
    if semantic_search_profiles is not None:
        contact = semantic_search_profiles["ice_contact"]
        board_material = semantic_search_profiles["background"]
        ice_material = semantic_search_profiles["ice_paint"]
        if contact.shape != gamma.shape or board_material.shape != gamma.shape or ice_material.shape != gamma.shape:
            raise ValueError("Semantic search support must share the exact original RGB profile samples")
        semantic_contact = (contact[:, :-1]+contact[:, 1:])*.5
        semantic_board = interval_mean(board_material)
        semantic_wrong_layer = interval_mean(ice_material)
        contact_rise = ((contact[:,1:]-contact[:,:-1])/step).clamp_min(0)
        board_fall = ((board_material[:,:-1]-board_material[:,1:])/step).clamp_min(0)
        semantic_transition = torch.sqrt(contact_rise*board_fall)
        # A segmentation class may occupy a broad ice-side strip. Its peak is
        # not the physical edge. Only the board-to-contact type transition
        # supplies a nearby search cue; UV still comes from native RGB.
        semantic_transition = F.max_pool1d(semantic_transition[:,None],33,stride=1,padding=16)[:,0]
        mode=contact.argmax(1)
        optical_axis=torch.arange(length,device=rgb.device)[None]
        distance_from_source_type_mode=(optical_axis-mode[:,None]).abs()*step
        source_type_neighbourhood=torch.exp(-distance_from_source_type_mode.square()/(2*8.**2))
        # These are image-model search/type hypotheses. They report context
        # disagreements without ranking physical RGB material edge positions.
        # Keep the coarse classes as search/context diagnostics. Their band
        # centre and broad type-rise are not a physical material boundary;
        # they must not pull a measured RGB strip exit towards an ice shadow.
    scores = torch.where(admissible, scores, -torch.inf)
    best, flattened = scores.flatten(1).max(1)
    entry, exit = flattened//length, flattened%length
    has_values = torch.isfinite(best)
    alternatives = admissible & (scores >= best[:, None, None]*.85)
    competing_upper = (alternatives & ((first[None]-entry[:, None, None]).abs()*step > 3)).flatten(1).sum(1)
    competing_lower = (alternatives & ((last[None]-exit[:, None, None]).abs()*step > 3)).flatten(1).sum(1)
    competing = (alternatives & (
        ((first[None]-entry[:, None, None]).abs()*step > 3) |
        ((last[None]-exit[:, None, None]).abs()*step > 3))).flatten(1).sum(1)
    usable = has_values & (competing_lower == 0)
    usable_upper = has_values & (competing_upper == 0)
    physical_fall = _first_iceward_yellow_material_fall(profiles,gamma,entry,exit,has_values)
    optical_exit=exit
    exit=physical_fall['gradient_index']
    usable &= physical_fall['supported']
    weak_type = torch.zeros_like(usable)
    selected_type = None
    if semantic_contact is not None:
        selected_type = semantic_contact[torch.arange(n, device=rgb.device), exit]
        selected_transition=semantic_transition[torch.arange(n, device=rgb.device),exit]
        weak_type = selected_transition < .01
        # The semantic target is a contact class band. A real RGB material
        # exit can lie on its plateau, where no further class rise exists.
        # Keep this transition as search diagnostics; it is not an optical
        # visibility or physical-role veto on the measured source boundary.
    candidate_indices = torch.nonzero(admissible)
    candidate_scores = scores[admissible]
    return dict(gamma=gamma, entry_index=entry, exit_index=exit,
        selected_exit_offset_native_px=physical_fall['offset_native_px'],
        selected_optical_pair_exit_index_before_material_role=optical_exit,
        first_iceward_yellow_material_fall_evidence=physical_fall,
        absolute_R_minus_G_used_as_RGB_candidate_veto=False,
        source_R_G_colour_reference_is_diagnostic_only=True,
        source_absolute_R_minus_G_profile=(rgb[..., 0]-rgb[..., 1]).abs(),
        source_R_G_colour_reference_sample_qualified=(rgb[..., 0]-rgb[..., 1]).abs() <= .065,
        selected_original_R_G_colour_reference_fraction=interval_mean(
            (warm_material & ((rgb[..., 0]-rgb[..., 1]).abs() <= .065)).float()
        )[torch.arange(n, device=rgb.device), entry, exit],
        usable=usable, usable_upper=usable_upper, missing_same_band=~has_values, competing_same_band_pairs=competing,
        competing_upper_material_interfaces=competing_upper,competing_lower_material_interfaces=competing_lower,
        weak_source_model_contact_type=weak_type, selected_source_model_contact_type=selected_type,
        source_board_to_contact_type_transition=semantic_transition if semantic_search_profiles is not None else None,
        selected_pair_score=best, selected_band_width_px=physical_fall['offset_native_px']-profiles['rgb_gradient_offsets_native_px'][entry],
        all_ordered_pair_indices_profile_entry_exit=candidate_indices,
        all_ordered_pair_scores=candidate_scores,
        selected_inside_warm_gamma=inside[torch.arange(n, device=rgb.device), entry, exit],
        selected_inside_pale_fraction=fixed_colour_fraction[torch.arange(n, device=rgb.device), entry, exit],
        selected_inside_source_warm_fraction=warm_fraction[torch.arange(n, device=rgb.device), entry, exit],
        maximum_absolute_yellow_chroma_used_as_RGB_candidate_veto=False,
        fixed_colour_reference_fraction_is_diagnostic_only=True,
        selected_band_gamma_variance=variance[torch.arange(n, device=rgb.device), entry, exit],
        selected_exit_has_native_RGB_iceward_brightening=bright_exit.expand(-1,length,-1)[torch.arange(n, device=rgb.device),entry,exit],
        selected_exit_has_native_RGB_transition_valley=valley_exit[torch.arange(n, device=rgb.device),entry,exit],
        native_RGB_luminance=luminance, ice_profile_search_membership=ice_support,
        profile_window_boundary_is_not_a_material_entry=True,
        warm_reflection_colour_tail_is_not_automatically_board_material=True,
        semantic_search_profiles=semantic_search_profiles,
        source_model_probabilities_used_only_to_rank_actual_RGB_optical_candidates=False,
        source_model_probabilities_used_only_as_context_disagreement=semantic_contact is not None,
        source_model_type_transition_is_navigation_diagnostic_not_RGB_usability_veto=True,
        upper_and_lower_selected_jointly=True, entry_platform_exit_order_verified_for_candidate=True,
        skin_or_colored_ad_not_automatically_a_board_foot=True,
        ice_mask_is_after_band_search_cue_not_physical_boundary=True)


def _first_iceward_yellow_material_fall(profiles,gamma,entry,optical_exit,has_values):
    """First RGB yellow-to-gray fall, keeping diffuse tails as uncertainty.

    This reuses the source-material half-height definition from ice_refine's
    existing detector. Actual gray-side RGB supplies its local baseline. No
    guide coordinate or learned contact-band centre supplies the output UV.
    """
    import torch
    rgb=profiles['rgb_profile'];valid=profiles['valid_mask'];n,length=gamma.shape
    sample=torch.arange(length,device=gamma.device)[None]
    rows=torch.arange(n,device=gamma.device)
    platform=(sample>entry[:,None])&(sample<=optical_exit[:,None])&valid
    peak_values,peak_index=torch.where(platform,gamma,-torch.inf).max(1)
    # 20/255 and 12/255 are the existing source detector's strongly-yellow
    # peak and weak warm-tail definitions, rather than a class confidence.
    gray=(sample>optical_exit[:,None]+2)&valid&((rgb.amax(-1)-rgb.amin(-1))<=.12)&(gamma<=12/255)
    gray_count=gray.sum(1)
    gray_gamma=torch.where(gray,gamma,torch.nan).nanmedian(1).values
    gray_gamma=torch.nan_to_num(gray_gamma)
    # A weak yellow reflection can remain above half the rail's chroma. Using
    # only a distant grey baseline would cross that reflection much later.
    # Reuse the selected ACTUAL optical fall's immediate weak-grey plateau
    # when it is already supported by three native samples. This changes the
    # material baseline, never substitutes a guide or semantic-mask boundary.
    immediate=(sample>optical_exit[:,None])&(sample<=optical_exit[:,None]+3)&valid
    weak_gray=immediate&((rgb.amax(-1)-rgb.amin(-1))<=.12)&(gamma<=12/255)
    immediate_count=weak_gray.sum(1)
    immediate_gamma=torch.where(weak_gray,gamma,torch.nan).nanmedian(1).values
    use_immediate=has_values&(immediate_count>=3)&torch.isfinite(immediate_gamma)
    distant_gray_gamma=gray_gamma.clone()
    gray_gamma=torch.where(use_immediate,immediate_gamma,gray_gamma)
    level=(peak_values+gray_gamma)*.5
    def first_crossing(threshold):
        below=(sample>peak_index[:,None])&valid&(gamma<=threshold[:,None])
        crossing=torch.where(below,sample,torch.full_like(sample,length)).min(1).values
        last=crossing.clamp(1,length-1);first=last-1
        left=gamma[rows,first];right=gamma[rows,last]
        fraction=((left-threshold)/(left-right).clamp_min(1e-12)).clamp(0,1)
        offset=profiles['offsets_native_px'][first]+fraction*profiles['metadata']['sample_step_native_px']
        return offset,crossing<length,first,last
    offset,found,first,last=first_crossing(level)
    high,high_found,_,_=first_crossing(gray_gamma+.8*(peak_values-gray_gamma))
    low,low_found,_,_=first_crossing(gray_gamma+.2*(peak_values-gray_gamma))
    supported=has_values&found&(gray_count>=3)&(peak_values>=20/255)&((peak_values-gray_gamma)>=.006)
    # A new strong warm rise before gray context is another material branch.
    rebound=(sample>last[:,None])&(sample<=optical_exit[:,None])&valid&(gamma>(gray_gamma+.8*(peak_values-gray_gamma))[:,None])
    competing=rebound.sum(1)>=3
    supported &= ~competing
    fallback=profiles['rgb_gradient_offsets_native_px'][optical_exit]
    offset=torch.where(found&has_values,offset,fallback)
    return dict(gradient_index=torch.where(found&has_values,first,optical_exit),offset_native_px=offset,
        supported=supported,dominant_warm_peak_index=peak_index,dominant_warm_peak_gamma=peak_values,
        actual_gray_side_sample_count=gray_count,actual_gray_side_gamma=gray_gamma,
        immediate_weak_gray_plateau_used=use_immediate,distant_gray_gamma_diagnostic=distant_gray_gamma,
        actual_warm_gray_half_height=level,first_fall_found=found,
        actual_warm_gray_80_to_20_fall_support_native_px=torch.stack((high,low),1),
        actual_warm_gray_80_to_20_fall_support_complete=high_found&low_found,
        observed_optical_fall_support_is_not_a_certified_physical_contact_interval=True,
        competing_strong_warm_rebound_before_gray=competing,
        weak_or_unresolved_actual_yellow_layer=~supported,
        guide_position_used_as_output=False,coarse_class_peak_used_as_output=False,
        last_warm_reflection_or_gray_shadow_brightening_used_as_output=False)


def _native_material_query_strip(rgb, prediction, classes, topology, spacing, device):
    """Scan the full native type domain and sample source profiles across it."""
    import torch
    import torch.nn.functional as F
    image=np.asarray(rgb)
    h,w=image.shape[:2]
    probability=prediction['class_probabilities'].to(device)
    if probability.shape[-2:]!=(h,w):
        probability=F.interpolate(probability,size=(h,w),mode='bilinear',align_corners=False)
    probability=probability[0]
    contact=probability[list(classes).index('ice_contact')]
    background=probability[list(classes).index('background')]
    transition=((contact[1:]-contact[:-1]).clamp_min(0)*
                (background[:-1]-background[1:]).clamp_min(0)).sqrt()
    x=np.arange(2,w-2,spacing)
    pair=topology.get('selected_pair')
    top=np.full(len(x),np.nan)
    if pair:
        top_seed=_run_points(topology['chains'][pair['top_chain']]['outer_top_source_pixel_runs_y_x_length'])
        columns=np.unique(top_seed[:,0])
        top_y=np.array([np.min(top_seed[top_seed[:,0]==column,1]) for column in columns])
        top=np.interp(x,columns,top_y)
    # Retain ALL native type components/runs, including tiny and competing
    # surfaces. Per-column maximum alone can choose a person's costume and
    # hide the independently predicted real board foot below it.
    import cv2
    mask=(probability.argmax(0)==list(classes).index('ice_contact')).cpu().numpy().astype(np.uint8)
    component_count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    contact_cpu=contact.cpu().numpy()
    transition_cpu=transition.cpu().numpy()
    queries=[];query_labels=[];strengths=[];seen=set()
    for column in np.rint(x).astype(int):
        source_rows=np.flatnonzero(mask[:,column])
        breaks=np.r_[0,np.flatnonzero(np.diff(source_rows)!=1)+1,len(source_rows)]
        for start,stop in zip(breaks[:-1],breaks[1:]):
            if start==stop:
                continue
            run=source_rows[start:stop]
            mode=int(run[np.argmax(contact_cpu[run,column])])
            queries.append([column,mode]);label=int(labels[mode,column]);query_labels.append(label);seen.add(label)
            strengths.append(float(transition_cpu[max(0,mode-8):min(h-1,mode+9),column].max()))
        if not len(source_rows):
            mode=int(transition_cpu[:,column].argmax())
            queries.append([column,mode+.5]);query_labels.append(0);strengths.append(float(transition_cpu[mode,column]))
    for label in range(1,component_count):
        if label in seen:
            continue
        left,upper,width,height,_=stats[label]
        local=contact_cpu[upper:upper+height,left:left+width]
        values=np.where(labels[upper:upper+height,left:left+width]==label,local,-np.inf)
        dy,dx=np.unravel_index(values.argmax(),values.shape)
        column=int(left+dx);mode=int(upper+dy)
        queries.append([column,mode]);query_labels.append(label)
        strengths.append(float(transition_cpu[max(0,mode-8):min(h-1,mode+9),column].max()))
    points=np.asarray(queries,float)
    normals=np.column_stack((np.zeros(len(points)),np.ones(len(points))))
    top_values=np.interp(points[:,0],x,top) if np.isfinite(top).any() else np.full(len(points),np.nan)
    heights=np.where(np.isfinite(top_values),points[:,1]-top_values,h*.25)
    strengths=np.asarray(strengths)
    return points,normals,heights,dict(
        query_source='all_native_original_RGB_model_contact_components_and_background_contact_transitions',
        query_native_columns=points[:,0].astype(int),query_component_labels=np.asarray(query_labels),
        contact_prediction_component_count=component_count-1,all_predicted_components_queried=True,
        query_strength=strengths,profile_direction='native_vertical_RGB_query_not_assumed_physical_normal',
        weak_query_model_support=strengths<.01,
        native_probability_column_domain_scanned=[0,w],
        profile_sample_spacing_native_px=float(spacing),
        profile_unique_query_column_count=int(len(np.unique(points[:,0]))),
        all_native_columns_queried=len(np.unique(points[:,0]))==w,
        weak_queries_not_removed=True,SAM_envelope_used_to_decide_visibility=False,
        fragment_scope_used_as_visibility_veto=False,
        source_type_mode_pixels_are_search_only=True)


def _original_RGB_profiles_on_CPU(image, points, normals, radius, binding):
    """Bilinear source-pixel profiles with unfiltered signed optical evidence."""
    import torch
    import torch.nn.functional as F
    h,w=image.shape[:2]
    values=torch.as_tensor(image.copy()).permute(2,0,1)[None].float()/255
    points=torch.as_tensor(points,dtype=torch.float32)
    normals=torch.as_tensor(normals,dtype=torch.float32)
    normals=normals/normals.norm(dim=1,keepdim=True).clamp_min(1e-12)
    half=int(np.ceil(radius/.5));offsets=torch.arange(-half,half+1,dtype=torch.float32)*.5
    uv=points[:,None]+normals[:,None]*offsets[None,:,None]
    grid=torch.stack((2*(uv[...,0]+.5)/w-1,2*(uv[...,1]+.5)/h-1),-1)[None]
    profile=F.grid_sample(values,grid,align_corners=False)[0].permute(1,2,0)
    valid=(uv[...,0]>=0)&(uv[...,0]<=w-1)&(uv[...,1]>=0)&(uv[...,1]<=h-1)
    profile=profile.masked_fill(~valid[...,None],float('nan'))
    gradients=(profile[:,1:]-profile[:,:-1])/.5
    return dict(rgb_profile=profile,valid_mask=valid,sampled_native_uv=uv,points_native_uv=points,
        normals_unit=normals,offsets_native_px=offsets,rgb_gradient_offsets_native_px=(offsets[1:]+offsets[:-1])*.5,
        rgb_gradient_valid_mask=valid[:,1:]&valid[:,:-1],rgb_gradient=gradients,
        source_binding=binding,flags=dict(native_frame_boundary_clipped=~valid.all(1),
            weak_or_blurred_optical_profiles_not_removed=True),
        metadata=dict(native_hw=[h,w],sample_step_native_px=.5,RGB_sampler='CPU_torch_bilinear_original_RGB',
            unsigned_or_wide_edge_thresholds_used_as_visibility_veto=False))


def source_graphic_rectangle_candidates(rgb, binding):
    """Read flat axis-aligned text rectangles in native RGB, without PTS rules."""
    import cv2
    image=np.asarray(rgb)
    if image.dtype!=np.uint8 or image.shape[:2]!=tuple(binding['native_hw']):
        raise ValueError('Graphic source candidates require unchanged native RGB')
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    contours,_=cv2.findContours(cv2.Canny(gray,12,36),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    accepted=[];ledger=[]
    for contour in contours:
        perimeter=cv2.arcLength(contour,True)
        polygon=cv2.approxPolyDP(contour,perimeter*.02,True)
        if len(polygon)!=4 or cv2.contourArea(contour)<500: continue
        x,y,width,height=cv2.boundingRect(contour)
        reasons=[]
        if width/max(height,1)<1.5 or height<12 or width<48:
            reasons.append('not_a_horizontal_text_panel')
        vertices=polygon[:,0]
        edges=np.roll(vertices,-1,0)-vertices
        if np.any(np.min(np.abs(edges),axis=1)>max(4,height*.1)):
            reasons.append('source_edges_are_perspective_or_not_axis_aligned')
        region=image[y:y+height,x:x+width]
        native_gray=gray[y:y+height,x:x+width]
        neutral=(region.max(2).astype(float)-region.min(2)<=40)&(region.mean(2)>=140)
        if neutral.mean()<.6: reasons.append('source_background_is_not_a_flat_light_neutral_panel')
        dark=(native_gray < np.quantile(native_gray,.85)-35).astype(np.uint8)
        count,_,stats,_=cv2.connectedComponentsWithStats(dark,8)
        glyphs=[]
        for gx,gy,gw,gh,area in stats[1:]:
            if 2<=gw<=height*.9 and 5<=gh<=height*.9 and 10<=area<=width*height*.1:
                glyphs.append([int(gx),int(gy),int(gw),int(gh),int(area)])
        tall=[glyph for glyph in glyphs if glyph[3]>=np.median([item[3] for item in glyphs])*.65] if glyphs else []
        if len(tall)<2 or np.std([glyph[1]+glyph[3] for glyph in tall])>max(2,height*.08):
            reasons.append('no_single_horizontal_dark_text_baseline')
        row=dict(box_xyxy=[int(x),int(y),int(x+width),int(y+height)],source_binding=binding,
            actual_source_polygon_uv=vertices.tolist(),source_glyph_components=glyphs,
            source_light_neutral_fraction=float(neutral.mean()),
            hypothesis='source_broadcast_graphic_rectangle_candidate',rejection_reasons=reasons,
            source_pixel_matte_ground_truth=False,PTS_or_human_box_coordinates_used=False)
        ledger.append(row)
        if not reasons and not any(np.max(abs(np.array(row['box_xyxy'])-np.array(other['box_xyxy'])))<4 for other in accepted):
            accepted.append(row)
    return dict(source_binding=binding,candidates=accepted,all_source_rectangle_hypotheses=ledger,
        source_scene_physical_edges_not_deleted=True,source_broadcast_graphic_precision_accepted=False)


def _source_material_ice_side_evidence(profiles, exit_index, semantic_profiles):
    """Actual RGB surface below a candidate supplies explicit layer conflicts."""
    import torch
    rgb=profiles['rgb_profile'];length=rgb.shape[1]
    positions=torch.arange(length,device=rgb.device)[None]
    offset=(positions-exit_index[:,None])*profiles['metadata']['sample_step_native_px']
    region=(offset>=24)&(offset<=64)&profiles['valid_mask']
    amount=region.sum(1)
    def mean(value):
        return torch.where(region,torch.nan_to_num(value),0.).sum(1)/amount.clamp_min(1)
    chroma=mean(rgb.amax(-1)-rgb.amin(-1))
    gray=mean(rgb.mean(-1))
    background=mean(semantic_profiles['background']) if semantic_profiles else torch.zeros_like(gray)
    ad=mean(semantic_profiles.get('ad_boundary',torch.zeros_like(gray[:,None]).expand(-1,length))) if semantic_profiles else torch.zeros_like(gray)
    nonice=(background+ad)>.65
    coloured_surface=chroma>.1
    return dict(observed_source_samples=amount,mean_actual_RGB_channel_range=chroma,
        mean_actual_RGB_luminance=gray,source_background_or_print_search_support=background+ad,
        printed_or_body_material_conflict=(amount>=12)&coloured_surface&nonice,
        clipped_or_weak_surface_context=amount<12,
        model_probability_alone_establishes_material_role=False,
        scene_material_context_does_not_delete_optical_measurements=True)


def _ordered_board_foot_profile_batches(profiles, ice, board_height_px, semantic_profiles, batch_size=48):
    """Bound transient pair matrices without removing any optical candidates."""
    import torch
    count=len(profiles['rgb_profile'])
    rows=[]
    for start in range(0,count,batch_size):
        stop=min(count,start+batch_size)
        part={key:value[start:stop] if isinstance(value,torch.Tensor) and value.ndim>=2 and
              value.shape[0]==count else value for key,value in profiles.items()}
        semantic={key:value[start:stop] for key,value in semantic_profiles.items()} if semantic_profiles else None
        row=_ordered_foot_material_pairs(part,ice,np.asarray(board_height_px)[start:stop],semantic)
        row['all_ordered_pair_indices_profile_entry_exit'][:,0]+=start
        rows.append(row)
    def merge(values):
        first=values[0]
        if isinstance(first,torch.Tensor):
            return torch.cat(values,0)
        if isinstance(first,dict):
            return {key:merge([value[key] for value in values]) for key in first}
        return first
    return {key:merge([row[key] for row in rows]) for key in rows[0]}


def source_material_transition_support(profiles, pair_evidence):
    """Preserve the measured colour slope and competing peaks as intervals.

    An optical warm-colour fall can include a blurred strip and its ice
    reflection. Its complete sampled support is not a uniquely identified
    physical contact pixel, even when a selected peak has a high class score.
    """
    gamma=_numpy(pair_evidence['gamma'])
    valid=_numpy(profiles['valid_mask']).astype(bool)
    uv=_numpy(profiles['sampled_native_uv'])
    entry=_numpy(pair_evidence['entry_index']).astype(int)
    exit=_numpy(pair_evidence['exit_index']).astype(int)
    missing=_numpy(pair_evidence['missing_same_band']).astype(bool)
    count,length=gamma.shape
    bounds=np.full((count,2),-1,np.int64)
    observed=np.zeros(count,bool)
    clipped=np.zeros(count,bool)
    gradient_runs=[]
    competing_peaks=[]
    for i in np.flatnonzero(~missing):
        start=int(entry[i]+np.argmax(gamma[i,entry[i]:exit[i]+1]))
        after=np.flatnonzero(valid[i,exit[i]+1:])+exit[i]+1
        if not len(after):
            clipped[i]=True
            continue
        limit=int(after[-1])
        candidates=np.arange(exit[i]+1,limit)
        troughs=candidates[(gamma[i,candidates]<=gamma[i,candidates-1])&
                           (gamma[i,candidates]<gamma[i,candidates+1])]
        stop=int(troughs[0]) if len(troughs) else limit
        clipped[i]=not len(troughs)
        if stop<=start:
            continue
        bounds[i]=[start,stop]
        observed[i]=True
        descending=(np.diff(gamma[i,start:stop+1])<0)&valid[i,start:stop]&valid[i,start+1:stop+1]
        starts=np.flatnonzero(descending&~np.r_[False,descending[:-1]])
        ends=np.flatnonzero(descending&~np.r_[descending[1:],False])+1
        gradient_runs.extend((int(i),int(start+a),int(start+b)) for a,b in zip(starts,ends))
        signed_fall=-np.diff(gamma[i])
        local=np.arange(start,stop)
        left=np.r_[-np.inf,signed_fall[:-1]]
        right=np.r_[signed_fall[1:],-np.inf]
        peaks=local[(signed_fall[local]>0)&(signed_fall[local]>=left[local])&
                    (signed_fall[local]>right[local])&valid[i,local]&valid[i,local+1]]
        competing_peaks.extend((int(i),int(j)) for j in peaks)
    endpoints=np.full((count,2,2),np.nan,dtype=uv.dtype)
    rows=np.flatnonzero(observed)
    endpoints[rows,0]=uv[rows,bounds[rows,0]]
    endpoints[rows,1]=uv[rows,bounds[rows,1]]
    return dict(source_binding=profiles['source_binding'],
        observed_native_warm_colour_fall_support_endpoints_uv=endpoints,
        source_profile_sample_index_bounds=bounds,observed_support=observed,
        source_window_clipped_before_following_colour_trough=clipped,
        actual_descending_gradient_runs_profile_first_last=np.asarray(gradient_runs,np.int64).reshape(-1,3),
        actual_parallel_falling_colour_peaks_profile_gradient_index=np.asarray(competing_peaks,np.int64).reshape(-1,2),
        source_profile_samples_include_blur_and_possible_ice_reflection=True,
        endpoints_are_observed_optical_support_not_physical_contact_points=True,
        zero_or_rising_gradient_gaps_not_filled=True,all_signed_native_RGB_profiles_retained=True,
        selected_peak_UV_changed=False,model_probability_plateau_endpoints_used=False,
        pixel_precision_or_physical_contact_interval_accepted=False)


def source_nested_material_band_conflicts(lower_uv, candidate_usable, pair_evidence, normals):
    """Keep a measured internal warm-band exit distinct from the whole strip.

    Different type components can query the same native column through
    different windows. A short local dip entirely inside another actually
    observed material interval is not an independently established board
    foot. Both optical measurements remain available as competing roles.
    """
    uv=_numpy(lower_uv)
    usable=_numpy(candidate_usable).astype(bool)&np.isfinite(uv).all(1)
    normal=_numpy(normals)
    width=_numpy(pair_evidence['selected_band_width_px'])
    entry=uv-normal*width[:,None]
    conflict=np.zeros(len(uv),bool)
    ledger=[]
    indices=np.flatnonzero(usable)
    for i in indices:
        # This is common source-profile evidence, not a fitted curve or a
        # model/class posterior. Require another observed interval on the
        # same source axis, with complete entry and exit support around it.
        delta=uv[indices]-uv[i]
        tangent_distance=np.abs(delta[:,0]*normal[i,1]-delta[:,1]*normal[i,0])
        alignment=(normal[indices]*normal[i]).sum(1)
        candidates=indices[(tangent_distance<=.5)&(alignment>=.98)&(width[indices]>=width[i]+4)]
        for j in candidates:
            entry_inside=float(np.dot(entry[i]-entry[j],normal[j]))
            exit_inside=float(np.dot(uv[j]-uv[i],normal[j]))
            if entry_inside<=2 or exit_inside<=3:
                continue
            conflict[i]=True
            ledger.append(dict(internal_original_query_index=int(i),enclosing_original_query_index=int(j),
                actual_internal_entry_source_uv=entry[i],actual_internal_exit_source_uv=uv[i],
                actual_enclosing_entry_source_uv=entry[j],actual_enclosing_exit_source_uv=uv[j],
                internal_observed_band_width_px=float(width[i]),enclosing_observed_band_width_px=float(width[j]),
                source_role_hypothesis='internal_material_or_reflection_exit_inside_another_observed_band',
                missing_gap_or_fitted_pixels_used=False,source_type_probability_used_as_role_veto=False))
            break
    return dict(internal_source_material_band_role_conflict=conflict,source_observed_nested_intervals=ledger,
        all_original_RGB_candidates_retained=True,nested_interval_establishes_precise_physical_boundary=False)


def source_foot_material_role_families(lower_uv, candidate_usable, pair_evidence, *, native_hw,
                                     source_transition_support=None, source_role_conflicts=None):
    """Separate source material curve families without discarding measurements.

    A high contact posterior on a blue rail is not enough to confer the foot
    role. The dominant observed pale-yellow paired curve is ranked by its
    original RGB material and native adjacency. Other optical families remain
    explicitly unresolved, with all points and candidates preserved.
    """
    uv=_numpy(lower_uv)
    usable=_numpy(candidate_usable).astype(bool)&np.isfinite(uv).all(1)
    h,w=native_hw
    indices=np.flatnonzero(usable)
    points=np.rint(uv[usable]).astype(int)
    inside=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
    indices=indices[inside];points=points[inside]
    primary=np.zeros(len(uv),bool)
    if not len(points):
        return dict(primary_role_samples=primary,unresolved_optical_samples=usable,
                    families=[],status='no_observed_source_material_family')
    unique,first=np.unique(points,axis=0,return_index=True)
    kinds=np.full(len(unique),'ice_contact',object)
    # Sparse normal profiles need native adjacency, not a Hough vote/minimum
    # fragment count. Every actual point participates, including short/weak
    # material sections. Gap links are navigation only and emit no pixels.
    parent=list(range(len(unique)))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]];index=parent[index]
        return index
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    links=[]
    for i,point in enumerate(unique):
        for j in range(i+1,len(unique)):
            dx=unique[j,0]-point[0]
            if dx>12: break
            if dx<=0: continue
            if abs(unique[j,1]-point[1])<=max(3.,dx*.8+3):
                join(i,j)
    groups={}
    for index in range(len(unique)):
        groups.setdefault(find(index),[]).append(index)
    segments=list(groups.values())
    for a in range(len(segments)):
        for b in range(a+1,len(segments)):
            left,right=(segments[a],segments[b]) if unique[segments[a],0].min()<=unique[segments[b],0].min() else (segments[b],segments[a])
            p,q=unique[left],unique[right];gap=q[:,0].min()-p[:,0].max()
            if gap<=12 or gap>min(360,w*.2): continue
            p=p[np.argsort(p[:,0])];q=q[np.argsort(q[:,0])]
            def tangent(values):
                if len(values)<2 or np.ptp(values[:,0])<2: return None
                return np.polyfit(values[:,0],values[:,1],1)
            first=tangent(p[-8:]);last=tangent(q[:8])
            if first is None or last is None: continue
            x=(p[-1,0]+q[0,0])/2
            error=abs(np.polyval(first,x)-np.polyval(last,x))
            if abs(np.arctan(first[0])-np.arctan(last[0]))>.35 or error>min(18.,4.+gap*.08): continue
            join(left[0],right[0]);links.append(dict(first_native_source_pixel=p[-1].tolist(),
                last_native_source_pixel=q[0].tolist(),source_x_gap_px=int(gap),
                source_tangent_navigation_error_px=float(error),gap_pixels_are_missing_and_not_output=True))
    groups={}
    for index in range(len(unique)): groups.setdefault(find(index),[]).append(index)
    chains=[]
    for members in groups.values():
        actual=unique[members]
        locations=set(map(tuple,actual))
        own_links=[link for link in links if
            tuple(link['first_native_source_pixel']) in locations and
            tuple(link['last_native_source_pixel']) in locations]
        chains.append(dict(actual=actual,x0=int(actual[:,0].min()),x1=int(actual[:,0].max()),links=own_links))
    warm=_numpy(pair_evidence['selected_inside_warm_gamma'])
    types=pair_evidence.get('selected_source_model_contact_type')
    types=_numpy(types) if types is not None else np.ones(len(uv))
    families=[]
    for family_id,chain in enumerate(chains):
        lookup=set(map(tuple,chain['actual']))
        members=np.array([index for index,pixel in zip(indices,points) if tuple(pixel) in lookup],int)
        if not len(members):
            continue
        # Count each original location once even when two source class runs
        # caused the exact same native RGB sample to be queried twice.
        _,once=np.unique(np.rint(uv[members]).astype(int),axis=0,return_index=True)
        real=members[once]
        strength=np.nan_to_num(warm[real])
        score=float(strength.sum())
        families.append(dict(family_id=family_id,original_RGB_query_indices=members.tolist(),
            unique_native_source_pixel_count=len(real),source_x_range=[chain['x0'],chain['x1']],
            mean_inside_yellow_chroma=float(np.nanmean(warm[real])),
            mean_source_contact_type=float(np.nanmean(types[real])),source_material_support_score=score,
            source_pixel_runs_y_x_length=_runs(chain['actual']),navigation_links=chain['links'],
            connected_gap_pixels_used_as_measurements=False,physical_role_precision_accepted=False))
    continuation_ledger=[]
    support_primary=np.zeros(len(uv),bool)
    role_conflicts=(np.zeros(len(uv),bool) if source_role_conflicts is None else
                    _numpy(source_role_conflicts).astype(bool))
    if families and source_transition_support is not None:
        support=_numpy(source_transition_support['observed_native_warm_colour_fall_support_endpoints_uv'])
        observed_support=_numpy(source_transition_support['observed_support']).astype(bool)
        dominant=max(families,key=lambda family:family['source_material_support_score'])
        # Class-component windows and lighting changes can split one native
        # strip into separate families. Cross-family support must come from
        # observed paired material intervals at the actual neighbouring ends,
        # rather than a ratio to one globally warmest appearance or a straight
        # line through the frame.
        supported={dominant['family_id']}
        pending=True
        while pending:
            pending=False
            for family in families:
                if family['family_id'] in supported:
                    continue
                members=np.array(family['original_RGB_query_indices'],int)
                own=members[observed_support[members]&~role_conflicts[members]]
                if not len(own):
                    continue
                for anchor in families:
                    if anchor['family_id'] not in supported:
                        continue
                    anchor_members=np.array(anchor['original_RGB_query_indices'],int)
                    other=anchor_members[observed_support[anchor_members]&~role_conflicts[anchor_members]]
                    if not len(other):
                        continue
                    left,right=(other,own) if uv[other,0].max()<uv[own,0].min() else (
                        (own,other) if uv[own,0].max()<uv[other,0].min() else (None,None))
                    if left is None:
                        continue
                    all_columns=np.unique(np.r_[uv[left,0],uv[right,0]])
                    cadence=float(np.median(np.diff(all_columns))) if len(all_columns)>1 else 1.
                    left_end=left[uv[left,0]>=uv[left,0].max()-2*cadence]
                    right_end=right[uv[right,0]<=uv[right,0].min()+2*cadence]
                    gap=float(uv[right,0].min()-uv[left,0].max())
                    actual_left=support[left_end];actual_right=support[right_end]
                    left_y=[float(np.min(actual_left[...,1])),float(np.max(actual_left[...,1]))]
                    right_y=[float(np.min(actual_right[...,1])),float(np.max(actual_right[...,1]))]
                    interval_gap=max(0.,left_y[0]-right_y[1],right_y[0]-left_y[1])
                    left_width=float(np.median(np.linalg.norm(actual_left[:,1]-actual_left[:,0],axis=1)))
                    right_width=float(np.median(np.linalg.norm(actual_right[:,1]-actual_right[:,0],axis=1)))
                    reach=3*cadence+left_width+right_width
                    accepted=(gap<=reach and interval_gap==0.)
                    continuation_ledger.append(dict(first_family_id=anchor['family_id'],second_family_id=family['family_id'],
                        accepted_source_material_curve_continuation=accepted,
                        actual_left_endpoint_query_indices=left_end.tolist(),actual_right_endpoint_query_indices=right_end.tolist(),
                        actual_left_endpoint_UV=uv[left_end],actual_right_endpoint_UV=uv[right_end],
                        actual_left_colour_fall_support_intervals_uv=actual_left,
                        actual_right_colour_fall_support_intervals_uv=actual_right,
                        source_x_gap_px=gap,observed_native_profile_cadence_px=cadence,
                        source_support_neighbourhood_reach_px=reach,source_optical_interval_gap_px=interval_gap,
                        global_colour_ratio_used_as_role_veto=False,model_class_confidence_establishes_role=False,
                        gap_pixels_are_missing_and_not_output=True,physical_identity_or_pixel_precision_accepted=False))
                    if accepted:
                        supported.add(family['family_id']);pending=True
                        break
        for family in families:
            members=np.array(family['original_RGB_query_indices'],int)
            supported_role=family['family_id'] in supported
            family['same_source_material_curve_continuation_supported']=supported_role
            family['global_colour_ratio_is_appearance_diagnostic_not_physical_identity']=True
            if supported_role:
                support_primary[members]=True
    competing_curve_samples=np.zeros(len(uv),bool)
    curve_competition=[]
    if families:
        best=max(families,key=lambda family:family['source_material_support_score'])
        anchor=np.array(best['original_RGB_query_indices'],int)
        band_width=pair_evidence.get('selected_band_width_px')
        band_width=_numpy(band_width) if band_width is not None else np.zeros(len(uv))
        for family in families:
            same_material=(family['mean_inside_yellow_chroma']>=best['mean_inside_yellow_chroma']*.6)
            observed_curve=family['unique_native_source_pixel_count']>=4
            family['matches_current_source_foot_material_appearance']=same_material
            family['has_observed_native_material_curve_neighbourhood']=observed_curve
            family['dominant_source_foot_family_id']=best['family_id']
            members=np.array(family['original_RGB_query_indices'],int)
            # A disconnected warm surface inside an already supported gap
            # is not established as the same board foot by colour/point count.
            # Both gap endpoints are observed RGB material points. The link
            # supplies a competing source-only search hypothesis, never a
            # fabricated measurement or a global horizontal-line condition.
            if family is not best and same_material:
                for link in best['navigation_links']:
                    first=np.array(link['first_native_source_pixel'],float)
                    last=np.array(link['last_native_source_pixel'],float)
                    local=members[(uv[members,0]>first[0])&(uv[members,0]<last[0])]
                    if not len(local):
                        continue
                    navigation_y=first[1]+(uv[local,0]-first[0])*(last[1]-first[1])/(last[0]-first[0])
                    endpoint_support=anchor[(np.abs(uv[anchor,0]-first[0])<=24)|(
                        np.abs(uv[anchor,0]-last[0])<=24)]
                    observed_width=np.nanmedian(band_width[endpoint_support]) if len(endpoint_support) else 0.
                    tolerance=max(6.,float(link['source_tangent_navigation_error_px'])+
                        float(np.nan_to_num(observed_width)))
                    disagreement=np.abs(uv[local,1]-navigation_y)
                    disputed=local[disagreement>tolerance]
                    if len(disputed):
                        competing_curve_samples[disputed]=True
                        curve_competition.append(dict(family_id=family['family_id'],
                            competing_source_foot_family_id=best['family_id'],
                            original_RGB_query_indices=disputed.tolist(),
                            actual_candidate_source_uv=uv[disputed],
                            source_navigation_y= navigation_y[disagreement>tolerance],
                            source_curve_disagreement_px=disagreement[disagreement>tolerance],
                            source_navigation_tolerance_px=tolerance,
                            source_gap_link=link,navigation_is_not_actual_boundary_UV=True,
                            separate_board_or_foreground_role_unresolved=True))
            if (same_material and observed_curve) or support_primary[members].any():
                primary[members]=~competing_curve_samples[members]
            family['same_board_source_curve_competition_count']=int(competing_curve_samples[members].sum())
        status='dominant_observed_paired_warm_board_foot_curve_hypothesis'
    else:
        status='optical_candidates_have_no_supported_board_foot_curve_family'
    return dict(primary_role_samples=primary,unresolved_optical_samples=usable&~primary,
        families=families,status=status,all_original_RGB_candidates_retained=True,
        same_board_source_curve_competing_samples=competing_curve_samples,
        same_board_source_curve_competition=curve_competition,
        source_material_curve_continuation_evidence=continuation_ledger,
        global_colour_ratio_is_not_a_physical_identity_or_visibility_veto=True,
        min_component_or_fragment_size_used_to_remove_RGB_measurements=False,
        source_type_component_is_not_automatically_physical_foot=True,
        source_material_family_is_a_role_hypothesis_not_pixel_GT=True)


def source_local_material_direction_evidence(profiles, pair_evidence, lower_uv,
        candidate_usable, families, source_role_conflicts=None):
    """Compare observed material bands within their native source neighbourhood.

    A spatial family can contain both board and foreground pixels. A different
    RGB colour direction AND a wider observed band are competing layer evidence.
    They withhold a physical role while retaining every RGB peak and interval.
    Reference directions come from this source's actual bands, never a fixed hue.
    """
    original_uv = _numpy(lower_uv)
    rgb = _numpy(profiles['rgb_profile']).astype(np.float64, copy=False)
    valid = _numpy(profiles['valid_mask']).astype(bool)
    uv = original_uv.astype(np.float64, copy=False)
    count = len(uv)
    entry = _numpy(pair_evidence['entry_index']).astype(int)
    exit = _numpy(pair_evidence['exit_index']).astype(int)
    values = np.where(valid[..., None], np.nan_to_num(rgb), 0)
    cumulative = np.pad(values.cumsum(1), ((0, 0), (1, 0), (0, 0)))
    valid_cumulative = np.pad(valid.cumsum(1), ((0, 0), (1, 0)))
    rows = np.arange(count)
    amount = valid_cumulative[rows, exit+1]-valid_cumulative[rows, entry+1]
    means = (cumulative[rows, exit+1]-cumulative[rows, entry+1])/np.maximum(amount[:, None], 1)
    vector = np.column_stack((means[:, 0]-means[:, 1], means[:, 1]-means[:, 2]))
    magnitude = np.linalg.norm(vector, axis=1)
    angle = np.arctan2(vector[:, 0], vector[:, 1])
    quant = np.arcsin(np.minimum(1, np.sqrt(2)/255/np.maximum(magnitude, 1e-12)))
    usable = _numpy(candidate_usable).astype(bool) & np.isfinite(uv).all(1) & (amount > 0)
    existing = (np.zeros(count, bool) if source_role_conflicts is None else
                _numpy(source_role_conflicts).astype(bool))
    conflict = np.zeros(count, bool)
    observed = np.zeros(count, bool)
    direction_disagreement = np.zeros(count, bool)
    width_disagreement = np.zeros(count, bool)
    center = np.full(count, np.nan)
    tolerance = np.full(count, np.nan)
    ref_count = np.zeros(count, np.int64)
    widths = _numpy(pair_evidence['selected_band_width_px'])
    references = []
    for family in families['families']:
        members = np.asarray(family['original_RGB_query_indices'], dtype=int)
        eligible = members[usable[members] & ~existing[members] & (vector[members, 1] > 0)
                           & (magnitude[members] > np.sqrt(2)/255)]
        if len(eligible) < 8:
            continue
        _, once = np.unique(np.rint(uv[eligible]).astype(int), axis=0, return_index=True)
        unique = eligible[once]
        if len(unique) < 8:
            continue
        mode = np.median(angle[unique])
        mad = np.median(abs(angle[unique]-mode))
        core = unique[abs(angle[unique]-mode) <= 2*1.4826*mad+quant[unique]]
        columns = np.unique(uv[unique, 0])
        spacing = np.median(np.diff(columns)) if len(columns) > 1 else 1
        radius = max(spacing*24, float(np.nanmedian(widths[unique]))*4)
        exclusion = spacing*3
        for i in members:
            if not usable[i] or existing[i]:
                continue
            distance = abs(uv[core, 0]-uv[i, 0])
            own = core[(distance <= radius) & (distance > exclusion)]
            if len(own) < 8:
                continue
            ref_angle = float(np.median(angle[own]))
            ref_mad = float(np.median(abs(angle[own]-ref_angle)))
            noise = float(quant[i]+np.median(quant[own]))
            interval = max(2*1.4826*ref_mad, noise)
            width_center = float(np.median(widths[own]))
            width_mad = float(np.median(abs(widths[own]-width_center)))
            width_tolerance = max(2*1.4826*width_mad,
                                  float(profiles['metadata']['sample_step_native_px'])*2)
            observed[i] = True
            center[i], tolerance[i], ref_count[i] = ref_angle, interval, len(own)
            direction_disagreement[i] = (angle[i]-ref_angle) > interval
            width_disagreement[i] = (widths[i]-width_center) > width_tolerance
            conflict[i] = direction_disagreement[i] & width_disagreement[i]
            references.append(dict(source_query_index=int(i), family_id=family['family_id'],
                source_reference_query_indices=own.astype(np.int64),
                source_reference_actual_uv=original_uv[own],
                reference_colour_direction_median_rad=ref_angle,
                reference_colour_direction_MAD_rad=ref_mad,
                source_colour_difference_quantisation_uncertainty_rad=noise,
                source_relative_direction_tolerance_rad=interval,
                reference_material_width_median_px=width_center,
                reference_material_width_MAD_px=width_mad,
                source_relative_material_width_tolerance_px=width_tolerance,
                query_observed_material_width_px=float(widths[i]),
                direction_disagreement=bool(direction_disagreement[i]),
                material_width_disagreement=bool(width_disagreement[i]),
                query_colour_direction_rad=float(angle[i]),
                reference_native_column_radius_px=radius, role_conflict=bool(conflict[i]),
                no_global_hue_or_absolute_R_G_limit=True))
    return dict(source_binding=dict(profiles['source_binding']),
        actual_selected_material_RGB_mean=means, actual_material_R_G_G_B_vector=vector,
        actual_material_direction_rad=angle, source_reference_direction_rad=center,
        source_relative_direction_tolerance_rad=tolerance, source_reference_count=ref_count,
        source_reference_observed=observed, source_material_direction_role_conflict=conflict,
        source_material_direction_disagreement=direction_disagreement,
        source_material_width_disagreement=width_disagreement,
        source_reference_evidence=references, raw_RGB_candidate_veto=False,
        RGB_or_candidate_UV_modified=False, world_or_annotation_pixels_used=False,
        colour_difference_quantisation_bound_is_not_physical_boundary_error=True,
        role_conflict_is_competing_surface_evidence_not_verified_skin_label=True)


def source_graphic_material_support_conflicts(profiles, pair_evidence, graphics):
    """Keep optical measurements but expose a graphic crossing the selected band.

    An interface just outside a graphic can still be its blurred edge. The
    original RGB samples supporting the entire selected entry/platform/exit
    must come from the material, rather than the graphic in front of it.
    Only samples already read from the source are inspected; no extra boundary
    is drawn and no sample is removed from the optical candidate ledger.
    """
    import torch
    coordinates=profiles['sampled_native_uv']
    valid=profiles['valid_mask']
    axis=torch.arange(coordinates.shape[1],device=coordinates.device)[None]
    selected=(axis>=pair_evidence['entry_index'][:,None]) & (
        axis<=pair_evidence['exit_index'][:,None]+1) & ~pair_evidence['missing_same_band'][:,None] & valid
    count=torch.zeros(len(coordinates),device=coordinates.device,dtype=torch.int64)
    per_graphic=[]
    for candidate in graphics.get('candidates',()):
        x0,y0,x1,y1=candidate['box_xyxy']
        inside=(coordinates[...,0]>=x0)&(coordinates[...,0]<x1)&(
            coordinates[...,1]>=y0)&(coordinates[...,1]<y1)
        support=(inside&selected).sum(1)
        count+=support
        per_graphic.append(dict(box_xyxy=candidate['box_xyxy'],
            actual_sampled_material_support_inside_source_graphic=support))
    return dict(selected_material_support_overlaps_source_graphic=count>0,
        actual_sampled_material_support_inside_source_graphic=count,per_graphic=per_graphic,
        support_domain='original_RGB_samples_of_selected_material_entry_platform_exit',
        source_graphic_is_a_physical_role_conflict_not_RGB_measurement_veto=True,
        all_original_optical_candidates_retained=True,fitted_or_human_UV_used=False)


def read_board_foot_material_profiles(rgb, instances, topology, *, material_role_authority,
                                    device="cuda", search_radius_px=48., sample_spacing_px=4.,
                                    prediction=None, classes=(),source_query_points_uv=None,source_query_normals=None):
    """Read both observed material interfaces; a board foot is part of the board.

    The caller supplies the event's directly confirmed warm-yellow foot role.
    Its location is NOT supplied. SAM and existing fragments only navigate;
    every reported UV is a fresh original-RGB signed material-profile peak.
    """
    import torch
    import torch.nn.functional as F
    from .vision_geometry import measure_boundary_profiles, _same_frame
    binding = instances["source_binding"]
    _same_frame(binding, topology["source_binding"])
    if prediction is not None:
        _same_frame(binding, prediction["source_binding"])
        if prediction.get("probabilities_conditioned_on_geometry", False):
            raise ValueError("Board-material type search must use original RGB without a geometry prior")
    if (material_role_authority.get("kind") not in ("direct_user_confirmation", "direct_user_drawing")
            or not material_role_authority.get("reference")
            or material_role_authority.get("source_sha256") != binding["source_sha256"]):
        raise ValueError("The source-bound board-foot role needs its actual user confirmation")
    if material_role_authority.get("board_foot_material") != "warm_yellow_strip_belongs_to_board":
        raise ValueError("This material reader requires its explicitly confirmed event material")
    image = np.asarray(rgb)
    h, w = image.shape[:2]
    graphics=source_graphic_rectangle_candidates(image,binding)
    ice_masks = [instance["region_mask"] for instance in instances["instances"] if instance["kind"] == "ice_region"]
    ice = torch.stack([mask.to(device) for mask in ice_masks]).any(0) if ice_masks else torch.zeros((h,w),device=device,dtype=torch.bool)
    if source_query_points_uv is not None:
        points=np.asarray(source_query_points_uv,float).reshape(-1,2)
        if not len(points)or not np.isfinite(points).all():raise ValueError('Finite actual-source navigation queries required')
        normals=(np.column_stack((np.zeros(len(points)),np.ones(len(points))))if source_query_normals is None
                 else np.asarray(source_query_normals,float).reshape(-1,2))
        if normals.shape!=points.shape or not np.isfinite(normals).all()or np.any(np.linalg.norm(normals,axis=1)==0):raise ValueError('Actual-source query normals differ')
        board_height=np.full(len(points),h*.25);valid=np.ones(len(points),bool);conflicts=np.zeros(len(points),bool)
        query_evidence=dict(query_source='explicit_current_RGB_material_candidates_navigation_only',
            all_optical_candidates_retained=True,navigation_pixels_are_not_output_measurements=True,
            profile_unique_query_column_count=int(len(np.unique(points[:,0]))),profile_sample_spacing_native_px=float(sample_spacing_px))
    elif prediction is not None:
        points,normals,board_height,query_evidence=_native_material_query_strip(
            image,prediction,classes,topology,sample_spacing_px,device)
        valid=np.ones(len(points),dtype=bool)
        conflicts=np.zeros(len(points),dtype=bool)
    else:
        pair=topology.get('selected_pair')
        if not pair:
            return dict(status='missing_source_type_or_fragment_navigation',readings=[],
                        material_role_authority=material_role_authority)
        seed=_run_points(topology['chains'][pair['contact_chain']]['outer_contact_source_pixel_runs_y_x_length'])
        columns=np.unique(seed[:,0]);source_y=np.array([np.max(seed[seed[:,0]==column,1]) for column in columns])
        x=np.arange(2,w-2,sample_spacing_px);y=np.interp(x,columns,source_y)
        points=np.column_stack((x,y));normals=np.column_stack((-np.gradient(y,x),np.ones(len(x))))
        board_height=np.full(len(x),h*.25)
        valid=np.ones(len(x),dtype=bool);conflicts=np.zeros(len(x),dtype=bool)
        query_evidence=dict(query_source='native_source_fragment_family_navigation_only',
            profile_sample_spacing_native_px=float(sample_spacing_px),
            profile_unique_query_column_count=int(len(np.unique(points[:,0]))),
            all_native_columns_queried=len(np.unique(points[:,0]))==w,
            fragment_scope_used_as_visibility_veto=False,
            SAM_envelope_used_to_decide_visibility=False)
    # The mask may start below the entire physical foot and include only its
    # reflection. Native board-fragment extent supplies a wider navigation
    # window; it supplies no measured material boundary or annotation pixel.
    effective_radius=max(search_radius_px,float(np.clip(np.quantile(board_height,.9)*.16+24,48,128)))
    values = torch.as_tensor(image.copy(), device=device).permute(2, 0, 1)[None].float()/255
    neutral = values.new_full((1, 1, h, w), .5)
    pts = torch.as_tensor(points, device=device, dtype=torch.float32)
    with torch.inference_mode():
        profiles = _original_RGB_profiles_on_CPU(image,points,normals,effective_radius,binding) if str(device)=='cpu' else measure_boundary_profiles(values, neutral, pts,
            torch.as_tensor(normals, device=device, dtype=torch.float32), class_index=0,
            search_radius=effective_radius, sample_step=.5,
            rgb_source_binding=binding, probability_source_binding=binding,
            projection_source_binding=binding, minimum_contrast=.008, maximum_edge_width=32.)
        profile = profiles["rgb_profile"]
        semantic_profiles = None
        if prediction is not None:
            probabilities=prediction["class_probabilities"].to(device)
            coordinates=profiles["sampled_native_uv"]
            grid=torch.stack((2*(coordinates[...,0]+.5)/w-1,2*(coordinates[...,1]+.5)/h-1),-1)[None]
            sampled=F.grid_sample(probabilities,grid,align_corners=False)[0].permute(1,2,0)
            semantic_profiles={name:sampled[...,list(classes).index(name)]
                               for name in ("ice_contact","background","ice_paint","ad_boundary","shadow_or_reflection")
                               if name in classes}
        ordered = _ordered_board_foot_profile_batches(profiles,ice,board_height,semantic_profiles)
        transition_support=source_material_transition_support(profiles,ordered)
        side_evidence=_source_material_ice_side_evidence(profiles,ordered['exit_index'],semantic_profiles)
        graphic_support=source_graphic_material_support_conflicts(profiles,ordered,graphics)
        material = ordered["gamma"]
        offsets = profiles["rgb_gradient_offsets_native_px"]
        readings = []
        for role, sign in (("ice_contact_board_foot_lower_edge", -1),
                           ("board_internal_material_interface_foot_upper_edge", 1)):
            index = ordered["exit_index"] if sign < 0 else ordered["entry_index"]
            accepted = ordered["usable"] if sign < 0 else ordered["usable_upper"]
            selected_offset=ordered['selected_exit_offset_native_px']if sign<0 else offsets[index]
            uv = profiles["points_native_uv"]+profiles["normals_unit"]*selected_offset[:, None]
            uv = uv.masked_fill(ordered["missing_same_band"][:, None], float("nan"))
            # Model person regions are conflicts, not authorized RGB vetoes.
            # Keep every optical candidate UV while withholding a physical
            # strip role where another current-source surface is plausible.
            conflict = torch.zeros(len(uv),device=device,dtype=torch.bool)
            for instance in instances["instances"]:
                if instance["kind"] not in ("occluder","broadcast_overlay"):
                    continue
                query=torch.stack((2*(uv[:,0]+.5)/w-1,2*(uv[:,1]+.5)/h-1),-1)[None,None]
                value=F.grid_sample(instance["region_mask"].to(device)[None,None].float(),query,align_corners=False)[0,0,0]
                conflict |= value >= .5
            graphic_conflict=torch.zeros(len(uv),device=device,dtype=torch.bool)
            for rectangle in graphics['candidates']:
                x0,y0,x1,y1=rectangle['box_xyxy']
                graphic_conflict|=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
            graphic_point_conflict=graphic_conflict.clone()
            graphic_conflict |= graphic_support['selected_material_support_overlaps_source_graphic']
            accepted = accepted & ~conflict
            selected = torch.nonzero(accepted).flatten()
            original_samples = np.flatnonzero(valid)
            ids = original_samples[selected.cpu().numpy()]
            component_labels=np.asarray(query_evidence.get('query_component_labels',np.zeros(len(points),int)))
            disconnected=(np.diff(ids)!=1) | (component_labels[ids[1:]]!=component_labels[ids[:-1]]) | (
                np.linalg.norm(np.diff(uv[selected].cpu().numpy(),axis=0),axis=1)>max(12,sample_spacing_px*3))
            breaks = np.r_[0, np.flatnonzero(disconnected)+1, len(ids)]
            actual = uv[selected]
            runs = [dict(actual_source_uv=actual[start:end], original_source_search_indices=ids[start:end])
                    for start, end in zip(breaks[:-1], breaks[1:]) if end > start]
            readings.append(dict(physical_role=role, actual_source_uv=actual, measured_runs=runs,
                candidate_source_uv=uv, source_indices=original_samples,
                candidate_usable=accepted, same_material_band_pair_score=ordered["selected_pair_score"],
                flags=dict(missing_same_material_band=ordered["missing_same_band"],
                    competing_same_material_bands=ordered["competing_same_band_pairs"],
                    current_source_occluder_or_overlay_role_conflict=conflict,
                    source_RGB_graphic_rectangle_role_conflict=graphic_conflict,
                    source_RGB_graphic_rectangle_point_role_conflict=graphic_point_conflict,
                    source_RGB_graphic_material_support_role_conflict=graphic_support['selected_material_support_overlaps_source_graphic'],
                    original_RGB_profile_flags=profiles["flags"]),
                material_role_authority=material_role_authority,
                mask_or_navigation_UV_used_as_output=False, source_pixel_accuracy_accepted=False))
        nested_material=source_nested_material_band_conflicts(readings[0]['candidate_source_uv'],
            readings[0]['candidate_usable'],ordered,profiles['normals_unit'])
        nested_conflict=torch.as_tensor(nested_material['internal_source_material_band_role_conflict'],device=device)
        families=source_foot_material_role_families(readings[0]['candidate_source_uv'],readings[0]['candidate_usable'],
            ordered,native_hw=(h,w),source_transition_support=transition_support,
            source_role_conflicts=nested_conflict|side_evidence['printed_or_body_material_conflict']|
                readings[0]['flags']['source_RGB_graphic_rectangle_role_conflict'])
        direction_evidence=source_local_material_direction_evidence(profiles,ordered,
            readings[0]['candidate_source_uv'],readings[0]['candidate_usable'],families,
            nested_conflict|side_evidence['printed_or_body_material_conflict']|
                readings[0]['flags']['source_RGB_graphic_rectangle_role_conflict'])
        direction_conflict=torch.as_tensor(direction_evidence['source_material_direction_role_conflict'],device=device)
        families['primary_role_samples_before_local_material_role_review']=families['primary_role_samples'].copy()
        families['primary_role_samples'] &= ~direction_evidence['source_material_direction_role_conflict']
        families['unresolved_optical_samples']=_numpy(readings[0]['candidate_usable']).astype(bool)&~families['primary_role_samples']
        families['source_material_direction_role_conflicts']=direction_evidence['source_material_direction_role_conflict']
        for family in families['families']:
            members=np.asarray(family['original_RGB_query_indices'],dtype=int)
            family['local_source_material_role_conflict_count']=int(
                direction_evidence['source_material_direction_role_conflict'][members].sum())
        family_mask=torch.as_tensor(families['primary_role_samples'],device=device)
        for reading in readings:
            primary=family_mask & reading['candidate_usable'] & ~reading['flags']['source_RGB_graphic_rectangle_role_conflict']
            if reading['physical_role'].startswith('ice_contact'):
                primary &= ~side_evidence['printed_or_body_material_conflict']
                primary &= ~nested_conflict
            observed=~ordered['missing_same_band']
            primary_indices=torch.nonzero(primary).flatten()
            indices=primary_indices.cpu().numpy()
            labels=np.asarray(query_evidence.get('query_component_labels',np.zeros(len(points),int)))
            primary_points=reading['candidate_source_uv'][primary]
            interrupted=(np.diff(indices)!=1) | (labels[indices[1:]]!=labels[indices[:-1]]) | (
                np.linalg.norm(np.diff(primary_points.cpu().numpy(),axis=0),axis=1)>max(12,sample_spacing_px*3))
            boundaries=np.r_[0,np.flatnonzero(interrupted)+1,len(indices)]
            primary_runs=[dict(actual_source_uv=primary_points[start:end],
                original_source_search_indices=indices[start:end])
                for start,end in zip(boundaries[:-1],boundaries[1:]) if end>start]
            reading.update(primary_role_samples=primary,primary_source_uv=reading['candidate_source_uv'][primary],
                primary_measured_runs=primary_runs,primary_source_search_indices=primary_indices,
                all_observed_optical_source_uv=reading['candidate_source_uv'][observed],
                unresolved_source_uv=reading['candidate_source_uv'][observed&~primary],
                physical_role_is_source_material_hypothesis=True,
                upper_material_entry_precision_accepted=False)
            reading['flags']['source_observed_nested_material_band_role_conflict']=nested_conflict
            reading['flags']['source_local_material_direction_role_conflict']=direction_conflict
    return dict(schema="rinkspace.actual-board-foot-material-profiles.v1", source_binding=binding,
        status="both_material_interfaces_read_as_separate_source_hypotheses", readings=readings,
        search_points_uv=points, navigation_excluded_original_indices=np.flatnonzero(conflicts),
        source_query_evidence=query_evidence,
        requested_search_radius_native_px=search_radius_px,effective_search_radius_native_px=effective_radius,
        navigation_missing_original_indices=np.flatnonzero(~valid),
        original_RGB_profiles=profiles, signed_warm_material_profile=material,
        same_material_band_pair_evidence=ordered,
        source_foot_physical_role_families=families,source_material_ice_side_evidence=side_evidence,
        source_observed_nested_material_band_evidence=nested_material,
        source_material_transition_support=transition_support,
        source_local_material_direction_evidence=direction_evidence,
        source_broadcast_graphic_candidates=graphics,
        source_broadcast_graphic_material_support_evidence=graphic_support,
        direct_user_material_role_confirmed=True, material_role_authority=material_role_authority,
        annotation_pixel_coordinates_used_as_input=False,
        source_model_semantics_used_only_for_search_and_candidate_role_ranking=False,
        source_model_semantics_used_only_for_search_and_context_disagreement=prediction is not None,
        source_model_sha256=prediction.get("model_sha256") if prediction else None,
        model_occluder_masks_used_as_RGB_measurement_veto=False,
        upper_foot_interface_is_ice_contact=False, fitted_points_used_as_output=False)
