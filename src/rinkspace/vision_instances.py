"""Source-bound SAM2 object candidates with independent native RGB edge reading.

SAM2 masks locate search regions and possible occluders. They never establish
an advertisement's printed perimeter, a physical number, or accurate pixels.
The full local video model supplies its decoder and object memory; no rink or
camera fit is performed here. All learned/model-prompted identities stay
expected identities unless a separate human source authority is supplied.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROMPT_ORIGINS = frozenset(("user_confirmed_visual_supervision",
                            "model_projection_search_hint", "original_rgb_candidate"))
INSTANCE_KINDS = frozenset(("ice_region", "board_region", "advertisement_panel", "occluder",
                            "broadcast_overlay"))
HUMAN_IDENTITIES = frozenset(("direct_user_confirmation", "user_confirmed_visual_supervision",
                             "human_confirmed_event_catalog"))
SOURCE_CORRESPONDENCE_IDENTITY = "source_reference_context_and_print_identity_transfer"


def source_correspondence_identity_verified(authority, binding, physical_id):
    """Authenticate machine identity transfer separately from human pixel facts."""
    if not isinstance(authority, Mapping) or authority.get("kind") != SOURCE_CORRESPONDENCE_IDENTITY:
        return False
    if binding.get("event_id", authority.get("event_id")) != authority.get("event_id"):
        raise ValueError("Source identity transfer belongs to a different event")
    same_frame(binding, authority.get("query_source_binding", {}))
    reference = source_binding(authority.get("reference_source_binding", {}))
    if any(reference[key] != binding[key] for key in ("source_sha256", "time_base")):
        raise ValueError("Source identity transfer belongs to a different original source/clock")
    human = authority.get("confirmed_reference_identity_authority", {})
    from .vision_native_references import registered_source_identity_verified
    reference_valid = (isinstance(human, Mapping) and human.get("kind") in HUMAN_IDENTITIES and human.get("reference"))
    reference_valid = reference_valid or registered_source_identity_verified(human, reference, physical_id)
    if not (reference_valid and authority.get("reference_physical_id") == physical_id):
        return False
    for key in ("source_packet_sha256", "query_RGB_sha256"):
        digest = str(authority.get(key, ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return False
    if binding.get("decoded_native_rgb_sha256", authority["query_RGB_sha256"]) != authority["query_RGB_sha256"]:
        raise ValueError("Source identity transfer belongs to different query RGB")
    context = authority.get("static_context_evidence", {})
    print_match = authority.get("current_print_correspondence", {})
    points = np.asarray(print_match.get("actual_source_keypoints_uv", []), float)
    return bool(context.get("distributed_distinctive_context_support") is True
        and context.get("event_order_consistent") is True
        and context.get("source_packet_sha256") == authority["source_packet_sha256"]
        and int(context.get("distinctive_descriptor_match_count", 0)) >= 6
        and print_match.get("appearance_match") is True
        and print_match.get("source_packet_sha256") == authority["source_packet_sha256"]
        and points.ndim == 2 and points.shape[1] == 2 and len(points) >= 4
        and np.isfinite(points).all() and authority.get("query_human_pixel_ground_truth") is False
        and authority.get("boundary_UV_verified_by_identity_transfer") is False)


def source_binding(value: Mapping[str, Any], native_hw=None) -> dict:
    """Validate original pixel/time identity without assigning an exposure rule."""
    result = dict(value)
    digest = str(result.get("source_sha256", "")).lower()
    stamp = result.get("source_pts")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("An actual source SHA256 is required")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)):
        raise ValueError("Original source PTS must be an integer")
    clock = Fraction(str(result.get("time_base", result.get("source_time_base", "0"))))
    if clock <= 0 or not result.get("shot_id"):
        raise ValueError("A positive original time base and shot_id are required")
    result.update(source_sha256=digest, source_pts=int(stamp), time_base=str(clock))
    if native_hw is not None:
        height, width = map(int, native_hw)
        if min(height, width) < 2 or result.get("native_hw", [height, width]) != [height, width]:
            raise ValueError("Actual RGB and declared native dimensions differ")
        result["native_hw"] = [height, width]
    if result.get("image_domain", "native_brown_pixels") != "native_brown_pixels":
        raise ValueError("Object reading requires unrectified native source pixels")
    return result


def same_frame(first, second):
    a, b = source_binding(first), source_binding(second)
    for key in ("source_sha256", "source_pts", "time_base", "shot_id", "event_id", "rig_id", "installation_id", "native_hw"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Prompt and RGB source binding differ: {key}")
    return a


def validate_prompt(prompt: Mapping[str, Any], binding: Mapping[str, Any]) -> dict:
    """A physical identity requires named human authority, independently of SAM."""
    result = dict(prompt)
    same_frame(binding, result.get("source_binding", {}))
    object_id = result.get("object_id")
    if isinstance(object_id, bool) or not isinstance(object_id, (int, np.integer)):
        raise ValueError("SAM object_id must be an integer; it is not the physical ad number")
    if result.get("kind") not in INSTANCE_KINDS or result.get("origin") not in PROMPT_ORIGINS:
        raise ValueError("Declare an object kind and source/model/human prompt origin")
    height, width = binding["native_hw"]
    points = result.get("points_uv")
    box = result.get("box_xyxy")
    if points is None and box is None:
        raise ValueError("Provide source-bound point/box search prompts")
    if points is not None:
        points = np.asarray(points, dtype=float)
        labels = np.asarray(result.get("point_labels"), dtype=int)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or labels.shape != (len(points),) or not np.isin(labels, (0, 1)).all()
                or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()):
            raise ValueError("Point prompts/positive-negative labels must lie in native RGB")
        result.update(points_uv=points.tolist(), point_labels=labels.tolist())
    if box is not None:
        box = np.asarray(box, dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError("Box search prompts must lie in native RGB")
        result["box_xyxy"] = box.tolist()
    claimed = result.get("physical_id") or result.get("expected_id")
    authority = result.get("identity_authority")
    human_verified = (isinstance(authority, Mapping) and authority.get("kind") in HUMAN_IDENTITIES
                and bool(authority.get("reference")))
    machine_verified = source_correspondence_identity_verified(authority, binding, claimed)
    verified = human_verified or machine_verified
    result.update(object_id=int(object_id), expected_id=claimed,
                  physical_id=claimed if verified else None,
                  identity_authority=dict(authority) if verified else None,
                  identity_confirmed_by_source_human=bool(human_verified),
                  identity_confirmed_by_source_correspondence=bool(machine_verified),
                  source_binding=dict(binding), prompt_is_boundary_measurement=False)
    if not verified and isinstance(authority, Mapping):
        result["expected_identity_authority"] = dict(authority)
    return result


def merge_prompt_identity(previous, current):
    """Refresh search coordinates without erasing a human fact; accept corrections."""
    result = dict(current)
    if previous and previous.get("physical_id") and not result.get("physical_id"):
        result.update(physical_id=previous["physical_id"], expected_id=previous["physical_id"],
                      identity_authority=previous["identity_authority"],
                      identity_confirmed_by_source_human=bool(previous.get("identity_confirmed_by_source_human")),
                      identity_confirmed_by_source_correspondence=bool(previous.get("identity_confirmed_by_source_correspondence")))
    elif (previous and previous.get("identity_confirmed_by_source_human")
          and not result.get("identity_confirmed_by_source_human")):
        result.update(physical_id=previous["physical_id"], expected_id=previous["physical_id"],
                      identity_authority=previous["identity_authority"],
                      identity_confirmed_by_source_human=True,
                      identity_confirmed_by_source_correspondence=False,
                      competing_machine_identity= current.get("physical_id"))
    for key in ("boundary_definition", "boundary_definition_authority"):
        if previous and key not in result and key in previous:
            result[key] = previous[key]
    return result


def transport_search_box(source_box, initial_region_box, current_region_box):
    """Move a search support with a region hypothesis, never a measured edge."""
    source, initial, current = map(lambda value: np.asarray(value, dtype=float),
                                  (source_box, initial_region_box, current_region_box))
    if any(value.shape != (4,) or not np.isfinite(value).all() for value in (source, initial, current)):
        raise ValueError("Search transport requires finite boxes")
    old_size, new_size = initial[2:]-initial[:2], current[2:]-current[:2]
    if (old_size <= 0).any() or (new_size <= 0).any():
        raise ValueError("A region search box cannot have zero dimensions")
    relative = (source.reshape(2, 2)-initial[:2])/old_size
    return (current[:2]+relative*new_size).reshape(4).tolist()


def current_material_region_search_box(prompt, search_box, current_region_box):
    """An unnamed material seed may navigate beyond its initial partial extent."""
    if (not prompt.get('source_material_proposal_unverified') or current_region_box is None
            or prompt.get('shared_SAM_board_search_only')):
        return search_box
    current=np.asarray(current_region_box,float)
    initial=current if search_box is None else np.asarray(search_box,float)
    if current.shape!=(4,) or initial.shape!=(4,) or not np.isfinite(np.r_[current,initial]).all():
        raise ValueError('Material navigation requires finite current native search extents')
    # A segmentation may extend the search; every returned boundary remains
    # a new RGB profile measurement, rather than any side of these boxes.
    return np.r_[np.minimum(initial[:2],current[:2]),np.maximum(initial[2:],current[2:])].tolist()


def source_RGB_tensors(rgb, binding, *, device="cuda"):
    """One authenticated native RGB tensor for all current source searches.

    The cache contains no prediction, projected point or boundary measurement.
    Its original clock, image bytes and device must agree before reuse.
    """
    import torch
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Source RGB cache requires original native uint8 RGB')
    binding = source_binding(binding, image.shape[:2])
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if binding.get('decoded_native_rgb_sha256', digest) != digest:
        raise ValueError('Source RGB cache differs from its declared native bytes')
    tensor = torch.as_tensor(image.copy(), device=device).permute(2, 0, 1)[None].float()/255.
    centered = tensor[0]-tensor[0].mean(0, keepdim=True)
    return dict(source_binding=binding, source_RGB_sha256=digest, image=tensor,
                neutral_probability=tensor.new_full((1, 1, *image.shape[:2]), .5),
                luminance=tensor[0].mean(0), chroma=tensor[0].amax(0)-tensor[0].amin(0),
                centered_RGB=centered, centered_RGB_length=centered.square().sum(0).sqrt(),
                source_tensor_is_boundary_measurement=False)


def _checked_source_RGB_tensors(rgb, cache, *, binding=None, device="cuda"):
    import torch
    if binding is not None:
        same_frame(binding, cache['source_binding'])
    image = np.asarray(rgb)
    if (image.dtype != np.uint8 or image.shape != (*cache['source_binding']['native_hw'], 3)
            or hashlib.sha256(image.tobytes()).hexdigest() != cache['source_RGB_sha256']):
        raise ValueError('Source RGB tensor reuse changed the native image bytes')
    expected_device = torch.device(device)
    if expected_device.type == 'cuda' and expected_device.index is None:
        expected_device = torch.device('cuda', torch.cuda.current_device())
    if cache['image'].device != expected_device:
        raise ValueError('Source RGB tensor cache uses a different device')
    return cache


def material_prompts_with_shared_board_search(rgb, binding, prompts, *, board_search,
                                             board_instance=None, board_object_id=299999):
    """Many material layers share one board search; no physical objects merge.

    Current RGB contours remain independent. A missing SAM board creates one
    navigation object, never one decoder per palette layer. The observed local
    rail/foot search supplies point navigation only, including its uncertainty.
    """
    image = np.asarray(rgb)
    binding = source_binding(binding, image.shape[:2])
    same_frame(binding, board_search['source_binding'])
    mask = np.asarray(board_search['search_mask'])
    if image.dtype != np.uint8 or mask.dtype != bool or mask.shape != image.shape[:2]:
        raise ValueError('Shared board search requires native RGB and a bool navigation region')
    if board_search.get('RGB_hard_veto_allowed'):
        raise ValueError('Shared board search cannot use a navigation mask as an RGB veto')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if board_search.get('RGB_sha256', digest) != digest:
        raise ValueError('Shared board search has different original RGB bytes')
    parent_prompt = None
    if board_instance is not None:
        same_frame(binding, board_instance['source_binding'])
        if board_instance['kind'] != 'board_region':
            raise ValueError('Only a board region may provide shared material search')
        parent_id = int(board_instance['object_id'])
    else:
        parent_id = int(board_object_id)
        rows = []
        luminance = image.mean(2)/255.
        chroma = (image.max(2).astype(float)-image.min(2))/255.
        for x in np.linspace(0, image.shape[1]-1, 16).round().astype(int):
            ys = np.flatnonzero(mask[:, x])
            if not len(ys):
                continue
            middle = (ys[0]+ys[-1])/2.
            neutral = ys[(luminance[ys, x] > .45) & (chroma[ys, x] < .12)]
            eligible = neutral if len(neutral) else ys
            y = int(eligible[np.argmin(np.abs(eligible-middle))])
            rows.append([float(x), float(y)])
        if not rows:
            return dict(source_binding=binding, status='missing_current_board_search_support',
                        prompts=[], material_layer_count=len(prompts), new_SAM_board_object_count=0)
        parent_prompt = dict(object_id=parent_id, kind='board_region', origin='original_rgb_candidate',
            source_binding=binding, points_uv=rows, point_labels=[1]*len(rows),
            physical_id=None, expected_id=None, source_material_board_search_only=True,
            prompt_target_role='current_board_navigation_region_not_physical_boundary')
    group = f"source_board_material_search:{binding['source_sha256'][:12]}:{binding['shot_id']}:{parent_id}"
    layers = []
    for original in prompts:
        same_frame(binding, original['source_binding'])
        if (not original.get('source_material_proposal_unverified') or original.get('physical_id')
                or int(original['object_id']) == parent_id):
            raise ValueError('Shared board search cannot merge named physical instances or its parent')
        layers.append({**original, 'sam_search_object_id': parent_id,
            'source_original_material_search_group_id': original.get('source_material_search_group_id'),
            'source_material_search_group_id': group, 'shared_SAM_board_search_only': True,
            'independent_RGB_material_layer_preserved': True,
            'shared_SAM_search_is_physical_identity_verification': False})
    return dict(source_binding=binding, source_RGB_sha256=digest, status='shared_current_board_material_search',
                prompts=([parent_prompt] if parent_prompt else [])+layers,
                material_layer_count=len(layers), SAM_search_parent_object_id=parent_id,
                new_SAM_board_object_count=int(parent_prompt is not None),
                physical_instances_merged=False, independent_material_boundaries_preserved=True,
                navigation_is_measurement=False)


def propose_source_region_prompts(rgb, binding, class_probabilities, classes, *,
                                 probabilities_conditioned_on_geometry,
                                 first_object_id=1000, person_boxes=(), overlay_boxes=()):
    """Reusable original-RGB/typed-boundary topology -> SAM2 search seeds.

    Long source-gradient components are required before contact/top predictions
    propose regions. Short timer boxes and separate board letters remain in the
    rejected-component audit. All grouping/closing is search topology only;
    its coordinates are never emitted as actual ice or rail boundary pixels.
    Person/overlay boxes must come from independent source observations. They
    retain that provenance; an overlay candidate is never an occlusion veto.
    """
    import cv2
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Region proposals require original RGB uint8")
    binding = source_binding(binding, image.shape[:2])
    if probabilities_conditioned_on_geometry is not False:
        raise ValueError("Automatic source proposals require explicitly no-geometry probabilities")
    probability = class_probabilities
    if hasattr(probability, "detach"):
        probability = probability.detach().float().cpu().numpy()
    probability = np.asarray(probability)
    if probability.ndim == 4 and probability.shape[0] == 1:
        probability = probability[0]
    names = tuple(classes)
    height, width = image.shape[:2]
    if (probability.shape != (len(names), height, width) or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()):
        raise ValueError("No-prior probabilities must share the original frame domain")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    magnitude = np.hypot(gx, gy)
    transverse = (magnitude >= .015) & (np.abs(gy) >= np.abs(gx)*.7)
    hypotheses, rejected = [], []
    for name in ("ice_contact", "board_top"):
        if name not in names:
            continue
        typed = probability[names.index(name)]
        evidence = transverse & (cv2.dilate(typed, np.ones((3, 3), np.uint8)) >= .2)
        # Close small gaps only to group search evidence; true UV remains RGB.
        grouped = cv2.morphologyEx(evidence.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((3, max(3, width//100)), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
        for component in range(1, count):
            x0, y0, cw, ch, area = map(int, stats[component])
            native_support = evidence & (labels == component)
            yy, xx = np.nonzero(native_support)
            if not len(xx):
                continue
            span = cw/width
            reason = ("short_isolated_timer_or_letter_or_other_component" if span < .3 else
                      "large_vertical_or_wall_structure_not_a_rail_candidate" if ch > height*.28 else None)
            record = dict(kind=name, source_bbox_xyxy=[x0, y0, x0+cw, y0+ch],
                          source_gradient_pixels=len(xx), horizontal_span_fraction=span,
                          coordinates_are_search_topology_only=True, physical_identity_verified=False)
            if reason:
                rejected.append({**record, "reason": reason})
                continue
            sample = np.linspace(0, len(xx)-1, min(len(xx), 128)).astype(int)
            xs, ys = xx[sample], yy[sample]
            below, above = np.clip(ys+32, 0, height-1), np.clip(ys-32, 0, height-1)
            pale_below = (gray[below, xs] > .5).mean()
            light_change = float(np.median(gray[below, xs]-gray[above, xs]))
            # Pale wall/window panels can have a stronger luminance step than
            # the actual rail. Keep that appearance diagnostic out of ranking.
            type_mean = float(typed[yy, xx].mean())
            score = type_mean+span*.2
            starts = np.r_[0, np.nonzero((np.diff(yy) != 0) | (np.diff(xx) != 1))[0]+1]
            stops = np.r_[starts[1:], len(xx)]
            source_runs = np.column_stack((yy[starts], xx[starts], stops-starts)).tolist()
            hypotheses.append({**record, "mean_source_y": float(np.median(yy)),
                               "candidate_score": score, "pale_RGB_below_fraction": float(pale_below),
                               "mean_no_prior_type_probability": type_mean,
                               "source_gradient_pixel_runs_y_x_length": source_runs,
                               "source_runs_include_only_actual_RGB_typed_evidence": True,
                               "closed_grouping_pixels_used_as_source_evidence": False,
                               "source_gradient_pixels_are_exact_boundary_points": False,
                               "same_board_structure_verified": False,
                               "RGB_below_minus_above": light_change})
    prompts = []
    contacts = [row for row in hypotheses if row["kind"] == "ice_contact"]
    tops = [row for row in hypotheses if row["kind"] == "board_top"]
    pairs = []
    pair_audit = []
    ad_signal = (magnitude >= .015) & (probability[names.index("ad_boundary")] >= .2) if "ad_boundary" in names else None
    for top in tops:
        for contact in contacts:
            separation = contact["mean_source_y"]-top["mean_source_y"]
            if not max(2., height*.01) < separation < height*.3:
                continue
            x0 = max(top["source_bbox_xyxy"][0], contact["source_bbox_xyxy"][0])
            x1 = min(top["source_bbox_xyxy"][2], contact["source_bbox_xyxy"][2])
            if x1-x0 < width*.25:
                continue
            floor = int(contact["mean_source_y"])
            corridor_start = max(0, int(floor-height*.3))
            print_below = None
            print_pixels = 0
            if ad_signal is not None:
                rows = ad_signal[corridor_start:floor, x0:x1].sum(1)
                print_pixels = int(rows.sum())
                if print_pixels:
                    print_below = float(rows[max(0, int(top["mean_source_y"])-corridor_start):].sum()/print_pixels)
            intervening = [other for other in tops if
                top["mean_source_y"]+2 < other["mean_source_y"] < contact["mean_source_y"]-max(2., height*.01) and
                other["mean_no_prior_type_probability"] >= top["mean_no_prior_type_probability"]*.6 and
                min(other["source_bbox_xyxy"][2], x1)-max(other["source_bbox_xyxy"][0], x0) >= width*.25]
            if ad_signal is not None and print_pixels:
                intervening = [other for other in intervening if
                    float(ad_signal[max(0, int(other["mean_source_y"])):floor, x0:x1].sum()/print_pixels) >= .8]
            score = top["candidate_score"]+contact["candidate_score"]-.3*bool(intervening)
            if print_below is not None:
                score += .15*print_below
                score -= .3*(print_below < .8)
            record = dict(top_hypothesis=top, contact_hypothesis=contact, candidate_pair_score=score,
                          overlapping_source_span_pixels=x1-x0, board_print_source_gradient_pixels=print_pixels,
                          fraction_board_print_below_top_candidate=print_below,
                          intervening_contact_proximal_top_hypotheses=intervening,
                          same_board_structure_verified=False,
                          candidate_only=True, ranking_uses_source_type_and_print_parent=True,
                          whiteness_or_pale_wall_used_for_ranking=False)
            pair_audit.append(record)
            pairs.append((top, contact, score))
    if pairs:
        top, contact, _ = max(pairs, key=lambda pair: pair[2])
        top["preferred_region_search_hypothesis"] = True
        contact["preferred_region_search_hypothesis"] = True
        lower_y = int(contact["mean_source_y"])
        top_y = int(top["mean_source_y"])
        # These two boxes propose large *regions*, never a fitted rail or rink edge.
        for kind, box in (("ice_region", [0, min(height-2, lower_y), width, height]),
                          ("board_region", [0, max(0, top_y), width, min(height, lower_y+1)])):
            x0, y0, x1, y1 = box
            prompts.append(dict(object_id=first_object_id+len(prompts), kind=kind,
                                origin="original_rgb_candidate", source_binding=binding,
                                box_xyxy=box, points_uv=[[(x0+x1-1)/2, (y0+y1-1)/2]],
                                point_labels=[1], prompt_target_role="source_topology_region_hypothesis",
                                same_board_structure_verified=False,
                                competing_top_hypotheses=tops,
                                topology_evidence=[top, contact]))
    elif contacts:
        contact = max(contacts, key=lambda row: row["candidate_score"])
        yy = min(height-2, max(0, int(contact["mean_source_y"])))
        prompts.append(dict(object_id=first_object_id, kind="ice_region", origin="original_rgb_candidate",
                            source_binding=binding, box_xyxy=[0, yy, width, height],
                            points_uv=[[(width-1)/2, (yy+height-1)/2]], point_labels=[1],
                            prompt_target_role="contact_supported_ice_region_hypothesis", topology_evidence=[contact]))
    for kind, observations in (("occluder", person_boxes), ("broadcast_overlay", overlay_boxes)):
        for observation in observations:
            same_frame(binding, observation["source_binding"])
            prompts.append(dict(object_id=first_object_id+len(prompts), kind=kind,
                                origin="original_rgb_candidate", source_binding=binding,
                                box_xyxy=observation["box_xyxy"],
                                prompt_target_role="source_person_region_candidate" if kind == "occluder" else
                                                   "screen_overlay_candidate_not_verified_static",
                                source_observation_provenance=observation.get("provenance"),
                                exclusion_authority=observation.get("exclusion_authority")))
    return dict(source_binding=binding, prompts=prompts, boundary_topology_hypotheses=hypotheses,
                board_contact_pair_hypotheses=pair_audit,
                same_board_structure_verified=False,
                rejected_short_or_vertical_components=rejected,
                missing_region_reason=None if prompts else "No source-supported long contact/top pair or person seeds",
                no_geometry_or_camera_input=True, automatic_identity_confirmed=False,
                coordinates_are_search_regions_only=True, accuracy_accepted=False)


def automatic_source_region_prompts(rgb, binding, *, class_probabilities=None, classes=(),
                                    probabilities_conditioned_on_geometry=False,
                                    person_boxes=(), overlay_boxes=()):
    """No manual test seed: source RGB topology, optionally assisted by no-prior types."""
    learned = class_probabilities is not None
    if not learned:
        # Neutral gate, not invented semantic probabilities. RGB gradients and
        # topology alone organize hypotheses; the report marks that limitation.
        classes = ("ice_contact", "board_top")
        class_probabilities = np.full((2, *np.asarray(rgb).shape[:2]), .5, np.float32)
    result = propose_source_region_prompts(
        rgb, binding, class_probabilities, classes,
        probabilities_conditioned_on_geometry=probabilities_conditioned_on_geometry,
        person_boxes=person_boxes, overlay_boxes=overlay_boxes)
    result.update(type_prediction_used=learned,
                  type_prediction_origin="no_prior_vision_module" if learned else "not_provided_RGB_topology_only",
                  manual_test_annotations_used=False)
    return result


def board_prompt_from_ice_region(rgb, binding, ice_mask, *, object_id=1001,
                                 top_hypotheses=(), excluded_source_boxes=()):
    """Source mask's upper envelope proposes a perspective-aware board region.

    The mask contour is a search hint, never the contact measurement. Negative
    source clicks above/below the proposed strip keep upper walls/ice separate.
    Independently detected people are avoided when selecting positive clicks.
    """
    if hasattr(ice_mask, "detach"):
        ice_mask = ice_mask.detach().cpu().numpy()
    mask = np.asarray(ice_mask, dtype=bool)
    height, width = mask.shape
    good_columns = mask.any(0)
    columns = np.nonzero(good_columns)[0]
    if len(columns) < width*.3:
        return None
    upper = mask.argmax(0)
    tops = [row for row in top_hypotheses if row["kind"] == "board_top"]
    preferred = [row for row in tops if row.get("preferred_region_search_hypothesis")]
    top = (max(preferred or tops, key=lambda row: row["candidate_score"])["mean_source_y"] if tops else
           float(np.median(upper[columns]))-height*.08)
    points, labels = [], []
    for index in np.linspace(0, len(columns)-1, 7).astype(int):
        x = int(columns[index])
        y = int(upper[x])
        positive_y = max(0, y-24)
        if any(box[0] <= x < box[2] and box[1] <= positive_y < box[3] for box in excluded_source_boxes):
            continue
        points.extend([[x, positive_y], [x, min(height-1, y+32)]])
        labels.extend([1, 0])
    if not points:
        return None
    return dict(object_id=object_id, kind="board_region", origin="original_rgb_candidate",
                source_binding=source_binding(binding, [height, width]),
                box_xyxy=[int(columns.min()), max(0, int(top)-4), int(columns.max())+1,
                          min(height, int(upper[columns].max())+4)],
                points_uv=points, point_labels=labels,
                prompt_target_role="board_region_above_source_ice_candidate_envelope",
                same_board_structure_verified=False, competing_top_hypotheses=tops,
                mask_envelope_is_contact_measurement=False)


class SourcePersonBoxes:
    """Reuse a local person model solely for source occluder box proposals."""
    def __init__(self, weights_path, *, vendor_path=None, dependency_paths=(),
                 runtime_directory=None, device="cuda"):
        import os
        import sys
        import torch
        weights_path = Path(weights_path).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        configured = runtime_directory or os.environ.get("YOLO_CONFIG_DIR")
        if configured:
            runtime = Path(configured).resolve()
            runtime.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(runtime)
        if vendor_path is not None:
            vendor_path = Path(vendor_path).resolve()
            if not (vendor_path/"ultralytics"/"__init__.py").is_file():
                raise FileNotFoundError(vendor_path)
            if str(vendor_path) not in sys.path:
                sys.path.append(str(vendor_path))
        for dependency in dependency_paths:
            dependency = Path(dependency).resolve()
            if not dependency.is_dir():
                raise FileNotFoundError(dependency)
            # Append fallback libraries; never replace the active GPU runtime.
            if str(dependency) not in sys.path:
                sys.path.append(str(dependency))
        from ultralytics import YOLO
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Source person proposals use CUDA")
        self.model = YOLO(str(weights_path))
        self.weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    def read(self, rgb, binding):
        image = np.asarray(rgb)
        binding = source_binding(binding, image.shape[:2])
        # Ultralytics ndarray input is BGR; this conversion preserves source pixels.
        prediction = self.model.predict(source=image[:, :, ::-1].copy(),
                                        device=self.device.index or 0, verbose=False)[0]
        rows = []
        for box, confidence, cls in zip(prediction.boxes.xyxy.cpu().numpy(),
                                        prediction.boxes.conf.cpu().numpy(),
                                        prediction.boxes.cls.cpu().numpy()):
            if int(cls) != 0:
                continue
            rows.append(dict(source_binding=dict(binding), box_xyxy=box.tolist(),
                             provenance=dict(kind="local_pretrained_person_box_prediction",
                                             model_sha256=self.weights_sha256, confidence=float(confidence),
                                             body_pose_used=False, is_pixel_ground_truth=False)))
        return rows


def source_person_refresh_prompts(binding, person_boxes, current_instances, object_prompts):
    """Current detector boxes refresh SAM search; they never exclude RGB pixels.

    A propagated mask can retain a torso while losing a leg. Check its extent
    and support inside the independently detected current box rather than only
    checking whether the whole mask is empty. Numeric object association here
    is search memory, not a person's physical identity.
    """
    binding = source_binding(binding)
    height, width = binding["native_hw"]
    tracked = [item for item in current_instances if item["kind"] == "occluder"]
    used, prompts, ledger = set(), [], []
    next_id = max(object_prompts, default=999)+1

    def array(mask):
        return mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)

    def iou(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
        return float(intersection/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection, 1e-9))

    masks = {}
    for item in tracked:
        mask = array(item["region_mask"]).astype(bool)
        if mask.shape != (height, width):
            raise ValueError("Person search mask must share the current native source")
        yy, xx = np.nonzero(mask)
        masks[item["object_id"]] = (mask, [xx.min(), yy.min(), xx.max()+1, yy.max()+1] if len(xx) else None)

    for observation in person_boxes:
        same_frame(binding, observation["source_binding"])
        box = np.asarray(observation["box_xyxy"], float)
        # Reuse the public prompt validator for actual native box bounds.
        validate_prompt(dict(object_id=next_id, kind="occluder", origin="original_rgb_candidate",
                             source_binding=binding, box_xyxy=box), binding)
        options = []
        for item in tracked:
            object_id = item["object_id"]
            if object_id in used:
                continue
            mask, current_box = masks[object_id]
            old_box = object_prompts.get(object_id, {}).get("box_xyxy")
            score = iou(box, current_box) if current_box else 0.
            # Original prompt boxes only associate an absent search object;
            # they cannot provide its current mask or prove its identity.
            score = max(score, iou(box, old_box) if old_box is not None else 0.)
            options.append((score, object_id, mask))
        options.sort(key=lambda row: row[0], reverse=True)
        selected = options[0] if options and options[0][0] > .1 else None
        if selected is None:
            object_id, mask = next_id, np.zeros((height, width), bool)
            next_id += 1
        else:
            _, object_id, mask = selected
        used.add(object_id)
        x0, y0 = np.floor(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:]).astype(int)
        local = mask[y0:y1, x0:x1]
        yy, xx = np.nonzero(local)
        fraction = float(local.mean())
        vertical = float((yy.max()-yy.min()+1)/(y1-y0)) if len(yy) else 0.
        refresh = selected is None or fraction < .25 or vertical < .85
        record = dict(object_id=object_id, source_binding=dict(binding), current_detector_box_xyxy=box.tolist(),
            propagated_mask_box_support_fraction=fraction, propagated_mask_vertical_extent_fraction=vertical,
            mask_support_threshold_is_pixel_accuracy_bound=False,
            status="source_box_reidentification_search_requested" if refresh else "propagated_person_search_support_present",
            detector_box_used_as_RGB_veto=False, detector_box_is_occlusion_ground_truth=False,
            physical_person_identity_inferred=False, source_observation_provenance=observation.get("provenance"))
        ledger.append(record)
        if refresh:
            prompts.append(dict(object_id=object_id, kind="occluder", origin="original_rgb_candidate",
                source_binding=dict(binding), box_xyxy=box.tolist(),
                prompt_target_role="current_source_person_reidentification_search",
                source_observation_provenance=observation.get("provenance"),
                source_person_refresh_evidence=record))
    return dict(source_binding=dict(binding), prompts=prompts, ledger=ledger,
                source_boxes_are_search_only=True, RGB_measurements_erased=False)


def region_contours(mask, *, stride=4, role_prefix="SAM2_region") -> list[dict]:
    """Mask-outline proposals only, with frame-edge exits kept explicitly open."""
    import cv2
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Contour sampling stride must be a positive integer")
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("Supply one native candidate region mask")
    height, width = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    result = []
    for contour_index, contour in enumerate(contours):
        hole = hierarchy[0, contour_index, 3] >= 0
        role = f"{role_prefix}_internal_hole" if hole else f"{role_prefix}_external_contour"
        full = contour[:, 0].astype(np.float32)
        if len(full) < 3:
            continue
        xy = full[::stride]
        if len(xy) < 3:
            xy = full
        on_frame = ((xy[:, 0] <= 1) | (xy[:, 0] >= width-2) |
                    (xy[:, 1] <= 1) | (xy[:, 1] >= height-2))
        # Never join a mask's clipped side as an observed scene boundary.
        start = 0
        if not on_frame.any():
            result.append(dict(uv=xy, closed=True, role=role, internal_hole=bool(hole),
                               source_component_id=contour_index))
            continue
        for end in range(len(xy)+1):
            if end == len(xy) or on_frame[end]:
                if end-start >= 3:
                    result.append(dict(uv=xy[start:end], closed=False,
                                       role=role, internal_hole=bool(hole),
                                       source_component_id=contour_index))
                start = end+1
    return result


def printed_pattern_candidates(rgb, region_mask, *, source_box=None, source_print_appearance=None,
                               search_pattern_keypoints_uv=(), device="cuda", source_rgb_cache=None,
                               region_search_support=None, background_luminance=None):
    """Actual source-ink topology inside a declared search support.

    A source box/dilated SAM region bounds the search only. Ink components and
    internal holes remain distinct. Their union extent is metadata, never a
    fabricated perimeter. This source color proposal is not complete pattern
    segmentation or calibrated edge precision.
    """
    import torch
    import torch.nn.functional as F
    cache = (_checked_source_RGB_tensors(rgb, source_rgb_cache, device=device)
             if source_rgb_cache is not None else None)
    image = (cache['image'][0] if cache is not None else
             torch.as_tensor(np.asarray(rgb).copy(), device=device).permute(2, 0, 1).float()/255)
    height, width = image.shape[-2:]
    if region_search_support is None:
        support = F.max_pool2d(torch.as_tensor(region_mask, device=device).float()[None, None],
                               25, stride=1, padding=12)[0, 0] > .5
    else:
        support = torch.as_tensor(region_search_support, device=device)
        if support.dtype != torch.bool or tuple(support.shape) != (height, width):
            raise ValueError('Shared region support must be the native bool dilation search')
    if source_box is not None:
        # This is a source navigation/support rectangle, expressly not an edge.
        x0, y0, x1, y1 = source_box
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        navigation = torch.zeros_like(support)
        navigation[y0:y1, x0:x1] = True
        support &= navigation
    luminance = cache['luminance'] if cache is not None else image.mean(0)
    if not bool(support.any()):
        return dict(ink_mask=torch.zeros_like(support), search_support=support, extent=None,
                    extent_is_physical_edge=False, source_pattern_complete=False)
    background = torch.quantile(luminance[support], .85) if background_luminance is None else background_luminance
    chroma = cache['chroma'] if cache is not None else image.amax(0)-image.amin(0)
    ink = support & ((luminance < background-.04) | ((chroma > .12) & (luminance < background-.015)))
    unlinked = torch.zeros_like(ink)
    appearance = source_print_appearance or {}
    if appearance.get("dominant_chromatic_print_background"):
        # A human reference's actual print material is a search descriptor,
        # not a panel rectangle or pixel truth. Link its chromatic components
        # to independently matched pattern keypoints, keeping unrelated rail,
        # skin and foreground candidates separate from the expected print.
        import cv2
        direction = torch.as_tensor(appearance["RGB_chroma_direction"], device=device).float()
        centered = cache['centered_RGB'] if cache is not None else image-image.mean(0, keepdim=True)
        length = cache['centered_RGB_length'] if cache is not None else centered.square().sum(0).sqrt()
        cosine = (centered*direction[:, None, None]).sum(0)/length.clamp_min(1e-6)
        # The original print can contain a dark, weakly chromatic gradient.
        # A fixed saturation cutoff broke that material around white letters
        # and lost their entire parent component. This is a RGB search floor
        # at two source quantization levels, not an ink/pixel accuracy label.
        material = ink & (chroma > 2/255.) & (cosine >= .7)
        count, labels = cv2.connectedComponents(material.detach().cpu().numpy().astype(np.uint8), connectivity=8)
        material_contours, material_hierarchy = cv2.findContours(
            material.detach().cpu().numpy().astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        seeds = np.asarray(search_pattern_keypoints_uv, float).reshape(-1, 2)
        selected = set()
        links = []
        for seed in seeds:
            if not np.isfinite(seed).all():
                continue
            x, y = np.rint(seed).astype(int)
            if 0 <= x < width and 0 <= y < height:
                nearby = set(int(v) for v in labels[max(0, y-3):min(height, y+4),
                                                    max(0, x-3):min(width, x+4)].ravel() if v)
                parents = set()
                for index, contour in enumerate(material_contours):
                    parent = int(material_hierarchy[0, index, 3])
                    if parent >= 0 and cv2.pointPolygonTest(contour, tuple(seed), False) >= 0:
                        px, py = material_contours[parent][0, 0]
                        if labels[py, px]:parents.add(int(labels[py, px]))
                selected.update(nearby | parents)
                links.append(dict(search_keypoint_uv=seed.tolist(), search_keypoint_is_boundary_measurement=False,
                    nearby_source_material_labels=sorted(nearby),
                    enclosing_source_hole_parent_labels=sorted(parents), correspondence_used_as_boundary_UV=False))
        linked = torch.as_tensor(np.isin(labels, list(selected)), device=device)
        unlinked = ink & ~linked
        ink = linked
    yx = torch.nonzero(ink)
    extent = None
    if len(yx):
        lo, hi = yx.amin(0), yx.amax(0)
        extent = [int(lo[1]), int(lo[0]), int(hi[1])+1, int(hi[0])+1]
    return dict(ink_mask=ink, search_support=support, extent=extent,
                source_chromatic_material_candidate_mask=material if appearance.get("dominant_chromatic_print_background") else None,
                source_navigation_box_replaces_object_support=False,
                source_pattern_material_component_links=links if appearance.get("dominant_chromatic_print_background") else [],
                source_chroma_search_quantization_floor=2/255. if appearance.get("dominant_chromatic_print_background") else None,
                unlinked_source_foreground_candidate_mask=unlinked,
                original_print_appearance_is_search_descriptor=True,
                pattern_keypoints_used_as_boundary_measurements=False,
                extent_is_physical_edge=False, source_pattern_complete=False,
                source_RGB_foreground_rule="local_bright_background_dark_or_chromatic_ink_candidate",
                background_luminance=float(background), internal_holes_separate=True)


def source_print_boundary_contours(pattern, *, stride=4, appearance=None):
    """Keep printed outer paths distinct from a selected color layer's holes.

    A white glyph surrounded by blue print is a blue/white material boundary;
    the selected blue mask's hole is not evidence of an unprinted whole-pattern
    hole. These are still search contours, requiring the actual RGB profile.
    """
    contours = region_contours(pattern["ink_mask"], stride=stride, role_prefix="printed_pattern")
    chromatic = bool((appearance or {}).get("dominant_chromatic_print_background"))
    for contour in contours:
        contour["source_boundary_is_output_measurement"] = False
        contour["solver_outer_boundary_eligible"] = not contour["internal_hole"]
        if chromatic:
            contour["print_material_scope"] = (appearance or {}).get(
                "source_material_authority_scope", "authenticated_chromatic_print_background_layer")
            contour["whole_printed_pattern_hole_verified"] = False
            contour["selected_material_hole"] = contour["internal_hole"]
            if contour["internal_hole"]:
                contour["role"] = "printed_pattern_internal_material_boundary"
                contour["internal_hole_scope"] = "selected_chromatic_material_layer_not_whole_print_pattern"
        else:
            contour["print_material_scope"] = "source_ink_component"
            if contour["internal_hole"]:
                contour["internal_hole_scope"] = "source_ink_component_complement_candidate"
    return contours


def source_print_component_continuity(rgb, binding, pattern, *, previous_evidence=None,
                                      previous_rgb=None, source_person_search_mask=None):
    """Actual source material tracks keep a known instance's visible fragments.

    Tracks originate in the previous linked source component. Current search
    covers all source color candidates in that object's SAM/navigation support,
    before glyph linking, so an occluded glyph cannot recursively erase its
    still-visible color fragment. No H transports a boundary or invents UV.
    """
    import cv2
    _add_existing_geometry_dependencies()
    binding = source_binding(binding, np.asarray(rgb).shape[:2])
    base = dict(source_binding=binding, search_only=True, actual_source_boundary_UV_transported=False,
                physical_instance_identity_reassigned=False, source_component_link_is_pixel_ground_truth=False,
                current_source_feature_uv=np.empty((0,2),np.float32))
    if previous_evidence is None or previous_rgb is None:
        return {**base, "status": "reference_pattern_initialization_without_previous_source"}
    old = source_binding(previous_evidence["source_binding"], np.asarray(previous_rgb).shape[:2])
    if any(old.get(key) != binding.get(key) for key in
           ("source_sha256", "time_base", "shot_id", "event_id", "rig_id", "installation_id", "native_hw")):
        raise ValueError("Print material tracks cannot cross a source or shot binding")
    if hashlib.sha256(np.asarray(previous_rgb).tobytes()).hexdigest() != previous_evidence.get("source_RGB_sha256"):
        raise ValueError("Print material track memory RGB changed")
    if old["source_pts"] == binding["source_pts"]:
        points = previous_evidence.get("_current_source_feature_uv_only", np.empty((0,2),np.float32))
        overlap_points = previous_evidence.get("_current_RGB_component_overlap_search_points_uv", np.empty((0,2),np.float32))
        pair = previous_evidence.get("source_material_instance_continuity")
        if pair and pair.get("status") == "same_source_frame_component_search_points_reused":
            pair = pair.get("actual_previous_native_pair_evidence")
        return {**base, "status": "same_source_frame_component_search_points_reused", "current_source_feature_uv": np.asarray(points,np.float32),
                "current_RGB_component_overlap_search_points_uv": np.asarray(overlap_points,np.float32),
                "actual_previous_native_pair_evidence": pair}
    if old["source_pts"] > binding["source_pts"]:
        raise ValueError("Print material tracks require chronological actual source frames")
    previous = previous_evidence.get("_source_component_material_mask")
    current = pattern.get("source_chromatic_material_candidate_mask")
    if previous is None or current is None:
        return {**base, "status": "missing_previous_linked_or_current_source_material_support"}
    current = current.detach().cpu().numpy() if hasattr(current,"detach") else np.asarray(current)
    previous = np.asarray(previous,bool)
    if previous.shape != rgb.shape[:2] or current.shape != rgb.shape[:2]:
        raise ValueError("Print material support must retain actual native dimensions")
    previous_valid = cv2.dilate(previous.astype(np.uint8),np.ones((15,15),np.uint8))>0
    current_valid = cv2.dilate(current.astype(np.uint8),np.ones((15,15),np.uint8))>0
    previous_person = previous_evidence.get("_source_person_search_mask")
    if previous_person is not None:previous_valid &= ~np.asarray(previous_person,bool)
    if source_person_search_mask is not None:
        person = source_person_search_mask.detach().cpu().numpy() if hasattr(source_person_search_mask,"detach") else np.asarray(source_person_search_mask)
        if person.shape != rgb.shape[:2]:raise ValueError("Person navigation differs from current native source")
        current_valid &= ~person.astype(bool)
    from .camera_context_reuse import temporal_correspondences
    previous_gray = cv2.cvtColor(np.asarray(previous_rgb),cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2GRAY)
    a,b,audit = temporal_correspondences([previous_gray,current_gray],
        [previous_valid.astype(np.uint8)*255,current_valid.astype(np.uint8)*255],0,1,min_points=6)
    if a is None or b is None:
        return {**base, "status": "missing_actual_material_tracks", "tracking_audit": audit}
    correlations = []
    for left,right in zip(a,b):
        p=cv2.getRectSubPix(previous_gray,(11,11),tuple(left)).astype(float)
        q=cv2.getRectSubPix(current_gray,(11,11),tuple(right)).astype(float)
        correlations.append(float(np.mean((p-p.mean())*(q-q.mean()))/max(p.std()*q.std(),1e-9)))
    correlations = np.asarray(correlations,np.float32)
    supported=correlations>=.6
    a,b,correlations=a[supported],b[supported],correlations[supported]
    H,selected = cv2.findHomography(a,b,cv2.RANSAC,2.5) if len(a)>=6 else (None,None)
    if H is None or selected is None or selected.sum()<6:
        return {**base, "status": "material_source_plane_consensus_missing", "tracking_audit": audit,
                "source_patch_correlations": correlations}
    selected=selected.ravel().astype(bool)
    # A temporarily plain/occluded printed fragment can have no reliable
    # corner of its own. The measured same-plane source pair only navigates
    # the previous linked support; current source RGB components supply any
    # new search point. Warped old mask coordinates are never boundary UV.
    warped = cv2.warpPerspective(previous.astype(np.uint8),H,(rgb.shape[1],rgb.shape[0]),
        flags=cv2.INTER_NEAREST)>0
    eligible=current.astype(bool)
    if source_person_search_mask is not None:eligible &= ~person.astype(bool)
    count,labels,stats,_=cv2.connectedComponentsWithStats(eligible.astype(np.uint8),connectivity=8)
    overlaps=[];search_points=[]
    for label in range(1,count):
        x,y,width,height,area=map(int,stats[label])
        component=labels[y:y+height,x:x+width]==label
        overlap=component&warped[y:y+height,x:x+width]
        pixels=int(overlap.sum())
        linked=pixels>0 and pixels/area>=.5
        if pixels:
            overlaps.append(dict(current_source_material_component_label=label,current_source_material_pixels=area,
                previous_linked_navigation_overlap_pixels=pixels,source_navigation_overlap_fraction=pixels/area,
                source_material_component_search_supported=linked,overlap_is_pixel_identity_ground_truth=False))
        if linked:
            yx=np.argwhere(overlap)
            pixel=yx[np.argmin(((yx-yx.mean(0))**2).sum(1))]
            search_points.append([float(pixel[1]+x),float(pixel[0]+y)])
    return {**base, "status": "actual_linked_source_material_component_tracks", "previous_source_binding": old,
        "previous_source_feature_uv": a[selected].astype(np.float32), "current_source_feature_uv": b[selected].astype(np.float32),
        "source_patch_correlations": correlations[selected], "actual_pair_image_H_search_only": H,
        "current_RGB_component_overlap_search_points_uv": np.asarray(search_points,np.float32).reshape(-1,2),
        "current_RGB_material_component_navigation_overlap": overlaps,
        "prior_material_support_projected_for_navigation_only": True,
        "overlap_search_points_are_current_material_pixels_not_boundary_UV": True,
        "source_plane_consensus_is_world_camera": False, "previous_linked_material_support_only": True,
        "current_material_search_precedes_glyph_component_linking": True,
        "person_masks_used_for_feature_navigation_only": True, "tracking_audit": audit}


def source_colour_material_evidence(rgb, binding, pattern, appearance):
    """Current RGB print-to-background material support, separate from atlas UV.

    Matched original pattern components establish the search parent. This
    compares the original current colors on both sides of their visible paths;
    a foreground interruption is retained as an optical role conflict.
    """
    import cv2
    binding = source_binding(binding, np.asarray(rgb).shape[:2])
    support = pattern["search_support"]
    support = support.detach().cpu().numpy() if hasattr(support, "detach") else np.asarray(support)
    image = np.asarray(rgb).astype(np.float32)/255.
    luminance = image.mean(2)
    chroma = np.linalg.norm(image-luminance[:, :, None], axis=2)
    direction = np.asarray(appearance["RGB_chroma_direction"], np.float32)
    if not support.any():
        return dict(source_binding=binding, status="missing_current_print_search_material", source_masks_are_edge_measurements=False)
    background = support & (luminance >= np.quantile(luminance[support], .85))
    levels = luminance[background]
    center = float(np.median(levels))
    noise = 1.4826*float(np.median(np.abs(levels-center)))
    lower = center-max(3*noise, 2/255.)
    neutral_chroma = float(np.quantile(chroma[background], .95))+2/255.

    def measure(points):
        points = np.asarray(points, np.float32)
        tangent = np.gradient(points, axis=0)
        normal = np.stack((tangent[:, 1], -tangent[:, 0]), 1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        uv = points[:, None]+normal[:, None]*np.array([-7., -5., 5., 7.], np.float32)[None, :, None]
        colors = cv2.remap(image, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
        colors = np.stack((colors[:, :2].mean(1), colors[:, 2:].mean(1)), 1)
        light = colors.mean(2)
        centered = colors-light[:, :, None]
        norms = np.linalg.norm(centered, axis=2)
        cosine = (centered*direction).sum(2)/np.maximum(norms, 1e-9)
        # The linked source material remains chromatic under dark/weak print
        # gradients. Compare its chroma to the opposite observed side rather
        # than imposing the bright-board distribution on both materials.
        print_side = (cosine >= .7) & (norms > 2/255.) & (norms > norms[:, ::-1]+2/255.)
        # The same neutral board can be darker at the other end of a printed
        # panel. Its absolute brightness is not fixed by a bright ROI quantile.
        # Require either that observed background distribution or the local
        # signed bright transition away from the independently linked print.
        local_bright_transition = light-light[:, ::-1] > max(3*noise, 2/255.)
        background_side = ((light >= lower) | local_bright_transition) & (norms <= neutral_chroma)
        supported = (print_side[:, 0] & background_side[:, 1]) | (print_side[:, 1] & background_side[:, 0])
        # A slight warm tint in the actual neutral board is not a foreground
        # material conflict. Preserve the observed background chroma tolerance.
        warm_or_different_material = ((norms > neutral_chroma) & (cosine < 0)).any(1)
        supported &= ~warm_or_different_material
        return dict(supported=supported, current_source_sided_luminance=light,
            current_source_sided_chroma_norm=norms, original_print_direction_cosine=cosine,
            observed_print_material_side=print_side, observed_bright_background_side=background_side,
            observed_local_signed_bright_transition=local_bright_transition,
            source_foreground_or_material_role_conflict=warm_or_different_material)

    return dict(source_binding=binding, status="current_RGB_linked_print_and_background_material_candidate",
        original_print_descriptor_is_search_only=True, current_source_background_luminance_median=center,
        current_source_background_luminance_noise=noise, current_source_background_lower_candidate=lower,
        current_source_background_chroma_upper_candidate=neutral_chroma,
        source_material_rule_is_pixel_ground_truth=False, _current_source_material_support=measure)


def source_print_material_frame_evidence(rgb, binding):
    """Bind unnamed material memory to the exact current image for later reread."""
    image=np.asarray(rgb)
    binding=source_binding(binding,image.shape[:2])
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError("Material frame evidence requires original native uint8 RGB")
    digest=hashlib.sha256(image.tobytes()).hexdigest()
    if binding.get('decoded_native_rgb_sha256',digest)!=digest:
        raise ValueError("Material memory RGB differs from its current source binding")
    # Keep the established print-layout/temporal API spelling; a missing or
    # mismatched canonical RGB key is intentionally a strict rejection.
    return dict(source_binding=binding,source_RGB_sha256=digest,
        status="current_source_material_proposal_without_reference_glyph_identity",
        current_source_material_proposal_identity_unverified=True,
        is_pixel_ground_truth=False,H_is_search_only=True,
        reference_H_available=False,reference_layout_identity_verified=False)


def source_print_layout_evidence(rgb, binding, reference, *, query_features=None,
                                 previous_evidence=None, previous_rgb=None,
                                 source_print_search_mask=None, source_person_search_mask=None):
    """Actual current feature correspondences authenticate a glyph search atlas.

    The atlas is an original reference image warped solely for search/role
    comparison. None of its coordinates becomes an output boundary or label.
    Fresh query matches are required: SAM mask-box motion is not an image H.
    """
    import cv2
    _add_existing_geometry_dependencies()
    from .camera_reuse_localization import descriptors
    from .camera_classification import descriptor_matches
    binding = source_binding(binding, np.asarray(rgb).shape[:2])
    base = dict(source_binding=binding, reference_source_binding=reference["source_binding"],
                source_RGB_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
                H_is_search_only=True, atlas_pixels_are_output_measurements=False,
                status="missing_current_source_glyph_correspondence", query_physical_identity_verified=False)
    ref_path = Path(reference["source_frame_path"])
    if _file_sha256(ref_path) != reference["source_frame_sha256"]:
        raise ValueError("Authenticated print reference PNG changed")
    image = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    ref_rgb = image[:, :, ::-1].copy()
    if hashlib.sha256(ref_rgb.tobytes()).hexdigest() != reference["source_rgb_sha256"]:
        raise ValueError("Print atlas reference differs from original native RGB")
    ref_points = np.asarray(reference["feature_native_uv"], np.float32)
    ref_descriptor = np.asarray(reference["descriptors"], np.float32)
    query = np.asarray(rgb)[:, :, ::-1].copy()
    points, descriptor = query_features or descriptors(query, np.full(query.shape[:2], 255, np.uint8))
    current_valid = None
    if source_print_search_mask is not None:
        raw = source_print_search_mask.detach().cpu().numpy() if hasattr(source_print_search_mask, "detach") else np.asarray(source_print_search_mask)
        if raw.shape != rgb.shape[:2]:raise ValueError("Print navigation must share current native RGB")
        current_valid = cv2.dilate(raw.astype(np.uint8), np.ones((15,15), np.uint8)) > 0
        if source_person_search_mask is not None:
            person = source_person_search_mask.detach().cpu().numpy() if hasattr(source_person_search_mask, "detach") else np.asarray(source_person_search_mask)
            if person.shape != rgb.shape[:2]:raise ValueError("Person navigation must share current native RGB")
            current_valid &= ~person.astype(bool)
        if descriptor is not None:
            xy = np.rint(points).astype(int)
            keep = current_valid[xy[:, 1].clip(0, rgb.shape[0]-1), xy[:, 0].clip(0, rgb.shape[1]-1)]
            points, descriptor = points[keep], descriptor[keep]
            if not len(points):descriptor = None
    if len(ref_descriptor) < 4:
        return base
    pairs = descriptor_matches(ref_descriptor, descriptor) if descriptor is not None else []
    reverse = descriptor_matches(descriptor, ref_descriptor) if descriptor is not None else []
    backward = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse if len(pair) == 2 and
                pair[0].distance < .8*pair[1].distance}
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
            if a.distance < .8*b.distance and backward.get(a.trainIdx) == a.queryIdx]
    H = None
    motion = None
    left = right = np.empty((0, 2), np.float32)
    if len(good) >= 6:
        left, right = ref_points[[m.queryIdx for m in good]], points[[m.trainIdx for m in good]]
        H, keep = cv2.findHomography(left, right, cv2.RANSAC, 3.)
        if H is None or keep is None or int(keep.sum()) < 6:
            H = None
        else:
            left, right = left[keep.ravel().astype(bool)], right[keep.ravel().astype(bool)]
            if len(np.unique(np.rint(right*2).astype(int), axis=0)) < 6 or min(np.ptp(right, axis=0)) < 3:
                H = None
    if H is None and previous_evidence is not None and previous_rgb is not None:
        old = source_binding(previous_evidence["source_binding"], np.asarray(previous_rgb).shape[:2])
        if any(old.get(key) != binding.get(key) for key in
               ("source_sha256", "time_base", "shot_id", "event_id", "rig_id", "installation_id", "native_hw")):
            raise ValueError("Print source memory cannot cross a source or shot binding")
        if hashlib.sha256(np.asarray(previous_rgb).tobytes()).hexdigest() != previous_evidence.get("source_RGB_sha256"):
            raise ValueError("Print source memory RGB changed")
        same_frame(reference["source_binding"], previous_evidence["reference_source_binding"])
        if old["source_pts"] < binding["source_pts"] and previous_evidence.get("H_query_native_from_reference_native"):
            # Reuse the existing native forward/backward LK interface. Actual
            # intervening source features supply a search H; never transport
            # the old print boundary UV or use a SAM-box transform as proof.
            from .camera_context_reuse import temporal_correspondences
            previous_gray = cv2.cvtColor(np.asarray(previous_rgb), cv2.COLOR_RGB2GRAY)
            current_gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
            old_valid = previous_evidence.get("_source_temporal_valid_mask")
            old_distance = previous_evidence.get("_distance_to_reference_ink_search")
            if old_valid is not None or old_distance is not None:
                valid = old_valid.astype(np.uint8)*255 if old_valid is not None else (
                    old_distance <= previous_evidence["reference_layout_search_radius_native_px"]+2).astype(np.uint8)*255
                a, b, audit = temporal_correspondences([previous_gray, current_gray],
                    [valid, current_valid.astype(np.uint8)*255 if current_valid is not None else
                     np.full(current_gray.shape, 255, np.uint8)], 0, 1, min_points=6)
                if a is not None and b is not None:
                    correlations = []
                    for p, q in zip(a, b):
                        pa = cv2.getRectSubPix(previous_gray, (11,11), tuple(p)).astype(float)
                        pb = cv2.getRectSubPix(current_gray, (11,11), tuple(q)).astype(float)
                        correlations.append(float(np.mean((pa-pa.mean())*(pb-pb.mean()))/max(pa.std()*pb.std(),1e-9)))
                    supported = np.asarray(correlations) >= .6
                    a, b = a[supported], b[supported]
                    pair_H, selected = cv2.findHomography(a, b, cv2.RANSAC, 2.5) if len(a) >= 6 else (None, None)
                    if pair_H is not None and selected is not None and selected.sum() >= 6:
                        selected = selected.ravel().astype(bool)
                        a, b = a[selected], b[selected]
                        previous_H = np.asarray(previous_evidence["H_query_native_from_reference_native"], float)
                        left = cv2.perspectiveTransform(a[None].astype(np.float32), np.linalg.inv(previous_H))[0]
                        right, H = b, pair_H@previous_H
                        motion = dict(previous_source_binding=old, current_source_binding=binding,
                            algorithm="existing_native_forward_backward_LK_and_actual_patch_check",
                            existing_tracking_audit=audit, previous_actual_feature_uv=a.tolist(),
                            current_actual_feature_uv=b.tolist(), supported_actual_feature_count=len(b),
                            source_patch_correlations=np.asarray(correlations)[supported][selected].tolist(),
                            actual_print_search_supported_in_both_source_frames=current_valid is not None and old_valid is not None,
                            person_candidate_regions_excluded_from_feature_navigation_only=source_person_search_mask is not None,
                            H_is_search_only=True, source_boundary_UV_transported=False)
    if H is None:
        return {**base, "actual_feature_match_count": len(good),
                "missing_reason": "fresh_reference_match_and_verified_source_memory_unavailable"}
    height, width = query.shape[:2]
    aligned = cv2.warpPerspective(image, H, (width, height))
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = map(int, reference["navigation_roi"])
    source_support = np.zeros(ref_rgb.shape[:2], np.uint8)
    source_support[y0:y1, x0:x1] = 255
    lum = ref_rgb.astype(np.float32).mean(2)/255
    patch = lum[y0:y1, x0:x1]
    background = float(np.quantile(patch, .85))
    chroma = ref_rgb.max(2).astype(np.int16)-ref_rgb.min(2).astype(np.int16)
    original_ink = ((lum < background-.04) | ((chroma > .12*255) & (lum < background-.015))) & (source_support > 0)
    original_border = cv2.morphologyEx(original_ink.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    border_search = cv2.warpPerspective(original_border, H, (width, height), flags=cv2.INTER_NEAREST) > 0
    distance = cv2.distanceTransform((~border_search).astype(np.uint8), cv2.DIST_L2, 5)
    projected = cv2.perspectiveTransform(left[None], H)[0]
    error = np.linalg.norm(projected-right, axis=1)

    def correlations(native_points, size):
        values = []
        for point in native_points:
            a = cv2.getRectSubPix(aligned_gray, (size, size), tuple(map(float, point))).astype(float)
            b = cv2.getRectSubPix(query_gray, (size, size), tuple(map(float, point))).astype(float)
            values.append(float(np.mean((a-a.mean())*(b-b.mean()))/max(a.std()*b.std(), 1e-9)))
        return np.asarray(values, float)

    matches_patch = correlations(right, 11)
    median = float(np.median(matches_patch))
    mad = float(np.median(np.abs(matches_patch-median)))*1.4826
    # Derive a generous lower support limit from independently matched source
    # glyph patches, keeping the measured distribution in the audit. This is
    # a role hypothesis, never a calibrated pixel or visibility bound.
    minimum = max(0., median-max(3*mad, .25))
    material_pair = None
    material_metadata = None
    if reference.get("observed_kind") == "printed_colour_background":
        # A foreground interruption can create a genuine RGB border of the
        # blue material while being no printed border. Compare both sides to
        # the authenticated original layout. The H only locates that search
        # comparison; the current contour and later RGB peak remain separate.
        native = query[:, :, ::-1].astype(np.float32)/255.
        atlas = aligned[:, :, ::-1].astype(np.float32)/255.
        original_patch = ref_rgb[y0:y1, x0:x1].astype(np.float32)/255.
        original_chroma = original_patch-original_patch.mean(2, keepdims=True)
        bright = original_patch.mean(2) >= np.quantile(original_patch.mean(2), .85)
        neutral_norm = np.linalg.norm(original_chroma, axis=2)[bright]
        neutral_median = float(np.median(neutral_norm))
        neutral_noise = 1.4826*float(np.median(np.abs(neutral_norm-neutral_median)))
        chroma_floor = max(2/255., neutral_median+3*neutral_noise)

        def material_pair(native_points):
            points = np.asarray(native_points, np.float32)
            tangent = np.gradient(points, axis=0)
            normal = np.stack((tangent[:, 1], -tangent[:, 0]), 1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
            uv = points[:, None]+normal[:, None]*np.array([-7., -5., 5., 7.], np.float32)[None, :, None]
            actual = cv2.remap(native, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
            expected = cv2.remap(atlas, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR)
            actual = np.stack((actual[:, :2].mean(1), actual[:, 2:].mean(1)), 1)
            expected = np.stack((expected[:, :2].mean(1), expected[:, 2:].mean(1)), 1)
            a, b = actual-actual.mean(2, keepdims=True), expected-expected.mean(2, keepdims=True)
            an, bn = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
            cosine = (a*b).sum(2)/np.maximum(an*bn, 1e-9)
            # Opposing chromatic material is a role conflict, not a reason to
            # erase its optical edge or call the entire detector box occluded.
            conflict = ((an > chroma_floor) & (bn > chroma_floor) & (cosine < 0)).any(1)
            return dict(supported=~conflict, source_sided_chroma_norm=an, reference_sided_chroma_norm=bn,
                        source_reference_sided_chroma_cosine=cosine,
                        opposing_reference_material_role_conflict=conflict)

        material_metadata = dict(reference_source_neutral_chroma_floor=chroma_floor,
            source_material_sides_reexamined=True, sides_are_actual_RGB_not_box_occupancy=True,
            material_comparison_is_role_evidence_not_pixel_ground_truth=True)
    return {**base, "status": "actual_query_glyph_correspondence_search_atlas" if motion is None else
                              "source_temporal_RGB_feature_correspondence_search_atlas",
            "source_temporal_correspondence": motion,
            "actual_query_feature_inlier_count": len(right), "median_image_transfer_error_px": float(np.median(error)),
            "actual_query_pattern_keypoints_uv": right.tolist(), "matched_reference_pattern_keypoints_uv": left.tolist(),
            "H_query_native_from_reference_native": H.tolist(),
            "matched_source_patch_correlations": matches_patch.tolist(),
            "minimum_local_patch_support_correlation": minimum,
            "reference_layout_search_radius_native_px": max(2., 3*float(np.median(error))),
            "_distance_to_reference_ink_search": distance, "_patch_correlation": correlations,
            "_source_material_pair_support": material_pair, "source_material_pair_evidence": material_metadata,
            "_source_temporal_valid_mask": current_valid,
            "atlas_source_original_RGB_sha256": reference["source_rgb_sha256"],
            "is_pixel_ground_truth": False, "occlusion_verified": False}


def verify_source_print_contours(contours, evidence):
    """Split real RGB-ink contours into glyph-supported and role-conflict runs."""
    accepted, conflicts = [], []
    distance = evidence.get("_distance_to_reference_ink_search")
    for contour in contours:
        uv = np.asarray(contour["uv"], float)
        if distance is None:
            keep = np.zeros(len(uv), bool)
            correlations = np.full(len(uv), np.nan)
            nearby = np.zeros(len(uv), bool)
        else:
            xy = np.rint(uv).astype(int)
            nearby = distance[xy[:, 1], xy[:, 0]] <= evidence["reference_layout_search_radius_native_px"]
            correlations = evidence["_patch_correlation"](uv, 11)
            keep = nearby & (correlations >= evidence["minimum_local_patch_support_correlation"])
        reference_keep = keep.copy()
        material = evidence.get("_source_material_pair_support")
        material_evidence = material(uv) if material is not None and len(uv) >= 3 else None
        if material_evidence is not None:
            keep &= material_evidence["supported"]
        current_material = evidence.get("_current_source_material_support")
        current_evidence = current_material(uv) if current_material is not None and len(uv) >= 3 else None
        if current_evidence is not None:
            if not contour.get("internal_hole") or (
                    evidence.get("current_source_material_proposal_identity_unverified") and
                    contour.get("role") == "printed_pattern_internal_material_boundary"):
                # An image H is a search hypothesis, not a pixel label or a
                # veto of a clearly observed linked print-to-board transition.
                keep |= current_evidence["supported"]
            keep &= ~current_evidence["source_foreground_or_material_role_conflict"]
        # Preserve connectivity only inside contiguous original contour runs.
        # A missing/occluding foreground section is never joined across a gap.
        starts = np.r_[0, np.nonzero(np.diff(keep.astype(int)) != 0)[0]+1]
        ends = np.r_[starts[1:], len(keep)]
        for start, end in zip(starts, ends):
            supported = bool(keep[start])
            record = {**contour, "uv": uv[start:end], "closed": bool(contour["closed"] and keep.all()),
                      "original_source_contour_indices": np.arange(start, end),
                      "reference_patch_correlations": correlations[start:end],
                      "reference_layout_neighborhood_supported": nearby[start:end],
                      "source_print_role_supported_by_reference_correspondence": bool(reference_keep[start:end].all()),
                      "source_print_role_supported_by_current_RGB_material": bool(
                          current_evidence is not None and current_evidence["supported"][start:end].all()),
                      "source_original_boundary_role": contour.get("role"),
                      "source_contour_is_output_measurement": False}
            if end-start < 3:
                supported = False
                record["source_print_role_supported_by_reference_correspondence"] = False
                record["short_source_contour_run_retained_without_fabricated_normals"] = True
            if material_evidence is not None:
                record["source_reference_material_pair_evidence"] = {
                    key: value[start:end] for key, value in material_evidence.items()}
            if current_evidence is not None:
                record["current_source_material_boundary_evidence"] = {
                    key: value[start:end] for key, value in current_evidence.items()}
            if not supported:
                record["role"] = "printed_pattern_source_role_conflict"
                record["possible_occlusion_or_missing_print"] = True
                record["solver_outer_boundary_eligible"] = False
            (accepted if supported else conflicts).append(record)
    return accepted, conflicts


def reread_region_edges(rgb, contours, binding, *, device="cuda", search_radius_px=12.,
                        printed_contour_search_radius_px=3.,
                        source_exclusion_mask=None, source_rgb_cache=None) -> list[dict]:
    """Use SAM contours only to search native RGB; return real RGB peak candidates.

    Every edge retains its support/ambiguity flags. None is called an ad outer
    frame or ice contact merely because the corresponding SAM mask was named.
    No learned type probability vetoes an original RGB edge in this operation.
    """
    import torch
    import torch.nn.functional as F
    from .vision_geometry import measure_boundary_profiles
    cache = (_checked_source_RGB_tensors(rgb, source_rgb_cache, binding=binding, device=device)
             if source_rgb_cache is not None else source_RGB_tensors(rgb, binding, device=device))
    image = cache['image']
    # A neutral unused typed channel is required by the reusable RGB profiler.
    # It is not a learned semantic prediction; only rgb_peak is consumed below.
    neutral = cache['neutral_probability']
    height, width = image.shape[-2:]
    readings = []
    with torch.inference_mode():
        for contour in contours:
            points = torch.as_tensor(contour["uv"], device=device, dtype=torch.float32)
            if len(points) < 3:
                continue
            tangent = torch.roll(points, -1, 0)-torch.roll(points, 1, 0)
            if not contour["closed"]:
                tangent[0], tangent[-1] = points[1]-points[0], points[-1]-points[-2]
            normals = torch.stack((-tangent[:, 1], tangent[:, 0]), 1)
            good_normals = torch.linalg.vector_norm(normals, dim=1) > 1e-6
            original_indices = torch.nonzero(good_normals).flatten()
            if not len(original_indices):
                continue
            printed = contour.get("role", "").startswith("printed_pattern_")
            material_support = contour.get("current_source_material_boundary_evidence", {}).get("supported")
            supported_material = torch.zeros(len(original_indices), dtype=torch.bool, device=device)
            if material_support is not None and not contour.get("internal_hole"):
                supported_material = torch.as_tensor(material_support, device=device).bool()[original_indices]
            profile_radius = (search_radius_px if bool(supported_material.any()) else
                              printed_contour_search_radius_px) if printed else search_radius_px
            profiles = measure_boundary_profiles(
                image, neutral, points[original_indices], normals[original_indices],
                class_index=0, search_radius=profile_radius, sample_step=.5,
                rgb_source_binding=binding, probability_source_binding=binding,
                projection_source_binding=binding,
                segment_ids=torch.zeros(len(original_indices), device=device, dtype=torch.long),
                minimum_contrast=.025, maximum_edge_width=4.)
            peak = profiles["rgb_peak"]
            flags = dict(profiles["flags"])
            transition_intervals = signed_peak_pool = None
            if printed:
                # These contours came from the source dark/chromatic ink mask,
                # not a projected region. Both external and hole contour
                # orientations point from ink into its brighter complement.
                # Opposite transitions on the other side of a narrow letter
                # stroke are a different true edge, not a competing reading
                # of this source contour. Keep the unsigned audit separately.
                signed = (profiles["rgb_profile"][:, 1:].mean(-1)-
                          profiles["rgb_profile"][:, :-1].mean(-1))/.5
                gradient = profiles["rgb_gradient_profile"]
                usable = profiles["rgb_gradient_valid_mask"] & (signed >= .025/4.)
                score = torch.where(usable, gradient, -torch.inf)
                maximum, index = score.max(1)
                has_values = torch.isfinite(maximum)
                position = torch.arange(score.shape[1], device=image.device)[None]
                previous = F.pad(score[:, :-1], (1, 0), value=-float("inf"))
                following = F.pad(score[:, 1:], (0, 1), value=-float("inf"))
                local = usable & (score >= previous) & (score > following)
                half = usable & (score >= maximum[:, None]*.5)
                lo = torch.where((position < index[:, None]) & ~half, position, -1).amax(1)
                hi = torch.where((position > index[:, None]) & ~half, position, score.shape[1]).amin(1)
                # Quantized local maxima inside one connected signed gradient
                # transition are one observed material interval, not separate
                # physical borders. A distinct peak across a lower valley
                # remains a competing boundary; no contour is filled in.
                # FWHM valleys can occur inside one blurred but monotonic
                # source RGB step. Connectivity uses its actual signed
                # lightening samples, rather than a fraction of the largest
                # peak. Flat or reversing material separates true steps.
                signed_support = profiles["rgb_gradient_valid_mask"] & (signed > 0)
                transition_lo = torch.where((position < index[:, None]) & ~signed_support, position, -1).amax(1)
                transition_hi = torch.where((position > index[:, None]) & ~signed_support, position, score.shape[1]).amin(1)
                same_transition = (position > transition_lo[:, None]) & (position < transition_hi[:, None])
                competing = (local & ~same_transition & ((position-index[:, None]).abs()*.5 > 1.) &
                             (score >= maximum[:, None]*.6)).sum(1)
                transition_width = (hi-lo-1).float()*.5
                offsets = profiles["rgb_gradient_offsets_native_px"][index]
                uv = profiles["points_native_uv"]+profiles["normals_unit"]*offsets[:, None]
                sampled_offsets = profiles["rgb_gradient_offsets_native_px"]
                interval_offsets = torch.stack((sampled_offsets[(lo+1).clamp(0, score.shape[1]-1)]-.25,
                                                sampled_offsets[(hi-1).clamp(0, score.shape[1]-1)]+.25), 1)
                transition_intervals = dict(normal_interval_native_px=interval_offsets,
                    same_material_signed_transition_normal_interval_native_px=torch.stack((
                        sampled_offsets[(transition_lo+1).clamp(0, score.shape[1]-1)]-.25,
                        sampled_offsets[(transition_hi-1).clamp(0, score.shape[1]-1)]+.25),1),
                    actual_RGB_interval_uv=profiles["points_native_uv"][:, None]+
                        profiles["normals_unit"][:, None]*interval_offsets[:, :, None],
                    source_indices=original_indices, interval_profile_clipped=(lo < 0) | (hi >= score.shape[1]),
                    interval_is_exact_physical_boundary=False)
                pool_row, pool_column = torch.nonzero(local, as_tuple=True)
                signed_peak_pool = dict(source_indices=original_indices[pool_row],
                    actual_source_peak_uv=profiles["points_native_uv"][pool_row]+
                        profiles["normals_unit"][pool_row]*sampled_offsets[pool_column, None],
                    actual_signed_gradient=signed[pool_row, pool_column],
                    RGB_gradient=gradient[pool_row, pool_column],
                    peak_sample_indices=pool_column, physical_role_accepted=False,
                    all_observed_local_signed_source_peaks_retained=True)
                flags.update(no_rgb_contrast=~has_values | (peak["contrast"] < .025),
                             multi_edge_ambiguity=competing > 0, blur_review=transition_width > 4.,
                             peak_at_search_boundary=(index == 0) | (index == score.shape[1]-1),
                             print_outward_lightening_transition_absent=~has_values)
                retained_wide = flags["blur_review"] & supported_material & (competing == 0)
                flags["wide_source_material_RGB_interval_retained"] = retained_wide
                candidate = (has_values & ~flags["point_out_of_frame"] & ~flags["partial_profile"] &
                             ~flags["no_rgb_contrast"] & ~flags["multi_edge_ambiguity"] &
                             (~flags["blur_review"] | retained_wide) & ~flags["peak_at_search_boundary"])
                peak = {**peak, "uv": uv, "gradient": maximum, "fwhm_px": transition_width,
                        "candidate_evidence_usable": candidate}
            valid = peak["candidate_evidence_usable"].clone()
            occluded = torch.zeros_like(valid)
            if source_exclusion_mask is not None:
                excluded = torch.as_tensor(source_exclusion_mask, device=device).bool()
                if excluded.shape != (height, width):
                    raise ValueError("Occlusion candidate must share native RGB dimensions")
                uv = peak["uv"]
                grid = torch.stack((2*(uv[:, 0]+.5)/width-1, 2*(uv[:, 1]+.5)/height-1), -1)[None, :, None]
                occluded = F.grid_sample(excluded[None, None].float(), grid,
                                         mode="nearest", align_corners=False)[0, 0, :, 0] > .5
                valid &= ~occluded
            selected = torch.nonzero(valid).flatten()
            indices, actual = original_indices[selected], peak["uv"][selected]
            runs = []
            ids = indices.cpu().tolist()
            start = 0
            for end in range(1, len(ids)+1):
                if end == len(ids) or ids[end] != ids[end-1]+1:
                    runs.append(dict(actual_source_uv=actual[start:end],
                                     source_indices=indices[start:end], closed=False))
                    start = end
            readings.append(dict(
                source_binding=dict(binding), actual_source_uv=actual, source_indices=indices,
                measured_runs=runs, candidate_source_uv=peak["uv"], candidate_usable=valid,
                flags={**flags, "occluder_region_candidate": occluded},
                original_unsigned_profile_flags=profiles["flags"] if printed else None,
                source_print_outward_transition_used=printed,
                observed_RGB_transition_intervals=transition_intervals,
                observed_RGB_signed_peak_pool=signed_peak_pool,
                search_radius_native_px=profile_radius,
                support_fwhm_px=peak["fwhm_px"], source_gradient=peak["gradient"],
                boundary_role=contour.get("role", "region_outline_RGB_edge_candidate"),
                internal_hole=bool(contour.get("internal_hole", False)),
                print_material_scope=contour.get("print_material_scope"),
                source_reference_component_scope=contour.get('source_reference_component_scope'),
                whole_advertisement_outer_boundary_eligible=contour.get('whole_advertisement_outer_boundary_eligible',True),
                selected_material_hole=contour.get("selected_material_hole",False),
                internal_hole_scope=contour.get("internal_hole_scope"),
                whole_printed_pattern_hole_verified=contour.get("whole_printed_pattern_hole_verified",False),
                solver_outer_boundary_eligible=contour.get("solver_outer_boundary_eligible",False),
                source_component_id=contour.get("source_component_id"), physical_boundary_verified=False,
                mask_outline_used_as_measurement=False, model_or_camera_projection_used=False,
                typed_probability_used=False, prompt_pixels_used_as_measurements=False,
                is_training_label=False, boundary_accuracy_accepted=False))
    return readings


def _validate_material_search_alias(prompt, parent, binding):
    search_id = prompt.get('sam_search_object_id')
    if isinstance(search_id, bool) or not isinstance(search_id, (int, np.integer)):
        raise ValueError('SAM material search aliases require an integer object ID')
    if not parent:
        raise ValueError('A material search alias requires an existing same-source parent')
    for key in ('source_sha256', 'time_base', 'shot_id', 'event_id', 'rig_id', 'installation_id', 'native_hw'):
        if parent['source_binding'].get(key) != binding.get(key):
            raise ValueError('Shared material search cannot cross original source or shot')
    if prompt.get('shared_SAM_board_search_only'):
        if (parent['kind'] != 'board_region' or not prompt.get('source_material_proposal_unverified')
                or prompt.get('physical_id')):
            raise ValueError('Shared board navigation cannot establish a material physical identity')
    elif (not prompt.get('source_material_proposal_unverified')
            or not parent.get('source_material_proposal_unverified')
            or prompt.get('physical_id') or parent.get('physical_id')
            or prompt.get('source_material_search_group_id') != parent.get('source_material_search_group_id')):
        raise ValueError('Only same-source unnamed material layers may share a SAM search object')


class Sam2SourceInstances:
    """Sequential original RGB -> full SAM2 decoder/memory object candidates.

    ``read(rgb, binding, prompts=(), reset=False)`` accepts point/box prompts
    in original pixels and keeps distinct numeric object IDs. No instance is
    silently merged on visual resemblance. Source/shot changes reset memory.
    ``region_mask`` stays on CUDA; callers decide which artifacts to save.
    """
    def __init__(self, checkpoint_path, *, device="cuda", contour_stride=4,
                 search_radius_px=12., printed_contour_search_radius_px=3., reread_rgb=True):
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAM2 object reading requires the configured CUDA GPU")
        checkpoint_path = Path(checkpoint_path).resolve()
        weights = checkpoint_path / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.model_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.model, loading = Sam2VideoModel.from_pretrained(
            str(checkpoint_path), local_files_only=True, output_loading_info=True)
        failures = {key: loading.get(key) for key in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
        if failures:
            raise RuntimeError(f"Full local SAM2 decoder/memory failed strict load: {failures}")
        self.model.requires_grad_(False).eval().to(self.device, dtype=torch.bfloat16)
        self.processor = Sam2VideoProcessor.from_pretrained(str(checkpoint_path), local_files_only=True)
        self.checkpoint_path = str(checkpoint_path)
        self.contour_stride, self.search_radius_px = contour_stride, search_radius_px
        self.printed_contour_search_radius_px = float(printed_contour_search_radius_px)
        self.reread_rgb = bool(reread_rgb)
        self.reset()

    def reset(self):
        import torch
        self.session = self.processor.init_video_session(
            inference_device=self.device, inference_state_device=self.device,
            video_storage_device="cpu", max_vision_features_cache_size=1, dtype=torch.bfloat16)
        self.objects = {}
        self.previous_binding = self.previous_rgb_sha256 = self.previous_masks = None
        self.previous_rgb = None
        self.print_layout_memory = {}
        self.print_layout_RGB_memory = {}
        self.current_RGB_tensor_cache = None
        self.frame_index = -1

    def source_RGB_profile_cache(self, rgb, binding):
        """Reuse only this exact native exposure's unmodified RGB tensors."""
        normalized = source_binding(binding, np.asarray(rgb).shape[:2])
        cache = self.current_RGB_tensor_cache
        if cache is not None:
            try:
                same_frame(normalized, cache['source_binding'])
            except ValueError:
                cache = None
        if cache is None:
            cache = source_RGB_tensors(rgb, normalized, device=self.device)
            self.current_RGB_tensor_cache = cache
        return _checked_source_RGB_tensors(rgb, cache, binding=normalized, device=self.device)

    def bootstrap_regions(self, rgb, binding, *, class_probabilities=None, classes=(),
                          probabilities_conditioned_on_geometry=False, person_boxes=(),
                          overlay_boxes=(), reset=True):
        """Automatic two-stage regional initialization, with every failure exposed."""
        proposal = automatic_source_region_prompts(
            rgb, binding, class_probabilities=class_probabilities, classes=classes,
            probabilities_conditioned_on_geometry=probabilities_conditioned_on_geometry,
            person_boxes=person_boxes, overlay_boxes=overlay_boxes)
        if not reset:
            # Reidentify a lost region at this same native frame without
            # deleting separately named ads or restarting the video index.
            available = {}
            for object_id, prompt in self.objects.items():
                available.setdefault(prompt["kind"], []).append(object_id)
            next_id = max(self.objects, default=999)+1
            for prompt in proposal["prompts"]:
                candidates = available.get(prompt["kind"], [])
                if candidates:
                    prompt["object_id"] = candidates.pop(0)
                else:
                    prompt["object_id"] = next_id
                    next_id += 1
        result = self.read(rgb, binding, prompts=proposal["prompts"], reset=reset)
        ice = next((row for row in result["instances"] if row["kind"] == "ice_region"), None)
        if ice is not None:
            board = board_prompt_from_ice_region(
                rgb, result["source_binding"], ice["region_mask"],
                top_hypotheses=proposal["boundary_topology_hypotheses"],
                excluded_source_boxes=[row["box_xyxy"] for row in person_boxes])
            if board is not None:
                # Refresh the same board object, preserving every other memory.
                existing = next((row for row in result["instances"] if row["kind"] == "board_region"), None)
                if existing is not None:
                    board["object_id"] = existing["object_id"]
                else:
                    board["object_id"] = max(self.objects, default=1000)+1
                result = self.read(rgb, binding, prompts=[board])
        result["bootstrap_source_proposals"] = proposal
        result["class_presence"] = {kind: dict(
            present=any(row["kind"] == kind and bool(row["region_mask"].any()) for row in result["instances"]),
            state="candidate_only" if any(row["kind"] == kind for row in result["instances"]) else
                  "missing_source_region_seed") for kind in ("ice_region", "board_region", "occluder", "broadcast_overlay")}
        return result

    def read_auto(self, rgb, binding, *, class_probabilities=None, classes=(),
                  probabilities_conditioned_on_geometry=False, person_boxes=(), overlay_boxes=(), reset=False):
        """Bootstrap at a cut; propagate memory; reidentify actually lost regions."""
        normalized = source_binding(binding, np.asarray(rgb).shape[:2])
        old = self.previous_binding
        cut = old and any(old.get(key) != normalized.get(key) for key in
                          ("source_sha256", "time_base", "shot_id", "event_id", "rig_id", "installation_id", "native_hw"))
        if reset or cut or not self.objects:
            return self.bootstrap_regions(
                rgb, normalized, class_probabilities=class_probabilities, classes=classes,
                probabilities_conditioned_on_geometry=probabilities_conditioned_on_geometry,
                person_boxes=person_boxes, overlay_boxes=overlay_boxes)
        result = self.read(rgb, normalized)
        tracked = [row for row in result["instances"] if row["kind"] in ("ice_region", "board_region")]
        if tracked and any(not bool(row["region_mask"].any()) for row in tracked):
            frame_index = self.frame_index
            result = self.bootstrap_regions(
                rgb, normalized, class_probabilities=class_probabilities, classes=classes,
                probabilities_conditioned_on_geometry=probabilities_conditioned_on_geometry,
                person_boxes=person_boxes, overlay_boxes=overlay_boxes, reset=False)
            if self.frame_index != frame_index:
                raise RuntimeError("Same source region reidentification advanced the SAM frame index")
            result["memory_update"] = "source_reidentification_after_region_lost"
        refresh = source_person_refresh_prompts(normalized, person_boxes, result["instances"], self.objects)
        if refresh["prompts"]:
            frame_index = self.frame_index
            proposal = result.get("bootstrap_source_proposals")
            result = self.read(rgb, normalized, prompts=refresh["prompts"], reset=False)
            if self.frame_index != frame_index:
                raise RuntimeError("Same source person refresh advanced the SAM frame index")
            if proposal is not None:
                result["bootstrap_source_proposals"] = proposal
        result["source_person_reidentification"] = refresh
        return result

    def _prune_architecture_memory(self):
        """Keep the architecture's attention horizon, not all streamed images.

        Conditioning memories persist. Nonconditioning frames outside both
        memory attention and object-pointer horizons cannot be read by this
        forward-only interface. This is an implementation cache policy, not a
        user task duration/capacity limit.
        """
        # This installed Transformers SAM2 implementation uses stride=1
        # (modeling_sam2_video._select_memory_frames), not a configurable field.
        stride = 1
        horizon = max(int(self.model.config.num_maskmem)*stride+2,
                      int(self.model.config.max_object_pointers_in_encoder)+2)
        before = self.frame_index-horizon
        if self.session.processed_frames:
            # Streaming forward never reads an old source frame again.
            self.session.processed_frames = {self.frame_index: self.session.processed_frames[self.frame_index]}
        for index, outputs in self.session.output_dict_per_obj.items():
            noncond = outputs["non_cond_frame_outputs"]
            for stamp in list(noncond):
                if stamp < before:
                    del noncond[stamp]
            tracked = self.session.frames_tracked_per_obj[index]
            for stamp in list(tracked):
                if stamp < before:
                    del tracked[stamp]

    def read(self, rgb, binding, *, prompts=(), reset=False, material_search_navigation=None):
        import torch
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Supply unchanged decoded original uint8 RGB")
        binding = source_binding(binding, rgb.shape[:2])
        material_navigation = None
        if material_search_navigation is not None:
            same_frame(binding, material_search_navigation['source_binding'])
            material_navigation = np.asarray(material_search_navigation['search_mask'])
            if (material_navigation.dtype != bool or material_navigation.shape != rgb.shape[:2]
                    or material_search_navigation.get('RGB_hard_veto_allowed')):
                raise ValueError('Material refinement requires current native navigation, never an RGB veto')
            if material_search_navigation.get('RGB_sha256', hashlib.sha256(rgb.tobytes()).hexdigest()) != hashlib.sha256(rgb.tobytes()).hexdigest():
                raise ValueError('Material refinement navigation differs from the current source RGB')
        old = self.previous_binding
        cut = old and any(old.get(key) != binding.get(key) for key in
                          ("source_sha256", "time_base", "shot_id", "event_id", "rig_id", "installation_id", "native_hw"))
        if reset or cut:
            self.reset()
            old = None
        if old and binding["source_pts"] < old["source_pts"]:
            raise ValueError("Reverse-time source reading needs an explicit reset")
        prepared = [validate_prompt(prompt, binding) for prompt in prompts]
        had_prepared_prompts = bool(prepared)
        if len({prompt["object_id"] for prompt in prepared}) != len(prepared):
            raise ValueError("Each frame prompt object ID must be distinct")
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        same_stamp = bool(old and binding["source_pts"] == old["source_pts"])
        if same_stamp and rgb_sha != self.previous_rgb_sha256:
            raise ValueError("Different original RGB was supplied for the same source frame")
        repeated = bool(old and binding.get("canonical_exposure_pts") is not None and
                        binding.get("exposure_proof_sha256") and
                        binding.get("exposure_proof_sha256") == old.get("exposure_proof_sha256") and
                        binding["canonical_exposure_pts"] == old.get("canonical_exposure_pts"))
        # Virtual material layers consume an already processed board search.
        # No encoder, decoder or fictitious temporal update is needed to add
        # such bookkeeping at the same native exposure.
        if same_stamp and prepared and all(p.get('sam_search_object_id',p['object_id']) != p['object_id'] for p in prepared):
            for prompt in prepared:
                prompt = merge_prompt_identity(self.objects.get(prompt['object_id']), prompt)
                _validate_material_search_alias(prompt, self.objects.get(prompt['sam_search_object_id']), binding)
                self.objects[prompt['object_id']] = prompt
            prepared = []
        reuse = (same_stamp or repeated) and not prepared
        if not reuse:
            if not same_stamp:
                self.frame_index += 1
            pending_ids = list(self.session.obj_with_new_inputs)
            for prompt in prepared:
                prior = self.objects.get(prompt["object_id"])
                prompt = merge_prompt_identity(prior, prompt)
                search_id=prompt.get('sam_search_object_id',prompt['object_id'])
                if isinstance(search_id,bool) or not isinstance(search_id,(int,np.integer)):
                    raise ValueError('SAM material search aliases require an integer object ID')
                if search_id!=prompt['object_id']:
                    parent=self.objects.get(search_id)
                    _validate_material_search_alias(prompt, parent, binding)
                    self.objects[prompt['object_id']]=prompt
                    continue
                self.objects[prompt['object_id']]=prompt
                kwargs = dict(inference_session=self.session, frame_idx=self.frame_index,
                              obj_ids=prompt["object_id"], original_size=tuple(rgb.shape[:2]))
                if prompt.get("points_uv") is not None:
                    kwargs.update(input_points=[[prompt["points_uv"]]], input_labels=[[prompt["point_labels"]]])
                if prompt.get("box_xyxy") is not None:
                    kwargs["input_boxes"] = [[prompt["box_xyxy"]]]
                self.processor.add_inputs_to_inference_session(**kwargs)
                if prompt["object_id"] not in pending_ids:
                    pending_ids.append(prompt["object_id"])
            # The installed processor assigns (rather than appends) this public
            # session queue on every call. Retain all separately prompted objects.
            self.session.obj_with_new_inputs = pending_ids
            if not self.objects:
                self.previous_binding, self.previous_rgb_sha256 = binding, rgb_sha
                return dict(source_binding=binding, instances=[], status="missing_source_prompts",
                            missing_reason="No source-bound region seeds supplied; no object identity invented",
                            model_sha256=self.model_sha256, memory_update="shot_reset", precision_accepted=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = self.processor(images=rgb.copy(), device=str(self.device),
                                         return_tensors="pt")["pixel_values"].to(self.device)
                output = self.model(inference_session=self.session, frame=encoded, frame_idx=self.frame_index)
                masks = self.processor.post_process_masks([output.pred_masks],
                            original_sizes=[list(rgb.shape[:2])], binarize=True)[0]
            self.previous_masks = {object_id: masks[index, 0].bool()
                                   for index, object_id in enumerate(output.object_ids)}
            self._prune_architecture_memory()
        masks = self.previous_masks or {}
        exclusion = torch.zeros(rgb.shape[:2], device=self.device, dtype=torch.bool)
        reviewed_exclusion = torch.zeros_like(exclusion)
        for object_id, mask in masks.items():
            if self.objects[object_id]["kind"] == "occluder":
                exclusion |= mask
                authority = self.objects[object_id].get("exclusion_authority") or {}
                if (authority.get("kind") in ("direct_user_confirmation", "source_visual_review")
                        and authority.get("reference")):
                    reviewed_exclusion |= mask
        instances = []
        current_print_memory_RGB = rgb.copy()
        region_geometry = {}
        pattern_region_support = {}
        RGB_tensor_cache = self.source_RGB_profile_cache(rgb, binding)
        query_features = None
        if any(prompt.get("source_pattern_reference", {}).get("observed_kind") in
               ("printed_ink_components", "printed_colour_background")
               for prompt in self.objects.values()):
            import cv2
            _add_existing_geometry_dependencies()
            from .camera_reuse_localization import descriptors
            query_features = descriptors(rgb[:, :, ::-1].copy(), np.full(rgb.shape[:2], 255, np.uint8))
        for object_id, prompt in self.objects.items():
            search_id = prompt.get('sam_search_object_id',object_id)
            mask = masks.get(search_id, torch.zeros_like(exclusion))
            if search_id not in region_geometry:
                yx = torch.nonzero(mask)
                current_region_box = None
                if len(yx):
                    lo, hi = yx.amin(0), yx.amax(0)
                    current_region_box = [int(lo[1]), int(lo[0]), int(hi[1])+1, int(hi[0])+1]
                region_geometry[search_id] = (current_region_box, region_contours(mask, stride=self.contour_stride))
            current_region_box, contours = region_geometry[search_id]
            if prompt.get("box_xyxy") and current_region_box and not prompt.get("initial_region_box"):
                prompt["initial_region_box"] = current_region_box
            pattern = None
            layout_evidence = None
            conflict_contours = []
            boundary_contours = contours
            raw_print_contours = []
            component_continuity = None
            definition = prompt.get("boundary_definition")
            deferred_material = bool(prompt.get('shared_SAM_board_search_only') and material_navigation is None)
            if definition == "printed_pattern_external_contour" and not deferred_material:
                pattern_region = mask
                if prompt.get('shared_SAM_board_search_only'):
                    # The board mask is a shared navigation hypothesis. The
                    # current source-derived wide domain retains visible print
                    # where that mask is missing or uncertain.
                    pattern_region = mask | torch.as_tensor(material_navigation, device=self.device)
                support_key = (search_id, bool(prompt.get('shared_SAM_board_search_only')))
                if support_key not in pattern_region_support:
                    import torch.nn.functional as F
                    pattern_region_support[support_key] = F.max_pool2d(pattern_region.float()[None,None],
                        25, stride=1, padding=12)[0,0] > .5
                search_box = prompt.get("box_xyxy")
                search_points = prompt.get("actual_pattern_keypoints_uv", [])
                if (search_box and current_region_box and prompt.get("initial_region_box") and
                        prompt["source_binding"]["source_pts"] != binding["source_pts"]):
                    search_box = transport_search_box(search_box, prompt["initial_region_box"], current_region_box)
                    if len(search_points):
                        old_box = np.asarray(prompt["initial_region_box"], float)
                        new_box = np.asarray(current_region_box, float)
                        search_points = (new_box[:2]+(np.asarray(search_points)-old_box[:2])/
                                         (old_box[2:]-old_box[:2])*(new_box[2:]-new_box[:2])).tolist()
                search_box=current_material_region_search_box(prompt,search_box,current_region_box)
                pattern = printed_pattern_candidates(
                    rgb, pattern_region, source_box=search_box,
                    source_print_appearance=prompt.get("source_print_appearance"),
                    search_pattern_keypoints_uv=search_points,
                    device=self.device, source_rgb_cache=RGB_tensor_cache,
                    region_search_support=pattern_region_support[support_key])
                cached = self.print_layout_memory.get(object_id)
                cached_rgb = getattr(self, 'print_layout_RGB_memory', {}).get(object_id)
                if cached is not None and cached_rgb is None:
                    # Compatibility for an in-memory caller created before
                    # paired object snapshots existed. Never pair stale
                    # evidence with this frame merely to bypass the check.
                    if (self.previous_rgb is None or hashlib.sha256(self.previous_rgb.tobytes()).hexdigest()
                            != cached.get('source_RGB_sha256')):
                        raise ValueError('Print object evidence is missing its authenticated paired RGB')
                    cached_rgb = self.previous_rgb
                if (prompt.get("source_print_appearance") or {}).get("dominant_chromatic_print_background"):
                    component_continuity = source_print_component_continuity(rgb,binding,pattern,
                        previous_evidence=cached,previous_rgb=cached_rgb,source_person_search_mask=exclusion)
                    tracked = component_continuity["current_source_feature_uv"]
                    overlap_points = component_continuity.get("current_RGB_component_overlap_search_points_uv")
                    if overlap_points is not None and len(overlap_points):
                        tracked=np.concatenate((tracked,overlap_points))
                    if len(tracked):
                        xy=np.rint(tracked).astype(int)
                        candidate_person=exclusion.detach().cpu().numpy()
                        inside=(xy[:,0]>=0)&(xy[:,0]<rgb.shape[1])&(xy[:,1]>=0)&(xy[:,1]<rgb.shape[0])
                        picked=np.flatnonzero(inside)
                        inside[picked]&=~candidate_person[xy[picked,1],xy[picked,0]]
                        tracked=tracked[inside]
                        component_continuity["current_source_feature_uv_used_for_material_search"] = tracked
                        if len(tracked):
                            pattern = printed_pattern_candidates(rgb,pattern_region,source_box=search_box,
                                source_print_appearance=prompt.get("source_print_appearance"),
                                search_pattern_keypoints_uv=np.concatenate((np.asarray(search_points,float).reshape(-1,2),tracked)),
                                device=self.device,source_rgb_cache=RGB_tensor_cache,
                                region_search_support=pattern_region_support[support_key])
                boundary_contours = source_print_boundary_contours(pattern,stride=self.contour_stride,
                    appearance=prompt.get("source_print_appearance"))
                reference_scope = prompt.get('source_pattern_reference', {}).get('source_component_scope')
                if reference_scope:
                    for contour in boundary_contours:
                        contour['source_reference_component_scope'] = reference_scope
                        contour['whole_advertisement_outer_boundary_eligible'] = prompt['source_pattern_reference'].get(
                            'whole_advertisement_outer_boundary_eligible', True)
                        if not contour['whole_advertisement_outer_boundary_eligible']:
                            contour['solver_outer_boundary_eligible'] = False
                raw_print_contours = boundary_contours
                reference = prompt.get("source_pattern_reference")
                if reference and reference.get("observed_kind") in ("printed_ink_components", "printed_colour_background"):
                    if cached is not None and cached["source_binding"]["source_pts"] == binding["source_pts"]:
                        # Same original-frame prompt refinement must not
                        # create a fictitious extra temporal step.
                        layout_evidence = cached
                    else:
                        layout_evidence = source_print_layout_evidence(rgb, binding, reference,
                            query_features=query_features, previous_evidence=cached, previous_rgb=cached_rgb,
                            source_print_search_mask=pattern.get("source_chromatic_material_candidate_mask") if
                                pattern.get("source_chromatic_material_candidate_mask") is not None else pattern["ink_mask"],
                            source_person_search_mask=exclusion)
                if (prompt.get("source_print_appearance") or {}).get("dominant_chromatic_print_background"):
                    if layout_evidence is None:
                        layout_evidence=source_print_material_frame_evidence(rgb,binding)
                    material = source_colour_material_evidence(rgb, binding, pattern, prompt["source_print_appearance"])
                    layout_evidence = {**layout_evidence, "current_source_colour_material_evidence": {
                        key:value for key,value in material.items() if not key.startswith("_")},
                        "_current_source_material_support": material.get("_current_source_material_support")}
                if layout_evidence is not None:
                    import cv2
                    # Same-frame refinement can change object search masks,
                    # but does not advance the atlas/temporal correspondence.
                    navigation = cv2.dilate(pattern["ink_mask"].detach().cpu().numpy().astype(np.uint8),
                        np.ones((15,15),np.uint8)) > 0
                    navigation &= ~exclusion.detach().cpu().numpy()
                    layout_evidence["_source_temporal_valid_mask"] = navigation
                    layout_evidence["_source_component_material_mask"] = pattern["ink_mask"].detach().cpu().numpy().copy()
                    layout_evidence["_source_person_search_mask"] = exclusion.detach().cpu().numpy().copy()
                    if component_continuity is not None:
                        layout_evidence["source_material_instance_continuity"] = component_continuity
                        layout_evidence["_current_actual_component_search_points_uv"] = component_continuity.get(
                            "current_source_feature_uv_used_for_material_search",component_continuity["current_source_feature_uv"])
                        layout_evidence["_current_source_feature_uv_only"] = component_continuity["current_source_feature_uv"]
                        layout_evidence["_current_RGB_component_overlap_search_points_uv"] = component_continuity.get(
                            "current_RGB_component_overlap_search_points_uv",np.empty((0,2),np.float32))
                    self.print_layout_memory[object_id] = layout_evidence
                    if not hasattr(self,'print_layout_RGB_memory'):
                        self.print_layout_RGB_memory = {}
                    self.print_layout_RGB_memory[object_id] = current_print_memory_RGB
                    boundary_contours, conflict_contours = verify_source_print_contours(boundary_contours, layout_evidence)
            readings = (reread_region_edges(rgb, boundary_contours, binding, device=self.device,
                        search_radius_px=self.search_radius_px,
                        printed_contour_search_radius_px=self.printed_contour_search_radius_px,
                        source_exclusion_mask=reviewed_exclusion if prompt["kind"] != "occluder" else None,
                        source_rgb_cache=RGB_tensor_cache)
                        if self.reread_rgb and not deferred_material else [])
            conflict_readings = (reread_region_edges(rgb, conflict_contours, binding, device=self.device,
                search_radius_px=self.search_radius_px,
                printed_contour_search_radius_px=self.printed_contour_search_radius_px,source_rgb_cache=RGB_tensor_cache)
                if self.reread_rgb and conflict_contours else [])
            raw_print_readings = (reread_region_edges(rgb, raw_print_contours, binding, device=self.device,
                search_radius_px=self.search_radius_px,
                printed_contour_search_radius_px=self.printed_contour_search_radius_px,source_rgb_cache=RGB_tensor_cache)
                if self.reread_rgb and raw_print_contours else [])
            measured = sum(len(reading["actual_source_uv"]) for reading in readings)
            status = ("awaiting_current_source_material_navigation" if deferred_material else
                      "missing_region_candidate" if not bool(mask.any()) and not pattern else
                      "RGB_edge_candidates_physical_role_unverified" if measured else
                      "region_only_no_unambiguous_RGB_boundary")
            instances.append(dict(
                object_id=object_id, kind=prompt["kind"], physical_id=prompt["physical_id"],
                expected_id=prompt["expected_id"], identity_authority=prompt["identity_authority"],
                expected_identity_authority=prompt.get("expected_identity_authority"),
                candidate_physical_ids=prompt.get("candidate_physical_ids", []),
                content_id=prompt.get("content_id"), source_component_id=prompt.get("source_component_id"),
                source_material_proposal_unverified=bool(prompt.get("source_material_proposal_unverified")),
                sam_search_object_id=prompt.get('sam_search_object_id',object_id),
                source_material_search_group_id=prompt.get('source_material_search_group_id'),
                shared_SAM_search_is_physical_identity_verification=False,
                shared_SAM_board_search_only=bool(prompt.get('shared_SAM_board_search_only')),
                current_source_material_navigation_present=material_navigation is not None,
                material_RGB_read_deferred_until_current_navigation=deferred_material,
                source_pattern_seed_kind=prompt.get("source_pattern_seed_kind"),
                identity_confirmed_by_source_human=prompt["identity_confirmed_by_source_human"],
                identity_confirmed_by_source_correspondence=bool(prompt.get("identity_confirmed_by_source_correspondence")),
                prompt_origin=prompt["origin"], prompt_source_binding=prompt["source_binding"],
                prompt_target_role=prompt.get("prompt_target_role", "region_role_unverified"),
                source_observation_provenance=prompt.get("source_observation_provenance"),
                boundary_definition=definition,
                boundary_definition_authority=prompt.get("boundary_definition_authority"),
                source_binding=dict(binding), status=status,
                missing_reason=None if measured else "SAM region absent or RGB boundary weak/ambiguous/clipped",
                region_mask=mask, region_contours=contours, boundary_readings=readings,
                printed_pattern_contours=boundary_contours if pattern is not None else [],
                source_print_unfiltered_contours=raw_print_contours,
                source_print_raw_boundary_readings=raw_print_readings,
                source_print_unfiltered_contours_are_measured_UV=False,
                source_print_role_conflict_contours=conflict_contours,
                source_print_role_conflict_readings=conflict_readings,
                source_print_layout_evidence={k: v for k, v in layout_evidence.items() if not k.startswith("_")} if layout_evidence else None,
                source_print_visibility_status="current_RGB_material_identity_and_print_relation_unverified" if
                    prompt.get('source_material_proposal_unverified') else
                    "source_glyph_parts_conflict_possible_occlusion_or_missing" if conflict_contours else
                                               "visible_source_glyph_candidates_not_precision_accepted",
                printed_pattern_extent=pattern["extent"] if pattern is not None else None,
                source_pattern_material_component_links=pattern.get("source_pattern_material_component_links", []) if pattern else [],
                source_material_instance_continuity=component_continuity,
                source_chromatic_material_candidate_mask=pattern.get("source_chromatic_material_candidate_mask") if pattern else None,
                printed_pattern_extent_is_edge=False,
                unlinked_source_foreground_candidate_mask=pattern.get("unlinked_source_foreground_candidate_mask") if pattern else None,
                printed_pattern_complete=False,
                candidate_only=True, mask_is_pixel_ground_truth=False,
                advertisement_outer_frame_verified=False,
                missing_edges_fabricated=False, appearance_based_identity_merging=False))
        self.previous_binding, self.previous_rgb_sha256 = binding, rgb_sha
        self.previous_rgb = current_print_memory_RGB
        return dict(source_binding=binding, instances=instances, status="source_region_candidates",
                    occluder_region_candidate_mask=exclusion,
                    reviewed_source_exclusion_mask=reviewed_exclusion,
                    unreviewed_occluder_predictions_used_as_RGB_veto=False,
                    model_sha256=self.model_sha256, full_decoder_and_video_memory_used=True,
                    SAM_forward_executed_this_call=not reuse,
                    material_layer_search_parent_count=len({p.get('sam_search_object_id',i) for i,p in self.objects.items()
                        if p.get('source_material_proposal_unverified')}),
                    memory_update="reused_verified_exposure" if repeated and reuse else
                                  "source_frame_prompt_refinement" if same_stamp and (had_prepared_prompts or material_navigation is not None) else
                                  "reused_source_frame" if same_stamp and reuse else
                                  "shot_reset" if old is None else "SAM2_object_memory_propagated",
                    no_model_grid_shown=True, world_or_camera_optimized=False, precision_accepted=False)


def _read_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _add_existing_geometry_dependencies():
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()  # Append existing dependencies; keep active Torch/NumPy precedence.


def _existing_renderer_geometry():
    _add_existing_geometry_dependencies()
    import sys
    renderer_directory = Path(__file__).resolve().parents[2]/"calibration/program_tracking"
    if str(renderer_directory) not in sys.path:
        # The reusable renderer's adaptive module retains script-style imports.
        sys.path.append(str(renderer_directory))
    from calibration.program_tracking import render_result
    return render_result


def named_scene_search_paths(scene):
    """Attach names to the existing renderer's exact finite geometry inventory.

    No grid/standard-reference geometry is added. Lane open edges stay open;
    advertisement producer retains only actual modeled sides and print paths.
    """
    renderer = _existing_renderer_geometry()
    paths = list(renderer.fixed_scene_paths(scene, draw_grid=False))
    descriptors = [dict(id="ice_board_contact", kind="ice_contact", primitive="free_curve",
                        physical_role="ice_board_contact"),
                   dict(id="board_top", kind="board_top", primitive="free_curve",
                        physical_role="board_top_edge")]
    for feature in scene["ice_features"]:
        if feature["kind"] == "circle":
            for ring in feature.get("rings", [{"role": "outer"}]):
                if ring["role"] == "outer":
                    descriptors.append(dict(id=f"{feature['id']}:outer", feature_id=feature["id"],
                                            kind="ice_paint", primitive="circle"))
        elif feature["kind"] == "polyline":
            for index, _ in enumerate(feature["segments_xyz"]):
                descriptors.append(dict(id=f"{feature['id']}:segment{index}", feature_id=feature["id"],
                                        kind="ice_paint", primitive="line"))
    for feature in scene["ice_features"]:
        if feature["kind"] == "polygon":
            omitted = set(feature.get("open_edges", []))
            if not omitted:
                descriptors.append(dict(id=f"{feature['id']}:boundary", feature_id=feature["id"],
                                        kind="ice_paint", primitive="free_curve"))
            else:
                for index in range(len(feature["polygon_xyz"])):
                    if index not in omitted:
                        descriptors.append(dict(id=f"{feature['id']}:edge{index}", feature_id=feature["id"],
                                                kind="ice_paint", primitive="line"))
        elif feature["kind"] == "circle":
            for ring in feature.get("rings", []):
                if ring["role"] != "outer":
                    descriptors.append(dict(id=f"{feature['id']}:{ring['role']}", feature_id=feature["id"],
                                            kind="ice_paint", primitive="circle"))
    if len(paths) != len(descriptors):
        raise ValueError("Named inventory no longer matches the reusable scene renderer")
    result = [{**descriptor, "world_xyz": xyz, "renderer_kind": renderer_kind}
              for descriptor, (renderer_kind, xyz) in zip(descriptors, paths)]
    for path in renderer.advertisement_review_paths(scene):
        number, role = int(path["number"]), path["role"]
        circle = "circle_outer_stroke" in role
        result.append(dict(id=f"Ad{number}:{role}", feature_id=f"Ad{number}", advertisement_number=number,
                           kind="ad_boundary", primitive="circle" if circle else "free_curve",
                           physical_role="printed_stroke_outer_boundary" if circle else "advertisement_visible_boundary",
                           world_xyz=np.asarray(path["xyz"], float),
                           extent_scope=path.get("extent"), renderer_kind="actual_advertisement_review_path"))
    return result


def projected_search_samples(projected, native_hw, *, spacing_px=8., margin_px=48.):
    """Clip/densify engineering hints only; never produce source measurements."""
    clip_segments = _existing_renderer_geometry().clip_segments
    if spacing_px <= 0 or margin_px < 0:
        raise ValueError("Search sampling units must be positive native pixels")
    height, width = native_hw
    points, segments = [], []
    segment_id = -1
    previous = None
    for run in projected.segments:
        for first, second in zip(run[:-1], run[1:]):
            a, b, good = clip_segments(projected.pixels[[first]]+margin_px,
                                       projected.pixels[[second]]+margin_px,
                                       width+int(2*margin_px), height+int(2*margin_px))
            if not bool(good[0]):
                previous = None
                continue
            a, b = a[0]-margin_px, b[0]-margin_px
            connected = previous is not None and np.linalg.norm(a-previous) < 1e-5
            if not connected:
                segment_id += 1
            samples = np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing_px))+1))
            if connected:
                samples = samples[1:]
            points.extend(samples)
            segments.extend([segment_id]*len(samples))
            previous = b
        previous = None
    return np.asarray(points, float).reshape(-1, 2), np.asarray(segments, np.int64)


class SceneGuideSearchAdapter:
    """Exact existing camera PTS -> named geometry-neighborhood RGB search hints.

    Frozen parameters are authenticated and read-only. Missing source/camera
    states produce explicit missing output; neither closest timestamps nor a
    different programme's camera are substituted. Search guides contain no
    ``actual_source_uv`` and cannot be used as training/calibration targets.
    """
    def __init__(self, scene_path, camera_path, *, role_contract_path, event_id,
                 advertisement_catalog_path=None, source_shots_path=None,
                 expected_scene_sha256=None, expected_camera_sha256=None,
                 search_radius_px=48., sample_spacing_px=8., identity_authorities=None,
                 exposure_registry=None):
        import json
        _add_existing_geometry_dependencies()
        from calibration.program_tracking.npz_readonly import mapped_array
        self.scene_path, self.camera_path = map(lambda value: Path(value).resolve(), (scene_path, camera_path))
        self.scene_sha256, self.camera_sha256 = map(_file_sha256, (self.scene_path, self.camera_path))
        for actual, expected, label in ((self.scene_sha256, expected_scene_sha256, "scene"),
                                         (self.camera_sha256, expected_camera_sha256, "camera")):
            if expected is not None and actual != expected:
                raise ValueError(f"Frozen {label} changed")
        self.scene = _read_json(self.scene_path)
        self.event_id = event_id
        self.search_radius_px, self.sample_spacing_px = float(search_radius_px), float(sample_spacing_px)
        self.array_routes = {}
        def array(name):
            try:
                value = mapped_array(self.camera_path, name)
                self.array_routes[name] = "existing_npz_readonly_mapped_array"
            except ValueError:
                # This existing camera export is compressed (~0.8MB), unlike
                # complete solver checkpoints. Load it once, forbid writes.
                with np.load(self.camera_path, allow_pickle=False) as archive:
                    value = archive[name]
                value.flags.writeable = False
                self.array_routes[name] = "immutable_compressed_NPZ_member_loaded_once"
            return value
        self.pts, self.cameras, self.representatives = array("pts"), array("camera25"), array("exposure_representative_pts")
        self.installations = array("installation_id")
        self.metadata = json.loads(str(array("metadata_json").item()))
        if (self.pts.ndim != 1 or np.any(np.diff(self.pts) <= 0) or
                self.cameras.shape != (len(self.pts), 25) or not np.isfinite(self.cameras).all()):
            raise ValueError("Existing camera export does not have exact ordered source PTS")
        self.source_sha256 = self.metadata["source_sha256"]
        self.time_base = str(Fraction(self.metadata["source_time_base"]))
        self.lookup = {int(stamp): index for index, stamp in enumerate(self.pts)}
        self.role_contract_path = Path(role_contract_path).resolve()
        roles = _read_json(self.role_contract_path)
        if roles.get("scene_sha256") != self.scene_sha256:
            raise ValueError("Source paint role contract is bound to a different scene")
        self.roles = {row["curve_id"]: row for row in roles["curve_records"]}
        self.inventory = named_scene_search_paths(self.scene)
        self.identity_authorities = dict(identity_authorities or {})
        self.catalog_path, self.catalog = None, None
        if advertisement_catalog_path is not None:
            self.catalog_path = Path(advertisement_catalog_path).resolve()
            self.catalog = _read_json(self.catalog_path)
            self.catalog_numbers = {int(row["number"]) for row in self.catalog["positions"]}
        self.source_shots_path = Path(source_shots_path).resolve() if source_shots_path else None
        self.source_shots = _read_json(self.source_shots_path) if self.source_shots_path else None
        if self.source_shots and self.source_shots["source_sha256_from_timeline"] != self.source_sha256:
            raise ValueError("Camera shot inventory belongs to another source video")
        # A source-specific registry owns any verified exposure alias. Other
        # events/sources use native PTS without loading this event's proof.
        from .vision import confirmed_exposure_groups
        self.exposure_groups, self.exposure_proof = confirmed_exposure_groups(
            self.source_sha256, self.time_base, registry=exposure_registry)
        if self.exposure_proof["source_sha256"] != self.source_sha256:
            raise ValueError("Camera and existing exposure proof belong to different videos")
        if str(Fraction(self.exposure_proof.get("time_base", self.time_base))) != self.time_base:
            raise ValueError("Camera and existing exposure proof have different native clocks")
        for stamp, representative in zip(self.pts, self.representatives):
            stamp, representative = int(stamp), int(representative)
            proof = self.exposure_groups.get(stamp)
            if ((proof is None and representative != stamp) or
                    (proof is not None and representative != proof["canonical"])):
                raise ValueError("Saved camera exposure alias differs from the verified original-pixel proof")
            other = self.lookup.get(representative)
            if other is None or not np.array_equal(self.cameras[self.lookup[stamp]], self.cameras[other]):
                raise ValueError("Saved cameras differ within one confirmed original exposure")

    @classmethod
    def from_okada_review(cls, project_root, *, role_contract_path, **kwargs):
        root = Path(project_root).resolve()
        base = root/"events/ankara_2026/programmes/okada_mei_sp/models/review_20261005"
        frozen = _read_json(base/"FROZEN_INPUTS_UNACCEPTED.json")
        kwargs.setdefault("exposure_registry", root/"events/ankara_2026/model_constraints/source_clock_registry.json")
        return cls(base/"frozen_shared_radius_scene_4075_UNACCEPTED.json",
                   base/"complete_physical_camera_model_UNACCEPTED_917c204dafde4027.npz",
                   role_contract_path=role_contract_path, event_id="ankara_2026",
                   advertisement_catalog_path=root/"configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json",
                   source_shots_path=root/"calibration/program_tracking/source_shots.json",
                   expected_scene_sha256=frozen["scene"]["sha256"],
                   expected_camera_sha256=frozen["camera"]["sha256"],
                   **kwargs)

    def guides_for(self, binding):
        from .vision_geometry import CameraShapeAdapter, source_annotations_to_guides
        binding = source_binding(binding)
        base = dict(source_binding=binding, event_id=self.event_id, guides=[], unresolved=[],
                    model_curve_inventory_count=len(self.inventory), projected_pixels_are_search_only=True,
                    pixel_training_labels_generated=False, world_or_camera_optimized=False,
                    accuracy_accepted=False, scene_sha256=self.scene_sha256, camera_sha256=self.camera_sha256)
        base["source_clock_evidence"] = dict(self.exposure_proof)
        if binding["source_sha256"] != self.source_sha256 or binding["time_base"] != self.time_base:
            return {**base, "status": "missing_camera_source_binding", "missing_reason": "Different source/clock; saved camera not reused"}
        index = self.lookup.get(binding["source_pts"])
        if index is None:
            return {**base, "status": "missing_exact_camera_PTS", "missing_reason": "No saved camera for this frame; no nearest-PTS/other-programme substitution"}
        proof = self.exposure_groups.get(binding["source_pts"])
        representative = int(self.representatives[index])
        if binding.get("canonical_exposure_pts", representative) != representative:
            raise ValueError("Caller exposure alias contradicts existing original-pixel proof")
        binding.update(canonical_exposure_pts=representative, installation_id=str(self.installations[index]))
        if proof is not None:
            binding.update(official_exposure_pts=proof["official_pts"],
                           official_exposure_time_base=proof["official_time_base"],
                           exposure_proof_sha256=self.exposure_proof["proof_sha256"])
        if "native_hw" not in binding:
            raise ValueError("Native RGB dimensions are required for source search projection")
        camera = self.cameras[index]
        adapter = CameraShapeAdapter()
        guides, unresolved, inventory = [], [], []
        for path in self.inventory:
            projection = adapter.project(path["world_xyz"], camera, geometry_id=path["id"],
                model_revision=self.scene_sha256, source_binding=binding, camera_source_binding=binding,
                native_hw=binding["native_hw"])
            uv, segments = projected_search_samples(projection, binding["native_hw"],
                spacing_px=self.sample_spacing_px, margin_px=self.search_radius_px)
            inventory.append(dict(id=path["id"], kind=path["kind"], search_sample_count=len(uv),
                                  visible_search_support=bool(len(uv)), identity_source="event_named_geometry"))
            if not len(uv):
                continue
            definition = self.roles.get(path["id"], {})
            authority = self.identity_authorities.get(path["id"])
            if authority is None and definition.get("user_confirmed_visual_supervision_present"):
                authority = dict(kind="direct_user_confirmation", reference=str(self.role_contract_path),
                                 source_curve_id=path["id"], scope="event_physical_identity_not_this_frame_pixels")
            number = path.get("advertisement_number")
            if number is not None and self.catalog and number in self.catalog_numbers:
                authority = dict(kind="human_confirmed_event_catalog", reference=str(self.catalog_path),
                                 physical_number=number, scope="same_event_physical_group_and_perimeter_order",
                                 same_content_pairs_are_distinct_physical_instances=True)
            annotation = dict(id=path["id"], feature_id=path["id"], group_id=path["id"],
                              kind=path["kind"], primitive=path["primitive"], uv=uv, segment_ids=segments,
                              source_binding=binding, state="model_projection_search_hint",
                              search_radius_px=self.search_radius_px, source_identity_verified=False,
                              identity_authority=authority)
            if path.get("physical_role"):
                annotation["physical_role"] = path["physical_role"]
            packet = source_annotations_to_guides([annotation], source_binding=binding,
                annotation_source_binding=binding, role_definitions=self.roles)
            unresolved.extend(packet["unresolved"])
            if packet["guides"]:
                guide = packet["guides"][0]
            else:
                # Preserve a known model curve whose physical raster role is
                # unresolved, without pretending that every circle is a disc.
                guide = dict(id=path["id"], kind=path["kind"], primitive=path["primitive"],
                             points_uv=uv, segment_ids=segments, physical_role="source_paint_role_unresolved",
                             identity_confirmed=False, identity_authority=authority,
                             source_binding=binding, search_radius_px=self.search_radius_px)
            guide.update(origin="model_projection_search_hint", expected_id=path["id"],
                         source_group_id=path["id"], display_label=path.get("feature_id", path["id"]),
                         model_search_projection=projection.metadata,
                         guide_pixels_are_source_measurements=False,
                         source_role_definition=definition, advertisement_number=number,
                         event_perimeter_identity_binding=number is not None,
                         event_identity_confirmed=bool(authority),
                         source_pixel_to_expected_identity_verified=False,
                         training_supervision_from_projection=False)
            guides.append(guide)
        return {**base, "source_binding": binding, "status": "exact_saved_camera_search_guides",
                "guides": guides, "unresolved": unresolved, "search_inventory": inventory,
                "camera25": camera.copy(), "camera_source_binding": binding,
                "array_read_routes": dict(self.array_routes)}


def advertisement_identity_from_constraints(candidate_numbers, matched_source_uv, binding, *,
                                           model_search=None, tracked_identities=(), source_identity_anchors=()):
    """Resolve appearance aliases only with separately bound spatial evidence.

    Permanent catalogue identity is an event fact, not a current-frame pixel
    assertion. Exact model neighborhoods may reduce expected IDs, but retain
    their candidate authority. A reviewed source anchor or a previously source-
    verified video object may establish the transferred physical identity.
    """
    binding = source_binding(binding)
    candidates = set(map(int, candidate_numbers))
    points = np.asarray(matched_source_uv, float).reshape(-1, 2)
    evidence, verified = [], []
    for row in (*tracked_identities, *source_identity_anchors):
        same_frame(binding, row["source_binding"])
        authority = row.get("identity_authority", {})
        if not (row.get("source_pixel_to_expected_identity_verified") and
                authority.get("kind") in HUMAN_IDENTITIES and authority.get("reference")):
            continue
        number = int(str(row["physical_id"]).removeprefix("Ad"))
        if number not in candidates:
            continue
        box = np.asarray(row["box_xyxy"], float)
        inside = ((points >= box[:2]) & (points <= box[2:])).all(1)
        if len(points) and inside.mean() >= .7:
            verified.append((number, authority))
            evidence.append(dict(kind="source_verified_instance_support", physical_number=number,
                                 actual_source_keypoint_count=int(inside.sum()), authority=authority))
    verified_numbers = {number for number, _ in verified}
    if len(verified_numbers) == 1:
        number = verified[0][0]
        return dict(physical_id=f"Ad{number}", expected_id=f"Ad{number}", candidate_numbers=[number],
                    identity_status="source_verified_instance_correspondence",
                    source_pixel_to_expected_identity_verified=True,
                    identity_authority=verified[0][1], constraint_evidence=evidence)
    if len(verified_numbers) > 1:
        return dict(physical_id=None, expected_id=None, candidate_numbers=sorted(verified_numbers),
                    identity_status="conflicting_source_identity_authorities",
                    source_pixel_to_expected_identity_verified=False, identity_authority=None,
                    constraint_evidence=evidence)
    if model_search is not None:
        same_frame(binding, model_search["source_binding"])
        if model_search.get("status") == "exact_saved_camera_search_guides":
            neighborhoods = set()
            for guide in model_search["guides"]:
                number = guide.get("advertisement_number")
                if number not in candidates:
                    continue
                uv = np.asarray(guide["points_uv"], float).reshape(-1, 2)
                radius = float(guide["search_radius_px"])
                lo, hi = uv.min(0)-radius, uv.max(0)+radius
                inside = ((points >= lo) & (points <= hi)).all(1)
                if len(points) and inside.mean() >= .7:
                    neighborhoods.add(number)
                    evidence.append(dict(kind="exact_saved_camera_search_neighborhood",
                                         expected_number=number, actual_source_keypoint_count=int(inside.sum()),
                                         scene_sha256=model_search["scene_sha256"],
                                         camera_sha256=model_search["camera_sha256"], is_pixel_identity_proof=False))
            if len(neighborhoods) == 1:
                candidates = neighborhoods
    expected = f"Ad{next(iter(candidates))}" if len(candidates) == 1 else None
    return dict(physical_id=None, expected_id=expected, candidate_numbers=sorted(candidates),
                identity_status="model_neighborhood_instance_candidate" if expected else "same_content_physical_instances_ambiguous",
                source_pixel_to_expected_identity_verified=False, identity_authority=None,
                constraint_evidence=evidence)


def advertisement_descriptor_search_mask(rgb, binding, *, source_search=None,
                                         person_boxes=(), overlay_boxes=()):
    """Restrict descriptor navigation, never measured ink or physical identity.

    A current board/print support may come from an independently source-bound
    region proposal. Detector boxes remove descriptor centres only. In
    particular, neither the box nor a SAM region may delete RGB edge evidence.
    Empty support remains an explicit search failure, rather than silently
    searching an unrelated wall. The usual broadcast UI exclusion is reused
    from the existing event classifier's 426x240 feature basis.
    """
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Descriptor search requires actual native RGB uint8")
    binding = source_binding(binding, image.shape[:2])
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if binding.get("decoded_native_rgb_sha256", digest) != digest:
        raise ValueError("Descriptor navigation RGB differs from its source binding")
    h, w = image.shape[:2]
    valid = np.full((h, w), 255, np.uint8)
    sources = []
    if source_search is not None:
        same_frame(binding, source_search["source_binding"])
        if source_search["source_binding"].get("decoded_native_rgb_sha256", digest) != digest:
            raise ValueError("Descriptor region navigation RGB differs from the current source")
        candidate = np.asarray(source_search["search_mask"])
        if candidate.shape != (h, w) or candidate.dtype != bool:
            raise ValueError("Descriptor region navigation requires a native bool search mask")
        valid[~candidate] = 0
        sources.append(dict(kind=source_search.get("origin", "current_source_region_candidate"),
                            source_binding=source_search["source_binding"],
                            search_mask_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
                            mask_is_pixel_ground_truth=False))
    # This is an existing classifier search convention, not detected overlay
    # geometry or a claim that every broadcast has the same UI layout.
    valid[:int(h*57/240), :int(w*145/426)] = 0
    observations = []
    for kind, records in (("person", person_boxes), ("overlay", overlay_boxes)):
        for observation in records:
            same_frame(binding, observation["source_binding"])
            if observation["source_binding"].get("decoded_native_rgb_sha256", digest) != digest:
                raise ValueError("Descriptor box navigation RGB differs from the current source")
            box = np.asarray(observation["box_xyxy"], float)
            if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
                raise ValueError("Descriptor navigation requires a finite ordered source box")
            lo = np.maximum(np.floor(box[:2]).astype(int), 0)
            hi = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
            valid[lo[1]:hi[1], lo[0]:hi[0]] = 0
            observations.append(dict(kind=kind, source_binding=observation["source_binding"],
                                     box_xyxy=box.tolist(), provenance=observation.get("provenance"),
                                     box_used_as_RGB_veto=False))
    evidence = dict(source_binding=binding, source_rgb_sha256=digest,
        status="source_descriptor_search_support" if valid.any() else "empty_source_descriptor_search_support",
        allowed_descriptor_center_pixels=int(np.count_nonzero(valid)),
        region_navigation_sources=sources, source_box_navigation=observations,
        broadcast_UI_search_convention=dict(reference="existing_event_classifier_426x240_feature_basis",
                                           candidate_only=True),
        search_mask_is_boundary_measurement=False, RGB_measurements_erased=False,
        physical_identity_confirmed_by_navigation=False)
    return valid, evidence


def source_print_material_search_proposals(rgb, binding, *, print_parents=(), board_search=None,
                                          person_boxes=(), overlay_boxes=(), first_object_id=300000,
                                          device="cpu"):
    """Current RGB material -> unnamed print search objects, independently of SIFT.

    Existing source-bound print parents navigate a broad board neighborhood;
    their boxes are never returned as measured contours. A supplied current
    board search region can replace that neighborhood. The palette and every
    component/seed come from the current RGB, not an advertisement number or
    a reference homography. Small/unlinked regions remain a proposal ledger.
    """
    import cv2
    image = np.asarray(rgb)
    binding = source_binding(binding, image.shape[:2])
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Print material search requires native uint8 RGB")
    boxes, parents = [], []
    for row in print_parents:
        same_frame(binding, row["source_binding"])
        if row["source_binding"].get("decoded_native_rgb_sha256", digest) != digest:
            raise ValueError("Print parent search RGB differs from the current source")
        box = row.get("printed_pattern_extent") or row.get("box_xyxy")
        if box is None:
            continue
        box = np.asarray(box, float)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
            raise ValueError("Current print parent requires an ordered native search extent")
        boxes.append(box)
        parents.append(dict(object_id=row.get("object_id"), source_binding=row["source_binding"],
            search_extent=box.tolist(), extent_is_edge=False,
            origin="current_print_parent_neighborhood_search"))
    h, w = image.shape[:2]
    base = dict(source_binding=binding, source_rgb_sha256=digest, prompts=[], material_candidates=[],
                rejected_component_ledger=[], parent_search_evidence=parents,
                source_test_manual_UV_loaded=False, physical_identity_inferred=False,
                source_boxes_are_boundaries=False, source_material_pixels_are_measured_boundary_UV=False,
                world_projection_used=False, sparse_ad_class_used_as_hard_domain=False)
    parent_core_navigation=None
    navigation=None
    independently_supplied_board_navigation=board_search is not None
    if boxes:
        array = np.stack(boxes)
        centers = (array[:, 0]+array[:, 2])/2
        height = float(np.median(array[:, 3]-array[:, 1]))
        order = np.argsort(centers)
        # This extends a search neighborhood along the observed board. It is
        # expressly not a fitted rail, advertisement edge or camera model.
        x = np.arange(w)
        lower = np.interp(x, centers[order], array[order, 1])
        upper = np.interp(x, centers[order], array[order, 3])
        if len(array) > 1 and np.ptp(centers) > 1:
            slope0=(array[order[-1],1]-array[order[0],1])/np.ptp(centers)
            slope1=(array[order[-1],3]-array[order[0],3])/np.ptp(centers)
            left,right=x<centers[order[0]],x>centers[order[-1]]
            lower[left]=array[order[0],1]+slope0*(x[left]-centers[order[0]])
            lower[right]=array[order[-1],1]+slope0*(x[right]-centers[order[-1]])
            upper[left]=array[order[0],3]+slope1*(x[left]-centers[order[0]])
            upper[right]=array[order[-1],3]+slope1*(x[right]-centers[order[-1]])
        y=np.arange(h)[:,None]
        navigation=(y>=lower[None]-height*.5)&(y<=upper[None]+height*.5)
        parent_core_navigation=(y>=lower[None]-height*.1)&(y<=upper[None]+height*.1)
        if independently_supplied_board_navigation:
            # A left print parent must not veto a source-observed slanted
            # board farther right. Parent ranges fill search gaps only.
            parent_core_navigation=None
    if board_search is None:
        if navigation is None:
            return {**base, "status": "missing_current_source_print_or_board_navigation"}
        board_search=dict(source_binding=binding,search_mask=navigation,
            origin="current_print_parent_extended_neighborhood_navigation_only")
    elif navigation is not None:
        # A missing source boundary column is uncertainty, not an RGB veto.
        # Only that uncertain search can fall back to current print neighbors.
        board_search=dict(board_search)
        candidate=np.asarray(board_search['search_mask'])
        if candidate.shape!=(h,w) or candidate.dtype!=bool:
            raise ValueError('Current board navigation requires a native bool mask')
        candidate=candidate.copy()
        fallback=sorted(set(board_search.get('missing_source_support_columns',())) |
                        set(board_search.get('uncertain_search_columns',())))
        if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0 or x>=w for x in fallback):
            raise ValueError('Source board uncertainty columns must be native integer columns')
        if fallback:candidate[:,fallback]|=navigation[:,fallback]
        board_search['search_mask']=candidate
        base['board_missing_or_uncertain_parent_fallback_columns']=fallback
        base['parent_fallback_is_measurement']=False
    valid, search_evidence = advertisement_descriptor_search_mask(image,binding,
        source_search=board_search,person_boxes=person_boxes,overlay_boxes=overlay_boxes)
    support=valid>0
    base["source_search_evidence"]=search_evidence
    base['effective_source_material_search_navigation'] = {
        **board_search, 'RGB_sha256': digest,
        'origin': 'current_board_search_and_print_parent_uncertainty_navigation_only',
        'mask_is_navigation_only': True, 'RGB_hard_veto_allowed': False,
        'parent_missing_or_uncertain_fallback_columns': base.get('board_missing_or_uncertain_parent_fallback_columns', [])}
    if not support.any():
        return {**base, "status": "empty_current_source_print_navigation"}
    # Reuse the same source-ink/chromatic component reader. Initial palette
    # peaks are search seeds; the existing reader can follow weakly chromatic
    # portions of their connected material at its source quantization floor.
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    strong=support&(hsv[:,:,1]>46)&(hsv[:,:,2]>25)
    bins=hsv[:,:,0]//10
    histogram=np.bincount(bins[strong],minlength=18)
    minimum_area=max(64,int(np.count_nonzero(support)*.0005))
    peaks=[i for i,count in enumerate(histogram) if count>=minimum_area and
           count>=histogram[(i-1)%18] and count>=histogram[(i+1)%18]]
    seen=np.zeros((h,w),bool)
    source_cache=source_RGB_tensors(image,binding,device=device)
    # The native bool dilation is the same 25-pixel search support used by
    # every palette pass. Compute it once instead of one full-image max pool
    # per colour. It is not a measured material mask.
    dilated_support=cv2.dilate(support.astype(np.uint8),np.ones((25,25),np.uint8))>0
    import torch
    region_support=torch.as_tensor(dilated_support,device=device)
    background=torch.quantile(source_cache['luminance'][region_support],.85)
    object_id=int(first_object_id)
    for index in peaks:
        samples=image[strong&(bins==index)].astype(np.float32)/255.
        centered=samples-samples.mean(1,keepdims=True)
        direction=centered.mean(0);direction/=max(float(np.linalg.norm(direction)),1e-9)
        appearance=dict(dominant_chromatic_print_background=True,RGB_chroma_direction=direction.tolist(),
            colour_is_search_descriptor_not_pixel_truth=True,
            source_material_authority_scope="current_RGB_material_candidate_not_reference_identity",
            descriptor_source_binding=binding,descriptor_source_rgb_sha256=digest)
        pattern=printed_pattern_candidates(image,support,source_print_appearance=appearance,device=device,
            source_rgb_cache=source_cache,region_search_support=region_support,background_luminance=background)
        material=pattern['source_chromatic_material_candidate_mask']
        material=material.detach().cpu().numpy() if hasattr(material,'detach') else np.asarray(material)
        count, labels, stats, centers_cc=cv2.connectedComponentsWithStats(material.astype(np.uint8),connectivity=8)
        for label in range(1,count):
            x0,y0,width,height_cc,area=map(int,stats[label])
            candidate=labels==label
            duplicate=float(np.count_nonzero(candidate&seen))/max(area,1)
            reason="small_current_source_material_component" if area<minimum_area else (
                "same_current_material_component_already_proposed" if duplicate>.7 else None)
            if reason is None and parent_core_navigation is not None:
                central_support=float(np.count_nonzero(candidate&parent_core_navigation))/max(area,1)
                if central_support<.5:
                    reason='source_material_outside_current_print_parent_neighborhood'
            if reason is None and width>12*height_cc:
                reason='long_thin_source_material_band_not_new_print_object_seed'
            occupancy=float(area)/max(width*height_cc,1)
            if reason is None and occupancy<.2:
                reason='sparse_branching_source_material_not_coherent_print_search_support'
            if reason is not None:
                base['rejected_component_ledger'].append(dict(palette_hue_bin=index,component_label=label,
                    original_RGB_candidate_pixels=area,reason=reason,RGB_evidence_erased=False))
                continue
            # Suppress duplicate navigation of a parent already being read,
            # retaining its material in the existing object's own raw ledger.
            parent_overlap=[]
            for parent in boxes:
                ax,ay,bx,by=map(int,parent)
                overlap=np.count_nonzero(candidate[max(0,ay):min(h,by),max(0,ax):min(w,bx)])/max(area,1)
                parent_overlap.append(float(overlap))
            if parent_overlap and max(parent_overlap)>.7:
                base['rejected_component_ledger'].append(dict(palette_hue_bin=index,component_label=label,
                    original_RGB_candidate_pixels=area,reason='existing_print_parent_material_search',
                    parent_search_extent_overlap=parent_overlap,RGB_evidence_erased=False))
                continue
            seen|=candidate
            distance=cv2.distanceTransform(candidate.astype(np.uint8),cv2.DIST_L2,5)
            yy,xx=np.unravel_index(int(distance.argmax()),distance.shape)
            margin=max(4,int(min(width,height_cc)*.08))
            box=[max(0,x0-margin),max(0,y0-margin),min(w,x0+width+margin),min(h,y0+height_cc+margin)]
            local_id=f"source_material:{binding['source_pts']}:{index}:{label}"
            prompt=dict(object_id=object_id,kind='advertisement_panel',origin='original_rgb_candidate',
                source_binding=binding,box_xyxy=box,points_uv=[[float(xx),float(yy)]],point_labels=[1],
                physical_id=None,expected_id=None,content_id=None,candidate_physical_ids=[],
                source_component_id=local_id,actual_pattern_keypoints_uv=[[float(xx),float(yy)]],
                source_pattern_seed_kind='actual_current_RGB_material_interior_search_not_SIFT_correspondence',
                source_print_appearance=appearance,boundary_definition='printed_pattern_external_contour',
                boundary_definition_authority='user_confirmed_printed_outer_contours_holes_extent_separate',
                whole_pattern_extent_is_edge=False,internal_holes_have_separate_role=True,
                source_material_proposal_unverified=True)
            base['prompts'].append(prompt)
            base['material_candidates'].append(dict(source_component_id=local_id,object_id=object_id,
                source_binding=binding,source_rgb_sha256=digest,palette_hue_bin=index,source_material_candidate_mask=candidate,
                original_RGB_candidate_pixels=area,search_box_xyxy=box,
                source_material_component_extent_occupancy=occupancy,
                search_point_native_uv=[float(xx),float(yy)],physical_id=None,
                source_material_candidate_is_print_identity_proof=False,search_extent_is_measured_edge=False))
            object_id+=1
    grouped=group_source_material_prompt_search(base['prompts'],base['material_candidates'],binding)
    base['prompts']=grouped['prompts'];base['source_material_search_groups']=grouped['groups']
    return {**base,"status":"current_source_material_search_proposals" if base['prompts'] else
                    "no_current_source_material_component_proposal"}


def group_source_material_prompt_search(prompts, candidates, binding):
    """Share overlapping current-material search, retaining independent RGB layers."""
    by_id={int(x['object_id']):dict(x) for x in prompts}
    materials={int(x['object_id']):x for x in candidates}
    ids=sorted(by_id)
    groups=[];visited=set()
    for start in ids:
        if start in visited:continue
        pending=[start];members=[];visited.add(start)
        while pending:
            index=pending.pop();members.append(index)
            same_frame(binding,by_id[index]['source_binding'])
            same_frame(binding,materials[index]['source_binding'])
            if not by_id[index].get('source_material_proposal_unverified') or by_id[index].get('physical_id'):
                raise ValueError('Material search grouping cannot merge confirmed physical objects')
            a=np.asarray(materials[index]['source_material_candidate_mask'],bool)
            for other in ids:
                if other in visited:continue
                b=np.asarray(materials[other]['source_material_candidate_mask'],bool)
                if a.shape!=b.shape or a.shape!=tuple(binding['native_hw']):
                    raise ValueError('Material grouping requires the same native source dimensions')
                overlap=np.count_nonzero(a&b)/max(min(np.count_nonzero(a),np.count_nonzero(b)),1)
                if overlap>=.7:visited.add(other);pending.append(other)
        primary=min(members)
        name=f"source_material_search:{binding['source_sha256'][:12]}:{binding['source_pts']}:{primary}"
        boxes=np.array([by_id[x]['box_xyxy'] for x in members],float)
        seeds=[point for x in members for point in by_id[x]['points_uv']]
        by_id[primary]['box_xyxy']=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)].tolist()
        by_id[primary]['points_uv']=seeds;by_id[primary]['point_labels']=[1]*len(seeds)
        for index in members:
            by_id[index].update(sam_search_object_id=primary,source_material_search_group_id=name,
                shared_SAM_search_is_physical_identity_verification=False,
                independent_RGB_material_layer_preserved=True)
        groups.append(dict(source_binding=binding,source_material_search_group_id=name,
            sam_search_object_id=primary,independent_material_layer_object_ids=sorted(members),
            grouping_authority='overlapping_actual_current_RGB_material_support_for_search_only',
            physical_instances_merged=False,content_or_logo_identity_used=False))
    return dict(prompts=[by_id[x] for x in ids],groups=groups)


def transfer_source_context_identities(rows, prompts, contexts, binding, *, event_id,
                                       query_RGB_sha256, minimum_matches=6):
    """Transfer a confirmed reference copy through actual print and venue matches.

    Repeated content remains in the candidate ledger. Conflicting authenticated
    copies matching the same query keypoints cannot acquire a physical ID.
    Image homographies and descriptor points never become boundary UV here.
    """
    binding = source_binding(binding)
    proposals = {}
    for context in contexts:
        if not (context.get("distributed_distinctive_context_support") and context.get("event_order_consistent")):
            continue
        same_frame(binding, context["query_source_binding"])
        reference = source_binding(context["reference_source_binding"])
        if any(reference[key] != binding[key] for key in ("source_sha256", "time_base")):
            continue
        for row in rows:
            if not row.get("appearance_match") or row["source_packet_sha256"] != context["source_packet_sha256"]:
                continue
            same_frame(binding, row["source_binding"])
            if row.get("reference_source_binding") != context["reference_source_binding"]:
                continue
            human = row.get("event_identity_authority", {})
            from .vision_native_references import registered_source_identity_verified
            if not ((human.get("kind") in HUMAN_IDENTITIES and human.get("reference")) or
                    registered_source_identity_verified(human, reference, row["reference_physical_id"])):
                continue
            authority = dict(kind=SOURCE_CORRESPONDENCE_IDENTITY, event_id=str(event_id),
                reference=human["reference"], confirmed_reference_identity_authority=dict(human),
                confirmed_reference_source_identity_fact=row.get("reference_physical_identity_authority"),
                reference_physical_id=row["reference_physical_id"],
                reference_source_binding=dict(reference), query_source_binding=dict(binding),
                query_RGB_sha256=query_RGB_sha256, source_packet_sha256=context["source_packet_sha256"],
                current_print_correspondence={key:row.get(key) for key in (
                    "appearance_match", "source_packet_sha256", "template_id", "actual_source_keypoints_uv",
                    "template_corresponding_keypoints_uv", "inlier_count", "median_image_transfer_error_px")},
                static_context_evidence={key:context.get(key) for key in (
                    "source_packet_sha256", "distributed_distinctive_context_support", "event_order_consistent",
                    "background_inlier_count", "stable_patch_match_count", "distinctive_descriptor_match_count",
                    "matched_reference_order", "distinctive_reference_keypoints_uv", "distinctive_actual_source_keypoints_uv",
                    "reference_descriptor_novelty_ratios")},
                query_human_pixel_ground_truth=False, boundary_UV_verified_by_identity_transfer=False,
                scope="confirmed_source_instance_transferred_by_current_RGB_context_and_print_correspondence")
            if source_correspondence_identity_verified(authority, binding, row["reference_physical_id"]):
                proposals[row["template_id"]] = authority
    conflicts = set()
    proposed = [row for row in rows if row["template_id"] in proposals]
    for index, row in enumerate(proposed):
        first = {tuple(value) for value in np.rint(np.asarray(row["actual_source_keypoints_uv"])*2).astype(np.int64)}
        for other in proposed[index+1:]:
            if row["content_id"] != other["content_id"] or row["reference_physical_id"] == other["reference_physical_id"]:
                continue
            second = {tuple(value) for value in np.rint(np.asarray(other["actual_source_keypoints_uv"])*2).astype(np.int64)}
            if len(first & second) >= minimum_matches:
                conflicts.update((row["template_id"], other["template_id"]))
    for row in rows:
        authority = proposals.get(row["template_id"])
        if authority is None:
            continue
        row["original_repeated_content_candidates_retained"] = True
        previous_authority = row.get("identity_authority")
        human_query_identity = bool(row.get("physical_id") and isinstance(previous_authority, Mapping)
            and previous_authority.get("kind") in HUMAN_IDENTITIES and previous_authority.get("reference"))
        if human_query_identity:
            # A contradictory machine copy correspondence is a conflict with
            # the source evidence, not permission to retire a user's identity.
            row["human_query_identity_retained_over_machine_context"] = True
            if row["template_id"] in conflicts or row["physical_id"] != row["reference_physical_id"]:
                row["competing_machine_reference_physical_id"] = row["reference_physical_id"]
        elif row["template_id"] in conflicts:
            row.update(physical_id=None, expected_id=None, identity_authority=None,
                identity_status="conflicting_authenticated_reference_copy_correspondences",
                source_physical_identity_correspondence_verified=False,
                source_pixel_to_expected_identity_verified=False)
        elif not row.get("physical_id"):
            row.update(physical_id=row["reference_physical_id"], expected_id=row["reference_physical_id"],
                identity_authority=authority, expected_identity_authority=authority,
                identity_status="source_reference_context_and_print_identity_transferred",
                source_physical_identity_correspondence_verified=True,
                source_pixel_to_expected_identity_verified=False,
                identity_confirmed_by_source_human=False,
                identity_confirmed_by_source_correspondence=True)
        for prompt in prompts:
            if prompt["appearance_correspondence_template_id"] == row["template_id"]:
                prompt.update(physical_id=row["physical_id"], expected_id=row.get("expected_id"),
                    identity_authority=row.get("identity_authority"),
                    expected_identity_authority=row.get("expected_identity_authority"),
                    source_physical_identity_correspondence_verified=row.get("source_physical_identity_correspondence_verified", False),
                    source_pixel_to_expected_identity_verified=bool(row.get("source_pixel_to_expected_identity_verified")),
                    candidate_physical_ids=[f"Ad{n}" for n in row["candidate_numbers"]])
    return rows, prompts


class SourceAdvertisementMatcher:
    """Reusable event RGB appearance bank; SIFT correspondence is not an edge.

    Source crops are navigation supports authenticated against native RGB.
    Features reuse the existing RootSIFT implementation, and descriptor matches
    reuse the event classifier's CPU/OpenCL backend. Only image homographies
    organize correspondence; no camera/world parameters are read or optimized.
    """
    def __init__(self, source_packet_paths, *, event_id, cache_directory=None,
                 template_scale=3., minimum_matches=6, ratio=.8, ransac_px=3.):
        import cv2
        import json
        _add_existing_geometry_dependencies()
        from .camera_reuse_localization import descriptors
        self.event_id = str(event_id)
        self.minimum_matches, self.ratio, self.ransac_px = int(minimum_matches), float(ratio), float(ransac_px)
        if self.minimum_matches < 4 or not 0 < self.ratio < 1 or template_scale <= 0 or ransac_px <= 0:
            raise ValueError("Use finite source feature matching parameters")
        self.templates, self.catalogs, self.contexts = [], {}, []
        self.cache_directory = Path(cache_directory).resolve() if cache_directory else None
        if self.cache_directory:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
        for packet_path in map(Path, source_packet_paths):
            packet_path = packet_path.resolve()
            packet = _read_json(packet_path)
            source = packet["source"]
            ref_binding = source_binding(source, source["native_hw"])
            catalogue = packet["identity_catalog"]
            catalog_path = Path(catalogue["path"]).resolve()
            if _file_sha256(catalog_path) != catalogue["sha256"]:
                raise ValueError("Human event advertisement catalogue changed")
            catalog = _read_json(catalog_path)
            catalog_event = catalog.get("event_id")
            if catalog_event is None and catalog.get("model_id", "").endswith("_shared"):
                catalog_event = catalog["model_id"].removesuffix("_shared")
            if catalog_event is not None and catalog_event != self.event_id:
                raise ValueError("Advertisement catalogue belongs to a different event")
            if packet.get("event_id", self.event_id) != self.event_id:
                raise ValueError("Appearance source packet belongs to a different event")
            self.catalogs[str(catalog_path)] = dict(path=str(catalog_path), sha256=catalogue["sha256"],
                                                   same_content_pairs=catalog["same_content_pairs"])
            alias_by_number = {int(n): row for row in catalog["same_content_pairs"] for n in row["numbers"]}
            image_path = Path(source["source_frame_path"])
            if _file_sha256(image_path) != source["source_frame_sha256"]:
                raise ValueError("Appearance source PNG changed")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2]) != source["native_hw"]:
                raise ValueError("Native appearance source is unavailable")
            rgb_hash = hashlib.sha256(image[:, :, ::-1].copy().tobytes()).hexdigest()
            if rgb_hash != source["decoded_native_rgb_sha256"]:
                raise ValueError("Appearance image is not the authenticated original RGB")
            packet_sha = _file_sha256(packet_path)
            cutoff = min(c["actual_source_navigation_roi"][1] for ad in packet["advertisements"]
                         for c in ad["components"] if c.get("actual_source_navigation_roi"))
            context_mask = self._context_mask(image[:, :, ::-1].copy(), cutoff-8)
            context_points, context_descriptors = descriptors(image, context_mask)
            self.contexts.append(dict(source_binding=ref_binding, source_packet_path=str(packet_path),
                                      source_packet_sha256=packet_sha, points=context_points,
                                      descriptors=context_descriptors, image=image, mask=context_mask,
                                      reference_order=packet["image_left_to_right_physical_order"]))
            for advertisement in packet["advertisements"]:
                number = int(advertisement["advertisement_number"])
                from .vision_native_references import registered_source_identity_verified
                registered = advertisement.get("registered_source_identity_authority")
                registered_valid = registered_source_identity_verified(registered, ref_binding, f"Ad{number}")
                if not advertisement.get("physical_identity_human_confirmed") and not registered_valid:
                    continue
                aliases = alias_by_number[number]
                for component in advertisement["components"]:
                    roi = component.get("actual_source_navigation_roi")
                    if roi is None:
                        continue
                    x0, y0, x1, y1 = map(int, roi)
                    crop = image[y0:y1, x0:x1]
                    if not crop.size:
                        raise ValueError("Empty original source navigation crop")
                    scaled = cv2.resize(crop, None, fx=template_scale, fy=template_scale, interpolation=cv2.INTER_CUBIC)
                    points, descriptor = descriptors(scaled, np.full(scaled.shape[:2], 255, np.uint8))
                    points = points/template_scale+np.array([x0, y0], np.float32)
                    template_id = f"{packet_sha[:12]}:Ad{number}:{component['component_id']}"
                    record = dict(template_id=template_id, source_binding=ref_binding,
                        source_packet_path=str(packet_path), source_packet_sha256=packet_sha,
                        source_frame_path=str(image_path), source_rgb_sha256=rgb_hash,
                        source_frame_sha256=source["source_frame_sha256"],
                        original_RGB_navigation_patch_sha256=hashlib.sha256(crop[:, :, ::-1].copy().tobytes()).hexdigest(),
                        navigation_roi=roi, navigation_roi_is_ad_boundary=False,
                        reference_physical_id=f"Ad{number}", content_id=aliases["content_id"],
                        candidate_numbers=list(aliases["numbers"]), source_component_id=component["component_id"],
                        observed_kind=component.get("observed_kind"),
                        observed_print_variant_id=template_id,
                        same_content_pair_is_print_variant_verification=False,
                        event_identity_authority=dict(kind="human_confirmed_event_catalog",
                            reference=str(catalog_path), catalogue_sha256=catalogue["sha256"],
                            source_instance_binding_reference=str(packet_path),
                            scope="confirmed_reference_instance_not_query_pixels"),
                        feature_native_uv=points, descriptors=descriptor,
                        feature_scale_is_navigation_only=True,
                        boundary_definition="printed_pattern_external_contour",
                        whole_pattern_extent_is_edge=False, internal_holes_have_separate_role=True)
                    if registered_valid:
                        record["event_identity_authority"] = registered
                        record["reference_identity_human_confirmed"] = False
                    record.update(source_component_scope=component.get('source_component_scope'),
                        whole_advertisement_outer_boundary_eligible=component.get(
                            'whole_advertisement_outer_boundary_eligible',True),
                        reference_physical_identity_authority=advertisement.get('physical_identity_authority'))
                    if component.get("observed_kind") == "printed_colour_background":
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        chromatic = (hsv[:, :, 1] > 46) & (hsv[:, :, 2] > 25)
                        bins = (hsv[:, :, 0]//10)[chromatic]
                        if len(bins):
                            peak_bin = int(np.bincount(bins, minlength=18).argmax())
                            dominant = chromatic & (hsv[:, :, 0]//10 == peak_bin)
                            if dominant.sum() >= crop.shape[0]*crop.shape[1]*.2:
                                colours = crop[:, :, ::-1][dominant].astype(float)/255.
                                direction = (colours-colours.mean(1, keepdims=True)).mean(0)
                                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                                record["source_print_appearance"] = dict(dominant_chromatic_print_background=True,
                                    RGB_chroma_direction=direction.tolist(), original_source_RGB_pixels=int(dominant.sum()),
                                    colour_is_search_descriptor_not_pixel_truth=True,
                                    source_native_navigation_patch_sha256=record["original_RGB_navigation_patch_sha256"])
                    self.templates.append(record)
                    if self.cache_directory:
                        key = hashlib.sha256(template_id.encode()).hexdigest()[:24]
                        np.savez_compressed(self.cache_directory/f"{key}.npz", feature_native_uv=points,
                            descriptors=np.empty((0, 128), np.float32) if descriptor is None else descriptor)
                        metadata = {k: v for k, v in record.items() if k not in ("feature_native_uv", "descriptors")}
                        metadata.update(feature_count=len(points), descriptor_path=str(self.cache_directory/f"{key}.npz"))
                        metadata["descriptor_file_sha256"] = _file_sha256(self.cache_directory/f"{key}.npz")
                        (self.cache_directory/f"{key}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _context_mask(rgb, cutoff):
        import cv2
        h, w = rgb.shape[:2]
        valid = np.zeros((h, w), np.uint8)
        valid[:max(0, min(h, int(cutoff)))] = 255
        # Same usual broadcast-overlay exclusion as the existing event
        # classifier (its fixed 426x240 feature basis), expressed natively.
        valid[:int(h*57/240), :int(w*145/426)] = 0
        # Saturated red LED pixels are dynamic clock candidates, not venue
        # anchors. Dilation removes their immediate descriptor support.
        v = rgb.astype(np.int16)
        red_led = ((v[:, :, 0] > 160) & (v[:, :, 0] > v[:, :, 1]+60) &
                   (v[:, :, 0] > v[:, :, 2]+50)).astype(np.uint8)
        valid[cv2.dilate(red_led, np.ones((31, 31), np.uint8)) > 0] = 0
        return valid

    def _context_matches(self, rgb, binding, rows):
        import cv2
        from .camera_reuse_localization import descriptors, match_features, ReuseConfig
        successful = [row for row in rows if row.get("appearance_match")]
        if not successful:
            return []
        query = rgb[:, :, ::-1].copy()
        results = []
        for context in self.contexts:
            result = dict(reference_source_binding=context["source_binding"], query_source_binding=binding,
                          source_packet_sha256=context["source_packet_sha256"],
                          role="static_context_identity_search_not_print_boundary",
                          physical_wing_identity_verified=False, camera_model_created=False,
                          status="insufficient_static_context_correspondence")
            source_rows = [row for row in successful if row['source_packet_sha256']==context['source_packet_sha256']]
            if not source_rows:
                result['status']='no_print_correspondence_to_this_reference_source'
                results.append(result)
                continue
            cutoff = min(float(np.asarray(row['actual_source_keypoints_uv'])[:,1].min()) for row in source_rows)-20
            qpoints, qdesc = descriptors(query, self._context_mask(rgb,cutoff))
            matched = match_features((context["points"], context["descriptors"]), (qpoints, qdesc),
                                     ReuseConfig(min_matches=10, ransac_px=self.ransac_px))
            if matched is None:
                results.append(result)
                continue
            left, right, H = matched
            # Check actual RGB patches after the context homography. Changed
            # people, clock digits and reflections are not stable wall evidence.
            aligned = cv2.warpPerspective(query, np.linalg.inv(H), (query.shape[1], query.shape[0]))
            grey_ref = cv2.cvtColor(context["image"], cv2.COLOR_BGR2GRAY)
            grey_query = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            correlations = []
            for point in left:
                a = cv2.getRectSubPix(grey_ref, (31, 31), tuple(point)).astype(float)
                b = cv2.getRectSubPix(grey_query, (31, 31), tuple(point)).astype(float)
                correlations.append(float(np.mean((a-a.mean())*(b-b.mean()))/max(a.std()*b.std(), 1e-9)))
            stable = np.asarray(correlations) >= .8
            left_stable, right_stable = left[stable], right[stable]
            # A matched window corner can be repeated in another bay. Compare
            # against spatially distant descriptors in the authenticated same
            # source image; record the novelty margin rather than declaring
            # every SIFT inlier a distinctive architectural anchor.
            ref_points, ref_desc = context["points"], context["descriptors"]
            query_nearest = np.argmin(np.linalg.norm(right_stable[:, None]-qpoints[None], axis=2), axis=1)
            ref_nearest = np.argmin(np.linalg.norm(left_stable[:, None]-ref_points[None], axis=2), axis=1)
            novelty, unique = [], []
            for ri, qi in zip(ref_nearest, query_nearest):
                distant = np.linalg.norm(ref_points-ref_points[ri], axis=1) > 64
                alternatives = np.linalg.norm(ref_desc[distant]-ref_desc[ri], axis=1)
                nearest_other = float(alternatives.min()) if len(alternatives) else 0.
                actual_distance = float(np.linalg.norm(qdesc[qi]-ref_desc[ri]))
                margin = actual_distance/max(nearest_other, 1e-12)
                novelty.append(margin)
                unique.append(margin < self.ratio)
            unique = np.asarray(unique, bool)
            unique_left, unique_right = left_stable[unique], right_stable[unique]
            distributed = bool(len(unique_left) >= 6 and np.ptp(unique_left[:, 0]) > 180 and
                               np.ptp(unique_left[:, 1]) > 35)
            # Independent print matches must preserve the catalogue-linked
            # source order. This still cannot make an ambiguous glass-only
            # background into an affirmative physical-wing assertion.
            observed = sorted(source_rows, key=lambda row: np.mean(row["actual_source_keypoints_uv"], axis=0)[0])
            # Multiple separately printed components may be one human
            # catalogue group (e.g. Ad17). Preserve their separate contours;
            # evaluate physical perimeter order once per known group.
            numbers = list(dict.fromkeys(int(row["reference_physical_id"][2:]) for row in observed))
            reference_order = context["reference_order"]
            indices = [reference_order.index(n) for n in numbers if n in reference_order]
            order_consistent = len(indices) == len(numbers) and all(b > a for a, b in zip(indices[:-1], indices[1:]))
            result.update(status="source_context_correspondence_candidate",
                background_inlier_count=len(left), stable_patch_match_count=int(stable.sum()),
                distinctive_descriptor_match_count=int(unique.sum()),
                distributed_distinctive_context_support=distributed,
                reference_static_keypoints_uv=left.tolist(), actual_source_static_keypoints_uv=right.tolist(),
                stable_patch_correlations=correlations, reference_descriptor_novelty_ratios=novelty,
                distinctive_reference_keypoints_uv=unique_left.tolist(),
                distinctive_actual_source_keypoints_uv=unique_right.tolist(),
                H_query_native_from_reference_native=H.tolist(),
                matched_reference_order=numbers, event_order_consistent=order_consistent,
                repeat_glass_geometry_alone_is_identity_proof=False,
                identity_status="distributed_static_context_and_print_order_candidate" if distributed and order_consistent else
                                "physical_wing_ambiguous_insufficient_distinctive_static_context",
                named_static_anchor_semantics_verified=False,
                observed_print_order_scope='this_authenticated_reference_packet_only',
                fitted_background_pixels_used_as_print_boundary=False)
            results.append(result)
        return results

    def read(self, rgb, binding, *, model_search=None, tracked_identities=(), source_identity_anchors=(),
             source_search=None, person_boxes=(), overlay_boxes=()):
        import cv2
        from .camera_reuse_localization import descriptors
        from .camera_classification import descriptor_matches
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Match the actual native uint8 RGB")
        binding = source_binding(binding, rgb.shape[:2])
        if binding.get("event_id", self.event_id) != self.event_id:
            raise ValueError("Query source belongs to a different event")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if binding.get("decoded_native_rgb_sha256", digest) != digest:
            raise ValueError("Query RGB differs from its native source binding")
        search_mask, search_evidence = advertisement_descriptor_search_mask(
            rgb, binding, source_search=source_search, person_boxes=person_boxes, overlay_boxes=overlay_boxes)
        points, descriptor = descriptors(rgb[:, :, ::-1].copy(), search_mask)
        rows, prompts = [], []
        for template_index, template in enumerate(self.templates):
            row = {k: template[k] for k in ("template_id", "reference_physical_id", "content_id", "candidate_numbers",
                    "source_component_id", "event_identity_authority", "source_packet_sha256")}
            row.update(source_binding=binding, reference_source_binding=template["source_binding"],
                       reference_physical_identity_authority=template.get("reference_physical_identity_authority"),
                       physical_id=None, identity_status="appearance_not_matched", actual_source_keypoints_uv=[],
                       template_corresponding_keypoints_uv=[], matching_is_boundary_reading=False,
                       observed_print_variant_id=template["observed_print_variant_id"],
                       reference_observed_kind=template["observed_kind"],
                       reference_patch_sha256=template["original_RGB_navigation_patch_sha256"],
                       same_content_pair_is_print_variant_verification=False,
                       descriptor_search_evidence=search_evidence)
            row.update(source_reference_component_scope=template.get('source_component_scope'),
                whole_advertisement_outer_boundary_eligible=template.get('whole_advertisement_outer_boundary_eligible',True))
            reference = template["descriptors"]
            if reference is None or descriptor is None or len(reference) < 4 or len(descriptor) < 4:
                row["reason"] = "insufficient_actual_source_features"
                rows.append(row)
                continue
            forward = descriptor_matches(reference, descriptor)
            backward = descriptor_matches(descriptor, reference)
            mutual = {pair[0].queryIdx: pair[0].trainIdx for pair in backward if len(pair) == 2 and
                      pair[0].distance < self.ratio*pair[1].distance}
            good = [a for pair in forward if len(pair) == 2 for a, b in [pair]
                    if a.distance < self.ratio*b.distance and mutual.get(a.trainIdx) == a.queryIdx]
            row["tentative_feature_matches"] = len(good)
            row["tentative_query_keypoints_uv"] = points[[m.trainIdx for m in good]].tolist()
            row["tentative_reference_keypoints_uv"] = template["feature_native_uv"][[m.queryIdx for m in good]].tolist()
            row["tentative_correspondences_are_boundary_measurements"] = False
            if len(good) < self.minimum_matches:
                row["reason"] = "insufficient_unambiguous_descriptor_matches"
                rows.append(row)
                continue
            left = template["feature_native_uv"][[m.queryIdx for m in good]]
            right = points[[m.trainIdx for m in good]]
            H, keep = cv2.findHomography(left, right, cv2.RANSAC, self.ransac_px)
            row["image_correspondence_inlier_count"] = int(keep.sum()) if keep is not None else 0
            if H is None or keep is None or int(keep.sum()) < self.minimum_matches:
                row["reason"] = "inconsistent_image_correspondence"
                rows.append(row)
                continue
            keep = keep.ravel().astype(bool)
            left, right = left[keep], right[keep]
            unique_right = np.unique(np.rint(right*2).astype(int), axis=0)
            if len(unique_right) < self.minimum_matches or min(np.ptp(right, axis=0)) < 3:
                row["reason"] = "feature_support_collapsed_or_collinear"
                rows.append(row)
                continue
            projected = cv2.perspectiveTransform(left[None].astype(np.float32), H)[0]
            errors = np.linalg.norm(projected-right, axis=1)
            identity = advertisement_identity_from_constraints(template["candidate_numbers"], right, binding,
                model_search=model_search, tracked_identities=tracked_identities, source_identity_anchors=source_identity_anchors)
            x0, y0, x1, y1 = template["navigation_roi"]
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            support = cv2.perspectiveTransform(corners[None], H)[0]
            low, high = support.min(0), support.max(0)
            low = np.maximum(low, 0)
            high = np.minimum(high, [rgb.shape[1], rgb.shape[0]])
            usable_support = bool(np.isfinite(support).all() and (high > low).all() and
                                  np.ptp(support, axis=0).max() < 2*max(rgb.shape[:2]))
            row.update(identity, appearance_match=True, status="matched_source_content_correspondence",
                       actual_source_keypoints_uv=right.tolist(), template_corresponding_keypoints_uv=left.tolist(),
                       inlier_count=len(right), median_image_transfer_error_px=float(np.median(errors)),
                       H_query_native_from_reference_native=H.tolist(),
                       search_support_polygon_uv=support.tolist() if usable_support else None,
                       search_support_is_visible_print_boundary=False,
                       image_homography_is_camera_model=False, boundary_precision_accepted=False)
            rows.append(row)
            if usable_support:
                prompts.append(dict(object_id=2000+template_index, kind="advertisement_panel",
                    origin="original_rgb_candidate", source_binding=binding, box_xyxy=np.r_[low, high].tolist(),
                    expected_id=identity["expected_id"], physical_id=identity["physical_id"],
                    identity_authority=identity["identity_authority"], candidate_physical_ids=[f"Ad{n}" for n in identity["candidate_numbers"]],
                    content_id=template["content_id"], source_component_id=template["source_component_id"],
                    observed_print_variant_id=template["observed_print_variant_id"],
                    same_content_pair_is_print_variant_verification=False,
                    appearance_correspondence_template_id=template["template_id"],
                    actual_pattern_keypoints_uv=right.tolist(),
                    source_print_appearance=template.get("source_print_appearance"),
                    source_pattern_reference={k: template[k].tolist() if isinstance(template[k], np.ndarray) else template[k] for k in (
                        "source_binding", "source_frame_path", "source_frame_sha256", "source_rgb_sha256",
                        "navigation_roi", "feature_native_uv", "descriptors", "observed_kind")},
                    boundary_definition="printed_pattern_external_contour",
                    boundary_definition_authority="user_confirmed_printed_outer_contours_holes_extent_separate",
                    whole_pattern_extent_is_edge=False, internal_holes_have_separate_role=True))
                prompts[-1]['source_pattern_reference'].update(
                    source_component_scope=template.get('source_component_scope'),
                    whole_advertisement_outer_boundary_eligible=template.get('whole_advertisement_outer_boundary_eligible',True),
                    reference_physical_identity_authority=template.get('reference_physical_identity_authority'))
        contexts = self._context_matches(rgb, binding, rows)
        rows, prompts = transfer_source_context_identities(rows, prompts, contexts, binding,
            event_id=self.event_id, query_RGB_sha256=digest, minimum_matches=self.minimum_matches)
        return dict(source_binding=binding, event_id=self.event_id, source_rgb_sha256=digest,
                    status="source_content_matching_complete", matches=rows, prompts=prompts,
                    source_context_correspondences=contexts,
                    content_matched_count=sum(bool(row.get("appearance_match")) for row in rows),
                    source_verified_identity_count=sum(bool(row.get("source_pixel_to_expected_identity_verified")) for row in rows),
                    source_correspondence_identity_count=sum(bool(row.get("source_physical_identity_correspondence_verified")) for row in rows),
                    query_feature_count=len(points), template_count=len(self.templates),
                    descriptor_search_evidence=search_evidence,
                    world_or_camera_optimized=False, actual_source_keypoints_are_print_boundary=False,
                    same_logo_instances_merged=False, pixel_training_labels_generated=False,
                    full_advertisement_coverage_accepted=False)


def review_source_packet(source_packet_path, checkpoint_path, output_directory):
    """Save real-frame candidates using existing source ROIs, never model geometry.

    This small runnable review is deliberately explicit about ink/navigation
    ROIs. It does not convert those ROIs into verified panel rectangles. Full
    continuous reading is available through the same class's ``read`` method.
    """
    import json
    import cv2
    import torch
    from PIL import Image
    packet_path = Path(source_packet_path).resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    source = packet["source"]
    rgb = np.asarray(Image.open(source["source_frame_path"]).convert("RGB"))
    native_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    if native_sha != source["decoded_native_rgb_sha256"]:
        raise ValueError("Review PNG does not match the recorded original video RGB")
    binding = source_binding(source, rgb.shape[:2])
    prompts = []
    for advertisement in packet["advertisements"]:
        for component in advertisement["components"]:
            roi = component.get("actual_source_navigation_roi")
            if roi is None:
                continue
            authority = None
            if advertisement.get("physical_identity_human_confirmed"):
                authority = dict(kind="human_confirmed_event_catalog", reference=str(packet_path),
                                 source_fact=advertisement["physical_identity_authority"])
            prompts.append(dict(object_id=len(prompts)+1, kind="advertisement_panel",
                                origin="original_rgb_candidate", source_binding=binding,
                                box_xyxy=roi, physical_id=f"Ad{advertisement['advertisement_number']}",
                                identity_authority=authority,
                                prompt_target_role=component.get("observed_kind", "source_navigation_ROI")))
            if advertisement["advertisement_number"] in (21, 23):
                # Latest direct user instruction: these have no solid frame;
                # read external printed contours, extent and holes separately.
                prompts[-1].update(boundary_definition="printed_pattern_external_contour",
                    boundary_definition_authority=dict(kind="direct_user_confirmation",
                        reference="events/ankara_2026/model_constraints/visual_boundary_roles_user_confirmed_20261006.json"))
    reader = Sam2SourceInstances(checkpoint_path)
    result = reader.read(rgb, binding, prompts=prompts, reset=True)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    preview = rgb.copy()
    rows = []
    colors = [(30, 230, 80), (255, 100, 30), (20, 170, 255), (240, 30, 160)]
    for index, instance in enumerate(result["instances"]):
        mask = instance["region_mask"].cpu().numpy()
        color = np.asarray(colors[index % len(colors)])
        preview[mask] = np.rint(preview[mask]*.78+color*.22).astype(np.uint8)
        mask_name = f"region_object_{instance['object_id']}.png"
        Image.fromarray(mask.astype(np.uint8)*255).save(output_directory/mask_name)
        points = []
        for reading in instance["boundary_readings"]:
            for run in reading["measured_runs"]:
                uv = run["actual_source_uv"].cpu().numpy()
                if len(uv) > 1:
                    cv2.polylines(preview, [np.rint(uv*256).astype(np.int32)], False,
                                  tuple(map(int, color)), 1, cv2.LINE_AA, shift=8)
            points.extend(reading["actual_source_uv"].cpu().tolist())
        if mask.any():
            y, x = np.nonzero(mask)
            label = f"{instance['physical_id'] or instance['expected_id'] or instance['object_id']} candidate"
            origin = (int(x.mean()), int(max(18, y.min()-5)))
            cv2.putText(preview, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(preview, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .45,
                        tuple(map(int, color)), 1, cv2.LINE_AA)
        def serial(value):
            if isinstance(value, torch.Tensor):
                return serial(value.detach().cpu().tolist())
            if isinstance(value, np.ndarray):
                return serial(value.tolist())
            if isinstance(value, dict):
                return {key: serial(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serial(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
        row = {key: value for key, value in instance.items() if key != "region_mask"}
        row.update(mask_path=str(output_directory/mask_name), region_pixels=int(mask.sum()),
                   actual_RGB_edge_candidate_count=len(points))
        rows.append(serial(row))
    preview_path = output_directory/"source_region_candidates.png"
    Image.fromarray(preview).save(preview_path)
    report = dict(schema="rinkspace.SAM2-source-instance-review.v1", source_binding=binding,
                  source_frame_path=source["source_frame_path"], actual_RGB_sha256=native_sha,
                  model_sha256=reader.model_sha256, checkpoint_path=reader.checkpoint_path,
                  full_decoder_and_video_memory_used=True, source_seed_packet=str(packet_path),
                  source_seed_is_panel_geometry=False, instances=rows, preview_path=str(preview_path),
                  world_or_camera_optimized=False, precise_advertisement_perimeters_verified=False,
                  all_visible_instance_accuracy_accepted=False)
    (output_directory/"source_instance_review.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Review full local SAM2 source-region candidates")
    parser.add_argument("--source-packet", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    result = review_source_packet(options.source_packet, options.checkpoint, options.output)
    print(json.dumps(dict(preview_path=result["preview_path"],
                          source_pts=result["source_binding"]["source_pts"],
                          instances=len(result["instances"]),
                          precision_accepted=False)))
