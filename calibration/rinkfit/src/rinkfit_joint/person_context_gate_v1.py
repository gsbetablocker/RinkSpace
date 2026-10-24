"""Source-only occlusion context candidate; never a person truth or packet override.

An empty foreground-mask pixel does not prove visible ice. Convex interiors of
separate same-exposure foreground components are retained as UNKNOWN context.
Source rows and every original measurement remain unchanged in the review.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .io_utils import file_sha256, object_sha256
from .native_material_evidence import direct_ice_measurement_reason

SCHEMA = 'rinkfit.person_context_source_gate_candidate.v1'


def build_person_context(frame_record, *, rgb):
    """Return (person bitmap, unknown-context bitmap, pinned source receipt).

    frame_record is an existing native person-mask index entry, not a board-body
    mask or a visible-ice mask. rgb must be the same original native exposure.
    """
    import cv2
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Native uint8 RGB image required')
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    if digest != frame_record['source_RGB_sha256']:
        raise ValueError('Person context is not bound to this original exposure')
    if frame_record.get('graphics_mask_included') is not False:
        raise ValueError('Typed person mask without graphics is required')
    ref = dict(file=frame_record['mask_file'], sha256=frame_record['mask_sha256'])
    if file_sha256(ref['file']) != ref['sha256']:
        raise ValueError('Original native person mask changed')
    bitmap = cv2.imread(str(ref['file']), cv2.IMREAD_GRAYSCALE)
    if bitmap is None or bitmap.shape != rgb.shape[:2]:
        raise ValueError('Person mask native dimensions differ')
    person = bitmap > 0
    boolean_sha = hashlib.sha256(person.tobytes()).hexdigest()
    if frame_record.get('mask_boolean_sha256', boolean_sha) != boolean_sha:
        raise ValueError('Original person-mask interpretation differs')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(person.astype('u1'), 8)
    envelope = person.astype('u1')
    components = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        local = (labels[y:y+h, x:x+w] == label).astype('u1')
        contours, hierarchy = cv2.findContours(local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        points = np.concatenate([c.reshape(-1, 2) for k, c in enumerate(contours)
                                 if hierarchy[0, k, 3] < 0]) + [x, y]
        hull = cv2.convexHull(points.astype('i4'))
        cv2.fillConvexPoly(envelope, hull, 1)
        components.append(dict(component=label, original_mask_pixels=area,
            bounding_box_xyxy=[x, y, x+w, y+h], source_space_hull=hull.reshape(-1, 2).tolist(),
            holes=int(np.sum(hierarchy[0, :, 3] >= 0)),
            hull_area_native_pixels_squared=float(cv2.contourArea(hull))))
    unknown = (envelope > 0) & ~person
    receipt = dict(schema=SCHEMA, shot_id=frame_record['shot_id'],
        native_pts=int(frame_record['native_pts']), time_base='1/12800',
        source_video_sha256=frame_record['source_video_sha256'], source_RGB_sha256=digest,
        native_hw=list(person.shape), person_mask_ref=ref,
        original_mask_boolean_sha256=boolean_sha,
        original_mask_pixels=int(person.sum()), unknown_context_pixels=int(unknown.sum()),
        unknown_context_boolean_sha256=hashlib.sha256(unknown.tobytes()).hexdigest(),
        components=components, separate_components_not_merged=True,
        hull_is_unknown_context_not_person_segmentation=True,
        zero_person_mask_is_not_visible_ice_proof=True,
        source_RGB_or_person_mask_changed=False, source_world_model_used=False,
        calibrated_occlusion_uncertainty_claimed=False, packet_installed=False)
    return person, unknown, receipt


def _recorded_profile_footprint(row, shape):
    """Bilinear native support of recorded quantile interval x tangent [-3,3].

    This is an exclusion test on the recorded material support, not a claim to
    reconstruct the original profile search, nor a new measured coordinate.
    """
    uv = np.asarray(row['uv'], float)
    evidence = row.get('pixel_support', {})
    normal = np.asarray(evidence.get('normal', []), float)
    if uv.shape != (2,) or normal.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
        return np.empty((0, 2), int), True
    normal /= np.linalg.norm(normal)
    values = [uv[None]]
    for q in (row.get('paired_edge_uv'), evidence.get('side_interface_quartiles_uv')):
        if q is not None:
            q = np.asarray(q, float).reshape(-1, 2)
            if not np.isfinite(q).all():
                return np.empty((0, 2), int), True
            values.append(q)
    points = np.concatenate(values)
    offsets = (points - uv) @ normal
    steps = max(1, int(np.ceil((offsets.max()-offsets.min())/.25)))
    along = uv + np.linspace(offsets.min(), offsets.max(), steps+1)[:, None]*normal
    tangent = np.array([-normal[1], normal[0]])
    sites = (along[:, None] + np.arange(-3, 4)[None, :, None]*tangent).reshape(-1, 2)
    h, w = shape
    outside = bool(((sites < [0, 0]) | (sites > [w-1, h-1])).any())
    taps = np.concatenate([np.column_stack([x, y]) for x in (np.floor(sites[:, 0]), np.ceil(sites[:, 0]))
                           for y in (np.floor(sites[:, 1]), np.ceil(sites[:, 1]))]).astype(int)
    taps = np.unique(taps, axis=0)
    return taps[((taps >= [0, 0]) & (taps < [w, h])).all(axis=1)], outside


def _material_reason(row):
    reason = direct_ice_measurement_reason(row)
    if reason is not None:
        return reason
    if 'boundary' in row.get('definition', ''):
        q = np.asarray(row.get('pixel_support', {}).get('side_interface_quartiles_uv', []), float)
        uv = np.asarray(row['uv'], float)
        n = np.asarray(row['pixel_support']['normal'], float)
        n /= np.linalg.norm(n)
        if q.ndim != 3 or q.shape[1:] != (3, 2) or not len(q) or not np.isfinite(q).all():
            return 'missing_actual_boundary_transition_quartiles'
        # A ring has ONE material interface. The historical complete_both flag
        # means two paint flanks and is false by construction for ring readers.
        for interface in q:
            offset = (interface-uv) @ n
            if np.linalg.norm(interface[1]-uv) > 1e-6 or not offset[0] < offset[1] < offset[2]:
                return 'unordered_or_unmatched_actual_boundary_transition'
    return None


def review_row(row, context, *, source_finding=None):
    """Preserve row; return operator-aware, source-only exclusion candidate."""
    person, unknown, receipt = context
    if int(row['native_pts']) != receipt['native_pts'] or row['shot_id'] != receipt['shot_id'] or row.get('source_sha256') != receipt['source_video_sha256']:
        raise ValueError('Source row and person context belong to different exposures')
    pixels, outside = _recorded_profile_footprint(row, person.shape)
    hits_person = int(person[pixels[:, 1], pixels[:, 0]].sum()) if len(pixels) else 0
    hits_unknown = int(unknown[pixels[:, 1], pixels[:, 0]].sum()) if len(pixels) else 0
    material_reason = _material_reason(row)
    if source_finding is not None:
        if source_finding['source_row_id'] != row['source_row_id'] or int(source_finding['native_pts']) != int(row['native_pts']) or source_finding['source_UV'] != row['uv'] or source_finding['source_RGB_sha256'] != receipt['source_RGB_sha256']:
            raise ValueError('Independent source finding does not bind this original row')
        if source_finding['classification'] != 'nonmaterial_occlusion' or not source_finding.get('source_review_finding'):
            raise ValueError('Explicit original RGB nonmaterial finding required')
        for ref in source_finding['evidence_refs']:
            if file_sha256(ref['file']) != ref['sha256']:
                raise ValueError('Independent source finding evidence changed')
        decision = 'source_RGB_confirmed_nonmaterial_occlusion'
    elif outside or not len(pixels):
        decision = 'unknown_incomplete_native_profile_support'
    elif hits_person:
        decision = 'existing_person_mask_overlap'
    elif hits_unknown:
        decision = 'unknown_person_component_context'
    elif material_reason:
        decision = 'unknown_material_evidence'
    else:
        decision = 'clear_of_person_context_and_recorded_material_gate'
    return dict(schema=SCHEMA, original_source_record=copy.deepcopy(row),
        original_source_record_sha256=object_sha256(row), decision=decision,
        candidate_gate_clear=decision == 'clear_of_person_context_and_recorded_material_gate',
        original_UV_source_ID_PTS_E_weights_unchanged=True,
        recorded_support=dict(native_pixels=len(pixels), outside_native_image=outside,
            existing_person_mask_pixels=hits_person, unknown_context_pixels=hits_unknown),
        material_evidence_unresolved_reason=material_reason,
        complete_both_material_sides_visible_means_two_paint_flanks=True,
        ring_material_interface_does_not_require_two_paint_flanks=True,
        independent_source_finding=copy.deepcopy(source_finding),
        named_entity_identity_and_visibility_still_require_independent_source_evidence=True,
        fit_admission_issued=False, packet_installed=False)


def write_candidate_review(*, rows, frame_record, rgb, output, source_refs,
                           source_findings=()):
    """Write one immutable small review generation, retaining every input row.

    This never emits admission arrays, a production mask, or a modified packet.
    """
    import gzip
    output = Path(output)
    if output.exists():
        raise FileExistsError('Candidate evidence generation already exists')
    for ref in source_refs:
        if file_sha256(ref['file']) != ref['sha256']:
            raise ValueError('Pinned original source evidence changed')
    context = build_person_context(frame_record, rgb=rgb)
    findings = {f['source_row_id']: f for f in source_findings}
    if len(findings) != len(source_findings):
        raise ValueError('Duplicate independent source finding')
    output.mkdir(parents=True)
    context_path = output/'context.json'
    context_path.write_text(json.dumps(context[2], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    counts = Counter()
    seen = set()
    rows_path = output/'rows.jsonl.gz'
    with gzip.open(rows_path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            key = row['source_row_id']
            if key in seen:
                raise ValueError('Duplicate original source row')
            seen.add(key)
            result = review_row(row, context, source_finding=findings.get(key))
            counts[result['decision']] += 1
            stream.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
    if set(findings)-seen:
        raise ValueError('Independent finding has no preserved original source row')
    def ref(path):
        return dict(file=str(Path(path).resolve()), sha256=file_sha256(path))
    manifest = dict(schema=SCHEMA, shot_id=frame_record['shot_id'],
        native_pts=int(frame_record['native_pts']), source_refs=source_refs,
        original_person_mask_index_record=frame_record, context=ref(context_path),
        rows=ref(rows_path), implementation=ref(__file__), input_rows=len(seen),
        preserved_original_records=len(seen), decisions=dict(counts),
        confirmed_nonmaterial_findings=len(findings),
        exclusions_do_not_claim_all_unknown_pixels_are_person=True,
        gate_clear_does_not_certify_named_material_identity=True,
        original_source_UV_or_measurements_changed=False,
        prior_costs_or_history_changed=False, fit_admission_issued=False,
        production_masks_written=False, packet_installed=False)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest
