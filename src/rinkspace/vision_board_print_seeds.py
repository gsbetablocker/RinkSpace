"""Existing frozen print contours navigated by original-image correspondence.

No scene XYZ or camera enters the source navigation. The returned coordinates
are queries; BoardProgrammeSourceReader still reads the current RGB boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .vision_programme_board_review import _root_sift, _source_orientation_preserved


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_image(reference):
    if _digest(reference['path']) != reference['sha256']:
        raise ValueError('Existing source print image changed')
    image = cv2.imread(reference['path'])
    if image is None or image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
        raise ValueError('Original native source print image required')
    return image


def _features(image, roi, sift):
    x0, y0, x1, y1 = map(int, roi)
    keys, descriptors = sift.detectAndCompute(
        cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None)
    coordinates = np.asarray([k.pt for k in keys], np.float32).reshape(-1, 2)+[x0, y0]
    return coordinates.astype(np.float32), descriptors


def _navigation(source_image, source_roi, target_image, target_roi, sift):
    source, source_desc = _features(source_image, source_roi, sift)
    target, target_desc = _features(target_image, target_roi, sift)
    if source_desc is None or target_desc is None or min(len(source), len(target)) < 8:
        return None, dict(status='pending_insufficient_actual_print_descriptors')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(_root_sift(source_desc), _root_sift(target_desc), k=2)
    reverse = matcher.knnMatch(_root_sift(target_desc), _root_sift(source_desc), k=2)
    reverse_good = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse
                    if len(pair) == 2 and pair[0].distance < .82*pair[1].distance}
    matches = [pair[0] for pair in forward if len(pair) == 2
               and pair[0].distance < .82*pair[1].distance
               and reverse_good.get(pair[0].trainIdx) == pair[0].queryIdx]
    if len(matches) < 8:
        return None, dict(status='pending_insufficient_mutual_actual_print_matches', matches=len(matches))
    a = source[[m.queryIdx for m in matches]]
    b = target[[m.trainIdx for m in matches]]
    matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.)
    take = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(a), bool)
    if (matrix is None or take.sum() < 8 or take.mean() < .6
            or np.ptp(a[take], axis=0).min() < 8.
            or np.ptp(b[take], axis=0).min() < 8.
            or not _source_orientation_preserved(matrix, a[take])):
        return None, dict(status='pending_actual_print_correspondence_or_orientation', matches=len(matches),
                          inliers=int(take.sum()))
    transformed = cv2.perspectiveTransform(a[take][None], matrix)[0]
    error = np.linalg.norm(transformed-b[take], axis=1)
    return matrix, dict(kind='same_print_original_RGB_mutual_SIFT_navigation',
        status='source_navigation_requires_current_visible_transition_read',
        matches=len(matches), inliers=int(take.sum()), median_inlier_px=float(np.median(error)),
        model_projection_or_fit_residual_used=False,
        source_texture_inliers_not_pixel_precision_or_whole_contour_truth=True)


def source_print_contour_seeds(scene_path, source_packet_path, identity_authority,
                               *, advertisement_numbers=(21, 22, 23)):
    """Map only existing model components of explicitly reviewed instances.

    Identity authority names exact source frame/ROI independently of this
    frozen scene. A homologous printed graphic transfers navigation only.
    No missing letter, hidden contour, or polygon is fabricated.
    """
    scene_path, packet_path = Path(scene_path).resolve(), Path(source_packet_path).resolve()
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    packet = json.loads(packet_path.read_text(encoding='utf-8-sig'))
    authority_path = Path(identity_authority['path']).resolve()
    if _digest(authority_path) != identity_authority['sha256']:
        raise ValueError('Actual source physical identity authority changed')
    authority = json.loads(authority_path.read_text(encoding='utf8'))
    approved = {(int(r['advertisement_number']), int(r['native_pts'])): r
                for r in authority['reviewed_physical_instances']
                if r.get('physical_instance_identity_confirmed') is True
                and r.get('original_source_image_actually_reviewed') is True
                and r.get('model_projection_or_fit_residual_used_for_identity') is False}
    prints = scene['source_conditioned_board_prints']
    if prints['schema'] != 'rinkspace.current-source-conditioned-board-print-contours.v1':
        raise ValueError('Exact existing frozen print model required')
    patterns = {int(r['advertisement_id']): r for r in prints['patterns']}
    source_patterns = {int(r['advertisement_id']): r for r in packet['source_conditioned_print_contours']}
    sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.014)
    cv2.setNumThreads(2)
    image_cache = {}
    seeds, checks = [], []
    for number in advertisement_numbers:
        pattern = patterns[int(number)]
        original = source_patterns[int(number)]
        review = approved.get((int(number), int(pattern['anchor_pts'])))
        if review is None or review['source_image'] != pattern['anchor_frame']:
            checks.append(dict(advertisement_number=int(number), status='pending_exact_source_instance_identity'))
            continue
        canonical_number = int(number)-13 if int(number) >= 14 else int(number)
        canonical = source_patterns[canonical_number]
        current_reference = pattern['anchor_frame']
        current_key = current_reference['path']
        if current_key not in image_cache:
            image_cache[current_key] = _source_image(current_reference)
        target_image = image_cache[current_key]
        if int(number) == 22:
            # This model's canonical CAN contour was frozen on the actual Ad22
            # source itself, even though the opposite instance also references it.
            canonical = original
        source_reference = canonical['anchor_frame']
        source_key = source_reference['path']
        if source_key not in image_cache:
            image_cache[source_key] = _source_image(source_reference)
        if source_key == current_key:
            matrix = np.eye(3)
            correspondence = dict(kind='exact_original_source_print_component',
                model_projection_or_fit_residual_used=False)
        else:
            matrix, correspondence = _navigation(image_cache[source_key], canonical['source_only_navigation_roi'],
                target_image, original['source_only_navigation_roi'], sift)
        checks.append(dict(advertisement_number=int(number), **correspondence))
        if matrix is None:
            continue
        original_contours = {r['id']: r for r in canonical['external_contours']}
        for component in pattern['external_contours']:
            source_component = original_contours.get(component['id'])
            if source_component is None:
                continue
            native_uv = np.asarray(source_component['uv'], np.float32)
            # Match the existing component to its preserved raw-source contour,
            # rather than using its conditional world vertices as observations.
            if not np.array_equal(np.asarray(component['source_native_uv'], np.float32), native_uv):
                raise ValueError('Frozen component and preserved original contour differ')
            navigation = cv2.perspectiveTransform(native_uv[None], matrix)[0]
            if (navigation < [0, 0]).any() or (navigation >= [1920, 1080]).any():
                continue
            target = dict(kind='current_scene_board_print_contour', advertisement_id=int(number),
                contour_id=component['id'], source_anchor_actual_pts=int(pattern['anchor_actual_pts']),
                scene_q_sha256=prints['q_sha256'])
            seeds.append(dict(anchor_id=f"original_print_Ad{number}_{component['id']}_{pattern['anchor_pts']}",
                physical_material_id=f"Ad{number}.print_contour.{component['id']}",
                advertisement_number=int(number), edge='visible_print_component_external_contour',
                uv=navigation.tolist(), native_pts=int(pattern['anchor_pts']),
                actual_pts=int(pattern['anchor_actual_pts']), curve={}, model_target=target,
                shot_id=review['shot_id'], load_preserved_source_template=True,
                source_raw_frame_reference=dict(current_reference),
                source_annotation_reference=dict(identity_authority),
                source_component_correspondence=dict(correspondence,
                    original_source_contour_id=component['id'], original_source_image=source_reference,
                    current_source_image=current_reference, scene_reference=dict(path=str(scene_path),
                        sha256=_digest(scene_path)), original_source_packet=dict(path=str(packet_path),
                        sha256=_digest(packet_path))),
                source_identity_confirmed=True, source_identity_reference_requires_current_shot=True,
                raw_source_navigation_not_output_UV=True,
                model_projection_or_fit_residual_used_for_identity=False))
    return seeds, checks
