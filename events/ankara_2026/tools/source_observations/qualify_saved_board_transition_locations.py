"""Restore existing RGB location estimators without changing observations.

Uses original source ROIs and pure estimator functions on existing PNGs in RAM.
No model, source video, new ROI, image output, Jacobian or fitting is used.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

OLD_QUALIFIER = Path(__file__).with_name('qualify_saved_board_transition_midpoints.py')
_spec = importlib.util.spec_from_file_location('frozen_board_midpoint_qualification', OLD_QUALIFIER)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
canonical, digest, reference, require, exact, save_new = (
    old.canonical, old.digest, old.reference, old.require, old.exact, old.save_new)
ROOT, H, PARENT, PARENT_SHA256 = old.ROOT, old.H, old.PARENT, old.PARENT_SHA256
ROW_INDICES = tuple(sorted((*old.ROW_INDICES, 0, 1, 2, 3, 4, 6, 19, 20)))
MODE_FIELDS = old.MODE_FIELDS
MODE = 'measured_transition_location'
RECEIPT_SCHEMA = 'rinkspace.saved-RGB-transition-location-revision.v1'
PACKET = ROOT / 'events/ankara_2026/observations/source_packets/board191_saved_RGB_location_estimators_20261011/packet.json'
RECEIPT = H / 'source_pixel_review_additional/board_top_contact_material_semantics/board191_saved_RGB_location_estimators_revision_20261011.json'
PAINT_PRODUCER = old.DATA / 'calibration/runs/primary823_main4_user_priority_review_20261005/extract_visible_source_paint.py'
PAINT_PRODUCER_SHA = '670ac13976b9dfd3a663ad52fd48ce3f395c861e1ec7714abed22a9bd1ba8ce4'
SUPPORT_PRODUCER = old.DATA / 'history/work_sessions/20261002_01a0fd26-33ba-7873-9aa9-bbd9d2ca1328/low5_repair/F2_inner_source.py'
SUPPORT_AST_SHA = '65649bffd6467eeeff15b5f37f024f894889a79a08cbd1b4d9a5f4c4989ecbf5'
RAIL_PRODUCER = old.DATA / 'observations/vision_training/continuous_review/read_clear_cut6_source_constraints_CPU.py'
RAIL_PRODUCER_SHA = 'f888a954e4decce7a9bc62da46d0e1facff5a1c44faab589be9e014bbaa32b3c'


def pure_functions(path, names, namespace):
    """Execute definitions only; original scripts have image-writing bodies."""
    text = Path(path).read_text(encoding='utf8')
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in functions} == set(names), 'Original estimator functions absent')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(text, n).encode('utf8')).hexdigest()
            for n in functions}


def source_image(row, images):
    path = row['source_raw_frame']
    if path not in images:
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == row['source_file_sha256'], 'Original native PNG changed')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.shape == (1080, 1920, 3), 'Original native frame shape changed')
        images[path] = image
    return images[path]


def measured_row(row):
    result = {k: deepcopy(row[k]) for k in ('id', 'split', 'curve', 'uv', 'normal',
        'normal_interval_px', 'sigma_px', 'source_raw_frame', 'source_file_sha256',
        'source_visible_paths', 'point_integration_mass') if k in row}
    result['actual_pts'] = int(row.get('actual_pts', row['pts']))
    return result


def review(index):
    if index in (0, 1, 2):
        finding = 'All four y350/380/410/440 windows show the unobscured blue panel external left transition; internal ISU text and athlete are outside the selected ROIs.'
    elif index == 3:
        return dict(reviewer='root', date='2026-10-11', basis='existing_native_RGB_opened_without_model',
            finding='x1713..1718 at y350/380/410/440 is the unobscured external left edge.')
    elif index == 4:
        finding = 'The image-left magenta panel border at x411..417, y570/640/710/760 is visible and unobscured; athlete and ANKARA lettering are to its right.'
    elif index == 6:
        finding = 'The blue panel external right edge at x108..110, y271/282/294 is visible below the broadcast graphic; no body, lettering or graphic covers these ROIs.'
    else:
        finding = 'The five x450/550/1250/1450/1650 upper blue-rim entry windows are unobscured; the athlete/cameraman are away from those windows. The inner blue-to-white interface is a separate stored diagnostic.'
    return dict(reviewer='solver_continuation', date='2026-10-11',
                basis='existing_native_RGB_opened_without_model', finding=finding)


def qualify_row(row, index, images):
    require(index in ROW_INDICES and row['split'] == 'fit', 'Only reviewed existing FIT rows may be revised')
    require(not any(k in row for k in MODE_FIELDS), 'An existing location operator must remain unchanged')
    if index in old.ROW_INDICES:
        result = old.qualify_row(row, index)
        result.update(location_estimator_kind='original_signed_transition_q50',
                      estimator_is_gradient_peak=False, estimator_is_not_verified_latent_edge_truth=True)
        return result
    raw = source_image(row, images)
    uv, normal, intervals = (np.asarray(row[k], float) for k in ('uv', 'normal', 'normal_interval_px'))
    require(uv.ndim == 2 and uv.shape[1:] == (2,) and uv.shape == normal.shape == intervals.shape,
            'Original point arrays disagree')
    proofs, producers = [], []
    if index in (0, 1, 2, 3):
        extraction = row['extraction']
        producer = Path(extraction['function_file'])
        require(digest(producer) == extraction['function_file_sha256'], 'Original grayscale estimator changed')
        require(extraction.get('model_or_camera_used_for_brackets_UV_or_identity',
                               extraction.get('model_or_camera_used_for_bracket_or_source_UV')) is False,
                'Model-selected scan is ineligible')
        namespace = {'np': np, 'cv2': cv2}
        ast_hashes = pure_functions(producer, ['gray_edge'], namespace)
        producers.append(dict(**reference(producer), function_AST_sha256=ast_hashes))
        samples = extraction['samples']
        require(len(samples) == len(uv), 'Original scan/point count changed')
        for p, scan in enumerate(samples):
            x0, y0, x1, y1 = scan['source_roi_xyxy']
            y = int(uv[p, 1]); require((y0, y1) == (y-2, y+2), 'Original five-row scan changed')
            answer = namespace['gray_edge'](raw, y, x0, x1+1)
            for k, values in (('uv', uv[p]), ('normal', normal[p]), ('normal_interval_px', intervals[p])):
                exact(answer[k], values, 'Replayed grayscale ' + k)
            exact(answer['q10_q50_q90_x'], scan['q10_q50_q90_x'], 'Saved grayscale q10/q50/q90')
            proofs.append(dict(point_index=p, original_profile_pointer=f'/rows/{index}/extraction/samples/{p}',
                source_profile_canonical_json_sha256=hashlib.sha256(canonical(scan)).hexdigest(),
                original_roi_xyxy=scan['source_roi_xyxy'], q10_q50_q90=answer['q10_q50_q90_x'],
                actual_source_uv=answer['uv'], original_estimator_replay_UV_and_support_bitexact=True))
        kind = 'integrated_negative_grayscale_gradient_mass_q50'
        identity = row['source_identity_authority']
    elif index == 4:
        require(digest(PAINT_PRODUCER) == PAINT_PRODUCER_SHA, 'Original magenta estimator changed')
        namespace = {'np': np}; ast_hashes = pure_functions(PAINT_PRODUCER, ['quantiles'], namespace)
        producers.append(dict(**reference(PAINT_PRODUCER), function_AST_sha256=ast_hashes,
            original_fixed_x_inclusive=[345, 465], original_fixed_y=[570, 640, 710, 760]))
        require(len(uv) == 4, 'Original magenta scan count changed')
        for p, y in enumerate((570, 640, 710, 760)):
            signal = raw[y-3:y+4, 345:466].mean(axis=0)
            score = (signal[:, 0]+signal[:, 2])/2-signal[:, 1]
            q = np.asarray(namespace['quantiles'](np.arange(345+.5, 465+.5), np.maximum(np.diff(score), 0.)))
            exact(uv[p], [q[1], float(y)], 'Original magenta UV')
            exact(normal[p], [1., 0.], 'Original magenta normal')
            exact(intervals[p], q[[0, 2]]-q[1], 'Original magenta support')
            proofs.append(dict(point_index=p, original_roi_xyxy=[345, y-3, 465, y+3],
                q10_q50_q90=q.tolist(), actual_source_uv=uv[p].tolist(),
                original_estimator_replay_UV_and_support_bitexact=True))
        kind = 'integrated_positive_magenta_chroma_diff_mass_q50'
        identity = row['source_identity_authority']
    elif index == 6:
        namespace = {'np': np, 'cv2': cv2}
        ast_hashes = pure_functions(SUPPORT_PRODUCER, ['support'], namespace)
        require(ast_hashes['support'] == SUPPORT_AST_SHA, 'Original plateau-transition estimator changed')
        producers.append(dict(**reference(SUPPORT_PRODUCER), function_AST_sha256=ast_hashes))
        samples = row['source_support']; require(len(samples) == len(uv), 'Original support count changed')
        for p, scan in enumerate(samples):
            require(scan['axis'] == 'x' and scan['side'] == 'visible_external_right', 'Different original material edge')
            answer = namespace['support'](raw, 'x', scan['fixed'], scan['native_search_support'])
            require(answer is not None, 'Original source contrast/crossing gate failed')
            point, band, profile = answer
            exact(point, uv[p], 'Original plateau q50 UV'); exact(band, intervals[p], 'Original plateau support')
            exact(normal[p], [1., 0.], 'Original plateau normal')
            exact(profile['q10_q50_q90_native'], scan['original_profile']['q10_q50_q90_native'], 'Original plateau quantiles')
            proofs.append(dict(point_index=p, original_profile_pointer=f'/rows/{index}/source_support/{p}/original_profile',
                source_profile_canonical_json_sha256=hashlib.sha256(canonical(scan['original_profile'])).hexdigest(),
                original_x_support_inclusive=scan['native_search_support'], original_fixed_y=scan['fixed'],
                q10_q50_q90=profile['q10_q50_q90_native'], actual_source_uv=point,
                original_estimator_replay_UV_and_support_bitexact=True))
        kind = 'smoothed_negative_gray_plateau_half_level_q50'
        identity = {'source_role': row['source_role'], 'original_candidate_curve_identity': row['original_candidate_curve_identity']}
    else:
        require(index in (19, 20) and digest(RAIL_PRODUCER) == RAIL_PRODUCER_SHA, 'Original blue-rim estimator changed')
        rgb = raw[:, :, ::-1].copy()
        require(hashlib.sha256(rgb.tobytes()).hexdigest() == row['source_native_RGB_sha256'], 'Original RGB pixels changed')
        namespace = {'np': np, 'rgb': rgb, 'sha': lambda v: hashlib.sha256(v).hexdigest()}
        ast_hashes = pure_functions(RAIL_PRODUCER, ['subpixel_peak', 'transition', 'rail'], namespace)
        producers.append(dict(**reference(RAIL_PRODUCER), function_AST_sha256=ast_hashes))
        require(len(row['measurements']) == len(uv), 'Original rail measurement count changed')
        for p, sample in enumerate(row['measurements']):
            answer = namespace['rail'](int(uv[p, 0]))
            for k, values in (('uv', uv[p]), ('normal', normal[p]), ('normal_interval_px', intervals[p])):
                exact(answer[k], values, 'Original blue-rim ' + k)
            require(answer['source_RGB_profile_sha256'] == sample['source_RGB_profile_sha256'], 'Original rail ROI pixels changed')
            proofs.append(dict(point_index=p, original_profile_pointer=f'/rows/{index}/measurements/{p}',
                source_profile_canonical_json_sha256=hashlib.sha256(canonical(sample)).hexdigest(),
                source_native_RGB_ROI_sha256=answer['source_RGB_profile_sha256'],
                original_search_scope=answer['search_scope'], actual_source_uv=answer['uv'],
                original_estimator_replay_UV_and_support_bitexact=True,
                estimator_definition='strongest positive blue-minus-red-green chroma gradient before blue maximum, parabolic subpixel peak',
                support_definition='contiguous gradient above one-quarter peak, expanded by half pixel',
                inner_blue_to_white_interface_is_not_this_location=True))
        kind = 'positive_blue_chroma_gradient_subpixel_peak'
        identity = row['physical_identity_authority']
    require(len(proofs) == len(uv), 'Original point proof count changed')
    return dict(original_row_index=index, id=row['id'], split=row['split'],
        source_row_canonical_json_sha256=hashlib.sha256(canonical(row)).hexdigest(),
        measured_boundary_row=measured_row(row), physical_identity_authority=deepcopy(identity),
        quality_review=review(index), location_estimator_kind=kind, source_location_point_proofs=proofs,
        original_estimator_producer_references=producers, estimator_is_not_verified_latent_edge_truth=True,
        source_sigma_and_optical_support_unchanged=True, sigma_is_not_new_measured_covariance=True)


def build_revision(parent_path=PARENT, packet_path=PACKET, receipt_path=RECEIPT):
    parent_ref = reference(parent_path)
    require(parent_ref['sha256'] == PARENT_SHA256, 'Exact original BOARD191 packet required')
    parent = json.loads(Path(parent_path).read_text(encoding='utf8'))
    require(len(parent['rows']) == 191, 'Original board source inventory changed')
    # The original 17-row reproducible qualification and receipt stay frozen.
    inherited_receipt = json.loads(old.RECEIPT.read_text(encoding='utf8'))
    require(reference(OLD_QUALIFIER) == inherited_receipt['reproducer_reference'], 'Frozen midpoint qualifier changed')
    images = {}
    records = [qualify_row(parent['rows'][i], i, images) for i in ROW_INDICES]
    native_refs = {}
    for record in records:
        row = record['measured_boundary_row']; path = row['source_raw_frame']
        if path not in native_refs: native_refs[path] = reference(path)
        require(native_refs[path]['sha256'] == row['source_file_sha256'], 'Original native image changed')
        record['source_row_reference'] = dict(parent_ref, pointer=f"/rows/{record['original_row_index']}")
    require(len(records) == 25 and sum(len(r['measured_boundary_row']['uv']) for r in records) == 83,
            'Exact reviewed 25-row/83-point scope required')
    receipt = dict(schema=RECEIPT_SCHEMA, parent_packet_reference=parent_ref,
        reproducer_reference=reference(__file__), inherited_midpoint_qualification_reference=reference(old.RECEIPT),
        inherited_frozen_midpoint_qualifier_reference=reference(OLD_QUALIFIER), changed_row_indices=list(ROW_INDICES),
        changed_row_ids=[r['id'] for r in records], points=83, rows=records,
        native_image_references=list(native_refs.values()),
        existing_native_PNG_decoded_in_RAM_for_original_estimator_reproduction=True,
        no_video_decode_or_new_source_image=True, new_ROI_source_point_or_estimator_selected=False,
        model_state_projection_or_residual_used_for_qualification=False,
        all_original_UV_sigma_normal_optical_support_split_and_point_order_unchanged=True,
        human_support_and_HELD_roles_unchanged=True,
        qualification_is_soft_source_estimator_location_not_accuracy_or_hard_guard=True)
    save_new(receipt_path, receipt); receipt_ref = reference(receipt_path)
    revised = deepcopy(parent)
    for i in ROW_INDICES:
        revised['rows'][i][MODE_FIELDS[0]] = MODE
        revised['rows'][i][MODE_FIELDS[1]] = receipt_ref
    revised['source_location_operator_revision'] = dict(parent_packet_reference=parent_ref,
        receipt_reference=receipt_ref, changed_row_indices=list(ROW_INDICES),
        changed_row_ids=receipt['changed_row_ids'], all_original_source_rows_except_two_location_fields_unchanged=True)
    strip = lambda r: {k: v for k, v in r.items() if k not in MODE_FIELDS}
    require(all(canonical(strip(a)) == canonical(strip(b)) for a, b in zip(parent['rows'], revised['rows'])),
            'Original source fields changed')
    require(all(canonical(a) == canonical(b) for i, (a, b) in enumerate(zip(parent['rows'], revised['rows'])) if i not in ROW_INDICES),
            'Unselected/manual/HELD source changed')
    save_new(packet_path, revised)
    return dict(packet=reference(packet_path), receipt=receipt_ref, changed_rows=25, points=83,
                original_source_fields_except_two_mode_fields_unchanged=True, new_source_images=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--packet', type=Path, default=PACKET)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(build_revision(args.parent, args.packet, args.receipt), ensure_ascii=False))


if __name__ == '__main__':
    main()
