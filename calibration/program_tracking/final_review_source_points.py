"""Display the frozen dataset and retained original hard-guard queries.

This final-review consumer never installs a factor, changes a support or
restores an old FIT objective. It draws only existing original query UV on
their own exposure. Dense derived guards and out-of-programme wide UV stay
out of this display archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .current_source_points import (CurrentSourcePoints, file_sha,
    freeze_current_source_point_inputs, save_current_source_point_binding)
from .strict_model_scope import authenticate_source_reference
from .strict_source_fit_dataset import SCHEMA
from calibration.solver_runtime.strict_dataset_read_runtime import load_strict_dataset_rows_once


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def same_reference(left, right):
    return (Path(left['path']).resolve() == Path(right['path']).resolve()
            and left['sha256'] == right['sha256'])


def read_guard_query_inputs(manifest, truth, actual_pts, source):
    """Authenticate retained authorities, without constructing any old factor."""
    allowed = set(map(int, actual_pts))
    native = manifest['highest_direct_human_hard_truth']['native']
    confirmed = manifest['highest_user_confirmed_native_sources']
    contracts = dict(native_guard_only=native, confirmed_guard_only=confirmed)
    rows, receipts = [], []
    packets = {}
    for role, contract in contracts.items():
        family = 'native' if role == 'native_guard_only' else 'confirmed'
        reference = contract['authority']
        authenticate_source_reference(reference)
        authority = read(reference['path'])
        saved = truth['families'][family]
        if not same_reference(saved['authority'], reference):
            raise ValueError('Final display authority differs from retained hard guard')
        records = authority['records']
        ids = [r['id'] for r in records]
        if len(ids) != len(set(ids)) or ids != saved['group_ids']:
            raise ValueError('Every retained original hard-guard group must be enumerated in order')
        if family == 'native':
            if (authority['schema'] != 'rinkspace.registered-direct-human-hard-truth-authorities.v1'
                    or authority['source_video_sha256'] != source['video_sha256']
                    or authority['time_base'] != source['time_base']
                    or contract['source_groups'] != len(records)):
                raise ValueError('Native hard-guard source/clock/inventory differs')
        elif (authority['schema'] != 'rinkspace.user-confirmed-native-source-priority-authorities.v1'
                or contract['ids'] != ids):
            raise ValueError('Confirmed original hard-guard authority differs')
        receipt = dict(family=role, complete_active_input=True, authority_reference=reference,
            display_role='guard_only', not_in_FIT=True, used_as_new_FIT=False,
            original_groups=len(records), original_query_points=0, displayed_original_query_points=0,
            missing_original_queries=[], original_queries=[],
            source_UV_support_sigma_and_operator_not_modified=True,
            derived_dense_guard_queries_included=False, wide_outside_programme_included=False)
        for ordinal, record in enumerate(records):
            packet_ref = record['packet_reference']
            key = (str(Path(packet_ref['path']).resolve()), packet_ref['sha256'])
            if key not in packets:
                authenticate_source_reference(packet_ref)
                packets[key] = read(packet_ref['path'])
            original = packets[key]['rows'][int(record['original_row_index'])]
            canonical = json.dumps(original, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode('utf8')
            source_id = record.get('source_row_id', record['id'])
            if (hashlib.sha256(canonical).hexdigest() != record['source_row_canonical_json_sha256']
                    or original['id'] != source_id):
                raise ValueError('Original hard-guard source row identity changed')
            stamp = int(record['actual_pts'])
            uv = np.asarray(record['uv'], np.float64)
            original_uv = np.asarray(original['uv'], np.float64)
            if (stamp not in allowed or int(original['actual_pts']) != stamp
                    or uv.ndim != 2 or uv.shape[1:] != (2,)
                    or uv.tobytes() != original_uv.tobytes() or uv.shape != original_uv.shape
                    or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= [1920, 1080])):
                raise ValueError('Only exact original guard UV on its admitted programme exposure may be shown')
            receipt['original_query_points'] += len(uv)
            model_query = any(value.get(flag) is True for value in (record, original)
                for flag in ('model_projection_used_to_generate_or_select_uv',
                             'model_or_RGB_transport_used_to_generate_UV', 'source_uv_is_model_projection'))
            if model_query:
                receipt['missing_original_queries'].append(dict(id=record['id'], points=len(uv),
                    reason='original UV is declared model-derived; no display point invented'))
                continue
            if family == 'native' and (record.get('direct_human_GT') is not True
                    or record.get('model_or_RGB_transport_used_to_generate_UV') is not False):
                raise ValueError('Direct original human query authority required')
            if family == 'confirmed' and record.get('model_projection_used_to_generate_or_select_uv') is not False:
                raise ValueError('Original confirmed source query authority required')
            if 'source_raw_frame' in record:
                authenticate_source_reference(dict(path=record['source_raw_frame'],
                                                   sha256=record['source_file_sha256']))
            operator = record.get('original_source_operator', original.get('curve_observation_kind'))
            kind = record.get('curve_observation_kind', original.get('curve_observation_kind'))
            horizontal = (kind in ('horizontal_extent', 'clipped_horizontal_extent')
                          or operator in ('printed_component_horizontal_extent',
                                          'original_complete_curve_horizontal_extent'))
            # Preserve existing scalar-operator context, especially the five
            # x-only R4/Ad23 queries. The stored y is never promoted to a new
            # measured axis, and no midpoint/support is constructed here.
            fields = ('curve', 'curve_observation_kind', 'original_source_operator',
                'original_source_support', 'normal', 'normal_interval_px', 'extent_side',
                'source_y_is_not_a_world_point_correspondence',
                'no_new_tangent_endpoint_or_context_y_hard_constraint', 'raw_support_companion_of')
            query = dict(id=record['id'], original_group_index=ordinal, actual_pts=stamp,
                native_pts=int(record['pts']), original_uv=record['uv'], points=len(uv),
                packet_reference=packet_ref, original_row_index=record['original_row_index'],
                source_row_canonical_json_sha256=record['source_row_canonical_json_sha256'],
                original_operator_fields={name: record.get(name, original.get(name)) for name in fields
                    if name in record or name in original},
                active_pixel_axes=['x'] if horizontal else None,
                context_pixel_axes=['y'] if horizontal else [],
                original_query_location_display_only=True, new_2D_measurement_claimed=False,
                not_in_FIT=True, missing_axis_or_support_filled=False)
            receipt['original_queries'].append(query)
            receipt['displayed_original_query_points'] += len(uv)
            rows.append((stamp, uv))
        if family == 'native' and receipt['original_query_points'] != contract['source_UV_queries']:
            raise ValueError('Native original query count differs from active guard')
        receipts.append(receipt)
    return rows, receipts


def prepare_final_display_inputs(state_directory, *, input_manifest):
    """Read-only complete-state/authority preparation; no source pack scan."""
    state = Path(state_directory)
    identity = read(state / 'state_identity.json')
    manifest_path = Path(input_manifest)
    if file_sha(manifest_path) != identity['input_manifest_sha256']:
        raise ValueError('Final display input manifest differs from complete q')
    manifest = read(manifest_path)
    checkpoint = state / 'complete_state_UNACCEPTED.npz'
    if file_sha(checkpoint) != identity['state_npz_sha256']:
        raise ValueError('Final complete-state file changed')
    with np.load(checkpoint, allow_pickle=False) as values:
        if hashlib.sha256(values['q'].tobytes()).hexdigest() != identity['q_sha256']:
            raise ValueError('Final complete q bytes changed')
    camera_path = Path(identity['camera_path'])
    if file_sha(camera_path) != identity['camera_sha256']:
        raise ValueError('Final saved camera changed')
    strict = manifest['strict_source_dataset']
    if (strict.get('schema') != 'rinkspace.strict-reviewed-finite-programme-source.v1'
            or manifest.get('legacy_pixel_objective_replaced') is not True
            or strict.get('archived_pixel_factors_in_FIT') is not False
            or strict.get('all_pixel_families_routed_to_strict_input') is not True):
        raise ValueError('Guard-only display roles require the actual strict dataset FIT route')
    reference = strict['finite_dataset_contract']['dataset_reference']
    authenticate_source_reference(reference)
    dataset = read(reference['path'])
    with np.load(camera_path, allow_pickle=False) as camera:
        actual = np.unique(camera['exposure_representative_pts'].astype(np.int64))
        metadata = json.loads(str(camera['metadata_json'].item()))
    if (dataset['schema'] != SCHEMA or not np.array_equal(actual, dataset['expected_actual_pts'])
            or dataset['source']['video_sha256'] != metadata['source_sha256']
            or dataset['source']['time_base'] != metadata['source_time_base']
            or dataset['source']['pixel_domain'] != 'native1920x1080'):
        raise ValueError('Final dataset and camera source/exposure clock differ')
    truth = read(state / 'truth_restoration_state.json')
    if (truth['complete_q_sha256'] != identity['q_sha256']
            or not same_reference(truth['complete_state_reference'], dict(path=str(checkpoint),
                sha256=identity['state_npz_sha256']))
            or not same_reference(truth['source_input_manifest_reference'], dict(path=str(manifest_path),
                sha256=identity['input_manifest_sha256']))):
        raise ValueError('Displayed hard guards do not belong to this complete state')
    guards, receipts = read_guard_query_inputs(manifest, truth, actual, dataset['source'])
    return dict(identity=identity, dataset_reference=reference, dataset=dataset,
                camera_path=camera_path, actual_pts=actual, guards=guards, guard_receipts=receipts)


def freeze_final_review_source_points(shared_path, binding_path, state_directory, *, input_manifest):
    """Freeze all dataset pixels plus retained original guard queries once."""
    if Path(shared_path).exists() or Path(binding_path).exists():
        raise FileExistsError('Preserve earlier final display archives')
    prepared = prepare_final_display_inputs(state_directory, input_manifest=input_manifest)
    dataset = prepared['dataset']
    fit, held, validation = load_strict_dataset_rows_once(prepared['dataset_reference'])
    expected = dict(fit=int(dataset['fit_points']), held=int(dataset['held_points']))
    if (validation.get('full_source_pack_passes') != 1
            or validation.get('every_original_row_proof_UV_sigma_clock_and_role_checked') is not True
            or validation.get('source_pixels_sigma_roles_locators_and_groups_unchanged') is not True
            or any(validation.get(role + '_points') != count for role, count in expected.items())):
        raise ValueError('Complete single-pass dataset validation receipt required')
    receipts = prepared['guard_receipts']
    summary = dict(family='final_complete_current_source_queries', complete_active_input=True,
        dataset_reference=prepared['dataset_reference'], single_pass_source_validation=validation,
        display_original_points_by_role=dict(expected, **{r['family']:r['original_query_points'] for r in receipts}),
        fit_points=expected['fit'], held_points=expected['held'], fitting_roles=['fit'],
        held_used_in_fit=False, guard_queries_used_as_new_FIT=False,
        held_missing_sigma_displayed_points=0, held_source_sigma_not_filled=True,
        missing_support_points_not_drawn=int(dataset['pending_support_points']),
        pending_named_points_not_drawn=int(dataset.get('pending_named_points', 0)),
        pending_or_retired_FIT_pixels_added=False, dense_derived_guard_queries_added=False,
        out_of_programme_wide_pixels_added=False, all_current_source_UV_unchanged=True,
        exact_same_exposure_pixel_coordinates_share_one_marker=True)
    summary['all_original_guard_queries_displayed'] = not any(r['missing_original_queries'] for r in receipts)
    summary['display_original_source_queries'] = sum(summary['display_original_points_by_role'].values())
    def groups():
        for role, rows in [('fit', fit), ('held', held)]:
            consumed = 0
            for row in rows:
                uv, sigma = np.asarray(row['uv']), np.asarray(row['sigma'])
                if (row['split'] != role or row['scope'] not in ('ice', 'board')
                        or uv.shape != (row['points'], 2) or sigma.shape != (len(uv),)
                        or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)
                        or (role == 'fit' and not np.isfinite(sigma).all())):
                    raise ValueError('Exact original admitted dataset role/UV/sigma required')
                consumed += len(uv)
                if role == 'held':
                    summary['held_missing_sigma_displayed_points'] += int(np.isnan(sigma).sum())
                yield int(row['actual_pts']), uv
            if consumed != expected[role]:
                raise ValueError('Every original dataset point must be displayed')
        yield from prepared['guards']
    shared = freeze_current_source_point_inputs(shared_path, groups(), actual_pts=prepared['actual_pts'],
        input_manifest=input_manifest, family_receipts=[summary, *receipts])
    binding = save_current_source_point_binding(binding_path, prepared['camera_path'], shared,
        q_sha256=prepared['identity']['q_sha256'], input_manifest=input_manifest)
    CurrentSourcePoints(binding_path, prepared['camera_path'], q_sha256=prepared['identity']['q_sha256'])
    return dict(binding, original_queries_by_role=summary['display_original_points_by_role'],
                original_source_queries=summary['display_original_source_queries'],
                missing_guard_queries=[m for r in receipts for m in r['missing_original_queries']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-directory', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--shared-points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_final_review_source_points(args.shared_points, args.output,
        args.state_directory, input_manifest=args.input_manifest), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
