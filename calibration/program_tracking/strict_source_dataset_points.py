"""Freeze admitted dataset source pixels for the existing review renderer.

This is a display consumer of the authenticated source dataset. It does not
fit, requalify, project, subsample or fill source observations. Equal current
pixel coordinates share one marker, as in the normal solver point archive.
FIT is the default; explicitly included HELD pixels remain display-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .current_source_points import (
    file_sha, freeze_current_source_point_inputs, save_current_source_point_binding,
)
from .strict_model_scope import authenticate_source_reference
from .strict_source_fit_dataset import SCHEMA
from calibration.solver_runtime.strict_dataset_read_runtime import load_strict_dataset_rows_once


def freeze_dataset_source_point_inputs(path, dataset_reference, *, input_manifest,
                                      camera_path, include_held=False):
    """Seal current UV once; optional HELD display never promotes it to FIT."""
    authenticate_source_reference(dataset_reference)
    dataset = json.loads(Path(dataset_reference['path']).read_text(encoding='utf-8-sig'))
    if dataset.get('schema') != SCHEMA:
        raise ValueError('Authenticated frozen finite-source dataset required')
    actual = np.asarray(dataset['expected_actual_pts'], np.int64)
    source = dataset['source']
    with np.load(camera_path, allow_pickle=False) as camera:
        camera_actual = np.unique(camera['exposure_representative_pts'].astype(np.int64))
        metadata = json.loads(str(camera['metadata_json'].item()))
    if (not np.array_equal(actual, camera_actual)
            or source['video_sha256'] != metadata['source_sha256']
            or source['time_base'] != metadata['source_time_base']
            or source['pixel_domain'] != 'native1920x1080'):
        raise ValueError('Dataset and camera original source/exposure clock differ')
    roles = ['fit', 'held'] if include_held else ['fit']
    expected = dict(fit=int(dataset['fit_points']), held=int(dataset['held_points']))
    receipt = dict(family=('frozen_current_source_dataset_review' if include_held else
                          'frozen_finite_source_dataset_fit'), complete_active_input=True,
        dataset_reference=dict(dataset_reference), fit_points=expected['fit'],
        held_points=expected['held'], held_points_not_drawn=0 if include_held else expected['held'],
        display_roles=roles, fitting_roles=['fit'], held_used_in_fit=False,
        display_original_points_by_role={role: expected[role] for role in roles},
        display_original_source_points=sum(expected[role] for role in roles),
        displayed_source_rows_by_role={role: 0 for role in roles},
        displayed_groups_by_role={role: 0 for role in roles},
        held_missing_sigma_displayed_points=0, held_source_sigma_not_filled=True,
        missing_support_points_not_drawn=int(dataset['pending_support_points']),
        pending_named_points_not_drawn=int(dataset.get('pending_named_points', 0)),
        pending_or_historical_points_included=False,
        original_source_UV_unchanged=True, all_admitted_FIT_points_enumerated=True,
        all_current_HELD_points_enumerated=bool(include_held),
        only_identical_current_pixel_coordinates_share_display_marker=True)

    def groups():
        fit, held, validation = load_strict_dataset_rows_once(dataset_reference)
        if (validation.get('schema') != 'rinkspace.single-pass-strict-dataset-read.v1'
                or validation.get('full_source_pack_passes') != 1
                or validation.get('fit_points') != expected['fit']
                or validation.get('held_points') != expected['held']
                or validation.get('every_original_row_proof_UV_sigma_clock_and_role_checked') is not True
                or validation.get('source_pixels_sigma_roles_locators_and_groups_unchanged') is not True):
            raise ValueError('Complete single-pass source validation receipt required')
        compact = dict(fit=fit, held=held)
        receipt['single_pass_source_validation'] = validation
        for role in roles:
            consumed = 0
            for row in compact[role]:
                if row['split'] != role or row['scope'] not in ('ice', 'board'):
                    raise ValueError('Only current admitted model-scope ' + role.upper() + ' observations may be drawn')
                uv = np.asarray(row['uv'])
                sigma = np.asarray(row['sigma'])
                points = row['points']
                if (uv.shape != (points, 2) or sigma.shape != (len(uv),)
                        or np.isinf(sigma).any() or np.any(sigma[np.isfinite(sigma)] <= 0)
                        or (role == 'fit' and not np.isfinite(sigma).all())):
                    raise ValueError('Exact measured source support required; missing sigma is HELD display only')
                # A shared target group can hold several original rows.
                # Count original source rows, with their exact UV ranges,
                # rather than relabelling the group as one measurement.
                offset = 0
                for locator in row['source_locators']:
                    lo, hi = locator['group_point_range']
                    if lo != offset or hi <= lo or hi > len(uv):
                        raise ValueError('Exact compact source row locator coverage required')
                    offset = hi
                if offset != len(uv):
                    raise ValueError('Exact compact source row locator coverage required')
                source_rows = len(row['source_locators'])
                receipt['displayed_source_rows_by_role'][role] += source_rows
                receipt['displayed_groups_by_role'][role] += 1
                if role == 'held':
                    receipt['held_missing_sigma_displayed_points'] += int(np.isnan(sigma).sum())
                consumed += len(uv)
                yield int(row['actual_pts']), uv
            if consumed != expected[role]:
                raise ValueError('Every admitted dataset ' + role.upper() + ' point must be enumerated')

    result = freeze_current_source_point_inputs(path, groups(), actual_pts=actual,
        input_manifest=input_manifest, family_receipts=[receipt])
    return dict(result, display_roles=roles,
        original_points_by_role=receipt['display_original_points_by_role'],
        original_source_points=receipt['display_original_source_points'], held_used_in_fit=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--camera', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--shared-points', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--q-sha256', required=True)
    parser.add_argument('--include-held', action='store_true',
                        help='Display all current FIT and HELD source pixels; HELD never enters fitting')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier state point bindings')
    reference = dict(path=str(args.dataset.resolve()), sha256=file_sha(args.dataset))
    shared = freeze_dataset_source_point_inputs(args.shared_points, reference,
        input_manifest=args.input_manifest, camera_path=args.camera, include_held=args.include_held)
    binding = save_current_source_point_binding(args.output, args.camera, shared,
        q_sha256=args.q_sha256, input_manifest=args.input_manifest)
    print(json.dumps(binding, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
