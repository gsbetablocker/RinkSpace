"""Compact full-clock locators for immutable source review parts.

This keeps measured native UV in their existing archives. A coverage index
records which actual source frames were read, not that all identities passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .strict_model_scope import authenticate_source_reference, file_sha, verify_source_identities_unchanged


def _load(reference):
    authenticate_source_reference(reference)
    with np.load(reference['path'], allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def consolidate_source_review_coverage(native_references, retirement_references,
        *, identity_reference, output_path):
    """Index exact native frames and old-point retirements without copying UV.

    References are ordered from earlier measurements to newer explicit source
    deltas. A repeated native frame must retain its exact clock and BGR hash.
    Its newer summary has an explicit additional locator; every earlier
    source row remains in the compact history. Explicit human role-absence
    decisions stay preserved.
    """
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError('Do not replace an earlier source review coverage index')
    identities = {}
    for ref in [identity_reference, *native_references, *retirement_references]:
        path, current_identity = authenticate_source_reference(ref)
        identities[str(path)] = current_identity
    identity = json.loads(Path(identity_reference['path']).read_text(encoding='utf8'))
    camera_reference = dict(path=identity['camera_path'], sha256=identity['camera_sha256'])
    path, current_identity = authenticate_source_reference(camera_reference)
    identities[str(path)] = current_identity
    with np.load(camera_reference['path'], allow_pickle=False) as camera:
        native_pts = camera['pts'].copy()
        actual_pts = camera['exposure_representative_pts'].copy()
        camera_metadata = json.loads(str(camera['metadata_json'].item()))
    source = dict(video_sha256=camera_metadata['source_sha256'],
        time_base=camera_metadata['source_time_base'], pixel_domain='native1920x1080')
    if (native_pts.ndim != 1 or actual_pts.shape != native_pts.shape
            or not np.array_equal(native_pts, np.unique(native_pts))):
        raise ValueError('Authenticated exact native source clock required')
    size = len(native_pts)
    part_index = np.full(size, -1, np.int16)
    local_index = np.full(size, -1, np.int32)
    bgr = np.full(size, b'', dtype='S64')
    measured = np.zeros(size, bool)
    board_checked = np.zeros(size, bool)
    status_counts = np.zeros((size, 5), np.int32)
    candidate_counts = np.zeros(size, np.int32)
    representative = native_pts == actual_pts
    history = [[] for _ in range(size)]
    lookup = {int(p):i for i,p in enumerate(native_pts)}
    for number, reference in enumerate(native_references):
        values = _load(reference)
        metadata = json.loads(str(values['metadata_json'].item()))
        if (metadata.get('schema') != 'rinkspace.strict-source-native-alias-material-review.v1'
                or metadata['source'] != source
                or metadata.get('new_FIT_observations_created') is not False
                or metadata['expected_native_pts_reference']['sha256'] != identity_reference['sha256']):
            raise ValueError('Native review part differs from the exact original source')
        pts = values['native_pts']
        if not np.array_equal(pts, np.unique(pts)):
            raise ValueError('A native review part must contain each source frame exactly once')
        local = {int(p):i for i,p in enumerate(pts)}
        for i, p in enumerate(pts):
            index = lookup.get(int(p))
            if index is None or int(values['actual_pts'][i]) != int(actual_pts[index]):
                raise ValueError('Native source exposure alias mapping changed')
            sha = values['BGR_sha256'][i]
            if len(sha) != 64 or (bgr[index] and bgr[index] != sha):
                raise ValueError('One native source frame has competing actual BGR bytes')
            if bool(values['representative'][i]) != bool(representative[index]):
                raise ValueError('Native source representative identity changed')
            part_index[index] = number
            local_index[index] = i
            history[index].append((number,i))
            bgr[index] = sha
            measured[index] = bool(values['native_material_checked'][i])
            status_counts[index] = 0
            candidate_counts[index] = 0
            board_checked[index] = False
        from .compact_source_json import unpack_source_json
        boards = unpack_source_json(values['alias_board_checks_json'])
        if len(boards) != len(values['alias_native_pts']):
            raise ValueError('Native board-check rows differ from actual alias frames')
        for i, p in enumerate(values['alias_native_pts']):
            index = lookup[int(p)]
            if int(p) not in local or representative[index]:
                raise ValueError('Alias checks refer to an undeclared native source frame')
            lo, hi = map(int, values['alias_candidate_offsets'][i:i+2])
            status = values['canonical_candidate_status'][lo:hi]
            if status.dtype.kind not in 'iu' or np.any(status > 4):
                raise ValueError('Exact native source material decisions required')
            candidate_counts[index] = hi-lo
            status_counts[index] = np.bincount(status, minlength=5)
            row = boards[i]
            if int(row['native_pts']) != int(p) or int(row['actual_pts']) != int(actual_pts[index]):
                raise ValueError('Native advertisement material check source clock differs')
            board_checked[index] = row.get('board_current_native_material_checked') is True
    if not retirement_references:
        raise ValueError('Original point retirement parts required')
    disposition = None
    old_reference = None
    old_pts = offsets = None
    families = set()
    for reference in retirement_references:
        values = _load(reference)
        metadata = json.loads(str(values['metadata_json'].item()))
        if (metadata.get('schema') != 'rinkspace.strict-source-old-pixel-retirement.v1'
                or metadata['source'] != source
                or metadata.get('default_legacy_admission') != 'forbidden'
                or metadata.get('unknown_is_not_negative_training_truth') is not True):
            raise ValueError('Strict old-point retirement source provenance required')
        if disposition is None:
            old_reference = metadata['original_pixel_reference']
            path, current_identity = authenticate_source_reference(old_reference)
            identities[str(path)] = current_identity
            old_pts, offsets = values['exposure_pts'], values['exposure_offsets']
            disposition = np.zeros_like(values['old_pixel_disposition'])
        if (metadata['original_pixel_reference'] != old_reference
                or not np.array_equal(old_pts, values['exposure_pts'])
                or not np.array_equal(offsets, values['exposure_offsets'])
                or disposition.shape != values['old_pixel_disposition'].shape):
            raise ValueError('Retirement parts use different original point locators')
        current = values['old_pixel_disposition']
        if current.dtype.kind not in 'iu' or np.any(current > 4):
            raise ValueError('Strict original source point retirement decisions required')
        update = (current > 0) & (disposition != 4)
        disposition[update] = current[update]
        families.update(metadata['legacy_pixel_families'])
    if (not np.array_equal(old_pts, np.unique(actual_pts))
            or offsets.shape != (len(old_pts)+1,) or offsets[0] != 0
            or offsets[-1] != len(disposition)):
        raise ValueError('Retirement parts omit the original complete exposure clock')
    all_native_processed = bool(np.all(part_index >= 0))
    aliases = ~representative
    metadata = dict(schema='rinkspace.strict-source-review-coverage-index.v1', source=source,
        native_review_parts=list(native_references), old_retirement_parts=list(retirement_references),
        identity_reference=identity_reference, camera_clock_reference=camera_reference,
        original_pixel_reference=old_reference, legacy_pixel_families=sorted(families),
        native_UV_stays_in_original_review_parts=True, unreviewed_legacy_default='excluded',
        all_previous_native_review_rows_preserved=True,
        repeated_native_reviews_are_independent_deltas=True,
        all_native_frames_RGB_processed=all_native_processed,
        all_native_aliases_material_measurement_attempted=bool(np.all(measured[aliases])),
        all_native_aliases_admitted_advertisements_material_checked=bool(np.all(board_checked[aliases])),
        alias_physical_identity_pending_candidate_count=int(status_counts[:,0].sum()),
        all_old_points_received_source_disposition=bool(np.all(disposition > 0)),
        unknown_or_scope_excluded_is_not_negative_training_truth=True,
        source_processing_is_not_manual_visual_completion=True, solver_started=False)
    verify_source_identities_unchanged(identities)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_offsets=np.r_[0,np.cumsum([len(items) for items in history])].astype(np.int64)
    history_locators=np.asarray([item for items in history for item in items],np.int32).reshape(-1,2)
    np.savez_compressed(output_path, native_pts=native_pts, actual_pts=actual_pts,
        BGR_sha256=bgr, representative=representative, native_review_part_index=part_index,
        native_review_local_index=local_index, native_material_measurement_attempted=measured,
        native_review_history_offsets=history_offsets,
        native_review_history_part_index=history_locators[:,0],
        native_review_history_local_index=history_locators[:,1],
        alias_admitted_advertisement_material_checked=board_checked,
        alias_candidate_count=candidate_counts, alias_candidate_status_counts=status_counts,
        old_pixel_disposition=disposition, exposure_pts=old_pts, exposure_offsets=offsets,
        metadata_json=json.dumps(metadata, ensure_ascii=False, separators=(',', ':')))
    return dict(path=str(output_path), sha256=file_sha(output_path),
        processed_native_frames=int(np.sum(part_index >= 0)), expected_native_frames=size,
        retired_original_points=int(np.sum(disposition > 0)),
        pending_alias_identity_checks=int(status_counts[:,0].sum()),
        all_native_frames_RGB_processed=all_native_processed,
        source_processing_is_not_manual_visual_completion=True, solver_started=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Index exact native source coverage without copying measured UV')
    parser.add_argument('--native-review', nargs=2, action='append', required=True, metavar=('PATH','SHA256'))
    parser.add_argument('--retirement', nargs=2, action='append', required=True, metavar=('PATH','SHA256'))
    parser.add_argument('--identity', nargs=2, required=True, metavar=('PATH','SHA256'))
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    ref = lambda item:dict(path=item[0], sha256=item[1])
    receipt = consolidate_source_review_coverage(list(map(ref,options.native_review)),
        list(map(ref,options.retirement)), identity_reference=ref(options.identity), output_path=options.output)
    print(json.dumps(receipt, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
