"""Combine authenticated source packs without restoring any legacy UV family.

Only measurements already admitted by the exact-source consumer are copied.
This operation neither reviews missing frames nor infers new source pixels.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack, _target_key,
    _measured_board_readout_support,
)
from .strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA


def _slice_row(row, indices):
    result = deepcopy(row)
    indices = np.asarray(indices, np.int64)
    count = len(row['uv'])
    for name in ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
            'source_side_quantiles_uv', 'original_point_indices', 'normal',
            'source_candidate_indices', 'original_candidate_indices',
            'anchor_point_indices', 'source_query_indices', 'candidate_sides', 'actual_source_normal_uv'):
        if name not in result:
            continue
        value = np.asarray(result[name])
        if value.ndim and len(value) == count:
            result[name] = value[indices].copy()
    uncertainty = result.get('source_readout_uncertainty', {})
    for name in ('actual_readout_blur_interval_px', 'normal_interval_px',
            'quantization_margin_px'):
        if name in uncertainty:
            value = np.asarray(uncertainty[name])
            if value.ndim and len(value) == count:
                uncertainty[name] = value[indices].copy()
    if 'candidate_local_source_clearance' in result:
        source_ids = set(map(int, result['original_point_indices']))
        local=result['candidate_local_source_clearance']
        if isinstance(local,dict):
            # Board readers retain exact source-anchor indices as JSON keys;
            # ice normalizers retain a list with explicit candidate indices.
            result['candidate_local_source_clearance']={key:proof for key,proof in local.items()
                if int(key) in source_ids}
        else:
            result['candidate_local_source_clearance'] = [proof for proof in local if
                int(proof.get('original_candidate_index', proof.get('candidate_index', -1))) in source_ids]
    restored_scope=result.get('source_scope_restored_from_authenticated_actual_graph_nodes')
    if restored_scope:
        for name in ('original_candidate_indices','frozen_original_scope_flags'):
            value=np.asarray(restored_scope[name])
            if len(value)!=count:raise ValueError('Recovered source scope locator count differs')
            restored_scope[name]=value[indices].tolist()
    return result


def merge_strict_absolute_source_packs(references, output_path):
    """Merge full-clock partial packs and deltas, preserving explicit HELD.

    ``references`` are immutable ``{path, sha256}`` records. Repeated pixels
    are identified by exact source UV and physical target, never a distance
    tolerance. For the same pixel, a measured working support is preferred to
    a missing support; equally complete readouts retain the first record.
    The complete original packs and their source indices remain referenced.
    """
    references = [dict(path=str(Path(r['path']).resolve()), sha256=r['sha256'])
        for r in references]
    if not references:
        raise ValueError('At least one authenticated strict source pack required')
    packs = [StrictAbsoluteSourcePack(r['path'], expected_sha256=r['sha256'])
        for r in references]
    source = packs[0].metadata['source']
    expected = packs[0].metadata['expected_actual_pts']
    for pack in packs[1:]:
        if pack.metadata['source'] != source or pack.metadata['expected_actual_pts'] != expected:
            raise ValueError('Source packs must share the same video and complete exposure clock')
    original_refs = {}
    for ref in references + [r for pack in packs for r in pack.metadata['original_references']]:
        path = str(Path(ref['path']).resolve())
        if path in original_refs and original_refs[path]['sha256'] != ref['sha256']:
            raise ValueError('Source packs cite conflicting bytes for one authority')
        original_refs[path] = dict(path=path, sha256=ref['sha256'])
    writer = StrictAbsolutePackWriter(output_path, source=source,
        expected_actual_pts=expected, original_references=list(original_refs.values()),
        legacy_pixel_families=set().union(*(p.metadata['legacy_pixel_families'] for p in packs)))
    held_features = {(r['actual_pts'], r['feature_id']) for p in packs for r in p.rows
        if r['split'] == 'held'}
    held_materials = {(r['actual_pts'], r['physical_material_id']) for p in packs for r in p.rows
        if r['split'] == 'held'}
    frames = {}
    processed = set()
    rows = []
    choices, candidate_pixels = {}, {}
    duplicate_count = 0
    for pack_number, pack in enumerate(packs):
        processed.update(pack.metadata['processed_actual_pts'])
        for pts, binding in pack.frame_bindings.items():
            if pts in frames and frames[pts] != binding:
                raise ValueError('Source packs bind one exposure to differing native pixels')
            frames[pts] = binding
        for row_number, header in enumerate(pack.rows):
            lo, hi = map(int, pack.values['row_offsets'][row_number:row_number+2])
            row = dict(deepcopy(header), **{key:pack.values[key][lo:hi].copy() for key in
                ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
                    'source_side_quantiles_uv', 'original_point_indices')})
            row['strict_pack_origin'] = dict(source_pack=references[pack_number],
                source_row_index=row_number)
            if row['scope'] == 'board' and not np.isfinite(row['sigma_px']).any():
                # Older packs already retain the actual contour readout and
                # adjacent measured-pixel normals. This only derives working
                # support from those immutable source measurements; missing
                # readout/normal/quantization evidence remains missing.
                _measured_board_readout_support(row)
                if np.isfinite(row['sigma_px']).all():
                    row['source_working_support_reconstructed_from_saved_actual_readout'] = True
            if ((row['actual_pts'], row['feature_id']) in held_features or
                    (row['actual_pts'], row['physical_material_id']) in held_materials):
                row['split'] = 'held'
            proofs = pack.values['source_proof_template_index'][lo:hi]
            row_record = len(rows)
            rows.append((pack, row, proofs))
            identity = (row['actual_pts'], row['scope'], row['physical_material_id'],
                _target_key(row['model_target']), row['observation_semantics'])
            for point_index, point in enumerate(row['uv']):
                pixel = np.asarray(point, np.float64).tobytes()
                # Candidate indices identify one frozen ice readout. A changed
                # UV is a source correction, not an exact duplicate to hide.
                locator = row.get('original_locator', {})
                if locator.get('kind') == 'current_ice_source_candidates':
                    original = (identity, int(row['original_point_indices'][point_index]),
                        json.dumps(locator, sort_keys=True, separators=(',', ':')))
                    if original in candidate_pixels and candidate_pixels[original] != pixel:
                        raise ValueError('One original source candidate carries competing actual UV readouts')
                    candidate_pixels[original] = pixel
                key = (identity, pixel)
                current = (row_record, point_index)
                if key in choices:
                    duplicate_count += 1
                    old_record, old_index = choices[key]
                    if (np.isfinite(row['sigma_px'][point_index]) and
                            not np.isfinite(rows[old_record][1]['sigma_px'][old_index])):
                        choices[key] = current
                else:
                    choices[key] = current
    selected = {}
    for row_number, point_index in choices.values():
        selected.setdefault(row_number, []).append(point_index)
    for pts, binding in frames.items():
        writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'])
    for pts in sorted(processed):
        writer.mark_exposure_processed(pts)
    for row_number, (pack, row, proof_ids) in enumerate(rows):
        indices = sorted(selected.get(row_number, ()))
        if not indices:
            continue
        row = _slice_row(row, indices)
        proofs = [dict(deepcopy(pack.proof_templates[int(proof_ids[i])]),
            uv=rows[row_number][1]['uv'][i].tolist()) for i in indices]
        review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
            source=source, reviewed_endpoints=proofs))
        if writer.add_row(row, review) != len(indices):
            raise ValueError('An already-admitted source pixel lost its exact proof during merge')
    # Rejected/pending source evidence remains part of the merged review. It
    # never becomes a row and must not disappear when an overlapping delta
    # supplies other valid pixels from the same exposure.
    for pack in packs:
        writer.exclusions.extend(deepcopy(pack.exclusions))
    receipt = writer.finish()
    return dict(receipt, exact_duplicate_points_removed=duplicate_count,
        processed_actual_exposures=len(processed), expected_actual_exposures=len(expected),
        all_exposures_processed=processed == set(expected),
        original_input_exclusions_preserved=True,
        source_review_is_not_manual_visual_completion=True, solver_started=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge authenticated strict source packs; never solve')
    parser.add_argument('--pack', nargs=2, action='append', metavar=('PATH', 'SHA256'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = merge_strict_absolute_source_packs(
        [dict(path=p, sha256=s) for p, s in options.pack], options.output)
    print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
