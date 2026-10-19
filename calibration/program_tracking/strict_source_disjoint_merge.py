"""Lossless sequential merge for source packs with disjoint exposure clocks.

Overlapping deltas must first use the ordinary exact-pixel merge. This path
holds one decoded source pack at a time and never compacts or samples pixels.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path

import numpy as np

from .strict_absolute_source import PACK_SCHEMA, StrictAbsolutePackWriter, StrictAbsoluteSourcePack
from .strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference


def _append_disjoint_pack(writer, reference, processed_pts):
    """Keep input decoder and its loop temporaries inside one lexical scope."""
    pack = StrictAbsoluteSourcePack(reference['path'], expected_sha256=reference['sha256'])
    if set(pack.metadata['processed_actual_pts']) != processed_pts:
        raise ValueError('Disjoint source processing clock changed after preflight')
    if pack.metadata['source'] != writer.source or pack.metadata['expected_actual_pts'] != writer.expected_pts.tolist():
        raise ValueError('Disjoint source video or expected clock changed after preflight')
    for header in pack.rows:
        if header['actual_pts'] not in processed_pts:
            raise ValueError('A disjoint source row belongs outside its processed source cut')
    if not set(pack.frame_bindings).issubset(processed_pts):
        raise ValueError('A disjoint pack binds pixels outside its processed source clock')
    # The public writer stores these canonical dtypes. Reject a representation
    # it would convert rather than silently claim a byte-preserving merge.
    fields = ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
        'source_side_quantiles_uv', 'original_point_indices')
    for name in fields:
        expected_dtype = np.dtype('int64' if name == 'original_point_indices' else 'float64')
        if pack.values[name].dtype != expected_dtype:
            raise ValueError('Disjoint merge cannot literally preserve noncanonical source dtype: '+name)
    for pts, binding in pack.frame_bindings.items():
        writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'])
    for pts in sorted(processed_pts):
        writer.mark_exposure_processed(pts)
    for ri, header in enumerate(pack.rows):
        lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
        row = dict(deepcopy(header), **{name:pack.values[name][lo:hi] for name in fields})
        # Preserve prior ordinary-merge provenance rather than overwrite it.
        row['strict_disjoint_pack_origin'] = dict(source_pack=reference, source_row_index=ri)
        proofs = [dict(deepcopy(pack.proof_templates[int(code)]), uv=point.tolist())
            for code,point in zip(pack.values['source_proof_template_index'][lo:hi], row['uv'])]
        review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
            source=writer.source, reviewed_endpoints=proofs))
        # No support reconstruction here: saved NaN, sigma, sides and quantiles
        # must retain their original literal values and missingness.
        if writer.add_row(row, review) != hi-lo:
            raise ValueError('A disjoint source pixel lost its existing proof')
    writer.exclusions.extend(deepcopy(pack.exclusions))
    return len(pack.values['uv']), len(pack.rows), len(pack.frame_bindings)


def merge_disjoint_strict_absolute_source_packs(references, output_path, *, additional_references=()):
    references = [dict(path=str(Path(r['path']).resolve()), sha256=r['sha256']) for r in references]
    if not references:
        raise ValueError('Authenticated disjoint source packs are required')
    original_refs = {}
    source = expected = None
    processed = set()
    legacy = set()
    cuts = []
    for reference in references:
        authenticate_source_reference(reference)
        with np.load(reference['path'], allow_pickle=False) as values:
            metadata = json.loads(str(values['metadata_json'].item()))
        if (metadata.get('schema') != PACK_SCHEMA or metadata.get('legacy_family_fallback') != 'forbidden'
                or metadata.get('physically_compacted_approved_pixels_only') is not True):
            raise ValueError('Strict compact source metadata is required before disjoint merge')
        if source is None:
            source, expected = metadata['source'], metadata['expected_actual_pts']
        if metadata['source'] != source or metadata['expected_actual_pts'] != expected:
            raise ValueError('Disjoint packs must share the exact video and expected source clock')
        current = set(metadata['processed_actual_pts'])
        if (len(current) != len(metadata['processed_actual_pts'])
                or any(type(pts) is not int for pts in metadata['processed_actual_pts'])
                or not current.issubset(set(expected))
                or metadata['all_exposures_processed'] != (current == set(expected))):
            raise ValueError('Disjoint source processing clock is invalid')
        if processed.intersection(current):
            raise ValueError('Overlapping exposures require the ordinary exact-pixel merge first')
        processed.update(current)
        cuts.append(current)
        legacy.update(metadata['legacy_pixel_families'])
        for ref in [reference, *metadata['original_references']]:
            key = str(Path(ref['path']).resolve())
            if key in original_refs and original_refs[key]['sha256'] != ref['sha256']:
                raise ValueError('One source authority cannot bind conflicting bytes')
            original_refs[key] = dict(ref, path=key)
    for ref in additional_references:
        authenticate_source_reference(ref)
        key = str(Path(ref['path']).resolve())
        if key in original_refs and original_refs[key]['sha256'] != ref['sha256']:
            raise ValueError('Additional source authority conflicts with immutable provenance')
        original_refs[key] = dict(ref, path=key)
    writer = StrictAbsolutePackWriter(output_path, source=source,
        expected_actual_pts=expected, original_references=list(original_refs.values()),
        legacy_pixel_families=legacy)
    input_points = input_rows = input_frames = 0
    for reference, current in zip(references, cuts):
        points, rows, frames = _append_disjoint_pack(writer, reference, current)
        input_points += points
        input_rows += rows
        input_frames += frames
        # Only the writer's canonical output and numeric structures remain.
        # The full input graph and even its last loop variables are released.
        gc.collect()
    if (writer.processed_pts != processed or len(writer.frame_bindings) != input_frames
            or not set(writer.frame_bindings).issubset(processed)):
        raise ValueError('Disjoint merge changed original source frame bindings')
    receipt = writer.finish()
    if receipt['points'] != input_points or receipt['rows'] != input_rows:
        raise ValueError('Sequential disjoint merge changed source point or row count')
    return dict(receipt, exact_duplicate_points_removed=0,
        input_points=input_points, input_rows=input_rows,
        processed_actual_exposures=len(processed), expected_actual_exposures=len(expected),
        all_exposures_processed=processed == set(expected),
        source_review_is_not_manual_visual_completion=True,
        merge_method='one_decoded_disjoint_source_pack_at_a_time',
        missing_source_sigma_reconstructed=False, original_pixel_payload_values_preserved=True,
        original_input_exclusions_preserved=True,
        pixels_or_time_subsampled=False, solver_started=False)
