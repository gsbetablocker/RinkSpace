"""Bind a measured cut to its authenticated whole-programme source clock.

Only the expected clock denominator and empty out-of-cut slots expand.
Processed exposures, measured pixels and their native image bindings stay put.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .strict_absolute_source import StrictAbsolutePackWriter, StrictAbsoluteSourcePack, _literal
from .strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference, file_sha


def expand_strict_source_programme_clock(reference, camera_reference, output_path, *, recover_completed_output=False):
    authenticate_source_reference(camera_reference)
    with np.load(camera_reference['path'], allow_pickle=False) as camera:
        clock = np.unique(camera['exposure_representative_pts'])
        native = camera['pts']
        aliases = camera['exposure_representative_pts']
        meta = json.loads(str(camera['metadata_json'].item()))
    if (native.ndim != 1 or aliases.shape != native.shape or
            not np.array_equal(native, np.unique(native))):
        raise ValueError('An authenticated ordered native/actual source clock is required')
    pack = StrictAbsoluteSourcePack(reference['path'], expected_sha256=reference['sha256'])
    source = pack.metadata['source']
    if (source['video_sha256'] != meta['source_sha256'] or
            source['time_base'] != meta['source_time_base']):
        raise ValueError('Expected programme clock belongs to another source')
    if not set(pack.metadata['expected_actual_pts']).issubset(set(map(int, clock))):
        raise ValueError('Measured cut expected exposures are outside this programme clock')
    refs = {}
    for ref in [reference, camera_reference, *pack.metadata['original_references']]:
        key = str(Path(ref['path']).resolve())
        if key in refs and refs[key]['sha256'] != ref['sha256']:
            raise ValueError('Source clock expansion cannot change an authority')
        refs[key] = dict(ref, path=key)
    fields = ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv',
        'source_side_quantiles_uv', 'original_point_indices')
    recovered = Path(output_path).exists() and recover_completed_output
    if recovered:
        receipt = dict(path=str(Path(output_path).resolve()), sha256=file_sha(output_path),
            points=len(pack.values['uv']), rows=len(pack.rows))
    else:
        writer = StrictAbsolutePackWriter(output_path, source=source,
            expected_actual_pts=clock, original_references=list(refs.values()),
            legacy_pixel_families=pack.metadata['legacy_pixel_families'])
        for pts, binding in pack.frame_bindings.items():
            writer._register_frame(pts, binding['native_pts'], binding['decoded_BGR_sha256'])
        for pts in pack.metadata['processed_actual_pts']:
            writer.mark_exposure_processed(pts)
        for ri, header in enumerate(pack.rows):
            lo, hi = map(int, pack.values['row_offsets'][ri:ri+2])
            row = dict(deepcopy(header), **{name: pack.values[name][lo:hi] for name in fields})
            proofs = [dict(deepcopy(pack.proof_templates[int(code)]), uv=point.tolist())
                for code, point in zip(pack.values['source_proof_template_index'][lo:hi], row['uv'])]
            review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
                source=source, reviewed_endpoints=proofs))
            if writer.add_row(row, review) != hi-lo:
                raise ValueError('An existing measured source point lost its proof')
        writer.exclusions.extend(deepcopy(pack.exclusions))
        receipt = writer.finish()
    derived = StrictAbsoluteSourcePack(receipt['path'], expected_sha256=receipt['sha256'])
    if (derived.metadata['source'] != source or derived.metadata['expected_actual_pts'] != clock.tolist()
            or any(ref not in derived.metadata['original_references'] for ref in refs.values())):
        raise ValueError('Expected clock expansion lost its exact programme authority')
    for name in (*fields, 'row_offsets', 'source_proof_template_index'):
        old, new = pack.values[name], derived.values[name]
        if old.dtype != new.dtype or old.shape != new.shape or old.tobytes() != new.tobytes():
            raise ValueError('Expected clock expansion changed measured source payload: '+name)
    if (derived.frame_bindings != pack.frame_bindings or _literal(derived.rows) != _literal(pack.rows) or
            _literal(derived.proof_templates) != _literal(pack.proof_templates) or _literal(derived.exclusions) != _literal(pack.exclusions) or
            derived.metadata['processed_actual_pts'] != pack.metadata['processed_actual_pts']):
        raise ValueError('Expected clock expansion changed source evidence or actual clock')
    return dict(receipt, input_source_pack=reference, camera_clock_authority=camera_reference,
        recovered_and_fully_verified_completed_output=recovered,
        old_expected_actual_exposures=len(pack.metadata['expected_actual_pts']),
        expected_actual_exposures=len(clock),
        processed_actual_exposures=len(pack.metadata['processed_actual_pts']),
        exact_measured_point_payload_and_evidence_preserved=True,
        native_RGB_bindings_changed=False, out_of_cut_exposures_marked_processed=False,
        all_exposures_processed=derived.metadata['all_exposures_processed'], solver_started=False)
