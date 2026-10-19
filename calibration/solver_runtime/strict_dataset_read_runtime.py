"""Single-pass strict dataset consumption with exact array proof indexing.

Each used proof template is passed through the original evidence reviewer.
Every pixel still checks its own proof index, exact UV bytes, source frame,
material role and measured centre construction. No legacy matching map is
built: this reader can only return authenticated dataset FIT/HELD rows.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from calibration.program_tracking.compact_source_json import unpack_source_json
from calibration.program_tracking.strict_absolute_source import (
    PACK_SCHEMA, _center_construction_mask, _row_admission_problem,
)
from calibration.program_tracking.strict_model_scope import (
    ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference,
    approved_source_pixel_indices, verify_source_identities_unchanged,
)
from calibration.program_tracking.strict_source_fit_dataset import (
    SCHEMA, POINT_FIELDS, partition_support, finite_source_row, _named_pending_indices,
)


class StrictDatasetValidatedPack:
    """Original strict validation without per-pixel evidence object copies."""

    def __init__(self, path, *, expected_sha256):
        started = time.perf_counter()
        self.path, self.file_identity = authenticate_source_reference(
            dict(path=str(path), sha256=expected_sha256))
        with np.load(self.path, allow_pickle=False) as archive:
            self.values = {key: archive[key].copy() for key in archive.files}
        self.metadata = json.loads(str(self.values.pop('metadata_json').item()))
        self.rows = unpack_source_json(self.values.pop('rows_json'))
        self.exclusions = unpack_source_json(self.values.pop('exclusions_json'))
        self.proof_templates = unpack_source_json(self.values.pop('source_proof_templates_json'))
        if (self.metadata.get('schema') != PACK_SCHEMA
                or self.metadata.get('legacy_family_fallback') != 'forbidden'
                or self.metadata.get('physically_compacted_approved_pixels_only') is not True):
            raise ValueError('Strict compact source pack is required')
        evidence_identities = {str(self.path): self.file_identity}
        for reference in self.metadata['original_references']:
            p, identity = authenticate_source_reference(reference)
            evidence_identities[str(p)] = identity
        offsets, uv = self.values['row_offsets'], self.values['uv']
        if (offsets.shape != (len(self.rows)+1,) or offsets[0] != 0
                or offsets[-1] != len(uv) or np.any(np.diff(offsets) < 0)):
            raise ValueError('Exact compact absolute row offsets required')
        shapes = (('uv', (len(uv), 2)), ('raw_candidate_uv', (len(uv), 2)),
            ('sigma_px', (len(uv),)), ('source_sides_uv', (len(uv), 2, 2)),
            ('source_side_quantiles_uv', (len(uv), 2, 3, 2)),
            ('source_proof_template_index', (len(uv),)), ('original_point_indices', (len(uv),)))
        for name, shape in shapes:
            if self.values[name].shape != shape:
                raise ValueError('Strict source array shape differs: ' + name)
        sigma = self.values['sigma_px']
        if (not np.isfinite(uv).all() or np.isinf(sigma).any()
                or np.any(sigma[np.isfinite(sigma)] <= 0)):
            raise ValueError('Strict absolute storage has invalid UV/sigma')
        proof_ids = self.values['source_proof_template_index']
        if (proof_ids.dtype.kind not in 'iu' or np.any(proof_ids < 0)
                or np.any(proof_ids >= len(self.proof_templates))):
            raise ValueError('Exact source proof template indices required')
        expected = set(self.metadata['expected_actual_pts'])
        frame_pts = self.values['source_frame_actual_pts']
        frame_native = self.values['source_frame_native_pts']
        frame_sha = self.values['source_frame_BGR_sha256']
        if (not np.array_equal(frame_pts, self.metadata['expected_actual_pts'])
                or frame_native.shape != frame_pts.shape or frame_sha.shape != frame_pts.shape):
            raise ValueError('Exact native source exposure catalogue differs')
        self.frame_bindings = {int(p): dict(native_pts=int(n), decoded_BGR_sha256=digest.decode())
            for p, n, digest in zip(frame_pts, frame_native, frame_sha) if n >= 0 and digest}
        if self.metadata['all_exposures_exact_native_BGR_bound'] != (set(self.frame_bindings) == expected):
            raise ValueError('Source frame binding coverage differs')
        processed = self.metadata['processed_actual_pts']
        if (len(processed) != len(set(processed)) or not set(processed).issubset(expected)
                or self.metadata['all_exposures_processed'] != (set(processed) == expected)):
            raise ValueError('Source exposure processing inventory differs')
        # The original reviewer checks source identity even for an empty row.
        # Its non-UV authority/image/navigation predicates are invariant for
        # one immutable proof-table entry, so authenticate each used entry once.
        ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
            source=self.metadata['source'], reviewed_endpoints=[]))
        template_authenticated = np.zeros(len(self.proof_templates), bool)
        template_checks = 0; template_row_bindings = 0
        has_fit = False; all_fit_ready = True
        for ordinal, row in enumerate(self.rows):
            lo, hi = map(int, offsets[ordinal:ordinal+2])
            points = uv[lo:hi]
            if (row['split'] not in ('fit', 'held') or row['actual_pts'] not in expected
                    or row['point_count'] != len(points) or _row_admission_problem(row)):
                raise ValueError('Unconfirmed row entered strict absolute storage')
            frame = self.frame_bindings.get(row['actual_pts'])
            if (frame is None or frame['native_pts'] != row['native_pts']
                    or frame['decoded_BGR_sha256'] != row['decoded_BGR_sha256']):
                raise ValueError('Strict source row differs from its native exposure catalogue')
            ready = bool(np.isfinite(sigma[lo:hi]).all())
            if (row['working_sigma_status'] == 'explicit_source_working_sigma') != ready:
                raise ValueError('Missing source uncertainty cannot become an explicit source weight')
            temporal = row['temporal_evidence']
            if temporal['kind'] in ('same_shot_actual_RGB_temporal_identity',
                    'actual_RGB_material_field_order_and_continuity_with_model_navigation'):
                previous = self.frame_bindings.get(temporal['previous_actual_pts'])
                if previous is not None and previous['decoded_BGR_sha256'] != temporal['previous_decoded_BGR_sha256']:
                    raise ValueError('Temporal source BGR binding differs from actual processed frame')
            # source_pixel_key's float64 exact-byte domain is applied to every
            # point. Signed zero stays distinct, just as the original keys.
            exact = np.ascontiguousarray(points, dtype=np.float64)
            if len(points) and (not isinstance(row['actual_pts'], (int, np.integer))
                    or isinstance(row['actual_pts'], bool)):
                raise ValueError('Exact integer actual exposure PTS required')
            if np.any(exact < 0) or np.any(exact >= [1920., 1080.]):
                raise ValueError('Exact native 1920x1080 source pixel required')
            if len(np.unique(exact.view(np.dtype('V16')).reshape(-1))) != len(points):
                raise ValueError('Conflicting or duplicate exact-pixel review record')
            if not _center_construction_mask(row, points, self.values['source_sides_uv'][lo:hi]).all():
                raise ValueError('Strict paint centerline differs from actual source sides')
            codes, first = np.unique(proof_ids[lo:hi], return_index=True)
            for code, position in zip(codes, first):
                proof = self.proof_templates[int(code)]
                if (proof['scope'] != row['scope'] or proof['actual_pts'] != row['actual_pts']
                        or proof['native_pts'] != row['native_pts']
                        or proof['decoded_BGR_sha256'] != row['decoded_BGR_sha256']):
                    raise ValueError('Strict pixel evidence belongs to another scope/exposure')
                point = exact[int(position)]
                if not template_authenticated[int(code)]:
                    review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
                        source=self.metadata['source'], reviewed_endpoints=[dict(proof, uv=point.tolist())]))
                    evidence_identities.update(review.evidence_identities)
                    if len(approved_source_pixel_indices(row['actual_pts'], [point], review,
                            physical_material_id=proof['physical_material_id'])) != 1:
                        raise ValueError('Strict absolute storage contains an unapproved exact source pixel')
                    template_authenticated[int(code)] = True
                    template_checks += 1
                # Every original point references one of these validated
                # table entries. A different row material cannot inherit it.
                if proof['physical_material_id'] != row['physical_material_id']:
                    raise ValueError('Strict absolute storage contains an unapproved exact source pixel')
                template_row_bindings += 1
            if row['split'] == 'fit':
                has_fit = True; all_fit_ready &= ready
        if (has_fit and all_fit_ready) != self.metadata['source_uncertainty_ready_for_fit']:
            raise ValueError('Source weight readiness differs from physically stored FIT rows')
        verify_source_identities_unchanged(evidence_identities)
        self.validation_receipt = dict(path=str(self.path), sha256=expected_sha256,
            rows=len(self.rows), original_points=len(uv),
            unique_proof_templates_authenticated=template_checks,
            proof_template_row_bindings_checked=template_row_bindings,
            every_original_UV_and_proof_index_checked=True,
            original_evidence_reviewer_reused_for_every_used_template=True,
            legacy_pixel_matching_map_constructed=False,
            seconds=time.perf_counter()-started)


_FACTOR_INPUT_FIELDS = frozenset((
    'split', 'scope', 'actual_pts', 'native_pts', 'decoded_BGR_sha256',
    'observation_semantics', 'physical_material_id', 'model_target', 'uv', 'sigma_px',
    'source_visible_paths', 'source_dataset_locator', 'curve_observation_kind',
    'must_preserve_upper_branch', 'source_optical_normal_support_authorized',
    'normal_interval_operator_authorized', 'source_point_bounds_authorized',
    'normal_interval', 'normal_interval_px', 'point_support_bounds_xy_px',
    'source_tangent_interval_px', 'clip_boundary_x', 'extent_side',
    'scanline_branch', 'source_operator_array_fields', 'point_correspondence',
    'quadratic', 'loss_weights', 'loss_weight', 'loss', 'loss_function',
    'robust_loss', 'huber_delta', 'trace_objective', 'source_integration_mass_rule',
    'source_integration_weights',
))


def load_strict_dataset_rows_once(manifest_reference):
    """Authenticate each source once and split both roles in original order."""
    from .strict_source_dataset_adapter import compact_dataset_rows
    started = time.perf_counter()
    authenticate_source_reference(manifest_reference)
    manifest = json.loads(Path(manifest_reference['path']).read_text(encoding='utf8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Frozen finite-support source dataset required')
    thin = dict(fit=[], held=[]); receipts = []
    for cut in manifest['cuts']:
        authenticate_source_reference(cut['support_index'])
        reference = cut['source_pack']
        pack = StrictDatasetValidatedPack(reference['path'], expected_sha256=reference['sha256'])
        if pack.metadata['source'] != manifest['source']:
            raise ValueError('Dataset source pack has another original video/clock')
        roles = [dict(split=row['split'], point_count=row['point_count']) for row in pack.rows]
        with np.load(cut['support_index']['path'], allow_pickle=False) as stored:
            selections = {role: stored[role+'_point_indices'] for role in ('fit', 'held')}
            if 'promoted_held_rows' in stored:
                for row_id in stored['promoted_held_rows']:
                    if roles[int(row_id)]['split'] != 'fit':
                        raise ValueError('Only FIT rows can acquire HELD protection')
                    roles[int(row_id)]['split'] = 'held'
            expected = partition_support(roles, pack.values['row_offsets'], pack.values['sigma_px'])
            if 'pending_named_point_indices' in stored:
                named = stored['pending_named_point_indices']
                bindings = {p: (v['native_pts'], v['decoded_BGR_sha256']) for p, v in pack.frame_bindings.items()}
                actual = _named_pending_indices(pack.rows, pack.values['row_offsets'], reference,
                    pack.metadata['source'], bindings, cut.get('pending_filter_references', []))
                if not np.array_equal(named, actual):
                    raise ValueError('Exact named pending selection changed')
                expected = {key: np.setdiff1d(value, named, assume_unique=True) for key, value in expected.items()}
            for key, selected in expected.items():
                if not np.array_equal(stored[key], selected):
                    raise ValueError('Stored source role/support selection changed')
            if not np.array_equal(stored['row_offsets'], pack.values['row_offsets']):
                raise ValueError('Stored source row boundaries changed')
        for ordinal, header in enumerate(pack.rows):
            lo, hi = map(int, pack.values['row_offsets'][ordinal:ordinal+2])
            for role, selection in selections.items():
                a, b = np.searchsorted(selection, [lo, hi]); local = selection[a:b]-lo
                if len(local):
                    row = dict(header, **{field: pack.values[field][lo:hi] for field in POINT_FIELDS})
                    selected = finite_source_row(row, local, reference, ordinal, output_split=role)
                    thin[role].append({key: value for key, value in selected.items() if key in _FACTOR_INPUT_FIELDS})
        receipts.append(pack.validation_receipt)
        # Drop the last row's ndarray views before loading the next pack.
        # They otherwise retain a complete large decoded UV/sides allocation.
        if 'row' in locals():
            del row, selected
        if pack.rows:
            del header
        del pack, roles, expected
    result = {role: compact_dataset_rows(values, split=role) for role, values in thin.items()}
    for role in ('fit', 'held'):
        if sum(len(row['uv']) for row in result[role]) != manifest[role+'_points']:
            raise ValueError('Single-pass strict source lost original ' + role.upper() + ' points')
    return result['fit'], result['held'], dict(
        schema='rinkspace.single-pass-strict-dataset-read.v1', source_packs=receipts,
        full_source_pack_passes=1, source_points=sum(item['original_points'] for item in receipts),
        fit_points=manifest['fit_points'], held_points=manifest['held_points'],
        every_original_row_proof_UV_sigma_clock_and_role_checked=True,
        source_pixels_sigma_roles_locators_and_groups_unchanged=True,
        seconds=time.perf_counter()-started)
