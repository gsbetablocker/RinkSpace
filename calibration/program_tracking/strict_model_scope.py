"""Exact-source admission of static ice/board pixels; unreviewed means excluded.

This module does not classify an image, infer a surface from a fitted plane,
or admit a polygon. Reviewers supply exact original pixels and source-image
evidence. An LK pair requires two approved endpoints of the same material.
Historical files remain immutable; a partition keeps their original indices.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .source_identity_navigation import admissible_source_identity_context


REVIEW_SCHEMA = 'rinkspace.exact-source-static-model-scope-review.v1'
PARTITION_SCHEMA = 'rinkspace.strict-static-model-scope-motion-partition.v1'
ALLOWED_SCOPES = frozenset(('ice', 'board'))
AUTHORITIES = frozenset(('agent_original_source_review',
    'human_original_source_review', 'source_measurement_with_reviewed_identity'))
REASONS = dict(unreviewed_endpoint=1, outside_model_scope=2,
    nonstatic_or_unconfirmed_material=4, inaccurate_or_unconfirmed_pixel=8,
    differing_physical_material=16)
_AUTHENTICATED_FILES = {}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _file_identity(path):
    value = Path(path).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def authenticate_source_reference(reference):
    """Read large immutable source bytes once for an unchanged file identity."""
    path = Path(reference['path']).resolve()
    identity = _file_identity(path)
    expected = reference['sha256']
    if _AUTHENTICATED_FILES.get(str(path)) != (identity, expected):
        if file_sha(path) != expected or _file_identity(path) != identity:
            raise ValueError('Original source/evidence bytes changed')
        _AUTHENTICATED_FILES[str(path)] = (identity, expected)
    return path, identity


def verify_source_identities_unchanged(identities):
    """Check source/evidence identities at stream finish without rehashing video."""
    for path, identity in identities.items():
        if _file_identity(path) != identity:
            raise ValueError('Original source/evidence changed during source review')


def source_pixel_key(source_sha256, time_base, actual_pts, uv):
    """No rounded coordinate, proximity lookup, or fitted projection is allowed."""
    values = np.asarray(uv, np.float64)
    if (values.shape != (2,) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values >= [1920, 1080])):
        raise ValueError('Exact native 1920x1080 source pixel required')
    if not isinstance(actual_pts, (int, np.integer)) or isinstance(actual_pts, bool):
        raise ValueError('Exact integer actual exposure PTS required')
    return (source_sha256, time_base, int(actual_pts), values.tobytes())


def _source_identity(source):
    if (not isinstance(source, dict) or len(source.get('video_sha256', '')) != 64
            or not source.get('time_base') or source.get('pixel_domain') != 'native1920x1080'):
        raise ValueError('Explicit video hash, source clock and native pixel domain required')
    return source['video_sha256'], source['time_base']


class ExactSourceScopeReview:
    """Sparse exact-pixel allow/reject ledger. Missing records never enter FIT."""
    def __init__(self, packet, *, reference=None):
        if packet.get('schema') != REVIEW_SCHEMA:
            raise ValueError('Exact-source scope review schema required')
        self.source = deepcopy(packet['source'])
        source_sha, time_base = _source_identity(self.source)
        self.reference = deepcopy(reference)
        self.records = {}
        self.evidence_identities = {}
        for item in packet.get('reviewed_endpoints', []):
            key = source_pixel_key(source_sha, time_base, item['actual_pts'], item['uv'])
            if key in self.records:
                raise ValueError('Conflicting or duplicate exact-pixel review record')
            authority = item.get('review_authority')
            source_reviewed = (item.get('original_source_image_actually_reviewed') is True
                if authority != 'source_measurement_with_reviewed_identity' else
                item.get('source_feature_physical_identity_confirmed') is True
                and item.get('actual_source_RGB_measured') is True)
            if (authority not in AUTHORITIES
                    or not str(item.get('reviewer', '')).strip()
                    or not source_reviewed
                    or not admissible_source_identity_context(item,source=self.source)
                    or not isinstance(item.get('native_pts'), int)
                    or len(item.get('decoded_BGR_sha256', '')) != 64
                    or not str(item.get('review_note', '')).strip()):
                raise ValueError('Actual original source-image review evidence required')
            evidence = item.get('source_image_evidence')
            if (not isinstance(evidence, dict) or not evidence.get('path')
                    or len(evidence.get('sha256', '')) != 64):
                raise ValueError('Hash-bound original image/evidence reference required')
            if evidence.get('kind') == 'original_video_exact_native_frame':
                if (evidence['sha256'] != source_sha
                        or evidence.get('native_pts') != item['native_pts']
                        or evidence.get('decoded_BGR_sha256') != item['decoded_BGR_sha256']):
                    raise ValueError('Exact native source-video frame evidence differs')
            evidence_path, identity = authenticate_source_reference(evidence)
            self.evidence_identities[str(evidence_path)] = identity
            if item.get('model_projection_used_as_navigation_and_identity_context') is True:
                navigation=item['source_identity_navigation_evidence']
                for reference_key in ('source_identity_authority','source_model_navigation_reference'):
                    path,identity=authenticate_source_reference(navigation[reference_key])
                    self.evidence_identities[str(path)]=identity
            decision = item.get('decision')
            if decision not in ('allow', 'exclude'):
                raise ValueError('Explicit allow/exclude source decision required')
            if decision == 'allow' and (item.get('scope') not in ALLOWED_SCOPES
                    or item.get('static_material_confirmed') is not True
                    or item.get('pixel_location_confirmed') is not True
                    or not str(item.get('physical_material_id', '')).strip()):
                raise ValueError('Only accurate, confirmed static ice/board pixels may be allowed')
            self.records[key] = deepcopy(item)
        self.verify_evidence_unchanged()

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        path = Path(path).resolve()
        digest = file_sha(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Exact-source scope review changed')
        return cls(json.loads(path.read_text(encoding='utf8')),
                   reference=dict(path=str(path), sha256=digest))

    def endpoint(self, actual_pts, uv):
        sha, clock = _source_identity(self.source)
        return self.records.get(source_pixel_key(sha, clock, actual_pts, uv))

    def verify_evidence_unchanged(self):
        verify_source_identities_unchanged(self.evidence_identities)


def partition_motion_inputs(a, b, source_pts, target_pts, review):
    """Return ordered physical keep indices and a reason for every excluded pair."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    source_pts, target_pts = np.asarray(source_pts), np.asarray(target_pts)
    n = len(a)
    if (a.shape != (n, 2) or b.shape != a.shape
            or source_pts.shape != (n,) or target_pts.shape != (n,)
            or source_pts.dtype.kind not in 'iu' or target_pts.dtype.kind not in 'iu'
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a < 0) or np.any(b < 0)
            or np.any(a >= [1920, 1080]) or np.any(b >= [1920, 1080])):
        raise ValueError('Exact source endpoint inventory required')
    reasons = np.zeros(n, np.uint8)
    if not review.records:
        reasons.fill(REASONS['unreviewed_endpoint'])
    source_sha, clock = _source_identity(review.source)
    for i, (uv_a, uv_b, pa, pb) in enumerate(zip(a, b, source_pts, target_pts)):
        if not review.records:
            break
        # Bounds and finiteness were checked once above. This exact byte key
        # avoids repeating image-array validation for millions of old rows.
        endpoints = (review.records.get((source_sha, clock, int(pa), uv_a.tobytes())),
                     review.records.get((source_sha, clock, int(pb), uv_b.tobytes())))
        for endpoint in endpoints:
            if endpoint is None:
                reasons[i] |= REASONS['unreviewed_endpoint']
            elif endpoint['decision'] != 'allow' or endpoint.get('scope') not in ALLOWED_SCOPES:
                reasons[i] |= REASONS['outside_model_scope']
            elif endpoint.get('static_material_confirmed') is not True:
                reasons[i] |= REASONS['nonstatic_or_unconfirmed_material']
            elif endpoint.get('pixel_location_confirmed') is not True:
                reasons[i] |= REASONS['inaccurate_or_unconfirmed_pixel']
        if not reasons[i] and (endpoints[0]['physical_material_id'] != endpoints[1]['physical_material_id']
                or endpoints[0]['scope'] != endpoints[1]['scope']
                or not str(endpoints[0].get('physical_material_point_id', '')).strip()
                or endpoints[0].get('physical_material_point_id') != endpoints[1].get('physical_material_point_id')):
            reasons[i] |= REASONS['differing_physical_material']
    keep = np.flatnonzero(reasons == 0)
    old_to_active = np.full(n, -1, np.int64)
    old_to_active[keep] = np.arange(len(keep))
    review.verify_evidence_unchanged()
    return dict(keep_active_point_indices=keep, excluded_active_point_indices=np.flatnonzero(reasons),
                excluded_reason_mask=reasons, old_active_point_to_strict_point=old_to_active)


def write_motion_partition(path, *, block, a, b, ia, ib, stamps, sigma,
                           active_to_original, review, original_references):
    """Write only approved active inputs plus indices back into immutable history.

    The complete excluded coordinates stay in original_references; they are
    not copied into the active a/b arrays or represented by zero residuals.
    """
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Do not overwrite an earlier source partition')
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ia, ib, stamps = np.asarray(ia), np.asarray(ib), np.asarray(stamps)
    n = len(a)
    sigma = np.asarray(sigma, np.float64).reshape(-1, 2)
    original = np.asarray(active_to_original, np.int64)
    if (ia.shape != (n,) or ib.shape != (n,) or sigma.shape != (n, 2)
            or original.shape != (n,) or len(np.unique(original)) != n
            or ia.dtype.kind not in 'iu' or ib.dtype.kind not in 'iu'
            or stamps.dtype.kind not in 'iu' or np.any(ia < 0) or np.any(ib < 0)
            or np.any(ia >= len(stamps)) or np.any(ib >= len(stamps))):
        raise ValueError('Stable raw indices and exact endpoint clock required')
    for reference in original_references:
        authenticate_source_reference(reference)
    partition = partition_motion_inputs(a, b, stamps[ia], stamps[ib], review)
    keep = partition['keep_active_point_indices']
    metadata = dict(schema=PARTITION_SCHEMA, block=block, source=review.source,
        review=review.reference, original_references=original_references,
        original_point_count=n, strict_active_point_count=len(keep),
        reason_mask_bits=REASONS, unknown_or_unreviewed_default='excluded',
        physically_removed_from_active_arrays=True, source_UV_sigma_and_clock_preserved=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, a=a[keep], b=b[keep], ia=ia[keep], ib=ib[keep],
        stamps=stamps, motion_sigma=sigma[keep], active_to_original=original[keep],
        previous_active_to_original=original, metadata_json=json.dumps(metadata, ensure_ascii=False),
        **partition)
    return dict(path=str(path), sha256=file_sha(path), block=block,
                before_points=n, after_points=len(keep))


def load_motion_partition(path, *, expected_sha256, source_sha256, time_base):
    """The actual consumer reads only the compact approved arrays."""
    if file_sha(path) != expected_sha256:
        raise ValueError('Strict active source partition changed')
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(values.pop('metadata_json').item()))
    if (metadata.get('schema') != PARTITION_SCHEMA
            or _source_identity(metadata['source']) != (source_sha256, time_base)
            or metadata.get('physically_removed_from_active_arrays') is not True
            or metadata.get('unknown_or_unreviewed_default') != 'excluded'):
        raise ValueError('Strict active source contract required')
    keep = values['keep_active_point_indices']
    if (len(values['a']) != metadata['strict_active_point_count']
            or len(keep) != len(values['a'])
            or not np.array_equal(values['active_to_original'], values['previous_active_to_original'][keep])
            or not np.array_equal(np.flatnonzero(values['excluded_reason_mask'] == 0), keep)):
        raise ValueError('Physical source removal map differs from active arrays')
    if metadata['review'] is None:
        raise ValueError('Persisted strict input requires a saved exact-source review')
    for reference in metadata['original_references']:
        authenticate_source_reference(reference)
    review = ExactSourceScopeReview.load(metadata['review']['path'], expected_sha256=metadata['review']['sha256'])
    verification = partition_motion_inputs(values['a'], values['b'],
        values['stamps'][values['ia']], values['stamps'][values['ib']], review)
    if len(verification['keep_active_point_indices']) != len(values['a']):
        raise ValueError('Every active strict motion row needs two exact approved endpoints')
    return values, metadata


def approved_source_pixel_indices(actual_pts, uv, review, *, physical_material_id=None):
    """Use the same hard gate for absolute ice/board measurements and new points."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    keep = []
    for index, point in enumerate(uv):
        endpoint = review.endpoint(actual_pts, point)
        if (endpoint is not None and endpoint['decision'] == 'allow'
                and endpoint.get('scope') in ALLOWED_SCOPES
                and endpoint.get('static_material_confirmed') is True
                and endpoint.get('pixel_location_confirmed') is True
                and (physical_material_id is None
                     or endpoint['physical_material_id'] == physical_material_id)):
            keep.append(index)
    return np.asarray(keep, np.int64)
