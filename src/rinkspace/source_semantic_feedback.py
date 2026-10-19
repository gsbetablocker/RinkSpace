"""Human-confirmed source roles at existing UVs, separate from FIT/positions.

A role statement can confirm a reflection at an already sampled source pixel.
It does not draw a new boundary, provide a normal, or qualify a static scene
observation. Every record retains its original source point and RGB identity.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.strict_absolute_source import _exact_binding
from calibration.program_tracking.strict_model_scope import (
    ExactSourceScopeReview, REVIEW_SCHEMA, authenticate_source_reference,
    file_sha, verify_source_identities_unchanged,
)
from .vision_supervision import active_source_supervision_mask


SCHEMA = 'rinkspace.known-source-semantic-point-feedback.v1'
STATE = 'human_confirmed_semantic_role_at_existing_source_uv'


def _negative_records(reference):
    authenticate_source_reference(reference)
    with np.load(reference['path'], allow_pickle=False) as saved:
        metadata = json.loads(str(saved['metadata_json'].item()))
        rows = json.loads(str(saved['rows_json'].item()))
        uv = saved['uv'].copy()
    if (metadata.get('schema') != 'rinkspace.strict-source-role-absence-pixels.v1'
            or metadata.get('other_positive_class_assigned') is not False
            or metadata.get('default_exclusions_used_as_negatives') is not False
            or metadata.get('pending_used_as_truth') is not False
            or uv.shape != (len(rows), 2)):
        raise ValueError('An immutable actual-source role-absence pack is required')
    review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA,
        source=metadata['source'], reviewed_endpoints=[dict(r, uv=p.tolist())
            for r, p in zip(rows, uv)]))
    for row in rows:
        if (row.get('decision') != 'exclude'
                or row.get('reject_only_authority') != 'actual_visual_role_absence'
                or row.get('actual_source_RGB_role_absence_confirmed') is not True
                or row.get('confidence_used_as_rejection_evidence') is not False
                or row.get('model_probability_used_as_rejection_evidence') is not False):
            raise ValueError('Pending/masked/model-probability points cannot supply known roles')
    return metadata['source'], rows, uv, review


def write_known_source_role_feedback(path, *, original_negative_reference,
        source_png_reference, human_role_confirmation):
    """Write a separate affirmative role pack; never change the negative pack.

    The caller supplies an explicit human statement and its physical material
    identity. Only the original negative pack's unchanged UVs are copied.
    """
    import cv2
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Do not replace earlier human semantic source evidence')
    confirmation = deepcopy(human_role_confirmation)
    if (confirmation.get('authority_kind') != 'human_current_conversation_role_statement'
            or not str(confirmation.get('statement', '')).strip()
            or not str(confirmation.get('kind', '')).strip()
            or not str(confirmation.get('physical_material_id', '')).strip()
            or confirmation.get('model_probability_used_for_role') is not False):
        raise ValueError('An explicit human source role statement is required')
    source, originals, uv, review = _negative_records(original_negative_reference)
    if not originals:
        raise ValueError('No original reviewed source pixels exist')
    authenticate_source_reference(source_png_reference)
    image = cv2.imread(str(source_png_reference['path']))
    if image is None or image.shape != (1080, 1920, 3):
        raise ValueError('The complete original native BGR image is required')
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if any(r['decoded_BGR_sha256'] != digest for r in originals):
        raise ValueError('Semantic source image differs from the actual reviewed pixels')
    rows = []
    for i, original in enumerate(originals):
        if original['physical_material_id'] != confirmation['physical_material_id']:
            raise ValueError('A human role statement cannot rename another source material')
        if confirmation['kind'] in original['excluded_kinds']:
            raise ValueError('Affirmative semantic feedback conflicts with the existing human role-absence evidence')
        human_reference = original['human_region_authority']
        authenticate_source_reference(human_reference)
        rows.append(dict(actual_pts=original['actual_pts'], native_pts=original['native_pts'],
            shot_id=original['shot_id'], decoded_BGR_sha256=digest,
            physical_material_id=original['physical_material_id'],
            kind=confirmation['kind'], state=STATE,
            role_confirmation=deepcopy(confirmation),
            human_region_reference=dict(path=human_reference['path'], sha256=human_reference['sha256']),
            source_image_evidence=deepcopy(original['source_image_evidence']),
            source_png_reference=deepcopy(source_png_reference),
            original_negative_point_index=i,
            original_frozen_pixel_index=original['original_frozen_pixel_index'],
            original_pixel_inventory_reference=deepcopy(original['original_pixel_inventory_reference']),
            coordinate_origin='unchanged_existing_reviewed_source_negative_UV',
            actual_source_RGB_role_confirmed=True,
            semantic_supervision_only=True, fitting_admissible=False,
            position_supervision_allowed=False, new_human_positions_created=False,
            normal_or_boundary_target_created=False,
            model_probability_used_for_role=False))
    review.verify_evidence_unchanged()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, uv=uv,
        rows_json=json.dumps(rows, ensure_ascii=False, separators=(',', ':')),
        metadata_json=json.dumps(dict(schema=SCHEMA, source=source,
            original_negative_reference=deepcopy(original_negative_reference),
            source_png_reference=deepcopy(source_png_reference),
            role_confirmation=confirmation, semantic_supervision_only=True,
            fitting_admissible=False, position_supervision_allowed=False,
            new_human_positions_created=False, negative_pack_modified=False,
            whole_shot_training_plan_must_remain_unchanged=True),
            ensure_ascii=False, separators=(',', ':')))
    return dict(path=str(path), sha256=file_sha(path), points=len(uv))


class KnownSourceSemanticFeedback:
    """Authenticated semantic-only rows; source RGB and whole-shot role gate.

    ``catalog`` supplies scheduling metadata only. ``frame_supervision`` must
    check the shared decoded BGR before that source frame can train.
    """
    def __init__(self, reference, *, classes):
        self.reference = deepcopy(reference)
        self.classes = tuple(classes)
        self.identities = {}
        self._authenticate(reference)
        with np.load(reference['path'], allow_pickle=False) as saved:
            self.metadata = json.loads(str(saved['metadata_json'].item()))
            self.rows = json.loads(str(saved['rows_json'].item()))
            self.uv = saved['uv'].copy()
        metadata = self.metadata
        if (metadata.get('schema') != SCHEMA
                or metadata.get('semantic_supervision_only') is not True
                or metadata.get('fitting_admissible') is not False
                or metadata.get('position_supervision_allowed') is not False
                or metadata.get('new_human_positions_created') is not False
                or metadata.get('negative_pack_modified') is not False
                or metadata.get('whole_shot_training_plan_must_remain_unchanged') is not True):
            raise ValueError('A separate semantic-only, non-FIT source pack is required')
        source, originals, original_uv, review = _negative_records(metadata['original_negative_reference'])
        self._authenticate(metadata['original_negative_reference'])
        self._authenticate(metadata['source_png_reference'])
        if (metadata['source'] != source or self.uv.shape != (len(self.rows), 2)
                or len(self.rows) != len(originals) or not np.array_equal(self.uv, original_uv)):
            raise ValueError('Semantic feedback may only retain the exact original reviewed UVs')
        self.source = source
        self.source_metadata = deepcopy(source)
        self.by_native = {}
        self.frame_bindings = {}
        for i, (row, original) in enumerate(zip(self.rows, originals)):
            required = dict(state=STATE, semantic_supervision_only=True,
                fitting_admissible=False, position_supervision_allowed=False,
                new_human_positions_created=False, normal_or_boundary_target_created=False,
                actual_source_RGB_role_confirmed=True, model_probability_used_for_role=False,
                coordinate_origin='unchanged_existing_reviewed_source_negative_UV',
                original_negative_point_index=i)
            if (any(row.get(k) != v for k, v in required.items()) or row.get('kind') not in self.classes
                    or row.get('role_confirmation') != metadata['role_confirmation']
                    or row['role_confirmation'].get('authority_kind') != 'human_current_conversation_role_statement'
                    or not row['role_confirmation'].get('statement')
                    or row['role_confirmation'].get('kind') != row['kind']
                    or row['kind'] in original['excluded_kinds']
                    or row['role_confirmation'].get('physical_material_id') != row['physical_material_id']
                    or row['role_confirmation'].get('model_probability_used_for_role') is not False
                    or any(k in row for k in ('normal', 'interval_px', 'primitive', 'model_target'))):
                raise ValueError('Known semantic source feedback cannot manufacture position targets')
            for name in ('actual_pts', 'native_pts', 'shot_id', 'decoded_BGR_sha256',
                    'physical_material_id', 'original_frozen_pixel_index',
                    'source_image_evidence', 'original_pixel_inventory_reference'):
                if row[name] != original[name]:
                    raise ValueError('Known semantic role source identity or original locator changed')
            if (row['human_region_reference'] != {k:original['human_region_authority'][k]
                    for k in ('path', 'sha256')}
                    or row['source_png_reference'] != metadata['source_png_reference']):
                raise ValueError('Known semantic role human/RGB evidence changed')
            for evidence in (row['human_region_reference'], row['original_pixel_inventory_reference'],
                    row['source_image_evidence']):
                self._authenticate(evidence)
            native = int(row['native_pts'])
            binding = dict(actual_pts=row['actual_pts'], native_pts=native,
                source_sha256=source['video_sha256'], time_base=source['time_base'],
                source_path=row['source_image_evidence']['path'],
                decoded_BGR_sha256=row['decoded_BGR_sha256'], shot_id=row['shot_id'])
            if native in self.frame_bindings and self.frame_bindings[native] != binding:
                raise ValueError('One semantic source native frame has competing bindings')
            self.frame_bindings[native] = binding
            self.by_native.setdefault(native, []).append(i)
        # Check the saved exact source PNG once, rather than reading it per crop.
        import cv2
        image = cv2.imread(str(metadata['source_png_reference']['path']))
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() if image is not None else None
        if image is None or image.shape != (1080, 1920, 3) or any(r['decoded_BGR_sha256'] != digest for r in self.rows):
            raise ValueError('Known role source PNG does not reproduce the actual native BGR')
        review.verify_evidence_unchanged()

    def _authenticate(self, reference):
        path, identity = authenticate_source_reference(reference)
        self.identities[str(path)] = identity

    def verify_evidence_unchanged(self):
        verify_source_identities_unchanged(self.identities)

    def _matching_indices(self, binding):
        if any(isinstance(binding.get(k), bool) or not isinstance(binding.get(k), (int, np.integer))
                for k in ('actual_pts', 'native_pts')):
            raise ValueError('Known semantic source PTS must remain exact integers')
        if any(binding[k] != self.source['video_sha256'] for k in
                ('source_sha256', 'video_sha256', 'source_hash') if k in binding):
            raise ValueError('Known semantic source hash aliases disagree')
        if (binding.get('decoded_BGR_sha256') is not None
                and binding.get('decoded_native_BGR_sha256') is not None
                and binding['decoded_BGR_sha256'] != binding['decoded_native_BGR_sha256']):
            raise ValueError('Known semantic source BGR hash aliases disagree')
        binding = _exact_binding(binding)
        if (binding['video_sha256'] != self.source['video_sha256']
                or binding['time_base'] != self.source['time_base']):
            raise ValueError('Known semantic feedback belongs to a different exact source clock')
        indices = self.by_native.get(binding['native_pts'], [])
        for i in indices:
            row = self.rows[i]
            if (any(binding.get(k) != row[k] for k in
                    ('actual_pts', 'native_pts', 'decoded_BGR_sha256', 'shot_id'))
                    or Path(binding['path']).resolve() != Path(row['source_image_evidence']['path']).resolve()):
                raise ValueError('Known semantic feedback belongs to another actual native frame')
        return indices

    def catalog(self, binding, *, shot_plan, human_annotations=()):
        indices = self._matching_indices(binding)
        shots = [s for s in shot_plan['shots'] if s['source_sha256'] == self.source['video_sha256']
            and s['time_base'] == self.source['time_base'] and s['id'] == binding.get('shot_id')]
        if not shots:
            return []
        if len(shots) != 1 or shots[0].get('whole_shot') is not True:
            raise ValueError('Known roles must use the unchanged complete-shot training plan')
        if shots[0]['split'] != 'train':
            return []
        groups = {}
        for i in indices:
            row = self.rows[i]
            groups.setdefault((row['kind'], row['physical_material_id']), []).append(i)
        annotations = []
        for (kind, material), ids in groups.items():
            points = self.uv[ids]
            _protect_human_semantics(points, kind, human_annotations)
            annotations.append(dict(group_id=f"human_source_role_{binding['native_pts']}_{material}",
                kind=kind, physical_material_id=material, state=STATE,
                uv=points.tolist(), original_point_indices=ids,
                original_frozen_pixel_indices=[self.rows[i]['original_frozen_pixel_index'] for i in ids],
                authority=deepcopy(self.reference), source_binding=deepcopy(self.frame_bindings[binding['native_pts']]),
                semantic_supervision_only=True, position_supervision_allowed=False,
                fitting_admissible=False, new_human_positions_created=False,
                vision_training_role='train'))
        return annotations

    def frame_supervision(self, bgr, binding, *, shot_plan, human_annotations=()):
        indices = self._matching_indices(binding)
        if indices:
            bgr = np.asarray(bgr)
            if (bgr.shape != (1080, 1920, 3) or bgr.dtype != np.uint8
                    or hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
                    != self.rows[indices[0]]['decoded_BGR_sha256']):
                raise ValueError('Shared actual native RGB differs from human semantic role evidence')
        return self.catalog(binding, shot_plan=shot_plan, human_annotations=human_annotations)

    def supervision_catalog(self, native, *, shot_plan, human_annotations=()):
        """Planning-only semantic annotations; this does not read current RGB."""
        if isinstance(native, bool) or not isinstance(native, (int, np.integer)):
            raise ValueError('Known semantic native PTS must remain an exact integer')
        binding = self.frame_bindings.get(int(native))
        if binding is None:
            return []
        return self.catalog(binding, shot_plan=shot_plan, human_annotations=human_annotations)


def _protect_human_semantics(points, kind, human_annotations):
    source_keys = {np.asarray(p, np.float64).tobytes() for p in points}
    for annotation in human_annotations:
        if annotation.get('state') != 'user_confirmed_visual_supervision' or annotation.get('kind') == kind:
            continue
        uv = np.asarray(annotation['uv'], np.float64).reshape(-1, 2)
        active = np.asarray(active_source_supervision_mask(annotation), bool)
        visible = np.asarray(annotation.get('source_visible_mask', np.ones(len(uv), bool)), bool)
        if active.shape != (len(uv),) or visible.shape != (len(uv),):
            raise ValueError('Human semantic protection must preserve original visible UV indices')
        if any(p.tobytes() in source_keys for p in uv[active & visible]):
            raise ValueError('Known source role conflicts with direct human semantics at the same original UV')


@lru_cache(maxsize=8)
def _feedback(reference_path, reference_sha256, classes):
    return KnownSourceSemanticFeedback(dict(path=reference_path, sha256=reference_sha256), classes=classes)


def known_source_semantic_point_loss(logits, annotations, classes, *, source_binding,
        origin_xy=(0, 0), human_annotations=()):
    """CE only for authenticated known roles at unchanged source UVs.

    The separate frame_supervision call checks original BGR. This loss checks
    the source binding and exact saved UVs again; crops only select in-bounds
    existing points. It supplies no boundary/normal/primitive/position loss.
    """
    import torch
    from torch.nn import functional as F
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != len(classes):
        raise ValueError('Known semantic point loss needs one aligned source class field')
    h, w = logits.shape[-2:]
    if min(h, w) < 2 or not torch.isfinite(logits).all():
        raise ValueError('Known semantic source class field must be finite')
    values, count = [], 0
    for annotation in annotations:
        authority = annotation['authority']
        reader = _feedback(str(Path(authority['path']).resolve()), authority['sha256'], tuple(classes))
        reader.verify_evidence_unchanged()
        eligible = reader._matching_indices(source_binding)
        ids = annotation['original_point_indices']
        points = np.asarray(annotation['uv'], np.float64).reshape(-1, 2)
        if (annotation.get('state') != STATE or annotation.get('semantic_supervision_only') is not True
                or annotation.get('position_supervision_allowed') is not False
                or annotation.get('fitting_admissible') is not False
                or annotation.get('new_human_positions_created') is not False
                or annotation.get('vision_training_role') != 'train'
                or any(k in annotation for k in ('normal', 'interval_px', 'primitive', 'model_target'))
                or len(ids) != len(set(ids))
                or any(i not in eligible or reader.rows[i]['kind'] != annotation['kind'] for i in ids)
                or not np.array_equal(points, reader.uv[ids])
                or annotation['original_frozen_pixel_indices'] !=
                    [reader.rows[i]['original_frozen_pixel_index'] for i in ids]
                or any(reader.rows[i]['physical_material_id'] != annotation['physical_material_id'] for i in ids)
                or annotation['source_binding'] != reader.frame_bindings[source_binding['native_pts']]):
            raise ValueError('Semantic annotations cannot alter source roles, UVs, or positional evidence')
        _protect_human_semantics(points, annotation['kind'], human_annotations)
        cropped = points - np.asarray(origin_xy)
        inside = (cropped >= 0).all(1) & (cropped < [w, h]).all(1)
        if not inside.any():
            continue
        uv = torch.as_tensor(cropped[inside], dtype=torch.float32, device=logits.device)
        grid = uv * uv.new_tensor([2/(w-1), 2/(h-1)]) - 1
        scores = F.grid_sample(logits.float(), grid[None, :, None], align_corners=True)[0, :, :, 0].T
        targets = torch.full((len(uv),), classes.index(annotation['kind']), device=logits.device, dtype=torch.long)
        values.append(F.cross_entropy(scores, targets, reduction='sum'))
        count += len(uv)
    return (torch.stack(values).sum()/count if count else logits.sum()*0), count
