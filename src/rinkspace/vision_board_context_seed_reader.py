"""Install a missing edge template from its confirmed original board entity.

The original boundary guide stays navigation. Wider actual entity descriptors
establish template identity; the inherited current RGB edge reader still owns
every output pixel and uncertainty readout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from calibration.program_tracking.strict_model_scope import authenticate_source_reference
from .vision_board_material_point_profiles import BoardMaterialPointSourceReader
from .vision_programme_board_review import _identity_reference_allows
from .vision_board_known_interface_profiles import read_known_board_interface_profiles


class ContextTemplateBoardSourceReader(BoardMaterialPointSourceReader):
    def _template_matches(self, gray, exclusion, bands, shot_id, *, tracked_names=()):
        found=super()._template_matches(gray,exclusion,bands,shot_id,tracked_names=tracked_names)
        clearance=getattr(self,'box_clearance',None)
        current=clearance.current if clearance is not None else None
        if (current is None or self.previous_gray is None or self.previous_shot!=shot_id
                or self.actual_successor.get(self.previous_actual_pts)!=int(current['binding']['actual_pts'])):
            return found
        # A discovery configured with one named entity has no competing-name
        # descriptor bank. It cannot replace an independently verified exact
        # anchor's current actual RGB continuity with an unconfirmed label.
        # Keep only the independently recomputed, hard-mask source track; no
        # discovery is promoted and no UV/identity is supplied by the model.
        tracked=self._track(gray,current['hard'])
        for name,row in list(found.items()):
            anchor=row['template']['anchor'];continued=tracked.get(name)
            if (anchor.get('template_descriptors_from_confirmed_original_entity_context') is not True
                    or row.get('current_instance_identity_confirmed') is True or continued is None
                    or continued.get('current_instance_identity_confirmed') is not True
                    or continued['template']['anchor']['anchor_id']!=anchor['anchor_id']):continue
            edge=np.asarray(row['search_uv'],float);centre=edge.mean(0)
            _,_,axes=np.linalg.svd(edge-centre,full_matrices=False)
            mismatch=np.median(np.abs((continued['search_uv']-centre)@axes[-1]))
            if mismatch<=8.:found.pop(name)
        return found

    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        points=super()._read_edge(image,exclusion,bands,track,
            known_source_board_material=known_source_board_material)
        template=track['template'];anchor=template['anchor'];membership=known_source_board_material or {}
        confirmed_context=anchor.get('template_descriptors_from_confirmed_original_entity_context') is True
        if (not confirmed_context and membership.get('source_entity_identity_method')==
                'same_actual_exposure_native_alias_source_material_profile_confirmation_required'):
            reference=anchor.get('source_annotation_reference')
            if reference and membership.get('source_identity_authority')==reference:
                path,_=authenticate_source_reference(reference)
                fact=json.loads(Path(path).read_text(encoding='utf8'))
                confirmed_context=(fact.get('original_source_image_actually_reviewed') is True
                    and fact.get('static_board_material_identity_confirmed') is True
                    and fact.get('physical_instance_identity_confirmed') is True
                    and anchor['physical_material_id'].startswith('Ad'+str(fact.get('advertisement_number'))+'.'))
        if (not confirmed_context or anchor.get('curve',{}).get('kind')!='advertisement_boundary'
                or 'vertical' not in anchor.get('edge','')
                or membership.get('physical_material_id')!=anchor['physical_material_id']
                or membership.get('source_identity_confirmed') is not True
                or membership.get('actual_original_source_printed_board_membership_confirmed') is not True
                or membership.get('model_projection_or_fit_residual_used_for_identity') is not False):
            return points
        extra=read_known_board_interface_profiles(image,exclusion,track['search_uv'],track['normals'],template['contrast'])
        existing={int(p[0]) for p in points}
        return sorted([*points,*(p for p in extra if int(p[0]) not in existing)],key=lambda p:p[0])

    def add_source_seed(self, seed):
        super().add_source_seed(seed)
        if (seed.get('source_identity_confirmed') is not True
                or seed.get('load_preserved_source_template') is not True
                or seed.get('source_entity_context_original_RGB_membership_confirmed') is not True
                or seed.get('curve', {}).get('kind') != 'advertisement_boundary'
                or any(t['anchor']['anchor_id'] == seed['anchor_id'] for t in self.templates)):
            return
        reference = seed['source_annotation_reference']
        if not _identity_reference_allows(reference, seed['advertisement_number'],
                seed['shot_id'], self.identity_authorities):
            return
        authority_path, _ = authenticate_source_reference(reference)
        authority = json.loads(Path(authority_path).read_text(encoding='utf8'))
        binding = authority.get('source_binding', {})
        if (authority.get('original_source_image_actually_reviewed') is not True
                or authority.get('static_board_material_identity_confirmed') is not True
                or authority.get('physical_instance_identity_confirmed') is not True
                or authority.get('advertisement_number') != seed['advertisement_number']
                or binding.get('actual_pts') != seed['actual_pts']
                or binding.get('native_pts') != seed['native_pts']
                or binding.get('shot_id') != seed['shot_id']
                or binding.get('source_sha256') != self.source_sha256
                or binding.get('time_base') != self.time_base):
            return
        frame_ref = seed['source_raw_frame_reference']
        if authority.get('source_image') != frame_ref:
            return
        path = str(Path(frame_ref['path']).resolve())
        cached = self.source_seed_descriptor_cache.get(path)
        if cached is None:
            return
        digest, image, coordinates, descriptors = cached
        if (digest != frame_ref['sha256']
                or hashlib.sha256(image.tobytes()).hexdigest() != binding.get('decoded_native_BGR_sha256')):
            raise ValueError('Original board entity context differs from its reviewed exact source BGR')
        context = np.asarray(seed.get('source_entity_context_uv', ()), float)
        if (context.ndim != 2 or context.shape[1:] != (2,) or len(context) < 2
                or not np.isfinite(context).all() or (context < 0).any()
                or (context >= [1920, 1080]).any()):
            return
        lo, hi = context.min(0), context.max(0)
        if np.any(hi <= lo):
            return
        take = np.flatnonzero((coordinates >= lo).all(1) & (coordinates <= hi).all(1))
        if descriptors is None or len(take) < 8:
            return
        anchor = dict(seed, source_BGR_sha256=binding['decoded_native_BGR_sha256'],
            source_identity_reference_requires_current_shot=True,
            template_descriptors_from_confirmed_original_entity_context=True,
            boundary_guide_stays_navigation_only=True,
            original_entity_descriptor_context_uv=context.tolist())
        self._install_template(anchor, image, np.asarray(seed['uv'], np.float32),
            coordinates[take], descriptors[take])
        self.discovery_indices.clear()
