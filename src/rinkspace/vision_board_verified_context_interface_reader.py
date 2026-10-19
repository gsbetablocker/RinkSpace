"""Reread one wide interface for an authenticated complete board source face."""
from __future__ import annotations
import json
from pathlib import Path
from calibration.program_tracking.strict_model_scope import authenticate_source_reference
from .vision_programme_board_review import _identity_reference_allows
from .vision_board_context_seed_reader import ContextTemplateBoardSourceReader
from .vision_board_known_interface_profiles import read_known_board_interface_profiles
from .vision_board_backward_context_reader import BackwardContextTemplateBoardSourceReader

class VerifiedContextInterfaceBoardSourceReader(ContextTemplateBoardSourceReader):
    def _reviewed_full_entity(self,anchor):
        if (anchor.get('source_entity_context_original_RGB_membership_confirmed') is not True
                or anchor.get('load_preserved_source_template') is not True
                or anchor.get('source_identity_confirmed') is not True):return False
        reference=anchor['source_annotation_reference']
        if not _identity_reference_allows(reference,anchor['advertisement_number'],anchor['shot_id'],self.identity_authorities):return False
        path,_=authenticate_source_reference(reference);fact=json.loads(Path(path).read_text(encoding='utf8'));binding=fact.get('source_binding',{})
        return bool(fact.get('original_source_image_actually_reviewed') is True
            and fact.get('static_board_material_identity_confirmed') is True
            and fact.get('physical_instance_identity_confirmed') is True
            and fact.get('advertisement_number')==anchor['advertisement_number']
            and fact.get('source_image')==anchor.get('source_raw_frame_reference')
            and binding.get('decoded_native_BGR_sha256')==anchor.get('source_BGR_sha256')
            and binding.get('native_pts')==anchor['native_pts'] and binding.get('actual_pts')==anchor['actual_pts']
            and binding.get('shot_id')==anchor['shot_id'] and binding.get('source_sha256')==self.source_sha256
            and binding.get('time_base')==self.time_base)

    def _read_edge(self,image,exclusion,bands,track,*,known_source_board_material=None):
        points=super()._read_edge(image,exclusion,bands,track,known_source_board_material=known_source_board_material)
        template=track['template'];anchor=template['anchor'];membership=known_source_board_material or {}
        if (anchor.get('curve',{}).get('kind')!='advertisement_boundary' or 'vertical' not in anchor.get('edge','')
                or membership.get('physical_material_id')!=anchor['physical_material_id']
                or membership.get('source_identity_confirmed') is not True
                or membership.get('actual_original_source_printed_board_membership_confirmed') is not True
                or membership.get('model_projection_or_fit_residual_used_for_identity') is not False
                or not self._reviewed_full_entity(anchor)):return points
        extra=read_known_board_interface_profiles(image,exclusion,track['search_uv'],track['normals'],template['contrast'])
        present={int(p[0]) for p in points}
        return sorted([*points,*(p for p in extra if int(p[0]) not in present)],key=lambda p:p[0])

class BackwardVerifiedContextInterfaceBoardSourceReader(BackwardContextTemplateBoardSourceReader,VerifiedContextInterfaceBoardSourceReader):
    pass
