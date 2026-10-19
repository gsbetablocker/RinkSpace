"""Navigate only on authenticated static board-face source descriptors.

The inherited source UV/profile/blur and current identity gates are untouched.
This separate reader excludes surrounding ice/reflection/caption/person texture
from template navigation; no descriptor is itself a fitting pixel.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
from .vision_board_late_original_context_reader import LateOriginalContextBoardSourceReader
from .vision_board_backward_context_reader import BackwardContextTemplateBoardSourceReader


class NamedFaceContextBoardSourceReader(LateOriginalContextBoardSourceReader):
    def __init__(self, *args, source_seed_exclusion_provider=None, **kwargs):
        self.source_seed_exclusion_provider=source_seed_exclusion_provider
        self.named_face_source_descriptor_checks=[]
        if kwargs.get('source_annotation_seeds') and source_seed_exclusion_provider is None:
            raise ValueError('Exact source-frame exclusion provider is required for named face navigation')
        super().__init__(*args, **kwargs)

    def add_source_seed(self, seed):
        super().add_source_seed(seed)
        matches=[i for i,t in enumerate(self.templates) if t['anchor']['anchor_id']==seed['anchor_id']]
        if self.source_seed_exclusion_provider is None:return
        index=matches[-1] if matches else None
        anchor=deepcopy(self.templates[index]['anchor'] if index is not None else seed)
        raw_ref=anchor.get('source_raw_frame_reference')
        if not raw_ref:return
        path=str(Path(raw_ref['path']).resolve())
        cached=self.source_seed_descriptor_cache.get(path)
        if cached is None:return
        if index is None:anchor['source_BGR_sha256']=hashlib.sha256(cached[1].tobytes()).hexdigest()
        if not self._reviewed_full_entity(anchor):return
        _,image,xy,descriptors=cached
        context=np.asarray(anchor['source_entity_context_uv'],float)
        if context.ndim!=2 or context.shape[1:]!=(2,) or len(context)<2 or not np.isfinite(context).all():raise ValueError('Actual entity context is incomplete')
        lo,hi=context.min(0),context.max(0)
        mask=np.asarray(self.source_seed_exclusion_provider(int(anchor['native_pts'])),bool)
        if mask.shape!=image.shape[:2]:raise ValueError('Original source exclusion pixel domain differs')
        pixels=np.clip(np.rint(xy).astype(int),[0,0],[1919,1079])
        take=np.flatnonzero((xy>=lo).all(1)&(xy<=hi).all(1)&~mask[pixels[:,1],pixels[:,0]])
        check=dict(anchor_id=anchor['anchor_id'],physical_material_id=anchor['physical_material_id'],
            source_actual_pts=anchor['actual_pts'],source_native_pts=anchor['native_pts'],
            source_native_BGR_sha256=anchor['source_BGR_sha256'],actual_static_face_descriptors=len(take),
            entity_context_is_navigation_only=True,descriptor_coordinates_are_not_source_pixel_truth=True,
            source_excluded_texture_not_used_for_template_identity=True)
        if index is not None:self.templates.pop(index)
        if descriptors is None or len(take)<8:
            check['status']='pending_insufficient_unexcluded_actual_static_face_descriptors'
        else:
            anchor.update(source_navigation_descriptors_from_actual_board_entity_context_only=True,
                template_descriptors_from_confirmed_original_entity_context=True,
                original_entity_descriptor_context_uv=context.tolist())
            self._install_template(anchor,image,np.asarray(seed['uv'],np.float32),xy[take],descriptors[take])
            check['status']='actual_static_board_face_NAV_descriptors_installed_requires_current_source_identity_and_profile'
        self.named_face_source_descriptor_checks.append(check);self.discovery_indices.clear()


class BackwardNamedFaceContextBoardSourceReader(
        BackwardContextTemplateBoardSourceReader, NamedFaceContextBoardSourceReader):
    pass
