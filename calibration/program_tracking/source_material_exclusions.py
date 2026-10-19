"""Authenticated exact-native user-known unmodeled source material."""
from __future__ import annotations
import json
import numpy as np
from .strict_model_scope import authenticate_source_reference


def load_known_source_material_exclusions(references):
    result=[]
    for reference in references:
        path,_=authenticate_source_reference(reference)
        with np.load(path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata_json'].item()))
            mask=saved['actual_source_material_mask'].copy()
        if (metadata.get('schema')!='rinkspace.known-original-source-external-material-exclusion.v1'
                or metadata.get('mask_is_source_exclusion_only') is not True
                or metadata.get('actual_source_selection_is_not_pixel_GroundTruth') is not True
                or metadata.get('source_fit_or_model_change') is not False
                or mask.shape!=(1080,1920) or mask.dtype!=np.bool_):
            raise ValueError('Known source material exclusion contract differs')
        binding=metadata['source_binding']
        if (any(type(binding.get(key)) is not int for key in ('actual_pts','native_pts'))
                or len(binding.get('source_sha256',''))!=64
                or len(binding.get('decoded_native_BGR_sha256',''))!=64
                or not binding.get('time_base') or not binding.get('shot_id')):
            raise ValueError('Known material exclusion needs exact native source binding')
        for key in ('source_identity_authority','original_source_image'):
            authenticate_source_reference(metadata[key])
        result.append(dict(source_binding=binding,actual_source_material_mask=mask,
            authority=metadata['source_identity_authority'],original_source_image=metadata['original_source_image'],
            source_material_exclusion_reference=reference))
    return result


def exact_source_material_exclusion(binding, exclusions):
    """Do not copy a canonical material mask into another native presentation."""
    current={key:binding[key] for key in ('source_sha256','time_base','actual_pts','native_pts',
        'shot_id','decoded_native_BGR_sha256')}
    mask=np.zeros((1080,1920),bool)
    for evidence in exclusions:
        source=evidence['source_binding']
        if source['native_pts']!=current['native_pts']:continue
        if source!=current:raise ValueError('Known negative material belongs to different source RGB bytes')
        mask|=evidence['actual_source_material_mask']
    return mask
