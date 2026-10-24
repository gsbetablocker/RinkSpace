"""Withdraw unsupported source-seed semantics while retaining all raw evidence."""
from pathlib import Path
import json,numpy as np
from .compact_identity_revision import _array
from .compact_identity_withhold_v2 import _source_identities
from .dense_joint_proofs import _ref,_checked
from .io_utils import write_json


def review(*,packet,source_evidence,issue,output):
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
    if Path(output).exists():raise FileExistsError('Immutable semantic source review exists')
    p=read(packet);e=read(source_evidence);facts=read(issue)
    if facts['source_model_projection_used']or facts['qualification_uses_model_residual']:raise ValueError('Original-image semantic facts required')
    seed=_array(e['columns']['seed_index']);a={k:_array(p['arrays'][k])for k in ['material_index','native_pts','uv','admission']}
    chosen={int(x['seed_index']):x for x in facts['seeds']};local=np.flatnonzero(np.isin(seed,list(chosen)));indices=local+facts['source_row_start'];identities,_=_source_identities(Path(packet),indices.tolist());rows=[]
    for li,i in zip(local,indices):
        i=int(i);si=int(seed[li]);source=chosen[si]
        if not a['admission'][i]or int(a['material_index'][i])!=facts['unresolved_material_index']:continue
        if not identities[i].startswith(e['source_dataset_id']+'/')or not identities[i].endswith('/'+str(int(a['native_pts'][i]))+'/seed'+str(si)):raise ValueError('Source seed/row identity differs')
        rows.append(dict(raw_row=i,source_row_id=identities[i],uv=a['uv'][i].tolist(),native_pts=int(a['native_pts'][i]),qualification='unknown',finding=facts['finding'],native_images=source['native_images'],native_image_role='Original issuer anchor and independently recorded canonical material; no claim that every propagated exposure was visually classified.',source_seed_index=si,source_anchor_pts=source['native_pts'],source_seed_uv=source['uv'],source_annotation=source['source_annotation'],declared_material_identity=facts['declared_material_identity'],source_geometry_classification='unsupported_inner_glyph_identity'))
    value=dict(schema='rinkfit.native_material_identity_review.v1',qualification_uses_model_residual=False,native_frame_extractor=facts['native_frame_extractor'],source_reviews=[_ref(issue),_ref(source_evidence),_ref(packet),_ref(__file__)],rows=rows,qualification_change_is_fitting_improvement=False,source_pixels_or_E_changed=False,source_rows_deleted=0)
    write_json(output,value);return dict(review=_ref(output),unknown_rows=len(rows),original_source_seeds=len(chosen),corrected_canonical_rows_retained=True)
