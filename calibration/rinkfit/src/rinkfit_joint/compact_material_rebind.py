"""Correct proven printed-material identity while retaining measured pixels."""
from pathlib import Path
import json
import numpy as np
from .dense_joint_proofs import _ref, _write
from .compact_identity_revision import _json, _array

def rebind(*, parent_packet, parent_qualification, parent_weights, review, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    doc=_json(parent_packet);q=_json(parent_qualification);w=_json(parent_weights);v=_json(review)
    if v['schema']!='rinkfit.native_print_identity_rebind_review.v1' or v['source_model_projection_used'] or v['source_pixels_changed']:
        raise ValueError('Actual-source identity review required')
    arrays=dict(doc['arrays']);mat=np.array(_array(arrays['material_index']),copy=True);uv=_array(arrays['uv']);pts=_array(arrays['native_pts']);path=_array(arrays['path']);active=_array(arrays['admission'])
    rows=[]
    for e in v['rows']:
        i=e['raw_row']
        if not active[i] or int(mat[i])!=e['old_material_index'] or uv[i].tolist()!=e['uv'] or int(pts[i])!=e['native_pts'] or int(path[i])!=0:
            raise ValueError('Reviewed source changed')
        if not 0<=e['new_station']<=1 or e['old_label'].split('.')[0]!=e['new_label'].split('.')[0]:
            raise ValueError('Not a same-panel printed-boundary identity correction')
        mat[i]=e['new_material_index'];rows.append(i)
    if not rows or len(set(rows))!=len(rows):raise ValueError('Empty or duplicate rebind')
    file=out/'material_index.i32';mat.tofile(file);arrays['material_index']=dict(_ref(file),dtype=mat.dtype.str,shape=list(mat.shape),layout='C',byte_order='little')
    revision=dict(schema='rinkfit.native_print_identity_rebind.v1',parent_packet=_ref(parent_packet),parent_qualification=_ref(parent_qualification),review=_ref(review),rebound_rows=sorted(rows),source_rows_deleted=0,raw_rows=doc['row_count'],source_UV_ID_PTS_and_membership_unchanged=True,issuer=_ref(__file__))
    rr=_write(out/'revision.json',revision)
    packet=_write(out/'packet.json',dict(schema=doc['schema'],dense_parent_packet=_ref(parent_packet),row_count=doc['row_count'],arrays=arrays,active_rows=doc['active_rows'],admitted_rows=doc['admitted_rows'],native_print_identity_rebind=rr,dense_parent_append=dict(source_manifests=[rr,_ref(review),_ref(__file__)])))
    q.update(schema='rinkfit.native_print_identity_rebind_qualification.v1',source_packet=packet,parent_packet=_ref(parent_packet),parent_qualification=_ref(parent_qualification),revision=rr,issuer=_ref(__file__),registry_kind='native_ad_source_binding',membership_array_sha256={k:r['sha256']for k,r in arrays.items()},all_raw_pixels_retained=True,quality_accepted=False)
    qr=_write(out/'source_qualification.json',q)
    w.update(source_packet=packet,dense_source_qualification=qr,source_identity_rebind=rr)
    # Existing per-exposure, per-panel budgets are unchanged by letter identity.
    wr=_write(out/'engineering_weights.json',w)
    return dict(packet=packet,qualification=qr,weights=wr,revision=rr,rebound=len(rows))
