"""Retain raw observations and withhold only source identities reviewed in native RGB."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from .joint_metrics import PacketArrays
from .io_utils import read_json
from .dense_joint_proofs import _ref,_write,_checked

def withhold(*,parent_packet,parent_qualification,parent_weights,review,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);p=PacketArrays(parent_packet);doc=p.packet;q=read_json(parent_qualification);w=read_json(parent_weights);v=read_json(review)
    if v['schema']!='rinkfit.native_material_identity_review.v1' or v['qualification_uses_model_residual']:raise ValueError('Native identity review required')
    _checked(v['native_frame_extractor'],Path(review).parent)
    for item in v['rows']:
        for image in item['native_images']:_checked(image,Path(review).parent)
    arrays=dict(doc['arrays']);uv=p.load('uv');pts=p.load('native_pts');idx=[]
    for item in v['rows']:
        i=int(item['raw_row']);assert doc['source_row_ids'][i]==item['source_row_id'] and list(uv[i])==item['uv'] and int(pts[i])==item['native_pts']
        assert item['qualification']=='unknown' and item['native_images'] and item['finding']
        idx.append(i)
    assert len(idx)==len(set(idx)) and idx
    idx.sort()
    for key in ('admission','fit_role'):
        a=np.array(p.load(key),copy=True);assert np.all(a[idx]==1);a[idx]=0;f=out/(key+'.u8');a.tofile(f);arrays[key]=dict(_ref(f),dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
    revision=dict(schema='rinkfit.native_material_identity_revision.v1',parent_packet=_ref(parent_packet),parent_qualification=_ref(parent_qualification),review=_ref(review),withheld_rows=idx,raw_rows=doc['row_count'],source_UV_ID_PTS_and_operators_unchanged=True,source_rows_deleted=0,issuer=_ref(__file__))
    rev=_write(out/'revision.json',revision)
    packet=_write(out/'packet.json',dict(schema=doc['schema'],dense_parent_packet=_ref(parent_packet),row_count=doc['row_count'],arrays=arrays,active_rows=int(doc['active_rows'])-len(idx),admitted_rows=int(np.count_nonzero(p.load('admission')==1))-len(idx),native_material_identity_revision=rev,dense_parent_append=dict(source_manifests=[rev,_ref(review),_ref(__file__)])))
    qual=deepcopy(q);qual.update(schema='rinkfit.native_material_identity_revision_qualification.v1',source_packet=packet,parent_packet=_ref(parent_packet),parent_qualification=_ref(parent_qualification),revision=rev,issuer=_ref(__file__),registry_kind='native_ad_source_binding' if 'native_ad_source_binding' in doc else 'dense_source_binding',membership_array_sha256={k:x['sha256']for k,x in arrays.items()},all_raw_pixels_retained=True,quality_accepted=False)
    registry=doc[qual['registry_kind']];newheld=sum(i>=registry['row_start']for i in idx);qual['active_new_rows']-=newheld;qual['unknown_new_rows']=int(registry['count'])-qual['active_new_rows'];qr=_write(out/'source_qualification.json',qual)
    weights=deepcopy(w);weights.update(source_packet=packet,dense_source_qualification=qr,active_rows=int(w['active_rows'])-len(idx),new_active_rows=int(w['new_active_rows'])-newheld)
    values={}
    for k in ('weights.f64','active.u8','information_class.u8'):
        ref=w['files'][k];a=np.fromfile(ref['file'],dtype=ref['dtype']).copy();a[idx]=0;values[k]=a;f=out/k;a.tofile(f);weights['files'][k]=dict(_ref(f),dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
    tail=int(w['parent_rows']);current_active=values['active.u8'].astype(bool)
    weights.update(active_rows=int(current_active.sum()),new_active_rows=int(current_active[tail:].sum()),retained_parent_budget=float(values['weights.f64'][:tail].sum()),new_dense_budget=float(values['weights.f64'][tail:].sum()),actual_total_engineering_budget=float(values['weights.f64'].sum()))
    if 'component_budgets' in weights:
        group_rows={g['id']:g['rows'] for g in doc['engineering_pixel_dependency_groups']['groups']}
        budgets=[]
        for budget in weights['component_budgets']:
            rows=np.asarray(group_rows[budget['id']],dtype=np.int64);rows=rows[rows>=tail];count=int(current_active[rows].sum())
            if count:budgets.append(dict(budget,rows=count,engineering_budget=count*budget['per_row_weight']))
        weights['component_budgets']=budgets
    wr=_write(out/'engineering_weights.json',weights);return dict(packet=packet,qualification=qr,weights=wr,withheld=len(idx),raw_rows=int(doc['row_count']))
