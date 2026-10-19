"""Explicit RGB identity quarantine for original ordinary source rows.

No residual-dependent rejection, W slicing, source remeasurement or new noise.
The original mixed-identity extraction remains archived outside FIT.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from types import MethodType
import numpy as np
from .first_shot_source_transition import read, reference


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def quarantine_ordinary_rows(frames, manifest_reference):
    from .first_shot_fit import FrozenWhitening
    manifest=read(manifest_reference)
    if (manifest.get('schema')!='rinkspace.reviewed-ordinary-native-source-quarantine.v1'
            or manifest.get('review_complete')is not True
            or manifest.get('source_UV_ID_sigma_W_approvals_and_native_PTS_unchanged')is not True
            or manifest.get('all_correlated_W_groups_must_be_retained_byte_exact')is not True):
        raise ValueError('Explicit original-RGB ordinary-source identity review required')
    for ref in manifest['evidence_references']:
        if reference(ref['path'])!=ref:raise ValueError('Original RGB quarantine evidence changed')
    specs={}
    for item in manifest['specifications']:
        address=int(item['native_pts']),item['source_point_id']
        if (address in specs or item.get('authority')!='agent_rgb_review'
                or item.get('material_identity_confirmed')is not False
                or item.get('identity_confirmed_by_user')is not False
                or item.get('ordinary_two_pixel_components')is not True
                or not item.get('reason')or not item.get('evidence_references')
                or any(ref not in manifest['evidence_references']for ref in item.get('evidence_references',()))):
            raise ValueError('Distinct reviewed uncertain ordinary source rows required')
        specs[address]=item
    if not specs:raise ValueError('An empty identity quarantine is not a transition')
    result=[];archive=[];used=set();formal=[];raw=[];formal_offset=raw_offset=0
    for f in frames:
        old=FrozenWhitening(f);rows=f['rows'];flat=[(r,j)for r in rows for j in range(len(r['uv']))]
        keep=np.ones(len(flat),bool);grouped={int(i)for g in f['whitening_groups']for i in g['point_indices']}
        local_archive=[]
        for i,(r,j)in enumerate(flat):
            address=int(f['native_pts']),r['source_point_ids'][j]
            if address not in specs:continue
            item=specs[address]
            if (i in grouped or len(r['uv'])!=1 or not np.array_equal(r['uv'][j],item['original_UV'])
                    or float(r['sigma'][j])!=item['original_sigma']or r['target']!=item['original_target']
                    or r['source_packet']!=item['source_packet']):
                raise ValueError('Quarantine must match one complete original ordinary row, never a correlated W member')
            keep[i]=False;used.add(address)
            local_archive.append(dict(frame=int(f['index']),native_pts=address[0],original_row=r,
                review=deepcopy(item),manifest=deepcopy(manifest_reference),excluded_from_FIT=True))
        if keep.all():
            result.append(f);formal.extend(range(formal_offset,formal_offset+old.count))
        else:
            mapping=np.full(len(flat),-1,int);mapping[keep]=np.arange(int(keep.sum()))
            kept_rows=[];cursor=0
            for r in rows:
                mask=keep[cursor:cursor+len(r['uv'])];cursor+=len(r['uv'])
                if mask.any()and not mask.all():raise ValueError('Ordinary identity quarantine cannot split a native row')
                if mask.all():kept_rows.append(r)
            groups=[dict(g,point_indices=mapping[np.asarray(g['point_indices'],int)].tolist())for g in f['whitening_groups']]
            current=dict(f,rows=kept_rows,whitening_groups=groups,source_point_count=int(keep.sum()),
                inactive_ordinary_source_rows=list(f.get('inactive_ordinary_source_rows',()))+local_archive,
                ordinary_identity_quarantine=deepcopy(manifest_reference))
            new=FrozenWhitening(current);ordinary_keep=np.repeat(keep,2)[old.ordinary]
            selected=list(range(old.ordinary_start))+(old.ordinary_start+np.flatnonzero(ordinary_keep)).tolist()
            if (new.count!=len(selected)or not np.array_equal(new.scales,old.scales[selected])
                    or len(new.groups)!=len(old.groups)):
                raise ValueError('Ordinary identity quarantine changed retained likelihood or scales')
            for (oi,A,os),(ni,B,ns)in zip(old.groups,new.groups,strict=True):
                expected=2*mapping[np.asarray(oi)//2]+np.asarray(oi)%2
                if not np.array_equal(A,B)or not np.array_equal(expected,ni):
                    raise ValueError('Ordinary identity quarantine changed a complete correlated W')
            formal.extend((formal_offset+np.asarray(selected)).tolist());result.append(current);archive.extend(local_archive)
        raw.extend((raw_offset+np.flatnonzero(keep)).tolist());raw_offset+=len(flat);formal_offset+=old.count
    if used!=set(specs):raise ValueError('Reviewed ordinary source row absent from exact current source')
    return result,dict(retained_formal_indices=np.asarray(formal,np.int64),retained_source_point_indices=np.asarray(raw,np.int64),
        archive=archive,removed_source_points=len(used),removed_formal_rows=2*len(used),native_frames=len(frames),
        original_correlated_W_groups_and_member_order_byte_exact=True,original_datasets_UV_ID_sigma_and_approvals_modified=False)


def install_ordinary_quarantine(owner, manifest_reference):
    from .first_shot_fit import FrozenWhitening
    p=owner.programme;b=p.bundle
    if any(x is not None for x in (p.pool,b._source_pool,b._geometry_pool)):
        raise ValueError('Ordinary source identity transition must precede all workers')
    if getattr(p,'initial_J_manifest',None)is not None:raise ValueError('New source identity requires a fresh complete J')
    old_scales=p.scales.copy();old_quad=p.quadratic.copy();n=len(old_scales)
    stack=[owner];objects=[];seen=set();snapshots=[]
    while stack:
        obj=stack.pop()
        if id(obj)in seen:continue
        seen.add(id(obj));objects.append(obj);values=vars(obj)if hasattr(obj,'__dict__')else{}
        for name in ('programme','base','original','bundle','adapter','physical_stage'):
            if name in values:stack.append(values[name])
        if obj is p or obj is b:continue
        snapshots.append((obj,{k:values[k].copy()for k in ('scales','quadratic')if k in values}))
    frames,proof=quarantine_ordinary_rows(b.frames,manifest_reference)
    selected=proof['retained_formal_indices'];whitening=[FrozenWhitening(f)for f in frames]
    scales=np.concatenate([w.scales for w in whitening]);mask=old_quad[selected].copy()
    if not np.array_equal(scales,old_scales[selected]):raise ValueError('Retained original source scales changed')
    descriptors=OrderedDict();ledgers=[]
    for f,w,ledger in zip(frames,whitening,b.ledgers,strict=True):
        h=hashlib.sha256()
        for r in f['rows']:
            descriptors.setdefault(r['_target_key'],r['target']);h.update(json.dumps(r['source_point_ids'],separators=(',',':')).encode())
            h.update(r['uv'].tobytes());h.update(r['sigma'].tobytes())
        for g in f['whitening_groups']:h.update(g['operator'].matrix.tobytes())
        ledgers.append(dict(ledger,source_points=f['source_point_count'],formal_rows=w.count,
            source_inventory_sha256=h.hexdigest(),ordinary_identity_quarantine=deepcopy(manifest_reference)))
    b.frames=frames;b.whitening=whitening;b.ledgers=ledgers;b.scales=scales;b.quadratic=mask
    b.descriptors=descriptors;b.keys=tuple(descriptors);b.target_descriptors=tuple(descriptors.values())
    p.scales=scales.copy();p.quadratic=mask.copy();p.formal_offsets=np.r_[0,np.cumsum([w.count for w in whitening])]
    for obj,fields in snapshots:
        for name,value in fields.items():
            original=old_scales if name=='scales'else old_quad
            if len(value)<n or not np.array_equal(value[:n],original):raise ValueError('Outer source prefix differs from current programme')
            prefix=scales if name=='scales'else mask
            setattr(obj,name,np.r_[prefix,value[n:]].astype(value.dtype,copy=False))
    for obj,name in ((b,'_world_cache'),(b,'_finite_coverage_cache'),(p,'transverse_world_cache'),(p,'guard_cache')):
        getattr(obj,name).clear()
    p.metric=None;p._initial_complete_J_reuse=False
    receipt={k:deepcopy(v)for k,v in proof.items()if k not in ('archive','retained_formal_indices','retained_source_point_indices')}
    receipt.update(manifest=deepcopy(manifest_reference),producer=reference(__file__),
        retained_formal_indices_sha256=_sha(selected),retained_source_point_indices_sha256=_sha(proof['retained_source_point_indices']),
        old_source_r_J_H_reuse=False,original_native_line_support_and_W_retained=True)
    previous=owner.checkpoint_identity
    def identity(this):return dict(previous(),reviewed_ordinary_identity_quarantine=deepcopy(receipt))
    owner.checkpoint_identity=MethodType(identity,owner)
    for obj in objects:
        for name in ('source_producers','recovery_references','_references'):
            values=vars(obj).get(name)
            if isinstance(values,list):
                for ref in (manifest_reference,reference(__file__)):
                    if ref not in values:values.append(deepcopy(ref))
    owner.guard(owner.start)
    return receipt,proof
