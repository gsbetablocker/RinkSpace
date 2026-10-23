"""Withdraw dense ice identities lacking positive native-material evidence.

A finite navigation search region is not a source identity certificate.
Preserve explicit positive reviews and every original pixel, ID and PTS.
"""
import json, types
import numpy as np
from . import dense_instance_support_batch as base
from .io_utils import read_json
from .dense_joint_proofs import _checked
from .joint_metrics import PacketArrays

REASON='source_navigation_only_without_positive_material_boundary_review'

def _multi_reviews(refs):
    sources=[];rows={};evidence=[];expected=None
    for ref in refs:
        doc=read_json(_checked(ref,base.ROOT));audit=read_json(_checked(doc['source_audit'],base.ROOT))
        if doc.get('schema')!='rinkfit.positive_source_support_absence_reviews.v1' or doc.get('model_residual_used_for_qualification') is not False:
            raise ValueError('Positive-source-support absence review required')
        if audit.get('schema')!='rinkfit.actual_positive_source_support_absence.v1' or audit.get('unreviewed_identity_is_unknown') is not True or audit.get('model_residual_used') is not False:
            raise ValueError('Actual source evidence audit required')
        if expected is None:
            p=PacketArrays(_checked(audit['source_packet'],base.ROOT));active=(p.load('admission')==1)&(p.load('fit_role')==1)
            registry=read_json(_checked(audit['existing_positive_registry'],base.ROOT));ring=read_json(_checked(audit['native_boundary_reviews'],base.ROOT))
            if registry['unreviewed_rows_default_to_confirmed'] is not False or ring['source_packet_sha256']!=audit['source_packet']['sha256']:
                raise ValueError('Source support identity/default changed')
            positive={r['packet_row']for r in registry['rows']if r['instance_state']==1}
            uv=p.load('uv');native_pts=p.load('native_pts')
            for r in ring['rows']:
                i=r['packet_row']
                if not r['visible_static_ice_material'] or p.packet['source_row_ids'][i]!=r['source_row_id'] or uv[i].tolist()!=r['uv'] or int(native_pts[i])!=r['native_pts']:
                    raise ValueError('Native positive evidence no longer matches source')
                positive.add(i)
            with np.load(_checked(audit['finite_navigation_membership'],base.ROOT),allow_pickle=False)as a:
                expected={int(o)for i,o in zip(a['packet_row'],a['source_store_ordinal'])if active[int(i)] and int(i)not in positive}
            evidence.extend([audit[k]for k in ('source_packet','existing_positive_registry','native_boundary_reviews','finite_navigation_membership','finite_navigation_report')])
        source=doc['source_rows'];sources.append(source);evidence.extend([ref,doc['source_audit'],doc['actual_review_consumer']])
        for review in doc['reviews']:
            if review.get('qualification_reason_code')!=REASON or review.get('source_instance_support_state')!='unknown' or review.get('positive_world_instance_support_refs')!=[]:
                raise ValueError('Only unconfirmed world-identity withdrawal is supported')
            for ordinal in review['source_store_ordinals']:
                ordinal=int(ordinal)
                if ordinal in rows:raise ValueError('Overlapping source identities')
                rows[ordinal]=dict(review,source_rows=source)
    if set(rows)!=expected:raise ValueError('Withdrawal differs from source-only evidence absence')
    for ref in evidence:_checked(ref,base.ROOT)
    return sources,rows,list({json.dumps(r,sort_keys=True):r for r in evidence}.values())

def _control():
    namespace=dict(base.__dict__,__file__=__file__,_multi_reviews=_multi_reviews);functions={}
    for name in ('commit_current_instance_support_reviews','consume_actual_current_instance_delta','build_instance_support_qualification_packet'):
        original=getattr(base,name);fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;functions[name]=fn
    namespace.update(functions);return functions

def commit_current_instance_support_reviews(**kwargs):
    return _control()['commit_current_instance_support_reviews'](**kwargs)

def build_instance_support_qualification_packet(**kwargs):
    return _control()['build_instance_support_qualification_packet'](**kwargs)
