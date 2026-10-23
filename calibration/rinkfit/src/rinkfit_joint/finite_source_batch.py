"""Apply source-span withdrawals without editing original observations.

Unlike a fit-residual filter, membership comes only from replay of the original
finite source annotation. Passing the scope gate does not grant new identity.
"""
import json,types
from pathlib import Path
import numpy as np
from . import dense_instance_support_batch as base
from .io_utils import read_json
from .dense_joint_proofs import _ref,_checked
from .dense_joint_packet import _array

REASONS={'outside_original_finite_source_spans','missing_finite_source_entity_support','historical_navigation_without_actual_finite_instance_support','source_guide_has_no_actual_finite_span_for_named_curve','template_navigation_without_actual_finite_instance_support'}

def _multi_reviews(refs):
    sources=[];rows={};evidence=[]
    for ref in refs:
        doc=read_json(_checked(ref,base.ROOT));audit=read_json(_checked(doc['source_audit'],base.ROOT))
        if doc.get('schema')!='rinkfit.accepted_finite_source_scope_reviews.v1' or doc.get('model_residual_used_for_qualification') is not False:raise ValueError('Expected finite source-only reviews')
        if audit.get('schema')!='rinkfit.actual_finite_source_scope_batch.v1' or audit.get('source_model_residual_used') is not False or audit.get('world_identity_granted') is not False:raise ValueError('Actual original finite source audit required')
        source=doc['source_rows'];sources.append(source)
        evidence.extend([ref,doc['source_audit'],doc['actual_review_consumer'],doc['membership']]+audit['source_anchors']+[audit['geometry_consumer'],audit['reader']])
        for review in doc['reviews']:
            if review.get('qualification_reason_code') not in REASONS or review.get('source_instance_support_state')!='unknown' or review.get('positive_world_instance_support_refs')!=[]:raise ValueError('Unknown source instance required')
            for ordinal in review['source_store_ordinals']:
                ordinal=int(ordinal)
                if ordinal in rows:raise ValueError('Overlapping source rows')
                rows[ordinal]=dict(review,source_rows=source)
    for ref in evidence:_checked(ref,base.ROOT)
    return sources,rows,list({json.dumps(r,sort_keys=True):r for r in evidence}.values())


def _validate(packet,refs):
    doc=read_json(packet);n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],Path(packet).parent)
    _,reviews,_=_multi_reviews(refs)
    expected={}
    for ref in refs:
        review=read_json(_checked(ref,base.ROOT))
        with np.load(_checked(review['membership'],base.ROOT),allow_pickle=False)as cols:
            for i,o,ok in zip(cols['packet_row'],cols['source_store_ordinal'],cols['supported']):
                if i<n0 or int(ordinal[int(i)-n0])!=int(o):raise ValueError('Original source ordinal changed')
                if not ok:expected[int(o)]=int(i)
    if set(reviews)!=set(expected):raise ValueError('Withdrawals differ from actual full source finite-scope result')


def _control():
    namespace=dict(base.__dict__,__file__=__file__,_multi_reviews=_multi_reviews)
    functions={}
    for name in ('commit_current_instance_support_reviews','consume_actual_current_instance_delta','build_instance_support_qualification_packet'):
        original=getattr(base,name);fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;functions[name]=fn
    namespace.update(functions);return functions


def commit_current_instance_support_reviews(**kwargs):
    _validate(kwargs['packet'],kwargs['review_refs'])
    return _control()['commit_current_instance_support_reviews'](**kwargs)


def build_instance_support_qualification_packet(**kwargs):
    _validate(kwargs['parent_packet'],kwargs['review_refs'])
    return _control()['build_instance_support_qualification_packet'](**kwargs)
