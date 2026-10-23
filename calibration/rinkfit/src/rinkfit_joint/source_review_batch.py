"""Withdraw unconfirmed material identities after native-source visual review.

Reuse the existing immutable source storage and exact membership consumer.
Original pixels, IDs, source times and previous qualification history remain.
"""
import json, types
from . import dense_instance_support_batch as base
from .io_utils import read_json
from .dense_joint_proofs import _checked

def _multi_reviews(refs):
    sources=[];rows={};evidence=[]
    for ref in refs:
        doc=read_json(_checked(ref,base.ROOT))
        if doc.get('schema')!='rinkfit.native_visual_identity_withdrawal.v1' or doc.get('model_residual_used_for_qualification') is not False:
            raise ValueError('Native visual identity review required')
        source=doc['source_rows'];sources.append(source)
        evidence.extend([ref,doc['actual_review_consumer'],doc['source_audit']]+doc['evidence_refs'])
        for review in doc['reviews']:
            if review.get('qualification_reason_code')!='native_material_boundary_identity_unconfirmed' or review.get('qualification_action')!='withhold_world_identity' or review.get('to_qualification')!='unknown' or review.get('qualification_granted') is not False or review.get('source_instance_support_state')!='unknown' or review.get('positive_world_instance_support_refs')!=[]:
                raise ValueError('Only explicit unknown-identity withdrawal is supported')
            if not review.get('native_source_images') or not review.get('source_pixel_finding'):
                raise ValueError('Native image and actual source finding required')
            evidence.extend(review['native_source_images'])
            for ordinal in review['source_store_ordinals']:
                ordinal=int(ordinal)
                if ordinal in rows:raise ValueError('Overlapping reviewed source identities')
                rows[ordinal]=dict(review,source_rows=source)
    if not rows:raise ValueError('No reviewed source withdrawals')
    if len({json.dumps(s,sort_keys=True)for s in sources})!=len(sources):raise ValueError('Repeated semantic source part')
    for ref in evidence:_checked(ref,base.ROOT)
    return sources,rows,list({json.dumps(r,sort_keys=True):r for r in evidence}.values())

def _control():
    namespace=dict(base.__dict__,__file__=__file__,_multi_reviews=_multi_reviews)
    functions={}
    for name in ('commit_current_instance_support_reviews','consume_actual_current_instance_delta','build_instance_support_qualification_packet'):
        original=getattr(base,name);fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;functions[name]=fn
    namespace.update(functions);return functions

def commit_current_instance_support_reviews(**kwargs):
    return _control()['commit_current_instance_support_reviews'](**kwargs)

def build_instance_support_qualification_packet(**kwargs):
    return _control()['build_instance_support_qualification_packet'](**kwargs)
