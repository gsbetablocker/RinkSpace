"""Install explicit finite-ring source reviews with the issued dense batch control.

The existing multi-source writer and immutable packet builder are reused. Only
source-reviewed membership is admitted to this policy; retained absolute weights
and measured UV stay unchanged. Positive finite arcs are never withdrawn.
"""
from pathlib import Path
import types,json
from . import dense_instance_support_batch as base
from .io_utils import read_json
from .dense_joint_proofs import _ref,_checked
from .dense_joint_packet import _array
from .dense_qualification_packet import _reviews

REASONS={'source_visibility_excluded_current_finite_ring','finite_ring_layer_positive_source_support_unresolved','observed_material_not_claimed_finite_ring_instance'}
CASES={('s02',600320),('s05',1035776)}


def _multi_reviews(refs):
    sources=[];rows={};evidence=[]
    for ref in refs:
        source,members,deps=_reviews([ref]);sources.append(source);evidence.extend(deps)
        for ordinal,r in members.items():
            if ordinal in rows or (r['shot_id'],int(r['native_pts']))not in CASES or r.get('qualification_reason_code')not in REASONS or r.get('source_instance_support_state')!='unknown' or r.get('positive_world_instance_support_refs')!=[]:raise ValueError('Only the explicit finite-source ring review cases may withhold instance support')
            rows[ordinal]=dict(r,source_rows=source)
    if len({json.dumps(s,sort_keys=True)for s in sources})!=len(sources):raise ValueError('One review document per actual source semantic part is required')
    evidence.extend([_ref(base.__file__),_ref(__file__)])
    unique={json.dumps(r,sort_keys=True):r for r in evidence}
    return sources,rows,list(unique.values())


def _validate_actual_unknown_membership(packet,refs):
    path=Path(packet).resolve();doc=read_json(path);n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent);positions={int(o):i+n0 for i,o in enumerate(ordinal)}
    _,reviews,_=_multi_reviews(refs)
    for ref in refs:
        review=read_json(_checked(ref,base.ROOT));audit=read_json(_checked(review['source_audit'],base.ROOT));items=audit.get('per_row',audit.get('source_rows',[]));expected={int(r['packet_row']):r for r in items if int(r['instance_state'])==0};positive={int(r['packet_row'])for r in items if int(r['instance_state'])==1}
        actual={positions[int(o)] for r in review['reviews']for o in r['source_store_ordinals']}
        if actual!=set(expected)or actual&positive:raise ValueError('Withdrawal differs from the complete actual finite-source audit or includes a positive arc')
        for i in actual:
            if doc['source_row_ids'][i]!=expected[i]['selected_row_id']:raise ValueError('Finite ring actual selected source ID differs')
    return True


def _control():
    namespace=dict(base.__dict__,__file__=__file__,_multi_reviews=_multi_reviews)
    functions={}
    for name in('commit_current_instance_support_reviews','consume_actual_current_instance_delta','build_instance_support_qualification_packet'):
        original=getattr(base,name);fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;functions[name]=fn
    namespace.update(functions)
    return functions


def commit_current_instance_support_reviews(**kwargs):
    _validate_actual_unknown_membership(kwargs['packet'],kwargs['review_refs'])
    return _control()['commit_current_instance_support_reviews'](**kwargs)


def build_instance_support_qualification_packet(**kwargs):
    _validate_actual_unknown_membership(kwargs['parent_packet'],kwargs['review_refs'])
    return _control()['build_instance_support_qualification_packet'](**kwargs)
