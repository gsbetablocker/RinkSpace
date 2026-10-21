"""Install the reviewed finite-side identity delta at an immutable batch boundary.

The existing multi-source control is reused. True turns remain finite; gray
material success and guide extrapolation never authorize a physical instance.
"""
from pathlib import Path
import types,json
import numpy as np
from . import dense_instance_support_batch as base
from .io_utils import read_json
from .dense_joint_proofs import _ref,_checked
from .dense_joint_packet import _array
from .dense_qualification_packet import _reviews

REASONS={
 's04':{'finite_side_before_source_turn_instance_unsupported','finite_physical_side_instance_support_unverified'},
 's06':{'finite_source_guide_span_instance_unverified'}
}


def _multi_reviews(refs):
    sources=[];rows={};evidence=[]
    for ref in refs:
        source,members,deps=_reviews([ref]);sources.append(source);evidence.extend(deps)
        for ordinal,r in members.items():
            if ordinal in rows or r['shot_id']not in REASONS or r.get('qualification_reason_code')not in REASONS[r['shot_id']]or r.get('source_instance_support_state')!='unknown'or r.get('positive_world_instance_support_refs')!=[]:raise ValueError('Only explicit finite-source-side reviews may withhold this identity')
            rows[ordinal]=dict(r,source_rows=source)
    if len({json.dumps(s,sort_keys=True)for s in sources})!=len(sources):raise ValueError('One review document per actual semantic source part required')
    evidence.extend([_ref(base.__file__),_ref(__file__)])
    return sources,rows,list({json.dumps(r,sort_keys=True):r for r in evidence}.values())


def _validate_actual_unknown_membership(packet,refs):
    path=Path(packet).resolve();doc=read_json(path);n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent);positions={int(o):i+n0 for i,o in enumerate(ordinal)}
    _multi_reviews(refs)
    for ref in refs:
        review=read_json(_checked(ref,base.ROOT));audit=read_json(_checked(review['source_audit'],base.ROOT));schema=audit.get('schema')
        if schema=='rinkfit.actual_finite_side_full_reader_instance_audit.v1':
            if audit['material_index']!=423 or audit['shot_id']!='s04'or audit['current_source_cohort_all_ID_UV_surface_definition_verified']is not True:raise ValueError('S04 actual finite-side source cohort not verified')
            aa={k:_array(audit['arrays'][k],Path(review['source_audit']['file']).parent)for k in('packet_row','source_store_ordinal','instance_state')};expected=set(map(int,aa['packet_row'][aa['instance_state']==0]));positive=set(map(int,aa['packet_row'][aa['instance_state']==1]))
            if len(expected)!=audit['current_instance_unknown_rows']or len(positive)!=audit['current_positive_finite_source_rows']:raise ValueError('Finite-side audit denominator differs')
        elif schema=='rinkfit.actual_S06_lane4_full_finite_source_guide_scope.v1':
            if audit['actual_selected_ID_UV_PTS_canonical_exact']is not True or audit['ice_plane_H_used_to_grant_world_instance']is not False or audit['world_model_residual_used']is not False:raise ValueError('S06 original finite-guide input evidence missing')
            with np.load(_checked(audit['scope_columns'],base.ROOT),allow_pickle=False)as cols:
                aa={k:cols[k].copy()for k in('packet_row','source_store_ordinal','outside_original_source_finite_support','canonical_provenance_query_replayed')}
            if not np.all(aa['canonical_provenance_query_replayed']==1):raise ValueError('Finite-guide original queries were not replayed')
            expected=set(map(int,aa['packet_row'][aa['outside_original_source_finite_support']==1]));positive=set()
            if len(expected)!=audit['outside_original_actual_source_finite_span']:raise ValueError('S06 explicit unknown cohort denominator differs')
        else:raise ValueError('Only the actual reviewed finite-side cohort formats supported')
        for ri,oo in zip(aa['packet_row'],aa['source_store_ordinal']):
            if int(ri)<n0 or positions[int(oo)]!=int(ri):raise ValueError('Finite-side source ordinal changes original identity')
        actual={positions[int(o)]for r in review['reviews']for o in r['source_store_ordinals']}
        if actual!=expected or actual&positive:raise ValueError('Unknown withdrawals differ from the actual source audit or include a positive finite entity')
    return True


def _control():
    namespace=dict(base.__dict__,__file__=__file__,_multi_reviews=_multi_reviews)
    functions={}
    for name in('commit_current_instance_support_reviews','consume_actual_current_instance_delta','build_instance_support_qualification_packet'):
        original=getattr(base,name);fn=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;functions[name]=fn
    namespace.update(functions);return functions


def commit_current_instance_support_reviews(**kwargs):
    _validate_actual_unknown_membership(kwargs['packet'],kwargs['review_refs']);return _control()['commit_current_instance_support_reviews'](**kwargs)


def build_instance_support_qualification_packet(**kwargs):
    _validate_actual_unknown_membership(kwargs['parent_packet'],kwargs['review_refs']);return _control()['build_instance_support_qualification_packet'](**kwargs)
