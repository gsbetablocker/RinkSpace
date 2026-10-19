"""Withhold only the exact source associations explicitly withdrawn by a human.

A candidate query number is not an original point ID. No nearest UV matching,
new-frame coordinate transfer, background negative, or physical-target rejection
is performed. The original candidate packet remains intact when admission stops.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

from rink_feedback.core import source_key
from .feedback_runtime import (ROOT, atomic_json, authenticate_refs, feedback_store,
                               native_source, resolve_scope)
from .storage import AssetStore

VERSION='exact-human-association-withdrawal.v1'


class WithdrawnSourceAssociations(ValueError):
    def __init__(self, receipt):
        self.receipt=receipt
        super().__init__('Exact human-withdrawn source associations withheld before model admission; '
                         + str(receipt.get('receipt',{}).get('path','see exception.receipt')))


def _image_hashes(source):
    result={}
    for space,names in [('RGB',('decoded_RGB_sha256','decoded_native_RGB_sha256','decoded_native_rgb_sha256')),
                        ('BGR',('decoded_BGR_sha256','decoded_native_BGR_sha256','decoded_native_bgr_sha256'))]:
        values={source[name] for name in names if source.get(name)}
        if len(values)>1:raise ValueError('Conflicting decoded source image SHA aliases')
        if values:result[space]=values.pop()
    return result


def _withdrawal(case):
    claims=case.get('assertions',{})
    withdrawal=claims.get('withdrawal')
    if withdrawal and withdrawal.get('applicability')=='exact_source_frame_and_point_ids_only':
        return withdrawal.get('source'),withdrawal.get('points',[])
    edit=claims.get('source_edit')
    if edit and edit.get('exact_source_frame_and_point_ids_only') is True:
        return edit.get('source'),edit.get('deleted_points',[])
    return None,[]


def _candidate_key(row):
    pid,target,uv=row.get('point_id'),row.get('target_id'),row.get('uv')
    if (not isinstance(pid,str) or not pid or not isinstance(target,str) or not target
            or not isinstance(uv,(list,tuple)) or len(uv)!=2
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in uv)):
        return None
    return pid,target,tuple(float(v) for v in uv)


def check_candidate_withdrawals(source,candidates,*,project=ROOT,event_id=None,run_id=None,
                                gate='candidate_admission',packet_reference=None,record=True):
    """Return a receipt; the caller enforces non-admission if withheld is nonempty."""
    project=Path(project).resolve()
    own=native_source(source)
    own_hashes=_image_hashes(source)
    keys=[_candidate_key(row) for row in candidates]
    base=dict(schema='rinkspace.exact-human-withdrawal-check.v1',runtime_version=VERSION,
        source=own,candidate_count=len(candidates),stable_identity_count=sum(key is not None for key in keys),
        withheld=[],used_case_refs=[],gate=gate,background_negative_created=False,
        current_source_UV_changed=False,candidate_packet_preserved=True,
        cross_frame_or_cross_source_transfer=False,new_frame_visual_improvement_verified=False,
        packet_reference=packet_reference)
    if not any(key is not None for key in keys):
        return dict(base,status='not_applicable_missing_stable_point_ids_and_targets',case_use_recorded=False)
    resolved_event,programme=resolve_scope(own,project)
    if event_id and resolved_event and event_id!=resolved_event:
        raise ValueError('Withdrawal event conflicts with current native source registry')
    event_id=event_id or resolved_event
    if not event_id:
        return dict(base,status='unregistered_source_no_withdrawal_scope',case_use_recorded=False)
    if record:
        from .feedback_outbox import registered_source_dataset
        path=packet_reference.get('path') if isinstance(packet_reference,dict) else None
        registered,registration=registered_source_dataset(project,path,source) if path else (False,dict(reason='missing_registered_candidate_snapshot'))
        if (not registered or not Path(path).is_file()
                or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=packet_reference.get('sha256')):
            return dict(base,status='diagnostic_candidate_not_registered',case_use_recorded=False,
                production_feedback_index_consulted=False,registration=registration)
    context=dict(source=own,event_id=event_id,programme_id=programme,
        entity_ids=sorted({key[1] for key in keys if key}),tags=['manual_edit'],include_related_cases=False)
    run_id=run_id or 'admission_'+uuid.uuid4().hex
    matches=[];blocked=[];withheld={}
    with feedback_store(project,event_id) as store:
        for match in store.query(context):
            # Portable withdrawal cases are advisory until this adapter proves
            # the exact original source key, opaque point ID, target and UV.
            if match['match_status'] not in ('applicable','advisory'):continue
            bound,points=_withdrawal(match['case'])
            if not bound or not points or source_key(bound)!=source_key(own):continue
            source_hashes=_image_hashes(bound)
            reference={key:match[key] for key in ('case_id','version','content_sha256')}
            withdrawn_keys={key for row in points if (key:=_candidate_key(row)) is not None}
            if any(source_hashes[space]!=own_hashes[space] for space in source_hashes.keys() & own_hashes.keys()):
                blocked.append(dict(case_id=match['case_id'],reason='decoded_native_image_sha_differs'))
                for index,key in enumerate(keys):
                    if key is not None and key in withdrawn_keys:
                        withheld.setdefault(index,dict(candidate_index=index,**deepcopy(candidates[index]),
                            status='withheld_source_version_recheck_required',reason='decoded_native_image_sha_differs',
                            allowed_for_fit=False,background_negative=False,case_refs=[],blocking_case_refs=[reference]))
                continue
            authenticate_refs(match['case'].get('evidence_refs',[]),project)
            matches.append(reference)
            for index,key in enumerate(keys):
                if key is not None and key in withdrawn_keys:
                    record_row=withheld.setdefault(index,dict(candidate_index=index,**deepcopy(candidates[index]),
                        status='withheld_human_withdrawn_exact_association',reason='human_withdrawal_reason_unknown',
                        allowed_for_fit=False,background_negative=False,case_refs=[]))
                    record_row['case_refs'].append(reference)
        used={ref['case_id']:ref for row in withheld.values() for ref in row['case_refs']}
        result=dict(base,status='withheld' if withheld else 'checked_no_exact_withdrawal_match',
            event_id=event_id,run_id=run_id,withheld=list(withheld.values()),used_case_refs=list(used.values()),
            retrieved_case_refs=matches,blockers=blocked,case_use_recorded=False)
        if record and (matches or blocked):
            # The receipt records a real decision on these input IDs/UVs. It does
            # not claim a current image was visually reinterpreted by this gate.
            directory=AssetStore(project).data/'events'/event_id/'feedback'/'withdrawals'
            token=hashlib.sha256(json.dumps(dict(source=own,candidates=candidates,gate=gate),
                ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()[:20]
            ref=atomic_json(directory/(token+'_'+uuid.uuid4().hex[:12]+'.json'),result)
            toolref=dict(ref,kind='source_review_receipt',source=own)
            uses=[]
            for reference in used.values():
                uses.append(store.record_use(dict(context,**reference,run_id=run_id,
                    use_kind='exact_source_association_withdrawal',observation_ids=[row['point_id'] for row in withheld.values()
                        if any(r['case_id']==reference['case_id'] for r in row['case_refs'])],
                    tool_evidence_refs=[toolref],outcome='exact_human_withdrawn_association_withheld_before_'+gate)))
            result.update(receipt=ref,case_use_recorded=bool(uses),case_use_ids=[use['use_id'] for use in uses])
        return result


def packet_binding_candidates(packet,binding):
    """Read the binding's real stable IDs; never synthesize them from queries."""
    features={feature['id']:feature for feature in packet.get('features',[])}
    result=[];seen=set()
    for row in binding.get('observations',[]):
        feature=features[row['source_feature_id']]
        segment=feature['segments'][row['source_segment_index']]
        indices,ids=row['source_vertex_indices'],row['source_point_ids']
        if len(indices)!=len(ids):raise ValueError('Binding source point identity length differs')
        for index,pid in zip(indices,ids):
            if pid in seen:raise ValueError('Candidate source point IDs are not unique')
            seen.add(pid)
            if type(index) is not int or index<0 or index>=len(segment['uv']):
                raise ValueError('Binding source point address differs')
            if segment.get('point_ids') is not None and segment['point_ids'][index]!=pid:
                raise ValueError('Binding and packet stable point IDs differ')
            result.append(dict(point_id=pid,target_id=row['target_id'],uv=deepcopy(segment['uv'][index]),
                feature_id=row['source_feature_id'],source_segment_index=row['source_segment_index'],
                source_vertex_index=index))
    return result


def enforce_packet_withdrawals(packet,binding,*,project=ROOT,packet_path=None,gate='model_binding'):
    """Stop candidate admission and retain original packet plus withheld receipt."""
    if not packet_path:
        return dict(schema='rinkspace.exact-human-withdrawal-check.v1',runtime_version=VERSION,
            status='not_applicable_missing_registered_candidate_snapshot',stable_identity_count=0,
            withheld=[],case_use_recorded=False,production_feedback_index_consulted=False,
            candidate_packet_preserved=True,background_negative_created=False)
    if packet_path:
        from .feedback_outbox import registered_source_dataset
        registered,registration=registered_source_dataset(project,packet_path,packet['source'])
        if not registered:
            return dict(schema='rinkspace.exact-human-withdrawal-check.v1',runtime_version=VERSION,
                status='diagnostic_candidate_outside_registered_source_dataset',stable_identity_count=0,
                withheld=[],case_use_recorded=False,production_feedback_index_consulted=False,
                candidate_packet_preserved=True,background_negative_created=False,registration=registration)
    reference=None
    if packet_path and Path(packet_path).is_file():
        reference=dict(path=str(Path(packet_path).resolve()),
            sha256=hashlib.sha256(Path(packet_path).read_bytes()).hexdigest(),
            in_memory_candidate_sha256=hashlib.sha256(json.dumps(packet,ensure_ascii=False,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest())
    result=check_candidate_withdrawals(packet['source'],packet_binding_candidates(packet,binding),
        project=project,event_id=packet['source'].get('event_id'),gate=gate,packet_reference=reference)
    if result['withheld']:raise WithdrawnSourceAssociations(result)
    return result


def measurement_withdrawal_applicability(measurement):
    """Cache query arrays alone cannot identify a previously withdrawn point."""
    return dict(runtime_version=VERSION,
        status='not_applicable_missing_stable_point_ids_and_targets',
        source=deepcopy(measurement.get('source_binding')),
        candidate_count=len(measurement.get('candidate_uv',[])),
        candidate_query_is_not_original_point_identity=True,
        candidate_flags_changed=False,case_use_recorded=False,background_negative_created=False)
