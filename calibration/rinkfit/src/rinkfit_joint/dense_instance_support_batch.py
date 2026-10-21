"""Install a subsequent source-only unknown-identity batch on the current stream.

This narrow dense-six continuation retains the original raw geometry registry,
previous unknown rows, parent W/Query and engineering-prefix bits. Only explicit
new accepted reader evidence can extend the cumulative unknown membership.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import numpy as np
from .io_utils import read_json
from .dense_joint_packet import _array
from .dense_joint_proofs import _ref,_checked,_write
from .dense_qualification_packet import _reviews
ROOT=Path(__file__).resolve().parents[4]



def _multi_reviews(refs):
    from .dense_qualification_packet import _reviews
    sources=[];reviews={};evidence=[]
    for ref in refs:
        source,rows,deps=_reviews([ref]);sources.append(source);evidence.extend(deps)
        for ordinal,row in rows.items():
            if ordinal in reviews:raise ValueError('Accepted source-instance reviews overlap')
            if row.get('qualification_reason_code')not in ('inference_only_circle_donor_world_instance_unverified','template_only_circle_fallback_world_instance_unverified')or row.get('source_instance_support_state')!='unknown'or row.get('positive_world_instance_support_refs')!=[]:raise ValueError('Explicit source-positive-support absence required; wrong pixels are not inferred')
            reviews[ordinal]=dict(row,source_rows=source)
    if len({json.dumps(s,sort_keys=True)for s in sources})!=len(sources):raise ValueError('Each changed semantic source part must be reviewed once')
    unique={json.dumps(r,sort_keys=True):r for r in evidence}
    return sources,reviews,list(unique.values())


def commit_current_instance_support_reviews(*,packet,review_refs,base_frames_by_stream,output,new_stream_prefix):
    path=Path(packet).resolve();doc=read_json(path);n0=int(doc['dense_source_binding']['row_start']);sources,reviews,evidence=_multi_reviews(review_refs)
    binding=read_json(_checked(doc['dense_qualification_binding'],path.parent));parts=binding['qualified_source_parts']
    ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent);positions={int(o):i+n0 for i,o in enumerate(ordinal)if int(o)in reviews}
    if set(positions)!=set(reviews):raise ValueError('Reviewed source ordinals are absent from the current packet')
    columns={k:_array(doc['arrays'][k],path.parent)for k in('shot_index','native_pts','uv')}
    from .dense_fit_feedback import _one_frame
    groups={json.dumps(s,sort_keys=True):defaultdict(dict)for s in sources};frames={};originals={}
    for o,review in reviews.items():
        index=positions[o];sid=doc['shot_ids'][int(columns['shot_index'][index])];pts=int(columns['native_pts'][index]);source=review['source_rows'];part=next((p for p in parts if p['reference']==source),None)
        if part is None or not int(part['ordinal_start'])<=o<int(part['ordinal_start'])+int(part['row_count']):raise ValueError('Reviewed ordinal belongs to another current source part')
        if (sid,pts)!=(review['shot_id'],int(review['native_pts'])):raise ValueError('Reviewed native clock differs')
        key=json.dumps(source,sort_keys=True),sid,pts
        if key not in frames:frames[key]=_one_frame(source,sid,pts)[0]
        ident=doc['source_row_ids'][index];before=frames[key][ident]
        if before['uv']!=list(columns['uv'][index])or before.get('source_fit_eligible')is False or before.get('physical_instance_qualification')=='unknown':raise ValueError('Source geometry changed or withdrawal repeats an existing unknown')
        groups[key[0]][sid,pts][ident]=dict(physical_instance_qualification='unknown',source_fit_eligible=False,qualification_reason_code=review['qualification_reason_code'],qualification_review_id=review['review_id'],physical_instance_support_state='unknown')
        originals[str(o)]=dict(selected_row_id=ident,source_point_id=before['source_point_id'],feature_id=before['feature_id'],shot_id=sid,native_pts=pts,prior_physical_instance_qualification=before.get('physical_instance_qualification','not_explicitly_withheld'))
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True)
    spec=importlib.util.spec_from_file_location('rinkspace_public_source_storage',ROOT/'scripts/source_storage.py');public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    commits=[]
    for source in sources:
        part=next(p for p in parts if p['reference']==source);key=json.dumps(source,sort_keys=True);change=groups[key];name=part['name'];sub=out/name;sub.mkdir()
        frames_ref=base_frames_by_stream[source['stream']]
        request=dict(schema='rinkspace.source_storage_commit_request.v1',db=source['store'],base_rows=source,base_frames=frames_ref,new_rows_stream=new_stream_prefix+'/'+name+'/selected_rows',new_frames_stream=new_stream_prefix+'/'+name+'/selected_frames',changes=[dict(shot_id=k[0],native_pts=k[1],update=v)for k,v in sorted(change.items())],source_evidence=[dict(file=str(_checked(r,ROOT)),sha256=r['sha256'])for r in evidence])
        request_ref=_write(sub/'source_request.json',request);result=public.commit(request_ref['file'],sub/'source_manifest.json');published=read_json(result['manifest'])
        transition=published['actual_validation']['actual_transition']
        if transition['removed_selected_rows']!=0 or published['source_UV_unchanged']is not True:raise ValueError('Source-instance qualification changed original readings')
        commits.append(dict(name=name,parent_reference=source,selected_rows=published['selected_rows'],selected_frames=published['selected_frames'],source_generation=_ref(result['manifest']),new_unknown_rows=sum(len(v)for v in change.values()),actual_full_source_validation=published['actual_validation']))
    generation=_write(out/'source_generation.json',dict(schema='rinkfit.multi_part_source_qualification_generation.v1',complete=True,source_commits=commits,issuer=_ref(__file__),source_UV_ID_PTS_unchanged=True,source_pixels_decoded=0,source_instance_unknown_is_not_wrong_pixel=True))
    membership=_write(out/'review_membership.json',dict(schema='rinkfit.source_unknown_identity_review_membership.v1',source_commit=generation,accepted_review_refs=review_refs,original_identity_membership=originals,source_UV_changed=False,source_rows_removed=False,qualification_only=True))
    return dict(source_generation=generation,selected_parts=commits,review_membership=membership,withdrawn_source_ordinals=sorted(reviews),evidence_refs=evidence+[p['source_generation']for p in commits])


def consume_actual_current_instance_delta(*,packet,qualification_binding,commit,review_refs,output):
    from .dense_fit_feedback import _one_frame
    path=Path(packet).resolve();doc=read_json(path);binding_path=Path(qualification_binding).resolve();binding=read_json(binding_path);sources,reviews,evidence=_multi_reviews(review_refs)
    n0=int(doc['dense_source_binding']['row_start']);ordinal=_array(doc['dense_source_binding']['source_store_ordinal'],path.parent);positions={int(o):i+n0 for i,o in enumerate(ordinal)if int(o)in reviews}
    cols={k:_array(doc['arrays'][k],path.parent)for k in('uv','native_pts','shot_index','admission','fit_role')};seen=set();frames={}
    for o,review in reviews.items():
        index=positions[o];part=next(p for p in commit['selected_parts']if p['parent_reference']==review['source_rows']);source=part['selected_rows'];sid=doc['shot_ids'][int(cols['shot_index'][index])];pts=int(cols['native_pts'][index]);key=source['stream'],sid,pts
        if key not in frames:frames[key]=_one_frame(source,sid,pts)[0]
        row=frames[key][doc['source_row_ids'][index]]
        if row['uv']!=list(cols['uv'][index])or int(row['native_pts'])!=pts or row['shot_id']!=sid or row.get('source_fit_eligible')is not False or row.get('physical_instance_qualification')!='unknown'or row.get('qualification_reason_code')!=review['qualification_reason_code']or int(cols['admission'][index])!=0 or int(cols['fit_role'][index])!=0:raise ValueError('New logical source qualification was not actually consumed by the new packet')
        seen.add(o)
    if seen!=set(reviews):raise ValueError('New source-instance delta membership was not fully consumed')
    receipt=dict(schema='rinkfit.actual_multi_source_instance_qualification_consumption.v1',passed=True,source_packet=_ref(path),source_qualification_binding=_ref(binding_path),source_observation_generation=commit['source_generation'],actual_consumer=_ref(__file__),new_unknown_source_rows_checked=len(seen),actual_source_native_exposure_frames=len(frames),reason_counts=dict(__import__('collections').Counter(r['qualification_reason_code']for r in reviews.values())),source_UV_ID_PTS_unchanged=True,all_new_membership_exact=True,full_source_semantics_verified_by_public_source_commit=True,prior_full_geometry_qualification_inherited_without_redecoding=True,source_pixels_decoded=0,physical_fit_started=False)
    return _write(output,receipt)

def build_instance_support_qualification_packet(*,parent_packet,parent_source_qualification,parent_dense_weights,commit,review_refs,output):
    """Write exact-source child, next binding, qualification delta and budgets."""
    path=Path(parent_packet).resolve();doc=read_json(path);raw=json.loads(path.read_text(encoding="utf-8-sig"))
    prior_q_path=Path(parent_source_qualification).resolve();prior_q=read_json(prior_q_path)
    prior_w_path=Path(parent_dense_weights).resolve();prior_w=read_json(prior_w_path)
    if prior_q["source_packet"]!=_ref(path) or prior_w["source_packet"]!=_ref(path):raise ValueError("Parent proof binds another packet")
    sources,reviews,evidence=_multi_reviews(review_refs)
    newly_withdrawn=set(map(int,commit["withdrawn_source_ordinals"]))
    if newly_withdrawn!=set(reviews):raise ValueError("New commit differs from accepted current-generation reviews")
    current_binding_ref=doc["dense_qualification_binding"];current_binding=read_json(_checked(current_binding_ref,path.parent))
    prior_withdrawn=set(map(int,_array(current_binding["withdrawn_source_store_ordinals"],path.parent)))
    if prior_withdrawn&newly_withdrawn:raise ValueError("Withdraw-only batch repeats an already unknown source row")
    withdrawn=np.asarray(sorted(prior_withdrawn|newly_withdrawn),dtype="<u8")
    n0=int(doc["dense_source_binding"]["row_start"]);n=int(doc["row_count"]);nnew=n-n0
    ordinals=_array(doc["dense_source_binding"]["source_store_ordinal"],path.parent)
    hit=np.isin(ordinals,withdrawn);indices=np.flatnonzero(hit)+n0
    if len(indices)!=len(withdrawn):raise ValueError("Withdrawn source ordinals are not unique packet members")
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True)
    withdrawn.tofile(out/"withdrawn_source_store_ordinals.u64")
    ordinal_ref=dict(_ref(out/"withdrawn_source_store_ordinals.u64"),dtype="<u8",shape=[len(withdrawn)])
    geometry_binding=doc["dense_source_binding"]["input_binding"]
    old_binding=read_json(_checked(geometry_binding,path.parent));parts=deepcopy(current_binding["qualified_source_parts"])
    if any(not any(part['reference']==source for part in parts)for source in sources):raise ValueError('Accepted reviews are not bound to current semantic source parts')
    for part in parts:
        changed=next((p for p in commit['selected_parts']if p['parent_reference']==part['reference']),None)
        if changed is not None:part['reference']=changed['selected_rows']
    next_binding=dict(schema="rinkfit.dense_source_fit_qualification_delta_input.v1",geometry_binding=geometry_binding,
        parent_packet=_ref(path),source_observation_generation=commit["source_generation"],
        qualified_source_parts=parts,withdrawn_source_store_ordinals=ordinal_ref,
        total_raw_rows=nnew,total_active_rows=nnew-len(withdrawn),total_unknown_rows=len(withdrawn),
        selected_source_ID_UV_nativePTS_order_unchanged=True,original_geometric_binding_preserved=True,
        review_membership=commit["review_membership"],accepted_source_review_refs=review_refs,issuer=_ref(__file__),
        parent_qualification_binding=current_binding_ref,new_unknown_rows=len(newly_withdrawn),
        changed_source_part_references=[p['selected_rows']for p in commit['selected_parts']],prior_unknown_rows=len(prior_withdrawn))
    next_binding_ref=_write(out/"source_binding.json",next_binding)
    review_set_ref=_write(out/"source_review_evidence.json",dict(schema="rinkfit.accepted_dense_source_review_set.v1",
        accepted_source_reviews=review_refs,evidence_refs=evidence,source_parts=sources,
        unknown_source_store_ordinals=ordinal_ref,actual_qualified_source_generation=commit["source_generation"],
        original_review_membership=commit["review_membership"],parent_source_qualification=_ref(prior_q_path),
        parent_source_review_evidence=read_json(_checked(doc["source_qualification_delta"],path.parent))["source_review_evidence"],
        raw_rows=nnew,unknown_rows=len(withdrawn),new_unknown_rows=len(newly_withdrawn),
        model_residual_used_for_qualification=False,source_UV_changed=False,issuer=_ref(__file__)))
    delta=dict(schema="rinkfit.dense_source_qualification_delta.v1",parent_source_qualification=_ref(prior_q_path),
        source_packet=_ref(path),actual_review_consumer=_ref(__file__),
        source_observation_generation=commit["source_generation"],source_review_evidence=review_set_ref,
        withdrawn_source_store_ordinals=ordinal_ref,from_qualification="eligible",to_qualification="unknown",
        source_semantics_verified=False,qualification_granted=False,selected_by_model_residual=False,
        cumulative_withdraw_only=True,new_unknown_rows=len(newly_withdrawn),prior_unknown_rows=len(prior_withdrawn),
        original_source_UV_ID_PTS_material_path_preserved=True,
        source_qualification_binding=next_binding_ref)
    delta_ref=_write(out/"qualification_delta.json",delta)
    arrays=dict(doc["arrays"])
    for key in ("admission","fit_role"):
        original=_array(arrays[key],path.parent);a=np.array(original,copy=True);a[indices]=0
        target=out/(key+".u8");a.tofile(target)
        arrays[key]=dict(_ref(target),dtype=a.dtype.str,shape=list(a.shape),layout="C",byte_order="little")
        if a[:n0].tobytes()!=np.asarray(original[:n0]).tobytes():raise ValueError("Qualification changed parent membership prefix")
    active_new=nnew-len(withdrawn)
    child=dict(schema=doc["schema"],dense_parent_packet=_ref(path),arrays=arrays,
        active_rows=int(doc["active_rows"])-len(newly_withdrawn),dense_qualification_binding=next_binding_ref,
        source_qualification_delta=delta_ref,
        dense_parent_append=dict(source_manifests=[next_binding_ref,delta_ref,review_set_ref,_ref(__file__)]))
    packet_ref=_write(out/"packet.json",child)
    actual_consumption=consume_actual_current_instance_delta(packet=packet_ref['file'],qualification_binding=next_binding_ref['file'],commit=commit,review_refs=review_refs,output=out/'actual_qualification_delta_consumption.json')
    q=deepcopy(prior_q);q.update(source_packet=packet_ref,issuer=_ref(__file__),qualification_delta=delta_ref,
        active_new_rows=active_new,unknown_new_rows=len(withdrawn),source_qualification_binding=next_binding_ref,
        source_geometry_qualification_inherited_exact=True,source_accuracy_accepted=False,
        membership_array_sha256={k:v["sha256"]for k,v in arrays.items()},
        actual_input_references=prior_q["actual_input_references"]+evidence+[_ref(__file__),delta_ref,review_set_ref,
            next_binding_ref,commit["source_generation"],commit["review_membership"],actual_consumption])
    qual_ref=_write(out/"source_qualification.json",q)
    # Qualification withdrawal never redistributes its budget to other rows.
    inherited={k:_array(prior_w['files'][k],prior_w_path.parent)for k in('weights.f64','active.u8','information_class.u8')}
    current_weights=np.array(inherited['weights.f64'],copy=True);current_active=np.array(inherited['active.u8'],copy=True);current_classes=np.array(inherited['information_class.u8'],copy=True)
    newly_indices=np.flatnonzero(np.isin(ordinals,np.asarray(sorted(newly_withdrawn),dtype='<u8')))+n0
    current_weights[newly_indices]=0;current_active[newly_indices]=0;current_classes[newly_indices]=0
    untouched=np.ones(n,dtype=bool);untouched[newly_indices]=False
    for key,a in [('weights.f64',current_weights),('active.u8',current_active),('information_class.u8',current_classes)]:
        if a[untouched].tobytes()!=np.asarray(inherited[key][untouched]).tobytes():raise ValueError('Retained source absolute weight/class/active bits changed')
    files={}
    for key,dtype,a in [('weights.f64','<f8',current_weights),('active.u8','|u1',current_active),('information_class.u8','|u1',current_classes)]:
        target=out/key;a.tofile(target);files[key]=dict(_ref(target),dtype=dtype,shape=[n])
    budgets=[];summ=defaultdict(lambda:dict(rows=0,exposures=set(),groups=0,budget=0.));covered=np.zeros(nnew,dtype='u1')
    columns={k:_array(arrays[k],path.parent)for k in('shot_index','native_pts')}
    for group in doc['engineering_pixel_dependency_groups']['groups']:
        ii=np.asarray(group['rows'],dtype=np.int64)
        if not len(ii)or np.max(ii)<n0:continue
        if np.min(ii)<n0:raise ValueError('Dense component crosses original source prefix')
        ii=ii[current_active[ii]!=0]
        if not len(ii):continue
        if np.any(covered[ii-n0]):raise ValueError('Retained active engineering groups overlap')
        covered[ii-n0]=1;values=current_weights[ii];per=float(values[0])
        if not np.isfinite(values).all()or per<=0 or np.any(values!=per):raise ValueError('Retained component requires its exact positive inherited per-row engineering weight')
        amount=float(per*len(ii));budgets.append(dict(id=group['id'],rows=len(ii),per_row_weight=per,engineering_budget=amount))
        sid=doc['shot_ids'][int(columns['shot_index'][ii[0]])];pts=int(columns['native_pts'][ii[0]]);s=summ[sid];s['rows']+=len(ii);s['exposures'].add(pts);s['groups']+=1;s['budget']+=amount
    if not np.array_equal(covered.astype(bool),current_active[n0:]!=0):raise ValueError('Retained groups do not partition active suffix')
    per_shot=[]
    for sid,s in sorted(summ.items()):
        old=next(x for x in prior_w['per_shot_new_budgets']if x['shot_id']==sid);per_shot.append(dict(old,new_rows=s['rows'],covered_native_exposures=len(s['exposures']),named_material_exposure_components=s['groups'],engineering_budget=s['budget'],absolute_prior_per_row_weights_preserved=True))
    w=deepcopy(prior_w);retained=float(np.sum(current_weights[:n0]));new=float(np.sum(current_weights[n0:]));active=current_active
    w.update(source_packet=packet_ref,issuer=_ref(__file__),dense_source_qualification=qual_ref,active_rows=int(np.sum(active!=0)),new_active_rows=active_new,files=files,component_budgets=budgets,retained_parent_budget=retained,new_dense_budget=new,actual_total_engineering_budget=retained+new,per_shot_new_budgets=per_shot,source_qualification_delta=delta_ref,unknown_source_rows_have_zero_budget=True,withdrawn_budget_not_redistributed=True,retained_source_absolute_weight_bits_preserved=True,withdrawn_absolute_budget=float(np.sum(np.asarray(inherited['weights.f64'])[newly_indices])),parent_dense_weights=_ref(prior_w_path))
    weight_ref=_write(out/"engineering_weights.json",w)
    result=dict(schema="rinkfit.dense_qualification_next_packet.v1",complete=True,packet=packet_ref,
        source_qualification=qual_ref,engineering_weights=weight_ref,source_binding=next_binding_ref,
        qualification_delta=delta_ref,source_generation=commit["source_generation"],raw_rows=n,
        active_rows=int(np.sum(active!=0)),new_raw_rows=nnew,new_active_rows=active_new,new_unknown_rows=len(withdrawn),
        original_source_UV_ID_nativePTS_material_path_ordinals_preserved=True,parent_all_weights_bits_preserved=True,
        original_statistical_W_Query_and_connected_raw_groups_preserved=True,
        source_only_qualification_delta=True,source_pixels_moved=False,new_source_geometry_issued=False,
        numeric_auxiliary_reused_for_retained_source_rows=True,unknown_rows_in_raw_total_denominator=True,
        physical_solver_started=False,quality_accepted=False)
    _write(out/"receipt.json",result)
    return result

