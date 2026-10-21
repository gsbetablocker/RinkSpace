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

def commit_current_unknown_instance_reviews(*,packet,review_refs,base_frames,output,new_rows_stream,new_frames_stream):
    """Commit qualification-only fields; keep every selected source reading."""
    path=Path(packet).resolve();doc=read_json(path);n0=int(doc["dense_source_binding"]["row_start"])
    base_rows,reviews,evidence=_reviews(review_refs)
    binding=read_json(_checked(doc["dense_qualification_binding"],path.parent))
    part=next((x for x in binding["qualified_source_parts"] if x["reference"]==base_rows),None)
    if part is None or part["ordinal_start"]!=0:
        raise ValueError("This qualification branch requires the current S02-S04 semantic base")
    ordinals=_array(doc["dense_source_binding"]["source_store_ordinal"],path.parent)
    lookup={int(o):i+n0 for i,o in enumerate(ordinals) if int(o) in reviews}
    source_columns={k:_array(doc["arrays"][k],path.parent)for k in ("shot_index","native_pts","uv")}
    if set(lookup)!=set(reviews):raise ValueError("Review includes a source row absent from the actual physical packet")
    from .dense_fit_feedback import _one_frame
    changes=defaultdict(dict);originals={};frames={}
    for ordinal,review in reviews.items():
        index=lookup[ordinal];sid=doc["shot_ids"][int(source_columns["shot_index"][index])]
        pts=int(source_columns["native_pts"][index]);key=sid,pts
        if key!=(review["shot_id"],int(review["native_pts"])):raise ValueError("Review native exposure differs")
        if key not in frames:frames[key]=_one_frame(base_rows,*key)[0]
        ident=doc["source_row_ids"][index];before=frames[key][ident]
        if before.get("source_fit_eligible")is False or before.get("physical_instance_qualification")=="unknown":raise ValueError("An already unknown source row cannot be newly withdrawn")
        actual_uv=source_columns["uv"][index]
        if list(actual_uv)!=before["uv"]:raise ValueError("Original source UV differs from the packet")
        changes[key][ident]=dict(physical_instance_qualification="unknown",source_fit_eligible=False,
            qualification_reason_code="source_navigation_homography_pole_chord",qualification_review_id=review["review_id"])
        originals[str(ordinal)]=dict(selected_row_id=ident,source_point_id=before["source_point_id"],
            feature_id=before["feature_id"],shot_id=sid,native_pts=pts,
            prior_physical_instance_qualification=before.get("physical_instance_qualification","not_explicitly_withheld"))
    out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True)
    request=dict(schema="rinkspace.source_storage_commit_request.v1",db=base_rows["store"],base_rows=base_rows,
        base_frames=base_frames,new_rows_stream=new_rows_stream,new_frames_stream=new_frames_stream,
        changes=[dict(shot_id=k[0],native_pts=k[1],update=v)for k,v in sorted(changes.items())],
        source_evidence=[dict(file=str(_checked(ref,ROOT)),sha256=ref["sha256"])for ref in evidence])
    request_ref=_write(out/"source_request.json",request)
    spec=importlib.util.spec_from_file_location("rinkspace_public_source_storage",ROOT/"scripts/source_storage.py")
    public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    result=public.commit(request_ref["file"],out/"source_manifest.json")
    committed=read_json(result["manifest"])
    if committed["actual_validation"]["actual_transition"]["removed_selected_rows"]!=0:
        raise ValueError("A qualification withdrawal removed original source readings")
    _write(out/"review_membership.json",dict(schema="rinkfit.source_unknown_identity_review_membership.v1",
        source_commit=_ref(result["manifest"]),accepted_review_refs=review_refs,original_identity_membership=originals,
        source_UV_changed=False,source_rows_removed=False,qualification_only=True))
    return dict(source_generation=_ref(result["manifest"]),selected_rows=committed["selected_rows"],
        selected_frames=committed["selected_frames"],review_membership=_ref(out/"review_membership.json"),
        withdrawn_source_ordinals=sorted(reviews),evidence_refs=evidence)


def build_cumulative_qualification_packet(*,parent_packet,parent_source_qualification,parent_dense_weights,commit,review_refs,output):
    """Write exact-source child, next binding, qualification delta and budgets."""
    path=Path(parent_packet).resolve();doc=read_json(path);raw=json.loads(path.read_text(encoding="utf-8-sig"))
    prior_q_path=Path(parent_source_qualification).resolve();prior_q=read_json(prior_q_path)
    prior_w_path=Path(parent_dense_weights).resolve();prior_w=read_json(prior_w_path)
    if prior_q["source_packet"]!=_ref(path) or prior_w["source_packet"]!=_ref(path):raise ValueError("Parent proof binds another packet")
    source,reviews,evidence=_reviews(review_refs)
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
    if not any(part["reference"]==source for part in parts):raise ValueError("Accepted reviews are not bound to the actual current semantic source part")
    for part in parts:
        if part["reference"]==source:part["reference"]=commit["selected_rows"]
    next_binding=dict(schema="rinkfit.dense_source_fit_qualification_delta_input.v1",geometry_binding=geometry_binding,
        parent_packet=_ref(path),source_observation_generation=commit["source_generation"],
        qualified_source_parts=parts,withdrawn_source_store_ordinals=ordinal_ref,
        total_raw_rows=nnew,total_active_rows=nnew-len(withdrawn),total_unknown_rows=len(withdrawn),
        selected_source_ID_UV_nativePTS_order_unchanged=True,original_geometric_binding_preserved=True,
        review_membership=commit["review_membership"],accepted_source_review_refs=review_refs,issuer=_ref(__file__),
        parent_qualification_binding=current_binding_ref,new_unknown_rows=len(newly_withdrawn),
        changed_source_part_reference=commit["selected_rows"],prior_unknown_rows=len(prior_withdrawn))
    next_binding_ref=_write(out/"source_binding.json",next_binding)
    review_set_ref=_write(out/"source_review_evidence.json",dict(schema="rinkfit.accepted_dense_source_review_set.v1",
        accepted_source_reviews=review_refs,evidence_refs=evidence,source_rows=source,
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
    q=deepcopy(prior_q);q.update(source_packet=packet_ref,issuer=_ref(__file__),qualification_delta=delta_ref,
        active_new_rows=active_new,unknown_new_rows=len(withdrawn),source_qualification_binding=next_binding_ref,
        source_geometry_qualification_inherited_exact=True,source_accuracy_accepted=False,
        membership_array_sha256={k:v["sha256"]for k,v in arrays.items()},
        actual_input_references=prior_q["actual_input_references"]+evidence+[_ref(__file__),delta_ref,review_set_ref,
            next_binding_ref,commit["source_generation"],commit["review_membership"]])
    qual_ref=_write(out/"source_qualification.json",q)
    # Re-budget only the actual remaining families of each source exposure.
    rates={r["rig_id"]:float(r["engineering_budget_per_exposure"])for r in prior_w["per_rig_exposure_rates"]}
    shots_rigs={r["shot_id"]:r["rig_id"]for r in prior_w["per_shot_new_budgets"]}
    columns={k:_array(arrays[k],path.parent)for k in ("shot_index","native_pts")}
    mask=np.ones(n,dtype=bool);mask[:n0]=False;mask[indices]=False
    inherited={k:_array(prior_w["files"][k],prior_w_path.parent)for k in ("weights.f64","active.u8","information_class.u8")}
    weights=np.memmap(out/"weights.f64",dtype="<f8",mode="w+",shape=(n,));active=np.memmap(out/"active.u8",dtype="u1",mode="w+",shape=(n,));classes=np.memmap(out/"information_class.u8",dtype="u1",mode="w+",shape=(n,))
    weights[:n0]=inherited["weights.f64"][:n0];weights[n0:]=0;active[:n0]=inherited["active.u8"][:n0];active[n0:]=1;active[indices]=0;classes[:n0]=inherited["information_class.u8"][:n0];classes[n0:]=5;classes[indices]=0
    context=defaultdict(list);groups=[]
    for g in doc["engineering_pixel_dependency_groups"]["groups"]:
        ii=np.asarray(g["rows"],dtype=np.int64)
        if not len(ii) or np.max(ii)<n0:continue
        if np.min(ii)<n0:raise ValueError("Dense engineering group crosses retained source prefix")
        ii=ii[mask[ii]]
        if not len(ii):continue
        si=int(columns["shot_index"][ii[0]]);pts=int(columns["native_pts"][ii[0]]);sid=doc["shot_ids"][si]
        if np.any(columns["shot_index"][ii]!=si)or np.any(columns["native_pts"][ii]!=pts):raise ValueError("Dense group crosses exposure")
        item=dict(id=g["id"],indices=ii,shot=sid,pts=pts);groups.append(item);context[(sid,pts)].append(item)
    budgets=[];summ=defaultdict(lambda:dict(rows=0,exposures=set(),groups=0,budget=0.))
    covered=np.zeros(nnew,dtype=np.uint8)
    for item in groups:
        ii=item["indices"];sid=item["shot"];pts=item["pts"]
        if np.any(covered[ii-n0]):raise ValueError("Remaining engineering components overlap")
        covered[ii-n0]=1;per=rates[shots_rigs[sid]]/len(context[(sid,pts)])/len(ii);weights[ii]=per
        budgets.append(dict(id=item["id"],rows=len(ii),per_row_weight=per,engineering_budget=float(per*len(ii))))
        ss=summ[sid];ss["rows"]+=len(ii);ss["exposures"].add(pts);ss["groups"]+=1;ss["budget"]+=per*len(ii)
    if not np.array_equal(covered.astype(bool),mask[n0:]):raise ValueError("Remaining engineering groups do not partition qualified suffix")
    files={}
    for key,dtype,a in (("weights.f64","<f8",weights),("active.u8","|u1",active),("information_class.u8","|u1",classes)):
        a.flush()
        if a[:n0].tobytes()!=inherited[key][:n0].tobytes():raise ValueError("Parent engineering prefix changed")
        files[key]=dict(_ref(out/key),dtype=dtype,shape=[n])
    per_shot=[]
    for sid,s in sorted(summ.items()):
        old=next(x for x in prior_w["per_shot_new_budgets"] if x["shot_id"]==sid)
        expected=rates[shots_rigs[sid]]*len(s["exposures"])
        if not np.isclose(s["budget"],expected,rtol=2e-12,atol=1e-8):raise ValueError("Remaining budget increased with point count")
        per_shot.append(dict(old,new_rows=s["rows"],covered_native_exposures=len(s["exposures"]),named_material_exposure_components=s["groups"],engineering_budget=expected))
    w=deepcopy(prior_w);retained=float(np.sum(weights[:n0]));new=float(np.sum(weights[n0:]))
    w.update(source_packet=packet_ref,issuer=_ref(__file__),dense_source_qualification=qual_ref,
        active_rows=int(np.sum(active!=0)),new_active_rows=active_new,files=files,component_budgets=budgets,
        retained_parent_budget=retained,new_dense_budget=new,actual_total_engineering_budget=retained+new,
        per_shot_new_budgets=per_shot,source_qualification_delta=delta_ref,unknown_source_rows_have_zero_budget=True)
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


def assert_cumulative_qualification_binding(binding_path,packet_path):
    """Consume the committed current stream and verify every withdrawn source row."""
    from .source_record_io import checked_records
    binding_path=Path(binding_path).resolve();binding=read_json(binding_path)
    path=Path(packet_path).resolve();doc=read_json(path);n0=int(doc["dense_source_binding"]["row_start"])
    ordinal=_array(doc["dense_source_binding"]["source_store_ordinal"],path.parent)
    withdrawn=set(map(int,_array(binding["withdrawn_source_store_ordinals"],binding_path.parent)))
    positions={int(o):i+n0 for i,o in enumerate(ordinal)if int(o)in withdrawn}
    uv=_array(doc["arrays"]["uv"],path.parent);pts=_array(doc["arrays"]["native_pts"],path.parent)
    si=_array(doc["arrays"]["shot_index"],path.parent);seen=set();count=0
    changed=[p for p in binding["qualified_source_parts"]if p["reference"]==binding["changed_source_part_reference"]]
    if len(changed)!=1:raise ValueError("This qualification consumer expects one explicit changed semantic source part")
    part=changed[0];start=int(part["ordinal_start"])
    for number,row in enumerate(checked_records(part["reference"])):
        global_ordinal=start+number;count+=1
        if global_ordinal not in withdrawn:continue
        index=positions[global_ordinal]
        if row["selected_row_id"]!=doc["source_row_ids"][index] or row["uv"]!=list(uv[index]) or int(row["native_pts"])!=int(pts[index]) or row["shot_id"]!=doc["shot_ids"][int(si[index])]:
            raise ValueError("Committed current qualification changed original source geometry or identity")
        if row.get("source_fit_eligible")is not False or row.get("physical_instance_qualification")!="unknown" or row.get("qualification_reason_code")!="source_navigation_homography_pole_chord":
            raise ValueError("Committed current source row did not install accepted unknown qualification")
        seen.add(global_ordinal)
    if count!=int(part["row_count"]) or seen!=withdrawn:raise ValueError("Current qualification stream denominator or withdrawal membership differs")
    return dict(schema="rinkfit.current_dense_qualification_actual_consumption.v1",passed=True,
        source_qualification_binding=_ref(binding_path),packet=_ref(path),
        full_qualified_source_stream_semantics_consumed=True,current_semantic_rows=count,
        current_unknown_source_rows_checked=len(seen),original_UV_ID_nativePTS_unchanged=True,
        original_pixel_readings_regenerated=False,physical_solver_started=False)


