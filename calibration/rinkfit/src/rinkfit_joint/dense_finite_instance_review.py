"""Commit source-only finite-instance uncertainty without changing measured UV.

A real material profile on an extended cross-shot guide does not qualify the
finite named entity. Only explicit reviewed source evidence is installed; the
legacy source arrays, numeric auxiliaries and accepted controls are preserved.
"""
from __future__ import annotations
from collections import defaultdict
import importlib.util
from pathlib import Path
from .io_utils import read_json
from .dense_joint_packet import _array
from .dense_joint_proofs import _ref,_checked,_write
from .dense_qualification_packet import _reviews
ROOT=Path(__file__).resolve().parents[4]

def commit_finite_instance_reviews(*,packet,review_refs,base_frames,output,new_rows_stream,new_frames_stream):
    """Commit qualification-only fields; keep every selected source reading."""
    path=Path(packet).resolve();doc=read_json(path);n0=int(doc["dense_source_binding"]["row_start"])
    base_rows,reviews,evidence=_reviews(review_refs)
    for review in reviews.values():
        if review.get("qualification_reason_code")!="finite_cross_shot_material_instance_unsupported" or review.get("finite_source_identity_supported")is not False or review.get("source_UV_retained")is not True:
            raise ValueError("This consumer requires explicit finite-source-instance review evidence")
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
            qualification_reason_code=review["qualification_reason_code"],qualification_review_id=review["review_id"])
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



def assert_current_instance_qualification_binding(binding_path,packet_path):
    """Consume the sealed current stream and each explicit cumulative reason."""
    from .source_record_io import checked_records
    binding_path=Path(binding_path).resolve();binding=read_json(binding_path)
    path=Path(packet_path).resolve();doc=read_json(path);n0=int(doc["dense_source_binding"]["row_start"])
    ordinal=_array(doc["dense_source_binding"]["source_store_ordinal"],path.parent)
    withdrawn=set(map(int,_array(binding["withdrawn_source_store_ordinals"],binding_path.parent)))
    positions={int(o):i+n0 for i,o in enumerate(ordinal)if int(o)in withdrawn}
    expected={};cursor=binding;ancestors=set()
    while True:
        for ref in cursor["accepted_source_review_refs"]:
            reviews=read_json(_checked(ref,binding_path.parent))
            if reviews.get("accepted_for_unknown_identity_withdrawal")is not True or reviews.get("model_residual_used_for_qualification")is not False:raise ValueError("Explicit source-only accepted reviews required")
            for review in reviews["reviews"]:
                if review.get("qualification_action")!="withhold_world_identity"or review.get("to_qualification")!="unknown"or review.get("model_residual_used_for_qualification")is not False:raise ValueError("Invalid cumulative source review action")
                reason=review.get("qualification_reason_code","source_navigation_homography_pole_chord")
                if reason not in ("source_navigation_homography_pole_chord","finite_cross_shot_material_instance_unsupported"):raise ValueError("Unsupported source-review reason")
                for o in review["source_store_ordinals"]:
                    o=int(o)
                    if o in expected:raise ValueError("Cumulative source reviews overlap")
                    expected[o]=reason
        parent=cursor.get("parent_qualification_binding")
        if parent is None:break
        if parent["sha256"]in ancestors:raise ValueError("Cyclic source qualification ancestry")
        ancestors.add(parent["sha256"]);cursor=read_json(_checked(parent,binding_path.parent))
    if set(expected)!=withdrawn:raise ValueError("Explicit cumulative review reasons differ from unknown membership")
    uv=_array(doc["arrays"]["uv"],path.parent);pts=_array(doc["arrays"]["native_pts"],path.parent)
    si=_array(doc["arrays"]["shot_index"],path.parent);seen=set();count=0
    changed=[p for p in binding["qualified_source_parts"]if p["reference"]==binding["changed_source_part_reference"]]
    if len(changed)!=1:raise ValueError("One explicit current semantic source part is required")
    part=changed[0];start=int(part["ordinal_start"])
    for number,row in enumerate(checked_records(part["reference"])):
        global_ordinal=start+number;count+=1
        if global_ordinal not in withdrawn:continue
        index=positions[global_ordinal]
        if row["selected_row_id"]!=doc["source_row_ids"][index]or row["uv"]!=list(uv[index])or int(row["native_pts"])!=int(pts[index])or row["shot_id"]!=doc["shot_ids"][int(si[index])]:raise ValueError("Current qualification changed original source geometry or identity")
        if row.get("source_fit_eligible")is not False or row.get("physical_instance_qualification")!="unknown"or row.get("qualification_reason_code")!=expected[global_ordinal]:raise ValueError("Current source row did not install its explicit reviewed unknown reason")
        seen.add(global_ordinal)
    if count!=int(part["row_count"])or seen!=withdrawn:raise ValueError("Current source denominator or explicit reviewed membership differs")
    from collections import Counter
    return dict(schema="rinkfit.current_dense_source_instance_qualification_consumption.v1",passed=True,
        source_qualification_binding=_ref(binding_path),packet=_ref(path),issuer=_ref(__file__),
        full_qualified_source_stream_semantics_consumed=True,current_semantic_rows=count,
        current_unknown_source_rows_checked=len(seen),reason_counts=dict(Counter(expected.values())),
        original_UV_ID_nativePTS_unchanged=True,source_pixels_decoded=0,physical_solver_started=False)
