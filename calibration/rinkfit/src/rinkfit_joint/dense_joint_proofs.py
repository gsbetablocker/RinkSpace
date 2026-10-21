"""Issue the dense6 source and engineering-channel proofs from real packet facts.

No observations are decoded, moved, dropped, or newly qualified here. A passed
full-source gate binds the actual source facts; weights are capture budgets,
not an independently calibrated likelihood or source measurement precision.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import math
import numpy as np
from .io_utils import read_json, file_sha256


def _ref(path):
    path=Path(path).resolve()
    return {"file":str(path),"sha256":file_sha256(path)}


def _checked(ref,base):
    path=Path(ref["file"])
    path=(path if path.is_absolute() else Path(base)/path).resolve()
    if file_sha256(path)!=ref["sha256"]:
        raise ValueError("Changed dense proof dependency: "+str(path))
    return path


def _array(spec,base):
    path=_checked(spec,base)
    dt=np.dtype(spec["dtype"]);shape=tuple(spec["shape"])
    offset=int(spec.get("offset_bytes",0))
    if offset<0 or path.stat().st_size!=offset+math.prod(shape)*dt.itemsize:
        raise ValueError("Dense proof array byte length differs")
    return np.memmap(path,dtype=dt,mode="r",offset=offset,shape=shape)


def _write(path,document):
    path=Path(path).resolve()
    if path.exists():
        raise FileExistsError("Immutable proof already exists: "+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    return _ref(path)


def build_dense_source_qualification(packet,actual_source_gate,inherited_native_qualification,output):
    """Bind one exact passed full-source gate; native rechecks actual columns."""
    path=Path(packet).resolve();raw=json.loads(path.read_text(encoding="utf-8-sig"))
    gate_path=Path(actual_source_gate).resolve();gate_doc=read_json(gate_path)
    gate=gate_doc.get("actual_full_source_guard",gate_doc)
    source=_ref(path)
    if not gate_doc.get("passed") or not gate.get("passed") or gate.get("packet")!=source:
        raise ValueError("A passed full-source gate for this exact packet is required")
    if not gate.get("actual_full_stream_semantics_consumed") or not gate.get("source_UV_ID_native_clock_exact"):
        raise ValueError("Full source UV, ID and native clock consumption was not verified")
    if gate.get("unknown_source_qualification_granted"):
        raise ValueError("Unknown source qualification is not permitted")
    registry=raw["dense_source_binding"];append=raw["dense_joint_source_append"]
    n0=int(registry["row_start"]);nnew=int(registry["count"]);n=int(raw["row_count"])
    if n0+nnew!=n or int(gate["actual_rows"])!=nnew:
        raise ValueError("Dense source gate denominator differs")
    identity=append["actual_column_identity_sha256"]
    if identity!=gate["actual_column_identity_sha256"]:
        raise ValueError("Dense source gate actual-column identity differs")
    binding_path=_checked(append["source_binding"],path.parent);binding=read_json(binding_path)
    consumption_path=binding_path.parent/"actual_consumption.json";consumption=read_json(consumption_path)
    if consumption.get("actual_consumed_eligible_rows")!=nnew or not consumption.get("full_store_stream_semantic_checks_completed"):
        raise ValueError("Bridge did not consume every eligible source row")
    if consumption.get("actual_UV_ID_PTS_material_normalized_sha256")!=binding["eligible_rows_normalized_sha256"]:
        raise ValueError("Bridge actual normalized source identity differs")
    if binding["total_rows"]!=nnew:
        raise ValueError("Bridge source count differs")
    inherited_path=_checked(inherited_native_qualification,path.parent)
    refs=[_ref(binding_path),_ref(consumption_path),_ref(gate_path),_ref(inherited_path)]
    refs += [binding[k] for k in ("source_storage_bindings","material_map","eligibility_index","admission","reader")]
    refs += binding.get("policy_refs",[])
    refs += [_ref(Path(__file__).with_name(name)) for name in ("dense_joint_packet.py","io_utils.py","packet_storage.py","exact_dependency_aliases.py")]
    if raw.get("exact_dependency_aliases"):
        refs.append(raw["exact_dependency_aliases"])
    unique={}
    for ref in refs:
        actual=_checked(ref,path.parent)
        unique[(str(actual).casefold(),ref["sha256"])]=dict(ref)
    membership={k:v["sha256"] for k,v in raw["arrays"].items()}
    result=dict(schema="rinkfit.dense6_native_source_qualification.v1",source_packet=source,
        parent_packet=raw["dense_parent_packet"],inherited_native_qualification=inherited_native_qualification,
        issuer=_ref(__file__),dense_binding=append["source_binding"],actual_source_consumption=_ref(consumption_path),
        parent_rows=n0,raw_rows=n,new_rows=nnew,
        source_video_sha256="e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d",
        time_base=[1,12800],coordinate_domain="native_1920x1080_pixel_centres",
        sourceUV_from_model=False,qualification_granted_for_unknown=False,
        actual_column_identity_sha256=identity,membership_array_sha256=membership,
        compact_registry=registry,actual_input_references=list(unique.values()),
        actual_full_source_gate=_ref(gate_path),all_unselected_rows_preserved_in_columnar_eligibility=True,
        source_accuracy_accepted=False,numerical_auxiliary_is_source_truth=False,
        source_RS_roots_or_optical_domain_certified=False,physical_solver_started=False)
    return _write(output,result)


def build_dense_engineering_weights(packet,parent_weights,model,dense_source_qualification,output_dir):
    """Preserve old budget bytes and split rig exposure rates over new families."""
    path=Path(packet).resolve();doc=read_json(path);raw=json.loads(path.read_text(encoding="utf-8-sig"))
    registry=raw["dense_source_binding"];n0=int(registry["row_start"]);n=int(doc["row_count"])
    nnew=int(registry["count"])
    pw_path=Path(parent_weights).resolve();pw=read_json(pw_path)
    if pw["source_packet"]!=raw["dense_parent_packet"] or int(pw["raw_rows"])!=n0:
        raise ValueError("Parent weights do not bind this exact immutable packet prefix")
    qualification_path=Path(dense_source_qualification).resolve();qualification=read_json(qualification_path)
    if qualification["source_packet"]!=_ref(path):
        raise ValueError("Dense qualification binds a different packet")
    model_path=Path(model).resolve();model_doc=read_json(model_path)
    shot_ids=doc["shot_ids"];rig_of={sid:model_doc["shots"][sid]["rig_id"] for sid in shot_ids}
    cols={k:_array(v,path.parent) for k,v in doc["arrays"].items()}
    original={k:_array(pw["files"][k],pw_path.parent) for k in ("weights.f64","active.u8","information_class.u8")}
    parent_active=np.asarray(original["active.u8"])!=0;parent_w=np.asarray(original["weights.f64"])
    if len(parent_w)!=n0 or not np.isfinite(parent_w).all() or (parent_w<0).any():
        raise ValueError("Invalid parent engineering weights")
    if np.any(parent_w[~parent_active]!=0) or np.any(parent_w[parent_active]<=0):
        raise ValueError("Parent active engineering budget does not match active mask")
    by_rig=defaultdict(lambda:dict(rows=0,budget=0.,exposures=set(),shots=set()))
    parent_exposures=set()
    for si,sid in enumerate(shot_ids):
        ii=np.flatnonzero((np.asarray(cols["shot_index"][:n0])==si)&parent_active)
        if not len(ii):continue
        pts=set(map(int,cols["native_pts"][ii]));rig=rig_of[sid];r=by_rig[rig]
        r["rows"]+=len(ii);r["budget"]+=float(np.sum(parent_w[ii],dtype=np.float64));r["shots"].add(sid)
        r["exposures"].update((sid,p) for p in pts);parent_exposures.update((sid,p) for p in pts)
    rates={rig:r["budget"]/len(r["exposures"]) for rig,r in by_rig.items() if r["exposures"]}
    engineering=raw["dense_parent_append"]["engineering_pixel_dependency_groups.groups"]
    family_index=_array(registry["source_family_index"],path.parent)
    families=registry["source_family_ids"]
    used=np.zeros(nnew,dtype=np.uint8);contexts=defaultdict(list);group_rows=[]
    for group in engineering:
        ii=np.asarray(group["rows"],dtype=np.int64)
        if not len(ii) or np.any(ii<n0) or np.any(ii>=n):
            raise ValueError("Engineering dense component has an invalid member")
        local=ii-n0
        if np.any(used[local]):raise ValueError("Dense engineering components overlap")
        used[local]=1
        shots=np.unique(cols["shot_index"][ii]);pts=np.unique(cols["native_pts"][ii]);ff=np.unique(family_index[local])
        if len(shots)!=1 or len(pts)!=1 or len(ff)!=1:
            raise ValueError("Engineering component crosses native exposure or named source material")
        si=int(shots[0]);sid=shot_ids[si];t=int(pts[0]);family=families[int(ff[0])]
        if group["id"]!=family+":native_exposure:"+str(t):
            raise ValueError("Engineering component does not bind actual source family and native exposure")
        if rig_of[sid] not in rates:
            raise ValueError("New rig has no retained parent exposure budget baseline")
        context=(sid,t)
        if any(g["family"]==family for g in contexts[context]):
            raise ValueError("Named family split into duplicate engineering groups")
        item=dict(id=group["id"],indices=ii,family=family,context=context)
        contexts[context].append(item);group_rows.append(item)
    if not np.all(used):raise ValueError("Dense engineering components do not partition every suffix row")
    if np.any(cols["kind"][n0:]!=0):raise ValueError("Dense engineering suffix contains a fixed point")
    if any(int(q["row_index"])>=n0 for q in doc["queries"] if q is not None):
        raise ValueError("Dense engineering suffix has an unqualified Query")
    stat_used=np.zeros(nnew,dtype=np.uint8)
    for g in raw["dense_parent_append"]["groups"]:
        ii=np.asarray(g["rows"],dtype=np.int64)
        if g.get("rank")!=0 or g.get("W")!=[]:
            raise ValueError("Unknown dense source W or rank was forged")
        if np.any(ii<n0) or np.any(ii>=n) or np.any(stat_used[ii-n0]):
            raise ValueError("Dense unknown-W component membership invalid")
        stat_used[ii-n0]=1
    if not np.all(stat_used):raise ValueError("Unknown-W components do not cover dense suffix")
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ("weights.f64","active.u8","information_class.u8","manifest.json"):
        if (out/name).exists():raise FileExistsError("Immutable engineering proof output already exists")
    weights=np.memmap(out/"weights.f64",dtype="<f8",mode="w+",shape=(n,))
    active=np.memmap(out/"active.u8",dtype="u1",mode="w+",shape=(n,))
    classes=np.memmap(out/"information_class.u8",dtype="u1",mode="w+",shape=(n,))
    weights[:n0]=original["weights.f64"];active[:n0]=original["active.u8"];classes[:n0]=original["information_class.u8"]
    active[n0:]=1;classes[n0:]=5
    budgets=[];shots_summary=defaultdict(lambda:dict(rows=0,exposures=set(),groups=0,budget=0.,overlap=set()))
    for item in group_rows:
        sid,t=item["context"];ii=item["indices"];rig=rig_of[sid]
        family_count=len(contexts[(sid,t)]);budget=rates[rig]/family_count;per_row=budget/len(ii)
        if not math.isfinite(per_row) or per_row<=0:raise ValueError("Invalid dense engineering per-row budget")
        weights[ii]=per_row
        actual=float(len(ii)*per_row)
        budgets.append(dict(id=item["id"],rows=len(ii),per_row_weight=per_row,engineering_budget=actual))
        s=shots_summary[sid];s["rows"]+=len(ii);s["exposures"].add(t);s["groups"]+=1;s["budget"]+=actual
        if (sid,t) in parent_exposures:s["overlap"].add(t)
    for a in (weights,active,classes):a.flush()
    specs={}
    for key,dtype,a in (("weights.f64","<f8",weights),("active.u8","|u1",active),("information_class.u8","|u1",classes)):
        b=np.asarray(a[:n0]).tobytes();expected=Path(pw["files"][key]["file"]).read_bytes()
        if b!=expected:raise ValueError("Parent engineering bytes changed: "+key)
        specs[key]=dict(_ref(out/key),dtype=dtype,shape=[n])
    per_rig=[dict(rig_id=rig,parent_shots=sorted(r["shots"]),parent_active_rows=r["rows"],
        parent_active_exposures=len(r["exposures"]),retained_parent_budget=r["budget"],
        engineering_budget_per_exposure=rates[rig]) for rig,r in sorted(by_rig.items())]
    per_shot=[]
    for sid,s in sorted(shots_summary.items()):
        expected=rates[rig_of[sid]]*len(s["exposures"])
        if not math.isclose(s["budget"],expected,rel_tol=2e-12,abs_tol=1e-8):
            raise ValueError("Dense engineering budget changed with sampling density")
        per_shot.append(dict(shot_id=sid,rig_id=rig_of[sid],new_rows=s["rows"],
            covered_native_exposures=len(s["exposures"]),named_material_exposure_components=s["groups"],
            parent_correlated_native_exposures=len(s["overlap"]),
            engineering_budget_per_exposure=rates[rig_of[sid]],engineering_budget=expected,
            parent_overlap_is_auxiliary_capture_channel_not_independent_information=bool(s["overlap"])))
    retained=float(np.sum(weights[:n0],dtype=np.float64));newbudget=float(np.sum(weights[n0:],dtype=np.float64))
    result=dict(schema="rinkfit.dense6_engineering_source_weights.v1",source_packet=_ref(path),
        parent_packet=raw["dense_parent_packet"],parent_weights=_ref(pw_path),issuer=_ref(__file__),
        dense_source_qualification=_ref(qualification_path),parent_rows=n0,raw_rows=n,
        active_rows=int(np.sum(active!=0)),new_active_rows=nnew,files=specs,component_budgets=budgets,
        actual_total_engineering_budget=retained+newbudget,retained_parent_budget=retained,new_dense_budget=newbudget,
        source_noise_W_or_sigma_issued=False,statistical_precision_or_noise_independence_claimed=False,
        engineering_policy="retained_parent_rig_native_exposure_rate_shared_across_visible_named_material_families",
        rate_model=_ref(model_path),per_rig_exposure_rates=per_rig,per_shot_new_budgets=per_shot,
        parent_budget_weight_active_information_class_prefix_bytes_exact=True,
        new_information_class=5,new_source_W_rank=0,
        original_2D_glyph_and_corner_budget_diluted=False,
        source_pixel_density_increases_exposure_budget=False,
        repeated_source_readings_are_not_independent_noise=True,
        physical_solver_started=False,quality_accepted=False)
    return _write(out/"manifest.json",result)
