"""One current S01 preparation generation, with stable semantic crosswalk.

Preparation consumes self-contained current source observations and the current
new model. Normal loading opens only this generation and current source arrays;
it never reopens a historical packet, source audit, or optimizer implementation.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
from scipy.sparse import csr_matrix,save_npz,load_npz
from . import schema as S
from .arrays import ArrayStore
from .observations import load_import,load_current_support
from .factors import GroupWhitening,WhiteningPlan,RationalProfile

VERSION="rinksolve.prepared_s01.v1"
# Raw evidence stays immutable. Every excluded association has explicit bits.
REASONS={1:"current_not_admitted",2:"current_not_fit_role",4:"current_model_retired",8:"current_withheld",
         16:"no_exact_current_geometry_crosswalk",32:"corner_semantics_not_fixed_entity",
         64:"selected_path_outside_current_allowed_paths",128:"query_descriptor_missing",256:"intersection_support_replacement",
         1024:"unverified_current_source_entity_association",2048:"unresolved_current_finite_glyph_boundary",4096:"Ad5_specific_source_association_requires_finite_branch_review"}


def _text(value):
    return value.decode("utf-8") if isinstance(value,(bytes,np.bytes_)) else str(value)


def _json(path,value):
    p=Path(path);tmp=p.with_name(p.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,p)


def _sha_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _actual_array(descriptor,data_root):
    p=Path(descriptor.get("path",descriptor.get("file",""))).resolve();p.relative_to(Path(data_root).resolve())
    dt=np.dtype(descriptor["dtype"]);shape=tuple(descriptor["shape"]);offset=int(descriptor.get("offset_bytes",0))
    if dt.hasobject or dt.kind not in "biuf" or offset<0 or len(shape)!=2:
        raise ValueError("invalid current W descriptor")
    if offset+int(np.prod(shape))*dt.itemsize>p.stat().st_size:
        raise ValueError("current W exceeds delivered file")
    return np.memmap(p,mode="r",dtype=dt,offset=offset,shape=shape)


def _columns(delivery,data_root):
    store=ArrayStore(data_root);refs={a.id:a for a in delivery.bundle.array_refs}
    return {t.id:{c.name:store.open(refs[c.array_id]) for c in t.columns} for t in delivery.bundle.column_tables}


def reason_names(mask):
    return [name for bit,name in REASONS.items() if int(mask)&bit]


def _curve_key(curve):
    d=curve.definition
    return (curve.label,str(d.get("original_path_id",d.get("source_path_id",""))))


def _current_path(material,path_index):
    return next((p for p in material.get("paths",[]) if int(p["index"])==int(path_index)),None)


@dataclass(frozen=True)
class PreparedS01:
    directory: Path
    manifest: dict
    arrays: dict
    whitening: csr_matrix

    @property
    def row_count(self):
        return int(self.manifest["row_count"])

    @property
    def profiles(self):
        return tuple(RationalProfile(**v) for v in self.manifest["profiles"])

    def assert_current_constraint_snapshot(self,ledger_path=None):
        """Check only the current prepared target, without source requalification."""
        binding=self.manifest.get("constraints_binding")
        if binding is None:
            return {"bound":False,"reason":"this preparation generation predates an explicit relevant-target snapshot"}
        path=Path(ledger_path or binding["path"])
        ledger=json.loads(path.read_text(encoding="utf-8-sig"))
        entries={v["id"]:v for v in ledger["constraints"]}
        for prior in binding["relevant_constraints"]:
            current=entries.get(prior["id"])
            if current is None:
                raise ValueError("prepared current engineering constraint was removed: "+prior["id"])
            relevant={"id":current["id"],"enabled":current["enabled"],"expression":current["expression"]}
            digest=hashlib.sha256(json.dumps(relevant,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
            expected=binding.get("relevant_sha256_by_id",{}).get(prior["id"],binding.get("relevant_sha256"))
            if digest!=expected:
                raise ValueError("current engineering target/support policy changed: "+prior["id"]+"; update only this affected prepared family")
        return {"bound":True,"current_revision":ledger["revision"],"prepared_revision":binding["revision"],
            "unrelated_revision_change":ledger["revision"]!=binding["revision"],"source_requalification_required":False}

    def observation_ids(self,rows=None):
        """Exact original stable observation IDs from the current readonly SoA.

        Lazy current bundle/array access is ordinary input consumption, not a
        historical qualification replay. No old packet or audit is opened.
        """
        column=getattr(self,"_observation_id_column",None)
        if column is None:
            source=load_import(self.manifest["source_observation_manifest"])
            table=next(t for t in source.bundle.column_tables if t.id=="s01.raw")
            aid=next(c.array_id for c in table.columns if c.name=="observation_id")
            ref=next(a for a in source.bundle.array_refs if a.id==aid)
            column=ArrayStore(self.manifest["data_root"]).open(ref)
            if len(column)!=self.row_count:
                raise ValueError("current stable observation ID count differs")
            object.__setattr__(self,"_observation_id_column",column)
        result=column if rows is None else column[rows]
        if isinstance(result,np.ndarray):
            result.setflags(write=False)
        return result

    def selection_current_active_rows(self):
        """Union of exact current statistical and engineering-budget support."""
        return np.union1d(self.whitening.indices//2,self.arrays["budget_ready_rows"])

    def nuisance_groups(self):
        """Disjoint local owners for exact full-W Schur grouping.

        One original W group owns all its row auxiliaries together. Shared
        physical corners/landmarks are not local stations and remain global.
        """
        claimed=np.zeros(self.row_count,dtype=bool)
        for i,g in enumerate(self.manifest["groups"]):
            if not g["active_statistical"]:
                continue
            lo,hi=self.arrays["group_member_offsets"][i:i+2]
            rows=self.arrays["group_member_rows"][lo:hi]
            if np.any(claimed[rows]):
                raise ValueError("current W local auxiliary groups overlap")
            claimed[rows]=True
            yield {"id":"nuisance/"+g["id"],"source_group_index":i,"rows":rows,"kind":"complete_W_source_group",
                "owner":"observation_auxiliary","contains_shared_landmark_positions":False}
        for row in self.selection_current_active_rows():
            if not claimed[row]:
                yield {"id":"nuisance/"+_text(self.observation_ids([row])[0]),"source_group_index":None,
                    "rows":np.asarray([row],dtype=np.int64),"kind":"individual_current_auxiliary",
                    "owner":"observation_auxiliary","contains_shared_landmark_positions":False}

    def source_group(self,index):
        """Return complete original W, including explicitly inactive groups."""
        g=self.manifest["groups"][index];a=self.arrays
        lo,hi=a["group_member_offsets"][index:index+2];wo,we=a["group_weight_offsets"][index:index+2]
        rows=a["group_member_rows"][lo:hi];w=a["group_weight_values"][wo:we].reshape(g["rank"],2*len(rows))
        return GroupWhitening(g["id"],rows,w,g["statistic_kind"],g.get("source_group_id"))

    @property
    def curve_ids(self):
        return tuple(v["id"] for v in self.manifest["curve_registry"])

    def bound_curve_ids(self, rows=None):
        rows=np.arange(self.row_count) if rows is None else np.asarray(rows,dtype=np.int64)
        indices=self.arrays["curve_index"][rows]
        if np.any(indices<0):
            raise ValueError("selected observation has an explicitly unresolved current curve")
        return np.asarray(self.curve_ids)[indices]

    def statistical_residual(self,native_error):
        error=np.asarray(native_error,dtype=np.float64)
        if error.shape!=(self.row_count,2):
            raise ValueError("prepared native error shape differs")
        used=np.unique(self.whitening.indices//2)
        if not np.isfinite(error[used]).all():
            raise ValueError("active source factor is nonfinite; do not drop it during a solve")
        clean=np.zeros_like(error);clean[used]=error[used]
        return self.whitening@clean.reshape(-1)

    def statistical_jacobian(self,native_jacobian):
        j=np.asarray(native_jacobian,dtype=np.float64)
        if j.ndim!=3 or j.shape[:2]!=(self.row_count,2):
            raise ValueError("prepared native Jacobian shape differs")
        used=np.unique(self.whitening.indices//2)
        if not np.isfinite(j[used]).all():
            raise ValueError("active factor derivative is nonfinite")
        clean=np.zeros_like(j);clean[used]=j[used]
        return self.whitening@clean.reshape(2*self.row_count,-1)



def _current_source_domains(quality,curve_index,kinds,query_index,normal,unknown_rows):
    """Current all-source E and independent Query selections, never W-only."""
    quality=np.asarray(quality);normal=np.asarray(normal,dtype=np.float64);n=len(quality)
    if normal.shape!=(n,2) or not np.isfinite(normal).all():raise ValueError("invalid current original profile normal column")
    candidate=np.flatnonzero((quality&np.uint16(0xFFFF ^ 16))==0).astype(np.int64)
    unknown=np.asarray(sorted(set(map(int,unknown_rows))),dtype=np.int64)
    unknown=unknown[(quality[unknown]&np.uint16(0xFFFF ^ 16))==0]
    query=np.flatnonzero((np.asarray(kinds)==2)&(quality==0)).astype(np.int64)
    if np.any(query_index[query]<0) or np.any(np.abs(np.linalg.norm(normal[query],axis=1)-1.)>1e-8):
        raise ValueError("current genuine profile lacks original Query or source normal")
    return {"budget_candidate_rows":candidate,"budget_ready_rows":candidate[curve_index[candidate]>=0],
        "budget_pending_binding_rows":candidate[curve_index[candidate]<0],
        "unknown_noise_budget_candidate_rows":unknown,"unknown_noise_budget_ready_rows":unknown[curve_index[unknown]>=0],
        "unknown_noise_budget_pending_binding_rows":unknown[curve_index[unknown]<0],
        "active_query_rows":query,"active_query_indices":query_index[query],"profile_normal":normal}


def _install_current_kernel_contract(meta):
    """One explicit source-statistic contract shared by fresh/current adapters."""
    meta.update(noise_contract_revision=2,statistical_error_definition="predicted_native_uv - immutable_source_raw_uv for ALL semantic kinds",
        statistical_Query_alpha_dependency=False,range_C_hard_equality_installed=False)
    meta["semantic_kind_definitions"]["2"]="genuine profile: raw-pixel W statistic; original source-normal E domain; independent exact predicted_uv-Q(alpha)=0 two-component equality; original finite alpha range"
    meta["source_budget_family"]={"id":"current_all_active_source_engineering_domain","sense":"le","rows":"budget_ready_rows",
        "ordinary_q":"(du*du+dv*dv)/(E*E)","genuine_profile_q":"(source_normal dot native_raw_error)^2/(E*E)","value":"q-budget_sq",
        "source_error":"predicted_uv-immutable_raw_uv","profile_normal_array":"profile_normal","semantic_kind_array":"semantic_kind",
        "E_array":"engineering_budget_px","E_unit":"native_pixel","E_is_sigma":False,
        "budget_rhs":{"id":"numerical/shared_budget_sq","scope":"one global shared scalar for S01","unit":"dimensionless","lower":0.,
            "feasibility_phase":"free; objective minimize budget_sq; initialize max(1,max q_i)","precision_phase":"fixed exactly 1"},
        "statistical_groups_also_have_engineering_domain":True,"original_independent_rows_also_have_engineering_domain":True}
    meta["engineering_constraint_family"]=dict(meta["source_budget_family"])
    meta["query_equality_family"]={"id":"current_genuine_profile_native_Query","sense":"eq","rows":"active_query_rows",
        "value":"predicted_native_uv-Q(alpha)","components":2,"unit":"native_pixel","alpha_bounds":"original profile Query interval",
        "does_not_replace_statistical_pixel_error":True,"does_not_subtract_Query_derivative_from_W_Jacobian":True}
    if meta.get("new_fixed_intersections"):
        meta["new_fixed_intersections"]["engineering_target_shared_budget_rhs"]="numerical/shared_budget_sq; precision phase fixed 1"



def _incremental_ref(name,path,value,generation,prior_refs):
    """Preserve unit/clock/role when an association creates a new array file."""
    from dataclasses import replace
    prior={d["id"].removeprefix("preparation."):S.from_json_value(d,S.TypedArrayRef) for d in prior_refs}
    if name in ("ad7_fixed_native_pts","intersection_native_pts"):
        template=prior["native_pts"];role=S.ArrayRole.IMMUTABLE_EVIDENCE
    elif name in ("ad7_fixed_uv","intersection_uv","immutable_uv"):
        template=prior["immutable_uv"];role=S.ArrayRole.IMMUTABLE_EVIDENCE
    elif name=="source_sigma" or name.endswith("engineering_budget_px"):
        template=prior["engineering_budget_px"];role=S.ArrayRole.DERIVED_EVIDENCE
    elif name.endswith("row_auxiliary_seed"):
        template=prior["row_auxiliary_seed"];role=S.ArrayRole.AUXILIARY_STATE
    elif name.endswith("seed"):
        template=prior["station_seed"];role=S.ArrayRole.AUXILIARY_STATE
    elif name=="profile_normal":
        template=prior.get(name,prior["curve_index"]);role=S.ArrayRole.IMMUTABLE_EVIDENCE
    else:
        template=prior.get(name,prior["curve_index"]);role=template.role
    return replace(template,id="preparation."+name,path=str(path),storage="npy",dtype=value.dtype.str,shape=value.shape,
        generation=int(generation),read_only=True,member=None,sha256=None,role=role)


def prepare_s01(observation_directory,model,d78_directory,output,*,data_root,
                intersection_source=None,intersection_budget_px=2.0,enable_intersections=True,constraint_ledger=None):
    """Prepare actual current values once, preserving current confirmation.

    d78 is only a successful initialization and an explicitly keyed crosswalk
    witness. The current observation associations override changed old labels.
    Their authority is not borrowed from numerical proximity to that initializer.
    """
    begun=time.perf_counter();data_root=Path(data_root).resolve();out=Path(output).resolve();out.relative_to(data_root)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("use a new preparation generation, never overwrite a mapped generation")
    constraint_binding=None
    if constraint_ledger is not None:
        ledger_path=Path(constraint_ledger).resolve();ledger=json.loads(ledger_path.read_text(encoding="utf-8-sig"))
        target=next(v for v in ledger["constraints"] if v["id"]=="numerical.intersection_budget")
        enable_intersections=bool(target["enabled"])
        intersection_budget_px=float(target["expression"]["radius_px"])
        relevant={"id":target["id"],"enabled":target["enabled"],"expression":target["expression"]}
        constraint_binding={"path":str(ledger_path),"revision":int(ledger["revision"]),"relevant_constraints":[relevant],
            "relevant_sha256":hashlib.sha256(json.dumps(relevant,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest(),
            "unrelated_constraint_change_requires_source_requalification":False}
    delivery=load_import(observation_directory);support=load_current_support(delivery);columns=_columns(delivery,data_root)
    raw,ctx=columns["s01.raw"],columns["s01.current_context"];n=len(raw["original_row"]);original=np.asarray(raw["original_row"],dtype=np.int64)
    if n==0 or np.any(original[1:]<=original[:-1]):
        raise ValueError("current original-row lookup must be strictly increasing")
    if not np.array_equal(raw["observation_id"],ctx["observation_id"]):
        raise ValueError("current context must join exact observation stable IDs")
    uv=np.column_stack((raw["u"],raw["v"]))
    if not np.isfinite(uv).all():
        raise ValueError("nonfinite immutable current pixels")
    registry=json.loads((Path(observation_directory)/"source_registry.json").read_text(encoding="utf-8"))
    current_materials={int(m["index"]):m for m in registry["legacy_material_binding"]["materials"]}
    curves=tuple(model.scene.curves);curve_index={c.id:i for i,c in enumerate(curves)}
    keyed=defaultdict(list)
    for i,curve in enumerate(curves):
        key=_curve_key(curve)
        if key[1]:
            keyed[key].append(i)
    result_curve=np.full(n,-1,dtype=np.int32);binding_source=np.zeros(n,dtype=np.uint8)
    quality=np.zeros(n,dtype=np.uint16)
    quality[ctx["legacy_admission"]==0]|=1;quality[ctx["legacy_fit_role"]==0]|=2
    quality[ctx["legacy_model_retired"]!=0]|=4
    withheld=set(map(int,support["current_withheld_rows"]))
    quality[np.isin(original,list(withheld))]|=8
    kinds=np.asarray(ctx["legacy_kind"],dtype=np.uint8)
    if not np.isin(kinds,[0,1,2]).all():
        raise ValueError("unknown current source semantic kind")
    # Current association is keyed by material identity AND exact path identity.
    # Same brand text or same old integer index alone never identifies geometry.
    missing=Counter();ambiguous=Counter()
    for (mi,pi) in set(zip(map(int,ctx["legacy_material_index"]),map(int,ctx["legacy_path"]))):
        selection=(ctx["legacy_material_index"]==mi)&(ctx["legacy_path"]==pi)
        material=current_materials.get(mi)
        if material is None:
            missing[("<retired>",str(mi),pi)]+=int(selection.sum());continue
        path=_current_path(material,pi)
        if path is None:
            missing[(material["label"],"<path missing>",pi)]+=int(selection.sum());continue
        choices=keyed.get((material["label"],str(path["id"])),[])
        if len(choices)==1:
            result_curve[selection]=choices[0];binding_source[selection]=1
        elif len(choices)>1:
            ambiguous[(material["label"],str(path["id"]),pi)]+=int(selection.sum())
        else:
            missing[(material["label"],str(path["id"]),pi)]+=int(selection.sum())
    # Targeted equality check enables correct inheritance of the current prefix,
    # including resolved finite-branch initialization. This is not replayed by load.
    d78=Path(d78_directory);semantic=json.loads((d78/"reference_semantic.json").read_text(encoding="utf-8"))
    with np.load(d78/"forward_comparison.npz",allow_pickle=False) as z:
        seed={k:np.array(z[k],copy=True) for k in ("source_rows","native_pts","material_index","branches","stations","new_row","frozen_profile_alpha")}
    prefix_count=len(seed["source_rows"]);positions=np.searchsorted(original,seed["source_rows"])
    if np.any(positions>=n) or not np.array_equal(original[positions],seed["source_rows"]):
        raise ValueError("current delivery no longer contains the exact d78 keyed prefix")
    if not np.array_equal(raw["native_pts"][positions],seed["native_pts"]):
        raise ValueError("current prefix native exposure identity differs from initialization")
    for key,a in (("source_point_ids",np.asarray([_text(s) for s in raw["source_id"][positions]],dtype=semantic["immutable_source_arrays"]["source_point_ids"]["dtype_str"])),
                  ("source_uv",uv[positions].astype("<f8",copy=False)),("native_pts",raw["native_pts"][positions].astype("<i8",copy=False))):
        if _sha_bytes(a)!=semantic["immutable_source_arrays"][key]["sha256_c_contiguous_array_bytes"]:
            raise ValueError("current prefix exact original "+key+" differs; do not align by count")
    d78_paths=json.loads((d78/"paths.json").read_text(encoding="utf-8"));old_path={(int(p["material_index"]),int(p["branch_index"])):p for p in d78_paths}
    station=np.asarray(ctx["legacy_station"],dtype=np.float64).copy();row_seed=np.asarray(raw["v"],dtype=np.float64).copy()
    alpha_seed=np.asarray(ctx["legacy_alpha"],dtype=np.float64).copy();changed=Counter();resolved_seeds=0
    for at,mi,bi,s,r,a in zip(positions,seed["material_index"],seed["branches"],seed["stations"],seed["new_row"],seed["frozen_profile_alpha"]):
        prior=old_path.get((int(mi),int(bi)))
        if prior is None:
            raise ValueError("successful initialization references absent current d78 branch")
        target=result_curve[at]
        if target>=0 and curves[target].id==prior["id"]:
            station[at]=s;row_seed[at]=r;alpha_seed[at]=a;resolved_seeds+=1
        elif target>=0:
            changed[(prior["label"],curves[target].label)]+=1
        # A changed current association keeps its own current station seed; it
        # does not acquire the older material geometry from the d78 initializer.
    row_seed_status=np.zeros(n,dtype=np.uint8)
    row_seed_status[positions]=1
    fallback=~np.isfinite(row_seed) | (row_seed<0) | (row_seed>1079)
    row_seed[fallback]=raw["v"][fallback]
    row_seed_status[fallback]=2
    if not np.isfinite(station).all() or np.any(station<0) or np.any(station>1):
        raise ValueError("current finite curve seed outside [0,1]")
    for row,allowed in support.get("allowed_paths",{}).items():
        i=int(np.searchsorted(original,int(row)))
        if i<n and original[i]==int(row) and int(ctx["legacy_path"][i]) not in allowed:
            quality[i]|=64
    quality[result_curve<0]|=16
    # Only explicitly delivered entity-corner semantics can make a landmark.
    # Sampling vertices, coverage caps and curve endpoints remain curve loci.
    corner_bad=[]
    for i in np.flatnonzero((kinds==1)&(result_curve>=0)):
        curve=curves[result_curve[i]];d=curve.definition
        one_point=(curve.kind=="board_material_polyline" and len(d.get("unwrapped_station_height",[]))==1) or (curve.kind=="world_polyline" and len(d.get("points_xyz",[]))==1)
        if not one_point:
            quality[i]|=32;corner_bad.append(int(original[i]))
        else:
            station[i]=0.
    profiles=[];qindex=np.full(n,-1,dtype=np.int32)
    for q in support["queries"]:
        i=int(np.searchsorted(original,int(q["row_index"])))
        if i>=n or original[i]!=int(q["row_index"]) or kinds[i]!=2:
            raise ValueError("current Query does not reference its original true_profile row")
        profile=RationalProfile.from_current(q,i,ctx["legacy_alpha_bounds"][i])
        nodes=np.asarray(q["alpha"],dtype=np.float64);target=np.asarray(q["native_uv"],dtype=np.float64)
        if not np.array_equal(profile.evaluate(nodes),target):
            if not np.allclose(profile.evaluate(nodes),target,rtol=0,atol=1e-12):
                raise ValueError("current rational Query and delivered native samples disagree")
        qindex[i]=len(profiles);profiles.append(profile)
        # Older optimum is a seed only; source alpha range is unchanged.
        if not profile.alpha_bounds[0]<=alpha_seed[i]<=profile.alpha_bounds[1]:
            raise ValueError("current profile seed outside original source range")
    quality[(kinds==2)&(qindex<0)]|=128
    native_evidence=next(e for e in delivery.bundle.evidence if e.id=="source.video")
    source_id=delivery.bundle.frames[0].id.rsplit(":",1)[0]
    clock_id=delivery.bundle.frames[0].clock_id
    intersections=_prepare_intersections(model,data_root,intersection_source,intersection_budget_px,native_evidence.sha256,source_id) if enable_intersections else None
    replacement_rows=np.zeros(n,dtype=bool)
    if intersections is not None:
        if intersections["metadata"]["source_video_sha256"] != native_evidence.sha256:
            raise ValueError("new intersection source differs from current official native video")
        source_materials={int(v["index"]):v for v in registry["legacy_source_material_binding"]["materials"]}
        is_ice=np.asarray([(current_materials.get(int(mi)) or source_materials.get(int(smi)) or {}).get("label","").startswith("ice.")
            for mi,smi in zip(ctx["legacy_material_index"],ctx["legacy_source_material_index"])],dtype=bool)
        replacement_rows=is_ice & np.isin(raw["native_pts"],intersections["native_pts"])
        quality[replacement_rows]|=256
    eligible=quality==0;group_meta=[];weight_values=[];group_rows=[];row_offsets=[0];weight_offsets=[0]
    active_groups=[];stat_coverage=np.zeros(n,dtype=np.uint8);unknown_rows=[]
    for g in support["groups"]:
        rows=np.asarray(g["rows"],dtype=np.int64);ix=np.searchsorted(original,rows)
        if np.any(ix>=n) or not np.array_equal(original[ix],rows) or len(np.unique(ix))!=len(ix):
            raise ValueError("current source group incomplete or repeated in S01")
        if np.any(stat_coverage[ix]):
            raise ValueError("current support assigns a source to multiple statistical groups")
        stat_coverage[ix]=1
        w=_actual_array(g["W"],data_root) if isinstance(g["W"],dict) else np.asarray(g["W"],dtype=np.float64).reshape(int(g["rank"]),2*len(ix))
        group=GroupWhitening(g["id"],ix,w,g["statistic_kind"],g.get("source_group_id"))
        weights=group.matrix.reshape(-1);weight_values.extend(weights.tolist());weight_offsets.append(weight_offsets[-1]+len(weights))
        group_rows.extend(ix.tolist());row_offsets.append(len(group_rows))
        union=int(np.bitwise_or.reduce(quality[ix],initial=0));active=not union and group.rank>0
        if active:
            active_groups.append(group)
        if group.rank==0:
            # No statistical model exists. Keep every current constraint source,
            # with row-wise binding readiness, rather than discarding its group.
            unknown_rows.extend(ix[(quality[ix] & np.uint16(0xFFFF ^ 16))==0].tolist())
        group_meta.append({"id":g["id"],"source_group_id":g.get("source_group_id"),"statistic_kind":g["statistic_kind"],
            "rank":group.rank,"member_count":len(ix),"active_statistical":active,"exclusion_reason_union":reason_names(union),
            "unknown_noise_budget_group":group.rank==0,"affected_member_original_rows":rows[quality[ix]!=0].tolist(),
            "current_row_indices_are_addresses_not_identities":True})
    independent=np.asarray(support["independent_rows"],dtype=np.int64);ii=np.searchsorted(original,independent)
    if np.any(ii>=n) or not np.array_equal(original[ii],independent) or len(np.unique(ii))!=len(ii) or np.any(stat_coverage[ii]):
        raise ValueError("independent source support duplicates or misses actual current source")
    stat_coverage[ii]=1
    sigma=np.asarray(ctx["legacy_source_sigma"],dtype=np.float64)
    if not np.isfinite(sigma[ii]).all() or np.any(sigma[ii]<=0):
        raise ValueError("original independent scalar sigma invalid")
    independent_active=ii[eligible[ii]];plan=WhiteningPlan(n,active_groups,independent_active,sigma[independent_active],
        independent_ids=["independent/"+_text(x) for x in raw["observation_id"][independent_active]])
    domains=_current_source_domains(quality,result_curve,kinds,qindex,ctx["legacy_profile_normal"],unknown_rows)
    budget_candidates=domains["budget_candidate_rows"];budget_ready=domains["budget_ready_rows"];budget_pending=domains["budget_pending_binding_rows"]
    budget=np.asarray(ctx["legacy_E"],dtype=np.float64)
    if not np.isfinite(budget[budget_candidates]).all() or np.any(budget[budget_candidates]<=0):
        raise ValueError("current engineering pixel budgets invalid")
    arrays={"original_row":original,"native_pts":np.asarray(raw["native_pts"]),"immutable_uv":uv,
        "semantic_kind":kinds,"curve_index":result_curve,"binding_source":binding_source,"exclusion_bits":quality,
        "station_seed":station,"row_auxiliary_seed":row_seed,"row_auxiliary_seed_source":row_seed_status,"profile_alpha_seed":alpha_seed,"query_index":qindex,
        "source_sigma":sigma,"engineering_budget_px":budget,"independent_source_rows":ii,"active_independent_rows":independent_active,
        "group_member_rows":np.asarray(group_rows,dtype=np.int64),"group_member_offsets":np.asarray(row_offsets,dtype=np.int64),
        "group_weight_values":np.asarray(weight_values,dtype=np.float64),"group_weight_offsets":np.asarray(weight_offsets,dtype=np.int64),
        "budget_candidate_rows":budget_candidates,"budget_ready_rows":budget_ready,"budget_pending_binding_rows":budget_pending,
        "uncovered_statistical_rows":np.flatnonzero(stat_coverage==0).astype(np.int64)}
    arrays.update(domains)
    if intersections is not None:
        arrays.update({"intersection_"+k:v for k,v in intersections["arrays"].items()})
    refs=[]
    for name,value in arrays.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        unit=S.Unit.PIXEL if name in ("immutable_uv","row_auxiliary_seed","source_sigma","engineering_budget_px","intersection_uv","intersection_row_auxiliary_seed","intersection_engineering_budget_px") else S.Unit.NATIVE_PTS if name in ("native_pts","intersection_native_pts") else S.Unit.DIMENSIONLESS
        ref=S.TypedArrayRef("preparation."+name,str(path),"npy",value.dtype.str,value.shape,S.UnitRef(unit,clock_id if unit==S.Unit.NATIVE_PTS else source_id if unit==S.Unit.PIXEL else None),
            S.ArrayRole.IMMUTABLE_EVIDENCE if name in ("immutable_uv","original_row","native_pts","intersection_uv","intersection_native_pts") else S.ArrayRole.AUXILIARY_STATE if name.endswith("seed") else S.ArrayRole.DERIVED_EVIDENCE if name in ("source_sigma","engineering_budget_px","group_weight_values","intersection_engineering_budget_px") else S.ArrayRole.SEMANTIC_FOREIGN_KEYS,True,1,
            clock_id=clock_id if unit==S.Unit.NATIVE_PTS else None)
        refs.append(S.to_json_value(ref))
    save_npz(out/"statistical_whitening.npz",plan.operator,compressed=False)
    missing_records=[{"current_label":k[0],"current_path_id":k[1],"path_index":k[2],"observation_count":v} for k,v in missing.most_common()]
    report={"schema":VERSION,"row_count":n,"status":"current_s01_prepared_with_explicit_pending_bindings",
        "ordinary_load_requires_legacy_packet":False,"source_evidence_modified":False,"inherited_current_confirmation_cleared":False,
        "exact_prefix_source_identity_uv_pts_matched":True,"prefix_rows":prefix_count,"projection_witness_seed_rows_inherited":resolved_seeds,
        "row_seed_reinitialized_from_current_native_v_because_historic_root_outside_sensor":original[fallback].tolist(),
        "current_association_revisions_override_old_initialization":[{"old_label":k[0],"current_label":k[1],"rows":v} for k,v in changed.most_common()],
        "exact_current_geometry_bound_rows":int(np.sum(result_curve>=0)),"pending_exact_geometry_rows":int(np.sum(result_curve<0)),
        "active_original_W_groups":len(active_groups),"original_W_groups":sum(g["rank"]>0 for g in group_meta),
        "active_joint_W_residual_rows":sum(g.rank for g in active_groups),"active_independent_sources":len(independent_active),
        "statistical_residual_rows":plan.residual_count,"statistical_nonzero_coefficients":plan.operator.nnz,
        "unknown_noise_groups":sum(g["rank"]==0 for g in group_meta),"engineering_budget_ready_rows":len(budget_ready),
        "engineering_budget_pending_binding_rows":len(budget_pending),"unknown_noise_is_sigma":False,
        "corner_observation_rows":int(np.sum(kinds==1)),"corner_rows_with_inconsistent_geometry":corner_bad,
        "fixed_corner_feature_count":len(set(int(x) for x in result_curve[(kinds==1)&eligible])),
        "true_profile_queries":len(profiles),"derived_intersections_used_in_statistical_objective":0,"derived_intersections_used_in_objective":0,
        "new_fixed_intersection_budget_constraints":len(intersections["observations"]) if intersections else 0,
        "new_fixed_intersection_shared_feature_stations":len(intersections["landmarks"]) if intersections else 0,
        "intersection_replaced_original_ice_rows":int(np.sum(replacement_rows)),
        "intersection_replaced_native_pts":list(map(int,np.unique(raw["native_pts"][replacement_rows]))),
        "derived_intersection_reason":"新交点的两曝光旧ice支持被完整组替代；仅5共享C线上固定实体/8工程预算，不把共用C父样本或审阅估计伪成独立噪声。",
        "exclusion_counts":{name:int(np.sum((quality&bit)!=0)) for bit,name in REASONS.items()},
        "no_source_statistical_support_rows":int(np.sum(stat_coverage==0)),"preparation_seconds":time.perf_counter()-begun}
    manifest={"schema":VERSION,"generation":1,"row_count":n,"shot_id":"S01","next_shot_allowed":False,
        "source_observation_manifest":str(delivery.manifest_path),"data_root":str(data_root),"constraints_binding":constraint_binding,"source_identity_column":"s01.raw.observation_id",
        "model_id":model.id,"model_coordinate_system_id":model.scene.coordinate_system_id,
        "prepared_against_model_revision":model.revision,"parameter_initialization_is_fixed_physical_fact":False,
        "curve_registry":[{"id":c.id,"label":c.label,"entity_id":c.entity_id,"kind":c.kind,"original_path_id":_curve_key(c)[1]} for c in curves],
        "array_refs":refs,"whitening_path":str(out/"statistical_whitening.npz"),"groups":group_meta,
        "statistical_factors":[{"id":f.id,"rows":f.rows.tolist(),"residual_start":f.residual_start,"residual_count":f.residual_count,"kind":f.kind} for f in plan.factors],
        "profiles":[{k:getattr(p,k) for k in p.__dataclass_fields__} for p in profiles],
        "semantic_kind_definitions":{"0":"finite CurveLocus: one observation-local station in exact allowed branch, no endpoint landmark assumption",
            "1":"current confirmed entity corner: one shared physical world position, no per-frame station",
            "2":"genuine profile with raw-UV statistics, separate exact Query equality, and original normal E domain"},
        "reason_bits":{str(k):v for k,v in REASONS.items()},"binding_source_codes":{"0":"unresolved","1":"exact current material label AND original path identity"},
        "landmark_features":[{"id":"landmark/"+curves[i].id,"curve_id":curves[i].id,"position_kind":"current_confirmed_fixed_entity_corner","station_initialization":0.,"station_is_free_per_observation":False} for i in sorted(set(result_curve[(kinds==1)&eligible].tolist()))],
        "engineering_pixel_dependency_groups":support["engineering_pixel_dependency_groups"],
        "missing_exact_geometry":missing_records,"ambiguous_exact_geometry":[{"label":k[0],"path_id":k[1],"rows":v} for k,v in ambiguous.items()],
        "new_fixed_intersections":intersections["metadata"] if intersections else {"active":False},
        "new_intersection_landmarks":intersections["landmarks"] if intersections else [],
        "new_intersection_observations":intersections["observations"] if intersections else [],
        "report":report}
    _install_current_kernel_contract(manifest)
    report.update(noise_contract_revision=2,unknown_noise_budget_ready_rows=len(domains["unknown_noise_budget_ready_rows"]),
        active_original_profile_observations=len(domains["active_query_rows"]),active_original_profile_Query_equalities=2*len(domains["active_query_rows"]),
        statistical_Query_alpha_dependency=False,ordinary_statistical_sources_have_E_domain=True,global_budget_sq_is_single_shared_scalar=True,normal_run_requires_historical_code=False)
    _json(out/"report.json",report);_json(out/"missing_geometry.json",missing_records);_json(out/"manifest.json",manifest)
    return load_prepared(out,data_root=data_root)



def _prepare_intersections(model,data_root,source_path,budget_px,expected_video_sha256,source_id):
    """Shared fixed points on confirmed C, without guessing gray global IDs.

    The explicit engineering 2 px target is not a covariance or a calibrated
    annotation error bound. Source precision estimates and shared parents remain
    visible. Two entire exposure ice supports are replaced by these constraints.
    """
    from scipy.optimize import minimize_scalar
    source=Path(source_path) if source_path else Path(data_root)/"runs/solve2_review/s01_intersections.json"
    review=json.loads(source.read_text(encoding="utf-8"))
    if not np.isfinite(budget_px) or budget_px<=0:
        raise ValueError("explicit positive intersection engineering target required")
    candidates=[c for c in model.scene.curves if c.label=="ice.C:segment0"]
    if len(candidates)!=1:
        raise ValueError("new fixed intersections require one explicitly confirmed current C material centerline")
    curve=candidates[0];operator=model.operator();features=review["fixed_features"];observations=review["observations"]
    if review["source"]["sha256"] != expected_video_sha256:
        raise ValueError("intersection observation source is not this S01 native video")
    feature_ids=[v["id"] for v in features];feature_index={v:i for i,v in enumerate(feature_ids)}
    if len(feature_index)!=len(features):
        raise ValueError("duplicate source fixed material intersection")
    pts=np.asarray([v["native_pts"] for v in observations],dtype=np.int64)
    uv=np.asarray([v["uv_native_pixels"] for v in observations],dtype=np.float64)
    shot=operator.shot
    if np.any(pts<shot.first_native_pts) or np.any(pts>shot.last_native_pts) or np.any((pts-shot.first_native_pts)%shot.step_native_pts) or not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv[:,0]>operator.rig.image_width-1) or np.any(uv[:,1]>operator.rig.image_height-1):
        raise ValueError("new intersection pixels or native exposure identities invalid")
    if any(v["source_frame_id"]!=source_id+":"+str(int(v["native_pts"])) for v in observations):
        raise ValueError("intersection source frame disagrees with its native PTS")
    indices=np.asarray([feature_index[v["fixed_feature_id"]] for v in observations],dtype=np.int32)
    station=[];landmarks=[];seed_errors=[]
    for i,feature in enumerate(features):
        chosen=np.flatnonzero(indices==i)
        def cost(value):
            xyz=model.scene.world_path(curve.id,float(value),model.state)
            out=operator.project_rs(np.broadcast_to(xyz,(len(chosen),3)),pts[chosen])
            error=np.asarray(out["uv"])-uv[chosen]
            return float(np.sum(error*error)) if np.isfinite(error).all() else 1e100
        # This is only a numerical initializer of one shared station, not a fit
        # acceptance or a global material identity chosen by nearest prediction.
        grid=np.linspace(0,1,65);scores=np.asarray([cost(x) for x in grid]);j=int(np.argmin(scores))
        lo,hi=grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]
        result=minimize_scalar(cost,bounds=(lo,hi),method="bounded",options={"xatol":1e-13})
        best=min([(float(result.fun),float(result.x)),(float(scores[0]),0.),(float(scores[-1]),1.)])[1]
        station.append(best);seed_errors.append(cost(best))
        key="feature/"+feature["id"]+"/shared_C_station"
        position=S.SharedCurvePointPosition(curve.id,key,S.Interval(0.,1.,S.UnitRef(S.Unit.DIMENSIONLESS)))
        variable=S.VariableBlock(key,S.OwnerRef(S.OwnerKind.FEATURE,feature["id"]),(S.ParameterComponent("station",S.UnitRef(S.Unit.DIMENSIONLESS)),),S.VariableMode.FREE)
        landmarks.append({"id":feature["id"],"curve_id":curve.id,"entity_id":"ice.static_material","position_kind":"shared_curve_point",
            "station_variable_block_id":key,"station_seed":best,"station_bounds":[0.,1.],"station_owner":"semantic_feature",
            "station_is_free_per_observation":False,"source_parent_curve_ids":feature["parent_curve_ids"],
            "gray_parent_global_role":"pending_without_blocking_fixed_entity_on_confirmed_C",
            "typed_position":S.to_json_value(position),"typed_variable_block":S.to_json_value(variable),
            "cross_frame_observation_ids":feature["observation_ids"],"initial_seed_sum_squared_px":seed_errors[-1]})
    meta={"active":True,"source_path":str(source),"source_video_sha256":review["source"]["sha256"],"C_curve_id":curve.id,"fixed_features":len(features),"observations":len(observations),
        "parameter_owner":"one static semantic feature per shared station","per_frame_material_sliding":False,
        "statistical_sigma":None,"source_precision_estimates_are_sigma":False,"source_precision_estimates_are_hard_bounds":False,
        "engineering_target_px":float(budget_px),"engineering_target_origin":"root-authorized programme engineering pixel target; configurable; not user physical fact",
        "support_replacement_policy":"all original ice rows at these two native exposures; complete W groups replaced, independent ice rows replaced; no new parent samples also counted",
        "native_pts":np.unique(pts).tolist(),"shared_parent_evidence_groups":[v["id"] for v in review["evidence_groups"]],
        "new_parent_pixels_used_as_separate_residuals":False,"measurement_confidence_claimed":False,
        "source_precision_estimates":[{"observation_id":v["id"],"estimate":v.get("precision_limit_estimate_px")} for v in observations]}
    return {"native_pts":np.unique(pts),"landmarks":landmarks,"observations":observations,"metadata":meta,
        "arrays":{"native_pts":pts,"uv":uv,"landmark_index":indices,"shared_station_seed":np.asarray(station,dtype=np.float64),
            "row_auxiliary_seed":uv[:,1].copy(),"engineering_budget_px":np.full(len(pts),float(budget_px))}}


def load_prepared(directory,*,data_root):
    """Load current prepared arrays only; no history, d78, or old code reads."""
    path=Path(directory).resolve();path.relative_to(Path(data_root).resolve());meta=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if meta.get("schema")!=VERSION or meta.get("shot_id")!="S01":
        raise ValueError("unknown prepared schema or unlocked shot")
    meta.setdefault("data_root",str(Path(data_root).resolve()))
    store=ArrayStore(data_root);arrays={}
    for d in meta["array_refs"]:
        ref=S.from_json_value(d,S.TypedArrayRef);arrays[ref.id.removeprefix("preparation.")]=store.open(ref)
    n=int(meta["row_count"])
    for key in ("original_row","native_pts","immutable_uv","semantic_kind","curve_index","exclusion_bits","station_seed","query_index"):
        if arrays[key].shape[0]!=n:
            raise ValueError("prepared current array lengths differ")
    whitening=load_npz(meta["whitening_path"]).tocsr()
    if whitening.shape!=(meta["report"]["statistical_residual_rows"],2*n) or not np.isfinite(whitening.data).all():
        raise ValueError("prepared full source W operator shape/value mismatch")
    if np.any(arrays["curve_index"]>=len(meta["curve_registry"])) or np.any(arrays["curve_index"] < -1):
        raise ValueError("prepared stable curve foreign key invalid")
    whitening.data.setflags(write=False);whitening.indices.setflags(write=False);whitening.indptr.setflags(write=False)
    return PreparedS01(path,meta,arrays,whitening)



def rebind_current_model(prepared,model,output,*,data_root):
    """Rebind only changed current model ownership; do not replay source work.

    This is valid for rig-chart/mount changes with identical stable geometry
    identity. A semantic/topology change instead reports affected curve IDs.
    Existing current prepared arrays and W remain readonly shared generation.
    """
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():
        raise FileExistsError("new current model binding generation required")
    if model.scene.coordinate_system_id!=prepared.manifest["model_coordinate_system_id"]:
        raise ValueError("current model coordinate identity changed; explicit geometry conversion required")
    current={c.id:c for c in model.scene.curves};mismatch=[]
    for c in prepared.manifest["curve_registry"]:
        got=current.get(c["id"])
        if got is None or got.label!=c["label"] or (got.kind!=c["kind"] and model.scene.resolved_definition(got,model.state).get("kind")!=c["kind"]) or _curve_key(got)[1]!=c["original_path_id"]:
            mismatch.append(c["id"])
    if mismatch:
        raise ValueError("current geometry identity changed for "+repr(mismatch))
    meta=json.loads(json.dumps(prepared.manifest,ensure_ascii=False));meta.update(generation=int(meta["generation"])+1,
        model_id=model.id,prepared_against_model_revision=model.revision,
        current_model_state_initialization_sha256=model.state.digest(),
        previous_current_preparation_manifest=str(prepared.directory/"manifest.json"),
        rebind_reason="current rig chart/mount changed; unchanged source confirmation, geometry identity, W and all source arrays retained")
    a=prepared.arrays;pending=int(np.sum((a["curve_index"]<0)&((a["exclusion_bits"] & np.uint16(0xFFFF ^ 16))==0)))
    meta["report"].update(status="current_s01_prepared" if pending==0 else "current_s01_prepared_with_explicit_pending_bindings",
        pending_eligible_exact_geometry_rows=pending,
        unbound_retired_source_rows=int(np.sum((a["curve_index"]<0)&((a["exclusion_bits"]&4)!=0))),
        ordinary_rebind_replayed_source_preparation=False,prepared_against_model_revision=model.revision)
    meta["array_refs"]=[S.to_json_value(_incremental_ref(d["id"].removeprefix("preparation."),d["path"],a[d["id"].removeprefix("preparation.")],d["generation"],prepared.manifest["array_refs"])) for d in meta["array_refs"]]
    meta["typed_reference_units_checked_against_current_source_clock"]=True
    out.mkdir(parents=True,exist_ok=True);_json(out/"manifest.json",meta);_json(out/"report.json",meta["report"])
    return load_prepared(out,data_root=data_root)



def correct_current_noise_contract(prepared,output,*,data_root,noise_contract,model=None):
    """Incrementally correct statistical/Query/E semantics in current inputs.

    This does not re-read old packets, historical source production or code.
    Only the current profile-normal column and existing necessary prepared
    selections are consumed. Original W and immutable UV/PTS are unchanged.
    """
    from dataclasses import replace
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():
        raise FileExistsError("corrected contract requires a new current generation")
    audit=json.loads(Path(noise_contract).read_text(encoding="utf-8"))
    source=load_import(prepared.manifest["source_observation_manifest"])
    table=next(t for t in source.bundle.column_tables if t.id=="s01.current_context")
    aid=next(c.array_id for c in table.columns if c.name=="legacy_profile_normal")
    original_ref=next(a for a in source.bundle.array_refs if a.id==aid)
    normal=ArrayStore(data_root).open(original_ref)
    a=prepared.arrays;n=prepared.row_count;kinds=a["semantic_kind"]
    if normal.shape!=(n,2) or not np.isfinite(normal).all():
        raise ValueError("current original profile-normal column shape/value mismatch")
    candidate=np.flatnonzero((a["exclusion_bits"] & np.uint16(0xFFFF ^ 16))==0).astype(np.int64)
    ready=candidate[a["curve_index"][candidate]>=0];pending=candidate[a["curve_index"][candidate]<0]
    active_queries=np.flatnonzero((kinds==2)&(a["exclusion_bits"]==0)).astype(np.int64)
    if np.any(a["query_index"][active_queries]<0) or np.any(np.abs(np.linalg.norm(normal[active_queries],axis=1)-1.)>1e-8):
        raise ValueError("active genuine profile lost its original Query or source normal")
    E=a["engineering_budget_px"]
    if not np.isfinite(E[candidate]).all() or np.any(E[candidate]<=0):
        raise ValueError("all active current source E values must be positive native-pixel scales")
    meta=json.loads(json.dumps(prepared.manifest,ensure_ascii=False));meta["generation"]=int(meta["generation"])+1
    changes={"unknown_noise_budget_ready_rows":a["budget_ready_rows"],"unknown_noise_budget_candidate_rows":a["budget_candidate_rows"],
        "unknown_noise_budget_pending_binding_rows":a["budget_pending_binding_rows"],
        "budget_ready_rows":ready,"budget_candidate_rows":candidate,"budget_pending_binding_rows":pending,
        "active_query_rows":active_queries,"active_query_indices":a["query_index"][active_queries]}
    out.mkdir(parents=True,exist_ok=True);refs=[d for d in meta["array_refs"] if d["id"].removeprefix("preparation.") not in changes and d["id"]!="preparation.profile_normal"]
    for name,value in changes.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        ref=_incremental_ref(name,path,value,meta["generation"],prepared.manifest["array_refs"])
        refs.append(S.to_json_value(ref))
    refs.append(S.to_json_value(replace(original_ref,id="preparation.profile_normal",role=S.ArrayRole.IMMUTABLE_EVIDENCE)))
    meta["array_refs"]=refs
    group_origins=audit["group_source_metadata"]
    for group in meta["groups"]:
        if group["rank"]>0:
            origin=group_origins.get(group["id"])
            if origin is None:
                raise ValueError("current exact W group/source definition differs: "+repr((group["id"],group["rank"],group["member_count"],origin)))
            group["current_matrix_shape"]=[group["rank"],2*group["member_count"]]
            group["source_original_matrix_shape"]=origin["matrix_shape"]
            group["source_original_rank"]=origin["rank"]
            group["current_retained_rank"]=group["rank"]
            group["retained_support_operator_preserved_without_recomputation"]=origin["matrix_shape"]!=group["current_matrix_shape"]
            group["covariance_origin"]=origin["covariance_origin"]
            group["original_matrix_sha256"]=origin["original_matrix_sha256"]
            group["standalone_full_pixel_likelihood_claimed"]=False
    if model is not None:
        if model.scene.coordinate_system_id!=meta["model_coordinate_system_id"]:
            raise ValueError("changed model coordinate identity requires explicit current conversion")
        current={c.id:c for c in model.scene.curves}
        for c in meta["curve_registry"]:
            got=current.get(c["id"])
            if got is None or got.label!=c["label"] or _curve_key(got)[1]!=c["original_path_id"]:
                raise ValueError("affected current model semantic identity changed: "+c["id"])
        known={v["id"] for v in meta["curve_registry"]}
        meta["curve_registry"].extend({"id":c.id,"label":c.label,"entity_id":c.entity_id,"kind":c.kind,"original_path_id":_curve_key(c)[1]} for c in model.scene.curves if c.id not in known)
        meta.update(model_id=model.id,prepared_against_model_revision=model.revision,current_model_state_initialization_sha256=model.state.digest())
        meta["model_rebind_note"]="same confirmed sources/exact semantic branches; current Ad7 HD aliases read through resolved_definition; changed shape station seeds are numerical initialization only"
        meta["geometry_station_seed_reinitialization_required"]= [v["id"] for v in meta["curve_registry"] if v["label"].startswith("Ad7.")]
    meta["noise_contract_revision"]=2
    meta["statistical_error_definition"]="predicted_native_uv - immutable_source_raw_uv for ALL semantic kinds"
    meta["statistical_Query_alpha_dependency"]=False
    meta["semantic_kind_definitions"]["2"]="genuine profile: raw-pixel W statistic; original source-normal E domain; independent exact predicted_uv-Q(alpha)=0 two-component equality; original finite alpha range"
    meta["source_budget_family"]={"id":"current_all_active_source_engineering_domain","sense":"le","rows":"budget_ready_rows",
        "ordinary_q":"(du*du+dv*dv)/(E*E)","genuine_profile_q":"(source_normal dot native_raw_error)^2/(E*E)","value":"q-budget_sq",
        "source_error":"predicted_uv-immutable_raw_uv","profile_normal_array":"profile_normal","semantic_kind_array":"semantic_kind",
        "E_array":"engineering_budget_px","E_unit":"native_pixel","E_is_sigma":False,
        "budget_rhs":{"id":"numerical/shared_budget_sq","scope":"one global shared scalar for S01","unit":"dimensionless","lower":0.,
            "feasibility_phase":"free; objective minimize budget_sq; initialize max(1,max q_i)","precision_phase":"fixed exactly 1"},
        "statistical_groups_also_have_engineering_domain":True,"original_independent_rows_also_have_engineering_domain":True}
    meta["query_equality_family"]={"id":"current_genuine_profile_native_Query","sense":"eq","rows":"active_query_rows",
        "value":"predicted_native_uv-Q(alpha)","components":2,"unit":"native_pixel","alpha_bounds":"original profile Query interval",
        "does_not_replace_statistical_pixel_error":True,"does_not_subtract_Query_derivative_from_W_Jacobian":True}
    meta["engineering_constraint_family"]=dict(meta["source_budget_family"])
    meta["range_C_hard_equality_installed"]=False
    meta["new_fixed_intersections"]["engineering_target_shared_budget_rhs"]="numerical/shared_budget_sq; precision phase fixed 1"
    meta["contract_correction"]={"supersedes_semantic_contract_of":str(prepared.directory/"manifest.json"),
        "previous_wrong_profile_statistic":"predicted_uv-Q(alpha)","previous_wrong_budget_scope":"unknown-noise groups only",
        "raw_source_or_original_W_values_modified":False,"old_code_required_at_runtime":False,
        "failure_record":str(Path(data_root)/"runs/solve2_s01/preparation/semantic_failure_v1/README.json")}
    contract={"schema":"rinksolve.current_noise_and_budget_contract.v2","revision":2,"definitions":audit["definitions"],
        "current_original_W_groups":audit["current_original_W_groups"],"covariance_origins":audit["original_covariance_origins"],
        "historical_code_is_runtime_dependency":False,"range_C_hard_equality_installed":False,
        "evidence":"Read-only source/statistic review preserved in preparation/current_noise_contract.json; active kernel consumes only this current contract."}
    _json(out/"noise_contract.json",contract);meta["noise_contract_path"]=str(out/"noise_contract.json")
    report=meta["report"];report.update(noise_contract_revision=2,engineering_budget_ready_rows=len(ready),engineering_budget_pending_binding_rows=len(pending),
        unknown_noise_budget_ready_rows=len(a["budget_ready_rows"]),unknown_noise_budget_pending_binding_rows=len(a["budget_pending_binding_rows"]),
        active_original_profile_Query_equalities=len(active_queries)*2,active_original_profile_observations=len(active_queries),
        statistical_profile_raw_UV_preserved=True,statistical_Query_alpha_dependency=False,
        ordinary_statistical_sources_have_E_domain=True,global_budget_sq_is_single_shared_scalar=True,
        original_W_values_modified=False,normal_run_requires_historical_code=False,
        status="current_s01_prepared" if len(pending)==0 else "current_s01_prepared_with_explicit_pending_bindings")
    _json(out/"manifest.json",meta);_json(out/"report.json",report)
    return load_prepared(out,data_root=data_root)



def apply_current_association_delta(prepared,delta_path,fixed_observations_path,model,output,*,data_root,constraint_ledger):
    """Apply an explicitly identified current association correction only.

    Evidence columns remain immutable. Unverified entities leave the graph;
    they cannot be rescued by a free station or observation-enabled toggle.
    Full original W operators are retained, never cropped when support leaves.
    New fixed artwork corners use typed model landmarks and engineering domains.
    """
    import copy
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():
        raise FileExistsError("association correction requires a new current generation")
    if prepared.manifest.get("noise_contract_revision")!=2:
        raise ValueError("current raw-pixel statistic/all-source E contract required")
    delta=json.loads(Path(delta_path).read_text(encoding="utf-8"))
    archive=Path(delta["selection_archive_path"])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=delta["selection_archive_sha256"]:
        raise ValueError("changed association delta archive hash differs")
    with np.load(archive,allow_pickle=False) as z:
        changed={k:np.array(z[k],copy=True) for k in z.files}
    identity=prepared.observation_ids();lookup={value:i for i,value in enumerate(identity)}
    if len(lookup)!=prepared.row_count or len(set(changed["observation_id"]))!=len(changed["observation_id"]):
        raise ValueError("current/delta observation identity is not unique")
    try:
        affected=np.asarray([lookup[v] for v in changed["observation_id"]],dtype=np.int64)
    except KeyError as exc:
        raise ValueError("changed source observation is missing from current delivery") from exc
    columns=_columns(load_import(prepared.manifest["source_observation_manifest"]),data_root)
    raw,context=columns["s01.raw"],columns["s01.current_context"]
    checks=((raw["source_id"],changed["source_id"]),(raw["source_family_id"],changed["source_family_id"]),
        (raw["source_dataset_id"],changed["source_dataset_id"]),(raw["native_pts"],changed["native_pts"]),
        (raw["original_row"],changed["original_raw_row"]),(context["legacy_material_index"],changed["previous_material_index"]),
        (context["legacy_path"],changed["previous_path_index"]))
    if any(not np.array_equal(current[affected],expected) for current,expected in checks):
        raise ValueError("affected association delta does not match current necessary source references")
    if model.scene.coordinate_system_id!=prepared.manifest["model_coordinate_system_id"]:
        raise ValueError("affected model coordinate identity changed")
    current_curves={c.id:c for c in model.scene.curves}
    for prior in prepared.manifest["curve_registry"]:
        got=current_curves.get(prior["id"])
        if got is None or got.label!=prior["label"] or _curve_key(got)[1]!=prior["original_path_id"]:
            raise ValueError("unaccounted model semantic change: "+prior["id"])
    meta=copy.deepcopy(prepared.manifest);a=prepared.arrays;n=prepared.row_count
    rejected=np.zeros(n,dtype=bool);rejected[affected]=True
    groups=meta["groups"];withdrawn=[];collateral=[]
    for i,group in enumerate(groups):
        lo,hi=a["group_member_offsets"][i:i+2];members=a["group_member_rows"][lo:hi]
        hit=members[rejected[members]]
        if not len(hit):continue
        group["exclusion_reason_union"]=sorted(set(group.get("exclusion_reason_union",[])+[REASONS[1024]]))
        group["affected_source_observation_ids"]=[_text(v) for v in identity[hit]]
        if group["rank"]>0:
            group["active_statistical"]=False
            group["association_exclusion_reason"]="complete original W withdrawn; changed source entity association is unverified; original W/source support retained"
            collateral.extend(int(v) for v in members[~rejected[members]])
            rejected[members]=True;withdrawn.append(group["id"])
    kept=[];matrix_rows=[];cursor=0
    for f in meta["statistical_factors"]:
        if np.any(rejected[f["rows"]]):continue
        start,count=f["residual_start"],f["residual_count"]
        matrix_rows.extend(range(start,start+count));f["residual_start"]=cursor;cursor+=count;kept.append(f)
    whitening=prepared.whitening[matrix_rows,:].tocsr()
    meta["statistical_factors"]=kept
    changes={"exclusion_bits":np.array(a["exclusion_bits"],copy=True)}
    changes["exclusion_bits"][rejected]|=1024
    for name in ("active_independent_rows","budget_ready_rows","budget_candidate_rows","budget_pending_binding_rows",
                 "unknown_noise_budget_ready_rows","unknown_noise_budget_candidate_rows","unknown_noise_budget_pending_binding_rows",
                 "active_query_rows"):
        rows=a[name];changes[name]=rows[~rejected[rows]]
    changes["active_query_indices"]=a["active_query_indices"][~rejected[a["active_query_rows"]]]
    changes["association_retired_rows"]=np.sort(affected)
    changes["association_collateral_rows"]=np.asarray(sorted(set(collateral)),dtype=np.int64)
    ledger_path=Path(constraint_ledger).resolve();ledger=json.loads(ledger_path.read_text(encoding="utf-8-sig"))
    policy=next(v for v in ledger["constraints"] if v["id"]=="numerical.ad7_fixed_corner_budget")
    E=float(policy["expression"]["radius_px"])
    if not np.isfinite(E) or E<=0 or policy["expression"].get("is_statistical_sigma") is not False:
        raise ValueError("current Ad7 fixed-corner engineering budget is not valid")
    source=json.loads(Path(fixed_observations_path).read_text(encoding="utf-8"))
    features=source["features"];observations=source["observations"]
    if len(features)!=policy["expression"]["fixed_entities"] or len(observations)!=policy["expression"]["observations"]:
        raise ValueError("current Ad7 fixed source scope differs from explicit engineering policy")
    landmarks={v.id:v for v in model.landmarks};feature_index={v["id"]:i for i,v in enumerate(features)}
    if len(feature_index)!=len(features):raise ValueError("duplicate new fixed feature identity")
    for feature in features:
        landmark=landmarks.get(feature["id"])
        if landmark is None or not isinstance(landmark.position,S.StructuralPointPosition):
            raise ValueError("new fixed glyph corner is not a typed current structural landmark")
        if tuple(landmark.position.geometry_feature_ids)!=(feature["curve_id"],) or feature["station_is_optimizer_variable"]:
            raise ValueError("fixed glyph corner semantic ownership differs")
        if not np.isfinite(model.scene.landmark_world(landmark,model.state)).all():
            raise ValueError("new fixed glyph landmark cannot be evaluated from current shared geometry")
    ids=set(map(_text,identity));seen=set();supplements=[]
    allowed_pts=set(policy["expression"]["source_native_pts"])
    for observation in observations:
        item=copy.deepcopy(observation);oid=item["id"]
        if oid in seen or oid in ids:raise ValueError("duplicate new source observation identity")
        seen.add(oid)
        if item["feature_id"] not in feature_index or item["native_pts"] not in allowed_pts:
            raise ValueError("new fixed source feature/exposure outside policy scope")
        if item["source_sigma_px"] is not None or item["uv_from_projected_template_or_nearest"]:
            raise ValueError("fixed glyph source cannot be given invented noise or model-filled pixels")
        if item["measurement_scope"]!="ice_board_static_print" or not item.get("dependency_family"):
            raise ValueError("fixed glyph source scope or correlation family missing")
        item.update(fixed_feature_id=item["feature_id"],uv_native_pixels=item["uv"],engineering_budget_px=E,
            engineering_budget_constraint_id=policy["id"],statistical_objective_weight=None,
            shared_global_budget_id="numerical/shared_budget_sq",fit_admission="current_static_board_glyph_fixed_corner",
            per_observation_world_station=False)
        if policy["enabled"]:supplements.append(item)
    uv=np.asarray([v["uv_native_pixels"] for v in supplements],dtype=np.float64).reshape(-1,2)
    if not np.isfinite(uv).all() or np.any(uv<0) or np.any(uv[:,0]>=1920) or np.any(uv[:,1]>=1080):
        raise ValueError("new glyph corner source is outside native image")
    changes.update(ad7_fixed_native_pts=np.asarray([v["native_pts"] for v in supplements],dtype=np.int64),ad7_fixed_uv=uv,
        ad7_fixed_landmark_index=np.asarray([feature_index[v["fixed_feature_id"]] for v in supplements],dtype=np.int32),
        ad7_fixed_row_auxiliary_seed=uv[:,1].copy(),ad7_fixed_engineering_budget_px=np.full(len(supplements),E,dtype=np.float64))
    meta.update(generation=int(meta["generation"])+1,model_id=model.id,prepared_against_model_revision=model.revision,
        current_model_state_initialization_sha256=model.state.digest(),previous_current_preparation_manifest=str(prepared.directory/"manifest.json"),
        ad7_fixed_observations=supplements,ad7_fixed_landmarks=[S.to_json_value(landmarks[v["id"]]) for v in features],
        ad7_fixed_source_features=features,selection_revision=0)
    meta["reason_bits"]["1024"]=REASONS[1024]
    meta["source_association_delta"]={"id":"Ad7_unverified_word_retirement","revision":delta["revision"],
        "source_delta_path":str(Path(delta_path).resolve()),"reason":delta["reason_code"],"excluded_source_observations":len(affected),
        "excluded_current_rows_array":"association_retired_rows","collateral_current_rows_array":"association_collateral_rows",
        "withdrawn_original_W_groups":withdrawn,"raw_UV_PTS_IDs_or_W_values_modified":False,
        "readmitting_requires_corrected_source_entity_association":True,"selection_enabled_cannot_override_admission":True,
        "current_unverified_word_bindings_are_inactive":True,"named_ISU_symbol_sources_excluded":False}
    meta["ad7_fixed_corner_family"]={"id":policy["id"],"sense":"le","value":"squared_norm(predicted_uv-immutable_uv)/E^2-budget_sq",
        "E_px":E,"E_is_sigma":False,"statistical_weight":None,"shared_global_budget_id":"numerical/shared_budget_sq",
        "source_noise_known":False,"source_observations_path":str(Path(fixed_observations_path).resolve()),
        "same_glyph_same_exposure_and_nearby_exposure_dependencies_retained":True,
        "existing_unverified_word_support_withdrawn_not_double_counted":True,
        "fixed_entity_position":"current model StructuralPointPosition; shared layout/board varies, no local station",
        "source_candidate_status_superseded_by_current_root_admission":True,"whole_shot_visual_acceptance":False}
    binding=meta["constraints_binding"];prior=next(v for v in binding["relevant_constraints"] if v["id"]=="numerical.intersection_budget")
    previous_digest=binding.get("relevant_sha256_by_id",{}).get(prior["id"],binding.get("relevant_sha256"))
    relevant={"id":policy["id"],"enabled":policy["enabled"],"expression":policy["expression"]}
    binding.update(path=str(ledger_path),revision=ledger["revision"],relevant_constraints=[prior,relevant],
        relevant_sha256_by_id={prior["id"]:previous_digest,relevant["id"]:hashlib.sha256(json.dumps(relevant,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()})
    out.mkdir(parents=True,exist_ok=True);refs=[d for d in meta["array_refs"] if d["id"].removeprefix("preparation.") not in changes]
    for name,value in changes.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        ref=_incremental_ref(name,path,value,meta["generation"],prepared.manifest["array_refs"])
        refs.append(S.to_json_value(ref))
    meta["array_refs"]=refs;save_npz(out/"whitening.npz",whitening);meta["whitening_path"]=str(out/"whitening.npz")
    report=meta["report"];report.update(engineering_budget_ready_rows=len(changes["budget_ready_rows"]),
        unknown_noise_budget_ready_rows=len(changes["unknown_noise_budget_ready_rows"]),
        active_original_W_groups=sum(g["active_statistical"] for g in groups),
        active_joint_W_residual_rows=sum(g["rank"] for g in groups if g["active_statistical"]),
        active_independent_sources=len(changes["active_independent_rows"]),statistical_residual_rows=cursor,
        statistical_nonzero_coefficients=whitening.nnz,ad7_unverified_word_observations_excluded=len(affected),
        ad7_named_symbol_sources_preserved=delta["named_ISU_symbol_rows_unchanged"],
        ad7_association_whole_W_withdrawals=len(withdrawn),ad7_association_collateral_rows=len(changes["association_collateral_rows"]),
        new_fixed_ad7_corner_features=len(features),new_fixed_ad7_corner_budget_constraints=len(supplements),
        new_fixed_ad7_corner_statistical_weight=None,new_fixed_ad7_per_observation_world_station=False,
        prepared_against_model_revision=model.revision,association_source_requalification_replayed=False,
        whole_shot_visual_acceptance=False)
    report["exclusion_counts"][REASONS[1024]]=int(rejected.sum())
    _json(out/"manifest.json",meta);_json(out/"report.json",report)
    return load_prepared(out,data_root=data_root)



def restore_current_source_associations(prepared,delta_path,model,output,*,data_root):
    """Admit reviewed per-observation finite contours, not material-wide labels.

    The current source association delta is input data. Only affected current
    IDs, immutable UV/PTS and typed finite branch references are checked.
    Unknown source noise remains unknown; restored sources have E domains only.
    """
    import copy
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():raise FileExistsError("new source association generation required")
    delta=json.loads(Path(delta_path).read_text(encoding="utf-8"));archive=Path(delta["archive"])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=delta["archive_sha256"]:
        raise ValueError("current reviewed association delta archive differs")
    with np.load(archive,allow_pickle=False) as z: values={k:np.array(z[k],copy=True) for k in z.files}
    if not np.isin(values["action"],[0,1]).all():raise ValueError("unknown source association action")
    identity=prepared.observation_ids();lookup={v:i for i,v in enumerate(identity)}
    try:rows=np.asarray([lookup[v] for v in values["observation_id"]],dtype=np.int64)
    except KeyError as exc:raise ValueError("reviewed changed source missing from current delivery") from exc
    if len(set(rows))!=len(rows) or set(rows)!=set(prepared.arrays["association_retired_rows"]):
        raise ValueError("current reviewed delta must exactly cover its previously withdrawn associations")
    source=_columns(load_import(prepared.manifest["source_observation_manifest"]),data_root)
    raw,ctx=source["s01.raw"],source["s01.current_context"]
    for name,current in (("source_id",raw["source_id"]),("source_family_id",raw["source_family_id"]),
        ("source_dataset_id",raw["source_dataset_id"]),("native_pts",raw["native_pts"]),("original_raw_row",raw["original_row"]),
        ("previous_material_index",ctx["legacy_material_index"]),("previous_path_index",ctx["legacy_path"])):
        if not np.array_equal(current[rows],values[name]):raise ValueError("reviewed association current "+name+" differs")
    restored=rows[values["action"]==1];target_ids=values["target_curve_id"][values["action"]==1]
    registry=prepared.manifest["curve_registry"];curve_indices={v["id"]:i for i,v in enumerate(registry)}
    curves={v.id:v for v in model.scene.curves};reviewed={v["observation_id"]:v for v in delta["restores"]}
    if len(restored)!=delta["restore_approved_budget_only"] or len(reviewed)!=len(restored):
        raise ValueError("reviewed source association scope differs")
    a=prepared.arrays;changes={name:np.array(a[name],copy=True) for name in ("curve_index","binding_source","exclusion_bits","station_seed","source_sigma")}
    used_statistical=np.unique(prepared.whitening.indices//2)
    if np.isin(restored,used_statistical).any() or np.isin(restored,a["active_independent_rows"]).any():
        raise ValueError("unknown-noise restored material unexpectedly has a statistical factor")
    if np.any((a["exclusion_bits"][restored]&np.uint16(0xFFFF ^ 1024))!=0):
        raise ValueError("source association cannot bypass other current admission failures")
    if model.id!=prepared.manifest["model_id"] or model.revision!=prepared.manifest["prepared_against_model_revision"]:
        raise ValueError("source association delta requires its exact current model")
    for j in np.flatnonzero(values["action"]==1):
        row=int(rows[j]);oid=_text(values["observation_id"][j]);record=reviewed[oid];cid=_text(values["target_curve_id"][j])
        curve=curves.get(cid)
        if curve is None or cid not in curve_indices:raise ValueError("reviewed exact finite material branch absent")
        definition=model.scene.resolved_definition(curve,model.state)
        if definition.get("source_feature_id")!=_text(values["target_source_feature_id"][j]):
            raise ValueError("reviewed actual glyph boundary does not match target source feature")
        if record["target_curve_id"]!=cid or record["position_kind"]!="curve_locus_not_fixed_point" or record["locus_interval"]!=[0.,1.]:
            raise ValueError("reviewed finite source branch semantics differ")
        if record["current_model_projection_used_for_assignment"] or record["sigma_px"] is not None:
            raise ValueError("reviewed source identity/noise cannot be inferred from model proximity")
        if not np.array_equal(a["immutable_uv"][row],record["uv_unchanged"]):raise ValueError("reviewed raw UV changed")
        changes["curve_index"][row]=curve_indices[cid];changes["binding_source"][row]=2
        changes["exclusion_bits"][row]&=np.uint16(0xFFFF ^ 1024);changes["source_sigma"][row]=np.nan
    # This chooses only a numerical station on the already reviewed exact finite
    # boundary. It does not choose the entity, create UV, or fix the station.
    operator=model.operator();grid=np.linspace(0.,1.,257)
    seed_error=[]
    for cid_raw in np.unique(target_ids):
        cid=_text(cid_raw);chosen=restored[target_ids==cid_raw]
        xyz=model.scene.world_path(cid,grid,model.state)
        result=operator.project_with_row_auxiliary(xyz[None,:,:],a["native_pts"][chosen,None],a["row_auxiliary_seed"][chosen,None])
        error=result["uv"]-a["immutable_uv"][chosen,None,:];cost=np.sum(error*error,axis=-1)
        cost=np.where(result["valid"],cost,np.inf)
        at=np.argmin(cost,axis=1)
        if not np.isfinite(cost[np.arange(len(chosen)),at]).all():raise ValueError("current known branch station initializer invalid")
        changes["station_seed"][chosen]=grid[at];seed_error.extend(np.sqrt(cost[np.arange(len(chosen)),at]).tolist())
    for name in ("budget_ready_rows","budget_candidate_rows","unknown_noise_budget_ready_rows","unknown_noise_budget_candidate_rows"):
        changes[name]=np.union1d(a[name],restored).astype(np.int64)
    changes["association_retired_rows"]=np.sort(rows[values["action"]==0])
    changes["association_restored_rows"]=np.sort(restored)
    meta=copy.deepcopy(prepared.manifest);meta.update(generation=int(meta["generation"])+1,
        previous_current_preparation_manifest=str(prepared.directory/"manifest.json"),
        current_reviewed_association_delta={"path":str(Path(delta_path).resolve()),"revision":delta["revision"],
            "remaining_unverified":len(changes["association_retired_rows"]),"restored_budget_only":len(restored),
            "per_observation_overrides_only":True,"material_wide_restore":False,"source_UV_PTS_ID_unchanged":True,
            "exact_source_feature_identity_required":True,"numerical_station_seed_is_not_source_identity":True},
        current_source_association_overrides=delta["restores"],current_association_dependency_clusters=delta["close_detection_clusters"])
    meta["binding_source_codes"]["2"]="visually reviewed per-observation actual glyph/finite source contour; material-level prior not authoritative"
    out.mkdir(parents=True,exist_ok=True);refs=[d for d in meta["array_refs"] if d["id"].removeprefix("preparation.") not in changes]
    for name,value in changes.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        refs.append(S.to_json_value(_incremental_ref(name,path,value,meta["generation"],prepared.manifest["array_refs"])))
    meta["array_refs"]=refs
    report=meta["report"];report.update(engineering_budget_ready_rows=len(changes["budget_ready_rows"]),
        unknown_noise_budget_ready_rows=len(changes["unknown_noise_budget_ready_rows"]),
        ad7_unverified_word_observations_excluded=len(changes["association_retired_rows"]),ad7_reviewed_per_source_glyph_associations_restored=len(restored),
        ad7_restored_observation_noise_unknown=True,ad7_material_wide_restore=False,
        ad7_restored_station_initializer_max_native_error_px=max(seed_error),
        ad7_close_detection_dependency_clusters=len(delta["close_detection_clusters"]))
    report["exclusion_counts"][REASONS[1024]]=int(np.sum((changes["exclusion_bits"]&1024)!=0))
    _json(out/"manifest.json",meta);_json(out/"report.json",report)
    return load_prepared(out,data_root=data_root)



def refresh_current_target_snapshot(prepared,ledger_path):
    """Rebind editable current engineering radii/enabled flags in memory only.

    No source qualification, producer, model or historical code is reopened.
    Call before the user's selection overlay. Turning off intersection replacement
    restores eligible original ice support from the delivered complete operators;
    turning it on withdraws complete original W groups at those two exposures.
    """
    import copy
    if prepared.manifest.get("selection_revision",0):
        raise ValueError("refresh engineering targets before applying the current observation selection overlay")
    path=Path(ledger_path).resolve();ledger=json.loads(path.read_text(encoding="utf-8-sig"));entries={v["id"]:v for v in ledger["constraints"]}
    meta=copy.deepcopy(prepared.manifest);arrays=dict(prepared.arrays);binding=meta.get("constraints_binding")
    if not binding:raise ValueError("current engineering targets have no delivered snapshot")
    changed=[];relevant=[];digests={}
    for prior in binding["relevant_constraints"]:
        current=entries.get(prior["id"])
        if current is None:raise ValueError("current engineering target removed: "+prior["id"])
        old_expression=copy.deepcopy(prior["expression"]);expression=copy.deepcopy(current["expression"])
        old_radius=old_expression.pop("radius_px",None);radius=expression.pop("radius_px",None)
        if expression!=old_expression:
            raise ValueError("engineering scope/noise/support semantics changed, beyond editable radius/enabled: "+prior["id"])
        if type(current["enabled"]) is not bool or not isinstance(radius,(int,float)) or not np.isfinite(radius) or radius<=0:
            raise ValueError("current engineering target needs a positive radius and boolean enabled")
        if current["enabled"]!=prior["enabled"] or radius!=old_radius:changed.append(prior["id"])
        value={"id":current["id"],"enabled":current["enabled"],"expression":current["expression"]};relevant.append(value)
        digests[current["id"]]=hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
    reservoir=meta.setdefault("current_target_source_registry",{})
    for family,prefix,list_name,keys in (
        ("numerical.intersection_budget","intersection","new_intersection_observations",("native_pts","uv","landmark_index","row_auxiliary_seed","engineering_budget_px")),
        ("numerical.ad7_fixed_corner_budget","ad7_fixed","ad7_fixed_observations",("native_pts","uv","landmark_index","row_auxiliary_seed","engineering_budget_px"))):
        policy=entries.get(family)
        if family not in {v["id"] for v in relevant}:continue
        if family not in reservoir:
            reservoir[family]={"observations":copy.deepcopy(meta.get(list_name,[])),
                "arrays":{key:arrays[prefix+"_"+key].tolist() for key in keys},
                "replacement_rows":np.flatnonzero((arrays["exclusion_bits"]&256)!=0).tolist() if prefix=="intersection" else []}
        source=reservoir[family];enabled=policy["enabled"];radius=float(policy["expression"]["radius_px"])
        observations=copy.deepcopy(source["observations"]) if enabled else []
        for item in observations:item["engineering_budget_px"]=radius
        meta[list_name]=observations
        for key in keys:
            original=arrays[prefix+"_"+key];value=np.asarray(source["arrays"][key],dtype=original.dtype)
            if key=="uv":value=value.reshape(-1,2)
            value=np.full(len(value),radius,dtype=np.float64) if key=="engineering_budget_px" else value
            arrays[prefix+"_"+key]=np.ascontiguousarray(value if enabled else value[:0])
        if prefix=="intersection":
            meta["new_fixed_intersections"].update(active=enabled,engineering_target_px=radius)
        else:meta["ad7_fixed_corner_family"].update(enabled=enabled,E_px=radius)
    old_intersection=next((v for v in binding["relevant_constraints"] if v["id"]=="numerical.intersection_budget"),None)
    toggle=old_intersection is not None and old_intersection["enabled"]!=entries[old_intersection["id"]]["enabled"]
    whitening=prepared.whitening
    if toggle:
        quality=np.array(arrays["exclusion_bits"],copy=True);quality&=np.uint16(0xFFFF ^ 256)
        if entries[old_intersection["id"]]["enabled"]:
            pts=meta["new_fixed_intersections"]["native_pts"]
            ice_curves=[i for i,c in enumerate(meta["curve_registry"]) if c["label"].startswith("ice.")]
            replace_rows=np.isin(arrays["native_pts"],pts)&np.isin(arrays["curve_index"],ice_curves)
            replace_rows[reservoir[old_intersection["id"]]["replacement_rows"]]=True
            quality[replace_rows]|=256
            for i,group in enumerate(meta["groups"]):
                lo,hi=arrays["group_member_offsets"][i:i+2];rows=arrays["group_member_rows"][lo:hi]
                if group["rank"]>0 and np.any(replace_rows[rows]):quality[rows]|=256
        active_groups=[]
        for i,group in enumerate(meta["groups"]):
            lo,hi=arrays["group_member_offsets"][i:i+2];rows=arrays["group_member_rows"][lo:hi]
            union=int(np.bitwise_or.reduce(quality[rows],initial=0));active=group["rank"]>0 and union==0
            group.update(active_statistical=active,exclusion_reason_union=reason_names(union))
            if active:active_groups.append(prepared.source_group(i))
        independent=arrays["independent_source_rows"];independent=independent[quality[independent]==0]
        plan=WhiteningPlan(prepared.row_count,active_groups,independent,arrays["source_sigma"][independent],
            independent_ids=["independent/"+_text(v) for v in prepared.observation_ids(independent)])
        whitening=plan.operator;arrays["active_independent_rows"]=independent;arrays["exclusion_bits"]=quality
        candidates=np.flatnonzero((quality&np.uint16(0xFFFF ^ 16))==0).astype(np.int64)
        arrays["budget_candidate_rows"]=candidates
        arrays["budget_ready_rows"]=candidates[arrays["curve_index"][candidates]>=0]
        arrays["budget_pending_binding_rows"]=candidates[arrays["curve_index"][candidates]<0]
        unknown=[]
        for i,g in enumerate(meta["groups"]):
            if g["rank"]!=0:continue
            lo,hi=arrays["group_member_offsets"][i:i+2];rows=arrays["group_member_rows"][lo:hi]
            unknown.extend(rows[(quality[rows]&np.uint16(0xFFFF ^ 16))==0])
        unknown=np.asarray(sorted(set(map(int,unknown))),dtype=np.int64)
        arrays["unknown_noise_budget_candidate_rows"]=unknown
        arrays["unknown_noise_budget_ready_rows"]=unknown[arrays["curve_index"][unknown]>=0]
        arrays["unknown_noise_budget_pending_binding_rows"]=unknown[arrays["curve_index"][unknown]<0]
        query=np.flatnonzero((arrays["semantic_kind"]==2)&(quality==0)).astype(np.int64)
        arrays["active_query_rows"]=query;arrays["active_query_indices"]=arrays["query_index"][query]
        meta["statistical_factors"]=[{"id":f.id,"rows":f.rows.tolist(),"residual_start":f.residual_start,"residual_count":f.residual_count,"kind":f.kind} for f in plan.factors]
    binding.update(path=str(path),revision=ledger["revision"],relevant_constraints=relevant,relevant_sha256_by_id=digests)
    meta["current_target_refresh"]={"changed_families":changed,"source_requalification_replayed":False,
        "original_W_values_changed":False,"in_memory_only":True,"must_precede_selection_overlay":True}
    meta["report"].update(statistical_residual_rows=whitening.shape[0],statistical_nonzero_coefficients=whitening.nnz,
        active_original_W_groups=sum(g["active_statistical"] for g in meta["groups"]),
        active_joint_W_residual_rows=sum(g["rank"] for g in meta["groups"] if g["active_statistical"]),
        active_independent_sources=len(arrays["active_independent_rows"]),engineering_budget_ready_rows=len(arrays["budget_ready_rows"]),
        unknown_noise_budget_ready_rows=len(arrays["unknown_noise_budget_ready_rows"]),
        active_original_profile_observations=len(arrays["active_query_rows"]),active_original_profile_Query_equalities=2*len(arrays["active_query_rows"]),
        new_fixed_intersection_budget_constraints=len(meta.get("new_intersection_observations",[])),
        new_fixed_ad7_corner_budget_constraints=len(meta.get("ad7_fixed_observations",[])))
    meta["report"]["exclusion_counts"][REASONS[256]]=int(np.sum((arrays["exclusion_bits"]&256)!=0))
    for value in arrays.values():value.setflags(write=False)
    for key in ("data","indices","indptr"):getattr(whitening,key).setflags(write=False)
    return PreparedS01(prepared.directory,meta,arrays,whitening)



def retire_current_budget_only_sources(prepared,delta_path,output,*,data_root,reason_bit=2048,association_scope="ad8",review_path=None):
    """Withdraw reviewed-unresolved finite associations from current E support.

    Only an explicit stable-ID delta is consumed. This fast path is deliberately
    limited to budget-only sources: a changed statistical support instead needs
    the complete-W-group update, never a hidden cropped covariance. Raw records,
    prior human confirmations, geometry, original W and unrelated sources stay.
    """
    import copy
    if reason_bit not in (1024,2048,4096):raise ValueError("use an existing registered base source-association reason")
    if not association_scope or not association_scope.replace("_","").isalnum():raise ValueError("invalid current association metadata scope")
    rows_name=association_scope+"_unverified_retired_rows"
    if prepared.manifest.get("selection_revision",0):raise ValueError("apply source admission before the separate observation selection overlay")
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():raise FileExistsError("new current admission generation required")
    delta=json.loads(Path(delta_path).read_text(encoding="utf-8"));archive=Path(delta["selection_archive_path"])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=delta["selection_archive_sha256"]:
        raise ValueError("changed source admission archive differs")
    if delta["base_model_revision"]!=prepared.manifest["prepared_against_model_revision"] or delta["restore_count"]!=0:
        raise ValueError("current unresolved association model/scope differs")
    with np.load(archive,allow_pickle=False) as z:values={k:np.array(z[k],copy=True) for k in z.files}
    ids=prepared.observation_ids();lookup={v:i for i,v in enumerate(ids)}
    try:rows=np.asarray([lookup[v] for v in values["observation_id"]],dtype=np.int64)
    except KeyError as exc:raise ValueError("changed stable source ID absent from current delivery") from exc
    if len(rows)!=delta["retirement_count"] or len(set(rows))!=len(rows) or np.any(values["action"]!=0):
        raise ValueError("current exact retirement coverage/action differs")
    source=_columns(load_import(prepared.manifest["source_observation_manifest"]),data_root)
    raw,ctx=source["s01.raw"],source["s01.current_context"];a=prepared.arrays
    for name,current in (("source_id",raw["source_id"]),("source_family_id",raw["source_family_id"]),
        ("source_dataset_id",raw["source_dataset_id"]),("native_pts",a["native_pts"]),("immutable_uv",a["immutable_uv"]),
        ("original_raw_row",a["original_row"]),("previous_material_index",ctx["legacy_material_index"]),
        ("previous_path_index",ctx["legacy_path"]),("existing_exclusion_bits",a["exclusion_bits"])):
        if name not in values:
            if name in ("source_family_id","previous_material_index","previous_path_index"):continue
            raise ValueError("affected current source column missing: "+name)
        if not np.array_equal(current[rows],values[name]):raise ValueError("affected current source "+name+" differs")
    actual_curves=prepared.bound_curve_ids(rows)
    if not np.array_equal(actual_curves,np.asarray([_text(v) for v in values["previous_curve_id"]])):
        raise ValueError("affected current finite branch identity differs")
    per_curve_counts={str(cid):int(np.sum(actual_curves==cid)) for cid in np.unique(actual_curves)}
    if "per_previous_curve" in delta and per_curve_counts!=delta["per_previous_curve"]:
        raise ValueError("affected finite branches/counts exceed reviewed current scope")
    if np.isin(rows,prepared.whitening.indices//2).any() or np.isin(rows,a["active_independent_rows"]).any():
        raise ValueError("current retirement involves statistical support; use complete original W group withdrawal")
    if not np.isin(rows,a["unknown_noise_budget_ready_rows"]).all():
        raise ValueError("current fast retirement requires admitted budget-only sources")
    rejected=np.zeros(prepared.row_count,dtype=bool);rejected[rows]=True
    meta=copy.deepcopy(prepared.manifest);changes={"exclusion_bits":np.array(a["exclusion_bits"],copy=True)}
    changes["exclusion_bits"][rows]|=reason_bit
    for name in ("budget_ready_rows","budget_candidate_rows","budget_pending_binding_rows",
        "unknown_noise_budget_ready_rows","unknown_noise_budget_candidate_rows","unknown_noise_budget_pending_binding_rows"):
        candidate=a[name];selected=candidate[~rejected[candidate]]
        if len(candidate)!=len(selected):changes[name]=selected
    query=a["active_query_rows"];keep=~rejected[query]
    if not np.all(keep):changes.update(active_query_rows=query[keep],active_query_indices=a["active_query_indices"][keep])
    changes[rows_name]=np.sort(rows)
    for i,g in enumerate(meta["groups"]):
        lo,hi=a["group_member_offsets"][i:i+2];members=a["group_member_rows"][lo:hi];hit=members[rejected[members]]
        if not len(hit):continue
        if g["rank"]!=0:raise ValueError("unresolved finite association unexpectedly belongs to original W")
        g["exclusion_reason_union"]=sorted(set(g.get("exclusion_reason_union",[])+[REASONS[reason_bit]]))
        g["current_unverified_finite_association_observation_ids"]=sorted(set(g.get("current_unverified_finite_association_observation_ids",[])+[_text(v) for v in ids[hit]]))
    review_records=[]
    if review_path is not None:
        review=json.loads(Path(review_path).read_text(encoding="utf-8"));review_records=review["reviewed_rows"]
        if {_text(v) for v in values["observation_id"]}!={v["stable_id"] for v in review_records}:
            raise ValueError("current detailed review covers different source IDs")
    meta.update(generation=int(meta["generation"])+1,previous_current_preparation_manifest=str(prepared.directory/"manifest.json"))
    meta[association_scope+"_current_association_retirement"]={"source_delta_path":str(Path(delta_path).resolve()),"revision":delta["revision"],
        "reason":delta["reason_code"],"reason_bit":reason_bit,"count":len(rows),"rows_array":rows_name,
        "source_observation_ids":[_text(v) for v in values["observation_id"]],
        "previous_source_bindings":[{"observation_id":_text(values["observation_id"][j]),"source_id":_text(values["source_id"][j]),
            "native_pts":int(values["native_pts"][j]),"uv":values["immutable_uv"][j].tolist(),"previous_curve_id":_text(values["previous_curve_id"][j])} for j in range(len(rows))],
        "review_path":str(Path(review_path).resolve()) if review_path else None,"review_records":review_records,
        "review_evidence_path":delta.get("review_evidence"),"fixed_corner_candidates_installed":False,"finite_candidates_rebound":False,
        "readmitting_requires_corrected_source_finite_branch_association":True,"selection_enabled_cannot_override_admission":True,
        "per_curve_counts":per_curve_counts,"raw_UV_PTS_IDs_modified":False,"W_modified":False,
        "source_production_or_qualification_replayed":False,"model_revision_or_state_modified":False,
        "unaffected_confirmed_source_associations_preserved":True}
    meta["reason_bits"][str(reason_bit)]=REASONS[reason_bit]
    out.mkdir(parents=True,exist_ok=True);refs=[d for d in meta["array_refs"] if d["id"].removeprefix("preparation.") not in changes]
    for name,value in changes.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        refs.append(S.to_json_value(_incremental_ref(name,path,value,meta["generation"],prepared.manifest["array_refs"])))
    meta["array_refs"]=refs;report=meta["report"]
    report.update(engineering_budget_ready_rows=len(changes.get("budget_ready_rows",a["budget_ready_rows"])),
        unknown_noise_budget_ready_rows=len(changes.get("unknown_noise_budget_ready_rows",a["unknown_noise_budget_ready_rows"])),
        **{association_scope+"_unverified_finite_glyph_source_associations_excluded":len(rows),
           association_scope+"_current_scope_restored_sources":0,association_scope+"_current_delta_statistical_factors_affected":0})
    if association_scope=="ad8":report["ad8_confirmed_d78_source_associations_retained"]=True
    report["exclusion_counts"][REASONS[reason_bit]]=int(np.sum((changes["exclusion_bits"]&reason_bit)!=0))
    _json(out/"manifest.json",meta);_json(out/"report.json",report)
    return load_prepared(out,data_root=data_root)



def detach_reviewed_source_associations(prepared,review_path,output,*,data_root,reason_bit=4096,association_scope=None,delta_path=None,audit_report_path=None):
    """Detach specific visually reviewed source bindings, retaining all evidence.

    Stable source IDs select associations, never an entire material label.
    Independent sigma factors leave individually. Touching any complete original
    W support withdraws that whole factor and records its collateral support.
    A possible glyph name is not an approved finite-branch rebind.
    """
    import copy
    if reason_bit not in (1024,2048,4096):raise ValueError("use an existing registered base source-association reason")
    if prepared.manifest.get("selection_revision",0):raise ValueError("apply source admission before the separate observation selection overlay")
    out=Path(output).resolve();out.relative_to(Path(data_root).resolve())
    if (out/"manifest.json").exists():raise FileExistsError("new current association generation required")
    review=json.loads(Path(review_path).read_text(encoding="utf-8"));records=review["reviewed_rows"]
    scope=association_scope or review["scope"]["advertisement"].lower()
    if not scope or not scope.replace("_","").isalnum():raise ValueError("invalid current association metadata scope")
    direct_name=scope+"_specific_detached_rows";collateral_name=scope+"_W_collateral_rows";independent_name=scope+"_removed_independent_rows"
    if review["model_revision"]!=prepared.manifest["prepared_against_model_revision"]:
        raise ValueError("reviewed source associations refer to a different current model")
    ids=prepared.observation_ids();lookup={_text(v):i for i,v in enumerate(ids)}
    try:direct=np.asarray([lookup[v["stable_id"]] for v in records],dtype=np.int64)
    except KeyError as exc:raise ValueError("reviewed stable source ID absent from current delivery") from exc
    if len(direct)!=review["summary"]["point_candidates"] or len(set(direct))!=len(direct):
        raise ValueError("reviewed exact source association count/identity differs")
    a=prepared.arrays;registry=prepared.manifest["curve_registry"]
    for row,record in zip(direct,records):
        ci=int(a["curve_index"][row])
        if ci<0 or registry[ci]["label"]!=record["current_label"] or not record["current_label"].startswith(review["scope"]["advertisement"]+"."):
            raise ValueError("reviewed current source entity binding differs")
        if int(a["native_pts"][row])!=record["native_pts"] or not np.array_equal(a["immutable_uv"][row],record["uv"]):
            raise ValueError("reviewed immutable source UV/PTS differs")
        if int(a["exclusion_bits"][row])!=record["excluded"] or record["recommended_fit_action"]!="detach_current_association_pending_review":
            raise ValueError("reviewed current source admission/action differs")
    delta=None;audit_summary=None
    if delta_path is not None:
        delta=json.loads(Path(delta_path).read_text(encoding="utf-8"));archive=Path(delta["selection_archive_path"])
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=delta["selection_archive_sha256"]:
            raise ValueError("affected current association archive differs")
        if delta["base_model_revision"]!=prepared.manifest["prepared_against_model_revision"] or delta["restore_count"]!=0:
            raise ValueError("affected current association model/action differs")
        with np.load(archive,allow_pickle=False) as z:changed={k:np.array(z[k],copy=True) for k in z.files}
        if {_text(v) for v in changed["observation_id"]}!={v["stable_id"] for v in records} or len(changed["observation_id"])!=len(direct):
            raise ValueError("archive and current detailed review source IDs differ")
        archive_rows=np.asarray([lookup[_text(v)] for v in changed["observation_id"]],dtype=np.int64)
        if len(archive_rows)!=delta["retirement_count"] or np.any(changed["action"]!=0):raise ValueError("affected archive count/action differs")
        current_columns=_columns(load_import(prepared.manifest["source_observation_manifest"]),data_root)["s01.raw"]
        for name,current in (("source_id",current_columns["source_id"]),("source_dataset_id",current_columns["source_dataset_id"]),
            ("native_pts",a["native_pts"]),("immutable_uv",a["immutable_uv"]),("original_raw_row",a["original_row"]),
            ("existing_exclusion_bits",a["exclusion_bits"]),("station_seed_unchanged",a["station_seed"])):
            if not np.array_equal(current[archive_rows],changed[name]):raise ValueError("affected archive current "+name+" differs")
        if not np.array_equal(prepared.bound_curve_ids(archive_rows),np.asarray([_text(v) for v in changed["previous_curve_id"]])):
            raise ValueError("affected archive current finite branch differs")
    if audit_report_path is not None:
        audit=json.loads(Path(audit_report_path).read_text(encoding="utf-8"));audit_summary=audit["specific_delta"]
        if audit_summary["excluded_candidate_count"]!=len(direct) or audit_summary["restore_count"]!=0:
            raise ValueError("current audit's specific source action differs")
        if delta is not None and audit_summary["sha256"]!=delta["selection_archive_sha256"]:
            raise ValueError("current audit and changed archive differ")
    meta=copy.deepcopy(prepared.manifest);rejected=np.zeros(prepared.row_count,dtype=bool);rejected[direct]=True
    withdrawn=[];collateral=[]
    for i,g in enumerate(meta["groups"]):
        lo,hi=a["group_member_offsets"][i:i+2];members=a["group_member_rows"][lo:hi]
        if not np.any(rejected[members]):continue
        g["exclusion_reason_union"]=sorted(set(g.get("exclusion_reason_union",[])+[REASONS[reason_bit]]))
        if g["rank"]>0:
            collateral.extend(int(v) for v in members[~rejected[members]])
            rejected[members]=True;g["active_statistical"]=False;withdrawn.append(g["id"])
            g["specific_association_exclusion_reason"]="complete original W withdrawn; original matrix and source records retained"
    factors=[];matrix_rows=[];cursor=0;independent_removed=[]
    for f in meta["statistical_factors"]:
        if np.any(rejected[f["rows"]]):
            if f["kind"]=="original_independent_scalar_sigma":independent_removed.extend(f["rows"])
            continue
        start,count=f["residual_start"],f["residual_count"];matrix_rows.extend(range(start,start+count));f["residual_start"]=cursor;cursor+=count;factors.append(f)
    whitening=prepared.whitening[matrix_rows,:].tocsr();meta["statistical_factors"]=factors
    changes={"exclusion_bits":np.array(a["exclusion_bits"],copy=True)};changes["exclusion_bits"][rejected]|=reason_bit
    for name in ("active_independent_rows","budget_ready_rows","budget_candidate_rows","budget_pending_binding_rows",
        "unknown_noise_budget_ready_rows","unknown_noise_budget_candidate_rows","unknown_noise_budget_pending_binding_rows"):
        candidate=a[name];selected=candidate[~rejected[candidate]]
        if len(candidate)!=len(selected):changes[name]=selected
    query=a["active_query_rows"];keep=~rejected[query]
    if not np.all(keep):changes.update(active_query_rows=query[keep],active_query_indices=a["active_query_indices"][keep])
    changes.update({direct_name:np.sort(direct),collateral_name:np.asarray(sorted(set(collateral)),dtype=np.int64),
        independent_name:np.asarray(sorted(independent_removed),dtype=np.int64)})
    meta.update(generation=int(meta["generation"])+1,previous_current_preparation_manifest=str(prepared.directory/"manifest.json"))
    meta[scope+"_current_specific_association_detachments"]={"review_path":str(Path(review_path).resolve()),"reason_bit":reason_bit,
            "direct_count":len(direct),"direct_rows_array":direct_name,"direct_records":records,
            "withdrawn_original_W_groups":withdrawn,"W_collateral_rows_array":collateral_name,"W_collateral_count":len(set(collateral)),
            "removed_independent_rows_array":independent_name,"removed_independent_count":len(independent_removed),
            "possible_named_glyphs_not_rebound":True,"material_wide_propagation":False,
            "readmitting_requires_corrected_source_finite_branch_association":True,"selection_enabled_cannot_override_admission":True,
            "raw_UV_PTS_IDs_or_original_W_matrices_modified":False,"source_confirmation_revoked":False,
            "source_production_or_qualification_replayed":False,"model_revision_or_state_modified":False,
            "source_delta_path":str(Path(delta_path).resolve()) if delta_path else None,
            "source_archive_sha256":delta["selection_archive_sha256"] if delta else None,
            "current_audit_report_path":str(Path(audit_report_path).resolve()) if audit_report_path else None,
            "current_audit_specific_summary":audit_summary,"fixed_corner_candidates_installed":False,"HD_geometry_added":False}
    meta["reason_bits"][str(reason_bit)]=REASONS[reason_bit]
    out.mkdir(parents=True,exist_ok=True);refs=[d for d in meta["array_refs"] if d["id"].removeprefix("preparation.") not in changes]
    for name,value in changes.items():
        value=np.ascontiguousarray(value);path=out/(name+".npy");np.save(path,value,allow_pickle=False)
        refs.append(S.to_json_value(_incremental_ref(name,path,value,meta["generation"],prepared.manifest["array_refs"])))
    meta["array_refs"]=refs;save_npz(out/"statistical_whitening.npz",whitening,compressed=False);meta["whitening_path"]=str(out/"statistical_whitening.npz")
    report=meta["report"];report.update(engineering_budget_ready_rows=len(changes.get("budget_ready_rows",a["budget_ready_rows"])),
        unknown_noise_budget_ready_rows=len(changes.get("unknown_noise_budget_ready_rows",a["unknown_noise_budget_ready_rows"])),
        active_independent_sources=len(changes.get("active_independent_rows",a["active_independent_rows"])),
        active_original_W_groups=sum(g["active_statistical"] for g in meta["groups"]),
        active_joint_W_residual_rows=sum(g["rank"] for g in meta["groups"] if g["active_statistical"]),
        statistical_residual_rows=cursor,statistical_nonzero_coefficients=whitening.nnz,
        **{scope+"_specific_source_associations_detached":len(direct),scope+"_current_whole_W_groups_withdrawn":len(withdrawn),
            scope+"_current_W_collateral_count":len(set(collateral)),scope+"_current_independent_statistics_removed":len(independent_removed),
            scope+"_current_scope_restored_sources":0})
    report["exclusion_counts"][REASONS[reason_bit]]=int(np.sum((changes["exclusion_bits"]&reason_bit)!=0))
    _json(out/"manifest.json",meta);_json(out/"report.json",report)
    return load_prepared(out,data_root=data_root)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--observations",required=True);p.add_argument("--model",required=True)
    p.add_argument("--d78",required=True);p.add_argument("--out",required=True);p.add_argument("--data-root",required=True);p.add_argument("--constraints");a=p.parse_args(argv)
    from .model import load_model_bundle
    result=prepare_s01(a.observations,load_model_bundle(a.model),a.d78,a.out,data_root=a.data_root,constraint_ledger=a.constraints)
    print(json.dumps(result.manifest["report"],ensure_ascii=False))


if __name__=="__main__":
    main()
