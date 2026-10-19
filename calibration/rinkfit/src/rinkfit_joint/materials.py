"""Operator-aware identities and source-priority decisions, without fitting pixels.

Typed source names propose shared entities. A numerical material index from an
old shot model is never a joint identity. Neither a name match nor a small
reprojection residual is an automatic proof of physical identity.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import re
from pathlib import Path
from functools import lru_cache
import math
from .io_utils import object_sha256, file_sha256, read_json
from .measurement_contract import EventMeasurementContract

@lru_cache(maxsize=1)
def _measurement_contract():
    return EventMeasurementContract(Path(__file__).resolve().parents[4], 'ankara_2026')


def measurement_definition(row):
    target, category = row.get("target_id"), row.get("category")
    raw = row.get("measurement_operator")
    contract=_measurement_contract();declared=contract.expected(target,category)
    expected=declared.get('operator') if declared else None
    authority=declared.get('authority') if declared else 'physical_definition_not_closed_against_existing_user_facts'
    # A missing categorical operator can be filled only by the rules above.
    # An explicitly conflicting ice operator is never silently rewritten.
    conflict = bool(raw and expected and raw != expected) or bool(declared and declared.get('prohibited_as_board_boundary'))
    operator = raw or expected or "unspecified"
    return {"raw_operator":raw,"operator":operator,"expected_operator":expected,
            "authority":authority,"conflict":conflict,"measurement_contract_sha256":contract.digest,
            "expected_physical_definition":declared,"definition_match_is_not_actual_RGB_proof":True}


def semantic_key(row):
    if not row.get("target_id"): return None
    definition = measurement_definition(row)
    return "|".join((row["target_id"], str(row.get("physical_surface") or "unspecified"),
                     definition["operator"], str(row.get("boundary_role") or "unspecified")))


def build_registry(rows, reference_model, reference_model_sha256):
    rows = list(rows)
    old = {m["label"]:i for i,m in enumerate(reference_model["materials"])}
    grouped = defaultdict(list)
    for row in rows:
        key = semantic_key(row)
        if key: grouped[key].append(row)
    materials, conflicts = [], []
    for key, members in sorted(grouped.items()):
        row = members[0]; target = row["target_id"]
        definition = measurement_definition(row)
        category = row.get("category")
        entity = target.split(".print.")[0] if category == "ad_print" else (
                 target.split(".")[0] if target.startswith("Ad") else target.split(":")[0])
        local_index = old.get(target)
        paths = [] if local_index is None else reference_model["materials"][local_index]["paths"]
        dependency_names = set()
        for path in paths:
            for field in ("geometry_block","advertisement_block"):
                index = path.get(field,-1)
                if index >= 0: dependency_names.add(reference_model["blocks"][index]["name"])
        conflict = definition["conflict"]
        if conflict: conflicts.append({"material_id":key,"rows":len(members),**definition})
        materials.append({"id":key,"entity_id":entity,"source_target":target,
            "category":category,"surface":row.get("physical_surface"),"measurement":definition,
            "point_correspondence":bool(row.get("point_correspondence")),
            "row_count":len(members),"shots":sorted({r["shot_id"] for r in members}),
            "rigs":sorted({r["rig_id"] for r in members}),
            "seed_match":None if local_index is None else {
                "model_sha256":reference_model_sha256,"model_local_material_index":local_index,
                "path_ids":[p["id"] for p in paths],"parameter_block_names":sorted(dependency_names)},
            "shared_geometry_status":"operator_conflict_requires_source_remeasurement" if conflict else (
                "reference_label_candidate_requires_semantic_replay" if local_index is not None
                else "requires_automatic_crosswalk_or_new_shared_geometry"),
            "identity_accepted":False,"geometry_frozen":False,
            "initialisation_method":("continuous_board_chart_and_multiview_print_identity" if category == "ad_print"
                else "shared_unwrapped_poster_rectangle" if category in {"ad_edge","ad_corner"}
                else "typed_ice_or_board_source_geometry")})
    connectivity = []
    for rig in ("main","secondary","end"):
        connectivity.append({"rig_id":rig,"materials":sum(rig in m["rigs"] for m in materials),
            "cross_rig_material_candidates":sum(rig in m["rigs"] and len(m["rigs"])>1 for m in materials)})
    return {"schema":"rinkfit.operator_material_registry.v1","materials":materials,
        "summary":{"typed_material_variants":len(materials),
            "unique_named_targets":len({m["source_target"] for m in materials}),
            "reference_label_matches":len({m["source_target"] for m in materials if m["seed_match"]}),
            "operator_conflict_variants":len(conflicts),"operator_conflict_rows":sum(c["rows"] for c in conflicts),
            "unbound_rows":sum(not r.get("target_id") for r in rows),"connectivity":connectivity},
        "conflicts":conflicts,"semantic_rules_version":"okada_joint9_20261018_v1",
        "warning":"reference matching is an initialisation index, not automatic measurement/geometry acceptance"}


def plan_admission(rows, groups, registry):
    """Executable source triage. Final promotion requires actual evidence receipts.

    This separates preserved historical user_accepted/allowed_for_fit fields
    from the new user-authorised automated review. It does not require the user
    to click 201 frames, and does not claim unimplemented pixel QA has passed.
    """
    materials = {m["id"]:m for m in registry["materials"]}
    group_index = {g["group_id"]:g for g in groups}
    counts = Counter(); reasons = Counter(); output = []
    tasks = defaultdict(lambda:{"row_count":0,"row_ids":[],"shot_ids":set(),"actions":set()})
    for row in rows:
        key = semantic_key(row); g = group_index.get(row["provenance_group_id"],{})
        definition = measurement_definition(row)
        flags = row.get("qa_flags",[])
        why, actions = [], set()
        if row.get("inside_supplied_person_mask"):
            why.append("source_person_mask_overlap");actions.add("recheck_source_occlusion_and_mask")
        if "exact_duplicate_same_target_uv" in flags:
            why.append("duplicate_observation_not_independent");actions.add("deduplicate_statistical_membership_keep_provenance")
        if not key:
            why.append("unbound_physical_identity");actions.add("resolve_static_ice_or_board_identity_from_source")
        if row.get("physical_surface") not in {"ice","board"}:
            why.append("physical_surface_not_in_fit_scope");actions.add("resolve_surface_without_free_depth")
        if definition["conflict"] or definition["operator"] == "unspecified":
            why.append("measurement_operator_conflict_or_missing");actions.add("remeasure_correct_physical_operator")
        if row.get("manual_authoritative"):
            actions.add("preserve_final_manual_visible_intervals_and_deletions")
        if row.get("evidence_class") in {"derived_source_curve","shared_source_contour"}:
            actions.add("fit_from_raw_support_or_reconstruct_group_measurement_covariance")
        feature = group_index.get(g.get("feature_source_group_id"),{})
        if feature.get("joint_group_navigation"):
            actions.add("remove_old_geometry_ratio_from_measurement_generator")
        if key and not materials[key].get("seed_match"):
            actions.add("crosswalk_legacy_typed_template_into_one_shared_scene")
        actions.add("native_pixel_shape_polarity_and_temporal_identity_review")
        actions.add("build_numeric_source_group_statistics")
        status = "auto_resolution_required" if why else "auto_registration_candidate"
        result = {"source_row_id":row["source_row_id"],"shot_id":row["shot_id"],
            "native_pts":row["native_pts"],"joint_material_id":key,
            "decision":status,"reasons":why,"next_actions":sorted(actions),
            "original_allowed_for_fit":row.get("original_allowed_for_fit"),
            "source_user_accepted":row.get("source_user_accepted"),
            "solver_eligible":False,"requires_frame_by_frame_user_acceptance":False}
        counts[status]+=1;reasons.update(why);output.append(result)
        # Action units are physical materials/source groups, not one question per point.
        task_key = key or f"unbound/{row['shot_id']}/{row.get('unbound_candidate_id') or row['provenance_group_id']}"
        job = tasks[task_key];job["row_count"]+=1;job["row_ids"].append(row["source_row_id"])
        job["shot_ids"].add(row["shot_id"]);job["actions"].update(actions)
    jobs=[{"id":object_sha256(k)[:20],"material_or_candidate":k,"row_count":v["row_count"],
        "source_row_ids":v["row_ids"],"shot_ids":sorted(v["shot_ids"]),"actions":sorted(v["actions"]),
        "owner":"Codex automated source audit", "human_review":"only unresolved ambiguous physical identity after automatic retries"}
          for k,v in sorted(tasks.items())]
    return output,jobs,{"schema":"rinkfit.auto_admission_plan.v1","row_counts":dict(counts),
        "reason_counts":dict(reasons),"automatic_job_count":len(jobs),"solver_eligible_rows":0,
        "source_flags_modified":False,"manual_frame_acceptance_required":False,
        "current_phase":"structural triage; source pixel and numeric likelihood receipts required before solve"}


class PromotionArtifactVerifier:
    """Hash shared immutable artifacts once, then rehash before batch release.

    No promotion output is committed until verify_unchanged succeeds.  This
    avoids reading the same 1920x1080 PNG once per derived representative.
    """
    def __init__(self, root):
        self.root=Path(root).resolve();self.hashes={};self.closed=False;self.source_definition_documents={}
        self.measurement_contract=_measurement_contract();self.measurement_contract.verify_unchanged()
    def digest(self, path):
        path=Path(path).resolve()
        if self.closed or self.root not in path.parents: raise ValueError("Invalid promotion artifact batch")
        if path not in self.hashes:self.hashes[path]=file_sha256(path)
        return self.hashes[path]
    def verify_unchanged(self):
        self.measurement_contract.verify_unchanged()
        for path,digest in self.hashes.items():
            if file_sha256(path)!=digest:raise ValueError(f"Promotion artifact changed during batch: {path}")
        self.closed=True
        return {str(k.relative_to(self.root)):v for k,v in self.hashes.items()}

    def source_definition_document(self,path):
        path=Path(path).resolve()
        if path not in self.source_definition_documents:
            if len(self.source_definition_documents)>=2:self.source_definition_documents.pop(next(iter(self.source_definition_documents)))
            self.source_definition_documents[path]=read_json(path)
        return self.source_definition_documents[path]



def current_physical_source_artifact_verified(row, certificate, evidence_root, artifact_verifier=None):
    contract=_measurement_contract();root=Path(evidence_root).resolve();proofs=certificate.get('source_evidence') or []
    actual_definition_bound=False
    for reference in proofs:
        if not isinstance(reference,dict):continue
        candidate=(root/reference.get('path','')).resolve()
        if candidate.suffix.lower()!='.json' or root not in candidate.parents or not candidate.is_file():continue
        try:doc=artifact_verifier.source_definition_document(candidate) if isinstance(artifact_verifier,PromotionArtifactVerifier) else read_json(candidate)
        except (ValueError,OSError):continue
        if doc.get('native_pts')!=row.get('native_pts') or doc.get('measurement_contract_sha256')!=contract.digest:continue
        matches=[q for q in doc.get('actual_outer_source_rows',doc.get('qualified_source_rows',[])) if q.get('source_row_id')==row.get('source_row_id') and q.get('current_uv')==row.get('current_uv') and q.get('joint_material_id')==row.get('joint_material_id') and q.get('measurement_operator',q.get('measurement_operator_resolved'))==certificate.get('operator')]
        for q in matches:
            if q.get('pixels_from_model') is not False:continue
            if certificate.get('operator') in ('board_upper_material_edge','ice_board_interface'):
                ca=q.get('source_measurement_audit') or {};ci=q.get('source_definition_certificate_index',-1);cs=doc.get('actual_outer_material_certificates',[])
                if not isinstance(ci,int) or not 0<=ci<len(cs):continue
                source=cs[ci];config=source.get('configuration',{})
                if (ca.get('actual_outer_material_order_verified') is True and ca.get('reflection_excluded') is True and ca.get('internal_interface_used') is False and config.get('internal_band_body_interfaces_used_as_measurements') is False and config.get('model_predictions_used') is False and source.get('measurement_contract_sha256')==contract.digest):actual_definition_bound=True
            elif q.get('finite_physical_identity_source_verified') is True and q.get('source_local_material_operator_verified') is True:
                actual_definition_bound=True
    return actual_definition_bound


def apply_promotion(row, decision, receipt, *, video_sha256, policy_sha256,
                    material_registry=None, statistics_registry=None, locked_roles=None,
                    evidence_root=None, decoded_frame_registry=None, artifact_verifier=None):
    """Strict, reusable promotion gate for the Codex pixel/identity/statistics worker.

    High residual alone is not grounds to drop a source observation. A failed
    promotion stays in an explicit unresolved ledger and remains visible in
    coverage/denominator reporting.
    """
    required = ("native_pixel_review_passed","physical_identity_passed","operator_passed",
                "source_lineage_complete","numeric_statistics_ready","manual_authority_preserved")
    errors=[]
    contract=_measurement_contract();definition=contract.inspect(row);certificate=row.get('measurement_contract_certificate') or {}
    errors.extend(definition['definition_problems'])
    if receipt.get('measurement_contract_sha256')!=contract.digest or certificate.get('contract_sha256')!=contract.digest:
        errors.append('promotion_not_bound_to_existing_user_measurement_facts')
    expected=definition['expected']
    if (certificate.get('source_row_id')!=row.get('source_row_id') or certificate.get('native_pts')!=row.get('native_pts') or certificate.get('joint_material_id')!=row.get('joint_material_id')):
        errors.append('source_definition_certificate_row_binding_mismatch')
    if expected and (certificate.get('operator')!=expected.get('operator') or certificate.get('physical_role')!=expected.get('physical_role')):
        errors.append('source_definition_certificate_wrong_physical_transition')
    semantic_current=object_sha256(dict(native_pts=row.get('native_pts'),current_uv=row.get('current_uv'),source_row_id=row.get('source_row_id')))
    if certificate.get('current_measurement_sha256')!=semantic_current:errors.append('source_definition_certificate_not_for_current_mean')
    if certificate.get('actual_RGB_material_identity_verified') is not True or certificate.get('operator_selected_correct_material_transition') is not True or certificate.get('pixels_from_model') is not False:
        errors.append('actual_RGB_physical_source_definition_not_proved')
    if receipt.get('measurement_contract_certificate_sha256')!=object_sha256(certificate):errors.append('physical_source_definition_not_bound_to_promotion_receipt')
    proofs=certificate.get('source_evidence') or []
    if not proofs:errors.append('missing_actual_physical_source_definition_artifacts')
    bound={(a.get('path'),a.get('sha256')) for a in receipt.get('evidence_artifacts',[]) if isinstance(a,dict)}
    if any(not isinstance(a,dict) or (a.get('path'),a.get('sha256')) not in bound for a in proofs):errors.append('physical_source_definition_artifact_not_bound_to_verified_artifact_batch')
    if receipt.get("source_row_id") != row["source_row_id"]: errors.append("row_identity_mismatch")
    if receipt.get("source_packet_sha256") != row["source_packet_sha256"]: errors.append("source_packet_mismatch")
    expected_material=row.get("joint_material_id") or semantic_key(row)
    if receipt.get("joint_material_id")!=expected_material:errors.append("source_material_binding_mismatch")
    if receipt.get("video_sha256") != video_sha256: errors.append("source_video_mismatch")
    if receipt.get("policy_sha256") != policy_sha256: errors.append("review_policy_mismatch")
    if receipt.get("evidence_basis") not in {"current_native_source_remeasurement", "native_source_manual_click_preserved"}:
        errors.append("unsupported_or_circular_measurement_basis")
    expected_identity = object_sha256({"native_pts":row["native_pts"],"source_uv":row["source_uv"],
                                      "source_row_id":row["source_row_id"]})
    if receipt.get("source_measurement_sha256") != expected_identity: errors.append("source_measurement_mismatch")
    errors.extend(k for k in required if receipt.get(k) is not True)
    artifact_roles=set()
    artifacts = receipt.get("evidence_artifacts",[])
    frame_identity=(decoded_frame_registry or {}).get(f"{video_sha256}:{row['native_pts']}",{})
    if (frame_identity.get("source_video_sha256")!=video_sha256 or
            frame_identity.get("native_pts")!=row["native_pts"] or
            frame_identity.get("decode_sequence_verified") is not True):
        errors.append("native_frame_not_bound_to_verified_video_and_pts")
    artifact_roles=set()
    if not artifacts or evidence_root is None: errors.append("missing_verifiable_source_artifacts")
    else:
        root=Path(evidence_root).resolve()
        artifact_roles=set()
        for a in artifacts:
            if not isinstance(a,dict): errors.append("invalid_artifact_reference");continue
            candidate=(root / a.get("path","")).resolve()
            if root not in candidate.parents or not candidate.is_file() or candidate.is_symlink():
                errors.append("artifact_outside_evidence_root_or_missing");continue
            digest=(artifact_verifier.digest(candidate) if isinstance(artifact_verifier,PromotionArtifactVerifier)
                    and artifact_verifier.root==root else file_sha256(candidate))
            if digest != a.get("sha256"): errors.append("artifact_hash_mismatch")
            if a.get("role")=="source_policy" and digest!=policy_sha256:
                errors.append("policy_artifact_differs_from_bound_review_policy")
            if a.get("role")=="native_video" and digest!=video_sha256:
                errors.append("native_video_differs_from_bound_official_source")
            if a.get("role")=="native_decode_receipt" and digest!=frame_identity.get("native_decode_receipt_sha256"):
                errors.append("native_decode_receipt_differs_from_verified_decoder")
            if a.get("role")=="native_frame" and digest!=frame_identity.get("native_png_sha256"):
                errors.append("native_frame_differs_from_verified_decoder_receipt")
            artifact_roles.add(a.get("role"))
        frame_artifact_bound=("native_frame" in artifact_roles or {"native_video","native_decode_receipt"}.issubset(artifact_roles))
        if not frame_artifact_bound or "source_measurement_audit" not in artifact_roles:
            errors.append("missing_native_frame_or_measurement_audit")
        if "native_frame" not in artifact_roles and {"native_video","native_decode_receipt"}.issubset(artifact_roles):
            decode_binding=frame_identity.get("native_decode_row_sha256")
            if (not isinstance(frame_identity.get("decoded_native_BGR_sha256"),str) or len(frame_identity["decoded_native_BGR_sha256"])!=64
                    or not decode_binding or receipt.get("native_decode_row_sha256")!=decode_binding):
                errors.append("native_decode_row_not_bound_to_actual_source_pixels")
            if row.get("decoded_native_BGR_sha256")!=frame_identity.get("decoded_native_BGR_sha256"):
                errors.append("current_native_BGR_differs_from_bound_decoder_row")
            if "current_native_profiles" not in artifact_roles:
                errors.append("missing_current_native_profile_artifact")
        if not current_physical_source_artifact_verified(row,certificate,root,artifact_verifier):errors.append('physical_definition_certificate_not_supported_by_actual_current_source_artifact_rows')
    material=(material_registry or {}).get(receipt.get("joint_material_id"),{})
    if material.get("identity_accepted") is not True or material.get("surface") not in {"ice","board"}:
        errors.append("material_identity_not_accepted")
    if receipt.get("material_definition_sha256") != object_sha256(material): errors.append("material_definition_mismatch")
    role=receipt.get("fit_role")
    if role not in {"train","historical_fit","development","sealed_validation","validation_guard"}:
        errors.append("missing_locked_role")
    locked=(locked_roles or {}).get(row["source_row_id"],{})
    if not locked or locked.get("fit_role") != role or receipt.get("locked_role_sha256") != object_sha256(locked):
        errors.append("role_not_in_locked_assignment")
    statistics=(statistics_registry or {}).get(receipt.get("statistical_group_id"),{})
    source_hashes=statistics.get("source_measurements_sha256",{})
    if not isinstance(source_hashes,dict):source_hashes={}
    if (statistics.get("numeric_whitening_verified") is not True
            or row["source_row_id"] not in statistics.get("row_ids",[])
            or source_hashes.get(row["source_row_id"]) != expected_identity
            or receipt.get("statistics_definition_sha256") != object_sha256(statistics)):
        errors.append("numeric_statistics_not_verified_for_source_row")
    if "current_uv" in row:
        if "source_policy" not in artifact_roles:errors.append("missing_bound_source_policy_artifact")
        uv=row.get("current_uv")
        if not isinstance(uv,list) or len(uv)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in uv):
            errors.append("missing_finite_current_native_measurement")
        else:
            current_identity=object_sha256({"native_pts":row["native_pts"],"current_uv":uv,
                "source_row_id":row["source_row_id"],"measurement_operator":row.get("measurement_operator_resolved")})
            if receipt.get("current_measurement_sha256")!=current_identity:
                errors.append("current_measurement_receipt_mismatch")
            current_hashes=statistics.get("current_measurements_sha256",{})
            if not isinstance(current_hashes,dict):current_hashes={}
            if current_hashes.get(row["source_row_id"])!=current_identity:
                errors.append("statistics_current_native_measurement_mismatch")
    if not isinstance(artifact_verifier,PromotionArtifactVerifier):contract.verify_unchanged()
    allowed = not errors and role in {"train","historical_fit","development"}
    return {**decision,"decision":"auto_admitted" if not errors else "auto_resolution_required",
            "admission_status":"admitted" if not errors else "pending",
            "solver_eligible":allowed,"promotion_errors":errors,"fit_role":role,
            "physical_identity_passed":not errors and receipt.get('physical_identity_passed') is True,
            "current_native_source_support_passed":not errors and receipt.get('native_pixel_review_passed') is True,
            "joint_material_id":receipt.get("joint_material_id"),"receipt_sha256":object_sha256(receipt),
            "measurement_contract_sha256":contract.digest,"measurement_contract_certificate_sha256":object_sha256(certificate)}
