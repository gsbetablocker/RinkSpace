"""Fit-checkpoint feedback orchestration, with no solver or visual oracle.

Queries are not recorded as successful consumption. Verified review declarations
can generate a bounded advisory episode automatically; only a live source reader
and observation transaction can change the actual project measurement state.
"""
from __future__ import annotations

from .core import Store, ValidationError, _hash, _sha, normalize_source, source_key


ACTIONS = {
    "measurement_error": "propose_native_remeasurement",
    "operator_error": "propose_role_corrected_native_reread",
    "identity_error": "propose_source_supported_identity_revision",
    "coverage_gap": "propose_measurement_of_current_visible_material",
    "model_mismatch": "retain_source_UV_and_diagnose_model",
    "unresolved": "keep_unknown",
}

RECIPES = {
    "material_flanks": {
        "trigger_tags": ["ice_contact", "board_foot", "yellow_strip", "material_boundary"],
        "purpose": "Compare both sides of nearby interfaces; select the registered physical material boundary, including the competing stronger edge.",
        "existing_entry": "vision_source_fit.reread_original_fit_source_frame / registered board material profile reader",
    },
    "finite_width_center": {
        "trigger_tags": ["paint_center", "C0", "A1", "A2", "A3", "A4", "measurement_operator"],
        "purpose": "Resolve the current physical operator; measure both paint flanks and the finite-width center where the definition requires a centerline.",
        "existing_entry": "vision_ice_source_role_override.effective_source_role_inventory / remeasure_confirmed_source_roles",
    },
    "weak_trace_native_relocation": {
        "trigger_tags": ["weak_trace", "F3", "outer_ring", "low_contrast", "blur"],
        "purpose": "Use identity and the coarse trace only to search; retain uncertainty and independently locate current native-pixel material support.",
        "existing_entry": "IceProgrammeSourceReader.read / OriginalFITSourceReader.read",
    },
    "distinct_installation": {
        "trigger_tags": ["advertisement", "same_template", "identity_conflict", "physical_instance"],
        "purpose": "Compare contextual neighbours and alternative physical installations; printed-template recognition does not decide installation identity.",
        "existing_entry": "vision_instances.advertisement_identity_from_constraints / transfer_source_context_identities",
    },
    "native_recheck": {
        "trigger_tags": [],
        "purpose": "Check source binding, class, physical instance, operator, visibility and position independently; retain good observations when the model disagrees.",
        "existing_entry": "vision_source_fit.OriginalFITSourceReader.read",
    },
}


def dependency_fingerprint(record):
    """Only a complete explicit dependency declaration permits review reuse."""
    deps = record.get("review_dependencies")
    fields = ("source_binding_sha256", "observation_snapshot_sha256", "reader_code_sha256",
              "reader_config_sha256", "effective_facts_sha256", "navigation_state_sha256")
    if not isinstance(deps, dict) or not all(name in deps for name in fields):
        return None
    try:
        for name in fields:
            if name == "navigation_state_sha256" and deps[name] == "not_used":
                continue
            _sha(deps[name], name)
    except ValidationError:
        return None
    return _hash(deps)


def same_review_subject(left, right):
    left_ids = sorted(set(left.get("observation_ids", [])))
    right_ids = sorted(set(right.get("observation_ids", [])))
    if left_ids or right_ids:
        return bool(left_ids) and left_ids == right_ids
    a, b = left.get("review_subject_sha256"), right.get("review_subject_sha256")
    try:
        return _sha(a, "review_subject_sha256") == _sha(b, "review_subject_sha256")
    except ValidationError:
        return False


class FeedbackCycle:
    def __init__(self, store: Store):
        self.store = store

    def before_read(self, context: dict) -> dict:
        """Prepare candidates; this is retrieval, not proof they were used."""
        context = dict(context)
        context["source"] = normalize_source(context.get("source"))
        context.setdefault("include_related_cases", True)
        matches = self.store.query(context)
        tags = set(context.get("tags") or context.get("context_tags") or [])
        for item in matches:
            tags.update(item["matched_tags"])
            if item["match_status"] == "applicable":
                tags.update(item["case"]["tags"])
        recipes = [name for name, spec in RECIPES.items() if name != "native_recheck" and tags.intersection(spec["trigger_tags"])]
        recipes.append("native_recheck")
        payload = {
            "context": context,
            "cases": matches,
            "recipe_ids": recipes,
            "recipes": {name: RECIPES[name] for name in recipes},
            "retrieval_only": True,
            "case_use_recorded": False,
            "current_pixel_measurement_performed": False,
            "fit_mutated": False,
        }
        payload["read_ticket_id"] = "read_" + _hash(payload)[:24]
        return payload

    def checkpoint(self, run_id: str, findings: list[dict]) -> dict:
        """Convert already-selected diagnostic signals into traceable jobs.

        Thresholds and representative selection belong to the live project and
        its actual measurement budget. This method invents neither pixel gates
        nor iteration quotas, and never interprets residual as a truth label.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValidationError("checkpoint requires a run_id")
        tasks = []
        for incoming in findings:
            item = dict(incoming)
            item.setdefault("run_id", run_id)
            if item["run_id"] != run_id:
                raise ValidationError("finding run_id conflicts with checkpoint run_id")
            saved = self.store.record_finding(item)
            context = {
                "source": saved["source"],
                "event_id": saved.get("event_id"),
                "programme_id": saved.get("programme_id"),
                "camera_id": saved.get("camera_id"),
                "entity_ids": saved.get("entity_ids", []),
                "tags": saved.get("context_tags", []),
                "include_related_cases": True,
                "split_context": saved.get("split_context", {}),
            }
            ticket = self.before_read(context)
            ticket.update({
                "finding_id": saved["finding_id"],
                "group_id": saved["group_id"],
                "group_occurrences": saved["group_occurrences"],
                "exact_finding_repeated": saved["idempotent"],
                "recurrence_requires_current_source_check": saved["group_occurrences"] > 1,
                "signal_type": saved["signal_type"],
                "observation_ids": saved.get("observation_ids", []),
                "run_id": run_id,
                "next_action": "source_review_before_any_observation_revision",
            })
            previous = self.store.resolutions_for_source_group(saved["source"], saved["group_id"])
            ticket["previous_resolution_refs"] = [{"resolution_id": r["resolution_id"], "content_sha256": r["content_sha256"], "verified_dimensions": r.get("verified_dimensions", [])} for r in previous]
            current_deps = dependency_fingerprint(saved)
            reusable = next((r for r in reversed(previous)
                             if current_deps and dependency_fingerprint(r) == current_deps
                             and r.get("source_evidence_present") is True
                             and same_review_subject(self.store.get_finding(r["finding_id"]), saved)), None)
            ticket["source_review_required"] = reusable is None
            if reusable:
                ticket["reuse_resolution_id"] = reusable["resolution_id"]
                ticket["next_action"] = "reuse_exact_source_review_if_evidence_artifact_hashes_still_match; recheck_current_admission"
            elif previous:
                ticket["reuse_blocker"] = "missing_or_changed_review_dependencies_or_native_evidence"
            tasks.append(ticket)
        groups = {}
        for task in tasks:
            groups.setdefault(task["group_id"], []).append(task["finding_id"])
        return {
            "run_id": run_id,
            "tasks": tasks,
            "groups": groups,
            "grouping_is_not_root_cause_proof": True,
            "source_review_executed": False,
            "fit_mutated": False,
        }

    def acknowledge_use(self, ticket: dict, review: dict) -> list[dict]:
        """Called by the reader AFTER it used a case and produced a receipt.

        A ticket alone does not suffice. The caller supplies exactly the cases
        it actually used and the tool receipt; metadata checks do not establish
        that its visual interpretation was correct.
        """
        context = ticket["context"]
        if source_key(review.get("source")) != source_key(context["source"]):
            raise ValidationError("reader receipt source differs from retrieval ticket")
        if not review.get("tool_evidence_refs"):
            raise ValidationError("case consumption requires a reader/tool receipt")
        supplied = review.get("used_case_refs")
        if not isinstance(supplied, list):
            raise ValidationError("used_case_refs must list actually consumed query references")
        available = {(x["case_id"], x["version"], x["content_sha256"]): x for x in ticket["cases"]}
        output = []
        for ref in supplied:
            key = (ref.get("case_id"), ref.get("version"), ref.get("content_sha256"))
            if key not in available:
                raise ValidationError("used case is not a version retrieved in this ticket")
            use = {
                **context,
                "run_id": review.get("run_id", ticket.get("run_id")),
                "case_id": key[0],
                "version": key[1],
                "content_sha256": key[2],
                "read_ticket_id": ticket["read_ticket_id"],
                "finding_id": ticket.get("finding_id"),
                "observation_ids": review.get("observation_ids", ticket.get("observation_ids", [])),
                "tool_evidence_refs": review["tool_evidence_refs"],
                "outcome": review.get("outcome", "reader_used_case; visual_quality_unassessed"),
                "use_kind": review.get("use_kind", "source_review_guidance"),
            }
            output.append(self.store.record_use(use))
        return output

    def learn(self, resolution: dict, lesson: dict | None = None) -> dict:
        """Persist a review and automatically build a bounded retrieval case.

        Automatic activation means an advisory case becomes searchable. It is
        not formal observation promotion, training admission or a new event fact.
        """
        resolution = dict(resolution)
        lesson = dict(lesson or {})
        if resolution.get("learning_lesson") is not None and resolution["learning_lesson"] != lesson:
            raise ValidationError("learning_lesson conflicts with supplied lesson")
        resolution["learning_lesson"] = lesson
        resolved = self.store.record_resolution(resolution)
        kind = resolved["resolution_type"]
        output = {"resolution": resolved, "observation_action": ACTIONS[kind], "learned_case": None,
                  "observation_transaction_applied": False, "training_eligible": False,
                  "visual_quality_verified_by_this_package": False}
        supported = resolved.get("source_evidence_supported") is True
        dimensions = resolved.get("verified_dimensions", [])
        if resolved.get("source_evidence_present") is not True or not dimensions:
            output["observation_action"] = "keep_unknown"
            output["reason"] = "No dimension has declared current-source support; retain discovery without manufacturing a learned truth."
            return output
        finding = self.store.get_finding(resolved["finding_id"])
        if not supported:
            output["observation_action"] = "keep_unknown"
        scope = {
            "event_id": finding.get("event_id"),
            "source_video_sha256": resolved["source"]["video_sha256"],
            "camera_id": finding.get("camera_id"),
            "entity_ids": finding.get("entity_ids", []),
        }
        tags = sorted(set(finding.get("context_tags", []) + [kind, "source_recheck"]))
        semantic_claims = {name: {k: v for k, v in resolved.get("checks", {}).get(name, {}).items()
                                  if k not in {"evidence_refs", "source", "evidence_id"}}
                           for name in dimensions if name != "position"}
        pattern = {"scope": scope, "resolution_type": kind, "verified_dimensions": dimensions,
                   "semantic_claims": semantic_claims, "tags": tags, "lesson": lesson}
        case_id = "lesson_" + _hash(pattern)[:24]
        old_case = self.store.get_case(case_id)
        if old_case is not None:
            if old_case["status"] != "active":
                output["reason"] = "Matching lesson was retracted; retain new evidence for review without automatic reactivation."
                return output
            output["learned_case"] = {"case_id": case_id, "version": old_case["version"], "content_sha256": old_case["content_sha256"], "idempotent": True}
            output["new_case_created"] = False
            output["reinforced_by_resolution_id"] = resolved["resolution_id"]
            output["reason"] = "Same scoped lesson already exists; this source occurrence remains in resolution history, not another prompt patch."
            return output
        # Stable lesson identity coalesces equivalent scoped lessons. Individual
        # source occurrences remain immutable resolution records. Case activation
        # permits advisory retrieval only, with the first episode as an exemplar.
        case = {
            "case_id": case_id,
            "version": 1,
            "status": "active",
            "kind": "recipe",
            "title": lesson.get("summary") or f"{kind}: {', '.join(finding.get('entity_ids', [])) or 'source episode'}",
            "scope": scope,
            "tags": tags,
            "assertions": {
                "episode_source": resolved["source"],
                "episode_resolution_type": kind,
                "next_matching_frame_action": ACTIONS[kind] if supported else "reread_unresolved_dimensions_keep_partial_claims",
                "verified_dimensions": dimensions,
                "unresolved_dimensions": resolved.get("unresolved_dimensions", []),
                "dimension_claims": {name: resolved.get("checks", {}).get(name, {"status": "unknown"}) for name in dimensions},
                "current_frame_native_reread_required": True,
                "transfer_pixel_coordinates_to_new_frame": False,
                "transfer_instance_identity_without_current_context": False,
                "generalization_status": "source_scoped_advisory_episode_only",
                "position_truth_for_future_frame": "unknown_until_current_native_review",
                "lesson": lesson,
                "regression_request": {
                    "source": resolved["source"],
                    "expected_action": ACTIONS[kind] if supported else "preserve_verified_dimensions_without_inventing_position",
                    "visual_regression_executed": False,
                },
            },
            "evidence_refs": resolved.get("evidence_refs", []),
            "provenance": {
                "authority": "source_review_declared_by_caller; not_user_confirmation",
                "finding_id": finding["finding_id"],
                "resolution_id": resolved["resolution_id"],
                "resolution_sha256": resolved["content_sha256"],
                "created_by": "FeedbackCycle.learn",
                "evidence_assurance": resolved["evidence_assurance"],
            },
            "lineage": {"source_video_sha256": resolved["source"]["video_sha256"], "source_key": source_key(resolved["source"])},
            "split_role": "synthetic_mechanics" if finding.get("simulation") else "development",
            "training_eligible": False,
            "fit_admissible": False,
        }
        output["learned_case"] = self.store.upsert_case(case)
        output["new_case_created"] = True
        return output
