"""Production source feedback: query before pixels, acknowledge after the reader.

Cases constrain an implemented operator/check. They never supply future UV or
physical membership. Pixel interpretation and observation installation retain
separate receipts. All runtime evidence is outside the synchronized checkout.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import uuid

from rink_feedback.core import Store, normalize_source
from rink_feedback.cycle import FeedbackCycle
from rink_feedback.adapters import import_project_seed_bundle
from .storage import AssetStore

VERSION = "source-feedback-runtime.v1.2"
ROOT = Path(__file__).resolve().parents[2]
_SEEDS = {}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    from .path_policy import guard_project_write
    path = guard_project_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = guard_project_write(path.with_name(path.name + "." + uuid.uuid4().hex[:12] + ".tmp"))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)
    return dict(path=str(path.resolve()), sha256=file_sha(path))


def native_source(binding):
    """Map declared aliases, never convert a PTS from a different video clock."""
    shas = [binding[k] for k in ("video_sha256", "source_video_sha256", "source_sha256") if binding.get(k)]
    stamps = [binding[k] for k in ("pts", "source_pts", "native_pts") if binding.get(k) is not None]
    if not shas or len(set(shas)) != 1 or not stamps:
        raise ValueError("Feedback needs one unambiguous native source identity")
    if any(isinstance(v, bool) or not hasattr(v, "__index__") for v in stamps) or len(set(map(int, stamps))) != 1:
        raise ValueError("Feedback PTS aliases must agree as original integers")
    result = dict(video_sha256=shas[0], pts=int(stamps[0]), time_base=binding["time_base"],
        coordinate_domain=binding.get("coordinate_domain", "native_pixels"))
    if binding.get("native_hw") is not None:result["native_hw"] = list(binding["native_hw"])
    return normalize_source(result)


def resolve_scope(source, project=ROOT):
    """Resolve exact source SHA/clock from existing programme/event registries."""
    project = Path(project)
    matches = set()
    for path in (project / "events").glob("*/programmes/*/primary_source_selection.json"):
        data = json.loads(path.read_text("utf-8-sig"))
        # Each representation keeps its own hash. No equality or PTS transfer is inferred.
        for key in ("primary_source", "existing_verified_copy"):
            row = data.get(key, {})
            clock = row.get("time_base", data.get("primary_source", {}).get("time_base"))
            if row.get("sha256") == source["video_sha256"] and clock and Fraction(clock) == Fraction(source["time_base"]):
                matches.add((data["event_id"], data["programme_id"]))
    if len(matches) == 1:return next(iter(matches))
    if matches:raise ValueError("Source belongs to ambiguous programme feedback scopes")
    for path in (project / "events").glob("*/model_constraints/source_clock_registry.json"):
        data = json.loads(path.read_text("utf-8-sig"))
        for row in data.get("sources", []):
            if row.get("source_sha256") == source["video_sha256"] and Fraction(row["time_base"]) == Fraction(source["time_base"]):
                matches.add((data["event_id"], None))
    if len(matches) > 1:raise ValueError("Ambiguous event feedback source registry")
    return next(iter(matches)) if matches else (None, None)


@contextmanager
def feedback_store(project, event_id):
    if not isinstance(event_id, str) or not event_id or Path(event_id).name != event_id or event_id in (".", ".."):
        raise ValueError("Bounded event feedback identity required")
    assets = AssetStore(project)
    database = assets.data / "events" / event_id / "feedback" / "feedback.sqlite"
    if database.resolve().is_relative_to(Path(project).resolve()):
        raise ValueError("Feedback SQLite must remain on the registered local data drive")
    with Store(database) as store:
        manifest = Path(project) / "events" / event_id / "feedback" / "seed_manifest.json"
        if manifest.is_file():
            stat = manifest.stat()
            key = (str(database), str(manifest), stat.st_mtime_ns, stat.st_size)
            # Opening a replaced/rebuilt DB must restore its seeds too.
            if _SEEDS.get(key) != database.stat().st_ino:
                import_project_seed_bundle(store, manifest)
                _SEEDS[key] = database.stat().st_ino
        # Durable project knowledge can rebuild a deleted derived index. Do
        # not rewrite its content or reactivate a newer retraction.
        cases = [json.loads(path.read_text("utf-8-sig")) for path in
            (Path(project) / "events" / event_id / "feedback" / "learned").glob("*.json")]
        for case in sorted(cases, key=lambda row: (row['case_id'], row['version'])):
            store.upsert_case(case)
        from .feedback_outbox import restore_manual_cases, restore_installation_cases
        restore_manual_cases(store, project=project, event_id=event_id)
        restore_installation_cases(store, project=project, event_id=event_id)
        yield store


def authenticate_refs(refs, project=ROOT):
    """Check real evidence bytes; changed live authority is a recheck blocker."""
    for ref in refs:
        path = Path(ref["path"])
        if file_sha(path) != ref["sha256"]:raise ValueError("Feedback evidence bytes changed: " + str(path))
        original = ref.get("original_address_metadata_only")
        if original:
            q = Path(original)
            if q.is_relative_to(Path(project)) and q.is_file() and file_sha(q) != ref["sha256"]:
                raise ValueError("Live fact supersedes imported feedback evidence: " + str(q))


class SourceFeedbackRead:
    """One actual read ticket. The reader calls finish only after reading RGB."""
    def __init__(self, binding, *, entity_ids=(), tags=(), project=ROOT,
                 event_id=None, programme_id=None, run_id=None, split_context=None):
        self.project = Path(project).resolve()
        self.source = native_source(binding)
        resolved_event, resolved_programme = resolve_scope(self.source, self.project)
        if event_id and resolved_event and event_id != resolved_event:raise ValueError("Feedback event conflicts with source registry")
        self.event_id, self.programme_id = event_id or resolved_event, programme_id or resolved_programme
        self.run_id = run_id or "reader_" + uuid.uuid4().hex
        if Path(self.run_id).name != self.run_id or self.run_id in (".", "..") or any(c in self.run_id for c in '/\\:'):
            raise ValueError("Feedback run_id must be one bounded directory name")
        mode = binding.get('evaluation_role', binding.get('fit_role', 'unknown'))
        if mode == 'held_out':mode = 'heldout'
        if binding.get('training_role_verified') is True:mode = 'development'
        self.split_context = deepcopy(split_context or binding.get('split_context') or {'mode': mode})
        self.binding = deepcopy(binding)
        self.effects, self.used, self.blockers = [], {}, []
        self.ticket = None
        if not self.event_id:return
        context = dict(source=self.source, event_id=self.event_id, programme_id=self.programme_id,
            shot_id=binding.get("shot_id"), camera_id=binding.get("camera_id"),
            entity_ids=sorted(set(entity_ids)), tags=sorted(set(tags)), include_related_cases=True,
            split_context=self.split_context)
        with feedback_store(self.project, self.event_id) as store:
            self.ticket = FeedbackCycle(store).before_read(context)
        self.applicable = []
        for match in self.ticket["cases"]:
            if match["match_status"] != "applicable":
                # Learned recipes are intentionally advisory in the portable
                # core. A same-event/source procedure may guide this tool;
                # it still cannot become a transferred fact, instance or UV.
                case = match["case"]
                scope = case.get("scope", {})
                same_scope = (scope.get("event_id") == self.event_id and
                    scope.get("source_video_sha256") == self.source["video_sha256"] and
                    scope.get("camera_id") in (None, binding.get("camera_id")) and
                    bool(set(scope.get("entity_ids", [])) & set(entity_ids)))
                if not (case.get("kind") == "recipe" and same_scope and
                        case.get("assertions", {}).get("lesson", {}).get("reader_check") in {"ordered_board_material_flanks", "single_actual_stroke_two_flanks_topology",
                            "declared_target_material_interface"}):
                    continue
            try:authenticate_refs(match["case"].get("evidence_refs", []), self.project)
            except (OSError, ValueError) as error:
                self.blockers.append(dict(case_id=match["case_id"], reason=str(error)))
                continue
            self.applicable.append(match)

    def _use(self, match, action, **effect):
        self.used[match["case_id"]] = {k: match[k] for k in ("case_id", "version", "content_sha256")}
        self.effects.append(dict(case_id=match["case_id"], action=action, **effect))

    def apply_inventory(self, inventory):
        """Apply an authoritative finite-width operator; keep all current UV."""
        result = [dict(row) for row in inventory]
        for match in getattr(self, "applicable", []):
            claims = match["case"].get("assertions", {})
            roles = claims.get("physical_material_roles", {}).get("finite_width_circular_lines", {})
            if roles.get("measurement_operator") != "paint_center":continue
            for row in result:
                if row.get("feature_id") not in roles.get("ids", []):continue
                previous = row.get("operator")
                row.update(operator="paint_center", material_transition=None,
                    source_role_authority="authenticated_feedback_finite_width_midline",
                    source_role_evidence=deepcopy(match["case"]["evidence_refs"]))
                self._use(match, "finite_width_operator", curve_id=row["curve_id"],
                    previous_operator=previous, resolved_operator="paint_center", old_UV_transferred=False)
        return result

    def require_operator(self, entity_ids, operator):
        """Validate a registered FIT definition, withholding unsupported edits."""
        allowed = True
        for match in getattr(self, 'applicable', []):
            claims = match['case'].get('assertions', {})
            roles = claims.get('physical_material_roles', {}).get('finite_width_circular_lines', {})
            if roles.get('measurement_operator') != 'paint_center':continue
            if not set(entity_ids).intersection(match['case']['scope']['entity_ids']):continue
            correct = operator in ('paint_center', 'paired_material_boundary_midline')
            self._use(match, 'registered_FIT_operator_checked', requested_operator=operator,
                resolved_operator='paint_center', operator_supported=correct,
                conflicting_revision_withheld=not correct, old_UV_transferred=False)
            allowed = allowed and correct
        return allowed

    def use_current_reader_recipe(self, reader_check, *, postcondition):
        """Acknowledge a known advisory recipe after its current RGB check.

        This records a current read, never transfers the exemplar's identity,
        coordinates, visibility, or source qualification into another frame.
        """
        known = {"single_actual_stroke_two_flanks_topology",
                 "declared_target_material_interface"}
        if reader_check not in known:
            raise ValueError("Unregistered actual ice reader check")
        if (not isinstance(postcondition, dict)
                or postcondition.get("actual_current_RGB_read_executed") is not True
                or postcondition.get("source_UV_or_identity_transferred") is not False
                or postcondition.get("pixels_from_model") is not False):
            raise ValueError("Recipe consumption needs executed current-source checks")
        consumed = []
        for match in getattr(self, "applicable", []):
            case = match["case"]
            if (case.get("kind") != "recipe"
                    or case.get("assertions", {}).get("lesson", {}).get("reader_check") != reader_check):
                continue
            self._use(match, reader_check, **postcondition,
                advisory_guidance_only=True, source_qualification_granted=False)
            consumed.append(match["case_id"])
        return consumed

    def material_authority(self, authority):
        """Bind the stored board-material distinction to the existing profile tool."""
        result = dict(authority)
        for match in getattr(self, "applicable", []):
            roles = match["case"].get("assertions", {}).get("material_roles", {})
            if roles.get("bottom_strip_region_belongs_to") == "board" and roles.get("primary_boundary") == "visible_lower_edge_of_board_bottom_strip_at_ice":
                result.update(feedback_case_reference={k:match[k] for k in ("case_id", "version", "content_sha256")},
                    feedback_material_order=deepcopy(roles), original_annotation_pixels_transferred=False)
                self._use(match, "check_both_strip_interfaces_and_keep_lower_iceward_role",
                    current_native_positions_required=True, upper_edge_is_competitor=True)
            elif match["case"].get("kind") == "recipe":
                lesson = match["case"].get("assertions", {}).get("lesson", {})
                if lesson.get("reader_check") == "ordered_board_material_flanks":
                    result.setdefault("feedback_regression_checks", []).append(match["case_id"])
                    self._use(match, "regression_check_current_board_material_order", current_native_positions_required=True)
        return result

    def check_top_face_exclusion(self, near_board):
        """Record use only after the current source material/exclusion actually ran."""
        if not near_board.get('readings'):
            return
        if (near_board.get('world_geometry_modified') is not False
                or near_board.get('exterior_top_deck_cameras_judges_people_are_advertisements') is not False):
            raise ValueError('Near-board source check conflicts with the zero-thickness contract')
        for match in getattr(self, 'applicable', []):
            rule=match['case'].get('assertions', {}).get('top_face_rule', {})
            if rule.get('model_board_thickness') == 0 and rule.get('outside_surface_and_objects_are_ads') is False:
                self._use(match,'current_top_face_iceward_boundary_and_outside_ad_exclusion',
                    current_native_positions_required=True, old_UV_transferred=False,
                    excluded_pixel_count=near_board['exclusion_pixel_count'],
                    native_geometry_modified=False)

    def finish(self, rgb, summary, *, tool, current_evidence=(), observation_ids=()):
        import numpy as np
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:raise ValueError("Feedback requires actual native uint8 RGB")
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        expected = self.binding.get("decoded_native_rgb_sha256", self.binding.get("decoded_native_RGB_sha256"))
        if expected and expected != digest:raise ValueError("Feedback receipt differs from actual current RGB")
        authenticate_refs(current_evidence, self.project)
        if not self.ticket:
            return dict(version=VERSION, status="unregistered_source_no_feedback_scope", source=self.source,
                actual_native_RGB_sha256=digest, case_use_recorded=False)
        directory = AssetStore(self.project).data / "events" / self.event_id / "feedback" / "reads" / self.run_id
        receipt = dict(schema="rinkspace.production_source_feedback_read.v1", runtime_version=VERSION,
            run_id=self.run_id, source=self.source, tool=tool, reader_code_sha256=file_sha(Path(__file__)),
            read_ticket_id=self.ticket["read_ticket_id"], actual_native_RGB_sha256=digest,
            native_hw=list(image.shape[:2]), effects=self.effects,
            used_case_refs=list(self.used.values()), split_context=self.split_context,
            consumed_fact_sha256=sorted({ref['sha256'] for match in self.applicable
                if match['case_id'] in self.used for ref in match['case'].get('evidence_refs', [])}),
            retrieved_case_refs=[{k:m[k] for k in
                ("case_id", "version", "content_sha256", "match_status")} for m in self.ticket["cases"]],
            evidence_refs=list(current_evidence), summary=summary, blockers=self.blockers,
            observation_ids=list(observation_ids), pixel_quality_independently_verified=False,
            observation_transaction_applied=False, fit_mutated=False)
        tool_name = tool.rsplit(".", 1)[-1]
        if not tool_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in tool_name):
            raise ValueError("Feedback tool name must be a bounded identifier")
        ref = atomic_json(directory / (str(self.source["pts"]) + "_" + tool_name + ".json"), receipt)
        toolref = dict(ref, kind="native_reader_receipt", source=self.source)
        with feedback_store(self.project, self.event_id) as store:
            uses = FeedbackCycle(store).acknowledge_use(self.ticket, dict(source=self.source, run_id=self.run_id,
                used_case_refs=list(self.used.values()), tool_evidence_refs=[toolref],
                observation_ids=list(observation_ids), use_kind="source_review_guidance",
                outcome="implemented_current_RGB_check_executed; inspect_effects_and_summary_for_result"))
        return dict(version=VERSION, status="actual_reader_completed", receipt=ref, source=self.source,
            read_ticket_id=self.ticket["read_ticket_id"], actual_native_RGB_sha256=digest,
            used_case_refs=list(self.used.values()), effects=self.effects, blockers=self.blockers,
            case_use_ids=[u["use_id"] for u in uses], case_use_recorded=bool(uses),
            retrieved_only_case_ids=[m["case_id"] for m in self.ticket["cases"] if m["case_id"] not in self.used])


def learn_source_review(*, project=ROOT, event_id, finding_id, source, checks,
                        evidence_refs, resolution_type, lesson=None, observation_transaction=None):
    """Authenticated current review writes knowledge plus persistent regression.

    Actual observation candidates remain in the existing editor/installer. A
    model mismatch retains UV; a declaration never becomes a committed packet.
    """
    authenticate_refs(evidence_refs, project)
    with feedback_store(project, event_id) as store:
        result = FeedbackCycle(store).learn(dict(finding_id=finding_id, source=native_source(source),
            resolution_type=resolution_type, checks=checks, evidence_refs=evidence_refs,
            request_recheck=True, observation_transaction=observation_transaction), lesson)
        learned = result.get("learned_case")
        if learned:
            case = store.get_case(learned["case_id"], learned["version"])
            for key in ("content_sha256", "content_hash", "latest_recorded_version"):
                case.pop(key, None)
            # Immutable, bounded knowledge is durable; image/profile arrays stay on Data.
            path = Path(project) / "events" / event_id / "feedback" / "learned" / (learned["case_id"] + ".v" + str(learned["version"]) + ".json")
            from .path_policy import guard_project_write
            guard_project_write(path, project_root=project)
            if not path.exists():atomic_json(path, case)
    result["consumers"] = dict(knowledge=bool(learned), regression=bool(learned),
        observation=False, observation_transaction=observation_transaction,
        good_source_UV_retained=resolution_type == "model_mismatch")
    # This function has not run the public installation transaction. A supplied
    # candidate/receipt cannot change that fact; its observer has a separate outbox.
    result['observation_receipt_supplied'] = observation_transaction is not None
    ref = atomic_json(AssetStore(project).data / "events" / event_id / "feedback" / "resolutions" /
        (result["resolution"]["resolution_id"] + ".json"), result)
    result["receipt"] = ref
    return result
