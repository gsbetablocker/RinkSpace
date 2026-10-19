"""SQLite feedback ledger. Evidence declarations are checked, not visual truth.

No method changes observation UV, a fit set, network weights or source files.
Candidate cases never replace an active version until an explicit active version
is recorded. All evidence paths are opaque references: this module opens none.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


class ValidationError(ValueError):
    """Invalid or unsafe-to-consume feedback input."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _copy(value):
    return json.loads(_json(value))


def _object(value, name):
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return _copy(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValidationError(f"{name} must be a 64-character SHA256 hexadecimal digest")
    return value.lower()


def _strings(value, name):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    return sorted(set(_text(item, name) for item in value))


def normalize_source(source: dict) -> dict:
    if not isinstance(source, dict):
        raise ValidationError("source must be an object")
    result = _copy(source)
    digest = source.get("video_sha256", source.get("source_video_sha256"))
    result["video_sha256"] = _sha(digest, "source.video_sha256")
    if source.get("source_video_sha256") and _sha(source["source_video_sha256"], "source.source_video_sha256") != result["video_sha256"]:
        raise ValidationError("conflicting source SHA256 aliases")
    result.pop("source_video_sha256", None)
    clock = source.get("time_base")
    if not isinstance(clock, str) or not re.fullmatch(r"[0-9]+/[0-9]+", clock):
        raise ValidationError("source.time_base must be a positive rational string, e.g. 1/12800")
    try:
        fraction = Fraction(clock)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValidationError("invalid source.time_base") from exc
    if fraction <= 0:
        raise ValidationError("source.time_base must be positive")
    result["time_base"] = f"{fraction.numerator}/{fraction.denominator}"
    if not isinstance(source.get("pts"), int) or isinstance(source.get("pts"), bool):
        raise ValidationError("source.pts must be an integer in the original source clock")
    result["coordinate_domain"] = _text(source.get("coordinate_domain"), "source.coordinate_domain")
    return result


def source_key(source: dict) -> str:
    source = normalize_source(source)
    return _hash({key: source[key] for key in ("video_sha256", "time_base", "pts", "coordinate_domain")})


def _scope(value):
    result = _copy(value or {})
    if not isinstance(result, dict):
        raise ValidationError("scope must be an object")
    for key in ("event_id", "camera_id"):
        if result.get(key) is not None:
            result[key] = _text(result[key], f"scope.{key}")
    if result.get("source_video_sha256") is not None:
        result["source_video_sha256"] = _sha(result["source_video_sha256"], "scope.source_video_sha256")
    result["entity_ids"] = _strings(result.get("entity_ids"), "scope.entity_ids")
    return result


def _context(value):
    value = _object(value, "context")
    result = dict(value.get("scope") or {})
    result.update({key: value[key] for key in ("event_id", "source_video_sha256", "camera_id", "entity_ids", "tags", "split_context", "include_related_cases") if key in value})
    if value.get("source"):
        source = normalize_source(value["source"])
        if result.get("source_video_sha256") and result["source_video_sha256"].lower() != source["video_sha256"]:
            raise ValidationError("context source SHA256 conflicts with source binding")
        result["source_video_sha256"] = source["video_sha256"]
    result.update(_scope(result))
    result["tags"] = _strings(result.get("tags", value.get("context_tags")), "context.tags")
    return result


def _tokens(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return set().union(*(_tokens(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return _tokens(list(value.values()))
    return set()


def _match(case, context):
    scope = case["scope"]
    for key in ("event_id", "source_video_sha256", "camera_id"):
        if scope.get(key) is not None and scope[key] != context.get(key):
            return None
    protected = {"heldout", "test", "final_test", "final_validation", "independent_validation", "validation_holdout"}
    if case.get("split_role", "development") in protected:
        return None
    split = context.get("split_context") or {}
    if isinstance(split, str):
        split = {"mode": split}
    if not isinstance(split, dict):
        raise ValidationError("split_context must be an object or mode string")
    blocked = _tokens(split.get("blocked_lineage_ids", [])) | _tokens(split.get("heldout_lineage_ids", []))
    if split.get("mode") in protected and context.get("source_video_sha256"):
        blocked.add(context["source_video_sha256"])
        blocked.add("capture:" + context["source_video_sha256"])
    lineage = _tokens(case.get("lineage", {})) | _tokens(case.get("lineage_ids", [])) | _tokens(scope.get("source_video_sha256"))
    if blocked & lineage:
        return None
    tags = set(case["tags"]) & set(context["tags"])
    entities = set(scope["entity_ids"])
    related = entities and not entities.intersection(context["entity_ids"])
    if related and not (context.get("include_related_cases") is True and tags):
        return None
    generic = not scope.get("event_id") and not scope.get("source_video_sha256")
    withdrawal = case.get("assertions", {}).get("withdrawal", {})
    local_withdrawal = isinstance(withdrawal, dict) and withdrawal.get("applicability") == "exact_source_frame_and_point_ids_only"
    advisory = bool(related or generic or local_withdrawal or case.get("kind") in {"recipe", "generic_recipe"})
    reason = "historical point withdrawal; adapter must match original source, point IDs and saved UV; no cross-frame negative transfer" if local_withdrawal else "related entity; inspect the example, do not transfer its identity" if related else "generic procedure; no event/instance fact transfer" if advisory else "explicit scope matches; each assertion retains its original evidence/precision limits"
    return {"match_status": "advisory" if advisory else "applicable", "match_reason": reason, "matched_tags": sorted(tags)}


class Store:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.path = str(Path(path).expanduser()) if str(path) != ":memory:" else ":memory:"
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO metadata VALUES ('schema_version','1');
            CREATE TABLE IF NOT EXISTS sources(source_key TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cases(case_id TEXT NOT NULL, version INTEGER NOT NULL,
              status TEXT NOT NULL, content_sha256 TEXT NOT NULL, body TEXT NOT NULL,
              created_at TEXT NOT NULL, PRIMARY KEY(case_id,version));
            CREATE TABLE IF NOT EXISTS findings(finding_id TEXT PRIMARY KEY, content_sha256 TEXT UNIQUE NOT NULL,
              group_id TEXT NOT NULL, source_key TEXT NOT NULL REFERENCES sources(source_key), body TEXT NOT NULL,
              created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS uses(use_id TEXT PRIMARY KEY, content_sha256 TEXT UNIQUE NOT NULL,
              source_key TEXT NOT NULL REFERENCES sources(source_key), body TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resolutions(resolution_id TEXT PRIMARY KEY, content_sha256 TEXT UNIQUE NOT NULL,
              finding_id TEXT NOT NULL REFERENCES findings(finding_id), body TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS finding_groups ON findings(group_id);
            CREATE INDEX IF NOT EXISTS finding_source_groups ON findings(source_key,group_id);
        """)
        if self.db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] != "1":
            raise ValidationError("unsupported store schema_version; do not overwrite this store")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def record_source(self, source: dict) -> dict:
        source = normalize_source(source)
        key = source_key(source)
        identity = {field: source[field] for field in ("video_sha256", "time_base", "pts", "coordinate_domain")}
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sources VALUES (?,?)", (key, _json(identity)))
        return {"source_key": key, "source": source}

    def upsert_case(self, case: dict) -> dict:
        case = _copy(case)
        if not isinstance(case, dict):
            raise ValidationError("case must be an object")
        case["case_id"] = _text(case.get("case_id"), "case_id")
        if not isinstance(case.get("version"), int) or isinstance(case["version"], bool) or case["version"] < 1:
            raise ValidationError("case.version must be a positive integer")
        if case.get("status") not in {"candidate", "active", "retracted"}:
            raise ValidationError("case.status must be candidate, active or retracted")
        case["title"] = _text(case.get("title"), "case.title")
        case["scope"] = _scope(case.get("scope"))
        case["tags"] = _strings(case.get("tags"), "case.tags")
        if not isinstance(case.get("assertions"), dict):
            raise ValidationError("case.assertions must separate identity, operator, visibility and position claims")
        for field, typ in (("provenance", dict), ("evidence_refs", list)):
            if not isinstance(case.get(field), typ):
                raise ValidationError(f"case.{field} must be {typ.__name__}")
        digest = _hash(case)
        old = self.db.execute("SELECT content_sha256 FROM cases WHERE case_id=? AND version=?", (case["case_id"], case["version"])).fetchone()
        if old and old[0] != digest:
            raise ValidationError("case version is immutable; record a new version")
        latest = self.db.execute("SELECT MAX(version) FROM cases WHERE case_id=?", (case["case_id"],)).fetchone()[0]
        if not old and latest is not None and case["version"] <= latest:
            raise ValidationError("new case version must increase monotonically")
        if not old:
            with self.db:
                self.db.execute("INSERT INTO cases VALUES (?,?,?,?,?,?)", (case["case_id"], case["version"], case["status"], digest, _json(case), self._now()))
        return {"case_id": case["case_id"], "version": case["version"], "content_sha256": digest, "idempotent": bool(old), "case": case}

    def _published(self, case_id=None):
        sql = "SELECT * FROM cases WHERE status IN ('active','retracted')"
        rows = self.db.execute(sql + (" AND case_id=?" if case_id else "") + " ORDER BY version", (case_id,) if case_id else ()).fetchall()
        latest = {}
        for row in rows:
            latest[row["case_id"]] = row
        return latest

    def query(self, context: dict) -> list[dict]:
        context = _context(context)
        result = []
        for row in self._published().values():
            if row["status"] != "active":
                continue
            case = json.loads(row["body"])
            match = _match(case, context)
            if match:
                result.append({"case_id": row["case_id"], "version": row["version"], "content_sha256": row["content_sha256"], "case": case, **match})
        return sorted(result, key=lambda item: (not item["case"].get("mandatory", False), -len(item["matched_tags"]), item["case_id"]))

    def get_case(self, case_id, version=None):
        case_id = _text(case_id, "case_id")
        if version is None:
            row = self._published(case_id).get(case_id)
        else:
            if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                raise ValidationError("case version must be a positive integer")
            row = self.db.execute("SELECT * FROM cases WHERE case_id=? AND version=?", (case_id, version)).fetchone()
        if row is None:
            return None
        latest = self.db.execute("SELECT MAX(version) FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]
        return {**json.loads(row["body"]), "content_sha256": row["content_sha256"],
                "content_hash": row["content_sha256"], "latest_recorded_version": latest}

    def _append(self, table, id_field, value, extra=()):
        value = _copy(value)
        supplied_id = value.pop(id_field, None)
        digest = _hash(value)
        record_id = _text(supplied_id, id_field) if supplied_id is not None else f"{table[:-1]}_{digest[:24]}"
        collision = self.db.execute(f"SELECT content_sha256 FROM {table} WHERE {id_field}=?", (record_id,)).fetchone()
        if collision and collision[0] != digest:
            raise ValidationError(f"{id_field} is immutable; use a new ID for a new occurrence")
        old = self.db.execute(f"SELECT * FROM {table} WHERE content_sha256=?", (digest,)).fetchone()
        if old:
            if old["content_sha256"] != digest:
                raise ValidationError(f"{id_field} is immutable; use a new ID for a new occurrence")
            result = json.loads(old["body"])
            return {**result, id_field: old[id_field], "content_sha256": digest, "content_hash": digest, "idempotent": True}
        value[id_field] = record_id
        columns = [id_field, "content_sha256", *(key for key, _ in extra), "body", "created_at"]
        values = [record_id, digest, *(item for _, item in extra), _json(value), self._now()]
        with self.db:
            self.db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", values)
        return {**value, "content_sha256": digest, "content_hash": digest, "idempotent": False}

    def record_finding(self, finding: dict) -> dict:
        finding = _object(finding, "finding")
        source = self.record_source(finding.get("source"))
        finding["source"] = source["source"]
        context = _context(finding)
        finding["event_id"] = context.get("event_id")
        finding["camera_id"] = context.get("camera_id")
        entities = context["entity_ids"] or ([finding["entity_id"]] if finding.get("entity_id") else [])
        finding["entity_ids"] = _strings(entities, "entity_ids")
        finding["signal_type"] = _text(finding.get("signal_type", finding.get("kind")), "signal_type")
        finding["observation_ids"] = _strings(finding.get("observation_ids"), "observation_ids")
        finding["context_tags"] = context["tags"]
        if not isinstance(finding.get("evidence_refs", []), list):
            raise ValidationError("finding.evidence_refs must be a list")
        group = {"event_id": finding["event_id"], "programme_id": finding.get("programme_id", source["source"]["video_sha256"]), "source_video_sha256": source["source"]["video_sha256"], "time_base": source["source"]["time_base"], "coordinate_domain": source["source"]["coordinate_domain"], "camera_id": context.get("camera_id"), "entity_ids": finding["entity_ids"], "problem": finding.get("problem_key", finding["signal_type"])}
        group_id = "group_" + _hash(group)[:24]
        finding["group_id"] = group_id
        finding["group_meaning"] = "related triage occurrences, not a proven common root cause"
        result = self._append("findings", "finding_id", finding, (("group_id", group_id), ("source_key", source["source_key"])))
        result["group_occurrences"] = self.db.execute("SELECT COUNT(*) FROM findings WHERE group_id=?", (group_id,)).fetchone()[0]
        return result

    def record_use(self, use: dict) -> dict:
        use = _object(use, "use")
        use["run_id"] = _text(use.get("run_id"), "use.run_id")
        use["source"] = normalize_source(use.get("source"))
        case_id = _text(use.get("case_id"), "use.case_id")
        row = self._published(case_id).get(case_id)
        if not row or row["status"] != "active":
            raise ValidationError("case is unknown, candidate-only or retracted")
        if use.get("version") != row["version"] or use.get("content_sha256") != row["content_sha256"]:
            raise ValidationError("case reference is stale or has the wrong content_sha256; query again")
        matches = {item["case_id"]: item for item in self.query(use)}
        if case_id not in matches:
            raise ValidationError("case does not apply to this source/scope/split context")
        match = matches[case_id]
        if match["match_status"] == "advisory" and use.get("use_kind") == "fact_application":
            raise ValidationError("advisory retrieval cannot transfer an event/entity fact")
        refs = use.get("tool_evidence_refs", [])
        receipt_bound = False
        for ref in refs if isinstance(refs, list) else []:
            if not isinstance(ref, dict) or ref.get("kind") not in {"reader_receipt", "native_reader_receipt", "source_review_receipt", "profile_receipt", "tool_call_receipt"}:
                continue
            locator = ref.get("path") or ref.get("evidence_id")
            if not isinstance(locator, str) or not locator.strip():
                continue
            try:
                _sha(ref.get("sha256"), "tool receipt.sha256")
                receipt_bound |= source_key(ref.get("source")) == source_key(use["source"])
            except ValidationError:
                continue
        if not receipt_bound:
            raise ValidationError("use requires a structured current-source reader/tool receipt with kind, SHA256, source and path or evidence_id")
        use["match_status"] = match["match_status"]
        use["case_content_sha256"] = use.pop("content_sha256")
        use["consumption_assurance"] = "receipt/source declarations checked; tool execution and pixel interpretation are not independently verified by this ledger"
        source = self.record_source(use["source"])
        return self._append("uses", "use_id", use, (("source_key", source["source_key"]),))

    def _get(self, table, field, record_id):
        row = self.db.execute(f"SELECT * FROM {table} WHERE {field}=?", (record_id,)).fetchone()
        if not row:
            raise ValidationError(f"unknown {field}: {record_id}")
        return {**json.loads(row["body"]), "content_sha256": row["content_sha256"], "content_hash": row["content_sha256"]}

    def get_finding(self, finding_id):
        return self._get("findings", "finding_id", finding_id)

    def get_resolution(self, resolution_id):
        return self._get("resolutions", "resolution_id", resolution_id)

    def resolutions_for_finding(self, finding_id):
        self.get_finding(finding_id)
        rows = self.db.execute("SELECT * FROM resolutions WHERE finding_id=? ORDER BY rowid", (finding_id,)).fetchall()
        return [{**json.loads(row["body"]), "content_sha256": row["content_sha256"],
                 "content_hash": row["content_sha256"]} for row in rows]

    def resolutions_for_source_group(self, source: dict, group_id: str) -> list[dict]:
        """Retrieve prior reviews for the exact source/problem group across runs.

        Callers still compare effective review dependencies and artifact hashes.
        Group membership is a triage aid, never proof of a common root cause.
        """
        rows = self.db.execute(
            "SELECT r.* FROM resolutions r JOIN findings f ON r.finding_id=f.finding_id "
            "WHERE f.source_key=? AND f.group_id=? ORDER BY r.rowid",
            (source_key(source), _text(group_id, "group_id")),
        ).fetchall()
        return [{**json.loads(row["body"]), "content_sha256": row["content_sha256"], "content_hash": row["content_sha256"]} for row in rows]

    def record_resolution(self, resolution: dict) -> dict:
        resolution = _object(resolution, "resolution")
        if resolution.get("fit_admissible") is True:
            raise ValidationError("this ledger cannot set fit_admissible; only an adapter recheck may be recommended")
        finding = self.get_finding(_text(resolution.get("finding_id"), "resolution.finding_id"))
        kind = resolution.get("resolution_type", "unresolved")
        if kind not in {"measurement_error", "operator_error", "identity_error", "coverage_gap", "model_mismatch", "unresolved"}:
            raise ValidationError("unknown resolution_type")
        resolution["resolution_type"] = kind
        source = normalize_source(resolution.get("source", finding["source"]))
        if source_key(source) != source_key(finding["source"]):
            raise ValidationError("resolution source must exactly match the finding source, clock, PTS and coordinate domain")
        resolution["source"] = source
        evidence = resolution.get("evidence_refs", [])
        if not isinstance(evidence, list):
            raise ValidationError("resolution.evidence_refs must be a list")
        native_ids = set()
        for item in evidence:
            if not isinstance(item, dict) or item.get("kind") not in {"native_pixel", "native_pixels", "source_native_pixel"}:
                continue
            if item.get("independent_of_model") is not True or item.get("current_frame") is not True:
                continue
            try:
                _sha(item.get("sha256"), "native evidence.sha256")
                same = source_key(item.get("source")) == source_key(source)
            except ValidationError:
                same = False
            if same:
                native_ids.update(str(item[key]) for key in ("evidence_id", "path", "sha256") if item.get(key))
                native_ids.add(_hash(item))
        blockers = []
        if not native_ids:
            blockers.append("missing exact-source native-pixel evidence with independent/current declarations and SHA256")
        checks = resolution.get("checks", {})
        if not isinstance(checks, dict):
            raise ValidationError("resolution.checks must be an object")
        dimensions = ("semantic_class", "physical_instance", "measurement_operator", "visibility", "position")
        verified = []
        for name in dimensions:
            check = checks.get(name, {})
            refs = check.get("evidence_refs", []) if isinstance(check, dict) else []
            ids = {_hash(ref) if isinstance(ref, dict) else str(ref) for ref in refs} if isinstance(refs, list) else set()
            if not isinstance(check, dict) or check.get("status") != "verified" or not ids.intersection(native_ids):
                blockers.append(f"{name} needs a distinct verified check linked to this resolution's current native evidence")
            else:
                verified.append(name)
        visibility = checks.get("visibility", {})
        if isinstance(visibility, dict) and (visibility.get("visible") is False or visibility.get("value") in ("occluded", "invisible", "unknown")):
            blockers.append("visibility check does not support a visible measurement")
        resolution["source_evidence_supported"] = not blockers
        resolution["source_evidence_present"] = bool(native_ids)
        resolution["verified_dimensions"] = verified
        resolution["unresolved_dimensions"] = [name for name in dimensions if name not in verified]
        if kind == "unresolved":
            blockers.append("resolution_type remains unresolved")
        requested = resolution.get("request_recheck") is True
        resolution["status"] = "eligible_for_adapter_recheck" if requested and not blockers else "unknown"
        resolution["recheck_blockers"] = blockers
        resolution["evidence_assurance"] = "source identity and declarations checked; pixel interpretation is not verified by this ledger"
        resolution["fit_mutated"] = False
        return self._append("resolutions", "resolution_id", resolution, (("finding_id", finding["finding_id"]),))

    def stats(self) -> dict:
        result = {table: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("sources", "cases", "findings", "uses", "resolutions")}
        result["active_cases"] = sum(row["status"] == "active" for row in self._published().values())
        result["finding_groups"] = self.db.execute("SELECT COUNT(DISTINCT group_id) FROM findings").fetchone()[0]
        result["groups_with_repeated_occurrences"] = self.db.execute("SELECT COUNT(*) FROM (SELECT group_id FROM findings GROUP BY group_id HAVING COUNT(*)>1)").fetchone()[0]
        result["eligible_for_adapter_recheck"] = sum(json.loads(row[0]).get("status") == "eligible_for_adapter_recheck" for row in self.db.execute("SELECT body FROM resolutions"))
        result["fit_mutations"] = 0
        result["training_updates"] = 0
        return result

    def export(self) -> dict:
        output = {"schema_version": 1, "stats": self.stats()}
        for table in ("sources", "cases", "findings", "uses", "resolutions"):
            output[table] = [{**dict(row), "body": json.loads(row["body"])} for row in self.db.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        return output
