"""Current, self-contained inputs and an atomic, provenance-preserving edit ledger."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
import uuid
from .atomic_io import read_text_shared as _read_text_shared, atomic_replace as _replace_atomic


class StoreError(Exception):
    def __init__(self, message, status=422, code="invalid_input"):
        super().__init__(message)
        self.status = status
        self.code = code


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def checked_path(path):
    """Validate the final absolute Windows UTF-16 path, including temporary names."""
    result = Path(path).resolve()
    if len(str(result).encode("utf-16-le")) // 2 >= 260:
        raise StoreError(f"路径必须少于260个UTF-16单元: {result}")
    if any(len(part.encode("utf-16-le")) // 2 > 255 for part in result.parts):
        raise StoreError(f"路径单个名称超过255个UTF-16单元: {result}")
    return result


def read_object(path):
    path = Path(path)
    try:
        result = json.loads(_read_text_shared(path))
    except FileNotFoundError as exc:
        raise StoreError(f"当前数据待登记: {path}", 503, "data_not_registered") from exc
    except (OSError, ValueError) as exc:
        raise StoreError(f"当前JSON不可读取: {path}: {exc}", 503, "invalid_current_data") from exc
    if not isinstance(result, dict):
        raise StoreError(f"当前JSON须为对象: {path}", 503, "invalid_current_data")
    return result


def atomic_json(path, value):
    path = checked_path(path)
    tmp = checked_path(path.with_name(path.name + "." + uuid.uuid4().hex[:12] + ".tmp"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_atomic(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


ASSUMPTION_KINDS = {"assumption", "physical_assumption", "physical_hypothesis", "physics_assumption", "hypothesis", "statistical_prior", "physical_prior"}
APPROVED = {"user_confirmed"}
EDITABLE = {"title", "expression", "scope", "owner", "notes", "enabled", "approval"}


def is_assumption(item):
    return str(item.get("kind", "")).lower() in ASSUMPTION_KINDS


def approved(item):
    return item.get("approval") in APPROVED


def requires_confirmation(item):
    return is_assumption(item) or item.get("approval", "not_recorded") in {"unapproved", "withdrawn", "rejected", "not_recorded"}


def public_constraint(item, execution=None):
    result = copy.deepcopy(item)
    result.setdefault("approval", "unapproved" if is_assumption(result) else "not_recorded")
    result.setdefault("enabled", False)
    # An inherited enabled bit is never a substitute for the user's approval.
    result["eligible_enabled"] = bool(result["enabled"]) and (not requires_confirmation(result) or approved(result))
    # The runner must report an actual binding to the same current model. Merely
    # enabling a ledger entry is not evidence that its implementation is assembled.
    result["effective_enabled"] = False
    result["effective_reason"] = "当前运行尚未报告此约束的实际装配"
    if execution and execution.get("matches_visible_model") is True:
        run = execution["run"]
        bindings = execution.get("constraint_bindings", run.get("constraint_bindings", []))
        binding = next((b for b in bindings if isinstance(b, dict) and b.get("id") == item["id"]), None)
        if binding is not None:
            fields = ("enabled", "approval", "expression", "scope")
            matching = all(binding.get(key) == result.get(key) for key in fields)
            result["effective_enabled"] = bool(result["eligible_enabled"] and matching)
            result["execution_status"] = run["status"]
            result["implementation"] = binding.get("implementation")
            result["effective_reason"] = ("已装配至当前模型的该次运行；状态：" + run["status"]) if result["effective_enabled"] else "配置与该次运行的约束快照不一致；编辑尚未应用"
        elif item["id"] in run.get("bound_constraint_ids", []) and run.get("constraint_revision") == execution.get("current_constraint_revision"):
            result["effective_enabled"] = bool(result["eligible_enabled"])
            result["execution_status"] = run["status"]
            result["effective_reason"] = "运行已报告当前台账修订的实际绑定；状态：" + run["status"]
    elif execution:
        result["effective_reason"] = "该次运行与当前可见模型结构不一致；不能作为当前约束生效证据"
    if requires_confirmation(result) and not approved(result):
        result["enabled"] = False
    result.setdefault("evidence", [])
    result.setdefault("notes", "")
    return result


class CurrentStore:
    def __init__(self, root, data, event="ankara_2026", programme="okada_mei_sp"):
        self.root = Path(root).resolve()
        self.data = Path(data).resolve()
        self.architecture_path = checked_path(self.root / "calibration" / "solve2" / "architecture.json")
        self.constraints_path = checked_path(self.root / "events" / event / "solve2" / "constraints.json")
        self.input_path = checked_path(self.root / "events" / event / "programmes" / programme / "solve2" / "input.json")
        self.review_path = checked_path(self.data / "runs" / "solve2_review" / "review.json")
        self.ledger_path = checked_path(self.data / "runs" / "solve2_control" / "edits")
        self._lock = threading.RLock()

    def architecture(self):
        with self._lock:
            return read_object(self.architecture_path)

    def contracts(self):
        with self._lock:
            return read_object(self.root / "calibration" / "solve2" / "contracts.json")

    def _constraints_raw(self):
        value = read_object(self.constraints_path)
        if type(value.get("revision")) is not int or not isinstance(value.get("constraints"), list):
            raise StoreError("约束数据必须包含整数revision与constraints数组", 503, "invalid_current_data")
        ids = []
        for item in value["constraints"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise StoreError("每个约束必须有字符串id", 503, "invalid_current_data")
            ids.append(item["id"])
        if len(set(ids)) != len(ids):
            raise StoreError("约束id重复", 503, "invalid_current_data")
        return value

    def constraints(self):
        with self._lock:
            result = self._constraints_raw()
            from .run_monitor import current_run
            execution = current_run(self.programme_input(), self.data)
            if execution is not None:
                execution["current_constraint_revision"] = result["revision"]
            result["constraints"] = [public_constraint(item, execution) for item in result["constraints"]]
            return result

    def update_constraint(self, constraint_id, payload):
        if not isinstance(payload, dict) or set(payload) != {"expected_revision", "changes"}:
            raise StoreError("提交必须包含expected_revision和changes")
        changes = payload["changes"]
        if not isinstance(changes, dict) or not changes:
            raise StoreError("changes必须是非空对象")
        forbidden = set(changes) - EDITABLE
        if forbidden:
            raise StoreError("原始来源与证据不可由普通编辑改写；不支持字段: " + ", ".join(sorted(forbidden)))
        if "enabled" in changes and type(changes["enabled"]) is not bool:
            raise StoreError("enabled必须是布尔值")
        for key in {"title", "owner", "notes", "approval"} & set(changes):
            if not isinstance(changes[key], str):
                raise StoreError(f"{key}必须是文本")
        for key in {"expression", "scope"} & set(changes):
            if not isinstance(changes[key], (str, dict, list)):
                raise StoreError(f"{key}必须是文本或JSON对象/数组")
        if "approval" in changes and changes["approval"] not in {"user_confirmed", "unapproved", "rejected", "withdrawn", "not_recorded"}:
            raise StoreError("未知批准状态")
        with self._lock:
            before = self._constraints_raw()
            if type(payload["expected_revision"]) is not int or payload["expected_revision"] != before["revision"]:
                raise StoreError("约束已被其他编辑更新；请刷新后重试", 409, "revision_conflict")
            after = copy.deepcopy(before)
            item = next((x for x in after["constraints"] if x["id"] == constraint_id), None)
            if item is None:
                raise StoreError("约束不存在", 404, "not_found")
            old_item = copy.deepcopy(item)
            item.update(changes)
            if requires_confirmation(item) and not approved(item):
                if changes.get("enabled") is True:
                    raise StoreError("未经用户确认的物理假设不能启用", 422, "unapproved_assumption")
                item["enabled"] = False
            diff = {key: {"before": old_item.get(key), "after": item.get(key)} for key in set(old_item) | set(item) if old_item.get(key) != item.get(key)}
            if not diff:
                return self.constraints()
            after["revision"] = before["revision"] + 1
            stamp = now_iso()
            edit_id = uuid.uuid4().hex[:16]
            edit_dir = checked_path(self.ledger_path / edit_id)
            old_path = checked_path(edit_dir / "before.json")
            event_path = checked_path(edit_dir / "event.json")
            event = {"id": edit_id, "time": stamp, "type": "constraint_edit", "constraint_id": constraint_id,
                     "previous_revision": before["revision"], "revision": after["revision"], "diff": diff,
                     "source_path": str(self.constraints_path), "before_path": str(old_path),
                     "approval_action": changes.get("approval"), "actor": "local_ui", "status": "prepared"}
            # The full previous revision and diff are durable before replacing current input.
            atomic_json(old_path, before)
            atomic_json(event_path, event)
            try:
                atomic_json(self.constraints_path, after)
            except Exception as exc:
                event.update(status="failed", error=str(exc))
                atomic_json(event_path, event)
                raise
            event["status"] = "committed"
            try:
                atomic_json(event_path, event)
            except OSError:
                # A prepared record still contains the complete old revision and diff.
                # The committed revision is authoritative, so report the successful edit.
                pass
            return self.constraints()

    def programme_input(self):
        with self._lock:
            return read_object(self.input_path)

    def datasets(self):
        datasets = []
        pending = []
        programme = None
        with self._lock:
            try:
                architecture = self.architecture()
                datasets.extend(copy.deepcopy(architecture.get("datasets", [])))
            except StoreError as exc:
                pending.append(str(exc))
            try:
                programme = self.programme_input()
                registered = programme.get("datasets", [])
                if isinstance(registered, list):
                    datasets.extend(copy.deepcopy(registered))
                datasets.append({"id": "programme_input", "title": "当前节目绑定", "path": str(self.input_path),
                                 "kind": "programme_binding", "status": "registered", "data": programme})
            except StoreError as exc:
                pending.append(str(exc))
        for item in datasets:
            if item.get("id")=="source_constraints":
                ledger=self._constraints_raw();item["rows"]=len(ledger["constraints"]);item["revision"]=ledger["revision"]
        return {"datasets": datasets, "programme": programme, "pending": pending}

    def review(self):
        with self._lock:
            return read_object(self.review_path)

    def events(self):
        with self._lock:
            events = []
            paths=list(self.ledger_path.glob("*/event.json"))
            paths.extend((self.data/"runs/solve2_s01/selection_edits").glob("*/event.json"))
            for path in paths:
                try:
                    events.append(read_object(path))
                except StoreError as exc:
                    events.append({"type": "ledger_error", "path": str(path), "message": str(exc)})
            return sorted(events, key=lambda x: x.get("time", ""), reverse=True)

    def sequence(self, current_runtime=None):
        pending = []
        shots = []
        try:
            programme = self.programme_input()
            candidates = programme.get("shots", programme.get("shot_order", programme.get("sequence", {}).get("shots", [])))
            for index, shot in enumerate(candidates):
                if isinstance(shot, str):
                    shot = {"id": shot}
                if isinstance(shot, dict):
                    entry = copy.deepcopy(shot)
                    entry.setdefault("id", f"s{index + 1:02d}")
                    shots.append(entry)
        except StoreError as exc:
            pending.append(str(exc))
        if not any(str(x.get("id", "")).lower() == "s01" for x in shots):
            shots.insert(0, {"id": "s01"})
        for shot in shots:
            first = str(shot["id"]).lower() == "s01"
            shot.update(status="awaiting_full_shot_validation" if first else "locked", locked=not first,
                        reason="须以同一模型全开冰线、围挡、广告及网格连续验证S01全段；实际数值阶段见运行监控" if first else "S01全段真实验证尚未通过，禁止进入后续镜头")
        return {"active_shot": "s01", "shots": shots, "advance_allowed": False,
                "validation_status": "not_passed",
                "engine_available": (current_runtime or {}).get("engine", {}).get("available", False),
                "numerical_status": (current_runtime or {}).get("run", {}).get("status"), "pending": pending}
