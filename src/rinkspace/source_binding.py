"""Read-only migration of event/rig/physical-installation source identities."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def canonical_source_binding(binding, scope, *, allow_legacy_event_installation=False):
    """Validate against the registered shot table without rewriting old packets."""
    result = deepcopy(binding)
    if isinstance(scope, (str, Path)):
        scope = json.loads(Path(scope).read_text(encoding="utf-8-sig"))
    event = scope["event_id"]
    shots = [row for row in scope["shots"] if row["id"] == result.get("shot_id")]
    if len(shots) != 1:
        raise ValueError("Source binding needs one explicitly registered shot")
    shot = shots[0]
    rig = shot["rig_id"]
    installation = scope["rig_bindings"]["ids"].get(rig)
    if not installation or installation == event:
        raise ValueError("Registered rig has no distinct physical installation identity")
    expected_clock = Fraction(scope["source_clock"]["time_base_num"], scope["source_clock"]["time_base_den"])
    stamp = result.get("source_pts")
    first, stop = shot["processing"]["first_pts"], shot["processing"]["stop_pts_exclusive"]
    step = scope["source_clock"]["native_pts_step"]
    if (type(stamp) is not int or type(step) is not int or step <= 0 or not first <= stamp < stop or (stamp-first) % step
            or result.get("source_sha256") != scope["source"]["sha256"]
            or Fraction(str(result.get("time_base", "0"))) != expected_clock
            or result.get("native_hw") != [scope["source"]["height"], scope["source"]["width"]]):
        raise ValueError("Source binding differs from registered video/clock/PTS/dimensions")
    if result.get("event_id", event) != event or result.get("rig_id", rig) != rig:
        raise ValueError("Source event/rig identity conflicts with registered shot")
    old = result.get("installation_id")
    legacy = old == event
    if old is not None and old != installation and not (legacy and allow_legacy_event_installation):
        raise ValueError("Source physical installation identity conflicts with registered rig")
    result.update(event_id=event, rig_id=rig, installation_id=installation, time_base=str(expected_clock),
                  identity_schema="rinkspace.source_installation.v2")
    result["installation_mapping"] = {"event_id": event, "rig_id": rig, "installation_id": installation,
        "scope_sha256": hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "legacy_event_installation_adapted": legacy, "historical_packet_modified": False}
    return result


def registered_reference_binding(binding, scope):
    """Resolve a retained source exposure only when its registered interval is unique."""
    if isinstance(scope, (str, Path)):
        scope = json.loads(Path(scope).read_text(encoding="utf-8-sig"))
    candidates = [s for s in scope["shots"] if s["processing"]["first_pts"] <= binding["source_pts"] < s["processing"]["stop_pts_exclusive"]]
    if len(candidates) != 1:
        raise ValueError("Native reference exposure has no unique registered shot")
    result = dict(binding, shot_id=candidates[0]["id"])
    return canonical_source_binding(result, scope)
