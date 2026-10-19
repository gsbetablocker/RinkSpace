"""Preserve raw Lane observations while excluding unresolved identity from fit.

The active ice factor accepts fit/holdout only. Keeping these rows as holdout
would still imply their Lane2 identity during review, so retain their complete
original payload separately from the active rows in a derived source packet.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json


BASE = Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026/observations/vision_training/continuous_review/cut6_clear_source_constraints")
SOURCE = BASE / "complete_source_clear_anchor_append_20261007/ice_parent_plus_actual_clear_anchors.json"
OUTPUT = SOURCE.with_name("ice_clear_anchors_lane_identity_unresolved_20261007.json")
RECORD = SOURCE.with_name("lane_identity_unresolved_retirement_20261007.json")
IDS = {
    "cut6_source_230506752_lane2_upper_long_holdout",
    "cut6_source_230507776_lane2_upper_long_fit",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    packet = deepcopy(original)
    unresolved = [row for row in packet["rows"] if row["id"] in IDS]
    if {row["id"] for row in unresolved} != IDS:
        raise ValueError("Expected exactly the two source Lane identity records")
    packet["rows"] = [row for row in packet["rows"] if row["id"] not in IDS]
    if any(row != old for row, old in zip(packet["rows"], [r for r in original["rows"] if r["id"] not in IDS])):
        raise ValueError("Unrelated source payload changed")
    record = dict(
        schema="rinkspace.source-identity-unresolved-retirement.v1",
        source_packet=dict(path=str(SOURCE), sha256=sha(SOURCE)),
        derived_packet=str(OUTPUT),
        source_original_modified=False,
        disposition="Physical identity missing; original pixel observations retained without fitting or Lane2 accuracy scoring",
        observed_material="Visible gray painted long-side band on ice",
        independent_RGB_frames_reviewed=[230382336, 230419968, 230507776],
        old_identity_basis="User arrow interpreted as Lane2; its navigation chain carries the same initial identity assumption",
        continuous_RGB_correspondence_reference=str(BASE / "Lane2_507776_to_501120_actual_pixel_navigation.json"),
        continuity_establishes_same_source_band_not_Lane_number=True,
        evidence_does_not_prove_another_physical_line_identity=True,
        rows=[dict(id=r["id"], pts=r["pts"], original_split=r["split"], UV_count=len(r["uv"])) for r in unresolved],
        excluded_FIT_scalar_count=sum(2 * len(r["uv"]) for r in unresolved if r["split"] == "fit"),
        excluded_HELD_scalar_count=sum(2 * len(r["uv"]) for r in unresolved if r["split"] == "holdout"),
        source_pixel_coordinates_or_sigma_changed=False,
        new_physical_identity_guessed=False,
    )
    packet["identity_unresolved_source_rows"] = unresolved
    packet["source_identity_retirement"] = record
    packet["status"] = "DERIVED_ACTIVE_SOURCE_WITH_EXPLICIT_UNRESOLVED_LANE_IDENTITY"
    packet["observation_count"] = len(packet["rows"])
    packet["fit_observation_count"] = sum(r["split"] == "fit" for r in packet["rows"])
    packet["holdout_observation_count"] = sum(r["split"] == "holdout" for r in packet["rows"])
    packet["current_fit_scalar_count"] = sum(2 * len(r["uv"]) for r in packet["rows"] if r["split"] == "fit")
    packet["current_primary_held_scalar_count"] = sum(2 * len(r["uv"]) for r in packet["rows"] if r["split"] == "holdout")
    if OUTPUT.exists() or RECORD.exists():
        raise FileExistsError("Preserve prior derived evidence; do not overwrite")
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    record["derived_packet_sha256"] = sha(OUTPUT)
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(packet=str(OUTPUT), sha256=sha(OUTPUT), record=str(RECORD),
        excluded_FIT_scalars=record["excluded_FIT_scalar_count"], active_FIT_scalars=packet["current_fit_scalar_count"])))


if __name__ == "__main__":
    main()
