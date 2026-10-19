"""Frozen source-only readability gate for programme advertisement components.

The gate preserves every original UV.  It only selects which of its image-x
and image-y residuals may enter one explicitly configured graph comparison.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "rinkspace-ad-source-component-gate-v1"
_SIDECAR_SCHEMA = "rinkspace.ad-native-source-unreadable-components-candidate.v1"
_REASONS = {
    "wide_step_and_unstable_three_scan_corner_extrapolation",
    "two_wide_scans_but_parallel_widths_unstable",
    "source_visual_confirmed_foreground_occlusion",
}
_VISUAL_REASON = "source_visual_confirmed_foreground_occlusion"
_VISUAL_REVIEW_SCHEMA = "rinkspace-ad-source-occlusion-future-correction-note-v1"
_ALLOWED_COMPONENTS = {("x", "y"), ("x",), ("y",), ()}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _uv(value, label: str) -> np.ndarray:
    result = np.asarray(value, np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be one finite source UV")
    return result


def _key(row) -> tuple[int, int, str]:
    return int(row["pts"]), int(row["user_number"]), str(row["role"])


def _validate_visual_review(row: dict, axis: str, source_sha256: str,
                            ad_corners: dict, frozen_inputs: dict) -> None:
    """Bind an exceptional omission to a reviewed source pixel, not a label."""
    ref = row.get("visual_review")
    required = {"path", "sha256", "record_key", "native_png", "native_png_sha256"}
    if not isinstance(ref, dict) or set(ref) != required:
        raise ValueError("Visual ad component needs a frozen source review and image")
    review_path, image_path = Path(ref["path"]), Path(ref["native_png"])
    for path, sha in ((review_path, ref["sha256"]),
                      (image_path, ref["native_png_sha256"])):
        if frozen_inputs.get(str(path)) != sha or _digest(path) != sha:
            raise ValueError(f"Visual ad component source evidence hash changed: {path}")
    review = json.loads(review_path.read_text(encoding="utf8"))
    if (review.get("schema") != _VISUAL_REVIEW_SCHEMA or
            review.get("source_sha256") != source_sha256 or
            not isinstance(ref["record_key"], str)):
        raise ValueError("Visual ad component source review provenance differs")
    record = review.get(ref["record_key"])
    if not isinstance(record, dict) or record.get("exact_key", {}).get("kind") != "ad_corner":
        raise ValueError("Visual ad component has no confirmed corner record")
    key = _key(row)
    original = ad_corners.get(key)
    if (original is None or _key(record["exact_key"]) != key or
            not np.array_equal(_uv(record["exact_key"]["uv"], "review UV"),
                               _uv(row["uv"], "selection UV")) or
            not np.array_equal(_uv(row["uv"], "selection UV"),
                               _uv(original["uv"], "original UV"))):
        raise ValueError("Visual ad component review key or original UV differs")
    reviewed_source = record.get("original_observation", {}).get("source_file")
    if (not reviewed_source or
            Path(reviewed_source).resolve() != Path(original["input_file"]).resolve()):
        raise ValueError("Visual ad component review belongs to another observation input")
    if (axis not in record.get("current_v14_components", []) or
            axis in record.get("source_only_supported_components", []) or
            not isinstance(record.get("source_only_supported_components"), list)):
        raise ValueError("Visual ad component axis still has source support")
    pixel = record.get("source_pixel_evidence", {})
    if (pixel.get("native_png") != str(image_path) or
            pixel.get("native_png_sha256") != ref["native_png_sha256"] or
            not isinstance(pixel.get("finding"), str) or not pixel["finding"].strip() or
            review.get("source_paths_sha256", {}).get(str(image_path)) !=
            ref["native_png_sha256"]):
        raise ValueError("Visual ad component has no matching reviewed image evidence")


class SourceComponentGate:
    """Validate exact source observations and return their retained axes."""

    def __init__(self, spec: dict, source_sha256: str,
                 ad_corners: dict, input_hashes: dict):
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise ValueError("Ad source component gate needs path and SHA-256")
        path = Path(spec["path"])
        manifest_hash = _digest(path)
        if manifest_hash != spec["sha256"]:
            raise ValueError("Frozen ad source component gate hash changed")
        manifest = json.loads(path.read_text(encoding="utf8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown ad source component gate schema")
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError("Ad source component gate belongs to another video")
        if manifest.get("frozen_holdout_used_for_fit") is not False:
            raise ValueError("Ad component gate may not use frozen holdout")
        source_path = Path(manifest["source_path"])
        if source_path.stat().st_size != int(manifest["source_file_size"]):
            raise ValueError("Ad component gate source video size changed")

        frozen_inputs = manifest.get("input_sha256")
        if not isinstance(frozen_inputs, dict) or not frozen_inputs:
            raise ValueError("Ad component gate input hashes are missing")
        source_files = {str(Path(row["input_file"]).resolve())
                        for row in ad_corners.values()}
        if not source_files.issubset({str(Path(name).resolve()) for name in frozen_inputs}):
            raise ValueError("Ad component gate omits an observation input hash")
        selection = manifest.get("selection_sidecar")
        if not isinstance(selection, dict) or set(selection) != {"path", "sha256"}:
            raise ValueError("Ad component gate needs its frozen source selection")
        selection_path = Path(selection["path"])
        if (str(selection_path.resolve()) not in
                {str(Path(name).resolve()) for name in frozen_inputs}):
            raise ValueError("Ad component gate omits source selection input hash")
        if _digest(selection_path) != selection["sha256"]:
            raise ValueError("Frozen source selection hash changed")
        for filename, expected in frozen_inputs.items():
            item = Path(filename)
            actual = _digest(item)
            if actual != expected:
                raise ValueError(f"Ad component gate source input changed: {item}")
            resolved = str(item.resolve())
            if resolved in input_hashes and input_hashes[resolved] != actual:
                raise ValueError(f"Conflicting source input hash: {resolved}")
            input_hashes[resolved] = actual
        input_hashes[str(path.resolve())] = manifest_hash

        selection_data = json.loads(selection_path.read_text(encoding="utf8"))
        if (selection_data.get("schema") != _SIDECAR_SCHEMA or
                selection_data.get("source_sha256") != source_sha256 or
                selection_data.get("selection_uses_fit_residual") is not False):
            raise ValueError("Ad component gate source selection provenance differs")
        selection_flags = selection_data.get("flags")
        if (not isinstance(selection_flags, list) or
                len(selection_flags) != selection_data.get("selected_components")):
            raise ValueError("Ad component gate source selection count differs")
        if any(row.get("reason") == _VISUAL_REASON for row in selection_flags):
            base = selection_data.get("base_selection_sidecar")
            if not isinstance(base, dict) or set(base) != {"path", "sha256"}:
                raise ValueError("Visual ad selection needs its frozen base selection")
            base_path = Path(base["path"])
            if (frozen_inputs.get(str(base_path)) != base["sha256"] or
                    _digest(base_path) != base["sha256"]):
                raise ValueError("Visual ad base selection hash changed")
            base_data = json.loads(base_path.read_text(encoding="utf8"))
            base_flags = base_data.get("flags")
            if (base_data.get("schema") != _SIDECAR_SCHEMA or
                    base_data.get("source_sha256") != source_sha256 or
                    base_data.get("selection_uses_fit_residual") is not False or
                    not isinstance(base_flags, list) or
                    len(base_flags) != base_data.get("selected_components") or
                    selection_flags[:len(base_flags)] != base_flags or
                    any(row.get("reason") != _VISUAL_REASON
                        for row in selection_flags[len(base_flags):])):
                raise ValueError("Visual ad selection changed its frozen base flags")
        flagged = {}
        for row in selection_flags:
            key = _key(row)
            axis = {"image_x": "x", "image_y": "y"}.get(row.get("component"))
            if axis is None or row.get("reason") not in _REASONS:
                raise ValueError("Unknown frozen ad component selection reason")
            if row["reason"] == _VISUAL_REASON:
                _validate_visual_review(row, axis, source_sha256, ad_corners, frozen_inputs)
            position = key + (axis,)
            if position in flagged:
                raise ValueError(f"Duplicate frozen ad component selection: {position}")
            flagged[position] = (_uv(row["uv"], "selection UV"), row["reason"])

        observations = manifest.get("observations")
        if (not isinstance(observations, list) or
                len(observations) != int(manifest.get("expected_observation_count", -1)) or
                len(observations) != len(ad_corners)):
            raise ValueError("Ad component gate must cover every programme corner")
        self._components = {}
        self._reasons = {}
        for row in observations:
            if row.get("kind") != "ad_corner":
                raise ValueError("Ad component gate row kind differs")
            key = _key(row)
            if key in self._components:
                raise ValueError(f"Duplicate ad component gate corner: {key}")
            original = ad_corners.get(key)
            if original is None:
                raise ValueError(f"Ad component gate has an extra corner: {key}")
            uv = _uv(row["uv"], "gate UV")
            if not np.array_equal(uv, _uv(original["uv"], "original UV")):
                raise ValueError(f"Ad component gate UV differs: {key}")
            components = row.get("components")
            if not isinstance(components, list) or tuple(components) not in _ALLOWED_COMPONENTS:
                raise ValueError(f"Ad component gate axes differ: {key}")
            reasons = row.get("unreadable_reasons")
            if not isinstance(reasons, dict) or set(reasons) != {"x", "y"} - set(components):
                raise ValueError(f"Ad component gate reasons do not match axes: {key}")
            for axis, reason in reasons.items():
                frozen = flagged.get(key + (axis,))
                if (reason not in _REASONS or frozen is None or
                        reason != frozen[1] or not np.array_equal(uv, frozen[0])):
                    raise ValueError(f"Ad component gate differs from source selection: {key}, {axis}")
            self._components[key] = (uv, tuple(components))
            self._reasons[key] = reasons
        if set(self._components) != set(ad_corners):
            raise ValueError("Ad component gate is missing a programme corner")
        if sum(len(reasons) for reasons in self._reasons.values()) != len(flagged):
            raise ValueError("Ad component gate omits a frozen source selection")
        if (manifest.get("expected_unreadable_components") != len(flagged) or
                manifest.get("expected_ad_factor_count") !=
                sum(bool(components) for _, components in self._components.values()) or
                manifest.get("expected_active_residual_components") !=
                sum(len(components) for _, components in self._components.values())):
            raise ValueError("Ad component gate frozen factor dimensions differ")

        self._called = set()
        self._reason_counts = dict(Counter(reason for reasons in self._reasons.values()
                                           for reason in reasons.values()))
        self.provenance = {
            "path": str(path.resolve()), "sha256": manifest_hash,
            "source_sha256": source_sha256,
            "selection_sidecar_path": str(selection_path.resolve()),
            "selection_sidecar_sha256": selection["sha256"],
            "expected_corners": len(self._components),
            "unreadable_components": len(flagged),
            "unreadable_reasons": self._reason_counts,
            "input_file_count": len(frozen_inputs),
            "input_sha256": dict(frozen_inputs),
            "frozen_holdout_used_for_fit": False,
        }

    def components(self, pts: int, number: int, role: str, uv) -> tuple[str, ...]:
        key = int(pts), int(number), str(role)
        entry = self._components.get(key)
        if entry is None:
            raise ValueError(f"Ad component gate has no corner: {key}")
        if not np.array_equal(_uv(uv, "factor UV"), entry[0]):
            raise ValueError(f"Ad component gate factor UV differs: {key}")
        if key in self._called:
            raise ValueError(f"Ad component gate corner applied twice: {key}")
        self._called.add(key)
        return entry[1]

    def report(self, *, require_complete: bool = False) -> dict:
        if require_complete and self._called != set(self._components):
            raise ValueError("Ad component gate did not visit every programme corner")
        counts = Counter(components for _, components in self._components.values())
        excluded = [{"pts": key[0], "user_number": key[1], "role": key[2],
                     "uv": self._components[key][0].tolist(),
                     "component": axis, "reason": reason}
                    for key, reasons in sorted(self._reasons.items())
                    for axis, reason in sorted(reasons.items())]
        return {**self.provenance,
                "applied_corners": len(self._called),
                "factor_components_xy": counts[("x", "y")],
                "factor_components_x_only": counts[("x",)],
                "factor_components_y_only": counts[("y",)],
                "omitted_both_components": counts[()],
                "active_component_residuals": sum(len(components)
                                                  for _, components in self._components.values()),
                "component_exclusions": excluded,
                "original_uv_preserved_in_diagnostics": True}
