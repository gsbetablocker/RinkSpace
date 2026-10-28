"""Explicit, reversible camera ablations for the current native motion rig.

This module only selects existing chart coordinates and copies scalar seed
values. It does not calibrate a Sony camera, infer frame size or readout time,
change observations, choose mechanical bounds, or write model/state files.
NumPy is the only non-stdlib dependency; no native or GPU library is imported.

The native rig has 38/43 stored values but 27/31 independent chart directions.
In particular, rotations and axes cannot be overwritten as scalar chart values.
The mapping follows native/include/rinkspace/camera_chart.hpp and camera_state.hpp.
"""

from copy import deepcopy
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np


SCHEMA = "rinkspace.camera_experiment.v1"
CAMERA_NAMES = (
    "R0_x", "R0_y", "R0_z", "pan_axis_tangent1", "pan_axis_tangent2",
    "tilt_axis", "pivot_x", "pivot_y", "pivot_z", "lever_x", "lever_y",
    "lever_z", "axis_spacing", "pupil1", "pupil2", "cx", "cy", "eta",
    "cx_zoom", "cy_zoom", "k10", "k11", "k20", "k21", "p1", "p2",
    "readout", "wide_k1", "wide_k2", "wide_p1", "wide_p2",
)
_SCALAR_STORAGE_INDEX = {
    **dict(zip(CAMERA_NAMES[6:15], range(15, 24))),
    **dict(zip(CAMERA_NAMES[15:27], range(26, 38))),
    **dict(zip(CAMERA_NAMES[27:], range(39, 43))),
}
_PROFILE_FIELDS = frozenset((
    "schema", "id", "basis", "active_camera_names", "fixed_rig_values",
    "scene_mode",
))
_BASIS_FIELDS = frozenset(("kind", "not_sony_measurement", "description"))


def _exact_fields(value, fields, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = fields - value.keys()
    extra = value.keys() - fields
    if missing or extra:
        raise ValueError(f"{label}: missing fields {sorted(missing)!r}; "
                         f"unknown fields {sorted(extra, key=str)!r}")


def _finite_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite numeric scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validated_profile(profile):
    _exact_fields(profile, _PROFILE_FIELDS, "camera experiment")
    if profile["schema"] != SCHEMA:
        raise ValueError(f"camera experiment schema must be {SCHEMA!r}")
    if not isinstance(profile["id"], str) or not profile["id"].strip():
        raise ValueError("camera experiment id must be a nonempty string")
    basis = profile["basis"]
    _exact_fields(basis, _BASIS_FIELDS, "basis")
    if (basis["kind"] != "engineering_assumption"
            or basis["not_sony_measurement"] is not True):
        raise ValueError("basis must explicitly identify engineering assumptions, "
                         "not Sony measurements")
    if not isinstance(basis["description"], str) or not basis["description"].strip():
        raise ValueError("basis.description must explain the experiment assumptions")
    names = profile["active_camera_names"]
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise ValueError("active_camera_names must be a JSON array of strings")
    if len(set(names)) != len(names):
        raise ValueError("active_camera_names contains duplicate names")
    unknown = set(names) - set(CAMERA_NAMES)
    if unknown:
        raise ValueError(f"Unknown camera chart names: {sorted(unknown)!r}")
    fixed = profile["fixed_rig_values"]
    if not isinstance(fixed, dict):
        raise ValueError("fixed_rig_values must be a JSON object")
    unknown_fixed = set(fixed) - _SCALAR_STORAGE_INDEX.keys()
    if unknown_fixed:
        raise ValueError("Only independent scalar rig values may be overridden; "
                         f"unsupported names: {sorted(unknown_fixed, key=str)!r}")
    conflict = set(names) & fixed.keys()
    if conflict:
        raise ValueError(f"Camera names cannot be both active and fixed: {sorted(conflict)!r}")
    if profile["scene_mode"] not in ("frozen", "joint_current"):
        raise ValueError("scene_mode must be explicitly frozen or joint_current")
    result = deepcopy(profile)
    result["fixed_rig_values"] = {
        name: _finite_scalar(value, f"fixed_rig_values.{name}")
        for name, value in fixed.items()
    }
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def read_camera_experiment(path):
    """Read one explicit JSON file; reject duplicate keys and nonfinite values.

    No profile is selected implicitly. Layout compatibility is checked when the
    profile is applied to a completed 38/43-value rig, after any resume load.
    """
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        profile = json.load(stream, object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant)
    return _validated_profile(profile)


def _rig_copy(rig):
    array = np.asarray(rig)
    if array.shape not in ((38,), (43,)) or array.dtype.kind not in "iuf":
        raise ValueError("rig must be a numeric vector with exactly 38 or 43 stored values")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("rig contains nonfinite values")
    return result


def camera_chart_size(rig):
    """Return 27/31 independent directions, never the 38/43 storage length."""
    return 31 if len(_rig_copy(rig)) == 43 else 27


def apply_camera_experiment(rig, profile):
    """Return (copied rig, ordered active chart indices, JSON-safe metadata).

    Active names are ordered by the native chart, independent of JSON ordering.
    An inactive name without a scalar override retains the supplied initial
    storage and omits its independent chart direction, not measured truth.
    The native nonlinear retraction still enforces rotation/axis relations;
    an inactive axis coordinate does not freeze every related storage element.
    No 38-value rig is silently extended to 43, and fref/radius/wide_boundary
    cannot be changed through this interface because they are chart conventions.
    scene_mode is returned for the runner to enforce; this module has no scene.
    """
    config = _validated_profile(profile)
    copied = _rig_copy(rig)
    size = 31 if copied.size == 43 else 27
    available = CAMERA_NAMES[:size]
    requested = set(config["active_camera_names"]) | config["fixed_rig_values"].keys()
    unavailable = requested - set(available)
    if unavailable:
        raise ValueError(f"Camera names unavailable in a {copied.size}-value rig: "
                         f"{sorted(unavailable)!r}")
    initial_hash = hashlib.sha256(copied.astype("<f8").tobytes()).hexdigest()
    before = {}
    for name, value in config["fixed_rig_values"].items():
        index = _SCALAR_STORAGE_INDEX[name]
        before[name] = float(copied[index])
        copied[index] = value
    active = tuple(i for i, name in enumerate(available)
                   if name in config["active_camera_names"])
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    inactive = [name for name in available
                if name not in config["active_camera_names"]
                and name not in config["fixed_rig_values"]]
    metadata = {
        "schema": "rinkspace.camera_experiment_applied.v1",
        "profile": config,
        "profile_sha256": hashlib.sha256(canonical).hexdigest(),
        "input_rig_sha256": initial_hash,
        "camera_storage_size": int(copied.size),
        "camera_chart_size": size,
        "active_camera_chart": list(active),
        "active_camera_names": [available[i] for i in active],
        "fixed_rig_values_before": before,
        "inactive_retained_from_input": inactive,
        "inactive_scalar_values": {
            name: float(copied[_SCALAR_STORAGE_INDEX[name]])
            for name in inactive if name in _SCALAR_STORAGE_INDEX
        },
        "scene_mode": config["scene_mode"],
        "physical_camera_validated": False,
    }
    return copied, active, metadata


def load_camera_experiment(path, rig):
    """Read and apply an explicitly selected profile, without writing any file."""
    return apply_camera_experiment(rig, read_camera_experiment(path))


def validate_experiment_resume(saved_hash, metadata, *, allow_change=False):
    """Prevent an experimental snapshot silently resuming the unrestricted model.

    An untagged historical snapshot may seed any explicit experiment. A tagged
    snapshot needs the same profile, or an explicit model-change flag. Returns
    whether two tagged/selected profiles differ; no optimizer state is restored.
    """
    if not saved_hash:
        return False
    if metadata is None:
        raise ValueError("Experimental snapshot requires --camera-experiment; "
                         "the original unrestricted model will not be reopened implicitly")
    changed = saved_hash != metadata["profile_sha256"]
    if changed and not allow_change:
        raise ValueError("Snapshot camera experiment differs from the selected profile; "
                         "use --allow-experiment-change to declare a new model seeded "
                         "from these numeric parameters")
    return changed


def guard_experiment_run_directory(run_directory, profile_path):
    """Reject reuse of a baseline/other experiment's output before any writes.

    A fresh prepared input directory is allowed. A directory with checkpoints or
    diagnostics is reusable only for the same explicitly selected profile. A
    model change must use a new candidate directory, even when its numeric seed
    comes from a tagged snapshot of another model.
    """
    config = read_camera_experiment(profile_path)
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    expected = hashlib.sha256(canonical).hexdigest()
    directory = Path(run_directory)
    for name, key in (("activity.json", "camera_experiment"),
                      ("activity_pixels.json", "camera_experiment"),
                      ("fit_input_validation.json", "camera_experiment_applied_before_validation")):
        path = directory / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            saved = data.get(key) or {}
            if saved.get("profile_sha256") != expected:
                raise ValueError("Camera experiment requires a separate prepared --run-dir; "
                                 f"existing {name} belongs to the baseline or another profile")
    latest = directory / "accepted_pixels_latest.npz"
    if latest.exists():
        with np.load(latest, allow_pickle=False) as saved:
            digest = str(saved["camera_experiment_sha256"]) if "camera_experiment_sha256" in saved else None
        if digest != expected:
            raise ValueError("Camera experiment would overwrite a baseline/other-profile accepted state; "
                             "prepare a separate --run-dir")


def _column_layout(rig, active_camera_indices, motion_columns, scene_columns):
    size = camera_chart_size(rig)
    indices = tuple(active_camera_indices)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
           or not 0 <= i < size for i in indices):
        raise ValueError(f"active_camera_indices must contain chart indices in [0, {size})")
    if len(set(indices)) != len(indices):
        raise ValueError("active_camera_indices contains duplicates")
    counts = (motion_columns, scene_columns)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 0
           for n in counts):
        raise ValueError("motion_columns and scene_columns must be nonnegative integers")
    return size, tuple(int(i) for i in indices), int(motion_columns), int(scene_columns)


def build_camera_directions(rig, active_camera_indices, chart_scales, *,
                            motion_columns, scene_columns):
    """Build compact rd and its camera-column indices for the native runner.

    Output rd has shape (27 or 31, motion_columns + n_active + scene_columns).
    Only selected camera directions allocate solver columns. chart_scales must
    be an explicit full-chart vector of finite positive solver step scales;
    these scales are not physical bounds or statistical priors.
    """
    size, indices, nm, ns = _column_layout(
        rig, active_camera_indices, motion_columns, scene_columns)
    scales = np.asarray(chart_scales, dtype=np.float64)
    if scales.shape != (size,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"chart_scales must contain {size} finite positive values")
    q = nm + len(indices) + ns
    if q == 0:
        raise ValueError("The native retained problem requires at least one solver column")
    directions = np.zeros((size, q), dtype=np.float64)
    columns = np.arange(nm, nm + len(indices), dtype=np.int64)
    rows = np.asarray(indices, dtype=np.int64)
    directions[rows, columns] = scales[rows]
    return directions, columns


def retained_column_indices(rig, active_camera_indices, *,
                            motion_columns, scene_columns):
    """Select a compact layout from [motion, ALL camera chart, selected scene].

    Use this optional helper to slice existing direction arrays consistently:
    ``old_motion[..., keep]``, ``old_rd[:, keep]``, ``old_sd[:, keep]``.
    It is a column selection, not a zero mask. Set scene_columns=0 only after
    applying scene_mode; it does not remove scene columns from an older array.
    """
    size, indices, nm, ns = _column_layout(
        rig, active_camera_indices, motion_columns, scene_columns)
    return np.asarray((*range(nm), *(nm + i for i in indices),
                       *range(nm + size, nm + size + ns)), dtype=np.int64)
