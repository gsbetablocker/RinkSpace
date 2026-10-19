"""Explicit shared zoom law through the existing physical-camera interface.

The driver is final-effective log focal, not a measured lens encoder. Four
shared slopes describe principal point, radial distortion and axial pupil
motion. A fixed tripod and constant mechanical lever remain distinct from
the zoom-dependent optical centre. No per-exposure translations are added.
"""
from dataclasses import replace
import numpy as np

PARAMETERS = ("principal_x_px_per_logf", "principal_y_px_per_logf",
              "k1_milli_per_logf", "axial_working_m_per_logf")
SHARED_NAMES = ("principal_x_fraction_degree_1", "principal_y_fraction_degree_1",
                "k1_degree_1", "axial_degree_1_scene_div100")


def read_zoom_configuration(reference, binding):
    from .official_static_surface_dataset import read_reference
    config = read_reference(reference)
    if (config.get("schema") != "rinkspace.first-shot-shared-zoom-camera.v1"
            or config.get("source_video_sha256") != binding["source"]["sha256"]
            or config.get("source_clock") != binding["source_clock"]
            or config.get("parameters") != list(PARAMETERS)
            or config.get("driver") != "final_effective_domain_log_focal_ratio"
            or config.get("per_exposure_extra_parameters") != 0
            or config.get("manufacturer_calibrated_coefficients") is not False):
        raise ValueError("Explicit source-bound shared physical zoom law required")
    if not np.isfinite(float(config["reference_domain_log_focal_ratio"])):
        raise ValueError("Finite fixed zoom reference required")
    return config


def with_zoom_calibration(state, q, calibration):
    columns = getattr(state, "zoom_parameter_indices", ())
    if not len(columns):
        return calibration
    slopes = np.asarray(q)[columns]
    # The historical first-shot base was explicitly constant. Do not silently
    # replace an already calibrated zoom curve with another chart.
    if (any(np.any(np.asarray(getattr(calibration, k))[1:] != 0.) for k in
            ("principal_x_fraction", "principal_y_fraction", "k1", "k2"))
            or np.any(np.asarray(calibration.axial) != 0.)):
        raise ValueError("Zoom extension expects the authenticated constant base chart")
    return replace(calibration,
        q_reference=float(state.zoom_configuration["reference_domain_log_focal_ratio"]),
        principal_x_fraction=(calibration.principal_x_fraction[0], slopes[0] / 1920.),
        principal_y_fraction=(calibration.principal_y_fraction[0], slopes[1] / 1080.),
        k1=(calibration.k1[0], .001 * slopes[2]),
        axial=(0., slopes[3] / state.scene_unit_working_m))


def zoom_shared_mapping(state, calibration):
    columns = np.asarray(getattr(state, "zoom_parameter_indices", ()), dtype=int)
    if not len(columns):
        return columns, np.empty(0, dtype=int), np.empty(0)
    names = calibration.shared_state_names()
    selected = np.asarray([names.index(name) for name in SHARED_NAMES], dtype=int)
    scales = np.array([1. / 1920., 1. / 1080., .001,
                       1. / (100. * state.scene_unit_working_m)])
    return columns, selected, scales


def append_zoom_initializer(state, q):
    columns = np.asarray(getattr(state, "zoom_parameter_indices", ()), dtype=int)
    if not len(columns) or q.shape != (int(columns[0]),):
        return q, None
    original = q.copy()
    q = np.r_[q, np.zeros(len(columns))]
    assert np.array_equal(q[:len(original)], original)
    return q, dict(columns=columns.tolist(), values=q[columns].tolist(),
        parameters=list(PARAMETERS), reference=state.zoom_reference,
        existing_coordinates_byte_exact=True, parameter_values_only=True)


def _zoom_driver_limits(state, q):
    focal = np.asarray(q)[state.view_start + 3:state.view_stop:4]
    limits = np.r_[focal, 1.5 * focal[0] - .5 * focal[1],
                   1.5 * focal[-1] - .5 * focal[-2]]
    return limits - float(state.zoom_configuration["reference_domain_log_focal_ratio"])


def _mount_polygon(state, q):
    # Include the reference mechanical lever as well as every optical centre.
    driver = _zoom_driver_limits(state, q)
    x = np.array([0., driver.min(), driver.max()])
    a = np.c_[np.ones(3), x]
    return np.r_[a, -a], np.r_[np.full(3, state.mount_absolute_lower[2]),
                              np.full(3, -state.mount_absolute_upper[2])]


def zoom_mount_linear_constraints(state, q, selected):
    """Current-focal shared half-spaces; the actual trial is checked again.

    RF4 is eliminated by the public solver. Its change of focal driver is
    therefore handled by the explicit two-coordinate trial retraction below,
    with the original full-state cost and nonlinear guard still authoritative.
    """
    a, b = _mount_polygon(state, q)
    columns = [123, int(state.zoom_parameter_indices[3])]
    selected = np.asarray(selected)
    positions = [int(np.flatnonzero(selected == c)[0]) for c in columns]
    value = np.array([state.calibration(q).lever_xyz[2] * state.scene_unit_working_m,
                      q[columns[1]]])
    matrix = np.zeros((len(b), len(selected)))
    matrix[:, positions] = a
    return dict(matrix=matrix, lower=b-a@value,
        report=dict(kind="existing_mount_total_offset_at_current_zoom_extrema",
                    complete_state_parameter_indices=columns,
                    trial_focal_change_requires_original_envelope_recheck=True))


def recover_zoom_mount_trial(state, parent, proposed):
    """Exact Euclidean projection of two shared coordinates onto saved ranges.

    At a proposed RF4, the envelope is a convex polygon in mechanical lever-z
    and axial slope (both in the recorded working length chart). Enumerating
    its faces and vertices gives its closest point without changing RF4 or
    adding a penalty. The public solver evaluates this actual complete state,
    records the retraction, and accepts only with its unchanged source gate.
    """
    q = np.asarray(proposed).copy()
    a, b = _mount_polygon(state, q)
    column = int(state.zoom_parameter_indices[3])
    lever = state.calibration(q).lever_xyz[2] * state.scene_unit_working_m
    original = np.array([lever, q[column]])
    candidates = []
    def retain(value):
        if np.all(a@value >= b-1e-14):
            candidates.append(value)
    retain(original)
    for row, limit in zip(a, b, strict=True):
        retain(original + (limit-row@original)/(row@row)*row)
    for i in range(len(b)):
        for j in range(i):
            pair = a[[i,j]]
            if abs(np.linalg.det(pair)) > 1e-14:
                retain(np.linalg.solve(pair, b[[i,j]]))
    if not candidates:
        raise ValueError("Existing total optical offset polygon is empty")
    chosen = min(candidates, key=lambda value: np.sum((value-original)**2))
    q[123] += chosen[0]-original[0]
    q[column] = chosen[1]
    return dict(q=q, report=dict(kind="existing_mount_exact_two_coordinate_projection",
        coordinate_units="working metres and working metres per log focal",
        before=original.tolist(), after=chosen.tolist(),
        modified_parameter_indices=np.flatnonzero(q != proposed).tolist(),
        original_source_cost_and_complete_physical_guard_required=True))


def zoom_mount_envelope(state, q, calibration):
    """Apply the saved finite camera envelope to the total optical offset.

    Linear log-focal interpolation attains its extrema at the native knots
    or the two supported terminal half intervals. This checks all rows without
    inventing separate slope bounds or treating a fitted slope as measurement.
    """
    x = _zoom_driver_limits(state, q)
    axial = np.polynomial.polynomial.polyval(x, calibration.axial)
    lever = np.asarray(calibration.lever_xyz) * state.scene_unit_working_m
    envelope = np.repeat(lever[None], 2, axis=0)
    envelope[:, 2] += np.array([axial.min(), axial.max()]) * state.scene_unit_working_m
    low = np.maximum(state.mount_absolute_lower - envelope.min(0), 0.)
    high = np.maximum(envelope.max(0) - state.mount_absolute_upper, 0.)
    return dict(valid=bool(np.all(low <= 1e-12) and np.all(high <= 1e-12)),
        total_offset_min_working_m=envelope.min(0).tolist(),
        total_offset_max_working_m=envelope.max(0).tolist(),
        axial_displacement_range_working_m=(np.array([axial.min(), axial.max()]) * state.scene_unit_working_m).tolist(),
        all_native_rows_and_supported_terminal_intervals_checked=True,
        verified_metre_scale=False)
