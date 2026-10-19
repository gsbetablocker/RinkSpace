"""Actual RGB board-print readouts navigated by the unchanged scene.

Navigation coordinates and normals do not become measured observations.
The caller supplies separately verified physical instance/field-order identity.
This independent delta module does not change the frozen programme readers.
"""
from __future__ import annotations

import numpy as np

from .vision_programme_board_review import _sample


def read_circle_material_interface(image, exclusion, navigation_uv, *, outward_normal=None):
    """Read the named outer circular print against measured exterior board.

    All output locations and blur bounds come from the current source profile.
    Unknown/competing transitions are explicitly absent. No circle/ellipse is
    fitted to produce missing pixels, and no identity is approved here.
    """
    uv = np.asarray(navigation_uv, float)
    minimum_count = 8 if outward_normal is None else 1
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) < minimum_count or not np.isfinite(uv).all():
        raise ValueError('A finite existing circular-model navigation path is required')
    if len(uv)>1 and np.linalg.norm(uv[0]-uv[-1]) < .1:
        uv = uv[:-1]
    if outward_normal is None:
        tangent = np.roll(uv, -1, axis=0)-np.roll(uv, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.c_[tangent[:, 1], -tangent[:, 0]]
        normal[np.sum(normal*(uv-uv.mean(0)), axis=1) < 0] *= -1
    else:
        normal = np.asarray(outward_normal, float)[:len(uv)]
        if normal.shape != uv.shape:
            raise ValueError('Model navigation normal shape differs')
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    offsets = np.arange(-18., 18.001, .5)
    tangent = np.c_[-normal[:, 1], normal[:, 0]]
    xy = uv[:, None, None]+offsets[None, :, None, None]*normal[:, None, None]+np.array([-2., 0., 2.])[None, None, :, None]*tangent[:, None, None]
    colors = np.median(_sample(image, xy.reshape(-1, 3, 2)).reshape(len(uv), len(offsets), 3, 3), axis=2)
    pixels = np.rint(xy).astype(int)
    valid = ((pixels >= 0).all(axis=(1, 2, 3)) & (pixels < [1920, 1080]).all(axis=(1, 2, 3)))
    result = []
    for index in np.flatnonzero(valid):
        p = pixels[index]
        if exclusion[p[..., 1], p[..., 0]].any():
            continue
        current = colors[index]
        # The exterior plateau is observed native board material. Its
        # brightness, colour and stability are read afresh for this query.
        exterior = current[-7:]
        if (exterior.mean(0).min() < 80. or exterior.std(0).max() > 5.
                or np.ptp(exterior.mean(0)) > 18.):
            continue
        values = current.mean(1)
        delta = values[4:]-values[:-4]
        peaks = np.flatnonzero((delta[1:-1] >= delta[:-2]) & (delta[1:-1] >= delta[2:]) & (delta[1:-1] >= 6.))+1
        peaks = peaks[np.abs(offsets[peaks+2]) <= 12.]
        if not len(peaks):
            continue
        # Only the last transition into the measured exterior plateau can be
        # this outer print interface. A later dark component makes it unknown.
        peak = int(peaks[-1]); centre = peak+2
        first = max(0, centre-10)
        valley = first+int(np.argmin(values[first:centre+1]))
        low, high = float(values[valley]), float(np.median(values[-7:]))
        if high-low < 12.:
            continue
        quantiles = []
        for fraction in (.25, .5, .75):
            level = low+fraction*(high-low)
            crossings = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
            crossings = crossings[crossings >= valley]
            if len(crossings) != 1:
                break
            k = int(crossings[0])
            quantiles.append(offsets[k]+.5*(level-values[k])/(values[k+1]-values[k]))
        if len(quantiles) != 3 or abs(quantiles[1]-offsets[centre]) > 3.:
            continue
        result.append(dict(navigation_index=int(index),uv=uv[index]+quantiles[1]*normal[index],
            profile_navigation_normal=normal[index],actual_profile_quantile_uv=uv[index]+np.asarray(quantiles)[:, None]*normal[index],
            actual_readout_blur_interval_px=np.asarray([quantiles[0]-quantiles[1],quantiles[2]-quantiles[1]]),
            actual_inside_BGR=current[valley],actual_exterior_BGR=np.median(exterior,axis=0),
            actual_transition_gradient=float(delta[peak]),actual_source_RGB_measured=True,
            projected_UV_used_as_source_pixel_truth=False,
            current_source_pixel_context_person_exclusion_clear=True,
            physical_material_identity_not_approved_by_profile_reader=True))
    return dict(measurement_method='actual_original_RGB_outer_print_to_stable_board_q25_q50_q75',
        source_profile_sampling_step_px=.5,source_interpolation_grid_px=1./32.,
        source_native_pixel_quantization_step_px=1.,readings=result,
        navigation_query_count=len(uv),unmeasured_navigation_indices=sorted(set(range(len(uv)))-{r['navigation_index'] for r in result}),
        profile_navigation_normals_are_not_source_truth=True,
        no_missing_boundary_UV_created=True)
