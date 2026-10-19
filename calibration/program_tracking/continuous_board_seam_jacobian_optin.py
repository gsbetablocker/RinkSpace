"""Opt-in analytic branch Jacobian at verified, nearly C1 board-ring seams.

The factor's actual pixels, global first hit, residual, panel choice, and
visibility checks are untouched. A current-panel analytic Jacobian is a valid
one-sided/Clarke derivative only after this module verifies that the neighboring
panel is allowed by the original factor, both pieces meet, and both source-state directional
limits agree at the current accepted state. The default guard stays installed
until ``install_board_seam_optin`` is explicitly called.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from threading import local

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()

import numpy as np

from continuous_brown_board_pair import ContinuousBrownBoardPair
from reference_board_pair import BoardIntersectionError


_PINNED = {
    'continuous_brown_board_pair.py':
        '2a9b52e34df9813fbf99f7c002cf85bdabcd46670bf64ef078101e5a3a5adb0d',
    'native_board_pair.py':
        'ba95e259756c4a8d39282d829477633a2a287c708e609044913a740d9dc47759',
    'reference_board_pair.py':
        '1cfa3a2071a0be73f4c00b7e4416b124cdd65279b05eb5928fde0dc3472ccfbf',
}
_ORIGINAL_RAW = ContinuousBrownBoardPair._raw_jacobians
_ORIGINAL_SAME = ContinuousBrownBoardPair._same_local_plane
_INSTALLED = False
_COUNTS = dict(seam_rows_validated=0, rejected_geometry=0,
               rejected_directional=0, unchanged_callbacks=0)
_PROBES = (1e-6, 1e-7)


def _context(pair):
    value = getattr(pair, '_seam_optin_context', None)
    if value is None:
        value = local()
        pair._seam_optin_context = value
    return value


def _same_local_plane(self, baseline, other):
    result = _ORIGINAL_SAME(self, baseline, other)
    permit = getattr(_context(self), 'permit', None)
    if permit is not None:
        baseline = np.asarray(baseline, np.int32)
        other = np.asarray(other, np.int32)
        if baseline.shape == permit['panels'].shape == other.shape:
            result |= (permit['rows'] & (baseline == permit['panels'])
                       & (other == permit['neighbors']))
    return result


def _limit_probe(pair, states, row, panel, neighbor, base_uv):
    source, target = states
    pixel = pair.xy0[row:row+1]
    limits = []
    switched = False
    for step in _PROBES:
        by_axis = []
        for axis in range(len(source)):
            sides = []
            for sign in (-1, 1):
                moved = np.asarray(source, np.float64).copy()
                moved[axis] += sign*step
                try:
                    uv, panel_id = pair._project_pixels([moved, target], pixel)
                except BoardIntersectionError as exc:
                    raise BoardIntersectionError(
                        'seam_directional_domain_' + exc.reason, [row]) from exc
                branch = int(panel_id[0])
                if branch not in (panel, neighbor) or not np.isfinite(uv).all():
                    raise BoardIntersectionError('seam_directional_wrong_panel', [row])
                switched |= branch == neighbor
                slope = (uv[0]-base_uv)/(sign*step)
                sides.append((branch, slope, float(np.linalg.norm(uv[0]-base_uv))))
            minus, plus = sides
            scale = max(float(np.linalg.norm(minus[1])),
                        float(np.linalg.norm(plus[1])))
            # This is a numerical near-C1 gate, not a claim that the polygon
            # has an exact classical derivative at its vertex.
            if (max(minus[2], plus[2]) > .1
                    or np.linalg.norm(minus[1]-plus[1]) > .1+1e-3*scale):
                raise BoardIntersectionError('seam_directional_kink', [row])
            by_axis.append(sides)
        limits.append(by_axis)
    if not switched:
        raise BoardIntersectionError('seam_neighbor_not_probed', [row])
    for axis, (coarse, fine) in enumerate(zip(*limits)):
        for old, new in zip(coarse, fine):
            # A tenfold smaller physical perturbation must approach the same
            # UV value and stable one-sided slope.
            scale = max(float(np.linalg.norm(old[1])),
                        float(np.linalg.norm(new[1])))
            if (new[2] > .2*old[2]+1e-8
                    or np.linalg.norm(old[1]-new[1]) > .1+1e-3*scale):
                raise BoardIntersectionError('seam_directional_unstable', [row])
    return limits[0]


def _validated_seam(self, states, hits, row, panel, neighbor):
    if not self._allowed[panel] or not self._allowed[neighbor]:
        raise BoardIntersectionError('seam_neighbor_not_allowed', [row])
    count = len(self._edges)
    fraction = float(hits.panel_fraction[row])
    at_end = fraction >= 1.-self.seam_fraction_margin
    seam = ((panel+1) % count) if at_end else panel
    if (not np.isfinite(fraction) or
            not ((at_end and neighbor == (panel+1) % count)
                 or (not at_end and neighbor == (panel-1) % count))):
        raise BoardIntersectionError('seam_nonadjacent_panel', [row])
    point_xy = self._starts[seam]
    if (np.linalg.norm(hits.xyz[row, :2]-point_xy) > 1e-6
            or not self.bottom[seam, 2]-1e-7 <= hits.xyz[row, 2]
            <= self.top[seam, 2]+1e-7):
        raise BoardIntersectionError('seam_not_at_shared_vertical_edge', [row])
    edges = self._edges[[panel, neighbor]]
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) <= 1e-10:
        raise BoardIntersectionError('seam_degenerate_panel', [row])
    cosine = float(np.clip(np.dot(*edges)/np.prod(lengths), -1., 1.))
    angle = float(np.degrees(np.arccos(cosine)))
    # This opt-in handles the closely tessellated curved board, not a sharp
    # corner or a transition to another physical wall.
    if not angle <= 2.:
        raise BoardIntersectionError('seam_sharp_physical_corner', [row])
    base_uv, base_panel = self._project_pixels(
        states, self.xy0[row:row+1])
    if int(base_panel[0]) != panel or not np.isfinite(base_uv).all():
        raise BoardIntersectionError('seam_base_first_hit_changed', [row])
    limits = _limit_probe(self, states, row, panel, neighbor, base_uv[0])
    return limits


def _raw_jacobians(self, states):
    source, target = (np.asarray(q, np.float64) for q in states)
    if source.shape != (15,) or target.shape != (15,):
        _COUNTS['unchanged_callbacks'] += 1
        return _ORIGINAL_RAW(self, states)
    hits = self.source_hits(source)
    self._require(hits)
    fraction = hits.panel_fraction
    smooth = (np.isfinite(fraction)
              & (fraction > self.seam_fraction_margin)
              & (fraction < 1.-self.seam_fraction_margin))
    at_start = np.isfinite(fraction) & (fraction <= self.seam_fraction_margin)
    at_end = np.isfinite(fraction) & (fraction >= 1.-self.seam_fraction_margin)
    previous = (hits.panel-1) % len(self._edges)
    following = (hits.panel+1) % len(self._edges)
    smooth |= at_start & _ORIGINAL_SAME(self, hits.panel, previous)
    smooth |= at_end & _ORIGINAL_SAME(self, hits.panel, following)
    failing = np.flatnonzero(~smooth)
    if not len(failing):
        _COUNTS['unchanged_callbacks'] += 1
        return _ORIGINAL_RAW(self, states)
    panels = hits.panel.copy()
    neighbors = np.where(at_start, previous, following)
    permit = np.zeros(len(fraction), bool)
    checks = {}
    for row in failing:
        row = int(row)
        try:
            checks[row] = _validated_seam(
                self, [source, target], hits, row, int(panels[row]),
                int(neighbors[row]))
        except BoardIntersectionError as exc:
            _COUNTS['rejected_geometry' if exc.reason.startswith('seam_')
                    and exc.reason not in ('seam_directional_kink',
                                           'seam_directional_unstable')
                    else 'rejected_directional'] += 1
            raise
        permit[row] = True
    context = _context(self)
    if getattr(context, 'permit', None) is not None:
        raise RuntimeError('Nested seam Jacobian callback')
    context.permit = dict(rows=permit, panels=panels, neighbors=neighbors)
    try:
        result = _ORIGINAL_RAW(self, states)
    finally:
        context.permit = None
    source_jac = np.asarray(result[0], np.float64).reshape(len(self.xy0), 2, -1)
    for row, limits in checks.items():
        panel = int(panels[row])
        for axis, sides in enumerate(limits):
            matching = [item[1] for item in sides if item[0] == panel]
            if not matching:
                raise BoardIntersectionError('seam_active_branch_not_probed', [row])
            slope = matching[0]
            analytic = source_jac[row, :, axis]
            scale = max(float(np.linalg.norm(slope)),
                        float(np.linalg.norm(analytic)))
            if np.linalg.norm(slope-analytic) > .1+1e-3*scale:
                raise BoardIntersectionError('seam_analytic_branch_disagrees', [row])
    _COUNTS['seam_rows_validated'] += len(checks)
    self.last_jacobian_counts = {
        **self.last_jacobian_counts,
        'seam_current_panel_clarke_rows': len(checks),
        'seam_limit_steps': list(_PROBES),
    }
    return result


def install_board_seam_optin():
    """Explicitly enable guarded current-first-hit branch Jacobians."""
    global _INSTALLED
    if _INSTALLED:
        return metadata()
    for name, expected in _PINNED.items():
        path = Path(__file__).with_name(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned board callback changed: ' + name)
    if (ContinuousBrownBoardPair._raw_jacobians is not _ORIGINAL_RAW
            or ContinuousBrownBoardPair._same_local_plane is not _ORIGINAL_SAME):
        raise RuntimeError('Board callback already changed by another opt-in')
    ContinuousBrownBoardPair._same_local_plane = _same_local_plane
    ContinuousBrownBoardPair._raw_jacobians = _raw_jacobians
    _INSTALLED = True
    return metadata()


def metadata():
    return dict(schema='rinkspace.board-seam-current-panel-clarke-optin.v1',
                installed=_INSTALLED, pinned_callbacks=dict(_PINNED),
                optin_path=str(Path(__file__).resolve()),
                optin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                permitted_angle_deg=2., directional_steps=list(_PROBES),
                counters=dict(_COUNTS), residual_changed=False,
                first_hit_changed=False, source_uv_changed=False,
                default_callbacks_changed_on_disk=False)
