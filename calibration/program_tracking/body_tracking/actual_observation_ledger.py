"""Export the actual in-memory source ledger, retaining the original file identity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def source_ledger_export(observations, packet_pts, objective_frames, *, ledger_sha256):
    """Export the actual loaded ledger, separating admission from objective use.

    Must be called with problem.body.obs, not reference.p.source_qualified or
    the export_source packet created before later visibility edits.
    """
    pts = np.asarray(packet_pts)
    if pts.ndim != 1 or pts.dtype.kind not in 'iu' or not len(pts) or np.any(np.diff(pts) <= 0):
        raise ValueError('Strictly increasing integer packet PTS required')
    if not np.array_equal(np.asarray(observations['pts']), pts):
        raise ValueError('Actual observation ledger and exported source PTS differ')
    use = np.asarray(observations['use'])
    uv, sigma = np.asarray(observations['uv']), np.asarray(observations['sigma_px'])
    if use.dtype.kind != 'b' or use.shape != (len(pts), 17):
        raise ValueError('Actual COCO17 admission mask required')
    if uv.shape != (*use.shape, 2) or sigma.shape != use.shape:
        raise ValueError('Actual source UV/sigma shapes differ from the ledger')
    if not np.isfinite(uv[use]).all() or not np.isfinite(sigma[use]).all() or np.any(sigma[use] <= 0):
        raise ValueError('Admitted source measurements must have finite UV and positive sigma')
    frames = np.asarray(objective_frames)
    if frames.ndim != 1 or frames.dtype.kind not in 'iu' or np.any(frames < 0) or np.any(frames >= len(pts)):
        raise ValueError('Objective frames must be valid source indices')
    if not isinstance(ledger_sha256, str) or len(ledger_sha256) != 64 or any(c not in '0123456789abcdef' for c in ledger_sha256):
        raise ValueError('Exact observation-ledger SHA256 required')
    active = np.zeros(len(pts), bool)
    active[frames] = True
    return dict(source_observation_use=use.copy(),
                source_observation_objective_use=use & active[:, None],
                source_uv_native=uv.copy(), source_sigma_px=sigma.copy(),
                source_observation_ledger_sha256=np.array(ledger_sha256),
                source_observation_export_schema=np.array('rinkspace.actual-source-ledger.v1'),
                source_observation_use_semantics=np.array('actual loaded ledger admission; source_observation_objective_use additionally restricts the objective source frames'))


def update_actual_observation_ledger(packet, body):
    """Update an export from body.obs and the actual cached objective indices.

    The file SHA names original observations.npz. The loaded-array SHA additionally
    records later in-memory visibility, association, UV and sigma changes.
    """
    original = packet.get('source_observation_use')
    fields = source_ledger_export(body.obs, packet['pts'], body.frames,
        ledger_sha256=hashlib.sha256(Path(body.ledger_path).read_bytes()).hexdigest())
    if hasattr(body, 'of') and hasattr(body, 'oj'):
        frames = np.asarray(body.frames)
        of, oj = np.asarray(body.of), np.asarray(body.oj)
        if (of.ndim != 1 or oj.shape != of.shape or of.dtype.kind not in 'iu'
                or oj.dtype.kind not in 'iu' or np.any(of < 0) or np.any(of >= len(frames))
                or np.any(oj < 0) or np.any(oj >= 17)):
            raise ValueError('Invalid actual body objective scatter indices')
        objective = np.zeros_like(fields['source_observation_use'])
        objective[frames[of], oj] = True
        if np.any(objective & ~fields['source_observation_use']):
            raise ValueError('Body objective scatter is stale after ledger admission edits')
        fields['source_observation_objective_use'] = objective
    if original is not None and 'source_observation_initial_use' not in packet:
        packet['source_observation_initial_use'] = np.asarray(original).copy()
    arrays = hashlib.sha256()
    for key in ('pts', 'use', 'uv', 'sigma_px'):
        value = np.ascontiguousarray(body.obs[key])
        arrays.update(key.encode('ascii'))
        arrays.update(value.dtype.str.encode('ascii'))
        arrays.update(str(value.shape).encode('ascii'))
        arrays.update(value.tobytes())
    fields['source_observation_loaded_arrays_sha256'] = np.array(arrays.hexdigest())
    packet.update(fields)
    return packet


def admitted_support_image_mask(packet_pts, factor):
    """Report the actual admitted fit factors, including visibility/reliability."""
    pts = np.asarray(packet_pts)
    mask = np.zeros((len(pts), 2), bool)
    for row, observation, _ in factor.rows:
        if observation.role == 'fit' and row.get('optimization_enabled', True):
            mask[pts == observation.pts, observation.foot] = True
    return mask
