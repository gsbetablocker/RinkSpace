"""Optional, strictly post-fit comparison with a read-only RinkSpace export."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .solver import Layout, fit_one
from .geometry import symmetric_errors, error_stats
from .io import write_json, sha256


def read_current_camera(path: Path, frames: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        pts = z['source_pts']; values = z['instantaneous_camera']; status = z['camera_status']
    if values.shape != (len(pts), 24) or status.shape != (len(pts),):
        raise ValueError('Expected 1ce8ae1 instantaneous-camera export: N x 24')
    if len(np.unique(pts)) != len(pts): raise ValueError('Duplicate camera PTS')
    lookup = {int(p): k for k, p in enumerate(pts)}
    try: selected = np.array([lookup[int(f['pts'])] for f in frames])
    except KeyError as e: raise ValueError(f'No exact camera snapshot at source PTS {e.args[0]}') from e
    if np.any(status[selected] != 0): raise ValueError('Camera export contains failed selected snapshots')
    v = values[selected]
    if not np.isfinite(v).all() or np.any(v[:, 16:18] <= 0): raise ValueError('Invalid camera export')
    R = v[:, 4:13].reshape(-1, 3, 3)
    if np.max(np.abs(R @ np.transpose(R, (0, 2, 1))-np.eye(3))) > 1e-6 or np.any(np.linalg.det(R) < .999):
        raise ValueError('Non-rotation matrix in export')
    return v[:, 16:24].copy(), R


def compare_current(graph: dict, camera: Path, out: Path, max_nfev=180) -> dict:
    meta, data = graph['meta'], graph['data']; before = sha256(camera)
    optics, R = read_current_camera(camera, meta['frames'])
    relative = R @ R[0].T
    for component in meta.get('active_components', [list(range(len(R)))]):
        relative[component] = R[component] @ R[component[0]].T
    raw, valid = symmetric_errors(data, optics, relative)
    test = data['test']; screened = test & data['screened']
    layout = Layout(len(meta['frames']), meta['width'], meta['height'], 'fixed_optics', fixed_optics=optics, anchors=meta.get('anchors'))
    # Both comparisons happen AFTER independent fitting. No current pose used by the independent branch.
    starts = [None, Rotation.from_matrix(relative[layout.rot_frames]).as_rotvec().ravel()]
    runs = [fit_one(layout, meta, data, max_nfev=max_nfev, start=s) for s in starts]
    best = min(runs, key=lambda x: x['summary']['train_robust_cost'])
    summary = dict(schema='rinkspace.ptz-current-comparison.v1', camera_sha256=before,
        columns_source='src/rinkspace/core/native.py evaluate_motion_camera at 1ce8ae1',
        current_rotation_and_optics_midframe=dict(holdout_all=error_stats(raw[test], valid[test]),
            holdout_screened=error_stats(raw[screened], valid[screened])),
        current_optics_frozen_rotations_refit=best['summary'],
        warning='All comparisons assume a common optical centre and global shutter. Differences do not by themselves prove the production camera is wrong. No height/translation is estimated.',
        independent_fit_used_current_parameters=False,
        production_state_changed=False,
        current_fx_over_fy_range=[float(np.min(optics[:, 0]/optics[:, 1])), float(np.max(optics[:, 0]/optics[:, 1]))])
    if sha256(camera) != before: raise RuntimeError('Camera export was modified')
    write_json(out/'current_comparison.json', summary)
    np.savez_compressed(out/'current_comparison.npz', optics=optics, original_rotations=relative,
        refitted_rotations=best['rotations'], original_residual=raw, original_valid=valid,
        refitted_residual=best['residual'], refitted_valid=best['valid'])
    return summary
