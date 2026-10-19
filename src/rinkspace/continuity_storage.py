"""Lossless geometry records with compressed observed point pairs."""
from pathlib import Path
import numpy as np

POINT_FIELDS = ('fit_current_xy', 'fit_keyframe_xy', 'fit_reference_xy')


def save_observations(frames, path):
    arrays = {}
    for row in frames:
        # Keep the fixed-reference correspondences in JSON for readability;
        # repetitive per-frame local edges are compressed losslessly as f64.
        fit = row.get('local_registration', {})
        for field in POINT_FIELDS:
            if field in fit:
                key = f"frame_{row['sample_index']}_{field}"
                arrays[key] = np.asarray(fit.pop(field), dtype=np.float64).reshape(-1, 2)
                fit.setdefault('point_arrays', {})[field] = key
        if fit.get('point_arrays'):
            fit['point_storage'] = Path(path).name
    np.savez_compressed(path, **arrays)
    return {'file': Path(path).name, 'array_count': len(arrays), 'dtype': 'float64',
            'direction': 'current_to_observed_keyframe; component_reference_matrix_also_retained',
            'all_observed_inlier_pairs_retained': True}


def load_observations(frames, path):
    with np.load(path, allow_pickle=False) as arrays:
        for row in frames:
            fit = row.get('local_registration', {})
            for field, key in fit.get('point_arrays', {}).items():
                fit[field] = arrays[key].tolist()
    return frames
