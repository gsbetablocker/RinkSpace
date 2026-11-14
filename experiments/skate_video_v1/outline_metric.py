"""Comparable outline distances on one native rolling-shutter projection."""
import numpy as np
from scipy.spatial import cKDTree

from .inputs import distribute


def outline_distances(assembly, vertices, polygon):
    """Return the archived mean and canonical median on the same source mask."""
    if not np.isfinite(vertices).all() or not np.isfinite(polygon).all():
        return np.nan, np.nan
    try:
        boundary = np.concatenate(assembly.raster(vertices))
    except ValueError:
        return np.nan, np.nan
    model = cKDTree(boundary)
    uniform = distribute(polygon, 96, closed=True)
    legacy = .5 * (model.query(uniform)[0].mean() +
                   cKDTree(uniform).query(boundary)[0].mean())
    canonical = (.55 * np.median(model.query(polygon)[0]) +
                 .45 * np.median(cKDTree(polygon).query(boundary)[0]))
    return float(legacy), float(canonical)
