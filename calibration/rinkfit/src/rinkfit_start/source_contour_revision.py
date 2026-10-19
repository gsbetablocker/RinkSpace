"""Explicit finite contour deletions with source provenance and station transport.

A revision only retains existing vertices in separate open paths. It never adds
vertices, bridges a source gap, changes a parameter block, or alters source UV.
Rows in a deleted interval have no transported station and require a fresh
finite-path search against their original observation in the current model.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np

SCHEMA = 'rinkfit.source-contour-path-revision.v1'


def _patch(revision):
    if revision.get('schema') != SCHEMA:
        raise ValueError('unsupported source contour revision')
    patch = revision['template_path_patch']
    points = np.asarray(patch['expected_old_points'], float)
    n = len(points) - 1
    if points.ndim != 2 or points.shape[1] != 2 or n < 1 or not np.isfinite(points).all():
        raise ValueError('invalid original finite contour')
    if patch.get('old_path_closed') is not False:
        raise ValueError('source deletion requires an explicit open original path')
    last = -1
    for branch, part in enumerate(patch['retained_paths']):
        indices = part['old_vertex_indices']
        if len(indices) < 2 or any(type(i) is not int for i in indices):
            raise ValueError('retained path requires original vertices')
        if indices != list(range(indices[0], indices[-1] + 1)) or not (0 <= indices[0] < indices[-1] <= n):
            raise ValueError('a retained path cannot bridge or reorder original vertices')
        if indices[0] <= last or part.get('new_path_index') != branch or part.get('closed') is not False:
            raise ValueError('retained paths must be separate ordered open intervals')
        if not part.get('source_evidence'):
            raise ValueError('retained path lacks source evidence')
        last = indices[-1]
    if not patch['retained_paths']:
        raise ValueError('revision retains no physical path')
    return patch, points


def apply_source_contour_revision(model, revision, atol=1e-11):
    """Return an independent compatible-model copy; all blocks remain exact."""
    patch, expected = _patch(revision)
    found = [i for i, m in enumerate(model['materials']) if m['label'] == patch['material_label']]
    if len(found) != 1:
        raise ValueError('source revision material identity is absent or ambiguous')
    mi = found[0]
    old_paths = model['materials'][mi]['paths']
    if len(old_paths) != 1 or patch['old_path_index'] != 0:
        raise ValueError('source revision requires the original single open path')
    old = old_paths[0]
    if old.get('kind') not in (2, 'board_path') or old.get('closed') is not False:
        raise ValueError('source revision requires the recorded open board path')
    current = np.asarray(old['points'], float)
    if current.shape != expected.shape or not np.allclose(current, expected, atol=atol, rtol=0):
        raise ValueError('source revision original vertex signature mismatch')
    for key in ('geometry_block', 'advertisement_block'):
        i = old.get(key)
        if type(i) is not int or not 0 <= i < len(model['blocks']):
            raise ValueError('source revision requires valid original parameter block references')
    result = deepcopy(model)
    paths = []
    for part in patch['retained_paths']:
        path = deepcopy(old)
        path['points'] = [deepcopy(old['points'][i]) for i in part['old_vertex_indices']]
        path['id'] = old.get('id', 'path0') + '/' + part['path_suffix']
        path['closed'] = False
        path['source_contour_revision'] = {
            'revision_id': revision['revision_id'],
            'source_template_contour': patch['source_template_contour'],
            'old_vertex_indices': part['old_vertex_indices'],
            'source_evidence': part['source_evidence'],
            'no_gap_closing_edge': True,
        }
        paths.append(path)
    result['materials'][mi]['paths'] = paths
    result.setdefault('source_contour_revisions', []).append({
        'revision_id': revision['revision_id'], 'material_label': patch['material_label'],
        'source_scene_sha256': revision['source_scene']['sha256'],
        'old_paths': 1, 'new_paths': len(paths), 'parameter_blocks_changed': False,
        'fresh_RS_S_Q_required': True, 'PHR_or_linearization_resume_permitted': False,
    })
    return result


def map_retained_stations(old_path, old_station, revision):
    """Map only retained material positions; deleted positions return -1/NaN.

    Both finite endpoints survive. No nearest endpoint clipping is performed.
    This transport is a seed, never a substitute for current-model RS/S/Q.
    """
    patch, points = _patch(revision)
    branch, station = np.broadcast_arrays(np.asarray(old_path), np.asarray(old_station, float))
    if not np.isfinite(branch).all() or not np.isfinite(station).all():
        raise ValueError('nonfinite original path or station')
    if np.any(branch != np.floor(branch)) or np.any(branch != patch['old_path_index']):
        raise ValueError('unrecognized original path')
    if np.any((station < 0) | (station > 1)):
        raise ValueError('original station outside finite domain')
    coordinate = station * (len(points) - 1)
    new_branch = np.full(station.shape, -1, np.int32)
    new_station = np.full(station.shape, np.nan, float)
    for part in patch['retained_paths']:
        a, b = part['old_vertex_indices'][0], part['old_vertex_indices'][-1]
        keep = (coordinate >= a) & (coordinate <= b)
        new_branch[keep] = part['new_path_index']
        new_station[keep] = (coordinate[keep] - a) / (b - a)
    return new_branch, new_station, new_branch >= 0
