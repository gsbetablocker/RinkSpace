"""Source support integration shared by readers and existing fit adapters.

One actual exposure and authenticated physical curve has one engineering
budget. These are correlated source samples, not independent measurements.
Packet sigma and prior sampling correction are declared separately; neither
is guessed from a count or from a model residual.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def source_support_masses(uv, paths, *, measured_mask=None):
    """Length quadrature on explicit visible RGB paths, with no gap bridging.

    Identical UV and repeated path edges are counted once. Interpolated points
    are omitted before forming edges, so inserting them changes no mass.
    Isolated measured anchors have one numerical pixel of support. Callers
    must supply separate paths at occlusion or source-identity breaks. The
    returned mass is per UNIQUE anchor and is repeated for its stored copies;
    compile_curve_sampling divides those copies. For direct neural losses use
    source_support_integration_weights, which performs that division itself.
    """
    uv = np.asarray(uv, dtype='<f8')
    if uv.ndim != 2 or uv.shape[1:] != (2,) or not len(uv) or not np.isfinite(uv).all():
        raise ValueError('Finite source UV required for support integration')
    measured = np.ones(len(uv), bool) if measured_mask is None else np.asarray(measured_mask)
    if measured.shape != (len(uv),) or measured.dtype != bool:
        raise ValueError('Explicit measured-source point mask required')
    keys = [point.tobytes() for point in uv]
    positions = {key: point for key, point in zip(keys, uv)}
    mass, seen, covered = defaultdict(float), set(), set()
    for path in paths:
        if (not isinstance(path, (list, tuple)) or not path
                or any(type(i) is not int or not 0 <= i < len(uv) for i in path)):
            raise ValueError('Explicit in-range visible source path indices required')
        selected = [keys[i] for i in path if measured[i]]
        covered.update(selected)
        for first, last in zip(selected[:-1], selected[1:]):
            if first == last: continue
            edge = tuple(sorted((first, last)))
            if edge in seen: continue
            seen.add(edge)
            half = .5 * float(np.linalg.norm(positions[first] - positions[last]))
            mass[first] += half; mass[last] += half
    if covered != {key for key, keep in zip(keys, measured) if keep}:
        raise ValueError('Every measured source anchor needs explicit visible support coverage')
    for key in covered:
        if mass[key] == 0.: mass[key] = 1.
    return np.asarray([mass[key] if keep else 0. for key, keep in zip(keys, measured)], float)


def source_support_integration_weights(uv, paths, *, measured_mask=None):
    """Direct loss weights summing to one across a physical source curve.

    Combine all annotation portions of one curve/exposure before this call.
    Exact duplicate UV divide the original anchor's quality mass; they cannot
    multiply a point's influence in a neural or ordinary source loss.
    """
    uv = np.ascontiguousarray(uv, dtype='<f8')
    mass = source_support_masses(uv, paths, measured_mask=measured_mask)
    measured = np.ones(len(uv), bool) if measured_mask is None else np.asarray(measured_mask)
    counts = defaultdict(int)
    for point, keep in zip(uv, measured):
        if keep: counts[point.tobytes()] += 1
    weights = np.asarray([value/counts[point.tobytes()] if keep else 0.
        for point, value, keep in zip(uv, mass, measured)], float)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.:
        raise ValueError('Physical source curve has no finite measured support')
    return weights / total


def _positive(value, n, label):
    result = np.broadcast_to(np.asarray(value, float), (n,)).copy()
    if not np.isfinite(result).all() or np.any(result <= 0.):
        raise ValueError('Finite positive ' + label + ' required')
    return result


def compile_curve_sampling(records, *, duplicate_anchor_policy=None):
    """Compile separate working normalization and loss integration weights.

    Each record binds one FIT UV array, declared stored/working sigma, prior
    sampling multiplier, physical curve/exposure, operator identity, sampling
    kinds and RGB integration masses. ``interpolated`` entries need explicit
    measured parent references and receive zero FIT mass. Exact duplicates of
    the same operator and UV divide one source-anchor mass among their copies.
    The optional authenticated-anchor policy keeps different original working
    sigmas on those copies. It never resolves conflicting support masses.
    """
    preserve_copy_sigma=False
    if duplicate_anchor_policy is not None:
        policy=dict(duplicate_anchor_policy);authority=policy.get('source_anchor_identity_authority')
        if (policy.get('kind')!='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma'
                or not isinstance(authority,dict) or not isinstance(authority.get('path'),str) or not authority['path']
                or not isinstance(authority.get('sha256'),str) or len(authority['sha256'])!=64
                or any(c not in '0123456789abcdef' for c in authority['sha256'])):
            raise ValueError('Explicit authenticated original source-anchor identity policy required')
        preserve_copy_sigma=True
    rows, atoms, group_atoms = {}, {}, defaultdict(list)
    for source in records:
        row = dict(source); identity = row['id']
        if preserve_copy_sigma and row.get('source_anchor_identity_authority')!=authority:
            raise ValueError('Source-anchor identity authority differs from explicit duplicate policy')
        if not isinstance(identity, str) or not identity or identity in rows:
            raise ValueError('Unique source sampling row IDs required')
        if row.get('split') != 'fit':
            raise ValueError('Curve grouping consumes FIT only; HELD remains unchanged')
        if (type(row.get('actual_pts')) is not int or row['actual_pts'] < 0
                or not isinstance(row.get('source_video_sha256'), str) or len(row['source_video_sha256']) != 64
                or not isinstance(row.get('physical_curve_id'), str) or not row['physical_curve_id']
                or not isinstance(row.get('operator_id'), str) or not row['operator_id']):
            raise ValueError('Explicit source video, actual exposure, physical curve and operator identity required')
        uv = np.ascontiguousarray(row['uv'], dtype='<f8'); n = len(uv)
        if uv.ndim != 2 or uv.shape[1:] != (2,) or not n or not np.isfinite(uv).all():
            raise ValueError('Finite original source sampling UV required')
        stored = _positive(row['stored_sigma_px'], n, 'stored sigma')
        working = _positive(row['working_sigma_px'], n, 'working engineering sigma')
        prior = _positive(row['prior_sampling_multiplier'], n, 'declared prior sampling multiplier')
        if not np.allclose(stored, working * prior, rtol=1e-12, atol=1e-12):
            raise ValueError('Stored sigma disagrees with explicit working/sampling decomposition')
        mass = np.asarray(row['point_integration_mass'], float)
        kinds = row['sampling_kind']
        if (mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass < 0.)
                or not isinstance(kinds, list) or len(kinds) != n
                or any(kind not in ('measured', 'interpolated') for kind in kinds)):
            raise ValueError('Explicit source sampling kinds and finite nonnegative integration masses required')
        parents = row.get('interpolation_parents', [None] * n)
        if not isinstance(parents, list) or len(parents) != n:
            raise ValueError('Per-point interpolation lineage required')
        group = (row['source_video_sha256'], row['actual_pts'], row['physical_curve_id'])
        row.update(uv=uv, working=working, prior=prior, group=group, parents=parents,
            point_integration_weights=np.zeros(n), quadratic_multipliers=np.zeros(n))
        rows[identity] = row
        for index, kind in enumerate(kinds):
            if kind == 'interpolated':
                if mass[index] != 0.:
                    raise ValueError('Interpolated source points cannot acquire new FIT mass')
                continue
            if mass[index] <= 0. or parents[index] is not None:
                raise ValueError('Measured source anchors require positive mass and no interpolation lineage')
            key = (group, row['operator_id'], uv[index].tobytes())
            if key not in atoms:
                atoms[key] = dict(mass=mass[index], working=working[index], copies=[])
                group_atoms[group].append(key)
            atom = atoms[key]
            if not (np.isclose(atom['mass'], mass[index], rtol=1e-12, atol=1e-12)
                    and (preserve_copy_sigma or np.isclose(atom['working'], working[index], rtol=1e-12, atol=1e-12))):
                raise ValueError('Duplicate source anchor working sigma or integration mass conflicts')
            atom['copies'].append((identity, index))
    # Check actual UV lineage. A convenient "interpolated" label alone is not
    # authorization to replace a measured point or invent source support.
    for row in rows.values():
        for index, kind in enumerate(row['sampling_kind']):
            if kind != 'interpolated': continue
            refs = row['parents'][index]
            if not isinstance(refs, list) or len(refs) < 2:
                raise ValueError('Interpolation needs at least two exact measured source parents')
            uv = np.zeros(2); total = 0.
            for ref in refs:
                parent = rows.get(ref.get('row_id')); at = ref.get('point_index'); weight = ref.get('weight')
                if (parent is None or type(at) is not int or not 0 <= at < len(parent['uv'])
                        or parent['sampling_kind'][at] != 'measured' or parent['group'] != row['group']
                        or parent['operator_id'] != row['operator_id'] or isinstance(weight, bool)
                        or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0.):
                    raise ValueError('Interpolation parent source/curve/operator identity changed')
                uv += weight * parent['uv'][at]; total += weight
            if abs(total - 1.) > 1e-12 or not np.allclose(uv, row['uv'][index], rtol=0., atol=1e-9):
                raise ValueError('Interpolation is not the declared source-anchor affine combination')
    groups = []
    for group, keys in group_atoms.items():
        total = sum(atoms[key]['mass'] for key in keys)
        precision = 0.; count = 0
        for key in keys:
            atom = atoms[key]; share = atom['mass'] / total / len(atom['copies'])
            if not preserve_copy_sigma:
                precision += atom['mass'] / total / atom['working']**2
            for identity, index in atom['copies']:
                rows[identity]['point_integration_weights'][index] = share
                rows[identity]['quadratic_multipliers'][index] = rows[identity]['prior'][index] * np.sqrt(share)
                if preserve_copy_sigma:
                    precision += share / rows[identity]['working'][index]**2
                count += 1
        groups.append(dict(source_video_sha256=group[0], actual_pts=group[1], physical_curve_id=group[2],
            engineering_curve_budget=1., point_integration_weight_sum=1.,
            unique_source_operator_UV_anchors=len(keys), stored_measured_copies=count,
            source_support_mass=total, effective_sum_inverse_working_sigma_squared=float(precision)))
    result=dict(rows={identity: dict(point_integration_weights=row['point_integration_weights'],
        working_normalization_multiplier=row['prior'].copy(),
        quadratic_residual_J_multiplier=row['quadratic_multipliers']) for identity, row in rows.items()}, groups=groups,
        policy='one engineering budget per authenticated physical curve and actual exposure',
        sigma_decomposition_explicit=True, measured_statistical_independence_claimed=False)
    if duplicate_anchor_policy is not None:result['duplicate_anchor_policy']=dict(duplicate_anchor_policy)
    return result
