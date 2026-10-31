"""Whole source-ring hypotheses over unchanged RGB interface measurements.

Ellipses only organize source boundaries. Their fitted coordinates are never
observations, and a geometric match alone does not assign a physical instance.
"""
from __future__ import annotations

import cv2
import numpy as np


def ellipse_matrix(points):
    """Return a source image conic; do not interpret its centre as a 3D point."""
    center, axes, angle = cv2.fitEllipse(np.asarray(points, np.float32))
    axes = np.asarray(axes, float) / 2.
    if not np.isfinite([*center, *axes, angle]).all() or axes.min() < 1.:
        raise ValueError('Degenerate source ellipse')
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.asarray(center), rotation @ np.diag(1./axes**2) @ rotation.T


def conic_distance(uv, center, matrix):
    delta = uv-center
    gradient = 2.*delta@matrix
    return (np.einsum('ij,ij->i', delta@matrix, delta)-1.) / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)


def ellipse_vertices(center, matrix, count=256):
    eig, rotation = np.linalg.eigh(matrix)
    theta = np.arange(count)*2.*np.pi/count
    return center+np.c_[np.cos(theta), np.sin(theta)] @ np.diag(1./np.sqrt(eig)) @ rotation.T


def source_ring_hypotheses(rgb, measurement, curve, fragments=(), *, band_centers=False,
                           material_signal='red', eligible_candidates=None):
    """Compare complete source boundaries, retaining alternative ellipses.

    Candidate IDs and gradient quantiles remain untouched. The search strip
    limits where to look, while signed RGB contrast and angular support rank
    actual source points. A later identity/temporal check is still required.
    """
    qcurve = np.asarray(measurement['query_curve'])
    query = np.asarray(measurement['candidate_query'])
    clean = (np.asarray(measurement['candidate_flags']) & 3) == 0
    if eligible_candidates is not None:
        eligibility=np.asarray(eligible_candidates,bool)
        if eligibility.shape!=clean.shape:raise ValueError('Candidate eligibility shape mismatch')
        clean &= eligibility
    ids = np.flatnonzero(clean & (qcurve[query] == curve))
    queries = np.flatnonzero(qcurve == curve)
    if len(ids) < 12 or len(queries) < 12:
        return []
    quv = np.asarray(measurement['query_uv'], float)[queries]
    try:
        guide_center, guide_matrix = ellipse_matrix(quv)
    except (ValueError, cv2.error):
        return []
    uv = (np.asarray(measurement['candidate_uv'])[ids] if band_centers else
          np.asarray(measurement['candidate_side_quantiles_uv'])[ids, 0, 1])
    finite = np.isfinite(uv).all(1)
    ids, uv = ids[finite], uv[finite]
    if len(ids) < 12:
        return []
    normals = np.asarray(measurement['query_normal'], float)[query[ids]]
    # Contrast across the actual q25..q75 transition, not across a rendered ring.
    quantiles = np.asarray(measurement['candidate_side_quantiles_uv'])[ids]
    samples = np.stack([quantiles[:, 0, 0]-normals*3., quantiles[:, -1 if band_centers else 0, 2]+normals*3.], 1)
    pixel = cv2.remap(np.asarray(rgb, np.float32), samples[..., 0].astype('f'),
                      samples[..., 1].astype('f'), cv2.INTER_LINEAR)
    if band_centers:
        middle=cv2.remap(np.asarray(rgb,np.float32),uv[:,0,None].astype('f'),uv[:,1,None].astype('f'),cv2.INTER_LINEAR)[:,0]
        difference=middle-pixel.mean(1)
        if material_signal == 'dark':
            contrast=np.maximum(-difference.mean(1),0.)
        elif material_signal == 'blue':
            contrast=np.maximum(difference[:,2]-difference[:,0],0.)
        elif material_signal == 'red':
            contrast=np.maximum(difference[:,0]-difference[:,1],0.)
        else:
            raise ValueError('Unknown ring material signal: '+str(material_signal))
    else:
        contrast = np.linalg.norm(pixel[:, 1]-pixel[:, 0], axis=1)/np.sqrt(3.)
    unique_queries = np.unique(query[ids])
    by_query = [np.flatnonzero(query[ids] == q) for q in unique_queries]
    initial = [(guide_center, guide_matrix)]
    if band_centers:
        # A translated ellipse cannot be initialized by uniformly expanding
        # the old ellipse. Combine observed arcs before fitting, otherwise
        # a correct near/far pair can each become a spurious thin ellipse.
        from itertools import combinations
        groups=[set(f['candidate_ids']) & set(ids) for f in fragments if f['curve']==curve]
        unions=[ids]+[sorted(a|b) for a,b in combinations(groups,2) if len(a|b)>=12]
        source_uv={int(i):p for i,p in zip(ids,uv)}
        for group in unions:
            try:initial.append(ellipse_matrix(np.array([source_uv[int(i)] for i in group])))
            except (ValueError,cv2.error):pass
    # Source fragments provide alternate starts, including displaced boundaries.
    for fragment in fragments:
        if fragment['curve'] != curve or len(fragment['candidate_ids']) < 12:
            continue
        subset = np.intersect1d(ids, fragment['candidate_ids'])
        if len(subset) < 12:
            continue
        try:
            initial.append(ellipse_matrix(np.asarray(measurement['candidate_uv'])[subset]))
        except (ValueError, cv2.error):
            pass
    # Start on every occupied source normal-displacement mode. This is not a
    # fixed optimisation budget: the number comes from the measured candidates.
    displacement = np.einsum('ij,ij->i', uv-np.asarray(measurement['query_uv'])[query[ids]], normals)
    for offset in np.unique(np.rint(displacement/4.)*4.):
        try:
            initial.append(ellipse_matrix(quv+np.asarray(measurement['query_normal'])[queries]*offset))
        except (ValueError, cv2.error):
            pass
    hypotheses = {}
    for center, matrix in initial:
        seen = set()
        while True:
            if not np.isfinite(center).all() or not np.isfinite(matrix).all():
                break
            distance = np.abs(conic_distance(uv, center, matrix))
            selected = np.array([g[np.argmin(distance[g])] for g in by_query], int)
            selected = selected[distance[selected] <= 4.]
            key = tuple(selected)
            if len(selected) < 12 or key in seen:
                break
            seen.add(key)
            try:
                fitted_center, fitted_matrix = ellipse_matrix(uv[selected])
            except (ValueError, cv2.error):
                break
            # Only the measured arc constrains this source hypothesis. Testing
            # the entire extrapolated ellipse would reject a real partial ring
            # because its hidden/occluded tip is not independently observable.
            supported_queries = np.asarray(measurement['query_uv'])[query[ids[selected]]]
            if np.median(np.abs(conic_distance(supported_queries, fitted_center, fitted_matrix))) > 48.:
                break
            center, matrix = fitted_center, fitted_matrix
            error = np.abs(conic_distance(uv[selected], center, matrix))
            good = error <= 2.
            chosen = selected[good]
            if len(chosen) < 12:
                continue
            eigen, rotation = np.linalg.eigh(matrix)
            local = (uv[chosen]-center)@rotation@np.diag(np.sqrt(eigen))
            angle = np.arctan2(local[:, 1], local[:, 0])
            sectors = np.unique(np.floor((angle+np.pi)*12./(2.*np.pi)).astype(int))
            if len(sectors) < 7:
                continue  # A short almost-straight arc cannot determine a ring.
            score = float(np.sum(np.minimum(contrast[chosen], 20.)))
            item = dict(candidate_ids=ids[chosen].tolist(), source_center=center.tolist(),
                        source_conic=matrix.tolist(), score=score, angular_sectors=len(sectors),
                        median_normal_error_px=float(np.median(error[good])),
                        median_source_contrast=float(np.median(contrast[chosen])),
                        ellipse_coordinates_are_not_observations=True,
                        physical_instance_confirmed=False)
            old = hypotheses.get(tuple(item['candidate_ids']))
            if old is None or score > old['score']:
                hypotheses[tuple(item['candidate_ids'])] = item
    ranked = sorted(hypotheses.values(), key=lambda h:h['score'], reverse=True)
    distinct = []
    for item in ranked:
        vertices = ellipse_vertices(np.asarray(item['source_center']), np.asarray(item['source_conic']))
        if any(np.median(np.abs(conic_distance(vertices, np.asarray(h['source_center']),
                                              np.asarray(h['source_conic'])))) < 2. for h in distinct):
            continue
        distinct.append(item)
    return distinct


def nested_source_rings(hypotheses):
    """Select only source samples shared by competing nested material stacks.

    Three surrounding interfaces establish the ring pattern. A small centre
    contour is optional and is never inferred from an ellipse centre.
    """
    from itertools import product
    roles = ('inner', 'middle', 'outer')
    if any(not hypotheses.get(role) for role in roles):
        return dict(status='incomplete_material_stack', selected={}, alternatives=[])
    alternatives = []
    for indices in product(*(range(len(hypotheses[role])) for role in roles)):
        rings = [hypotheses[role][i] for role, i in zip(roles, indices)]
        nested = True
        for inside, outside in zip(rings, rings[1:]):
            vertices = ellipse_vertices(np.array(inside['source_center']), np.array(inside['source_conic']))
            d = conic_distance(vertices, np.array(outside['source_center']), np.array(outside['source_conic']))
            if np.max(d) >= -1.:
                nested = False
                break
        if nested:
            alternatives.append(dict(indices=dict(zip(roles, indices)), score=sum(h['score'] for h in rings)))
    alternatives.sort(key=lambda h:h['score'], reverse=True)
    if not alternatives:
        return dict(status='inconsistent_material_nesting', selected={}, alternatives=[])
    # Weak alternatives are retained in the receipt. All plausible alternatives
    # must agree on the exact source candidate before that candidate is emitted.
    plausible = [a for a in alternatives if a['score'] >= .85*alternatives[0]['score']]
    selected = {}
    for role in roles:
        choices = [set(hypotheses[role][a['indices'][role]]['candidate_ids']) for a in plausible]
        selected[role] = sorted(set.intersection(*choices))
    return dict(status='source_nested_stack_hypothesis', selected=selected,
                plausible_alternatives=len(plausible), alternatives=alternatives,
                physical_instance_confirmed=False, observations_are_original_candidate_ids=True)


def source_ring_color_evidence(rgb, measurement, selected):
    """Test the observed red/white/dark layers, including the null alternative.

    This is evidence for the event's observed house pattern, not a universal
    ellipse rule. The other dark layer can be almost neutral in this broadcast;
    it must not be required to carry an invented blue chroma signal.
    """
    statistics, evidence = {}, {}
    for role, ids in selected.items():
        ids=np.asarray(ids,int)
        if not len(ids):
            continue
        query=np.asarray(measurement['candidate_query'])[ids]
        normals=np.asarray(measurement['query_normal'])[query]
        quantiles=np.asarray(measurement['candidate_side_quantiles_uv'])[ids,0]
        samples=np.stack([quantiles[:,0]-normals*3.,quantiles[:,2]+normals*3.],1)
        pixels=cv2.remap(np.asarray(rgb,np.float32),samples[...,0].astype('f'),
            samples[...,1].astype('f'),cv2.INTER_LINEAR).astype(float)
        # The coloured side is inward for outer/inner and outward for middle.
        sign=-1. if role=='middle' else 1.
        difference=sign*(pixels[:,0]-pixels[:,1])
        red=difference[:,0]-.5*(difference[:,1]+difference[:,2])
        blue=difference[:,2]-.5*(difference[:,0]+difference[:,1])
        evidence[role]=(ids,red,blue)
        statistics[role]=dict(readouts=len(ids),red_median=float(np.median(red)),blue_median=float(np.median(blue)),
            red_supported=int((red>1.5).sum()),blue_supported=int((blue>1.5).sum()))
    roles=('inner','middle','outer')
    if any(role not in evidence for role in roles):
        return dict(status='color_layers_missing',statistics=statistics,selected={})
    candidates=[]
    for outer_color in ('red','neutral_dark'):
        colors=dict(outer=outer_color,middle=outer_color,inner='neutral_dark' if outer_color=='red' else 'red')
        support={}
        for role,color in colors.items():
            ids,red,blue=evidence[role]
            support[role]=ids[red>1.5 if color=='red' else red<=1.5].tolist()
        fraction=min(len(support[role])/len(evidence[role][0]) for role in roles)
        if fraction>=.5 and min(map(len,support.values()))>=6:
            candidates.append(dict(colors=colors,selected=support,minimum_supported_fraction=fraction))
    if len(candidates)!=1:
        return dict(status='colored_material_pattern_unresolved',statistics=statistics,selected={},alternatives=candidates)
    return dict(status='source_colored_ring_stack',statistics=statistics,**candidates[0],
        physical_instance_confirmed=False,observations_are_original_candidate_ids=True)
