"""Qualify whole nested source-ring alternatives before candidate consensus."""
import numpy as np
from .material_rings import nested_source_rings, source_ring_color_evidence


def color_constrained_ring_stack(rgb, measurement, hypotheses, *, nesting=None):
    """Use real material colors to resolve competing geometric source stacks.

    Geometric alternatives may share no candidate IDs. Intersecting them before
    inspecting their colors discards the evidence that distinguishes paint from
    neutral reflections. This function qualifies each entire stack first; its
    output remains original source readouts, never fitted conic coordinates.
    """
    nesting = nested_source_rings(hypotheses) if nesting is None else nesting
    # All alternatives inspect the same original frame. Convert it once,
    # rather than copying the full HD image for every candidate interface.
    rgb = np.asarray(rgb, np.float32)
    alternatives = []
    for item in nesting.get('alternatives', []):
        rings = {role: hypotheses[role][index] for role, index in item['indices'].items()}
        checked = source_ring_color_evidence(rgb, measurement,
            {role: ring['candidate_ids'] for role, ring in rings.items()})
        if not checked['selected']:
            continue
        score = sum(ring['score'] * len(checked['selected'][role]) / len(ring['candidate_ids'])
                    for role, ring in rings.items())
        alternatives.append(dict(indices=item['indices'], score=score,
            colors=checked['colors'], selected=checked['selected'],
            minimum_supported_fraction=checked['minimum_supported_fraction']))
    alternatives.sort(key=lambda item: item['score'], reverse=True)
    receipt = dict(method='whole_color_qualified_alternatives_before_source_consensus',
        geometric_alternatives=len(nesting.get('alternatives', [])),
        color_qualified_alternatives=len(alternatives), alternatives=alternatives,
        physical_instance_confirmed=False, observations_are_original_candidate_ids=True)
    if not alternatives:
        return dict(receipt, status='no_color_qualified_source_stack', selected={})
    plausible = [item for item in alternatives if item['score'] >= .85*alternatives[0]['score']]
    receipt['plausible_alternatives'] = len(plausible)
    selected = {role: sorted(set.intersection(*(set(item['selected'][role]) for item in plausible)))
                for role in ('inner', 'middle', 'outer')}
    checked = source_ring_color_evidence(rgb, measurement, selected)
    if not checked['selected']:
        return dict(receipt, status='source_colored_stack_still_ambiguous', selected={},
                    consensus_statistics=checked['statistics'])
    return dict(receipt, status='source_colored_ring_stack', selected=checked['selected'],
                colors=checked['colors'], statistics=checked['statistics'],
                minimum_supported_fraction=checked['minimum_supported_fraction'])


def matched_complete_ring_consensus(hypotheses, nesting, matched):
    """Keep matched readouts shared by plausible *complete* source stacks.

    A target may show only two matched interfaces, but those interfaces must
    coexist in a geometrically supported complete source-ring alternative.
    Neither missing layers nor fitted ellipse coordinates are returned.
    """
    alternatives = []
    for item in nesting.get('alternatives', []):
        selected = {role: set(hypotheses[role][index]['candidate_ids']) & set(matched.get(role, []))
                    for role, index in item['indices'].items()}
        if sum(len(ids) >= 3 for ids in selected.values()) < 2:
            continue
        alternatives.append(dict(selected=selected, score=sum(map(len, selected.values())), indices=item['indices']))
    if not alternatives:
        return dict(selected={}, status='no_complete_stack_supporting_two_matched_interfaces', plausible=0)
    best = max(a['score'] for a in alternatives)
    plausible = [a for a in alternatives if a['score'] >= .85*best]
    selected = {role: sorted(set.intersection(*(a['selected'][role] for a in plausible)))
                for role in ('inner', 'middle', 'outer')}
    if sum(len(ids) >= 3 for ids in selected.values()) < 2:
        return dict(selected={},status='matched_stack_ambiguous',plausible=len(plausible))
    return dict(selected=selected,status='matched_complete_source_stack',plausible=len(plausible))
