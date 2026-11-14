"""Qualify whole nested source-ring alternatives before candidate consensus."""
from operator import index as integer_index
import numpy as np
from .material_ring_bands import compatible_query_consensus
from .material_rings import nested_source_rings, source_ring_color_evidence


def _color_qualified_alternatives(rgb, measurement, hypotheses, nesting):
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
    return rgb, alternatives, receipt


def color_constrained_ring_stack(rgb, measurement, hypotheses, *, nesting=None):
    """Use real material colors to resolve competing geometric source stacks.

    Geometric alternatives may share no candidate IDs. Intersecting them before
    inspecting their colors discards the evidence that distinguishes paint from
    neutral reflections. This function qualifies each entire stack first; its
    output remains original source readouts, never fitted conic coordinates.
    """
    nesting = nested_source_rings(hypotheses) if nesting is None else nesting
    rgb, alternatives, receipt = _color_qualified_alternatives(rgb, measurement, hypotheses, nesting)
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


def _validate_ring_source_indices(measurement, hypotheses, nesting):
    def checked_index(value, count, name):
        try:
            index = integer_index(value)
        except TypeError as exc:
            raise ValueError(f'{name} must be an integer source index: {value!r}') from exc
        if isinstance(value, (bool, np.bool_)) or not 0 <= index < count:
            raise ValueError(f'{name} outside source range [0, {count}): {value!r}')
        return index

    inspected_rings, inspected_candidates = set(), set()
    for item in nesting.get('alternatives', []):
        for role, value in item['indices'].items():
            index = checked_index(value, len(hypotheses[role]), 'Ring hypothesis index')
            if (role, index) in inspected_rings:
                continue
            inspected_rings.add((role, index))
            for value in hypotheses[role][index]['candidate_ids']:
                candidate = checked_index(value, len(measurement['candidate_query']), 'Source candidate index')
                if candidate in inspected_candidates:
                    continue
                inspected_candidates.add(candidate)
                checked_index(candidate, len(measurement['candidate_side_quantiles_uv']), 'Source candidate index')
                if 'candidate_uv' in measurement:
                    checked_index(candidate, len(measurement['candidate_uv']), 'Source candidate index')
                checked_index(measurement['candidate_query'][candidate], len(measurement['query_normal']),
                              'Source query index')


def color_constrained_ring_query_consensus(rgb, measurement, hypotheses, *, nesting=None):
    """Keep uncontested source readouts across color-qualified ring alternatives.

    Missing arcs do not veto support; competing candidates at one query remain
    pending. The original three-layer RGB test rechecks the retained IDs, and
    fitted conic coordinates never become observations or physical identity.
    """
    nesting = nested_source_rings(hypotheses) if nesting is None else nesting
    _validate_ring_source_indices(measurement, hypotheses, nesting)
    rgb, alternatives, receipt = _color_qualified_alternatives(rgb, measurement, hypotheses, nesting)
    receipt['method'] = 'whole_color_qualified_alternatives_before_query_consensus'
    if not alternatives:
        return dict(receipt, status='no_color_qualified_source_stack', selected={})
    plausible = [item for item in alternatives if item['score'] >= .85*alternatives[0]['score']]
    receipt['plausible_alternatives'] = len(plausible)
    consensus = {role: compatible_query_consensus(
        [dict(candidate_ids=item['selected'][role]) for item in plausible], measurement['candidate_query'])
        for role in ('inner', 'middle', 'outer')}
    receipt['conflicting_queries'] = {role: result['conflicting_queries'] for role, result in consensus.items()}
    checked = source_ring_color_evidence(rgb, measurement,
        {role: result['selected'] for role, result in consensus.items()})
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
