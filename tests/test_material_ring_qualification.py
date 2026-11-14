from copy import deepcopy
import cv2
import numpy as np
import pytest
from rinkspace.session.material_rings import nested_source_rings
from rinkspace.session.material_ring_qualification import (
    color_constrained_ring_stack, color_constrained_ring_query_consensus)


@pytest.mark.parametrize('qualify', [color_constrained_ring_stack, color_constrained_ring_query_consensus])
def test_color_resolves_disjoint_geometric_alternatives_before_intersection(qualify):
    rgb = np.full((240, 520, 3), 230, np.uint8)
    hypotheses = {role: [] for role in ('inner', 'middle', 'outer')}
    normals, quantiles = [], []
    theta = np.arange(24)*2*np.pi/24
    radial = np.c_[np.cos(theta), np.sin(theta)]
    for k, cx in enumerate((130, 390)):
        center = np.array([cx, 120.])
        cv2.circle(rgb, (cx,120), 90, (100,100,100), -1)
        cv2.circle(rgb, (cx,120), 60, (230,230,230), -1)
        cv2.circle(rgb, (cx,120), 30, (170,100,100) if k == 0 else (100,100,100), -1)
        for role, radius in [('inner',30), ('middle',60), ('outer',90)]:
            begin = len(normals); normals.extend(radial)
            q = center + radial[:,None,:] * np.array([radius-1.,radius,radius+1.])[None,:,None]
            quantiles.extend(np.repeat(q[:,None,:,:],2,axis=1))
            hypotheses[role].append(dict(candidate_ids=list(range(begin,begin+24)),
                source_center=center.tolist(), source_conic=(np.eye(2)/radius**2).tolist(),
                score=100. if k == 0 else 101.))
    measurement = dict(candidate_query=np.arange(len(normals)),query_normal=np.array(normals),
                       candidate_side_quantiles_uv=np.array(quantiles))
    geometric = nested_source_rings(hypotheses)
    assert all(not ids for ids in geometric['selected'].values())
    result = qualify(rgb,measurement,hypotheses,nesting=geometric)
    assert result['status'] == 'source_colored_ring_stack'
    assert result['color_qualified_alternatives'] == 1
    for role,ids in result['selected'].items():
        assert ids == hypotheses[role][0]['candidate_ids']
    assert result['colors']['inner'] == 'red'
    assert not result['physical_instance_confirmed']


def test_missing_layer_is_not_reconstructed_from_color():
    result = color_constrained_ring_stack(np.zeros((10,10,3)),{},
        {'inner':[],'middle':[],'outer':[]})
    assert result['selected'] == {}


def test_cross_view_matches_cannot_mix_interfaces_from_incompatible_stacks():
    from rinkspace.session.material_ring_qualification import matched_complete_ring_consensus
    hypotheses = {role: [dict(candidate_ids=list(range(offset, offset+3))),
                          dict(candidate_ids=list(range(offset+30, offset+33)))]
                  for role, offset in [('inner',0),('middle',3),('outer',6)]}
    nesting = dict(alternatives=[dict(indices=dict(inner=0,middle=0,outer=0)),
                                 dict(indices=dict(inner=1,middle=1,outer=1))])
    mixed = matched_complete_ring_consensus(hypotheses,nesting,dict(middle=[3,4,5],outer=[36,37,38]))
    assert mixed['selected'] == {}
    coherent = matched_complete_ring_consensus(hypotheses,nesting,dict(middle=[3,4,5],outer=[6,7,8]))
    assert coherent['selected'] == dict(inner=[],middle=[3,4,5],outer=[6,7,8])


def _source_color_ring(*, outer=(100, 100, 100), inner=(170, 100, 100)):
    rgb = np.full((260, 260, 3), 230, np.uint8)
    cv2.circle(rgb, (130, 130), 90, outer, -1)
    cv2.circle(rgb, (130, 130), 60, (230, 230, 230), -1)
    cv2.circle(rgb, (130, 130), 30, inner, -1)
    theta = np.arange(24)*2*np.pi/24
    radial = np.c_[np.cos(theta), np.sin(theta)]
    normals, quantiles, hypotheses = [], [], {}
    for role, radius in [('inner', 30), ('middle', 60), ('outer', 90)]:
        begin = len(normals)
        normals.extend(radial)
        q = [130., 130.] + radial[:, None, :] * np.array([radius-1., radius, radius+1.])[None, :, None]
        quantiles.extend(np.repeat(q[:, None, :, :], 2, axis=1))
        hypotheses[role] = [dict(candidate_ids=list(range(begin, begin+24)), score=100.,
            source_center=[130., 130.], source_conic=(np.eye(2)/radius**2).tolist(),
            physical_instance_confirmed=False, ellipse_coordinates_are_not_observations=True)]
    measurement = dict(candidate_query=np.arange(72), query_normal=np.array(normals),
        candidate_side_quantiles_uv=np.array(quantiles), candidate_flags=np.arange(72, dtype=np.uint16))
    measurement['candidate_uv'] = measurement['candidate_side_quantiles_uv'][:, 0, 1].copy()
    return rgb, measurement, hypotheses


def _two_alternatives():
    return dict(alternatives=[dict(indices=dict.fromkeys(('inner', 'middle', 'outer'), index))
                              for index in (0, 1)])


def test_query_consensus_keeps_missing_arcs_without_changing_source_geometry_or_flags():
    rgb, measurement, hypotheses = _source_color_ring()
    expected = {role: rings[0]['candidate_ids'][:] for role, rings in hypotheses.items()}
    for role, rings in hypotheses.items():
        ring = rings[0]
        hypotheses[role] = [dict(ring, candidate_ids=ring['candidate_ids'][:12], median_normal_error_px=.01),
                            dict(ring, candidate_ids=ring['candidate_ids'][12:], median_normal_error_px=50.)]
    hypotheses['center'] = []  # A fourth layer is optional.
    original_hypotheses, original_measurement, original_rgb = deepcopy(hypotheses), deepcopy(measurement), rgb.copy()
    old = color_constrained_ring_stack(rgb, measurement, hypotheses)
    assert old['selected'] == {}  # Preserve the original API's intersection behavior.
    result = color_constrained_ring_query_consensus(rgb, measurement, hypotheses)
    assert result['status'] == 'source_colored_ring_stack'
    assert result['selected'] == expected
    assert result['color_qualified_alternatives'] == result['plausible_alternatives'] == 8
    assert result['conflicting_queries'] == dict(inner={}, middle={}, outer={})
    assert result['observations_are_original_candidate_ids'] and not result['physical_instance_confirmed']
    assert hypotheses == original_hypotheses
    for key, values in measurement.items():
        np.testing.assert_array_equal(values, original_measurement[key])
    np.testing.assert_array_equal(rgb, original_rgb)


@pytest.mark.parametrize('conflicts', [1, 19])
def test_query_conflicts_remain_pending_and_retained_readouts_still_need_six_per_layer(conflicts):
    rgb, measurement, hypotheses = _source_color_ring()
    original = {role: rings[0]['candidate_ids'][:] for role, rings in hypotheses.items()}
    disputed = [cid for ids in original.values() for cid in ids[:conflicts]]
    duplicates = dict(zip(disputed, range(72, 72+len(disputed))))
    for key in ('candidate_query', 'candidate_uv', 'candidate_side_quantiles_uv', 'candidate_flags'):
        measurement[key] = np.concatenate([measurement[key], measurement[key][disputed]], axis=0)
    # Distinct source readings at the same query, both on the observed pigment.
    measurement['candidate_side_quantiles_uv'][72:, :, :, 0] += .1
    measurement['candidate_uv'][72:, 0] += .1
    for role, rings in hypotheses.items():
        ring = rings[0]
        rings.append(dict(ring, candidate_ids=[duplicates.get(cid, cid) for cid in ring['candidate_ids']]))
    result = color_constrained_ring_query_consensus(rgb, measurement, hypotheses, nesting=_two_alternatives())
    assert result['conflicting_queries'] == {role: {str(cid): [cid, duplicates[cid]] for cid in ids[:conflicts]}
                                             for role, ids in original.items()}
    if conflicts == 1:
        assert result['status'] == 'source_colored_ring_stack'
        assert result['selected'] == {role: ids[conflicts:] for role, ids in original.items()}
    else:
        assert result['status'] == 'source_colored_stack_still_ambiguous'
        assert result['selected'] == {}
        assert all(stat['readouts'] == 5 for stat in result['consensus_statistics'].values())


@pytest.mark.parametrize('second_score, plausible', [(84., 1), (85., 2)])
def test_query_consensus_uses_original_point_eighty_five_plausible_threshold(second_score, plausible):
    rgb, measurement, hypotheses = _source_color_ring()
    for rings in hypotheses.values():
        ring = rings[0]
        rings.append(dict(ring, candidate_ids=ring['candidate_ids'][12:], score=second_score))
        rings[0] = dict(ring, candidate_ids=ring['candidate_ids'][:12])
    result = color_constrained_ring_query_consensus(rgb, measurement, hypotheses, nesting=_two_alternatives())
    assert result['plausible_alternatives'] == plausible
    assert result['selected'] == {role: sum([r['candidate_ids'] for r in rings[:plausible]], [])
                                  for role, rings in hypotheses.items()}


@pytest.mark.parametrize('outer, inner', [((100, 100, 100), (100, 100, 100)),
                                         ((170, 100, 100), (170, 100, 100))])
def test_query_consensus_rejects_real_rgb_without_the_original_color_order(outer, inner):
    rgb, measurement, hypotheses = _source_color_ring(outer=outer, inner=inner)
    result = color_constrained_ring_query_consensus(rgb, measurement, hypotheses)
    assert result['status'] == 'no_color_qualified_source_stack'
    assert result['selected'] == {}


def test_query_consensus_rechecks_color_ambiguity_after_combining_qualified_readouts():
    rgb, measurement, hypotheses = _source_color_ring()
    other_rgb, other_measurement, other_hypotheses = _source_color_ring(
        outer=(170, 100, 100), inner=(100, 100, 100))
    rgb = np.concatenate([rgb, other_rgb], axis=1)
    for key in ('candidate_uv', 'candidate_side_quantiles_uv'):
        other_measurement[key][..., 0] += 260.
    other_measurement['candidate_query'] += 72
    for key in measurement:
        measurement[key] = np.concatenate([measurement[key], other_measurement[key]], axis=0)
    for role, rings in hypotheses.items():
        other = other_hypotheses[role][0]
        rings.append(dict(other, candidate_ids=[cid+72 for cid in other['candidate_ids']], source_center=[390., 130.]))
    result = color_constrained_ring_query_consensus(rgb, measurement, hypotheses, nesting=_two_alternatives())
    assert result['color_qualified_alternatives'] == 2
    assert result['status'] == 'source_colored_stack_still_ambiguous'
    assert result['selected'] == {}
    assert all(stat['readouts'] == 48 and stat['red_supported'] == 24
               for stat in result['consensus_statistics'].values())


@pytest.mark.parametrize('target, value', [('candidate', -1), ('candidate', 72), ('candidate', 1.5),
                                         ('query', -1), ('query', 72), ('query', 1.5), ('hypothesis', -1)])
def test_query_consensus_rejects_invalid_source_indices_instead_of_wrapping_or_truncating(target, value):
    rgb, measurement, hypotheses = _source_color_ring()
    nesting = dict(alternatives=[dict(indices=dict(inner=0, middle=0, outer=0))])
    if target == 'candidate':
        hypotheses['inner'][0]['candidate_ids'][0] = value
    elif target == 'query':
        measurement['candidate_query'] = measurement['candidate_query'].astype(float) if isinstance(value, float) else measurement['candidate_query']
        measurement['candidate_query'][0] = value
    else:
        nesting['alternatives'][0]['indices']['inner'] = value
    with pytest.raises(ValueError, match='index'):
        color_constrained_ring_query_consensus(rgb, measurement, hypotheses, nesting=nesting)


def test_query_consensus_does_not_require_a_fourth_layer_or_fill_a_missing_required_layer():
    result = color_constrained_ring_query_consensus(np.zeros((10, 10, 3)), {},
        {'inner': [], 'middle': [], 'outer': [], 'center': []})
    assert result['selected'] == {}
