import cv2
import numpy as np
from rinkspace.session.material_rings import nested_source_rings
from rinkspace.session.material_ring_qualification import color_constrained_ring_stack


def test_color_resolves_disjoint_geometric_alternatives_before_intersection():
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
    result = color_constrained_ring_stack(rgb,measurement,hypotheses,nesting=geometric)
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
