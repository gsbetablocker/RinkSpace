import numpy as np

from rinkspace.session.source_shape_consensus import known_shape_profile_hypotheses


def measured_arcs(*, rival_line=False, excluded=False):
    angle = np.linspace(.3, 2.8, 73)
    q = np.c_[400*np.cos(angle), 60*np.sin(angle)]
    normal = q/np.array([400**2, 60**2])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    offset = normal@np.array([6., -3.])
    uv = q+normal*offset[:, None]
    query = np.arange(len(q))
    if rival_line:
        line_offset = (62-q[:, 1])/normal[:, 1]
        uv = np.vstack([uv, q+normal*line_offset[:, None]])
        query = np.r_[query, query]
    sides = uv[:, None, :]+np.array([-2., 2.])[None, :, None]*normal[query, None]
    quant = sides[:, :, None, :]+np.array([-.6, 0, .6])[None, None, :, None]*normal[query, None, None]
    flags = np.zeros(len(uv), np.uint32)
    if excluded:
        flags[10:20] = 1
    return dict(query_curve=np.zeros(len(q), int), query_uv=q, query_normal=normal,
        query_interval=np.tile([-48., 48.], (len(q), 1)), candidate_query=query,
        candidate_uv=uv, candidate_side_quantiles_uv=quant, candidate_flags=flags)


def test_known_partial_shape_distinguishes_straight_competing_material():
    z = measured_arcs(rival_line=True)
    original = {k: v.copy() for k, v in z.items()}
    hypotheses = known_shape_profile_hypotheses(z, [0])
    assert hypotheses
    assert len(hypotheses[0]['candidate_ids']) >= 68
    assert max(hypotheses[0]['candidate_ids']) < len(z['query_uv'])
    assert all(np.array_equal(z[k], original[k]) for k in z)
    assert hypotheses[0]['fit_eligible'] is False
    assert hypotheses[0]['physical_identity_confirmed'] is False


def test_excluded_source_pixels_cannot_supply_shape_support():
    z = measured_arcs(excluded=True)
    hypotheses = known_shape_profile_hypotheses(z, [0])
    assert hypotheses
    assert not any(set(h['candidate_ids']) & set(range(10, 20)) for h in hypotheses)


def test_missing_source_curve_is_not_filled_from_known_shape():
    z = measured_arcs()
    assert known_shape_profile_hypotheses(z, [3]) == []


def test_nested_layers_can_explain_common_camera_aspect_change():
    angle = np.linspace(.3, 2.8, 73)
    q = np.vstack([np.c_[radius*400*np.cos(angle), radius*60*np.sin(angle)]
                   for radius in (1., .7, .4)])
    normal = np.tile([0., 1.], (len(q), 1))
    actual = q*np.array([1., 1.12])+[0., 3.]
    side = np.repeat(actual[:, None], 2, axis=1)
    quant = side[:, :, None]+np.array([-.6, 0., .6])[None, None, :, None]*normal[:, None, None]
    z = dict(query_curve=np.repeat(np.arange(3), 73), query_uv=q, query_normal=normal,
        query_interval=np.tile([-48., 48.], (len(q), 1)), candidate_query=np.arange(len(q)),
        candidate_uv=actual, candidate_side_quantiles_uv=quant,
        candidate_flags=np.zeros(len(q), np.uint32))
    hypotheses = known_shape_profile_hypotheses(z, [0, 1, 2])
    assert hypotheses[0]['transform_model'] == 'shared_symmetric_affine'
    assert len(hypotheses[0]['candidate_ids']) >= 210
    assert hypotheses[0]['exact_shared_conic_inside_original_transition']
