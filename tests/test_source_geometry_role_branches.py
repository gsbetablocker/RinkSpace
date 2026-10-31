"""Regression for circles/parallel materials competing with a source line."""
import numpy as np

from rinkspace.vision_source_roles import organize_source_line_material_branches


def source_rows(xs, branches):
    return [dict(source_search_point_uv=np.array([x, 100.]),
        all_actual_source_dark_stroke_candidates=[dict(
            actual_source_uv=np.array([x, y]),
            disposition='measured_dark_stroke_center_candidate',
            actual_source_darkness_mass=mass,
            observed_dark_stroke_normal_footprint_px=np.array([-40., 40.]))
            for y, mass in branches(x)]) for x in xs]


def organize(rows, paths):
    return organize_source_line_material_branches(rows,
        navigation_tangent_uv=[1., 0.], source_visible_paths=paths,
        candidate_collection_key='all_actual_source_dark_stroke_candidates',
        candidate_disposition='measured_dark_stroke_center_candidate')


def test_stronger_circle_branch_cannot_steal_long_line_role():
    xs=np.arange(0., 801., 20.)
    rows=source_rows(xs, lambda x: [(100., 10.), (88.+(x-400.)**2/3000., 1000.)])
    # A real occlusion remains a path break, even though source portions
    # organize as one line. Stronger local circle mass is not a role vote.
    result=organize(rows, [list(range(20)), list(range(20, 41))])
    assert result['source_line_support_qualified']
    assert len(result['actual_source_uv'])>30
    assert np.all(result['actual_source_uv'][:, 1]==100.)
    assert len(result['source_visible_paths'])>=2
    for query, branch, uv in zip(result['source_query_indices'],
            result['source_candidate_indices'], result['actual_source_uv']):
        assert np.array_equal(uv, rows[query]['all_actual_source_dark_stroke_candidates'][branch]['actual_source_uv'])


def test_two_full_parallel_strokes_are_unknown_not_nearest_navigation():
    rows=source_rows(np.arange(0., 801., 20.), lambda x: [(100., 10.), (115., 1000.)])
    result=organize(rows,[list(range(len(rows)))])
    assert result['disposition']=='competing_source_line_families_unknown'
    assert not len(result['actual_source_uv'])


def test_dark_occupied_width_is_not_center_uncertainty():
    # Huge paint footprint above must not make a curved center within a
    # global straight-line role. Width is never used as center sigma.
    rows=source_rows(np.arange(0., 801., 20.), lambda x: [(100.+(x-400.)**2/2500., 100.)])
    result=organize(rows,[list(range(len(rows)))])
    assert not len(result['actual_source_uv'])
    assert not result['source_line_support_qualified']
    assert result['geometry_tolerance_is_not_solver_sigma_or_calibrated_covariance']


def test_explicit_hidden_gap_is_absent_from_distribution_denominator():
    xs=[0.,8.,16.,24.,32.,40.,48.,56.,88.,96.,104.]
    rows=source_rows(xs,lambda x:[(100.,10.)]if 8.<=x<=56. else[])
    # Missing, unmasked material at either end still counts. Only the declared
    # foreground gap between 56 and 88 is absent from visible source length.
    result=organize(rows,[list(range(8)),list(range(8,11))])
    assert result['source_line_support_qualified']
    audit=result['actual_source_line_organizer']
    assert audit['source_visible_station_length_px']==72.
    assert audit['selected_actual_station_span_px']==48.
    for query,uv in zip(result['source_query_indices'],result['actual_source_uv']):
        np.testing.assert_array_equal(uv,rows[query]['all_actual_source_dark_stroke_candidates'][0]['actual_source_uv'])
    # Treating that same interval as visible keeps its missing-source evidence.
    full=organize(rows,[list(range(len(rows)))])
    assert not full['source_line_support_qualified']


def test_hidden_gap_does_not_resolve_competing_actual_line_families():
    xs=[0.,8.,16.,24.,32.,40.,48.,56.,88.,96.,104.]
    rows=source_rows(xs,lambda x:[(100.,10.),(115.,20.)]if 8.<=x<=56. else[])
    result=organize(rows,[list(range(8)),list(range(8,11))])
    assert not result['source_line_support_qualified']
    assert result['disposition']=='competing_source_line_families_unknown'
