"""The source reader can refine one exposure without advancing it twice."""
import pytest
from test_vision_tracking_frame import reader_and_spies, source_rgb, binding, guide


def test_additional_query_shares_frontend_and_commits_once(monkeypatch):
    reader, calls = reader_and_spies(monkeypatch)
    actual = reader.vision.predict
    predictions = []
    def predict(*a, **kw):
        predictions.append(1)
        return actual(*a, **kw)
    reader.vision.predict = predict
    ctx = reader.begin_frame(source_rgb(8), binding(0), reset=True)
    reader.read_queries(ctx, [guide()])
    changed = dict(guide(), points_uv=[[2., 4.], [2., 6.]], search_radius_px=20)
    result = reader.read_queries(ctx, [changed])
    assert calls == {'global': 1, 'guided': 2}
    assert len(predictions) == 1
    assert reader.active_guides == []
    with pytest.raises(ValueError, match='Commit or abort'):
        reader.begin_frame(source_rgb(9), binding(256))
    reader.commit_frame(ctx)
    assert result['frame_context']['temporal_commits'] == 1
    assert reader.active_binding['source_pts'] == 0
    with pytest.raises(ValueError, match='already been committed'):
        reader.commit_frame(ctx)
    with pytest.raises(ValueError, match='stale'):
        reader.read_queries(ctx, [guide()])


def test_open_exposure_rgb_cannot_change(monkeypatch):
    reader, _ = reader_and_spies(monkeypatch)
    rgb = source_rgb(8)
    ctx = reader.begin_frame(rgb, binding(0), reset=True)
    rgb[0, 0] = 255
    with pytest.raises(ValueError, match='RGB changed'):
        reader.read_queries(ctx, [guide()])
    reader.abort_frame(ctx)


def test_disjoint_runs_of_same_material_are_not_deduplicated_by_path(monkeypatch):
    reader, _ = reader_and_spies(monkeypatch)
    ctx = reader.begin_frame(source_rgb(8), binding(0), reset=True)
    from rinkspace import vision_geometry
    single = vision_geometry.guided_source_reading
    def multiple(image, probability, guides, **kwargs):
        results = [single(image, probability, [g], **kwargs) for g in guides]
        return dict(results[0], readings=[r for result in results for r in result['readings']])
    monkeypatch.setattr(vision_geometry, 'guided_source_reading', multiple)
    first = dict(guide(), visible_arc_branch='upper')
    second = dict(guide(), visible_arc_branch='lower', points_uv=[[8., 1.], [8., 2.]])
    reader.read_queries(ctx, [first, second])
    assert len(ctx['queries']) == 2
    assert len({g['query_id'] for g in ctx['queries']}) == 2
    reader.read_queries(ctx, [dict(ctx['queries'][0], search_radius_px=20)])
    assert len(ctx['queries']) == 2
    reader.commit_frame(ctx)
