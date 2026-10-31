import cv2
import numpy as np
import pytest

from rinkspace.session.panel_boundaries import read_panel_boundary
from rinkspace.vision_programme_ice_review import _actual_profile_candidates, _batched_actual_profiles
from rinkspace.vision_source_shapes import _MaterialProfileFrame


def _panel_scene(blur_width=31):
    rgb = np.full((360, 640, 3), 235, np.uint8)
    rgb[80:280, 120:520] = [18, 42, 110]
    # Enclosed white lettering and a white QR patch create genuine interior
    # dark-to-light alternatives inside the vertical-edge search intervals.
    for x in (170, 205, 240, 275):
        rgb[125:150, x:x+12] = 235
        rgb[190:225, x:x+18] = 235
    rgb[125:235, 390:485] = 235
    for y in range(133, 230, 16):
        for x in range(398, 480, 16):
            if (x//16 + y//16) % 2:
                rgb[y:y+8, x:x+8] = [18, 42, 110]
    if blur_width > 1:
        rgb = cv2.blur(rgb, (blur_width, 1))
    corners = np.array([[108., 68.], [532., 68.], [532., 292.], [108., 292.]])
    paths = [np.array([corners[i], corners[(i+1) % 4]]) for i in range(4)]
    binding = dict(source_sha256="a"*64, source_pts=100, time_base="1/90000",
                   shot_id="synthetic_panel", native_hw=list(rgb.shape[:2]))
    return rgb, paths, binding


def _read(rgb, paths, binding, exclusion=None, **options):
    if exclusion is None:
        exclusion = np.zeros(rgb.shape[:2], dtype=bool)
    radius=options.pop('search_radius',96. if options.get('material_signal') in ('red','blue') else 140.)
    return read_panel_boundary(rgb, paths, exclusion, binding=binding,
                              search_radius=radius, spacing=20., **options)


def _side_uv(rows, side):
    return np.array([row["uv"] for row in rows if row["curve"] == side]).reshape(-1, 2)


@pytest.mark.parametrize('blur_width', [1, 31])
def test_blue_material_selects_silhouette_without_using_chroma_as_edge(blur_width):
    rgb,paths,binding=_panel_scene(blur_width)
    rows,report=_read(rgb,paths,binding,material_signal='blue')
    assert report['source_component_method']=='blue_source_flood_exterior'
    for side,x in ((1,519.5),(3,119.5)):
        points=_side_uv(rows,side)
        assert len(points)>=6
        np.testing.assert_allclose(points[:,0],x,atol=.75)
    for row in rows:
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])
        assert row['model_UV_synthesized'] is False
    # A neutral dark shape with identical geometry cannot stand in for blue ink.
    neutral=np.repeat(rgb.min(2)[...,None],3,axis=2)
    missing,reason=_read(neutral,paths,binding,material_signal='blue')
    assert missing==[]
    assert reason['status']=='no_source_blue_panel_component'


def test_closed_silhouette_keeps_native_shape_when_camera_search_is_translated():
    rgb,paths,binding=_panel_scene(31)
    closed=np.array([[119.5,79.5],[519.5,79.5],[519.5,279.5],[119.5,279.5],[119.5,79.5]])+[24.,9.]
    original=closed.copy()
    rows,report=_read(rgb,[closed],binding,material_signal='blue',closed_paths=[True])
    assert len(rows)>=15
    shape=report['reports'][0]
    assert shape['curve_geometry']=='native_closed_outline'
    assert shape['disposition']=='source_closed_outline_supported'
    assert shape['organizing_shape_is_not_output_measurement']
    for row in rows:
        u,v=row['uv']
        assert min(abs(u-119.5),abs(u-519.5),abs(v-79.5),abs(v-279.5))<.75
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])
    np.testing.assert_array_equal(closed,original)


@pytest.mark.parametrize("blur_width", [1, 31], ids=["sharp", "horizontal_motion_blur"])
def test_current_rgb_outer_q50_rejects_letter_and_qr_holes_despite_offset_guides(blur_width):
    rgb, paths, binding = _panel_scene(blur_width)
    original_rgb = rgb.copy()
    original_paths = [path.copy() for path in paths]
    rgb.flags.writeable = False
    for path in paths:
        path.flags.writeable = False
    rows, report = _read(rgb, paths, binding)
    assert report["status"] == "source_panel_exterior_read"
    assert report["source_UV_unchanged"]
    assert report["source_component_used_for_topology_only"]
    for side, actual_x in ((1, 519.5), (3, 119.5)):
        edge_rows = [row for row in rows if row["curve"] == side]
        edge_uv = _side_uv(rows, side)
        assert len(edge_rows) >= 6
        assert np.ptp(edge_uv[:, 1]) >= 120.
        np.testing.assert_allclose(edge_uv[:, 0], actual_x, atol=.75)
        assert np.all(np.abs(edge_uv[:, 0]-paths[side][0, 0]) > 8.)
        for row in edge_rows:
            quantiles = np.asarray(row["actual_source_quantiles"])
            assert quantiles.shape == (2, 3, 2)
            assert np.isfinite(quantiles).all()
            np.testing.assert_array_equal(row["uv"], quantiles[0, 1])
            np.testing.assert_array_equal(quantiles[0], quantiles[1])
            normal = np.asarray(row["normal"])
            assert np.all(np.diff(quantiles[0] @ normal) > 0.)
            if blur_width > 1:
                assert np.linalg.norm(quantiles[0, 2]-quantiles[0, 0]) > 10.
            assert row["model_UV_synthesized"] is False
            assert row["measurement_operator"] == "source_outer_panel_material_interface"
    # The RGB reader encountered interior transitions; topology must exclude
    # them rather than relying on the projected guide being the nearest edge.
    interior = [candidate["uv"] for profile in report["profile_debug"]
        for candidate in profile["candidates"] if 145. < candidate["uv"][0] < 495.]
    assert interior
    np.testing.assert_array_equal(rgb, original_rgb)
    for path, original in zip(paths, original_paths):
        np.testing.assert_array_equal(path, original)


@pytest.mark.parametrize("side,x_bounds", [(1, (495, 545)), (3, (90, 150))],
                         ids=["right_edge_masked", "left_edge_masked"])
def test_occluded_vertical_exterior_stays_missing_without_filling_from_guides(side, x_bounds):
    rgb, paths, binding = _panel_scene()
    baseline, _ = _read(rgb, paths, binding)
    baseline_uv = _side_uv(baseline, side)
    assert np.any((baseline_uv[:, 1] >= 145) & (baseline_uv[:, 1] < 205))
    exclusion = np.zeros(rgb.shape[:2], dtype=bool)
    exclusion[145:205, x_bounds[0]:x_bounds[1]] = True
    original_mask = exclusion.copy()
    exclusion.flags.writeable = False
    rows, report = _read(rgb, paths, binding, exclusion)
    assert report["status"] == "source_panel_exterior_read"
    measured = _side_uv(rows, side)
    assert len(measured)
    assert not np.any((measured[:, 1] >= 145) & (measured[:, 1] < 205))
    expected = baseline_uv[(baseline_uv[:, 1] < 145) | (baseline_uv[:, 1] >= 205)]
    np.testing.assert_array_equal(measured, expected)
    other_side = 3 if side == 1 else 1
    np.testing.assert_array_equal(_side_uv(rows, other_side), _side_uv(baseline, other_side))
    for row in rows:
        x, y = np.rint(row["uv"]).astype(int)
        assert not exclusion[y, x]
    np.testing.assert_array_equal(exclusion, original_mask)


def _red_panel_scene(blur_width=31):
    rgb, paths, binding = _panel_scene(1)
    blue = np.all(rgb == [18, 42, 110], axis=2)
    rgb[blue] = [210, 30, 30]
    if blur_width > 1:
        rgb = cv2.blur(rgb, (blur_width, 1))
    return rgb, paths, binding


@pytest.mark.parametrize('blur_width', [1, 31], ids=['sharp_red', 'blurred_red'])
def test_red_field_uses_rgb_q50_and_filled_exterior_to_reject_white_letter_edges(blur_width):
    rgb, paths, binding = _red_panel_scene(blur_width)
    original_rgb = rgb.copy()
    original_paths = [path.copy() for path in paths]
    rgb.flags.writeable = False
    for path in paths:
        path.flags.writeable = False
    rows, report = _read(rgb, paths, binding, material_signal='red')
    assert report['status'] == 'source_panel_exterior_read'
    assert report['source_component_used_for_topology_only']
    assert report['source_UV_unchanged']
    for side, boundary in ((1, 519.5), (3, 119.5)):
        edge = _side_uv(rows, side)
        assert len(edge) >= 6
        np.testing.assert_allclose(edge[:, 0], boundary, atol=.75)
        assert np.all(np.abs(edge[:, 0]-paths[side][0, 0]) > 8.)
        for row in (row for row in rows if row['curve'] == side):
            quantiles = np.asarray(row['actual_source_quantiles'])
            np.testing.assert_array_equal(row['uv'], quantiles[0, 1])
            assert np.all(np.diff(quantiles[0] @ np.asarray(row['normal'])) > 0.)
            if blur_width > 1:
                assert np.linalg.norm(quantiles[0, 2]-quantiles[0, 0]) > 10.
            assert row['model_UV_synthesized'] is False
    # Interior white-letter and icon transitions are present in the actual
    # profiles, but every selected readout lies on the red field's exterior.
    assert any(145. < candidate['uv'][0] < 495. for profile in report['profile_debug']
               for candidate in profile['candidates'])
    for row in rows:
        u, v = row['uv']
        assert min(abs(u-119.5), abs(u-519.5), abs(v-79.5), abs(v-279.5)) < .75
    np.testing.assert_array_equal(rgb, original_rgb)
    for path, original in zip(paths, original_paths):
        np.testing.assert_array_equal(path, original)


@pytest.mark.parametrize('side,x_bounds', [(1, (495, 545)), (3, (90, 150))])
def test_blurred_red_exterior_mask_does_not_create_or_fill_boundary_pixels(side, x_bounds):
    rgb, paths, binding = _red_panel_scene()
    baseline, _ = _read(rgb, paths, binding, material_signal='red')
    original = _side_uv(baseline, side)
    assert np.any((original[:, 1] >= 145) & (original[:, 1] < 205))
    exclusion = np.zeros(rgb.shape[:2], dtype=bool)
    exclusion[145:205, x_bounds[0]:x_bounds[1]] = True
    rows, report = _read(rgb, paths, binding, exclusion, material_signal='red')
    assert report['status'] == 'source_panel_exterior_read'
    edge = _side_uv(rows, side)
    expected = original[(original[:, 1] < 145) | (original[:, 1] >= 205)]
    np.testing.assert_array_equal(edge, expected)
    assert not np.any((edge[:, 1] >= 145) & (edge[:, 1] < 205))
    for row in rows:
        u, v = np.rint(row['uv']).astype(int)
        assert not exclusion[v, u]


def test_red_chroma_without_luminance_edge_does_not_invent_panel_q50():
    rgb, paths, binding = _red_panel_scene(1)
    outside = np.ones(rgb.shape[:2], dtype=bool)
    outside[80:280, 120:520] = False
    rgb[outside] = 84
    rgb = cv2.blur(rgb, (31, 1))
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert luminance[100, 100] == luminance[100, 140] == 84
    rows, _ = _read(rgb, paths, binding, material_signal='red')
    assert rows == []


@pytest.mark.parametrize('paths', [[], [np.empty((0, 2))], [[]],
    [np.full((2, 2), np.nan)], [np.array([[np.inf, 12.], [14., -np.inf]])]])
def test_empty_or_nonfinite_projected_panel_paths_are_outside_source(paths):
    rgb, _, binding = _red_panel_scene()
    for signal in ('dark', 'red'):
        rows, report = _read(rgb, paths, binding, material_signal=signal)
        assert rows == []
        assert report['status'] == 'outside_source'


def test_red_panel_clipped_at_image_border_does_not_emit_a_navigation_boundary():
    rgb, paths, binding = _red_panel_scene()
    # Crop through the physical left edge; a projected x=0 side has no RGB
    # transition or complete profile and cannot become an observed boundary.
    rgb = rgb[:, 150:].copy()
    paths = [path-[150., 0.] for path in paths]
    paths[3][:, 0] = 0.
    binding = dict(binding, native_hw=list(rgb.shape[:2]))
    rows, _ = _read(rgb, paths, binding, material_signal='red')
    assert not any(row['curve'] == 3 for row in rows)
    assert any(row['curve'] == 1 for row in rows)


def _nested_red_panel_with_low_resolution_chroma():
    rgb=np.full((360,640,3),180,np.uint8)
    rings=[((80,280,120,520),(150,70,70)),((82,278,122,518),(180,180,180)),
           ((83,277,123,517),(160,40,40)),((85,275,125,515),(180,180,180)),
           ((86,274,126,514),(170,30,30))]
    for (y0,y1,x0,x1),color in rings:rgb[y0:y1,x0:x1]=color
    for x in (170,205,240,275):rgb[125:150,x:x+12]=180
    rgb[125:235,390:485]=180
    # Retain the native sharp Y plane while deliberately lowering chroma
    # bandwidth: fine red/white rings become a monotonic chroma staircase.
    ycc=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)
    for channel in (1,2):
        low=cv2.resize(ycc[...,channel],(80,45),interpolation=cv2.INTER_AREA)
        ycc[...,channel]=cv2.resize(low,(640,360),interpolation=cv2.INTER_LINEAR)
    rgb=cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB)
    corners=np.array([[126.,86.],[514.,86.],[514.,274.],[126.,274.]])
    paths=[np.array([corners[i],corners[(i+1)%4]]) for i in range(4)]
    binding=dict(source_sha256='b'*64,source_pts=100,time_base='1/90000',
                 shot_id='synthetic_nested_rings',native_hw=list(rgb.shape[:2]))
    return rgb,paths,binding


def test_nested_red_white_rings_keep_outer_luminance_q50_despite_chroma_staircase_and_nearer_guide():
    rgb,paths,binding=_nested_red_panel_with_low_resolution_chroma()
    original=rgb.copy();rgb.flags.writeable=False
    chroma=rgb[75:91,300,0].astype(float)-rgb[75:91,300,1]
    assert np.all(np.diff(chroma)>=0.)
    rows,report=_read(rgb,paths,binding,material_signal='red')
    assert report['status']=='source_panel_exterior_read'
    for curve,axis,outer in ((0,1,79.5),(1,0,519.5),(2,1,279.5),(3,0,119.5)):
        edge=np.array([row['uv'] for row in rows if row['curve']==curve])
        assert len(edge)>=6
        np.testing.assert_allclose(edge[:,axis],outer,atol=.75)
    # At this same actual RGB query, luminance resolves all three outward
    # interfaces, while chroma has one midpoint between the physical rings.
    frame=_MaterialProfileFrame(rgb,binding)
    query=np.array([[306.,86.]]);normal=np.array([[0.,-1.]])
    candidates={}
    for signal,transition,separate in (('dark','dark_to_light',True),('red','light_to_dark',False)):
        offsets,profile,smooth,valid=_batched_actual_profiles(frame,query,normal,
            np.array([[-30.,30.]]),np.array([signal]))
        candidates[signal]=_actual_profile_candidates(query[0],normal[0],offsets[0,valid[0]],
            profile[0,valid[0]],smooth[0,valid[0]],operator='material_interface',signal=signal,
            transition=transition,with_quantiles=True,separate_bands=separate)
    luminance=[row for row in candidates['dark'] if not row[2]]
    assert len(luminance)==3 and len(candidates['red'])==1
    outer=min(luminance,key=lambda row:row[0][1])
    assert abs(candidates['red'][0][0][1]-outer[0][1])>2.
    assert min(abs(row[0][1]-query[0,1]) for row in luminance)<1.
    actual=[row for row in rows if row['curve']==0 and row['uv'][0]==306.]
    assert len(actual)==1
    np.testing.assert_allclose(actual[0]['uv'],outer[0],atol=1e-12)
    assert abs(actual[0]['uv'][1]-query[0,1])>5.
    np.testing.assert_array_equal(actual[0]['uv'],np.asarray(actual[0]['actual_source_quantiles'])[0,1])
    assert actual[0]['model_UV_synthesized'] is False
    np.testing.assert_array_equal(rgb,original)


@pytest.mark.parametrize('masked', [False,True], ids=['right_frame_clip','right_frame_clip_and_mask'])
def test_clipped_nested_red_rings_keep_real_outer_edges_without_closing_on_frame_or_mask(masked):
    rgb,paths,binding=_nested_red_panel_with_low_resolution_chroma()
    rgb=rgb[:,:480].copy()  # All three right-hand rings are physically outside this image.
    binding=dict(binding,native_hw=list(rgb.shape[:2]))
    exclusion=np.zeros(rgb.shape[:2],bool)
    if masked:exclusion[70:100,246:306]=True
    original=rgb.copy();original_paths=[path.copy() for path in paths]
    original_mask=exclusion.copy()
    rows,report=_read(rgb,paths,binding,exclusion,material_signal='red')
    assert report['status']=='source_panel_exterior_read'
    assert not any(row['curve']==1 for row in rows)
    for curve,axis,boundary in ((0,1,79.5),(2,1,279.5),(3,0,119.5)):
        points=np.array([row['uv'] for row in rows if row['curve']==curve])
        assert len(points)>=6
        np.testing.assert_allclose(points[:,axis],boundary,atol=.75)
    for row in rows:
        u,v=np.rint(row['uv']).astype(int)
        assert u<rgb.shape[1]-4  # The frame cannot supply the missing right boundary.
        assert not exclusion[v,u]
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])
        assert row['model_UV_synthesized'] is False
    np.testing.assert_array_equal(rgb,original)
    np.testing.assert_array_equal(exclusion,original_mask)
    for path,original_path in zip(paths,original_paths):np.testing.assert_array_equal(path,original_path)


def test_red_topology_without_reliable_non_frame_outside_seeds_stays_unresolved():
    rgb,paths,binding=_nested_red_panel_with_low_resolution_chroma()
    # The search ROI reaches all four original frame borders: none is a
    # reliable outside seed even though the frame contains visible background.
    rows,report=_read(rgb,paths,binding,material_signal='red',search_radius=200.)
    assert rows==[] and report['status']=='panel_topology_unresolved'
    # Artificial ROI borders covered by dark material are not outside either.
    rgb=np.full((360,640,3),[170,30,30],np.uint8)
    rgb[160:200,280:360]=180
    corners=np.array([[270.,150.],[370.,150.],[370.,210.],[270.,210.]])
    paths=[np.array([corners[i],corners[(i+1)%4]]) for i in range(4)]
    rows,report=_read(rgb,paths,binding,material_signal='red',search_radius=20.)
    assert rows==[] and report['status']=='panel_topology_unresolved'


def test_source_panel_split_by_person_retains_both_visible_halves():
    rgb, paths, binding = _panel_scene(blur_width=1)
    mask = np.zeros(rgb.shape[:2], bool)
    mask[45:310, 290:350] = True
    rows, _ = _read(rgb, paths, binding, mask)
    top = _side_uv(rows, 0)
    bottom = _side_uv(rows, 2)
    assert np.any(top[:, 0] < 280.) and np.any(top[:, 0] > 360.)
    assert np.any(bottom[:, 0] < 280.) and np.any(bottom[:, 0] > 360.)
    assert len(_side_uv(rows, 1)) >= 6 and len(_side_uv(rows, 3)) >= 6
    for row in rows:
        x, y = np.rint(row['uv']).astype(int)
        assert not mask[y, x]
        assert min(abs(x-120), abs(x-519), abs(y-80), abs(y-279)) < 2.


def test_disconnected_colored_stripes_keep_graphic_exterior_roles():
    rgb = np.full((360, 640, 3), 235, np.uint8)
    rgb[80:120, 120:520] = [18, 70, 170]
    rgb[240:280, 120:520] = [30, 150, 70]
    # A dark isolated word in the white interior is a separate source
    # component. Neither it nor the green stripe's inner edge is the top.
    rgb[170:185, 235:385] = 40
    _, paths, binding = _panel_scene(1)
    rows, _ = _read(rgb, paths, binding)
    top, bottom = _side_uv(rows, 0), _side_uv(rows, 2)
    assert len(top) >= 12 and len(bottom) >= 12
    np.testing.assert_allclose(top[:, 1], 79.5, atol=.75)
    np.testing.assert_allclose(bottom[:, 1], 279.5, atol=.75)
    for row in rows:
        q = np.asarray(row['actual_source_quantiles'])
        np.testing.assert_array_equal(row['uv'], q[0, 1])



@pytest.mark.parametrize('blur_width', [1, 31], ids=['sharp', 'horizontal_motion_blur'])
def test_disconnected_blue_green_fields_do_not_use_white_gap_as_material_support(blur_width):
    rgb = np.full((360, 640, 3), 235, np.uint8)
    rgb[80:160, 120:520] = [18, 70, 170]
    rgb[200:280, 120:520] = [30, 150, 70]
    # The open white seam has a small luminance step against the surround,
    # as can occur with a shaded white board or broadcast ringing. It offers
    # a real RGB quantile but no ink component on the inward side. A convex
    # envelope alone would admit that step as a measured vertical panel edge.
    rgb[160:200, 120:520] = 205
    if blur_width > 1:
        rgb = cv2.blur(rgb, (blur_width, 1))
    _, paths, binding = _panel_scene(1)
    original = rgb.copy()
    rows, report = _read(rgb, paths, binding)
    assert report['source_component_count'] == 2
    assert report['status'] == 'source_panel_exterior_read'
    for side, actual_x in ((1, 519.5), (3, 119.5)):
        edge = _side_uv(rows, side)
        assert np.any(edge[:, 1] < 160.) and np.any(edge[:, 1] >= 200.)
        assert not np.any((edge[:, 1] >= 160.) & (edge[:, 1] < 200.))
        np.testing.assert_allclose(edge[:, 0], actual_x, atol=.75)
    # The rejected seam readings still exist in the original RGB profiles;
    # the source material support, not an output UV adjustment, rejects them.
    assert any(160. <= profile['query'][1] < 200. and profile['candidates']
               for profile in report['profile_debug'])
    assert len(_side_uv(rows, 0)) >= 12 and len(_side_uv(rows, 2)) >= 12
    for row in rows:
        np.testing.assert_array_equal(row['uv'], np.asarray(row['actual_source_quantiles'])[0, 1])
        assert row['model_UV_synthesized'] is False
    np.testing.assert_array_equal(rgb, original)


def test_long_person_occlusion_does_not_erase_two_visible_side_fragments():
    rgb, paths, binding = _panel_scene(1)
    mask=np.zeros(rgb.shape[:2],bool)
    # A broad normal search extends past this body-shaped occluder. Its
    # unmasked far end is not evidence that the physical panel edge is visible.
    mask[105:255,490:550]=True
    rows,report=_read(rgb,paths,binding,mask)
    right=_side_uv(rows,1)
    assert len(right)>=2
    assert np.any(right[:,1]<105) and np.any(right[:,1]>255)
    assert not np.any((right[:,1]>=105)&(right[:,1]<255))
    np.testing.assert_allclose(right[:,0],519.5,atol=.75)
    for row in rows:
        np.testing.assert_array_equal(row['uv'],np.asarray(row['actual_source_quantiles'])[0,1])


def test_source_visibility_never_removes_measured_queries_or_unmasked_missing_spans():
    from rinkspace.session.panel_boundaries import _source_supported_visibility
    q=np.array([[10.,10.],[10.,20.],[10.,30.],[10.,40.]])
    measurement=dict(query_uv=q,query_normal=np.tile([1.,0.],(4,1)),query_interval=np.tile([-5.,5.],(4,1)),candidate_query=np.array([0,3]))
    report=dict(curve=0,disposition='insufficient_distributed_source_line_span',visible_source_queries=[0,1,2,3],source_organizer=dict(coefficients=[0.,2.],station_center=15.,competing_family_support_length_px=0.))
    mask=np.zeros((60,40),bool);mask[10,9:16]=True;mask[20,9:16]=True
    support,inferred=_source_supported_visibility(measurement,[report],mask)
    np.testing.assert_array_equal(support,[True,False,True,True])
    assert inferred=={0:[1]}


def test_rejected_raw_alternatives_do_not_block_source_line_mask_visibility():
    from rinkspace.session.panel_boundaries import _source_supported_visibility
    q=np.array([[10.,10.],[10.,20.],[10.,30.],[10.,40.]])
    measurement=dict(query_uv=q,query_normal=np.tile([1.,0.],(4,1)),query_interval=np.tile([-8.,8.],(4,1)),candidate_query=np.arange(4),candidate_flags=np.array([0,3,1,0]))
    report=dict(curve=0,disposition='insufficient_distributed_source_line_span',visible_source_queries=[0,1,2,3],source_geometry_tolerance_px=3.,source_organizer=dict(coefficients=[0.,2.],station_center=15.,competing_family_support_length_px=0.))
    mask=np.zeros((60,40),bool);mask[10,9:16]=True;mask[20,9:16]=True
    support,inferred=_source_supported_visibility(measurement,[report],mask)
    np.testing.assert_array_equal(support,[True,False,True,True])
    assert inferred=={0:[1]}
    # An unresolved competing source line cannot choose the mask denominator.
    report['source_organizer']['competing_family_support_length_px']=5.
    support,inferred=_source_supported_visibility(measurement,[report],mask)
    assert support.all() and not inferred
    report['source_organizer']['competing_family_support_length_px']=0.
    # One masked center pixel cannot hide the permitted RGB-line center halo.
    mask[:]=False;mask[20,12]=True
    support,inferred=_source_supported_visibility(measurement,[report],mask)
    assert support.all() and not inferred
