"""Native source/support linkage without a window, weights or inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_viewer_integrity import DATA, PREPARE, Surface, catalog, replace_npz
from test_viewer_comparison_framing import app_fixture, app_method, FRAMING


PATH=Path(__file__).resolve().parents[1]/'src/rinkspace/viewer3d/source.py'
SPEC=importlib.util.spec_from_file_location('_viewer_source_support',PATH)
SOURCE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SOURCE)


def test_saved_source_fields_survive_cache_independently_without_moving_geometry(tmp_path):
    c=catalog(tmp_path)
    original=PREPARE.prepare_variant(c,'fixture',Surface())
    phase=np.array(['spin','unknown','exit'])
    mode=np.array([['unknown','glide'],['unknown','unknown'],['unknown','pick']])
    confidence=np.array([[0.,.7],[0.,0.],[0.,.8]])
    used=np.array([[False,True],[False,False],[False,False]])
    replace_npz(c.root/'result.npz',source_action_phase_mid=phase,source_support_mode_mid=mode,
                source_support_confidence_mid=confidence,support_image_in_optimization=used)
    changed=PREPARE.prepare_variant(c,'fixture',Surface())
    before,after=DATA.CachedSequence(original),DATA.CachedSequence(changed)
    for key in ('P','Q','G','gear','joints','coco'):
        np.testing.assert_array_equal(before.state[key],after.state[key])
    np.testing.assert_array_equal(before.vertices,after.vertices)
    for key,expected in [('source_action_phase',phase),('source_support_mode',mode),
                         ('source_support_confidence',confidence),('support_image_in_optimization',used)]:
        np.testing.assert_array_equal(after.state[key],expected)
        assert key not in before.state
    assert SOURCE.source_value(before.state,'source_support_mode',0,1)=='unknown'
    assert SOURCE.source_value(after.state,'source_support_mode',1,1)=='unknown'
    assert SOURCE.source_value(after.state,'source_action_phase',2,1)=='exit'


@pytest.mark.parametrize('key,value',[
    ('source_support_mode_mid',np.array(['glide'])),
    ('source_support_confidence_mid',np.full((3,2),np.nan)),
    ('support_image_in_optimization',np.ones((3,2),float)),
])
def test_invalid_saved_source_metadata_is_not_published(tmp_path,key,value):
    c=catalog(tmp_path);replace_npz(c.root/'result.npz',**{key:value})
    with pytest.raises(ValueError):PREPARE.prepare_variant(c,'fixture',Surface())
    assert not list(c.cache_root.glob('fixture_*/metadata.json'))


def test_source_regions_keep_exact_pts_roles_and_unknown(tmp_path):
    row=dict(id='a',pts=283136,role='fit',foot=1,centre_uv=[100.,200.],
             region_shape_px2=[[9.,0.],[0.,4.]])
    review=dict(row,id='b',role='review')
    path=tmp_path/'regions.json'
    path.write_text(json.dumps(dict(time_base=[1,12800],shot_id='s05',observations=[row,review])))
    regions=SOURCE.SupportObservations(path,(1,12800),shot='s05')
    assert regions.at(283392)==[]
    assert [r['role'] for r in regions.at(283136)]==['fit','review']
    outline=SOURCE.region_outline(row)
    np.testing.assert_allclose(outline.max(0),[103.,202.])
    assert SOURCE.observation_used({},row,0)=='unknown'
    state=dict(support_image_in_optimization=np.array([[False,True]]))
    assert SOURCE.observation_used(state,row,0) is True
    assert SOURCE.observation_used(state,review,0) is False
    with pytest.raises(ValueError):regions.at(283136.0)
    with pytest.raises(ValueError):SOURCE.SupportObservations(path,(1,1000))


@pytest.mark.parametrize('view',['support_top','support_side'])
def test_fixed_support_foot_camera_changes_no_pts_or_saved_geometry(tmp_path,view):
    player,sequences,_=app_fixture(tmp_path,comparison=False)
    player.catalog.config['support_foot']=1
    player.support_view_center=app_method('support_view_center',{'np':np}).__get__(player)
    player.set_view=app_method('set_view',{'np':np}).__get__(player)
    # A much lower other foot must not change the explicit source association.
    player.sequence.state['gear'][0,0,:,2]=-100.
    before={key:value.copy() for key,value in player.sequence.state.items()}
    player.set_view(view)
    np.testing.assert_allclose(player.scene.camera.target,player.sequence.state['gear'][0,1].mean(0))
    assert player.scene.camera.ortho_size==.32 and player.frame==0
    for key in before:np.testing.assert_array_equal(player.sequence.state[key],before[key])


def test_source_video_refuses_neighbouring_frame_instead_of_fps_substitution():
    from fractions import Fraction
    video=SOURCE.SourceVideo.__new__(SOURCE.SourceVideo)
    video.time_base=Fraction(1,12800)
    video._last_pts=0;video._image=None
    video._frames=iter([SimpleNamespace(pts=512,time_base=Fraction(1,12800))])
    with pytest.raises(ValueError,match='No exact source frame'):
        video.read(256)
    assert video._frames is None


def test_source_panel_and_controls_leave_comparison_geometry_visible(tmp_path):
    player,sequences,_=app_fixture(tmp_path)
    player.catalog.source_video=Path('source.mp4')
    player.set_view('side')
    assert player.comparison_framing['right_inset_pixels']==650
    for sequence in sequences.values():
        bounds=FRAMING.comparison_frustum_bounds(player,sequence,0,left_inset=453,right_inset=650)
        assert bounds['all_body_and_gear_in_view_region']
