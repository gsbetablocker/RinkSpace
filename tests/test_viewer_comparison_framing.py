"""Diagnostic camera and clock regressions without a window or body assets."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest


SOURCE=Path(__file__).resolve().parents[1]/'src/rinkspace/viewer3d/verify_fullbody.py'
NAMES={'continuous_playback'}
TREE=ast.parse(SOURCE.read_text(encoding='utf8'))
FRAMING_PATH=SOURCE.with_name('framing.py')
SPEC=importlib.util.spec_from_file_location('_comparison_framing',FRAMING_PATH)
FRAMING=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FRAMING)
HELPERS={'np':np,'time':time,**{name:getattr(FRAMING,name) for name in ('comparison_points','fit_comparison_camera','comparison_frustum_bounds')}}
exec(compile(ast.Module(body=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in NAMES],type_ignores=[]),str(SOURCE),'exec'),HELPERS)


class Camera:
    def __init__(self,offset,up):
        self.target=np.array([0.,0.,.8]);self.position=self.target+offset;self.up=np.array(up,dtype=float)
        self.near=.01;self.far=100.;self.is_ortho=False;self.ortho_size=1.

    def update_matrices(self,width,height):
        back=self.position-self.target;back/=np.linalg.norm(back)
        right=np.cross(self.up,back);right/=np.linalg.norm(right);up=np.cross(back,right)
        camera_world=np.eye(4);camera_world[:3,:3]=np.stack([right,up,back],axis=1);camera_world[:3,3]=self.position
        projection=np.eye(4);projection[0,0]=height/(width*self.ortho_size);projection[1,1]=1/self.ortho_size
        projection[2,2]=-2/(self.far-self.near);projection[2,3]=-(self.far+self.near)/(self.far-self.near)
        self.matrix=projection@np.linalg.inv(camera_world)

    def get_view_projection_matrix(self):return self.matrix


def sequence(offset):
    offset=np.array(offset,dtype=float)
    joints=np.array([[0,0,0],[0,0,1.6],[-.7,0,1],[.7,0,1]])+offset
    gear=np.array([[[-.1,-.2,0],[.1,.2,0]],[[-.1,-.2,.1],[.1,.2,.1]]])+offset
    surface=np.array([[-.85,-.2,-.04],[.85,.2,1.75]])+offset
    return SimpleNamespace(state=dict(joints=joints[None],gear=gear[None],boot_surface=(gear+.02)[None],P=offset[None],pts=np.array([283136])),vertices=surface[None])


@pytest.mark.parametrize('offset,up',[([0,-8,0],[0,0,1]),([2,-2.7,1.3],[0,0,1]),([0,0,10],[0,1,0])])
@pytest.mark.parametrize('size',[(960,720),(720,960)])
def test_shared_camera_keeps_bodies_six_metres_apart_inside_including_surface(offset,up,size):
    sequences={'old':sequence([0,0,0]),'new':sequence([6,-2,.3])}
    before={name:seq.vertices.copy() for name,seq in sequences.items()}
    camera=Camera(offset,up);direction=(camera.target-camera.position)/np.linalg.norm(offset)
    player=SimpleNamespace(scene=SimpleNamespace(camera=camera),window_size=size)
    framing=HELPERS['fit_comparison_camera'](player,sequences,0)
    assert camera.is_ortho and framing['camera_only']
    np.testing.assert_allclose((camera.target-camera.position)/np.linalg.norm(camera.target-camera.position),direction,atol=1e-12)
    # The joint/gear AABB fixes the common center; no per-candidate recentering.
    np.testing.assert_allclose(camera.target,[3.,-1.,.95],atol=1e-12)
    for name,seq in sequences.items():
        assert HELPERS['comparison_frustum_bounds'](player,seq,0)['all_body_and_gear_inside']
        np.testing.assert_array_equal(seq.vertices,before[name])


def test_nonfinite_body_geometry_cannot_claim_valid_framing():
    seq=sequence([0,0,0]);seq.vertices[0,0,0]=np.nan
    player=SimpleNamespace(scene=SimpleNamespace(camera=Camera([0,-8,0],[0,0,1])),window_size=(960,720))
    with pytest.raises(ValueError,match='Invalid comparison framing geometry'):
        HELPERS['fit_comparison_camera'](player,{'broken':seq},0)


class ClockPlayer:
    playback_fps=50.
    def __init__(self,catalog,*,skip=False):
        self.catalog=catalog;self.sequence=SimpleNamespace(state=dict(pts=catalog.pts.copy()));self.frame=0;self.skip=skip
    def seek(self,frame):self.frame=frame;self.run_animations=False
    def toggle_animation(self,run):self.run_animations=run;self._last_frame_rendered_at=0.
    def on_render(self,render_time,frame_time,export):
        assert self.run_animations and export
        frames=int((render_time-self._last_frame_rendered_at)*self.playback_fps)
        self.frame+=frames+(1 if self.skip else 0)
        self._last_frame_rendered_at+=frames/self.playback_fps


def clock_catalog():
    pts=283136+np.arange(145)*256
    catalog=SimpleNamespace(pts=pts,seconds=pts/12800.,time_base=(1,12800))
    catalog.frame_for_seconds=lambda seconds:int(np.argmin(abs(catalog.seconds-seconds)))
    return catalog


def test_requested_opening_playback_records_all_145_source_exposures():
    catalog=clock_catalog();player=ClockPlayer(catalog)
    report=HELPERS['continuous_playback'](player,catalog,[22.12,25.0])
    assert report['no_skipped_or_repeated_exposures'] and report['exposures']==145
    assert report['rendered_steps']==144 and report['playback_fps']==50.
    assert report['source_first_last_pts']==[283136,320000]
    assert report['source_pts_deltas']==[256]*144
    assert [row['local_frame'] for row in report['records']]==list(range(145))
    assert not player.run_animations


def test_clock_check_rejects_skipped_exposure_and_still_pauses():
    catalog=clock_catalog();player=ClockPlayer(catalog,skip=True)
    with pytest.raises(AssertionError,match='skipped or repeated'):
        HELPERS['continuous_playback'](player,catalog,[22.12,25.0])
    assert not player.run_animations


def app_method(name,namespace):
    source=SOURCE.with_name('app.py');tree=ast.parse(source.read_text(encoding='utf8'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ResultsPlayer')
    node=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace[name]


def app_fixture(tmp_path,*,comparison=True):
    sequences={'old':sequence([0,0,0]),'new':sequence([6,-2,.3])};calls=[]
    for name,seq in sequences.items():
        seq.path=tmp_path/name;seq.path.mkdir();(seq.path/'metadata.json').write_text('{}')
    def cached(path):calls.append(path);return next(seq for seq in sequences.values() if seq.path==path)
    catalog=SimpleNamespace(config={'comparison_variants':['old','new']} if comparison else {},pts=np.array([283136]),
                            variant=lambda name:{'id':name},cache_path=lambda variant:(sequences[variant['id']].path,{}))
    player=SimpleNamespace(catalog=catalog,variant={'id':'new'},sequence=sequences['new'],frame=0,follow=True,
                           scene=SimpleNamespace(camera=Camera([2,-2.7,1.3],[0,0,1])),window_size=(1600,900),save_session=Mock())
    namespace={'np':np,'CachedSequence':cached,'fit_comparison_camera':FRAMING.fit_comparison_camera}
    player.comparison_sequences=app_method('comparison_sequences',namespace).__get__(player)
    player.set_view=app_method('set_view',namespace).__get__(player)
    return player,sequences,calls


@pytest.mark.parametrize('view',['free','top','side','end'])
def test_actual_app_view_frames_both_candidates_clear_of_controls_and_reuses_cache(tmp_path,view):
    player,sequences,calls=app_fixture(tmp_path)
    player.set_view(view)
    assert player.comparison_framing['left_inset_pixels']==453
    for seq in sequences.values():
        bounds=FRAMING.comparison_frustum_bounds(player,seq,0,left_inset=453)
        assert bounds['all_body_and_gear_inside'] and bounds['all_body_and_gear_in_view_region']
    assert calls==[sequences['old'].path]
    player.set_view(view)
    assert calls==[sequences['old'].path]  # No repeated sequence loading or inference.


@pytest.mark.parametrize('view,fit_comparison,comparison',[
    ('feet_oblique',True,True),('free',False,True),('free',True,False),
])
def test_feet_normal_and_restored_views_do_not_apply_comparison_framing(tmp_path,view,fit_comparison,comparison):
    player,sequences,calls=app_fixture(tmp_path,comparison=comparison)
    player.set_view(view,fit_comparison=fit_comparison)
    assert player.comparison_framing is None and calls==[]
    if view=='feet_oblique':assert player.scene.camera.ortho_size==.34
    else:assert not player.scene.camera.is_ortho


@pytest.mark.parametrize('view,axis',[('feet_side',1),('feet_end',0)])
def test_dense_skate_inspection_camera_uses_saved_template_frame_and_never_moves_mesh(tmp_path,view,axis):
    player,sequences,_=app_fixture(tmp_path);state=player.sequence.state
    lower=np.arange(2*24*3,dtype=float).reshape(1,2,24,3)/100
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    state.update(skate_lower=lower.copy(),skate_R=np.broadcast_to(R,(1,2,3,3)).copy())
    player.set_view(view)
    center=lower[0].reshape(-1,3).mean(axis=0)
    np.testing.assert_allclose(player.scene.camera.target,center)
    np.testing.assert_allclose(player.scene.camera.position,center+R[:,axis]*.95+[0.,0.,.16])
    np.testing.assert_array_equal(state['skate_lower'],lower)
