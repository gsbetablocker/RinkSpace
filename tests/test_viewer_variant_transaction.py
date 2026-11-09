"""Exercise the production switch method without importing a GUI or model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'src/rinkspace/viewer3d/app.py'


def switch_method(namespace, source=SOURCE, name='load_variant'):
    tree = ast.parse(Path(source).read_text(encoding='utf8'))
    player = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResultsPlayer')
    method = next(n for n in player.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def headless_player_class(namespace):
    """Use production methods with no GUI imports or window construction."""
    tree = ast.parse(SOURCE.read_text(encoding='utf8'))
    player = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResultsPlayer')
    namespace['Viewer'] = object
    exec(compile(ast.Module(body=[player], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['ResultsPlayer']


def restore_method(namespace, source=SOURCE):
    """Run configuration and session initialization around the GUI setup."""
    tree = ast.parse(source.read_text(encoding='utf8'))
    player = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResultsPlayer')
    method = next(n for n in player.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    setup = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Expr))
    begin = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'old' for t in n.targets))
    method.name = 'restore_session'
    method.body = method.body[:setup] + method.body[begin:]
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['restore_session']


class InjectedFailure(RuntimeError):
    pass


class RenderNode:
    def __init__(self, name, fixture=None):
        self.name = name
        self.fixture = fixture
        self.releases = 0
        self._frame = 0
        self._enabled = True

    @property
    def current_frame_id(self):
        return self._frame

    @current_frame_id.setter
    def current_frame_id(self, value):
        if self.fixture and self.fixture.fail_frame == self.name:
            raise self.fixture.error
        self._frame = value

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if self.fixture and self.fixture.fail_visibility == self.name:
            raise self.fixture.error
        self._enabled = value

    def release(self):
        self.releases += 1


class SwitchFixture:
    def __init__(self, path, *, gear_points=48, trajectory_reference='pelvis'):
        self.error = InjectedFailure('injected renderer failure')
        self.fail_constructor = self.fail_frame = self.fail_visibility = None
        self.fail_add = None
        self.fail_hull = False
        self.created = []
        self.mesh_inputs = []
        self.add_calls = 0
        self.clear_cache = Mock()
        self.pts = np.array([101, 111, 141], np.int64)
        self.seconds = self.pts / 1000.
        self.sequence = SimpleNamespace(
            vertices=np.zeros((3, 4, 3)), faces=np.array([[0, 1, 2]], np.int32),
            state=dict(pts=self.pts.copy(), parents=np.array([-1, 0, 1]),
                       joints=np.zeros((3, 3, 3)), gear=np.zeros((3, 2, gear_points, 3)),
                       P=np.arange(9.).reshape(3, 3),
                       G=np.column_stack([np.arange(3.), -np.arange(3.), np.zeros(3)])))
        if trajectory_reference == 'com':
            t = self.seconds - self.seconds[0]
            self.sequence.state['com_world_mid'] = np.column_stack([
                10. + 2*t + 40*t**2, -3. + 3*t - 15*t**2, .9 - t + 10*t**2])
        self.definition = dict(id='new')
        self.old_nodes = [RenderNode('old body'), RenderNode('old skeleton')]
        self.static = RenderNode('rink')
        self.old_layers = dict(surface=[self.old_nodes[0]], skeleton=[self.old_nodes[1]],
                               shoes=[], trajectory=[], pelvis=[], ice=[self.static])
        self.camera = SimpleNamespace(position=np.array([4., 5., 6.]), target=np.array([1., 2., 3.]))
        self.scene = SimpleNamespace(nodes=[self.static, *self.old_nodes], camera=self.camera,
                                     current_frame_id=1, add=self.add, remove=self.remove)
        self.namespace = dict(np=np, CachedSequence=lambda _: self.sequence,
                              Meshes=self.meshes, Skeletons=self.construct, Lines=self.construct, Spheres=self.construct,
                              shell_triangles=self.hull, ribbon_triangles=lambda _: np.array([[0, 1, 2]]))
        self.namespace['Meshes'] = Mock(side_effect=self.meshes)
        self.namespace['Meshes'].compute_vertex_and_face_normals.cache_clear = self.clear_cache
        player_class = headless_player_class(self.namespace)
        self.player = player_class.__new__(player_class)
        old_track = np.arange(4., 13.).reshape(3, 3)
        self.player.__dict__.update(
            catalog=SimpleNamespace(variant=lambda _: self.definition, cache_path=lambda _: (path, {}),
                                    pts=self.pts, seconds=self.seconds, config={}),
            sequence=SimpleNamespace(state=dict(gear=np.zeros((3, 2, gear_points, 3)))),
            variant=dict(id='old'), variant_nodes=self.old_nodes,
            layers=self.old_layers, tracking_points=old_track, last_tracking_position=old_track[1].copy(),
            trajectory_reference=trajectory_reference,
            com_velocity=np.ones((3, 3)) if trajectory_reference == 'com' else None,
            scene=self.scene, surface_alpha=.72, error='old error', follow=True, speed=2., playback_fps=100.,
            save_session=Mock(),
            visible=dict(surface=False, skeleton=True, shoes=False, trajectory=True, pelvis=False, ice=False,
                         com_projection=False, pelvis_trajectory=False))
        self.before = {key: getattr(self.player, key) for key in
                       ('sequence', 'variant', 'variant_nodes', 'layers', 'tracking_points',
                        'last_tracking_position', 'com_velocity')}
        self.scene_before = list(self.scene.nodes)
        self.layers_before = {key: list(nodes) for key, nodes in self.old_layers.items()}
        self.switch = player_class.load_variant

    def meshes(self, *args, **kwargs):
        self.mesh_inputs.append((args,kwargs))
        return self.construct(*args, **kwargs)

    def construct(self, *args, **kwargs):
        name = kwargs['name']
        if name == self.fail_constructor:
            raise self.error
        node = RenderNode(name, self)
        node.points = args[0]
        self.created.append(node)
        return node

    def hull(self, points, **kwargs):
        if self.fail_hull:
            raise self.error
        return np.array([[0, 1, 2]], np.int32)

    def add(self, *nodes, enabled=True):
        for node in nodes:
            self.add_calls += 1
            # No old member or scene node may be lost while adding candidates.
            for key, value in self.before.items():
                assert getattr(self.player, key) is value, key
            assert all(old in self.scene.nodes for old in self.scene_before)
            if self.fail_add == (self.add_calls, 'before'):
                raise self.error
            node._enabled = enabled  # The installed Node.add resets this field.
            self.scene.nodes.append(node)
            if self.fail_add == (self.add_calls, 'after'):
                raise self.error

    def remove(self, *nodes):
        for node in nodes:
            node.release()
            if node in self.scene.nodes:
                self.scene.nodes.remove(node)

    def assert_unchanged(self, case):
        for key, value in self.before.items():
            case.assertIs(getattr(self.player, key), value, key)
        case.assertEqual(self.player.layers, self.layers_before)
        case.assertEqual(self.scene.nodes, self.scene_before)
        case.assertEqual(self.player.frame, 1)
        case.assertEqual(self.scene.current_frame_id, 1)
        case.assertIs(self.scene.camera, self.camera)
        np.testing.assert_array_equal(self.camera.position, [4., 5., 6.])
        np.testing.assert_array_equal(self.camera.target, [1., 2., 3.])
        case.assertTrue(all(n.releases == 0 for n in self.scene_before))
        case.assertTrue(all(n.releases == 1 for n in self.created))
        self.clear_cache.assert_not_called()
        case.assertEqual(self.player.error, 'old error')


class VariantTransactionTests(unittest.TestCase):
    def setUp(self):
        from tempfile import TemporaryDirectory
        self.temp = TemporaryDirectory(prefix='rs_switch_')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'metadata.json').write_text('{}')

    def assert_switch_failure(self, fixture):
        with self.assertRaises(InjectedFailure) as caught:
            fixture.switch(fixture.player, 'new')
        self.assertIs(caught.exception, fixture.error)
        fixture.assert_unchanged(self)

    def test_constructor_failures_preserve_every_active_member(self):
        for name in ('Saved full SMPL-X surface', 'Native skeleton', 'Boot candidate 1',
                     'Blade lower edge strip 2', 'P pelvis', 'G pelvis projection'):
            with self.subTest(constructor=name):
                f = SwitchFixture(self.path)
                f.fail_constructor = name
                self.assert_switch_failure(f)
                self.assertEqual(f.add_calls, 0)

    def test_boot_hull_failure_occurs_before_scene_commit(self):
        f = SwitchFixture(self.path)
        f.fail_hull = True
        self.assert_switch_failure(f)
        self.assertEqual(f.add_calls, 0)

    def test_frame_and_visibility_failures_occur_before_scene_commit(self):
        for attribute in ('fail_frame', 'fail_visibility'):
            with self.subTest(attribute=attribute):
                f = SwitchFixture(self.path)
                setattr(f, attribute, 'Native skeleton')
                self.assert_switch_failure(f)
                self.assertEqual(f.add_calls, 0)

    def test_every_partial_addition_rolls_back_without_releasing_old_nodes(self):
        for reference, count in [('pelvis', 16), ('com', 19)]:
            for index in range(1, count + 1):
                for phase in ('before', 'after'):
                    with self.subTest(reference=reference, index=index, phase=phase):
                        f = SwitchFixture(self.path, trajectory_reference=reference)
                        f.fail_add = (index, phase)
                        self.assert_switch_failure(f)

    def test_source_clock_and_frame_failure_do_not_change_active_members(self):
        for invalid in ('clock', 'frame'):
            with self.subTest(invalid=invalid):
                f = SwitchFixture(self.path)
                if invalid == 'clock':
                    f.sequence.state['pts'] += 1
                else:
                    f.sequence.state['P'] = np.zeros((1, 3))
                with self.assertRaises((AssertionError, IndexError)):
                    f.switch(f.player, 'new')
                f.assert_unchanged(self)
                self.assertEqual(f.add_calls, 0)

    def test_success_preserves_source_frame_camera_and_layer_visibility(self):
        for gear_points in (5, 48):
            with self.subTest(gear_points=gear_points):
                f = SwitchFixture(self.path, gear_points=gear_points)
                state = f.sequence.state
                state['com_world_mid'] = state['P'] * 4.
                f.switch(f.player, 'new')
                self.assertIs(f.player.sequence, f.sequence)
                self.assertIs(f.player.variant, f.definition)
                self.assertEqual(f.player.variant_nodes, f.created)
                self.assertEqual(f.scene.nodes, [f.static, *f.created])
                self.assertTrue(all(n.releases == 1 for n in f.old_nodes))
                self.assertEqual(f.static.releases, 0)
                self.assertTrue(all(n.releases == 0 and n.current_frame_id == 1 for n in f.created))
                for key in ('surface', 'skeleton', 'shoes', 'trajectory', 'pelvis'):
                    self.assertTrue(all(n.enabled == f.player.visible[key] for n in f.player.layers[key]))
                self.assertIs(f.player.layers['ice'], f.old_layers['ice'])
                np.testing.assert_array_equal(f.player.tracking_points, state['P'])
                np.testing.assert_array_equal(f.player.last_tracking_position, state['P'][1])
                np.testing.assert_array_equal(f.player.layers['trajectory'][0].points, state['G'])
                self.assertEqual(f.player.layers['com_projection'], [])
                self.assertEqual(f.player.layers['pelvis_trajectory'], [])
                self.assertIsNone(f.player.com_velocity)
                self.assertIs(f.scene.camera, f.camera)
                np.testing.assert_array_equal(f.camera.position, [4., 5., 6.])
                np.testing.assert_array_equal(f.camera.target, [1., 2., 3.])
                self.assertEqual(f.scene.current_frame_id, 1)
                self.assertEqual(f.player.frame, 1)
                self.assertEqual(f.player.error, '')
                f.clear_cache.assert_called_once_with()
                f.scene.current_frame_id = 2
                f.player.frame_changed(2)
                delta = state['P'][2] - state['P'][1]
                np.testing.assert_array_equal(f.camera.position, [4., 5., 6.] + delta)
                np.testing.assert_array_equal(f.camera.target, [1., 2., 3.] + delta)

    def test_com_switch_uses_saved_trajectory_for_camera_and_source_clock_velocity(self):
        f = SwitchFixture(self.path, trajectory_reference='com')
        player = f.player
        player.visible['com_projection'] = True
        state = f.sequence.state
        com = state['com_world_mid'].copy()
        f.switch(player, 'new')
        self.assertIs(player.sequence, f.sequence)
        self.assertEqual(player.frame, 1)
        np.testing.assert_array_equal(f.camera.position, [4., 5., 6.])
        np.testing.assert_array_equal(f.camera.target, [1., 2., 3.])
        np.testing.assert_array_equal(player.last_tracking_position, com[1])
        np.testing.assert_array_equal(player.layers['trajectory'][0].points, com)
        np.testing.assert_array_equal(player.layers['com_projection'][0].points, com * [1., 1., 0.])
        np.testing.assert_array_equal(player.layers['pelvis_trajectory'][0].points, state['G'])
        self.assertTrue(player.layers['trajectory'][0].enabled)
        self.assertTrue(player.layers['com_projection'][0].enabled)
        self.assertFalse(player.layers['pelvis_trajectory'][0].enabled)
        np.testing.assert_array_equal(state['com_world_mid'], com)

        t = f.seconds - f.seconds[0]
        velocity = np.column_stack([2 + 80*t, 3 - 30*t, -1 + 20*t])
        np.testing.assert_allclose(player.com_velocity, velocity, atol=1e-11, rtol=0.)
        player.set_view('side')
        np.testing.assert_array_equal(f.camera.target, com[1])
        offset = f.camera.position - f.camera.target
        player.speed = .25
        player.playback_fps = 12.5
        f.scene.current_frame_id = 2
        player.frame_changed(2)
        np.testing.assert_allclose(f.camera.target, com[2])
        np.testing.assert_allclose(f.camera.position, com[2] + offset)
        np.testing.assert_allclose(player.com_velocity, velocity, atol=1e-11, rtol=0.)

    def test_two_exposure_com_switch_uses_the_shared_secant(self):
        f = SwitchFixture(self.path, trajectory_reference='com')
        f.player.catalog.pts = f.pts[:2]
        f.player.catalog.seconds = f.seconds[:2]
        for key in ('pts', 'joints', 'gear', 'P', 'G', 'com_world_mid'):
            f.sequence.state[key] = f.sequence.state[key][:2]
        f.sequence.vertices = f.sequence.vertices[:2]
        f.switch(f.player, 'new')
        velocity = np.diff(f.sequence.state['com_world_mid'], axis=0) / (f.seconds[1] - f.seconds[0])
        np.testing.assert_allclose(f.player.com_velocity, np.repeat(velocity, 2, axis=0))

    def test_com_switch_to_variant_without_com_clears_tracking_and_velocity(self):
        f = SwitchFixture(self.path, trajectory_reference='com')
        player = f.player
        del f.sequence.state['com_world_mid']
        player.visible['pelvis_trajectory'] = True
        f.switch(player, 'new')
        self.assertIs(player.sequence, f.sequence)
        self.assertIsNone(player.tracking_points)
        self.assertIsNone(player.last_tracking_position)
        self.assertIsNone(player.com_velocity)
        self.assertEqual(player.layers['trajectory'], [])
        self.assertEqual(player.layers['com_projection'], [])
        self.assertTrue(player.layers['pelvis_trajectory'][0].enabled)
        np.testing.assert_array_equal(player.layers['pelvis_trajectory'][0].points, f.sequence.state['G'])
        f.scene.current_frame_id = 2
        player.frame_changed(2)
        np.testing.assert_array_equal(f.camera.position, [4., 5., 6.])
        np.testing.assert_array_equal(f.camera.target, [1., 2., 3.])

        gear = f.sequence.state['gear']
        gear[2] = [.3, -.2, .1]
        f.scene.current_frame_id = 1
        player.set_view('feet_side')
        foot_camera = f.camera.position.copy()
        f.scene.current_frame_id = 2
        player.frame_changed(2)
        np.testing.assert_allclose(f.camera.target, [.3, -.2, .1])
        np.testing.assert_allclose(f.camera.position, foot_camera + [.3, -.2, .1])

    def test_com_mode_feet_side_keeps_following_shoe_geometry_after_variant_switch(self):
        f = SwitchFixture(self.path, trajectory_reference='com')
        player = f.player
        player.set_view('feet_side')
        old_position, old_target = f.camera.position.copy(), f.camera.target.copy()
        shoes = np.broadcast_to(
            np.array([[3., 2., .1], [4., 5., .2], [4.2, 4.5, .3]])[:, None, None, :],
            (3, 2, 24, 3)).copy()
        f.sequence.state['skate_lower'] = shoes
        f.switch(player, 'new')
        np.testing.assert_array_equal(f.camera.position, old_position)
        np.testing.assert_array_equal(f.camera.target, old_target)
        f.scene.current_frame_id = 2
        player.frame_changed(2)
        delta = shoes[2].mean(axis=(0, 1)) - shoes[1].mean(axis=(0, 1))
        np.testing.assert_allclose(f.camera.position, old_position + delta)
        np.testing.assert_allclose(f.camera.target, old_target + delta)

        player.follow = False
        f.scene.current_frame_id = 1
        player.frame_changed(1)
        np.testing.assert_allclose(f.camera.position, old_position + delta)
        np.testing.assert_allclose(f.camera.target, old_target + delta)
        player.follow = True
        f.scene.current_frame_id = 2
        player.frame_changed(2)
        np.testing.assert_allclose(f.camera.target, old_target + 2*delta)

    def test_stale_saved_variant_falls_back_and_keeps_source_pts_and_camera(self):
        camera=dict(position=[4.,5.,6.],target=[1.,2.,3.],up=[0.,0.,1.],is_ortho=True,ortho_size=2.7)
        old=dict(variant='j1',source_pts=33,camera=camera,follow=False)
        catalog=SimpleNamespace(config=dict(default_variant='full1102',initial_source_pts=11,initial_wear_inspection=True,initial_view='feet'),
                                variants=[dict(id=i) for i in ('ref1102','obs1102','full1102','b5','j2')],
                                pts=np.array([11,22,33]))
        player_class = headless_player_class({})
        player = player_class.__new__(player_class)
        player.__dict__.update(scene=SimpleNamespace(current_frame_id=0, camera=SimpleNamespace(), add=Mock()),
                               _read_session=lambda: old, save_session=Mock(), set_view=Mock(),
                               visible=dict(wear=False), layers={}, apply_visibility=Mock())
        player.set_wear_inspection=switch_method({},name='set_wear_inspection').__get__(player)
        def load_variant(selected):
            player.variant=next(v for v in catalog.variants if v['id']==selected)
            player.sequence=SimpleNamespace(state=dict(P=np.arange(9.).reshape(3, 3)))
            player.tracking_points = player.sequence.state['P']
        player.load_variant=load_variant
        restore=restore_method(dict(np=np,FrameObserver=lambda _:object()))
        restore(player,catalog)
        self.assertEqual(player.trajectory_reference, 'pelvis')
        self.assertEqual(player.variant['id'],'full1102')
        self.assertEqual(player.scene.current_frame_id,2)
        np.testing.assert_array_equal(player.last_tracking_position, [6., 7., 8.])
        for attr in ('position','target','up'):
            np.testing.assert_array_equal(getattr(player.scene.camera,attr),camera[attr])
        self.assertTrue(player.scene.camera.is_ortho)
        self.assertEqual(player.scene.camera.ortho_size,2.7)
        self.assertFalse(player.follow)
        self.assertEqual(player.view_label,'上次视角')
        player.set_view.assert_called_once_with('free',fit_comparison=False)
        self.assertFalse(player.wear_inspection)
        self.assertFalse(player.visible['wear'])

    def test_fresh_config_can_start_in_wear_inspection_without_reading_session(self):
        catalog=SimpleNamespace(config=dict(default_variant='boot_fitted',initial_source_pts=22,
                                           initial_wear_inspection=True,initial_view='feet',trajectory_reference='com'),
                                variants=[dict(id='boot_fitted')],pts=np.array([11,22,33]))
        player_class = headless_player_class({})
        player = player_class.__new__(player_class)
        player.__dict__.update(scene=SimpleNamespace(current_frame_id=0, camera=SimpleNamespace(), add=Mock()),
                               _read_session=Mock(), save_session=Mock(), set_view=Mock(),
                               visible=dict(wear=False), layers={}, apply_visibility=Mock())
        player.set_wear_inspection=switch_method({},name='set_wear_inspection').__get__(player)
        def load_variant(selected):
            player.variant=next(v for v in catalog.variants if v['id']==selected)
            player.sequence=SimpleNamespace(state=dict(P=np.zeros((3,3)), com_world_mid=np.arange(9.).reshape(3,3)))
            player.tracking_points = player.sequence.state['com_world_mid']
        player.load_variant=load_variant
        restore=restore_method(dict(np=np,FrameObserver=lambda _:object()))
        restore(player,catalog,restore=False)
        self.assertEqual(player.trajectory_reference, 'com')
        np.testing.assert_array_equal(player.last_tracking_position, [3., 4., 5.])
        player._read_session.assert_not_called()
        player.set_view.assert_called_once_with('feet')
        self.assertTrue(player.wear_inspection)
        self.assertTrue(player.visible['wear'])
        self.assertEqual(player.boot_section,0)
        self.assertEqual(player.scene.current_frame_id,1)

    def test_saved_dynamic_surface_uses_fixed_faces_and_section_without_hull(self):
        f=SwitchFixture(self.path);f.definition['shell_topology']='saved';s=f.sequence.state
        s['boot_surface']=np.arange(3*2*14*3,dtype=float).reshape(3,2,14,3)
        s['boot_faces']=np.array([[0,1,2],[7,8,9]],np.int32)
        f.fail_hull=True;f.player.boot_section=1;f.player.wear_inspection=True
        before=s['boot_surface'].copy();f.switch(f.player,'new')
        meshes=[(args,kw) for args,kw in f.mesh_inputs if kw['name'].startswith('Boot candidate')]
        self.assertEqual(len(meshes),2)
        for foot,(args,kw) in enumerate(meshes):
            np.testing.assert_array_equal(args[0],before[:,foot])
            np.testing.assert_array_equal(args[1],[[7,8,9]])
            self.assertEqual(kw['color'][-1],.40)
        np.testing.assert_array_equal(s['boot_surface'],before)
        np.testing.assert_array_equal(f.camera.position,[4.,5.,6.])
        self.assertEqual(f.scene.current_frame_id,1)

    def test_dynamic_surface_cannot_be_retriangulated_as_legacy_hull(self):
        f=SwitchFixture(self.path);f.definition['shell_topology']='closed_hull'
        f.sequence.state['boot_surface']=np.zeros((3,2,14,3))
        with self.assertRaises(ValueError):f.switch(f.player,'new')
        f.assert_unchanged(self)

    def test_automatic_section_removes_the_side_facing_the_observing_camera(self):
        f=SwitchFixture(self.path);f.definition['shell_topology']='saved';s=f.sequence.state
        side=np.column_stack([np.arange(7),np.zeros(7),np.zeros(7)])
        points=np.r_[side,side+[0,1,0]]
        s['boot_surface']=np.broadcast_to(points,(3,2,14,3)).copy()
        s['boot_faces']=np.array([[0,1,2],[7,8,9]],np.int32)
        f.player.boot_section=3;f.switch(f.player,'new')
        meshes=[args for args,kw in f.mesh_inputs if kw['name'].startswith('Boot candidate')]
        # Existing camera y=5 sees the positive-width side (vertices 7..13).
        for args in meshes:np.testing.assert_array_equal(args[1],[[0,1,2]])
        np.testing.assert_array_equal(s['boot_surface'],np.broadcast_to(points,(3,2,14,3)))

    def test_wear_layer_constructor_failure_rolls_back_every_candidate_node(self):
        f=SwitchFixture(self.path);f.sequence.state['axes']=np.zeros((3,2,1,2,3))
        f.sequence.state['axis_labels']=np.array(['shank_x'])
        f.fail_constructor='Axis shank_x';self.assert_switch_failure(f)

    def test_clean_wear_mode_disables_proxies_and_can_enable_one_diagnostic(self):
        visible=dict(surface=True,shoes=True,skeleton=True,trajectory=True,pelvis=True,rink_lines=True,boards=True,ice=True,
                     com_projection=True,pelvis_trajectory=True,wear=False,footbed=False,cavity=False,axes=False,collar=False)
        player=SimpleNamespace(visible=visible.copy(),layers={},apply_visibility=Mock())
        mode=switch_method({},name='set_wear_inspection')
        mode(player,True)
        for key in ('surface','shoes','collar'):self.assertTrue(player.visible[key])
        for key in ('skeleton','trajectory','pelvis','com_projection','pelvis_trajectory',
                    'rink_lines','boards','ice','footbed','cavity','axes'):self.assertFalse(player.visible[key])
        mode(player,True,['footbed'])
        self.assertTrue(player.visible['footbed'])
        self.assertFalse(player.visible['cavity'])
        self.assertFalse(player.visible['axes'])
        mode(player,False)
        self.assertEqual(player.visible,visible)

    def versioned_fixture(self):
        f=SwitchFixture(self.path);f.definition['shell_topology']='saved';s=f.sequence.state
        s.update(template_version=np.array('fixture.v1'),
                 skate_lower=np.arange(3*2*24*3,dtype=float).reshape(3,2,24,3),
                 skate_upper=np.arange(3*2*20*3,dtype=float).reshape(3,2,20,3),
                 skate_lower_faces=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]),
                 skate_lower_face_parts=np.array(['runner','toe_pick','plate','pillar']),
                 skate_upper_faces=np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]),
                 skate_upper_region_ids=np.arange(4),skate_R=np.broadcast_to(np.eye(3),(3,2,3,3)).copy(),
                 skate_edge=np.zeros((3,2,2,8,3)),skate_witness=np.zeros((3,2,3)),
                 skate_reference=np.ones((3,2,3)),skate_cavity=np.ones((3,2,18,3)),
                 skate_cavity_faces=np.array([[0,1,2]]),skate_footbed=np.ones((3,2,8,3)))
        f.fail_hull=True
        f.player.visible.update(contact=True,skate_reference=True)
        return f

    def test_versioned_skate_axis_layer_reads_exact_saved_normal_without_gui(self):
        from rinkspace.viewer3d.blade_review import skate_axis_primitives
        f=self.versioned_fixture()
        f.sequence.state['skate_origin']=np.arange(18.).reshape(3,2,3)
        f.sequence.meta={'uniform_unit_reexpression_ratio':2.}
        normal=np.broadcast_to([0.,0.,1.],(3,2,3)).copy()
        f.player.blade_review=SimpleNamespace(model_array=Mock(return_value=normal))
        f.player.visible['blade_axes']=True
        f.namespace['skate_axis_primitives']=skate_axis_primitives
        before={key:value.copy() for key,value in f.sequence.state.items()}
        f.switch(f.player,'new')
        f.player.blade_review.model_array.assert_called_once_with('new','ice_normal_world',f.pts)
        arrows=f.player.layers['blade_axes']
        self.assertEqual(len(arrows),8)
        for foot in range(2):
            for label in ('F','L','U','iceN'):
                self.assertTrue(any(node.name=='Model skate %s foot %d; not source contact'%(label,foot) for node in arrows))
        self.assertTrue(all(node.enabled and node.current_frame_id==1 for node in arrows))
        for key,value in before.items():np.testing.assert_array_equal(f.sequence.state[key],value)

    def test_versioned_skate_renders_full_saved_parts_edges_and_witness_without_old_adapter_shell(self):
        f=self.versioned_fixture();before={key:value.copy() for key,value in f.sequence.state.items()}
        f.switch(f.player,'new')
        meshes=[(args,kw) for args,kw in f.mesh_inputs if kw['name'].startswith('Boot candidate')]
        for foot,(args,kw) in enumerate(meshes):
            np.testing.assert_array_equal(args[0],before['skate_upper'][:,foot])
            np.testing.assert_array_equal(args[1],before['skate_upper_faces'])
        lower=[(args,kw) for args,kw in f.mesh_inputs if kw['name'].startswith('Saved connected skate solids')]
        self.assertEqual(len(lower),2)
        for args,kw in lower:
            foot=int(kw['name'][-1])-1
            np.testing.assert_array_equal(args[0],before['skate_lower'][:,foot])
            np.testing.assert_array_equal(args[1],before['skate_lower_faces'])
            self.assertEqual(kw['face_colors'].shape,(4,4))
        names=[node.name for node in f.created]
        self.assertEqual(sum(name.startswith('Exact blade edge') for name in names),4)
        self.assertEqual(sum(name.startswith('Nonpenetration witness') for name in names),2)
        self.assertEqual(sum(name.startswith('Fixed blade material reference trajectory') for name in names),2)
        self.assertFalse(any(name.startswith('Blade lower edge strip') for name in names))
        for key,value in before.items():np.testing.assert_array_equal(f.sequence.state[key],value)

    def test_versioned_skate_section_uses_saved_lateral_regions_and_preserves_tongue(self):
        f=self.versioned_fixture();f.player.boot_section=3;s=f.sequence.state
        # Camera is on the positive local-y side. Keep the bridges and tongue.
        s['skate_upper'][:]=0.
        f.switch(f.player,'new')
        meshes=[args for args,kw in f.mesh_inputs if kw['name'].startswith('Boot candidate')]
        for args in meshes:np.testing.assert_array_equal(args[1],s['skate_upper_faces'][[0,2,3]])
        np.testing.assert_array_equal(s['skate_upper'],0.)

    def test_versioned_contact_constructor_failure_preserves_current_variant(self):
        f=self.versioned_fixture();f.fail_constructor='Nonpenetration witness 2'
        self.assert_switch_failure(f)

    def test_millimetre_chart_leaves_unknown_support_intervals_open(self):
        calls=[]
        draw=SimpleNamespace(**{name:(lambda *args,_name=name:calls.append((_name,args)))
                                for name in ('add_line','add_circle_filled','add_rect_filled')})
        gui=SimpleNamespace(text=lambda _:None,get_cursor_screen_pos=lambda:(10.,20.),
                            get_content_region_available=lambda:(400.,200.),get_window_draw_list=lambda:draw,
                            get_color_u32_rgba=lambda *args:1,dummy=lambda *args:None)
        method=switch_method(dict(np=np,imgui=gui),name='gui_height_plot')
        player=SimpleNamespace(frame=1,catalog=SimpleNamespace(seconds=np.array([10.,10.02,10.05,10.10])))
        method(player,'support',[(np.array([np.nan,1.,np.nan,2.]),(1.,1.,1.,1.))],band=(-2.,3.))
        self.assertFalse(any(name=='add_line' and len(args)==6 for name,args in calls))
        self.assertEqual(sum(name=='add_circle_filled' for name,_ in calls),2)
        for _,args in calls:self.assertTrue(np.isfinite(args).all())


if __name__ == '__main__':
    unittest.main()
