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


def restore_method(namespace, source=SOURCE):
    """Run the actual post-window initialization against a headless scene."""
    tree=ast.parse(Path(source).read_text(encoding='utf8'))
    player=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ResultsPlayer')
    method=next(n for n in player.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    begin=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='old' for t in n.targets))
    method.name='restore_session';method.body=method.body[begin:]
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
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
    def __init__(self, path, *, gear_points=48):
        self.error = InjectedFailure('injected renderer failure')
        self.fail_constructor = self.fail_frame = self.fail_visibility = None
        self.fail_add = None
        self.fail_hull = False
        self.created = []
        self.mesh_inputs = []
        self.add_calls = 0
        self.clear_cache = Mock()
        self.pts = np.array([11, 22, 33], np.int64)
        self.sequence = SimpleNamespace(
            vertices=np.zeros((3, 4, 3)), faces=np.array([[0, 1, 2]], np.int32),
            state=dict(pts=self.pts.copy(), parents=np.array([-1, 0, 1]),
                       joints=np.zeros((3, 3, 3)), gear=np.zeros((3, 2, gear_points, 3)),
                       P=np.arange(9.).reshape(3, 3), G=np.zeros((3, 3))))
        self.definition = dict(id='new')
        self.old_nodes = [RenderNode('old body'), RenderNode('old skeleton')]
        self.static = RenderNode('rink')
        self.old_layers = dict(surface=[self.old_nodes[0]], skeleton=[self.old_nodes[1]],
                               shoes=[], trajectory=[], pelvis=[], ice=[self.static])
        self.camera = SimpleNamespace(position=np.array([4., 5., 6.]), target=np.array([1., 2., 3.]))
        self.scene = SimpleNamespace(nodes=[self.static, *self.old_nodes], camera=self.camera,
                                     current_frame_id=1, add=self.add, remove=self.remove)
        self.player = SimpleNamespace(
            catalog=SimpleNamespace(variant=lambda _: self.definition, cache_path=lambda _: (path, {}), pts=self.pts),
            sequence=object(), variant=dict(id='old'), variant_nodes=self.old_nodes,
            layers=self.old_layers, last_p=np.array([7., 8., 9.]), scene=self.scene, frame=1,
            surface_alpha=.72, error='old error',
            visible=dict(surface=False, skeleton=True, shoes=False, trajectory=True, pelvis=False, ice=False))
        self.player.apply_visibility = lambda: [setattr(node, 'enabled', self.player.visible[key])
                                               for key, nodes in self.player.layers.items() for node in nodes]
        self.before = {key: getattr(self.player, key) for key in ('sequence', 'variant', 'variant_nodes', 'layers', 'last_p')}
        self.scene_before = list(self.scene.nodes)
        self.layers_before = {key: list(nodes) for key, nodes in self.old_layers.items()}
        self.namespace = dict(np=np, CachedSequence=lambda _: self.sequence,
                              Meshes=self.meshes, Skeletons=self.construct, Lines=self.construct, Spheres=self.construct,
                              shell_triangles=self.hull, ribbon_triangles=lambda _: np.array([[0, 1, 2]]))
        self.namespace['Meshes'] = Mock(side_effect=self.meshes)
        self.namespace['Meshes'].compute_vertex_and_face_normals.cache_clear = self.clear_cache
        self.switch = switch_method(self.namespace)

    def meshes(self, *args, **kwargs):
        self.mesh_inputs.append((args,kwargs))
        return self.construct(*args, **kwargs)

    def construct(self, *args, **kwargs):
        name = kwargs['name']
        if name == self.fail_constructor:
            raise self.error
        node = RenderNode(name, self)
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
        for index in range(1, 15):
            for phase in ('before', 'after'):
                with self.subTest(index=index, phase=phase):
                    f = SwitchFixture(self.path)
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
                np.testing.assert_array_equal(f.player.last_p, f.sequence.state['P'][1])
                self.assertIs(f.scene.camera, f.camera)
                np.testing.assert_array_equal(f.camera.position, [4., 5., 6.])
                np.testing.assert_array_equal(f.camera.target, [1., 2., 3.])
                self.assertEqual(f.scene.current_frame_id, 1)
                self.assertEqual(f.player.frame, 1)
                self.assertEqual(f.player.error, '')
                f.clear_cache.assert_called_once_with()

    def test_stale_saved_variant_falls_back_and_keeps_source_pts_and_camera(self):
        camera=dict(position=[4.,5.,6.],target=[1.,2.,3.],up=[0.,0.,1.],is_ortho=True,ortho_size=2.7)
        old=dict(variant='j1',source_pts=33,camera=camera,follow=False)
        catalog=SimpleNamespace(config=dict(default_variant='full1102',initial_source_pts=11,initial_wear_inspection=True,initial_view='feet'),
                                variants=[dict(id=i) for i in ('ref1102','obs1102','full1102','b5','j2')],
                                pts=np.array([11,22,33]))
        player=SimpleNamespace(scene=SimpleNamespace(current_frame_id=0,camera=SimpleNamespace(),add=Mock()),
                               frame=0,_read_session=lambda:old,save_session=Mock(),set_view=Mock(),visible=dict(wear=False),layers={},apply_visibility=Mock())
        player.set_wear_inspection=switch_method({},name='set_wear_inspection').__get__(player)
        def load_variant(selected):
            player.variant=next(v for v in catalog.variants if v['id']==selected)
            player.sequence=SimpleNamespace(state=dict(P=np.zeros((3,3))))
        player.load_variant=load_variant
        restore=restore_method(dict(np=np,FrameObserver=lambda _:object()))
        restore(player,catalog)
        self.assertEqual(player.variant['id'],'full1102')
        self.assertEqual(player.scene.current_frame_id,2)
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
        catalog=SimpleNamespace(config=dict(default_variant='boot_fitted',initial_source_pts=22,initial_wear_inspection=True,initial_view='feet'),
                                variants=[dict(id='boot_fitted')],pts=np.array([11,22,33]))
        player=SimpleNamespace(scene=SimpleNamespace(current_frame_id=0,camera=SimpleNamespace(),add=Mock()),
                               frame=0,_read_session=Mock(),save_session=Mock(),set_view=Mock(),visible=dict(wear=False),layers={},apply_visibility=Mock())
        player.set_wear_inspection=switch_method({},name='set_wear_inspection').__get__(player)
        def load_variant(selected):
            player.variant=next(v for v in catalog.variants if v['id']==selected)
            player.sequence=SimpleNamespace(state=dict(P=np.zeros((3,3))))
        player.load_variant=load_variant
        restore=restore_method(dict(np=np,FrameObserver=lambda _:object()))
        restore(player,catalog,restore=False)
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
                     wear=False,footbed=False,cavity=False,axes=False,collar=False)
        player=SimpleNamespace(visible=visible.copy(),layers={},apply_visibility=Mock())
        mode=switch_method({},name='set_wear_inspection')
        mode(player,True)
        for key in ('surface','shoes','collar'):self.assertTrue(player.visible[key])
        for key in ('skeleton','trajectory','pelvis','rink_lines','boards','ice','footbed','cavity','axes'):self.assertFalse(player.visible[key])
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
