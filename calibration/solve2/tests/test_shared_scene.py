"""Geometric invariants of the shared scene, independent of camera fitting."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rinksolve import schema as S
from rinksolve.model import (StructuredIceFamily, ParameterState, ModelBundle,
    SceneModel, CurveModel, InstallationModel, block, save_model_bundle, load_model_bundle,
    shared_advertisement_candidate)


def fixture():
    values = {
        'board': [.5, .1, .12, 2.1, .02],
        'curling': [-.31, .25, -.18, .12, .03, .12, .33, .66],
        'lane': [.37, .08, -.39, -.38, -.4, -.37, .4, .41, .42, .43],
        'hockey': [.02, .12, .075, -.24, .24],
        'ends': [-.43, .42, .1, .11, .12, .13],
        'marks': [.03, .031, .05, .07],
    }
    specs=tuple(block(k,S.OwnerKind.SCENE,'scene',[str(i) for i in range(len(v))],S.Unit.SCENE_UNIT,'L1') for k,v in values.items())
    state=ParameterState('test',0,'L1',specs,tuple(S.ParameterValue(k,tuple(v)) for k,v in values.items()))
    family=StructuredIceFamily('test.v2','curling','lane','hockey',3,'ends',
        layout_version=2,board_block_id='board',hockey_end_block_id='marks')
    return family,state


class SharedSceneTests(unittest.TestCase):
    def check_relations(self,family,state):
        curves=family.curves(state)
        radii=[]
        for row in range(1,5):
            f=np.array(curves[f'ice.F{row}:outer']['center_xy'])
            r=np.array(curves[f'ice.R{row}:outer']['center_xy'])
            upper=np.array(curves[f'ice.lane{row}:edge0']['endpoints_xy']).reshape(2,2)
            lower=np.array(curves[f'ice.lane{row}:edge2']['endpoints_xy']).reshape(2,2)
            self.assertAlmostEqual(f[1],r[1],places=15)
            self.assertAlmostEqual(f[1],(upper[0,1]+lower[0,1])/2,places=15)
            self.assertNotIn(f'ice.lane{row}:edge3',curves)
            for prefix in ['F','R']:
                radii.append(curves[f'ice.{prefix}{row}:outer']['radius'])
        self.assertEqual(len(set(radii)),1)
        np.testing.assert_array_equal(curves['ice.C0:outer']['center_xy'],[0.,0.])
        for name in ['A1','A2','A3','A4']:
            self.assertEqual(curves[f'ice.{name}:outer']['radius'],curves['ice.C0:outer']['radius'])
        blue1=np.array(curves['ice.B1:segment0']['endpoints_xy']).reshape(2,2)
        blue2=np.array(curves['ice.B2:segment0']['endpoints_xy']).reshape(2,2)
        center=np.array(curves['ice.C:segment0']['endpoints_xy']).reshape(2,2)
        np.testing.assert_allclose((blue1+blue2)/2,center,atol=1e-15)
        W,rx,ry,p,_=state.get('board')
        for side in ['left','right']:
            for edge in ['upper','lower']:
                q=np.array(curves[f'ice.goal_trapezoid_{side}_{edge}_outer:segment0']['endpoints_xy'])[-2:]
                value=max(0.,(abs(q[0])-(.5-rx))/rx)**p+max(0.,(abs(q[1])-(W/2-ry))/ry)**p
                self.assertAlmostEqual(value,1.,places=12)

    def test_relations_hold_under_parameter_changes(self):
        family,state=fixture()
        rng=np.random.default_rng(34)
        for _ in range(20):
            curling=state.get('curling').copy();curling[2:5]=[rng.uniform(-.19,-.17),rng.uniform(.10,.125),rng.uniform(.025,.04)]
            lane=state.get('lane').copy();lane[1]=rng.uniform(.055,.095)
            hockey=state.get('hockey').copy();hockey[0]=rng.uniform(-.06,.06)
            self.check_relations(family,state.updated({'curling':curling,'lane':lane,'hockey':hockey}))

    def test_no_old_extra_degrees_of_freedom_in_v2(self):
        family,state=fixture()
        class BadState:
            def get(self,key):
                return np.r_[state.get(key),0.,0.,0.,0.] if key=='lane' else state.get(key)
        with self.assertRaises(ValueError):family.curves(BadState())
        with self.assertRaises(ValueError):replace(family,layout_version=99).curves(state)

    def test_legacy_v1_retains_its_saved_geometry(self):
        family,state=fixture()
        class Legacy:
            def get(self,key):
                if key=='lane':return np.r_[state.get(key)[:2],[-.17,-.05,.07,.19],state.get(key)[2:]]
                if key=='hockey':return np.r_[.01,.02,state.get(key)]
                return state.get(key)
        old=replace(family,layout_version=1,hockey_end_block_id=None)
        curves=old.curves(Legacy())
        np.testing.assert_array_equal(curves['ice.C0:outer']['center_xy'],[.01,.02])
        self.assertAlmostEqual(curves['ice.lane1:edge0']['endpoints_xy'][1],-.13)

    def test_scene_round_trip_and_board_seam(self):
        family,state=fixture()
        definitions=family.curves(state)
        curves=tuple(CurveModel(k,k,k,d['kind'],None,d) for k,d in definitions.items())
        scene=SceneModel('scene','L1','board',curves,structural_family=family)
        model=ModelBundle('test',0,scene,(),(),state)
        with tempfile.TemporaryDirectory() as directory:
            save_model_bundle(model,directory)
            other=load_model_bundle(directory)
            self.assertEqual(other.state.digest(),state.digest())
            for curve in curves:
                np.testing.assert_array_equal(scene.world_path(curve.id,[0.,.5,1.],state),other.scene.world_path(curve.id,[0.,.5,1.],other.state))
        board=scene.board(state)
        for seam in board.breaks:
            np.testing.assert_allclose(board.xyz([seam-1e-9],[0.]),board.xyz([seam+1e-9],[0.]),atol=2.1e-9)
            np.testing.assert_array_equal(board.xyz([seam],[0.])[0,:2],board.xyz([seam],[board.height])[0,:2])

    def test_paired_artwork_shares_shape_and_size_but_not_placement(self):
        family,state=fixture()
        paths=[]
        for n,s,w in [(2,.1,.12),(15,.8,.09),(12,1.8,.03)]:
            q=np.array([[s,.003],[s+w,.003],[s+w,.014],[s,.014],[s,.003]])
            paths.append(CurveModel(f'd78_Ad{n}' if n!=15 else 'demo_Ad15',f'Ad{n}.print.outline',f'Ad{n}',
                'board_material_polyline',None,{'kind':'board_material_polyline','unwrapped_station_height':q}))
        model=ModelBundle('ads',0,SceneModel('scene','L1','board',tuple(paths)),(),(),state)
        facts={'constraint':{'paired_advertisement_numbers':'x and x+13','same_printed_template':True,'shared_printed_dimensions':True}}
        candidate,report=shared_advertisement_candidate(model,facts)
        a=next(i for i in candidate.installations if i.id=='installation/Ad2')
        b=next(i for i in candidate.installations if i.id=='installation/Ad15')
        self.assertEqual(a.size_block_id,b.size_block_id)
        self.assertNotEqual(a.placement_block_id,b.placement_block_id)
        self.assertIn(25,report['missing_placement'])
        self.assertFalse(any(c.label.startswith('Ad25.') for c in candidate.scene.curves))
        source=next(c for c in candidate.scene.curves if c.label=='Ad2.print.outline')
        target=next(c for c in candidate.scene.curves if c.label=='Ad15.print.outline')
        self.assertEqual(target.definition['template_curve_id'],source.id)
        for scale in [1.,1.2]:
            updated=candidate.state.updated({a.size_block_id:candidate.state.get(a.size_block_id)*scale})
            x=candidate.scene.unwrapped_path(source.id,[0.,.25,.5,.75,1.],updated)
            y=candidate.scene.unwrapped_path(target.id,[0.,.25,.5,.75,1.],updated)
            np.testing.assert_allclose(x-x[0],y-y[0],atol=1e-15)
        with tempfile.TemporaryDirectory() as directory:
            save_model_bundle(candidate,directory);loaded=load_model_bundle(directory)
            np.testing.assert_array_equal(candidate.scene.world_path(target.id,[0.,.5,1.],candidate.state),loaded.scene.world_path(target.id,[0.,.5,1.],loaded.state))


if __name__=='__main__':unittest.main()
