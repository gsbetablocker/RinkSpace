"""Exact finite skin replays, with dense pose blends and all 55 weights."""
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.compact_body_derivatives import CompactBodyLinearization
from calibration.program_tracking.body_tracking.selected_surface_linearization import SelectedSurfaceLinearization,supports_surface
from rinkspace.viewer3d.prepare import FullSurface
from test_compact_body_performance import synthetic_model


def rotations(value):
    return Rotation.from_rotvec(value.reshape(-1,3)).as_matrix().reshape(*value.shape[:-1],3,3)


class SelectedSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.body=synthetic_model(Path(self.temp.name)/'synthetic.npz')
        self.rng=np.random.default_rng(431104)
        self.reference=rotations(self.rng.normal(0.,.8,(5,21,3)))
        self.delta=self.rng.normal(0.,.4,(5,21,3))
        self.delta[0]=0.;self.delta[1]*=1e-9;self.delta[2]*=4
        self.kinematics=CompactBodyLinearization(self.body,self.reference,self.delta,include_vertices=False)
        self.surface=FullSurface.__new__(FullSurface)
        self.surface.body=self.body
        self.surface.shaped=self.body.shaped.astype(np.float32)
        self.surface.posedirs=self.body.posedirs.astype(np.float32)
        self.surface.weights=self.body.W.astype(np.float32)
        self.surface._sample_double_cache=None
        self.ids=np.array([8,1,8,3,30,0,24,5])

    def context(self,*,chunk_size=2,ids=None):
        return SelectedSurfaceLinearization(self.surface,self.kinematics.native,self.kinematics.frames,
            self.ids if ids is None else ids,chunk_size=chunk_size)

    def compare_pair(self,context,column,step=1e-5,*,states=None,ids=None):
        ids=self.ids if ids is None else ids
        states=self.kinematics.kinematic_pair(column,step) if states is None else states
        actual=context.pair(column,*states)
        expected=tuple(self.surface.sample_vertices(s['native'],s['frames'],ids,double_precision=True) for s in states)
        for got,want in zip(actual,expected):np.testing.assert_allclose(got,want,atol=5e-13,rtol=5e-13)
        np.testing.assert_allclose((actual[0]-actual[1])/(2*step),(expected[0]-expected[1])/(2*step),atol=3e-9,rtol=2e-7)
        return actual,expected

    def test_all_63_exact_pairs_at_nonzero_chart_and_sparse_selection(self):
        context=self.context()
        expected=self.surface.sample_vertices(self.kinematics.native,self.kinematics.frames,self.ids,double_precision=True)
        np.testing.assert_allclose(context.samples,expected,atol=3e-14,rtol=3e-14)
        for column in range(6,69):
            with self.subTest(column=column):self.compare_pair(context,column)

    def test_nonnegligible_finite_steps_are_not_taylor_approximations(self):
        context=self.context()
        for column in (6,11,23,44,68):
            self.compare_pair(context,column,step=.13)

    def test_nonleg_pose_blend_survives_zero_subtree_skinning_weights(self):
        self.surface.weights[:]=0.;self.surface.weights[:,0]=1.
        context=self.context()
        pair,_=self.compare_pair(context,68)
        self.assertGreater(np.linalg.norm((pair[0]-pair[1])/2e-5),1e-4)

    def test_fixed_tail_joint_skinning_and_rotations_remain_live(self):
        self.surface.weights[:]=0.;self.surface.weights[:,54]=1.
        context=self.context()
        pair,_=self.compare_pair(context,9)
        self.assertGreater(np.linalg.norm((pair[0]-pair[1])/2e-5),.1)

    def test_localized_skin_weights_keep_all_pose_columns(self):
        self.surface.weights[:]=0.
        self.surface.weights[:16,[1,3,7,15]]=.25
        self.surface.weights[16:,[2,5,11,23]]=.25
        context=self.context()
        self.assertLess(len(context._skin_indices[1]),len(context._indices[1]))
        self.assertEqual(len(context._skin_indices[21]),0)
        for column in range(6,69):
            with self.subTest(column=column):self.compare_pair(context,column)

    def test_original_world_local_roundtrip_with_large_translation(self):
        q=rotations(self.rng.normal(0.,.5,(5,3)));p=self.rng.normal(0.,3.,(5,3))
        def roundtrip(state):
            state=dict(state)
            world=p[:,None]+np.einsum('nij,nkj->nki',q,state['native'])
            state['native']=np.einsum('nji,nkj->nki',q,world-p[:,None])
            return state
        baseline=roundtrip(dict(native=self.kinematics.native,frames=self.kinematics.frames))
        context=SelectedSurfaceLinearization(self.surface,baseline['native'],baseline['frames'],self.ids,chunk_size=3)
        for column in (6,24,27,65,68):
            pair=tuple(roundtrip(s) for s in self.kinematics.kinematic_pair(column))
            self.compare_pair(context,column,states=pair)

    def test_selection_order_duplicates_and_chunk_sizes(self):
        for ids in (np.arange(32),self.ids[::-1],np.array([3,3,3]),np.array([],dtype=int)):
            a=self.context(ids=ids,chunk_size=1);b=self.context(ids=ids,chunk_size=30)
            for column in (68,7,6,68):
                pair=self.kinematics.kinematic_pair(column)
                for x,y in zip(a.pair(column,*pair),b.pair(column,*pair)):
                    np.testing.assert_allclose(x,y,rtol=3e-14,atol=3e-14)
                self.compare_pair(a,column,ids=ids)

    def test_cache_reuse_and_concurrent_pairs_are_read_only(self):
        self.surface.sample_vertices(self.kinematics.native,self.kinematics.frames,self.ids,double_precision=True)
        cache=self.surface._sample_double_cache
        context=self.context();self.assertIs(self.surface._sample_double_cache,cache)
        self.assertIs(context._posedirs,cache[1][1]);self.assertFalse(context.samples.flags.writeable)
        columns=[68,6,17,68,49,7,6]
        def evaluate(column):return context.pair(column,*self.kinematics.kinematic_pair(column))
        expected=list(map(evaluate,columns))
        with ThreadPoolExecutor(max_workers=4) as workers:actual=list(workers.map(evaluate,columns))
        for got,want in zip(actual,expected):
            for x,y in zip(got,want):np.testing.assert_array_equal(x,y)
        self.assertIs(self.surface._sample_double_cache,cache)

    def test_capability_and_invalid_layouts_fail_closed(self):
        self.assertTrue(supports_surface(self.surface))
        class Custom(FullSurface):pass
        custom=Custom.__new__(Custom);custom.__dict__.update(self.surface.__dict__)
        self.assertFalse(supports_surface(custom))
        self.surface.sample_vertices=lambda *args,**kwargs:None
        self.assertFalse(supports_surface(self.surface))
        del self.surface.sample_vertices
        for ids in ([-1],[32],[.1]):
            with self.assertRaises(ValueError):self.context(ids=np.asarray(ids))
        with self.assertRaises(ValueError):self.context(chunk_size=0)
        context=self.context();pair=self.kinematics.kinematic_pair(6)
        for column in (0,69,True,6.5):
            with self.assertRaises(ValueError):context.pair(column,*pair)
        bad=dict(pair[0]);bad['native']=bad['native'][:-1]
        with self.assertRaises(ValueError):context.pair(6,bad,pair[1])

    def test_wrapped_instance_or_class_sampling_keeps_custom_semantics(self):
        original=FullSurface.sample_vertices
        @wraps(original)
        def changed(surface,*args,**kwargs):return original(surface,*args,**kwargs)+.1
        self.surface.sample_vertices=MethodType(changed,self.surface)
        self.assertFalse(supports_surface(self.surface))
        del self.surface.sample_vertices
        with patch.object(FullSurface,'sample_vertices',changed):
            self.assertFalse(supports_surface(self.surface))
        self.assertTrue(supports_surface(self.surface))


if __name__=='__main__':unittest.main()
