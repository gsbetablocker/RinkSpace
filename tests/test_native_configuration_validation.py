"""Python validation with recording ABI doubles; no native or CUDA execution."""
import ctypes as c
import unittest
import numpy as np
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.core.native import D, I, N

class Fn:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*a):return self.fn(*a)

class NativeRecorder:
    """Records ABI traffic; reproduces only the relevant validation/state gates."""
    def __init__(self):
        self.calls=[];self.profile_ids=None;self.scene=None;self.charts=None
        self.rs_gpu_attach_profiles=Fn(self.profiles)
        self.rs_gpu_set_profile_mode=Fn(self.mode)
        self.rs_gpu_attach_scene=Fn(self.attach_scene)
        self.rs_gpu_set_scene_chart=Fn(self.set_charts)
    def profiles(self,handle,ids,ni,data,nd):
        self.calls.append('attach_profiles')
        ids=np.ctypeslib.as_array(ids,shape=(ni,)).copy()
        data=np.ctypeslib.as_array(data,shape=(nd,)).reshape(-1,17).copy()
        if self.profile_ids is not None:return 1
        if np.any(ids< -1) or np.any(ids>=len(data)):return 1
        seen=np.bincount(ids[ids>=0],minlength=len(data))
        if (np.any(seen==0) or not np.isfinite(data).all() or
            np.any(data[:,12]>=data[:,13]) or np.any(data[:,14]>=data[:,15]) or
            np.any(data[:,16]<data[:,12]) or np.any(data[:,16]>data[:,13])):return 1
        self.profile_ids=ids;return 0
    def mode(self,*a):self.calls.append('set_profile_mode');return 0
    def attach_scene(self,handle,values,nv,records,nr,data,nd,q,directions,ns,lo,hi):
        self.calls.append('attach_scene')
        self.scene=np.ctypeslib.as_array(values,shape=(nv,)).copy()
        self.records=np.ctypeslib.as_array(records,shape=(nr,)).copy()
        return 0
    def set_charts(self,handle,charts,n):
        self.calls.append('set_scene_chart')
        a=np.ctypeslib.as_array(charts,shape=(n,)).copy();v=self.scene
        if np.any(a<0) or np.any(a>5) or np.any(v[a==1]<=0) or np.any(v[a==2]<=1):return 1
        for i in np.flatnonzero(a>=3):
            start=int(i)-int(a[i]-3)
            if start<0 or start+2>=n or not np.array_equal(a[start:start+3],[3,4,5]) or not 0<v[start]<v[start+1]<v[start+2]<1:return 1
        self.charts=a;return 0

class Problem(ResidentResidualProblem):
    """Use the real Python wrapper with a recording library, without a GPU handle."""
    def __init__(self,n=2):
        self._lib=NativeRecorder();self._handle=c.c_void_p(1);self.count=n

    def _check(self,status):
        if status:raise RuntimeError('native recorder rejected input')


def profile_kwargs(ids=(0,0),**changes):
    k=dict(row_ids=np.asarray(ids),numerator_u=[[0,1,0,0]],numerator_v=[[0,0,0,0]],denominator=[[1,0,0,0]],alpha_bounds=[[0,1]],station_bounds=[[0,1]],initial_alpha=[.5])
    k.update(changes);return k

def scene_kwargs(**changes):
    k=dict(values=[.2,.4,.6],records=np.zeros((1,12),np.int32),data=[],directions=np.eye(3),charts=np.zeros(3,np.int32))
    k.update(changes);return k

class ProfileTests(unittest.TestCase):
    def assert_rejected_without_call(self,**k):
        p=Problem()
        with self.assertRaises((ValueError,AttributeError)):p.configure_profiles(**profile_kwargs(**k))
        self.assertEqual(p._lib.calls,[])
        self.assertIsNone(p._lib.profile_ids)
    def test_large_id_not_wrapped_to_zero(self):self.assert_rejected_without_call(ids=np.array([2**32,0],np.uint64))
    def test_large_id_not_wrapped_to_no_profile(self):self.assert_rejected_without_call(ids=np.array([2**32-1,0],np.uint64))
    def test_negative_large_id_not_wrapped(self):self.assert_rejected_without_call(ids=np.array([-2**32,0],np.int64))
    def test_invalid_mode_has_no_side_effect(self):self.assert_rejected_without_call(mode='typo')
    def test_out_of_table_rejected_before_native(self):self.assert_rejected_without_call(ids=np.array([1,0],np.int32))
    def test_missing_mode_capability_has_no_side_effect(self):
        p=Problem();del p._lib.rs_gpu_set_profile_mode
        with self.assertRaises(AttributeError):p.configure_profiles(**profile_kwargs())
        self.assertEqual(p._lib.calls,[])
    def test_fractional_ids_rejected(self):self.assert_rejected_without_call(ids=[.1,0.])
    def test_valid_profiles_unchanged(self):
        p=Problem();p.configure_profiles(**profile_kwargs(ids=[-1,0]));self.assertEqual(p._lib.profile_ids.tolist(),[-1,0]);self.assertEqual(p.profile_mode,'exact')
    def test_restoration_mode_preserved(self):
        p=Problem();p.configure_profiles(**profile_kwargs(mode='feasibility_restoration'));self.assertEqual(p.profile_mode,'feasibility_restoration')
    def test_can_retry_after_bad_mode(self):
        p=Problem()
        with self.assertRaises(ValueError):p.configure_profiles(**profile_kwargs(mode='typo'))
        p.configure_profiles(**profile_kwargs());self.assertEqual(p.profile_count,1)

class SceneTests(unittest.TestCase):
    def assert_rejected_without_call(self,**k):
        p=Problem()
        with self.assertRaises((ValueError,AttributeError)):p.configure_scene_parameters(**scene_kwargs(**k))
        self.assertEqual(p._lib.calls,[]);self.assertIsNone(p._lib.scene)
    def test_chart_wrong_length_no_attach(self):self.assert_rejected_without_call(charts=[0,0])
    def test_chart_out_of_range_no_attach(self):self.assert_rejected_without_call(charts=[0,6,0])
    def test_fractional_chart_not_truncated(self):self.assert_rejected_without_call(charts=[0,.5,0])
    def test_large_chart_not_wrapped(self):self.assert_rejected_without_call(charts=np.array([0,2**32,0],np.uint64))
    def test_negative_log_domain_no_attach(self):self.assert_rejected_without_call(values=[-.2,.4,.6],charts=[1,0,0])
    def test_greater_than_one_domain_no_attach(self):self.assert_rejected_without_call(charts=[2,0,0])
    def test_bad_ordered_triple_no_attach(self):self.assert_rejected_without_call(charts=[3,4,0])
    def test_bad_ordered_values_no_attach(self):self.assert_rejected_without_call(values=[.2,.6,.4],charts=[3,4,5])
    def test_fractional_records_not_truncated(self):self.assert_rejected_without_call(records=np.full((1,12),.1))
    def test_large_records_not_wrapped(self):self.assert_rejected_without_call(records=np.full((1,12),2**32,dtype=np.uint64))
    def test_integral_float_records_preserved(self):
        p=Problem();p.configure_scene_parameters(**scene_kwargs(records=np.zeros((1,12),np.float64)));self.assertEqual(p._lib.records.tolist(),[0]*12)
    def test_integral_float_charts_preserved(self):
        p=Problem();p.configure_scene_parameters(**scene_kwargs(charts=[3.,4.,5.]));self.assertEqual(p._lib.charts.tolist(),[3,4,5])
    def test_valid_ordered_chart_unchanged(self):
        p=Problem();r=p.configure_scene_parameters(**scene_kwargs(charts=[3,4,5]));self.assertEqual(p._lib.charts.tolist(),[3,4,5]);self.assertEqual(r,{'scene_parameters':3,'paths':1})
    def test_valid_log_charts_unchanged(self):
        p=Problem();p.configure_scene_parameters(**scene_kwargs(values=[2.,2.,.6],charts=[1,2,0]));self.assertEqual(p._lib.charts.tolist(),[1,2,0])
    def test_none_charts_unchanged(self):
        p=Problem();p.configure_scene_parameters(**scene_kwargs(charts=None));self.assertEqual(p._lib.calls,['attach_scene'])
    def test_missing_chart_capability_has_no_side_effect(self):
        p=Problem();del p._lib.rs_gpu_set_scene_chart
        with self.assertRaises(AttributeError):p.configure_scene_parameters(**scene_kwargs())
        self.assertEqual(p._lib.calls,[])
