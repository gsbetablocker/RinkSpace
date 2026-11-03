"""Sampled real-triangle collar factors: topology, sign, cache and batching."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import skate_collar_factors as FACTORS
from calibration.program_tracking.body_tracking.skate_collar_review import _signed_gap_general


def prism(polygon,height=2.):
    polygon=np.asarray(polygon,float);n=len(polygon)
    vertices=np.r_[np.c_[polygon,np.zeros(n)],np.c_[polygon,np.full(n,height)]]
    faces=np.array([f for i in range(n) for f in ((i,(i+1)%n,n+i),((i+1)%n,n+(i+1)%n,n+i))])
    return vertices,faces


class Binding:
    def __init__(self,body=None,inner=None):
        body=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]) if body is None else np.asarray(body)
        inner=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]) if inner is None else np.asarray(inner)
        v,f=prism(body);self.skin=np.r_[v,v+[3.,0.,0.]]
        faces=np.r_[f,f+len(v)];weights=np.zeros((len(self.skin),55));weights[:len(v),7]=1;weights[len(v):,8]=1
        def sample(joints,frames,ids,**kwargs):return np.broadcast_to(self.skin[ids],(len(joints),len(ids),3)).copy()
        self.surface=SimpleNamespace(weights=weights,faces=faces,sample_vertices=sample)
        self.inner,self.faces=prism(inner,1.);self.region_ids=np.zeros(len(self.faces),int)
        self.collar=np.arange(len(inner),2*len(inner));self.ratio=1.;self.x=np.zeros(17)

    def packet(self):
        inner=self.inner.copy();inner[:,:2]*=np.exp(self.x[12])
        return dict(inner=inner,xs=np.arange(int(np.ceil(len(inner)/6))),collar=self.collar)

    def parameters(self):
        R=np.broadcast_to(np.eye(3),(2,3,3)).copy();t=self.x[:12].reshape(2,6)[:,:3].copy()
        return R,t,None

    def deformation(self,joints,frames):return np.zeros((len(joints),2,3))

    def deform(self,vertices,alpha):return np.broadcast_to(vertices,(*alpha.shape[:2],*vertices.shape)).copy()


def state(n):
    joints=np.zeros((n,22,3));joints[:,8,0]=3.
    frames=np.broadcast_to(np.eye(3),(n,22,3,3)).copy()
    return joints,frames


def test_real_triangle_sections_inside_and_outside_preserve_sign_and_input_arrays():
    b=Binding();spec=FACTORS.prepare(b);joints,frames=state(3)
    samples=b.surface.sample_vertices(joints,frames,spec.sample_ids)
    ids=np.flatnonzero(spec.sample_ids<len(b.skin)//2)
    samples[1,ids,0]+=1.2;samples[2,ids,2]+=3.
    before=samples.copy();result=FACTORS.evaluate(b,joints,frames,samples)
    np.testing.assert_allclose(result['breach_solver'][:2],[[0.,0.],[.7,0.]],atol=1e-12)
    assert not result['unknown'][:2].any()
    assert result['unknown'][2,0] and not result['unknown'][2,1]
    assert result['breach_solver'][2,0]==0 and result['unknown_reason'][2,0]=='body:no_triangle_intersection'
    np.testing.assert_array_equal(samples,before)
    np.testing.assert_array_equal(b.x,np.zeros(17))


def test_nonconvex_inner_preserves_notch_and_samples_actual_segment_midpoint():
    inner=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[.2,1.],[.2,0.],[-.2,0.],[-.2,1.],[-1.,1.]])
    body=np.array([[-.7,.7],[.7,.7],[0.,-.5]])
    b=Binding(body,inner);result=FACTORS.evaluate(b,*state(1))
    assert not result['unknown'].any()
    assert np.all(result['breach_solver']>.19) and np.all(result['breach_solver']<=.2)
    strict,lower,_,_=_signed_gap_general(body,inner,1e-10,1e-6)
    # The sampled surrogate may miss the strict segment minimum. It detects
    # the actual notch, and must not claim a stronger penetration than it.
    assert lower<=strict<=-result['breach_solver'][0,0]+1e-10


def test_batched_graph_rejects_multiple_closed_components_and_no_intersection():
    one,faces=prism([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    vertices=np.r_[one,one+[4.,0.,0.]]
    topology=FACTORS._triangle_spec(np.r_[faces,faces+len(one)])
    batch=np.broadcast_to(vertices,(3,*vertices.shape)).copy()
    batch[0,len(one):,2]+=3.;batch[2,:,2]+=3.
    points,mask,reason,count=FACTORS._section_samples(batch,topology,.999,1e-10)
    assert reason.tolist()==['','body:multiple_section_loops','body:no_triangle_intersection']
    assert count[0]>0 and count[1]>count[0] and count[2]==0
    assert points.shape[:2]==mask.shape


def test_open_body_and_excluded_tongue_wall_are_unknown_not_zero_gap_evidence():
    b=Binding();b.surface.faces=b.surface.faces[2:]
    result=FACTORS.evaluate(b,*state(1))
    assert result['unknown'][0,0] and result['breach_solver'][0,0]==0
    assert result['unknown_reason'][0,0]=='body:open_or_branched_section'
    assert not result['unknown'][0,1]
    b=Binding();b.region_ids[:2]=3
    result=FACTORS.evaluate(b,*state(1))
    assert result['unknown'].all() and np.all(result['breach_solver']==0.)
    assert np.all(result['unknown_reason']=='inner:open_section')


def test_cached_topology_is_immutable_and_shape_cache_isolated_across_worker_copy():
    b=Binding();first=FACTORS.prepare(b);worker=copy.copy(b);worker.x=b.x.copy()
    assert FACTORS.prepare(worker) is first
    worker.x[0]=.2
    assert FACTORS.prepare(worker) is first
    with pytest.raises(ValueError):first.sample_ids[0]=5
    with pytest.raises(ValueError):first.inner_vertices[0,0]=5
    worker.x[12]=.1;second=FACTORS.prepare(worker)
    assert second is not first and FACTORS.prepare(b) is first
    assert np.max(abs(second.inner_vertices[:,0]))>np.max(abs(first.inner_vertices[:,0]))
    j,f=state(1);samples=b.surface.sample_vertices(j,f,first.sample_ids)
    with pytest.raises(ValueError):FACTORS.evaluate(b,j,f,samples[:,:-1])


def test_vectorized_simple_loop_checks_do_not_close_a_bow_tie():
    points=np.array([[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]],
                     [[-1.,-1.],[1.,1.],[1.,-1.],[-1.,1.]]])
    np.testing.assert_array_equal(FACTORS._simple_loops(points,np.array([4,4]),1e-10),[True,False])


def test_batch_boundary_and_empty_batch_match_single_batch_values():
    b=Binding();joints,frames=state(129)
    values=FACTORS.evaluate(b,joints,frames)
    assert values['breach_solver'].shape==(129,2) and not values['unknown'].any()
    np.testing.assert_array_equal(values['breach_solver'],0.)
    empty=FACTORS.evaluate(b,*state(0))
    assert empty['breach_solver'].shape==(0,2) and empty['unknown'].shape==(0,2)


def test_translation_sweep_is_continuous_through_sampled_contact_and_batch_boundary():
    b=Binding();spec=FACTORS.prepare(b);joints,frames=state(257)
    samples=b.surface.sample_vertices(joints,frames,spec.sample_ids)
    shift=np.linspace(0.,2.,len(joints));foot0=spec.sample_ids<len(b.skin)//2
    samples[:,foot0,0]+=shift[:,None]
    result=FACTORS.evaluate(b,joints,frames,samples)
    assert not result['unknown'].any()
    expected=np.maximum(shift-.5,0.)
    np.testing.assert_allclose(result['breach_solver'][:,0],expected,atol=1e-12)
    assert np.all(np.diff(result['breach_solver'][:,0])<=np.diff(shift)+1e-12)


def test_vertex_on_plane_has_explicit_unknown_and_nearby_closed_sections_remain_valid():
    b=Binding();caps=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7]])
    b.surface.faces=np.r_[b.surface.faces,caps,caps+8]
    spec=FACTORS.prepare(b);joints,frames=state(3)
    samples=b.surface.sample_vertices(joints,frames,spec.sample_ids)
    vertex=int(np.flatnonzero(spec.sample_ids==0)[0])
    samples[:,vertex,2]=spec.height_solver+np.array([-1e-6,0.,1e-6])
    result=FACTORS.evaluate(b,joints,frames,samples)
    np.testing.assert_array_equal(result['unknown'][:,0],[False,True,False])
    assert result['unknown_reason'][1,0]=='body:vertex_on_section'
    assert result['breach_solver'][1,0]==0.
    np.testing.assert_allclose(result['breach_solver'][[0,2],0],0.)
