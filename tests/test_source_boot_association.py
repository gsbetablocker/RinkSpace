"""A visible boot's endpoints cannot be split between crossing physical feet."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import eye

from calibration.program_tracking.body_tracking import source_motion_refit as module


def test_latent_boot_association_keeps_heel_and_toe_together(monkeypatch):
    positions={('heel',0):[0.,0.],('toe',0):[20.,0.],
               ('heel',1):[4.,0.],('toe',1):[10.,0.]}
    def project(problem,x,rows,jacobian=False):
        return np.array([positions[r['feature'],r['foot']] for r in rows]),eye(2*len(rows),format='csr') if jacobian else None
    monkeypatch.setattr(module,'skate_projection',project)
    rows=[dict(id=name,role='fit',feature=name,foot=0,foot_candidates=[0,1],
               association_group='one_visible_boot',pts=256,uv=uv,normalization_px=1.,weight=1.)
          for name,uv in [('heel',[0.,0.]),('toe',[10.,0.])]]
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=rows))
    value=factor.evaluate(np.empty(0),True)
    np.testing.assert_array_equal(value['residual'],[4.,0.,0.,0.])
    np.testing.assert_array_equal(value['jac'].toarray(),np.eye(8)[[2,3,6,7]])
    assert [r['foot'] for r in value['metrics']['native_heel_toe_pixels']['inferred_foot_association']]==[1,1]
    factor.refresh_each_state=False
    positions['toe',0]=[10.,0.]
    np.testing.assert_array_equal(factor.evaluate(np.empty(0))['residual'],[4.,0.,0.,0.])
    factor.refresh_each_state=True
    np.testing.assert_array_equal(factor.evaluate(np.empty(0))['residual'],np.zeros(4))


def test_explicit_source_foot_does_not_reassociate(monkeypatch):
    seen=[]
    def project(problem,x,rows,jacobian=False):
        seen.extend(r['foot'] for r in rows)
        return np.array([[12.,5.]]),None
    monkeypatch.setattr(module,'skate_projection',project)
    row=dict(role='fit',feature='toe',foot=1,pts=256,uv=[10.,4.],normalization_px=2.,weight=9.)
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=[row]))
    value=factor.evaluate(np.empty(0))
    assert seen==[1]
    np.testing.assert_allclose(value['residual'],[3.,1.5])
    assert value['metrics']['native_heel_toe_pixels']['inferred_foot_association']==[]


def test_tracked_boot_cannot_switch_feet_at_one_overlapping_exposure(monkeypatch):
    def project(problem,x,rows,jacobian=False):
        return np.array([[0. if r['foot']==0 else 10.,0.] for r in rows]),None
    monkeypatch.setattr(module,'skate_projection',project)
    rows=[dict(id=str(i),role='fit',feature='toe',foot=0,foot_candidates=[0,1],
        association_group='boot' if i==0 else 'boot:flow:512',pts=256*(i+1),
        uv=uv,normalization_px=1.,weight=1.) for i,uv in enumerate(([0.,0.],[6.,0.]))]
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=rows))
    value=factor.evaluate(np.empty(0))
    assert [r['foot'] for r in value['metrics']['native_heel_toe_pixels']['inferred_foot_association']]==[0,0]


def test_two_observed_boot_tracks_cannot_both_select_same_physical_foot(monkeypatch):
    def project(problem,x,rows,jacobian=False):
        return np.array([[0. if r['foot']==0 else 10.,0.] for r in rows]),None
    monkeypatch.setattr(module,'skate_projection',project)
    rows=[dict(id=str(i),role='fit',feature='toe',foot=0,foot_candidates=[0,1],
        association_group=f'boot{i}',association_exclusion_group='pair',pts=256,
        uv=uv,normalization_px=1.,weight=1.) for i,uv in enumerate(([0.,0.],[4.,0.]))]
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=rows))
    value=factor.evaluate(np.empty(0))
    assert [r['foot'] for r in value['metrics']['native_heel_toe_pixels']['inferred_foot_association']]==[0,1]


def test_cropped_boot_region_does_not_invent_a_hidden_target(monkeypatch):
    def project(problem,x,rows,jacobian=False):
        return x.reshape(-1,2),eye(len(x),format='csr') if jacobian else None
    monkeypatch.setattr(module,'skate_projection',project)
    row=dict(role='fit',feature='cuff_center',foot=0,pts=256,uv=[960.,1600.],
             pixel_box=[[-10000.,1100.],[10000.,10000.]],normalization_px=20.,weight=4.)
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=[row]))
    # Only the observed empty image is excluded. No horizontal attraction and
    # no force toward the arbitrary centre of the allowed off-frame region.
    outside=factor.evaluate(np.array([400.,1060.]),True)
    np.testing.assert_allclose(outside['residual'],[0.,-4.])
    np.testing.assert_allclose(outside['jac'].toarray(),[[0.,0.],[0.,.1]])
    for position in ([400.,1200.],[1300.,2400.]):
        inside=factor.evaluate(np.array(position),True)
        np.testing.assert_array_equal(inside['residual'],[0.,0.])
        np.testing.assert_array_equal(inside['jac'].toarray(),np.zeros((2,2)))


def test_later_overlap_does_not_undo_an_earlier_two_boot_identity(monkeypatch):
    def project(problem,x,rows,jacobian=False):
        return np.array([[10.*r['foot'],0.] for r in rows]),None
    monkeypatch.setattr(module,'skate_projection',project)
    # B is one tracked boot across two frames. The later C observation favors
    # renaming B, but that would give A and B the same foot in the first frame.
    samples=[('A','first',256,0.,9.),('B','first',256,0.,1.),
             ('B','second',512,0.,1.),('C','second',512,10.,4.)]
    rows=[dict(id=f'{track}:{pts}',role='fit',feature='toe',foot=0,foot_candidates=[0,1],
        association_track=track,association_exclusion_group=pair,pts=pts,
        uv=[position,0.],normalization_px=1.,weight=weight)
        for track,pair,pts,position,weight in samples]
    factor=module.NativeShoePoints(SimpleNamespace(binding=SimpleNamespace(x=np.empty(0))),dict(samples=rows))
    associations=factor.evaluate(np.empty(0))['metrics']['native_heel_toe_pixels']['inferred_foot_association']
    assert [r['foot'] for r in associations]==[0,1,1,0]
