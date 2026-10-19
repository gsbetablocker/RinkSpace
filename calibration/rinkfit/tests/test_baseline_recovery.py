"""Targeted recovery tests: actual anchor arithmetic and unchanged baseline."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_joint.initialize_cameras import spline_seed,preserve_anchors,AnchorSeedError
from rinkfit_start.camera import UniformCubicBasis,CameraDomainError
from rinkfit_joint.baseline_recovery import restore_s01_into_joint_container,install_s01_supported_corrections,verify_zero_optimization_import

def test_actual_anchor_values_survive_without_surrogate_resampling():
    basis=UniformCubicBasis(-.8,.4,13);times=np.array([0.,.34,.8,1.7,2.46,3.2]);prior=np.column_stack((np.linspace(-.2,.3,13),np.linspace(.01,-.05,13),np.linspace(8.,9.,13)))
    target=np.asarray([basis.row(t)@prior for t in times]);target[:,0]+=np.array([0.,.002,-.001,.003,-.002,0.])
    controls,offset=spline_seed(times,target,dict(n_controls=13,dt=.4,first_control_time=-.8,duration=3.2),prior_total_controls=prior)
    total=np.asarray(controls['controls']);total[:,:2]+=offset
    assert np.max(np.abs(np.asarray([basis.row(t)@total for t in times])-target))<1e-12
    assert controls['initialization_receipt']['method']=='actual_anchor_basis_exact_transfer'
    assert controls['initialization_receipt']['source_pixel_accuracy_claimed'] is False


def test_duplicate_off_lattice_and_insufficient_halo_fail():
    plan=dict(n_controls=12,dt=.4,first_control_time=-.8,duration=3.2)
    with pytest.raises(AnchorSeedError,match='duplicate'):spline_seed([0.,0.],np.array([[0.,0.,9.],[.1,0.,9.]]),plan)
    with pytest.raises(AnchorSeedError,match='lattice'):preserve_anchors(np.array([256257]),256256,np.array([[0.,0.,9.]]),np.ones((12,3)),{k:plan[k] for k in ('n_controls','dt','first_control_time')},[-20,-20,4],[20,20,14],duration=3.2)
    with pytest.raises(CameraDomainError):spline_seed([0.,3.2],np.array([[0.,0.,9.],[.1,0.,9.]]),dict(plan,duration=3.21,n_controls=11))


def test_real_complete_baseline_import_and_affine_zero():
    root=Path('D:/OneDrive/codex_projs/RinkSpace');run=Path('D:/CodexData/RinkSpace_Data/runs/okada_joint9_1018a')
    bp=root/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/audit_model.json';cp=run/'fit_full_source_v98/generation_1/model.json'
    if not bp.exists() or not cp.exists():pytest.skip('real immutable recovery fixtures absent')
    base=json.loads(bp.read_text(encoding='utf8'));current=json.loads(cp.read_text(encoding='utf8'))
    q=restore_s01_into_joint_container(current,base,baseline_sha256='d78b7012c80664131f88134be0f9f868c1d6742ad7516155fc131b72ca3110e9',baseline_ref=str(bp))
    q=install_s01_supported_corrections(q,np.arange(256256,575489,256,dtype=np.int64))
    report=verify_zero_optimization_import(base,q)
    assert report['native_exposures']==1248
    assert report['maximum_spline_component_change']==[0.,0.,0.]
    assert report['maximum_material_world_change']==0.
    assert len(q['shots'])==9 and len(q['rigs'])==3
    for g in q['linear_control_manifolds']:
        B=np.array(g['basis']);assert np.max(np.abs(B.T@B-np.eye(B.shape[1])))<1e-12
        if g['constraint_matrix']:
            C=np.array(g['constraint_matrix']);assert np.max(np.abs(C@B))<1e-10
            assert np.max(np.abs(C@np.array(q['blocks'][g['block']]['values'])))<2e-10
