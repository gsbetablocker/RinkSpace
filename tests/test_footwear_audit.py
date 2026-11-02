"""Execute repository classes with synthetic body/camera providers, no inference or fit.

The source files under --repo are executed, not copied implementations. Heavy
model providers are replaced at their import boundary; NumPy/SciPy/OpenCV,
residual assembly, fast-path selection and transfer preflight are real.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix

REPO = Path(os.environ.get('RINKSPACE_AUDIT_REPO', Path(__file__).resolve().parents[1]))
PREFIX = Path('calibration/program_tracking/body_tracking')
INITIAL = np.zeros(14)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name:z[name].copy() for name in z.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute_source(name, injected):
    path = REPO/PREFIX/name
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    # Replace only project-relative imports. All function/class bodies remain
    # byte-for-byte source-derived; scientific library imports are not stubbed.
    tree.body = [node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.level)]
    scope = {'__name__':'audit_test_module','__file__':str(path),**injected}
    exec(compile(tree,str(path),'exec'),scope)
    return scope


class FakeBody:
    def forward(self, pose, delta, **kwargs):
        n = len(pose)
        coco = np.zeros((n,17,3));coco[:,:,2] = 1.
        return coco, np.zeros((n,22,3)), None, np.broadcast_to(np.eye(3),(n,22,3,3)).copy()


class FrozenBodyProvider:
    def __init__(self, root, mode, exposure_nodes):
        self.exposure_nodes=exposure_nodes;self.body=FakeBody();self.projection=object()
    def kinematics(self, x, frames, all_nodes):
        n=len(frames)*self.exposure_nodes
        return {'P':np.zeros((n,3)), 'Q':np.broadcast_to(np.eye(3),(n,3,3)).copy(),
                'inputs':{'pose':np.zeros((n,21,3))}, 'delta':np.zeros((n,1))}


def gear(P,Q,joints,rotations,x):
    n=len(P);theta=np.linspace(0,2*np.pi,14,endpoint=False)
    shell=np.c_[4+np.cos(theta),4+np.sin(theta),np.ones(14)]
    blade=np.c_[np.linspace(0,8,34),np.full(34,10.),np.ones(34)]
    one=np.r_[shell,blade]
    result=np.broadcast_to(one,(n,2,48,3)).copy()
    return result, {'shell':np.arange(14),'blade_left':np.arange(14,31)}, rotations, np.zeros((n,2,3))


def project(provider,world,frames):
    middle=world[:,world.shape[1]//2]
    return {'uv':middle[:,:,:2].copy(),'world':middle.copy(),'row_iterations':1}


def basis(times,spacing):
    knots=np.r_[np.repeat(times[0],4),np.repeat(times[-1],4)]
    return BSpline.design_matrix(times,knots,3).tocsr(),knots


@pytest.fixture
def module():
    return execute_source('footwear_fit.py',dict(
        load=load_npz,digest=digest,ExposureArticulatedSplineProblem=FrozenBodyProvider,
        moving_projection=project,_basis=basis,
        _second_derivative_basis=lambda t,k:csr_matrix((len(t),len(k)-4)),
        NAMES=[str(i) for i in range(14)],INITIAL=INITIAL.copy(),LOWER=np.full(14,-10.),UPPER=np.full(14,10.),
        LEG_DOFS=['synthetic'],decode=lambda x:x.tolist(),extra_rotations=lambda x:{},
        equipment_world=gear,lowest_blade_world=lambda R,O,x:O.copy()))


def fixture_tree(tmp_path, observed_kind='boot_external_contour', reverse=False,
                 missing=np.nan, qualified=False):
    root=tmp_path/'body';root.mkdir();(root/'pose').mkdir();out=root/'footwear_v2';out.mkdir()
    uv=np.zeros((3,17,2));mask=np.ones((3,17),bool)
    uv[0,13]=missing;mask[0,13]=qualified
    np.savez(root/'input.npz',pts=np.array([256,512,768]),fit_time_seconds=np.array([0.,.5,1.]),
             rig=np.zeros(38),source_uv=uv,source_qualified=mask)
    np.savez(root/'pose/result.npz',x=np.zeros(14),P_world_m=np.zeros((3,3)),Q_world=np.broadcast_to(np.eye(3),(3,3,3)))
    (root/'input.json').write_text('{}')
    review=root/'review.json';review.write_text('{}')
    masks=np.zeros((2,16,16),np.uint8);masks[0,2:12,2:9]=1;masks[1,2:12,12:15]=1
    maskfile=out/'masks.npz';np.savez(maskfile,masks=masks)
    records=[dict(pts=256,foot=0,status='admitted_visible_boundary',mask_file=str(maskfile),
                  observed_kind=observed_kind,reverse_boundary=reverse,partial=True,sigma_px=2.5,
                  source_boundary_native_uv=[[4.,10.],[3.,10.],[5.,10.]])]
    (out/'admission.json').write_text(json.dumps({'records':records}))
    (out/'source_index.json').write_text(json.dumps({'rows':[{'pts':256,'crop_xyxy':[0,0,16,16]}]}))
    return root,review,out


@pytest.mark.parametrize('missing',[np.nan,np.inf,-np.inf])
def test_unqualified_nonfinite_pixel_does_not_poison_residual(module,tmp_path,missing):
    root,review,out=fixture_tree(tmp_path,missing=missing)
    p=module['FootwearProblem'](root,review,'shared')
    source_before=p.p['source_uv'].copy();mask_before=p.p['source_qualified'].copy()
    result=p.evaluate(p.x0)
    assert np.isfinite(result['residual']).all()
    np.testing.assert_array_equal(result['residual'][3:5],np.zeros(2))
    np.testing.assert_array_equal(p.p['source_uv'],source_before)
    np.testing.assert_array_equal(p.p['source_qualified'],mask_before)
    assert result['pattern'] is None
    assert p.sparsity.shape==(len(result['residual']),p.size)


def test_qualified_invalid_pixel_not_silently_removed(module,tmp_path):
    root,review,out=fixture_tree(tmp_path,missing=np.nan,qualified=True)
    p=module['FootwearProblem'](root,review,'shared')
    assert not np.isfinite(p.evaluate(p.x0)['residual']).all()


def test_finite_knee_scaling_and_row_layout_unchanged(module,tmp_path):
    root,review,out=fixture_tree(tmp_path,missing=24.,qualified=True)
    p=module['FootwearProblem'](root,review,'shared')
    result=p.evaluate(p.x0)
    np.testing.assert_array_equal(result['residual'][3:7],[-2.,-2.,0.,0.])
    assert len(result['residual'])==3+4+14


@pytest.mark.parametrize('kind',['boot_external_contour','full_equipment_external_contour'])
@pytest.mark.parametrize('nodes',[3,5])
def test_fast_projection_preserves_requested_contour_objective(module,tmp_path,kind,nodes):
    root,review,out=fixture_tree(tmp_path,observed_kind=kind,missing=0.,qualified=True)
    p=module['FootwearProblem'](root,review,'shared',exposure_nodes=nodes)
    assert p.fast_projection
    fast=p.state(p.x0,fast=True);full=p.state(p.x0,fast=False)
    used=fast['parts']['shell'] if kind=='boot_external_contour' else np.arange(48)
    np.testing.assert_allclose(fast['uv'][0,0,used],full['uv'][0,0,used],rtol=0,atol=0)
    fast_res=p.evaluate(p.x0)['residual'].copy()
    p.cache_x=None;p.fast_projection=False
    np.testing.assert_allclose(p.evaluate(p.x0)['residual'],fast_res,rtol=0,atol=0)


def test_fast_boot_only_does_not_eagerly_project_unrequested_blades(module,tmp_path):
    root,review,out=fixture_tree(tmp_path,missing=0.,qualified=True)
    p=module['FootwearProblem'](root,review,'shared')
    result=p.state(p.x0,fast=True)
    assert np.isnan(result['uv'][0,0,14:]).all()


class ReachedOptimizer(Exception):
    pass


@pytest.mark.parametrize('defect',['knots','missing_knots','size','nonfinite','matrix','valid'])
def test_transfer_validates_exact_saved_spline_before_output(tmp_path,defect):
    root,review,out=fixture_tree(tmp_path,missing=0.,qualified=True)
    knots=np.r_[np.zeros(4),np.ones(4)];size=18
    p=SimpleNamespace(root=root,knots=knots,size=size,exposure_nodes=3,
        sparsity=csr_matrix((4,size)),bounds=(np.full(size,-10.),np.full(size,10.)))
    x=np.zeros(size);saved=knots.copy()
    if defect=='knots':saved[4:]=2.
    if defect=='size':x=np.zeros(size-1)
    if defect=='nonfinite':x[0]=np.nan
    if defect=='matrix':x=x.reshape(2,-1)
    values={'x':x}
    if defect!='missing_knots':values['correction_knots']=saved
    source=tmp_path/'saved.npz';np.savez(source,**values)
    scope=execute_source('footwear_pose_transfer.py',dict(
        FootwearProblem=lambda *a,**k:p,load=load_npz,digest=digest,
        INITIAL=INITIAL,decode=lambda x:x.tolist()))
    def optimizer(*a,**kw):raise ReachedOptimizer('synthetic stop: no optimization is run')
    scope['least_squares']=optimizer
    if defect=='valid':
        with pytest.raises(ReachedOptimizer):scope['run'](root,review,out,source)
        saved=load_npz(out/'observed_pose/initial_state.npz')
        np.testing.assert_array_equal(saved['x'],x)
        np.testing.assert_array_equal(saved['correction_knots'],knots)
    else:
        with pytest.raises(ValueError,match='Transferred footwear'):
            scope['run'](root,review,out,source)
        assert not (out/'observed_pose').exists()


@pytest.mark.parametrize('frames',[[1,0],[2,0,1],[0,0]])
def test_fast_equipment_frame_selection_uses_source_frame_identity(module,tmp_path,frames):
    root,review,out=fixture_tree(tmp_path,observed_kind='full_equipment_external_contour',missing=0.,qualified=True)
    p=module['FootwearProblem'](root,review,'shared')
    requested=np.asarray(frames,int)
    fast=p.state(p.x0,requested,fast=True);full=p.state(p.x0,requested,fast=False)
    relevant=requested==0
    np.testing.assert_allclose(fast['uv'][relevant],full['uv'][relevant],rtol=0,atol=0)
    assert np.isnan(fast['uv'][~relevant,:,14:]).all()
