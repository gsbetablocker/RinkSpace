"""Contact Jacobian regression without private models, Torch or viewer imports.

The production functions are loaded from their syntax tree to keep this test
independent of optional media/runtime dependencies.  Directional differences
exercise actual FK and a deforming upper witness, including nonzero root and
joint increments, both feet, row signs and noncontiguous spline supports.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from scipy.spatial.transform import Rotation


def load_kernel(project=None):
    project=Path(project) if project is not None else Path(__file__).resolve().parents[1]
    directory=project/'calibration/program_tracking/body_tracking'
    namespace=dict(np=np,coo_matrix=coo_matrix,csr_matrix=csr_matrix,Rotation=Rotation,NORMALIZATION=.005)
    for filename,names in [('fullbody_state.py',{'matrices'}),
                           ('fullbody_fit.py',{'skew','right_jacobian'}),
                           ('skate_physics.py',{'fk_state','body_constraint_jacobian'})]:
        path=directory/filename
        tree=ast.parse(path.read_text(encoding='utf-8'))
        selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
        if {node.name for node in selected}!=names:raise RuntimeError('Production kernel functions are missing: '+filename)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**namespace)


class SyntheticBinding:
    """Rigid ankle registration and ankle-relative upper deformation."""
    ratio=.9993616177797032

    def __init__(self):
        self.R_FB=Rotation.from_rotvec([[.13,-.08,.11],[-.06,.14,-.05]]).as_matrix()
        self.t_FB=np.array([[.025,-.041,-.083],[-.021,.036,-.078]])
        self.upper=np.array([[-.05,-.1,.05],[.06,-.09,.11],[.04,.04,.12],[-.03,.05,.09]])
        self.modes=np.arange(3*4*3,dtype=float).reshape(3,4,3)*.0007-.008

    def transforms(self,P,Q,joints,frames):
        ankles=frames[:,[7,8]]
        R=Q[:,None]@ankles@self.R_FB[None]
        local=joints[:,[7,8]]+np.einsum('nfij,fj->nfi',ankles,self.t_FB)
        O=P[:,None]+np.einsum('nij,nfj->nfi',Q,local)
        return R,O

    def deformation(self,joints,frames):
        relative=frames[:,[4,5]].transpose(0,1,3,2)@frames[:,[7,8]]
        return np.stack((relative[:,:,2,1],relative[:,:,0,2],relative[:,:,1,0]),axis=-1)

    def deform(self,vertices,alpha):
        return np.asarray(vertices)[None,None]+np.einsum('nfd,dvi->nfvi',alpha,self.modes)

    def packet(self):return dict(upper=self.upper)


def make_fixture(n=17,*,active=None,upper=True,seed=1104,controls=None):
    kernel=load_kernel();rng=np.random.default_rng(seed)
    parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
    joints=np.cumsum(rng.normal(0.,.06,(22,3)),axis=0)
    binding=SyntheticBinding()
    raw=dict(P=rng.normal(0.,.2,(n,3)),Q=Rotation.from_rotvec(rng.normal(0.,.3,(n,3))).as_matrix(),
             local_R=Rotation.from_rotvec(rng.normal(0.,.2,(n*21,3))).as_matrix().reshape(n,21,3,3))
    reference=SimpleNamespace(body=SimpleNamespace(J=joints,parents=parents,scale=1.07),binding=binding,at_calls=0)
    def at(times):
        reference.at_calls+=1
        if len(times)!=n:raise AssertionError('Unexpected reference time selection')
        return raw
    reference.at=at
    delta=rng.normal(0.,.17,(n,69));delta[:,:3]*=.1
    state=kernel.fk_state(reference,raw,delta)
    active=np.arange(69) if active is None else np.asarray(active,int)
    # Four or fewer local cubic-style nonzeros. Actual numeric values do not
    # depend on a particular spline implementation; derivatives must scatter
    # through precisely the supplied basis, including all-zero rows.
    m=max(6,min(51,n//8+4)) if controls is None else int(controls);B=np.zeros((n,m))
    for i in range(n):
        if i%19==18:continue
        k=min(i*m//max(n,1),m-4);weights=rng.uniform(.05,1.,4);weights/=weights.sum()
        B[i,k:k+4]=weights
    deformed=binding.deform(binding.upper,binding.deformation(state['native'],state['frames']))
    rows=[];upper_rows=[]
    for i in range(n):
        for foot in (0,1):
            vertex=(i+foot)%len(binding.upper) if upper and (i+foot)%3==0 else -1
            q=deformed[i,foot,vertex] if vertex>=0 else rng.normal(0.,.12,3)
            rows.append((i,foot,q.copy(),-1. if (i+foot)%5==0 else 1.));upper_rows.append(vertex)
    # Extra support inequalities repeat a physical time with opposite signs.
    if n:
        rows.extend([(0,1,rows[1][2].copy(),1.),(0,1,rows[1][2].copy(),-1.)])
        upper_rows.extend([upper_rows[1],upper_rows[1]])
    evaluation=dict(rows=rows,row_upper_vertex=upper_rows,times=np.arange(n)*.02)
    return kernel,reference,raw,state,delta,active,B,evaluation


def fixed_witness_values(kernel,reference,raw,delta,evaluation):
    state=kernel.fk_state(reference,raw,delta);binding=reference.binding
    upper=binding.deform(binding.upper,binding.deformation(state['native'],state['frames']))
    out=[]
    upper_rows=evaluation.get('row_upper_vertex',[-1]*len(evaluation['rows']))
    for row,(frame,foot,q,sign) in enumerate(evaluation['rows']):
        if upper_rows[row]>=0:q=upper[frame,foot,upper_rows[row]]
        point=state['O'][frame,foot]+state['R'][frame,foot]@q
        out.append(sign*point[2]*binding.ratio/kernel.NORMALIZATION)
    return np.asarray(out)


class ContactJacobianPerformanceTests(unittest.TestCase):
    def check_direction(self,active,*,upper=True,omit_upper=False):
        kernel,reference,raw,state,delta,active,B,evaluation=make_fixture(active=active,upper=upper)
        if omit_upper:evaluation.pop('row_upper_vertex')
        actual=kernel.body_constraint_jacobian(reference,state,delta,active,B,evaluation)
        self.assertIsInstance(actual,csr_matrix)
        self.assertEqual(actual.shape,(len(evaluation['rows']),B.shape[1]*len(active)))
        self.assertFalse(np.any(actual.data==0))
        rng=np.random.default_rng(225)
        for _ in range(3):
            direction=rng.normal(size=(B.shape[1],len(active)));direction/=np.linalg.norm(direction)
            physical=np.zeros_like(delta);physical[:,active]=B@direction
            h=1e-6
            numeric=(fixed_witness_values(kernel,reference,raw,delta+h*physical,evaluation)-
                     fixed_witness_values(kernel,reference,raw,delta-h*physical,evaluation))/(2*h)
            np.testing.assert_allclose(actual@direction.ravel(),numeric,atol=3e-7,rtol=3e-7)
        return reference,actual

    def test_all_69_columns_with_dynamic_upper_witnesses(self):
        reference,_=self.check_direction(np.arange(69))
        # Six ankle columns reuse exactly one immutable reference snapshot.
        self.assertEqual(reference.at_calls,1)

    def test_reordered_reduced_columns_and_repeated_time_rows(self):
        self.check_direction([24,3,2,9,25,0,6,28,5,1,8,27,4,26,7])

    def test_root_only_and_optional_upper_metadata(self):
        reference,_=self.check_direction([2,0,1],upper=False,omit_upper=True)
        self.assertEqual(reference.at_calls,0)

    def test_pose_only(self):self.check_direction(np.arange(3,69))

    def test_nonleg_rotations_are_zero_for_rigid_witnesses(self):
        _,actual=self.check_direction([12,13,14,30,31,32,60,61,62],upper=False)
        self.assertEqual(actual.nnz,0)

    def test_empty_constraint_set_and_empty_active_columns(self):
        kernel,reference,_,state,delta,active,B,evaluation=make_fixture(n=4,upper=False)
        evaluation['rows']=[];evaluation['row_upper_vertex']=[]
        actual=kernel.body_constraint_jacobian(reference,state,delta,active,B,evaluation)
        self.assertEqual(actual.shape,(0,B.shape[1]*69));self.assertEqual(actual.nnz,0)
        evaluation['rows']=[(0,0,np.zeros(3),1.)];evaluation['row_upper_vertex']=[-1]
        actual=kernel.body_constraint_jacobian(reference,state,delta,[],B,evaluation)
        self.assertEqual(actual.shape,(1,0));self.assertEqual(actual.nnz,0)


if __name__=='__main__':unittest.main()
