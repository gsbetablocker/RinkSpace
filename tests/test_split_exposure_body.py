import importlib.util,os,unittest
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation,Slerp
from calibration.program_tracking.body_tracking.rebuild_exposure_body import row_weights,moving_projection,ExposureArticulatedSplineProblem
from calibration.program_tracking.body_tracking.rebuild_body_fit import export_exposure_state
from calibration.program_tracking.body_tracking.footwear_fit import FootwearProblem


class SplitWeightsTests(unittest.TestCase):
    def test_piecewise_nodes_and_row_derivatives(self):
        rows=np.array([-2.,0.,100.,539.5,700.,1079.,1081.]);w,d=row_weights(rows,5)
        np.testing.assert_allclose(w.sum(-1),1);np.testing.assert_allclose(d.sum(-1),0,atol=1e-15)
        for i in [2,4]:
            a,_=row_weights(rows[i]+1e-4,5);b,_=row_weights(rows[i]-1e-4,5)
            np.testing.assert_allclose((a-b)/2e-4,d[i],atol=1e-10)
        np.testing.assert_array_equal(d[[0,-1]],0)
        knots=np.linspace(0,1079,5);v,_=row_weights(knots,5);np.testing.assert_allclose(v,np.eye(5),atol=1e-15)


@unittest.skipUnless(os.environ.get('RINKSPACE_S01_BODY_ROOT'),'requires selected local source experiment')
class NativeSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['RINKSPACE_S01_BODY_ROOT']);cls.z=dict(np.load(cls.root/'pose/result.npz'));cls.p3=ExposureArticulatedSplineProblem(cls.root,'pose',exposure_nodes=3);cls.p5=ExposureArticulatedSplineProblem(cls.root,'pose',exposure_nodes=5)
    def test_three_nodes_replay_saved_B_and_original_residual(self):
        import calibration.program_tracking.body_tracking as package
        name=package.__name__+'._pre_split_exposure';file=self.root/'next_round_20261029/pre_patch/rebuild_exposure_body.py';spec=importlib.util.spec_from_file_location(name,file);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        p=old.ExposureArticulatedSplineProblem(self.root,'pose');a=p.evaluate(self.z['x'],jacobian=True);b=self.p3.evaluate(self.z['x'],jacobian=True)
        np.testing.assert_allclose(a['residual'],b['residual'],atol=1e-12,rtol=0);d=a['jac']-b['jac'];self.assertLess(float(np.max(abs(d.data))) if d.nnz else 0.,1e-12)
        out=self.p3.output_projections(self.z['x'],np.arange(len(self.z['pts'])));np.testing.assert_allclose(out['uv'][:,:17],self.z['COCO_uv'],atol=1e-8,rtol=0)
    def test_five_node_body_and_contact_directional_derivatives(self):
        rng=np.random.default_rng(91)
        for mode in ['pose','pose_contact']:
            p=self.p5 if mode=='pose' else ExposureArticulatedSplineProblem(self.root,mode,exposure_nodes=5);x=np.load(self.root/mode/'result.npz')['x'];J=p.evaluate(x,jacobian=True)['jac']
            blocks=[slice(None),slice(0,3*p.m),slice(3*p.m,6*p.m),slice(6*p.m,None)] if mode=='pose' else [slice(None)]
            for block in blocks:
                d=np.zeros(len(x));d[block]=rng.normal(size=len(d[block]));d/=np.linalg.norm(d);h=1e-5;fd=(p.fun(x+h*d)-p.fun(x-h*d))/(2*h)
                self.assertLess(float(np.linalg.norm(fd-J@d)/np.linalg.norm(fd)),1e-6)
    def test_five_nodes_against_exact_row_FK_and_new_exports(self):
        p=self.p5;x=self.z['x'];frames=np.searchsorted(p.p['pts'],[315392,317952,350976,366592,367616,453632,461056]);approx=p.output_projections(x,frames);uv=approx['uv'].copy();t=p.p['fit_time_seconds'];tau=float(p.p['rig'][37]);poses=[Slerp(t,Rotation.from_rotvec(p.p['pose_axis_angle'][:,j])) for j in range(21)];qr=Slerp(t,Rotation.from_matrix(p.p['raw_root_prior_Q_world']))
        for _ in range(32):
            times=(t[frames,None]+tau*(np.clip(uv[:,:,1],0,1079)/1079-.5)).ravel();B=BSpline.design_matrix(times,p.knots,3,extrapolate=True);clipped=np.clip(times,t[0],t[-1]);pose=np.stack([s(clipped).as_rotvec() for s in poses],axis=1);P=B@x[:3*p.m].reshape(p.m,3);Q=Rotation.from_rotvec(B@x[3*p.m:6*p.m].reshape(p.m,3)).as_matrix()@qr(clipped).as_matrix();delta=B@x[6*p.m:].reshape(p.m,p.dof);c,_,blade=p.body.forward(pose,delta);local=np.concatenate([c,blade.reshape(len(c),10,3)],1);selected=local[np.arange(len(c)),np.tile(np.arange(27),len(frames))];world=(P+np.einsum('nij,nj->ni',Q,selected)).reshape(len(frames),27,3);new=p.projection.project(world,frames)
            if np.max(abs(new-uv))<1e-8:uv=new;break
            uv=new
        self.assertLess(float(np.max(np.linalg.norm(uv-approx['uv'],axis=2))),.02)
        fields=export_exposure_state(p,x);np.testing.assert_array_equal(fields['P_world_mid'],self.z['P_world_m']);np.testing.assert_array_equal(fields['G_world_mid'],self.z['G_world_m']);self.assertEqual(fields['P_world_at_row'].shape,(1143,3));self.assertEqual(fields['blade_row_time'].shape,(1143,2,5))
        expected=p.p['pts']/12800+tau*(np.clip(fields['P_uv_dynamic'][:,1],0,1079)/1079-.5);np.testing.assert_allclose(fields['P_row_time'],expected,atol=1e-12)
    def test_five_node_footwear_fast_full_and_sparse_support(self):
        dataset=self.root/'footwear_v2/semantic_transfer';review=dataset/'observed_pose/source_review_at_run.json';p=FootwearProblem(self.root,review,'articulated',dataset=dataset,exposure_nodes=5);x=np.load(dataset/'observed_pose/result.npz')['x'];a=p.state(x,fast=True);b=p.state(x,fast=False);m=np.isfinite(a['uv']);np.testing.assert_allclose(a['uv'][m],b['uv'][m],atol=1e-8,rtol=1e-12)
        for column in [0,10,13,28,210,451,780,867]:
            d=np.zeros(len(x));d[column]=1e-5;v=(p.fun(x+d)-p.fun(x-d))/2e-5;outside=~p.sparsity[:,column].toarray().ravel().astype(bool);self.assertLess(float(np.max(abs(v[outside]))),1e-8)

if __name__=='__main__':unittest.main()
