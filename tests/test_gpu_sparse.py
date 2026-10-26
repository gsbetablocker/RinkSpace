"""Sparse GPU LM regression: independent equations, bounds and actual support."""
import os
import unittest
import numpy as np
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.core.native import NativeCore
import test_g2_gpu as fixture


@unittest.skipUnless(os.environ.get("RINKSPACE_G4_LIBRARY"), "explicit CUDA/cuDSS library required")
class SparseGPU(unittest.TestCase):
    def setUp(self):
        previous=fixture.G2GPU();previous.setUp();self.kw=previous.kw
        self.core=NativeCore(os.environ["RINKSPACE_G4_LIBRARY"])

    def planar(self,n):
        kw=dict(self.kw);kw["rig"]=list(kw["rig"]);kw["rig"][-1]=0.
        kw.update(controls=np.c_[np.zeros(8),np.zeros(8),np.full(8,np.log(100.))],
            codes=np.zeros(n,np.int32),stations=np.full(n,.5),times=np.zeros(n),observed=np.zeros((n,2)))
        return kw

    def independent_plan(self,p,directions,ids):
        n=p.count;nl=int(np.max(ids))+1
        p.configure_projection_time_support(half_seconds=max(.01,abs(p.export_model()["rig"][37])/2))
        return p.configure_linearization(directions=directions,local_ids=ids,local_scales=np.full(nl,.001),
            group_offsets=np.arange(n+1),members=np.arange(n),ranks=np.full(n,2),
            weight_offsets=np.arange(n+1)*4,weights=np.tile(np.eye(2).ravel(),n),huber_delta=3.)

    def solver(self,p,**options):
        n=int(p.linearization_info["local_count"])
        return p.configure_solver(lower=np.zeros(n),upper=np.ones(n),
            device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30,**options)

    def test_near_bound_and_large_damping_cannot_claim_premature_convergence(self):
        # Analytic pinhole observations: x=-.2+.4*s, z_camera=3, f=100.
        # The constrained minimizer is exactly clip(target,0,1).
        target=np.array([-.3,.8,.2]);kw=self.planar(3)
        kw["observed"]=np.c_[50+100*(-.2+.4*target)/3,np.full(3,50.)]
        kw["stations"][0]=1e-12
        for initial_lambda in (0.,1e8):
            with self.subTest(initial_lambda=initial_lambda),ResidentResidualProblem(self.core,**kw) as p:
                p.configure_geometry([{"kind":0,"parameters":[-.2,0.,0.,.2,0.,0.]}])
                self.independent_plan(p,np.zeros((8,3,1)),np.arange(3))
                self.solver(p,initial_lambda=initial_lambda);history=[]
                result=p.solve(progress=history.append)
                self.assertTrue(result["numerically_converged"])
                self.assertGreater(result["accepted"],1)
                self.assertLessEqual(result["projected_gradient_inf"],result["tolerances"]["gradient"])
                np.testing.assert_allclose(p.export_state()["stations"],np.clip(target,0,1),atol=2e-6)
                self.assertAlmostEqual(result["final_cost"],8.,places=7)
                # Local support projection must not throttle unrelated parameters.
                if initial_lambda==0:self.assertEqual(history[0]["summary"]["feasible_alpha"],1.)

    def test_sparse_components_match_independent_full_whitened_system(self):
        n=10;nc=24;q=3*nc;kw=dict(self.kw)
        kw.update(controls=np.c_[np.linspace(-.01,.01,nc),np.linspace(-.02,.02,nc),np.full(nc,np.log(100.))],
            codes=np.full(n,100,np.int32),stations=np.array([.13,.3,.3,.6,.7,.3,.4,.4,.2,.6]),
            times=np.array([0.,.1,6.,6.1,12.,12.1,12.2,18.,18.1,18.2]),observed=np.zeros((n,2)))
        with ResidentResidualProblem(self.core,**kw) as target:
            kw["observed"]=target.evaluate("cpu")["values"][:,3:5]+np.array([-.02,.015])
        ids=np.array([0,1,1,2,-1,3,4,4,-1,5]);nl=6;offsets=np.array([0,2,5,7,10]);rng=np.random.default_rng(873)
        matrices=[np.eye(2*m)+rng.normal(size=(2*m,2*m))*.06 for m in np.diff(offsets)]
        woffset=np.r_[0,np.cumsum([w.size for w in matrices])]
        directions=np.eye(q).reshape(nc,3,q)*.001
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_projection_time_support(half_seconds=max(.01,abs(p.export_model()["rig"][37])/2))
            p.configure_linearization(directions=directions,local_ids=ids,local_scales=np.full(nl,.001),
                group_offsets=offsets,members=np.arange(n),ranks=2*np.diff(offsets),
                weight_offsets=woffset,weights=np.concatenate([w.ravel() for w in matrices]),huber_delta=1e6)
            self.assertLess(p.sparsity["max_component_q"],q)
            self.assertLess(p.sparsity["retained_nnz"],q*(q+1)//2)
            p.linearize();jac=p.probe_rows(np.arange(n));raw=p.evaluate("cpu")["values"][:,9:11]
            J=np.zeros((2*n,q+nl));J[:,:q]=jac[:,:,:q].reshape(2*n,q)
            for row,local in enumerate(ids):
                if local>=0:J[2*row:2*row+2,q+local]=jac[row,:,-1]
            whiteJ=[];whiter=[]
            for i,w in enumerate(matrices):
                a,b=2*offsets[i],2*offsets[i+1];whiteJ.extend(w@J[a:b]);whiter.extend(w@raw.ravel()[a:b])
            whiteJ=np.array(whiteJ);whiter=np.array(whiter);H=whiteJ.T@whiteJ;g=whiteJ.T@whiter
            blocks=p.export_sparse_blocks();assembled=np.zeros_like(H);gradient=np.zeros_like(g)
            for ci,c in enumerate(p.layout["components"]):
                lb,nlocal,_,_,dim,hoff,goff=map(int,c);begin,count=map(int,p.layout["component_columns"][ci])
                indices=np.r_[p.layout["columns"][begin:begin+count],q+p.layout["locals"][lb:lb+nlocal]]
                assembled[np.ix_(indices,indices)]+=blocks["H"][hoff:hoff+dim*dim].reshape(dim,dim)
                gradient[indices]+=blocks["g"][goff:goff+dim]
            np.testing.assert_allclose(assembled,H,rtol=2e-12,atol=1e-13)
            np.testing.assert_allclose(gradient,g,rtol=2e-12,atol=1e-13)
            p.linearize("cpu");reference=p.export_sparse_blocks("cpu")
            for key in ("H","g","shared_values","shared_g"):np.testing.assert_allclose(blocks[key],reference[key],rtol=1e-10,atol=1e-12)
            self.solver(p);step=p.lm_step();system=p.export_step_system()
            probe=p.probe_step_changes()
            np.testing.assert_allclose(probe[:,:2],raw+kw["observed"],atol=1e-12)
            np.testing.assert_allclose(probe[:,4:],(J@system["applied"]).reshape(n,2),atol=1e-12)
            np.testing.assert_allclose(probe[:,2:4],p.evaluate("cuda")["values"][:,3:5],atol=1e-12)
            self.assertEqual(step["summary"]["feasible_alpha"],1.)
            expected=np.linalg.solve(H+step["summary"]["lambda_used"]*np.eye(q+nl),-g)
            np.testing.assert_allclose(system["step"],expected,rtol=1e-7,atol=2e-8)
            np.testing.assert_allclose(system["S"]@system["step"][:q],system["rhs"],rtol=1e-8,atol=1e-11)
            import runpy
            from pathlib import Path
            checks=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/evaluate_g3.py"))
            checks["system_check"]({"directions":directions,"local_scales":np.full(nl,.001)},p.layout,
                {**blocks,"shared_H":H[:q,:q],"shared_g":g[:q]},system,step)
            sparse=p.export_step_system(sparse=True)
            self.assertEqual(sparse["csr_values"].size,p.sparsity["retained_nnz"])

    def test_jacobi_damping_matches_full_system_and_is_coordinate_scale_invariant(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw
        with ResidentResidualProblem(self.core,**kw) as p:kw['observed']=p.evaluate('cpu')['values'][:,3:5]+[.01,-.02]
        states=[]
        for scale in (np.ones(4),np.array([1e-3,1e3,1e2,1e-2])):
            plan={**f.plan,'directions':f.directions*scale[:2],'local_scales':f.plan['local_scales']*scale[2:]}
            with ResidentResidualProblem(self.core,**kw) as p:
                p.configure_linearization(**plan);p.linearize();blocks=p.export_blocks();H=blocks['H'].reshape(4,4);g=blocks['g']
                self.solver(p,damping_metric='jacobi');step=p.lm_step()['summary'];system=p.export_step_system()
                diagonal=np.diag(H);floor=64*np.finfo(float).eps*diagonal.max();diagonal=np.maximum(diagonal,floor if floor>0 else 1.)
                expected=np.linalg.solve(H+step['lambda_used']*np.diag(diagonal),-g)
                np.testing.assert_allclose(system['step'],expected,rtol=2e-8,atol=1e-9)
                self.assertTrue(step['accepted'],step);states.append(p.export_state())
        for key in ('controls','stations'):np.testing.assert_allclose(states[0][key],states[1][key],rtol=2e-10,atol=1e-10)

    def test_image_station_metric_preserves_source_likelihood_and_full_system(self):
        kw=self.planar(2);kw['observed']=np.array([[50.,51.],[50.,50.]])
        # The source chiefly measures vertical displacement. A nearly horizontal
        # curve makes its station weakly observable; raw image speed stays finite.
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry([dict(kind=0,parameters=[-.2,0.,0.,.2,.0004,0.])])
            directions=np.zeros((8,3,1));directions[:,1,0]=.001
            W=np.diag([1e-6,1.]);p.configure_linearization(directions=directions,
                local_ids=np.array([0,1]),local_scales=np.full(2,.001),group_offsets=np.arange(3),
                members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,
                weights=np.tile(W.ravel(),2),huber_delta=100.)
            before=p.linearize()['summary'];raw=p.evaluate('cuda')['values'][:,9:11];jac=p.probe_rows(np.arange(2))
            J=np.zeros((4,3));J[:,0]=jac[:,:,0].ravel()
            for row in range(2):J[2*row:2*row+2,row+1]=jac[row,:,-1]
            B=np.kron(np.eye(2),W);H=J.T@B.T@B@J;g=J.T@B.T@B@raw.ravel();diag=np.diag(H).copy()
            diag=np.maximum(diag,64*np.finfo(float).eps*diag.max());diag[1:]=np.maximum(diag[1:],np.sum(jac[:,:,-1]**2,axis=1))
            self.solver(p,damping_metric='jacobi_image_station',initial_lambda=1.)
            step=p.lm_step()['summary'];system=p.export_step_system()
            np.testing.assert_allclose(system['step'],np.linalg.solve(H+step['lambda_used']*np.diag(diag),-g),rtol=2e-9,atol=1e-10)
            self.assertAlmostEqual(step['old_cost'],before['huber_cost'],places=12)
            self.assertLess(abs(system['step'][1]),1.)

    def test_image_global_metric_preserves_source_likelihood_and_full_system(self):
        kw=self.planar(2);kw['observed']=np.array([[50.,51.],[50.,50.]])
        # The source chiefly measures vertical displacement. A nearly horizontal
        # curve makes its station weakly observable; raw image speed stays finite.
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry([dict(kind=0,parameters=[-.2,0.,0.,.2,.0004,0.])])
            directions=np.zeros((8,3,1));directions[:,1,0]=.001
            W=np.diag([1e-6,1.]);p.configure_linearization(directions=directions,
                local_ids=np.array([0,1]),local_scales=np.full(2,.001),group_offsets=np.arange(3),
                members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,
                weights=np.tile(W.ravel(),2),huber_delta=100.)
            before=p.linearize()['summary'];raw=p.evaluate('cuda')['values'][:,9:11];jac=p.probe_rows(np.arange(2))
            J=np.zeros((4,3));J[:,0]=jac[:,:,0].ravel()
            for row in range(2):J[2*row:2*row+2,row+1]=jac[row,:,-1]
            B=np.kron(np.eye(2),W);H=J.T@B.T@B@J;g=J.T@B.T@B@raw.ravel();diag=np.diag(H).copy()
            diag=np.maximum(diag,64*np.finfo(float).eps*diag.max());diag=np.maximum(1.,np.maximum(diag,np.sum(J*J,axis=0)))
            self.solver(p,damping_metric='jacobi_image',initial_lambda=1.)
            step=p.lm_step()['summary'];system=p.export_step_system()
            np.testing.assert_allclose(system['step'],np.linalg.solve(H+step['lambda_used']*np.diag(diag),-g),rtol=2e-9,atol=1e-10)
            self.assertAlmostEqual(step['old_cost'],before['huber_cost'],places=12)
            self.assertLess(abs(system['step'][1]),1.)

    def test_rs_support_union_keeps_columns_across_a_spline_knot(self):
        kw=dict(self.kw);nc=20;q=3*nc;n=4
        kw["rig"]=list(kw["rig"]);kw["rig"][-1]=.25
        kw.update(controls=np.c_[np.linspace(-.01,.01,nc),np.linspace(-.02,.02,nc),np.full(nc,np.log(100.))],
            codes=np.full(n,100,np.int32),stations=np.array([.1,.3,.6,.8]),times=np.array([1.99,2.,2.01,2.02]),observed=np.zeros((n,2)))
        directions=np.eye(q).reshape(nc,3,q)*.001
        with ResidentResidualProblem(self.core,**kw) as p:
            self.independent_plan(p,directions,np.arange(n));p.linearize();jac=p.probe_rows(np.arange(n))
            self.assertEqual(p.sparsity["max_row_q"],15)
            for col in (9,12,14,18,20,23):
                values=[];h=1e-4
                for sign in (1,-1):
                    with ResidentResidualProblem(self.core,**{**kw,"controls":kw["controls"]+sign*h*directions[:,:,col]}) as shifted:
                        values.append(shifted.evaluate("cpu")["values"][:,3:5])
                np.testing.assert_allclose(jac[:,:,col],(values[0]-values[1])/(2*h),rtol=3e-5,atol=2e-7)

    def test_ten_thousand_rows_do_not_repeat_global_camera_columns(self):
        n=10000;nc=160;q=nc*3;kw=dict(self.kw)
        kw.update(controls=np.c_[np.linspace(-.01,.01,nc),np.linspace(-.02,.02,nc),np.full(nc,np.log(100.))],
            codes=np.full(n,100,np.int32),stations=np.full(n,.3),times=np.linspace(-.5,154.5,n),observed=np.zeros((n,2)))
        with ResidentResidualProblem(self.core,**kw) as p:
            info=self.independent_plan(p,np.eye(q).reshape(nc,3,q)*.001,np.arange(n))
            self.assertLessEqual(info["max_row_q"],15)
            self.assertLessEqual(info["H_value_count"],n*16**2)
            self.assertLessEqual(info["jacobian_values"],n*2*16)
            self.assertLess(info["retained_nnz"],q*16)
            self.assertEqual(p.linearize()["summary"]["failed_rows"],0)
            self.solver(p)
            self.assertEqual(p.solver_info["local_matrix_values"],n)

    def test_ragged_local_workspace_does_not_pad_every_component(self):
        n=400;kw=dict(self.kw);kw.update(codes=np.full(n,100,np.int32),stations=np.full(n,.3),times=np.zeros(n),observed=np.zeros((n,2)))
        offsets=np.r_[np.arange(n-64+1),n];sizes=np.diff(offsets);wo=np.r_[0,np.cumsum((2*sizes)**2)]
        directions=np.zeros((8,3,2));directions[:,0,0]=.001;directions[:,2,1]=.001
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_linearization(directions=directions,local_ids=np.arange(n),local_scales=np.full(n,.001),
                group_offsets=offsets,members=np.arange(n),ranks=2*sizes,weight_offsets=wo,
                weights=np.concatenate([np.eye(2*m).ravel() for m in sizes]),huber_delta=3.)
            self.solver(p)
            self.assertEqual(p.solver_info["max_local_dimension"],64)
            self.assertEqual(p.solver_info["local_matrix_values"],n-64+64**2)
            p.lm_step()

    def test_polyline_kinks_select_both_sides_and_corner_minimum_without_locking(self):
        target=np.array([[-.3,.1],[.1,.3],[.2,-.2],[.2,.3]])
        kw=self.planar(4);kw['observed']=50+100*target/3;kw['stations']=np.array([.5,.5,.5,.4999])
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry([dict(kind=4,vertices=[[-1.,0.,0.],[0.,0.,0.],[0.,1.,0.]],parameters=[])])
            self.independent_plan(p,np.zeros((8,3,1)),np.arange(4))
            p.linearize('cuda');gpu=p.probe_rows(np.arange(4));p.linearize('cpu');np.testing.assert_allclose(gpu,p.probe_rows(np.arange(4),'cpu'),atol=1e-13)
            self.assertGreater(gpu[0,0,-1],0.);self.assertAlmostEqual(gpu[0,1,-1],0.)
            self.assertGreater(gpu[1,1,-1],0.);np.testing.assert_array_equal(gpu[2,:,-1],0.)
            self.solver(p);history=[];result=p.solve(progress=history.append)
            self.assertTrue(result['numerically_converged'],result)
            np.testing.assert_allclose(p.export_state()['stations'],[.35,.65,.5,.65],atol=2e-6)
            self.assertTrue(any(x['summary']['accepted'] for x in history))

    def test_correlated_shared_corner_station_uses_full_original_weight(self):
        kw=self.planar(2);kw['observed']=np.array([[46.,54.],[51.,55.]])
        vertices=np.array([[-1.,0.,0.],[0.,0.,0.],[0.,1.,0.]])
        W=np.array([[1.,.2,.1,0.],[0.,2.,.2,.1],[.1,0.,1.,.2],[0.,.1,0.,2.]])
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry([dict(kind=4,vertices=vertices,parameters=[])])
            p.configure_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,0]),local_scales=np.array([.001]),group_offsets=np.array([0,2]),members=np.arange(2),ranks=np.array([4]),weight_offsets=np.array([0,16]),weights=W.ravel(),huber_delta=100.)
            p.linearize('cuda');gpu=p.probe_rows(np.arange(2));p.linearize('cpu');np.testing.assert_allclose(gpu,p.probe_rows(np.arange(2),'cpu'),atol=1e-13)
            self.solver(p);result=p.solve();self.assertTrue(result['numerically_converged'],result)
            residual=(50-kw['observed']).ravel();j=np.tile([0.,200/3],2);expected=.5-np.dot(W@j,W@residual)/np.dot(W@j,W@j)
            np.testing.assert_allclose(p.export_state()['stations'],expected,atol=2e-6)

    def test_nonzero_corner_bound_enters_coupled_local_and_retained_rhs(self):
        kw=self.planar(3);kw['stations']=np.array([.499,.2,.3])
        geometry=[dict(kind=4,vertices=[[-1.,0.,0.],[0.,0.,0.],[0.,1.,0.]],parameters=[])]
        with ResidentResidualProblem(self.core,**kw) as target:
            target.configure_geometry(geometry);kw['observed']=target.evaluate('cpu')['values'][:,3:5]+[[4.,0.],[.2,.1],[0.,0.]]
        W=np.eye(6);W[0,2]=.2;W[2,4]=.3;W[4,0]=-.1
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(geometry);rigdirs=np.zeros((27,1));rigdirs[15,0]=1.
            p.configure_camera_parameters(directions=rigdirs,lower=np.full(38,-np.inf),upper=np.full(38,np.inf),max_readout=.1)
            p.configure_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,1,-1]),local_scales=np.full(2,.001),group_offsets=np.array([0,3]),members=np.arange(3),ranks=np.array([6]),weight_offsets=np.array([0,36]),weights=W.ravel(),huber_delta=100.)
            p.linearize();jac=p.probe_rows(np.arange(3));raw=p.evaluate('cpu')['values'][:,9:11].ravel()
            J=np.zeros((6,3));J[:,0]=jac[:,:,0].ravel();J[:2,1]=jac[0,:,-1];J[2:4,2]=jac[1,:,-1]
            H=J.T@W.T@W@J;g=J.T@W.T@W@raw
            self.solver(p,initial_lambda=.0001);step=p.lm_step()['summary'];system=p.export_step_system()
            A=H+step['lambda_used']*np.eye(3);expected=np.zeros(3);expected[1]=(.5-kw['stations'][0])/.001;free=np.array([0,2])
            expected[free]=np.linalg.solve(A[np.ix_(free,free)],-g[free]-A[free,1]*expected[1])
            np.testing.assert_allclose(system['step'],expected,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(system['applied'],expected,rtol=1e-10,atol=1e-10)
            self.assertGreater(step['bound_passes'],1);self.assertTrue(step['accepted'],step)
            self.assertAlmostEqual(step['gain_ratio'],1.,places=9)

    def test_retained_physical_bound_reallocates_correlated_joint_step(self):
        kw=self.planar(3);kw['stations']=np.array([.3,.4,.7]);geometry=[dict(kind=0,parameters=[-1.,0.,0.,1.,0.,0.])]
        with ResidentResidualProblem(self.core,**kw) as target:
            target.configure_geometry(geometry);kw['observed']=target.evaluate('cpu')['values'][:,3:5]+[4.,3.]
        W=np.eye(6);W[0,1]=.25;W[2,3]=.4;W[4,5]=-.15;W[1,2]=.2
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(geometry);rd=np.zeros((27,2));rd[15,0]=rd[16,1]=1.;lo=np.full(38,-np.inf);hi=-lo;hi[26]=50.2
            p.configure_camera_parameters(directions=rd,lower=lo,upper=hi,max_readout=.1)
            p.configure_linearization(directions=np.zeros((8,3,2)),local_ids=np.array([0,-1,-1]),local_scales=np.array([.001]),group_offsets=np.array([0,3]),members=np.arange(3),ranks=np.array([6]),weight_offsets=np.array([0,36]),weights=W.ravel(),huber_delta=100.)
            p.linearize();jac=p.probe_rows(np.arange(3));raw=p.evaluate('cpu')['values'][:,9:11].ravel();J=np.zeros((6,3));J[:,:2]=jac[:,:,:2].reshape(6,2);J[:2,2]=jac[0,:,-1]
            H=J.T@W.T@W@J;g=J.T@W.T@W@raw
            self.solver(p,initial_lambda=.0001);summary=p.lm_step()['summary'];system=p.export_step_system();A=H+summary['lambda_used']*np.eye(3)
            expected=np.zeros(3);expected[0]=.2;free=np.array([1,2]);expected[free]=np.linalg.solve(A[np.ix_(free,free)],-g[free]-A[free,0]*expected[0])
            np.testing.assert_allclose(system['step'],expected,rtol=1e-10,atol=1e-9);self.assertTrue(summary['accepted'],summary);self.assertAlmostEqual(summary['gain_ratio'],1.,places=9)
            self.assertLessEqual(p.export_state()['rig'][26],50.2)
