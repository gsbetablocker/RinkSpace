"""The common GPU factors consume every current native geometry family."""
import os
import json
from pathlib import Path
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.session.gpu_geometry import compile_geometry, NativeSceneGPU
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.domain.print_assets import evaluate_print_path
import test_g2_gpu as fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G4_LIBRARY"),"explicit CUDA library required")
class GPUGeometry(unittest.TestCase):
    def setUp(self):
        old=fixture.G2GPU();old.setUp();self.kw=old.kw
        self.core=NativeCore(os.environ["RINKSPACE_G4_LIBRARY"])
        self.previous=NativeCore(os.environ.get("RINKSPACE_GEOMETRY_REFERENCE_LIBRARY",os.environ["RINKSPACE_G4_LIBRARY"]))

    def test_current_entire_scene_matches_existing_native_forward(self):
        root=Path(__file__).resolve().parents[1]
        field=root/"events/ankara_2026/scene/field/model.json"
        scene=json.loads(field.read_text(encoding="utf8"))
        current,registry=compile_geometry(scene,field,self.core)
        paths=list(registry.values());stations=np.array([0.,.031,.137,.283,.417,.529,.683,.797,.893,.967,1.])
        codes=np.repeat(np.arange(len(paths)),len(stations));s=np.tile(stations,len(paths))
        kw={**self.kw,"board":current["board"],"layout":current["shared_layout"],
            "options":current["shared_geometry_options"],"codes":codes,"stations":s,
            "times":np.zeros(len(s)),"observed":np.zeros((len(s),2))}
        expected=[];assets_by_instance={}
        for ref,assets in read_scene_assets(scene,field):
            for identity in ref["instance_ids"]:assets_by_instance[identity]=assets
        for path in paths:
            if path["kind"]==8:
                xyz=evaluate_print_path(assets_by_instance[path["instance_id"]],path["instance_id"],
                    path["template_path_id"],current["board"],self.previous,stations,jacobian=False)["xyz"]
            else:
                xyz=self.previous.sample_path(current["board"],path["kind"],path["parameters"],path["vertices"],stations)
            expected.extend(xyz)
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(paths);cpu=p.evaluate("cpu");gpu=p.evaluate("cuda")
            np.testing.assert_array_equal(cpu["status"],gpu["status"])
            np.testing.assert_array_equal(gpu["status"],0)
            np.testing.assert_allclose(gpu["values"][:,:3],expected,rtol=1e-10,atol=2e-12)
            np.testing.assert_allclose(cpu["values"],gpu["values"],rtol=1e-10,atol=1e-9)
            expected_ids={path["id"] for path in scene["paths"]}
            for ref,assets in read_scene_assets(scene,field):
                for identity in ref["instance_ids"]:
                    from rinkspace.domain.print_assets import resolve_instance
                    _,template,size,placement=resolve_instance(assets,identity)
                    if size["values"] is not None and placement["values"] is not None:
                        expected_ids.update(identity+"::"+path["id"] for path in template["paths"])
            self.assertEqual(set(registry),expected_ids)
            self.assertTrue({0,1,6,7,8}.issubset({x["kind"] for x in paths}))

    def test_mixed_paths_full_jacobian_and_one_solver(self):
        paths=[{"kind":0,"parameters":[-.2,-.1,0,.2,.1,0]},
               {"kind":6,"parameters":[0,0,0,.1,.05,.2,0,3.14]},
               {"kind":7,"parameters":[1.]},
               {"kind":8,"vertices":[[.1,.005],[.2,.005],[.2,.015]],"knots":[0.,.3,1.]},
               {"kind":4,"vertices":[[.1,.1,0.]]}]
        n=len(paths);s=np.array([.2,.3,3.1,.6,0.]);dirs=np.zeros((8,3,3))
        for i in range(3):dirs[:,i,i]=.001
        kw={**self.kw,"codes":np.arange(n,dtype=np.int32),"stations":s,"times":np.zeros(n),"observed":np.zeros((n,2))}
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(paths);kw["observed"]=p.evaluate("cpu")["values"][:,3:5]
        # A shared camera change affects all families, including the fixed point.
        kw["controls"]=kw["controls"]+dirs[:,:,0]*.1
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(paths)
            p.configure_linearization(directions=dirs,local_ids=np.array([0,1,2,3,-1]),
                local_scales=np.full(4,.001),group_offsets=np.arange(n+1),members=np.arange(n),
                ranks=np.full(n,2),weight_offsets=np.arange(n+1)*4,weights=np.tile(np.eye(2).ravel(),n),huber_delta=3.)
            p.linearize();jac=p.probe_rows(np.arange(n))
            step=1e-5
            for j in range(4):
                plus={**kw};minus={**kw}
                if j<3:plus["controls"]=kw["controls"]+step*dirs[:,:,j];minus["controls"]=kw["controls"]-step*dirs[:,:,j]
                else:plus["stations"]=s+step*np.array([.001]*4+[0]);minus["stations"]=s-step*np.array([.001]*4+[0])
                projected=[]
                for values in (plus,minus):
                    with ResidentResidualProblem(self.core,**values) as check:
                        check.configure_geometry(paths);projected.append(check.evaluate("cpu")["values"][:,3:5])
                np.testing.assert_allclose(jac[:,:,j],(projected[0]-projected[1])/(2*step),rtol=2e-5,atol=2e-7)
            p.configure_solver(lower=np.array([0,0,-np.inf,0]),upper=np.array([1,1,np.inf,1]),
                device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
            result=p.solve()
            self.assertTrue(result["numerically_converged"])
            self.assertLess(result["final_cost"],1e-8)
            self.assertEqual(p.export_state()["stations"][-1],0.)

    def test_all_current_paths_run_through_continuous_gpu_solver(self):
        field=Path(__file__).resolve().parents[1]/"events/ankara_2026/scene/field/model.json"
        geometry=NativeSceneGPU(field,self.core)
        ids=np.repeat(list(geometry.paths),3).tolist();n=len(ids)
        stations=np.tile([.137,.529,.893],len(geometry.paths))
        camera={key:self.kw[key] for key in ("rig","first_control_time","control_dt","image_height")}
        camera.update(reference_pts=2**53+17,time_base_num=1,time_base_den=12800)
        pts=np.full(n,2**53+17,dtype=np.int64)+np.tile([-8960,16640,39680],len(geometry.paths))
        inputs=dict(camera=camera,controls=self.kw["controls"],path_ids=ids,stations=stations,pts=pts,observed=np.zeros((n,2)))
        with geometry.resident(**inputs) as target:
            wanted=target.evaluate("cpu")["values"][:,3:5]
        directions=np.eye(24).reshape(8,3,24)*.001
        inputs["controls"]=self.kw["controls"]+.0003*np.sin(np.arange(24).reshape(8,3)+.3)
        inputs["observed"]=wanted
        # Engineering-only measurements made in memory; no source dataset opened.
        locals=np.full(n,-1,np.int32);free=np.flatnonzero(np.arange(n)%3!=0);locals[free]=np.arange(len(free))
        periodic=np.array([geometry.paths[ids[row]]["kind"]==7 for row in free])
        lower=np.where(periodic,-np.inf,0.);upper=np.where(periodic,np.inf,1.)
        with geometry.resident(**inputs) as p:
            p.configure_linearization(directions=directions,local_ids=locals,local_scales=np.full(len(free),.001),
                group_offsets=np.arange(n+1),members=np.arange(n),ranks=np.full(n,2),
                weight_offsets=np.arange(n+1)*4,weights=np.tile(np.eye(2).ravel(),n),huber_delta=3.)
            p.configure_solver(lower=lower,upper=upper,device_budget_bytes=p.creation["free_after_alloc_bytes"],
                               host_budget_bytes=2**30)
            history=[];result=p.solve(progress=history.append,gradient_tolerance=1e-10,step_tolerance=1e-8,relative_cost_tolerance=1e-12)
            self.assertTrue(result["numerically_converged"])
            self.assertLess(result["final_cost"],1e-9)
            self.assertEqual(result["symbolic_analyses"],1)
            actual=p.evaluate("cuda")
            np.testing.assert_array_equal(actual["status"],0)
            np.testing.assert_allclose(actual["values"][:,3:5],wanted,atol=1e-5)
            self.assertGreater(len(geometry.paths),600)
            self.assertEqual(p.linearization_info["q"],24)
            self.assertTrue(all(x["summary"]["trial_domain_failures"]==0 for x in history))

    def test_geometry_authority_is_not_an_observation_input(self):
        import inspect
        params=inspect.signature(NativeSceneGPU.resident).parameters
        self.assertNotIn("scene",params)
        self.assertNotIn("board",params)
        self.assertNotIn("layout",params)
        self.assertNotIn("vertices",params)

    def test_nonzero_current_ad_state_keeps_exact_native_knots(self):
        from rinkspace.domain.parameters import SceneParameterModel
        field=Path(__file__).resolve().parents[1]/"events/ankara_2026/scene/field/model.json"
        scene=json.loads(field.read_text(encoding="utf8"))
        model=SceneParameterModel(scene,field);state=model.initial_state
        changes={}
        # Current native blocks only; create an in-memory engineering candidate.
        for identity,block in state.blocks.items():
            if block["available"] and block["kind"] in ("print_shape","print_square","print_rectangle"):
                proposed=state[identity].copy()
                active=list(block["independent_local_indices"])
                proposed[active]+=.002*np.sin(np.arange(len(active))+.3)
                changes[identity]=proposed
        self.assertTrue(changes)
        state=state.with_blocks(changes)
        current,paths=compile_geometry(scene,field,self.core,parameter_state=state)
        chosen=[p for p in paths.values() if p["kind"]==8]
        assets_by_instance={identity:assets for ref,assets in read_scene_assets(scene,field) for identity in ref["instance_ids"]}
        samples=np.array([.017,.093,.211,.337,.513,.671,.823,.989])
        expected=[]
        for path in chosen:
            expected.extend(evaluate_print_path(assets_by_instance[path["instance_id"]],path["instance_id"],path["template_path_id"],
                current["board"],self.previous,samples,jacobian=False,parameter_state=state)["xyz"])
        n=len(chosen)*len(samples)
        kw={**self.kw,"board":current["board"],"layout":current["shared_layout"],"options":current["shared_geometry_options"],
            "codes":np.repeat(np.arange(len(chosen)),len(samples)),"stations":np.tile(samples,len(chosen)),
            "times":np.zeros(n),"observed":np.zeros((n,2))}
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_geometry(chosen);actual=p.evaluate("cuda")
            np.testing.assert_array_equal(actual["status"],0)
            np.testing.assert_allclose(actual["values"][:,:3],expected,atol=3e-12,rtol=1e-10)
