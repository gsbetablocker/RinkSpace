"""All ordinary warm finite-path GN candidates; true RS upper bounds, not station KKT."""
from pathlib import Path
import sys,json,hashlib,time,numpy as np
R=Path("D:/CodexData/RinkSpace_Data/runs/okada_s1_r11");F=R/"physical_reduced_component262_rf29_r11_transport/python_frozen";sys.path.insert(0,str(F))
from rinkfit_start.material_foot import _certified_candidates,_implicit_capture,select_certified_upper_bound
def refine_all_ordinary(scene,camera,material,branch,station,row,times,observed,active,kind,*,model_reference):
 t0=time.perf_counter();bound=camera.at.assert_bound_model(model_reference);p=Path(model_reference["path"]);loaded=json.loads(p.read_text())
 if scene.model!=loaded:raise ValueError("Scene differs from the exact bound current camera model")
 material=np.asarray(material);branch=np.asarray(branch);station=np.asarray(station);row=np.asarray(row);active=np.asarray(active,bool);kind=np.asarray(kind);ordinary=active&(kind==0);ids=np.flatnonzero(ordinary)
 if not (len(ids)==179363 and active.sum()==182016 and np.flatnonzero(~active).tolist()==[25537]):raise ValueError("All current programme ordinary sources are required")
 before=_certified_candidates(scene,camera,material[ids],branch[ids],station[ids],np.asarray(times)[ids])
 initial_row=np.asarray(row)[ids].copy();initial_row[before["valid"]]=before["row"][before["valid"]]
 cs=_implicit_capture(scene,camera,material[ids],branch[ids],station[ids],np.asarray(times)[ids],np.asarray(observed)[ids],iterations=16,initial_row=initial_row)
 cr=_certified_candidates(scene,camera,material[ids],branch[ids],cs,np.asarray(times)[ids]);ss,bb,rr,take=select_certified_upper_bound(station[ids],branch[ids],before,cs,branch[ids],cr,np.asarray(observed)[ids])
 oldq=np.sum((before["uv"]-np.asarray(observed)[ids])**2,axis=1);newq=np.sum((rr["uv"]-np.asarray(observed)[ids])**2,axis=1)
 comparable=before["valid"]&rr["valid"]
 if not np.all(newq[comparable]<=oldq[comparable]):raise ArithmeticError("Same-camera candidate upper bound regressed")
 S=station.copy();B=branch.copy();V=row.copy();S[ids]=ss;B[ids]=bb;V[ids[rr["valid"]]]=rr["row"][rr["valid"]]
 fixed=~ordinary
 if not (np.array_equal(S[fixed],station[fixed])and np.array_equal(B[fixed],branch[fixed])and np.array_equal(V[fixed],row[fixed])):raise AssertionError("Nonordinary auxiliary state changed")
 report=dict(schema="rinkfit.all_ordinary_warm_true_RS_local_candidates.v1",model=bound["model"],bound_camera_state_digest=bound["state_digest"],ordinary_requested=len(ids),ordinary_initially_valid=int(before["valid"].sum()),ordinary_finally_valid=int(rr["valid"].sum()),strictly_better_candidates=int(take.sum()),invalid_rows_requiring_recovery=ids[~rr["valid"]].tolist(),old_legal_candidate_preserved_per_source=True,all_other_S_branch_row_IEEE_unchanged=True,alpha_not_an_input_or_output=True,local_iterations=16,original_finite_path_domain_only=True,no_numerical_C2_bound_applied=True,station_KKT_or_global_nearest_certified=False,native_independent_gate_required=True,elapsed_seconds=time.perf_counter()-t0)
 return dict(station=S,branch=B,row=V,source_rows=ids,ordinary_result=rr,ordinary_before=before,report=report)
