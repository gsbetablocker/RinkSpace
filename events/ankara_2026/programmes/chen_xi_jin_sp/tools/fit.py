"""Joint multi-shot pixel fit with one fixed scene and installation per rig."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
from time import monotonic

for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, kron

from common import HEIGHT, JOINT, OUTPUT, frozen_scene, observation_packet, read, save, scope, source_pts, write
from ice_evidence import ICE_EVIDENCE_FIELDS, assert_ice_evidence, ice_roles
from measure import DT, FIRST, clock, load_camera
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos


def row_layer(family):
    return np.where(np.char.startswith(family,"Ad"),"ad",
             np.where(np.char.startswith(family,"board."),"board","ice"))


def statistics(errors):
    q=np.asarray(errors)
    valid=np.isfinite(q)
    out=dict(rows=len(q),invalid=int((~valid).sum()))
    if valid.any():
        good=q[valid]
        out.update(rms=float(np.sqrt(np.mean(good**2))),p50=float(np.median(good)),
                   p90=float(np.quantile(good,.9)),p95=float(np.quantile(good,.95)),
                   worst=float(good.max()))
    return out


def measurement_rows(folder, shot, record, path_roles):
    sid=shot["id"]
    with np.load(observation_packet(folder,sid),allow_pickle=False) as a:
        assert a["source_sha256"].item()==record["source"]["sha256"]
        assert a["geometry_sha256"].item()==record["geometry_binding"]["value_sha256"]
        keep=a["eligible"]&(a["fold"]==0)
        assert_ice_evidence(a,keep,path_roles)
        rows={k:a[k][keep] for k in ("pts","idx","uv","path","station","family","material")}
        n=int(keep.sum())
        rows["source_bracket"]=a["source_bracket"][keep] if "source_bracket" in a else np.full((n,2,2),np.nan)
        rows["source_side_contrast"]=a["source_side_contrast"][keep] if "source_side_contrast" in a else np.full((n,2),np.nan)
        rows["ice_measurement_role"]=a["ice_measurement_role"][keep] if "ice_measurement_role" in a else np.full(n,"")
    np.testing.assert_array_equal(rows["pts"],source_pts(shot)[rows["idx"]])
    return rows


def row_weights(pts,family,layer):
    paired=np.rec.fromarrays([pts,family])
    _, index, count = np.unique(paired,return_inverse=True,return_counts=True)
    sigma = np.where(layer=="ice",1.35,np.where(layer=="board",1.40,1.64))
    # Normalize the influence of repeated samples on a single feature group
    # in one exposure. Keep material-group Huber loss explicit.
    return 2.0/(np.sqrt(np.maximum(1.,count[index]/4.))*sigma)


def supported_basis(c, t, sampled, stride):
    template=UniformMotionChart.create(len(c),FIRST,DT,stride,t[0])
    d=continue_exposure_halos(np.repeat(template.refinement[:,None],3,axis=1),FIRST,DT,t[[0,-1]])[:,0]
    if not len(sampled):
        return csr_matrix((len(c)*3,0))
    ii,w=cubic_samples(len(c),FIRST,DT,t[np.unique(sampled)])
    hits=np.einsum("ij,ijk->ik",w,d[ii])
    use=np.count_nonzero(abs(hits)>.075,axis=0)>=2
    return kron(csr_matrix(d[:,use]),csr_matrix(np.diag([.0014,.0014,.0030])),format="csr")


def bundle(record, folder, previous, stride, path_roles, active_shots=None):
    all_rows={k:[] for k in ("pts","idx","uv","path","station","family","material","times","camera","shot",
                            *ICE_EVIDENCE_FIELDS)}
    controls, blocks, spans = [],[],{}
    rigs={}
    cursor=0
    for s in record["shots"]:
        sid=s["id"]
        pts,c,rig=load_camera(s,previous)
        r=measurement_rows(folder,s,record,path_roles)
        if int(s["camera_index"]) in rigs:
            np.testing.assert_array_equal(rigs[int(s["camera_index"])],rig)
        rigs[int(s["camera_index"])]=rig
        t=clock(pts)
        spans[sid]=(cursor,cursor+len(c))
        controls.append(c)
        block=(supported_basis(c,t,r["idx"],stride)
               if active_shots is None or sid in active_shots else csr_matrix((c.size,0)))
        blocks.append(block)
        r["times"]=t[r["idx"]]+cursor*DT-FIRST
        r["camera"]=np.full(len(r["pts"]),s["camera_index"],dtype=np.int32)
        r["shot"]=np.full(len(r["pts"]),sid)
        for k,value in r.items():
            all_rows[k].append(value)
        cursor+=len(c)
    return ({k:np.concatenate(v) for k,v in all_rows.items()},np.vstack(controls),
            np.asarray([rigs[k] for k in range(3)]),block_diag(blocks,format="csr"),spans)


def row_loss(errors, weights, layer, delta=2.7):
    e=weights*errors
    loss=np.where(e<=delta,.5*e*e,delta*(e-.5*delta))
    return {name:float(np.sum(loss[layer==name])) for name in ("ice","board","ad")}


def checkpoint(path,raw,record,spans,rigs,native,optics,exposure_clocks=None,rig_active_indices=None,
               rig_slots=None):
    if "scene_values" in raw:
        np.testing.assert_array_equal(raw["scene_values"],native.parameter_state.values)
    r=np.asarray(raw["rig"])
    active=rig_active_indices if rig_active_indices is not None else {
        i:{26,27,29,30} if optics else set() for i in range(len(rigs))}
    for k,active_indices in active.items():
        fixed_orientation=[i for i in range(15) if i not in active_indices]
        np.testing.assert_allclose(r[k,fixed_orientation],rigs[k,fixed_orientation],atol=2.e-14,rtol=0.)
        fixed=[i for i in range(15,r.shape[1]) if i not in active_indices]
        np.testing.assert_array_equal(r[k,fixed],rigs[k,fixed])
    np.testing.assert_array_equal(raw["board"],native.scene["board"])
    np.testing.assert_array_equal(raw["layout"],native.scene["shared_layout"])
    saved_rig=r if optics or rig_active_indices is not None else rigs
    arrays=dict(scene_values_sha256=record["geometry_binding"]["value_sha256"],
                layout_revision=record["geometry_binding"]["layout_revision"],native_time_base=np.array([1,12800]))
    for shot in record["shots"]:
        sid=shot["id"]
        a,b=spans[sid]
        arrays[sid+"_pts"]=source_pts(shot)
        arrays[sid+"_controls"]=raw["controls"][a:b]
        slot=shot["camera_index"] if rig_slots is None else rig_slots[sid]
        arrays[sid+"_rig"]=saved_rig[slot]
        if exposure_clocks is not None and sid in exposure_clocks:
            arrays[sid+"_exposure_times"]=exposure_clocks[sid]
    save(path,**arrays)


def view_priors(native, record, previous, spans, rows):
    """Project weak view priors from the source-navigation state, never as observed UV."""
    screen=np.array([[270.,185.],[1610.,220.],[950.,535.],[310.,886.],[1590.,915.]])
    arrays={k:[] for k in ("xyz","times","camera","weight")}
    for shot in record["shots"]:
        sid=shot["id"]
        pts,c,rig=load_camera(shot,previous)
        t=clock(pts)
        counts=np.bincount(rows["idx"][rows["shot"]==sid],minlength=len(t))
        # The two closing closeups show little beyond the board plane.
        # Preserve the measured navigation between their narrow feature bands.
        closeup=sid in ("s01","s19","s20")
        use=np.arange(len(t)) if closeup else np.flatnonzero((counts<12)|(np.arange(len(t))%2==0))
        camera=native.core.evaluate_motion_camera(rig,c,FIRST,DT,t[use])
        if np.any(camera["status"]):
            raise ValueError("View prior origin outside motion support "+sid)
        norm=(screen[None,:,:]-camera["values"][:,None,18:20])/camera["values"][:,None,16:18]
        direction=np.concatenate((norm,np.ones((*norm.shape[:2],1))),axis=2)
        yy=camera["centre"][:,None,:]+.47*np.einsum("mni,mij->mnj",direction,camera["rotation"])
        arrays["xyz"].append(yy.reshape(-1,3))
        arrays["times"].append(np.repeat(t[use]+spans[sid][0]*DT-FIRST,len(screen)))
        arrays["camera"].append(np.full(len(yy)*len(screen),shot["camera_index"],np.int32))
        weight=np.where(counts[use]>=15,.30,.75)
        if closeup:
            weight=np.maximum(weight,1.15)
        arrays["weight"].append(np.repeat(weight,len(screen)))
    result={k:np.concatenate(v) for k,v in arrays.items()}
    result["uv"]=np.zeros((len(result["times"]),2))
    return result


def resident_problem(native, rows, controls, rigs, prior=None):
    motion=dict(first_control_time=0.,control_dt=DT,image_height=HEIGHT,reference_pts=0,
                time_base_num=1,time_base_den=12800)
    if prior is None:
        return native.resident(camera=motion,controls=controls,rigs=rigs,camera_ids=rows["camera"],
            path_ids=rows["path"],stations=rows["station"],pts=rows["pts"],
            observed=rows["uv"],motion_times=rows["times"])
    paths=list(dict.fromkeys(rows["path"]))
    code={name:i for i,name in enumerate(paths)}
    definitions=[native.paths[k] for k in paths]
    definitions.append(dict(kind=4,vertices=prior["xyz"],parameters=[]))
    n=len(prior["xyz"])
    resident=ResidentResidualProblem(native.core,board=native.scene["board"],
        layout=native.scene["shared_layout"],options=native.scene["shared_geometry_options"],
        rig=rigs,controls=controls,first_control_time=0.,control_dt=DT,image_height=HEIGHT,
        codes=np.r_[[code[k] for k in rows["path"]],np.full(n,len(paths))].astype(np.int32),
        stations=np.r_[rows["station"],np.linspace(0.,1.,n)],
        times=np.r_[rows["times"],prior["times"]],
        observed=np.vstack([rows["uv"],prior["uv"]]),
        camera_ids=np.r_[rows["camera"],prior["camera"]])
    try:
        resident.configure_geometry(definitions)
    except BaseException:
        resident.close()
        raise
    return resident


def configure(problem, rows, motion, rigs, optical_fields, prior):
    half=float(np.max(abs(rigs[:,37]))*.64 + 1.e-6)
    if optical_fields:
        from solve_joint import rig_columns
        tags=list(optical_fields)
        q=motion.shape[1]+sum(map(len,optical_fields.values()))
        directions,lo,hi=rig_columns(rigs,tags,q,motion.shape[1],optical_fields)
        problem.configure_camera_parameters(directions=directions,max_readout=2.*half,lower=lo,upper=hi)
        motion=hstack((motion,csr_matrix((motion.shape[0],q-motion.shape[1]))),format="csr")
    else:
        problem.configure_projection_time_support(half_seconds=half)
    support=problem._path_supports[problem._codes[:len(rows['pts'])]]
    stations=rows["station"]
    assert np.all(stations>=support[:,0]-1.e-10) and np.all(stations<=support[:,1]+1.e-10)
    window=np.where(np.char.startswith(rows["path"],"board/continuous/"),.018,.13)
    lower=np.maximum(stations-window,support[:,0])
    upper=np.minimum(stations+window,support[:,1])
    layer=row_layer(rows["family"])
    weights=row_weights(rows["pts"],rows["family"],layer)
    weights*=np.where(np.char.startswith(rows["material"],"dense_"),.70,1.)
    weights*=np.where((rows["shot"]=="s01")&(rows["material"]=="direct_purple_panel_boundary"),1.65,1.)
    pw=prior["weight"] if prior is not None else np.empty(0)
    weights_all=np.r_[weights,pw]
    white=np.zeros((len(weights_all),4))
    white[:,0]=white[:,3]=weights_all
    ids=np.arange(len(weights_all),dtype=np.int32)
    local_ids=np.r_[np.arange(len(stations),dtype=np.int32),np.full(len(pw),-1,dtype=np.int32)]
    plan=problem.configure_linearization(directions=motion,local_ids=local_ids,
        local_scales=np.where(layer=="board",.003,.009),
        group_offsets=np.arange(len(ids)+1,dtype=np.int32),members=ids,
        ranks=np.full(len(ids),2,np.int32),weight_offsets=np.arange(len(ids)+1,dtype=np.int32)*4,
        weights=white.ravel(),huber_delta=2.7)
    solver=problem.configure_solver(lower=lower,upper=upper,device_budget_bytes=7*1024**3,
         host_budget_bytes=12*1024**3,initial_lambda_relative=.0008,damping_metric="jacobi_image")
    return layer,weights,plan,solver


def trial_fit(folder, observations, previous=None, *, stride=6, optics=False, view_guard=False, active_shots=None):
    sys.path.insert(0,str(JOINT))
    from solve_joint import Journal, OPTICAL_FIELDS, finite_json

    folder.mkdir(exist_ok=True)
    journal=Journal(folder/"solve.jsonl")
    stage=folder.name
    record=scope()
    journal("start",variant=stage,device="project native CUDA / cuDSS",
            observations=str(observations),previous=str(previous) if previous else None,
            camera_installations=len({s["camera_index"] for s in record["shots"]}),frozen_scene=True,
            required_shots=len(record["shots"]),active_motion_shots=active_shots,
            control_stride=stride)
    optical_fields={k:OPTICAL_FIELDS for k in ("main","secondary","end")} if optics else {}
    before_clock=monotonic()
    try:
        native=frozen_scene(record)
        if active_shots and set(active_shots)-set(record["required_shot_ids"]):
            raise ValueError("Unknown motion shot")
        rows, controls, rigs, motion, spans=bundle(record,observations,previous,stride,ice_roles(native),active_shots)
        original_rows=len(rows["pts"])
        out=folder/"camera.npz"
        if not original_rows or not motion.shape[1]:
            raise ValueError("No usable current source factors")
        prior=view_priors(native,record,previous,spans,rows) if view_guard else None
        with resident_problem(native,rows,controls,rigs,prior) as initial:
            initial.configure_projection_time_support(half_seconds=float(np.max(abs(rigs[:,37]))*.64+1.e-6))
            raw=initial.evaluate("cuda")
        valid=(raw["status"][:original_rows]==0)
        if prior is not None:
            safe=(raw["status"][original_rows:]==0)
            prior={k:v[safe] for k,v in prior.items()}
            prior["uv"]=raw["values"][original_rows:,3:5][safe]
            journal("view_priors",variant=stage,rows=len(prior["uv"]),
                    excluded=int(np.count_nonzero(~safe)),measured_uv=False,
                    role="weak_source_navigation_motion_prior")
        prior_count=len(prior["uv"]) if prior is not None else 0
        if not np.all(valid):
            ix=np.flatnonzero(~valid)
            save(folder/"unprojectable_input_rows.npz",**{k:v[ix] for k,v in rows.items()},status=raw["status"][ix])
            journal("domain_failure",variant=stage,rows=original_rows,
                    failures=Counter(map(int,raw["status"][ix])),record=str(folder/"unprojectable_input_rows.npz"))
            rows={k:v[valid] for k,v in rows.items()}

        with resident_problem(native,rows,controls,rigs,prior) as problem:
            layer,weights,plan,solver=configure(problem,rows,motion,rigs,optical_fields,prior)
            real=len(rows["pts"])
            def errors():
                return problem.evaluate_errors()[:real]
            e0=errors()
            if not np.isfinite(e0).all():
                raise ValueError("Resident start has unresolved nonfinite rows")
            parts=row_loss(e0,weights,layer)
            journal("prepared",variant=stage,rows=len(e0),retained_columns=motion.shape[1],plan=plan,gpu=solver,
                    domain_failures=original_rows-len(e0),objective_by_layer=parts,checkpoint=str(out),optical_fields=optical_fields,
                    fit_by_shot=dict(Counter(rows["shot"])),setup_seconds=monotonic()-before_clock)
            checkpoint(out,problem.export_state(),record,spans,rigs,native,optics)
            small=0
            accepted=0

            def update(detail):
                nonlocal small,accepted,parts
                summary=detail["summary"]
                if summary["accepted"]:
                    accepted+=1
                    checkpoint(out,problem.export_state(),record,spans,rigs,native,optics)
                    parts=row_loss(errors(),weights,layer)
                    gain=abs(summary["actual_decrease"])/max(1.,abs(summary["old_cost"]))
                    small=small+1 if gain<.000025 and accepted>=4 else 0
                journal("trial",variant=stage,rows=len(e0),retained_columns=motion.shape[1],
                        objective_by_layer=parts,checkpoint=str(out),**detail)
                return small<2 and not (folder/"stop").exists()

            final=problem.solve(relative_cost_tolerance=8.e-6,step_tolerance=8.e-6,progress=update,
                       numerical_progress=lambda r:journal("factorization_retry",variant=stage,**r))
            state=problem.export_state()
            checkpoint(out,state,record,spans,rigs,native,optics)
            e1=errors()
            part={sid:dict(before=statistics(e0[rows["shot"]==sid]),after=statistics(e1[rows["shot"]==sid]))
                    for sid in record["required_shot_ids"]}
            save(folder/"active_rows.npz",**rows,station_fitted=state["stations"][:real],
                 before_px=e0,after_px=e1,source_sha256=record["source"]["sha256"],
                 geometry_sha256=record["geometry_binding"]["value_sha256"])
            report=dict(stage=stage,original_rows=original_rows,active_rows=len(e1),
                source_sha256=record["source"]["sha256"],
                geometry_sha256=record["geometry_binding"]["value_sha256"],
                active_by_layer=dict(Counter(layer)),accepted_trials=accepted,optical_fields=optical_fields,
                active_motion_shots=active_shots,view_prior_count=prior_count,solver=finite_json(final),
                before=statistics(e0),after=statistics(e1),by_shot=part,
                seconds=monotonic()-before_clock,camera_checkpoint=str(out),measurement_directory=str(observations))
            write(folder/"fit.json",report)
            journal("finish",variant=stage,rows=len(e1),reason=final.get("stop_reason"),
                    objective_by_layer=parts,checkpoint=str(out),result=str(folder/"fit.json"))
            return report
    except BaseException as error:
        journal("failure",variant=stage,reason=str(error))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    p.add_argument("--observations",required=True)
    p.add_argument("--previous")
    p.add_argument("--stride",type=int,default=6)
    p.add_argument("--optics",action="store_true")
    p.add_argument("--view-guard",action="store_true")
    p.add_argument("--active-shots",nargs="+")
    a=p.parse_args()
    trial_fit(OUTPUT/a.out,OUTPUT/a.observations,OUTPUT/a.previous if a.previous else None,
              stride=a.stride,optics=a.optics,view_guard=a.view_guard,active_shots=a.active_shots)


if __name__=="__main__":
    main()
