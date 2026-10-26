"""Prepare the existing G1 residual problem for correlation-aware G2.

This module compiles identities/weights/activity. Native owns all numerical
linearization and reduction. It does not infer new geometry or measurements.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .dataset import validate_current_noise_groups

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def prepare_g2(g1_dir, *, motion_activity="level"):
    if motion_activity not in ("level","affine"):
        raise ValueError("unsupported motion activity")
    root=Path(g1_dir);problem=read(root/"problem.json")
    z=dict(np.load(root/problem["arrays_ref"],allow_pickle=False))
    camera=read(root/problem["camera_ref"])
    scene=read(root/problem["scene_ref"])
    controls=np.load(root/camera["controls_ref"],allow_pickle=False)
    source_groups=read(root/"noise_groups.json")
    # G1 already validated these current references; only W is newly consumed.
    ref=source_groups["weights_source"];wp=Path(problem["prepared_path"])/ref["path"]
    if hashlib.sha256(wp.read_bytes()).hexdigest()!=ref["sha256"]:raise ValueError("original W changed")
    original_weights=np.load(wp,allow_pickle=False)
    row_map={int(raw):i for i,raw in enumerate(z["rows"])}
    offsets=[0];weights_offsets=[0];members=[];weights=[];ranks=[];records=[]
    indices=np.full(len(z["rows"]),-1,np.int32)
    current_groups=[]
    for record in source_groups["groups"]:
        old=record["group"];rr=record["original_member_rows"]
        if rr!=record["selected_member_rows"] or not all(int(r) in row_map for r in rr):
            raise ValueError("partial correlation group is not eligible for G2: "+old["id"])
        mapped=np.array([row_map[int(r)] for r in rr],dtype=np.int64)
        w=original_weights[old["weight_start"]:old["weight_start"]+old["rank"]*2*len(rr)].copy()
        if hashlib.sha256(w.astype("<f8").tobytes()).hexdigest()!=old["matrix_sha256"]:
            raise ValueError("W group identity mismatch")
        group={**old,"member_start":len(members),"weight_start":len(weights)}
        current_groups.append(group)
        indices[mapped]=len(current_groups)-1
        members.extend(mapped.tolist());weights.extend(w.tolist());ranks.append(old["rank"])
        offsets.append(len(members));weights_offsets.append(len(weights))
        records.append({"id":old["id"],"origin":"original_correlated_group","source":record,
                        "rows":mapped.tolist(),"rank":old["rank"],"matrix_sha256":old["matrix_sha256"]})
    # Reuse the current group binding validator before adding the true independent rows.
    validate_current_noise_groups(current_groups,np.asarray(members,dtype="<i8"),
        np.asarray(weights,dtype="<f8"),{"pts":z["pts"],"split":z["split"],"noise_group_index":indices})
    for raw in source_groups["independent_rows"]:
        row=row_map[int(raw)]
        if indices[row]!=-1:raise ValueError("independent row overlaps a correlated group")
        xx,xy,yy=z["covariance"][row]
        covariance=np.array([[xx,xy],[xy,yy]])
        # Inverse Cholesky is a square-root precision with A C A^T = I.
        w=np.linalg.solve(np.linalg.cholesky(covariance),np.eye(2)).astype("<f8")
        members.append(row);weights.extend(w.ravel().tolist());ranks.append(2)
        offsets.append(len(members));weights_offsets.append(len(weights))
        records.append({"id":"independent/"+z["observation_ids"][row].decode(),"origin":"original_independent_covariance",
                        "rows":[row],"rank":2,"covariance":covariance.tolist(),
                        "matrix_sha256":hashlib.sha256(w.tobytes()).hexdigest()})
    if sorted(members)!=list(range(len(z["rows"]))):raise ValueError("noise partition is not complete and disjoint")
    # Keep original local-variable identity; fixed observations do not create station.
    local_ids=np.full(len(z["rows"]),-1,np.int32)
    old_locals=np.unique(z["local_index"][z["kind"]==1])
    if np.any(old_locals<0) or np.any(z["local_index"][z["kind"]==0]!=-1):
        raise ValueError("invalid existing station/fixed-point binding")
    remap={int(v):i for i,v in enumerate(old_locals)}
    for row in np.flatnonzero(z["kind"]==1):local_ids[row]=remap[int(z["local_index"][row])]
    # Six slow window directions in the existing control grid: level and slope
    # of pan/tilt/logf. This retains all 105 old coefficients and knot times.
    lo,hi=float(z["times"].min()),float(z["times"].max())
    centre=(lo+hi)/2;half=(hi-lo)/2
    if not half>0:raise ValueError("affine window activity requires multiple source times")
    knot=camera["first_control_time"]+np.arange(len(controls))*camera["control_dt"]
    directions=np.zeros((len(controls),3,6),np.float64)
    names=("pan","tilt","log_f")
    parameters=[]
    for ch,name in enumerate(names):
        directions[:,ch,ch]=.001
        directions[:,ch,ch+3]=.001*(knot-centre)/half
        parameters.append({"column":ch,"id":"motion/"+name+"/level","physical_scale":.001,
            "unit":"radian" if ch<2 else "log_ratio","control_basis":"constant","initial":0.})
    for ch,name in enumerate(names):
        parameters.append({"column":ch+3,"id":"motion/"+name+"/slope","physical_scale":.001,
            "unit":"radian" if ch<2 else "log_ratio","control_basis":"(control_time-window_centre)/window_half_duration","initial":0.})
    tested_parameters=parameters
    if motion_activity=="level":
        directions=directions[:,:,:3].copy();parameters=parameters[:3]
    return dict(problem=problem,arrays=z,scene=scene,camera=camera,controls=controls,
        directions=directions,local_ids=local_ids,local_scales=np.full(len(old_locals),.001),
        group_offsets=np.array(offsets,np.int32),members=np.array(members,np.int32),
        ranks=np.array(ranks,np.int32),weight_offsets=np.array(weights_offsets,np.int32),
        weights=np.array(weights,np.float64),group_records=records,
        activity={"parameters":parameters,"original_local_indices":old_locals.tolist(),
            "local_scale":.001,"local_coordinate":"original finite station in [0,1]",
            "world_gauge":"current native L=1; original axes/origin; geometry held fixed",
            "camera_gauge":"reference pose/axes/lever/optics/readout frozen; original reference control lies outside this window",
            "original_control_indices":camera["snapshot_origin"]["control_indices"],
            "window_centre_relative_seconds":centre,"window_half_duration_seconds":half,
            "activity_scope":motion_activity+" window directions on existing controls; no new knots or per-exposure free poses",
            "frozen_motion_directions":tested_parameters[3:] if motion_activity=="level" else [],
            "frozen":["motion slopes in level activity","all shared geometry","all advertisements","rig installation","Brown optics coefficients","readout time"],
            "loss":{"family":"rank_normalized_huber","delta":3.0,"threshold_unit":"whitened group RMS",
                    "reason":"explicit G2 linearization setting; not a fitted tolerance or geometry acceptance"},
            "motion_prior":{"enabled":False,"reason":"G2 audits data terms only; no uncalibrated prior weight introduced"},
            "parameter_update_executed":False})

def load_linearization_problem(g1_dir, g2_dir):
    """Consume G2's prepared numeric plan; no historical producer/weight rebuild."""
    root=Path(g1_dir);linear=Path(g2_dir)
    problem=read(root/"problem.json");camera=read(root/problem["camera_ref"])
    p=dict(problem=problem,camera=camera,scene=read(root/problem["scene_ref"]),
        arrays=dict(np.load(root/problem["arrays_ref"],allow_pickle=False)),
        controls=np.load(root/camera["controls_ref"],allow_pickle=False),
        activity=read(linear/"activity.json"),group_records=read(linear/"noise_groups.json"))
    p.update(dict(np.load(linear/"plan.npz",allow_pickle=False)))
    nl=len(p["local_scales"]);lo=np.full(nl,-np.inf);hi=np.full(nl,np.inf)
    for row,local in enumerate(p["local_ids"]):
        if local<0:continue
        support=p["arrays"]["support"][row]
        lo[local]=max(lo[local],support[0]);hi[local]=min(hi[local],support[1])
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo>hi):
        raise ValueError("inconsistent original finite station supports")
    p["lower"]=lo;p["upper"]=hi
    return p


def compile_navigation_problem(prepared_dir, navigation_local_state, *, shot_id, core,
                               camera_state=None, activity='supported_controls'):
    """Compile one complete shot using the common frozen native GPU implementation."""
    from .dataset import load_prepared, relative
    from .gpu_geometry import NativeSceneGPU, bind_fixed_junctions
    root=Path(prepared_dir)
    manifest,arrays=load_prepared(root,purpose='camera_navigation',core=core)
    shots=manifest['shots'];shot_index=next(i for i,s in enumerate(shots) if s['id']==shot_id)
    shot=shots[shot_index]
    if camera_state is None:
        path=relative(root,shot['camera_ref']);camera=read(path)
        controls=np.load(relative(path.parent,camera['controls_ref']),allow_pickle=False)
    else:camera,controls=camera_state['camera'],np.asarray(camera_state['controls'],float)
    selected=np.flatnonzero((arrays['shot_index']==shot_index)&(arrays['active']!=0)&(arrays['split']==0))
    if not len(selected):raise ValueError('No qualified source measurements for '+shot_id)
    rowmap={int(old):i for i,old in enumerate(selected)}
    offsets=[0];weight_offsets=[0];members=[];ranks=[];weights=[];groups=[]
    all_members=arrays['noise_group_members'];all_weights=arrays['noise_group_weights']
    for g in manifest['noise_groups']:
        rr=all_members[g['member_start']:g['member_start']+g['member_count']]
        take=[int(r) in rowmap for r in rr]
        if not any(take):continue
        if not all(take):raise ValueError('Navigation selection truncates correlation group '+g['id'])
        if 'allowed_use' in g and 'camera_navigation' not in g['allowed_use']:raise ValueError('W not authorized for camera navigation')
        w=all_weights[g['weight_start']:g['weight_start']+g['rank']*2*len(rr)]
        if hashlib.sha256(np.asarray(w,dtype='<f8').tobytes()).hexdigest()!=g['matrix_sha256']:raise ValueError('Original W changed')
        members.extend(rowmap[int(r)] for r in rr);weights.extend(w);ranks.append(g['rank'])
        offsets.append(len(members));weight_offsets.append(len(weights));groups.append(g)
    for raw in selected[arrays['noise_group_index'][selected]<0]:
        xx,xy,yy=arrays['covariance'][raw];w=np.linalg.solve(np.linalg.cholesky([[xx,xy],[xy,yy]]),np.eye(2))
        members.append(rowmap[int(raw)]);weights.extend(w.ravel());ranks.append(2)
        offsets.append(len(members));weight_offsets.append(len(weights))
    if sorted(members)!=list(range(len(selected))):raise ValueError('Incomplete noise partition')
    features=manifest['features'];scene=NativeSceneGPU(relative(root,manifest['scene_ref']),core)
    bind_fixed_junctions(scene,features)
    states=navigation_local_state if isinstance(navigation_local_state,dict) else read(navigation_local_state)
    lookup={r['observation_id']:r for r in states['rows']}
    if len(lookup)!=len(states['rows']):raise ValueError('Duplicate navigation local state identity')
    paths=[];stations=[];locals_=[];localmap={};lower=[];upper=[];local_origin=[]
    # An isolated normal residual has only one observed direction. Giving its
    # station another free direction would annihilate that same camera factor.
    # Preserve its real initialized station as a fixed search coordinate; only
    # groups with extra observed rank may jointly refine stations.
    fixed_normal_rows=set()
    for g in groups:
        rr=all_members[g['member_start']:g['member_start']+g['member_count']]
        if g['rank']<=len(rr):fixed_normal_rows.update(map(int,rr))
    for raw in selected:
        f=features[int(arrays['feature_index'][raw])];path_id=f.get('path_id',f['id'])
        if f['kind']=='fixed_junction':path_id=f['id']
        paths.append(path_id)
        oid=arrays['observation_ids'][raw].decode()
        if f['kind']!='finite_curve':
            if int(arrays['local_index'][raw])!=-1:raise ValueError('Fixed point acquired station')
            stations.append(0.);locals_.append(-1);continue
        state=lookup.get(oid)
        if state is None or state['path_id']!=path_id or not np.isfinite(state['value']):
            raise ValueError('Missing or incompatible real navigation station for '+oid)
        value=float(state['value']);key=state['local_id']
        if int(raw) in fixed_normal_rows:
            stations.append(value);locals_.append(-1);continue
        lo,hi=(-np.inf,np.inf) if arrays['support_kind'][raw]==1 else arrays['support'][raw]
        if not lo<=value<=hi:raise ValueError('Initial station is outside real path support')
        if key not in localmap:
            localmap[key]=len(localmap);lower.append(lo);upper.append(hi);local_origin.append(state)
        local=localmap[key]
        if local_origin[local]['path_id']!=path_id or local_origin[local]['value']!=value:raise ValueError('Shared station identity disagrees')
        lower[local]=max(lower[local],lo);upper[local]=min(upper[local],hi)
        stations.append(value);locals_.append(local)
    native_pts=arrays['pts'][selected]
    times=(native_pts-int(camera['reference_pts']))*camera['time_base_num']/camera['time_base_den']
    if activity=='level':
        directions=np.zeros((len(controls),3,3));
        for j in range(3):directions[:,j,j]=.001
        active_controls=list(range(len(controls)))
    elif activity=='supported_controls':
        # Native cubic support is cell-1 through cell+2. Include signed readout
        # margins; controls with insufficient real source factors remain frozen.
        dt=camera['control_dt'];first=camera['first_control_time'];readout=abs(camera['rig'][37])
        cells=np.floor((times-first)/dt).astype(int)
        active_controls=[k for k in range(len(controls)) if np.count_nonzero((cells>=k-2)&(cells<=k+1))>=3]
        if not active_controls:raise ValueError('No supported camera control directions')
        directions=np.zeros((len(controls),3,len(active_controls)*3))
        for j,k in enumerate(active_controls):directions[k,:,3*j:3*j+3]=np.eye(3)*.001
    else:raise ValueError('Unknown camera activity')
    resident=scene.resident(camera=camera,controls=controls,path_ids=paths,stations=np.asarray(stations),pts=native_pts,observed=arrays['uv'][selected])
    try:
        plan=resident.configure_linearization(directions=directions,local_ids=np.asarray(locals_,np.int32),
            local_scales=np.full(len(localmap),.001),group_offsets=np.asarray(offsets,np.int32),members=np.asarray(members,np.int32),
            ranks=np.asarray(ranks,np.int32),weight_offsets=np.asarray(weight_offsets,np.int32),weights=np.asarray(weights),huber_delta=3.)
        # For the common engineering normal-only input, no nuisance station
        # is active. Exact weighted native Jacobians expose any unobservable
        # camera combinations before the LM solver is attached.
        observability=dict(method='native_weighted_camera_J',directions_before=directions.shape[2])
        if not localmap:
            resident.linearize('cuda')
            if hasattr(resident,'sparsity'):
                blocks=resident.export_sparse_blocks('cuda')
                h=np.zeros((directions.shape[2],directions.shape[2]))
                for row,(begin,end) in enumerate(zip(blocks['csr_offsets'][:-1],blocks['csr_offsets'][1:])):
                    columns=blocks['csr_columns'][begin:end];values=blocks['shared_values'][begin:end]
                    h[row,columns]=values;h[columns,row]=values
            else:h=resident.export_blocks('cuda')['shared_H']
            eigen,vectors=np.linalg.eigh(h)
            tolerance=np.finfo(float).eps*max(h.shape)*max(float(eigen[-1]),1.)*64
            keep=eigen>tolerance
            observability.update(retained_rank=int(keep.sum()),squared_singular_values=eigen.tolist(),
                numerical_rank_tolerance=tolerance,station_activity='normal_only_stations_frozen_no_false_nuisance_fit')
            if not keep.any():raise ValueError('No observable camera direction in actual source normals')
            if not keep.all():
                directions=np.einsum('ncq,qr->ncr',directions,vectors[:,keep])
                resident.close()
                resident=scene.resident(camera=camera,controls=controls,path_ids=paths,stations=np.asarray(stations),
                    pts=native_pts,observed=arrays['uv'][selected])
                plan=resident.configure_linearization(directions=directions,local_ids=np.asarray(locals_,np.int32),
                    local_scales=[],group_offsets=np.asarray(offsets,np.int32),members=np.asarray(members,np.int32),
                    ranks=np.asarray(ranks,np.int32),weight_offsets=np.asarray(weight_offsets,np.int32),weights=np.asarray(weights),huber_delta=3.)
        else:
            observability.update(station_activity='joint_native_station_groups',rank_requires_native_local_elimination=True)
        setup=resident.configure_solver(lower=lower,upper=upper,device_budget_bytes=8*1024**3,host_budget_bytes=8*1024**3)
    except BaseException:resident.close();raise
    return dict(resident=resident,scene=scene,camera=camera,controls=controls,source_rows=selected,
        observation_ids=[arrays['observation_ids'][r].decode() for r in selected],local_origins=local_origin,
        local_ids=np.asarray(locals_),path_ids=paths,groups=groups,plan=plan,setup=setup,
        active_control_indices=active_controls,activity=activity,observability=observability,source_UV_unchanged=True,
        frozen_normal_station_rows=len(fixed_normal_rows),
        frozen=['native_scene','print_placement','print_internal_shape','rig_installation','Brown_optics','readout'])
