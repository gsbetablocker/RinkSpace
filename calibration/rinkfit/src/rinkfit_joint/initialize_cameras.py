"""Navigation proposals only; native-source optimisation must calibrate these seeds."""
from pathlib import Path
import argparse,copy,json,hashlib
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .io_utils import read_json,write_json,file_sha256
from .camera_graph import CameraGraph,SHOT_TO_RIG,REFERENCE_SHOTS
from .role_guard import FrameRoleGuard
from rinkfit_start.native_camera import rotation
from rinkfit_start.camera import UniformCubicBasis

class AnchorSeedError(ValueError):
    """An initialization cannot preserve the supplied real navigation times."""


def spline_seed(times,values,plan,*,prior_total_controls=None,precision=(1e-10,1e-10,1e-10)):
    """Convert real navigation anchors without resampling their target values.

    The optional prior is a numerical nullspace tie-break.  Without one, a
    linear continuation of the measured endpoint secants supplies that prior;
    it is explicitly not observed halo motion or calibrated camera truth.
    """
    t=np.asarray(times,float);y=np.asarray(values,float)
    n=int(plan['n_controls']);dt=float(plan['dt']);first=float(plan['first_control_time'])
    if t.ndim!=1 or not len(t) or y.shape!=(len(t),3) or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise AnchorSeedError('finite [N] times and [N,3] anchor values required')
    order=np.argsort(t);t=t[order];y=y[order]
    if np.any(np.diff(t)<=0):raise AnchorSeedError('duplicate navigation times require explicit reconciliation')
    basis=UniformCubicBasis(first,dt,n);duration=float(plan['duration']);half=float(plan.get('tau_bound',.02))/2
    basis.require_coverage(0.,duration,2*half)
    A=np.asarray([basis.row(float(v)) for v in t])
    if prior_total_controls is None:
        knots=first+dt*np.arange(n);prior=np.column_stack([np.interp(knots,t,y[:,j]) for j in range(3)])
        if len(t)>1:
            left=knots<t[0];right=knots>t[-1]
            prior[left]=y[0]+(knots[left]-t[0])[:,None]*(y[1]-y[0])/(t[1]-t[0])
            prior[right]=y[-1]+(knots[right]-t[-1])[:,None]*(y[-1]-y[-2])/(t[-1]-t[-2])
        prior_policy='linear endpoint-secant continuation; numerical initializer only, unobserved halo remains unsupported'
    else:
        prior=np.asarray(prior_total_controls,float).copy();prior_policy='minimum coefficient change from explicit same-chart prior'
    if prior.shape!=(n,3) or not np.isfinite(prior).all():raise AnchorSeedError('prior must be finite [n_controls,3] in the same total-angle chart')
    delta,_,rank,singular=np.linalg.lstsq(A,y-A@prior,rcond=None);total=prior+delta
    error=np.max(np.abs(A@total-y),axis=0);tol=np.asarray(precision,float)
    if tol.shape!=(3,) or not np.isfinite(tol).all() or np.any(tol<=0):raise AnchorSeedError('positive arithmetic component tolerances required')
    if np.any(error>tol):raise AnchorSeedError('allocated cubic cannot preserve actual navigation anchors; no surrogate resampling or clipping allowed: '+str(error.tolist()))
    row0=basis.row(0.);ref=(row0@total)[:2];ctrl=total.copy();ctrl[:,:2]-=ref
    lower=np.broadcast_to(plan.get('lower',[-20.,-20.,np.log(100.)]),ctrl.shape)
    upper=np.broadcast_to(plan.get('upper',[20.,20.,np.log(1e6)]),ctrl.shape)
    if np.any(ctrl<lower) or np.any(ctrl>upper):raise AnchorSeedError('anchor-preserving local controls outside original coefficient bounds')
    zero=np.flatnonzero(np.max(np.abs(A),axis=0)==0).tolist()
    receipt=dict(method='actual_anchor_basis_exact_transfer',anchors=len(t),rank=int(rank),nullity=n-int(rank),
        maximum_anchor_transfer_error_components=error.tolist(),arithmetic_precision_components=tol.tolist(),
        nonzero_singular_condition=float(singular[0]/singular[int(rank)-1]) if rank else None,
        zero_anchor_support_controls=zero,unsupported_controls_must_not_be_freely_optimized=True,
        nullspace_policy=prior_policy,required_RS_interval_seconds=[-half,duration+half],
        allocated_time_domain_seconds=list(basis.domain),source_pixel_accuracy_claimed=False,
        halo_physics_verified=False,navigation_promoted_to_observations=False)
    return dict(basis=dict(dt=dt,first_control_time=first,n_controls=n),controls=ctrl.tolist(),
        reference_time=0.,angle_dependent_index=int(np.argmax(np.abs(row0))),initialization_receipt=receipt),ref


def preserve_anchors(native_pts,time_origin_pts,target_total_angles_logf,prior_total_controls,basis,lower,upper,*,time_base=(1,12800),precision=(1e-10,1e-10,1e-10),duration=None,tau_bound=.02):
    """Strict official50 integer-clock adapter used by production recovery."""
    pts=np.asarray(native_pts)
    if pts.dtype.kind not in 'iu' or pts.ndim!=1 or not len(pts):raise AnchorSeedError('integer native PTS required')
    if isinstance(time_origin_pts,bool) or not isinstance(time_origin_pts,(int,np.integer)):raise AnchorSeedError('integer shot origin PTS required')
    if tuple(time_base)!=(1,12800) or np.any((pts-int(time_origin_pts))%256):raise AnchorSeedError('verified official50 1/12800 PTS lattice required')
    times=(pts.astype(np.int64)-int(time_origin_pts))/12800.
    if duration is None:duration=float(np.max(times))
    controls,offset=spline_seed(times,target_total_angles_logf,dict(basis,duration=duration,tau_bound=tau_bound,lower=lower,upper=upper),prior_total_controls=prior_total_controls,precision=precision)
    return dict(camera_controls=controls,reference_angles=offset.tolist(),diagnostics=controls['initialization_receipt'],candidate_only=True,source_pixel_accuracy_claimed=False)

def angles_for(R,R0,n,b):
    w0=R0.T@b; out=[]
    for ri in R:
        wi=ri.T@b;aa=-np.arctan2(n@np.cross(w0,wi),w0@wi)
        bb=Rotation.from_matrix(ri@(R0@rotation(n,aa)).T).as_rotvec()@b
        fit=least_squares(lambda q:Rotation.from_matrix(ri@(rotation(b,q[1])@R0@rotation(n,q[0])).T).as_rotvec(),[aa,bb])
        out.append(fit.x)
    return np.unwrap(np.asarray(out),axis=0)

def initialise_navigation_graph(project,run,source_graph,out):
    project=Path(project);run=Path(run);out=Path(out)
    graph=copy.deepcopy(read_json(source_graph));guard=FrameRoleGuard(run/"roles/frame_roles.json")
    plans={s["shot_id"]:s for s in read_json(run/"control_plan.json")["shots"]}
    nav={};proof=[]
    for index in range(2,10):
        sid=f"s{index:02d}"; result=project/"__unused__"
        data=Path("D:/CodexData/RinkSpace_Data")/"events/ankara_2026/programmes/okada_mei_sp/results"/f"shot{index}_review_20261014"
        if index==2:data=data/"r2"
        mapping=read_json(data/"navigation_mapping.json")
        ref=mapping["model_seed"]["camera_seeds"]; cp=Path(ref["path"])
        if file_sha256(cp)!=ref["sha256"]:raise ValueError("Navigation source hash mismatch")
        archive_ref=mapping["model_seed"]["inherited_camera_archive"];archive_path=Path(archive_ref["path"])
        if file_sha256(archive_path)!=archive_ref["sha256"]:raise ValueError("Navigation archive hash mismatch")
        archive=np.load(archive_path,allow_pickle=False)
        pts=np.asarray([int(m["official_pts"]) for m in mapping["frames"]],dtype=np.int64)
        cams=np.asarray([archive["camera25"][int(m["inherited_camera_archive_row_0based"])] for m in mapping["frames"]],float)
        for j,m in enumerate(mapping["frames"]):
            if hashlib.sha256(np.ascontiguousarray(cams[j],dtype="<f8").tobytes()).hexdigest()!=m["camera25_float64_sha256"]:
                raise ValueError("Exact navigation archive row mismatch")
        guard.assert_training_pixels(pts.tolist(),shot_id=sid)
        sc=read_json(Path(mapping["model_seed"]["scene"]["path"]))
        bounds=np.asarray(sc["parameters"]["bounds"],float);scale=bounds[2]-bounds[0];origin=np.r_[(bounds[:2]+bounds[2:])/2,0.]
        if not scale>0:raise ValueError("Scene normalisation")
        nav[sid]={"pts":pts,"R":Rotation.from_rotvec(cams[:,3:6]).as_matrix(),"C":(cams[:,:3]-origin)/scale,
            "logf":np.mean(cams[:,6:8],axis=1),"cams":cams,"scale":scale}
        proof.append({"shot_id":sid,"mapping":str(data/"navigation_mapping.json"),"mapping_sha256":file_sha256(data/"navigation_mapping.json"),
            "seed":ref,"actual_navigation_archive":archive_ref,"rows":len(pts),"conditional_world_length":scale,"old_camera_values_used_as":"navigation-only initial proposal, never calibrated truth or final likelihood",
            "time_array_used":"pts (verified official50), never exposure_representative_pts/legacy60_pts"})
    for rig in ("secondary","end"):
        ref=REFERENCE_SHOTS[rig];ri=nav[ref]["R"][0];n=np.array([0.,0.,1.]);b=np.array([1.,0.,0.]);q=ri@n;b-=q*(q@b);b/=np.linalg.norm(b)
        members=[s for s in nav if SHOT_TO_RIG[s]==rig];cams=np.concatenate([nav[s]["cams"] for s in members]);centres=np.concatenate([nav[s]["C"] for s in members])
        fref=float(np.exp(np.mean(np.concatenate([nav[s]["logf"] for s in members]))))
        graph["rigs"][rig].update(status="navigation_seed_awaiting_native_source_calibration",tau_bound=.02,
            shared_head={"R0":ri.tolist(),"n":n.tolist(),"b":b.tolist(),"P":np.median(centres,axis=0).tolist(),"d":[0.,0.,0.],"a":0.,"pupil_linear":0.,"pupil_quadratic":0.},
            optics={"f_ref":fref,"radius_max":2.,"principal_point":np.median(cams[:,8:10],axis=0).tolist(),"eta":float(np.median(cams[:,6]-cams[:,7])),
                "principal_linear":[0.,0.],"k1":[0.,0.],"k2":[0.,0.],"p1":0.,"p2":0.},
            rolling_shutter={"signed_readout_seconds":0.})
    for sid,d in nav.items():
        s=graph["shots"][sid];rig=graph["rigs"][s["rig_id"]];h=rig["shared_head"]
        ab=angles_for(d["R"],np.asarray(h["R0"]),np.asarray(h["n"]),np.asarray(h["b"]))
        times=(d["pts"]-s["time_origin_pts"])/12800.;duration=(s["scope"]["last_pts"]-s["scope"]["first_pts"])/12800.
        plan={**plans[sid],"duration":duration};ctrl,delta=spline_seed(times,np.c_[ab,d["logf"]],plan)
        s.update(status="navigation_seed_awaiting_native_source_fit",camera_controls=ctrl,reference_angles=delta.tolist())
    for rig in ("secondary","end"):
        ref=REFERENCE_SHOTS[rig];s=graph["shots"][ref];da,db=s["reference_angles"];h=graph["rigs"][rig]["shared_head"]
        h["R0"]=(rotation(h["b"],db)@np.asarray(h["R0"])@rotation(h["n"],da)).tolist()
        for sid in nav:
            if graph["shots"][sid]["rig_id"]==rig:
                graph["shots"][sid]["reference_angles"]=(np.asarray(graph["shots"][sid]["reference_angles"])-[da,db]).tolist()
    graph["initialization_provenance"]={"method":"old pose navigation as unaccepted numerical seed; current native source fit required","sources":proof,
        "new_rig_optics":"independent neutral Brown starting values; source calibrated values not claimed","all_source_likelihood_unchanged":True,"formal_calibration_complete":False}
    graph["shared_scene"]["event_model_ref"]=str(run/"s01_snapshot_v2/checkpoint/model.json")
    graph["cache_generation"]=int(graph.get("cache_generation",0))+1
    check=CameraGraph(graph).validate(require_ready=True)
    graph["certificates"]={"joint_fit_complete":False,"all_source_replay_complete":False,"reason":"navigation-only proposal awaiting actual native source fit"}
    write_json(out,graph);write_json(out.with_name("navigation_seed_receipt.json"),{"graph_sha256":file_sha256(out),"check":check,"evidence":proof,"not_a_physical_calibration":True})
    return {"graph":str(out),"check":check,"calibration_complete":False}

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--project",required=True);a.add_argument("--run",required=True);a.add_argument("--graph",required=True);a.add_argument("--out",required=True)
    x=a.parse_args();print(json.dumps(initialise_navigation_graph(x.project,x.run,x.graph,x.out),ensure_ascii=False))
