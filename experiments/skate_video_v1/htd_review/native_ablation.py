"""Native root-translation ablation on real source exposures."""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron, vstack
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .baseline_audit import metric
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from experiments.skate_video_v1.body_skate_fit import Assembly
from experiments.skate_video_v1.inputs import atomic_json, atomic_npz, digest, emit
from experiments.skate_video_v1.programme_motion_followup import bind


def interval(x, lo, hi):
    return (x >= lo - 1e-8) & (x <= hi + 1e-8)


def diff(n, order, step):
    from math import comb
    return diags([(-1)**(order-i)*comb(order,i)/step**order for i in range(order+1)],
                 np.arange(order+1), shape=(n-order,n), format="csr")


def huber(z, cutoff):
    a=np.abs(z)
    return np.where(a<=cutoff,.5*z*z,cutoff*(a-.5*cutoff))


def projection_shape(camera, world, frames, B, ratio, active, jacobian):
    r=moving_projection(camera,world,frames,jacobian=jacobian)
    uv=r["uv"][active]
    if not jacobian:
        return uv
    n,k=active
    blocks=r["pixel_jac_nodes"][active].reshape(len(n)*2,15)/ratio
    offsets=n[:,None]*15+np.arange(15)
    cols=np.repeat(offsets,2,axis=0).ravel()
    rows=np.repeat(np.arange(len(n)*2),15)
    jac=coo_matrix((blocks.ravel(),(rows,cols)),shape=(len(n)*2,len(frames)*15)).tocsr()@B
    return uv,jac


class PixelEvidence:
    def __init__(self, camera, frames, basis, ratio, world, selected, uv, sigma, normal=None, label=""):
        self.camera,self.frames,self.basis,self.ratio=camera,frames,basis,ratio
        self.world,self.selected,self.uv=world,selected,uv
        self.sigma=np.broadcast_to(sigma,(len(uv),))
        self.normal,self.label=normal,label
        self.chain=kron(basis,eye(3,format="csr"),format="csr")
        if normal is not None:
            self.reducer=csr_matrix((normal.ravel(),(np.repeat(np.arange(len(normal)),2),np.arange(2*len(normal)))),
                                    shape=(len(normal),2*len(normal)))

    def evaluate(self, x, jacobian):
        shift=(self.basis@x).reshape(len(self.frames),5,3)
        world=self.world+shift[:,:,None]/self.ratio
        out=projection_shape(self.camera,world,self.frames,self.chain,self.ratio,self.selected,jacobian)
        uv=out[0] if jacobian else out
        d=(uv-self.uv)
        if self.normal is not None:
            raw=np.einsum("ni,ni->n",d,self.normal)
            residual=raw/self.sigma
            J=diags(1/self.sigma)@self.reducer@out[1] if jacobian else None
        else:
            residual=(d/self.sigma[:,None]).ravel()
            J=diags(np.repeat(1/self.sigma,2))@out[1] if jacobian else None
        return residual,J


class LinearEvidence:
    def __init__(self, label, matrix, offset, cutoff, unilateral=False):
        self.label,self.matrix,self.offset=label,matrix,offset
        self.cutoff,self.unilateral=cutoff,unilateral

    def evaluate(self, x, jacobian):
        r=self.matrix@x.ravel()+self.offset
        J=self.matrix if jacobian else None
        if self.unilateral:
            if jacobian:
                J=diags((r<0).astype(float))@J
            r=np.minimum(0,r)
        return r,J


def support_jac(matrix, vector):
    """Map three coordinates of each knot to one directed residual per row."""
    return vstack([diags(vector[:,i])@matrix for i in range(3)],format="csr")[
        np.arange(matrix.shape[0]*3).reshape(3,-1).T.ravel()].tocoo()


def dot_rows(matrix, direction):
    m=support_jac(matrix,direction)
    return coo_matrix((m.data,(m.row//3,m.col*3+m.row%3)),
                      shape=(matrix.shape[0],3*matrix.shape[1])).tocsr()


def node_data(ref, camera, frames, assembly=None):
    nt=ref.times[frames,None]+camera.readout_times(frames)[:,None]*np.linspace(-.5,.5,5)
    output=[]
    for sub in np.array_split(np.arange(len(frames)),max(1,int(np.ceil(len(frames)/48)))):
        t=nt[sub].ravel()
        s=ref.at(t)
        if assembly is None:
            local=ref.body.forward_rotations(s["local_R"],hand_local_R=s.get("hand_local_R"))[0]
            points=s["P"][:,None]+np.einsum("nij,nkj->nki",s["Q"],local)
        else:
            state=fk_state(ref,s,np.zeros((len(t),69)))
            points=np.stack([assembly.points(state,f,transforms=(state["R"],state["O"])) for f in range(2)],axis=1)
        output.append(points.reshape(len(sub),5,*points.shape[1:]))
    return nt,np.concatenate(output)


def source_shoe_pixels(ref, camera, assembly, frames, basis, data, guard):
    role=data["role"][frames]
    visit=np.flatnonzero((role==1).any(1) & ~guard)
    subframes=frames[visit]
    nt,world=node_data(ref,camera,subframes,assembly)
    npnts=world.shape[3]
    projected=moving_projection(camera,world.reshape(len(visit),5,2*npnts,3),subframes)["uv"]
    uv=projected.reshape(len(visit),2,npnts,2)
    selectors=[]
    for sub,f in np.argwhere(role[visit]==1):
        g=frames[visit[sub]]
        outline=data["region"][g,f]
        if not np.isfinite(outline).all():
            continue
        ids,frac,target,normal=assembly.associations(uv[sub,f],outline)
        weight=max(float(data["sigma_px"][g,f]),2.5)*np.sqrt(len(frac)/4)
        selectors.append((sub,f,ids,frac,target,normal,weight))
    if not selectors:
        return None
    size=len(selectors[0][3])
    points=np.broadcast_to(world[:,:,:,:1],(len(world),5,2,size,3)).copy()
    used=[];targets=[];normals=[];sigmas=[]
    for sub,f,idx,frac,goal,normal,weight in selectors:
        v=world[sub,:,f]
        points[sub,:,f]=v[:,idx[:,0]]*(1-frac[None,:,None])+v[:,idx[:,1]]*frac[None,:,None]
        used.extend((sub,f*size+j) for j in range(size))
        targets.extend(goal)
        normals.extend(normal)
        sigmas.extend([weight]*size)
    B=basis(nt.ravel())
    return PixelEvidence(camera,subframes,B,ref.binding.ratio,points.reshape(len(visit),5,2*size,3),
                         tuple(np.array(used).T),np.array(targets),np.array(sigmas),np.array(normals),"whole_skate")


def outline_distance(assembly, pixels, observed):
    boundary=np.concatenate(assembly.raster(pixels))
    return float(.55*np.median(cKDTree(boundary).query(observed)[0])
                 +.45*np.median(cKDTree(observed).query(boundary)[0]))


def source_shoe_audit(ref,camera,assembly,frames,basis,data,curves):
    xy=data["region"][frames]
    valid=(data["role"][frames]>0)&np.isfinite(xy).all(axis=(2,3))
    values={label:np.full(valid.shape,np.nan) for label in curves}
    visit=np.flatnonzero(valid.any(1))
    for lo in range(0,len(visit),40):
        take=visit[lo:lo+40]
        sub=frames[take]
        nt,points=node_data(ref,camera,sub,assembly)
        b=basis(nt.ravel())
        for label,x in curves.items():
            move=(b@x).reshape(len(take),5,1,1,3)/ref.binding.ratio
            uv=moving_projection(camera,(points+move).reshape(len(take),5,-1,3),sub)["uv"]
            uv=uv.reshape(len(take),2,points.shape[3],2)
            for i,f in np.argwhere(valid[take]):
                j=take[i]
                values[label][j,f]=outline_distance(assembly,uv[i,f],xy[j,f])
    return values


def nonlinear_fit(pixel_terms, linear_terms, dims, label):
    start=perf_counter()
    x=np.zeros(dims)
    damping=.3
    factors=[(term,2.0) for term in pixel_terms]+[(term,term.cutoff) for term in linear_terms]
    def eval(x,want_jac):
        errors={}
        vectors=[]
        matrices=[]
        for term,cut in factors:
            r,J=term.evaluate(x,want_jac)
            errors[term.label]=float(huber(r,cut).sum())
            if want_jac:
                wt=np.sqrt(np.minimum(1.,cut/np.maximum(1e-10,np.abs(r))))
                vectors.append(r*wt)
                matrices.append(diags(wt)@J)
        return sum(errors.values()), errors, np.concatenate(vectors) if want_jac else None, vstack(matrices,format="csr") if want_jac else None

    for step in range(9):
        total,parts,r,J=eval(x,True)
        G=(J.T@J).tocsc()
        u=J.T@r
        emit("linearized",method=label,step=step,current=total,components=parts,jacobian_shape=J.shape,
             normal_nnz=G.nnz,elapsed_s=perf_counter()-start)
        accepted=False
        for trial in range(5):
            dx=spsolve(G+diags(damping*np.maximum(1.,G.diagonal())), -u).reshape(dims)
            tested,cost,_,_=eval(x+dx,False)
            accepted=np.isfinite(tested) and tested<total
            emit("trial",method=label,step=step,trial=trial,objective=tested,components=cost,damping=damping,
                 max_control_step_m=float(np.linalg.norm(dx,axis=1).max()),accepted=bool(accepted),
                 elapsed_s=perf_counter()-start)
            if accepted:
                x+=dx
                damping*=.36
                break
            damping*=5.
        if not accepted or total-tested<1e-4:
            break
    return x


def assess(t, com, shoe, shift, body, silhouette, role, guard, episodes, physics_dt):
    outcome={}
    a=diff(len(t),2,physics_dt)@com
    speed=np.linalg.norm(np.diff(com,axis=0),axis=-1)/physics_dt
    for name,lo,hi in (("whole",t[0],t[-1]),("spin",87.5,92.5),("entry",80.76,81.6),("exit",92.74,94.44)):
        q=interval(t,lo,hi)
        ints=q[:-2]&q[1:-1]&q[2:]
        pair=q[:-1]&q[1:]
        outcome[name]=dict(exposures=int(q.sum()),root_correction_mm=metric(np.linalg.norm(shift[q],axis=1)*1000),
            com_acceleration_mps2=metric(np.linalg.norm(a[ints],axis=1)),com_speed_mps=metric(speed[pair]),
            peak_acceleration_source_seconds=float(t[1:-1][ints][np.argmax(np.linalg.norm(a[ints],axis=1))]),
            detector_residual_fit_px=metric(body[q&~guard]),detector_residual_guard_px=metric(body[q&guard]),
            whole_skate_role2_px=metric(silhouette[q[:,None]&(role==2)]))
    for name,lo,hi,foot,kind in episodes:
        q=interval(t,lo,hi)
        if kind=="flight_visible_candidate" and q.sum()>=3:
            inside=q[:-2]&q[1:-1]&q[2:]
            tt=t[q]-lo
            path=com[q]+np.c_[0*tt,0*tt,4.905*tt**2]
            design=np.c_[np.ones(len(tt)),tt]
            model=design@np.linalg.lstsq(design,path,rcond=None)[0]
            outcome[name]=dict(
                source_seconds=[lo,hi],acceleration_samples=int(inside.sum()),
                com_gravity_residual_mps2=metric(np.linalg.norm(a[inside]-[0,0,-9.81],axis=-1)),
                ballistic_distance_mm=metric(np.linalg.norm(path-model,axis=1)*1000),
                min_steel_mm=metric(shoe[q].min(1)*1000))
        if name=="spin_support":
            z=shoe[q,foot]
            outcome[name]=dict(source_seconds=[lo,hi],absolute_height_mm=metric(np.abs(z)*1000),
                signed_median_mm=float(np.median(z)*1000),
                above_30mm=int((z>.03).sum()),below_minus_10mm=int((z<-.01).sum()),
                free_foot_height_mm=metric(shoe[q,1-foot]*1000))
    return outcome


def main(args):
    cv2.setNumThreads(1)
    parameter=args.baseline/"solve/candidate.npz"
    ref,cam,info,doc,controls=bind(args.prepared,args.evidence,parameter)
    p=ref.p
    s=p["pts"]*info["source_time_base"][0]/info["source_time_base"][1]
    frames=np.flatnonzero(interval(s,args.start,args.end))
    if len(np.unique(p["shot_ids"][frames]))!=1:
        raise ValueError("Trial is on one original shot")
    t=s[frames]
    dt=float(np.diff(t)[0])
    np.testing.assert_allclose(np.diff(t),dt,atol=1e-9,rtol=0.)
    r=ref.binding.ratio
    curve=ref.absolute_curve
    active=np.flatnonzero((curve.knots[:-4]>=ref.times[frames[0]]+dt)&
                          (curve.knots[4:]<=ref.times[frames[-1]]-dt))
    def basis(query):
        return curve.basis_sparse(query)[:,active].tocsr()
    b=basis(ref.times[frames])
    whole=load(args.roles)
    np.testing.assert_array_equal(whole["source_pts"],p["pts"])
    if str(whole["camera_sha256"])!=info["state_sha256"]:
        raise ValueError("Outline roles use a different camera")
    obs=load(args.prepared/"observations.npz")
    np.testing.assert_array_equal(obs["pts"],p["pts"])
    with np.load(args.baseline/"fitted/result.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["source_pts"],p["pts"])
        P,com,saved_uv=[z[k][frames] for k in ("P_world_mid","com_world_mid","COCO_uv")]
    np.testing.assert_allclose(curve.position_derivative(ref.times[frames],0)*r,P,atol=1e-9,rtol=0)
    clock,body_world=node_data(ref,cam,frames)
    base_native=moving_projection(cam,body_world,frames)["uv"]
    np.testing.assert_allclose(base_native,saved_uv,atol=1e-7,rtol=0)

    obs_use=obs["use"][frames]&np.isfinite(obs["uv"][frames]).all(axis=-1)
    # One 0.40 s block every 1.60 s, and 40 ms around it, has no direct fitted pixels.
    ordinal=np.arange(len(frames))
    guard=((ordinal//20)%4)==2
    excluded=np.isin(ordinal%80,np.arange(38,62))
    f,o=np.nonzero(obs_use&~excluded[:,None])
    joint=obs["model_joint_ids"][frames[f],o]
    sigma=np.maximum(5.,obs["sigma_px"][frames[f],o])
    pixel=PixelEvidence(cam,frames,basis(clock.ravel()),r,body_world,(f,joint),obs["uv"][frames[f],o],sigma,label="body")
    assembly=Assembly(ref.binding)
    shoes=source_shoe_pixels(ref,cam,assembly,frames,basis,whole,excluded)

    native=fk_state(ref,ref.at(ref.times[frames]),np.zeros((len(frames),69)))
    features=batch_features(ref.binding,native["R"],native["O"])
    steel=features["height"][...,features["eligible"]].min(-1)
    second=diff(len(t),2,dt)
    third=diff(len(t),3,dt)
    episodes=[]
    for row in doc["source_episode_review"]["episodes"]:
        lo,hi=(row[key]*info["source_time_base"][0]/info["source_time_base"][1]
               for key in ("start_pts","end_pts_inclusive"))
        if lo>=t[0]-1e-8 and hi<=t[-1]+1e-8:
            episodes.append((row["episode_id"],lo,hi,row.get("foot"),row["kind"]))

    dims=(len(active),3)
    delta_savgol=savgol_filter(P,11,2,axis=0)-P
    gain=(b.T@b+eye(len(active))*1e-6).tocsc()
    curves={"A":np.zeros(dims),"B":spsolve(gain,b.T@delta_savgol)}

    optics=cam.midpoint["rotation"][frames[1:-2],2]
    A=dot_rows(third@b,optics)
    if args.control=="isotropic":
        first=diff(len(t),1,dt)
        # Follow the local mean acceleration, keeping signed low-frequency
        # motion and guarding correction velocity along every world axis.
        smoothed=savgol_filter(com,9,3,axis=0)
        acceleration_target=second@smoothed
        lin=[LinearEvidence("translation_guard",kron(b,eye(3),format="csr")/.035,np.zeros(len(t)*3),2.0),
             LinearEvidence("velocity_correction_guard",kron(first@b,eye(3),format="csr")/.36,
                            np.zeros((len(t)-1)*3),2.0),
             LinearEvidence("excess_com_acceleration",kron(second@b,eye(3),format="csr")/15.,
                            (second@com-acceleration_target).ravel()/15.,2.5)]
    else:
        lin=[LinearEvidence("translation_guard",kron(b,eye(3),format="csr")/.12,np.zeros(len(t)*3),2.0),
             LinearEvidence("line_of_sight_com_jerk",A/250.,np.einsum("ij,ij->i",third@com,optics)/250.,2.0)]
    zdir=np.tile([0.,0.,1.],(len(t),1))
    Z=dot_rows(b,zdir)
    lin.append(LinearEvidence("steel_non_penetration",Z/.025,(steel.min(1)+.005)/.025,2.,True))
    sup=np.full(len(t),-1)
    fit_air=np.zeros(len(t),bool)
    for name,lo,hi,foot,kind in episodes:
        q=interval(t,lo,hi)&~excluded
        if kind=="support_candidate" and foot in (0,1):
            sup[q]=foot
        if kind=="flight_visible_candidate" and name!="entry_flight_core":
            fit_air|=q
    support=(sup>=0)&~fit_air
    h=steel[support,sup[support]]
    support_sigma=.018 if args.control=="isotropic" else .050
    lin.append(LinearEvidence("source_support_candidate",Z[support]/support_sigma,h/support_sigma,2.0))
    air=fit_air[:-2]&fit_air[1:-1]&fit_air[2:]
    gravity=(second@com)[air]-[0,0,-9.81]
    if air.any():
        lin.append(LinearEvidence("flight_gravity",kron((second@b)[air],eye(3),format="csr")/14.,
                                  gravity.ravel()/14.,2.))
    pixels=[pixel]+([shoes] if shoes is not None else [])
    emit("native_ablation_started",source_seconds=[float(t[0]),float(t[-1])],source_exposures=len(t),
         controls=len(active),body_fit_points=len(joint),whole_skate_fit_samples=0 if shoes is None else len(shoes.uv),
         source_support_candidates=int(support.sum()),flight_fit_samples=int(air.sum()),excluded_exposures=int(excluded.sum()),
         device="CPU sparse native camera-Jacobian trial",baseline_sha256=digest(parameter))
    control_name="C2" if args.control=="isotropic" else "C"
    curves[control_name]=nonlinear_fit(pixels,lin,dims,control_name)
    if args.pva:
        pva=load(args.pva)
        ids=np.searchsorted(p["pts"],pva["sample_pts"])
        np.testing.assert_array_equal(p["pts"][ids],pva["sample_pts"])
        if not interval(s[ids],t[0],t[-1]).all():
            raise ValueError("PVA target outside current source window")
        baseline=load(args.baseline/"fitted/result.npz")
        bp=baseline["COCO_world_mid"][ids]
        dur=np.diff(s[ids])
        traj=np.diff(bp,axis=0)/dur[:,None,None]
        v=pva["velocity_world_mps"]
        if v.shape!=traj.shape:
            raise ValueError("Unrecognized PVA joint convention")
        # Equal-weight robust location over the shoulders and hips, using
        # independent PVA velocity in exactly the current joint correspondence.
        torso=[5,6,11,12]
        valid=pva["source_keypoints_valid"][:,torso]
        native_uv=baseline["COCO_uv"][ids][:,torso]
        consistent=np.linalg.norm(pva["source_image_keypoints"][:,torso]-native_uv,axis=-1)<22.
        agree=(valid&consistent)[:-1]&(valid&consistent)[1:]
        rows=agree.sum(1)>=3
        delta_v=np.full((len(v),3),np.nan)
        for i in np.flatnonzero(rows):
            delta_v[i]=np.median((traj[i,torso]-v[i,torso])[agree[i]],axis=0)
        rows &= ~pva["box_inferred"][:-1]&~pva["box_inferred"][1:]
        if not rows.any():
            raise ValueError("No compatible PVA velocity rows in the shared source correspondence")
        M=diags(1/dur)@(basis(ref.times[ids][1:])-basis(ref.times[ids][:-1]))
        learned=LinearEvidence("pva_torso_velocity",kron(M[rows],eye(3),format="csr")/.90,
                               delta_v[rows].ravel()/.90,2.0)
        emit("pva_velocity_rows",source_pairs=len(rows),admitted=int(rows.sum()),
             delta_velocity_mps=metric(np.linalg.norm(delta_v[rows],axis=1)),prediction_sha256=digest(args.pva))
        curves["D"]=nonlinear_fit(pixels,[*lin,learned],dims,"D")

    out=args.output
    outlines=source_shoe_audit(ref,cam,assembly,frames,basis,whole,curves)
    stats={}
    fields=dict(source_pts=p["pts"][frames],source_seconds=t,local_check=guard,
                source_role=whole["role"][frames],active_position_controls=active)
    for method,x in curves.items():
        shift=b@x
        shift_node=(basis(clock.ravel())@x).reshape(len(t),5,3)
        pix=moving_projection(cam,body_world+shift_node[:,:,None]/r,frames)["uv"]
        idx,k=np.nonzero(obs_use)
        mapped=obs["model_joint_ids"][frames[idx],k]
        body=np.full(obs_use.shape,np.nan)
        body[idx,k]=np.linalg.norm(pix[idx,mapped]-obs["uv"][frames[idx],k],axis=-1)
        corrected_com=com+shift
        corrected_steel=steel+shift[:,2,None]
        stats[method]=assess(t,corrected_com,corrected_steel,shift,body,outlines[method],whole["role"][frames],
                             guard,episodes,dt)
        fields.update({method+"_"+k:a for k,a in dict(shift=shift,control_correction_m=x,uv=pix,
            com=corrected_com,steel=corrected_steel,outline=outlines[method],body_error=body,
            com_accel=second@corrected_com,P=P+shift).items()})
        if method in ("C","C2","D"):
            updated=dict(controls)
            updated["absolute_position_controls"]=controls["absolute_position_controls"].copy()
            updated["absolute_position_controls"][active]+=x/r
            updated["x"]=controls["x"].copy()
            source_block=updated["x"][17:17+len(curve.position_controls)*69].reshape(-1,69)
            source_block[active,:3]=updated["absolute_position_controls"][active]-controls["absolute_seed_position_controls"][active]
            np.testing.assert_allclose(controls["absolute_seed_position_controls"]+source_block[:,:3],
                                       updated["absolute_position_controls"],atol=1e-9,rtol=0.)
            updated.update(experimental_parent_sha256=np.array(digest(parameter)),experimental_method=np.array(method),
                           historical_solver_diagnostics_apply_to_parent=np.array(True))
            atomic_npz(out/(method+"_candidate.npz"),**updated)
    atomic_npz(out/"ablation_values.npz",**fields)
    atomic_json(out/"ablation.json",dict(source_pts=p["pts"][frames[[0,-1]]].tolist(),source_seconds=t[[0,-1]].tolist(),
        shot=str(p["shot_ids"][frames[0]]),native_dt_seconds=dt,camera_sha256=info["state_sha256"],
        baseline_sha256=digest(parameter),state_scope="root translation only, same body/skate rotations and dimensions",
        method=dict(A="saved native state",B="11 native samples degree-2 Savitzky-Golay root translation",
                    C="native body and whole skate pixels, robust line-of-sight COM jerk, shared blade and flight terms",
                    C2="same pixels with all-axis excess acceleration relative to local signed motion, correction-velocity guard",
                    D=control_name+" with verified PVA torso velocity if present"),
        weights=(dict(root_correction_m=.035,correction_velocity_mps=.36,excess_acceleration_mps2=15.,
                      target_sg_window_exposures=9,support_m=.018) if args.control=="isotropic" else
                 dict(root_correction_m=.12,los_jerk_mps3=250.,support_m=.05)) |
                dict(penetration_m=.025,flight_mps2=14.,pva_velocity_mps=.90),
        holdout_scope="new local fit excludes contiguous body/skate blocks with two exposure padding each side; source streams reused historically",
        body_fit_points=len(joint),whole_skate_fit_samples=0 if shoes is None else len(shoes.uv),
        clips=stats,pva_used=args.pva is not None))
    for label,stat in stats.items():
        emit("ablation_result",method=label,spin_com=stat["spin"]["com_acceleration_mps2"],
             whole_guard=stat["whole"]["detector_residual_guard_px"],spin_support=stat.get("spin_support"),
             holdout_outline=stat["whole"]["whole_skate_role2_px"])


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__)
    for name in ("baseline","prepared","evidence","roles","output"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--start",type=float,default=76.)
    cli.add_argument("--end",type=float,default=95.)
    cli.add_argument("--pva",type=Path)
    cli.add_argument("--control",choices=("los","isotropic"),default="los")
    args=cli.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    main(args)
