"""Fit one free body scale and a continuous full-programme trajectory, or a matched fixed control."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsqr

from world_motion_refit import WorldMotion
from shared_stature import apply_body_scale
from source_joint_refit import POLICY, save, log
from source_review import digest
from opening_support import samples, verify, enforce
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from calibration.program_tracking.body_tracking.body_gpu_linear import CuDSS


class StatureFit(WorldMotion):
    def __init__(self,args):
        super().__init__(args.prepared,args.parent,args.shared/'final_body_observations.npz',args.output)
        self.source_factors(args.work,args.shared)
        self.edge_review=True
        self.root_trust=0.
        self.protect[:]=0.  # Projected positions from the fixed-height fit are not new measurements.
        self.original_scale=float(self.ref.body.scale)
        self.local_landmarks=self.attach.copy()
        self.mass_offsets=self.mass.offsets.copy()
        self.theta=0.
        y=self.ref.binding.surface.shaped[:,1]
        self.rest_span=float(np.ptp(y))
        self.nuisance_groups=[[-1,1,4,7],[-1,2,5,8],[-1,0],[3,6,9,12],[13,16,18,20],[14,17,19,21]]

    def set_height(self,theta):
        if theta==self.theta:return
        if not np.isfinite(theta):raise ValueError('Nonfinite body scale increment')
        multiplier=np.exp(theta)
        apply_body_scale(self.ref,self.original_scale*multiplier)
        self.attach=self.local_landmarks*multiplier
        self.mass.offsets=self.mass_offsets.copy()
        self.mass.offsets[:self.mass.count]*=multiplier
        self.theta=float(theta)

    def stature_cm(self):
        return 100*self.rest_span*self.ref.body.scale

    def root_height_step(self,curve,free,*,offset_cm=None,check_derivative=False,keep_offset=False):
        from calibration.program_tracking.body_tracking.skate_physics import fk_state
        old_theta=self.theta
        costs,H,g=self.system(curve,[-1,-2])
        n=len(curve.position_controls)
        # One scalar column shared by every spline basis function on every shot.
        column=np.zeros((n,6));column[:,3]=1.
        t=samples(self.ref,POLICY);B=curve.basis_sparse(t)[:,self.restricted]
        from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
        pose=evaluate_curve(curve,t)
        original=self.ref.at(t)
        fk=fk_state(self.ref,{**original,**pose},np.zeros((len(t),69)))
        rel=(fk['native_world'][:,[7,8],2]-fk['P'][:,None,2]).mean(1)
        # Reparametrize opening pelvis height around both ankles. All accepted
        # trials additionally pass the two-blade continuous contact guard.
        column[self.restricted,2]=lsqr(B,-rel,atol=1e-12,btol=1e-12)[0]
        active=np.zeros((n,6),bool);active[:,:3]=True;active[self.restricted,2]=False
        ix=np.flatnonzero(active.ravel());c=column.ravel();hg=H@c
        cross=hg[ix];scalar=float(c@hg);grad=float(c@g)
        A=H[ix][:,ix].tocsr()+sparse.eye(len(ix),format='csr')*2.
        A.sort_indices();A.eliminate_zeros()
        gpu=torch.sparse_csr_tensor(torch.as_tensor(A.indptr,device='cuda',dtype=torch.int32),
            torch.as_tensor(A.indices,device='cuda',dtype=torch.int32),torch.as_tensor(A.data,device='cuda'),size=A.shape)
        cudss=CuDSS()
        def solve(rhs):
            answer,receipt=cudss.solve(gpu,torch.as_tensor(rhs,device='cuda'))
            return answer.cpu().numpy(),receipt
        root,receipt=solve(-g[ix]);response,_=solve(-cross)
        curvature=scalar+float(cross@response);reduced_grad=grad+float(cross@root)
        if not np.isfinite(curvature) or curvature<=0:raise ValueError(('Underdetermined height',curvature,scalar))
        dh=-reduced_grad/curvature if free else 0.
        if offset_cm is not None:dh=np.log1p(offset_cm/self.stature_cm())
        def trial(alpha,ht,dx):
            step=column*ht
            step.ravel()[ix]+=dx
            pos=curve.position_controls+alpha*step[:,:3]
            self.set_height(old_theta+alpha*ht)
            moved=AbsoluteMotionCurve(curve.knots,pos,curve.rotation_controls,curve.edge_reference_quaternion)
            try:verify(self.ref,moved,POLICY)
            except ValueError:
                moved=enforce(self.ref,moved,POLICY,lambda event,**v:log(self.output,event,**v))
            return moved
        dx=root+response*dh
        if check_derivative:
            # Check the full objective in the very direction used in the global
            # solve, including every projection, mass and sole term.
            eps=2e-5;loss=[]
            for sign in (-1,1):
                q=trial(sign*eps,1.,response)
                loss.append(sum(self.system(q,[-1,-2],False).values()))
            self.set_height(old_theta)
            numeric=(loss[1]-loss[0])/(2*eps)
            analytic=2*float(grad+g[ix]@response)/len(self.ref.times)
            log(self.output,'height_derivative',finite=numeric,analytic=analytic,
                relative_difference=abs(numeric-analytic)/max(1.,abs(numeric),abs(analytic)))
        before=sum(costs.values());alphas=[1.] if offset_cm is not None else [1.,.5,.25,.125,.0625,.03125]
        for alpha in alphas:
            moved=trial(alpha,dh,dx)
            loss=self.system(moved,[-1,-2],False)
            current=sum(loss.values())
            if offset_cm is not None:
                if keep_offset:return moved,current
                self.set_height(old_theta)
                return current,loss
            accepted=current<before
            log(self.output,'shared_height_step',accepted=accepted,alpha=alpha,nominal_cm=self.stature_cm(),
                common_rink_cm=self.stature_cm()*self.ratio,before=costs,after=loss,
                conditional_curvature=curvature,unprofiled_curvature=scalar,controls=n,
                active_translations=len(ix),device=receipt['backend'],device_seconds=receipt['solve_seconds'])
            if accepted:return moved,(before-current)/before
        self.set_height(old_theta)
        return curve,0.


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.parent/'source.json',args.output/'source.json')
    log(args.output,'start',mode=args.mode,parent=str(args.parent),scale_prior=False,
        shared_scalar_count=int(args.mode=='free'),opening=POLICY,output=str(args.output))
    fit=StatureFit(args);curve=fit.seed;cycle=0
    checkpoint=args.output/'candidate.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            info=json.loads(str(z['joint_source_fit_json']))
            if info['parent_sha256']!=digest(args.parent/'candidate.npz') or info['mode']!=args.mode:
                raise ValueError('Height experiment checkpoint identity mismatch')
            fit.set_height(np.log(float(z['fitted_programme_body_scale'])/fit.original_scale))
            curve=curve_from_packet(z);cycle=info['cycle']+1
    elif args.start_cm is not None:
        if args.mode!='free':raise ValueError('Alternative initialization requires a free height')
        curve,loss=fit.root_height_step(curve,True,offset_cm=args.start_cm-fit.stature_cm(),keep_offset=True)
        log(args.output,'alternative_initialization',nominal_cm=fit.stature_cm(),loss=loss)
    parent_hash=digest(args.parent/'candidate.npz')
    while True:
        height_before=fit.stature_cm()
        curve,gain=fit.root_height_step(curve,args.mode=='free',check_derivative=(cycle==0 and args.mode=='free'))
        for group in fit.nuisance_groups:
            curve,change=fit.optimize(curve,group);gain+=change
        report=dict(stage='shared_stature_experiment',mode=args.mode,cycle=cycle,improvement=gain,
            body_scale=fit.ref.body.scale,nominal_cm=fit.stature_cm(),common_rink_cm=fit.stature_cm()*fit.ratio,
            camera='s14_boards_1118',ice='lane4_side_1118',parent_sha256=parent_hash,
            camera_sha256=fit.cam.info['state_sha256'],
            programme_camera_registration_sha256=fit.ref.source_document['programme_camera_candidate']['record_sha256'],
            source_exposures=len(fit.ref.times),scale_prior=False)
        packet=dict(fit.packet,fitted_programme_body_scale=np.array(fit.ref.body.scale))
        save(args.output,'candidate.npz',packet,curve,fit.ref,report)
        if cycle+1>=args.min_cycles and cycle and gain<.001 and abs(fit.stature_cm()-height_before)<.02:break
        cycle+=1
    if args.mode=='free' and args.start_cm is None:
        profile=[]
        for offset in [-4.,-2.,-1.,0.,1.,2.,4.]:
            loss,parts=fit.root_height_step(curve,True,offset_cm=offset)
            profile.append(dict(nominal_cm=fit.stature_cm()+offset,loss=loss,components=parts))
        (args.output/'profile.json').write_text(json.dumps(profile,indent=2),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['prepared','parent','shared','work','output']:ap.add_argument(k,type=lambda v:Path(v).resolve())
    ap.add_argument('--mode',choices=['fixed','free'],required=True)
    ap.add_argument('--min-cycles',type=int,default=1)
    ap.add_argument('--start-cm',type=float)
    run(ap.parse_args())
