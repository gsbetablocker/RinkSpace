"""Refine the two source-reviewed left-support windows on the native motion curve."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times, _log, _exp
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet,evaluate_curve
from calibration.program_tracking.body_tracking.skate_physics import fk_state,batch_features
from sitspin_refit import refresh_photo
from source_joint_refit import POLICY, log, save
from source_review import digest
from world_motion_refit import WorldMotion, project


def dense_sit_objective(stage,window):
    lo,hi=window
    times=np.arange(round(lo*200),round(hi*200)+1)/200-stage.ref.p['pts'][0]/12800
    prior=stage.ref.at(times)
    base_k=fk_state(stage.ref,prior,np.zeros((len(times),69)))['native_world']
    mid=np.arange(1,len(times)-1)
    def knee_geometry(points):
        thigh=points[:,1]-points[:,4];shin=points[:,7]-points[:,4]
        thigh/=np.linalg.norm(thigh,axis=1,keepdims=True);shin/=np.linalg.norm(shin,axis=1,keepdims=True)
        normal=np.cross(thigh,shin);sine=np.linalg.norm(normal,axis=1)
        return np.arctan2(sine,(thigh*shin).sum(1)),normal/np.maximum(1.e-8,sine[:,None])
    old_angle=knee_geometry(base_k)[0]
    def terms(curve,group,linear):
        nd=3*len(group)
        ev=evaluate_curve(curve,times,with_blocks=linear)
        fk=fk_state(stage.ref,{**prior,**ev},np.zeros((len(times),69)))
        axes=fk['Q'][:,None]@fk['frames'];pos=fk['native_world']*stage.ratio
        st=dict(pos=pos,axes=axes);R=fk['R'][:,0];O=fk['O'][:,0]*stage.ratio
        def witness(rot):
            steel=batch_features(stage.ref.binding,rot[:,None],np.zeros((len(rot),1,3)))
            ok=np.isin(steel['semantics'],['runner','toe_pick'])
            idx=steel['height'][:,0,ok].argmin(1)
            return steel['local_point'][:,0][:,ok][np.arange(len(rot)),idx]*stage.ratio
        local=witness(R[mid])
        a=O[:-2]+(R[:-2]@local[:,:,None])[:,:,0]
        b=O[2:]+(R[2:]@local[:,:,None])[:,:,0]
        vel=(b-a)*100
        h=R[mid,:,0];length=np.linalg.norm(h[:,:2],axis=1)
        side=np.stack((-h[:,1],h[:,0],np.zeros(len(h))),1)/length[:,None]
        slip_w=1./.0008
        slip=(vel*side).sum(1)*slip_w
        angle,normal=knee_geometry(pos)
        knee_w=2100.
        knee=(angle-old_angle)*knee_w
        cost=dict(sit_dense_slip=float(slip@slip)/len(stage.ref.times),sit_knee_angle=float(knee@knee)/len(stage.ref.times))
        if not linear:return cost,None,None
        blocks=ev['rotation_right_blocks'][:,:,np.maximum(group,0)].copy()
        if -1 in group:blocks[:,:,group.index(-1)]=ev['position_weights'][:,:,None,None]*np.eye(3)
        def columns(rows,der):
            values=np.einsum('ngj,nkgji->nkgi',der.reshape(len(rows),len(group),3),blocks[rows])
            col=ev['control_indices'][rows,:,None]*nd+np.arange(nd)
            rid=np.broadcast_to(np.arange(len(rows))[:,None,None],col.shape)
            return sparse.coo_matrix((values.ravel(),(rid.ravel(),col.ravel())),
                shape=(len(rows),len(curve.position_controls)*nd)).tocsr()
        ja=stage.jac(a[:,None],[7],group,st,mid-1,1)[:,0];jb=stage.jac(b[:,None],[7],group,st,mid+1,1)[:,0]
        dslip=columns(mid-1,-np.einsum('ni,nij->nj',side,ja)*100)+columns(mid+1,np.einsum('ni,nij->nj',side,jb)*100)
        rocker=np.stack([(witness(R[mid]@_exp(axis))-witness(R[mid]@_exp(-axis)))/.0002 for axis in np.eye(3)*.0001],2)
        dmiddle=np.zeros((len(mid),nd))
        for slot,j in enumerate(group):
            if j<0 or not stage.ancestor[7,j]:continue
            ax=axes[mid,j].swapaxes(1,2);diff=np.cross(ax,h[:,None])
            ds=np.stack((-diff[:,:,1],diff[:,:,0],np.zeros_like(diff[:,:,0])),2)
            ds=(ds-side[:,None]*(ds*side[:,None]).sum(-1,keepdims=True))/length[:,None,None]
            dv=(R[mid+1]-R[mid-1])@rocker@(R[mid].swapaxes(1,2)@axes[mid,j])*100
            dmiddle[:,3*slot:3*slot+3]=(vel[:,None]*ds).sum(-1)+np.einsum('ni,nij->nj',side,dv)
        dslip=(dslip+columns(mid,dmiddle))*slip_w
        dk=np.zeros((len(times),nd));col=3*group.index(4)
        dk[:,col:col+3]=np.einsum('ni,nij->nj',normal,axes[:,4])*knee_w
        return cost,sparse.vstack([dslip,columns(np.arange(len(times)),dk)],format='csr'),np.r_[slip,knee]
    return terms


def anatomical_frames(stage,fk,axes,boot,ankle,gt,ids,frame_idx,lo,hi):
    """Two-bone reach with a source knee marker and a continuous flexion plane."""
    def unit(x):
        return x/np.maximum(1.e-10,np.linalg.norm(x,axis=-1,keepdims=True))
    def plane(v,a):
        return unit(v-(v*a).sum(-1,keepdims=True)*a)
    old=fk['native_world']*stage.ratio;hip=old[:,1];old_knee=old[:,4]
    a=np.linalg.norm(old_knee-hip,axis=1);b=np.linalg.norm(old[:,7]-old_knee,axis=1)
    axis=unit(ankle-hip)
    reach=np.clip(np.linalg.norm(ankle-hip,axis=1),abs(a-b)+.0001,a+b-.0001)
    along=(a*a-b*b+reach*reach)/(2*reach)
    line=hip+axis*along[:,None];radius=np.sqrt(np.maximum(0,a*a-along*along))
    forward=plane(boot[:,:,0],axis)
    across=np.cross(axis,forward)
    angles=np.linspace(-np.pi,np.pi,72,endpoint=False)
    dirs=forward[:,None]*np.cos(angles)[None,:,None]+across[:,None]*np.sin(angles)[None,:,None]
    candidates=line[:,None]+radius[:,None,None]*dirs
    k=ids[np.abs(stage.ref.p['pts'][ids,None]/12800-gt).argmin(0)]
    cam=stage.cameras[k];meas=stage.target[k,13]
    offset=stage.parent_uv[k,13]-project(cam,stage.seed_state['pos'][2*k,4,None],stage.camera_scale)[:,0]
    uv=project(cam,candidates,stage.camera_scale)+offset[:,None]
    # Protect the preceding sit-spin knee. In layback, the visually reviewed
    # source marker lies inside the support leg where the earlier fit drifts.
    sit=lo>170
    target=np.where(stage.use[k,13,None],meas,stage.parent_uv[k,13])
    cost=((uv-target[:,None])**2).sum(-1)/(15. if sit else 6.)**2
    cost+=((uv-stage.parent_uv[k,13,None])**2).sum(-1)/(4. if sit else 22.)**2
    cost+=((candidates-old_knee[:,None])**2).sum(-1)/(.024 if sit else .075)**2
    cost+=1.5*(1-np.cos(angles))
    diff=angles[:,None]-angles[None,:]
    steps=(np.arctan2(np.sin(diff),np.cos(diff))/.23)**2
    best=cost[0];links=[]
    for row in cost[1:]:
        trial=best[:,None]+steps;links.append(trial.argmin(0));best=trial.min(0)+row
    path=[int(best.argmin())]
    for prev in reversed(links):path.append(prev[path[-1]])
    knee=candidates[np.arange(len(gt)),path[::-1]];ankle=hip+axis*reach[:,None]
    oldboot=fk['R'][:,0];border=(gt<lo+.11)|(gt>hi-.11)
    def bone_frame(j,child,world):
        rest=unit(stage.ref.body.J[child]-stage.ref.body.J[j])
        prior_y=(axes[:,j].swapaxes(1,2)@oldboot[:,:,1,None])[:,:,0]
        side=plane(np.median(prior_y[border],axis=0),rest)
        basis=np.stack((side,rest,np.cross(side,rest)),1)
        w=unit(world);lat=plane(boot[:,:,1],w)
        return np.stack((lat,w,np.cross(lat,w)),2)@basis.T
    return bone_frame(1,4,knee-hip),bone_frame(4,7,ankle-knee)


def image_seed(stage,review,anatomical=False):
    """Initialize bearing on the ice plane, then solve the rigid shin and boot chain."""
    curve=stage.seed
    controls=curve.rotation_controls.copy()
    g=greville_times(curve.knots)
    clock=g+stage.ref.p['pts'][0]/12800
    times=stage.ref.p['pts']/12800
    headings=np.linspace(-np.pi,np.pi,180,endpoint=False)
    direction=np.stack((np.cos(headings),np.sin(headings),np.zeros_like(headings)),1)
    length=np.linalg.norm(stage.mat[4]-stage.mat[3])*stage.ratio
    for lo,hi in review['windows']:
        ids=np.flatnonzero((times>=lo-1e-8)&(times<=hi+1e-8))
        pts=stage.ref.p['pts'][ids]
        camera=stage.cameras[ids]
        records={(r['pts'],r['feature']):r for r in review['records'] if r['role']=='fit'}
        use=np.array([int(x) in {p for p,_ in records} for x in pts])
        boxes=np.array([[records[int(x),k]['box_px'] for k in (20,21)] for x in pts[use]])
        target=boxes.mean(2).mean(1)
        # The measured centre is only used on visible training frames. The
        # neighbouring estimates supply a continuous start through occlusions.
        centres=np.column_stack([np.interp(times[ids],times[ids][use],target[:,k]) for k in range(2)])
        mid=stage.seed_state['shoes'][2*ids,0,3:5].mean(1)
        mid[:,2]=.009
        def project_mid(x):
            return project(camera,x[:,None],stage.camera_scale)[:,0]
        for _ in range(5):
            uv=project_mid(mid)
            jac=np.stack([(project_mid(mid+v*.0001)-uv)/.0001 for v in np.eye(3)[:2]],axis=-1)
            mid[:,:2]+=np.linalg.solve(jac,(centres-uv)[:,:,None])[:,:,0]
        pred=np.stack([mid[:,None]-direction*length/2,mid[:,None]+direction*length/2],axis=2)
        pred[:,:,0,2]=.006;pred[:,:,1,2]=.012
        px=project(camera,pred.reshape(len(ids),-1,3),stage.camera_scale).reshape(len(ids),len(headings),2,2)
        difference=px[use]-np.clip(px[use],boxes[:,None,:,0],boxes[:,None,:,1])
        data=np.zeros((len(ids),len(headings)))
        data[use]=np.sum(np.sqrt(1+(difference/3.)**2)-1,axis=(2,3))
        near=np.flatnonzero((times>lo-.35)&(times<hi+.35))
        rot=stage.seed_state['boot_R'][2*near,0]
        yaw=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]))
        advance=np.median(np.diff(yaw))
        bend=headings[None,:]-headings[:,None]-advance
        transitions=(np.arctan2(np.sin(bend),np.cos(bend))/.18)**2
        prefs=[];total=data[0].copy()
        for v in data[1:]:
            cost=total[:,None]+transitions
            prefs.append(cost.argmin(0))
            total=cost.min(0)+v
        path=[int(total.argmin())]
        for prev in reversed(prefs):path.append(prev[path[-1]])
        angles=np.unwrap(headings[np.asarray(path[::-1])])
        idx=np.where(stage.active_knots&(clock>=lo-.04)&(clock<=hi+.04))[0]
        gt=clock[idx]
        fk=fk_state(stage.ref,stage.ref.at(g[idx]),np.zeros((len(idx),69)))
        axes=fk['Q'][:,None]@fk['frames']
        oldboot=fk['R'][:,0]
        yaw=np.interp(gt,times[ids],angles)
        forwards=np.stack([np.cos(yaw),np.sin(yaw),np.zeros_like(yaw)],1)
        lateral=np.stack([-np.sin(yaw),np.cos(yaw),np.zeros_like(yaw)],1)
        up=np.broadcast_to([0.,0.,1.],forwards.shape)
        bank=np.arcsin(np.clip(oldboot[:,2,1],-.45,.45))[:,None]
        boot=np.stack((forwards,lateral*np.cos(bank)+up*np.sin(bank),up*np.cos(bank)-lateral*np.sin(bank)),2)
        rel=axes[:,7].swapaxes(1,2)@oldboot
        ankle_axes=boot@rel.swapaxes(1,2)
        old_ankle=fk['native_world'][:,7]*stage.ratio
        knee=fk['native_world'][:,4]*stage.ratio
        ankle_to_boot=(axes[:,7].swapaxes(1,2)@((fk['O'][:,0]*stage.ratio-old_ankle)[:,:,None]))[:,:,0]
        origin=np.stack([np.interp(gt,times[ids],mid[:,k]) for k in range(3)],1)-(boot@(stage.mat[3:5].mean(0)*stage.ratio)[:,None])[:,:,0]
        aim=origin-(ankle_axes@ankle_to_boot[:,:,None])[:,:,0]-knee
        aim/=np.linalg.norm(aim,axis=1,keepdims=True)
        shin=old_ankle-knee;shin/=np.linalg.norm(shin,axis=1,keepdims=True)
        cross=np.cross(shin,aim);s=np.linalg.norm(cross,axis=1)
        align=Rotation.from_rotvec(cross*(np.arctan2(s,(shin*aim).sum(1))/np.maximum(s,1e-9))[:,None]).as_matrix()
        knee_axes=align@axes[:,4]
        blend=np.clip(np.minimum((gt-lo+.02)/.10,(hi-gt+.02)/.10),0,1)
        blend=blend*blend*(3-2*blend)
        hip_axes=axes[:,1]
        goal=[]
        if anatomical:
            hip_axes,knee_axes=anatomical_frames(stage,fk,axes,boot,
                origin-(ankle_axes@ankle_to_boot[:,:,None])[:,:,0],gt,ids,idx,lo,hi)
            goal.append((1,axes[:,0].swapaxes(1,2)@hip_axes))
        goal.extend(((4,hip_axes.swapaxes(1,2)@knee_axes),(7,knee_axes.swapaxes(1,2)@ankle_axes)))
        for joint,wanted in goal:
            change=_log(fk['local_R'][:,joint-1].swapaxes(1,2)@wanted)
            controls[idx,joint]=controls[idx,joint]@_exp(change*blend[:,None])
        log(stage.output,'source_bearing_seed',window=[lo,hi],frames=len(ids),observed=int(use.sum()),
            median_prior_advance_deg=float(np.degrees(advance)),bearing_degrees=np.degrees(angles).round(1).tolist())
    new=AbsoluteMotionCurve(curve.knots,curve.position_controls,controls,curve.edge_reference_quaternion)
    new.check_continuous_edge_update(curve)
    return new


def run(args):
    out=args.output
    out.mkdir(exist_ok=True)
    log(out,'start',source_windows='159.08 and 175.72',parent=str(args.parent),opening=POLICY)
    inp=args.programme/'fit_v11/spin_pair_pixels_1119.json'
    obs=json.loads(inp.read_text('utf8'))
    source=json.loads((args.parent/'source.json').read_text('utf8'))
    assert obs['source_sha256']==source['source_sha256']
    identity=digest(args.parent/'candidate.npz')
    source['paired_spin_review']=dict(path=str(inp),sha256=digest(inp),parent_sha256=identity,
        windows=obs['windows'],protected_seconds=obs['preserve_seconds'])
    ledger=out/'body_observations.npz'
    if not ledger.exists():shutil.copyfile(args.parent/ledger.name,ledger)
    assert digest(ledger)==digest(args.parent/ledger.name)
    (out/'source.json').write_text(json.dumps(source,ensure_ascii=False,separators=(',',':')),'utf8')
    stage=WorldMotion(args.prepared,args.parent,ledger,out)
    stage.source_factors(args.work,args.shared)
    pts=stage.ref.p['pts']
    t=stage.stamps/12800
    intervals=[(t>=lo-1e-8)&(t<=hi+1e-8) for lo,hi in obs['windows']]
    tracked=intervals[0]|intervals[1]
    assert stage.support[tracked,0].all()
    stage.active_knots=np.zeros(len(stage.seed.position_controls),bool)
    stage.active_knots[np.unique(stage.B[tracked].indices)]=True
    stage.local_fit_rows=np.asarray(stage.B[:,stage.active_knots].getnnz(axis=1)).ravel()>0
    assert not stage.local_fit_rows[np.abs(t-177.54)<.35].any()
    stage.exact_rotation_basis=stage.differentiate_rocker=stage.preserve_step_direction=True
    if args.dense_sit:stage.control_objective=dense_sit_objective(stage,obs['windows'][1])
    stage.edge_review=True
    stage.root_trust=0.
    stage.protect[:]=0
    stage.protect[:,13]=10.
    stage.knee_weight=2.
    if args.root:
        mask=np.zeros_like(stage.active_knots)
        mask[np.unique(stage.B[intervals[0]].indices)]=True
        stage.joint_knots={-1:mask}
        stage.root_trust=250.
        stage.protect[:,[0,5,6,7,8,9,10,11,12,14,16]]=3.
        stage.protect[:,24:31]=4.
    if args.hip:
        stage.protect[intervals[0][::2],13]=2.
        stage.protect[intervals[1][::2],13]=30.
        stage.left_knee_world_weight=np.where(intervals[0],180.,1600.)
        stage.knee_weight=20.
    stage.com_weight=args.com
    stage.shoe_source_weight=3.
    stage.left_outline_weight=2.8
    stage.support_slip_scale=np.full(stage.support.shape,.08)
    stage.support_slip_scale[tracked,0]=args.slip
    stage.native_slip_scale=args.slip
    stage.review_contact_weight=np.full(stage.support.shape,1800.)
    stage.review_contact_weight[tracked,0]=args.ice
    stage.runner_height_cap=np.where(intervals[0],.027,.035)
    stage.runner_cap_weight=np.where(tracked,4200.,0.)
    with np.load(args.parent/'joint_source_measurements.npz') as z:
        assert str(z['candidate'])==identity
        surface=z['shoe'][:,0,7:]
    stage.outline_reference_px=np.abs((surface@stage.dirs.T).max(1)-stage.ext[:,0])+1.
    stage.outline_reference_px[~tracked[::2]]=np.inf
    if args.protect_knee:
        stage.protect[intervals[1][::2],13]=65.
        stage.left_knee_world_weight[intervals[1]]=7600.
        stage.knee_weight=36.
        stage.left_outline_weight=9.
        stage.root_trust=800.
    rows=[r for r in obs['records'] if r['role']=='fit']
    boxes=np.array([r['box_px'] for r in rows])
    frames=np.searchsorted(pts,[r['pts'] for r in rows])
    np.testing.assert_array_equal(pts[frames],[r['pts'] for r in rows])
    weight=np.array([args.photo*(.6 if r['end_on'] else 1.) for r in rows])
    focus=np.isin(pts[frames],np.rint(np.array([159.08,175.72])*12800))
    weight[focus]*=2.5
    # A visible interval supplies the main constraint; the centre of the
    # interval gives its weak, uncertainty-dependent location estimate.
    mid=boxes.mean(1)
    feature=np.array([r['feature'] for r in rows])
    stage.reviewed_image_regions=(np.tile(frames,2),np.tile(feature,2),
        np.r_[boxes[:,0],mid],np.r_[boxes[:,1],mid],np.r_[weight,weight*.27])
    curve=stage.seed
    checkpoint=out/'candidate.npz'
    initial=checkpoint if checkpoint.exists() else args.initial
    cycle=0
    if initial:
        with np.load(initial) as z:
            old=json.loads(str(z['joint_source_fit_json']))
            assert old['parent_sha256']==identity and old['review_sha256']==digest(inp)
            if initial==checkpoint:assert str(z['source_evidence_sha256'])==digest(out/'source.json')
            curve=curve_from_packet(z)
            if initial==checkpoint:cycle=old['cycle']+1
    elif args.image_seed:
        curve=image_seed(stage,obs,args.anatomy)
    if args.anatomy:
        if not args.hip:raise ValueError('An anatomical initialization uses the left hip')
        stage.pose_reference=image_seed(stage,obs,True) if initial else curve
        stage.pose_trust_scale=args.pose
        stage.pose_rate_trust_scale=args.rate
    nominal=100*stage.ref.body.scale*np.ptp(stage.ref.binding.surface.shaped[:,1])
    np.testing.assert_allclose(nominal,153.,rtol=0,atol=.001)
    if initial or args.image_seed:refresh_photo(stage,curve,args)
    focus_idx=np.searchsorted(pts,[2036224,2249216])
    def detail(curve):
        pose=stage.state(curve)
        uv=project(stage.cameras,pose['photo'],stage.camera_scale)+stage.offset
        u=uv[frames,feature]
        return dict(focus_uv=uv[focus_idx,20:22].tolist(),
                    focus_heights_mm=(1000*pose['shoes'][2*focus_idx,0,3:5,2]).tolist(),
                    source_box_p95_px=float(np.percentile(np.linalg.norm(u-np.clip(u,boxes[:,0],boxes[:,1]),axis=1),95)))
    log(out,'initial',**detail(curve))
    group=[1,4,7] if args.hip else [4,7]
    if args.root:group=[-1]+group
    while True:
        curve,gain=stage.optimize(curve,group)
        report=dict(stage='paired_spin_refit',mode='fixed',cycle=cycle,improvement=gain,nominal_cm=nominal,
              body_scale=stage.ref.body.scale,camera='s14_boards_1118',ice='lane4_side_1118',
              camera_sha256=stage.cam.info['state_sha256'],source_exposures=len(pts),parent_sha256=identity,
              programme_camera_registration_sha256=source['programme_camera_candidate']['record_sha256'],
              review_sha256=digest(inp),local_joints=group,com_weight=args.com,slip_scale=args.slip,
              contact_weight=args.ice,pixel_weight=args.photo,initial_sha256=digest(args.initial) if args.initial else None,
              image_seed=args.image_seed,anatomical_prior=args.anatomy,pose_weight=args.pose,rate_weight=args.rate,
              protect_knee=args.protect_knee,dense_sit=args.dense_sit,**detail(curve))
        save(out,'candidate.npz',stage.packet,curve,stage.ref,report)
        if gain<.000003:break
        cycle+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('prepared','parent','work','shared','programme','output'):
        parser.add_argument('--'+key,type=lambda v:Path(v).resolve(),required=True)
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--image-seed',action='store_true')
    parser.add_argument('--hip',action='store_true')
    parser.add_argument('--root',action='store_true')
    parser.add_argument('--protect-knee',action='store_true')
    parser.add_argument('--dense-sit',action='store_true')
    parser.add_argument('--anatomy',action='store_true')
    parser.add_argument('--pose',type=float,default=135.)
    parser.add_argument('--rate',type=float,default=15.)
    parser.add_argument('--com',type=float,default=8.)
    parser.add_argument('--photo',type=float,default=32.)
    parser.add_argument('--slip',type=float,default=.010)
    parser.add_argument('--ice',type=float,default=25000.)
    run(parser.parse_args())
