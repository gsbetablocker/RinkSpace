"""Restore the ending sit-spin left runner from source boxes and whole-boot outlines."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times, _exp, _log
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from limb_support_refit import load_regions
from source_joint_refit import POLICY, log, save
from source_review import digest
from world_motion_refit import WorldMotion, project


def refresh_photo(stage,curve,args):
    from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference
    from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
    ref=absolute_reference(stage.ref,curve)
    nodes=np.linspace(-.5,.5,5)
    st=stage.state(curve)
    surface=np.vstack([stage.mat,stage.surface])
    collar=np.unique(ref.binding.packet()['collar'])
    ids=np.flatnonzero(stage.local_fit_rows[::2])
    for group in np.array_split(ids,max(1,int(np.ceil(len(ids)/32)))):
        ts=ref.times[group,None]+stage.cam.readout_times(group)[:,None]*nodes
        fk=ref.world(ref.at(ts.ravel()))
        gear=fk['footwear']
        shoe=gear['boot_O'][:,:,None]+np.einsum('nfij,vj->nfvi',gear['boot_R'],surface)
        cuff=gear['upper_local'][:,:,collar].mean(2)
        shoe[:,:,6]=gear['boot_O']+np.einsum('nfij,nfj->nfi',gear['boot_R'],cuff)
        world=np.concatenate([fk['coco'],shoe.reshape(ts.size,-1,3)],axis=1)
        actual=moving_projection(stage.cam,world.reshape(len(group),5,-1,3),group)['uv']
        boots=actual[:,17:].reshape(len(group),2,-1,2)
        targets=np.concatenate([actual[:,:17],boots[:,:,:7].reshape(len(group),14,2)],axis=1)
        mid=stage.cameras[group]
        stage.offset[group]=targets-project(mid,st['photo'][group],stage.camera_scale)
        for foot in (0,1):
            xyz=stage.surface_xyz(st,foot)[group]
            stage.surface_offset[foot][group]=boots[:,foot,7:]-project(mid,xyz,stage.camera_scale)
    log(args.output,'native_projection_refresh',affected_exposures=len(ids),nodes=len(nodes))


def run(args):
    folder=args.output
    folder.mkdir(exist_ok=True)
    log(folder,'start',parent=str(args.parent),opening=POLICY,nominal_cm=153.)
    evidence=args.programme/'fit_v11/sitspin_pixels_1119.json'
    doc=json.loads(evidence.read_text('utf8'))
    source=json.loads((args.parent/'source.json').read_text('utf8'))
    if doc['source_sha256']!=source['source_sha256']:
        raise ValueError('Runner review source mismatch')
    source['sitspin_source_review']=dict(path=str(evidence),sha256=digest(evidence),
        parent=str(args.parent/'candidate.npz'),parent_sha256=digest(args.parent/'candidate.npz'),
        edited_local_joints=[1,4,7],scope_seconds=doc['scope_seconds'])
    if args.turn_ends:
        extra=json.loads(args.turn_ends.read_text('utf8'))
        if extra['source_sha256']!=doc['source_sha256']:raise ValueError('Different turn evidence')
        source['additional_turn_ends']=dict(path=str(args.turn_ends.resolve()),sha256=digest(args.turn_ends))
    ledger=folder/'body_observations.npz'
    if not ledger.exists():shutil.copyfile(args.parent/ledger.name,ledger)
    if digest(ledger)!=digest(args.parent/ledger.name):raise ValueError('Observation ledger mismatch')
    (folder/'source.json').write_text(json.dumps(source,ensure_ascii=False,separators=(',',':')),'utf8')
    stage=WorldMotion(args.prepared,args.parent,ledger,folder)
    stage.source_factors(args.work,args.shared)
    pts=stage.ref.p['pts']
    t=stage.stamps/12800
    start,end=doc['scope_seconds']
    in_spin=(t>=start-1.e-7)&(t<=end+1.e-7)
    assert stage.support[in_spin,0].all()
    stage.active_knots=np.zeros(len(stage.seed.position_controls),bool)
    control_scope=in_spin|((t>end)&(t<=end+.10)) if args.finish else in_spin
    stage.active_knots[np.unique(stage.B[control_scope].indices)]=True
    stage.local_fit_rows=np.asarray(stage.B[:,stage.active_knots].getnnz(axis=1)).ravel()>0
    stage.exact_rotation_basis=args.exact
    stage.differentiate_rocker=args.exact
    if args.native_slip:
        if not args.exact:raise ValueError('Native steel slip needs the local rocker Jacobian')
        stage.native_slip_scale=args.native_slip
    stage.edge_review=True
    stage.protect[:]=0
    stage.protect[:,13]=.55
    stage.knee_weight=args.knee_weight
    stage.preserve_step_direction=True
    stage.left_knee_world_weight=np.where(in_spin,args.knee_world_weight,0.)
    stage.left_outline_weight=args.outline_weight
    stage.shoe_source_weight=4.
    stage.com_weight=args.com_weight
    stage.root_trust=0.
    stage.support_slip_scale=np.full(stage.support.shape,.08)
    stage.support_slip_scale[in_spin,0]=args.slip
    stage.review_contact_weight=np.full(stage.support.shape,1800.)
    stage.review_contact_weight[in_spin,0]=args.contact_weight
    # A soft upper bound on either end still permits a rocked blade. Tighten
    # only around the independently visible side-on exposure, not at end-on views.
    stage.runner_height_cap=np.where(abs(t-177.54)<=.041,.015,.045)
    stage.runner_cap_weight=np.where(in_spin,2200.,0.)
    if args.finish:
        stage.review_contact_weight[in_spin&(t>=178.92),0]=110000.
        stage.runner_cap_weight[abs(t-177.54)<=.041]=11000.
        stage.runner_height_cap[abs(t-177.54)<=.041]=.014
        with np.load(args.finish/'joint_source_measurements.npz') as z:
            if str(z['candidate'])!=digest(args.finish/'candidate.npz'):raise ValueError('Outline prior mismatch')
            projected=z['shoe'][:,0,7:]
        stage.outline_reference_px=np.abs((projected@stage.dirs.T).max(1)-stage.ext[:,0])+.75
        stage.outline_reference_px[~in_spin[::2]]=np.inf
    rows=[dict(r,weight=6.5 if not r['end_on'] else 3.) for r in doc['records'] if r['role']=='fit']
    if args.turn_ends:
        rows.extend(dict(r,weight=22.) for r in extra['records'])
    for row in rows:
        if row['pts']==2272512 and not row['end_on']:row['weight']*=args.focus_weight
    if args.finish:
        # The screenshot exposure shows both termini separately. A weak
        # midpoint estimate complements its wider digitization intervals.
        for row in list(rows):
            if row['pts']!=2272512:continue
            centre=np.mean(row['box_px'],axis=0)
            rows.append(dict(row,box_px=[centre-1.5,centre+1.5],weight=row['weight']*.65))
    rows.extend(r for r in load_regions(args.programme) if r['role']=='fit' and
                r['kind']=='joint' and r['feature']==13 and start<=r['pts']/12800<=end)
    def regions(rows):
        fr=np.searchsorted(pts,[r['pts'] for r in rows])
        np.testing.assert_array_equal(pts[fr],[r['pts'] for r in rows])
        boxes=np.array([r['box_px'] for r in rows])
        return fr,np.array([r['feature'] for r in rows]),boxes[:,0],boxes[:,1],np.array([r['weight'] for r in rows])
    stage.reviewed_image_regions=regions(rows)
    hold=regions([dict(r,weight=1.) for r in doc['records'] if r['role']=='check'])
    j=int(np.searchsorted(pts,round(177.54*12800)))
    def photo_metrics(curve):
        st=stage.state(curve)
        uv=project(stage.cameras,st['photo'],stage.camera_scale)+stage.offset
        def boxes(reg):
            fr,idx,lo,hi,_=reg
            x=uv[fr,idx]
            return np.percentile(np.linalg.norm(x-np.clip(x,lo,hi),axis=-1),[50,95]).tolist()
        return dict(training_region_px=boxes(stage.reviewed_image_regions),withheld_px=boxes(hold),
            at_17754_uv=uv[j,20:22].tolist(),ends_mm=(st['shoes'][2*j,0,3:5,2]*1000).tolist())
    nominal=100*np.ptp(stage.ref.binding.surface.shaped[:,1])*stage.ref.body.scale
    np.testing.assert_allclose(nominal,153.,atol=.001,rtol=0)
    identity=digest(args.parent/'candidate.npz')
    curve=stage.seed
    checkpoint=folder/'candidate.npz'
    cycle=0
    initial=checkpoint if checkpoint.exists() else args.initial
    if initial:
        with np.load(initial) as z:
            info=json.loads(str(z['joint_source_fit_json']))
            expected_source=(initial.parent/'source.json') if initial==args.initial else folder/'source.json'
            if info['parent_sha256']!=identity or str(z['source_evidence_sha256'])!=digest(expected_source):
                raise ValueError('Resume source mismatch')
            cycle=info['cycle']+1 if checkpoint.exists() else 0
            curve=curve_from_packet(z)
    elif args.pose_seed:
        with np.load(args.pose_seed) as z:prior=curve_from_packet(z)
        np.testing.assert_array_equal(prior.knots,curve.knots)
        clock=greville_times(curve.knots)+pts[0]/12800
        blend=np.clip(np.minimum((clock-start+.04)/.10,(end-clock+.04)/.10),0.,1.)
        blend*=stage.active_knots
        blend=blend*blend*(3-2*blend)
        rot=curve.rotation_controls.copy()
        active=blend>0
        for joint in (1,4,7):
            old=rot[active,joint]
            delta=_log(old.swapaxes(-1,-2)@prior.rotation_controls[active,joint])
            rot[active,joint]=old@_exp(delta*blend[active,None])
        curve=AbsoluteMotionCurve(curve.knots,curve.position_controls,rot,curve.edge_reference_quaternion)
        curve.check_continuous_edge_update(stage.seed)
    if args.refresh:refresh_photo(stage,curve,args)
    log(folder,'initial_check',**photo_metrics(curve))
    while True:
        curve,gain=stage.optimize(curve,[1,4,7])
        report=dict(stage='sitspin_refit',mode='fixed',cycle=cycle,improvement=gain,
            nominal_cm=nominal,body_scale=stage.ref.body.scale,camera='s14_boards_1118',ice='lane4_side_1118',
            parent_sha256=identity,camera_sha256=stage.cam.info['state_sha256'],
            programme_camera_registration_sha256=stage.ref.source_document['programme_camera_candidate']['record_sha256'],
            source_exposures=len(pts),review_sha256=digest(evidence),support_slip_scale=args.slip,
            contact_weight=args.contact_weight,knee_weight=args.knee_weight,outline_weight=args.outline_weight,
            knee_world_weight=args.knee_world_weight,initial_sha256=digest(args.initial) if args.initial else None,
            pose_seed_sha256=digest(args.pose_seed) if args.pose_seed else None,
            exact_rotation_basis=args.exact,focus_weight=args.focus_weight,
            outline_reference_sha256=digest(args.finish/'candidate.npz') if args.finish else None,
            native_slip_scale=args.native_slip,extra_turn_sha256=digest(args.turn_ends) if args.turn_ends else None,
            com_weight=args.com_weight,refreshed_projection=args.refresh,
            new_fit_regions=sum(r['role']=='fit' for r in doc['records']),**photo_metrics(curve))
        save(folder,'candidate.npz',stage.packet,curve,stage.ref,report)
        if gain<.000003:break
        cycle+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('prepared','parent','work','shared','programme','output'):
        parser.add_argument('--'+key,type=lambda v:Path(v).resolve(),required=True)
    parser.add_argument('--slip',type=float,default=.055)
    parser.add_argument('--contact-weight',type=float,default=6500.)
    parser.add_argument('--knee-weight',type=float,default=1.85)
    parser.add_argument('--knee-world-weight',type=float,default=0.)
    parser.add_argument('--outline-weight',type=float,default=1.)
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--pose-seed',type=Path)
    parser.add_argument('--exact',action='store_true')
    parser.add_argument('--focus-weight',type=float,default=1.)
    parser.add_argument('--finish',type=Path)
    parser.add_argument('--turn-ends',type=Path)
    parser.add_argument('--native-slip',type=float)
    parser.add_argument('--com-weight',type=float,default=1.3)
    parser.add_argument('--refresh',action='store_true')
    run(parser.parse_args())
