"""Okada programme refit using frozen s14 calibration and the verified opening condition."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compare import bind
from source_review import digest
from source_camera_rebind import write
from opening_support import enforce, verify
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, greville_times, _exp, _log
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet, evaluate_curve

START = time.perf_counter()
POLICY = dict(pts=[283136,306688], time_base=[1,12800], feet=['left','right'], plane='z=0',
              grid_hz=400, tolerance_m=.001, provenance='用户确认 22.12–23.96 秒左右鞋刀持续着冰。')


def log(out, event, **values):
    row = dict(event=event, elapsed_s=round(time.perf_counter()-START,3), **values)
    with (out/'fit.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(event,values,flush=True)


def save(out, name, packet, curve, ref, provenance):
    report=verify(ref,curve,POLICY)
    values=dict(packet, absolute_position_controls=curve.position_controls,
        absolute_rotation_controls=curve.rotation_controls, absolute_seed_position_controls=curve.position_controls,
        absolute_seed_rotation_controls=curve.rotation_controls,
        absolute_rotation_edge_reference_quaternion=curve.edge_reference_quaternion,
        source_evidence_sha256=np.array(digest(out/'source.json')),
        programme_double_support_json=np.array(json.dumps(POLICY,ensure_ascii=False)),
        joint_source_fit_json=np.array(json.dumps(dict(provenance,contact=report),ensure_ascii=False)))
    path=out/name
    temp=path.with_suffix('.pending.npz')
    np.savez_compressed(temp,**values);temp.replace(path)
    log(out,'accepted_checkpoint',path=str(path),contact=report,**provenance)


def init(args):
    source=json.loads((args.parent/'source.json').read_text(encoding='utf8'))
    conf=source['scoped_candidate_import']['opening_lunge']
    if conf['right_support_pts']!=POLICY['pts'] or conf['left_support_pts'][1]<POLICY['pts'][1]:
        raise ValueError('Opening evidence clock mismatch')
    source['programme_double_support']=POLICY
    write(args.output/'source.json',source)
    ref,cam,info,_,pkt=bind(args.prepared,args.parent/'source.json',args.parent/'candidate.npz')
    curve=curve_from_packet(pkt)
    with np.load(args.early/'solve/candidate.npz') as z:
        prior=curve_from_packet(z)
    g=greville_times(curve.knots);clock=g+ref.p['pts'][0]/12800
    # Use the supported early stance as an initialization; the continuous controls
    # stay editable. Photo-calibrated horizontal displacement is retained.
    weight=np.clip((25.00-clock)/(.77),0,1)
    weight=weight**2*(3.-2.*weight)
    ids=np.flatnonzero(weight>0)
    donor=evaluate_curve(prior,g[ids])
    rot=curve.rotation_controls.copy();pos=curve.position_controls.copy()
    rel=_log(rot[ids].swapaxes(-1,-2)@donor['rotations'])
    rot[ids]=rot[ids]@_exp(rel*weight[ids,None,None])
    pos[ids,2]+=(donor['P'][:,2]-pos[ids,2])*weight[ids]
    chart=curve.edge_reference_quaternion.copy()
    q=Rotation.from_matrix((rot[:-1].swapaxes(-1,-2)@rot[1:]).reshape(-1,3,3)).as_quat().reshape(chart.shape)
    updated=np.arange(len(chart))<=ids[-1]
    chart[updated]=q[updated]*np.where(q[updated,...,3:4]>=0,1.,-1.)
    new=AbsoluteMotionCurve(curve.knots,pos,rot,chart)
    new=enforce(ref,new,POLICY,lambda e,**v:log(args.output,e,**v))
    save(args.output,'supported_seed.npz',pkt,new,ref,dict(stage='supported_initialization',parent=digest(args.parent/'candidate.npz')))


def review_arm(args):
    from source_continuous_refine import stage
    src=args.output/'refined_candidate.npz'
    ref,cam,_,_,packet=bind(args.prepared,args.output/'source.json',src)
    previous=curve_from_packet(packet)
    with np.load(args.output/'body_observations.npz') as z:
        ledger={k:z[k] for k in z.files}
    pts=ledger['pts'];ids=ledger['model_joint_ids']
    omissions=[]
    for ji,begin,end in [(7,63.42,63.48),(9,63.44,63.48)]:
        bad=(pts[:,None]>=round(begin*12800))&(pts[:,None]<=round(end*12800))&(ids==ji)
        ledger['use'][bad]=False
        omissions.append(dict(joint=ji,source_seconds=[begin,end],reason='The detector places the raised left limb on the exposed back.'))
    # Wrist boxes read from native consecutive frames. The hand stays on/above
    # the head through occlusion; the hidden elbow has only an admissible region.
    wrists=[(63.40,891,369),(63.42,909,371),(63.44,927,373),(63.46,940,375),
            (63.48,955,375),(63.50,969,378),(63.52,982,380)]
    points=dict(body_samples=[],body_regions=[],samples=[])
    for t,x,y in wrists:
        stamp=round(t*12800)
        points['body_regions'].append(dict(id=f'left_wrist_head_{stamp}',joint=9,pts=stamp,
            role='fit',pixel_box=[[x-10,y-10],[x+10,y+10]],normalization_px=5.,weight=6.))
        if 63.42<=t<=63.5:
            points['body_regions'].append(dict(id=f'partly_occluded_left_elbow_{stamp}',joint=7,pts=stamp,
                role='fit',pixel_box=[[x-43,394],[x+32,465]],normalization_px=8.,weight=5.))
    spec=dict(name='left_arm_passing_behind_head',seconds=[63.32,63.62],joints=[13,16,18,20],
        source_joints=[5,7,9],root_position=False,fit_shoe_points=False,
        motion_acceleration_weight=12.,control_update_weight=2.,detector_robust_loss='pseudo_huber')
    source_revision=dict(omissions=omissions,regions=points['body_regions'],stage=spec,
        source_sha256=ref.source_document['source_sha256'],source_sheet=str(args.output/'source_left_wrist.jpg'),
        semantics='Coordinates and boxes from the original image; occluded elbow position is not directly observed.')
    write(args.output/'arm_review.json',source_revision)
    ledger['arm_source_review_json']=np.array(json.dumps(source_revision,ensure_ascii=False))
    np.savez_compressed(args.output/'final_body_observations.npz',**ledger)
    def emit(e,**v):
        log(args.output,e,**{k:val for k,val in v.items() if k!='metrics'})
    refined=stage(ref,cam,packet,previous,points,ledger,spec,emit)
    frozen=np.ones(22,bool);frozen[spec['joints']]=False
    np.testing.assert_array_equal(refined.rotation_controls[:,frozen],previous.rotation_controls[:,frozen])
    np.testing.assert_array_equal(refined.position_controls,previous.position_controls)
    np.testing.assert_array_equal(refined._check_log_chart()[:,frozen],previous._check_log_chart()[:,frozen])
    save(args.output,'delivery_candidate.npz',packet,refined,ref,dict(stage='left_arm_source_revision',
        full_programme_parent=json.loads(str(packet['joint_source_fit_json'])),parent_sha256=digest(src),
        reviewed_source_sha256=digest(args.output/'arm_review.json'),ledger_sha256=digest(args.output/'final_body_observations.npz')))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('step',choices=['init','fit','refine','arm','legs'])
    for arg in ('output','parent','early','prepared','work','programme'):
        ap.add_argument('--'+arg,type=lambda s:Path(s).resolve(),required=True)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    log(args.output,'start',step=args.step,output=str(args.output))
    if args.step=='init':
        init(args)
    elif args.step=='arm':
        review_arm(args)
    else:
        from joint_source_objective import run
        try:
            run(args)
        except Exception as error:
            log(args.output,'fit_interrupted',reason=repr(error))
            raise


if __name__=='__main__':
    main()
