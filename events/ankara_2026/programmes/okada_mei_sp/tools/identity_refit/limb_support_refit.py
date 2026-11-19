"""Refine the fixed-stature programme using source limb and steel regions and reviewed ice support."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from source_joint_refit import POLICY, log, save
from source_review import digest, whole_skate_observations
from world_motion_refit import WorldMotion


def load_regions(programme):
    revised = programme / 'fit_v11/limb_support_1119.json'
    original = whole_skate_observations(programme)
    source = json.loads(revised.read_text('utf8'))
    archived = json.loads(original.read_text('utf8'))
    records = [dict(r, origin='direct') for r in source['rows']]
    # Interpolate short, visible tracks inside a shot. These are correlated
    # constraints, with additional uncertainty, not independently read pixels.
    tracks = defaultdict(list)
    for r in records:
        if r['kind']=='joint' and (107.70 <= r['pts']/12800 <= 107.90 or 160.97 <= r['pts']/12800 <= 161.15):
            tracks[r['feature']].append(r)
    withheld = {(r['pts'],r['feature']) for r in records if r['role']=='check'}
    for joint, track in tracks.items():
        fit = sorted((r for r in track if r['role']=='fit'), key=lambda r:r['pts'])
        for first,last in zip(fit,fit[1:]):
            duration=last['pts']-first['pts']
            if duration>1024:continue
            for stamp in range(first['pts']+256,last['pts'],256):
                if (stamp,joint) in withheld:continue
                f=(stamp-first['pts'])/duration
                region=(1-f)*np.array(first['box_px'])+f*np.array(last['box_px'])
                region+=np.array([[-5,-5],[5,5]])
                records.append(dict(pts=stamp,feature=joint,box_px=region.tolist(),role='fit',
                                    origin='short_track',weight=2.5,kind='joint'))
    for key in ('body_samples','body_regions'):
        for r in archived[key]:
            if r['role']!='fit':continue
            if key=='body_regions':region=r['pixel_box']
            else:
                radius=min(6.,r['normalization_px']*.4)
                region=(np.array(r['uv'])[None]+np.array([[-radius,-radius],[radius,radius]])).tolist()
            records.append(dict(pts=r['pts'],feature=r['joint'],box_px=region,role='fit',
                                origin=key,weight=2.2,kind='joint'))
    return records


def prepare(args):
    records=load_regions(args.programme)
    if getattr(args,'runner_review',None):
        runner=json.loads(args.runner_review.read_text('utf8'))
        camera=json.loads((args.programme/'s14_boards_1118.json').read_text('utf8'))
        if runner['source_sha256']!=camera['source']['sha256']:raise ValueError('Different runner footage')
        records.extend(runner['rows'])
    if getattr(args,'turn_review',None):
        turn=json.loads(args.turn_review.read_text('utf8'))
        if turn['source_sha256']!=runner['source_sha256']:raise ValueError('Different turn footage')
        records.extend(turn['rows'])
    with np.load(args.shared/'final_body_observations.npz') as z:ledger={k:z[k] for k in z}
    pts=ledger['pts']; ids=ledger['model_joint_ids'];disabled=[]
    fit=[r for r in records if r['role']=='fit']
    for r in fit:
        i=int(np.searchsorted(pts,r['pts']))
        if pts[i]!=r['pts']:raise ValueError(r)
        if r['kind']!='joint':continue
        slot=int(np.flatnonzero(ids[i]==r['feature'])[0]);u=ledger['uv'][i,slot]
        delta=np.linalg.norm(u-np.clip(u,*r['box_px']))
        if ledger['use'][i,slot] and delta>max(10.,2*ledger['sigma_px'][i,slot]):
            disabled.append(dict(pts=r['pts'],slot=slot,model_joint=r['feature'],
                                 detected_uv=u.tolist(),distance_outside_review_px=float(delta),origin=r['origin']))
            ledger['use'][i,slot]=False
    ledger['source_limb_regions_revision_json']=np.array(json.dumps(disabled))
    path=args.output/'body_observations.npz'
    with path.open('wb') as file:np.savez_compressed(file,**ledger)
    source=json.loads((args.parent/'source.json').read_text('utf8'))
    parent_ledger=args.shared/'final_body_observations.npz'
    revision=args.programme/'fit_v11/limb_support_1119.json'
    # Register the revised arrays against the prepared native observations.
    # The original per-slot sources and each later admission revision remain
    # present in the NPZ metadata and in the previous source document.
    original=args.prepared/'observations.npz'
    with np.load(parent_ledger) as observed,np.load(original) as raw:
        np.testing.assert_array_equal(observed['pts'],raw['pts'])
        assert observed['uv'].shape==raw['uv'].shape
    prior=dict(path=str(parent_ledger),sha256=digest(parent_ledger),
               parent_observations_sha256=digest(original),prepared_original_path=str(original),
               source_revision_archive=str(args.parent/'source.json'),
               source_track_inputs=[str(args.work/'input/body_track_source.npz'),str(args.shared/'body_observations.npz')])
    source['programme_body_observation_revision']=dict(path=str(path),sha256=digest(path),previous_revision=prior,
            parent_observations_sha256=digest(parent_ledger),parent_path=str(parent_ledger),
            source_video_sha256=source['source_sha256'],time_base=[1,12800],
            review_path=str(revision),review_sha256=digest(revision))
    source['limb_steel_review']=dict(path=str(revision),sha256=digest(revision),disabled=disabled,
                                    fit_regions=len(fit),retained_source_observations=int(ledger['use'].sum()))
    if getattr(args,'support_review',None):
        extra=json.loads(args.support_review.read_text('utf8'))
        if extra['source_sha256']!=source['source_sha256']:raise ValueError('Different footage in support review')
        existing={e['episode_id']:e for e in source['source_episode_review']['episodes']}
        for e in extra['episodes']:
            if e['episode_id'] in existing:
                if existing[e['episode_id']]!=e:raise ValueError('Different source support range')
            else:
                source['source_episode_review']['episodes'].append(e)
        source['limb_steel_review']['extra_support_review_sha256']=digest(args.support_review)
    if getattr(args,'runner_review',None):
        source['limb_steel_review']['runner_review']=str(args.runner_review)
        source['limb_steel_review']['runner_review_sha256']=digest(args.runner_review)
    if getattr(args,'turn_review',None):
        source['limb_steel_review']['turn_review']=str(args.turn_review)
        source['limb_steel_review']['turn_review_sha256']=digest(args.turn_review)
    pending=args.output/'source.pending.json'
    pending.write_text(json.dumps(source,ensure_ascii=False,separators=(',',':')),'utf8')
    pending.replace(args.output/'source.json')
    boxes=np.asarray([r['box_px'] for r in fit])
    factors=(np.searchsorted(pts,[r['pts'] for r in fit]), np.array([r['feature'] for r in fit]),
             boxes[:,0],boxes[:,1],np.array([r['weight'] for r in fit]))
    return path,factors


def run(args):
    args.output.mkdir(exist_ok=True)
    log(args.output,'start',parent=str(args.parent),nominal_cm=153.,opening=POLICY)
    ledger,regions=prepare(args)
    stage=WorldMotion(args.prepared,args.parent,ledger,args.output)
    stage.source_factors(args.work,args.shared)
    if args.support_review:
        review=json.loads(args.support_review.read_text('utf8'))
        for episode in review['episodes']:
            inside=(stage.stamps>=episode['start_pts']-.01)&(stage.stamps<=episode['end_pts_inclusive']+.01)
            stage.support[inside,episode['foot']]=True
    stage.reviewed_image_regions=regions
    stage.edge_review=True
    stage.root_trust=0.
    stage.protect[:]=0
    stage.support_slip_scale=np.full(stage.support.shape,.080)
    t=stage.stamps/12800
    if args.support_review:
        stage.review_contact_weight=np.full(stage.support.shape,1800.)
    for start,end,foot,sigma in [(76.70,76.90,1,.027),(174.76,179.02,0,.035)]:
        subset=(t>=start-1.e-8)&(t<=end+1.e-8)
        assert stage.support[subset,foot].all()
        stage.support_slip_scale[subset,foot]=sigma
        if args.support_review:
            stage.review_contact_weight[subset,foot]=10000. if foot==1 else 4500.
    nominal=100*np.ptp(stage.ref.binding.surface.shaped[:,1])*stage.ref.body.scale
    np.testing.assert_allclose(nominal,153,atol=.001,rtol=0)
    identity=digest(args.parent/'candidate.npz')
    curve=stage.seed;cycle=0;checkpoint=args.output/'candidate.npz'
    if args.runner_review:
        windows=json.loads(args.support_review.read_text('utf8'))['episodes']
        windows=windows+[e for e in stage.episodes if e['episode_id'] in ('support_76_8','ccosp_left_sit')]
        # Preserve the rest of the programme exactly. The padding permits the
        # adjacent free leg and root spline to join continuously on both sides.
        selected=np.zeros(len(t),bool)
        for e in windows:
            foot=e['foot'];t0=e['start_pts']/12800;t1=e['end_pts_inclusive']/12800
            keep=(t>=t0-1.e-8)&(t<=t1+1.e-8)
            selected|=(t>=t0-.32)&(t<=t1+.32)
            stage.support_slip_scale[keep,foot]=.014 if e['episode_id']=='ccosp_left_sit' else .010
            stage.review_contact_weight[keep,foot]=max(6500.,stage.review_contact_weight[keep,foot].max())
        if not args.balance:
            stage.active_knots=np.zeros(len(curve.position_controls),bool)
            stage.active_knots[np.unique(stage.B[selected].indices)]=True
    if args.turn_review:
        turn=json.loads(args.turn_review.read_text('utf8'))
        selected=np.zeros(len(t),bool)
        for start,end in turn['window_seconds']:
            selected|=(t>=start-.16)&(t<=end+.16)
            region=(t>=start-.021)&(t<=end+.021)
            stage.support_slip_scale[region,0]=args.turn_slip
        stage.active_knots=np.zeros(len(curve.position_controls),bool)
        stage.active_knots[np.unique(stage.B[selected].indices)]=True
    if args.balance or args.turn_review:
        stage.knee_weight=1.85
        stage.shoe_source_weight=2.20
        stage.com_weight=1.30
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            info=json.loads(str(z['joint_source_fit_json']))
            if (info['parent_sha256']!=identity or str(z['source_evidence_sha256'])!=digest(args.output/'source.json')):
                raise ValueError('Resume input changed')
            curve=curve_from_packet(z);cycle=info['cycle']+1
            if args.balance and not info.get('source_balance'):cycle=0
            if args.turn_review and info.get('turn_slip',.070)!=args.turn_slip:cycle=0
    if args.balance and not checkpoint.exists():raise ValueError('Balance requires a support checkpoint')
    while True:
        gain=0.
        groups=([-1,1,4,7],[-1,2,5,8],[-1,0],[3,6,9,12],[13,16,18,20],[14,17,19,21])
        if args.runner_review and not args.balance:
            groups=([-1],[-1,0,1,4,7],[-1,0,2,5,8],[7,8],[-1])
        if args.turn_review:
            groups=([-1,0,1,4,7],[1,4,7],[-1,0],[7],[-1])
        for group in groups:
            curve,change=stage.optimize(curve,group);gain+=change
        report=dict(stage='source_support_and_limbs',mode='fixed',cycle=cycle,improvement=gain,source_balance=args.balance,turn_review=bool(args.turn_review),
                    turn_slip=args.turn_slip if args.turn_review else None,
                    nominal_cm=nominal,body_scale=stage.ref.body.scale,camera='s14_boards_1118',ice='lane4_side_1118',
                    parent_sha256=identity,camera_sha256=stage.cam.info['state_sha256'],
                    programme_camera_registration_sha256=stage.ref.source_document['programme_camera_candidate']['record_sha256'],
                    source_exposures=len(stage.ref.times),region_count=len(regions[0]))
        save(args.output,'candidate.npz',stage.packet,curve,stage.ref,report)
        if cycle and gain<(.00025 if args.runner_review and not args.balance else .001):break
        cycle+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('prepared','programme','shared','work','parent','output'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    parser.add_argument('--support-review',type=lambda v:Path(v).resolve())
    parser.add_argument('--runner-review',type=lambda v:Path(v).resolve())
    parser.add_argument('--balance',action='store_true')
    parser.add_argument('--turn-review',type=lambda v:Path(v).resolve())
    parser.add_argument('--turn-slip',type=float,default=.070)
    run(parser.parse_args())
