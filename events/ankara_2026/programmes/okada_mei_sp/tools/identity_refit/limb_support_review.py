"""Source-space and world-space before/after checks on matching native exposures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from compare import bind
from joint_refit_review import crop, errors, metrics
from joint_source_objective import material_points, source_shoe_rows
from source_review import SourceVideo, digest, whole_skate_observations
from world_motion_review import paths, physics, phase_details
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state


def distances(records, predictions, pts):
    index=np.searchsorted(pts,[r['pts'] for r in records])
    feature=[r['feature'] for r in records]
    box=np.array([r['box_px'] for r in records]);u=predictions[index,feature]
    return np.linalg.norm(u-np.clip(u,box[:,0],box[:,1]),axis=1)


def draw_pose(img, uv, shoe):
    draw=ImageDraw.Draw(img)
    for a,b in [(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:
        draw.line([tuple(x) for x in uv[[a,b]]],fill='#20dcce' if b%2 else '#ffa038',width=3)
    for f,color in enumerate(['#20dcce','#ffa038']):
        xy=shoe[f,7:]; hull=ConvexHull(xy).vertices
        draw.line([tuple(x) for x in xy[np.r_[hull,hull[0]]]],fill=color,width=2)
    return img


def contact_path(ref,episode,hz):
    foot=episode['foot']
    start=episode['start_pts'];end=episode['end_pts_inclusive']
    times=(np.arange(start,end+1,12800//hz)-ref.p['pts'][0])/12800
    state=fk_state(ref,ref.at(times),np.zeros((len(times),69)))
    R=state['R'][:,foot];O=state['O'][:,foot]*ref.binding.ratio
    feature=batch_features(ref.binding,state['R'],state['O'])
    steel=np.isin(feature['semantics'],['runner','toe_pick'])
    z=feature['height'][:,foot,steel];idx=np.argmin(z,axis=1)
    local=feature['local_point'][:,foot][:,steel][np.arange(len(times)),idx]*ref.binding.ratio
    velocity=(O[2:]-O[:-2]+((R[2:]-R[:-2])@local[1:-1,:,None])[:,:,0])*hz*.5
    axis=R[1:-1,:2,0];axis=axis/np.linalg.norm(axis,axis=1,keepdims=True)
    side=np.stack([-axis[:,1],axis[:,0]],1)
    lateral=abs((velocity[:,:2]*side).sum(1))
    return dict(interior_samples=len(lateral),absolute_lateral_speed_mps=metrics(lateral),
                lateral_travel_interior_m=float(lateral.sum()/hz),
                clearance_absolute_mm=metrics(1000*abs(z.min(1))))


def compare(args):
    out=args.output; shots=np.load(args.work/'input/shoe_occlusion_source.npz')
    pts=shots['source_pts'];shot=shots['shot_ids'];shot_ids=list(dict.fromkeys(shot))
    dirs=np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1)
    good=(shots['role']==1)&shots['visible']&~shots['partial_boundary']&(shots['source_pair_overlap'][:,None]<.12)
    ext=np.max(np.nan_to_num(shots['region'])@dirs.T,axis=2)
    old_doc=json.loads(whole_skate_observations(args.programme).read_text('utf8'))
    for r in old_doc.get('source_outline_review_exclusions',[]):
        if 'pts' in r and 'foot' in r:good[np.searchsorted(pts,r['pts']),r['foot']]=False
    with np.load(args.shared/'before_source_measurements.npz') as inp:rows=source_shoe_rows(old_doc,pts,inp['shoe'])
    groups=json.loads((args.programme/'fit_v11/limb_support_1119.json').read_text('utf8'))['rows']
    review=json.loads((out/'source.json').read_text('utf8'))['limb_steel_review']
    if 'runner_review' in review:
        runner=Path(review['runner_review'])
        assert digest(runner)==review['runner_review_sha256']
        groups+=json.loads(runner.read_text('utf8'))['rows']
    if 'turn_review' in review:
        turn_path=Path(review['turn_review']);assert digest(turn_path)==review['turn_review_sha256']
        turn=json.loads(turn_path.read_text('utf8'))
        fitted={(r['pts'],r['feature']) for r in groups if r['role']=='fit'}
        for r in groups:
            if r['kind']=='steel' and r['pts'] in turn['reopened_check_pts'] and r['role']=='check':
                r['role']='development_recheck'
        for r in turn['rows']:
            if r['role']=='check' and (r['pts'],r['feature']) in fitted:r['role']='development_recheck'
        groups+=turn['rows']
    episodes=json.loads((out/'source.json').read_text('utf8'))['source_episode_review']['episodes']
    final_ledger=np.load(out/'body_observations.npz');parent_ledger=np.load(args.shared/'final_body_observations.npz')
    assert np.array_equal(final_ledger['pts'],pts)
    report=dict(frames=len(pts),shots=shot_ids,source_seconds=(pts[[0,-1]]/12800).tolist(),
                contact_evidence='single-view inferred except the user-confirmed opening',candidates={})
    states={};per_errors={};view=json.loads((args.programme/'s14_boards_1118.json').read_text('utf8'))
    for label,folder in [('fixed153',args.parent),('revised',out)]:
        ref,cam,info,_,packet=bind(args.prepared,folder/'source.json',folder/'candidate.npz')
        with np.load(folder/'joint_source_measurements.npz') as m:
            assert str(m['candidate'])==digest(folder/'candidate.npz')
            state={k:m[k] for k in ['uv','shoe','height','angles']}
        states[label]=state;predictions=np.concatenate((state['uv'],state['shoe'][:,:,:7].reshape(len(pts),14,2)),axis=1)
        all_obs=errors(label,state,final_ledger,rows,ext,good,dirs);per_errors[label]=all_obs
        historical=errors(label,state,parent_ledger,rows,ext,good,dirs)
        e=distances(groups,predictions,pts)
        def region_summary(selection):
            vals=e[np.array(selection,bool)]
            return dict(**metrics(vals),mean=float(np.mean(vals))) if len(vals) else {}
        regional={}
        for role in ('fit','check'):
            for cat in ('all','joint','steel','knees','arms'):
                regional[role+'_'+cat]=region_summary([r['role']==role and (cat=='all' or r['kind']==cat or
                    cat=='knees' and r['feature'] in (13,14) or cat=='arms' and r['feature'] in (7,8,9,10)) for r in groups])
        for key,(start,end) in {'right_runner':(76.70,76.90),'sitspin':(174.76,179.02),'waist_hand':(107.72,107.88),'layback_arms':(160.98,161.14)}.items():
            for cat in ('joint','steel'):
                regional[key+'_'+cat]=region_summary([start-1.e-7<=r['pts']/12800<=end+1.e-7 and r['kind']==cat for r in groups])
        with np.load(folder/'result.npz') as z:
            np.testing.assert_array_equal(z['source_pts'],pts);com=z['com_world_mid']
            np.testing.assert_allclose(z['com_projection_world_mid'][:,:2],com[:,:2])
            np.testing.assert_allclose(z['com_projection_world_mid'][:,2],0)
            blade=z['skate_origin_world_mid'][:,:,None]+np.einsum('nfij,kj->nfki',z['skate_R_world_mid'],material_points(ref.binding)[[3,4]])*ref.binding.ratio
            physical=physics(com,blade,state['height'],pts,shot,episodes)
            phases=phase_details(z,episodes)
            angular=z['native_body_local_R_mid']
            shared_scale=float(z['shared_body_scale'])
            np.testing.assert_allclose(shared_scale,float(packet['fitted_programme_body_scale']))
            assert str(z['transfer_parameter_sha256'])==digest(folder/'candidate.npz')
        t=np.linspace(ref.times[0],ref.times[-1],4*len(pts)-3)
        c,b,h=paths(ref,ref.absolute_curve,t)
        np.testing.assert_allclose(b[::4],blade,atol=2e-8)
        np.testing.assert_allclose(c[::4],com,atol=2e-8)
        dense=physics(c,b,h,pts[0]+np.arange(len(t))*64,shot[np.arange(len(t))//4],episodes)
        report['candidates'][label]=dict(sha256=digest(folder/'candidate.npz'),camera_sha256=info['state_sha256'],
            nominal_cm=100*shared_scale*np.ptp(ref.binding.surface.shaped[:,1]),physics=physical,
            physics_200hz=dense,phase=phases,
            support_200hz={ep['episode_id']:contact_path(ref,ep,200) for ep in episodes
                           if ep['kind']=='support_candidate' and ep.get('foot') in (0,1)},
            source_regions=regional,all_native={k:metrics(v) for k,v in all_obs.items()},
            original_population={k:metrics(v) for k,v in historical.items()},
            per_shot={s:{k:metrics(v[shot==s]) for k,v in all_obs.items()} for s in shot_ids},
            region_error_px=e.tolist(),fit=json.loads(str(packet['joint_source_fit_json'])))
    a=report['candidates']['fixed153'];b=report['candidates']['revised']
    assert a['camera_sha256']==b['camera_sha256'] and abs(a['nominal_cm']-b['nominal_cm'])<1.e-8
    visual=out/'source_review';visual.mkdir(exist_ok=True)
    video=SourceVideo(view['source']['path'],(1,12800))
    chronological=[];checks=[]
    early=np.load(args.shared/'early_source_measurements.npz')
    states['early']={k:early[k] for k in ['uv','shoe']}
    for s in shot_ids:
        ids=np.where(shot==s)[0]
        for win in np.array_split(ids,max(1,int(np.ceil(len(ids)/200)))):
            # The review sample includes the largest remaining limb residual in
            # each consecutive four-second window plus separately requested times.
            val=per_errors['revised'];risk=np.nan_to_num(val['arms'][win],nan=0.).max(1)+np.nan_to_num(val['knees'][win],nan=0.).max(1)
            i=win[np.argmax(risk)];checks.append(i)
            chronological.append(dict(shot=str(s),range_pts=pts[win[[0,-1]]].tolist(),example_pts=int(pts[i]),
                errors={n:{k:metrics(q[win]) for k,q in per_errors[n].items()} for n in per_errors}))
    special=[22.12,23.96,29.10,29.12,29.14,29.16,29.20,53.5,56.72,76.70,76.76,76.80,76.84,76.90,107.8,111.44,161.06,174.76,175.52,175.58,176.28,177.04,177.68,177.72,177.76,178.44,179.02,182.82]
    checks=sorted(set(checks)|set(np.searchsorted(pts,np.rint(np.array(special)*12800))))
    for offset in range(0,len(checks),8):
        keep=checks[offset:offset+8];paper=Image.new('RGB',(1000,len(keep)*144),'white');dr=ImageDraw.Draw(paper)
        for row,i in enumerate(keep):
            raw=Image.fromarray(video.read(int(pts[i])))
            region=crop(raw,np.concatenate([states['revised']['uv'][i],states['fixed153']['uv'][i],states['revised']['shoe'][i,:,:7].reshape(-1,2)]))
            for col,n in enumerate(['source','early','fixed153','revised']):
                pic=raw if n=='source' else draw_pose(raw.copy(),states[n]['uv'][i],states[n]['shoe'][i])
                tile=pic.crop(region);tile.thumbnail((248,126));x=col*250;y=row*144
                paper.paste(tile,(x+(250-tile.width)//2,y+18));dr.text((x+6,y+2),f'{shot[i]} {pts[i]/12800:.2f}s  {n}',fill='black')
        paper.save(visual/f'comparison_{1+offset//8}.jpg',quality=92)
    video.close()
    report.update(windows=chronological,visual_source_pts=pts[checks].tolist(),review_region_source=groups)
    (out/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('prepared','programme','parent','shared','work','output'):
        parser.add_argument('--'+k,required=True,type=lambda v:Path(v).resolve())
    compare(parser.parse_args())
