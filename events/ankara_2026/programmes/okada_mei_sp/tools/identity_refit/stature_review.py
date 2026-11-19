"""Like-for-like source and physical comparisons of the two height treatments."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull

from compare import bind
from joint_refit_review import errors,metrics,crop
from joint_source_objective import source_shoe_rows,material_points,FEATURES
from source_review import SourceVideo,digest, whole_skate_observations
from world_motion_review import physics,phase_details,paths
from calibration.program_tracking.body_tracking.absolute_motion_state import curve_from_packet
from world_motion_refit import project


def support_sensitivity(args, ref, cam):
    """Profile out per-frame positions to isolate the scale information from ice contact."""
    folder = args.output / 'free'
    with np.load(folder / 'result.npz') as saved:
        pts = saved['source_pts']
        body = saved['COCO_world_mid']
        pelvis = saved['native22_world_mid'][:, 0]
        ankles = saved['native22_world_mid'][:, [7, 8]]
        nominal_cm = float(np.ptp(ref.binding.surface.shaped[:, 1])) * 100 * float(saved['shared_body_scale'])
        opening = json.loads(str(saved['programme_double_support_json']))['pts']
    with np.load(args.shared / 'final_body_observations.npz') as ledger:
        use, ids, target = ledger['use'], ledger['model_joint_ids'], ledger['uv']
        np.testing.assert_array_equal(ledger['pts'], pts)
    with np.load(folder / 'joint_source_measurements.npz') as measurements:
        actual, height = measurements['uv'], measurements['height']
    support = np.zeros((len(pts), 2), bool)
    for episode in ref.source_document['source_episode_review']['episodes']:
        if episode['kind'] == 'support_candidate' and episode.get('foot') in (0, 1):
            inside = (pts >= episode['start_pts']) & (pts <= episode['end_pts_inclusive'])
            support[inside, episode['foot']] = True
    support[(pts >= opening[0]) & (pts <= opening[1])] = True

    camera = cam.midpoint['values']
    units = cam.scale * ref.binding.ratio
    projected = project(camera, body, units)
    step = 1.e-5
    jac = np.stack([(project(camera, body + axis * step, units) - projected) / step
                    for axis in np.eye(3)], axis=-1)
    frames = np.arange(len(pts))[:, None]
    jac = jac[frames, ids]
    residual = np.nan_to_num(actual[frames, ids] - target)
    weight = use.astype(float) / (1 + np.linalg.norm(residual, axis=-1)**2 / 7**2)**.25
    weight *= .33 * np.where((ids >= 5) & (ids <= 10), 2.2, np.where((ids == 13) | (ids == 14), 1.7, 1.))
    lever = (body - pelvis[:, None])[frames, ids]
    body_scale_jac = np.einsum('nkwd,nkd->nkw', jac, lever)
    translation = (jac * weight[:, :, None, None]).reshape(len(pts), -1, 3)
    scale = (body_scale_jac * weight[:, :, None]).reshape(len(pts), -1)
    error = (residual * weight[:, :, None]).reshape(len(pts), -1)

    def profile(position_columns, scale_column, residuals):
        inverse = np.linalg.pinv(position_columns, rcond=1.e-10)
        def remove_position(values):
            return values - (position_columns @ (inverse @ values[:, :, None]))[:, :, 0]
        remaining = remove_position(scale_column)
        return (remaining**2).sum(1), (remaining * remove_position(residuals)).sum(1)

    body_only, _ = profile(translation, scale, error)
    ice = support.astype(float) * 420 / np.sqrt(2)
    curvature, gradient = profile(
        np.concatenate([translation, ice[:, :, None] * np.array([0., 0., 1.])], axis=1),
        np.concatenate([scale, ice * (ankles[:, :, 2] - pelvis[:, None, 2])], axis=1),
        np.concatenate([error, ice * height], axis=1))
    keep = support.any(1)
    total = curvature[keep].sum()
    def estimate(mask):
        information = curvature[mask].sum()
        return float(nominal_cm * np.exp(-gradient[mask].sum()/information)) if information > 0 else None
    result = dict(method='Local fit using original body pixels and reviewed contacts. Final poses are fixed and per-frame pelvis translations are profiled out.',
                  body_only_median_to_supported_curvature=float(np.median(body_only[keep])/np.median(curvature[keep])),
                  frames=int(keep.sum()), full_conditional_cm=estimate(keep), per_shot={})
    for shot in np.unique(cam.shot_ids):
        hit = keep & (cam.shot_ids == shot)
        result['per_shot'][str(shot)] = dict(n=int(hit.sum()), information_fraction=float(curvature[hit].sum()/total),
                                             excluding_shot_cm=estimate(keep & ~hit))
    (args.output/'support_sensitivity.json').write_text(json.dumps(result, indent=2), encoding='utf8')


def review(args):
    roots=dict(previous=args.parent,fixed=args.output/'matched',free=args.output/'free')
    states={};scenes={};records={};phase_data={};scores={};profiles={}
    with np.load(args.shared/'final_body_observations.npz') as z:obs={k:z[k] for k in ['use','pts','model_joint_ids','uv']}
    pts=obs['pts'];doc=json.loads(whole_skate_observations(args.programme).read_text('utf8'))
    with np.load(args.work/'input/shoe_occlusion_source.npz') as z:
        np.testing.assert_array_equal(z['source_pts'],pts);shot=z['shot_ids'];dirs=np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1)
        good=(z['role']==1)&z['visible']&~z['partial_boundary']&(z['source_pair_overlap'][:,None]<.12)
        ext=np.max(np.nan_to_num(z['region'])@dirs.T,axis=2)
    for r in doc.get('source_outline_review_exclusions',[]):
        if 'pts' in r and 'foot' in r:good[np.searchsorted(pts,r['pts']),r['foot']]=False
    with np.load(args.shared/'before_source_measurements.npz') as z:rows=source_shoe_rows(doc,pts,z['shoe'])
    extra=[r for r in doc['samples'] if r['role'] in ('additional_check','review_only')]
    shots=list(dict.fromkeys(shot))
    for name,folder in roots.items():
        ref,cam,_,_,pkt=bind(args.prepared,folder/'source.json',folder/'candidate.npz')
        episodes=ref.source_document['source_episode_review']['episodes']
        with np.load(folder/'joint_source_measurements.npz') as z:
            assert str(z['candidate'])==digest(folder/'candidate.npz')
            m={k:z[k] for k in ('uv','shoe','height','angles')};arcs=json.loads(str(z['reviewed_arcs']))
        states[name]=m
        e=errors(name,m,obs,rows,ext,good,dirs)
        extra_err=np.array([np.linalg.norm(m['shoe'][np.searchsorted(pts,r['pts']),r['foot'],FEATURES.index(r['feature'])]-r['uv']) for r in extra])
        # The raw manual check rows are kept on their recorded side; no
        # matching against this estimate or choosing the smaller error.
        scores[name]=dict(full={k:metrics(v) for k,v in e.items()},extra=metrics(extra_err),extra_px=extra_err.tolist(),
            shots={str(s):{k:metrics(v[shot==s]) for k,v in e.items()} for s in shots},arcs=arcs)
        fit=json.loads(str(pkt['joint_source_fit_json']))
        with np.load(folder/'result.npz') as z:
            np.testing.assert_array_equal(z['source_pts'],pts)
            np.testing.assert_allclose(z['shared_body_scale'],ref.body.scale,rtol=0,atol=1.e-12)
            com=z['com_world_mid'];shoe=z['skate_origin_world_mid'][:,:,None]+np.einsum('nfij,kj->nfki',z['skate_R_world_mid'],material_points(ref.binding)[[3,4]])*ref.binding.ratio
            scenes[name]=dict(com=com,shoe=shoe,joints=z['native22_world_mid'],rot=z['native_body_local_R_mid'])
            phys=physics(com,shoe,m['height'],pts,shot,episodes);ph=phase_details(z,episodes)
            phase_data[name]=ph
            records[name]=dict(scale=float(z['shared_body_scale']),nominal_cm=float(np.ptp(ref.binding.surface.shaped[:,1]))*100*ref.body.scale,
                rink_ratio=float(z['body_length_ratio']),physics=phys,phase_details=ph,contact=fit['contact'],candidate=digest(folder/'candidate.npz'),
                maximum_cavity_violation_m=metrics(z['foot_cavity_max_violation_m_mid']),footbed_error_m=metrics(z['forefoot_error_mid']))
            if name!='previous':
                t=np.linspace(ref.times[0],ref.times[-1],4*len(pts)-3)
                c,b,h=paths(ref,curve_from_packet(pkt),t)
                np.testing.assert_allclose(c[::4],com,atol=2e-8)
                np.testing.assert_allclose(b[::4],shoe,atol=2e-8)
                dense=physics(c,b,h,pts[0]+np.arange(len(t))*64,shot[np.arange(len(t))//4],episodes)
                records[name]['physical_200hz']=dense
    support_sensitivity(args,ref,cam)
    b=scenes['fixed'];c=scenes['free']
    diffs={k:metrics(np.linalg.norm(c[k]-b[k],axis=-1)*100) for k in ['joints','shoe','com']}
    dr=b['rot'].swapaxes(-1,-2)@c['rot']
    angle=np.rad2deg(np.arccos(np.clip((dr.trace(axis1=-2,axis2=-1)-1)/2,-1,1)))
    diffs['local_rotations_degree']=metrics(angle)
    diffs['world_pelvis_cm']=metrics(np.linalg.norm(c['joints'][:,0]-b['joints'][:,0],axis=-1)*100)
    diffs['joints_relative_to_pelvis_cm']=metrics(np.linalg.norm((c['joints']-c['joints'][:,0,None])-(b['joints']-b['joints'][:,0,None]),axis=-1)*100)
    delta=c['com']-b['com']
    centre=cam.midpoint['centre']*cam.scale*ref.binding.ratio
    view=(c['com']+b['com'])*.5-centre
    view/=np.linalg.norm(view,axis=1,keepdims=True)
    along=np.sum(delta*view,axis=1)
    across=np.linalg.norm(delta-along[:,None]*view,axis=1)
    depth=dict(squared_displacement_fraction_along_view=float(along@along/np.sum(delta**2)),
        signed_depth_difference_cm=np.percentile(along*100,[5,50,95]).tolist(),
        transverse_difference_p95_cm=float(np.percentile(across*100,95)),
        free_farther_exposures=int((along>0).sum()),exposures=len(delta))
    (args.output/'depth_difference.json').write_text(json.dumps(depth,indent=2),encoding='utf8')
    dst=args.output/'source_checks';dst.mkdir(exist_ok=True)
    source=json.loads((args.programme/'s14_boards_1118.json').read_text('utf8'))['source']
    video=SourceVideo(source['path'],(1,12800))
    sheet_frames=[]
    for s in shots:
        ix=np.flatnonzero(shot==s)
        separation=np.nanmean(np.linalg.norm(states['free']['uv'][ix]-states['fixed']['uv'][ix],axis=-1),axis=1)
        sheet_frames.append(ix[np.argmax(separation)])
    special=[22.12,23.96,24.02,48.48,76.8,81.04,111.44,150.06,175.58,182.82]
    sheet_frames=sorted(set(sheet_frames)|set(np.searchsorted(pts,np.round(np.asarray(special)*12800).astype(int))))
    for a in range(0,len(sheet_frames),6):
        selected=sheet_frames[a:a+6];paper=Image.new('RGB',(880,len(selected)*214),'white')
        for row,i in enumerate(selected):
            raw=Image.fromarray(video.read(int(pts[i])))
            rect=crop(raw,np.concatenate((states['free']['uv'][i],states['fixed']['uv'][i],states['free']['shoe'][i,:,:7].reshape(-1,2))))
            for col,n in enumerate(['source','previous','fixed','free']):
                tile=raw.copy();dr=ImageDraw.Draw(tile)
                if n!='source':
                    for start,end in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
                        dr.line([tuple(p) for p in states[n]['uv'][i,[start,end]]],fill=('#20e0c6' if end%2 else '#e98b29'),width=2)
                    for foot,color in [(0,'#20e0c6'),(1,'#e98b29')]:
                        points=states[n]['shoe'][i,foot,7:];edge=ConvexHull(points).vertices
                        dr.line([tuple(p) for p in points[np.r_[edge,edge[0]]]],fill=color,width=2)
                tile=tile.crop(rect);tile.thumbnail((216,192));paper.paste(tile,(col*220,row*214+18))
                ImageDraw.Draw(paper).text((col*220+3,row*214+1),f'{shot[i]} {pts[i]/12800:.2f}s  {n}',fill='black')
        paper.save(dst/f'height_comparison_{1+a//6}.jpg',quality=92)
    video.close()
    report=dict(shots=[str(s) for s in shots],source_pts=[int(pts[0]),int(pts[-1])],frames=len(pts),
        source=source,quality=scores,models=records,matched_state_differences=diffs,
        review_seconds=(pts[sheet_frames]/12800).tolist(),additional_checks=[{k:r.get(k) for k in ['role','foot','feature','pts','uv']} for r in extra])
    (args.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['prepared','parent','shared','work','output','programme']:ap.add_argument(k,type=lambda v:Path(v).resolve())
    review(ap.parse_args())
