"""Audit the ending runner against identical native frames in three candidate versions."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from compare import bind
from joint_refit_review import metrics
from limb_support_review import contact_path
from source_review import SourceVideo, digest, whole_skate_observations
from joint_source_objective import material_points
from world_motion_review import phase_details


def run(a):
    doc=json.loads((a.programme/'fit_v11/sitspin_pixels_1119.json').read_text('utf8'))
    with np.load(a.work/'input/shoe_occlusion_source.npz') as z:
        pts=z['source_pts'];shots=z['shot_ids']
        dirs=np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1)
        complete=(z['role']==1)&z['visible']&~z['partial_boundary']&(z['source_pair_overlap'][:,None]<.12)
        ext=(np.nan_to_num(z['region'])@dirs.T).max(2)
    previous=json.loads(whole_skate_observations(a.programme).read_text('utf8'))
    for r in previous.get('source_outline_review_exclusions',[]):
        if 'pts' in r and 'foot' in r:complete[np.searchsorted(pts,r['pts']),r['foot']]=False
    time=pts/12800
    scope=(time>=doc['scope_seconds'][0])&(time<=doc['scope_seconds'][1])
    labels=list(doc['records'])
    source=json.loads((a.output/'source.json').read_text('utf8'))
    extra=source.get('additional_turn_ends')
    if extra:
        path=Path(extra['path'])
        assert digest(path)==extra['sha256']
        labels.extend(json.loads(path.read_text('utf8'))['records'])
    fr=np.searchsorted(pts,[r['pts'] for r in labels])
    feet=np.array([r['feature']-17 for r in labels]);bounds=np.array([r['box_px'] for r in labels])
    with np.load(a.output/'body_observations.npz') as z:
        ids=z['model_joint_ids'];use=z['use']&(ids==13)&scope[:,None];target=z['uv'][use]
    camera=json.loads((a.programme/'s14_boards_1118.json').read_text('utf8'))
    j=np.searchsorted(pts,round(177.54*12800));values={};states={}
    report=dict(frames=len(pts),source_seconds=time[[0,-1]].tolist(),shots=list(dict.fromkeys(shots.tolist())),
                phase_seconds=doc['scope_seconds'],phase_exposures=int(scope.sum()),candidates={})
    report['direct_runner_exposures']=len(set(r['pts'] for r in labels))
    for label,folder in [('fixed153',a.early),('previous',a.parent),('revised',a.output)]:
        ref,cam,info,_,packet=bind(a.prepared,folder/'source.json',folder/'candidate.npz')
        ep=next(e for e in ref.source_document['source_episode_review']['episodes'] if e['episode_id']=='ccosp_left_sit')
        with np.load(folder/'joint_source_measurements.npz') as m:
            assert str(m['candidate'])==digest(folder/'candidate.npz')
            shoe=m['shoe'];uv=m['uv'];height=m['height']
        outlines=np.abs((shoe[:,:,7:]@dirs.T).max(2)-ext).mean(-1)
        x=shoe[fr,0,feet];distance=np.linalg.norm(x-np.clip(x,bounds[:,0],bounds[:,1]),axis=-1)
        steel=shoe[j,0,3:5];delta=steel[1]-steel[0]
        with np.load(folder/'result.npz') as z:
            np.testing.assert_array_equal(z['source_pts'],pts)
            assert str(z['transfer_parameter_sha256'])==digest(folder/'candidate.npz')
            local=material_points(ref.binding)[3:5]*ref.binding.ratio
            ends=z['skate_origin_world_mid'][:,0,None]+np.einsum('nij,kj->nki',z['skate_R_world_mid'][:,0],local)
            com=z['com_world_mid'];native=z['native22_world_mid']
            np.testing.assert_array_equal(com[:,:2],z['com_projection_world_mid'][:,:2])
            np.testing.assert_array_equal(z['com_projection_world_mid'][:,2],0.)
            slip50=phase_details(z,[ep])[ep['episode_id']]['blade_lateral_speed_mps']
        states[label]=dict(shoe=shoe,uv=uv,com=com,native=native)
        focus=dict(pts=int(pts[j]),uv_rear_front=steel.tolist(),angle_deg=float(np.degrees(np.arctan2(delta[1],delta[0]))),
                   outline_px=float(outlines[j,0]),rear_front_height_mm=(ends[j,:,2]*1000).tolist(),lowest_mm=float(height[j,0]*1000))
        knee=np.linalg.norm(uv[np.nonzero(use)[0],13]-target,axis=1)
        values[label]=dict(outlines=outlines[complete[:,0]&scope,0],dist=distance,knee=knee)
        rr=dict(camera_sha256=info['state_sha256'],sha256=digest(folder/'candidate.npz'),fit=json.loads(str(packet['joint_source_fit_json'])),
              nominal_cm=100*ref.body.scale*np.ptp(ref.binding.surface.shaped[:,1]),focus=focus,
              outline_px=metrics(values[label]['outlines']),right_outline_px=metrics(outlines[scope&complete[:,1],1]),
              left_knee_px=metrics(knee),reviewed_support_50hz=slip50,
              turns_outline_px={str(sec):float(outlines[i,0]) if complete[i,0] else None
                                for sec in (175.58,178.10) for i in [np.searchsorted(pts,round(sec*12800))]},
              clearance_200hz=contact_path(ref,ep,200),rear_front_height_max_mm=metrics(ends[scope,:,2].max(1)*1000),
              com_accel_mps2=metrics(np.linalg.norm(np.diff(com[scope],n=2,axis=0)/.02**2,axis=1)))
        for role in ('check','fit'):
            select=np.array([r['role']==role for r in labels])
            rr[role+'_runner_box_px']=metrics(distance[select])
            worst=np.flatnonzero(select)[np.argmax(distance[select])]
            rr[role+'_largest_runner_residual']=dict(pts=labels[worst]['pts'],feature=labels[worst]['feature'],
                source_seconds=labels[worst]['pts']/12800,box_px=labels[worst]['box_px'],
                projected_uv=x[worst].tolist(),distance_px=float(distance[worst]))
        report['candidates'][label]=rr
    old=states['previous'];new=states['revised']
    for name,arr in [('right_boot_uv',np.subtract(new['shoe'][:,1],old['shoe'][:,1])),
                     ('upper_and_right_skeleton_world',np.subtract(new['native'][:,[0,1,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]],old['native'][:,[0,1,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]]))]:
        report[name+'_max_change']=float(np.nanmax(np.abs(arr)))
        assert report[name+'_max_change']<1e-8
    magnitude=np.linalg.norm(new['native']-old['native'],axis=-1).max(1)
    moved=magnitude>1e-9
    report['actual_leg_change_seconds']=time[moved][[0,-1]].tolist()
    assert not np.any(moved & ((time<doc['scope_seconds'][0]-.2)|(time>doc['scope_seconds'][1]+.2)))
    displacement=np.linalg.norm(new['com']-old['com'],axis=1)*1000
    report['com_change_in_spin_mm']=metrics(displacement[scope])
    report['com_change_outside_edited_interval_mm']=float(displacement[~moved].max())
    assert report['com_change_outside_edited_interval_mm']<1.e-5
    for key in ('fixed153','previous'):
        change=values['revised']['outlines']-values[key]['outlines']
        times=time[scope&complete[:,0]]
        report['comparison_to_'+key]=dict(improved_frames=int((change<0).sum()),max_worsening_px=float(change.max()),
            worst_seconds=float(times[change.argmax()]),more_than_2px_worse_seconds=times[change>2].tolist())
    best=report['candidates']['revised']
    assert len({z['camera_sha256'] for z in report['candidates'].values()})==1
    assert abs(best['nominal_cm']-153)<.001
    assert best['fit']['contact'] and best['fit']['review_sha256']==digest(a.programme/'fit_v11/sitspin_pixels_1119.json')
    worst=time[scope&complete[:,0]][np.argsort(values['revised']['outlines'])[-3:]]
    selected=sorted(set(np.round([175.58,176.4,177.50,177.54,177.58,177.76,178.10,178.74,*worst],2)))
    video=SourceVideo(camera['source']['path'],(1,12800))
    tw,th=250,145
    canvas=Image.new('RGB',(4*tw,len(selected)*(th+20)),(25,25,25));draw=ImageDraw.Draw(canvas)
    for row,sec in enumerate(selected):
        i=np.searchsorted(pts,round(sec*12800));raw=Image.fromarray(video.read(int(pts[i])))
        coords=np.vstack([st['shoe'][i,0] for st in states.values()])
        centre=(coords.max(0)+coords.min(0))/2
        size=max(160.,np.ptp(coords[:,0])+34,(np.ptp(coords[:,1])+26)*tw/th)
        l=int(np.clip(centre[0]-size/2,0,1920-size));top=int(np.clip(centre[1]-size*th/tw/2,0,1080-size*th/tw))
        box=(l,top,l+int(size),top+int(size*th/tw))
        for col,k in enumerate(['source','fixed153','previous','revised']):
            pic=raw.copy()
            if col:
                shoe=states[k]['shoe'][i,0];hull=ConvexHull(shoe[7:]).vertices
                g=ImageDraw.Draw(pic)
                outline=shoe[7:][np.append(hull,hull[0])]
                g.line([tuple(v) for v in outline],fill='#38db9b',width=2)
                g.line([tuple(v) for v in shoe[3:5]],fill='#ff4e32',width=2)
            canvas.paste(pic.crop(box).resize((tw,th)),(col*tw,row*(th+20)+20))
            draw.text((col*tw+4,row*(th+20)+3),f'{sec:.2f} s   {k}',fill='white')
    video.close()
    canvas.save(a.output/'source_review'/'spin_comparison.jpg',quality=94)
    (a.output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    for k,v in report['candidates'].items():
        print(k,v['focus'],v['outline_px'],v['clearance_200hz']['absolute_lateral_speed_mps'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('prepared','work','programme','early','parent','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    run(parser.parse_args())
