"""Validate the two contact intervals at the native camera exposures and on the continuous motion curve."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont
from calibration.program_tracking.body_tracking.rebuild_exposure_body import moving_projection
from calibration.program_tracking.body_tracking.skate_physics import fk_state, batch_features
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from compare import bind
from joint_refit_review import metrics
from limb_support_review import contact_path
from source_review import SourceVideo, digest, whole_skate_observations


def measure(prepared,folder,shared,work,programme,review):
    ref,cam,info,_,packet=bind(prepared,folder/'source.json',folder/'candidate.npz')
    pts=ref.p['pts']
    mask=np.zeros(len(pts),bool)
    for lo,hi in review['windows']:mask|=(pts>=round((lo-.12)*12800))&(pts<=round((hi+.12)*12800))
    mask|=(pts>=round(177.44*12800))&(pts<=round(177.64*12800))
    ids=np.flatnonzero(mask);clock=pts[ids]
    with np.load(shared/'parent_material_uv.npz') as z:mat=z['material']
    shoes=[];poses=[]
    for group in np.array_split(ids,(len(ids)+23)//24):
        t=ref.times[group,None]+cam.readout_times(group)[:,None]*np.linspace(-.5,.5,5)
        body=ref.world(ref.at(t.ravel()))
        gear=body['footwear']
        xyz=gear['boot_O'][:,0,None]+np.einsum('nij,vj->nvi',gear['boot_R'][:,0],mat)
        whole=np.concatenate((body['coco'],xyz),axis=1)
        uv=moving_projection(cam,whole.reshape(len(group),5,-1,3),group)['uv']
        poses.append(uv[:,:17]);shoes.append(uv[:,17:])
    uv=np.concatenate(poses);shoe=np.concatenate(shoes)
    state=fk_state(ref,ref.at(ref.times[ids]),np.zeros((len(ids),69)))
    mass=ContinuousMassGeometry(ref)
    com=mass.com(mass.from_pose(state)[0])
    native=state['native_world']*ref.binding.ratio
    R=state['R'][:,0];O=state['O'][:,0]*ref.binding.ratio
    ends=O[:,None]+np.einsum('nij,kj->nki',R,mat[3:5])*ref.binding.ratio
    steel=batch_features(ref.binding,state['R'],state['O'])
    lowest=1000*steel['height'][:,0,np.isin(steel['semantics'],['runner','toe_pick'])].min(1)
    knee=native[:,4];thigh=native[:,1]-knee;shin=native[:,7]-knee
    angle=np.degrees(np.arccos(np.clip((thigh*shin).sum(1)/(np.linalg.norm(thigh,axis=1)*np.linalg.norm(shin,axis=1)),-1,1)))
    with np.load(folder/'body_observations.npz') as z:
        use=z['use'][ids]&(z['model_joint_ids'][ids]==13)
        krows=np.nonzero(use)[0];kerr=np.linalg.norm(uv[krows,13]-z['uv'][ids][use],axis=1)
        markers=np.zeros((len(ids),17,2));valid=np.zeros((len(ids),17),bool)
        markers[np.arange(len(ids))[:,None],z['model_joint_ids'][ids]]=z['uv'][ids]
        valid[np.arange(len(ids))[:,None],z['model_joint_ids'][ids]]=z['use'][ids]
    def angle2d(x):
        a=x[:,11]-x[:,13];b=x[:,15]-x[:,13]
        return np.rad2deg(np.arctan2(abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]),(a*b).sum(1)))
    source_angle=np.abs(angle2d(markers)-angle2d(uv))
    directions=np.stack([np.cos(np.arange(12)*np.pi/6),np.sin(np.arange(12)*np.pi/6)],1)
    with np.load(work/'input/shoe_occlusion_source.npz') as z:
        np.testing.assert_array_equal(z['source_pts'],pts)
        good=(z['role'][ids,0]==1)&z['visible'][ids,0]&~z['partial_boundary'][ids,0]&(z['source_pair_overlap'][ids]<.12)
        ext=(np.nan_to_num(z['region'][ids,0])@directions.T).max(1)
    source_obs=json.loads(whole_skate_observations(programme).read_text('utf8'))
    for r in source_obs.get('source_outline_review_exclusions',[]):
        if r.get('foot')==0 and 'pts' in r:good[clock==r['pts']]=False
    outline=np.abs((shoe[:,7:]@directions.T).max(1)-ext).mean(1)
    rows=review['records'];fr=np.searchsorted(clock,[r['pts'] for r in rows]);feat=[r['feature']-17 for r in rows]
    bounds=np.array([r['box_px'] for r in rows]);coords=shoe[fr,feat]
    distances=np.linalg.norm(coords-np.clip(coords,bounds[:,0],bounds[:,1]),axis=1)
    old=json.loads((programme/'fit_v11/sitspin_pixels_1119.json').read_text('utf8'))
    historical=next(r for r in old['records'] if r['pts']==2249216 and r['feature']==20)
    initial_box=np.array(historical['box_px'])
    results={}
    for key,t0,(lo,hi) in zip(('layback','sitspin'),(159.08,175.72),review['windows']):
        i=np.searchsorted(clock,round(t0*12800));within=(clock>=round(lo*12800))&(clock<=round(hi*12800))
        episode=dict(foot=0,start_pts=round(lo*12800),end_pts_inclusive=round(hi*12800))
        focus=clock[fr]==clock[i]
        res=dict(source_seconds=[lo,hi],frames=int(within.sum()),focus_pts=int(clock[i]),focus_seconds=t0,
            focus_uv=shoe[i,3:5].tolist(),focus_distance_px=distances[focus].tolist(),
            focus_height_mm=(1000*ends[i,:,2]).tolist(),focus_lowest_mm=float(lowest[i]),outline_px=metrics(outline[within&good]),
            knee_source_angle_error_deg=metrics(source_angle[within&valid[:,[11,13,15]].all(1)]),
            com_accel_mps2=metrics(np.linalg.norm(np.diff(com[within],n=2,axis=0)/.02**2,axis=-1)),
            focus_knee_error_px=float(np.linalg.norm(uv[i,13]-markers[i,13])) if valid[i,13] else None,
            left_knee_error_px=metrics(kerr[within[krows]]),native_support=contact_path(ref,episode,50),
            dense_support=contact_path(ref,episode,200),support_400hz=contact_path(ref,episode,400))
        check_times=np.arange(round((lo-.08)*200),round((hi+.08)*200)+1)/200
        dense_state=fk_state(ref,ref.at(check_times-pts[0]/12800),np.zeros((len(check_times),69)))
        res['joint_rotation_step_deg']={}
        body_error=np.linalg.norm(uv-markers,axis=-1)
        for name,idx in [('upper_body',list(range(5,11))),('right_ankle',[16])]:
            use=within[:,None]&valid[:,idx]
            res[name+'_source_px']=metrics(body_error[:,idx][use])
        for rate,stride in ((50,4),(200,1)):
            for joint,name in ((1,'hip'),(4,'knee'),(7,'ankle')):
                rr=dense_state['local_R'][::stride,joint-1]
                delta=Rotation.from_matrix(rr[:-1].swapaxes(1,2)@rr[1:]).magnitude()
                res['joint_rotation_step_deg'][f'{name}_{rate}Hz']=metrics(np.rad2deg(delta))
        for role in ('fit','check'):
            select=within[fr]&np.array([r['role']==role for r in rows])
            res[role+'_box_px']=metrics(distances[select])
        if key=='sitspin':
            a=shoe[i,3]
            res['previous_17572_tail_box_px']=float(np.linalg.norm(a-np.clip(a,*initial_box)))
        results[key]=res
    state=dict(pts=clock,shoe=shoe,body=uv,native=native,knee_angle=angle,lowest_mm=lowest)
    return dict(sha256=digest(folder/'candidate.npz'),camera_sha256=info['state_sha256'],
                nominal_cm=100*ref.body.scale*np.ptp(ref.binding.surface.shaped[:,1]),windows=results),state


def images(a,states,review):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    field=json.loads((a.programme/'s14_boards_1118.json').read_text('utf8'))
    reader=SourceVideo(field['source']['path'],(1,12800))
    for key,centre in [('layback',159.08),('sitspin',175.72)]:
        paper=Image.new('RGB',(3*272,5*145),(22,26,29));dr=ImageDraw.Draw(paper)
        for row in range(5):
            t=centre+(row-2)*.02;stamp=round(t*12800);raw=Image.fromarray(reader.read(stamp))
            i=np.searchsorted(states[0]['pts'],stamp);cloud=np.vstack([state['shoe'][i] for state in states])
            mx=(cloud.min(0)+cloud.max(0))/2
            width=max(195.,np.ptp(cloud[:,0])+16,(np.ptp(cloud[:,1])+16)*272/120)
            height=width*120/272;left,top=np.clip(mx-[width/2,height/2],[0,0],[1920-width,1080-height]).astype(int)
            box=(left,top,left+int(width),top+int(height))
            for col,label in enumerate(('原片','上一交付版','本轮修正版')):
                im=raw.copy()
                if col:
                    xy=states[col-1]['shoe'][i];h=ConvexHull(xy[7:]).vertices
                    g=ImageDraw.Draw(im);g.line(list(map(tuple,xy[7:][np.append(h,h[0])])),fill='#20c789',width=2)
                    g.line(list(map(tuple,xy[3:5])),fill='#f44d34',width=2)
                x=col*272;y=row*145
                dr.text((x+4,y+2),f'{t:.2f} 秒　{label}',font=font,fill='white')
                paper.paste(im.crop(box).resize((272,120)),(x,y+25))
        paper.save(a.output/'source_review'/f'{key}_comparison.png')
    paper=Image.new('RGB',(246*3,192*6),(22,26,29));dr=ImageDraw.Draw(paper)
    for row,t in enumerate((159.04,159.08,159.12,175.68,175.72,175.76)):
        stamp=round(t*12800);raw=Image.fromarray(reader.read(stamp))
        i=np.searchsorted(states[0]['pts'],stamp)
        cloud=np.concatenate([np.vstack((st['body'][i,[11,13,15]],st['shoe'][i])) for st in states])
        mid=(cloud.min(0)+cloud.max(0))/2
        width=max(220.,np.ptp(cloud[:,0])+70.,(np.ptp(cloud[:,1])+56.)*246/166)
        height=width*166/246
        left,top=np.clip(mid-[width/2,height/2],[0,0],[1920-width,1080-height]).astype(int)
        box=(left,top,left+int(width),top+int(height))
        for col,label in enumerate(('原片','上一交付版','本轮修正版')):
            im=raw.copy()
            if col:
                st=states[col-1];line=st['body'][i,[11,13,15]];g=ImageDraw.Draw(im)
                g.line(list(map(tuple,line)),fill='#00beac',width=2)
                x,y=line[1];g.ellipse((x-4,y-4,x+4,y+4),outline='white',width=2)
                g.line(list(map(tuple,st['shoe'][i,3:5])),fill='#f44d34',width=2)
            x,y=246*col,192*row
            dr.text((x+4,y+2),f'{t:.2f} 秒　{label}',font=font,fill='white')
            paper.paste(im.crop(box).resize((246,166)),(x,y+26))
    paper.save(a.output/'source_review'/'knee_comparison.jpg',quality=94)
    reader.close()


def main(a):
    spec=json.loads((a.programme/'fit_v11/spin_pair_pixels_1119.json').read_text('utf8'))
    report={};states=[]
    for key,folder in [('previous',a.parent),('revised',a.output)]:
        report[key],state=measure(a.prepared,folder,a.shared,a.work,a.programme,spec);states.append(state)
    before,after=states;assert report['previous']['camera_sha256']==report['revised']['camera_sha256']
    np.testing.assert_allclose(report['revised']['nominal_cm'],153.,atol=.001)
    clock=before['pts'];protected=(clock>=round(177.44*12800))&(clock<=round(177.64*12800))
    for key in ('shoe','body','native'):np.testing.assert_array_equal(before[key][protected],after[key][protected])
    report['protected_source_seconds']=(clock[protected][[0,-1]]/12800).tolist()
    exact=np.flatnonzero(clock==round(177.54*12800)).item()
    report['protected_17754']=dict(blade_endpoints_px=after['shoe'][exact,3:5].tolist(),
        left_knee_px=after['body'][exact,13].tolist(),identical_source_projections=True)
    report['window_guard']={}
    for window,(lo,hi) in zip(('layback','sitspin'),spec['windows']):
        inside=(clock>=round(lo*12800))&(clock<=round(hi*12800))
        focus=np.flatnonzero(clock==report['revised']['windows'][window]['focus_pts']).item()
        report['window_guard'][window]=dict(knee_joint_centre_world_max_mm=1000*float(np.linalg.norm((after['native']-before['native'])[inside,4],axis=-1).max()),
            focus_knee_flexion_before_after_deg=[float(s['knee_angle'][focus]) for s in states],
            focus_pelvis_shift_xyz_mm=(1000*(after['native']-before['native'])[focus,0]).tolist(),
            pelvis_shift_mm=metrics(1000*np.linalg.norm((after['native']-before['native'])[inside,0],axis=-1)),
            free_ankle_source_shift_px=metrics(np.linalg.norm((after['body']-before['body'])[inside,16],axis=-1)),
            upper_source_shift_px=metrics(np.linalg.norm((after['body']-before['body'])[inside,5:11],axis=-1).ravel()),
            knee_marker_change_px=metrics(np.linalg.norm((after['body']-before['body'])[inside,13],axis=-1)),
            knee_flexion_change_deg=metrics(np.abs((after['knee_angle']-before['knee_angle'])[inside])))
    if a.full:
        with np.load(a.parent/'candidate.npz') as old,np.load(a.output/'candidate.npz') as new:
            for k in ('relative_skate_rotation_controls','fitted_programme_body_scale',
                      'programme_double_support_json','absolute_rotation_edge_reference_quaternion'):
                np.testing.assert_array_equal(old[k],new[k])
            changed_joints=json.loads(str(new['joint_source_fit_json']))['local_joints']
            if -1 not in changed_joints:np.testing.assert_array_equal(old['absolute_position_controls'],new['absolute_position_controls'])
            keep=[j for j in range(22) if j not in changed_joints]
            np.testing.assert_array_equal(old['absolute_rotation_controls'][:,keep],new['absolute_rotation_controls'][:,keep])
        with np.load(a.parent/'result.npz') as old,np.load(a.output/'result.npz') as new:
            pts=old['source_pts'];np.testing.assert_array_equal(pts,new['source_pts']);assert len(pts)==8329
            fixed=[0,1,2,3,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21]
            np.testing.assert_array_equal(old['skate_R_world_mid'][:,1],new['skate_R_world_mid'][:,1])
            root_window=np.zeros(len(pts),bool)
            if -1 in changed_joints:
                lo,hi=spec['windows'][0];root_window=(pts/12800>=lo-.15)&(pts/12800<=hi+.15)
            for k in ('native22_world_mid','skate_origin_world_mid'):
                select=fixed if k=='native22_world_mid' else [1]
                np.testing.assert_array_equal(old[k][~root_window][:,select],new[k][~root_window][:,select])
                delta=new[k][root_window][:,select]-old[k][root_window][:,select]
                root_delta=(new['native22_world_mid']-old['native22_world_mid'])[root_window,:1]
                np.testing.assert_allclose(delta,np.broadcast_to(root_delta,delta.shape),rtol=0,atol=2e-5)
            changed=np.any(old['skate_R_world_mid'][:,0]!=new['skate_R_world_mid'][:,0],axis=(1,2))
            allowed=np.zeros(len(pts),bool)
            for lo,hi in spec['windows']:allowed|=(pts/12800>=lo-.15)&(pts/12800<=hi+.15)
            assert not changed[~allowed].any()
            runs=np.split(np.flatnonzero(changed),np.flatnonzero(np.diff(np.flatnonzero(changed))>1)+1)
            report['changed_left_orientation_source_windows']=[(pts[ix[[0,-1]]]/12800).tolist() for ix in runs if len(ix)]
            keep=(pts/12800>=177.40)&(pts/12800<=177.68)
            for k in ('native22_world_mid','skate_R_world_mid','com_world_mid','skate_origin_world_mid'):
                np.testing.assert_array_equal(old[k][~allowed|keep],new[k][~allowed|keep])
            assert str(new['transfer_parameter_sha256'])==report['revised']['sha256']
            report['frames']=len(pts);report['programme_seconds']=(pts[[0,-1]]/12800).tolist()
            report['fit']=json.loads(str(new['joint_source_fit_json']))
    (a.output/'source_review').mkdir(exist_ok=True)
    images(a,states,spec)
    (a.output/('review.json' if a.full else 'probe.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf8')
    for key in ('previous','revised'):
        for name,r in report[key]['windows'].items():
            print(key,name,r['focus_distance_px'],r['focus_height_mm'],r['outline_px'],r['dense_support']['absolute_lateral_speed_mps'],flush=True)
    print('guards',report['window_guard'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('prepared','shared','work','programme','parent','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--full',action='store_true')
    main(p.parse_args())
