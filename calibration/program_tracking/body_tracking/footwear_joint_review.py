"""Read-only comparison of fixed-camera S01 research candidates and source evidence."""
from pathlib import Path
import argparse,csv,json
import av,cv2,numpy as np
from scipy.spatial.transform import Rotation
from .rebuild_prepare import load,digest
from .footwear_joint import JointFootwearProblem
from .footwear_fit import point_polyline_distance,hull
from .footwear_geometry import decode


def stats(values):
    values=np.asarray(values,float).ravel();values=values[np.isfinite(values)]
    return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max())) if len(values) else dict(count=0,median=None,p95=None,maximum=None)


def direction(equipment,record):
    points=equipment[record['foot']];h=points[[0,7]].mean(0);t=points[[1,8]].mean(0);v=t-h;u=np.asarray(record['toe'])-record['heel'];length=float(np.linalg.norm(v));angle=float(np.rad2deg(np.arccos(np.clip(v@u/max(length*np.linalg.norm(u),1e-15),-1,1))))
    endpoint=[float(point_polyline_distance([record[name]],points[pair])[0]) for name,pair in [('heel',[0,7]),('toe',[1,8])]]
    return dict(source_seconds=record['pts']/12800,pts=record['pts'],foot=record['foot'],angle_deg=angle,projected_length_px=length,source_length_px=float(np.linalg.norm(u)),heel_toe_edge_error_px=endpoint)


def run(root,review,dataset,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False);nxt=root/'next_round_20261029';rev=json.loads(Path(review).read_text(encoding='utf8'));programme=Path(review).parent
    p=JointFootwearProblem(root,review,dataset,nxt/'B5/result.npz','J1');source=p.p.p;times=source['pts']/12800.;visible=~source['lower_body_inferred'];indices=p.p.body_problem.indices;qualified=source['source_qualified'][:,indices];base=load(nxt/'B5/result.npz');baseP=base['P_world_mid'];C=source['camera_centre']*float(source['metres_per_scene_unit']);ray=baseP-C;ray/=np.linalg.norm(ray,axis=1)[:,None]
    specifications=[('B5',nxt/'B5/result.npz'),('shoe_previous',Path(rev['frozen_equipment_source'])),('J0_s2',nxt/'J0_s2/result.npz'),('J0_s2_footinit',nxt/'J0_s2_footinit/result.npz'),('J1_s2',nxt/'J1_s2/result.npz'),('J1_s2_alternative',nxt/'J1_s2_alternative/result.npz'),('J2_s2',nxt/'J2_s2/result.npz'),('FootMR_parent_J1',nxt/'J1/result.npz'),('FootMR_conditional',nxt/'FootMR_native/conditional_installation.npz'),('FootMR_zero_installation',nxt/'FootMR_native/zero_installation.npz')]
    packets={};metrics={};curves=[];directions=[];contours=[];temporal={}
    groups=[('fit',rev['blade_curves']),('development_geometry',json.loads((programme/'footwear_s01_geometry_check_20261029.json').read_text(encoding='utf8'))['blade_curves']),('development_earlier',json.loads((programme/'footwear_s01_check_20261029.json').read_text(encoding='utf8'))['blade_curves'])]
    for name,file in specifications:
        z=load(file);packets[name]=z;np.testing.assert_array_equal(z['pts'],source['pts']);error=np.linalg.norm(z['COCO_uv'][:,indices]-source['source_uv'][:,indices],axis=2);full=qualified&visible[:,None];bodyfit=np.zeros_like(qualified);bodyfit[p.frames]=qualified[p.frames]&~p.p.body_problem.hold[p.frames,None]
        m=dict(path=str(file.resolve()),sha256=digest(file),body_visible13_px=stats(error[full]),body_partial_visible_points_px=stats(error[qualified&~visible[:,None]]),body_used_sample_px=stats(error[bodyfit]),body_development_excluded_window_px=stats(error[full&p.p.body_problem.hold[:,None]]))
        arms=np.array([7,8,9,10]);arm_error=np.linalg.norm(z['COCO_uv'][:,arms]-source['source_uv'][:,arms],axis=2);m['elbow_wrist_visible_px']=stats(arm_error[(source['source_qualified'][:,arms]&visible[:,None])]);m['elbow_wrist_used_in_objective']=False
        worst=np.unravel_index(np.argmax(np.where(full,error,-1)),error.shape);m['worst_visible_body']=dict(source_seconds=float(times[worst[0]]),coco_joint=int(indices[worst[1]]),error_px=float(error[worst]))
        P=z.get('P_world_mid',z.get('P_world_m'));d=P-baseP;along=np.sum(d*ray,1);across=np.linalg.norm(d-along[:,None]*ray,axis=1);m['root_displacement_from_B5_conditional_m']=dict(visible=stats(np.linalg.norm(d[visible],axis=1)),inferred=stats(np.linalg.norm(d[~visible],axis=1)),visible_absolute_along_ray=stats(abs(along[visible])),visible_transverse=stats(across[visible]),visible_vertical=stats(abs(d[visible,2])))
        rp=file.with_suffix('.json')
        if rp.exists():m['optimizer']=json.loads(rp.read_text(encoding='utf8'))
        if 'equipment_uv' in z:
            for group,records in groups:
                ds=[]
                for r in records:
                    fi=int(np.searchsorted(source['pts'],r['pts']));gear=z['equipment_uv'][fi,r['foot']];dist=point_polyline_distance(r['uv'],gear[14:31]);other=point_polyline_distance(r['uv'],gear[31:48]);selected=0 if np.sum(dist*dist)<=np.sum(other*other) else 1;best=dist if selected==0 else other;ds.extend(dist)
                    curves.append(dict(mode=name,group=group,pts=r['pts'],source_seconds=r['pts']/12800,foot=r['foot'],original_vertices=len(dist),fixed_left_median_px=float(np.median(dist)),fixed_left_p95_px=float(np.percentile(dist,95)),candidate_best_edge=selected,candidate_best_median_px=float(np.median(best)),candidate_best_p95_px=float(np.percentile(best,95))))
                m[group+'_blade_fixed_left_px']=stats(ds)
            dr=[]
            for r in rev['boot_landmarks']:
                fi=int(np.searchsorted(source['pts'],r['pts']));row=dict(mode=name,**direction(z['equipment_uv'][fi],r));dr.append(row);directions.append(row)
            m['direction_angle_deg']=stats([r['angle_deg'] for r in dr]);m['heel_toe_edge_error_px']=stats([x for r in dr for x in r['heel_toe_edge_error_px']]);m['critical_direction']=[r for r in dr if r['foot']==1 and r['pts'] in [325120,336128]]
            cd=[]
            for r in p.p.visible_rows:
                fi=int(np.searchsorted(source['pts'],r['pts']));distance=point_polyline_distance(r['observed'],hull(z['equipment_uv'][fi,r['foot'],:14]),closed=True);cd.extend(distance);contours.append(dict(mode=name,pts=r['pts'],source_seconds=r['pts']/12800,foot=r['foot'],median_px=float(np.median(distance)),p95_px=float(np.percentile(distance,95))))
            m['admitted_boot_source_contour_px']=stats(cd)
            points=z['equipment_uv'][:,1];v=points[:,[1,8]].mean(1)-points[:,[0,7]].mean(1);temporal[name]=dict(direction_deg=np.rad2deg(np.unwrap(np.arctan2(v[:,1],v[:,0]))),projected_length_px=np.linalg.norm(v,axis=1))
        if 'nominal_blade_minimum_world_m' in z:
            h=z['nominal_blade_minimum_world_m'][visible,:,2];m['conditional_blade_height_diagnostic_m']=dict(values=stats(h),negative_value_fraction=float(np.mean(h<0)),contact_state_inferred=False)
        if 'shared_geometry' in z:m['equipment_parameters']=decode(z['shared_geometry'])
        if name in ['J0_s2_footinit','J1_s2','J2_s2']:
            raw=p.p.state(np.r_[z['shared_geometry'],z['leg_controls'].ravel()],np.arange(len(times)),root_controls=z['root_delta_controls'],project=False);R=raw['boot_rotation'].reshape(len(times),5,2,3,3)[:,2];dR=np.einsum('nfji,nfjk->nfik',R[:-1],R[1:]);speed=np.rad2deg(Rotation.from_matrix(dR.reshape(-1,3,3)).magnitude().reshape(-1,2))/.02;valid=visible[:-1]&visible[1:];m['model_boot_angular_speed_deg_s']=stats(speed[valid]);m['temporal_model_quantities_are_not_observed_truth']=True
        metrics[name]=m
    # Show the effect of the semantic correction on identical archived pixels.
    old=json.loads((programme/'footwear_s01_joint_20261029.json').read_text(encoding='utf8'));same_prediction=packets['shoe_previous']['equipment_uv'];semantics=[]
    for r in rev['boot_landmarks']:
        if 'original_semantic_binding' not in r:continue
        oldr=next(q for q in old['boot_landmarks'] if q['pts']==r['pts'] and q['foot']==r['foot']);fi=np.searchsorted(source['pts'],r['pts']);semantics.append(dict(pts=r['pts'],raw_uv_multiset_unchanged=sorted(map(tuple,[r['heel'],r['toe']]))==sorted(map(tuple,[oldr['heel'],oldr['toe']])),same_previous_prediction_angle_before_deg=direction(same_prediction[fi],oldr)['angle_deg'],same_previous_prediction_angle_after_deg=direction(same_prediction[fi],r)['angle_deg']))
    profile=json.loads((nxt/'profiles_s2/profiles.json').read_text(encoding='utf8'));profile_summary=[]
    for center in sorted({r['source_seconds'] for r in profile['rows']}):
        for axis in ['depth','lateral']:
            rows=[r for r in profile['rows'] if r['source_seconds']==center and r['axis']==axis];best=min(rows,key=lambda r:r['data_cost']);zero=next(r for r in rows if r['offset_conditional_m']==0);profile_summary.append(dict(source_seconds=center,axis=axis,lowest_sampled_data_offset_m=best['offset_conditional_m'],data_cost_at_zero=zero['data_cost'],data_cost_at_best=best['data_cost'],offsets_m=[r['offset_conditional_m'] for r in rows],data_cost_change_from_zero=[r['data_cost']-zero['data_cost'] for r in rows],probability_interval=False))
    alt_delta=packets['J1_s2_alternative']['P_world_mid']-packets['J1_s2']['P_world_mid'];worst_alt=int(np.argmax(np.linalg.norm(alt_delta,axis=1)));multistart=dict(reference='J1_s2',alternative='J1_s2_alternative',visible_root_difference_conditional_m=stats(np.linalg.norm(alt_delta[visible],axis=1)),visible_G_difference_conditional_m=stats(np.linalg.norm(alt_delta[visible,:2],axis=1)),worst_source_seconds=float(times[worst_alt]),both_numerically_converged=False,interpretation='A comparison of two terminated nonlinear basins, not proof of uniqueness or a physical uncertainty interval.')
    fitmask=np.zeros(len(times),bool);fitmask[p.frames]=True;np.savez_compressed(output/'fit_plan.npz',pts=source['pts'],joint_sample=fitmask,body_exclusion_window=p.p.body_problem.hold,source_qualified=source['source_qualified'],source_lower_body_inferred=source['lower_body_inferred'])
    result=dict(schema='rinkspace.s01-exposure-joint-footmr-comparison.v1',source_review_sha256=digest(review),source_exposures=len(times),native_time_base=[1,12800],visible_lower_body_frames=int(visible.sum()),inferred_lower_body_frames=int((~visible).sum()),camera_updated=False,scene_updated=False,metric_scale_status='conditional 60 metres per rink L=1; body stature conditional',contact_feedback=False,all_checks_are_development=True,independent_3d_truth=False,semantic_binding_corrections=semantics,multistart=multistart,body_source_indices=indices.tolist(),joint_source_exposures=p.n,admitted_boot_contours=len(p.p.visible_rows),fit_blade_curve_count=len(rev['blade_curves']),fit_blade_original_vertices=sum(len(r['uv']) for r in rev['blade_curves']),fit_blade_resampled_vertices_per_curve=24,metrics=metrics,profile_summary=profile_summary,curve_scoring_note='Original unsampled source vertices to the same fixed blade_left edge for historical comparison. Candidate-best entire edge is reported separately and is optimistic, not physical side confirmation.',temporal_note='Global splines and FootMR offline 120-frame windows use time context. Source-only 81-frame continuity review corrected two bindings; failed long LK tracks are not fitting data. Angular speeds are model outputs, not validation of true pose.')
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    for name,rows in [('blade_curves',curves),('directions',directions),('contours',contours)]:
        with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    for name in ['J0_s2_footinit','J1_s2','J2_s2']:
        P=packets[name]['P_world_mid'];axes[0].plot(times,np.linalg.norm((P-baseP)[:,:2],axis=1),label=name);axes[1].plot(times,P[:,2],label=name);axes[2].plot(times,temporal[name]['projected_length_px'],label=name)
    axes[0].set(ylabel='pelvis XY change from B5 (conditional m)');axes[1].set(ylabel='pelvis height (conditional m)');axes[2].set(ylabel='right boot heel-toe projection (px)',xlabel='native source seconds')
    for ax in axes:ax.axvspan(times[0],24.6,color='gray',alpha=.15);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fixed camera; development candidates; shaded opening lower body inferred');fig.savefig(output/'root_and_temporal.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained');window=(times>=25)&(times<=26.6)
    for name in ['J0_s2_footinit','J1_s2','J2_s2','FootMR_conditional']:
        angle=(temporal[name]['direction_deg']+180)%360-180;angle=angle.copy();angle[np.r_[False,abs(np.diff(angle))>180]]=np.nan;axes[0].plot(times[window],angle[window],label=name);axes[1].plot(times[window],temporal[name]['projected_length_px'][window],label=name)
    for r in rev['boot_landmarks']:
        if r['foot']!=1 or not 25<=r['pts']/12800<=26.6:continue
        v=np.asarray(r['toe'])-r['heel'];axes[0].scatter(r['pts']/12800,np.rad2deg(np.arctan2(v[1],v[0])),color='black',marker='x',s=50);axes[1].scatter(r['pts']/12800,np.linalg.norm(v),color='black',marker='x',s=50)
    axes[0].set(ylabel='source image H-to-T direction (degrees)',ylim=(-185,185));axes[1].set(ylabel='projected boot H-to-T length (px)',xlabel='native source seconds')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Temporal shoe projections; black crosses: outer-boot source semantics, development only');fig.savefig(output/'temporal_direction.png',dpi=140);plt.close(fig)
    source_images={};image_pts=[325120,336128,421888,461056,489216];worst_pts=round(metrics['J1_s2']['worst_visible_body']['source_seconds']*12800);worst_window=[worst_pts+j*256 for j in range(-4,5)];decode_pts=set(image_pts+worst_window)
    with av.open(str(root/'offline_review/source_s01.mp4')) as c:
        c.streams.video[0].codec_context.thread_count=2
        for f in c.decode(video=0):
            if f.pts in decode_pts:source_images[f.pts]=f.to_ndarray(format='bgr24')
            if f.pts>=max(decode_pts):break
    index=json.loads((Path(dataset)/'source_index.json').read_text(encoding='utf8'));bypts={r['pts']:r for r in index['rows']};panels=[]
    for pts in image_pts:
        fi=int(np.searchsorted(source['pts'],pts));r=bypts.get(pts);xyxy=r['crop_xyxy'] if r else [500,650,1400,1080];x0,y0,x1,y1=xyxy;offset=np.array([x0,y0]);crop=source_images[pts][y0:y1,x0:x1].copy();row=[]
        for name in ['source','J0_s2_footinit','J1_s2','J2_s2']:
            im=crop.copy()
            if name!='source':
                gear=packets[name]['equipment_uv'][fi]
                for foot,color in [(0,(50,230,100)),(1,(250,190,60))]:
                    for ids in [np.arange(14,31),np.arange(31,48)]:cv2.polylines(im,[np.rint(gear[foot,ids]-offset).astype(np.int32)],False,color,1,cv2.LINE_AA)
                    h=gear[foot,[0,7]].mean(0)-offset;t=gear[foot,[1,8]].mean(0)-offset;cv2.arrowedLine(im,tuple(np.rint(h).astype(int)),tuple(np.rint(t).astype(int)),color,1,cv2.LINE_AA,tipLength=.15)
            for record in rev['boot_landmarks']:
                if record['pts']!=pts:continue
                for key in ['heel','toe']:
                    xy=tuple(np.rint(np.asarray(record[key])-offset).astype(int));cv2.drawMarker(im,xy,(80,70,250),cv2.MARKER_CROSS,10,1);cv2.putText(im,'H' if key=='heel' else 'T',xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(80,70,250),1,cv2.LINE_AA)
            for group,rs in groups:
                for record in rs:
                    if record['pts']==pts:cv2.polylines(im,[np.rint(np.asarray(record['uv'])-offset).astype(int)],False,(245,70,245),1,cv2.LINE_AA)
            scale=min(420/im.shape[1],280/im.shape[0]);small=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);canvas=np.full((330,440,3),22,np.uint8);canvas[42:42+small.shape[0],10:10+small.shape[1]]=small;cv2.putText(canvas,f'{pts/12800:.3f}s / {name}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA);row.append(canvas)
        panels.append(np.hstack(row))
    cv2.imwrite(str(output/'source_joint_comparison.png'),np.vstack(panels))
    worst_tiles=[]
    for pts in worst_window:
        fi=int(np.searchsorted(source['pts'],pts));pred=packets['J1_s2']['COCO_uv'][fi];ids=np.array([13,14,15,16]);points=np.r_[source['source_uv'][fi,ids],pred[ids]];low=np.maximum(np.floor(points.min(0)-70),0).astype(int);high=np.minimum(np.ceil(points.max(0)+70),[1920,1080]).astype(int);im=source_images[pts][low[1]:high[1],low[0]:high[0]].copy()
        for label,xy,color in [('S',source['source_uv'][fi],(70,70,250)),('M',pred,(70,240,100))]:
            for j in ids:
                v=tuple(np.rint(xy[j]-low).astype(int));cv2.drawMarker(im,v,color,cv2.MARKER_CROSS,8,1);cv2.putText(im,str(j)+label,v,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        scale=min(400/im.shape[1],270/im.shape[0]);im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);tile=np.full((310,420,3),22,np.uint8);tile[35:35+im.shape[0],10:10+im.shape[1]]=im;cv2.putText(tile,f'Source {pts/12800:.3f}s; S=2D / M=J1',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1,cv2.LINE_AA);worst_tiles.append(tile)
    cv2.imwrite(str(output/'worst_body_temporal_source.png'),np.vstack([np.hstack(worst_tiles[i:i+3]) for i in [0,3,6]]))
    print(json.dumps(dict(output=str(output),candidates=len(metrics),profiles=len(profile['rows']),summary={k:{'body':v['body_visible13_px'],'blade':v.get('development_geometry_blade_fixed_left_px'),'root':v['root_displacement_from_B5_conditional_m']['visible']} for k,v in metrics.items() if k in ['J0_s2_footinit','J1_s2','J2_s2']})),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','review','dataset','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.root,a.review,a.dataset,a.output)
