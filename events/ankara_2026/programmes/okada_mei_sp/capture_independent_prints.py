"""Independent print-content registration followed by original RGB edge reading.

Source feature homographies only navigate material searches and initialize new
motion. They never become observations and never select a native residual.
"""
from pathlib import Path
import argparse,json,hashlib,time
import cv2,numpy as np
from scipy.optimize import least_squares
from rinkspace.core.native import NativeCore
from rinkspace.camera_reuse_localization import descriptors,match_features,ReuseConfig
from rinkspace.geometry import project
from rinkspace.session.registered_panel_edges import read_registered_panel_edges
from rinkspace.session.source_pattern_verification import check_source_pattern,warped_reference_support,source_pattern_overlap
from prepare_independent_navigation import frames
from capture_independent_materials import source_exclusion,spaced_indices,plain,write


def material_guard(rgb,row,ad):
    uv=np.asarray(row['uv']);normal=np.asarray(row['normal']);xy=np.rint(uv-np.array([2.,4.,6.,9.])[:,None]*normal).astype(int)
    xy[:,0]=np.clip(xy[:,0],0,1919);xy[:,1]=np.clip(xy[:,1],0,1079);c=rgb[xy[:,1],xy[:,0]].astype(float);r,g,b=c.T
    red=(r>g+8)&(r>b+8);blue=(b>r+8)&(b>g+3);purple=(r>g+8)&(b>g+8)&(r>b-35)
    if ad=='Ad6':good=red
    elif ad=='Ad2':good=purple
    elif ad in ('Ad3','Ad7'):good=blue
    elif ad=='Ad4':good=blue|purple
    elif ad=='Ad1':good=blue|(c.max(1)<80)
    else:good=np.ones(len(c),bool)
    return bool(good.sum()>=2),dict(inside_RGB=c.tolist(),inside_probe_uv=xy.tolist(),matched_coloured_material_samples=int(good.sum()),source_material_semantics=ad)


def read_bank(root):
    bank=json.loads((root/'print_references/references.json').read_text('utf8'));refs=[]
    for meta in bank['references']:
        z=dict(np.load(meta['descriptor_npz'],allow_pickle=False))
        z['source_image']=cv2.imread(str(root/meta['shot_id']/'originals'/f'{meta["shot_id"]}_pts{meta["pts"]}.png'))
        z['reference_mask']=cv2.imread(meta['descriptor_mask'],cv2.IMREAD_GRAYSCALE)>0
        refs.append((meta,z))
    return refs


def source_patch_check(image,exclusion,meta,z,h):
    result=check_source_pattern(image,exclusion,meta,z,h,minimum_highpass_ncc=.45,minimum_texture_rms=2.)
    result['supported']=bool(result['supported'] and (result['whole_visible_pattern_correlation']or -1)>=.45)
    result['source_colour_or_shared_letters_alone_are_not_identity']=True
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shots',nargs='+',default=['s07','s08','s09']);p.add_argument('--pts',type=int,nargs='+');p.add_argument('--stride',type=int,default=10);p.add_argument('--source-registration',type=Path);args=p.parse_args()
    root=args.root;index=json.loads((root/'index.json').read_text('utf8'));base=json.loads((root.parent/'baseline.json').read_text('utf8'));refs=read_bank(root);world=dict(np.load(root/'native_world.npz'));core=NativeCore(base['inputs']['library']['path']);args.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);start=time.monotonic()
    write(args.output/'binding.json',dict(source_sha256=index['source']['actual_sha256'],independent_source_capture=True,legacy_new_shot_inputs_read=False,appearance_references=str(root/'print_references/references.json'),source_feature_match_is_boundary_measurement=False,native_camera_prediction_is_identity_evidence=False,fit_eligible=False))
    for sid in args.shots:
        shot=next(s for s in index['shots']if s['shot_id']==sid);cam=2 if shot['known_rig_id']=='end'else 0;folder=args.output/sid;folder.mkdir(exist_ok=True)
        nav=dict(np.load(root/sid/'navigation/poses_reanchored.npz'));pose_by_pts={int(pt):q for pt,q in zip(nav['pts'],nav['poses'])};source_poses=[];representatives={f['actual_pts']for f in shot['frames']};total=0;processed=0;graph=dict(np.load(root/sid/'navigation/source_graph.npz'));graph_index={int(pt):i for i,pt in enumerate(graph['pts'])};last_support={}
        cached_support={}
        cumulative=np.empty((len(graph['pts']),3,3));cumulative[0]=np.eye(3)
        for j,hstep in enumerate(graph['adjacent_h'],1):
            cumulative[j]=hstep@cumulative[j-1];cumulative[j]/=cumulative[j,2,2]
        inverse_cumulative=np.linalg.inv(cumulative);bad_prefix=np.r_[0,np.cumsum(~graph['supported'])]
        if args.source_registration:
            cache_binding=json.loads((args.source_registration/'binding.json').read_text('utf8'))
            if cache_binding['source_sha256']!=index['source']['actual_sha256']:raise ValueError('Cached source appearance belongs to another original')
            for line in (args.source_registration/sid/'registration.jsonl').read_text('utf8').splitlines():
                cache=json.loads(line)
                for item in cache['prints']:
                    if item.get('status')=='source_content_registered' and item.get('source_pattern_check',{}).get('supported'):
                        cached_support.setdefault((item['ad'],item['reference_pts']),[]).append((cache['pts'],np.asarray(item['H_current_from_reference'])))
        path=folder/'source_rows.jsonl'
        if path.exists():raise FileExistsError(path)
        with path.open('w',encoding='utf8',buffering=1) as output,(folder/'registration.jsonl').open('w',encoding='utf8',buffering=1) as audit:
            for fi,(pts,image) in enumerate(frames(index,shot)):
                if args.pts and pts not in args.pts:continue
                if not args.pts and fi%args.stride and pts not in representatives and pts!=shot['processing']['last_pts']:continue
                mask,maskpath=source_exclusion(root,sid,pts);rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);features=None if args.source_registration else descriptors(image,(~mask).astype('uint8')*255);rows=[];pending=[];notes=[];nav_xyz=[];nav_uv=[];ref_poses=[]
                binding=dict(source_sha256=index['source']['actual_sha256'],source_pts=pts,time_base='1/12800',shot_id=sid,pixel_domain='native_distorted',width=1920,height=1080)
                hypotheses=[]
                for meta,z in refs:
                    ad=meta['ad_id'];parent=None;indirect_patch=None
                    if args.source_registration:
                        support=cached_support.get((ad,meta['pts']),[]);before=[q for q in support if q[0]<=pts];after=[q for q in support if q[0]>pts]
                        candidates=([before[-1]]if before else [])+([after[0]]if after else []);validated=[];i=graph_index[pts]
                        for source_pts,hh in candidates:
                            j=graph_index[source_pts];lo,hi=sorted([i,j])
                            if bad_prefix[hi+1]-bad_prefix[lo+1]:continue
                            h=cumulative[i]@inverse_cumulative[j]@hh;h/=h[2,2]
                            if not np.isfinite(h).all() or np.linalg.det(h[:2,:2])<=0:continue
                            check=source_patch_check(image,mask,meta,z,h)
                            if check['supported']:validated.append((check.get('whole_visible_pattern_correlation',0),source_pts,h,check))
                        if validated:
                            _,parent,h,indirect_patch=max(validated,key=lambda q:q[0]);match=(np.empty((0,2)),np.empty((0,2)),h)
                        else:match=None
                    else:match=match_features((z['keypoints'],z['descriptors']),features,ReuseConfig(min_matches=6,ransac_px=3.))
                    note=dict(ad=ad,source_content_matched=match is not None,native_projection_used_for_identity=False,reference_pts=meta['pts'],source_graph_parent_pts=parent,source_graph_transfer_is_navigation_only=parent is not None)
                    if match is None:
                        notes.append(note);continue
                    left,right,h=match;errors=np.linalg.norm(project(h,left)-right,axis=1)
                    if np.linalg.det(h[:2,:2])<=0 or not np.isfinite(h).all():
                        note['status']='invalid_source_transfer';notes.append(note);continue
                    patch=indirect_patch if indirect_patch is not None else source_patch_check(image,mask,meta,z,h);note['source_pattern_check']=patch
                    if not patch['supported']:
                        note['status']='source_pattern_mismatch';notes.append(note);continue
                    uv=project(h,z['projected_uv']);offsets=z['edge_offsets'];pieces=[uv[offsets[j]:offsets[j+1]]for j in range(4)]
                    if not np.isfinite(uv).all() or max(np.ptp(uv,axis=0))>10000:
                        note['status']='unsupported_source_extent';notes.append(note);continue
                    hypotheses.append((meta,z,h,patch,note,left,right,errors,uv,pieces,offsets))
                best={}
                for hypothesis in hypotheses:
                    identity=hypothesis[0]['ad_id'];score=hypothesis[3]['whole_visible_pattern_correlation']
                    if identity not in best or score>best[identity][3]['whole_visible_pattern_correlation']:best[identity]=hypothesis
                # Different physical print instances cannot occupy the same
                # actual source region. Resolve only score-dominant patterns;
                # unresolved competitors remain pending, never duplicate factors.
                choices=list(best.values());supports=[warped_reference_support(q[1]['reference_mask'],q[0]['roi_xyxy_inclusive'],q[2],mask.shape,mask)for q in choices];blocked=set()
                for i in range(len(choices)):
                    for j in range(i):
                        overlap=source_pattern_overlap(supports[i],supports[j])
                        if overlap['intersection_over_smaller']<=.5:continue
                        pi,pj=choices[i][3],choices[j][3];metrics=('whole_visible_pattern_correlation','highpass_pattern_correlation')
                        if all(pi[k]>pj[k]for k in metrics):blocked.add(j)
                        elif all(pj[k]>pi[k]for k in metrics):blocked.add(i)
                        else:blocked.update([i,j])
                        for k in(i,j):choices[k][4].setdefault('source_region_competitors',[]).append(dict(other_ad=choices[j if k==i else i][0]['ad_id'],overlap=overlap))
                for i in sorted(blocked):
                    choices[i][4]['status']='competing_source_pattern_region_pending';notes.append(choices[i][4])
                for meta,z,h,patch,note,left,right,errors,uv,pieces,offsets in (q for i,q in enumerate(choices)if i not in blocked):
                    ad=meta['ad_id']
                    local,record=read_registered_panel_edges(rgb,pieces,mask,binding=dict(binding,independent_source_pattern_match_verified=True),search_radius=20.,spacing=24.,candidate_material_validator=lambda row:material_guard(rgb,row,ad)[0])
                    note.update(status='source_content_registered',matches=len(left),source_keypoints_reference=left.tolist(),source_keypoints_current=right.tolist(),H_current_from_reference=h.tolist(),transfer_median_px=float(np.median(errors)) if len(errors) else None,source_readout=record)
                    for j,pathid in enumerate(meta['native_edge_path_ids']):
                        group=[q for q in local if q['curve']==j]
                        accepted=[]
                        for q in group:
                            supported,colour=material_guard(rgb,q,ad);item=dict(q,path_id=pathid,entity=ad,capture_family='source_registered_panel',source_colour=colour,source_identity_basis='independently read source print content matched across images; physical placement disambiguated by known main/end rig and far-board order',source_reference_pts=meta['pts'])
                            if supported:accepted.append(item)
                            else:pending.append(dict(item,reason='actual_inner_material_does_not_match_print_colour'))
                        for k in spaced_indices([q['uv']for q in accepted]):rows.append(accepted[k])
                    for j in range(4):
                        ids=np.linspace(offsets[j],offsets[j+1]-1,5).round().astype(int);q=uv[ids];good=(q>[-100,-100]).all(1)&(q<[2020,1180]).all(1)
                        nav_xyz.extend(z['world_xyz'][ids[good]]);nav_uv.extend(q[good])
                    if meta['camera_index']==cam:ref_poses.append(z['navigation_pose'])
                    notes.append(note)
                theta=pose_by_pts[pts].copy();navfit=None
                if len(nav_xyz)>=6:
                    xyz=np.asarray(nav_xyz);target=np.asarray(nav_uv)
                    def residual(value):
                        q=core.project_motion(world['rig'][cam],np.tile(value,(6,1)),-2.,1.,1080,0.,xyz);r=q['uv']-target
                        if np.any(q['status'])or not np.isfinite(r).all():return np.full(2*len(xyz),1e5)
                        return r.ravel()
                    choices=[theta,*ref_poses];seed=min(choices,key=lambda q:np.linalg.norm(residual(q)))
                    fit=least_squares(residual,seed,loss='soft_l1',f_scale=3.,x_scale=[.2,.1,1.],diff_step=1e-5,max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-6)
                    if np.isfinite(fit.x).all() and np.max(np.abs(residual(fit.x)))<1e4:theta=fit.x;source_poses.append((pts,theta))
                    navfit=dict(pose=theta.tolist(),rms=float(np.sqrt(np.mean(residual(theta)**2))),navigation_only=True,source_warped_pixels_are_not_observations=True)
                record=dict(pts=pts,source_BGR_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_video_sha256=index['source']['actual_sha256'],mask_path=str(maskpath),mask_sha256=hashlib.sha256(maskpath.read_bytes()).hexdigest(),fit_eligible=False,review_state='pending_source_registered_material_review',rows=rows,pending_rows=pending)
                output.write(json.dumps(plain(record),ensure_ascii=False)+'\n');audit.write(json.dumps(plain(dict(pts=pts,prints=notes,navigation=navfit)),ensure_ascii=False)+'\n')
                if args.pts or pts in representatives or fi%250==0:
                    proof=image.copy();groups={}
                    for row in rows:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),3,(0,0,255),-1);groups.setdefault(row['entity'],[]).append(row['uv'])
                    for ad,xy in groups.items():cv2.putText(proof,ad,tuple(np.rint(np.mean(xy,axis=0)).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,0),2)
                    cv2.imwrite(str(folder/f'{pts}_source_registered.jpg'),proof)
                processed+=1;total+=len(rows)
                if processed%10==0 or args.pts:print(json.dumps(dict(shot=sid,frames=processed,pts=pts,rows=total,matched_prints=sum(n['source_content_matched']for n in notes),current_rows=len(rows),seconds=time.monotonic()-start)),flush=True)
        poses=nav['poses'].copy()
        if source_poses:
            xp=np.array([x[0]for x in source_poses]);d=np.array([x[1]-pose_by_pts[x[0]]for x in source_poses])
            for k in range(3):poses[:,k]+=np.interp(nav['pts'],xp,d[:,k])
        np.savez_compressed(folder/'poses.npz',pts=nav['pts'],poses=poses)
        write(folder/'coverage.json',dict(frames=processed,rows=total,source_registered_navigation_frames=len(source_poses),fit_eligible=False))
if __name__=='__main__':main()
