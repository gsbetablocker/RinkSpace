"""Pending source-only board pilot, using current RGB/SAM and public readers.

Color components locate search profiles only. Optional frozen native projections
help distinguish board identities; they never supply measured UVs. Dense contact
profiles support searches; only explicit 48px columns become source observations.
"""
from pathlib import Path
import argparse,hashlib,json
from collections import Counter
import cv2,numpy as np,torch

from capture_independent_materials import source_exclusion
from programme_source_frames import iter_registration_source_frames
from rinkspace.path_policy import guard_project_write
from rinkspace.review.render import source_frames
from rinkspace.session.board_contacts import read_board_contacts
from rinkspace.session.chromatic_line_navigation import source_board_floor_limit
from rinkspace.vision_source_topology import board_foot_material_authority,read_source_board_rail_material_profiles

HERE=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()


def plain(value):
    if hasattr(value,'detach'):return plain(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float)and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v)for v in value]
    return value


def write(path,value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def _source_boundary_lines(active):
    """Long source-material/white interfaces, used only to organize searches."""
    height,width=active.shape
    proposals=cv2.HoughLinesP(active.astype('uint8')*255,1,np.pi/1440,80,
        minLineLength=250,maxLineGap=48)
    if proposals is None:return []
    yy,xx=np.where(active);lines=[]
    for x0,y0,x1,y1 in proposals.reshape(-1,4):
        if abs(x1-x0)<250 or abs(y1-y0)>.2*abs(x1-x0):continue
        slope=(y1-y0)/(x1-x0);intercept=y0-slope*x0
        support=abs(yy-(slope*xx+intercept))<=3
        if np.count_nonzero(support)<250:continue
        fit=cv2.fitLine(np.c_[xx[support],yy[support]].astype('float32'),
            cv2.DIST_L1,0,.01,.01).ravel()
        if abs(fit[0])<1e-6:continue
        slope=float(fit[1]/fit[0]);intercept=float(fit[3]-slope*fit[2])
        support=abs(yy-(slope*xx+intercept))<=3;columns=np.unique(xx[support])
        if len(columns)<250:continue
        if any(max(abs(intercept-line['intercept']),
            abs(intercept+slope*width-line['intercept']-line['slope']*width))<7
            for line in lines):continue
        lines.append(dict(slope=slope,intercept=intercept,columns=columns,
            source_support_columns=len(columns),navigation_only=True))
    return sorted(lines,key=lambda line:line['source_support_columns'],reverse=True)


def _source_band_queries(material,lower_support,line):
    """Find actual column-wise band entries near one continuous source strip."""
    height,width=material.shape;entries=[]
    for x in range(4,width-4,4):
        predicted=line['intercept']+line['slope']*x
        lo=max(0,int(np.floor(predicted))-4);hi=min(height,int(np.ceil(predicted))+5)
        if lo>=hi:continue
        candidates=np.flatnonzero(lower_support[lo:hi,x])+lo
        if not len(candidates):continue
        y=int(candidates[np.argmin(abs(candidates-predicted))]);last=y
        while y>0 and material[y-1,x]:y-=1
        entries.append([x,y,last-y+1])
    if len(entries)<2:return np.empty((0,2)),dict(status='source_band_entry_unresolved')
    entries=np.asarray(entries,float);thickness=float(np.median(entries[:,2]))
    # A colour region connected far above the narrow rail is not its upper
    # boundary. Missing columns remain absent rather than being interpolated.
    entries=entries[entries[:,2]<=max(6.,4*thickness)]
    if len(entries)<2:return np.empty((0,2)),dict(status='source_band_thickness_unresolved')
    fit=cv2.fitLine(entries[:,:2].astype('float32'),cv2.DIST_L1,0,.01,.01).ravel()
    slope=float(fit[1]/fit[0]);intercept=float(fit[3]-slope*fit[2])
    residual=entries[:,1]-(intercept+slope*entries[:,0])
    mad=float(np.median(abs(residual-np.median(residual))));tolerance=max(4.,3*1.4826*mad)
    supported=entries[abs(residual)<=tolerance]
    if len(supported)<2 or np.ptp(supported[:,0])<300:
        return np.empty((0,2)),dict(status='source_band_lacks_long_continuity')
    queries=supported[np.isin(supported[:,0],np.arange(48,width-1,48)),:2]
    return queries,dict(status='continuous_source_material_band',slope=slope,intercept=intercept,
        source_entry_tolerance_px=tolerance,public_entry_tolerance_px=tolerance+2.,
        source_support_columns=len(supported),source_support_span_px=float(np.ptp(supported[:,0])),
        source_median_band_thickness_px=thickness,source_entry_fit_is_navigation_only=True,
        fitted_UV_used_as_measurements=False)


def curve_distances(points,curve):
    points=np.asarray(points,float).reshape(-1,2);curve=np.asarray(curve,float)
    a,b=curve[:-1],curve[1:];valid=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    a,b=a[valid],b[valid];d=b-a;length2=np.sum(d*d,axis=1)
    if not len(a):return np.full(len(points),np.inf)
    t=np.clip(np.einsum('nki,ki->nk',points[:,None]-a,d)/np.maximum(length2,1e-15),0,1)
    return np.linalg.norm(points[:,None]-(a+t[:,:,None]*d),axis=2).min(1)


def color_navigation(rgb,mask,model_curves=None):
    # The public ice-floor minimum can point to warm seating or a graphic.
    # Keep its evidence, but choose a cap/foot pair from the current source.
    _,floor_info=source_board_floor_limit(rgb,mask)
    rgb=np.asarray(rgb,np.uint8);mask=np.asarray(mask,bool);height,width=mask.shape
    f=rgb.astype(float);hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    blue=(f[:,:,2]-f[:,:,0]>4)&(f[:,:,2]-f[:,:,1]>4)&~mask
    yellow=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=25)&(hsv[:,:,2]>=70)&~mask
    white=(rgb.min(2)>120)&(hsv[:,:,1]<25)&~blue&~yellow&~mask
    # The white front-face separator below blue and white ice below yellow
    # authenticate material order. Their pixels locate profiles, never UVs.
    white_below=np.zeros(mask.shape,np.uint8)
    for dy in range(2,7):white_below[:-dy]+=white[dy:]
    cap_lower=blue&(white_below>=3);foot_lower=yellow&(white_below>=3)
    caps=_source_boundary_lines(cap_lower);feet=_source_boundary_lines(foot_lower);pairs=[]
    for cap in caps:
        for foot in feet:
            columns=np.intersect1d(cap['columns'],foot['columns'])
            if len(columns)<300:continue
            upper=cap['intercept']+cap['slope']*columns
            lower=foot['intercept']+foot['slope']*columns
            if np.percentile(lower-upper,10)<24:continue
            samples=np.rint(upper[:,None]+np.linspace(.05,.95,31)*(lower-upper)[:,None]).astype(int)
            if np.any(samples<0)or np.any(samples>=height):continue
            clear=~mask[samples,columns[:,None]];face=white[samples,columns[:,None]]
            face_columns=int(np.count_nonzero(np.mean(face,axis=1)>.1))
            if face_columns<250:continue
            face_fraction=float(face.sum()/max(clear.sum(),1))
            guide=None
            if model_curves is not None:
                x=np.quantile(columns,[0,.25,.5,.75,1])
                dt=curve_distances(np.c_[x,cap['intercept']+cap['slope']*x],model_curves[0])
                db=curve_distances(np.c_[x,foot['intercept']+foot['slope']*x],model_curves[1])
                guide=float(np.mean(dt)+np.mean(db))
                if max(np.max(dt),np.max(db))>48.:continue
            pairs.append(dict(cap=cap,foot=foot,score=float(len(columns)*(1+face_fraction)),
                model_identity_navigation_distance=guide,
                shared_source_columns=len(columns),white_face_supported_columns=face_columns,
                white_face_fraction=face_fraction))
    audit=dict(kickplate=floor_info,source_navigation_only=True,
        selection='wide_continuous_blue_cap_white_front_face_yellow_foot_white_ice_pair',
        color_mask_boundaries_are_observations=False,uniform_output_query_spacing_px=48.,
        public_floor_minimum_used_as_navigation=False,paired_candidates=len(pairs))
    if not pairs:
        return np.empty((0,2)),np.empty((0,2)),dict(audit,status='paired_source_board_unresolved')
    chosen=(max(pairs,key=lambda pair:pair['score']) if model_curves is None else
        min(pairs,key=lambda pair:(pair['model_identity_navigation_distance'],-pair['score'])))
    bottom,bottom_boundary=_source_band_queries(yellow,foot_lower,chosen['foot'])
    top,top_boundary=_source_band_queries(blue,cap_lower,chosen['cap'])
    compact=lambda line:{key:value for key,value in line.items()if key!='columns'}
    audit.update(status='paired_source_board_supported',
        selected_pair={**{k:v for k,v in chosen.items()if k not in ('cap','foot')},
            'cap_lower_interface':compact(chosen['cap']),'foot_lower_interface':compact(chosen['foot'])},
        top_boundary=top_boundary,bottom_boundary=bottom_boundary,
        bottom_navigation='fresh_actual_yellow_band_entries_of_selected_continuous_source_pair',
        top_navigation='fresh_actual_blue_band_entries_of_selected_continuous_source_pair')
    return bottom,top,audit


def source_top_continuity(uv,navigation):
    """Reject isolated public UVs; never replace them with a fitted boundary."""
    uv=np.asarray(uv,float).reshape(-1,2);boundary=navigation.get('top_boundary',{})
    if boundary.get('status')!='continuous_source_material_band':return np.zeros(len(uv),bool)
    expected=boundary['intercept']+boundary['slope']*uv[:,0]
    return np.isfinite(uv).all(1)&(abs(uv[:,1]-expected)<=boundary['public_entry_tolerance_px'])


def source_bottom_continuity(uv,navigation):
    """Keep measured yellow exits on the same source strip, not a person's skin."""
    uv=np.asarray(uv,float).reshape(-1,2)
    boundary=navigation.get('bottom_boundary',{})
    if boundary.get('status')!='continuous_source_material_band':
        return np.zeros(len(uv),bool)
    valid=np.isfinite(uv).all(1)
    if np.count_nonzero(valid)<3:return valid
    # Colour masks can include the yellow reflection below the actual foot.
    # Check continuity of the public material measurements themselves instead.
    fit=cv2.fitLine(uv[valid].astype('float32'),cv2.DIST_L1,0,.01,.01).ravel()
    normal=np.array([-fit[1],fit[0]])
    signed=(uv-fit[2:4])@normal
    centre=float(np.median(signed[valid]));mad=float(np.median(abs(signed[valid]-centre)))
    return valid&(abs(signed-centre)<=max(4.,3*1.4826*mad))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shot',choices=['s07','s08','s09'],default='s07')
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--pts',type=int,nargs='+')
    selection.add_argument('--all-frames',action='store_true')
    parser.add_argument('--sparse-proofs',action='store_true',help='Save source PNGs only at representative exposures')
    parser.add_argument('--model-navigation',type=Path,help='Frozen camera/board curves for identity search only; never source UV')
    parser.add_argument('--proof-pts',type=int,nargs='*',default=[])
    args=parser.parse_args()
    root=args.root.resolve(strict=True);out=guard_project_write(args.output)
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(HERE.parents[3]):raise ValueError('RGB pilot artifacts belong outside OneDrive')
    cv2.setNumThreads(2);torch.set_num_threads(2)
    scope=json.loads((HERE/'rebuild_scope.json').read_text('utf8'));source={**scope['source'],**scope['source_clock']}
    video=Path(source['registered_path']);source_sha=sha(video)
    if source_sha!=source['sha256']:raise ValueError('Original video SHA differs')
    window=next(s['processing']for s in scope['shots']if s['id']==args.shot)
    if args.all_frames:args.pts=list(range(window['first_pts'],window['last_pts']+256,256))
    if len(set(args.pts))!=len(args.pts)or any(not window['first_pts']<=p<=window['last_pts']or(p-window['first_pts'])%256 for p in args.pts):
        raise ValueError(f'Pilot PTS must be distinct registered {args.shot.upper()} native exposures')
    authority=board_foot_material_authority(HERE/'board_source_roles.json')
    out.mkdir(parents=True);summaries=[]
    model_navigation=None
    if args.model_navigation:
        data=np.load(args.model_navigation,allow_pickle=False);model_navigation={}
        for pt,uv,status in zip(data['pts'],data['uv'],data['status']):
            uv=uv.copy();uv[status!=0]=np.nan;model_navigation[int(pt)]=uv
        write(out/'model_navigation_binding.json',dict(path=str(args.model_navigation.resolve()),
            sha256=sha(args.model_navigation),source_UV_synthesized=False,identity_search_only=True))
    proof_pts=set(args.pts if not(args.all_frames or args.sparse_proofs) else [args.pts[0],args.pts[-1]])
    proof_pts.update(args.proof_pts)
    if args.all_frames or args.sparse_proofs:
        proof_pts.update(round(t*50)*256 for t in (144.,146.,148.,150.,157.6,159.6,167.4,167.6,168.,175.6,178.,184.,187.6,188.6))
    with (out/'pending_source_rows.jsonl').open('w',encoding='utf8')as rows_out:
        for source_frame in iter_registration_source_frames(args.shot,bootstrap_root=root,selected_pts=args.pts):
            pts=source_frame['pts'];bgr=source_frame['image'];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            bgr_sha=hashlib.sha256(bgr.tobytes()).hexdigest();rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            mask=source_frame['source_exclusion_mask'];mask_path=source_frame['mask_path'];mask_sha=source_frame['mask_sha256']
            binding=dict(source_sha256=source_sha,source_pts=pts,shot_id=args.shot,time_base='1/12800',
                native_hw=[1080,1920],pixel_domain='native_distorted')
            original=out/f'{args.shot}_{pts}_original.png'
            if pts in proof_pts:cv2.imwrite(str(original),bgr)
            guide=None if model_navigation is None else model_navigation[pts]
            bottom_nav,top_nav,navigation=color_navigation(rgb,mask,guide)
            write(out/f'{pts}_navigation.json',dict(navigation,bottom_queries=bottom_nav,top_queries=top_nav,
                source_BGR_sha256=bgr_sha,old_observations_loaded=False,native_projection_loaded=guide is not None))
            rows=[];bottom_info={};top_info={}
            if len(bottom_nav)>=2:
                contact=read_board_contacts(rgb,bottom_nav,mask,binding=binding,authority=authority,spacing=24.)
                profile=contact.pop('original_RGB_profiles');profile_path=out/f'{pts}_bottom_profiles.npz'
                np.savez_compressed(profile_path,**profile)
                proof=dict(path=str(profile_path),sha256=sha(profile_path))
                selected=np.asarray(contact['old_query_indices'])[contact['old_query_supported']]
                continuity=np.zeros(len(contact['candidate_source_uv']),bool)
                continuity[selected]=source_bottom_continuity(contact['candidate_source_uv'][selected],navigation)
                contact['paired_source_boundary_continuity_supported']=continuity
                contact.update(original_RGB_profiles=proof,navigation_is_new_current_RGB=True,existing_old_observations_loaded=False)
                write(out/f'{pts}_bottom_readout.json',contact)
                for local,qi in enumerate(contact['old_query_indices']):
                    if not contact['old_query_supported'][local] or not continuity[qi]:continue
                    uv=contact['candidate_source_uv'][qi]
                    rows.append(dict(path_id='board/continuous/bottom',uv=uv.tolist(),kind=1,
                        measurement_operator='public_actual_RGB_first_iceward_yellow_material_fall_q50',
                        source_query_index=int(qi),source_profile_proof=proof,
                        actual_transition_offsets_80_20=contact['transition_offsets_80_20'][qi].tolist(),
                        model_UV_synthesized=False,fit_eligible=False))
                flags=contact['source_role_flags'];bottom_info=dict(navigation_queries=len(bottom_nav),
                    dense_profile_queries=len(contact['queries']),accepted_dense_searches=int(contact['source_material_accepted'].sum()),
                    retained_48px_rows=sum(r['path_id']=='board/continuous/bottom'for r in rows),
                    source_mask_conflicts=int(contact['source_mask_conflict'].sum()),
                    missing_material_bands=int(np.asarray(plain(flags['missing_same_material_band'])).sum()),
                    competing_band_queries=int(np.count_nonzero(contact['competing_same_band_pairs'])),
                    original_profiles=proof)
            else:bottom_info=dict(status='insufficient_current_source_yellow_navigation',retained_48px_rows=0)
            if len(top_nav)>=2:
                rail=read_source_board_rail_material_profiles(rgb,dict(source_binding=binding,RGB_sha256=rgb_sha,
                    actual_source_top_uv=top_nav),source_instances=[dict(object_id='current_source_exclusion',kind='occluder',
                    source_binding=binding,region_mask=mask)],sample_spacing_px=1,exit_requires_current_blue_material=True)
                profiles=rail.pop('actual_original_RGB_profiles');profile_path=out/f'{pts}_top_profiles.npz'
                np.savez_compressed(profile_path,**profiles);proof=dict(path=str(profile_path),sha256=sha(profile_path))
                rail['actual_original_RGB_profiles']=proof
                continuity=source_top_continuity(rail['actual_source_blue_rail_entry_uv'],navigation)
                rail['pilot_paired_source_boundary_continuity_supported']=continuity
                rail['pilot_continuity_guard_changes_public_UV']=False
                write(out/f'{pts}_top_readout.json',rail)
                for i,accepted in enumerate(rail['primary_role_samples']):
                    if not accepted or not continuity[i]:continue
                    evidence=rail['source_material_role_evidence'][i]
                    rows.append(dict(path_id='board/continuous/top',uv=rail['actual_source_blue_rail_entry_uv'][i].tolist(),kind=1,
                        measurement_operator='public_actual_RGB_blue_cap_upper_entry_signed_peak',
                        source_query_index=evidence['source_query_index'],source_profile_proof=proof,
                        source_material_role_evidence=evidence,model_UV_synthesized=False,fit_eligible=False))
                top_info=dict(navigation_queries=len(top_nav),actual_entry_candidates=len(rail['actual_source_blue_rail_entry_uv']),
                    public_primary_role_rows=int(rail['primary_role_samples'].sum()),
                    paired_continuity_rejected_rows=int(np.count_nonzero(rail['primary_role_samples']&~continuity)),
                    retained_48px_rows=int(np.count_nonzero(rail['primary_role_samples']&continuity)),
                    missing_reasons=dict(Counter(r['reason']for r in rail['missing_or_competing_source_roles'])),
                    conflicting_entry_exit_roles=int(np.count_nonzero(~rail['primary_role_samples'])),original_profiles=proof,
                    blue_to_white_exit_is_internal_not_output=True)
            else:top_info=dict(status='insufficient_current_source_blue_navigation',retained_48px_rows=0)
            canvas=bgr.copy()
            for row in rows:
                color=(0,0,255)if row['path_id'].endswith('/bottom')else(255,255,0)
                cv2.circle(canvas,tuple(np.rint(row['uv']).astype(int)),3,color,-1,cv2.LINE_AA)
            cv2.putText(canvas,f'PTS {pts} | pending RGB only | bottom red / top cyan | 48px',(590,35),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(20,240,255),1,cv2.LINE_AA)
            overlay=out/f'{args.shot}_{pts}_source_points.png'
            if pts in proof_pts:cv2.imwrite(str(overlay),canvas)
            frame=dict(pts=pts,shot_id=args.shot,source_video_sha256=source_sha,source_BGR_sha256=bgr_sha,
                source_RGB_sha256=rgb_sha,source_image=str(original)if pts in proof_pts else None,
                source_image_sha256=sha(original)if pts in proof_pts else None,
                mask_path=str(mask_path),mask_sha256=mask_sha,rows=rows,fit_eligible=False,
                review_state='pending_root_original_source_boundary_review',source_navigation_only=True,
                native_projection_loaded=guide is not None,old_observations_loaded=False)
            rows_out.write(json.dumps(plain(frame),ensure_ascii=False,allow_nan=False)+'\n')
            summary=dict(pts=pts,shot_id=args.shot,source_seconds=pts/12800,bottom=bottom_info,top=top_info,
                original=str(original),overlay=str(overlay),source_BGR_sha256=bgr_sha)
            summaries.append(summary)
            if len(summaries)%25==0 or pts==args.pts[-1] or not(args.all_frames or args.sparse_proofs):write(out/'summary.json',dict(schema='rinkspace.source_board_pilot.v1',frames=summaries,
                shot_id=args.shot,source_video=str(video),source_video_sha256=source_sha,authority=authority,
                only_requested_pts=args.pts,output_query_spacing_px=48.,dense_queries_are_observations=False,
                fit_eligible=False,fit_executed=False,native_state_loaded=False,old_observations_loaded=False))
            if len(summaries)%25==0 or pts==args.pts[-1] or not(args.all_frames or args.sparse_proofs):
                print(json.dumps(dict(pts=pts,frames=len(summaries),bottom=bottom_info['retained_48px_rows'],top=top_info['retained_48px_rows'])),flush=True)


if __name__=='__main__':main()
