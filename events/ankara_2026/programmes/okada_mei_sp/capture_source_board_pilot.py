"""Pending source-only board pilot, using current RGB/SAM and public readers.

Color components locate search profiles only. No old observations, native state
or projections are loaded. Dense contact profiles support searches; only the
explicit uniform 48px columns become pending source observations.
"""
from pathlib import Path
import argparse,hashlib,json
from collections import Counter
import cv2,numpy as np,torch

from capture_independent_materials import source_exclusion
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


def color_navigation(rgb,mask):
    floor,info=source_board_floor_limit(rgb,mask)
    h,w=mask.shape;columns=np.arange(48,w-1,48,dtype=int)
    f=rgb.astype(float);blue=(f[:,:,2]-f[:,:,0]>4)&(f[:,:,2]-f[:,:,1]>4)&~mask
    seed=cv2.morphologyEx(blue.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,11),'uint8'));seed[mask]=0
    _,_,stats,_=cv2.connectedComponentsWithStats(seed,8)
    cap_components=[dict(bbox=[int(x),int(y),int(x+cw),int(y+ch)],width=int(cw))
        for x,y,cw,ch,area in stats[1:]if cw>=300 and cw>=4*ch and area>=cw]
    if floor is not None and cap_components:
        cap=max(cap_components,key=lambda c:c['width'])
        below=[c for c in info['components']if c['bbox'][1]>cap['bbox'][3]]
        if below and floor<=cap['bbox'][3]:
            # Warm seating/HUD above the measured cap cannot be this front
            # face's foot. Retain it in the navigation audit; search at an
            # actually observed yellow component below the current RGB cap.
            previous=floor;floor=float(min(c['bbox'][1]for c in below))
            info=dict(info,minimum_y=floor,unpaired_original_minimum_y=previous,
                current_RGB_blue_cap_search_component=cap,
                source_material_order_navigation='blue_cap_above_yellow_front_face_foot',
                rejected_upper_components=[c for c in info['components']if c not in below])
    # The public RGB floor limit navigates the whole source width. Missing
    # chromatic segmentation at one column must not suppress its RGB search.
    bottom=([[float(x),float(floor)]for x in columns]if floor is not None else [])
    # Reuse the public rail reader's source blue-material predicate for search
    # navigation. Its actual signed RGB entry remains the output measurement.
    if floor is not None:blue[int(np.floor(floor)):]=False
    seed=cv2.morphologyEx(blue.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,11),'uint8'));seed[mask]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed,8)
    components=[]
    for i in range(1,count):
        x,y,cw,ch,area=map(int,stats[i])
        if cw>=300 and cw>=4*ch and area>=cw:
            components.append(dict(component=i,bbox=[x,y,x+cw,y+ch],navigation_only=True))
    top=[]
    for x in columns:
        possible=[]
        for component in components:
            x0,y0,x1,y1=component['bbox']
            if not x0<=x<x1:continue
            yy=np.flatnonzero(blue[y0:y1,x])+y0
            if len(yy):possible.append(float(yy.min()))
        if possible:top.append([float(x),min(possible)])
    return np.asarray(bottom).reshape(-1,2),np.asarray(top).reshape(-1,2),dict(
        kickplate=info,blue_components=components,source_navigation_only=True,
        bottom_navigation='fresh_RGB_source_board_floor_limit_across_uniform_native_columns',
        color_mask_boundaries_are_observations=False,uniform_output_query_spacing_px=48.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pts',type=int,nargs='+',required=True);args=parser.parse_args()
    root=args.root.resolve(strict=True);out=guard_project_write(args.output)
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(HERE.parents[3]):raise ValueError('RGB pilot artifacts belong outside OneDrive')
    cv2.setNumThreads(2);torch.set_num_threads(2)
    scope=json.loads((HERE/'rebuild_scope.json').read_text('utf8'));source={**scope['source'],**scope['source_clock']}
    video=Path(source['registered_path']);source_sha=sha(video)
    if source_sha!=source['sha256']:raise ValueError('Original video SHA differs')
    window=next(s['processing']for s in scope['shots']if s['id']=='s07')
    if len(set(args.pts))!=len(args.pts)or any(not window['first_pts']<=p<=window['last_pts']or(p-window['first_pts'])%256 for p in args.pts):
        raise ValueError('Pilot PTS must be distinct registered S07 native exposures')
    authority=board_foot_material_authority(HERE/'board_source_roles.json')
    out.mkdir(parents=True);summaries=[]
    with (out/'pending_source_rows.jsonl').open('w',encoding='utf8')as rows_out:
        for pts in args.pts:
            frames=list(source_frames(video,source,np.array([pts],np.int64)))
            if len(frames)!=1 or frames[0][0]!=pts:raise ValueError('Exact native source frame missing')
            bgr=frames[0][1];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            bgr_sha=hashlib.sha256(bgr.tobytes()).hexdigest();rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            mask,mask_path=source_exclusion(root,'s07',pts);mask_sha=sha(mask_path)
            binding=dict(source_sha256=source_sha,source_pts=pts,shot_id='s07',time_base='1/12800',
                native_hw=[1080,1920],pixel_domain='native_distorted')
            original=out/f's07_{pts}_original.png';cv2.imwrite(str(original),bgr)
            bottom_nav,top_nav,navigation=color_navigation(rgb,mask)
            write(out/f'{pts}_navigation.json',dict(navigation,bottom_queries=bottom_nav,top_queries=top_nav,
                source_BGR_sha256=bgr_sha,old_observations_loaded=False,native_projection_loaded=False))
            rows=[];bottom_info={};top_info={}
            if len(bottom_nav)>=2:
                contact=read_board_contacts(rgb,bottom_nav,mask,binding=binding,authority=authority,spacing=24.)
                profile=contact.pop('original_RGB_profiles');profile_path=out/f'{pts}_bottom_profiles.npz'
                np.savez_compressed(profile_path,**profile)
                proof=dict(path=str(profile_path),sha256=sha(profile_path))
                contact.update(original_RGB_profiles=proof,navigation_is_new_current_RGB=True,existing_old_observations_loaded=False)
                write(out/f'{pts}_bottom_readout.json',contact)
                for local,qi in enumerate(contact['old_query_indices']):
                    if not contact['old_query_supported'][local]:continue
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
                rail['actual_original_RGB_profiles']=proof;write(out/f'{pts}_top_readout.json',rail)
                for i,accepted in enumerate(rail['primary_role_samples']):
                    if not accepted:continue
                    evidence=rail['source_material_role_evidence'][i]
                    rows.append(dict(path_id='board/continuous/top',uv=rail['actual_source_blue_rail_entry_uv'][i].tolist(),kind=1,
                        measurement_operator='public_actual_RGB_blue_cap_upper_entry_signed_peak',
                        source_query_index=evidence['source_query_index'],source_profile_proof=proof,
                        source_material_role_evidence=evidence,model_UV_synthesized=False,fit_eligible=False))
                top_info=dict(navigation_queries=len(top_nav),actual_entry_candidates=len(rail['actual_source_blue_rail_entry_uv']),
                    retained_48px_rows=int(rail['primary_role_samples'].sum()),
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
            overlay=out/f's07_{pts}_source_points.png';cv2.imwrite(str(overlay),canvas)
            frame=dict(pts=pts,shot_id='s07',source_video_sha256=source_sha,source_BGR_sha256=bgr_sha,
                source_RGB_sha256=rgb_sha,source_image=str(original),source_image_sha256=sha(original),
                mask_path=str(mask_path),mask_sha256=mask_sha,rows=rows,fit_eligible=False,
                review_state='pending_root_original_source_boundary_review',source_navigation_only=True,
                native_projection_loaded=False,old_observations_loaded=False)
            rows_out.write(json.dumps(plain(frame),ensure_ascii=False,allow_nan=False)+'\n')
            summary=dict(pts=pts,source_seconds=pts/12800,bottom=bottom_info,top=top_info,
                original=str(original),overlay=str(overlay),source_BGR_sha256=bgr_sha)
            summaries.append(summary);write(out/'summary.json',dict(schema='rinkspace.source_board_pilot.v1',frames=summaries,
                source_video=str(video),source_video_sha256=source_sha,authority=authority,
                only_requested_pts=args.pts,output_query_spacing_px=48.,dense_queries_are_observations=False,
                fit_eligible=False,fit_executed=False,native_state_loaded=False,old_observations_loaded=False))
            print(json.dumps(plain(summary)),flush=True)


if __name__=='__main__':main()
