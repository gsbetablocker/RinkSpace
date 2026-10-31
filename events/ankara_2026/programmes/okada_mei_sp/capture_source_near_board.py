"""Read source-only near blue cap/ice boundaries on every selected exposure."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import cv2,numpy as np
from rinkspace.path_policy import guard_project_write
from rinkspace.vision_near_board_material import extract_candidate
from capture_independent_materials import spaced_indices
from programme_source_frames import iter_registration_source_frames

def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v)for v in x]
    return x
def write(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def full_profile_support(uv,normals,mask):
    # The public profile spans +/-20 from its search anchor, and q50 can move
    # at most 10px. This envelope keeps every original sampled pixel in-frame
    # and out of SAM2/graphics, including a negative-y iceward normal.
    xy=uv[:,None]+normals[:,None]*np.arange(-30.,30.01,.5)[None,:,None]
    inside=((xy>=0)&(xy<=[mask.shape[1]-1,mask.shape[0]-1])).all((1,2))
    out=np.zeros(len(uv),bool)
    for i in np.flatnonzero(inside):
        q=np.rint(xy[i]).astype(int);out[i]=not mask[q[:,1],q[:,0]].any()
    return out

def run(args):
    out=guard_project_write(args.output.resolve())
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cv2.setNumThreads(2);started=time.monotonic()
    frozen={str(p):sha(p)for p in [Path(__file__),Path(extract_candidate.__code__.co_filename),Path(iter_registration_source_frames.__code__.co_filename)]}
    write(out/'binding.json',dict(schema='rinkspace.current_near_blue_cap_capture.v1',shots=args.shots,
        selected_pts=args.pts,source_dependencies=frozen,native_geometry_loaded=False,
        navigation_from_original_RGB=True,source_UV_synthesized=False,
        retained_arc_spacing_px=48.,source_view='top_face_only',fit_eligible=False))
    totals={}
    for sid in args.shots:
        folder=out/sid;folder.mkdir();counts=Counter();coverage=[]
        with (folder/'source_rows.jsonl').open('x',encoding='utf8',buffering=1)as output,(folder/'readouts.jsonl').open('x',encoding='utf8',buffering=1)as raw:
            for fi,frame in enumerate(iter_registration_source_frames(sid,bootstrap_root=args.bootstrap,selected_pts=args.pts)):
                image=frame['image'];mask=frame['source_exclusion_mask'];pt=int(frame['pts'])
                captures=extract_candidate(image,pt,mask);retained=[];all_rows=[];notes=[]
                for ci,capture in enumerate(captures):
                    uv=capture['observed_uv'];normal=capture['outward_normal'];support=full_profile_support(uv,normal,mask)
                    rows=[]
                    for i in np.flatnonzero(support):
                        row=dict(path_id='board/continuous/top',uv=uv[i].tolist(),kind=1,
                            measurement_operator='original_RGB_near_blue_top_cap_ice_q50',
                            source_view='top_face_only',material='blue_cap_to_ice',source_candidate_component=ci,
                            source_query_index=int(i),normal=normal[i].tolist(),
                            normal_interval_px=capture['normal_interval_px'][i].tolist(),
                            source_color_contrast=float(capture['color_contrast'][i]),
                            current_source_RGB_measured=True,model_UV_synthesized=False,fit_eligible=False)
                        rows.append(row)
                    all_rows.extend(rows)
                    # Preserve each actually measured visible fragment, never
                    # thin across an occlusion or missing source span.
                    pieces=[];piece=[]
                    for row in rows:
                        if piece and np.linalg.norm(np.array(row['uv'])-piece[-1]['uv'])>40:
                            pieces.append(piece);piece=[]
                        piece.append(row)
                    if piece:pieces.append(piece)
                    for piece in pieces:retained.extend(piece[i]for i in spaced_indices([r['uv']for r in piece]))
                    notes.append(dict(component=ci,source_box=capture['component'],source_context=capture['source_context'],
                        original_source_rows=len(uv),full_profile_supported_rows=len(rows),
                        excluded_incomplete_or_foreground_profile_indices=np.flatnonzero(~support).tolist(),
                        source_BGR_sha256=capture['source_pixel_sha256']))
                    if capture['source_pixel_sha256']!=frame['source_BGR_sha256']:raise ValueError('Source RGB changed')
                record=dict(pts=pt,source_video_sha256=frame['source_video_sha256'],source_BGR_sha256=frame['source_BGR_sha256'],
                    mask_path=str(frame['mask_path']),mask_sha256=frame['mask_sha256'],mask_sources=frame['mask_sources'],
                    source_exclusion_sha256=frame['source_exclusion_sha256'],rows=retained,
                    fit_eligible=False,review_state='pending_near_top_deck_identity_review',model_UV_synthesized=False)
                output.write(json.dumps(record,ensure_ascii=False)+'\n')
                raw.write(json.dumps(dict(pts=pt,components=notes,all_actual_readouts=all_rows),ensure_ascii=False)+'\n')
                counts['frames']+=1;counts['rows']+=len(retained);counts['frames_with_rows']+=bool(retained)
                counts['source_components']+=len(captures)
                coverage.append(dict(pts=pt,rows=len(retained),components=len(captures)))
                if retained and (args.pts or fi%25==0):
                    proof=image.copy()
                    for row in retained:cv2.circle(proof,tuple(np.rint(row['uv']).astype(int)),2,(0,0,255),-1)
                    cv2.imwrite(str(folder/f'{pt}_near_points.png'),proof)
                if fi%100==0:print(json.dumps(dict(shot=sid,frames=fi+1,pts=pt,current_rows=len(retained),total_rows=counts['rows'],seconds=time.monotonic()-started)),flush=True)
        totals[sid]=dict(counts);write(folder/'coverage.json',dict(counts=dict(counts),frames=coverage))
    changed=[p for p,s in frozen.items()if sha(p)!=s]
    write(out/'summary.json',dict(shots=totals,seconds=time.monotonic()-started,changed_dependencies=changed,fit_executed=False))
    if changed:raise RuntimeError('Capture dependency changed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shots',nargs='+',required=True);p.add_argument('--pts',nargs='+',type=int)
    run(p.parse_args())
