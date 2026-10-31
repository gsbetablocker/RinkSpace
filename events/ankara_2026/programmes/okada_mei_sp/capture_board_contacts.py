"""Reread stored board-contact source rows; emit reviewable replacements only."""
from pathlib import Path
import argparse,hashlib,time
import cv2,numpy as np,torch
import run_native_programme as run
from rinkspace.session.board_contacts import read_board_contacts
from rinkspace.vision_source_topology import board_foot_material_authority


def plain(value):
    if hasattr(value,'detach'):return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v)for v in value]
    return value


def capture(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);torch.set_num_threads(2);cv2.setNumThreads(2)
    parent=args.run_root/'full_source'
    z=dict(np.load(parent/'point_observations.npz',allow_pickle=False))
    mapping=run.read(parent/'mapping.json');path='board/continuous/bottom'
    path_index=mapping['path_ids'].index(path)
    authority=board_foot_material_authority(run.HERE/'board_source_roles.json')
    run.write_json(args.output/'binding.json',dict(parent=str(parent),input_sha256=run.digest(parent/'point_observations.npz'),
        adapter_sha256=run.digest(__file__),reader_sha256=run.digest(read_board_contacts.__code__.co_filename),
        material_reader_sha256=run.digest(Path(run.HERE).parents[3]/'src/rinkspace/vision_source_topology.py'),
        authority=authority,only_source_family='board:dense_board/',fit_eligible=False))
    for shot in args.shots:
        folder=args.output/shot;folder.mkdir();rows=[];coverage=[];started=time.monotonic()
        vis,frames=run.source_frames(shot,selected_pts=args.pts)
        for frame in frames:
            pts=frame['pts'];indices=np.flatnonzero((z['pts']==pts)&(z['path_indices']==path_index))
            indices=np.asarray([i for i in indices if str(z['observation_ids'][i]).startswith('board:dense_board/')],int)
            if len(indices)<2:
                coverage.append(dict(pts=pts,status='insufficient_stored_source_navigation',replacement_candidates=0));continue
            im=cv2.imread(frame['image'])
            if hashlib.sha256(im.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source RGB changed')
            mask=cv2.imread(frame['mask_path'],0)>0
            if frame.get('near_outside_ref'):mask|=cv2.imread(frame['near_outside_ref']['file'],0)>0
            binding=dict(source_sha256=vis['source_video_sha256'],source_pts=pts,shot_id=shot,
                time_base='1/12800',native_hw=[1080,1920],pixel_domain='native_distorted')
            old=z['uv'][indices]
            packet=read_board_contacts(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),old,mask,binding=binding,authority=authority)
            profiles=packet.pop('original_RGB_profiles');profile_path=folder/f'{pts}_profiles.npz'
            np.savez_compressed(profile_path,**profiles)
            packet.update(source_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                mask_path=frame['mask_path'],mask_sha256=frame['mask_sha256'],
                original_RGB_profiles=dict(path=str(profile_path),sha256=run.digest(profile_path)),
                old_observation_ids=z['observation_ids'][indices],old_uv=old)
            record=folder/f'{pts}.json';run.write_json(record,plain(packet))
            chosen=packet['old_query_supported']&packet['old_outside_measured_transition']
            for local in np.flatnonzero(chosen):
                query=int(packet['old_query_indices'][local])
                rows.append(dict(pts=pts,path_id=path,kind=1,uv=packet['old_query_fresh_uv'][local].tolist(),
                    replaces_observation_id=str(z['observation_ids'][indices[local]]),old_uv=old[local].tolist(),
                    source_query_index=query,source_record=str(record),source_record_sha256=run.digest(record),
                    evidence_image=frame['image'],source_BGR_sha256=frame['source_BGR_sha256'],
                    mask_path=frame['mask_path'],mask_sha256=frame['mask_sha256'],
                    transition_offsets_80_20=packet['transition_offsets_80_20'][query].tolist(),
                    reason='Stored UV outside the freshly measured yellow-to-ice fall; paired material order and source exclusion checked.',
                    source_UV_synthesized=False))
            coverage.append(dict(pts=pts,old_rows=len(indices),supported_old_queries=int(packet['old_query_supported'].sum()),
                replacement_candidates=int(chosen.sum()),new_fit_rows_from_dense_queries=0))
        source=folder/'source_pixels.json';links=folder/'coverage.json'
        run.write_json(source,dict(shot=shot,candidates=rows,source_UV_synthesized=False,fit_eligible=False))
        run.write_json(links,dict(frames=coverage,seconds=time.monotonic()-started))
        run.write_json(folder/'source_verified.json',dict(shot=shot,accepted=[],source_packet=str(source),
            source_packet_sha256=run.digest(source),source_links=str(links),source_links_sha256=run.digest(links),
            fit_eligible=False,review_state='pending_original_source_review',user_visual_acceptance=False))
        print(shot,len(frames),len(rows),round(time.monotonic()-started,3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--shots',nargs='+',required=True);p.add_argument('--pts',nargs='+',type=int)
    capture(p.parse_args())
