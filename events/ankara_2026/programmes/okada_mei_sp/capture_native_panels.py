"""Capture modeled four-edge material boundaries and badge exteriors over source shots."""
from pathlib import Path
import argparse,json,time,hashlib,re
import cv2,numpy as np
import run_native_programme as run
from capture_native_centerlines import plain
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel,ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.domain.scene_assets import read_scene_assets
from rinkspace.review.scene import sample_scene
from rinkspace.session.panel_boundaries import read_panel_boundary


PANEL_EDGES=('top_edge','right_vertical','bottom_edge','left_vertical')


def panel_material_metadata(scene,scene_path):
    """Read roles and material meanings from verified current numeric assets."""
    metadata={}
    for reference,assets in read_scene_assets(scene,scene_path):
        asset_path=Path(reference['path'])
        if not asset_path.is_absolute():asset_path=Path(scene_path).parent/asset_path
        instances={item['id']:item for item in assets['instances']}
        templates={item['id']:item for item in assets['templates']}
        for identity in reference.get('instance_ids',[reference.get('instance_id')]):
            template=templates[instances[identity]['template_id']]
            for path in template['paths']:
                path_id=identity+'::'+path['id']
                metadata[path_id]=dict(path_id=path_id,template_id=template['id'],template_path_id=path['id'],
                    role=path.get('role'),boundary_side=path.get('boundary_side'),
                    logical_boundary_meaning=path.get('logical_boundary_meaning'),
                    asset_path=str(asset_path.resolve()),asset_sha256=reference['sha256'])
    return metadata


def with_material_metadata(panel,metadata):
    return dict(panel,edge_metadata=[metadata.get(path,dict(path_id=path,role=None,
        logical_boundary_meaning=None,metadata_status='current_asset_path_not_resolved'))for path in panel['path_ids']])


def native_panel_inventory(cuts,path_metadata=None):
    """Select only complete, unique native four-edge material entities.

    A terminal /pathN identifies an existing native subpath; stripping it is
    only for edge-name matching. Returned path IDs retain their exact spelling.
    No other contour or print bounds can create a panel edge.
    """
    instances={}
    for cut in cuts:
        path_id=cut['id'];normalized=re.sub(r'/path[0-9]+$','',path_id)
        match=re.fullmatch(r'print_instance/(Ad[0-9]+)::(.+)\.('+ '|'.join(PANEL_EDGES)+')',normalized)
        if match is None:continue
        instance,template_prefix,edge=match.groups()
        group=instances.setdefault(instance,{name:[] for name in PANEL_EDGES})
        group[edge].append((path_id,template_prefix))
    panels=[]
    for instance,group in sorted(instances.items(),key=lambda item:(int(item[0][2:]),item[0])):
        if any(len(group[name])!=1 for name in PANEL_EDGES):continue
        panels.append(dict(panel_instance=instance,path_ids=[group[name][0][0] for name in PANEL_EDGES],
            template_path_prefixes=sorted({group[name][0][1] for name in PANEL_EDGES}),
            material_signal='dark',source_identity_basis='complete unique named native boundary paths; physical material scope comes from current asset edge metadata'))
    return [with_material_metadata(panel,path_metadata or {})for panel in panels]


def single_panel(instance,badge_part):
    if instance=='Ad1':
        paths=['print_instance/Ad1::Ad1.'+name for name in PANEL_EDGES]
        material_signal='dark';identity_basis='Ad1 complete blue panel with QR at left and ALL SKATING / ONE APP print; registered instance topology'
    elif instance in ('Ad9','Ad22'):
        paths=[f'print_instance/{instance}::Ad9.print.red_field.boundary.0000']
        material_signal='red';identity_basis=f'{instance} registered red badge logo_outer_boundary; exterior of the complete red source component, excluding internal white lettering and symbols'
    elif instance in ('Ad12','Ad25'):
        paths=[f'print_instance/{instance}::Ad12.shield_print_component001']
        material_signal='blue';identity_basis=f'{instance} registered Ankara blue shield logo_outer_boundary; blue source material selects the component, actual luminance interface supplies UV, internal white marks and adjacent lettering are not its exterior'
    elif instance in ('Ad11','Ad24'):
        suffix='outer_ring.outer' if badge_part=='outer_ring' else 'red_shield'
        paths=[f'print_instance/{instance}::Ad11.{suffix}']
        material_signal='red';identity_basis=f'{instance} federation badge {suffix} exterior; registered complete badge topology distinguishes outer ring, inner shield, central red shield and its white holes'
    else:raise ValueError('Unsupported current native panel instance: '+instance)
    return dict(panel_instance=instance,path_ids=paths,material_signal=material_signal,source_identity_basis=identity_basis)


def capture(args):
    if args.output.exists():raise FileExistsError(args.output)
    m=run.read(args.run_root/'joint_manifest.json');v=dict(np.load(args.state));scene=run.read(m['scene_path'])
    model=SceneParameterModel(scene,Path(m['scene_path']),optimize_scene=True);core=NativeCore(m['native_library'])
    sample=sample_scene(scene,core,Path(m['scene_path']),parameter_state=ParameterState(model.initial_state.layout,v['scene_values']))
    batch=bool(getattr(args,'all_native_panels',False))
    path_metadata=panel_material_metadata(scene,Path(m['scene_path']))
    panels=(native_panel_inventory(sample['cuts'],path_metadata) if batch else
        [with_material_metadata(single_panel(getattr(args,'panel_instance','Ad1'),getattr(args,'badge_part','outer_ring')),path_metadata)])
    if not panels:raise ValueError('No complete unique native four-edge material entities in the current scene')
    paths=[path for panel in panels for path in panel['path_ids']]
    world={c['id']:sample['xyz'][c['begin']:c['end']] for c in sample['cuts'] if c['id'] in paths}
    closed={c['id']:bool(c.get('closed',False)) for c in sample['cuts'] if c['id'] in paths}
    if set(paths)!=set(world):raise ValueError('Requested native panel paths are missing from the current scene')
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time'])
    args.output.mkdir(parents=True);cv2.setNumThreads(2);started=time.monotonic()
    binding=dict(state=str(args.state),state_sha256=run.digest(args.state),scene_sha256=m['scene_file_sha256'],
        path_ids=paths,adapter_sha256=run.digest(__file__),reader_sha256=run.digest(read_panel_boundary.__code__.co_filename),
        visibility_adapter_sha256=run.digest(run.__file__),
        source_exclusion='SAM2 foreground/near-outside and actual broadcast graphics; board top/bottom search margins are not occlusion')
    if batch:binding.update(all_native_panels=True,panel_instances=[p['panel_instance'] for p in panels],panels=panels)
    else:binding.update(panels[0])
    run.write_json(args.output/'binding.json',binding)
    for sid in args.shots:
        vis,frames=run.source_frames(sid,selected_pts=args.pts);shot=next(s for s in m['shots'] if s['id']==sid)
        folder=args.output/sid;folder.mkdir();accepted=[];coverage=[]
        with (folder/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
          for fi,frame in enumerate(frames):
            t=(frame['pts']-shot['reference_pts'])/12800+timeline.offset(shot)
            image=cv2.imread(frame['image']);masks,_=run.surfaces(frame,image)
            if hashlib.sha256(image.tobytes()).hexdigest()!=frame['source_BGR_sha256']:raise ValueError('Source RGB changed')
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);exclusion=~masks['unoccluded_source'];rows=[];readouts={};panel_coverage={}
            source_binding=dict(source_sha256=vis['source_video_sha256'],source_pts=frame['pts'],time_base='1/12800',
                shot_id=sid,pixel_domain='native_distorted',width=1920,height=1080)
            source_record=str(folder/f'{frame["pts"]}.json')
            for panel in panels:
                instance=panel['panel_instance'];panel_paths=panel['path_ids']
                projections=[core.project_motion(v['rig'][shot['camera_index']],v['controls'],timeline.first,timeline.dt,1080,t,world[p]) for p in panel_paths]
                pieces=[]
                for result in projections:
                    piece=result['uv'].copy();piece[result['status']!=0]=np.nan;pieces.append(piece)
                if not any(np.any(np.isfinite(piece[:-1]).all(1)&np.isfinite(piece[1:]).all(1)) for piece in pieces):
                    local_rows=[];record=dict(status='no_supported_local_projection')
                else:local_rows,record=read_panel_boundary(rgb,pieces,exclusion,material_signal=panel['material_signal'],
                    binding=source_binding,closed_paths=[closed[path] for path in panel_paths])
                for row in local_rows:
                    row.update(pts=frame['pts'],path_id=panel_paths[row['curve']],identity=panel_paths[row['curve']],evidence_image=frame['image'],
                        source_BGR_sha256=frame['source_BGR_sha256'],mask_path=frame['mask_path'],mask_sha256=frame['mask_sha256'],source_record=source_record)
                    if batch:row['panel_instance']=instance
                rows.extend(local_rows);readouts[instance]=record
                panel_coverage[instance]=dict(status=record['status'],rows=len(local_rows),path_ids=panel_paths)
            accepted.extend(rows)
            if batch:
                coverage.append(dict(pts=frame['pts'],rows=len(rows),paths=sorted({r['path_id'] for r in rows}),panels=panel_coverage))
                payload=dict(readouts=readouts,rows=rows)
            else:
                record=readouts[panels[0]['panel_instance']]
                coverage.append(dict(pts=frame['pts'],rows=len(rows),paths=sorted({r['path_id'] for r in rows}),status=record['status']))
                payload=dict(readout=record,rows=rows)
            run.write_json(folder/f'{frame["pts"]}.json',plain(payload))
            if args.pts or fi in (0,len(frames)//2,len(frames)-1):
                for row in rows:cv2.circle(image,tuple(np.rint(row['uv']).astype(int)),3,(0,0,255),-1)
                cv2.imwrite(str(folder/f'{frame["pts"]}_source.jpg'),image)
            if fi%25==0 or fi==len(frames)-1:
                message=dict(shot=sid,frames=fi+1,total=len(frames),rows=len(accepted),seconds=time.monotonic()-started)
                log.write(json.dumps(message)+'\n');print(json.dumps(message),flush=True)
        run.write_json(folder/'source_pixels.json',dict(shot=sid,accepted=accepted,source_UV_synthesized=False))
        run.write_json(folder/'coverage.json',coverage)
        run.write_json(folder/'source_verified.json',dict(shot=sid,accepted=accepted,source_packet=str(folder/'source_pixels.json'),
            source_packet_sha256=run.digest(folder/'source_pixels.json'),source_links=str(folder/'coverage.json'),source_links_sha256=run.digest(folder/'coverage.json'),
            fit_eligible=False,review_state='awaiting_source_review',user_visual_acceptance=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shots',nargs='+',required=True);parser.add_argument('--pts',nargs='+',type=int)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--panel-instance',choices=['Ad1','Ad9','Ad22','Ad11','Ad24','Ad12','Ad25'],default='Ad1')
    mode.add_argument('--all-native-panels',action='store_true',help='Read every complete unique native four-edge material boundary per source frame')
    parser.add_argument('--badge-part',choices=['outer_ring','red_shield'],default='outer_ring')
    capture(parser.parse_args())
