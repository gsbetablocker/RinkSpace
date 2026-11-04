"""Read-only native source/3D review of the bounded support experiment."""
from pathlib import Path
import argparse,copy,io,json,subprocess,shutil
import numpy as np
import av
from PIL import Image,ImageDraw,ImageFont
from .support_refinement import PROGRAMME,reference_for
from .skate_integration import restore_checkpoint_layers
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .support_anchor_factors import lowest_runner_witness
from .action_clip_input import camera25_from_snapshot
from .camera_bridge import rays
from .rebuild_exposure_body import moving_projection
from .footwear_binding import transform
from .skate_checkpoint import validate_checkpoint_identity
from .skate_physics import fk_state


def selected_sources(base,output):
    old=base/'action_integration_1103/spin_s05'
    return [('original_A','原 A',old/'A/fitted/result.npz'),
            ('old_root','旧 root',old/'root_repair/A_pose/fitted/result.npz'),
            ('support_stage1','首轮保留2步',output/'original_A/fitted/result.npz'),
            ('support_new','新联合候选（未验收）',output/'stage1_retained/fitted/result.npz')]


def context_config(project,base,output):
    player=output/'player';player.mkdir(exist_ok=True)
    old=base/'action_integration_1103/spin_s05';p=load(old/'input.npz')
    keep=(p['pts']>=87.5*12800)&(p['pts']<=92.5*12800);n=len(p['pts'])
    sliced={k:v[keep].copy()if isinstance(v,np.ndarray)and v.ndim>0 and v.shape[0]==n else v for k,v in p.items()}
    np.savez_compressed(player/'input.npz',**sliced)
    compact=player/'compact_smplx.npz'
    if not compact.exists():compact.write_bytes((old/'compact_smplx.npz').read_bytes())
    variants=[]
    for key,label,path in selected_sources(base,output):
        packet=load(path);np.testing.assert_array_equal(packet['pts'],p['pts'])
        packet={k:v[keep].copy()if isinstance(v,np.ndarray)and v.ndim>0 and v.shape[0]==n else v for k,v in packet.items()}
        destination=player/key;destination.mkdir(exist_ok=True)
        np.savez_compressed(destination/'result.npz',**packet)
        variants.append(dict(id=key,label=label,kind='fullbody',result=key+'/result.npz',
            shell_topology='saved',body_reference_label='same_GVHMR_native_camera_and_shared_skate',
            body_refinement_intervals_seconds=[[87.5,92.5]]if key.startswith('support_')else[]))
    source=json.loads((output/'source_review/selection.json').read_text(encoding='utf-8'))
    original=json.loads((project/PROGRAMME/'viewer3d_action_spin_s05_1103.json').read_text(encoding='utf-8'))
    config=dict(original,shot='spin_support_1104',source_shot='s05',source_run=str(player),
        default_variant='support_new',initial_source_pts=88*12800,initial_view='side',
        initial_wear_inspection=False,comparison_variants=['original_A','old_root','support_new'],
        variants=variants,source_video=source['source_path'],source_time_base=[1,12800],
        support_observations=str(project/PROGRAMME/'support_spin_1104.json'),support_foot=1,
        support_world_regions=str(output/'support_world_regions.json'),
        review_status='New candidate is unaccepted: review-region and world-height failures remain.',
        root_repair_status='Historical root is a retained comparison; the new joint candidate is unaccepted.',
        body_window_scope='87.5–92.5 context; 88–92 core. Full 76–95 and original S01 retained separately.')
    target=project/PROGRAMME/'viewer3d_spin_support_1104.json'
    if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    write_json(target,config)
    return target


def references(project,base,output):
    A,camera,info,_=reference_for(project,base,output,'original_A')
    old,_,_,_=reference_for(project,base,output,'old_root')
    new=object.__new__(type(A));new.__dict__.update(A.__dict__)
    new.binding=copy.copy(A.binding);new.binding.x=A.binding.x.copy();new.new_refinements=[]
    restore_checkpoint_layers(new,load(output/'stage1_retained/fit/candidate.npz'),include_primary=True)
    return {'original_A':A,'old_root':old,'support_new':new},camera,info


def world_regions(document,camera,info):
    lookup={int(p):i for i,p in enumerate(camera.p['pts'])};result=[]
    for row in document['observations']:
        if row['kind']!='support_contact_region':continue
        i=lookup[row['pts']];v=row['centre_uv'][1]
        t=camera.p['motion_times'][i]+float(camera.p['rig'][37])*(v/1079.-.5)
        pose=camera.core.evaluate_motion_camera(camera.p['rig'],camera.p['controls'],
            float(camera.p['first_control_time']),float(camera.p['control_dt']),np.array([t]))
        cam25=camera25_from_snapshot(pose)[0]
        theta=np.linspace(0,2*np.pi,65)
        uv=np.asarray(row['centre_uv'])+np.c_[np.cos(theta),np.sin(theta)]@np.linalg.cholesky(row['region_shape_px2']).T
        direction,valid=rays(uv,cam25);C=cam25[:3]*info['metres_per_scene_unit']
        distance=-C[2]/direction[:,2];valid &= distance>0
        if not valid.all():raise ValueError('Source contact region has invalid conditional plane rays')
        xyz=C+distance[:,None]*direction
        result.append(dict(id=row['id'],pts=row['pts'],foot=row['foot'],role=row['role'],
            world_polyline_m=xyz.tolist(),source_region_px2=row['region_shape_px2'],
            interpretation='Fixed observed-row native camera / nominal Z=0 conditional image footprint; not contact truth or another optimization factor',
            unmodelled_uncertainty=['camera calibration','reference surface height','row timing']))
    return dict(schema='rinkspace.conditional-support-world-regions.v1',rows=result,
        units='conditional scene metres; exact native Brown inverse at fixed observed-row camera',
        independent_evidence=False,used_as_optimization_factor=False)


def projected_and_runner(project,base,output,refs,camera,info):
    original=load(base/'action_integration_1103/spin_s05/A/fitted/result.npz')
    frames=np.flatnonzero((original['pts']>=87.5*12800)&(original['pts']<=92.5*12800))
    times=(refs['original_A'].times[frames,None]+np.linspace(-.5,.5,5)*float(camera.p['rig'][37])).ravel()
    projections={};numeric={'pts':original['pts'][frames]};metrics={}
    for key,ref in refs.items():
        mid=ref.world(ref.at(ref.times[frames]));nodes=ref.world(ref.at(times))
        p=ref.binding.packet();edge=p['edges']
        R,O=nodes['footwear']['boot_R'],nodes['footwear']['boot_O']
        edge_world=np.stack([transform(R,O,edge[k])for k in range(2)],axis=2)
        uv=moving_projection(camera,edge_world.reshape(len(frames),5,324,3),frames)['uv'].reshape(len(frames),2,2,81,2)
        coco=moving_projection(camera,nodes['world'].reshape(len(frames),5,ref.layout.point_count,3),frames)['uv'][:,ref.layout.body_landmarks]
        metric_arcs=copy.deepcopy(p['template']['contact_arcs'])
        for arc in metric_arcs:
            arc['center']=(np.asarray(arc['center'])*info['length_export_ratio']).tolist()
            arc['radius_m']*=info['length_export_ratio']
        Rmid=mid['footwear']['boot_R'];Omid=mid['footwear']['boot_O']*info['length_export_ratio']
        runner=np.stack([lowest_runner_witness(metric_arcs,Rmid[i,1],Omid[i,1],
            plane_normal_world=[0,0,1])['world_point']for i in range(len(frames))])
        numeric[key+'_runner_world_m']=runner;numeric[key+'_P_world_m']=mid['P']*info['length_export_ratio']
        projections[key]=dict(edges=uv,coco=coco)
        core=(numeric['pts']>=88*12800)&(numeric['pts']<=92*12800);xyz=runner[core]
        metrics[key]=dict(core_exposures=int(core.sum()),runner_XY_span_m=np.ptp(xyz[:,:2],axis=0).tolist(),
            runner_XY_path_m=float(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1).sum()),
            runner_raw_reference_Z_range_m=[float(xyz[:,2].min()),float(xyz[:,2].max())],
            interpretation='Exact continuous model runner minimum, not a fixed material point or actual-contact truth')
    np.savez_compressed(output/'review_trajectories.npz',**numeric)
    np.savez_compressed(output/'review_projected_pixels.npz',pts=numeric['pts'],
        **{key+'_'+field:values for key,packet in projections.items()for field,values in packet.items()})
    write_json(output/'runner_comparison.json',metrics)
    return numeric['pts'],projections


def movie(output,document,pts,projected):
    source=json.loads((output/'source_review/selection.json').read_text(encoding='utf-8'))['source_path']
    old=output.parent/'action_integration_1103/source_cache/spin_s05.native50.npz'
    raw=load(old);lookup={int(p):i for i,p in enumerate(raw['pts'])}
    boxes=np.asarray(raw['bbox'],float).copy();finite=np.isfinite(boxes).all(1)
    display_crop_fallback_pts=raw['pts'][~finite].tolist()
    for axis in range(4):
        boxes[:,axis]=np.interp(raw['pts'],raw['pts'][finite],boxes[finite,axis])
    contact_rows=sorted([r for r in document['observations']if r['kind']=='support_contact_region'],key=lambda r:r['pts'])
    contact_clock=np.array([r['pts']for r in contact_rows]);contact_uv=np.array([r['centre_uv']for r in contact_rows])
    wanted={int(p):i for i,p in enumerate(pts)if 88*12800<=p<=92*12800}
    video=output/'source_A_root_new_88_92.mp4';ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('Existing ffmpeg not found')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    colors=[(248,120,32),(32,220,245)]
    keys=[('original_A','原 A'),('old_root','旧 root'),('support_new','新联合候选：未验收')]
    edges_coco=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    cmd=[ffmpeg,'-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1440x960',
        '-framerate','50','-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','veryfast','-pix_fmt','yuv420p',str(video)]
    if video.exists():
        archived=output/(video.stem+'_before_'+digest(video)[:12]+'.mp4')
        if not archived.resolve().is_relative_to(output.resolve()):raise ValueError('Archive outside output')
        if archived.exists():raise FileExistsError(archived)
        video.rename(archived)
    log=output/'video_encode.log';count=0;saved=[]
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW}if hasattr(subprocess,'CREATE_NO_WINDOW')else{}
    with log.open('wb')as err:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err,**kwargs)
        try:
            with av.open(source)as container:
                stream=container.streams.video[0];container.seek(88*12800,stream=stream,backward=True)
                for frame in container.decode(stream):
                    p=int(frame.pts)
                    if p>92*12800:break
                    if p not in wanted:continue
                    i=wanted[p];source_rgb=Image.fromarray(frame.to_ndarray(format='rgb24'));native_i=lookup[p]
                    bbox=boxes[native_i]
                    center=np.array([np.interp(p,contact_clock,contact_uv[:,axis])for axis in range(2)])
                    body_rect=[max(0,min(bbox[0]-80,center[0]-110)),0,
                        min(1920,max(bbox[2]+80,center[0]+110)),min(1080,max(bbox[3]+70,center[1]+50))]
                    foot_rect=[center[0]-170,center[1]-130,center[0]+170,center[1]+60]
                    canvas=Image.new('RGB',(1440,960),(22,27,35))
                    for column,(key,label)in enumerate(keys):
                        overlay=source_rgb.copy();draw=ImageDraw.Draw(overlay)
                        for foot in range(2):
                            for edge in range(2):
                                line=projected[key]['edges'][i,foot,edge]
                                draw.line([tuple(v)for v in line],fill=colors[foot],width=3)
                        coco=projected[key]['coco'][i]
                        for a,b in edges_coco:draw.line([tuple(coco[a]),tuple(coco[b])],fill=(170,230,190),width=2)
                        for row in document['observations']:
                            if row['pts']!=p:continue
                            colour=(70,245,100)if row['role']=='fit'else(255,210,40)
                            L=np.linalg.cholesky(row['region_shape_px2']);angle=np.linspace(0,2*np.pi,65)
                            line=np.asarray(row['centre_uv'])+np.c_[np.cos(angle),np.sin(angle)]@L.T
                            draw.line([tuple(v)for v in line],fill=colour,width=3)
                        body=overlay.crop(tuple(map(int,body_rect)));body.thumbnail((460,580))
                        canvas.paste(body,(column*480+(480-body.width)//2,75))
                        foot=overlay.crop(tuple(map(int,foot_rect))).resize((460,272))
                        canvas.paste(foot,(column*480+10,675))
                        ImageDraw.Draw(canvas).text((column*480+10,40),label,font=font,fill='white')
                    d=ImageDraw.Draw(canvas);d.text((12,6),f'源片 {p/12800:.3f}s | PTS {p} | 同一原片裁切；绿色fit／黄色review区域；模型未贴地',font=font,fill='white')
                    process.stdin.write(canvas.tobytes());count+=1
                    if p in [1133568,1151744,1159168,1170944]:
                        name=output/f'comparison_{p}.png';canvas.save(name);saved.append(str(name))
            process.stdin.close();code=process.wait()
            if code:raise RuntimeError('ffmpeg failed; see '+str(log))
        finally:
            if process.poll()is None:process.terminate();process.wait()
    assert count==201,count
    write_json(output/'comparison_video.json',dict(path=str(video),source_seconds=[88.,92.],source_pts=sorted(wanted),
        frames=count,encoded_fps=50,source_time_base=[1,12800],first_last_inclusive=True,
        note='201 original exposures / 4.02s encoded, starts at source88.00 and last source92.00. No synthesized frames.',
        diagnostic_images=saved,no_new_fit_or_inference=True))
    record=json.loads((output/'comparison_video.json').read_text(encoding='utf-8'))
    record['display_crop_fallback_pts']=[p for p in display_crop_fallback_pts if p in wanted]
    record['crop_fallback_semantics']='Adjacent real image boxes interpolated ONLY for display cropping; original exposure/observation/unknown are retained, not used as fit points.'
    record['foot_crop_semantics']='Same source-anchored crop for all candidates. Source-region centres interpolate ONLY the display zoom; review marks never enter fitting/seed selection.'
    write_json(output/'comparison_video.json',record)
    return video


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    packet=load(output/'review_trajectories.npz')
    pts=packet['pts'];core=(pts>=88*12800)&(pts<=92*12800);sec=pts[core]/12800.
    fig,(xy,height)=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for key,label,color in [('original_A','Original A','#dc8318'),('old_root','Old root','#3779bd'),('support_new','Joint source-region candidate (unaccepted)','#b53380')]:
        p=packet[key+'_runner_world_m'][core]
        xy.plot(p[:,0],p[:,1],label=label,color=color,lw=1.5)
        xy.scatter(p[[0,-1],0],p[[0,-1],1],color=color,s=16)
        height.plot(sec,p[:,2],label=label,color=color,lw=1.5)
    regions=json.loads((output/'support_world_regions.json').read_text(encoding='utf-8'))['rows']
    first=True
    for row in regions:
        if not 88*12800<=row['pts']<=92*12800:continue
        p=np.asarray(row['world_polyline_m'])
        xy.add_patch(Polygon(p[:,:2],fill=False,edgecolor='#268b46'if row['role']=='fit'else'#888888',
            alpha=.45,lw=.8,label='Source-region footprint, conditioned on Z=0/camera'if first else None))
        first=False
    xy.set_aspect('equal',adjustable='datalim');xy.set(xlabel='Frozen world X (conditional m)',ylabel='Frozen world Y (conditional m)')
    xy.set_title('Model runner minimum, foot 1\nSame absolute coordinates; no alignment',fontsize=10)
    xy.grid(alpha=.2);xy.legend(fontsize=7)
    height.axhline(0,color='black',ls=':',lw=1)
    height.set(xlabel='Original source seconds',ylabel='Raw height above nominal plane (m)',
        title='Reference height, not measured contact truth')
    height.grid(alpha=.2)
    fig.suptitle('88–92 s: image fitting improved locally; world support remains unresolved',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(output/('runner_source_comparison.'+suffix),dpi=170)
    plt.close(fig)


def verify_saved(project,base,output):
    import hashlib
    ref,camera,info,_=reference_for(project,base,output,'original_A')
    ref.new_refinements=[]
    path=project/PROGRAMME/'spin_support_joint_1104.npz';params=load(path)
    validate_checkpoint_identity(params,ref.fixed_checkpoint_identity)
    if str(params['source_region_sha256'])!=digest(output/'support_regions.json'):raise ValueError('Source regions changed')
    restore_checkpoint_layers(ref,params,include_primary=True)
    saved=load(output/'stage1_retained/fitted/result.npz');value=ref.at(ref.times);ratio=info['length_export_ratio']
    checks=dict(P_world_max=float(np.max(abs(value['P']*ratio-saved['P_world_mid']))),
        Q_max=float(np.max(abs(value['Q']-saved['Q_world_mid']))),
        local_R_max=float(np.max(abs(value['local_R']-saved['local_R_mid']))))
    state=fk_state(ref,value,np.zeros((len(ref.times),69)))
    checks.update(skate_R_max=float(np.max(abs(state['R']-saved['skate_R_world_mid']))),
        skate_O_max=float(np.max(abs(state['O']*ratio-saved['skate_origin_world_mid']))))
    if max(checks.values())>2e-10:raise ValueError(checks)
    pixels=load(output/'review_projected_pixels.npz');lookup={int(p):i for i,p in enumerate(saved['pts'])}
    indices=[lookup[int(p)]for p in pixels['pts']]
    checks['native_projected_COCO_max']=float(np.max(abs(pixels['support_new_coco']-saved['COCO_uv'][indices])))
    if checks['native_projected_COCO_max']>1e-7:raise ValueError('Saved source projection mismatch')
    protected=json.loads((output/'immutable_baselines.json').read_text(encoding='utf-8'))
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items()}
    if not all(unchanged.values()):raise ValueError('Protected baseline changed')
    mask=saved['support_image_in_optimization']
    if np.count_nonzero(mask[:,0])!=0 or np.count_nonzero(mask[:,1])!=8:raise ValueError('Unexpected source-use mask')
    review=json.loads((output/'profile.json').read_text(encoding='utf-8'))['review_exposures']
    if mask[np.isin(saved['pts'],review)].any():raise ValueError('Review labels used by fit')
    video=output/'source_A_root_new_88_92.mp4';times=[]
    with av.open(str(video))as container:
        for frame in container.decode(container.streams.video[0]):times.append(float(frame.pts*frame.time_base))
    if len(times)!=201 or max(abs(np.diff(times)-.02))>1e-9:raise ValueError('Diagnostic video cadence mismatch')
    sidecar=json.loads((output/'comparison_video.json').read_text(encoding='utf-8'))
    if sidecar['source_pts']!=list(range(88*12800,92*12800+1,256)):raise ValueError('Original exposure mapping changed')
    record=dict(schema='rinkspace.support-spin-replay-check.v1',parameters=str(path),parameter_sha256=digest(path),
        checks=checks,immutable_files=unchanged,source_pts_step=256,time_base=[1,12800],full_saved_exposures=len(saved['pts']),
        context_exposures=251,core_exposures=201,new_fit_observation_mask_counts=[0,8],review_not_used=True,
        source_support_nonunknown=int(np.count_nonzero(saved['source_support_mode_mid']!='unknown')),
        video_frames=len(times),encoded_duration_seconds=times[-1]+.02,original_source_seconds=[88.,92.],
        no_fit_performed=True,actual_gui_opened=False,native_library=info['native_library'])
    write_json(output/'parameter_replay_1104.json',record)
    print(json.dumps(record['checks']),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--base-root',required=True);parser.add_argument('--output',required=True);parser.add_argument('--video',action='store_true')
    parser.add_argument('--video-only',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv);project,base,output=map(Path,[args.project,args.base_root,args.output])
    document=json.loads((output/'support_regions.json').read_text(encoding='utf-8'))
    if args.verify_only:
        verify_saved(project,base,output);return
    if args.plot_only:
        plot(output);return
    if args.video_only:
        saved=load(output/'review_projected_pixels.npz')
        projected={key:{field:saved[key+'_'+field]for field in ['edges','coco']}for key in ['original_A','old_root','support_new']}
        video=movie(output,document,saved['pts'],projected)
        print(json.dumps(dict(video=str(video),source_seconds=[88.,92.]),ensure_ascii=False),flush=True)
        return
    refs,camera,info=references(project,base,output)
    write_json(output/'support_world_regions.json',world_regions(document,camera,info))
    config=context_config(project,base,output)
    pts,projected=projected_and_runner(project,base,output,refs,camera,info)
    if args.video:movie(output,document,pts,projected)
    print(json.dumps(dict(config=str(config),runner_comparison=str(output/'runner_comparison.json'),
        video=str(output/'source_A_root_new_88_92.mp4')if args.video else None),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
