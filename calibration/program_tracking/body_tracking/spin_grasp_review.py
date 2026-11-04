"""Read-only phase comparison and registration of the actual new spin candidate."""
from pathlib import Path
import json,shutil
import numpy as np
from .rebuild_prepare import load,digest
from .fullbody_round import write_json
from .exmr_integration import replay
from .spin_grasp_factors import export_grasp
from .spin_grasp_refinement import archive_paths
PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')


def stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(count=len(a),median=None if not len(a) else float(np.median(a)),p95=None if not len(a) else float(np.percentile(a,95)),maximum=None if not len(a) else float(a.max()))


def compare(base,output,label):
    case=output/'common'/label;root=base/'exmr_spin_compare_1104/native_protocol'
    paths={'gvhmr_fixed':root/'common/gvhmr/initial/fitted/result.npz','exmr_prefoot_old':root/'common/exmr_prefoot/initial/fitted/result.npz','hand_nofoot_initial':case/'initial_current_objective/fitted/result.npz','grasp_candidate':case/'fitted/result.npz'}
    active_path=output/'active_candidate.json'
    if active_path.exists():paths['grasp_candidate']=Path(json.loads(active_path.read_text(encoding='utf8'))['result'])
    located=output/'support_located_candidate.json'
    if located.exists():paths['support_located']=Path(json.loads(located.read_text(encoding='utf8'))['result'])
    midpoint=output/'support_midpoint_candidate.json'
    if midpoint.exists():paths['support_midpoint']=Path(json.loads(midpoint.read_text(encoding='utf8'))['result'])
    packets={k:load(v) for k,v in paths.items()};obs=load(case/'observations.npz');pts=obs['pts'];records={}
    phases={'entry_84_86':[84.,86.],'held_86_8_92_46':[86.8,92.46],'release_92_46_93_2':[92.46,93.2],'exit_93_2_94_8':[93.2,94.8]}
    for name,d in packets.items():
        np.testing.assert_array_equal(d['pts'],pts);by={}
        for phase,(a,b) in phases.items():
            mask=(pts>=round(a*12800))&(pts<=round(b*12800));P=d['P_world_mid'][mask];error=np.linalg.norm(d['COCO_uv']-obs['uv'],axis=-1)
            group=dict(exposures=int(mask.sum()),body_pixels=stats(error[mask&obs['use'].any(1)][obs['use'][mask&obs['use'].any(1)]]),
                pelvis_xy_span_m=np.ptp(P[:,:2],axis=0).tolist(),pelvis_xy_path_m=float(np.linalg.norm(np.diff(P[:,:2],axis=0),axis=1).sum()),raw_reference_min_clearance_m=stats(d['skate_clearance_m_mid'][mask]),source_support_raw_gap_m=stats(d['skate_support_gap_m_mid'][mask]))
            for key,j in [('left_ankle',15),('right_ankle',16),('left_wrist',9),('right_wrist',10)]:group[key+'_native_px']=stats(error[mask&obs['use'][:,j],j])
            if 'palm_to_segment_m_mid' in d:group['palm_to_finite_segment_model_m']=stats(d['palm_to_segment_m_mid'][mask])
            else:group['palm_to_finite_segment_model_m']=dict(status='unavailable in preserved historical default-hand export; wrist not substituted')
            by[phase]=group
        records[name]=dict(result=str(paths[name]),sha256=digest(paths[name]),hand_channel='hand_local_R_mid' in d,phases=by)
    report=dict(source_time_base=[1,12800],source_first_last_pts=pts[[0,-1]].tolist(),same_source_use_mask=True,cases=records,
        caution='G=Pxy and changing runner minima are model diagnostics, not measured athlete displacement; raw reference heights are not actual penetration or flight. No per-candidate trajectory alignment.',source_roles='development fit and reviewed holdout; not independent 3D truth')
    write_json(output/'phase_comparison.json',report);return packets,report


def register(project,base,output,label,packets):
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    player=output/'player';player.mkdir(exist_ok=True);case=output/'common'/label
    for name in ('input.npz','compact_smplx.npz'):
        target=player/name
        if not target.exists():target.write_bytes((case/name).read_bytes())
    original=json.loads((project/PROGRAMME/'viewer3d_spin_models_1104.json').read_text(encoding='utf8'));old=base/'exmr_spin_compare_1104/native_protocol'
    active=json.loads((output/'active_candidate.json').read_text(encoding='utf8')) if (output/'active_candidate.json').exists() else None
    descriptions=[('gvhmr_fixed','原GVHMR固定初值（旧默认手）',old/'common/gvhmr/initial/fitted/result.npz'),('exmr_prefoot_old','原ExMR细化前（旧默认手）',old/'common/exmr_prefoot/initial/fitted/result.npz'),('hand_nofoot_initial','HandMR开／FootMR关初值（真实手通道）',case/'initial_current_objective/fitted/result.npz'),('grasp_candidate','新全链候选（待源图审核）',case/'fitted/result.npz')]
    if active:
        descriptions[-1]=('grasp_candidate','新全链候选（4步，待源图审核）',Path(active['result']))
        descriptions.append(('grasp_stage1','第一阶段3步（保留对照）',case/'fitted/result.npz'))
    located_path=output/'support_located_candidate.json'
    if located_path.exists():
        located=json.loads(located_path.read_text(encoding='utf8'))
        descriptions.append(('support_located','支撑刀直接定位（姿态待审核）',Path(located['result'])))
    midpoint_path=output/'support_midpoint_candidate.json'
    if midpoint_path.exists():
        midpoint=json.loads(midpoint_path.read_text(encoding='utf8'))
        descriptions=[(k,('最低点误作中点（失败保留）' if k=='support_located' else t),p) for k,t,p in descriptions]
        descriptions.append(('support_midpoint','支撑刀中点定位（姿态待审核）',Path(midpoint['result'])))
    oldroot=base/'action_integration_1103/spin_s05/root_repair/A_pose/fitted/result.npz'
    if oldroot.is_file():descriptions.append(('old_root','旧根修正（失败对照）',oldroot))
    variants=[dict(id=k,label=t,kind='fullbody',result=str(p),shell_topology='saved',body_reference_label=label if k in ('hand_nofoot_initial','grasp_candidate','support_located','support_midpoint') else 'preserved_original',body_refinement_intervals_seconds=[[82.,94.8]] if k=='grasp_candidate' else []) for k,t,p in descriptions]
    config=dict(original,shot='spin_grasp_1104',source_run=str(player),default_variant='grasp_candidate',initial_source_pts=round(85.2*12800),initial_view='side',variants=variants,comparison_variants=['gvhmr_fixed','hand_nofoot_initial','grasp_candidate'],support_observations=str(output/'support_and_free_shoe_regions.json'),grasp_hand='left',free_foot=0,relation_document=str(output/'grasp_relations.json'),review_status='New closed-chain numerical candidate; source/3D accuracy not accepted',root_repair_status='Same native camera and world, finite ice and source-supported intervals; no fixed centre',body_window_scope='Single continuous 76-95s state, C2 correction 82-94.8s')
    names=archive_paths(project,output);stem=names['parameter_stem'];legacy=stem=='spin_grasp_1104'
    if located_path.exists():
        config.update(default_variant='support_located',comparison_variants=['hand_nofoot_initial','grasp_candidate','support_located'],support_location_document=str(project/PROGRAMME/'spin_support_location_anchors_1104.json'),root_repair_status='Source-supported conditional blade/ice placement is a separate candidate; articulation unchanged; not independent 3D truth')
    if midpoint_path.exists():
        config.update(default_variant='support_midpoint',comparison_variants=['grasp_candidate','support_midpoint'],support_location_document=str(project/PROGRAMME/'spin_support_midpoint_anchors_1104.json'),review_status='Image/pose failures remain; midpoint placement is a conditional review candidate, not an accepted fit')
    config_name='viewer3d_spin_grasp_1104.json' if legacy else stem+'_viewer.json'
    target=project/PROGRAMME/config_name;assert len(str(target).encode('utf-16-le'))//2<260;write_json(target,config)
    catalog=ResultCatalog(project=project,data_root=AssetStore(project).data,config=target);caches=[]
    for variant in variants:
        print(json.dumps(dict(event='grasp_player_cache',variant=variant['id'])),flush=True);cache=prepare_variant(catalog,variant);caches.append(dict(variant=variant['id'],path=str(cache)))
    write_json(output/'player_registration.json',dict(config=str(target),caches=caches,source_frames=len(catalog.pts),views=['free','support_top','support_side','grasp'],same_pts_and_world=True,no_visible_gui_opened=True))
    launcher_name='START_3D_GRASP' if legacy else 'START_3D_'+stem.upper()
    command=project/(launcher_name+'.ps1');text="""[CmdletBinding()]
param([string]$Variant='grasp_candidate',[switch]$Fresh,[switch]$Wait,[switch]$Check)
$ErrorActionPreference='Stop'
$configPath=Join-Path $PSScriptRoot 'events/ankara_2026/programmes/okada_mei_sp/viewer3d_spin_grasp_1104.json'
if ($Check) {
 $config=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
 if (-not (Test-Path -LiteralPath $config.source_video)) { throw 'Source video missing' }
 foreach ($item in $config.variants) { if (-not (Test-Path -LiteralPath $item.result)) { throw ('Result missing: '+$item.result) }; Write-Output $item.id }
 Write-Output '76-95s, same native PTS; no window opened.'; exit 0
}
& (Join-Path $PSScriptRoot 'START_3D_RESULTS.ps1') -Config $configPath -Variant $Variant -Fresh:$Fresh -Wait:$Wait
exit $LASTEXITCODE
"""
    text=text.replace('events/ankara_2026/programmes/okada_mei_sp/viewer3d_spin_grasp_1104.json',str((PROGRAMME/config_name).as_posix()))
    text=text.replace("$Variant='grasp_candidate'","$Variant='"+config['default_variant']+"'")
    command.write_text(text,encoding='utf-8-sig')
    cmd_lines=['@echo off','powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0'+launcher_name+'.ps1" %*','set "rinkspace_exit=%ERRORLEVEL%"','if not "%rinkspace_exit%"=="0" (','  echo.','  echo RinkSpace 3D failed. Review the error above.','  pause',')','exit /b %rinkspace_exit%']
    (project/(launcher_name+'.cmd')).write_bytes((chr(13)+chr(10)).join(cmd_lines).encode('ascii')+bytes([13,10]))
    return target


def plots(output,packets):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8));colors=['#bf4d45','#84785e','#5589b9','#15924b','#a84fc1','#dd851d']
    for (name,d),color in zip(packets.items(),colors):
        t=d['pts']/12800.;mask=(t>=84)&(t<=93.2);P=d['P_world_mid']
        axes[0,0].plot(P[mask,0],P[mask,1],label=name,color=color,alpha=.8)
        axes[0,1].plot(t,P[:,2],label=name,color=color,alpha=.8)
        axes[1,0].plot(t,d['skate_clearance_m_mid'][:,1],label=name,color=color,alpha=.8)
        if 'palm_to_segment_m_mid' in d:axes[1,1].plot(t,d['palm_to_segment_m_mid'],label=name,color=color)
    axes[0,0].set(xlabel='world X (m)',ylabel='world Y (m)',title='Pelvis projection, same world, no alignment')
    axes[0,1].set(xlabel='source seconds',ylabel='root height (m)');axes[1,0].set(xlabel='source seconds',ylabel='raw clearance diagnostic (m)');axes[1,1].set(xlabel='source seconds',ylabel='palm to finite grip segment (model m)')
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'grasp_world_comparison.png',dpi=140);plt.close(fig)


def movie(base,output,packets):
    import av
    from PIL import Image,ImageDraw,ImageFont
    from rinkspace.viewer3d.source import SourceVideo
    source=json.loads((base/'action_integration_1103/spin_s05/contract.json').read_text(encoding='utf8'))['source']['path']
    pts=next(iter(packets.values()))['pts'];pts=pts[(pts>=84*12800)&(pts<=94*12800)];indices={int(v):i for i,v in enumerate(next(iter(packets.values()))['pts'])}
    prepared=load(base/'action_integration_1103/spin_s05/prepared.npz');rect=np.stack([np.interp(pts,prepared['source_pts'],prepared['roi_native_xyxy'][:,j]) for j in range(4)],axis=1)
    target=output/('source_grasp_84_94_finished.mp4' if (output/'active_candidate.json').exists() else 'source_grasp_84_94.mp4')
    if target.exists():return
    v=SourceVideo(source,(1,12800));container=av.open(str(target),'w');stream=container.add_stream('libx264',rate=50);stream.width=1680;stream.height=600;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'};stream.codec_context.thread_count=4
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);bones=[(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    labels=['source','gvhmr_fixed','hand_nofoot_initial','grasp_candidate'];record=[]
    try:
        for i,pt in enumerate(pts):
            raw=Image.fromarray(v.read(int(pt)));box=rect[i];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;h=max(620.,(box[3]-box[1])*1.1);w=max(h*.7,(box[2]-box[0])*1.1);h=max(h,w/.7);crop=(int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2));canvas=Image.new('RGB',(1680,600));scale=np.array([420/(crop[2]-crop[0]),600/(crop[3]-crop[1])]);index=indices[int(pt)]
            for col,name in enumerate(labels):
                panel=raw.crop(crop).resize((420,600));draw=ImageDraw.Draw(panel)
                if name!='source':
                    d=packets[name];uv=(d['COCO_uv'][index]-crop[:2])*scale
                    for a,b in bones:
                        if np.isfinite(uv[[a,b]]).all():draw.line([tuple(uv[a]),tuple(uv[b])],fill=(0,235,120),width=2)
                    for f in (0,1):
                        points=(d['equipment_uv'][index,f,:14]-crop[:2])*scale
                        for q in points:
                            if np.isfinite(q).all():draw.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill=(255,155,45))
                draw.rectangle((0,0,420,44),fill=(0,0,0));draw.text((8,3),name+' source '+format(pt/12800.,'.2f')+'s',font=font,fill='white');canvas.paste(panel,(col*420,0))
            frame=av.VideoFrame.from_image(canvas);frame.pts=i
            for packet in stream.encode(frame):container.mux(packet)
            record.append(int(pt))
        for packet in stream.encode():container.mux(packet)
    finally:container.close();v.close()
    with av.open(str(target)) as check:decoded=[f.pts*f.time_base for f in check.decode(video=0)]
    assert len(decoded)==len(pts) and all(float(t)==i/50 for i,t in enumerate(decoded))
    write_json(output/('source_grasp_movie_clock_finished.json' if (output/'active_candidate.json').exists() else 'source_grasp_movie_clock.json'),dict(source=str(source),source_time_base=[1,12800],source_pts=record,encoded_frames=len(record),source_seconds=[84,94],all_encoded_pts_verified=True,viewport='same source ROI for all columns; not a world trajectory alignment',geometry='saved native five-node body and equipment projection; no new fitting'))


def deliver(project,base,output,label):
    case=output/'common'/label;packets,report=compare(base,output,label);plots(output,packets);config=register(project,base,output,label,packets)
    stem=archive_paths(project,output)['parameter_stem'];legacy=stem=='spin_grasp_1104'
    for name,target in [('grasp_relations.json','spin_grasp_relations_1104.json' if legacy else stem+'_relations.json'),('palm_grip_geometry.json','spin_grasp_geometry_1104.json' if legacy else stem+'_geometry.json'),('support_and_free_shoe_regions.json','spin_grasp_source_regions_1104.json' if legacy else stem+'_source_regions.json')]:
        path=project/PROGRAMME/target;assert len(str(path).encode('utf-16-le'))//2<260
        if path.exists():assert path.read_bytes()==(output/name).read_bytes()
        else:path.write_bytes((output/name).read_bytes())
    active=json.loads((output/'active_candidate.json').read_text(encoding='utf8')) if (output/'active_candidate.json').exists() else None
    checkpoint=Path(active['parameter']) if active else archive_paths(project,output)['parameter']
    previous=case/('replay_'+digest(checkpoint)[:12])/'replay.json'
    if previous.exists():
        replay_report=json.loads(previous.read_text(encoding='utf8'))
        assert replay_report['passed'] and replay_report['checkpoint_sha256']==digest(checkpoint)
        assert replay_report['original_result_sha256']==digest(Path(active['result']) if active else case/'fitted/result.npz')
    else:replay_report=replay(project,base,output,label,checkpoint)
    movie(base,output,packets)
    summary=json.loads((case/'run_summary.json').read_text(encoding='utf8'))
    if active:
        summary=dict(summary,accepted_steps=active['summary']['total_accepted'],retained_phase_step=active['summary']['total_retained'],parameter=active['parameter'],finish=active['summary'])
    lines=['S05 手—自由鞋闭链联合候选（未验收）',f"数值接受 {summary['accepted_steps']} 步；最终阶段保留步 {summary['retained_phase_step']}；真实阶段记录见 common/"+label+'/run_summary.json', '连续源76–95s；实际动作可动上下文82–94.8s；保持/释放按源帧，不固定世界旋转中心。','参数重放：'+str(replay_report['passed']),'播放器：'+str(config),'分相位人体/踝腕/原始参考高度与掌面有限段距离：phase_comparison.json','short movie source84–94s: source_grasp_84_94_finished.mp4；native full mesh76–95s: native_camera_overlay/before_after_mesh_76_95.mp4；供PotPlayer。','注意：所有高度、骨盆轨迹和掌面距离均为当前模型诊断，不是三维真值或统一毫米/厘米验收。']
    (output/'GRASP_RESULT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf8');write_json(output/'delivery.json',dict(config=str(config),run=summary,replay_passed=replay_report['passed'],phase_report=str(output/'phase_comparison.json'),no_visible_gui_opened=True,accuracy_accepted=False))
    print(json.dumps(dict(event='grasp_delivery_ready',config=str(config),replay_passed=replay_report['passed']),ensure_ascii=False),flush=True)


def audit_geometry(project,base,output,label):
    """Same-world source-region/runner and complete 951-exposure domain audit."""
    from types import SimpleNamespace
    from .skate_integration import build,restore_checkpoint_layers,SpinGraspWindow
    from .action_integration import ClipReference
    from .fullbody_state import camera_binding
    from .support_image import SupportImageFactors
    from .skate_physics import batch_features
    case=output/'common'/label;shared,_,_,_=build(project,base,output/'geometry_audit/shared')
    shared.binding.x=load(project/PROGRAMME/'skate_s01_1103.npz')['x'][:17].copy();shared.binding.enable_collar_constraints()
    p=load(case/'input.npz');camera,info,_,pose=camera_binding(project,SimpleNamespace(p=p),shot_id='s05')
    document=json.loads((output/'support_and_free_shoe_regions.json').read_text(encoding='utf8'));relations=json.loads((output/'grasp_relations.json').read_text(encoding='utf8'));active=json.loads((output/'active_candidate.json').read_text(encoding='utf8')) if (output/'active_candidate.json').exists() else None
    root=base/'exmr_spin_compare_1104/native_protocol';sources={
      'gvhmr_fixed':(root/'common/gvhmr',root/'common/gvhmr/initial/fitted/result.npz',None),
      'exmr_prefoot_old':(root/'common/exmr_prefoot',root/'common/exmr_prefoot/initial/fitted/result.npz',None),
      'hand_nofoot_initial':(case,case/'initial_current_objective/fitted/result.npz',None),
      'grasp_candidate':(case,Path(active['result']) if active else case/'fitted/result.npz',Path(active['parameter']) if active else archive_paths(project,output)['parameter'])}
    report={};phases={'entry_84_86':(84,86),'held_86_8_92_46':(86.8,92.46),'release_92_46_93_2':(92.46,93.2),'exit_93_2_94_8':(93.2,94.8)}
    initial_packet=load(sources['hand_nofoot_initial'][1]);C=p['camera_centre']*float(initial_packet['metres_per_scene_unit']);ray=initial_packet['P_world_mid']-C;ray/=np.linalg.norm(ray,axis=1)[:,None]
    for name,(folder,result,parameter) in sources.items():
        d=load(result);native=[];frames=[]
        for first in range(0,len(d['pts']),64):
            kw={} if 'hand_local_R_mid' not in d else {'hand_local_R':d['hand_local_R_mid'][first:first+64]}
            values=shared.body.forward_rotations(d['local_R_mid'][first:first+64],with_frames=True,**kw);native.append(values[1]);frames.append(values[3])
        native=np.concatenate(native);frames=np.concatenate(frames);kw={} if 'hand_local_R_mid' not in d else {'hand_local_R':d['hand_local_R_mid']}
        samples=shared.binding.surface.sample_vertices(native,frames,shared.binding.sample_ids,double_precision=True,**kw);collar=shared.binding.collar_metrics(native,frames,samples)
        unknown=[dict(pts=int(d['pts'][i]),seconds=float(d['pts'][i]/12800),foot=int(f),reason=str(collar['unknown_reason'][i,f]),sample_count=int(collar['sample_count'][i,f])) for i,f in zip(*np.nonzero(collar['unknown']))]
        valid=~collar['unknown'];valid_breach=collar['breach_solver'][valid]*shared.binding.ratio
        R=d['skate_R_world_mid'];O=d['skate_origin_world_mid']/shared.binding.ratio
        features=batch_features(shared.binding,R,O);ids=[i for i,s in enumerate(features['semantics']) if s=='runner'];h=features['height'][:,1,ids];best=np.asarray(ids)[np.argmin(h,axis=1)];idx=np.arange(len(h));q=features['local_point'][idx,1,best]
        runner=(O[:,1]+np.einsum('nij,nj->ni',R[:,1],q))*shared.binding.ratio
        inp=load(folder/'input.npz');initial=load(folder/'initial.npz');ref=ClipReference(shared,folder,inp,initial)
        if parameter:restore_checkpoint_layers(ref,load(parameter),include_primary=True)
        state=SpinGraspWindow(ref,(82.,94.8));body=SimpleNamespace(state=state,tau=float(camera.p['rig'][37]));stub=SimpleNamespace(reference=ref,binding=ref.binding,camera=camera,body=body,optimize_shared=False)
        rows=SupportImageFactors(stub,document,include_review=True).evaluate(np.zeros(state.size),False)['metrics']['observations']
        by={};delta=d['P_world_mid']-initial_packet['P_world_mid'];los=np.einsum('ni,ni->n',delta,ray)
        for phase,(a,b) in phases.items():
            mask=(d['pts']>=round(a*12800))&(d['pts']<=round(b*12800));group=dict(runner_xy_span_m=np.ptp(runner[mask,:2],axis=0).tolist(),runner_geometric_path_m=float(np.linalg.norm(np.diff(runner[mask,:2],axis=0),axis=1).sum()),runner_raw_height_m=stats(runner[mask,2]),root_change_from_hand_initial_los_m=stats(los[mask]),root_change_los_max_abs_m=float(abs(los[mask]).max()),regions={})
            for role in ('fit','review'):
                for foot in (0,1):
                    for kind in ('boot_marker_region','support_contact_region'):
                        selected=[r for r in rows if r['role']==role and r['foot']==foot and r['kind']==kind and round(a*12800)<=r['pts']<=round(b*12800)]
                        group['regions'][role+'_'+kind+'_foot'+str(foot)]=dict(outside_native_px=stats([r['outside_region_px'] for r in selected]),exposures=len({r['pts'] for r in selected}))
            by[phase]=group
        report[name]=dict(source_frames=len(d['pts']),collar_unknown_sections=len(unknown),unknown_rows=unknown,valid_collar_breach_max_m=float(valid_breach.max()) if len(valid_breach) else None,model_foot_cavity_breach_max_m=float(np.max(d['foot_cavity_max_violation_m_mid'])),phases=by,all_source_roles_unchanged=True,region_metrics='unweighted native distance outside source uncertainty region; not exact material point error',runner_identity='geometric minimum, not one tracked steel material point')
        print(json.dumps(dict(event='whole_spin_geometry_checked',variant=name,unknown_sections=len(unknown))),flush=True)
    write_json(output/'geometry_and_source_audit.json',dict(source_time_base=[1,12800],cases=report,not_accuracy_accepted=True,source_review_roles_preserved=True))
    # The old finish driver accidentally queried cached initial-factor samples
    # after attaching its final curve. Preserve that diagnostic and replace it
    # with source regions evaluated from the actual archived final reference.
    stale=case/'finished_factor_metrics.json'
    if stale.exists() and not (case/'finished_factor_metrics_initial_cache_discarded.json').exists():stale.replace(case/'finished_factor_metrics_initial_cache_discarded.json')
    write_json(case/'finished_factor_metrics.json',dict(source='geometry_and_source_audit.json, reconstructed from the exact archived final state',regions=report['grasp_candidate']['phases'],domain=report['grasp_candidate']['collar_unknown_sections'],discarded_cached_initial_measurement_not_used=True))
    return report
