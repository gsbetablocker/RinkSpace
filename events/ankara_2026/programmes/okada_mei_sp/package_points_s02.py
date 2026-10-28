"""Assemble S02 source evidence, native input and a complete connected review."""
from pathlib import Path
import argparse,collections,gzip,hashlib,itertools,json,shutil,time,zipfile
import cv2
import numpy as np
import points_s02 as p
from points_s02 import OUT,REC,SCENE,CAMERA,FIRST,LAST,STEP,read,write,emit,file_ref
from ice_points_s04 import select_representatives,current_graphics


def dump(f,r):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def ad_select():
    import ad_points_s04 as shared
    r,t,m,o,s,ats,chosen=p.ad_configure()
    shared.promote_audit()
    s.run();s.publish(source_prefix='s02_native_source_v1_')


def manual_records(geo):
    """Preserve exact user scope; resolve only explicitly named native roles."""
    import ad_replacement_s01 as b
    aa=b.assets();inventory=read(OUT/'evidence/source_inventory.json');records=[];history=[]
    for item in inventory['manual_rows']:
        r=item['record'];name=r['feature_id'];pt=r['native_pts'];target=geo.lookup(name);kind=1;station=.5
        if name.startswith('Ad'):
            ad=int(name.split('.')[0][2:]);names=set(aa[ad]['aliases'].get(name,[]))
            if name.endswith(('left_top','left_bottom')):
                names=set(aa[ad]['aliases'].get(f'Ad{ad}.left_vertical',[]));kind=0;station=1. if name.endswith('left_top')else 0.
            if len(names)==1:target=geo.current_paths.get(f'print_instance/Ad{ad}::'+names.pop())
        ref=r['source_pixel_or_annotation_ref']
        if 'gzip_member_offset' in ref:
            with open(ref['file'],'rb')as archive:
                archive.seek(ref['gzip_member_offset']);member=archive.read(ref['gzip_member_bytes'])
            if hashlib.sha256(member).hexdigest()!=ref['gzip_member_sha256']:raise ValueError('Manual source member changed')
            original=json.loads(gzip.decompress(member).splitlines()[ref['member_jsonl_line_0based']])
        else:original=next(q for i,q in p.rows(ref['file'])if i==ref['jsonl_line_0based'])
        if original['uv']!=r['uv'] or original['point_id']!=r['source_point_id']:raise ValueError('Manual archive identity mismatch')
        packet=read(original['source_file']);feature=packet['features'][int(original['source_pointer'].split('/')[2])]
        ledger=dict(original,source_archive_ref=ref,original_feature=feature,native_path_id=target['id']if target else None,
            binding_status='explicit_current_named_material'if target else'current_measurement_operator_not_established',
            user_confirmed=bool(r.get('user_confirmed',False)),
            user_accepted_scope='this exact original UV and PTS only'if r.get('user_confirmed',False)else None,approval_propagated_to_other_rows=False)
        history.append(ledger)
        if target is None:continue
        records.append(dict(observation_id=r['source_point_id']+'@pts='+str(pt),pts=pt,uv=r['uv'],native_path_id=target['id'],
            source_id=r['source_point_id'],kind=kind,station=station,origin='preserved_human_source_click',human_acceptance=bool(r.get('user_confirmed',False)),
            human_acceptance_scope=dict(source_pts=pt,original_point_id=r['source_point_id'],uv=r['uv'])if r.get('user_confirmed',False)else None,source_record=ledger,
            source_group_id=feature.get('source_evidence_group_id',f'manual:{pt}:{name}'),review_order=dict(family='manual',run='manual:'+target['id'],order=float(r['uv'][0]))))
    write(OUT/'evidence/manual_scope.json',dict(original=74,included=len(records),pending=len(history)-len(records),records=history,
        original_UV_PTS_and_acceptance_unchanged=True,ambiguity_not_resolved_by_renaming=True))
    return records


def assemble():
    cv2.setNumThreads(2);p.verify_authority();grid=p.NativeGrid();auth=read(OUT/'evidence/native_authority.json')
    geo=p.NativeSourceGeometry(SCENE,auth['native_library']);folder=OUT/'points';folder.mkdir(exist_ok=True)
    write(OUT/'evidence/native_entities.json',dict(entities=[dict(x,native_path_id=x['id'])for x in geo.current_paths.values()]))
    allrows=[];coverage=[];manual=manual_records(geo);manual_by=collections.defaultdict(list)
    for r in manual:manual_by[r['pts']].append(r)
    vis={v['pts']:v for v in read(OUT/'evidence/visibility.json')['frames']}
    ice_admission=[]
    for pt in range(FIRST,LAST+1,STEP):
        f={};source_files={}
        for mode in ['lines','circles','gap_lines','red_arc_RGB']:
            file=OUT/f'evidence/{"reread_"+mode if mode!="red_arc_RGB"else mode}/{pt}.json';packet=read(file)
            if mode=='red_arc_RGB':
                for key in list(f):
                    if key.startswith(('ice.A1:','ice.A2:','ice.A3:','ice.A4:')):f.pop(key)
            f.update(packet['features']);source_files.update({k:file_ref(file) for k in packet['features']})
        counts=collections.Counter()
        for name,feature in f.items():
            rows=feature['rows'];pid=feature['native_path_id'];path=geo.current_paths[pid]
            if not rows:continue
            rr=[dict(r,pixel_support=dict(normal=r['source_evidence']['profile_normal'],query_index=r['source_evidence'].get('query_index',i)))for i,r in enumerate(rows)]
            options={}
            if path['kind']in ('ice_circle','ice_ellipse'):
                options=dict(source_curve_guide=np.asarray([r['source_evidence']['query_uv']for r in rr]),source_curve_order=np.asarray([r['theta']for r in rr]))
            keep,why=select_representatives(rr,path,**options)
            m=[r for r in manual_by[pt]if r['native_path_id']==pid]
            kept=0
            for j in keep:
                r=rr[j]
                if any(np.linalg.norm(np.asarray(r['uv'])-v['uv'])<25 for v in m):continue
                key=r['source_row_id'];order=float(r.get('theta',r['source_evidence'].get('query_index',j)))
                allrows.append(dict(observation_id=key,source_id=key,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                    origin='current_original_RGB_readout',human_acceptance=False,source_BGR_sha256=vis[pt]['source_BGR_sha256'],
                    source_record=r,source_ref=source_files[name],source_navigation=feature.get('navigation'),
                    source_group_id=f's02/{pt}/{pid}/same_material_profile_family',
                    selection_reasons=sorted(why[j]),review_order=dict(family='ice',run=f'{pid}@{source_files[name]["path"]}',order=order)))
                counts['ice']+=1;kept+=1
            ice_admission.append(dict(pts=pt,path=pid,actual_reads=len(rows),selected=kept,manual_source_rows=len(m)))
        # Exact old board readings already authenticate blue-top and yellow-bottom.
        bp=REC/f'dense_board_v1/qualified_frames/{pt}.json';board=read(bp);im=cv2.imread(vis[pt]['image']);mask=cv2.imread(vis[pt]['mask_path'],0)
        if board['decoded_native_BGR_sha256']!=vis[pt]['source_BGR_sha256']:raise ValueError('Board source changed')
        if not board['paired_blue_cap_and_white_board_body_semantics']or not board['yellow_contact_from_current_global_band_not_accumulated_extrapolation']:raise ValueError('Board semantics not established')
        graphics,_=current_graphics(im);by=collections.defaultdict(list)
        for j,r in enumerate(board['rows']):
            if not r.get('pixels_measured_in_this_frame')or not r.get('pixel_support',{}).get('current_pixel_profile_reread'):continue
            x,y=np.rint(r['uv']).astype(int)
            if not(0<=x<1920 and 0<=y<1080)or mask[y,x]or graphics[y,x]or r.get('occlusion'):continue
            pid=geo.lookup('board.top'if r['role']=='board.top'else'board.bottom')['id']
            by[pid].append(dict(r,native_path_id=pid,source_row_index=j,feature_id=r['role'],pixel_support=dict(r['pixel_support'],query_index=j,normal=[0.,1.])))
        for pid,rr in by.items():
            keep,why=select_representatives(rr,dict(kind='ice_segment'))
            for j in keep:
                r=rr[j];sid='board:'+r['source_row_id'];allrows.append(dict(observation_id=sid,source_id=r['source_row_id'],pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                    origin='original_qualified_board_profile',human_acceptance=False,source_record=r,source_ref=file_ref(bp),source_BGR_sha256=vis[pt]['source_BGR_sha256'],
                    source_group_id=f's02/{pt}/{pid}/{r["segment_id"]}',selection_reasons=sorted(why[j]),
                    review_order=dict(family='board',run=r['segment_id'],order=float(r['uv'][0]))));counts['board']+=1
        border_file=OUT/f'evidence/reread_ad_borders/{pt}.json';extra=read(border_file)
        for pid,feature in extra['features'].items():
            for r in feature['rows']:
                if any(v['native_path_id']==pid and np.linalg.norm(np.asarray(r['uv'])-v['uv'])<25 for v in manual_by[pt]):continue
                sid=r['source_row_id'];allrows.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                    origin='current_RGB_panel_gap_readout',human_acceptance=False,source_BGR_sha256=vis[pt]['source_BGR_sha256'],source_record=r,
                    source_ref=file_ref(border_file),source_navigation=feature['navigation'],source_group_id=f's02/{pt}/{pid}/same_panel_boundary',
                    review_order=dict(family='ad_source_edge',run=pid,order=float(r['uv'][0]))));counts['ad_boundary_gap']+=1
        projected=grid.project(pt)
        for r in manual_by[pt]:
            if r['native_path_id']in projected and geo.current_paths[r['native_path_id']]['kind']in ('ice_circle','ice_ellipse'):
                r['review_order']['order']=float(grid.order(np.asarray(r['uv'])[None],projected[r['native_path_id']])[0]/256*2*np.pi)
            allrows.append(r);counts['manual']+=1
        coverage.append(dict(pts=pt,source_seconds=pt/12800,counts=dict(counts),ice_paths={k:len(v['rows'])for k,v in f.items()}))
        if len(coverage)%100==0:emit('assemble_sources',frames=len(coverage),rows=len(allrows))
    # Source observations retain instance identity independently from template name.
    work=OUT/'ads/source_review';chart=read(work/'source_identity_charts.json');ats=read(OUT/'ads/atlases.json')['atlases'];ad_rejected=[]
    model_grid=p.NativeGrid(full_model=True);projected={};advisible={};adcounts=collections.Counter()
    for _,c in p.rows(work/'pointwise_checks.jsonl.gz'):
        if c['decision']=='not_admitted':continue
        pt=c['pts'];pid=c['path_id'];ad=pid.split('::')[0];aid=c['identity']['atlas_id'];original_atlas=ats[aid]
        same_shot=any(v.get('proof',{}).get('physical_instance')==ad.split('/')[-1]for v in original_atlas['paths'])
        key=(pt,ad)
        if key not in advisible:
            if pt not in projected:projected={pt:model_grid.project(pt)}
            cloud=[x['uv'][x['valid']]for k,x in projected[pt].items()if k.startswith(ad+'::')and x['valid'].any()]
            q=np.vstack(cloud)if cloud else np.empty((0,2))
            # Only an unequivocal whole-instance contradiction is rejected.
            # A full image of navigation slack preserves edge-of-view cases.
            advisible[key]=bool(len(q)and(q.max(0)>[-1920,-1080]).all()and(q.min(0)<[3840,2160]).all())
        if not same_shot and not advisible[key]:
            ad_rejected.append(dict(c,reason='shared_artwork_match_conflicts_with_whole_physical_instance_and_S02_sequence',source_point_retained_in_dense_history=True));continue
        if any(r['native_path_id']==pid and np.linalg.norm(np.asarray(r['uv'])-c['uv'])<25 for r in manual_by[pt]):continue
        sid=str(c['stable_id'])+'@pts='+str(pt)
        allrows.append(dict(observation_id=sid,source_id=c['stable_id'],pts=pt,uv=c['uv'],native_path_id=pid,kind=c['kind'],station=c['station'],
            origin=c['decision'],human_acceptance=False,source_record=c,source_BGR_sha256=vis[pt]['source_BGR_sha256'],
            source_group_id=f's02/{pt}/{pid}/source_pattern_{aid}',review_order=dict(family='ad',atlas=aid,source_path=c['identity']['source_path_index'])));adcounts[pt]+=1
    with gzip.open(OUT/'evidence/ad_instance_conflicts.jsonl.gz','wt',encoding='utf8')as f:
        for r in ad_rejected:dump(f,r)
    allrows.sort(key=lambda r:(r['pts'],r['native_path_id'],r['observation_id']));ids=[r['observation_id']for r in allrows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate source point identity in S02 input')
    pathids=sorted({r['native_path_id']for r in allrows});lookup={k:i for i,k in enumerate(pathids)}
    if set(pathids)-geo.current_paths.keys():raise ValueError('Non-current path in S02 input')
    z=dict(source_rows=np.arange(len(allrows),dtype=np.int64),observation_ids=np.asarray(ids),uv=np.asarray([r['uv']for r in allrows]),pts=np.asarray([r['pts']for r in allrows],np.int64),
        kind=np.asarray([r['kind']for r in allrows],np.uint8),stations=np.asarray([r['station']for r in allrows]),path_indices=np.asarray([lookup[r['native_path_id']]for r in allrows],np.int32))
    np.savez_compressed(folder/'point_observations.npz',**z)
    with gzip.open(folder/'point_records.jsonl.gz','wt',encoding='utf8')as f:
        for i,r in enumerate(allrows):r['source_row']=i;dump(f,r)
    authority=read(OUT/'evidence/native_authority.json')
    write(folder/'mapping.json',dict(schema='rinkspace.native_source_points.v1',shot_id='s02',path_ids=pathids,
        scene_path=str(SCENE),scene_revision=read(OUT/'ads/source_review/native_entity_registry.json')['scene_revision'],scene_file_sha256=file_ref(SCENE)['sha256'],
        camera_sha256=file_ref(CAMERA)['sha256'],control_sha256=file_ref(CAMERA.parent/read(CAMERA)['controls_ref'])['sha256'],
        point_input_sha256=file_ref(folder/'point_observations.npz')['sha256'],source_video=authority['source']['registered_path'],
        source_video_sha256=authority['source']['sha256'],source_scope=dict(first_pts=FIRST,last_pts=LAST,time_base='1/12800',frames=567),
        pixel_domain='native_distorted',image_size=[1920,1080],native_numeric_state=authority['scene_state'],
        original_sources_rewritten=False,projected_or_flow_UV_emitted=False,human_scope_file='../evidence/manual_scope.json'))
    for r in coverage:r['counts']['ad']=adcounts[r['pts']]+r['counts'].pop('ad_boundary_gap',0)
    write(OUT/'evidence/coverage.json',dict(frames=coverage,ice_entities=ice_admission,ad_instance_conflicts=len(ad_rejected)))
    counts=collections.Counter(r['origin']for r in allrows);write(OUT/'result.json',dict(frames=567,rows=len(allrows),origins=dict(counts),native_paths=len(pathids),human_scope_exact=True,complete_visual_acceptance=False))
    p.verify_authority();emit('whole_S02_assembled',rows=len(allrows),origins=dict(counts),paths=len(pathids),instance_conflicts=len(ad_rejected))


def initialize():
    from rinkspace.session.gpu_geometry import NativeSceneGPU,initialize_curve_stations
    from prepare_native_s01 import load_pixel_input
    p.verify_authority();g=p.NativeGrid();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));mapping=read(folder/'mapping.json')
    native=NativeSceneGPU(SCENE,g.core,parameter_state=g.parameter_state);paths=[mapping['path_ids'][i]for i in z['path_indices']];before=file_ref(folder/'point_observations.npz')
    camera=read(CAMERA);stations,report=initialize_curve_stations(native,camera=camera,controls=g.state['controls'],path_ids=paths,stations=z['stations'],pts=z['pts'],observed=z['uv'],kinds=z['kind'],backend='cpu',progress=lambda r:emit('native_station_seed',**r))
    np.savez_compressed(folder/'initialized_points.npz',stations=stations);load_pixel_input(folder)
    with native.resident(camera=camera,controls=g.state['controls'],path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'])as resident:
        result=resident.evaluate('cpu')
    np.savez_compressed(folder/'native_evaluation.npz',status=result['status'],projected_UV=result['values'][:,3:5],residual=result['values'][:,9:11])
    report.update(rows=len(z['pts']),frames=len(np.unique(z['pts'])),invalid=int(np.count_nonzero(result['status'])),statuses=dict(zip(*[a.tolist()for a in np.unique(result['status'],return_counts=True)])),
        CPU_timings=result['timings'],GPU_used=False,camera_or_scene_parameters_updated=False,source_input_unchanged=file_ref(folder/'point_observations.npz')==before)
    write(OUT/'evidence/native_integration.json',report);p.verify_authority();emit('native_input_checked',**{k:v for k,v in report.items()if k!='unresolved'})


def verify():
    from prepare_native_s01 import load_pixel_input
    p.verify_authority();folder=OUT/'points';z,m,stations=load_pixel_input(folder);seen=set();human=0;groups=collections.defaultdict(list)
    if not np.array_equal(np.unique(z['pts']),np.arange(FIRST,LAST+1,STEP)):raise ValueError('Missing official exposure')
    inv=read(OUT/'evidence/source_inventory.json');withdrawn=set(inv['withdrawals'])|set(inv['historical_corrections']);needed={};by_path=collections.defaultdict(list)
    source_refs=inv['source_files']+read(OUT/'ads/source_inventory.json')['sources']
    for ref in source_refs:
        if file_ref(ref['path'])['sha256']!=ref['sha256']:raise ValueError('Historical source changed: '+ref['path'])
    evaluation=np.load(folder/'native_evaluation.npz')
    if len(evaluation['status'])!=len(z['pts'])or np.any(evaluation['status']):raise ValueError('Final native evaluation is missing or invalid')
    for _,r in p.rows(folder/'point_records.jsonl.gz'):
        i=r['source_row'];seen.add(i);human+=bool(r['human_acceptance']);groups[r['source_group_id']].append(r['observation_id'])
        if r['observation_id']!=str(z['observation_ids'][i])or r['pts']!=int(z['pts'][i])or not np.array_equal(r['uv'],z['uv'][i])or r['native_path_id']!=m['path_ids'][int(z['path_indices'][i])]:raise ValueError('Point record mismatch')
        if r['source_id']in withdrawn:raise ValueError('Withdrawn original point restored')
        source=r['source_record']
        if source['uv']!=r['uv']:raise ValueError('Source coordinate was moved')
        if r['origin']=='preserved_human_source_click'and r['human_acceptance']!=bool(source.get('user_confirmed',False)):raise ValueError('Human scope changed')
        if r['origin']=='include_original_pixel':needed[int(source['source_row'])]=r
        by_path[r['native_path_id']].append(r['observation_id'])
    if len(seen)!=len(z['pts']):raise ValueError('Evidence missing')
    with gzip.open(folder/'source_correlated_groups.jsonl.gz','wt',encoding='utf8')as f:
        for key,ids in groups.items():dump(f,dict(group_id=key,observation_ids=ids,independent_point_weight_claimed=False))
    with gzip.open(folder/'temporal_material_groups.jsonl.gz','wt',encoding='utf8')as f:
        for key,ids in by_path.items():dump(f,dict(native_path_id=key,observation_ids=ids,shared_physical_material_across_exposures=True,
            frame_source_groups_preserved=True,independence_not_claimed=True))
    found=set()
    with gzip.open(folder/'original_ad_provenance.jsonl.gz','wt',encoding='utf8')as dest:
        for _,source in p.rows(OUT/'ads/candidate_provenance.jsonl.gz'):
            row=needed.get(source['source_row'])
            if row is None:continue
            if source['stable_id']!=row['source_id']or source['pts']!=row['pts']or source['uv']!=row['uv']:raise ValueError('Original ad source differs from delivered row')
            found.add(source['source_row']);dump(dest,dict(source,observation_id=row['observation_id'],native_path_id=row['native_path_id']))
    if found!=needed.keys():raise ValueError('Missing original ad provenance')
    report=dict(frames=567,rows=len(z['pts']),first_pts=FIRST,last_pts=LAST,integer_source_pts=True,raw_UV_unmodified=True,
        source_records_cover_every_point=True,preserved_exact_human_points=human,correlated_source_groups=len(groups),
        current_native_authority_unchanged=True,raw_source_history_unchanged=True,projected_or_flow_points=0,automatic_rows_human_accepted=False,
        checked_original_ad_provenance=len(found),native_invalid_observations=0,
        input_sha256=file_ref(folder/'point_observations.npz')['sha256'])
    write(OUT/'evidence/source_consistency.json',report);emit('S02_source_consistency',**report)


def deliver():
    p.verify_authority();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));mapping=read(folder/'mapping.json')
    consistency=read(OUT/'evidence/source_consistency.json');video=read(OUT/'review/connected_video.json');integration=read(OUT/'evidence/native_integration.json')
    sha=file_ref(folder/'point_observations.npz')['sha256']
    if consistency['input_sha256']!=sha or video['point_input_sha256']!=sha or video['points_displayed']!=len(z['pts']):raise ValueError('Delivery does not describe the final input')
    entities={r['native_path_id']:r for r in read(OUT/'evidence/native_entities.json')['entities']}
    layers=np.asarray([entities[pid]['layer']for pid in mapping['path_ids']]);layer=layers[z['path_indices']]
    counts={name:int(np.sum(layer==name))for name in np.unique(layer)};frames=[];gaps=[]
    for pt in range(FIRST,LAST+1,STEP):
        take=z['pts']==pt;by_layer={name:int(np.sum(take&(layer==name)))for name in counts}
        frames.append(dict(pts=pt,source_seconds=pt/12800,counts=by_layer,native_paths=len(np.unique(z['path_indices'][take]))))
        if by_layer.get('ice',0)==0:
            diagnostics={}
            for mode in ['reread_lines','reread_circles','red_arc_RGB','reread_gap_lines']:
                doc=read(OUT/f'evidence/{mode}/{pt}.json');diagnostics[mode]=doc.get('diagnostics',doc.get('attempts',{}))
            gaps.append(dict(pts=pt,source_seconds=pt/12800,affected='ice source constraints',
                reason='No current named two-sided paint/shape readout passed; this does not prove absence of visible marks',
                source_image=str(OUT/f'source_frames/{pt}.png'),diagnostics=diagnostics))
    manual=read(OUT/'evidence/manual_scope.json');pending=[r for r in manual['records']if r['native_path_id']is None]
    reg=read(OUT/'ads/source_review/native_entity_registry.json');observed=set(mapping['path_ids'])
    unmeasured=[r['native_id']for r in reg['paths']if r['native_id']not in observed]
    write(OUT/'evidence/unresolved.json',dict(ice_empty_frames=gaps,manual_pending_current_operator=pending,
        current_registered_ad_paths_without_measurements=unmeasured,
        unmeasured_path_does_not_alone_prove_visibility=True,
        ad26_retained_small_text='Original current-asset reference recovered; blurred/cropped shape did not yield reliable S02 registration. Outer circle has actual readings.',
        navigation_camera='Existing S02 camera seed; its projection mismatch is not a measured pixel error or an updated fit',
        machine_results_user_accepted=False))
    write(OUT/'evidence/delivery_coverage.json',dict(expected_frames=567,frames=frames,point_counts=counts,ice_empty_frames=len(gaps),
        advertisement_empty_frames=sum(r['counts'].get('advertisement',0)==0 for r in frames),
        board_empty_frames=sum(r['counts'].get('board',0)==0 for r in frames),
        exposure_coverage_is_not_complete_visible_material_recall=True))
    summary=read(OUT/'result.json');summary.update(point_counts=counts,ice_empty_frames=len(gaps),preserved_human_source_rows=manual['included'],
        user_confirmed_rows=consistency['preserved_exact_human_points'],manual_original_rows=manual['original'],manual_pending=len(pending),
        native_invalid_rows=integration['invalid'],video=str(OUT/'review/S02_points_connected_entities.mp4'),
        source_seconds=[FIRST/12800,LAST/12800],source_pts=[FIRST,LAST],point_input_sha256=sha,
        native_geometry_changed=False,source_datasets_overwritten=False,accuracy_acceptance='not_user_accepted',
        protected_authority_files=len(read(OUT/'evidence/native_authority.json')['files']))
    write(OUT/'result.json',summary)
    note=f'''# S02 当前 native 数据集

完整原片范围：44.980–56.300 秒；PTS {FIRST}–{LAST}，步长 256，time base 1/12800，共 567 个原生曝光。

当前输入 {len(z['pts']):,} 点，{len(mapping['path_ids'])} 条当前 native 路径：冰面 {counts.get('ice',0):,}、广告 {counts.get('advertisement',0):,}、围挡 {counts.get('board',0):,}。
输入入口 `points/`，供现有 `prepare_native_s01.load_pixel_input()` 读取。七字段 NPZ、逐点来源、相关组和初始化 station 一起交付；原 UV、整数 PTS、源 ID 未移动。全部点已通过当前 DLL 的 CPU 求值，失效行 {integration['invalid']}，没有拟合相机、场景或广告参数。

视频：`review/S02_points_connected_entities.mp4`，用 PotPlayer 查看；视频时间 0 对应原片 44.980 秒。全部 567 帧逐帧显示当帧实际点、同一实体的完整连线、实体名称、当前 native 全网格与同模型俯视图。遮挡／缺段连接用虚线明确显示，连线不是新增观测。

来源使用 recover_v2 的完整 S02 冰面池 579,959 行和完整广告池 3,304,086 行、分支纠错、撤销清单，以及原始 shot2_r2 人工／实测记录；没有把 500 点展示选择当作全集。推算点不作实测。广告按当前实际 Ad1–Ad8、Ad22–Ad26 实例绑定，模板同名不自动代表同一块广告。稠密支持保留在原归档，当前输入按端点、转折及空间分散支持选原始实测代表点。

历史人工来源共 74 点；当前输入保留 {manual['included']} 点，其中明确用户确认 {consistency['preserved_exact_human_points']} 点，另外 12 个历史人工广告点原记录未确认，仍未标通过。其余 {len(pending)} 个 C0 点的原 UV、PTS、人工确认及原特征完整保留在 `evidence/manual_scope.json`。旧记录只说明选中了 C0 外圈，没有两侧材料中线依据，因此没有将它们改名后强行当作当前中线。不是根据 `outer` 字样删除；当前 RGB 中线已独立重读。

仍有 {len(gaps)} 帧冰面为空，具体源时刻、失败诊断在 `evidence/unresolved.json`；其余帧也可能存在局部缺段。广告与围挡每个曝光均有实际读数。Ad26 的当前保留小字细节未得到可靠 S02 配准，外圆已有实测。相机沿用导航初值，白色网格偏差保留，不能据此宣称点已准确或已经拟合。全部机器结果等待最终视觉验收。

模型依据为 `evidence/native_authority.json` 登记的当前 scene、13 个广告资产、DLL、S02 导航和冻结 scene_values。只读使用当前 native；不覆盖生产输入，不改变几何，不干扰已有拟合。`points/native_evaluation.npz` 是接口／物理域检查，不是准确性验收。

`evidence/source_inventory.json` 保留旧纠错、撤销与原来源；`points/point_records.jsonl.gz` 逐行对应 NPZ，`points/original_ad_provenance.jsonl.gz` 回溯原始广告成员，`source_correlated_groups.jsonl.gz` 和 `temporal_material_groups.jsonl.gz` 保留同材料关联，不宣称重复点独立。原视频、大型完整原始池和逐帧原 RGB 留在原路径，本包不重复打包。
'''
    (OUT/'HANDOFF.md').write_text(note,encoding='utf8')
    selected=[OUT/'HANDOFF.md',OUT/'result.json',*folder.glob('*')]
    selected+=[OUT/'evidence'/name for name in ['native_authority.json','current_native_geometry_state.npz','native_entities.json','native_integration.json','source_consistency.json','source_inventory.json','manual_scope.json','coverage.json','delivery_coverage.json','unresolved.json','visibility.json','ad_instance_conflicts.jsonl.gz','historical_ice_dispositions.jsonl.gz','panel_gap_prior_review.jsonl.gz','mask_center_check.json']]
    selected+=[OUT/'review'/name for name in ['S02_points_connected_entities.mp4','connected_frame_index.jsonl.gz','connected_video.json']]
    selected+=[OUT/'ads/source_inventory.json',OUT/'ads/source_review/sampling_report.json']
    manifest=dict(schema='rinkspace.S02_native_source_delivery.v1',input_sha256=sha,files=[dict(relative_path=str(f.relative_to(OUT)).replace('\\','/'),**file_ref(f))for f in selected if f.is_file()])
    write(OUT/'delivery_manifest.json',manifest);selected.append(OUT/'delivery_manifest.json')
    target=OUT/'S02_native_points_20261027.zip';tmp=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for f in selected:
            if f.is_file():archive.write(f,str(f.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if f.suffix in ['.gz','.npz','.mp4']else zipfile.ZIP_DEFLATED)
        archive.write(CAMERA,'navigation/s02.json');archive.write(CAMERA.parent/read(CAMERA)['controls_ref'],'navigation/s02_controls.npy')
    with zipfile.ZipFile(tmp)as archive:
        if archive.testzip()is not None:raise ValueError('Corrupt delivery archive')
    tmp.replace(target);p.verify_authority()
    published=p.PROG/'source_datasets/native_points_s02_20261027/result.json'
    if len(str(published.resolve()).encode('utf-16-le'))//2>=260 or len(published.name.encode('utf-16-le'))//2>255:raise ValueError('OneDrive result path too long')
    published.parent.mkdir(exist_ok=True)
    write(published,dict(summary,input_directory=str(folder),handoff=str(OUT/'HANDOFF.md'),package=file_ref(target),
        source_consistency=file_ref(OUT/'evidence/source_consistency.json'),remaining_issues=file_ref(OUT/'evidence/unresolved.json')))
    emit('S02_delivered',**summary,package=file_ref(target))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['ad_select','assemble','initialize','verify','deliver']);args=parser.parse_args();globals()[args.stage]()
