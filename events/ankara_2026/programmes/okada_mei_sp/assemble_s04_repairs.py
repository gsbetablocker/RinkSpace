"""Assemble all S04 actual-source repairs; never write model or fitting state."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from review_points_s04 import OUT,ADS,FIRST,LAST,STEP,read,write,rows,CurrentNativeGrid
from ice_points_s04 import select_representatives,source_domain,REC
from rinkspace.vision_native import file_ref
from rinkspace.vision_board_reader import _unexcluded_pixels

DEST=OUT/'points_shape_repaired'


def emit(stream,value):
    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')


def actual_support(row):
    p=row['source_evidence'];samples=[row['uv']]
    for key in ('actual_source_sides_uv','actual_source_side_quantiles_uv'):
        if key in p:samples.extend(np.asarray(p[key]).reshape(-1,2).tolist())
    return np.asarray(samples)


def assemble():
    cv2.setNumThreads(2);DEST.mkdir(exist_ok=True)
    parent=OUT/'points_repaired';mapping=read(parent/'mapping.json')
    original=dict(np.load(parent/'point_observations.npz',allow_pickle=False))
    original_ref=file_ref(parent/'point_observations.npz')
    original_ids={str(k):i for i,k in enumerate(original['observation_ids'])}
    paths=list(mapping['path_ids']);lookup={p:i for i,p in enumerate(paths)}
    visibility={r['pts']:r for r in read(ADS/'evidence/visibility.json')['frames']}
    native=CurrentNativeGrid()
    from reread_s04_house import NativeHouseReader
    circle_navigation=NativeHouseReader(native)
    fields={k:[] for k in original};counts=collections.Counter();frames=[];sampling=[]
    preserved=0;rebindings=0;nextrow=int(original['source_rows'].max())+1
    provenance=[];by_nonice=collections.defaultdict(list)
    nonice_file=OUT/'evidence/nonice_repaired_records.jsonl.gz';nonice_ref=file_ref(nonice_file)
    for number,r in enumerate(rows(nonice_file)):by_nonice[r.get('pts',r.get('native_pts'))].append((number,r))
    sparse=DEST/'point_records.jsonl.gz';start=time.monotonic()
    with gzip.open(sparse,'wt',encoding='utf8',compresslevel=3)as point_records,gzip.open(DEST/'source_dispositions.jsonl.gz','wt',encoding='utf8')as ledger:
        def append(key,uv,pt,pid,kind,station,metadata,evidence,correlation_group):
            nonlocal nextrow,preserved,rebindings
            old=original_ids.get(key)
            if old is not None:
                if int(original['pts'][old])!=pt or not np.array_equal(original['uv'][old],uv):raise ValueError('Original UV/PTS moved: '+key)
                source_row=int(original['source_rows'][old]);preserved+=1
                oldpath=paths[int(original['path_indices'][old])]
                if oldpath!=pid:rebindings+=1
                elif kind!=int(original['kind'][old]):raise ValueError('Original observation type changed')
                if kind==0:station=float(original['stations'][old])
            else:source_row=nextrow;nextrow+=1
            if pid not in lookup:lookup[pid]=len(paths);paths.append(pid)
            values=dict(source_rows=source_row,observation_ids=key,uv=uv,pts=pt,kind=kind,stations=station,path_indices=lookup[pid])
            for k,v in values.items():fields[k].append(v)
            emit(point_records,dict(observation_id=key,source_row=source_row,uv=uv,pts=pt,native_path_id=pid,kind=kind,
                review_order=metadata,source_evidence=evidence,source_BGR_sha256=visibility[pt]['source_BGR_sha256'],
                correlation_group=correlation_group,human_acceptance=False,
                acceptance_scope='machine_source_repair_only; prior exact human scopes remain in immutable source records'))
            counts[metadata['family']]+=1

        for frame_index,pt in enumerate(range(FIRST,LAST+1,STEP)):
            v=visibility[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Wrong RGB frame')
            graphics,border,_,_=source_domain(im,mask,read(REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
            frame=dict(pts=pt,source_BGR_sha256=v['source_BGR_sha256'],raw_ice=0,selected_ice=0,ice_features={},nonice=0)
            for directory,kind in (('native_family_line_RGB','ice_segment'),('native_shape_RGB','ice_circle')):
                source=OUT/'evidence'/directory/f'{pt}.json';packet=read(source);ref=file_ref(source);provenance.append(ref)
                if packet['source_pts']!=pt or packet['source_BGR_sha256']!=v['source_BGR_sha256']:raise ValueError('Reader source differs')
                if packet['model_or_fitted_curve_pixels_emitted']:raise ValueError('Synthetic source pixels')
                for name,feature in packet['features'].items():
                    candidates=[];frame['raw_ice']+=len(feature['rows'])
                    for index,r in enumerate(feature['rows']):
                        reason=None
                        # Reviewed actual-source crop and both adjacent exposures
                        # show only isolated competing strokes, not an A2 arc.
                        # Preserve these six candidates with the exact scope.
                        if name=='ice.A2:centreline' and pt==879360:
                            reason='isolated_A2_candidate_has_no_distinct_arc_source_support; adjacent_exposures_do_not_resolve_identity'
                        elif not _unexcluded_pixels(excluded,actual_support(r)).all():
                            reason='measured_material_support_intersects_existing_person_graphics_or_board_domain'
                        if reason:
                            emit(ledger,dict(observation_id='ice:'+r['source_row_id'],pts=pt,uv=r['uv'],native_path_id=r['native_path_id'],
                                reason=reason,action='retain_source_candidate_not_current_observation',source_ref=ref,row_index=index,
                                human_acceptance=False));continue
                        p=r['source_evidence']
                        candidates.append(dict(r,pixel_support=dict(normal=p['profile_normal'],query_index=r.get('query_index',p.get('query_index',index))),
                            visible_span_id=f's04/{pt}/{name}',source_ref=dict(**ref,feature=name,row_index=index)))
                    if not candidates:continue
                    options={}
                    if kind=='ice_circle':
                        fid,role=name.removeprefix('ice.').split(':')
                        if role=='centreline':role='outer'
                        theta=np.asarray([r['theta']for r in candidates])
                        guide=circle_navigation.projection(circle_navigation.groups[fid],pt,role,theta,
                            feature['navigation']['shared_source_affine_parameters'])[0]
                        options=dict(source_curve_guide=guide,source_curve_order=theta)
                    chosen,reasons=select_representatives(candidates,dict(kind=kind),**options)
                    # Display gaps explicitly while still drawing every link.
                    # A gap does not split the correlated material family.
                    order=np.argsort([r.get('theta',r['pixel_support']['query_index'])for r in candidates])
                    positions=np.asarray([candidates[i]['uv']for i in order])
                    delta=np.linalg.norm(np.diff(positions,axis=0),axis=1)
                    positive=delta[delta>.25]
                    gap=max(12.,.04*float(np.linalg.norm(np.ptp(positions,axis=0))),3*float(np.median(positive)))if len(positive)else 12.
                    spans=np.zeros(len(candidates),int);spans[order]=np.r_[0,np.cumsum(delta>gap)]
                    frame['ice_features'][name]=len(chosen);frame['selected_ice']+=len(chosen)
                    sampling.append(dict(pts=pt,native_path_id=feature['native_path_id'],feature=name,raw=len(feature['rows']),eligible=len(candidates),selected=len(chosen)))
                    for i in chosen:
                        r=candidates[i];p=r['pixel_support'];key='ice:'+r['source_row_id']+'@'+ref['sha256'][:16]
                        family=name.removeprefix('ice.').split(':')[0]
                        append(key,r['uv'],pt,r['native_path_id'],1,.5,
                            dict(family='ice',run=f's04/{pt}/{name}/span{spans[i]}',order=float(r.get('theta',p['query_index']))),
                            dict(source_ref=r['source_ref'],source_row_id=r['source_row_id'],feature_id=name,
                                measurement=r['source_evidence'],density_selection=sorted(reasons.get(i,[])),original_UV_unchanged=True),
                            f's04/{pt}/'+('nested_house/'+family if family.startswith(('F','R')) and kind=='ice_circle' else 'material/'+family))
            for number,r in by_nonice[pt]:
                if r['family']=='board':
                    key='board:'+r['source_row_id'];pid=r['native_path_id']
                    append(key,r['uv'],pt,pid,1,.5,dict(family='board',run=r['visible_span_id'],order=float(r['uv'][0])),
                        dict(source_ref=dict(**nonice_ref,jsonl_line_0based=number),original_record=r),r['segment_id'])
                else:
                    ident=r['identity'];pid=r['path_id']
                    if 'atlas_id'in ident:
                        meta=dict(family='ad',atlas=int(ident['atlas_id']),source_path=int(ident['source_path_index']))
                        group=f's04/{pt}/source_atlas/{ident["atlas_id"]}/{pid}'
                    else:
                        meta=dict(family='ad_source_edge',run=pid+'@'+str(ident['source_guide_pts']),order=float(r['stable_id'].rsplit('_',1)[1]))
                        group=f's04/{pt}/source_guide/{ident["source_guide_pts"]}/{pid}'
                    append(r['stable_id'],r['uv'],pt,pid,int(r['kind']),float(r['station']),meta,
                        dict(source_ref=dict(**nonice_ref,jsonl_line_0based=number),original_record=r),group)
                frame['nonice']+=1
            frames.append(frame)
            if frame_index%100==0:print(json.dumps(dict(stage='assemble',frames=frame_index+1,total=793,counts=dict(counts),seconds=time.monotonic()-start)),flush=True)
    z={k:np.asarray(v,dtype=(np.float64 if k in ('uv','stations') else str if k=='observation_ids' else np.int64))for k,v in fields.items()}
    if len(set(z['observation_ids']))!=len(z['pts'])or len(set(z['source_rows']))!=len(z['pts']):raise ValueError('Duplicate original point keys')
    if not np.array_equal(np.unique(z['pts']),np.arange(FIRST,LAST+1,STEP)):raise ValueError('Source frame coverage differs')
    np.savez_compressed(DEST/'point_observations.npz',**z)
    mapping.update(path_ids=paths,parent_input=original_ref,point_input_sha256=file_ref(DEST/'point_observations.npz')['sha256'],
        stations_status='see separate current native station initialization; UV and fixed stations immutable',
        source_row_namespace=dict(parent=mapping['source_row_namespace'],new_rows_start=int(original['source_rows'].max())+1,new_rows_stop=nextrow),
        actual_record_file=file_ref(sparse),native_replay_authority=native.receipt,
        corrections=[file_ref(OUT/'evidence/nonice_repair_changes.jsonl.gz'),file_ref(DEST/'source_dispositions.jsonl.gz'),mapping['source_review_records']],
        original_source_data_modified=False,human_acceptance=False,
        previous_ice_readings='preserved in parent input and original source files; superseded by full-entity current RGB rereads, not blanket classified as wrong')
    write(DEST/'mapping.json',mapping)
    write(DEST/'source_coverage.json',dict(frames=frames,sampling=sampling,full_exposures=793,
        counts=dict(counts),unchanged_original_pixels=preserved,source_supported_identity_rebindings=rebindings,
        original_ice_points_preserved_in_parent=int(np.char.startswith(original['observation_ids'],'ice:').sum()),
        source_reader_files=provenance,source_video_sha256=mapping['source_video_sha256'],
        native_model_or_peer_fit_modified=False,missing_reads_not_filled_from_projection=True,human_acceptance=False))
    native.unchanged()
    if file_ref(parent/'point_observations.npz')!=original_ref:raise ValueError('Original input changed')
    print(json.dumps(dict(stage='assembled',directory=str(DEST),points=len(z['pts']),counts=dict(counts),frames=len(frames))),flush=True)


def initialize():
    """Read-only native projection for material station seeds and coverage."""
    mapping=read(DEST/'mapping.json');z=dict(np.load(DEST/'point_observations.npz',allow_pickle=False))
    paths=mapping['path_ids'];observed={paths[int(i)] for i in np.unique(z['path_indices'])}
    grid=CurrentNativeGrid(full_model=True,observed_path_ids=observed)
    seed=np.empty(len(z['pts']));distance=np.empty(len(seed));start=time.monotonic();coverage=[]
    lookup=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,(pt,p)in enumerate(zip(z['pts'],z['path_indices'])):lookup[int(pt)][paths[int(p)]].append(i)
    for frame_index,pt in enumerate(range(FIRST,LAST+1,STEP)):
        projected=grid.project(pt);frame=[]
        for pid,p in projected.items():
            visible=p['valid']&(p['uv']>=[0,0]).all(1)&(p['uv']<[1920,1080]).all(1)
            indices=np.asarray(lookup[pt].get(pid,[]),dtype=int)
            if visible.any()or len(indices):frame.append(dict(path_id=pid,visible_native_samples=int(visible.sum()),actual_points=len(indices)))
            if not len(indices):continue
            order=grid.order(z['uv'][indices],p)
            if order is None:raise ValueError(f'No native station navigation for {pt}/{pid}')
            cut=grid.cuts[pid]
            if cut['layer']=='advertisement':
                stations=np.asarray(grid.samples['print_sampling_stations'][cut['instance_id']][cut['template_path_id']])
            else:
                bounds=cut.get('sampling_interval',cut['support'])
                stations=np.linspace(*bounds,len(p['uv']))
            seed[indices]=np.interp(order,np.arange(len(stations)),stations)
            pp=np.column_stack([np.interp(order,np.arange(len(p['uv'])),p['uv'][:,axis])for axis in (0,1)])
            distance[indices]=np.linalg.norm(pp-z['uv'][indices],axis=1)
        coverage.append(dict(pts=pt,paths=frame,visibility_is_native_projection_only_not_proof_of_observable_material=True))
        if frame_index%100==0:print(json.dumps(dict(stage='native_initialization',frames=frame_index+1,total=793,seconds=time.monotonic()-start)),flush=True)
    seed[z['kind']==0]=z['stations'][z['kind']==0]
    if not np.isfinite(seed).all():raise ValueError('Nonfinite material seed')
    np.savez_compressed(DEST/'initialized_points.npz',stations=seed)
    np.save(DEST/'seed_screen_distance.npy',distance)
    write(DEST/'station_initialization.json',dict(point_input_sha256=mapping['point_input_sha256'],
        native_authority=grid.receipt,frames=793,points=len(seed),fixed_stations_preserved=True,
        camera_geometry_and_source_UV_updated=False,GPU_used=False,
        method='nearest exact-native sampled material segment, preparation seed only; no fitting',
        screen_distance_is_navigation_mismatch_not_source_accuracy=True,
        screen_distance_quantiles_px=np.quantile(distance,[0,.5,.9,.99,1]).tolist(),seconds=time.monotonic()-start))
    write(DEST/'native_coverage.json',dict(frames=coverage,native_authority=grid.receipt))
    grid.unchanged()
    from prepare_native_s01 import load_pixel_input
    load_pixel_input(DEST)
    print(json.dumps(dict(stage='native_initialization_complete',points=len(seed),current_native_paths=len(grid.cuts))),flush=True)


def verify():
    from prepare_native_s01 import load_pixel_input
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    z,m,seed=load_pixel_input(DEST);parent=OUT/'points_repaired'
    old=dict(np.load(parent/'point_observations.npz',allow_pickle=False));oldmap=read(parent/'mapping.json')
    grid=CurrentNativeGrid();native=NativeSceneGPU(grid.manifest['scene_path'],grid.core,parameter_state=grid.parameter_state)
    point_ref=file_ref(DEST/'point_observations.npz')
    if point_ref['sha256']!=m['point_input_sha256']:raise ValueError('Stale current mapping')
    if read(DEST/'station_initialization.json')['point_input_sha256']!=point_ref['sha256']:raise ValueError('Stale station seed')
    ids={str(k):i for i,k in enumerate(z['observation_ids'])};oldids={str(k):i for i,k in enumerate(old['observation_ids'])}
    used={m['path_ids'][int(i)]for i in np.unique(z['path_indices'])}
    if used-native.paths.keys():raise ValueError('Non-native material path')
    if not np.array_equal(np.unique(z['pts']),np.arange(FIRST,LAST+1,STEP)):raise ValueError('Missing source exposure')
    source_files={};seen=set();retained=0;rebound=[];counts=collections.Counter();fixed=0;physical_cameras={}
    rebind_authority={r['observation_id']:r for r in rows(OUT/'evidence/nonice_repair_changes.jsonl.gz')if r['action']=='source_supported_identity_rebinding'}
    for r in rows(DEST/'point_records.jsonl.gz'):
        i=ids[r['observation_id']]
        if i in seen:raise ValueError('Duplicate source record')
        seen.add(i);pid=m['path_ids'][int(z['path_indices'][i])]
        if not np.array_equal(z['uv'][i],r['uv'])or int(z['pts'][i])!=r['pts']or pid!=r['native_path_id']or int(z['source_rows'][i])!=r['source_row']:
            raise ValueError('Raw source record does not match numerical input')
        source=r['source_evidence'];ref=source['source_ref']
        source_files[ref['path']]=dict(path=ref['path'],sha256=ref['sha256'])
        counts[r['review_order']['family']]+=1
        if r['review_order']['family']=='ice':
            e=source['measurement']
            if not e.get('actual_source_support_outside_person_and_broadcast_masks'):raise ValueError('Ice source support lost')
            if native.paths[pid]['label'].endswith((':centreline',':segment0'))or'ice.lane'in native.paths[pid]['label']:
                midpoint=np.asarray(e['actual_source_sides_uv']).mean(0)
                if e.get('width_midpoint_operator'):
                    from reread_s04_C_source import CurrentNativeGroundCamera
                    from calibration.program_tracking.physical_ice_width import ground_midpoint_from_source_sides
                    if r['pts']not in physical_cameras:physical_cameras[r['pts']]=CurrentNativeGroundCamera(grid,r['pts'])
                    result=ground_midpoint_from_source_sides(physical_cameras[r['pts']],e['actual_source_sides_uv'])
                    if result is None:raise ValueError('Current native measured-flank midpoint is unavailable')
                    midpoint=result[0]
                if np.linalg.norm(np.asarray(r['uv'])-midpoint)>1e-3:
                    raise ValueError('Centreline is not a measured two-sided midpoint')
        oldi=oldids.get(r['observation_id'])
        if oldi is not None:
            if not np.array_equal(z['uv'][i],old['uv'][oldi])or z['pts'][i]!=old['pts'][oldi]or z['source_rows'][i]!=old['source_rows'][oldi]:raise ValueError('Retained source pixel identity changed')
            retained+=1;oldpid=oldmap['path_ids'][int(old['path_indices'][oldi])]
            if oldpid!=pid:
                authority=rebind_authority.get(r['observation_id'])
                if not authority or authority['replacement_native_path_id']!=pid:raise ValueError('Unaudited material rebind')
                rebound.append(r['observation_id'])
            if old['kind'][oldi]==0:
                if z['kind'][i]!=0 or z['stations'][i]!=old['stations'][oldi]:raise ValueError('Fixed source corner changed')
                fixed+=1
    if len(seen)!=len(z['pts']):raise ValueError('Source evidence omission')
    for ref in source_files.values():
        if file_ref(ref['path'])!=ref:raise ValueError('Actual source evidence file changed: '+ref['path'])
    withdrawn=list(rows(Path(m['source_review_records']['path'])))
    if any(r['observation_id']in ids for r in withdrawn):raise ValueError('Reviewed source rejection reentered')
    oldpaths=np.asarray(oldmap['path_ids'])[old['path_indices']]
    late=np.flatnonzero(np.char.startswith(oldpaths,'print_instance/Ad4::Ad4.ANKARA.native440')&(old['pts']>=933120)&(old['pts']<=943360))
    if len(late)!=3929 or any(str(old['observation_ids'][i])not in ids for i in late):raise ValueError('Correct late ANKARA source lost')
    if fixed!=int((old['kind']==0).sum()):raise ValueError('Fixed source corner omitted')
    nonice=read(OUT/'evidence/nonice_repaired_check_summary.json')
    if nonice['issues'] or len(nonice['frames'])!=793:raise ValueError('Unrepaired advertisement/board source findings')
    bounds=[]
    for k in np.unique(z['path_indices']):
        pid=m['path_ids'][int(k)];path=native.paths[pid];part=seed[z['path_indices']==k]
        if path['kind']==7:support=[0.,grid.core.board_perimeter(native.scene['board'])]
        elif path['kind']==8:support=[path['knots'][0],path['knots'][-1]]
        else:support=path['support']
        if (part<support[0]-1e-8).any()or(part>support[1]+1e-8).any():raise ValueError('Seed outside current finite material path')
        bounds.append(dict(path_id=pid,observations=len(part),station_range=[float(part.min()),float(part.max())]))
    report=dict(point_input_sha256=point_ref['sha256'],source_exposures=793,points=len(z['pts']),counts=dict(counts),
        current_native_paths_used=len(used),current_native_inventory=native.inventory,
        exact_source_records=len(seen),unchanged_retained_original_pixels=retained,documented_identity_rebindings=len(rebound),
        fixed_material_corners_preserved=fixed,true_late_ANKARA_points_preserved=len(late),exact_previous_withdrawals_absent=len(withdrawn),
        original_raw_UV_PTS_IDs_not_overwritten=True,source_evidence_files_checked=len(source_files),
        final_nonice_full_frame_audit=file_ref(OUT/'evidence/nonice_repaired_check_summary.json'),
        ice_no_projection_or_flow_observations=True,native_path_station_ranges=bounds,
        native_geometry_or_peer_fit_written=False,human_acceptance=False)
    grid.unchanged();write(DEST/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items()if k not in ('current_native_inventory','native_path_station_ranges')},ensure_ascii=False),flush=True)


def deliver():
    import zipfile
    m=read(DEST/'mapping.json');check=read(DEST/'verification.json');v=read(OUT/'review/connected_video.json')
    for r in (check,v,read(DEST/'station_initialization.json')):
        if r['point_input_sha256']!=m['point_input_sha256']:raise ValueError('Delivery mixes different source inputs')
    if v['decoded_frames']!=793 or not v['native_PTS_equal']:raise ValueError('Incomplete review video')
    coverage=read(DEST/'source_coverage.json');counts=coverage['counts']
    gaps=[f['pts']for f in coverage['frames']if not f['selected_ice']]
    result=dict(schema='rinkspace.s04_source_delivery.v1',source_exposures=793,point_input_sha256=m['point_input_sha256'],
        counts=dict(ice=counts.get('ice',0),advertisement=counts.get('ad',0)+counts.get('ad_source_edge',0),board=counts.get('board',0)),
        current_input=str(DEST),source_time_base='1/12800',source_pts=[FIRST,LAST],source_seconds=[FIRST/12800,LAST/12800],
        original_source_video=m['source_video'],source_video_sha256=m['source_video_sha256'],
        current_native_authority=v['native_grid'],review_video=v,verification=file_ref(DEST/'verification.json'),
        original_and_superseded_inputs_preserved=[str(OUT/'points'),str(OUT/'points_repaired')],
        remaining_scope=dict(no_qualified_ice_readouts_source_pts=gaps,
            no_qualified_ice_readouts_source_seconds=[pt/12800 for pt in gaps],
            no_qualified_ice_readouts_reason='Current source has diffuse ice haze without a resolved, coherent pair of named material flanks; board and advertisement observations remain available',
            no_readout_does_not_mean_no_physical_material=True,
            faint_C_source_repair=file_ref(OUT/'evidence/faint_C_RGB/summary.json'),
            A2_unresolved_source_pts=[879360],A2_unresolved_reason='Isolated candidate has no distinct arc support in this exposure or its immediate neighbours; six raw candidates retained',
            unresolved_candidates=str(DEST/'source_dispositions.jsonl.gz'),
            projected_native_visibility_denominator=str(DEST/'native_coverage.json'),
            native_projection_residual_is_not_a_reason_to_move_UV_or_geometry=True,
            actual_visual_acceptance='pending_user_final_review'),
        native_geometry_or_peer_fit_modified=False,machine_results_written_as_human_acceptance=False)
    for name in ('delivery.json','result.json','package.json'):
        path=OUT/name;prior=OUT/'evidence'/('before_full_shape_repair_'+name)
        if path.exists()and not prior.exists():prior.write_bytes(path.read_bytes())
    write(OUT/'delivery.json',result);write(OUT/'result.json',dict(points=check['points'],frames=793,**result['counts'],current_input=str(DEST),human_acceptance=False))
    manifest=DEST/'README.txt'
    manifest.write_text('第四镜头 S04 修正数据集\n原片 58.280–74.120 秒，793 个官方 50fps 独立曝光。\n'
        '入口：point_observations.npz + mapping.json + initialized_points.npz。复用 prepare_native_s01.load_pixel_input。\n'
        '七个数值字段，无新增权重、sigma 或旧几何。原 UV、整数 PTS、来源和逐项纠错见 point_records.jsonl.gz。\n'
        '完整带连线视频：'+v['path']+'\n白线是当前 native 投影；彩色是本帧实际点。虚线仍连接缺段两端，不能作为实测曲线。\n'
        '本包不覆盖 native、相机、运行中的拟合输入。人工验收仍待用户最终审阅。\n',encoding='utf8')
    archive=OUT/'S04_native_points_repaired_20261027.zip'
    files=list(DEST.glob('*'))+[OUT/'delivery.json',OUT/'result.json',OUT/'review/connected_video.json',OUT/'review/connected_frame_index.jsonl.gz']
    files += [OUT/'evidence'/name for name in ('nonice_repair_changes.jsonl.gz','nonice_repaired_check_summary.json',
        'correction_sources.json','source_inventory.json','before_diffuse_band_repair.jsonl.gz',
        'before_fixed_circle_navigation.jsonl.gz','C_user_source_annotation_929536.png',
        'diffuse_band_source_comparison.json','final_remaining_ice_frames.jpg')]
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5)as package:
        for path in files:
            if path.is_file():package.write(path,path.relative_to(OUT),compress_type=zipfile.ZIP_STORED if path.suffix in('.gz','.npz')else zipfile.ZIP_DEFLATED)
        for name in ('native_shape_RGB','native_family_line_RGB'):
            for pt in range(FIRST,LAST+1,STEP):
                path=OUT/'evidence'/name/f'{pt}.json';package.write(path,path.relative_to(OUT))
        for path in (OUT/'evidence/faint_C_RGB').glob('*'):
            if path.is_file():package.write(path,path.relative_to(OUT))
        path=OUT/'evidence/C_source_identity_929536.json';package.write(path,path.relative_to(OUT))
        for name in ('reread_s04_house.py','reread_s04_lines.py','reread_s04_C_source.py','repair_s04_nonice.py',
                'audit_s04_nonice.py','ice_points_s04.py','assemble_s04_repairs.py','review_points_s04.py'):
            path=Path(__file__).parent/name;package.write(path,'implementation/'+name)
        for name in ('native_ice_material_repair.py','native_ice_short_arc_repair.py','extract_first_shot_ice.py','physical_ice_width.py'):
            path=ROOT/'calibration/program_tracking'/name;package.write(path,'implementation/calibration/program_tracking/'+name)
    with zipfile.ZipFile(archive)as package:
        if package.testzip()is not None:raise ValueError('Corrupt delivery archive')
    write(OUT/'package.json',dict(**file_ref(archive),bytes=archive.stat().st_size,point_input_sha256=m['point_input_sha256'],video_external_path=v['path']))
    print(json.dumps(dict(archive=str(archive),video=v['path'],points=check['points'],source_exposures=793)),flush=True)


if __name__=='__main__':
    if '--initialize'in sys.argv:initialize()
    elif '--verify'in sys.argv:verify()
    elif '--deliver'in sys.argv:deliver()
    else:assemble()
