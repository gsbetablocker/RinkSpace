"""Package and review the complete S02 source repair without changing the model."""
from pathlib import Path
from fractions import Fraction
import argparse,collections,gzip,json,hashlib,time,zipfile
import cv2,numpy as np,av
from PIL import Image,ImageDraw,ImageFont
import repair_points_s02 as repair
from ice_points_s04 import select_representatives
from rinkspace.review.point_connections import entity_color,polyline_order,source_connections
from rinkspace.review.render import source_frames,validate_video
old=repair.old;OUT=repair.OUT;read=repair.read;write=repair.fit.write

def dump(f,r):f.write(json.dumps(repair.fit.safe(r),ensure_ascii=False,separators=(',',':'))+'\n')

def assemble():
    cv2.setNumThreads(2);grid=repair.GuideGrid();photo=repair.SourceGuides(grid)
    entities={p['native_path_id']:p for p in read(old.OUT/'evidence/native_entities.json')['entities']}
    original=list(repair.rows(old.OUT/'points/point_records.jsonl.gz'));by=collections.defaultdict(list)
    for r in original:by[r['pts']].append(r)
    allrows=[];coverage=[];support_dir=OUT/'support';support_dir.mkdir(exist_ok=True);folder=OUT/'points';folder.mkdir(exist_ok=True)
    with gzip.open(OUT/'dispositions.jsonl.gz','wt',encoding='utf8',compresslevel=3)as ledger:
        for pt in range(old.FIRST,old.LAST+1,256):
            packet=read(OUT/f'ice/{pt}.json');nonice=read(OUT/f'nonice/{pt}.json');frame=[];supports={}
            packet['features'],ownership=repair.material_ownership(packet['features'],entities)
            for failure in ownership:dump(ledger,dict(failure,pts=pt,original_source_preserved=True,applies_to_all_reader_and_gap_outputs=True))
            ice_ref=repair.fit.file_ref(OUT/f'ice/{pt}.json');ad_ref=repair.fit.file_ref(OUT/f'nonice/{pt}.json')
            manual=[r for r in by[pt]if r['origin']=='preserved_human_source_click']
            manual_ids={r['observation_id']for r in manual};frame.extend(manual)
            frame.extend(r for r in nonice['retained_or_remeasured']if r['observation_id']not in manual_ids)
            for r in nonice['dispositions']:dump(ledger,r)
            for name,f in packet['features'].items():
                pid=f['native_path_id'];raw=f['rows'];path=entities[pid]
                if not raw:continue
                rr=[dict(r,pixel_support=dict(normal=r['source_evidence']['profile_normal'],query_index=r['source_evidence'].get('query_index',i)))for i,r in enumerate(raw)]
                opts={}
                if path['kind']in ('ice_circle','ice_ellipse') and path.get('role')!='physical_half_ellipse_boundary' and not f.get('navigation',{}).get('finite_actual_source_support_only'):
                    opts=dict(source_curve_guide=np.asarray([r['source_evidence']['query_uv']for r in rr]),source_curve_order=np.asarray([r['theta']for r in rr]))
                keep,why=select_representatives(rr,path,**opts)
                human=[r for r in manual if r['native_path_id']==pid]
                for i in keep:
                    r=rr[i]
                    if any(np.linalg.norm(np.asarray(r['uv'])-h['uv'])<25 for h in human):continue
                    sid=r['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='repaired_current_RGB_ice_material',human_acceptance=False,source_record=r,
                        source_ref=ice_ref,source_BGR_sha256=packet['source_BGR_sha256'],
                        source_group_id=f's02/{pt}/{pid}/same_material_profile_family',selection_reasons=sorted(why[i]),
                        review_order=dict(family='ice',run=pid,order=float(r.get('theta',r['source_evidence'].get('query_index',i))))))
                supports[pid]=dict(rows=raw,kind=path['kind'],closed=bool(grid.cuts[pid]['closed'])and not f.get('navigation',{}).get('finite_actual_source_support_only'),source=str(OUT/f'ice/{pt}.json'))
            # Do not erase an existing RGB line solely because the new search
            # missed it. Keep its original pixels if the whole actual stroke
            # still belongs to the current native direction family.
            oldice=collections.defaultdict(list)
            for r in by[pt]:
                if r['native_path_id'].startswith('demo_initial/')and r['origin']!='preserved_human_source_click':oldice[r['native_path_id']].append(r)
            present={r['native_path_id']for r in frame};retained=set()
            for pid,rr in oldice.items():
                path=entities[pid];name=path['label']
                if any(f['target_id']==name for f in ownership):continue
                if pid in present or path['kind']!='ice_segment' or len(rr)<2 or name not in photo.axes:continue
                direction=photo.direction_family(name,pt,np.asarray([r['uv']for r in rr]))
                if direction!=photo.axes[name]:continue
                source=photo.at(name,pt)
                if source is None:continue
                frame.extend(rr);retained.update(r['observation_id']for r in rr)
                supports[pid]=dict(rows=[dict(r['source_record'],uv=r['uv'])for r in rr],kind=path['kind'],closed=False,source='preserved_original_actual_material')
            for pid,rr in oldice.items():
                for r in rr:
                    if r['observation_id']in retained:continue
                    dump(ledger,dict(observation_id=r['observation_id'],original_uv=r['uv'],pts=pt,path_id=pid,
                        reason='current_RGB_material_replacement'if pid in supports else'current_material_not_recovered',
                        original_source_preserved=True,replacement_feature=str(OUT/f'ice/{pt}.json')))
            for pid,f in nonice['borders'].items():
                rr=f['rows'];supports[pid]=dict(rows=rr,kind='panel_border',closed=False,source=str(OUT/f'nonice/{pt}.json'))
                if not rr:continue
                keep,why=select_representatives(rr,dict(kind='ice_segment'))
                for i in keep:
                    r=rr[i]
                    if any(h['native_path_id']==pid and np.linalg.norm(np.asarray(r['uv'])-h['uv'])<25 for h in manual):continue
                    sid=r['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='repaired_current_RGB_panel_outer_border',human_acceptance=False,source_record=r,source_ref=ad_ref,
                        source_BGR_sha256=packet['source_BGR_sha256'],source_group_id=f's02/{pt}/{pid}/panel_material',selection_reasons=sorted(why[i]),
                        review_order=dict(family='ad_source_edge',run=pid,order=r['order'])))
            # The original qualified board traces are already real source
            # material reads. Preserve blue-top/yellow-bottom roles and masks.
            board=read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json')
            for pid in {r['native_path_id']for r in frame if r['native_path_id'].startswith('board/')}:
                role='board.top'if 'top'in pid else'board.contact'
                rr=[r for r in board['rows']if r['role']==role and r.get('pixels_measured_in_this_frame')and not r.get('occlusion')]
                supports[pid]=dict(rows=rr,kind='board',closed=False,source=str(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            write(support_dir/f'{pt}.json',supports)
            allrows.extend(frame);counts=collections.Counter(entities[r['native_path_id']]['layer']for r in frame)
            coverage.append(dict(pts=pt,source_seconds=pt/12800,counts=dict(counts),retained_existing_actual_line_points=len(retained),paths=sorted({r['native_path_id']for r in frame})))
    allrows.sort(key=lambda r:(r['pts'],r['native_path_id'],r['observation_id']));ids=[r['observation_id']for r in allrows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate repaired observation identity')
    paths=sorted({r['native_path_id']for r in allrows});lookup={p:i for i,p in enumerate(paths)}
    _,_,_,_,_,native=repair.fit.load()
    if not set(paths)<=native.paths.keys():raise ValueError('A repaired path is not in the current native model')
    z=dict(source_rows=np.arange(len(ids),dtype=np.int64),observation_ids=np.asarray(ids),uv=np.asarray([r['uv']for r in allrows]),pts=np.asarray([r['pts']for r in allrows],np.int64),
        kind=np.asarray([r['kind']for r in allrows],np.uint8),stations=np.asarray([r['station']for r in allrows]),path_indices=np.asarray([lookup[r['native_path_id']]for r in allrows],np.int32))
    np.savez_compressed(folder/'point_observations.npz',**z)
    groups=collections.defaultdict(list)
    with gzip.open(folder/'point_records.jsonl.gz','wt',encoding='utf8',compresslevel=3)as f:
        for i,r in enumerate(allrows):r['source_row']=i;dump(f,r);groups[r['source_group_id']].append(r['observation_id'])
    with gzip.open(folder/'source_correlated_groups.jsonl.gz','wt',encoding='utf8',compresslevel=3)as f:
        for group,observations in groups.items():dump(f,dict(group_id=group,observation_ids=observations,independent_samples=False))
    mapping=read(old.OUT/'points/mapping.json');mapping.update(path_ids=paths,point_input_sha256=repair.fit.file_ref(folder/'point_observations.npz')['sha256'],
        native_numeric_state=repair.fit.file_ref(repair.fit.OUT/'geometry_seed.npz'),auxiliary_camera=repair.fit.file_ref(OUT/'navigation_camera.npz'),human_scope_file=str(old.OUT/'evidence/manual_scope.json'),
        original_package_unchanged=repair.fit.file_ref(old.OUT/'points/point_observations.npz'),source_repair_dispositions=str(OUT/'dispositions.jsonl.gz'))
    write(folder/'mapping.json',mapping)
    originals={r['observation_id']:r for r in original if r['origin']=='preserved_human_source_click'}
    carried={r['observation_id']:r for r in allrows if r['origin']=='preserved_human_source_click'}
    if originals!=carried:
        # Only the output table's row index may change; the original scope may not.
        for key,value in originals.items():
            if key not in carried or any(value.get(k)!=carried[key].get(k)for k in ['uv','pts','native_path_id','human_acceptance','human_acceptance_scope','source_record']):raise ValueError('Original human scope changed')
    count=collections.Counter(entities[r['native_path_id']]['layer']for r in allrows)
    result=dict(frames=567,source_seconds=[44.98,56.3],rows=len(allrows),counts=dict(count),native_paths=len(paths),coverage=coverage,
        original_human_rows=len(originals),user_confirmed_rows=sum(r['human_acceptance']for r in carried.values()),
        native_geometry_changed=False,old_data_overwritten=False,machine_results_user_accepted=False,
        input_sha256=mapping['point_input_sha256'],ice_empty_frames=[r['pts']for r in coverage if not r['counts'].get('ice')])
    write(OUT/'result.json',result);grid.unchanged();print(json.dumps({k:v for k,v in result.items()if k!='coverage'},ensure_ascii=False),flush=True)

def initialize():
    from rinkspace.session.gpu_geometry import initialize_curve_stations
    from prepare_native_s01 import load_pixel_input
    m,_,_,g,core,native=repair.fit.load();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));mapping=read(folder/'mapping.json');state=dict(np.load(OUT/'navigation_camera.npz'))
    camera=read(repair.fit.OUT/'camera_seed.json');controls=np.load(repair.fit.OUT/'controls_seed.npy');paths=[mapping['path_ids'][i]for i in z['path_indices']]
    # The freely adjusted source-search camera is not the registered solver
    # initialization. Honour the camera hashes already declared by the input.
    stations,report=initialize_curve_stations(native,camera=camera,controls=controls,path_ids=paths,stations=z['stations'],pts=z['pts'],observed=z['uv'],kinds=z['kind'],backend='cpu',progress=lambda r:print(json.dumps(r),flush=True))
    np.savez_compressed(folder/'initialized_points.npz',stations=stations);load_pixel_input(folder)
    with native.resident(camera=camera,controls=controls,path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'])as resident:result=resident.evaluate('cpu')
    np.savez_compressed(folder/'native_evaluation.npz',status=result['status'],projected_UV=result['values'][:,3:5],residual=result['values'][:,9:11])
    auxiliary=dict(camera,rig=state['rig'].tolist())
    with native.resident(camera=auxiliary,controls=state['controls'],path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'])as resident:nav=resident.evaluate('cpu')
    np.savez_compressed(folder/'auxiliary_evaluation.npz',status=nav['status'],projected_UV=nav['values'][:,3:5],residual=nav['values'][:,9:11])
    report.update(rows=len(z['pts']),invalid=int(np.count_nonzero(result['status'])),auxiliary_projection_invalid=int(np.count_nonzero(nav['status'])),
        camera_or_scene_updated=False,projection_error_is_not_pixel_accuracy=True,initialization_camera=repair.fit.file_ref(repair.fit.OUT/'camera_seed.json'),initialization_controls=repair.fit.file_ref(repair.fit.OUT/'controls_seed.npy'))
    write(OUT/'native_integration.json',report);repair.fit.load();print(json.dumps({k:v for k,v in report.items()if k!='unresolved'}),flush=True)


def verify():
    from prepare_native_s01 import load_pixel_input
    folder=OUT/'points';z,m,stations=load_pixel_input(folder);inventory=read(old.OUT/'evidence/source_inventory.json')
    original={r['observation_id']:r for r in repair.rows(old.OUT/'points/point_records.jsonl.gz')}
    banned=set(inventory['withdrawals'])|set(inventory['historical_corrections']);source_refs={};groups=collections.defaultdict(list);temporal=collections.defaultdict(list)
    counts=collections.Counter();seen=set()
    if not np.array_equal(np.unique(z['pts']),np.arange(old.FIRST,old.LAST+1,256)):raise ValueError('An official exposure is missing')
    if not np.isfinite(z['uv']).all()or np.any(z['uv']<0)or np.any(z['uv']>[1919,1079]):raise ValueError('Invalid source UV')
    for r in repair.rows(folder/'point_records.jsonl.gz'):
        i=r['source_row'];seen.add(i);source=r['source_record'];counts[r['origin']]+=1
        if r['observation_id']!=str(z['observation_ids'][i])or r['pts']!=int(z['pts'][i])or not np.array_equal(r['uv'],z['uv'][i])or r['native_path_id']!=m['path_ids'][int(z['path_indices'][i])]:raise ValueError('Evidence/input mismatch')
        if r['origin']=='reuse_authenticated_current_RGB_profile_measurement':
            if r['uv']!=source['source_profile']['actual_readout_uv']:raise ValueError('Ad RGB profile was not used verbatim')
            parent=original[r['original_observation_id']]
            if r['original_uv']!=parent['uv']or r['source_group_id']!=parent['source_group_id']:raise ValueError('Original ad lineage changed')
        elif r['uv']!=source['uv']:raise ValueError('A fitted/search coordinate replaced an actual readout')
        sid=source.get('source_evidence',{}).get('original_source_row_id',r['source_id'])
        if sid in banned:raise ValueError('Historical withdrawal/correction was ignored')
        if r['human_acceptance']:
            parent=original.get(r['observation_id'])
            if parent is None or not parent['human_acceptance']or any(r.get(k)!=parent.get(k)for k in ['uv','pts','native_path_id','human_acceptance_scope']):raise ValueError('Human acceptance expanded')
        ref=r.get('source_ref')
        if ref and ref.get('sha256'):source_refs[ref['path']]=ref['sha256']
        groups[r['source_group_id']].append(r['observation_id']);temporal[r['native_path_id']].append(r['observation_id'])
    if seen!=set(range(len(z['pts']))):raise ValueError('Missing row evidence')
    for ref in inventory['source_files']:source_refs[ref['path']]=ref['sha256']
    for path,sha in source_refs.items():
        if repair.fit.file_ref(Path(path))['sha256']!=sha:raise ValueError('Source evidence changed: '+path)
    declared={g['group_id']:g['observation_ids']for g in repair.rows(folder/'source_correlated_groups.jsonl.gz')}
    if dict(groups)!=declared:raise ValueError('Correlated groups do not cover the exact delivered rows')
    with gzip.open(folder/'temporal_material_groups.jsonl.gz','wt',encoding='utf8')as f:
        for key,ids in temporal.items():dump(f,dict(native_path_id=key,observation_ids=ids,shared_physical_material_across_exposures=True,frame_source_groups_preserved=True,independence_not_claimed=True))
    evaluation=np.load(folder/'native_evaluation.npz')
    if len(evaluation['status'])!=len(z['pts']):raise ValueError('Native evaluation has missing rows')
    if np.any(evaluation['status']):raise ValueError('Registered S02 native initialization has invalid rows')
    # A navigation camera cannot certify unoptimized finite geometry. Keep
    # those actual source rows and expose every camera-domain failure.
    camera_failures=[]
    auxiliary=np.load(folder/'auxiliary_evaluation.npz')
    for i in np.flatnonzero(auxiliary['status']):
        camera_failures.append(dict(observation_id=str(z['observation_ids'][i]),pts=int(z['pts'][i]),uv=z['uv'][i].tolist(),
            path_id=m['path_ids'][int(z['path_indices'][i])],status=int(auxiliary['status'][i]),
            reason='behind_auxiliary_camera'if auxiliary['status'][i]==5 else'outside_auxiliary_camera_optical_domain',point_not_removed=True))
    write(OUT/'auxiliary_projection_failures.json',dict(rows=camera_failures,scene_geometry_not_modified=True,auxiliary_camera_not_used_as_solver_initialization=True,requires_camera_or_identity_diagnosis_for_failed_auxiliary_projection=True))
    repair.fit.load()
    result=dict(frames=567,rows=len(z['pts']),origins=dict(counts),source_refs_verified=len(source_refs),human_source_rows=counts['preserved_human_source_click'],
        native_geometry_and_original_input_unchanged=True,actual_uv_and_integer_pts_verified=True,correlated_groups=len(groups),machine_results_user_accepted=False,
        auxiliary_projection_invalid_rows=len(camera_failures),numerical_ingestion_does_not_certify_visual_accuracy=True)
    write(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


def bundle():
    verify();path=OUT/'S02_repaired_native_points_20261027.zip'
    files=[(f,'points/'+f.name)for f in (OUT/'points').iterdir()if f.is_file()]
    files.extend((OUT/f,f)for f in ['HANDOFF.md','result.json','verification.json','native_integration.json','dispositions.jsonl.gz','auxiliary_projection_failures.json','material_coverage.json','material_identity_conflicts.json','remaining_issues.json'])
    files.extend([(old.OUT/'evidence/manual_scope.json','history/manual_scope.json'),(old.OUT/'evidence/source_inventory.json','history/source_inventory.json'),
                  (repair.fit.OUT/'geometry_seed.npz','navigation/geometry_seed.npz'),(OUT/'navigation_camera.npz','navigation/auxiliary_camera.npz'),(repair.fit.OUT/'camera_seed.json','navigation/camera_seed.json'),(repair.fit.OUT/'controls_seed.npy','navigation/controls_seed.npy'),
                  (repair.fit.OUT/'manifest.json','native_authority.json'),(OUT/'review/video.json','review/video.json')])
    manifest=dict(schema='rinkspace.s02_source_repair_bundle.v1',shot='s02',source_seconds=[44.98,56.3],frames=567,
        point_folder='points',original_input=repair.fit.file_ref(old.OUT/'points/point_observations.npz'),files=[dict(archive=name,**repair.fit.file_ref(file))for file,name in files],
        shared_native_files_changed=False,raw_source_history_kept=True,machine_results_user_accepted=False)
    with zipfile.ZipFile(path.with_suffix('.partial.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5)as archive:
        for file,name in files:archive.write(file,name)
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    path.with_suffix('.partial.zip').replace(path)
    with zipfile.ZipFile(path)as archive:
        if archive.testzip()is not None:raise ValueError('Archive checksum failed')
    write(OUT/'package.json',dict(archive=repair.fit.file_ref(path),bytes=path.stat().st_size));print(str(path),flush=True)

def source_trace(rows,mask,kind,closed=False):
    """Draw actual support order, never an unobserved chord or a model curve."""
    if len(rows)<2:return [],[]
    uv=np.asarray([r['uv']for r in rows],float)
    circular=kind in ('ice_circle','ice_ellipse')
    if circular:order=np.asarray([r.get('source_order_arc_px',r.get('theta',i))for i,r in enumerate(rows)])
    elif kind in ('board','panel_border'):order=uv[:,1]if np.ptp(uv[:,1])>np.ptp(uv[:,0])else uv[:,0]
    else:
        direction=np.linalg.svd(uv-uv.mean(0),full_matrices=False)[2][0];order=uv@direction
    ids=np.argsort(order,kind='stable');q=uv[ids]
    # These are the existing reader's actual support intervals: 260px for a
    # coherent line, three 35px queries for a curved material. No UV is deleted.
    limit=105. if circular else 260.
    gaps=np.linalg.norm(np.diff(q,axis=0),axis=1)>limit
    if circular and all('query_index'in r.get('source_evidence',{})for r in rows):
        query=np.asarray([r['source_evidence']['query_index']for r in rows])[ids]
        # Nearby sides of a very flat circle can straddle an unobserved turn.
        # Source-query continuity, not pixel proximity, determines that link.
        gaps|=np.abs(np.diff(query))>3
    runs=np.r_[0,np.cumsum(gaps)]
    pairs,breaks=source_connections(q,runs,np.arange(len(q),dtype=float),mask)
    if circular and closed and not any('source_order_arc_px'in r for r in rows) and len(q)>2 and np.linalg.norm(q[0]-q[-1])<=limit:
        # Require near-complete angular coverage as well as image proximity;
        # the two turns of a flattened partial arc can otherwise appear close.
        angle=order[ids]
        positive=np.diff(angle);step=np.median(positive[positive>0])if np.any(positive>0)else 0
        if step and 2*np.pi-angle[-1]+angle[0]<=3*step:
            closing,_=source_connections(q[[-1,0]],['closure']*2,np.array([0.,1.]),mask)
            if closing:pairs.append((len(q)-1,0))
    return [q[[a,b]]for a,b in pairs],breaks

class Review:
    def __init__(self):
        self.mapping=read(OUT/'points/mapping.json');self.z=dict(np.load(OUT/'points/point_observations.npz'));self.by=collections.defaultdict(list)
        self.shot_label=self.mapping.get('shot_id','s02').upper()
        for r in repair.rows(OUT/'points/point_records.jsonl.gz'):self.by[r['pts']].append(r)
        self.vis={v['pts']:v for v in read(old.OUT/'evidence/visibility.json')['frames']}
        self.entities={v['native_path_id']:v for v in read(old.OUT/'evidence/native_entities.json')['entities']}
        self.atlases=read(old.OUT/'ads/atlases.json')['atlases']+read(old.OUT/'ads/source_review/additional_source_atlases.json')['atlases']
        self.charts=read(old.OUT/'ads/source_review/source_identity_charts.json')['records']
        self.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);self.small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)

    def ad_edges(self,group,mask,pt):
        uv=np.asarray([r['uv']for r in group]);refs=[]
        for ai,pi in {(r['review_order'].get('atlas'),r['review_order'].get('source_path'))for r in group if r['review_order'].get('family')=='ad'}:
            chart=self.charts.get(str(ai),{}).get(str(pt))
            if chart is None:continue
            p=cv2.perspectiveTransform(np.asarray(self.atlases[ai]['paths'][pi]['pixels'],float)[None],np.asarray(chart['H'],float))[0]
            order,distance=polyline_order(uv,p);refs.append((float(np.median(distance)),order,p))
        if not refs:return [],[]
        _,order,p=min(refs,key=lambda x:x[0]);ids=np.argsort(order);edges=[];rejected=[]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        proposed=list(zip(ids[:-1],ids[1:]))
        if np.linalg.norm(p[0]-p[-1])<2 and len(ids)>2:proposed.append((ids[-1],ids[0]))
        for a,b in proposed:
            lo,hi=order[a],order[b];between=p[(arc>=lo)&(arc<=hi)]if hi>=lo else p[(arc>=lo)|(arc<=hi)]
            q=uv[[a,b]];clear,blocked=source_connections(q,['source']*2,np.array([0.,1.]),mask)
            # A missing material turn cannot be replaced by its straight chord.
            if len(between):
                d=q[1]-q[0];u=np.clip((between-q[0])@d/max(float(d@d),1e-9),0.,1.)
                departure=np.linalg.norm(between-q[0]-u[:,None]*d,axis=1).max()
            else:departure=0.
            if clear and departure<=3.:edges.append(q)
            else:rejected.append(dict(a=int(a),b=int(b),reason='unobserved_material_turn_or_occlusion',source_contour_chord_departure_px=float(departure)))
        return edges,rejected

    def frame(self,pt,im):
        if hashlib.sha256(im.tobytes()).hexdigest()!=self.vis[pt]['source_BGR_sha256']:raise ValueError('Review source changed')
        from ice_points_s04 import current_graphics
        mask=cv2.imread(self.vis[pt]['mask_path'],0)>0;graphics,_=current_graphics(im);mask|=graphics
        canvas=np.full((1160,2400,3),20,np.uint8);canvas[:1080,:1920]=im
        grouped=collections.defaultdict(list)
        for row in self.by[pt]:grouped[row['native_path_id']].append(row)
        support=read(OUT/f'support/{pt}.json');ledger=[];labels=[]
        for pid,group in sorted(grouped.items(),key=lambda x:('::'in x[0],x[0])):
            color=entity_color(pid)[::-1]
            if pid in support:
                s=support[pid];connection_mask=mask
                if s.get('source_material_exclusion_mask'):
                    material=cv2.imread(s['source_material_exclusion_mask'],0)
                    if material is None or material.shape!=mask.shape:raise ValueError('Missing source material connection domain')
                    connection_mask=mask|(material>0)
                if s.get('branches'):
                    edges=[];missing=[]
                    for branch in s['branches']:
                        links,gaps=source_trace(branch['rows'],connection_mask,s['kind'],False)
                        edges.extend(links);missing.extend(gaps)
                else:edges,missing=source_trace(s['rows'],connection_mask,s['kind'],s['closed'])
            else:edges,missing=self.ad_edges(group,mask,pt)
            for q in edges:cv2.line(canvas,tuple(np.rint(q[0]).astype(int)),tuple(np.rint(q[1]).astype(int)),color,1,cv2.LINE_AA)
            for r in group:
                xy=tuple(np.rint(r['uv']).astype(int));cv2.circle(canvas,xy,3,(15,15,15),1,cv2.LINE_AA);cv2.circle(canvas,xy,2,color,1,cv2.LINE_AA)
            if '::'in pid:
                instance,template=pid.split('::');instance=instance.rsplit('/',1)[-1]
                label=(instance+'.'+template.split('.',1)[-1]).replace('.print.','.').replace('.glyph_','.').replace('boundary.','b').replace('_part','p').replace('/path','/p').replace('bottom_','下').replace('top_','上').replace('SKATING_','SK_')
            else:
                label=self.entities[pid]['label'].removeprefix('ice.').replace(':segment0','').replace('house_connection_','').replace('goal_trapezoid_right_upper_outer','右梯形/上').replace('goal_trapezoid_right_lower_outer','右梯形/下').replace('goal_crease_right','右蓝区轮廓').replace(':centreline','/中线').replace('board.top','围挡上沿').replace('board.bottom','围挡下沿')
            labels.append((label,len(group),color[::-1]));ledger.append(dict(path_id=pid,observation_ids=[r['observation_id']for r in group],drawn_source_links=len(edges),unsupported_links=missing))
        pic=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pic)
        draw.text((1938,18),f'{self.shot_label}  原片 {pt/12800:.3f} 秒',font=self.font,fill='white');draw.text((1938,50),f'PTS {pt} · {len(self.by[pt])} 点',font=self.font,fill='white')
        # Every group has a label; no top-down scene or dense native mesh covers
        # the source. Two columns are used only when the whole frame needs them.
        columns=2 if len(labels)>45 else 1;step=22
        for i,(name,n,color) in enumerate(labels):
            col=i//46 if columns==2 else 0;row=i%46 if columns==2 else i
            x=1937+col*225;y=92+row*step
            text=name+f' {n}';width=205 if columns==2 else 425
            while draw.textbbox((0,0),text,font=self.small)[2]>width and len(name)>4:name=name[:-1];text=name+'…'+f' {n}'
            draw.line((x,y+10,x+12,y+10),fill=color,width=2);draw.text((x+17,y),text,font=self.small,fill=color)
        draw.text((18,1090),'彩点：进入数据集的原片实测点；细线：同一材料的实测支持连接。所有点保持真实读数。',font=self.font,fill='white')
        draw.text((18,1124),'缺段、遮挡及缺少材料转折支持的位置留空；不强行闭合圆弧，不显示模型投影。机器结果尚未经人工验收。',font=self.font,fill=(210,210,210))
        return cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR),ledger

def review(video=True):
    cv2.setNumThreads(2);r=Review();folder=OUT/'review';folder.mkdir(exist_ok=True);pts=np.arange(old.FIRST,old.LAST+1,256)
    if not video:
        for pt in [575744,601344,619264,644608,682496,703744,720640]:
            im,ledger=r.frame(pt,cv2.imread(r.vis[pt]['image']));cv2.imwrite(str(folder/f'{pt}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,96])
        return
    path=folder/'S02_repaired_points_connected.mp4';tmp=path.with_suffix('.partial.mp4');start=time.monotonic();displayed=0
    with gzip.open(folder/'frame_index.jsonl.gz','wt',encoding='utf8')as ledger,av.open(str(tmp),'w')as container:
        stream=container.add_stream('libx264',rate=50);stream.width=2400;stream.height=1160;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,12800);stream.options={'crf':'17','preset':'fast'}
        for i,(pt,im)in enumerate(source_frames(Path(r.mapping['source_video']),dict(time_base_num=1,time_base_den=12800,width=1920,height=1080),pts)):
            canvas,groups=r.frame(pt,im);dump(ledger,dict(pts=pt,entities=groups));displayed+=len(r.by[pt]);v=av.VideoFrame.from_ndarray(canvas,format='bgr24');v.pts=pt-old.FIRST;v.time_base=Fraction(1,12800)
            for packet in stream.encode(v):container.mux(packet)
            if i%100==0:print(json.dumps(dict(stage='review',frames=i+1,seconds=time.monotonic()-start)),flush=True)
        for packet in stream.encode():container.mux(packet)
    result=validate_video(tmp,pts,Fraction(1,12800));tmp.replace(path)
    if displayed!=len(r.z['pts']):raise ValueError('Review omitted input observations')
    write(folder/'video.json',dict(video=repair.fit.file_ref(path),validation=result,points_displayed=displayed,point_input_sha256=r.mapping['point_input_sha256'],projection_shown=False,source_seconds=[44.98,56.3]));print(str(path),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','initialize','review','previews','verify','bundle']);a=p.parse_args()
    if a.stage=='previews':review(False)
    else:globals()[a.stage]()
