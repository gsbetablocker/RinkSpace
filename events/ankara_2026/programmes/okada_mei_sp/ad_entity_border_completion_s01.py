"""Complete clear outer panel arcs using local source profiles and existing identities."""
from pathlib import Path
import sys,json,gzip,collections,hashlib,itertools,shutil,time
import cv2,numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_measure_s01 as m
r=m.r;b=r.b;OUT=r.OUT

def read_arc(pt,name,q,n):
    im,gray,mask=r.current(pt);profiles=m.read_profiles(im,r.VIS[pt],q,np.tile(n,(len(q),1)));runs=[];run=[]
    for p in profiles:
        if m.profile_reason(p) is None:run.append(np.asarray(p['actual_source_uv']))
        elif run:
            if len(run)>=4:runs.append(np.asarray(run))
            run=[]
    if len(run)>=4:runs.append(np.asarray(run))
    return [dict(path_id=name,pixels=a.tolist(),contour_sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),proof=dict(kind='actual_partial_panel_interface_with_local_plateaus',source_pts=pt,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],partial_arc_only=True,not_a_fixed_corner_reference=True,other_board_rail_is_not_the_ad_boundary=True)) for a in runs]

def run():
    cv2.setNumThreads(2);extra=b.read(OUT/'additional_source_atlases.json');doc=b.read(OUT/'source_identity_charts.json')
    if any(a['id']>=84 for a in extra['atlases']):raise ValueError('Border completion already built')
    specs=[(84,82,4,368640,[('Ad4.top_edge/path0',np.column_stack([np.arange(355.,960.,3.),np.full(202,404.)]),[0.,-1.]),('Ad4.bottom_edge/path0',np.column_stack([np.arange(355.,900.,3.),np.full(182,535.)]),[0.,1.])]),
    (85,83,7,507136,[('Ad7.top_edge/path0',np.column_stack([np.arange(475.,1900.,3.),np.full(475,488.)]),[0.,-1.]),('Ad7.bottom_edge/path0',np.column_stack([np.arange(475.,1900.,3.),np.full(475,709.)]),[0.,1.]),('Ad7.left_vertical/path0',np.column_stack([np.full(65,442.),np.arange(500.,695.,3.)]),[-1.,0.])])]
    refs={}
    for ai,old,ad,pt,queries in specs:
        paths=[]
        for name,q,n in queries:paths.extend(read_arc(pt,name,q,np.array(n)))
        f=r.VIS[pt];a=dict(id=ai,ad=ad,image=f['image'],image_sha256=b.sha(f['image']),paths=paths,partial_source_reference=True)
        extra['atlases'].append(a);doc['records'][str(ai)]={p:dict(c,atlas_id=ai,method='actual_partial_panel_interface_local_material_plateaus') for p,c in doc['records'][str(old)].items()};refs[ai]=r.reference(a)
        print(json.dumps(dict(source_atlas=ai,source_arcs=len(paths),source_points=sum(len(p['pixels']) for p in paths))),flush=True)
    b.write(OUT/'additional_source_atlases.json',extra);b.write(OUT/'source_identity_charts.json',doc)
    z=dict(np.load(OUT/'verified_ads.npz'));mapping=b.read(OUT/'verified_mapping.json');paths=mapping['path_ids'];lookup={p.split('::')[1]:i for i,p in enumerate(paths)};nextrow=int(z['source_rows'].max())+1;added=[];bypt=collections.defaultdict(list);start=time.monotonic();reason=collections.Counter()
    pts=sorted({int(pt) for ai in refs for pt in doc['records'][str(ai)] if int(pt)>=283136})
    for fi,pt in enumerate(pts):
        im,gray,mask=r.current(pt);near=collections.defaultdict(list)
        for j in np.flatnonzero(z['pts']==pt):near[paths[z['path_indices'][j]].split('::')[1]].append(z['uv'][j])
        qq=[];nn=[];vv=[];which=[]
        for ai,ref in refs.items():
            chart=doc['records'][str(ai)].get(str(pt))
            if chart is None:continue
            for v in m.chart_paths(ref,chart,im,gray,mask):
                vi=len(vv);vv.append(v)
                for qi in range(1,len(v['q'])-1,4):
                    if not v['valid'][qi] or v['ncc'][qi]<.05:continue
                    q=v['q'][qi]
                    if near[v['path']] and np.min(np.linalg.norm(np.asarray(near[v['path']])-q,axis=1))<3.:continue
                    qq.append(q);nn.append(v['n'][qi]);which.append((vi,qi))
        profiles=m.read_profiles(im,r.VIS[pt],np.array(qq),np.array(nn)) if qq else []
        for j,(p,(vi,qi)) in enumerate(zip(profiles,which)):
            v=vv[vi];name=v['path']
            if m.profile_reason(p):reason['local_material_transition_unresolved']+=1;continue
            uv=np.asarray(p['actual_source_uv']);off=np.asarray(p['source_q10_q50_q90_normal_offsets_px']);norm=np.asarray(nn[j])
            rgb=r.sample(im,(qq[j]+np.array([off[0]-3,off[0]-1,off[2]+1,off[2]+3])[:,None]*norm)[None])[0];sides=np.stack([rgb[:2].mean(0),rgb[2:].mean(0)]);chroma=np.ptp(sides,axis=1)
            if not(chroma.max()>20 and chroma.min()<max(15.,chroma.max()*.60)):reason['no_actual_paint_background_interface']+=1;continue
            if near[name] and np.min(np.linalg.norm(np.asarray(near[name])-uv,axis=1))<3.:continue
            sid=f's01_native_source_v2_border_{pt}_{lookup[name]:03d}_{len(bypt[pt]):04d}'
            rec=dict(source_row=nextrow,stable_id=sid,pts=pt,uv=uv.tolist(),path_id=paths[lookup[name]],kind=1,station=.5,decision='new_actual_source_boundary',identity=dict(atlas_id=v['atlas_id'],source_path_index=v['source_path_index'],source_contour_sha256=v['source_contour_sha256'],source_chart_pts=pt,current_neighborhood_correlation=float(v['ncc'][qi]),unique_native_entity=True),source_profile=dict(search_uv=np.asarray(qq[j]).tolist(),normal=norm.tolist(),offsets=off.tolist(),search_radius_px=p['source_search_radius_px'],flags=p['source_flags']),source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],source_navigation_not_observed_UV=True,native_geometry_used_for_identity=False,current_observability=dict(fine_detail=False,actual_colored_panel_to_background_transition=True,partial_material_arc_visible=True,source_reference_is_actual_RGB=True))
            nextrow+=1;added.append(rec);bypt[pt].append(rec);near[name].append(uv)
        if fi%100==0:print(json.dumps(dict(frame=fi,frames=len(pts),added=len(added),seconds=round(time.monotonic()-start,1))),flush=True)
    a=dict(source_rows=np.array([c['source_row'] for c in added],np.int64),observation_ids=np.array([c['stable_id'] for c in added]),uv=np.array([c['uv'] for c in added]).reshape(-1,2),pts=np.array([c['pts'] for c in added],np.int64),kind=np.ones(len(added),np.uint8),stations=np.full(len(added),.5),path_indices=np.array([paths.index(c['path_id']) for c in added],np.int32))
    merged={k:np.concatenate([z[k],a[k]]) for k in z}
    if len(set(merged['observation_ids']))!=len(merged['pts']) or len(set(merged['source_rows']))!=len(merged['pts']):raise ValueError('Duplicate source identity')
    for k in z:
        if not np.array_equal(merged[k][:len(z['pts'])],z[k]):raise ValueError('Existing point changed')
    history=OUT/'before_border_completion';history.mkdir(exist_ok=True)
    for name in ['verified_ads.npz','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz','measurement_summary.json']:
        src=(OUT/name).resolve();dst=(history/name).resolve()
        if dst.exists() or not dst.is_relative_to(OUT.resolve()):raise ValueError('Unsafe/repeated history target')
        src.rename(dst)
    np.savez_compressed(OUT/'verified_ads.npz',**merged)
    with gzip.open(history/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f,gzip.open(OUT/'pointwise_checks.jsonl.gz','wt',encoding='utf8') as out:
        for pt,rows in itertools.groupby((json.loads(s) for s in f),key=lambda c:c['pts']):
            for c in rows:out.write(json.dumps(c,ensure_ascii=False)+'\n')
            for c in bypt[pt]:out.write(json.dumps(c,ensure_ascii=False)+'\n')
    count=collections.Counter((c['pts'],c['path_id']) for c in added)
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(OUT/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as out:
        for s in f:
            c=json.loads(s);n=count[(c['pts'],c['path_id'])];c['included_points']+=n;c['supplemental_current_material_arc_points']=n;out.write(json.dumps(c,ensure_ascii=False)+'\n')
    summary=b.read(history/'measurement_summary.json');summary['ad_rows']=len(merged['pts']);summary['by_path']=dict(collections.Counter(paths[k] for k in merged['path_indices']));summary['by_ad']=dict(collections.Counter(paths[k].split('::')[0].split('/')[-1] for k in merged['path_indices']));summary['decisions']['new_actual_source_boundary']+=len(added)
    for f in summary['frames']:
        extra=bypt[f['pts']];f['ad_points']+=len(extra);f['new_source_measurements']+=len(extra);cc=collections.Counter(f['by_ad']);cc.update(c['path_id'].split('::')[0].split('/')[-1] for c in extra);f['by_ad']=dict(cc)
    b.write(OUT/'measurement_summary.json',summary);mapping.update(identity_charts_sha256=b.sha(OUT/'source_identity_charts.json'),point_input_sha256=b.sha(OUT/'verified_ads.npz'));b.write(OUT/'verified_mapping.json',mapping)
    b.write(OUT/'border_completion.json',dict(added_points=len(added),frames_with_additions=len(bypt),current_existing_points_unchanged=True,all_new_points_actual_current_RGB=True,no_native_geometry_change=True,unresolved_searches=dict(reason),new_source_atlases=[84,85]))
    print(json.dumps(dict(done=True,added=len(added),total=len(merged['pts']),seconds=round(time.monotonic()-start,1))),flush=True)
if __name__=='__main__':run()
