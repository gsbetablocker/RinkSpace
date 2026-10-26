"""Select geometric representatives from already source-verified observations.

Selection never changes UV, PTS, native identity, or loss weights. Dense validated
measurements remain an evidence pool; only selected representatives reach native.
"""
from pathlib import Path
import sys,json,gzip,itertools,collections,time,hashlib
import cv2,numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_measure_s01 as m
r=m.r;b=r.b;OUT=r.OUT
EPS=.75
REFERENCES={}

def priority(c):
    return (c['kind']==0,c['decision']=='include_original_pixel',-float(c['source_profile']['offsets'][2]-c['source_profile']['offsets'][0]))

def reduce_run(records,indices,geometry,protect,why):
    if not len(indices):return set()
    selected=set()
    if len(indices)<=2:
        for i in indices:why[i].add('visible_run_support');selected.add(i)
        return selected
    q=np.asarray(geometry,float)
    epsilon=max(EPS,.01*float(np.linalg.norm(np.ptp(q,axis=0))))
    approx=cv2.approxPolyDP(q.astype(np.float32),epsilon,False)[:,0]
    # Sharp, actually supported turns survive scale-normalized smooth-arc sampling.
    coarse=cv2.approxPolyDP(q.astype(np.float32),epsilon,False)[:,0]
    if len(coarse)>2:
        d=np.diff(coarse,axis=0);length=np.linalg.norm(d,axis=1);unit=d/np.maximum(length[:,None],1e-9);turn=(np.sum(unit[:-1]*unit[1:],axis=1)<np.cos(np.deg2rad(55)))&(np.minimum(length[:-1],length[1:])>2.)
        approx=np.vstack([approx,coarse[1:-1][turn]])
    tree=cKDTree(q);_,nodes=tree.query(approx)
    for k in np.unique(nodes):
        i=int(indices[int(k)]);selected.add(i);why[i].add('source_shape_turn_or_visible_run_endpoint')
    # Across a wide image span, keep camera-field coverage as well as endpoints.
    # A tiny local line does not receive a regular dense grid.
    for axis,cuts in [(0,[480.,960.,1440.]),(1,[270.,540.,810.])]:
        for cut in cuts:
            if np.ptp(q[:,axis])>(240. if axis==0 else 135.) and q[:,axis].min()+15<cut<q[:,axis].max()-15:
                k=int(np.argmin(abs(q[:,axis]-cut)));i=int(indices[k]);selected.add(i);protect.add(i);why[i].add('separated_image_field_support')
    return selected

def select_path(pt,records,doc,ats):
    why=collections.defaultdict(set);protect={i for i,c in enumerate(records) if c['kind']==0}
    for i in protect:why[i].add('verified_fixed_material_corner')
    selected=set(protect);name=records[0]['path_id'].split('::')[1]
    if len(records)<=2:return set(range(len(records))),{i:{'visible_run_support'} for i in range(len(records))}
    border=any(k in name for k in ['_edge','_vertical'])
    if border:
        uv=np.array([c['uv'] for c in records]);centre=np.median(uv,axis=0);_,_,vt=np.linalg.svd(uv-centre,full_matrices=False);s=(uv-centre)@vt[0];order=np.argsort(s);s=s[order]
        # Median normal position suppresses repeated pixel-level wobble only for
        # deciding which original observations to retain. UV is never averaged.
        q=uv[order].copy();normal=vt[1];tangent=vt[0];normal_pos=q@normal
        smooth=np.array([np.median(normal_pos[max(0,j-2):min(len(q),j+3)]) for j in range(len(q))])
        geometry=np.outer(q@tangent,tangent)+np.outer(smooth,normal)
        positive=np.diff(s);positive=positive[positive>.25];gap=max(12.,3*float(np.median(positive))) if len(positive) else 12.
        cuts=np.r_[0,np.flatnonzero(np.diff(s)>gap)+1,len(order)]
        for a,c in zip(cuts[:-1],cuts[1:]):
            local=q[a:c];u=local@tangent;y=local@normal
            if len(local)>=5 and np.ptp(u)>1:
                curve=np.polynomial.Polynomial.fit(u,y,min(4,max(1,len(local)//5)))
                shape=np.outer(u,tangent)+np.outer(curve(u),normal)
            else:shape=local
            selected|=reduce_run(records,order[a:c],shape,protect,why)
    else:
        groups=collections.defaultdict(list)
        for i,c in enumerate(records):groups[(c['identity']['atlas_id'],c['identity']['source_path_index'])].append(i)
        curves={};closed=[]
        for key,indices in groups.items():
            ai,pi=key;p=np.asarray(ats[ai]['paths'][pi]['pixels'],float);chart=doc['records'][str(ai)][str(pt)];p=cv2.perspectiveTransform(p[None],np.asarray(chart['H']))[0];v=m.interp_path(p,.75)
            if v is None:continue
            q,n,station=v
            if ai not in REFERENCES:REFERENCES[ai]=r.reference(ats[ai])
            H=np.asarray(chart['H']);inv=np.linalg.inv(H);left=cv2.perspectiveTransform((q-n*1.25)[None],inv)[0];right=cv2.perspectiveTransform((q+n*1.25)[None],inv)[0];delta=r.sample(REFERENCES[ai]['gray'],right[:,None,:]).ravel()-r.sample(REFERENCES[ai]['gray'],left[:,None,:]).ravel();n[delta<0]*=-1
            curves[key]=(q,n,cKDTree(q))
            if np.linalg.norm(p[0]-p[-1])<2.:closed.append(key)
        # All photographic references of one complete contour share ONE ordering.
        # This consolidates repeated evidence without merging native entities.
        if closed:
            uv=np.array([c['uv'] for c in records]);norm=np.array([c['source_profile']['normal'] for c in records]);scores={}
            for key in closed:
                q,n,tree=curves[key];dd,kk=tree.query(uv,k=min(12,len(q)));dd=dd.reshape(len(uv),-1);kk=kk.reshape(len(uv),-1);compatible=np.einsum('ijk,ik->ij',n[kk],norm)>.5;rank=np.where(compatible,dd,dd+100.);scores[key]=float(np.median(rank.min(1)))
            key=min(scores,key=scores.get);groups={key:list(range(len(records)))}
        for key,indices in groups.items():
            if key not in curves:selected.update(indices);continue
            q,n,tree=curves[key]
            if closed:
                search=np.array([records[i]['uv'] for i in indices]);norm=np.array([records[i]['source_profile']['normal'] for i in indices]);dd,kk=tree.query(search,k=min(12,len(q)));dd=dd.reshape(len(indices),-1);kk=kk.reshape(len(indices),-1);compatible=np.einsum('ijk,ik->ij',n[kk],norm)>.5;rank=np.where(compatible,dd,dd+100.);k=kk[np.arange(len(indices)),rank.argmin(1)]
                for j in np.flatnonzero(~compatible.any(1)):protect.add(indices[int(j)]);selected.add(indices[int(j)]);why[indices[int(j)]].add('distinct_current_material_direction')
            else:
                search=np.array([records[i]['source_profile'].get('search_uv',records[i]['uv']) for i in indices]);dist,k=tree.query(search)
            unique={}
            for i,j in zip(indices,k):
                if int(j) not in unique or priority(records[i])>priority(records[unique[int(j)]]):unique[int(j)]=i
            jj=np.array(sorted(unique));ii=np.array([unique[int(j)] for j in jj]);arc=jj*.75
            positive=np.diff(arc);positive=positive[positive>.25];gap=max(12.,3*float(np.median(positive))) if len(positive) else 12.
            cuts=np.r_[0,np.flatnonzero(np.diff(arc)>gap)+1,len(ii)]
            for a,c in zip(cuts[:-1],cuts[1:]):selected|=reduce_run(records,ii[a:c],q[jj[a:c]],protect,why)
    # Overlapping references of the same contour must not create duplicate weight.
    order=sorted(selected,key=lambda i:priority(records[i]),reverse=True);kept=[]
    for i in order:
        uv=np.asarray(records[i]['uv']);n=np.asarray(records[i]['source_profile']['normal']);duplicate=False
        if i not in protect:
            for j in kept:
                if np.linalg.norm(uv-records[j]['uv'])<2.5 and np.dot(n,records[j]['source_profile']['normal'])>.9:duplicate=True;break
        if not duplicate:kept.append(i)
    if not kept:
        kept=[max(range(len(records)),key=lambda i:priority(records[i]))];why[kept[0]].add('visible_entity_support')
    return set(kept),why

def run():
    cv2.setNumThreads(2);sample='--sample' in sys.argv;chosen={int(round(float(x)*12800)) for x in sys.argv[sys.argv.index('--sample')+1].split(',')} if sample else None
    doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases']+b.read(OUT/'additional_source_atlases.json')['atlases'];z=dict(np.load(OUT/'verified_ads.npz'));mapping=b.read(OUT/'verified_mapping.json');paths=mapping['path_ids'];selected=set();counts=[];start=time.monotonic();tag='sample_' if sample else ''
    output=OUT/(tag+'representative_checks.jsonl.gz')
    with gzip.open(OUT/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f,gzip.open(output,'wt',encoding='utf8') as dest:
        for fi,(pt,lines) in enumerate(itertools.groupby((json.loads(s) for s in f),key=lambda c:c['pts'])):
            rows=list(lines)
            if chosen is not None and pt not in chosen:continue
            by=collections.defaultdict(list)
            for c in rows:
                if c['decision']!='not_admitted':by[c['path_id']].append(c)
            picked=set();reasons={}
            for name,records in by.items():
                keep,why=select_path(pt,records,doc,ats)
                for i in keep:picked.add(records[i]['source_row']);reasons[records[i]['source_row']]=sorted(why[i])
                counts.append(dict(pts=pt,path_id=name,verified_dense_points=len(records),fit_representatives=len(keep),native_identity_unchanged=True))
            for c in rows:
                if c['decision']!='not_admitted':
                    if c['source_row'] in picked:
                        c['fit_sampling']=dict(selected=True,reasons=reasons[c['source_row']],source_geometry_selection_tolerance_px=EPS,observed_UV_unchanged=True)
                    else:
                        c.update(previous_decision=c['decision'],decision='not_admitted',reason='redundant_same_visible_contour',source_observation_valid=True,fit_sampling=dict(selected=False,identity_and_visibility_still_valid=True))
                dest.write(json.dumps(c,ensure_ascii=False)+'\n')
            selected.update(picked)
            if fi%150==0 or sample:print(json.dumps(dict(stage='geometric_representatives',pts=pt,before=sum(map(len,by.values())),after=len(picked),seconds=round(time.monotonic()-start,1))),flush=True)
    keep=np.isin(z['source_rows'],list(selected));zz={k:v[keep] for k,v in z.items()}
    np.savez_compressed(OUT/(tag+'representative_ads.npz'),**zz)
    b.write(OUT/(tag+'sampling_report.json'),dict(dense_verified_rows=int(sum(x['verified_dense_points'] for x in counts)),selected_rows=len(zz['pts']),source_frames=len(np.unique(zz['pts'])),native_entity_frame_pairs_before=len(counts),native_entity_frame_pairs_after=sum(x['fit_representatives']>0 for x in counts),all_verified_fixed_corners_preserved=bool(np.all(keep[z['kind']==0])) if not sample else None,source_UV_PTS_ID_path_unchanged=True,loss_weights_changed=False,native_projection_used_for_sampling=False,selection='Per visible contour run: source shape turns and endpoints; additional spatially separated image-field support; overlapping-reference duplicates removed',source_shape_selection_tolerance='max(0.75 native px, 1 percent of visible source-run diagonal); sharp source turns retained independently',border_selection_shape='Degree <= 4 smooth curve used only to choose actual observed representatives; observed pixels are not changed',tolerance_is_not_fit_residual_allowance=True,per_entity_frame=counts))
    print(json.dumps(dict(done=True,before=sum(x['verified_dense_points'] for x in counts),after=len(zz['pts']),seconds=round(time.monotonic()-start,1))),flush=True)

def publish(source_prefix="s01_native_source_v2_"):
    report=b.read(OUT/'sampling_report.json');z=dict(np.load(OUT/'verified_ads.npz'));a=dict(np.load(OUT/'representative_ads.npz'));ids={int(i):j for j,i in enumerate(z['source_rows'])};ii=np.array([ids[int(x)] for x in a['source_rows']])
    for k in z:
        if not np.array_equal(z[k][ii],a[k]):raise ValueError('Representative selection changed an observation')
    if not report['all_verified_fixed_corners_preserved'] or report['native_entity_frame_pairs_before']!=report['native_entity_frame_pairs_after']:raise ValueError('Sampling lost fixed corners or an observable entity')
    history=OUT/'dense_verified_pool';history.mkdir(exist_ok=True)
    for name in ['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz']:
        src=(OUT/name).resolve();dst=(history/name).resolve()
        if dst.exists() or not dst.is_relative_to(OUT.resolve()):raise ValueError('Unsafe/repeated dense-pool target')
        src.rename(dst)
    (OUT/'representative_ads.npz').rename(OUT/'verified_ads.npz');(OUT/'representative_checks.jsonl.gz').rename(OUT/'pointwise_checks.jsonl.gz')
    summary=b.read(history/'measurement_summary.json');summary['dense_verified_rows']=len(z['pts']);summary['ad_rows']=len(a['pts']);summary['by_path']=dict(collections.Counter(paths for paths in np.asarray(b.read(OUT/'verified_mapping.json')['path_ids'])[a['path_indices']]));mapping=b.read(OUT/'verified_mapping.json');paths=mapping['path_ids'];summary['by_ad']=dict(collections.Counter(paths[k].split('::')[0].split('/')[-1] for k in a['path_indices']));fresh=np.char.startswith(a['observation_ids'],source_prefix);summary['decisions']=dict(include_original_pixel=int((~fresh).sum()),new_actual_source_boundary=int(fresh.sum()))
    for f in summary['frames']:
        rr=np.flatnonzero(a['pts']==f['pts']);f['ad_points']=len(rr);f['new_source_measurements']=int(fresh[rr].sum());f['verified_original']=len(rr)-f['new_source_measurements'];f['by_ad']=dict(collections.Counter(paths[k].split('::')[0].split('/')[-1] for k in a['path_indices'][rr]))
    b.write(OUT/'measurement_summary.json',summary);count=collections.Counter((int(pt),paths[k]) for pt,k in zip(a['pts'],a['path_indices']))
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(OUT/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as dest:
        for line in f:
            c=json.loads(line);c['dense_verified_points']=c['included_points'];c['included_points']=count[(c['pts'],c['path_id'])];dest.write(json.dumps(c,ensure_ascii=False)+'\n')
    mapping.update(point_input_sha256=b.sha(OUT/'verified_ads.npz'),redundancy_reduced=True);b.write(OUT/'verified_mapping.json',mapping)
    print(json.dumps({k:v for k,v in report.items() if k!='per_entity_frame'}),flush=True)
if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:run()
