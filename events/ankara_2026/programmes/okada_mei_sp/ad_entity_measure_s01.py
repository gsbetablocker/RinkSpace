"""Per-observation source identity and all-retained-contour coverage readout.

Source charts only navigate the image. Current RGB supplies every measurement.
No current native projection, fit residual, or previous row label admits a row.
"""
from pathlib import Path
import sys,json,gzip,collections,time,hashlib
import cv2,numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_registration_s01 as r
import ad_replace_package_s01 as pack
b=r.b;OUT=r.OUT;INC=b.OUT/'increments/001'
sys.path.insert(0,str(b.ROOT/'src'))
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

def interp_path(p,spacing):
    dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if dist[-1]<.1:return None
    ts=np.arange(0,dist[-1]+.001,spacing)
    if len(ts)<2:ts=np.array([0.,dist[-1]])
    q=np.column_stack([np.interp(ts,dist,p[:,k]) for k in (0,1)])
    left=np.column_stack([np.interp(np.maximum(0,ts-.8),dist,p[:,k]) for k in (0,1)])
    right=np.column_stack([np.interp(np.minimum(dist[-1],ts+.8),dist,p[:,k]) for k in (0,1)])
    tangent=right-left;normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
    return q,normal,ts/max(dist[-1],1e-9)

def physical_color(pid,im,q):
    q=np.asarray(q);off=np.mgrid[-4:5:4,-4:5:4].reshape(2,-1).T[:,::-1]
    rgb=r.sample(im,q[:,None,:]+off);B,G,R=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    purple=((R-G>15)&(B-G>15)&(R-B>-55)).mean(1)
    blue=((B-R>22)&(B-G>9)).mean(1);red=((R-B>22)&(R-G>22)).mean(1)
    ok=np.ones(len(q),bool)
    if pid.startswith('Ad2.print.'):ok=purple>.02
    if pid.startswith('Ad4.'):
        if 'ANKARA' in pid:ok=purple>.02
        elif 'shot4_' in pid:ok=(blue>.02)&(purple<.40)
    if pid.startswith('Ad6.') and 'print.' in pid:ok=red>.02
    if pid.startswith('Ad7.'):ok=blue>.02
    if pid.startswith('Ad8.'):ok=blue>.02
    if pid.startswith('Ad9.'):ok=red>.02
    return ok

def patch_score(ref,H,im,q,n,offset=0.):
    off=np.mgrid[-6:7:3,-6:7:3].reshape(2,-1).T[:,::-1]
    target=q[:,None,:]+off
    source=cv2.perspectiveTransform(target.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(target.shape)
    a=r.sample(ref['gray'],source);c=r.sample(im,target)
    a-=a.mean(1,keepdims=True);c-=c.mean(1,keepdims=True)
    norm=np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1)
    return np.divide((a*c).sum(1),norm,out=np.zeros(len(a)),where=norm>5)

def chart_paths(ref,chart,im,gray,mask):
    H=np.asarray(chart['H'],float);inv=np.linalg.inv(H);answer=[]
    for pi,path in enumerate(ref['atlas']['paths']):
        p=cv2.perspectiveTransform(np.asarray(path['pixels'],float)[None],H)[0]
        v=interp_path(p,1.5)
        if v is None:continue
        q,n,s=v;xy=np.rint(q).astype(int);inside=(xy[:,0]>5)&(xy[:,0]<1914)&(xy[:,1]>185)&(xy[:,1]<1074)
        rr=np.flatnonzero(inside);occluded=np.ones(len(q),bool);occluded[rr]=mask[xy[rr,1],xy[rr,0]]>0
        q=q[inside];n=n[inside];s=s[inside];occluded=occluded[inside]
        if not len(q):continue
        # Orient from the immutable source pattern, never from the desired current edge.
        a=cv2.perspectiveTransform((q-n*1.25)[None],inv)[0]
        c=cv2.perspectiveTransform((q+n*1.25)[None],inv)[0]
        da=r.sample(ref['gray'],a[:,None,:]).ravel();dc=r.sample(ref['gray'],c[:,None,:]).ravel();delta=dc-da
        n[delta<0]*=-1
        photometric=patch_score(ref,H,gray,q,n)
        colors=physical_color(path['path_id'],im,q)
        polarity=np.abs(delta)>2.
        valid=(~occluded)&colors&polarity
        answer.append(dict(path=path['path_id'],atlas_id=int(ref['atlas']['id']),source_path_index=pi,source_contour_sha256=path['contour_sha256'],H=H,reference_pixels=np.asarray(path['pixels'],float),q=q,n=n,s=s,valid=valid,occluded=occluded,color_ok=colors,polarity_ok=polarity,ncc=photometric,tree=cKDTree(q)))
    return answer

def profile_reason(profile):
    flags=profile['source_flags'];q=np.asarray(profile['source_q10_q50_q90_normal_offsets_px'])
    if any(flags.values()) or not np.isfinite(q).all():return 'current_local_material_transition_unresolved'
    return None

def read_profiles(im,frame,q,n):
    binding=dict(source_sha256=frame['source_video_sha256'],source_pts=frame['pts'],time_base='1/12800',shot_id=frame.get('shot_id','s01'),event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080)
    rgb=np.ascontiguousarray(im[:,:,::-1]);q=np.asarray(q);n=np.asarray(n)
    records=read_source_material_transition_profiles(rgb,binding=binding,search_points_uv=q,normals=n,search_intervals_px=np.tile([-4.,4.],(len(q),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
    # A clipped search is not proof that a boundary is unobservable. Expand only
    # toward actual plateaus; never narrow away competing or reversed interfaces.
    radii=np.full(len(q),4.)
    for radius in [8.,16.,32.]:
        ix=np.array([i for i,p in enumerate(records) if p['source_flags']['search_interval_truncated'] and not any(p['source_flags'].get(k,False) for k in ['source_image_border_truncated','competing_material_reversal','separated_same_sign_transitions'])],int)
        if not len(ix):break
        wide=read_source_material_transition_profiles(rgb,binding=binding,search_points_uv=q[ix],normals=n[ix],search_intervals_px=np.tile([-radius,radius],(len(ix),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.25)['readings']
        for i,p in zip(ix,wide):records[int(i)]=p;radii[i]=radius
    mask=r.current(frame['pts'])[2]
    for i,p in enumerate(records):
        radius=radii[i];off=np.arange(-radius-2,radius+3);t=np.array([-n[i,1],n[i,0]])
        pixels=q[i]+off[:,None]*n[i];grid=np.concatenate([pixels-t,pixels,pixels+t]);v=r.sample(mask,grid[:,None,:]).ravel()
        p['source_flags']['source_profile_foreground_intersection']=bool(np.any(v>0))
        if np.any(v>0):p['actual_source_uv']=np.empty((0,2));p['disposition']='source_material_role_unknown'
        p['source_search_radius_px']=float(radius)
    return records

def run(*, source=None, banned=None, first_pts=283136, source_prefix="s01_native_source_v2_", query_new=True):
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True);doc=b.read(OUT/'source_identity_charts.json');ats=b.read(b.OUT/'atlases.json')['atlases'];extra_atlases=OUT/'additional_source_atlases.json';ats+=b.read(extra_atlases)['atlases'] if extra_atlases.exists() else [];refs={int(i):r.reference(ats[int(i)]) for i in doc['records'] if doc['records'][i]}
    registry=b.read(OUT/'native_entity_registry.json');paths=[p['native_id'] for p in registry['paths']];lookup={p.split('::')[1]:i for i,p in enumerate(paths)};charts=collections.defaultdict(list)
    for ai,rows in doc['records'].items():
        for pt,c in rows.items():
            if c.get('source_track_conflict_px',0)>8:continue
            charts[int(pt)].append((int(ai),c))
    if source is None:
        source=b.load();extra=dict(np.load(INC/'additions.npz'));extra_rows=np.flatnonzero(extra['source_rows']>=len(source['pts']))
        N=len(source['pts']);source=dict(source)
        source['uv']=np.concatenate([source['uv'],extra['uv'][extra_rows]])
        source['pts']=np.r_[source['pts'],extra['pts'][extra_rows]]
        source['ids']=np.r_[source['ids'],extra['observation_ids'][extra_rows]]
        source['kind']=np.r_[source['kind'],np.zeros(len(extra_rows),np.uint8)]
        source['ad']=np.r_[source['ad'],np.full(len(extra_rows),7)]
        source['labels']=np.r_[source['labels'],np.full(len(extra_rows),'increment001_unverified_entity_hint')]
        source_rows=np.r_[np.arange(N),extra['source_rows'][extra_rows]]
    else:
        source=dict(source);source_rows=np.asarray(source.get('source_rows',np.arange(len(source['pts']))),np.int64)
    if banned is None:
        index=b.read(b.OUT/'source_evidence_index.json');banned=set(map(int,index['withdrawals']))|set(map(int,index['retired']))
        banned|={c['row'] for c in b.read(b.OUT/'boundary_identity_review.json')['changes'] if c['decision']=='exclude'}
    else:banned=set(banned)
    perframe=collections.defaultdict(list)
    for k in np.flatnonzero(source['ad']>0):perframe[int(source['pts'][k])].append(int(k))
    core,model,native=pack.current_native();assets=b.assets()
    accepted=[];coverage=[];decisions=collections.Counter();start=time.monotonic();nextrow=int(source_rows.max())+1
    testpts=None
    if '--sample' in sys.argv:testpts={int(round(float(v)*12800)) for v in sys.argv[sys.argv.index('--sample')+1].split(',')}
    tag='sample_' if testpts else ''
    with gzip.open(OUT/(tag+'pointwise_checks.jsonl.gz'),'wt',encoding='utf8') as log,gzip.open(OUT/(tag+'coverage_intervals.jsonl.gz'),'wt',encoding='utf8') as coverlog:
        for fi,(pt,frame) in enumerate(sorted(r.VIS.items())):
            if pt<first_pts:continue
            if testpts and pt not in testpts:continue
            im,gray,mask=r.current(pt);mask=cv2.dilate(mask,np.ones((3,3),np.uint8));variants=[];by_path=collections.defaultdict(list)
            for ai,chart in charts[pt]:
                for v in chart_paths(refs[ai],chart,im,gray,mask):
                    if v['path'] not in lookup:raise ValueError('Unretained native entity')
                    by_path[v['path']].append(len(variants));variants.append(v)
            rows=np.array(perframe[pt],int);rawq=source['uv'][rows];candidates=[[] for _ in rows]
            for vi,v in enumerate(variants):
                if not len(rows):continue
                d,ix=v['tree'].query(rawq)
                use=(d<4.0)&v['valid'][ix]
                for j in np.flatnonzero(use):
                    k=int(ix[j]);candidates[j].append((float(d[j]),float(v['ncc'][k]),vi,k))
            selected=[];oldreasons={}
            for j,k in enumerate(rows):
                k=int(k);sr=int(source_rows[k])
                if sr in banned:oldreasons[j]='explicit_source_withdrawal_or_retired_entity';continue
                x,y=np.rint(rawq[j]).astype(int)
                if not (5<x<1914 and 185<y<1074) or mask[y,x]:oldreasons[j]='actual_foreground_occlusion_or_image_boundary';continue
                options=candidates[j]
                if not options:oldreasons[j]='current_source_entity_not_resolved';continue
                if source['kind'][k]==1:
                    options=[op for op in options if not refs[variants[op[2]]['atlas_id']]['atlas'].get('partial_source_reference')]
                    if not options:oldreasons[j]='partial_source_arc_cannot_define_material_corner';continue
                # Collapse photographs of the same entity before testing distinct identities.
                best={}
                for op in options:
                    d,ncc,vi,qi=op;v=variants[vi];score=d-.65*ncc
                    if v['path'] not in best or score<best[v['path']][0]:best[v['path']]=(score,op)
                opts=sorted(best.values(),key=lambda x:x[0]);_,chosen=opts[0];d,ncc,vi,qi=chosen;v=variants[vi]
                if len(opts)>1:
                    other=opts[1][1];ov=variants[other[2]]
                    if other[0]<d+.75 and np.dot(v['n'][qi],ov['n'][other[3]])>.5:oldreasons[j]='distinct_source_entities_remain_ambiguous';continue
                if ncc<.05:oldreasons[j]='current_entity_neighborhood_not_supported';continue
                station=.5;kind=1;target=v['path'];corner=None
                if source['kind'][k]==1:
                    if refs[v['atlas_id']]['atlas'].get('partial_source_reference'):oldreasons[j]='partial_source_arc_cannot_define_material_corner';continue
                    bound=pack.corner_binding(k,dict(path_id=target),source,assets,core,model)
                    if bound is None:oldreasons[j]='named_native_material_corner_not_resolved';continue
                    target,station,corner=bound;kind=0
                    rp=v['reference_pixels'];name=corner['corner']
                    xside='left' if 'left' in name.split('_') else 'right'
                    yside='top' if name.startswith('top_') or name.endswith('_top') else 'bottom'
                    extremum=rp[:,0].min() if xside=='left' else rp[:,0].max()
                    choices=np.flatnonzero(np.abs(rp[:,0]-extremum)<=1.)
                    ci=int(choices[np.argmin(rp[choices,1]) if yside=='top' else np.argmax(rp[choices,1])])
                    hh,ww=refs[v['atlas_id']]['gray'].shape
                    if not (5<rp[ci,0]<ww-5 and 5<rp[ci,1]<hh-5):oldreasons[j]='reference_image_clips_named_material_corner';continue
                    endpoint=cv2.perspectiveTransform(rp[ci:ci+1][None],v['H'])[0,0]
                    corner_distance=float(np.linalg.norm(endpoint-source['uv'][k]))
                    if corner_distance>4.:oldreasons[j]='source_pixel_is_not_the_named_material_corner';continue
                    corner.update(actual_source_reference_corner_uv=endpoint.tolist(),source_corner_distance_px=corner_distance)
                selected.append(dict(row=k,source_row=sr,local_row=j,v=vi,qi=qi,path=target,station=station,kind=kind,corner=corner,distance=d,ncc=ncc))
            readings=read_profiles(im,frame,np.array([source['uv'][s['row']] for s in selected]),np.array([variants[s['v']]['n'][s['qi']] for s in selected])) if selected else []
            frame_added=[];near=collections.defaultdict(list)
            for s,p in zip(selected,readings):
                reason=profile_reason(p);offset=np.asarray(p['source_q10_q50_q90_normal_offsets_px'])
                if reason is None and not offset[0]-1.<=0<=offset[2]+1.:reason='original_pixel_outside_actual_transition_support'
                if reason:oldreasons[s['local_row']]=reason;continue
                k=s['row'];v=variants[s['v']];rec=dict(source_row=s['source_row'],stable_id=str(source['ids'][k]),pts=pt,uv=source['uv'][k].tolist(),path_id=paths[lookup[s['path']]],kind=s['kind'],station=s['station'],decision='include_original_pixel',identity=dict(atlas_id=v['atlas_id'],source_path_index=v['source_path_index'],source_contour_sha256=v['source_contour_sha256'],source_chart_pts=pt,source_contour_distance_px=s['distance'],current_neighborhood_correlation=s['ncc'],unique_native_entity=True),source_profile=dict(offsets=offset.tolist(),normal=variants[s['v']]['n'][s['qi']].tolist(),actual_readout_uv=np.asarray(p['actual_source_uv']).tolist(),search_radius_px=p['source_search_radius_px'],flags=p['source_flags']),source_BGR_sha256=frame['source_BGR_sha256'],original_UV_PTS_ID_unchanged=True,native_geometry_used_for_identity=False,corner=s['corner'])
                accepted.append(rec);frame_added.append(rec);near[s['path']].append(rec['uv']);log.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');decisions['include_original_pixel']+=1
            for j,reason in oldreasons.items():
                k=int(rows[j]);rec=dict(source_row=int(source_rows[k]),stable_id=str(source['ids'][k]),pts=pt,uv=source['uv'][k].tolist(),decision='not_admitted',reason=reason,source_BGR_sha256=frame['source_BGR_sha256'])
                log.write(json.dumps(rec,ensure_ascii=False)+'\n');decisions[reason]+=1
            # Search every visible modeled contour, including parts with no old observations.
            queries=[];normals=[];which=[];covercounts=collections.defaultdict(collections.Counter)
            for name,vis in (by_path.items() if query_new else []):
                # All source references remain candidates; duplicates are removed only after current measurement.
                for vi in vis:
                    v=variants[vi]
                    for qi in range(1,len(v['q'])-1,4):
                        uv=v['q'][qi];count=covercounts[name];count['source_navigation_samples']+=1
                        if v['occluded'][qi]:count['foreground_occluded']+=1;continue
                        if not v['color_ok'][qi]:count['source_material_color_conflict']+=1;continue
                        if not v['polarity_ok'][qi]:count['reference_material_side_unresolved']+=1;continue
                        if v['ncc'][qi]<.05:count['local_source_pattern_not_supported']+=1;continue
                        if near[name] and np.min(np.linalg.norm(np.asarray(near[name])-uv,axis=1))<3.:count['existing_verified_point_nearby']+=1;continue
                        queries.append(uv);normals.append(v['n'][qi]);which.append((vi,qi))
            # Collapse duplicate photographic navigation samples of the same
            # physical contour before rereading RGB; keep the best source context.
            order=sorted(range(len(queries)),key=lambda j:-variants[which[j][0]]['ncc'][which[j][1]])
            grids=collections.defaultdict(dict);keep=[]
            for j in order:
                vi,qi=which[j];name=variants[vi]['path'];q=queries[j];cell=tuple(np.floor(q/2.8).astype(int));grid=grids[name];duplicate=False
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        for other in grid.get((cell[0]+dx,cell[1]+dy),[]):
                            if np.linalg.norm(q-queries[other])<2.8 and np.dot(normals[j],normals[other])>.7:duplicate=True;break
                        if duplicate:break
                    if duplicate:break
                if duplicate:covercounts[name]['duplicate_source_reference_navigation']+=1;continue
                grid.setdefault(cell,[]).append(j);keep.append(j)
            queries=[queries[j] for j in keep];normals=[normals[j] for j in keep];which=[which[j] for j in keep]
            profiles=read_profiles(im,frame,np.array(queries),np.array(normals)) if queries else []
            # Batch all competing entity checks. Multiple photographs of one
            # contour do not count as separate physical identities.
            ambiguous_flags=np.zeros(len(queries),bool);usable=np.array([j for j,p in enumerate(profiles) if profile_reason(p) is None],int)
            if len(usable):
                uq=np.array([profiles[j]['actual_source_uv'] for j in usable]);un=np.array([normals[j] for j in usable]);own=np.array([variants[which[j][0]]['path'] for j in usable]);limit=np.minimum(2.,np.linalg.norm(uq-np.array(queries)[usable],axis=1)+.5)
                for other,ivs in by_path.items():
                    pp=[];nn=[]
                    for oi in ivs:
                        ov=variants[oi];valid=ov['valid']&(ov['ncc']>.05);pp.extend(ov['q'][valid]);nn.extend(ov['n'][valid])
                    if not pp:continue
                    pp=np.asarray(pp);nn=np.asarray(nn);tree=cKDTree(pp);dd,kk=tree.query(uq,k=min(4,len(pp)))
                    if dd.ndim==1:dd=dd[:,None];kk=kk[:,None]
                    bad=(own!=other)&np.any((dd<limit[:,None])&(np.einsum('ijk,ik->ij',nn[kk],un)>.6),axis=1)
                    ambiguous_flags[usable[bad]]=True
            for j,(p,(vi,qi)) in enumerate(zip(profiles,which)):
                v=variants[vi];name=v['path'];count=covercounts[name];reason=profile_reason(p)
                if reason:count[reason]+=1;continue
                uv=np.asarray(p['actual_source_uv']);x,y=np.rint(uv).astype(int)
                if mask[y,x]:count['measured_point_foreground_occluded']+=1;continue
                if near[name] and np.min(np.linalg.norm(np.asarray(near[name])-uv,axis=1))<2.8:count['verified_point_already_covers_contour']+=1;continue
                # An unresolved neighbor with the same paint-side direction cannot be silently selected.
                if ambiguous_flags[j]:count['distinct_source_entity_ambiguity']+=1;continue
                sid=f'{source_prefix}{pt}_{lookup[name]:03d}_{len(near[name]):04d}'
                rec=dict(source_row=nextrow,stable_id=sid,pts=pt,uv=uv.tolist(),path_id=paths[lookup[name]],kind=1,station=.5,decision='new_actual_source_boundary',identity=dict(atlas_id=v['atlas_id'],source_path_index=v['source_path_index'],source_contour_sha256=v['source_contour_sha256'],source_chart_pts=pt,reference_arc_fraction=float(v['s'][qi]),current_neighborhood_correlation=float(v['ncc'][qi]),unique_native_entity=True),source_profile=dict(search_uv=queries[j].tolist(),normal=normals[j].tolist(),offsets=np.asarray(p['source_q10_q50_q90_normal_offsets_px']).tolist(),search_radius_px=p['source_search_radius_px'],flags=p['source_flags']),source_BGR_sha256=frame['source_BGR_sha256'],source_navigation_not_observed_UV=True,native_geometry_used_for_identity=False)
                nextrow+=1;accepted.append(rec);frame_added.append(rec);near[name].append(rec['uv']);log.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');count['new_actual_source_boundary']+=1;decisions['new_actual_source_boundary']+=1
            for pid in lookup:
                rec=dict(pts=pt,path_id=paths[lookup[pid]],source_identity_chart_available=pid in by_path,counts=dict(covercounts[pid]),included_points=len(near[pid]),missing_chart_is_not_a_blur_verdict=True)
                coverlog.write(json.dumps(rec,ensure_ascii=False)+'\n')
            coverage.append(dict(pts=pt,source_seconds=pt/12800,ad_points=len(frame_added),verified_original=sum(c['decision']=='include_original_pixel' for c in frame_added),new_source_measurements=sum(c['decision']=='new_actual_source_boundary' for c in frame_added),by_ad=dict(collections.Counter(c['path_id'].split('::')[0].split('/')[-1] for c in frame_added)),charted_entities=len(by_path)))
            if fi%30==0 or testpts:print(json.dumps(dict(frame=fi+1,pts=pt,ad_points=len(frame_added),accepted=len(accepted),decisions=dict(decisions),seconds=round(time.monotonic()-start,1))),flush=True)
    z=dict(source_rows=np.array([c['source_row'] for c in accepted],np.int64),observation_ids=np.array([c['stable_id'] for c in accepted]),uv=np.array([c['uv'] for c in accepted]).reshape(-1,2),pts=np.array([c['pts'] for c in accepted],np.int64),kind=np.array([c['kind'] for c in accepted],np.uint8),stations=np.array([c['station'] for c in accepted]),path_indices=np.array([paths.index(c['path_id']) for c in accepted],np.int32))
    np.savez_compressed(OUT/(tag+'verified_ads.npz'),**z)
    b.write(OUT/(tag+'verified_mapping.json'),dict(path_ids=paths,scene_revision=registry['scene_revision'],scene_sha256=registry['scene_sha256'],identity_charts_sha256=b.sha(OUT/'source_identity_charts.json'),source_video_sha256=next(iter(r.VIS.values()))['source_video_sha256'],source_identity_uses_unfitted_native_geometry=False))
    b.write(OUT/(tag+'measurement_summary.json'),dict(decisions=dict(decisions),ad_rows=len(accepted),source_frames=len(coverage),frames=coverage,by_ad=dict(collections.Counter(c['path_id'].split('::')[0].split('/')[-1] for c in accepted)),by_path=dict(collections.Counter(c['path_id'] for c in accepted)),native_parameters_unchanged=True,old_rows_not_overwritten=True,human_visual_acceptance=False))
    print(json.dumps(dict(done=True,rows=len(accepted),decisions=dict(decisions),frames=len(coverage),seconds=round(time.monotonic()-start,1))),flush=True)
if __name__=='__main__':run()
