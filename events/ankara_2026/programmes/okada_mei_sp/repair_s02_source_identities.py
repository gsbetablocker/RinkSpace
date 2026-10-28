"""Incremental S02 source-identity repair, preserving the delivered baseline.

Existing readers perform new measurements. Review exclusions, source domains,
and source-motion guides never become observed UV. Native geometry is read-only.
"""
from pathlib import Path
import argparse,collections,copy,functools,gzip,hashlib,json,time,shutil,zipfile
import cv2,numpy as np
import repair_points_s02 as prior
from calibration.program_tracking.audit_first_shot_circle_material_identity import centreline_house_conflicts

BASE=prior.old.OUT
BEFORE=prior.OUT
OUT=BEFORE/'identity_repair'
EXPECTED='5eef9ad919afbd4c169c2392e40bf7ecbbe447126a0e45ab68ece832352cbf05'
read=prior.read
write=prior.fit.write


@functools.lru_cache(maxsize=1)
def inputs():
    if hashlib.sha256((BEFORE/'points/point_observations.npz').read_bytes()).hexdigest()!=EXPECTED:
        raise ValueError('Repair baseline changed')
    data=list(prior.rows(BEFORE/'points/point_records.jsonl.gz'))
    by=collections.defaultdict(list)
    for row in data:by[row['pts']].append(row)
    entities={e['native_path_id']:e for e in read(BASE/'evidence/native_entities.json')['entities']}
    vis={v['pts']:v for v in read(BASE/'evidence/visibility.json')['frames']}
    return data,by,entities,vis


def reviewed_fill_mask(features,reviewed):
    """Source-painted regions explicitly inspected in the previous audit."""
    mask=np.zeros((1080,1920),np.uint8)
    for house,band in sorted({(r['house'],r['band'])for r in reviewed}):
        outer,inner=band.split('-');aa=features.get(f'ice.{house}:{outer}',{}).get('rows',[]);bb=features.get(f'ice.{house}:{inner}',{}).get('rows',[])
        if min(len(aa),len(bb))<5:continue
        a=cv2.convexHull(np.array([r['uv']for r in aa],np.float32));b=cv2.convexHull(np.array([r['uv']for r in bb],np.float32))
        patch=np.zeros_like(mask);cv2.fillPoly(patch,[np.rint(a).astype(np.int32)],1);cv2.fillPoly(patch,[np.rint(b).astype(np.int32)],0);mask|=patch
    return mask.astype(bool)


def ice(requested=None):
    from ice_points_s04 import source_domain
    from reread_s04_lines import NativeLineReader
    cv2.setNumThreads(2);_,by,entities,vis=inputs();folder=OUT/'ice';folder.mkdir(parents=True,exist_ok=True)
    # All 78 candidate groups were inspected, including the remaining R2/A1
    # and late R3/R4 crops. This is an assistant review, not user acceptance.
    reviewed=read(BEFORE/'peer_review/ring_band_candidates.json');reviewed_by=collections.defaultdict(list)
    for r in reviewed:reviewed_by[r['pts']].append(r)
    grid=prior.GuideGrid();photo=prior.SourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=BASE/'evidence/source_navigation.json',source_prefix='s02_identity',source_cross_line_guides=True)
    for p in reader.paths:p.update(_source_query_spacing_px=35,_separate_profile_support=True)
    a=photo.annotation
    reader.identity.guides.append(dict(kind='line',frame=a['source_pts'],curve_id=a['feature_id'].removeprefix('ice.'),uv=a['approximate_source_search_uv'],authority='user_identity_only_source_search'))
    summary=[]
    for j,pt in enumerate(sorted(vis)if requested is None else requested):
        source=read(BEFORE/f'ice/{pt}.json');features=source['features'];bad,mask,patches=centreline_house_conflicts(features,entities)
        reviewed_mask=reviewed_fill_mask(features,reviewed_by[pt]);mask|=reviewed_mask
        remove={r['source_row_id']:r for r in bad}
        for row in reviewed_by[pt]:
            remove[row['id']]=dict(source_row_id=row['id'],target_id=row['label'],uv=row['uv'],reason='source_visual_review_confirms_house_fill_not_claimed_centreline',review=str(BEFORE/'peer_review/ring_candidate_index.json'))
        # Apply the inspected material region to dense support as well, so
        # review lines cannot silently connect across the excluded ring band.
        for name,f in features.items():
            if entities[f['native_path_id']].get('measurement')!='centreline':continue
            for row in f['rows']:
                x,y=np.rint(row['uv']).astype(int)
                if 0<=x<1920 and 0<=y<1080 and reviewed_mask[y,x]:
                    remove[row['source_row_id']]=dict(source_row_id=row['source_row_id'],target_id=name,uv=row['uv'],reason='same_visually_reviewed_current_house_fill_region')
        affected={r['target_id']for r in remove.values()};features={name:dict(f,rows=[r for r in f['rows']if r['source_row_id']not in remove])for name,f in features.items()}
        recovered={};diagnostic={}
        if affected:
            v=vis[pt];im=cv2.imread(v['image']);person=cv2.imread(v['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Source exposure changed')
            graphics,border,_,_=source_domain(im,person,read(prior.old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            candidates,diagnostic=reader.read(pt,im,(person>0)|graphics|(np.arange(1080)[:,None]<=border[None])|mask,v['source_video_sha256'])
            for name in affected:
                if name not in candidates:continue
                candidate=candidates[name];uv=np.array([r['uv']for r in candidate['rows']])
                direction=photo.direction_family(name,pt,uv)
                if direction is not None and direction!=photo.axes[name]:continue
                f=features.get(name);existing=f['rows']if f else[];new=[]
                for row in candidate['rows']:
                    q=np.array(row['uv']);x,y=np.rint(q).astype(int)
                    if mask[y,x]or any(np.linalg.norm(q-r['uv'])<25 for r in existing+new):continue
                    new.append(row)
                if new:
                    if f is None:features[name]=dict(candidate,rows=[])
                    features[name]['rows'].extend(new);recovered[name]=new
        write(folder/f'{pt}.json',dict(pts=pt,source_BGR_sha256=source['source_BGR_sha256'],features=features,
            excluded_source_rows=list(remove.values()),recovered=recovered,
            current_source_material_patches=patches,reviewed_material_scopes=reviewed_by[pt],
            reread_diagnostics=diagnostic,native_authority=source.get('native_authority'),
            original_data_unchanged=True,human_acceptance=False))
        summary.append(dict(pts=pt,rejected_dense_rows=len(remove),affected=sorted(affected),
            rejected_formal_rows=sum(r['observation_id']in remove for r in by[pt]),new_dense_rows=sum(map(len,recovered.values()))))
        if requested is not None or j%50==0 or j==len(vis)-1:print(json.dumps(dict(stage='ice_identity',frame=j+1,**summary[-1])),flush=True)
    write(OUT/'ice_repair.json',dict(input_sha256=EXPECTED,frames=len(summary),records=summary,original_native_unchanged=True));grid.unchanged()


def glyphs(requested=None):
    from ad_entity_measure_s01 import current_glyph_material_pair
    from scipy.spatial import cKDTree
    cv2.setNumThreads(2);_,by,entities,vis=inputs();folder=OUT/'glyphs';folder.mkdir(parents=True,exist_ok=True)
    r,t,m,o,s,atlases,_=prior.old.ad_configure();atlases+=read(BASE/'ads/source_review/additional_source_atlases.json')['atlases']
    charts=read(BASE/'ads/source_review/source_identity_charts.json')['records'];summary=[]
    for j,pt in enumerate(sorted(vis)if requested is None else requested):
        im,gray,mask=r.current(pt);groups=collections.defaultdict(list)
        for row in by[pt]:
            if '::'in row['native_path_id']and not any(k in row['native_path_id']for k in ['_edge','_vertical']):groups[row['native_path_id']].append(row)
        decisions=[];replacements=[];contexts=[];context_by={};trees={};not_applicable=[]
        for pid,rr in groups.items():
            if len(rr)<2:continue
            name=pid.split('::')[1];choices=[]
            ais={v.get('source_record',{}).get('current_observability',{}).get('identity_reference_atlas')for v in rr}
            ais|={v.get('review_order',{}).get('atlas')for v in rr}
            for ai in ais-{None}:
                chart=charts.get(str(ai),{}).get(str(pt))
                if chart is None:continue
                parts=[p for p in atlases[int(ai)]['paths']if p['path_id']==name]
                if not parts:continue
                q=cv2.perspectiveTransform(np.vstack([p['pixels']for p in parts]).astype(float)[None],np.asarray(chart['H'],float))[0]
                choices.append(q)
            context=np.vstack(choices)if choices else np.array([v['uv']for v in rr])
            context_by[pid]=context
            dense=[m.interp_path(q,1.5)for q in choices]
            dense=[v[0]for v in dense if v is not None]
            if dense:trees[pid]=cKDTree(np.vstack(dense))
            contexts.append(dict(path=pid,actual_source_reference_bounds=[context.min(0).tolist(),context.max(0).tolist()]))
        # A neighboring material remains a competitor even when it has no
        # selected formal point in this exposure. Use the whole photographic
        # chart, not only the subset already admitted to the input.
        for instance in {p.split('::')[0]for p in groups}:
            ais={v.get('source_record',{}).get('current_observability',{}).get('identity_reference_atlas')for pid,rr in groups.items()if pid.split('::')[0]==instance for v in rr}
            ais|={v.get('review_order',{}).get('atlas')for pid,rr in groups.items()if pid.split('::')[0]==instance for v in rr}
            expanded=collections.defaultdict(list)
            for ai in ais-{None}:
                chart=charts.get(str(ai),{}).get(str(pt))
                if chart is None:continue
                for path in atlases[int(ai)]['paths']:
                    pid=instance+'::'+path['path_id']
                    q=cv2.perspectiveTransform(np.asarray(path['pixels'],float)[None],np.asarray(chart['H'],float))[0]
                    dense=m.interp_path(q,1.5)
                    if dense is not None:expanded[pid].append(dense[0])
            for pid,pieces in expanded.items():trees[pid]=cKDTree(np.vstack(pieces))
        for pid,rr in groups.items():
            if pid not in context_by:continue
            if not str(entities[pid].get('role','')).startswith('letter_'):
                not_applicable.append(dict(path=pid,role=entities[pid].get('role'),rows=len(rr),
                    reason='native_logo_or_multimaterial_outline_must_not_use_a_single_glyph_material_split',all_prior_observations_preserved=True))
                continue
            context=context_by[pid]
            for row in rr:
                if row['human_acceptance']or row['kind']!=1:continue
                sp=row.get('source_record',{}).get('source_profile')
                if not sp:continue
                q=np.asarray(sp.get('search_uv',row.get('original_uv',row['uv'])),float);n=np.asarray(sp['normal'],float)
                check=current_glyph_material_pair(gray,mask,context,q,n,sp['offsets'])
                if check['resolved']:continue
                decision=dict(observation_id=row['observation_id'],source_row=row['source_row'],pts=pt,uv=row['uv'],path_id=pid,
                    original_group=row['source_group_id'],before_check=check,action='unresolved_material_identity',original_source_preserved=True)
                # A misplaced narrow query must not turn a recoverable diffuse
                # edge into a deletion. Locate this glyph's class transition,
                # then reuse the existing full RGB material-profile reader.
                if 'current_material_class_boundary'in check:
                    offsets=np.arange(-32.,32.01,.5);xy=q+offsets[:,None]*n
                    values=r.sample(gray,xy[None])[0];smooth=cv2.GaussianBlur(values.reshape(1,-1).astype('float32'),(0,0),.7).ravel()
                    level=check['current_material_class_boundary'];cross=np.flatnonzero((smooth[:-1]<level)&(smooth[1:]>=level))
                    if len(cross):
                        k=int(cross[np.argmin(abs(offsets[cross]))]);f=(level-smooth[k])/max(float(smooth[k+1]-smooth[k]),1e-9);query=q+(offsets[k]+.5*f)*n
                        profile=m.read_profiles(im,vis[pt],query[None],n[None])[0]
                        if m.profile_reason(profile)is None:
                            off=profile['source_q10_q50_q90_normal_offsets_px'];after=current_glyph_material_pair(gray,mask,context,query,n,off)
                            decision['reread_check']=after
                            if after['resolved']:
                                uv=np.asarray(profile['actual_source_uv'])
                                if pid in trees:
                                    distance=float(trees[pid].query(uv)[0])
                                    competitors=[name for name,tree in trees.items()if name!=pid and name.split('::')[0]==pid.split('::')[0]and float(tree.query(uv)[0])+2<distance]
                                    if competitors:
                                        decision['reread_competing_entities']=competitors
                                        decision['reread_check']=dict(after,resolved=False,reason='new_material_transition_is_closer_to_another_named_source_contour')
                                        decisions.append(decision);continue
                                sid=row['observation_id']+'@material_pair_RGB'
                                replacement=dict(row,observation_id=sid,source_id=sid,uv=uv.tolist(),origin='identity_repaired_current_RGB_glyph',
                                    original_observation_id=row['observation_id'],original_uv=row['uv'],human_acceptance=False,
                                    source_record=dict(uv=uv.tolist(),source_evidence=dict(query_uv=query.tolist(),profile_normal=n.tolist(),
                                        source_q10_q50_q90_normal_offsets_px=np.asarray(off).tolist(),source_flags=profile['source_flags'],
                                        measurement_origin='existing_current_RGB_material_transition_reader',glyph_material_pair=after),
                                        original_source_record=row['source_record']))
                                replacement.pop('source_ref',None);replacements.append(replacement)
                                decision.update(action='current_RGB_replacement',replacement_observation_id=sid,replacement_uv=uv.tolist())
                decisions.append(decision)
        previous=read(folder/f'{pt}.json')if(folder/f'{pt}.json').exists()else{}
        discarded_candidates=[v for v in previous.get('decisions',[])if not str(entities[v['path_id']].get('role','')).startswith('letter_')]
        packet=dict(pts=pt,source_BGR_sha256=vis[pt]['source_BGR_sha256'],decisions=decisions,replacements=replacements,contexts=contexts,
            native_role_scope=not_applicable,not_applied_preliminary_multimaterial_candidates=discarded_candidates,
            no_whole_frame_or_glyph_rejection=True,human_acceptance_unchanged=True)
        write(folder/f'{pt}.json',packet);summary.append(dict(pts=pt,candidates=len(decisions),recovered=len(replacements),unresolved=len(decisions)-len(replacements)))
        if requested is not None or j%50==0 or j==len(vis)-1:print(json.dumps(dict(stage='glyph_identity',frame=j+1,**summary[-1])),flush=True)
    write(OUT/'glyph_repair.json',dict(input_sha256=EXPECTED,frames=len(summary),records=summary))


def border_failures(group,feature,cap,image,contact=()):
    """Diagnose old pixels using current photographed surfaces and materials."""
    failures={};raw=feature.get('rows',[])
    for row in group:
        if row['human_acceptance']:continue
        uv=np.asarray(row['uv']);why=None
        if len(cap)>=2 and uv[1]<=np.interp(uv[0],cap[:,0],cap[:,1],left=np.nan,right=np.nan)+2:
            why='ad_pixel_on_actual_blue_board_cap_not_ad_material'
        if len(contact)>=2 and uv[1]>=np.interp(uv[0],contact[:,0],contact[:,1],left=np.nan,right=np.nan)-2:
            why='ad_pixel_on_actual_yellow_foot_ice_contact_not_ad_material'
        if why is None and feature.get('native_material_role')=='logo_outer_boundary'and raw:
            near=min(raw,key=lambda r:np.linalg.norm(uv-r['uv']));n=np.asarray(near['source_evidence']['profile_normal'])
            delta=uv-near['uv'];tangent=abs(delta@np.array([-n[1],n[0]]))
            if tangent<80 and abs(delta@n)>3:
                oldxy=np.rint(uv-6*n).astype(int);newxy=np.rint(np.asarray(near['uv'])-6*n).astype(int)
                if (oldxy>=[0,0]).all()and(oldxy<[1920,1080]).all()and(newxy>=[0,0]).all()and(newxy<[1920,1080]).all():
                    a=image[oldxy[1],oldxy[0]].astype(float);b=image[newxy[1],newxy[0]].astype(float)
                    ac=a-a.mean();bc=b-b.mean()
                    if min(np.ptp(a),np.ptp(b))>20 and ac@bc/(np.linalg.norm(ac)*np.linalg.norm(bc))<.7:
                        why='old_pixel_reads_different_colored_material_from_current_native_logo_outline'
        if why:failures[row['observation_id']]=dict(observation_id=row['observation_id'],pts=row['pts'],path_id=row['native_path_id'],uv=row['uv'],reason=why,original_source_preserved=True)
    return failures


def assemble(requested=None):
    """Apply only diagnosed changes to the immutable delivered input."""
    from ice_points_s04 import select_representatives
    import package_s02_repaired as package
    cv2.setNumThreads(2);original,by,entities,vis=inputs();allrows=[];coverage=[];counts=collections.Counter()
    for name in ['points','support','material_domains']:(OUT/name).mkdir(exist_ok=True)
    shutil.copyfile(BEFORE/'navigation_camera.npz',OUT/'navigation_camera.npz')
    with gzip.open(OUT/'dispositions.jsonl.gz','wt',encoding='utf8',compresslevel=3)as ledger:
        for number,pt in enumerate(sorted(by)):
            ice=read(OUT/f'ice/{pt}.json');ads=read(OUT/f'nonice/{pt}.json');glyph=read(OUT/f'glyphs/{pt}.json')
            supports=read(BEFORE/f'support/{pt}.json');old_by_path=collections.defaultdict(list)
            for row in by[pt]:old_by_path[row['native_path_id']].append(row)
            bad={r['source_row_id']:dict(r,category='ice_material_identity')for r in ice['excluded_source_rows']}
            bad.update({r['observation_id']:dict(r,category='glyph_material_identity')for r in glyph['decisions']})
            cap=np.asarray(ads.get('board_cap_domain',{}).get('lower_blue_cap_uv',[]),float)
            if len(cap):cap=cap[np.argsort(cap[:,0])]
            contact=np.asarray([r['uv']for r in supports.get('board/continuous/bottom',{}).get('rows',[])],float)
            if len(contact):contact=contact[np.argsort(contact[:,0])]
            image=cv2.imread(vis[pt]['image']);affected_borders={}
            for pid,feature in ads['borders'].items():
                if len(contact)>=2:feature=dict(feature,rows=[r for r in feature['rows']if r['uv'][1]<np.interp(r['uv'][0],contact[:,0],contact[:,1],left=np.nan,right=np.nan)-2])
                failures=border_failures(old_by_path[pid],feature,cap,image,contact)
                if failures:
                    bad.update({k:dict(v,category='ad_surface_identity')for k,v in failures.items()});affected_borders[pid]=feature
            frame=[dict(r)for r in by[pt]if r['observation_id']not in bad or r['human_acceptance']]
            formal={r['observation_id']for r in by[pt]}
            for sid,r in bad.items():
                if sid not in formal:continue
                parent=next(v for v in by[pt]if v['observation_id']==sid)
                if parent['human_acceptance']:raise ValueError('A machine disposition attempted to modify human-accepted data')
                package.dump(ledger,dict(r,observation_id=sid,original_uv=parent['uv'],original_source_group=parent['source_group_id'],original_source_ref=parent.get('source_ref'),baseline=str(BEFORE/'points/point_records.jsonl.gz')));counts[r['category']+'_old']+=1
            ice_ref=prior.fit.file_ref(OUT/f'ice/{pt}.json');ad_ref=prior.fit.file_ref(OUT/f'nonice/{pt}.json');glyph_ref=prior.fit.file_ref(OUT/f'glyphs/{pt}.json')
            material=None
            affected_ice={r['target_id']for r in ice['excluded_source_rows']}
            if affected_ice:
                source=read(BEFORE/f'ice/{pt}.json')['features'];_,material,_=centreline_house_conflicts(source,entities)
                material|=reviewed_fill_mask(source,ice['reviewed_material_scopes']);mask_path=OUT/f'material_domains/{pt}.png';cv2.imwrite(str(mask_path),material.astype('uint8')*255)
            for name in affected_ice|set(ice['recovered']):
                if name not in ice['features']:continue
                f=ice['features'][name];pid=f['native_path_id'];path=entities[pid]
                support=dict(rows=f['rows'],kind=path['kind'],closed=supports.get(pid,{}).get('closed',False),source=str(OUT/f'ice/{pt}.json'))
                if material is not None and path.get('measurement')=='centreline':support['source_material_exclusion_mask']=str(mask_path)
                supports[pid]=support
                raw=ice['recovered'].get(name,[])
                if not raw:continue
                rr=[dict(r,pixel_support=dict(normal=r['source_evidence']['profile_normal'],query_index=r['source_evidence'].get('query_index',i)))for i,r in enumerate(raw)]
                keep,why=select_representatives(rr,path)
                for i in keep:
                    row=rr[i];uv=np.array(row['uv'])
                    if any(r['native_path_id']==pid and np.linalg.norm(uv-r['uv'])<25 for r in frame):continue
                    sid=row['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=row['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='identity_repaired_current_RGB_ice',human_acceptance=False,source_record=row,source_ref=ice_ref,source_BGR_sha256=vis[pt]['source_BGR_sha256'],
                        source_group_id=f's02/{pt}/{pid}/same_material_profile_family',selection_reasons=sorted(why[i]),review_order=dict(family='ice',run=pid,order=float(row.get('theta',row['source_evidence'].get('query_index',i))))));counts['ice_new']+=1
            for pid,f in affected_borders.items():
                supports[pid]=dict(rows=f['rows'],kind='panel_border',closed=False,source=str(OUT/f'nonice/{pt}.json'))
                if not f['rows']:continue
                keep,why=select_representatives(f['rows'],dict(kind='ice_segment'))
                for i in keep:
                    row=f['rows'][i];uv=np.array(row['uv'])
                    if any(r['native_path_id']==pid and np.linalg.norm(uv-r['uv'])<25 for r in frame):continue
                    sid=row['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=row['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='identity_repaired_current_RGB_ad_border',human_acceptance=False,source_record=row,source_ref=ad_ref,source_BGR_sha256=vis[pt]['source_BGR_sha256'],
                        source_group_id=f's02/{pt}/{pid}/panel_material',selection_reasons=sorted(why[i]),review_order=dict(family='ad_source_edge',run=pid,order=row['order'])));counts['ad_border_new']+=1
            for row in glyph['replacements']:
                if any(r['native_path_id']==row['native_path_id']and np.linalg.norm(np.array(row['uv'])-r['uv'])<2.5 for r in frame):
                    package.dump(ledger,dict(observation_id=row['observation_id'],pts=pt,uv=row['uv'],reason='new_actual_support_redundant_with_retained_same_material_point',source_ref=glyph_ref));counts['glyph_recovered_support_only']+=1;continue
                frame.append(dict(row,source_ref=glyph_ref));counts['glyph_new']+=1
            # Preserve every untouched observation exactly, including the source
            # group. New observations retain their own current RGB provenance.
            write(OUT/f'support/{pt}.json',supports);allrows.extend(frame)
            coverage.append(dict(pts=pt,source_seconds=pt/12800,rows=len(frame),counts=dict(collections.Counter(entities[r['native_path_id']]['layer']for r in frame)),paths=sorted({r['native_path_id']for r in frame})))
            if number%100==0:print(json.dumps(dict(stage='incremental_assemble',frame=number+1,rows=len(allrows))),flush=True)
    allrows.sort(key=lambda r:(r['pts'],r['native_path_id'],r['observation_id']));ids=[r['observation_id']for r in allrows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate observation ID')
    paths=sorted({r['native_path_id']for r in allrows});lookup={p:i for i,p in enumerate(paths)}
    _,_,_,_,_,native=prior.fit.load()
    if not set(paths)<=native.paths.keys():raise ValueError('Non-native geometry entered input')
    z=dict(source_rows=np.arange(len(ids),dtype=np.int64),observation_ids=np.asarray(ids),uv=np.asarray([r['uv']for r in allrows]),pts=np.asarray([r['pts']for r in allrows],np.int64),kind=np.asarray([r['kind']for r in allrows],np.uint8),stations=np.asarray([r['station']for r in allrows]),path_indices=np.asarray([lookup[r['native_path_id']]for r in allrows],np.int32))
    np.savez_compressed(OUT/'points/point_observations.npz',**z);groups=collections.defaultdict(list)
    with gzip.open(OUT/'points/point_records.jsonl.gz','wt',encoding='utf8',compresslevel=3)as f:
        for i,row in enumerate(allrows):row['source_row']=i;package.dump(f,row);groups[row['source_group_id']].append(row['observation_id'])
    with gzip.open(OUT/'points/source_correlated_groups.jsonl.gz','wt',encoding='utf8',compresslevel=3)as f:
        for gid,group in groups.items():package.dump(f,dict(group_id=gid,observation_ids=group,independent_samples=False))
    mapping=read(BEFORE/'points/mapping.json');mapping.update(path_ids=paths,point_input_sha256=prior.fit.file_ref(OUT/'points/point_observations.npz')['sha256'],parent_repaired_input=prior.fit.file_ref(BEFORE/'points/point_observations.npz'),auxiliary_camera=prior.fit.file_ref(OUT/'navigation_camera.npz'),source_repair_dispositions=str(OUT/'dispositions.jsonl.gz'));write(OUT/'points/mapping.json',mapping)
    old_by_id={r['observation_id']:r for r in original};same=0;human=[]
    for row in allrows:
        if row['observation_id']in old_by_id:
            old=old_by_id[row['observation_id']]
            if {k:v for k,v in old.items()if k!='source_row'}!={k:v for k,v in row.items()if k!='source_row'}:raise ValueError('Retained observation changed')
            same+=1
        if row['human_acceptance']:human.append(row['observation_id'])
    if set(human)!={r['observation_id']for r in original if r['human_acceptance']}:raise ValueError('Human scope changed')
    result=dict(frames=len(coverage),source_seconds=[44.98,56.3],rows=len(allrows),counts=dict(collections.Counter(entities[r['native_path_id']]['layer']for r in allrows)),native_paths=len(paths),coverage=coverage,repairs=dict(counts),unchanged_original_observations=same,human_accepted_observations=len(human),original_source_rows_preserved_at=str(BEFORE/'points'),native_geometry_changed=False,machine_results_user_accepted=False,input_sha256=mapping['point_input_sha256'])
    write(OUT/'result.json',result);inputs();prior.fit.load();print(json.dumps({k:v for k,v in result.items()if k!='coverage'},ensure_ascii=False),flush=True)


def integrate(requested=None):
    import package_s02_repaired as package
    package.OUT=OUT;package.initialize();package.verify()


def review(requested=None):
    import package_s02_repaired as package
    package.OUT=OUT;package.review(video=requested is None)


def audit(requested=None):
    original,_,entities,vis=inputs();rows=list(prior.rows(OUT/'points/point_records.jsonl.gz'));ids={r['observation_id']for r in rows}
    findings=read(BEFORE/'peer_review/confirmed_point_findings.json');remaining=[r for r in findings if r['observation_id']in ids]
    if remaining:raise ValueError('A confirmed wrong identity remains in the formal input')
    protected=read(BEFORE/'peer_review/review.json')['protected_files'];verified=[]
    for ref in protected:
        current=prior.fit.file_ref(Path(ref['path']))
        if current['sha256']!=ref['sha256']:raise ValueError('A protected input or native model changed')
        verified.append(dict(current,unchanged=True))
    prior.fit.load()
    by=collections.defaultdict(list)
    for row in rows:by[row['pts']].append(row)
    material_conflicts=[];ad_domain_conflicts=[];unresolved=collections.defaultdict(list)
    for pt in sorted(by):
        ice=read(OUT/f'ice/{pt}.json');ads=read(OUT/f'nonice/{pt}.json');supports=read(OUT/f'support/{pt}.json')
        cap=np.asarray(ads.get('board_cap_domain',{}).get('lower_blue_cap_uv',[]),float)
        if len(cap):cap=cap[np.argsort(cap[:,0])]
        contact=np.array([r['uv']for r in supports.get('board/continuous/bottom',{}).get('rows',[])])
        if len(contact):contact=contact[np.argsort(contact[:,0])]
        _,mask,_=centreline_house_conflicts(ice['features'],entities)
        if ice['reviewed_material_scopes']:mask|=reviewed_fill_mask(read(BEFORE/f'ice/{pt}.json')['features'],ice['reviewed_material_scopes'])
        distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,3)
        for row in by[pt]:
            x,y=row['uv'];ix,iy=np.rint([x,y]).astype(int);path=entities[row['native_path_id']]
            if path.get('measurement')=='centreline'and distance[iy,ix]>2:material_conflicts.append(dict(observation_id=row['observation_id'],pts=pt,path=row['native_path_id'],uv=row['uv']))
            if path['layer']=='advertisement'and ((len(cap)>=2 and y<=np.interp(x,cap[:,0],cap[:,1],left=np.nan,right=np.nan)+2)or(len(contact)>=2 and y>=np.interp(x,contact[:,0],contact[:,1],left=np.nan,right=np.nan)-2)):
                ad_domain_conflicts.append(dict(observation_id=row['observation_id'],pts=pt,path=row['native_path_id'],uv=row['uv']))
        for row in read(OUT/f'glyphs/{pt}.json')['decisions']:
            if row['action']=='unresolved_material_identity':unresolved[row['path_id']].append(row)
    issues=[]
    for pid,rr in sorted(unresolved.items()):
        pp=sorted({r['pts']for r in rr});cuts=np.r_[0,np.flatnonzero(np.diff(pp)>256)+1,len(pp)]
        spans=[dict(first_pts=pp[a],last_pts=pp[b-1],source_seconds=[pp[a]/12800,pp[b-1]/12800],frames=b-a)for a,b in zip(cuts[:-1],cuts[1:])]
        issues.append(dict(path_id=pid,old_candidate_rows=len(rr),spans=spans,reason='current_glyph_material_pair_not_resolved_or_another_source_contour_competes',
            effect='Those old candidate pixels have no accepted replacement; this is not a count of all visible missing pixels.',
            next_evidence='A resolved neighboring exposure and the same complete source contour may guide a new current-RGB measurement; no flow or projected coordinate may replace a readout.'))
    prior_remaining=read(BEFORE/'remaining_issues.json')
    write(OUT/'remaining_issues.json',dict(glyph_material_readout_failures=issues,unresolved_glyph_candidate_rows=sum(len(v)for v in unresolved.values()),
        inherited_unresolved_scope=prior_remaining,material_identity_conflicts_after_repair=material_conflicts,ad_surface_conflicts_after_repair=ad_domain_conflicts,
        whole_visible_material_recall_not_certified=True,machine_results_user_accepted=False))
    result=dict(frames=len(by),rows=len(rows),point_input_sha256=prior.fit.file_ref(OUT/'points/point_observations.npz')['sha256'],confirmed_prior_wrong_identities=len(findings),confirmed_wrong_ids_remaining=len(remaining),
        same_house_material_conflicts=len(material_conflicts),ad_surface_conflicts=len(ad_domain_conflicts),protected_files=verified,
        native_manifest=prior.fit.file_ref(prior.fit.OUT/'manifest.json'),native_geometry_unchanged=True,raw_history_unchanged=True,solver_started=False,gpu_used=False,
        original_human_clicks=sum(r['origin']=='preserved_human_source_click'for r in rows),human_accepted_rows=sum(r['human_acceptance']for r in rows),
        all_detected_changes_have_point_dispositions=True,visual_accuracy_is_not_inferred_from_native_ingestion=True)
    write(OUT/'final_source_audit.json',result);print(json.dumps({k:v for k,v in result.items()if k not in ['protected_files','native_manifest']},ensure_ascii=False),flush=True)


def deliver(requested=None):
    result=read(OUT/'result.json');verification=read(OUT/'verification.json');video=read(OUT/'review/video.json')
    if not(OUT/'final_source_audit.json').exists()or read(OUT/'final_source_audit.json').get('point_input_sha256')!=result['input_sha256']:audit()
    checked=read(OUT/'final_source_audit.json')
    if checked['same_house_material_conflicts']or checked['ad_surface_conflicts']:raise ValueError('Diagnosed source-domain conflicts remain; inspect them before delivery')
    if video['point_input_sha256']!=result['input_sha256']:raise ValueError('Video belongs to another dataset')
    files=[(p,'points/'+p.name)for p in (OUT/'points').iterdir()if p.is_file()]
    files.extend((OUT/name,name)for name in ['result.json','verification.json','native_integration.json','dispositions.jsonl.gz','auxiliary_projection_failures.json','remaining_issues.json','final_source_audit.json','ice_repair.json','glyph_repair.json','review/video.json','review/frame_index.jsonl.gz'])
    files.extend([(BASE/'evidence/manual_scope.json','history/manual_scope.json'),(BASE/'evidence/source_inventory.json','history/source_inventory.json'),(prior.fit.OUT/'geometry_seed.npz','navigation/geometry_seed.npz'),(prior.fit.OUT/'camera_seed.json','navigation/camera_seed.json'),(prior.fit.OUT/'controls_seed.npy','navigation/controls_seed.npy'),(OUT/'navigation_camera.npz','navigation/auxiliary_camera.npz'),(prior.fit.OUT/'manifest.json','native_authority.json')])
    name=OUT/'S02_native_points_identity_repaired_20261027.zip';manifest=dict(schema='rinkspace.s02_source_identity_repair_bundle.v1',source_pts=[575744,720640],time_base=[1,12800],frames=567,point_folder='points',baseline=prior.fit.file_ref(BEFORE/'points/point_observations.npz'),
        files=[dict(archive=entry,**prior.fit.file_ref(path))for path,entry in files],external_original_evidence_is_addressed_by_hash=True,video=video['video'],machine_results_user_accepted=False)
    with zipfile.ZipFile(name.with_suffix('.partial.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5)as archive:
        for path,entry in files:archive.write(path,entry)
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    name.with_suffix('.partial.zip').replace(name)
    with zipfile.ZipFile(name)as archive:
        if archive.testzip()is not None:raise ValueError('Archive checksum failed')
    write(OUT/'package.json',dict(archive=prior.fit.file_ref(name),bytes=name.stat().st_size))
    entry=Path(__file__).parent/'source_datasets/native_points_s02_20261027/result.json';record=read(entry)
    record['latest_identity_repair']=dict(result=prior.fit.file_ref(OUT/'result.json'),points=prior.fit.file_ref(OUT/'points/point_observations.npz'),mapping=prior.fit.file_ref(OUT/'points/mapping.json'),verification=prior.fit.file_ref(OUT/'verification.json'),audit=prior.fit.file_ref(OUT/'final_source_audit.json'),package=prior.fit.file_ref(name),video=video['video'],machine_results_user_accepted=False)
    write(entry,record);print(str(name),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['ice','glyphs','assemble','integrate','review','audit','deliver']);parser.add_argument('--pts',nargs='+',type=int)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);globals()[args.stage](args.pts)
