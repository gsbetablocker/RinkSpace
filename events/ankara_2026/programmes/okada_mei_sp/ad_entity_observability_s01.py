"""Audit actual local visibility separately from recognizing a whole advertisement."""
from pathlib import Path
from functools import lru_cache
import sys,json,gzip,re,collections,time,itertools
import numpy as np,cv2
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_entity_measure_s01 as m
r=m.r;b=r.b;OUT=r.OUT
CANONICAL_BY_AD={}
CANONICAL_BY_PATH={}
LOCAL_PRINT_SHARPNESS_ADS=set()
BEST_INDEPENDENT_SHAPE_ADS=set()

@lru_cache(maxsize=4)
def native_material_roles(registry_path):
    return {p['native_id']:p.get('role')for p in b.read(registry_path)['paths']}

def group_name(name):
    if name.startswith('Ad1.'):return re.sub(r'_inner\d*$', '', name)
    if name.startswith('Ad7.'):
        q=re.match(r'(Ad7\.(?:top|bottom)_\d+_[A-Z])',name)
        if q:return q.group(1)
    if name.startswith(('Ad8.','Ad10.')):return name.split('.boundary')[0]
    if name.startswith('Ad2.'):return name.replace('_inner','')
    if name.startswith('Ad4.print.'):return re.sub(r'_inner|__part\d+','',name)
    return name

def correlation(a,c):
    a=a-a.mean();c=c-c.mean()
    return float(np.sum(a*c)/max(np.linalg.norm(a)*np.linalg.norm(c),1e-9))

def complete_named_source_reference(v,ref):
    """A full photographed glyph can validate portions absent in an old crop.

    This identifies reference scope only. Current RGB observability remains
    required; neither artwork geometry nor a high score changes the native path.
    """
    paths=[p for p in ref['atlas']['paths']if p['path_id']==v['path']]
    return (bool(paths) and not ref['atlas'].get('partial_source_reference',True) and
        all(p.get('proof',{}).get('kind')=='visually_identified_actual_source_glyph_by_word_order'
            and len(p['pixels'])>=4 and np.allclose(p['pixels'][0],p['pixels'][-1]) for p in paths))

def reference_normal_width(ref,H,q,n):
    offsets=np.arange(-8.,8.001,.25)
    xy=q+offsets[:,None]*n
    source=cv2.perspectiveTransform(xy[None],np.linalg.inv(H))[0]
    values=r.sample(ref['gray'],source[:,None,:]).ravel()
    values=cv2.GaussianBlur(values.reshape(1,-1).astype(np.float32),(0,0),2.4).ravel()
    gradient=np.diff(values);mid=(offsets[:-1]+offsets[1:])*.5
    central=np.flatnonzero(abs(mid)<=3.)
    peak=int(central[np.argmax(gradient[central])]);height=gradient[peak]
    if height<=0:return None
    lo=peak;hi=peak
    while lo>0 and gradient[lo-1]>.02*height:lo-=1
    while hi+1<len(gradient) and gradient[hi+1]>.02*height:hi+=1
    mass=np.maximum(gradient[lo:hi+1],0);cdf=np.cumsum(mass);cdf/=cdf[-1]
    positions=np.interp([.1,.9],cdf,mid[lo:hi+1])
    return float(positions[1]-positions[0])

def group_score(ref,H,group,gray,mask):
    paths=[p for p in ref['atlas']['paths'] if group_name(p['path_id'])==group]
    p=np.concatenate([np.asarray(p['pixels'],float) for p in paths]);q=cv2.perspectiveTransform(p[None],H)[0]
    span=np.ptp(q,axis=0);lo=np.maximum(np.floor(q.min(0)-4).astype(int),[5,185]);hi=np.minimum(np.ceil(q.max(0)+5).astype(int),[1914,1074])
    fine=bool(span[1]<54 or span[0]<12 or any(s in group for s in ['glyph_','monogram_W','.star','.crescent','native850']) or re.match(r'Ad7\.(top|bottom)_\d+',group) or group.startswith('Ad4.print.shot4_'))
    # Borders remain partial observable arcs even when their clipped bbox is narrow.
    if any(s in group for s in ['_edge','_vertical','blue_bar','green_bar','blue_tile','red_field']):fine=False
    out=dict(fine_detail=fine,projected_source_extent=span.tolist(),current_structure_resolved=True)
    if not fine:
        out['current_structure_resolved']=None
        out['local_material_confirmation_required']=True
        return out
    if min(hi-lo)<7 or max(hi-lo)>1000:
        out.update(current_structure_resolved=False,reason='fine_entity_not_visible_as_a_two_dimensional_shape');return out
    yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];xy=np.stack([xx,yy],-1).astype(float)
    src=cv2.perspectiveTransform(xy.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(xy.shape)
    aa=cv2.GaussianBlur(r.sample(ref['gray'],src).astype(np.float32),(3,3),.7)
    best=None
    for dy in [-2,0,2]:
        for dx in [-2,0,2]:
            cc=cv2.GaussianBlur(gray[lo[1]+dy:hi[1]+dy,lo[0]+dx:hi[0]+dx].astype(np.float32),(3,3),.7)
            valid=mask[lo[1]+dy:hi[1]+dy,lo[0]+dx:hi[0]+dx]==0
            if valid.mean()<.55 or valid.sum()<30:continue
            score=correlation(aa[valid],cc[valid]);gx=[]
            for axis in [1,0]:
                va=np.diff(aa,axis=axis);vc=np.diff(cc,axis=axis)
                vm=(valid[:,:-1]&valid[:,1:]) if axis==1 else (valid[:-1]&valid[1:])
                gx.append(correlation(va[vm],vc[vm]) if vm.sum()>10 else -1.)
            candidate=dict(unblurred_shape_correlation=score,horizontal_structure_correlation=gx[0],vertical_structure_correlation=gx[1],actual_luminance_std=float(np.std(cc[valid])),visible_fraction=float(valid.mean()),navigation_translation_px=[dx,dy])
            if best is None or score>best['unblurred_shape_correlation']:best=candidate
    if best is None:
        out.update(current_structure_resolved=False,reason='fine_shape_foreground_obscured');return out
    out.update(best)
    out['current_structure_resolved']=bool(best['unblurred_shape_correlation']>=.66 and best['horizontal_structure_correlation']>=.14 and best['vertical_structure_correlation']>=.25 and best['actual_luminance_std']>=3.)
    if not out['current_structure_resolved']:out['reason']='current_fine_shape_strokes_not_individually_resolved'
    out['synthetic_motion_blur_used_for_admission']=False
    return out

def frame_check(pt,records,doc,refs):
    im,gray,mask=r.current(pt);mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    roles=native_material_roles(OUT/'native_entity_registry.json')
    present={int(ai):row[str(pt)] for ai,row in doc['records'].items() if str(pt) in row}
    by=collections.defaultdict(list)
    for ai,c in present.items():
        for v in m.chart_paths(refs[ai],c,im,gray,mask):by[v['path']].append(v)
    canonical={7:[76],8:[71],10:[73],11:[67],12:[68],3:[70],**CANONICAL_BY_AD};cache={};best={}
    def getscore(v):
        ai=v['atlas_id'];g=group_name(v['path']);key=(ai,g)
        if key not in cache:cache[key]=group_score(refs[ai],v['H'],g,gray,mask)
        return cache[key]
    for name,vs in by.items():
        ad=int(name.split('.')[0][2:]);ca=CANONICAL_BY_PATH.get(name,[])+canonical.get(ad,[]);cvs=sorted([v for v in vs if v['atlas_id'] in ca],key=lambda v:ca.index(v['atlas_id']))
        photographed=[v for v in vs if complete_named_source_reference(v,refs[v['atlas_id']]) and getscore(v)['current_structure_resolved'] is not False]
        if photographed:
            best[name]=max(photographed,key=lambda v:getscore(v).get('unblurred_shape_correlation',-1.))
        elif cvs and ad in BEST_INDEPENDENT_SHAPE_ADS:
            best[name]=max(cvs,key=lambda v:(getscore(v)['current_structure_resolved'] is not False,getscore(v).get('unblurred_shape_correlation',float(np.median(v['ncc'][v['valid']])) if v['valid'].any() else -1.)))
        else:best[name]=cvs[0] if cvs else max(vs,key=lambda v:np.median(v['ncc'][v['valid']]) if v['valid'].any() else -1.)
    for v in best.values():getscore(v)
    word=collections.defaultdict(list)
    for (ai,g),s in cache.items():
        if ai==76 and re.match(r'Ad7\.(top|bottom)_\d+',g) and 'unblurred_shape_correlation' in s:word[g.split('_')[0]].append(s)
    word_bad={g for g,ss in word.items() if len(ss)>=3 and (np.median([s['unblurred_shape_correlation'] for s in ss])<.66 or np.median([s['horizontal_structure_correlation'] for s in ss])<.14)}
    pp=[];nn=[];names=[]
    for name,v in best.items():
        valid=v['valid'];pp.extend(v['q'][valid]);nn.extend(v['n'][valid]);names.extend([name]*int(valid.sum()))
    if pp:
        tree=cKDTree(pp);nn=np.asarray(nn);names=np.asarray(names)
        dd,kk=tree.query(np.array([c['uv'] for c in records]),k=min(12,len(pp)))
        dd=dd.reshape(len(records),-1);kk=kk.reshape(len(records),-1)
    pair_trees={}
    for ai in present:
        vv=[v for vs in by.values() for v in vs if v['atlas_id']==ai]
        if vv:
            pq=np.concatenate([v['q'] for v in vv]);pn=np.concatenate([v['n'] for v in vv]);pair_trees[ai]=(cKDTree(pq),pq,pn)
    output=[]
    for j,c in enumerate(records):
        name=c['path_id'].split('::')[1];v=best.get(name);reason=None;check={}
        if v is None:reason='current_source_entity_chart_unavailable'
        else:
            check=dict(getscore(v),identity_reference_atlas=v['atlas_id']);g=group_name(name)
            if check['current_structure_resolved'] is False:reason=check.get('reason','current_fine_shape_strokes_not_individually_resolved')
            if v['atlas_id']==76 and g.split('_')[0] in word_bad:reason='small_text_row_smeared_into_bands'
            if re.match(r'Ad7\.(top|bottom)_\d+',name) and v['atlas_id']!=76:reason='fine_letter_has_no_independent_complete_source_identity'
            dist,qi=v['tree'].query(c['uv']);offsets=c['source_profile']['offsets'];width=float(offsets[2]-offsets[0]);limit=max(4.,width/2+2.)
            check.update(distance_to_independent_source_entity_px=float(dist),same_entity_support_limit_px=limit)
            ad=int(name.split('.')[0][2:])
            if ad in LOCAL_PRINT_SHARPNESS_ADS and '.print.component' in name:
                expected=reference_normal_width(refs[v['atlas_id']],v['H'],v['q'][int(qi)],v['n'][int(qi)])
                if expected is not None:
                    resolved_limit=max(3.,2.*expected)
                    check.update(independent_normal_transition_width_px=expected,local_resolved_transition_width_limit_px=resolved_limit,current_normal_transition_width_px=width)
                    if width>resolved_limit:reason='current_local_print_transition_smeared_vs_independent_reference'
            # Compare the observed blur support with the separation of opposite
            # material interfaces in the SAME actual-source reference. A thin
            # stroke merged by motion cannot supply two distinct edge identities.
            tree2,pq,pn=pair_trees[v['atlas_id']];anchor=v['q'][int(qi)];normal2=v['n'][int(qi)];radius=max(3.,width+1.);nearby=tree2.query_ball_point(anchor,radius)
            if nearby:
                delta=pq[nearby]-anchor;along=np.abs(delta@normal2);across=np.abs(delta@np.array([-normal2[1],normal2[0]]));opposed=(pn[nearby]@normal2)<-.7
                matches=opposed&(across<1.5)&(along>.5)
                if matches.any():
                    separation=float(along[matches].min());check['opposed_source_interface_separation_px']=separation;check['observed_transition_width_px']=width
                    if width>separation:reason='individual_paint_interfaces_overlap_at_current_blur_scale'
            border=any(k in name for k in ['_edge','_vertical'])
            if str(roles.get(c['path_id'],'')).startswith('letter_'):
                sp=c['source_profile'];query=np.asarray(sp.get('search_uv',c['uv']))
                pair=m.current_glyph_material_pair(gray,mask,v['q'],query,sp['normal'],sp['offsets'])
                check['actual_glyph_material_pair']=pair
                if not pair['resolved']:reason=pair['reason']
            if dist>limit and not border:reason='source_reference_disagrees_with_independent_entity_identity'
            if border and name.split('.')[0] in ['Ad2','Ad3','Ad4','Ad7']:
                sp=c['source_profile'];base=np.asarray(sp.get('search_uv',c['uv']));normal=np.asarray(sp['normal']);off=np.array([sp['offsets'][0]-3,sp['offsets'][0]-1,sp['offsets'][2]+1,sp['offsets'][2]+3]);rgb=r.sample(im,(base+off[:,None]*normal)[None])[0];sides=np.stack([rgb[:2].mean(0),rgb[2:].mean(0)]);chroma=np.ptp(sides,axis=1)
                material_ok=bool(chroma.max()>20 and chroma.min()<max(15.,chroma.max()*.60))
                check['actual_colored_panel_to_background_transition']=material_ok
                if not material_ok:reason='claimed_panel_edge_has_no_observed_paint_background_interface'
            if pp:
                normal=np.asarray(c['source_profile']['normal']);same=names[kk[j]]==name;samepaint=(nn[kk[j]]@normal)>.5
                other=(~same)&samepaint&(dd[j]<max(0.,float(dist)-2.))
                if other.any() and not border:
                    reason='measured_transition_matches_different_source_entity'
                    check['competing_source_entities']=sorted(set(names[kk[j][other]].tolist()))
        c['current_observability']=check
        if reason:c.update(previous_decision=c['decision'],decision='not_admitted',reason=reason)
        output.append(c)
    return output,cache

def run(source_file=None, mapping_file=None):
    cv2.setNumThreads(2);doc=b.read(OUT/'source_identity_charts.json')
    ats=b.read(b.OUT/'atlases.json')['atlases']+b.read(OUT/'additional_source_atlases.json')['atlases']
    refs={int(ai):r.reference(ats[int(ai)]) for ai in doc['records'] if doc['records'][ai]};sample='--sample' in sys.argv
    selected={int(round(float(x)*12800)) for x in sys.argv[sys.argv.index('--sample')+1].split(',')} if sample else None
    source=Path(source_file) if source_file is not None else OUT/'pointwise_checks.jsonl.gz';out=OUT/('sample_observability_checks.jsonl.gz' if sample else 'observability_checked_points.jsonl.gz')
    scores=[];reasons=collections.Counter();accepted=[];frames=[];start=time.monotonic();mapping=b.read(mapping_file or OUT/'verified_mapping.json');paths=mapping['path_ids']
    with gzip.open(source,'rt',encoding='utf8') as f,gzip.open(out,'wt',encoding='utf8') as dest:
        for fi,(pt,lines) in enumerate(itertools.groupby((json.loads(s) for s in f),key=lambda c:c['pts'])):
            rows=list(lines)
            if selected is not None and pt not in selected:continue
            candidates=[c for c in rows if c['decision']!='not_admitted'];existing=[c for c in rows if c['decision']=='not_admitted']
            checked,cache=frame_check(pt,candidates,doc,refs) if candidates else ([],{})
            keep=[c for c in checked if c['decision']!='not_admitted'];accepted.extend(keep)
            reasons.update(c['reason'] for c in checked if c['decision']=='not_admitted')
            for c in existing+checked:dest.write(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n')
            for (ai,g),s in cache.items():scores.append(dict(pts=pt,atlas_id=ai,source_group=g,**s))
            frames.append(dict(pts=pt,source_seconds=pt/12800,ad_points=len(keep),verified_original=sum(c['decision']=='include_original_pixel' for c in keep),new_source_measurements=sum(c['decision']=='new_actual_source_boundary' for c in keep),by_ad=dict(collections.Counter(c['path_id'].split('::')[0].split('/')[-1] for c in keep))))
            if fi%50==0 or sample:print(json.dumps(dict(stage='current_entity_observability',pts=pt,kept=len(keep),candidates=len(candidates),reasons=dict(reasons),seconds=round(time.monotonic()-start,1))),flush=True)
    tag='sample_observability_' if sample else 'observability_'
    z=dict(source_rows=np.array([c['source_row'] for c in accepted],np.int64),observation_ids=np.array([c['stable_id'] for c in accepted]),uv=np.array([c['uv'] for c in accepted]).reshape(-1,2),pts=np.array([c['pts'] for c in accepted],np.int64),kind=np.array([c['kind'] for c in accepted],np.uint8),stations=np.array([c['station'] for c in accepted]),path_indices=np.array([paths.index(c['path_id']) for c in accepted],np.int32))
    np.savez_compressed(OUT/(tag+'ads.npz'),**z)
    b.write(OUT/(tag+'summary.json'),dict(rejection_reasons=dict(reasons),ad_rows=len(accepted),source_frames=len(frames),frames=frames,by_ad=dict(collections.Counter(c['path_id'].split('::')[0].split('/')[-1] for c in accepted)),by_path=dict(collections.Counter(c['path_id'] for c in accepted)),native_parameters_unchanged=True,human_visual_acceptance=False,decisions=dict(collections.Counter(c['decision'] for c in accepted))))
    with gzip.open(OUT/(tag+'regions.jsonl.gz'),'wt',encoding='utf8') as f:
        for c in scores:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    print(json.dumps(dict(done=True,rows=len(accepted),seconds=round(time.monotonic()-start,1))),flush=True)
if __name__=='__main__':run()
