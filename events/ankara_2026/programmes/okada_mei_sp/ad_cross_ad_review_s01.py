"""Check old advertisement-owner mistakes against independently verified source words."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_source_verify_s01 as v
import cv2,numpy as np,collections,time

def run():
    cv2.setNumThreads(2);data=b.load();root=b.OUT/'verified_final';docs={}
    for p in root.glob('*.json'):
        d=b.read(p);docs[(d['ad'],d['pts'])]=(p,d)
    reviewed=sum(len(d['confirmed'])+len(d['excluded']) for _,d in docs.values())
    if reviewed!=np.count_nonzero(data['ad']):raise ValueError('Wait for complete within-ad review before cross-ad review')
    index=b.read(b.OUT/'source_evidence_index.json')
    banned={int(k) for k in index['withdrawals']}|{int(k) for k in index['retired']}
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012)
    refs={a['id']:b.prepare_reference(a,sift) for a in b.read(b.OUT/'atlases.json')['atlases']}
    from ad_finalize_s01 import exclusion_contours
    negatives=exclusion_contours(refs,[r['metadata'] for r in refs.values()])
    from scipy.spatial import cKDTree
    statistics=collections.Counter();changes=[];start=time.monotonic()
    for frame_no,pt in enumerate(sorted(set(pt for _,pt in docs))):
        present={n:docs[(n,pt)] for n,p in docs if p==pt}
        if len(present)<2:continue
        candidates={}
        for n,(file,d) in present.items():
            for r in d['excluded']:
                row=r['row']
                if row not in banned and data['kind'][row]==0:candidates[row]=(n,r)
        if not candidates:continue
        sample=next(iter(present.values()))[1]
        gray=b.gray_read(sample['source_frame']['image'])
        votes=collections.defaultdict(list)
        for target,(target_file,d) in present.items():
            if len(d['confirmed'])<5:continue
            current=np.array([x['row'] for x in d['confirmed']])
            boxlo=data['uv'][current].min(0)-16;boxhi=data['uv'][current].max(0)+16
            rr=np.array([row for row,(previous,_) in candidates.items() if previous!=target and np.all(data['uv'][row]>=boxlo) and np.all(data['uv'][row]<=boxhi)],int)
            if not len(rr):continue
            statistics['cross_owner_candidate_checks']+=len(rr)
            # Use every chart that has already passed actual entity checks on
            # this very exposure. Proximity is only a search proposal.
            reg_ids=sorted(set(x['registration_index'] for x in d['confirmed']))
            for ri in reg_ids:
                reg=d['registrations'][ri];ref=refs[reg['atlas_id']]
                try:matches,_=v.local_evidence(ref,reg,gray,data['uv'][rr])
                except (cv2.error,np.linalg.LinAlgError):continue
                tree=None
                if reg['atlas_id'] in negatives and matches:
                    q=cv2.perspectiveTransform(negatives[reg['atlas_id']][None],np.asarray(reg['H']))[0]
                    keep=np.isfinite(q).all(1)&(np.abs(q).max(1)<10000)
                    if keep.any():tree=cKDTree(q[keep])
                for match in matches:
                    row=int(rr[match.pop('local_row')])
                    if tree is not None and tree.query(data['uv'][row]-np.asarray(match['local_flow']))[0]<match['source_reference_distance']+.4:continue
                    score=float(v.patch_ncc(ref['gray'],gray,np.asarray(reg['H']),data['uv'][row:row+1],39,np.asarray(match['local_flow'])[None])[0])
                    if score<.72:continue
                    votes[row].append(dict(row=row,stable_id=str(data['ids'][row]),native_target_advertisement=target,
                        previous_advertisement=int(data['ad'][row]),cross_ad_entity_repaired=True,word_context_correlation=score,
                        target_frame_evidence=str(target_file),target_registration_index=ri,**match))
        modified={}
        for row,choices in votes.items():
            groups=collections.defaultdict(list)
            for r in choices:groups[r['path_id']].append(r)
            ranks=sorted([(max(x['word_context_correlation'] for x in vv),label) for label,vv in groups.items()],reverse=True)
            if len(ranks)>1 and ranks[0][0]-ranks[1][0]<.10:
                statistics['cross_owner_ambiguous']+=1;continue
            best=max(groups[ranks[0][1]],key=lambda x:x['word_context_correlation'])
            previous=candidates[row][0];file,d=present[previous]
            if previous not in modified:
                archive=b.OUT/'review_revisions'/('before_cross_owner_'+file.name);archive.parent.mkdir(exist_ok=True)
                if not archive.exists():archive.write_bytes(file.read_bytes())
                modified[previous]=(file,d,archive)
            target_doc=present[best['native_target_advertisement']][1]
            reg=target_doc['registrations'][best['target_registration_index']]
            best['registration_index']=len(d['registrations']);d['registrations'].append(reg)
            d['confirmed'].append(best);d['excluded']=[x for x in d['excluded'] if x['row']!=row];d['conflicts']=[x for x in d['conflicts'] if x['row']!=row]
            changes.append(dict(row=row,stable_id=best['stable_id'],pts=pt,previous_advertisement=previous,target_advertisement=best['native_target_advertisement'],path_id=best['path_id'],word_context_correlation=best['word_context_correlation'],previously_active=bool(data['old_active'][row])))
            statistics['unique_cross_owner_repairs']+=1
        for file,d,archive in modified.values():
            counts=collections.Counter(x['reason'] for x in d['excluded']);counts.update(confirmed=len(d['confirmed']),checked=len(d['confirmed'])+len(d['excluded']))
            d['counts']=dict(counts);d['cross_owner_review_prior']=dict(path=str(archive),sha256=b.sha(archive))
            b.write(file,d)
        if frame_no%80==0:b.emit('cross_owner_source_review_progress',frame=frame_no,counts=dict(statistics),seconds=round(time.monotonic()-start,1))
    b.write(b.OUT/'cross_owner_review.json',dict(schema='rinkspace.actual_source_cross_ad_entity_review.v1',counts=dict(statistics),changes=changes,native_parameters_used_for_correspondence=False,original_UV_PTS_IDs_unchanged=True,code=str(Path(__file__).resolve()),code_sha256=b.sha(__file__)))
    b.emit('cross_owner_source_review_complete',counts=dict(statistics),seconds=round(time.monotonic()-start,1))
if __name__=='__main__':run()
