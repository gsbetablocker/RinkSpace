"""Recover wrong advertisement owners using complete color and word evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b,ad_source_verify_s01 as v,ad_color_review_s01 as c
import cv2,numpy as np,collections,time
from scipy.spatial import cKDTree

def run():
 cv2.setNumThreads(2);data=b.load();root=b.OUT/'verified_color_final';docs={};bypts=collections.defaultdict(list)
 for file in root.glob('*.json'):
  d=b.read(file);docs[(d['ad'],d['pts'])]=(file,d);bypts[d['pts']].append((file,d))
 if sum(len(d['confirmed'])+len(d['excluded']) for file,d in docs.values())!=int(np.count_nonzero(data['ad'])):raise ValueError('Within-ad final review must finish first')
 ix=b.read(b.OUT/'source_evidence_index.json');banned=set(map(int,ix['withdrawals']))|set(map(int,ix['retired']));sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012);refs={a['id']:b.prepare_reference(a,sift) for a in b.read(b.OUT/'atlases.json')['atlases']}
 from ad_finalize_s01 import exclusion_contours
 negatives=exclusion_contours(refs,[r['metadata'] for r in refs.values()]);stats=collections.Counter();changes=[];start=time.monotonic()
 for no,pt in enumerate(sorted(bypts)):
  pairs=bypts[pt];targets=collections.defaultdict(list);candidates={}
  for file,d in pairs:
   for r in d['confirmed']:targets[int(r['path_id'].split('.')[0][2:])].append((file,d,r))
   for r in d['excluded']:
    if r['row'] not in banned and data['kind'][r['row']]==0:candidates[r['row']]=(file,d,r)
  if not candidates or not targets:continue
  vis=pairs[0][1]['source_frame'];im=c.color_read(vis['image']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=cv2.imread(vis['mask_path'],0);purple=c.purple_region(im);votes=collections.defaultdict(list)
  for target,points in targets.items():
   if len(points)<5:continue
   cr=np.array([r['row'] for _,_,r in points]);clarity=c.directional_clarity(gray,data['uv'][cr],mask)
   if clarity['detail_unresolved_directions']:continue
   lo=data['uv'][cr].min(0)-16;hi=data['uv'][cr].max(0)+16;rr=np.array([row for row in candidates if int(data['ad'][row])!=target and np.all(data['uv'][row]>=lo) and np.all(data['uv'][row]<=hi)],int)
   if not len(rr):continue
   stats['cross_owner_candidates']+=len(rr);used=set()
   for targetfile,td,r in points:
    regkey=(str(targetfile),r['registration_index'])
    if regkey in used:continue
    used.add(regkey);reg=td['registrations'][r['registration_index']];ref=refs[reg['atlas_id']]
    if ref['metadata']['ad']!=target:continue
    try:mm,_=v.local_evidence(ref,reg,gray,data['uv'][rr])
    except (cv2.error,np.linalg.LinAlgError):continue
    tree=None
    if reg['atlas_id'] in negatives and mm:
     q=cv2.perspectiveTransform(negatives[reg['atlas_id']][None],np.array(reg['H']))[0];ok=np.isfinite(q).all(1)&(np.abs(q).max(1)<10000)
     if ok.any():tree=cKDTree(q[ok])
    for match in mm:
     row=int(rr[match.pop('local_row')]);uv=data['uv'][row];flow=np.array(match['local_flow']);x,y=np.rint(uv).astype(int);magenta=bool(purple[y,x]);path=match['path_id']
     if target==4 and (('shot4_' in path and magenta) or ('ANKARA' in path and not magenta)):continue
     if tree is not None and tree.query(uv-flow)[0]<match['source_reference_distance']+.4:continue
     a=ref['metadata'];color=c.chroma_check(c.color_read(a['image']),im,np.array(reg['H']),uv,flow,bool(a.get('photometric_transform')))
     if not color['verified']:continue
     word=c.wide_context(ref['gray'],gray,np.array(reg['H']),uv,flow,mask)
     if word<.72:continue
     votes[row].append(dict(row=row,stable_id=str(data['ids'][row]),native_target_advertisement=target,previous_advertisement=int(data['ad'][row]),cross_ad_entity_repaired=True,wide_word_context_correlation=word,physical_color_verified=True,directional_word_clarity_verified=True,color_evidence=color,directional_clarity=clarity,target_frame_evidence=str(targetfile),target_registration_index=r['registration_index'],**match))
  modified={}
  for row,choices in votes.items():
   groups=collections.defaultdict(list)
   for x in choices:groups[x['path_id']].append(x)
   ranked=sorted([(max(x['wide_word_context_correlation'] for x in vv),label) for label,vv in groups.items()],reverse=True)
   if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.1:stats['ambiguous_cross_owner']+=1;continue
   best=max(groups[ranked[0][1]],key=lambda x:x['wide_word_context_correlation']);file,d,old=candidates[row]
   if str(file) not in modified:
    archive=b.OUT/'review_revisions'/('before_color_cross_'+file.name)
    if not archive.exists():archive.write_bytes(file.read_bytes())
    modified[str(file)]=(file,d,archive)
   targetdoc=next(td for tf,td in pairs if str(tf)==best['target_frame_evidence']);reg=targetdoc['registrations'][best['target_registration_index']];best['registration_index']=len(d['registrations']);d['registrations'].append(reg);d['confirmed'].append(best);d['excluded']=[x for x in d['excluded'] if x['row']!=row];d['conflicts']=[x for x in d.get('conflicts',[]) if x['row']!=row]
   changes.append(dict(row=row,pts=pt,previous=int(data['ad'][row]),native_target=best['native_target_advertisement'],path_id=best['path_id'],wide_word_context=best['wide_word_context_correlation']));stats['unique_color_word_cross_owner_repairs']+=1
  for file,d,archive in modified.values():
   d['counts']=dict(collections.Counter(r['reason'] for r in d['excluded']));d['counts'].update(confirmed=len(d['confirmed']),checked=len(d['confirmed'])+len(d['excluded']));d['color_cross_review_prior']=dict(path=str(archive),sha256=b.sha(archive));b.write(file,d)
  if no%150==0:b.emit('color_cross_progress',frame=no,counts=dict(stats),seconds=round(time.monotonic()-start,1))
 b.write(b.OUT/'cross_color_review.json',dict(counts=dict(stats),changes=changes,script_sha256=b.sha(__file__),native_geometry_used_for_identity=False));b.emit('color_cross_complete',counts=dict(stats),seconds=round(time.monotonic()-start,1))
if __name__=='__main__':run()
