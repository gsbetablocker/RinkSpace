"""Resolve remaining entities with full color regions and word identity, then publish decisions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b,ad_source_verify_s01 as v,ad_color_review_s01 as c
import numpy as np,cv2,collections,time
from scipy.spatial import cKDTree

def run():
 cv2.setNumThreads(2);data=b.load();ats=b.read(b.OUT/'atlases.json')['atlases'];sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012);refs={a['id']:b.prepare_reference(a,sift) for a in ats};out=b.OUT/'verified_color_final';out.mkdir(exist_ok=True)
 from ad_finalize_s01 import exclusion_contours
 negatives=exclusion_contours(refs,ats);index=b.read(b.OUT/'source_evidence_index.json');banned=set(map(int,index['withdrawals']))|set(map(int,index['retired']));stats=collections.Counter();start=time.monotonic();explicit={337718:'Ad4.ANKARA.native440.boundary_5',337614:'Ad4.ANKARA.native440.boundary_4',29614:'Ad2.print.K',29615:'Ad2.print.K'}
 for number,p in enumerate(sorted((b.OUT/'verified_final').glob('*.json'))):
  d=b.read(p);destination=out/p.name
  if not d['confirmed']:
   d.update(version='source_entity_color_word_v4',previous_source_review=dict(path=str(p),sha256=b.sha(p)));b.write(destination,d);continue
  checkfile=b.OUT/'color_review_checks'/p.name;checks=b.read(checkfile)
  if checks['source_evidence_sha256']!=b.sha(p):raise ValueError('Source decisions changed during color review')
  ck={r['row']:r for r in checks['rows']};original={r['row']:r for r in d['confirmed']};im=c.color_read(d['source_frame']['image']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=cv2.imread(d['source_frame']['mask_path'],0);purple=c.purple_region(im)
  kept=[];withdrawn=[];retry=[];targets={}
  for row,r in original.items():
   audit=ck[row];target=r['path_id'].split('.')[0];detail=checks['clarity'][target]['detail_unresolved_directions'];macro=any(x in r['path_id'] for x in ['top_edge','bottom_edge','vertical'])
   if detail and not macro:
    reason='whole_word_directional_blur_detail_unresolved'
    withdrawn.append(dict(row=row,stable_id=r['stable_id'],reason=reason,identified_native_entity=explicit.get(row,r['path_id'] if not audit['wrong_Ad4_word_region'] else None),identified_physical_word='ANKARA' if target=='Ad4' and audit['actual_magenta_panel'] else None,color_review=str(checkfile),prior_path_id=r['path_id'],directional_blur=checks['clarity'][target]));stats[reason]+=1
   elif audit['reason'] is None and (audit['wide_word_context']>=.65 or macro):
    r.update(color_word_review=dict(path=str(checkfile),sha256=b.sha(checkfile)),wide_word_context_correlation=audit['wide_word_context'],physical_color_verified=True,directional_word_clarity_verified=True);kept.append(r);stats['original_entity_confirmed_with_full_color_word_context']+=1
   else:retry.append(row);targets[row]=int(target[2:])
  # On clear Ad4 exposures, revisit prior unresolved points under the actual
  # physical word region as well; native geometry never supplies a candidate.
  if d['ad']==4 and not checks['clarity'].get('Ad4',{}).get('detail_unresolved_directions'):
   for r in d['excluded']:
    row=r['row']
    if row not in banned and data['kind'][row]==0:retry.append(row);targets[row]=4
  retry=sorted(set(retry));votes=collections.defaultdict(list)
  if retry:
   rr=np.array(retry);uv=data['uv'][rr]
   for ri,reg in enumerate(d['registrations']):
    ref=refs[reg['atlas_id']];target=ref['metadata']['ad'];wanted=np.flatnonzero([targets[int(row)]==target for row in rr])
    if not len(wanted):continue
    try:matches,_=v.local_evidence(ref,reg,gray,uv[wanted])
    except (cv2.error,np.linalg.LinAlgError):continue
    negative=negatives.get(reg['atlas_id']);tree=None
    if negative is not None and matches:
     xy=cv2.perspectiveTransform(negative[None],np.asarray(reg['H']))[0];finite=np.isfinite(xy).all(1)&(np.abs(xy).max(1)<10000)
     if finite.any():tree=cKDTree(xy[finite])
    for match in matches:
     row=int(rr[wanted[match.pop('local_row')]]);point=data['uv'][row];x,y=np.rint(point).astype(int);magenta=bool(purple[y,x]);path=match['path_id']
     if target==4 and (('shot4_' in path and magenta) or ('ANKARA' in path and not magenta)):continue
     if tree is not None and tree.query(point-np.array(match['local_flow']))[0]<match['source_reference_distance']+.4:continue
     a=ref['metadata'];color=c.chroma_check(c.color_read(a['image']),im,np.array(reg['H']),point,np.array(match['local_flow']),bool(a.get('photometric_transform')))
     if not color['verified']:continue
     word=c.wide_context(ref['gray'],gray,np.array(reg['H']),point,np.array(match['local_flow']),mask)
     if word<.65:continue
     votes[row].append(dict(row=row,stable_id=str(data['ids'][row]),registration_index=ri,wide_word_context_correlation=word,color_evidence=color,physical_color_verified=True,directional_word_clarity_verified=True,color_word_review=dict(path=str(checkfile),sha256=b.sha(checkfile)),**match))
   for row in retry:
    choices=votes[row];bylabel=collections.defaultdict(list)
    for r in choices:bylabel[r['path_id']].append(r)
    ranking=sorted([(max(r['wide_word_context_correlation'] for r in rs),name) for name,rs in bylabel.items()],reverse=True)
    if ranking and (len(ranking)==1 or ranking[0][0]-ranking[1][0]>=.10):
     best=max(bylabel[ranking[0][1]],key=lambda r:r['wide_word_context_correlation']);old=original.get(row)
     if old:best['prior_path_id']=old['path_id']
     kept.append(best);stats['rechecked_entity_in_clear_word_context']+=1
     if old and old['path_id']!=best['path_id']:stats['corrected_physical_entity']+=1
     if not old:stats['previously_unresolved_clear_Ad4_rows_recovered']+=1
    elif row in original:
     old=original[row];reason=ck[row]['reason'] or 'physical_entity_unconfirmed_in_whole_word_context'
     withdrawn.append(dict(row=row,stable_id=old['stable_id'],reason=reason,prior_path_id=old['path_id'],color_review=str(checkfile),identified_native_entity=explicit.get(row)));stats[reason]+=1
  handled={r['row'] for r in kept}|{r['row'] for r in withdrawn};d['excluded']=[r for r in d['excluded'] if r['row'] not in handled]+withdrawn;d['confirmed']=kept
  d['counts']=dict(collections.Counter(r['reason'] for r in d['excluded']));d['counts'].update(confirmed=len(kept),checked=len(kept)+len(d['excluded']))
  d.update(version='source_entity_color_word_v4',previous_source_review=dict(path=str(p),sha256=b.sha(p)),full_color_word_review=dict(path=str(checkfile),sha256=b.sha(checkfile)),direct_color_visual_review=str(b.OUT/'direct_color_source_review.json'))
  b.write(destination,d)
  if number%150==0:b.emit('apply_color_word_progress',files=number,counts=dict(stats),seconds=round(time.monotonic()-start,1))
 b.write(b.OUT/'final_color_decisions.json',dict(schema='rinkspace.source_entity_color_word_decisions.v4',counts=dict(stats),all_final_evidence=str(out),code_sha256=b.sha(__file__),native_geometry_used_for_identity=False,original_UV_PTS_IDs_changed=False));b.emit('apply_color_word_complete',counts=dict(stats),seconds=round(time.monotonic()-start,1))
if __name__=='__main__':run()
