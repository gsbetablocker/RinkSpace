"""Check printed panel sides on actual color fields; never use native projection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b,ad_color_review_s01 as c
import cv2,numpy as np,collections

def fields(im,n):
 B,G,R=cv2.split(im.astype('float32'));purple=(R-G>24)&(B-G>20)&(R-B>-30);blue=(B-R>18)&(B-G>10)
 if n==2:mask=purple|blue
 elif n==4:mask=purple|(blue&(G>65)&(R>45))
 elif n==6:mask=(R-G>30)&(R-B>20)
 else:mask=blue
 mask[:180]=False;mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);result=[]
 for contour in contours:
  x,y,w,h=cv2.boundingRect(contour)
  if w<60 or h<25 or cv2.contourArea(contour)<1800:continue
  if n==2:
   region=np.zeros(mask.shape,np.uint8);cv2.drawContours(region,[contour],0,1,-1)
   if np.count_nonzero(purple&(region>0))<.025*np.count_nonzero(region):continue
  result.append((contour[:,0].astype(float),(x,y,w,h)))
 return result

def run():
 data=b.load();assets=b.assets();changes=[];stats=collections.Counter()
 for file in (b.OUT/'verified_color_final').glob('*.json'):
  d=b.read(file);todo=[r for r in d['confirmed'] if any(k in r['path_id'] for k in ['top_edge','bottom_edge','vertical']) and r.get('wide_word_context_correlation',1)<.65]
  if not todo:continue
  archive=b.OUT/'review_revisions'/('before_boundary_'+file.name)
  if not archive.exists():archive.write_bytes(file.read_bytes())
  im=c.color_read(d['source_frame']['image']);new=[];excluded=[];cache={}
  for r in todo:
   row=r['row'];uv=data['uv'][row];n=int(r['path_id'].split('.')[0][2:]);options=[]
   if n not in cache:cache[n]=fields(im,n)
   for points,box in cache[n]:
    distances=np.linalg.norm(points-uv,axis=1);j=int(np.argmin(distances));dist=float(distances[j]);q=points[j];x,y,w,h=box;t=points[(j+4)%len(points)]-points[(j-4)%len(points)];side=None
    if abs(t[0])>=abs(t[1]):
     if q[1]<y+.2*h:side='top_edge'
     elif q[1]>y+.8*h:side='bottom_edge'
    else:
     if q[0]<x+.2*w and x>1:side='left_vertical'
     elif q[0]>x+.8*w and x+w<im.shape[1]-1:side='right_vertical'
    if dist<=3.5 and side:
     target=next((p for p in assets[n]['paths'] if p==f'Ad{n}.{side}' or p==f'Ad{n}.{side}/path0'),None)
     if target:options.append((dist,target,box,q.tolist(),side))
   options.sort()
   if options and (len(options)==1 or options[0][1]==options[1][1] or options[1][0]-options[0][0]>1):
    dist,target,box,q,side=options[0];old=dict(r);r['path_id']=target;r['superseded_local_chart_relation']=dict(path_id=old['path_id'],registration_index=old['registration_index'],source_curve_fraction=old.get('source_curve_fraction'),wide_word_context=old.get('wide_word_context_correlation'));r.pop('source_curve_fraction',None);r.pop('wide_word_context_correlation',None)
    r['physical_boundary_side_evidence']=dict(method='actual_colored_field_external_contour_and_visible_side',source_image=d['source_frame']['image'],source_BGR_sha256=d['source_frame'].get('source_BGR_sha256'),box=box,nearest_actual_exterior_pixel=q,distance_pixels=dist,side=side,native_geometry_used=False,prior_image_chart_is_not_entity_authority=True);new.append(r);stats['confirmed_actual_field_side']+=1
    if target!=old['path_id']:stats['corrected_side']+=1
    changes.append(dict(row=row,pts=d['pts'],previous=old['path_id'],decision='include',current=target,evidence=r['physical_boundary_side_evidence']))
   else:
    reason='not_a_uniquely_visible_advertisement_panel_exterior';excluded.append(dict(row=row,stable_id=r['stable_id'],reason=reason,previous_native_entity=r['path_id'],direct_color_field_review=str(file),source_image=d['source_frame']['image']));changes.append(dict(row=row,pts=d['pts'],previous=r['path_id'],decision='exclude',reason=reason));stats['withdrawn_wrong_or_unconfirmed_panel_edge']+=1
  ids={r['row'] for r in todo};d['confirmed']=[r for r in d['confirmed'] if r['row'] not in ids]+new;d['excluded']+=excluded;d['counts']=dict(collections.Counter(r['reason'] for r in d['excluded']));d['counts'].update(confirmed=len(d['confirmed']),checked=len(d['confirmed'])+len(d['excluded']));d['boundary_identity_prior']=dict(path=str(archive),sha256=b.sha(archive));b.write(file,d)
 b.write(b.OUT/'boundary_identity_review.json',dict(counts=dict(stats),changes=changes,code_sha256=b.sha(__file__),native_geometry_changed=False));b.emit('boundary_identity_complete',counts=dict(stats))
if __name__=='__main__':run()
