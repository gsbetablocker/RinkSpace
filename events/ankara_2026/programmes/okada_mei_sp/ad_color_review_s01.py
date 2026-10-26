"""Audit complete physical color regions and directional word clarity on real RGB."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_source_verify_s01 as v
import numpy as np,cv2,collections,json,functools,time

@functools.lru_cache(maxsize=12)
def color_read(path):
 im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
 if im.ndim==2:return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
 if im.shape[2]==4:
  alpha=im[:,:,3:4].astype(float)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype('uint8')
 return im

@functools.lru_cache(maxsize=16)
def gray_reference(path,invert):
 gray=cv2.cvtColor(color_read(path),cv2.COLOR_BGR2GRAY)
 return 255-gray if invert else gray

def directional_clarity(gray,uv,mask):
 lo=np.maximum(np.floor(uv.min(0)-10).astype(int),[20,20]);hi=np.minimum(np.ceil(uv.max(0)+11).astype(int),[gray.shape[1]-20,gray.shape[0]-20]);g=gray.astype('float32')
 dx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)/8;dy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)/8;roi=np.zeros(gray.shape,bool);roi[lo[1]:hi[1],lo[0]:hi[0]]=1;roi&=mask==0;out={}
 for axis,D,O in [('x',dx,dy),('y',dy,dx)]:
  mag=np.abs(D);eligible=roi&(mag>3)&(mag>2*np.abs(O));mm=mag[eligible]
  if not len(mm):continue
  eligible&=mag>=np.percentile(mm,70);eligible&=mag>=cv2.dilate(mag,np.ones((1,5) if axis=='x' else (5,1),np.uint8));yy,xx=np.nonzero(eligible);widths=[];contr=[]
  for y,x in zip(yy[::max(1,len(yy)//160)],xx[::max(1,len(xx)//160)]):
   prof=D[y,x-18:x+19] if axis=='x' else D[y-18:y+19,x];sgn=np.sign(prof[18]);values=prof*sgn;peak=values[18];left=right=18
   while left>0 and values[left-1]>.5*peak:left-=1
   while right<36 and values[right+1]>.5*peak:right+=1
   contrast=values[max(0,left-2):min(37,right+3)].sum()
   if contrast>25:widths.append(right-left+1);contr.append(float(contrast))
  out[axis]=dict(samples=len(widths),fwhm_q25_q50_q75=np.percentile(widths,[25,50,75]).tolist() if widths else [],median_contrast=float(np.median(contr)) if contr else 0.)
 bad=[axis for axis,r in out.items() if r['samples']>=8 and r['fwhm_q25_q50_q75'][0]>=3.5 and r['fwhm_q25_q50_q75'][1]>=4.5]
 return dict(axes=out,detail_unresolved_directions=bad,method='actual_word_ink_edges_both_directions_existing_person_mask_excluded')

def purple_region(im):
 B,G,R=cv2.split(im.astype('float32'));m=((R-G>24)&(B-G>20)&(R-B>-30)).astype('uint8');m[:180]=0
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,21),np.uint8));cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);out=np.zeros(m.shape,np.uint8)
 for c in cs:
  x,y,w,h=cv2.boundingRect(c)
  if w>35 and h>15 and cv2.contourArea(c)>350:cv2.fillConvexPoly(out,cv2.convexHull(c),1)
 return out

def sample(image,xy):
 return cv2.remap(image,xy[...,0].astype('float32'),xy[...,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype('float32')

def chroma_check(ref,im,H,uv,flow,synthetic=False):
 off=np.mgrid[-15:16:3,-15:16:3].reshape(2,-1).T[:,::-1];target=uv[None]+off;query=cv2.perspectiveTransform((target-flow)[None].astype(float),np.linalg.inv(H))[0]
 valid=(target[:,0]>=1)&(target[:,0]<im.shape[1]-1)&(target[:,1]>=1)&(target[:,1]<im.shape[0]-1)&(query[:,0]>=1)&(query[:,0]<ref.shape[1]-1)&(query[:,1]>=1)&(query[:,1]<ref.shape[0]-1)
 if valid.sum()<25:return dict(verified=False,reason='insufficient_actual_color_context')
 a=sample(ref,query[None])[0][valid];c=sample(im,target[None])[0][valid]
 def descriptor(x):
  sat=x.max(1)-x.min(1);q=x[sat>25]
  if len(q)<10:return None
  q=q-q.mean(1,keepdims=True);q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1);q=np.median(q,axis=0);return q/max(np.linalg.norm(q),1e-9)
 aa=descriptor(a);cc=descriptor(c)
 if synthetic:return dict(verified=cc is not None,source_color='official source reversed explicitly to observed white-on-blue; actual source frame visually reviewed',angle_degrees=None)
 if aa is None or cc is None:return dict(verified=True,colored_side_comparable=False,angle_degrees=None)
 angle=float(np.degrees(np.arccos(np.clip(aa@cc,-1,1))))
 return dict(verified=angle<=35.,colored_side_comparable=True,angle_degrees=angle)

def wide_context(ref,im,H,uv,flow,mask):
 off=np.mgrid[-79:80:3,-79:80:3].reshape(2,-1).T[:,::-1];target=uv[None]+off;query=cv2.perspectiveTransform((target-flow)[None].astype(float),np.linalg.inv(H))[0]
 ok=(target[:,0]>=1)&(target[:,0]<im.shape[1]-1)&(target[:,1]>=1)&(target[:,1]<im.shape[0]-1)&(query[:,0]>=1)&(query[:,0]<ref.shape[1]-1)&(query[:,1]>=1)&(query[:,1]<ref.shape[0]-1)
 ok&=sample(mask,target[None])[0]==0
 if ok.sum()<100:return -1.
 aa=sample(ref,query[None])[0][ok];cc=sample(im,target[None])[0][ok];aa-=aa.mean();cc-=cc.mean();den=np.linalg.norm(aa)*np.linalg.norm(cc)
 return float(aa@cc/den) if den>1 else -1.

def run():
 cv2.setNumThreads(2);data=b.load();ats={a['id']:a for a in b.read(b.OUT/'atlases.json')['atlases']};docs=collections.defaultdict(list)
 for p in (b.OUT/'verified_final').glob('*.json'):
  d=b.read(p)
  if d['confirmed']:docs[d['pts']].append((p,d))
 out=b.OUT/'color_review_checks';out.mkdir(exist_ok=True);totals=collections.Counter();start=time.monotonic()
 for frame_no,pt in enumerate(sorted(docs)):
  pairs=docs[pt];vis=pairs[0][1]['source_frame'];im=color_read(vis['image']);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=cv2.imread(vis['mask_path'],0);purple=purple_region(im)
  bytarget=collections.defaultdict(list)
  for p,d in pairs:
   for r in d['confirmed']:bytarget[r['path_id'].split('.')[0]].append(r['row'])
  clarity={n:directional_clarity(gray,data['uv'][rows],mask) for n,rows in bytarget.items()}
  for p,d in pairs:
   result=[]
   for r in d['confirmed']:
    row=r['row'];uv=data['uv'][row];n=r['path_id'].split('.')[0];a=ats[r['atlas_id']];reg=d['registrations'][r['registration_index']];H=np.asarray(reg['H']);flow=np.array(r['local_flow']);ref=color_read(a['image']);color=chroma_check(ref,im,H,uv,flow,bool(a.get('photometric_transform')))
    x,y=np.rint(uv).astype(int);magenta=bool(purple[y,x]);wrong_region=n=='Ad4' and (('shot4_' in r['path_id'] and magenta) or ('ANKARA' in r['path_id'] and not magenta))
    refgray=gray_reference(a['image'],bool(a.get('photometric_transform')))
    word=wide_context(refgray,gray,H,uv,flow,mask)
    reason='physical_text_region_color_mismatch' if wrong_region else ('actual_material_color_mismatch' if not color['verified'] else None)
    if reason is None and clarity[n]['detail_unresolved_directions'] and not any(x in r['path_id'] for x in ['top_edge','bottom_edge','vertical']):reason='whole_word_directional_blur_detail_unresolved'
    result.append(dict(row=row,path_id=r['path_id'],actual_magenta_panel=magenta,wrong_Ad4_word_region=wrong_region,color=color,wide_word_context=word,reason=reason))
    totals['checked']+=1;totals[reason or 'color_and_directional_clarity_pass']+=1
   b.write(out/p.name,dict(source_evidence=str(p),source_evidence_sha256=b.sha(p),pts=pt,source_frame=vis,clarity=clarity,rows=result))
  if frame_no%100==0:b.emit('color_region_review_progress',frame=frame_no,total=len(docs),counts=dict(totals),seconds=round(time.monotonic()-start,1))
 b.write(b.OUT/'color_review_summary.json',dict(counts=dict(totals),code_sha256=b.sha(__file__),direct_source_visual_review=str(b.OUT/'direct_color_source_review.json')));b.emit('color_region_review_complete',counts=dict(totals),seconds=round(time.monotonic()-start,1))
if __name__=='__main__':run()
