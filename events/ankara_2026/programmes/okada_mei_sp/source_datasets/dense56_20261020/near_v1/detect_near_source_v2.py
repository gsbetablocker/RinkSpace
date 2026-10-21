from pathlib import Path
import av,cv2,numpy as np,json,gzip,hashlib,sys,time,math
O=Path(sys.argv[1]);N=O/'near_board_v1';cfg=json.load(open(O/'consumer_s06.json',encoding='utf8'));src=cfg['source']['file'];sys.path.insert(0,str(O));from source_graphics import current_graphics
cv2.setNumThreads(2)
rows={r['native_pts']:r for r in map(json.loads,gzip.open(cfg['mask_index']['file'],'rt',encoding='utf8')) if r['shot_id']=='s06'}
def scan(b,m,bb):
 g=cv2.GaussianBlur(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),1.1)
 n=np.array([1.,-m]);n/=np.linalg.norm(n);t=np.array([m,1.]);t/=np.linalg.norm(t)
 pts=[];ys=range(310,1074,12)
 for y in ys:
  x=m*y+bb
  if x<12 or x>620:continue
  ds=np.arange(-40.,40.01,.5);xy=np.array([x,y])+ds[:,None]*n
  xx=np.broadcast_to(xy[:,0][None,:],(5,len(ds))).astype(np.float32);yy=(xy[:,1][None,:]+np.arange(-2,3)[:,None]).astype(np.float32)
  vals=cv2.remap(g,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(0)
  lo=float(np.median(vals[(ds<-17)&(ds>-30)]));hi=float(np.median(vals[(ds>17)&(ds<30)]));contrast=hi-lo
  if not(12<contrast and 60<lo<150 and hi>140):continue
  center=np.where((ds>-15)&(ds<15))[0];grad=np.gradient(vals);j=int(center[np.argmax(grad[center])]);a=max(0,j-50);z=min(len(ds)-1,j+50)
  if grad[j]<.5:continue
  crossings=[]
  for q in [.1,.5,.9]:
   target=lo+q*contrast;cs=[k for k in range(a,z)if vals[k]<=target<vals[k+1]]
   if not cs:break
   k=min(cs,key=lambda k:abs(k-j));crossings.append(float(ds[k]+(target-vals[k])/(vals[k+1]-vals[k])*.5))
  if len(crossings)<3 or not(crossings[0]<=crossings[1]<=crossings[2]) or crossings[2]-crossings[0]>27:continue
  uv=np.array([x,y])+crossings[1]*n
  pts.append(dict(uv=uv.tolist(),normal=n.tolist(),interval=[(np.array([x,y])+crossings[0]*n).tolist(),(np.array([x,y])+crossings[2]*n).tolist()],contrast=contrast,gray_plateaus=[lo,hi],profile_normal_offsets=crossings))
 return pts
def detect(b,guard):
 crop=b[280:1080,:650];gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(0,0),1.1);edge=cv2.Canny(gray,8,20);edge[guard[280:1080,:650]]=0
 h=cv2.HoughLinesP(edge,1,np.pi/1800,30,minLineLength=45,maxLineGap=25)
 cand=[]
 if h is not None:
  for seg in h[:,0]:
   x1,y1,x2,y2=map(float,seg);y1+=280;y2+=280
   if abs(y2-y1)<45:continue
   m=(x2-x1)/(y2-y1);bb=x1-m*y1
   if not(.2<m<.85 and 0<m*1075+bb<650 and 280<-bb/m<1080):continue
   if any(abs(m-q['m'])<.025 and abs((m*800+bb)-(q['m']*800+q['b']))<8 for q in cand):continue
   pts=scan(b,m,bb)
   if len(pts)<4:continue
   ya=np.array([p['uv'][1]for p in pts]);span=float(ya.max()-ya.min());q=dict(m=m,b=bb,points=pts,span=span,score=span*len(pts)/max(1,(1074-max(310,-bb/m))/12))
   if span<75 or ya.max()<1000:continue
   cand.append(q)
 return sorted(cand,key=lambda q:q['score'],reverse=True),edge
all=[];thumb=[];c=av.open(src);s=c.streams.video[0];c.seek(1301248,stream=s,backward=True);start=time.perf_counter()
for f in c.decode(s):
 if f.pts<1301248:continue
 if f.pts>1425408:break
 b=f.to_ndarray(format='bgr24');rr=rows[f.pts];assert hashlib.sha256(b.tobytes()).hexdigest()==rr['source_BGR_sha256'];person=cv2.imread(rr['mask_file'],0)>0;guard=cv2.dilate(person.astype('uint8'),np.ones((17,17),np.uint8))>0;guard|=current_graphics(b)[0]
 cand,edge=detect(b,guard);best=cand[0]if cand else None
 all.append(dict(native_pts=f.pts,detected=best is not None,best=best,alternatives=[{k:q[k]for k in('m','b','span','score')}for q in cand[1:3]]))
 if best and len(all)%4==0:
  im=b.copy()
  for p in best['points']:cv2.circle(im,tuple(np.round(p['uv']).astype(int)),3,(0,255,0),-1)
  im=cv2.resize(im,(640,360));cv2.putText(im,str(f.pts),(10,345),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);thumb.append((f.pts,im))
 if len(all)%100==0:print(json.dumps({'done':len(all),'detected':sum(x['detected']for x in all),'elapsed':time.perf_counter()-start}),flush=True)
json.dump(all,open(N/'detect_probe.json','w',encoding='utf8'),ensure_ascii=False)
for page in range((len(thumb)+5)//6):
 canvas=np.zeros((1080,1280,3),np.uint8)
 for j,(p,im)in enumerate(thumb[page*6:page*6+6]):canvas[(j//2)*360:(j//2+1)*360,(j%2)*640:(j%2+1)*640]=im
 cv2.imwrite(str(N/f'detect_page_{page:02d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,91])
print(json.dumps({'frames':len(all),'visible':sum(x['detected']for x in all),'first':next((x['native_pts']for x in all if x['detected']),None),'last':next((x['native_pts']for x in all[::-1]if x['detected']),None),'thumbs':len(thumb)}),flush=True)
