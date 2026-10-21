exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_footwear_v1/build.py',encoding='utf8').read().split('def main():')[0])
with av.open(source['path'])as c:
 st=c.streams.video[0];c.seek(1235200,stream=st,backward=True)
 for fr in c.decode(st):
  if int(fr.pts)<1235200:continue
  if int(fr.pts)>1235200:break
  im=fr.to_ndarray(format='bgr24');m=cv2.imread(str(O/'clean_parent52_v2/masks/1235200.png'),0);guard=cv2.dilate((m>0).astype('u1'),np.ones((17,17),'u1'));cnt,_=cv2.findContours(guard,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,255),1);cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,0,0),1)
  x0,y0,x1,y1=750,500,930,740
  for x in range(760,930,20):cv2.line(im,(x,y0),(x,y1),(0,190,210),1);cv2.putText(im,str(x),(x+1,y0+12),0,.3,(0,0,255),1)
  for y in range(500,741,20):cv2.line(im,(x0,y),(x1,y),(0,190,210),1);cv2.putText(im,str(y),(x0+1,y-2),0,.3,(0,0,255),1)
  cv2.imwrite(str(O/'clean52_footwear_delta/1235200_freeboot_native.jpg'),cv2.resize(im[y0:y1,x0:x1],(540,720)),[cv2.IMWRITE_JPEG_QUALITY,99])
