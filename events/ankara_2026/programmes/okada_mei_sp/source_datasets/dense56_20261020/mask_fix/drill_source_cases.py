exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/audit_static.py',encoding='utf8').read().split('mode=sys.argv')[0])
O=O/'background_v1';rs=rz(O/'instances.jsonl.gz');wanted={108,112,122,125};cells=[];out=[]
for i,r in enumerate(rs):
 if r['tracklet_id']not in wanted:continue
 pg=cv2.imread(str(O/'QA'/f'page_{i//12:03}.jpg'));k=i%12;cell=pg[(k//4)*280:(k//4+1)*280,(k%4)*360:(k%4+1)*360].copy();cells.append(cell)
 if len(cells)==12:
  f=O/f'drill_{len(out):02}.jpg';cv2.imwrite(str(f),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8)]),[cv2.IMWRITE_JPEG_QUALITY,98]);out.append(str(f));cells=[]
if cells:
 cells+=[np.full((280,360,3),240,'u1')for _ in range(12-len(cells))];f=O/f'drill_{len(out):02}.jpg';cv2.imwrite(str(f),np.vstack([np.hstack(cells[k:k+4])for k in(0,4,8)]),[cv2.IMWRITE_JPEG_QUALITY,98]);out.append(str(f))
print(json.dumps(out))
