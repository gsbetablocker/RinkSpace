exec(open(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense56_v1/shoe_fix/audit_static.py',encoding='utf8').read().split('mode=sys.argv')[0])
b=rz(O/'background_v1/instances.jsonl.gz');old={q['native_pts']:q for q in rz(R/'dense56_footwear_v1/index.jsonl.gz')};ans=[]
for r in b:
 if r['tracklet_id']!=84:continue
 p=r['native_pts'];m=cv2.imread(r['mask_file'],0)>0;base=cv2.imread(old[p]['mask_file'],0)>0;n,la,stats,_=cv2.connectedComponentsWithStats(m.astype('u1'),8);ans.append(dict(native_pts=p,components=[dict(component=j,bbox_xywh=stats[j,:4].tolist(),pixels=int(stats[j,4]),old_person_overlap=int(((la==j)&base).sum()))for j in range(1,n)]))
print(json.dumps(ans,indent=2));(O/'background_v1/T84_old_pixel_check.json').write_text(json.dumps(ans,indent=2),encoding='utf8')
