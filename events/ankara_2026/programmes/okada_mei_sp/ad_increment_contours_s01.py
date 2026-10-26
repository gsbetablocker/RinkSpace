"""Use verified source-artwork identity only as search navigation; read current pixels."""
from pathlib import Path
import sys,json,gzip,collections,hashlib,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_increment_letters_s01 as letters
b=letters.b;OUT=letters.OUT
from rinkspace.vision_source_shapes import read_source_material_transition_profiles

def usable(reg):
    H=np.asarray(reg['H']);q=np.array([[0.,0.],[2400.,0.],[2400.,330.],[0.,330.],[1200.,165.]],np.float32)
    p=cv2.perspectiveTransform(q[None],H)[0]
    if not np.isfinite(p).all():return False
    scales=np.linalg.norm(np.diff(p[:4],axis=0),axis=1)/np.array([2400.,330.,2400.])
    if scales.min()<.05 or scales.max()>.9 or scales.max()/scales.min()>1.7:return False
    if np.cross(p[1]-p[0],p[3]-p[0])<=0:return False
    return reg.get('inliers',0)>=8 and reg.get('p95',100)<2.5

def measure():
    if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten; prepare a new increment')
    flow_mode='--source-flow' in sys.argv;tag='flow' if flow_mode else 'contour';flow=b.read(OUT/'source_flow_navigation.json') if flow_mode else None;cv2.setNumThreads(2);data=b.load();z=dict(np.load(OUT/'additions.npz'));old=np.load(b.OUT/'ads_only/point_observations.npz');mapping=b.read(OUT/'mapping.json');lookup={p:i for i,p in enumerate(mapping['path_ids'])}
    if any(str(s).startswith(f's01_ad7_{tag}_') for s in z['observation_ids']):raise ValueError('Already appended')
    atlas=b.read(b.OUT/'atlases.json')['atlases'][76];vis={f['pts']:f for f in b.read(b.OUT/'visibility.json')['frames']};accepted=[];summaries=[];pending=collections.Counter();start=time.monotonic()
    with gzip.open(OUT/('new_flow_contours.jsonl.gz' if flow_mode else 'new_visible_contours.jsonl.gz'),'wt',encoding='utf8') as log:
        for fi,pt in enumerate(sorted(map(int,np.unique(data['pts'][data['ad']==7])))):
            proof=b.OUT/'verified_final'/f'ad7_{pt}.json';evidence=b.read(proof);regs=[r for r in evidence['registrations'] if r['atlas_id'] in [79,80] and usable(r)]
            if flow_mode:
                if regs or str(pt) not in flow['records']:continue
                regs=[flow['records'][str(pt)]]
            if not regs:continue
            chosen={}
            for ai in [79,80]:
                opts=[r for r in regs if r['atlas_id']==ai]
                if opts:chosen[ai]=max(opts,key=lambda r:(r['inliers'], -r['p95']))
            f=vis[pt];im=cv2.imread(f['image']);mask=cv2.imread(f['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:raise ValueError('Source changed')
            safe=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,3)
            gray=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float),(3,3),.6);gx=cv2.Scharr(gray,cv2.CV_64F,1,0)/32;gy=cv2.Scharr(gray,cv2.CV_64F,0,1)/32
            queries=[];normals=[];identities=[]
            for path in atlas['paths']:
                pid='print_instance/Ad7::'+path['path_id']
                if pid not in lookup:continue
                ai=80 if '.bottom_' in pid else 79;reg=chosen.get(ai,chosen.get(79,chosen.get(80)));H=np.asarray(reg['H']);ref=np.asarray(path['pixels'],np.float32)
                contour=cv2.perspectiveTransform(ref[None],H)[0];s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(contour,axis=0),axis=1))]
                if s[-1]<5:continue
                ts=np.arange(1,s[-1]-1,6.);q=np.column_stack([np.interp(ts,s,contour[:,k]) for k in range(2)]);left=np.column_stack([np.interp(ts-1,s,contour[:,k]) for k in range(2)]);right=np.column_stack([np.interp(ts+1,s,contour[:,k]) for k in range(2)])
                tangents=right-left;normal=np.column_stack([-tangents[:,1],tangents[:,0]]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-10)
                near=np.concatenate([a['uv'][(a['pts']==pt)&(a['path_indices']==lookup[pid])] for a in [z,old]])
                for qi,(uv,n) in enumerate(zip(q,normal)):
                    x,y=np.rint(uv).astype(int)
                    if not (6<=x<1914 and 186<=y<1074) or safe[y,x]<6:continue
                    if len(near) and np.min(np.linalg.norm(near-uv,axis=1))<3.2:continue
                    patch=im[y-4:y+5,x-4:x+5].astype(float);B,G,R=cv2.split(patch)
                    if np.mean((B-R>15)&(B-G>8))<.12:continue
                    grad=np.array([gx[y,x],gy[y,x]]);gn=np.linalg.norm(grad)
                    if gn<1 or abs(np.dot(grad,n))/gn<.65:continue
                    if np.dot(grad,n)<0:n=-n
                    queries.append(uv);normals.append(n);identities.append(dict(path_id=pid,registration=reg,atlas_id=76,reference_contour_sha256=path['contour_sha256']))
            if not queries:continue
            queries=np.array(queries);normals=np.array(normals);binding=dict(source_sha256=f['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id='s01',event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080)
            result=read_source_material_transition_profiles(np.ascontiguousarray(im[:,:,::-1]),binding=binding,search_points_uv=queries,normals=normals,search_intervals_px=np.tile([-4.,4.],(len(queries),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.6,engineering_support_margin_px=.2)
            gained=0
            for qi,p in enumerate(result['readings']):
                if any(p['source_flags'].values()) or np.asarray(p['actual_source_uv']).shape!=(2,):pending['local_material_transition_unresolved']+=1;continue
                uv=np.asarray(p['actual_source_uv']);pid=identities[qi]['path_id'];x,y=np.rint(uv).astype(int)
                if safe[y,x]<4:continue
                near=np.concatenate([a['uv'][(a['pts']==pt)&(a['path_indices']==lookup[pid])] for a in [z,old]])
                if len(near) and np.min(np.linalg.norm(near-uv,axis=1))<2.7:continue
                sid=f's01_ad7_{tag}_{pt}_{qi:04d}'
                record=dict(stable_id=sid,source_row=int(max(data['pts'].size-1,z['source_rows'].max()))+1+len(accepted),pts=pt,uv=uv.tolist(),path_id=pid,source_video_sha256=f['source_video_sha256'],source_BGR_sha256=f['source_BGR_sha256'],source_image=f['image'],time_base='1/12800',method='actual_source_material_transition_with_existing_verified_artwork_identity',source_identity_proof=str(proof),source_identity_proof_sha256=b.sha(proof),identity=identities[qi],query_uv=queries[qi].tolist(),normal=normals[qi].tolist(),profile_offsets=np.asarray(p['source_q10_q50_q90_normal_offsets_px']).tolist(),native_geometry_used_for_identity_or_uv=False,query_projection_used_as_observation=False)
                accepted.append(record);log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');gained+=1
            summaries.append(dict(pts=pt,added=gained,identity_registrations=chosen));
            if fi%30==0:print(json.dumps(dict(frame=fi,new=len(accepted),seconds=round(time.monotonic()-start,1))),flush=True)
    n=len(accepted);a=dict(source_rows=np.array([r['source_row'] for r in accepted],np.int64),path_indices=np.array([lookup[r['path_id']] for r in accepted],np.int32),stations=np.full(n,.5),pts=np.array([r['pts'] for r in accepted],np.int64),uv=np.array([r['uv'] for r in accepted]).reshape(-1,2),kind=np.ones(n,np.uint8),observation_ids=np.array([r['stable_id'] for r in accepted]))
    np.savez_compressed(OUT/'additions.npz',**{k:np.concatenate([z[k],a[k]]) for k in z});report=dict(added_rows=n,total_increment_rows=len(z['pts'])+n,source_frames=len(set(r['pts'] for r in accepted)),by_path=dict(collections.Counter(r['path_id'] for r in accepted)),pending=dict(pending),per_frame=summaries,baseline_rewritten=False,native_geometry_used_for_identity_or_uv=False)
    b.write(OUT/('flow_contour_measurements.json' if flow_mode else 'visible_contour_measurements.json'),report);print(json.dumps({k:v for k,v in report.items() if k not in ('per_frame','by_path')}),flush=True)
if __name__=='__main__':measure()
