"""Original-image comparisons for the public review, including difficult source cases."""
from collections import defaultdict
import cv2
import numpy as np
from lane_end_evidence import BASE, WORK
from repair_motion import project
from review import ji,native_for,camera_state,source_frames
from shared import PREPARED, arrays, prepared_packet_path, read, write
from motion_delivery import FINAL,review_directory
from source_bundle import relative


def title(im,value):
    out=cv2.copyMakeBorder(im,25,0,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
    cv2.putText(out,value,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(238,238,238),1,cv2.LINE_AA)
    return out


def crop_box(uv,width,height):
    mid=np.median(uv,axis=0);x=int(np.clip(mid[0]-width//2,0,1920-width));y=int(np.clip(mid[1]-height//2,0,1080-height))
    return x,y


def curve_crop(image,poly,observed,box,color):
    x,y,w,h=box;c=image[y:y+h,x:x+w].copy()
    for uv,status in poly:
        good=(status==0)&np.isfinite(uv).all(1)&(np.max(abs(uv),axis=1)<1e6)
        ix=np.flatnonzero(good)
        for group in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
            if len(group)>1:cv2.polylines(c,[np.rint((uv[group]-[x,y])*16).astype('i4')],False,color,1,cv2.LINE_AA,4)
    for u,v in observed:cv2.circle(c,(round(u-x),round(v-y)),2,(35,45,225),1,cv2.LINE_AA)
    return c


def run():
    dest=FINAL/'inspection';dest.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json')['shots'];bound=read(FINAL/'source_bindings.json')
    candidates=[BASE,relative(FINAL)]
    states=[camera_state(p) for p in candidates];natives=[native_for(p,'jin') for p in candidates]
    summary={}
    motion_at={'jin':('jin_s10',78.44),'okada':('okada_s07',111.44),
               'chen':('chen_s14',155.06),'nayeon':('nayeon_s04',55.02)}
    for prog,(moving,clock) in motion_at.items():
        cases=defaultdict(list)
        for name,s in setup.items():
            if s['program']!=prog:continue
            a=arrays(prepared_packet_path(name))
            use=np.isin(a['material'],['paired_lane_end_source','real_paired_annulus_edges'])
            if not use.any():continue
            errors=[arrays(bound[name][k]['path'])['error'] for k in ('baseline_errors','matched_errors')]
            ix=np.flatnonzero(use&(a['fold']<=6))
            for path in np.unique(a['path'][ix]):
                take=ix[a['path'][ix]==path]
                for pt in np.unique(a['pts'][take]):
                    rr=take[a['pts'][take]==pt]
                    if len(rr)<5:continue
                    med=[float(np.median(e[rr])) for e in errors]
                    if not np.isfinite(med).all():continue
                    kind='lane' if a['material'][rr[0]]=='paired_lane_end_source' else 'annulus'
                    cases[kind].append((med[0]-med[1],med[1],name,int(pt),str(path),med))
        choice=[]
        for label,rows in cases.items():
            if rows:choice.append((label,max(rows,key=lambda v:v[0])))
        if cases['lane']:choice.append(('lane, high residual',max(cases['lane'],key=lambda v:v[1])))
        cards=[];records=[]
        for label,(_,_,name,pt,path,med) in choice:
            s=setup[name];a=arrays(prepared_packet_path(name));i=np.flatnonzero(states[0][name+'_pts']==pt).item()
            ids=(a['pts']==pt)&(a['path']==path)&np.isin(a['material'],['paired_lane_end_source','real_paired_annulus_edges'])&(a['fold']<=6)
            uv=a['uv'][ids];img=next(source_frames(s['source'],[pt]))[1];x,y=crop_box(uv,330,145);box=(x,y,330,145)
            pictures=[title(img[y:y+145,x:x+330].copy(),f'{name} {pt/12800:.3f}  {label}')]
            for j,(native,b) in enumerate(zip(natives,states)):
                w=ji.native_paths(native,[path],count=513)[path][0];r=b[name+'_rigs'][i]
                z=native.core.project_motion(r,b[name+'_controls'],s['first'],s['dt'],1080,np.full(len(w),b[name+'_times'][i]),w)
                c=curve_crop(img,[(z['uv'],z['status'])],uv,box,(40,180,235) if j==0 else (80,200,60))
                pictures.append(title(c,('Previous' if j==0 else 'Current')+f'  median {med[j]:.2f} px'))
            cards.append(np.hstack(pictures));records.append(dict(kind=label,shot=name,pts=pt,path=path,rows=int(ids.sum()),median_px=med))
        cv2.imwrite(str(dest/(prog+'_geometry.jpg')),np.vstack(cards))
        s=setup[moving];z=arrays(WORK/'texture'/(moving+'.npz'));q=arrays(WORK/'texture/quality'/(moving+'.npz'))
        i=np.flatnonzero(z['pts']==round(clock*12800)).item();motion=[];records_motion=[]
        for j in (i-1,i,i+1):
            rr=np.flatnonzero(z['valid']&q['keep']&z['held']&(z['index']==j)&(z['lag']==1))
            if not len(rr):continue
            pt=int(z['pts'][j]);w=z['world'][rr];next_index=np.full(len(rr),j)
            pred=[z['left_uv'][rr]+project(n,s,b,moving,next_index,w)-project(n,s,b,moving,next_index-1,w) for n,b in zip(natives,states)]
            error=np.stack([np.linalg.norm(u-z['uv'][rr],axis=1) for u in pred]);k=np.argmin(abs(error[0]-np.median(error[0])))
            actual=z['uv'][rr[k]];x,y=crop_box([actual],165,78);img=next(source_frames(s['source'],[pt]))[1][y:y+78,x:x+165]
            pictures=[];points=[actual,pred[0][k],pred[1][k]];labels=['Measured source patch','Previous displacement','Current displacement']
            for point,label,color in zip(points,labels,[(70,215,90),(40,180,235),(65,210,110)]):
                c=cv2.resize(img,(330,156),interpolation=cv2.INTER_LINEAR)
                uv=np.rint((point-[x,y])*2+.5).astype(int);cv2.drawMarker(c,uv,color,cv2.MARKER_CROSS,15,1,cv2.LINE_AA)
                pictures.append(title(c,label+f'  {pt/12800:.3f}'))
            motion.append(np.hstack(pictures));records_motion.append(dict(pts=pt,source_row=int(rr[k]),actual=actual.tolist(),
                previous_px=float(error[0,k]),current_px=float(error[1,k]),held_patches=len(rr)))
        cv2.imwrite(str(dest/(prog+'_motion.jpg')),np.vstack(motion))
        summary[prog]=dict(geometry=records,motion=records_motion)
    write(dest/'examples.json',summary)


def contact_sheets():
    dest=FINAL/'inspection';dest.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json')['shots'];b=camera_state(relative(FINAL));summary={}
    for prog in ('jin','okada','chen','nayeon'):
        folder=review_directory(prog);rows=[];pages=[]
        names=[n for n,s in setup.items() if s['program']==prog]
        for n in names:
            pts=b[n+'_pts'];tiles=[]
            for j in (0,len(pts)//2,-1):
                pt=int(pts[j]);im=cv2.imread(str(folder/(n+'_'+str(pt)+'.jpg')))
                if im is None:raise ValueError((n,pt,'missing full render still'))
                thumb=cv2.resize(im,(340,152),interpolation=cv2.INTER_AREA)
                tiles.append(title(thumb,f'{n}   {pt/12800:.3f} s'))
            rows.append(np.hstack(tiles))
            if len(rows)==6 or n==names[-1]:
                path=dest/(prog+'_shots_'+str(len(pages)+1)+'.jpg')
                cv2.imwrite(str(path),np.vstack(rows));pages.append(str(path));rows=[]
        summary[prog]=dict(shots=len(names),screens=3*len(names),pages=pages)
    write(dest/'shot_contacts.json',summary)


if __name__=='__main__':run()
