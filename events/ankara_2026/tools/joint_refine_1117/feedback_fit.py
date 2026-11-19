"""Prepare the reviewed source observations and fit one geometry for all four programmes."""
from collections import defaultdict
import shutil

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from feedback_source import CASES, DEST, PARENT, selected_paths, read, save, write
from review import camera_state, ji, native_for, source_frames
from shared import OUT, PREPARED, PROGRAMMES, arrays, prepared_packet_path
from source_paint import matte
from lane_end_evidence import foreground, WORK
from source_bundle import native_run, relative, flow

RINGS=('F4','R4','goal_crease_left')
PRIOR=WORK/'checked'


def extreme(poly,stations,direction):
    ix=int(np.argmax(poly[:,0]*direction))
    if ix==0 or ix==len(stations)-1:
        if np.linalg.norm(poly[0]-poly[-1])>1:return None
        ix=0
        ids=np.array([len(stations)-2,0,1])
    else:ids=np.arange(ix-1,ix+2)
    x=poly[ids,0]*direction
    c=(x[0]+x[2])/2-x[1]
    u=np.clip((x[0]-x[2])/(4*c),-.5,.5) if c<0 else 0.
    st=(stations[ix]+u*(stations[1]-stations[0]))%1.
    q=poly[ix]+u*.5*(poly[ids[2]]-poly[ids[0]])+.5*u*u*(poly[ids[2]]+poly[ids[0]]-2*poly[ix])
    return st,q


def measure_tip(img,mask,poly,station,shift,direction,sign):
    loc=extreme(poly,station,direction)
    if loc is None:return None
    st,model=loc
    x0,y0=model+shift
    if not 12<x0<1907 or not 12<y0<1067:return None
    yy=np.arange(max(3,int(y0)-21),min(1076,int(y0)+22))
    poly=poly+shift
    found=[]
    for y in yy:
        segments=np.flatnonzero(((poly[:-1,1]<=y)&(poly[1:,1]>y))|((poly[1:,1]<=y)&(poly[:-1,1]>y)))
        if not len(segments):continue
        f=(y-poly[segments,1])/(poly[segments+1,1]-poly[segments,1])
        xx=poly[segments,0]+f*(poly[segments+1,0]-poly[segments,0])
        px=xx.max() if direction>0 else xx.min()
        if abs(px-x0)>68:continue
        radius=48
        xs=np.arange(max(0,round(px)-radius-12),min(1920,round(px)+radius+13))
        if len(xs)<40:continue
        rgb=img[y-1:y+2,xs].mean(0)
        lum=rgb@np.array([.114,.587,.299])
        v=gaussian_filter1d(lum,2.)
        g=np.gradient(v)*direction*sign
        peaks=np.flatnonzero((g[1:-1]>=g[:-2])&(g[1:-1]>g[2:])&(g[1:-1]>.34))+1
        best=None
        for k in peaks[(peaks>8)&(peaks<len(xs)-9)]:
            if mask[y-2:y+3,xs[k]-6:xs[k]+7].any():continue
            strength=(v[k+3:k+8].mean()-v[k-7:k-2].mean())*direction*sign
            if strength<4 or min(v[k-6],v[k+6])<111:continue
            score=min(35.,strength)/(1+abs(xs[k]-px)*.047)
            if best is None or score>best[0]:
                sub=np.clip((g[k+1]-g[k-1])/(2*max(.001,2*g[k]-g[k-1]-g[k+1])),-.55,.55)
                best=(score,xs[k]+sub,strength)
        if best:found.append([y-y0,best[1],best[2]])
    if len(found)<7:return None
    y,x,contrast=np.asarray(found).T
    A=np.c_[np.ones(len(y)),y,y*y];ok=np.ones(len(y),bool)
    for _ in range(4):
        if ok.sum()<7:return None
        coeff=np.linalg.lstsq(A[ok],x[ok],rcond=None)[0]
        resid=abs(x-A@coeff)
        ok=resid<max(2.1,2.8*np.median(resid[ok]))
    if ok.sum()<7 or direction*coeff[2]>=-.004:return None
    vy=-coeff[1]/(2*coeff[2]);vx=coeff@[1.,vy,vy**2]
    if (abs(vx-x0)>35 or abs(vy)>17 or (y[ok]-vy).min()>-2 or (y[ok]-vy).max()<2
            or np.median(resid[ok])>2.2):return None
    return st,np.array([vx,y0+vy]),model,float(np.median(contrast[ok])),np.c_[x[ok],y0+y[ok]]


def bind_extrema(native,state,setup,rows,extent):
    tab=ji.native_paths(native,np.unique(rows['path'][extent]),count=513)
    world={k:v[0] for k,v in tab.items()}
    for name in np.unique(rows['shot'][extent]):
        s=setup['shots'][name];rr=np.flatnonzero(extent&(rows['shot']==name))
        t=state[name+'_times']
        groups=defaultdict(list)
        for r in rr:groups[int(rows['idx'][r]),rows['path'][r]].append(r)
        jobs=[(i,k,t[i]) for i,k in groups]
        curves=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],jobs,world)
        for ((i,k),ids),(uv,status) in zip(groups.items(),curves):
            if np.any(status!=0):raise ValueError(('Unprojectable outline',name,i,k))
            for r in ids:
                material=str(rows['material'][r])
                axis=1 if material.endswith(('ymin','ymax')) else 0
                side=1 if material.endswith('max') else -1
                oriented=uv if axis==0 else uv[:, ::-1]
                hit=extreme(oriented,tab[k][1],side)
                if hit is None:
                    if not material.startswith('ink_extent_'):
                        raise ValueError(('Missing interior outline extremum',name,i,k,material))
                    # A finite printed stroke can terminate at its extremum.
                    station=tab[k][1][int(np.argmax(oriented[:,0]*side))]
                else:
                    station,_=hit
                rows['station'][r]=station


def tips(name,state,native,s,dense=False):
    if dense:
        audit=arrays(DEST/'end_sweep'/(name+'.npz'))
        labels=np.asarray([native.paths[k]['label'][4:].split(':')[0] for k in audit['keys']])
        a=dict(idx=audit['index'],path=audit['keys'][audit['path']],family=labels[audit['path']],shift=audit['shift'])
    else:
        a=arrays(DEST/'reads'/(name+'.npz'))
    if 'family' not in a:return []
    ids=np.flatnonzero(np.isin(a['family'],RINGS))
    if not len(ids):return []
    tab=ji.native_paths(native,np.unique(a['path'][ids]),count=513)
    t=state[name+'_times'];pts=state[name+'_pts'];packed=matte(s,pts)
    jobs=[];shifts={}
    for i in np.unique(a['idx'][ids]):
        rows=ids[a['idx'][ids]==i]
        for path in np.unique(a['path'][rows]):
            rr=rows[a['path'][rows]==path]
            if len(rr)<(8 if dense else 15):continue
            jobs.append((i,path,t[i]));shifts[i,path]=a['shift'][rr[0]]
    if not jobs:return []
    world={k:z[0] for k,z in tab.items()}
    by_frame=defaultdict(list)
    for start in range(0,len(jobs),128):
        slab=jobs[start:start+128]
        curves=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],s['first'],s['dt'],slab,world)
        for (i,k,_),(uv,status) in zip(slab,curves):
            if np.all(status==0) and np.isfinite(uv).all():by_frame[i].append((k,uv))
    rows=[]
    if not by_frame:return rows
    indices=sorted(by_frame)
    for i,(pt,img) in zip(indices,source_frames(s['source'],pts[indices]),strict=True):
        mask=foreground(s,packed,i,pt)
        if mask is None:continue
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8));mask[:270,:492]=255;mask[936:,1670:]=255
        for k,uv in by_frame[i]:
            label=native.paths[k]['label'];fam=label[4:].split(':')[0]
            for side in (-1,1):
                hit=measure_tip(img,mask,uv,tab[k][1],shifts[i,k],side,-1 if label.endswith('middle') else 1)
                if hit:
                    st,xy,model,contrast,pixels=hit
                    rows.append(dict(idx=i,pts=pt,path=k,station=st,uv=xy,guide=model,normal=[side,0.],
                        contrast=contrast,shift=xy-model,family=fam,material='outline_xmin' if side<0 else 'outline_xmax',
                        pixels=pixels))
    return rows


def extract_tips(programme,dense=False):
    state=camera_state(PARENT);native=native_for(PARENT,programme)
    for name,s in read(PREPARED/'setup.json')['shots'].items():
        if s['program']!=programme:continue
        if dense and s['camera_index']!=2:continue
        out=DEST/('dense_tips' if dense else 'tips')/(name+'.npz')
        if out.exists():continue
        rows=tips(name,state,native,s,dense)
        d={k:np.asarray([z[k] for z in rows]) for k in rows[0] if k!='pixels'} if rows else {}
        pixels=[z['pixels'] for z in rows]
        save(out,**d,pixel_offsets=np.r_[0,np.cumsum([len(z) for z in pixels])],
             edge_pixels=np.concatenate(pixels) if pixels else np.empty((0,2)),source_sha256=s['source_sha256'])
        print(name,'source extrema',len(rows),flush=True)


def admit(name,s,new,is_tip):
    if 'pts' not in new:return np.empty(0,np.int8)
    role=np.full(len(new['pts']),9,np.int8)
    windows=CASES.get(name,[])
    for fam in np.unique(new['family']):
        use=np.flatnonzero(new['family']==fam)
        frames,first,num=np.unique(new['idx'][use],return_index=True,return_counts=True)
        times=new['pts'][use[first]]/12800.
        shifts=new['shift'][use[first]]
        for f,(i,pt) in enumerate(zip(frames,times)):
            near=[abs(pt-sec) for sec,group in windows if fam in group]
            marked=bool(near and min(near)<1.15)
            rr=use[new['idx'][use]==i]
            if is_tip:
                if s['camera_index']!=2 and not marked:continue
                valid=rr
            elif fam=='board':
                if not marked:continue
                keep=np.zeros(len(rr),bool)
                for path in np.unique(new['path'][rr]):
                    kk=np.flatnonzero(new['path'][rr]==path);r=rr[kk]
                    dist=np.sum((new['uv'][r]-new['guide'][r])*new['normal'][r],axis=1)
                    for j in range(len(r)):
                        if np.linalg.norm(new['guide'][r[j]]-new['guide'][r[max(0,j-1)]])>95:continue
                        neighbour=(abs(np.arange(len(r))-j)<=4)&(abs(np.arange(len(r))-j)>0)
                        keep[kk[j]]=np.count_nonzero(neighbour&(abs(dist-dist[j])<3.7))>=2
                valid=rr[keep]
            else:
                peers=(abs(frames-i)<=95)&(frames!=i)
                if fam in RINGS or fam.startswith('Ad'):
                    peers&=(abs(shifts[:,0]-shifts[f,0])<(35 if fam in RINGS else 13))&(abs(shifts[:,1]-shifts[f,1])<8)
                # Additional unrelated angles supply checks on the earlier
                # full packet. Admit new automatic marks only when supported.
                if not marked and (not peers.any() or num[f]<(25 if fam in RINGS else 12)):continue
                if fam.startswith('Ad') and not marked:continue
                if fam in RINGS and not marked and s['camera_index']!=2:continue
                if fam in ('B1','C0') and not marked:continue
                valid=rr
            # Hold the marked exposure and a narrow surrounding interval for
            # comparison. Those shots already contributed other original points.
            held=bool(near and min(near)<.049)
            if not marked and s['camera_index']==2 and name in ('chen_s13','jin_s14'):held=True
            role[valid]=6 if held else 0
    return role


def prepare():
    native=native_for(PARENT,'jin');base=camera_state(PARENT)
    setup=read(PREPARED/'setup.json');folder=DEST/'prepared';folder.mkdir(exist_ok=True)
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    audit={}
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);s['observations']=str(old)
        a=None;entries=[]
        for kinds in ('reads','tips'):
            path=DEST/kinds/(name+'.npz');q=arrays(path);is_tip=kinds=='tips'
            roles=admit(name,s,q,is_tip);rows=np.flatnonzero(roles<9)
            if not len(rows):continue
            if a is None:a=arrays(old)
            if q['source_sha256'].item()!=s['source_sha256']:raise ValueError(name)
            code=max(map(int,s['source_packets']))+1
            fam=q['family'][rows];xy=q['uv'][rows]
            instance=fam.copy().astype('U24')
            board=fam=='board'
            instance[board]=np.where(np.char.endswith(q['path'][rows][board],'bottom'),'board.bottom','board.top')
            if is_tip:material=q['material'][rows]
            else:material=np.select([board,np.isin(fam,RINGS),np.isin(fam,['B1','C0'])],
                ['review_rail_edge','review_ice_contour','review_paint_midpoint'],default='review_print_edge')
            ext=dict(pts=q['pts'][rows],idx=q['idx'][rows],uv=xy,station=q['station'][rows],path=q['path'][rows],
                family=instance,material=material,fold=roles[rows],source_fold=roles[rows],
                station_fixed=False,source_packet=code,source_row=rows,baseline_display=False,
                origin_row=-(60000000 if is_tip else 50000000)-rows,
                admission_reason='native_review_material',restored_admission_reason='native_review_material')
            for key,col in a.items():
                val=np.broadcast_to(ext[key],(len(rows),)+col.shape[1:])
                a[key]=np.concatenate((col,val.astype(col.dtype if col.dtype.kind!='U' else str)))
            s['source_packets'][str(code)]=dict(path=str(path),source_row='source measurement row',
                source_sha256=s['source_sha256'])
            entries.append(dict(path=str(path),sampled_exposures=len(q.get('sampled',np.unique(q.get('pts',[])))),
                eligible=int((roles==0).sum()),check=int((roles==6).sum()),unused=int((roles==9).sum())))
        if a is not None:
            target=folder/(name+'.npz');save(target,**a);s['observations']=str(target)
        audit[name]=dict(parent=str(old),measurements=entries)
    setup['parent']=str(PREPARED);setup['guide_state']=str(PRIOR/'state.npz')
    write(folder/'setup.json',setup);write(folder/'source_audit.json',audit)
    bind=read(PRIOR/'source_bindings.json')
    for name,entry in bind.items():
        with np.load(entry['matched_errors']['path']) as z:
            save(DEST/'stations'/(name+'.npz'),station=z['station'])
    write(DEST/'weights.json',{'review_rail_edge':2.2,'review_ice_contour':2.6,'review_paint_midpoint':3.4,
        'review_print_edge':2.5,'outline_xmin':5.5,'outline_xmax':5.5})


def geometry(branch):
    ice='scene/ankara_2026.field/ice_tolerance'
    blocks=[ice]
    if branch=='houses':
        fam=['F4','R4','lane4','house_connection_F4_R4','F3','R3','lane3','F2','R2','lane2','F1','R1','lane1']
        coords=['row_gap/2','row_centre/0','F_column_position/3','R_column_position/3','FR_row_alignment/3',
                'house_ring_radius/14','house_ring_radius/15','house_ring_radius/30','house_ring_radius/31',
                'lane_width_at_F/3','lane_width_at_R/3']
    elif branch=='crease':
        fam=['goal_crease_left','goal_line_left','lane1','lane2','lane3','lane4',
             'goal_trapezoid_left_lower_outer','goal_trapezoid_left_upper_outer']
        coords=['crease_semiaxes/0','crease_semiaxes/1','hockey_carrier_position/3','hockey_carrier_slope/3']
    elif branch=='hockey':
        fam=['C','C0','B1']
        coords=['hockey_circle_radius/0','hockey_carrier_position/0','hockey_carrier_slope/0',
                'hockey_carrier_position/1','hockey_carrier_slope/1']
    elif branch=='print':
        fam=['Ad1','Ad4','Ad7','Ad8','Ad9','Ad10','Ad11','Ad12','Ad24','Ad25','Ad26']
        blocks=['installation/'+k+'/shared_placement' for k in fam]
        coords=[]
    else:raise ValueError(branch)
    args=['--mode','scene','--families',*fam,'--scene-blocks',*blocks,
          '--unit-weights','--material-weights',DEST/'weights.json','--station-seed-dir',DEST/'stations']
    if coords:args+=['--ice-coordinates',*coords]
    native_run(DEST/'geometry'/branch,PARENT,args)


def stripe_evidence(programme):
    from feedback_source import read_stripe
    cv2.setNumThreads(1)
    setup=read(PREPARED/'setup.json')['shots'];state=camera_state(PARENT)
    for name,s in setup.items():
        if s['program']!=programme:continue
        output=DEST/'stripes'/(name+'.npz')
        if output.is_file():continue
        a=arrays(DEST/'reads'/(name+'.npz'));data=defaultdict(list)
        if 'family' in a:ids=np.flatnonzero(np.isin(a['family'],['C0','B1']))
        else:ids=[]
        if len(ids):
            pts=state[name+'_pts'];m=matte(s,pts);ii=np.unique(a['idx'][ids])
            for i,(pt,img) in zip(ii,source_frames(s['source'],pts[ii]),strict=True):
                mask=foreground(s,m,i,pt)
                if mask is None:continue
                mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
                mask[:270,:492]=255;mask[936:,1670:]=255
                subset=ids[a['idx'][ids]==i]
                for fam in np.unique(a['family'][subset]):
                    rr=subset[a['family'][subset]==fam];path=a['path'][rr[0]]
                    if len(rr)<8:continue
                    part=(path,a['guide'][rr],a['normal'][rr],a['station'][rr],1)
                    hits=read_stripe(img,mask,[part],fam,True)
                    if len(hits)<8:continue
                    st=np.asarray([h[1] for h in hits]);uv=np.array([h[2] for h in hits])
                    guide=np.array([h[3] for h in hits]);norm=np.array([h[4] for h in hits])
                    offset=np.sum((uv-guide)*norm,axis=1)
                    val=(st-st.mean())/max(.0001,np.ptp(st))
                    A=np.c_[np.ones(len(st)),val,val**2] if fam=='B1' else np.c_[np.ones(len(st)),np.sin(2*np.pi*st),np.cos(2*np.pi*st)]
                    ok=np.ones(len(st),bool)
                    for _ in range(5):
                        if ok.sum()<7:break
                        resid=abs(offset-A@np.linalg.lstsq(A[ok],offset[ok],rcond=1e-6)[0])
                        ok=resid<max(2.2,min(9.,2.7*np.median(resid[ok])))
                    if ok.sum()<7 or np.linalg.norm(np.ptp(uv[ok],axis=0))<80:continue
                    for loc in np.flatnonzero(ok):
                        h=hits[loc];br=h[7];width=np.linalg.norm(br[1]-br[0])
                        if h[5]<3.1 or width<3.:continue
                        for k,v in dict(pts=pt,idx=i,path=path,station=h[1],uv=h[2],guide=h[3],normal=h[4],
                                contrast=h[5],bracket=br,family=fam,material='review_verified_stripe').items():data[k].append(v)
        save(output,**{k:np.asarray(v) for k,v in data.items()},source_sha256=s['source_sha256'])
        print(name,len(data.get('pts',[])),flush=True)


def seed():
    from shared import NativeSceneGPU, ParameterState, numeric_sha, sha
    from rinkspace.core.native import NativeError
    state=arrays(PRIOR/'state.npz');original=state['scene_values'];delta=np.zeros_like(original)
    branches={};native=native_for(PARENT,'jin')
    for branch in ('houses','crease','hockey','print'):
        folder=DEST/'geometry'/branch
        assert (folder/'report.json').is_file()
        candidate=arrays(folder/'state.npz')['scene_values']
        changes=candidate-original
        if np.any((changes!=0)&(delta!=0)):raise ValueError('Overlapping geometry directions')
        delta+=changes;branches[branch]=sha(folder/'state.npz')
    # Each block solve has its own coupled lane/goal feasibility checks.
    # Recheck the common step at their shared finite intersection.
    for scale in (1.,.998,.99,.97,.94,.90,.85,.75,.5):
        values=original+scale*delta
        try:
            NativeSceneGPU(native.scene_path,native.core,parameter_state=ParameterState(native.parameter_state.layout,values))
            break
        except NativeError:
            continue
    else:raise ValueError('No feasible combined step')
    state['scene_values']=values;state['scene_values_sha256']=numeric_sha(values)
    save(DEST/'scene/state.npz',**state)
    write(DEST/'scene/inputs.json',dict(parent=PARENT,geometry_branches=branches,combined_fraction=scale))
    shots=read(PREPARED/'setup.json')['shots']
    special={'jin_s07':'zoom_texture','jin_s10':'blue_texture','chen_s20':'corner_texture'}
    write(DEST/'texture/sources.json',{n:str(WORK/special.get(n,'texture')) for n in shots})
    weights=read(WORK/'blue_corner_precision.json')
    weights.update(read(DEST/'weights.json'))
    write(DEST/'camera_weights.json',weights)


def cameras(programme,camera,stage='mount',parent=None,names=None):
    setup=read(PREPARED/'setup.json')['shots']
    names=names or [n for n,s in setup.items() if s['program']==programme and s['camera_index']==camera]
    parent=parent or relative(DEST/'scene')
    opts=['--mode','camera','--shots',*names,'--stride',22,'--unit-weights','--huber-delta',4.,
          '--material-weights',DEST/'camera_weights.json','--station-seed-dir',DEST/'stations',
          '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
          '--texture-dir',DEST/'texture','--rebind-texture','--triplet-scale',3.8,'--flow-gain',1.4]
    if stage=='mount':
        opts+=['--rig','--rig-roll','--rig-fields','pivot_x','pivot_y','pivot_z','R0_z','eta','k10']
    if programme=='nayeon':opts+=['--photo-bindings',OUT/'repair/prepared_v2/nayeon_photo_bindings.npz']
    native_run(DEST/stage/(programme+'_cam'+str(camera)),parent,opts)


if __name__=='__main__':
    from argparse import ArgumentParser
    ap=ArgumentParser(description=__doc__);ap.add_argument('task',choices=['tips','prepare','geometry'])
    ap.add_argument('--programme');ap.add_argument('--branch')
    args=ap.parse_args()
    if args.task=='tips':extract_tips(args.programme)
    elif args.task=='prepare':prepare()
    else:geometry(args.branch)
