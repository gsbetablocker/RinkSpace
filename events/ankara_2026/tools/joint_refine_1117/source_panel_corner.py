"""Direct board print border and corner readings from the original exposures."""
import shutil
import cv2
import numpy as np
from lane_end_evidence import WORK
from review import ji, localize, native_for, camera_state, source_frames
from shared import PREPARED, arrays, prepared_packet_path, read, save, write
from source_bundle import relative
from flow_depths import bind


def border(img,origin,along,normal,*,distances=None,radius=38,minimum=6,span=65,mask=None,blurred=False):
    d=np.arange(16.,176.,9.) if distances is None else distances;off=np.arange(-radius,radius+1.)
    if len(d)<minimum:return None
    guide=origin+d[:,None]*along
    grid=guide[:,None,:]+normal*off[None,:,None]
    rgb=cv2.remap(img,grid.astype('f4'),None,cv2.INTER_LINEAR).astype(float)
    signal=rgb[:,:,0]-rgb[:,:,1]+.42*(rgb[:,:,2]-rgb[:,:,1])-.2*(rgb.mean(2)-160)
    points=[];weight=[]
    for j,line in enumerate(signal):
        if (grid[j]<[3,3]).any() or (grid[j]>[1916,1076]).any():continue
        v=cv2.GaussianBlur(line[:,None],(1,7),1.3).ravel()
        g=-np.gradient(v);k=int(np.argmax(g[8:-8]))+8
        si,so=(slice(1,8),slice(-8,-1)) if blurred else (slice(k-8,k-3),slice(k+4,k+9))
        inside=float(np.median(v[si]));outside=float(np.median(v[so]))
        if inside-outside<(16 if blurred else 12) or np.median(rgb[j,si,0]-rgb[j,si,1])<8:continue
        if np.median(rgb[j,so,0]-rgb[j,so,1])>(8 if blurred else 10):continue
        threshold=(inside+outside)/2
        crossings=np.flatnonzero((v[:-1]>=threshold)&(v[1:]<threshold)&(abs(np.arange(len(v)-1)-k)<(19 if blurred else 6)))
        if not len(crossings):continue
        u=crossings[np.argmin(abs(crossings-k))];sub=(v[u]-threshold)/(v[u]-v[u+1])
        if mask is not None:
            check=np.rint(grid[j,max(0,u-4):u+6]).astype(int)
            if mask[check[:,1],check[:,0]].any():continue
        points.append(guide[j]+normal*(off[u]+sub));weight.append(inside-outside)
    if len(points)<minimum:return None
    xy=np.array(points);keep=np.ones(len(xy),bool)
    if blurred:
        signed=(xy-origin)@normal;keep=abs(signed-np.median(signed))<4.5
    for _ in range(3):
        if keep.sum()<minimum:return None
        centre=xy[keep].mean(0);_,_,v=np.linalg.svd(xy[keep]-centre,full_matrices=False)
        keep=abs((xy-centre)@v[1])<1.7
    if keep.sum()<minimum or np.ptp((xy[keep]-centre)@v[0])<span:return None
    return xy[keep],centre,v[1]


def prepare():
    name='chen_s20';parent=relative(WORK/'release_cameras'/name);b=camera_state(parent)
    setup=read(PREPARED/'setup.json');s=setup['shots'][name];native=native_for(parent,'chen')
    bottom='print_instance/Ad2::Ad2.bottom_edge';right='print_instance/Ad2::Ad2.right_vertical'
    keys=[bottom,right];tab=ji.native_paths(native,keys,count=49);world={k:v[0] for k,v in tab.items()}
    pts=b[name+'_pts'];jobs=[(i,k,b[name+'_times'][i]) for i in range(len(pts)) for k in keys]
    curves=ji.project_groups(native,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
    source=[];corners={};edge_values=[]
    for i,(pt,img) in enumerate(source_frames(s['source'],pts)):
        pb=np.where(curves[2*i][1][:,None]==0,curves[2*i][0],np.nan)
        pr=np.where(curves[2*i+1][1][:,None]==0,curves[2*i+1][0],np.nan);origin=pb[0]
        if not np.isfinite(origin).all() or (origin<[45,45]).any() or (origin>[1875,1035]).any():continue
        left=pb[1]-pb[0];left/=np.linalg.norm(left);up=pr[-2]-pr[-1];up/=np.linalg.norm(up)
        obs=[border(img,origin,left,-up),border(img,origin,up,-left)]
        if any(e is None for e in obs):continue
        normals=np.array([e[2] for e in obs])
        if abs(np.linalg.det(normals))<.7:continue
        corner=np.linalg.solve(normals,[e[2]@e[1] for e in obs])
        if np.linalg.norm(corner-origin)>40:continue
        corners[i]=corner
        source.append((pt,i,bottom,0.,corner,True))
        for key,poly,e in zip(keys,[pb,pr],obs):
            st,_=localize(poly,tab[key][1],e[0])
            for point,u in zip(e[0],st):
                if not np.isfinite(u):raise ValueError((pt,key,'missing finite border'))
                source.append((pt,i,key,u,point,False))
    if not corners:raise ValueError('No two-border readings')
    pt,ix,path,station,uv,fixed=map(np.asarray,zip(*source))
    folder=WORK/'delivery_prepared';raw=folder/'chen_s20_Ad2.npz'
    save(raw,pts=pt,idx=ix,path=path,station_seed=station,uv=uv,corner=fixed,source_sha256=s['source_sha256'])
    original=prepared_packet_path(name);a=arrays(original);new=np.arange(len(ix))
    code=max(map(int,s['source_packets']))+1
    ext=dict(pts=pt,idx=ix,path=path,station=station,uv=uv,station_fixed=fixed,
        family='Ad2',material=np.where(fixed,'native_purple_panel_corner','native_purple_panel_border'),
        source_fold=0,fold=0,source_packet=code,source_row=new,origin_row=-20000000-new,
        baseline_display=False,restored_admission_reason='two_original_purple_white_boundaries')
    for k,col in a.items():
        new_values=np.broadcast_to(ext[k],(len(ix),)+col.shape[1:])
        a[k]=np.concatenate([col,new_values.astype(col.dtype if col.dtype.kind!='U' else str)])
    for other,rec in setup['shots'].items():rec['observations']=str(prepared_packet_path(other))
    save(folder/(name+'.npz'),**a);s['observations']=str(folder/(name+'.npz'))
    s['source_packets'][str(code)]=dict(path=str(raw),source_row='row in measured purple panel border packet')
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz');setup['parent']=str(PREPARED)
    audit=read(PREPARED/'source_audit.json');audit[name]['added_Ad2_rows']=len(ix)
    audit[name]['added_Ad2_corners']=len(corners);write(folder/'source_audit.json',audit);write(folder/'setup.json',setup)
    # Deterministic held strips in the original texture bank retain their role.
    old=arrays(WORK/'texture'/(name+'.npz'));quality=arrays(WORK/'texture/quality'/(name+'.npz'))
    pairs=[(i,lag,corners[i-lag],point) for i,point in corners.items() for lag in (1,8) if i-lag in corners]
    idx,lag,left,end=map(np.asarray,zip(*pairs))
    world,err=bind(native,s,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],idx-lag,left,np.zeros(len(idx),int))
    vals=dict(index=idx,lag=lag,left_uv=left,uv=end,world=world,inverse_error=err,quality=np.ones(len(idx)),
              held=np.zeros(len(idx),bool),valid=np.isfinite(world).all(1))
    for k,v in vals.items():old[k]=np.concatenate((old[k],v.astype(old[k].dtype)))
    td=WORK/'corner_texture';save(td/(name+'.npz'),**old)
    quality['keep']=np.r_[quality['keep'],vals['valid']]
    quality['transient']=np.unique(np.r_[quality['transient'],idx[lag==1]])
    save(td/'quality'/(name+'.npz'),**quality)
    print('visible corners',len(corners),'source rows',len(ix),'motion pairs',len(pairs),flush=True)


def jin_panel():
    from lane_end_evidence import foreground
    name='jin_s10';parent=relative(WORK/'selected');b=camera_state(parent)
    setup=read(PREPARED/'setup.json');s=setup['shots'][name];native=native_for(parent,'jin')
    bottom='print_instance/Ad1::Ad1.bottom_edge';right='print_instance/Ad1::Ad1.right_vertical'
    keys=[bottom,right];tab=ji.native_paths(native,keys,count=65);world={k:v[0] for k,v in tab.items()}
    pts=b[name+'_pts'];jobs=[(i,k,b[name+'_times'][i]) for i in range(len(pts)) for k in keys]
    curves=ji.project_groups(native,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
    source=[];dist=np.array([6,10,14,19,24,30,38,49,63,78,95,118,145,170],float)
    for i,(pt,img) in enumerate(source_frames(s['source'],pts)):
        poly=[np.where(curves[2*i+j][1][:,None]==0,curves[2*i+j][0],np.nan) for j in (0,1)]
        pb,pr=poly;origin=pb[0]
        if not np.isfinite(origin).all() or (origin<[22,22]).any() or (origin>[1897,1057]).any():continue
        left=pb[1]-pb[0];up=pr[-2]-pr[-1]
        if not np.isfinite(np.r_[left,up]).all():continue
        left/=np.linalg.norm(left);up/=np.linalg.norm(up)
        h=np.linalg.norm(pr[-1]-pr[0]);dv=dist[dist<h*.77] if np.isfinite(h) else dist[:9]
        mask=foreground(s,None,i,pt)
        if mask is None:continue
        mask[:158,:590]=255;mask[:292,:451]=255;mask[928:,1618:]=255
        obs=[border(img,origin,left,-up,distances=dist,radius=25,minimum=4,span=14,mask=mask,blurred=True),
             border(img,origin,up,-left,distances=dv,radius=29,minimum=4,span=18,mask=mask,blurred=True)]
        if any(v is None for v in obs):continue
        normals=np.array([v[2] for v in obs])
        if abs(np.linalg.det(normals))<.78:continue
        corner=np.linalg.solve(normals,[v[2]@v[1] for v in obs])
        if np.linalg.norm(corner-origin)>30:continue
        xy=np.rint(corner).astype(int)
        if not ((xy>=[0,0]).all() and (xy<[1920,1080]).all()) or mask[xy[1],xy[0]]:continue
        source.append((pt,i,bottom,0.,corner,True))
        for key,line,edge in zip(keys,poly,obs):
            st,_=localize(line,tab[key][1],edge[0])
            for u,v in zip(st,edge[0]):
                if np.isfinite(u):source.append((pt,i,key,u,v,False))
    pt,i,path,st,uv,fixed=map(np.asarray,zip(*source));frames=np.unique(i[fixed])
    available=frames[np.isin(frames-1,frames)|np.isin(frames+1,frames)]
    good=np.isin(i,available);folder=WORK/'blue_inputs';raw=folder/'jin_s10_Ad1.npz'
    save(raw,pts=pt,idx=i,path=path,station_seed=st,uv=uv,corner=fixed,temporal_support=good,source_sha256=s['source_sha256'])
    a=arrays(prepared_packet_path(name));ids=np.flatnonzero(good);code=max(map(int,s['source_packets']))+1
    ext=dict(pts=pt[good],idx=i[good],path=path[good],station=st[good],uv=uv[good],station_fixed=fixed[good],
        family='Ad1',material=np.where(fixed[good],'native_blue_panel_corner','native_blue_panel_border'),
        source_fold=0,fold=0,source_packet=code,source_row=ids,origin_row=-30000000-ids,
        baseline_display=False,admission_reason='two_original_blue_white_boundaries',restored_admission_reason='two_original_blue_white_boundaries')
    for k,col in a.items():
        extra=np.broadcast_to(ext[k],(len(ids),)+col.shape[1:])
        a[k]=np.concatenate((col,extra.astype(col.dtype if col.dtype.kind!='U' else str)))
    for k,v in setup['shots'].items():v['observations']=str(prepared_packet_path(k))
    save(folder/(name+'.npz'),**a);s['observations']=str(folder/(name+'.npz'))
    s['source_packets'][str(code)]=dict(path=str(raw),source_row='independent native panel flank measurements')
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz');setup['parent']=str(PREPARED)
    audit=read(PREPARED/'source_audit.json');audit[name]['added_Ad1_corners']=len(available)
    audit[name]['added_Ad1_rows']=len(ids);write(folder/'source_audit.json',audit);write(folder/'setup.json',setup)
    corners=dict(zip(i[fixed&good],uv[fixed&good]))
    pairs=[(j,lag,corners[j-lag],v) for j,v in corners.items() for lag in (1,8) if j-lag in corners]
    idx,lag,start,end=map(np.asarray,zip(*pairs))
    world,err=bind(native,s,b[name+'_rigs'],b[name+'_controls'],b[name+'_times'],idx-lag,start,np.zeros(len(idx),int))
    values=dict(index=idx,lag=lag,left_uv=start,uv=end,world=world,inverse_error=err,quality=np.ones(len(idx)),
                held=np.zeros(len(idx),bool),valid=np.isfinite(world).all(1))
    old=arrays(WORK/'texture'/(name+'.npz'));quality=arrays(WORK/'texture/quality'/(name+'.npz'))
    for k,v in values.items():old[k]=np.concatenate((old[k],v.astype(old[k].dtype)))
    td=WORK/'blue_texture';save(td/(name+'.npz'),**old)
    quality['keep']=np.r_[quality['keep'],values['valid']]
    quality['transient']=np.unique(np.r_[quality['transient'],idx[lag==1]])
    save(td/'quality'/(name+'.npz'),**quality)
    print('source corners',len(available),'admitted rows',len(ids),'motion pairs',len(idx),flush=True)


if __name__=='__main__':
    from argparse import ArgumentParser
    p=ArgumentParser(description=__doc__);p.add_argument('--jin',action='store_true');args=p.parse_args()
    if args.jin:jin_panel()
    else:prepare()
