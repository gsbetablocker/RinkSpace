"""Current source displacement factors; reference depths are not rink measurements."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from shared import DATA, OUT, arrays, read, sha


def packet(name, shot):
    if shot['program'] in ('jin','okada'):
        return DATA/'runs/ankara_joint_1113/stability_1113/source_tracks'/(name+'.npz')
    root=DATA/'events/ankara_2026/programmes'
    sub=('chen_xi_jin_sp/fit_1115/programme_recheck_1116/source_tracks' if shot['program']=='chen'
         else 'nayeon_ko_sp/fit_1116/review_all_1116/pitch_probe_bound/tracks')
    return root/sub/(shot['shot']+'.npz')


def off_graphics(uv):
    x,y=uv.T
    return ((y>158.)|(x>590.))&((y>292.)|(x>451.))&((y<928.)|(x<1618.))


def load(setup, base, spans, rig_spans, *, native=None, current_depths=False,all_development=False,source_visibility=False,triplet_scale=0.,dense_source_tracks=False):
    output=defaultdict(list);manifest={}
    for name,(start,stop) in spans.items():
        shot=setup['shots'][name];path=packet(name,shot);a=arrays(path)
        pts=base[name+'_pts'];np.testing.assert_array_equal(a['pts'],pts)
        if 'source_sha256' in a:
            assert a['source_sha256'].item()==shot['source_sha256']
        keep=((a['role']<4) if all_development else (a['role']==0)) if 'role' in a else (a['fold']<(4 if all_development else 2))
        keep&=(a['quality']<16.)&np.isfinite(a['world']).all(1)
        visibility_note=None
        if source_visibility and shot['program']=='okada':
            ref=OUT/'source_flow_visibility'/(name+'.npz');visible=arrays(ref)
            np.testing.assert_array_equal(visible['pts'],pts)
            assert visible['source_packet_sha256'].item()==sha(path)
            assert visible['source_sha256'].item()==shot['source_sha256']
            keep&=visible['keep']
            visibility_note=dict(path=str(ref),sha256=sha(ref),
                role='actual source patch support avoids candidate foreground; no residual-based rejection')
        if not bool(a.get('graphics_mask_per_exposure',False)):
            keep&=off_graphics(a['uv'])&off_graphics(a['left_uv'])&off_graphics(a['anchor_uv'])
        ix=np.flatnonzero(keep)
        # Retain original pixels of the best source patch in each image strip,
        # lag and exposure. Adjacent patches do not count as independent frames.
        if not dense_source_tracks:
            key=np.rec.fromarrays((a['index'][ix],a['lag'][ix],a['family'][ix],
                                  (a['uv'][ix,0]//160).astype(int)))
            order=np.lexsort((a['quality'][ix],key));ordered=ix[order];kk=key[order]
            first=np.r_[True,kk[1:]!=kk[:-1]];ix=ordered[first]
        group=np.rec.fromarrays((a['index'][ix],a['lag'][ix],a['family'][ix]))
        _,inv,count=np.unique(group,return_inverse=True,return_counts=True)
        ix=ix[count[inv]>=2]
        group=np.rec.fromarrays((a['index'][ix],a['lag'][ix],a['family'][ix]))
        _,inv,count=np.unique(group,return_inverse=True,return_counts=True)
        # The historical source-flow objective normalizes each exposure/lag's
        # total mass. This is a declared robust engineering loss, not measured
        # sensor covariance or proof of independent statistical observations.
        mass=np.where(a['lag'][ix]==1,13.,9.5)*np.where(a['family'][ix]==0,1.,.68)
        weight=mass/np.sqrt(count[inv])
        frames=np.c_[a['index'][ix]-a['lag'][ix],a['index'][ix]]
        slot=rig_spans[name]
        earlier=np.maximum(a['index'][ix]-2,0)
        prev2=a.get('prev2_uv',np.full_like(a['uv'],np.nan))[ix]
        values=dict(world=a['world'][ix],uv=np.stack((a['left_uv'][ix],a['uv'][ix]),axis=1),
            times=base[name+'_times'][frames]+start*.02-shot['first'],pts=pts[frames],
            camera=slot['inverse'][frames]+slot['start'],weight=weight,
            shot=np.repeat(name,len(ix)),source_row=ix,lag=a['lag'][ix],
            triplet=(triplet_scale>0.)&(a['lag'][ix]==1)&(a['index'][ix]>=2)&np.isfinite(prev2).all(1),
            prev2_uv=prev2,prev2_times=base[name+'_times'][earlier]+start*.02-shot['first'],
            prev2_pts=pts[earlier],prev2_camera=slot['inverse'][earlier]+slot['start'],
            triplet_scale=np.full(len(ix),triplet_scale))
        depth_note=None
        if current_depths:
            from flow_depths import bind
            if native is None:raise ValueError('Current depths require the active native geometry')
            world,error=bind(native,shot,base[name+'_rigs'],base[name+'_controls'],base[name+'_times'],
                frames[:,0],a['left_uv'][ix],a['family'][ix])
            good=np.isfinite(world).all(1)
            values['world']=world
            values={k:v[good] for k,v in values.items()}
            depth_note=dict(reference='left actual source pixel on current board/ice; conditional depth only',
                valid_pairs=int(good.sum()),invalid_inverse_pairs=int((~good).sum()),
                maximum_native_inverse_roundtrip_px=float(error[good].max()) if good.any() else None,
                world_movement_p50=float(np.median(np.linalg.norm(world[good]-a['world'][ix[good]],axis=1))) if good.any() else None)
        for k,v in values.items():output[k].append(v)
        manifest[name]=dict(path=str(path),selected_pairs=len(ix),dense_source_tracks=bool(dense_source_tracks),
            source_sha256=shot['source_sha256'],reference_geometry_hash=str(a.get('geometry_sha256','legacy binding')),
            role='source displacement only; stored world is conditional reference depth',current_depth_binding=depth_note,
            source_visibility=visibility_note)
    return {k:np.concatenate(v) for k,v in output.items()},manifest


def append_board_texture(tracks,folder,setup,base,spans,rig_spans,native=None):
    """Keep measured native-pixel displacement, with one mass per exposure and lag."""
    report={}
    sources=read(folder/'sources.json') if (folder/'sources.json').is_file() else {}
    for name,(start,_) in spans.items():
        root=Path(sources.get(name,folder))
        path=root/(name+'.npz');a=arrays(path);s=setup['shots'][name]
        assert a['source_sha256'].item()==s['source_sha256']
        pts=base[name+'_pts'];np.testing.assert_array_equal(a['pts'],pts)
        certificate=arrays(root/'quality'/(name+'.npz'))
        np.testing.assert_array_equal(certificate['pts'],pts)
        use=a['valid']&certificate['keep']
        ix=np.flatnonzero(use&~a['held'])
        if not len(ix):
            report[name]=dict(path=str(path),rows=0);continue
        end=a['index'][ix];lag=a['lag'][ix];frames=np.c_[end-lag,end]
        surface=a['family'] if 'family' in a else np.zeros(len(a['index']),np.int8)
        if not np.isin(surface,[0,1]).all():raise ValueError('Unknown measured texture surface in '+str(path))
        group=(end*16+lag)*2+surface[ix]
        _,inv,count=np.unique(group,return_inverse=True,return_counts=True)
        # Several nearby patches measure one displacement. Preserve source
        # support without giving a textured logo the mass of many exposures.
        world=a['world'][ix]
        if native is not None:
            from flow_depths import bind
            world,_=bind(native,s,base[name+'_rigs'],base[name+'_controls'],base[name+'_times'],
                         frames[:,0],a['left_uv'][ix],surface[ix])
            bound=np.isfinite(world).all(1)
            ix=ix[bound];frames=frames[bound];world=world[bound];end=end[bound];lag=lag[bound]
            _,inv,count=np.unique((end*16+lag)*2+surface[ix],return_inverse=True,return_counts=True)
        mass_count=count[inv]
        if 'spatial_bin' in a:
            groups=np.column_stack((end,lag,surface[ix],a['spatial_bin'][ix]))
            cells,cell_id,num=np.unique(groups,axis=0,return_inverse=True,return_counts=True)
            _,offset,ncells=np.unique(cells[:,:3],axis=0,return_inverse=True,return_counts=True)
            mass_count=num[cell_id]*ncells[offset[cell_id]]
        w=np.where(lag==1,21.,12.)/np.sqrt(mass_count)
        slot=rig_spans[name]
        value=dict(world=world,uv=np.stack((a['left_uv'][ix],a['uv'][ix]),axis=1),
            times=base[name+'_times'][frames]+start*.02-s['first'],pts=pts[frames],
            camera=slot['inverse'][frames]+slot['start'],weight=w,shot=np.repeat(name,len(ix)),
            source_row=ix,lag=lag,triplet=np.zeros(len(ix),bool),prev2_uv=np.full((len(ix),2),np.nan),
            prev2_times=base[name+'_times'][end]+start*.02-s['first'],prev2_pts=pts[end],
            prev2_camera=slot['inverse'][end]+slot['start'],triplet_scale=np.zeros(len(ix)))
        for k in tracks:tracks[k]=np.concatenate((tracks[k],value[k].astype(tracks[k].dtype,copy=False)))
        report[name]=dict(path=str(path),sha256=sha(path),quality_sha256=sha(root/'quality'/(name+'.npz')),
                          rows=len(ix),held=int((use&a['held']).sum()))
    return report


def attach(native, rows, tracks):
    """A fixed auxiliary point per track pair, never a changed event path."""
    key='__source_flow_reference_depths__'
    count=len(tracks['world'])
    native.paths[key]=dict(id=key,kind=4,parameters=[],vertices=tracks['world'],knots=[],
        support=[0.,1.],layer='auxiliary',label='conditional source flow depths')
    mask=np.column_stack((tracks['triplet'],np.ones((count,2),bool)))
    mask[:,1]=tracks['lag']!=0
    length=mask.sum(1)
    uv=np.concatenate((tracks['prev2_uv'][:,None,:],tracks['uv']),axis=1)
    pts=np.column_stack((tracks['prev2_pts'],tracks['pts']))
    times=np.column_stack((tracks['prev2_times'],tracks['times']))
    camera=np.column_stack((tracks['prev2_camera'],tracks['camera']))
    paths=np.repeat(key,count).astype('U200')
    stations=np.linspace(0.,1.,count)
    if 'material_path' in tracks:
        photo=tracks['lag']==0
        if np.any(tracks['material_path'][photo]=='') or not np.isfinite(tracks['material_station'][photo]).all():
            raise ValueError('Photo factor lacks a fixed typed material binding')
        paths[photo]=tracks['material_path'][photo];stations[photo]=tracks['material_station'][photo]
    return dict(path=np.r_[rows['path'],np.repeat(paths,length)],
        station=np.r_[rows['station'],np.repeat(stations,length)],
        pts=np.r_[rows['pts'],pts[mask]],uv=np.vstack((rows['uv'],uv[mask])),
        camera=np.r_[rows['camera'],camera[mask]].astype(np.int32),
        times=np.r_[rows['times'],times[mask]])


def factors(n, white, tracks):
    m=len(tracks['weight']);pair=np.kron([[-1.,1.]],np.eye(2)).ravel()
    triple=tracks['triplet'];photo=tracks['lag']==0;pair_only=~(triple|photo)
    length=(2+triple-photo).astype(np.int32)
    ranks=np.where(photo,1,np.where(triple,4,2)).astype(np.int32)
    offsets=np.r_[0,np.cumsum(2*length*ranks,dtype=np.int32)]
    relative=np.empty(offsets[-1]);w=tracks['weight']
    relative[offsets[:-1][pair_only,None]+np.arange(8)]=w[pair_only,None]*pair
    relative[offsets[:-1][photo,None]+np.arange(2)]=w[photo,None]*np.array([1.,0.])
    for scale in np.unique(tracks['triplet_scale'][triple]):
        selected=triple&(tracks['triplet_scale']==scale)
        contrast=np.kron(np.array([[-1.,0.,1.],[1.,-2.,1.]])*np.array([[1.],[scale/np.sqrt(3.)]]),np.eye(2)).ravel()
        relative[offsets[:-1][selected,None]+np.arange(24)]=w[selected,None]*contrast
    if 'source_image_metric' in tracks:
        # Transform each 2-D displacement and acceleration into its measured
        # source-patch axes. A single blurred edge weakly supports motion along it.
        for mask,rank,members in [(pair_only,2,2),(triple,4,3)]:
            chosen=np.flatnonzero(mask&np.any(tracks['source_image_metric']!=np.eye(2),axis=(1,2)))
            if not len(chosen):continue
            jj=offsets[chosen,None]+np.arange(rank*members*2)
            weights=relative[jj].reshape(len(chosen),rank,members,2)
            weights=np.einsum('njmi,nik->njmk',weights,tracks['source_image_metric'][chosen])
            relative[jj]=weights.reshape(len(chosen),-1)
    total=int(length.sum())
    return dict(local_ids=np.r_[np.arange(n,dtype=np.int32),np.full(total,-1,np.int32)],
        group_offsets=np.r_[np.arange(n,dtype=np.int32),n+np.r_[0,np.cumsum(length,dtype=np.int32)]],
        members=np.arange(n+total,dtype=np.int32),ranks=np.r_[np.full(n,2,np.int32),ranks],
        weight_offsets=np.r_[4*np.arange(n,dtype=np.int32),4*n+offsets],
        weights=np.r_[white.ravel(),relative])


def append_photos(tracks,source,setup,base,spans,rig_spans,native,*,dynamic=False):
    """Retain actual photo UV and its fixed typed material station."""
    from rebind_source_photos import rebind_world
    a=arrays(source)
    world=rebind_world(native,np.c_[a['path'],a['alternate_path']],
        np.c_[a['station'],a['alternate_station']])
    if dynamic:
        tracks['material_path']=np.full(len(tracks['world']),'',dtype='U200')
        tracks['material_station']=np.full(len(tracks['world']),np.nan)
    additions=[];counts={}
    for name,(start,stop) in spans.items():
        s=setup['shots'][name]
        if s['program']!='nayeon':continue
        assert a['source_sha256'].item()==s['source_sha256']
        use=(a['shot']==s['shot'])&a['usable'];rows=np.flatnonzero(use)
        if not len(rows):continue
        pts=a['pts'][rows];index=np.searchsorted(base[name+'_pts'],pts)
        np.testing.assert_array_equal(base[name+'_pts'][index],pts)
        times=base[name+'_times'][index]+start*.02-s['first']
        slot=rig_spans[name];camera=slot['inverse'][index]+slot['start']
        values=dict(world=world[rows],uv=np.repeat(a['uv'][rows,None,:],2,axis=1),
            times=np.repeat(times[:,None],2,axis=1),pts=np.repeat(pts[:,None],2,axis=1),
            camera=np.repeat(camera[:,None],2,axis=1),weight=a['weight'][rows],
            shot=np.repeat(name,len(rows)),source_row=rows,lag=np.zeros(len(rows),np.int8),
            triplet=np.zeros(len(rows),bool),prev2_uv=np.full((len(rows),2),np.nan),
            prev2_times=times,prev2_pts=pts,prev2_camera=camera,triplet_scale=np.zeros(len(rows)))
        if dynamic:values.update(material_path=a['path'][rows],material_station=a['station'][rows])
        additions.append(values);counts[name]=len(rows)
    for addition in additions:
        for key in tracks:tracks[key]=np.concatenate((tracks[key],addition[key].astype(tracks[key].dtype,copy=False)))
    return dict(path=str(source),sha256=sha(source),rows_by_shot=counts,
        source_rows=len(a['pts']),usable_material_bindings=int(a['usable'].sum()),
        retained_unbound_group_statistics=int((~a['usable']).sum()),
        role=('source horizontal photometric registration; rank1; fixed path/station with native scene derivatives'
              if dynamic else 'source horizontal photometric registration; rank1; current fixed geometry material references'))


def resident(native, *, directions=None, **kwargs):
    """Bind real scene rows plus explicit constant-depth motion references."""
    p=native.resident(**kwargs)
    if directions is None:return p
    from rinkspace.session.gpu_geometry import compile_joint_scene_records,scene_coordinate_charts
    from rinkspace.domain.ice_tolerance import scene_physical_bounds
    order=list(dict.fromkeys(kwargs['path_ids']))
    key='__source_flow_reference_depths__'
    real=[k for k in order if k!=key]
    records,data=compile_joint_scene_records(native.scene,native.scene_path,native.parameter_state,real,core=native.core)
    by_name=dict(zip(real,records))
    by_name[key]=np.zeros(12,np.int32)  # Static native kind-4 vertices, zero scene derivative.
    lo,hi=scene_physical_bounds(native.parameter_state,native.ice_relative_limit,free_metric=True)
    try:
        p.configure_scene_parameters(values=native.parameter_state.values,
            records=np.asarray([by_name[k] for k in order]),data=data,directions=directions,
            lower=lo,upper=hi,charts=scene_coordinate_charts(native.parameter_state))
    except BaseException:
        p.close();raise
    return p
