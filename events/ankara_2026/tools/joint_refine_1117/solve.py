"""Joint native CUDA/cuDSS geometry and camera refinement on current inputs."""
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path
import json
import os
import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, kron
from shared import prepared_packet_path, shared_reader
from shared import (EVENT, FIELD, LIBRARY, OUT, PREPARED, NativeCore, NativeSceneGPU,
    ParameterState, SceneParameterModel, arrays, numeric_sha, read, save, sha, write)
from rinkspace.domain.motion_chart import UniformMotionChart, cubic_samples
from rinkspace.domain.multi_shot_motion import continue_exposure_halos
from rinkspace.domain.camera_experiment import CAMERA_NAMES, _SCALAR_STORAGE_INDEX


def finite(value):
    if isinstance(value,dict):return {str(k):finite(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite(v) for v in value]
    if isinstance(value,np.ndarray):return finite(value.tolist())
    if isinstance(value,np.generic):return finite(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


class Journal:
    def __init__(self,path):
        self.stream=path.open('x',encoding='utf-8',buffering=1)
        self.path=path
        main=os.environ.get('RINKSPACE_JOINT_MAIN_LOG')
        self.main=Path(main).open('a',encoding='utf-8',buffering=1) if main else None
    def __call__(self,event,**values):
        item=finite(dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**values))
        self.stream.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
        if self.main:
            self.main.write(json.dumps(dict(item,stage_log=str(self.path)),ensure_ascii=False,separators=(',',':'))+'\n')
        if event=='trial':
            s=item['summary'];print(s['trial_count'],s['reason'],s['accepted'],round(s['trial_cost'],3),flush=True)
        else:print(event,values.get('rows',''),values.get('reason',''),flush=True)


def stats(e):
    v=e[np.isfinite(e)]
    return dict(n=len(e),invalid=int(len(e)-len(v)),p50=float(np.median(v)),
        p90=float(np.quantile(v,.90)),p95=float(np.quantile(v,.95)),rms=float(np.sqrt(np.mean(v*v)))) if len(v) else dict(n=len(e),invalid=len(e))


def layer(f):
    return np.where(np.char.startswith(f,'Ad'),'ad',np.where(np.char.startswith(f,'board.'),'board','ice'))


def summaries(errors,rows):
    out={}
    for p in np.unique(rows['program']):
        use=rows['program']==p
        out[p]={key:stats(errors[use&(rows['layer']==key)]) for key in ('ice','board','ad')}
    return out


def scene_columns(state, rows, mode, selected=None,board=False,ice_coordinates=None):
    columns=[]
    if mode=='camera':return columns
    for key,b in state.blocks.items():
        if selected and key not in selected:continue
        if b['kind']=='board_shape' and board:
            for i in b['independent_local_indices']:
                columns.append((b['offset']+i,.003 if i==4 else .00008 if i==5 else .0003,key+'/'+str(i)))
        elif key.endswith('/ice_tolerance'):
            for i in b['independent_local_indices']:
                if ice_coordinates and b['values_meaning'][i] not in ice_coordinates:
                    continue
                if 16<=i<48 and (i-16)%4==0:
                    continue  # Innermost contour identity is unresolved in current source audit.
                columns.append((b['offset']+i,.008,b['values_meaning'][i]))
        elif b['kind']=='print_placement':
            ad=key.split('/')[1]
            if np.count_nonzero(rows['family']==ad)<12:continue
            for i in b['independent_local_indices']:
                columns.append((b['offset']+i,.0006 if i<2 else .005,key+'/'+str(i)))
        elif b['kind']=='print_size':
            for i in b['independent_local_indices']:
                columns.append((b['offset']+i,.008,key+'/'+str(i)))
        # Entire print artwork affine fields; no individual letters or vertices.
        elif b['kind']=='print_shape':
            for i in b['independent_local_indices']:
                if i<4:columns.append((b['offset']+i,.003,key+'/'+str(i)))
    return columns


def motion_basis(controls,times,first,dt,indices,stride,detail_indices=None,local_indices=None,source_basis=None):
    if source_basis is None:
        chart=UniformMotionChart.create(len(controls),first,dt,stride,times[0])
        matrix=chart.refinement
    else:
        matrix=np.asarray(source_basis,dtype='f8')
        if matrix.ndim!=2 or len(matrix)!=len(controls) or not np.isfinite(matrix).all() or local_indices is None:
            raise ValueError('Source-supported motion columns must match the local control timeline')
    if source_basis is None and detail_indices is not None and len(detail_indices):
        if stride!=2:raise ValueError('Local odd-knot complement requires the two-exposure coefficient chart')
        knots=np.rint((times[detail_indices]-first)/dt).astype(int)
        nearby=np.unique((knots[:,None]+np.arange(-3,4)).ravel())
        nearby=nearby[(nearby%2==1)&(nearby>1)&(nearby<len(controls)-2)]
        detail=np.zeros((len(controls),len(nearby)))
        detail[nearby,np.arange(len(nearby))]=1.
        matrix=np.column_stack((matrix,detail))
    if local_indices is not None:
        support=np.rint((times[local_indices]-first)/dt).astype(int)
        allowed=np.unique((support[:,None]+np.arange(-4,5)).clip(0,len(controls)-1))
        mask=np.zeros(len(controls),bool);mask[allowed]=True
        matrix=np.where(mask[:,None],matrix,0.)
    grid=(times[[0,-1]]-first)/dt
    if np.max(abs(grid-np.rint(grid)))<1.e-8:
        matrix=continue_exposure_halos(np.repeat(matrix[:,None],3,axis=1),first,dt,times[[0,-1]])[:,0]
    ii,w=cubic_samples(len(controls),first,dt,times[np.unique(indices)])
    response=np.einsum('ni,nik->nk',w,matrix[ii])
    active=np.count_nonzero(abs(response)>.06,axis=0)>=2
    return kron(csr_matrix(matrix[:,active]),csr_matrix(np.diag([.001,.001,.003])),format='csr')


def assemble(setup,base,mode,stride,programmes=None,families=None,shots=None,all_development=False,
             camera_shots=None,passive_families=None,required_shots=None,paths=None,detail_dir=None,passive_paths=None,local_material=None,local_detail=False,materials=None):
    chunks=defaultdict(list);controls=[];rigs=[];blocks=[];spans={};rig_spans={};cursor=0
    for name,s in setup['shots'].items():
        if shots and name not in shots:continue
        if programmes and s['program'] not in programmes:continue
        a=arrays(prepared_packet_path(name))
        use=(a['fold']<5) if all_development else (a['fold']==0)
        if families:use&=np.isin(a['family'],families)
        if paths:use&=np.isin(a['path'],paths)
        if materials:use&=np.isin(a['material'],materials)
        active_camera=camera_shots is None or name in camera_shots
        if not active_camera and passive_families:use&=np.isin(a['family'],passive_families)
        if not active_camera and passive_paths:use&=np.isin(a['path'],passive_paths)
        x={k:v[use] for k,v in a.items()}
        x['station_fixed']=a.get('station_fixed',np.zeros(len(a['pts']),bool))[use]
        if not len(x['pts']) and (required_shots is None or name not in required_shots):continue
        c=base[name+'_controls'];t=base[name+'_times'];r=base[name+'_rigs']
        table,inv=np.unique(r,axis=0,return_inverse=True)
        rig_spans[name]=dict(start=len(rigs),stop=len(rigs)+len(table),inverse=inv)
        rids=inv[x['idx']]+len(rigs)
        rigs.extend(table)
        spans[name]=(cursor,cursor+len(c))
        x['times']=t[x['idx']]+cursor*.02-s['first']
        x['camera']=rids.astype(np.int32)
        x['program']=np.repeat(s['program'],len(x['pts']))
        x['shot']=np.repeat(name,len(x['pts']))
        for k in ('uv','pts','path','station','family','material','times','camera','program','shot','idx','origin_row','fold','station_fixed'):
            chunks[k].append(x[k])
        detail_packet=arrays(Path(detail_dir)/(name+'.npz')) if detail_dir and active_camera else {}
        detail=detail_packet.get('transient')
        local=np.unique(x['idx'][x['material']==local_material]) if local_material else None
        if local_detail:
            if detail is None or local_material:raise ValueError('Local exposure motion requires a source detail file')
            local=detail
        if local is not None and not len(local):raise ValueError('No admitted local source material in '+name)
        blocks.append(motion_basis(c,t,s['first'],s['dt'],x['idx'],stride,detail,local,detail_packet.get('motion_basis')) if mode!='scene' and active_camera else csr_matrix((c.size,0)))
        controls.append(c);cursor+=len(c)
    rows={k:np.concatenate(v) for k,v in chunks.items()}
    rows['layer']=layer(rows['family'])
    return rows,np.vstack(controls),np.asarray(rigs),block_diag(blocks,format='csr'),spans,rig_spans


def weights(rows,reference=None):
    group=np.rec.fromarrays((rows['shot'],rows['pts'],rows['family']))
    _,inv,num=np.unique(group,return_inverse=True,return_counts=True)
    sigma=np.where(rows['layer']=='ice',1.5,np.where(rows['layer']=='board',1.35,1.6))
    w=2./(sigma*np.sqrt(np.maximum(1.,num[inv]/4.)))
    masses={p:float(np.sum(w[rows['program']==p]**2)) for p in np.unique(rows['program'])}
    normalization=reference if reference is not None else masses
    target=float(np.median(list(normalization.values())))
    for p in masses:w[rows['program']==p]*=np.sqrt(target/normalization[p])
    return w,masses


def run(args):
    if not np.isfinite(args.huber_delta) or args.huber_delta<=0:raise ValueError('Invalid Huber scale')
    if args.raw_pixels and args.tracks:
        raise ValueError('Raw pixel likelihood excludes correlated source-flow groups')
    fixed_photo_geometry=(args.mode=='camera' or (args.mode=='joint' and args.ice_coordinates
        and args.scene_blocks and all(k.endswith('/ice_tolerance') for k in args.scene_blocks) and not args.board))
    if args.dynamic_photos and not args.photo_bindings:raise ValueError('Dynamic photos require material bindings')
    if args.photo_bindings and ((not fixed_photo_geometry and not args.dynamic_photos) or not args.tracks):
        raise ValueError('Photo material references require unchanged advertisement geometry and source-track groups')
    folder=OUT/args.out;folder.mkdir(exist_ok=True)
    log=Journal(folder/'solve.jsonl')
    log('start',mode=args.mode,library=str(LIBRARY),scene=str(FIELD),input=str(PREPARED),
        parent=args.parent,reset_camera=args.reset_camera,coefficient_stride=args.stride,
        coordinate_policy='nominal chart fixed; no nominal metric boxes')
    setup=read(PREPARED/'setup.json');original=arrays(PREPARED/'baseline.npz');base=dict(original)
    scene=arrays(OUT/'initial_scene.npz')['scene_values'];field=FIELD
    if args.parent:
        parent=arrays(OUT/args.parent/'state.npz');scene=parent['scene_values']
        field=Path(str(parent.get('scene_path',FIELD)))
        for key in base:
            if key in parent:base[key]=parent[key]
        for key,value in parent.items():
            if key.endswith('_active_folds'):base[key]=value
        if args.reset_camera:
            for name,s in setup['shots'].items():
                if args.programmes and s['program'] not in args.programmes:continue
                if args.shots and name not in args.shots:continue
                for suffix in ('_pts','_times','_controls','_rigs'):
                    base[name+suffix]=original[name+suffix]
    model=SceneParameterModel(read(field),field,optimize_scene=True)
    state=ParameterState(model.initial_state.layout,np.ascontiguousarray(scene))
    if args.scene_blocks and set(args.scene_blocks)-set(state.blocks):
        raise ValueError('Unknown scene blocks: '+str(sorted(set(args.scene_blocks)-set(state.blocks))))
    native=NativeSceneGPU(field,NativeCore(LIBRARY),parameter_state=state)
    photo_shots=None
    if args.dynamic_photos:
        photo=arrays(args.photo_bindings)
        photo_shots={'nayeon_'+str(s) for s in np.unique(photo['shot'][photo['usable']])}
    rows,c,rigs,md,spans,rig_spans=assemble(setup,base,args.mode,args.stride,args.programmes,args.families,args.shots,args.all_development,
        args.camera_shots,args.passive_families,photo_shots,args.paths,args.detail_dir,args.passive_paths,args.local_material,args.local_detail,args.materials)
    active_spans={name:span for name,span in spans.items() if args.camera_shots is None or name in args.camera_shots}
    if args.camera_shots and set(active_spans)!=set(args.camera_shots):raise ValueError('An active camera shot has no source rows')
    if args.freeze_motion:md=csr_matrix((md.shape[0],0))
    if args.station_seed or args.station_seed_dir:
        seed={}
        if args.station_seed:
            with shared_reader(OUT/args.station_seed/'curve_errors.npz') as stream, np.load(stream) as archive:
                seed={name+'_station':archive[name+'_station'] for name in spans}
        for name in spans:
            a=arrays(prepared_packet_path(name));use=rows['shot']==name
            all_stations=a['station'].copy()
            known=seed.get(name+'_station',np.empty(0))
            if args.station_seed_dir and (Path(args.station_seed_dir)/(name+'.npz')).is_file():
                with np.load(Path(args.station_seed_dir)/(name+'.npz')) as archive:known=archive['station']
            valid_indices=np.flatnonzero(np.isfinite(known))
            all_stations[valid_indices]=known[valid_indices]
            mask=(a['fold']<5) if args.all_development else (a['fold']==0)
            if args.families:mask&=np.isin(a['family'],args.families)
            if args.paths:mask&=np.isin(a['path'],args.paths)
            if args.materials:mask&=np.isin(a['material'],args.materials)
            if name not in active_spans and args.passive_families:mask&=np.isin(a['family'],args.passive_families)
            if name not in active_spans and args.passive_paths:mask&=np.isin(a['path'],args.passive_paths)
            value=all_stations[mask]
            if len(value)!=int(use.sum()) or not np.isfinite(value).all():
                raise ValueError('Station seed does not bind current source rows: '+name)
            rows['station'][use]=np.where(rows['station_fixed'][use],rows['station'][use],value)
    geo=scene_columns(state,rows,args.mode,args.scene_blocks,args.board,args.ice_coordinates)
    if args.ice_coordinates and set(args.ice_coordinates)-{column[2] for column in geo}:
        raise ValueError('Requested ice coordinates were not activated')
    rig_fields={'pivot_x':.00045,'pivot_y':.00045,'pivot_z':.0003,
        'eta':.003,'cx':3.,'cy':3.,'cy_zoom':.6,'k10':.012,'k11':.020,
        'p1':.002,'p2':.002,'wide_k1':.025,'wide_k2':.035} if args.rig else {}
    if args.rig_roll:
        if not args.rig:raise ValueError('--rig-roll requires --rig')
        rig_fields['R0_z']=.001
    if args.fit_readout:
        if not args.rig:raise ValueError('--fit-readout requires --rig')
        rig_fields['readout']=.00035
    if args.rig_kinematics:
        if not args.rig:raise ValueError('--rig-kinematics requires --rig')
        rig_fields.update(pan_axis_tangent1=.001,pan_axis_tangent2=.001,tilt_axis=.001,
            lever_x=.0002,lever_y=.0002,lever_z=.0002,axis_spacing=.0001,
            pupil1=.00015,pupil2=.00015,cx_zoom=.6,k20=.006,k21=.008,wide_p1=.002,wide_p2=.002)
    if args.rig_fields:
        if not args.rig:raise ValueError('--rig-fields requires --rig')
        absent=set(args.rig_fields)-set(rig_fields)
        if absent:raise ValueError('Unavailable rig fields: '+str(sorted(absent)))
        rig_fields={key:rig_fields[key] for key in args.rig_fields}
    rig_terms={key:{key:scale} for key,scale in rig_fields.items()}
    if args.rig_profile:
        if not args.rig or args.rig_fields:raise ValueError('Choose an explicit installation profile or individual rig fields')
        rig_terms=read(args.rig_profile)
        for coefficients in rig_terms.values():
            if not coefficients or set(coefficients)-set(CAMERA_NAMES) or not np.isfinite(list(coefficients.values())).all():
                raise ValueError('Invalid installation profile direction')
        rig_fields=rig_terms
    groups=sorted({(setup['shots'][name]['program'],setup['shots'][name]['camera_index']) for name in active_spans})
    rig_count=len(groups)*len(rig_fields)
    q=md.shape[1]+len(geo)+rig_count
    if not q:raise ValueError('No active directions')
    motion=hstack((md,csr_matrix((md.shape[0],len(geo)+rig_count))),format='csr')
    directions=np.zeros((len(scene),q)) if geo else None
    for j,(index,scale,label) in enumerate(geo,md.shape[1]):directions[index,j]=scale
    optical=None
    if args.rig:
        optical=np.zeros((len(rigs)*31,q))
        for name,rspan in rig_spans.items():
            if name not in active_spans:continue
            s=setup['shots'][name];group=groups.index((s['program'],s['camera_index']))
            for k,coefficients in enumerate(rig_terms.values()):
                column=md.shape[1]+len(geo)+group*len(rig_fields)+k
                for rig_field,scale in coefficients.items():
                    optical[np.arange(rspan['start'],rspan['stop'])*31+CAMERA_NAMES.index(rig_field),column]=scale
    mcam=dict(first_control_time=0.,control_dt=.02,image_height=1080,reference_pts=0,time_base_num=1,time_base_den=12800)
    n=len(rows['pts'])
    if args.raw_pixels or args.unit_weights:
        w=np.ones(n);masses={p:int(np.count_nonzero(rows['program']==p)) for p in np.unique(rows['program'])}
    else:w,masses=weights(rows,read(args.mass_reference)['masses'] if args.mass_reference else None)
    if args.weight_profile:
        if args.raw_pixels or args.unit_weights:raise ValueError('Select either source profile or unit weights')
        profile=read(args.weight_profile)
        for programme,record in profile['programmes'].items():
            selected=rows['program']==programme
            sigma=np.where(rows['layer']=='ice',1.35,np.where(rows['layer']=='board',1.40,1.64))
            default=np.where(rows['layer']=='ice',1.5,np.where(rows['layer']=='board',1.35,1.6))
            w[selected]*=(default/sigma)[selected]
            w[selected&np.char.startswith(rows['material'],'dense_')]*=.70
            for profile_field,label in [('path','path_weights'),('family','family_weights')]:
                for shot,items in record[label].items():
                    for key,multiplier in items.items():
                        if not np.isfinite(multiplier) or multiplier<=0:raise ValueError('Invalid source weight profile')
                        w[selected&(rows['shot']==programme+'_'+shot)&(rows[profile_field]==key)]*=multiplier
    if args.material_weights:
        for material,gain in read(args.material_weights).items():
            if not np.isfinite(gain) or gain<=0:raise ValueError('Invalid source-material whitening')
            w[rows['material']==material]*=gain
    if not np.isfinite(args.ice_pixel_gain) or args.ice_pixel_gain<=0:
        raise ValueError('Ice measurement precision must be positive')
    w[rows['layer']=='ice']*=args.ice_pixel_gain
    white=np.zeros((n,4));white[:,0]=white[:,3]=w
    vertical_extent=np.isin(rows['material'],['ink_extent_ymin','ink_extent_ymax'])
    extent=np.isin(rows['material'],['outline_xmin','outline_xmax','ink_extent_xmin','ink_extent_xmax']) | vertical_extent
    if extent.any():
        if args.raw_pixels:
            raise ValueError('A measured silhouette extent requires a one-dimensional residual')
        # The material station of an image extremum depends on the camera.
        # Below, the rank-one factor keeps only the measured coordinate.
        from feedback_fit import bind_extrema
        bind_extrema(native,base,setup,rows,extent)
        white[vertical_extent,0]=0.
        white[vertical_extent,1]=w[vertical_extent]
    tracks=None;bound=rows
    if args.tracks:
        import motion_tracks
        tracks,track_manifest=motion_tracks.load(setup,base,active_spans,rig_spans,native=native,
            current_depths=args.current_track_depths,all_development=args.all_development,
            source_visibility=args.source_visibility,triplet_scale=args.triplet_scale,dense_source_tracks=args.dense_source_tracks)
        source_count=len(tracks['weight'])
        if args.texture_dir:
            track_manifest['additional_board_texture']=motion_tracks.append_board_texture(
                tracks,Path(args.texture_dir),setup,base,active_spans,rig_spans,native if args.rebind_texture else None)
        if args.flow_gain!=1.:
            if not np.isfinite(args.flow_gain) or args.flow_gain<=0:raise ValueError('Invalid displacement weight')
            tracks['weight']*=args.flow_gain
            track_manifest['source_displacement_gain']=args.flow_gain
        if args.photo_bindings:
            track_manifest['photo_registration']=motion_tracks.append_photos(tracks,Path(args.photo_bindings),
                setup,base,spans if args.dynamic_photos else active_spans,rig_spans,native,dynamic=args.dynamic_photos)
        if args.source_image_metrics:
            metrics=np.tile(np.eye(2),(len(tracks['weight']),1,1))
            registered={}
            for name in active_spans:
                path=Path(args.source_image_metrics)/(name+'.npz')
                if not path.is_file():continue
                q=arrays(path)
                assert q['source_sha256'].item()==setup['shots'][name]['source_sha256']
                assert q['packet_sha256'].item()==sha(motion_tracks.packet(name,setup['shots'][name]))
                ix=np.flatnonzero(tracks['shot'][:source_count]==name)
                metrics[ix]=q['metric'][tracks['source_row'][ix]]
                registered[name]=dict(path=str(path),sha256=sha(path))
            tracks['source_image_metric']=metrics
            track_manifest['source_image_metrics']=registered
        bound=motion_tracks.attach(native,rows,tracks)
        write(folder/'source_tracks.json',track_manifest)
        save(folder/'used_tracks.npz',**tracks)
        log('source_displacement_factors',pairs=len(tracks['weight']),
            role='actual source pixel differences; fixed conditional reference depths; no geometry measurement')
    log('binding',rows=n,scene_path=str(field),motion_columns=md.shape[1],scene_columns=[g[2] for g in geo],
        group_masses=masses,shots=list(spans),rig_rows=len(rigs),rig_fields=rig_fields,
        objective='unweighted_squared_pixel_distance' if args.raw_pixels else 'correlated_rank_normalized_huber',
        absolute_weights='unit' if args.raw_pixels or args.unit_weights else 'per_source_feature_group',
        source_track_triplet_scale=args.triplet_scale,
        checkpoint=str(folder/'state.npz'))
    write(folder/'input_binding.json',dict(scene=str(field),scene_sha256=sha(field),
        parent_state_sha256=sha(OUT/args.parent/'state.npz') if args.parent else None,
        source_packets={name:sha(prepared_packet_path(name)) for name in spans},
        prepared_directory=str(PREPARED),
        station_seed=dict(path=str(OUT/args.station_seed/'curve_errors.npz'),
            sha256=sha(OUT/args.station_seed/'curve_errors.npz')) if args.station_seed else None,
        station_seed_directory=args.station_seed_dir,active_camera_shots=list(active_spans),
        passive_families=args.passive_families,passive_paths=args.passive_paths,
        mass_reference=dict(path=args.mass_reference,sha256=sha(args.mass_reference)) if args.mass_reference else None,
        weight_profile=dict(path=str(args.weight_profile),sha256=sha(args.weight_profile)) if args.weight_profile else None,
        material_weights=dict(path=args.material_weights,sha256=sha(args.material_weights)) if args.material_weights else None,
        ice_pixel_gain=args.ice_pixel_gain,
        camera_local_material=args.local_material,
        camera_local_source_detail=args.local_detail,
        objective='unweighted_squared_pixel_distance' if args.raw_pixels else 'correlated_rank_normalized_huber',
        camera_increment_groups=groups,rig_fields=rig_fields,huber_delta=None if args.raw_pixels else args.huber_delta,
        fit_roles=[0,1,2,3,4] if args.all_development else [0],
        role_meaning='inherited roles have already been reused for model selection; no independent final test claim',
        rig_policy='shared increments per programme and assigned camera; original exposure-specific rig differences retained',
        source_uv_policy='current source-admitted points; original and corrected source measurements retained separately'))
    save(folder/'selected_rows.npz',shot=rows['shot'],origin_row=rows['origin_row'],pts=rows['pts'],original_fold=rows['fold'],weight=w)
    begin=monotonic()
    binding=dict(camera=mcam,controls=c,rigs=rigs,camera_ids=bound['camera'],path_ids=bound['path'],
          stations=bound['station'],pts=bound['pts'],observed=bound['uv'],motion_times=bound['times'])
    problem=(native.resident(**binding,scene_directions=directions) if tracks is None else
             motion_tracks.resident(native,**binding,directions=directions))
    with problem as p:
        half=float(np.max(abs(rigs[:,37]))*.65+1e-6)
        if args.fit_readout:
            half=max(half,.0101)
        if optical is not None:
            lower=np.full(rigs.shape,-np.inf);upper=np.full(rigs.shape,np.inf)
            p.configure_camera_parameters(directions=optical,max_readout=2*half,
                lower=lower.ravel(),upper=upper.ravel())
            for name,rspan in rig_spans.items():
                if name not in active_spans:continue
                side='negative_x_end' if setup['shots'][name]['camera_index']==2 else 'negative_y'
                for index in range(rspan['start'],rspan['stop']):p.configure_mount_constraints(side=side,camera_index=index)
        else:p.configure_projection_time_support(half_seconds=half)
        bounds=p._path_supports[p._codes[:n]]
        window=np.where(np.char.startswith(rows['path'],'board/continuous/'),.035,.30)
        lo=np.maximum(bounds[:,0],rows['station']-window)
        hi=np.minimum(bounds[:,1],rows['station']+window)
        ids=np.arange(n,dtype=np.int32)
        factors=dict(local_ids=ids.copy(),
            group_offsets=np.arange(n+1,dtype=np.int32),members=ids,ranks=np.full(n,2,np.int32),
            weight_offsets=np.arange(n+1,dtype=np.int32)*4,weights=white.ravel())
        if tracks is not None:factors=motion_tracks.factors(n,white,tracks)
        if extent.any():
            keep=np.ones(len(factors['weights']),bool)
            keep[(4*np.flatnonzero(extent)[:,None]+[2,3]).ravel()]=False
            sizes=np.diff(factors['weight_offsets']).copy()
            sizes[:n]-=2*extent
            factors['weights']=factors['weights'][keep]
            factors['weight_offsets']=np.r_[0,np.cumsum(sizes,dtype=np.int32)].astype(np.int32)
            factors['ranks'][:n][extent]=1
        variable=~(rows['station_fixed']|extent)
        factors['local_ids'][:n]=np.where(variable,np.cumsum(variable)-1,-1)
        scales=np.full(int(variable.sum()),.0015)
        if args.raw_pixels:
            plan=p.configure_pixel_linearization(directions=motion,local_scales=scales,
                local_ids=factors['local_ids'])
        else:
            plan=p.configure_linearization(directions=motion,local_scales=scales,huber_delta=args.huber_delta,**factors)
        gpu=p.configure_solver(lower=lo[variable],upper=hi[variable],device_budget_bytes=11*1024**3,
              host_budget_bytes=24*1024**3,initial_lambda_relative=.001,damping_metric='jacobi_image',
              profile_trial_stations=args.profile_stations)
        all_initial=p.evaluate_errors();initial=all_initial[:n]
        if not np.isfinite(all_initial).all():raise ValueError('Invalid source projections in current input')
        start_stats=summaries(initial,rows)
        log('prepared',rows=n,plan=plan,gpu=gpu,setup_seconds=monotonic()-begin,by_programme_layer=start_stats)
        def checkpoint():
            a=p.export_state();result=dict(base)
            result['scene_values']=a.get('scene_values',scene)
            result['scene_path']=str(field)
            result['layout_revision']=state.layout['layout_revision']
            result['scene_values_sha256']=numeric_sha(result['scene_values'])
            for name,(start,stop) in spans.items():
                result[name+'_controls']=a['controls'][start:stop]
                result[name+'_active_folds']=np.arange(5,dtype=np.int8) if args.all_development else np.array([0],np.int8)
                if optical is not None:
                    slot=rig_spans[name]
                    exported_rigs=np.asarray(a['rig']).reshape(rigs.shape)
                    result[name+'_rigs']=exported_rigs[slot['start']:slot['stop']][slot['inverse']]
                    assert result[name+'_rigs'].shape==base[name+'_rigs'].shape
            save(folder/'state.npz',**result)
        def progress(rec):
            if rec['summary']['accepted']:checkpoint()
            log('trial',**rec,checkpoint=str(folder/'state.npz'))
            return not (folder/'stop').exists()
        answer=p.solve(relative_cost_tolerance=1e-6,step_tolerance=1e-6,progress=progress,
            numerical_progress=lambda rec:log('factorization_retry',**rec))
        checkpoint()
        errors=p.evaluate_errors()[:n]
        save(folder/'fitted_stations.npz',shot=rows['shot'],origin_row=rows['origin_row'],
             stations=p.export_state()['stations'][:n])
        save(folder/'errors.npz',before=initial,after=errors,shot=rows['shot'],family=rows['family'],pts=rows['pts'])
        report=dict(before=start_stats,after=summaries(errors,rows),solver=finite(answer),
                    seconds=monotonic()-begin,source_rows=n,user_visual_acceptance=False)
        write(folder/'report.json',report)
        log('finish',**report)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=('scene','joint','camera'),default='joint')
    ap.add_argument('--parent')
    ap.add_argument('--stride',type=int,default=19)
    ap.add_argument('--programmes',nargs='+')
    ap.add_argument('--shots',nargs='+')
    ap.add_argument('--camera-shots',nargs='+',help='Only these camera trajectories/calibrations vary; remaining source cameras are fixed')
    ap.add_argument('--passive-families',nargs='+',help='Full-density relevant geometry families from fixed source cameras')
    ap.add_argument('--materials',nargs='+',help='Source material types for a measurement-specific ablation')
    ap.add_argument('--passive-paths',nargs='+',help='Exact finite geometry paths consumed at full density from the fixed cameras')
    ap.add_argument('--mass-reference',help='Fixed programme normalization computed from all current full source packets')
    ap.add_argument('--rig',action='store_true')
    ap.add_argument('--rig-fields',nargs='+',help='Explicit subset of enabled native physical calibration coordinates')
    ap.add_argument('--rig-profile',help='Explicit named linear combinations of physical chart coordinates shared by installation')
    ap.add_argument('--freeze-motion',action='store_true',help='Hold every existing exposure trajectory fixed while fitting scene/calibration directions')
    ap.add_argument('--rig-roll',action='store_true',help='Release existing physical reference roll per camera increment group')
    ap.add_argument('--fit-readout',action='store_true',help='Fit the signed native rolling-shutter readout from moving source features')
    ap.add_argument('--rig-kinematics',action='store_true',help='Release existing physical axes, lever, pupil and additional zoom lens coordinates')
    ap.add_argument('--tracks',action='store_true')
    ap.add_argument('--texture-dir',help='Additional forward/backward verified native source board patches')
    ap.add_argument('--source-image-metrics',help='Optional two-axis source patch precision by original displacement row')
    ap.add_argument('--rebind-texture',action='store_true',help='Recompute conditional depths from measured left texture pixels after an installation change')
    ap.add_argument('--flow-gain',type=float,default=1.,help='Scale measured displacement groups before adding absolute photo observations')
    ap.add_argument('--detail-dir',help='Source-supported exposure indices for local single-exposure trajectory corrections')
    ap.add_argument('--local-material',help='Limit changed spline coefficients to exposures surrounding this actual measured material')
    ap.add_argument('--local-detail',action='store_true',help='Limit changed spline coefficients to the measured anomalous exposure window')
    ap.add_argument('--paths',nargs='+',help='Select exact paths for a geometry-only ablation')
    ap.add_argument('--raw-pixels',action='store_true',help='Preserve unweighted squared pixel likelihood; excludes correlated flow factors')
    ap.add_argument('--unit-weights',action='store_true',help='Unit absolute source pixel weights with explicit correlated Huber/flow objective')
    ap.add_argument('--huber-delta',type=float,default=2.7,help='Transition scale for the native robust pixel and source-track likelihood')
    ap.add_argument('--triplet-scale',type=float,default=0.,help='Actual source three-exposure curvature contrast in the same track group')
    ap.add_argument('--weight-profile',help='Current prepared per-programme source path and family weighting')
    ap.add_argument('--material-weights',help='Source measurement precision multipliers by material type')
    ap.add_argument('--ice-pixel-gain',type=float,default=1.,help='Precision multiplier for original ice pixels, including mixed-material packets')
    ap.add_argument('--photo-bindings',help='Current prepared typed rank1 photo material references for a fixed scene')
    ap.add_argument('--dynamic-photos',action='store_true',help='Keep fixed material stations on their native paths so photo factors follow scene changes')
    ap.add_argument('--profile-stations',action='store_true',help='Native inner station profiling for raw pixel factors')
    ap.add_argument('--all-development',action='store_true',help='Final refit of all admitted, previously reused development roles; preserves original role labels')
    ap.add_argument('--current-track-depths',action='store_true',help='Rebind conditional depths from unchanged left source UV through the current native camera')
    ap.add_argument('--dense-source-tracks',action='store_true',help='Retain all eligible source patches; normalize existing per-exposure/lag mass without image-strip thinning')
    ap.add_argument('--source-visibility',action='store_true',help='Apply source foreground patch exclusion to the affected Okada flow rows')
    ap.add_argument('--station-seed')
    ap.add_argument('--station-seed-dir',help='Per-shot finite nearest material stations from full-pixel review')
    ap.add_argument('--scene-blocks',nargs='+')
    ap.add_argument('--ice-coordinates',nargs='+',help='Explicit named ice coordinates for a source-driven geometry ablation')
    ap.add_argument('--families',nargs='+')
    ap.add_argument('--board',action='store_true',help='Open the five existing continuous-board shape coordinates, retaining length gauge')
    ap.add_argument('--reset-camera',action='store_true',help='Use registered original cameras with the selected parent geometry')
    run(ap.parse_args())
