"""Combine cross-angle source measurements and re-estimate physical cameras."""
import shutil

import numpy as np

from feedback_fit import PRIOR, native_run, relative
from feedback_source import CASES, DEST, PARENT
from review import camera_state, native_for
from shared import (OUT, PREPARED, NativeSceneGPU, ParameterState, arrays, numeric_sha,
                    prepared_packet_path, read, save, sha, write)
from rinkspace.core.native import NativeError


def roles(name,q,kind):
    n=len(q.get('idx',[]))
    result=np.full(n,9,dtype='i1')
    if not n:return result
    for path in np.unique(q['path']):
        ix=np.flatnonzero(q['path']==path)
        for material in np.unique(q['material'][ix]):
            kk=ix[q['material'][ix]==material]
            idx,at=np.unique(q['idx'][kk],return_index=True)
            for j,i in enumerate(idx):
                rows=kk[q['idx'][kk]==i];r=rows[0];fam=q['family'][r];sec=q['pts'][r]/12800.
                near=min([abs(sec-time) for time,fs in CASES.get(name,[]) if fam in fs] or [999.])
                if kind=='dense_tips':
                    peers=at[(abs(idx-i)<=12)&(idx!=i)]
                    if not len(peers) or not np.any(abs(q['shift'][kk[peers],0]-q['shift'][r,0])<15.):continue
                if kind=='print_reference':
                    if q['correlation'][r]<.76 or q['peak_gap'][r]<.035:continue
                    if name=='nayeon_s03' and q['blur_width'][r]>31:continue
                held=(near<.049 or name in ('chen_s13','jin_s14') or (near>1.2 and i%7==1))
                result[rows]=6 if held else 0
    return result


def prepare():
    folder=DEST/'verified';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');report={}
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);s['observations']=str(old);a=None;entry=[]
        for kind,number in [('stripes',7),('dense_tips',8),('print_reference',9)]:
            file=DEST/kind/(name+'.npz')
            if not file.exists():continue
            q=arrays(file);role=roles(name,q,kind);rows=np.flatnonzero(role<9)
            if not len(rows):continue
            if q['source_sha256'].item()!=s['source_sha256']:raise ValueError(name)
            if a is None:a=arrays(old)
            same=np.zeros(len(a['idx']),bool)
            for p in np.unique(q['path'][rows]):
                rr=rows[q['path'][rows]==p]
                for m in np.unique(q['material'][rr]):
                    mm=rr[q['material'][rr]==m]
                    previous={'stripes':'review_paint_midpoint','dense_tips':m,'print_reference':'review_print_edge'}[kind]
                    same|=(a['path']==p)&(a['material']==previous)&np.isin(a['idx'],q['idx'][mm])
            same&=~a['station_fixed']
            # The exact old UV and original role remain recoverable.
            a['fold'][same]=9
            a['admission_reason']=np.where(same,'replaced_by_material_remeasurement',
                                           a.get('admission_reason','inherited_source_role'))
            code=max(map(int,s['source_packets']))+1
            vals=dict(pts=q['pts'][rows],idx=q['idx'][rows],uv=q['uv'][rows],station=q['station'][rows],
                path=q['path'][rows],family=q['family'][rows],material=q['material'][rows],fold=role[rows],
                source_fold=role[rows],station_fixed=False,source_packet=code,source_row=rows,
                origin_row=-number*10000000-rows,baseline_display=False,
                admission_reason='source_review_verified',restored_admission_reason='source_review_verified')
            for k,col in a.items():
                v=np.broadcast_to(vals[k],(len(rows),)+col.shape[1:])
                a[k]=np.concatenate((col,v.astype(col.dtype if col.dtype.kind!='U' else str)))
            s['source_packets'][str(code)]=dict(path=str(file),source_row='source measurement row',source_sha256=s['source_sha256'])
            entry.append(dict(source=str(file),fit=int((role==0).sum()),check=int((role==6).sum()),
                superseded=int(same.sum()),rejected=int((role==9).sum())))
        if a is not None:
            out=folder/(name+'.npz');save(out,**a);s['observations']=str(out)
        report[name]=entry
        print(name,sum(r['fit'] for r in entry),flush=True)
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup);write(folder/'source_audit.json',report)
    wt=read(DEST/'camera_weights.json')
    wt.update(outline_xmin=13.,outline_xmax=13.,review_verified_stripe=6.,registered_same_print=3.8)
    write(DEST/'verified_weights.json',wt)


def combine(label,parent,branches,shots=None):
    b=camera_state(relative(parent));initial=b['scene_values'].copy();delta=np.zeros_like(initial)
    n=native_for(relative(parent),'jin');files={}
    for path,scale in branches:
        z=camera_state(relative(path));change=(z['scene_values']-initial)*scale
        if np.any((delta!=0)&(change!=0)):raise ValueError(('overlap',path))
        delta+=change;files[relative(path)]=dict(sha256=sha(path/'state.npz'),fraction=scale)
    for frac in (1.,.999,.99,.97,.93,.85,.65,.4):
        values=initial+frac*delta
        try:
            NativeSceneGPU(n.scene_path,n.core,parameter_state=ParameterState(n.parameter_state.layout,values))
            break
        except NativeError:continue
    else:raise ValueError('Infeasible native common scene')
    b['scene_values']=values;b['scene_values_sha256']=numeric_sha(values)
    origins={}
    for name,source in (shots or {}).items():
        z=camera_state(relative(source))
        for suffix in ('_controls','_rigs','_active_folds'):b[name+suffix]=z[name+suffix]
        origins[name]=dict(source=relative(source),sha256=sha(source/'state.npz'))
    save(DEST/label/'state.npz',**b)
    write(DEST/label/'inputs.json',dict(parent=relative(parent),geometry=files,combined_fraction=frac,cameras=origins))


def geometry(branch,parent='mount_candidate'):
    fam,coords={
        'r4':(['R4','lane4','house_connection_F4_R4'],['R_column_position/3','FR_row_alignment/3','house_ring_radius/30','house_ring_radius/31']),
        'f4':(['F4','house_connection_F4_R4'],['F_column_position/3','house_ring_radius/14','house_ring_radius/15']),
        'crease':(['goal_crease_left','goal_line_left','lane1','lane2','lane3','lane4'],
             ['crease_semiaxes/0','crease_semiaxes/1','hockey_carrier_position/3','hockey_carrier_slope/3']),
        'blue':(['B1'],['hockey_carrier_position/1','hockey_carrier_slope/1']),
        'c0':(['C','C0'],['hockey_carrier_position/0','hockey_carrier_slope/0','hockey_circle_radius/0'])
    }[branch]
    opts=['--mode','scene','--families',*fam,'--scene-blocks','scene/ankara_2026.field/ice_tolerance',
          '--ice-coordinates',*coords,'--unit-weights','--station-seed-dir',DEST/'stations',
          '--material-weights',DEST/'verified_weights.json']
    native_run(DEST/'verified_geometry'/branch,relative(DEST/parent),opts)


def camera(name,parent='refined_scene',stage='refit',stride=12):
    opts=['--mode','camera','--shots',name,'--stride',stride,'--unit-weights','--huber-delta',3.5,
          '--station-seed-dir',DEST/'stations','--material-weights',DEST/'verified_weights.json',
          '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
          '--texture-dir',DEST/'texture','--rebind-texture','--triplet-scale',4.,'--flow-gain',1.6]
    if name.startswith('nayeon'):opts+=['--photo-bindings',OUT/'repair/prepared_v2/nayeon_photo_bindings.npz']
    native_run(DEST/stage/name,relative(DEST/parent),opts)


def balance():
    """Admit visible counterviews of the same rings, including physical horizontal extents."""
    folder=DEST/'balanced';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    report={}
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);s['observations']=str(old);a=None;report[name]=[]
        if s['camera_index']==2:continue
        for tip in (False,True):
            f=DEST/('tips' if tip else 'reads')/(name+'.npz');q=arrays(f)
            if 'family' not in q:continue
            role=np.full(len(q['pts']),9,dtype=np.int8)
            for fam in ('F4','R4','goal_crease_left'):
                rr=np.flatnonzero(q['family']==fam)
                idx,at=np.unique(q['idx'][rr],return_index=True)
                for j,i in enumerate(idx):
                    ix=rr[q['idx'][rr]==i];r=ix[0];delta=q['shift'][r]
                    if tip:
                        if q['contrast'][r]<5.5:continue
                    else:
                        neighbours=at[(abs(idx-i)<=100)&(idx!=i)]
                        if (len(ix)<23 or not len(neighbours) or np.linalg.norm(np.ptp(q['uv'][ix],axis=0))<80
                                or not np.any(np.max(abs(q['shift'][rr[neighbours]]-delta),axis=1)<20)):continue
                    role[ix]=6 if (name in ('chen_s11','jin_s18') or j%6==2) else 0
            rr=np.flatnonzero(role<9)
            if not len(rr):continue
            if a is None:a=arrays(old)
            assert q['source_sha256'].item()==s['source_sha256']
            source_code=max(map(int,s['source_packets']))+1
            origin=-(60000000 if tip else 50000000)-rr;new=~np.isin(origin,a['origin_row'])
            rr=rr[new];origin=origin[new]
            if not len(rr):continue
            ext=dict(pts=q['pts'][rr],idx=q['idx'][rr],uv=q['uv'][rr],station=q['station'][rr],
                path=q['path'][rr],family=q['family'][rr],material=q['material'][rr] if tip else 'review_ice_counterview',
                fold=role[rr],source_fold=role[rr],station_fixed=False,source_packet=source_code,source_row=rr,
                origin_row=origin,baseline_display=False,admission_reason='independent_view_material',
                restored_admission_reason='independent_view_material')
            for k,col in a.items():
                val=np.broadcast_to(ext[k],(len(rr),)+col.shape[1:])
                a[k]=np.concatenate((col,val.astype(col.dtype if col.dtype.kind!='U' else str)))
            s['source_packets'][str(source_code)]=dict(path=str(f),source_row='source measurement row',source_sha256=s['source_sha256'])
            report[name].append(dict(source=str(f),fit=int(np.sum(role[rr]==0)),check=int(np.sum(role[rr]==6))))
        if a is not None and report[name]:
            path=folder/(name+'.npz');save(path,**a);s['observations']=str(path)
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup);write(folder/'source_audit.json',report)
    weights=read(DEST/'verified_weights.json');weights['review_ice_counterview']=4.;write(DEST/'balanced_weights.json',weights)


def new_layout(parent):
    from shared import EVENT,LIBRARY,NativeCore,SceneParameterModel,safe_path
    state=camera_state(relative(parent));old=native_for(relative(parent),'jin')
    doc=read(old.scene_path);doc['shared_geometry_options']=1023
    doc['revision']='ankara_reviewed_lane_centres_and_left_crease_1118'
    doc['ice_metric_freedom'].update(schema='rinkspace.free_ice_metric.v3',native_option=1023,
        carrier_offsets='Independent left crease centre along the finite goal line and lane midlines independent of house centres; values start at zero.')
    doc['ice_metric_freedom']['retained']=[v for v in doc['ice_metric_freedom']['retained'] if 'crease' not in v or 'midpoint' not in v]
    oldsize=len(doc['shared_layout']);add=['left_crease_carrier_offset/0',
            *[f'lane_centre_at_{end}/{i}' for end in ('F','R') for i in range(4)]]
    assert oldsize==146;doc['shared_layout']+=len(add)*[0.]
    descriptor=next(q for q in doc['parameter_blocks'] if q['id']=='ice_tolerance')
    offset=descriptor['size'];descriptor['size']+=len(add);descriptor['tangent_size']+=len(add)
    descriptor['independent_local_indices']+=list(range(offset,offset+len(add)));descriptor['values_meaning']+=add
    path=safe_path(EVENT/'scene/field/crease_offset_1118.json');write(path,doc)
    model=SceneParameterModel(doc,path,optimize_scene=True);new=model.initial_state;values=new.values.copy()
    for k,block in old.parameter_state.blocks.items():
        if not block['available']:continue
        original=old.parameter_state[k];start=new.blocks[k]['offset']
        values[start:start+len(original)]=original
    native=NativeSceneGPU(path,NativeCore(LIBRARY),parameter_state=ParameterState(new.layout,values))
    for k in old.paths:
        np.testing.assert_allclose(native.paths[k]['parameters'],old.paths[k]['parameters'],rtol=0,atol=2e-14)
    state.update(scene_path=str(path),scene_values=values,scene_values_sha256=numeric_sha(values),layout_revision=new.layout['layout_revision'])
    save(DEST/'free_seed/state.npz',**state)
    write(DEST/'free_seed/inputs.json',dict(parent=relative(parent),structure=str(path),library=str(LIBRARY),
          zero_offset_validated_paths=len(old.paths),added=add))


def free_solve(which):
    if which=='crease':
        fs=['goal_crease_left'];coords=['left_crease_carrier_offset/0','crease_semiaxes/0','crease_semiaxes/1']
    elif which.startswith('lane'):
        i=int(which[-1])-1;fs=['lane'+str(i+1)]
        coords=[f'{field}/{i}' for field in ('lane_centre_at_F','lane_centre_at_R','lane_width_at_F','lane_width_at_R','lane_right_position','lane_right_slope')]
    else:
        i=3 if which=='f4' else 7
        fs=[which.upper()];coords=[('F' if i==3 else 'R')+'_column_position/3',f'house_ring_radius/{4*i+2}',f'house_ring_radius/{4*i+3}']
        if i==7:coords+=['FR_row_alignment/3']
    native_run(DEST/'free_geometry'/which,relative(DEST/'free_seed'),['--mode','scene','--families',*fs,
        '--scene-blocks','scene/ankara_2026.field/ice_tolerance','--ice-coordinates',*coords,
        '--unit-weights','--material-weights',DEST/'balanced_weights.json','--station-seed-dir',DEST/'stations'])
