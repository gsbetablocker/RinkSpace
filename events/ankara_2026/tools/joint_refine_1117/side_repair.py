"""Rebind source measurements and inspect the Ankara side and camera corrections."""
from collections import defaultdict
from pathlib import Path
import shutil

import numpy as np

from jin_board_1118 import CASES
from lane_side_1118 import BASE, WORK, CAP, SIDE, TRANSVERSE
from review import ji, camera_state, native_for
from shared import OUT, PREPARED, arrays, numeric_sha, prepared_packet_path, read, save, sha, write
from source_bundle import native_run, relative


PREVIOUS = OUT / 'review_fixes_1118'
REPORTED = {**CASES, 'nayeon_s03':[49.98], 'nayeon_s06':[125.12]}


def native_table(n,keys,count):
    result={}
    for key in keys:
        path=n.paths[key]
        if path['kind']==7:
            board=n.scene['board'];t=np.linspace(0.,n.core.board_perimeter(board),count)
            xyz=n.core.sample_path(board,7,path['parameters'],path.get('vertices',[]),t)
            result[key]=(np.asarray(xyz),t)
        else:result.update(ji.native_paths(n,[key],count=count))
    return result


def positions(candidate,name,z):
    """Evaluate the measured station in its original exposure, including sensor readout."""
    b=camera_state(candidate);n=native_for(candidate,name.split('_')[0])
    s=read(PREPARED/'setup.json')['shots'][name]
    keys,pi=np.unique(z['path'],return_inverse=True)
    tab={}
    for k in keys:
        tab.update(native_table(n,[k],count=4097 if k.startswith('board/') else 513))
    world={k:v[0] for k,v in tab.items()}
    group=z['idx']*len(keys)+pi;order=np.argsort(group,kind='stable')
    blocks=np.split(order,np.flatnonzero(np.diff(group[order]))+1)
    result=np.full_like(z['uv'],np.nan,dtype='f8');normal=result.copy()
    for lo in range(0,len(blocks),64):
        hits=blocks[lo:lo+64]
        jobs=[(z['idx'][rr[0]],z['path'][rr[0]],b[name+'_times'][z['idx'][rr[0]]]) for rr in hits]
        projected=ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for rr,(_,k,_),(u,status) in zip(hits,jobs,projected,strict=True):
            stations=tab[k][1];t=z['station'][rr];j=np.searchsorted(stations,t,side='right').clip(1,len(stations)-1)
            mix=np.clip((t-stations[j-1])/np.maximum(1.e-12,stations[j]-stations[j-1]),0.,1.)
            d=u[j]-u[j-1];ok=(status[j]==0)&(status[j-1]==0)&(np.linalg.norm(d,axis=1)>1.e-8)
            result[rr[ok]]=u[j[ok]-1]+mix[ok,None]*d[ok]
            tangent=d[ok]/np.linalg.norm(d[ok],axis=1)[:,None]
            normal[rr[ok]]=np.c_[-tangent[:,1],tangent[:,0]]
    return result,normal


def measured_scores(candidate,name,file,folder=None):
    q=arrays(file);guide,norm=positions(candidate,name,q)
    residual=np.sum((guide-q['uv'])*norm,axis=1)
    signed=np.where(abs(norm[:,0])>abs(norm[:,1]),residual*np.sign(norm[:,0]),residual*np.sign(norm[:,1]))
    q['signed']=signed;q['axes']=np.argmax(abs(norm),axis=1)
    if folder:
        save(folder/(name+'.npz'),signed=signed,axes=q['axes'],source_measurements_sha256=sha(file))
    return q


def case_rows(q,name):
    rows={}
    for sec in REPORTED[name]:
        at=q['pts']==round(sec*12800)
        for f in np.unique(q['family'][at]):
            for axis in (0,1):
                take=at&(q['family']==f)&(q['axes']==axis)&np.isfinite(q['signed'])
                for key,region in [('all',True),('left',q['uv'][:,0]<480),('right',q['uv'][:,0]>1280)]:
                    vals=q['signed'][take&region]
                    if len(vals):
                        rows[f'{sec:.3f}/{f}/{"xy"[axis]}/{key}']=dict(n=len(vals),median=float(np.median(vals)),
                            p90_abs=float(np.percentile(abs(vals),90)))
    return rows


def score(candidate,name,source='jin_source'):
    target=WORK/'checks'/candidate.replace('/','_')/source
    file=WORK/source/(name+'.npz');q=measured_scores(candidate,name,file,target)
    write(target/(name+'.json'),case_rows(q,name) if name in REPORTED else {})


def append(a,s,file,role,number):
    q=arrays(file)
    # Earlier icon registrations have no fold column. They map a template by
    # one correlation displacement, so their vertices cannot be direct UVs.
    role=np.where(q['material']=='same_source_icon',9,role).astype('i1')
    rows=np.flatnonzero(role<9)
    if len(q['idx'])!=len(role) or not np.array_equal(q['pts'],np.asarray(s['_pts'])[q['idx']]):
        raise ValueError(('Source time mismatch',file))
    if 'source_sha256' in q and q['source_sha256'].item()!=s['source_sha256']:raise ValueError(file)
    code=max(map(int,s['source_packets']))+1
    origin=-number*10000000-rows
    while np.isin(origin,a['origin_row']).any():
        number+=1
        origin=-number*10000000-rows
    material=q['material'].astype('U48')
    if file.parent.name=='verified_inputs' and file.stem=='nayeon_s03':
        foreground=(q['uv'][:,0]<630)&(q['uv'][:,1]>440)&(q['pts']>=round(49.6*12800))&(q['pts']<=round(51.34*12800))
        material[foreground&(q['family']=='board.bottom')]='grazing_kickplate'
    values=dict(pts=q['pts'][rows],idx=q['idx'][rows],uv=q['uv'][rows],station=q['station'][rows],
        path=q['path'][rows],family=q['family'][rows],material=material[rows],fold=role[rows],
        source_fold=role[rows],station_fixed=q.get('station_fixed',np.zeros(len(role),bool))[rows],
        source_packet=code,source_row=rows,origin_row=origin,baseline_display=False,
        admission_reason='measured_original_source',restored_admission_reason='measured_original_source')
    for k,col in a.items():
        v=np.broadcast_to(values[k],(len(rows),)+col.shape[1:])
        a[k]=np.concatenate((col,v.astype(col.dtype if col.dtype.kind!='U' else str)))
    s['source_packets'][str(code)]=dict(path=str(file),source_row='row in the source measurement packet',
        source_sha256=s['source_sha256'])
    return dict(packet=str(file),code=code,fit=int((role<5).sum()),check=int(((role>=5)&(role<9)).sum()),
        rejected=int((role==9).sum()))


def camera_roles(name,q):
    if 'fold' in q:return q['fold']
    held=np.zeros(len(q['idx']),bool)
    for sec in REPORTED[name]:
        held|=abs(q['pts']/12800.-sec)<.013
    held|=(q['idx']//14)%11==5
    return np.where(held,6,0).astype('i1')


def prepare(label='prepared',names=None,extra=()):
    folder=WORK/label;folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json');b=camera_state(BASE);report={}
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,s in setup['shots'].items():
        old=prepared_packet_path(name);s['observations']=str(old)
        if names and name not in names:continue
        src=[(kind,WORK/('nayeon_source/verified_inputs' if kind=='nayeon_source' else kind)/(name+'.npz'))
             for kind in ['lane_source','jin_source','nayeon_source',*extra]]
        historical=WORK/'historic'/(name+'.npz')
        if not any(f.is_file() for _,f in src) and not historical.is_file():continue
        a=arrays(old);s['_pts']=b[name+'_pts'];entry=[];before=a['fold'].copy()
        if historical.is_file():
            tentative=(a['path']==SIDE)&(a['fold']<9)&~a['station_fixed']
            a['fold'][tentative]=9
            a['admission_reason']=np.where(tentative,'upper_side_entity_unverified_on_original',a['admission_reason'])
        # A new source measurement replaces a machine reading only when its
        # physical material ambiguity is documented. Fixed human marks keep their roles.
        if name=='nayeon_s03' and (WORK/'nayeon_source/verified_inputs'/('superseded_'+name+'.npz')).is_file():
            retired=arrays(WORK/'nayeon_source/verified_inputs'/('superseded_'+name+'.npz'))['rows']
            if np.any(a['station_fixed'][retired]):raise ValueError('Fixed source mark in automatic replacement')
            a['fold'][retired]=9
            a['admission_reason']=np.where(np.isin(np.arange(len(before)),retired),'rail_surface_instead_of_front_edge',a['admission_reason'])
        changed=np.flatnonzero(before!=a['fold'])
        if len(changed):save(folder/'role_changes'/(name+'.npz'),row=changed,previous=before[changed],current=a['fold'][changed],reason=a['admission_reason'][changed])
        for number,(kind,file) in enumerate(src,12):
            if not file.exists():continue
            q=arrays(file);role=q['fold'] if kind=='lane_source' else camera_roles(name,q)
            if kind=='jin_source' and 'panel_source' in extra:
                # The original single-gradient reader can select a printed stroke,
                # or the top edge of an ad. These are historical diagnostics only.
                role=np.where((q['material']=='plate_transition')|(q['family']=='board.top'),9,role).astype('i1')
            entry.append(append(a,s,file,role,number))
        if entry or len(changed):
            dst=folder/(name+'.npz');save(dst,**a);s['observations']=str(dst)
        s.pop('_pts');report[name]=dict(parent=str(old),changed_previous_roles=len(changed),new=entry)
        print(label,name,len(changed),sum(k['fit'] for k in entry),flush=True)
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup);write(folder/'source_audit.json',report)


def texture(label='texture',extended=False):
    old=read(PREVIOUS/'texture/sources.json')
    new=WORK/'motion_evidence'
    for name in ['jin_s14','jin_s22']:
        root=Path(old[name]);original=arrays(root/(name+'.npz'));quality=arrays(root/'quality'/(name+'.npz'))
        roots=[new]
        if extended:
            if name=='jin_s14':roots=[new/'s14_extended']
            else:roots=[new,new/'onset_probe']
        packs=[original];surfaces=[0];flags=[quality['keep']]
        for folder in roots:
            for surface,sub in [(0,folder),(1,folder/'ice')]:
                p=arrays(sub/(name+'.npz'));good=p['keep'].copy()
                if folder.name=='onset_probe' and surface:
                    good[[554,556,557,558]]=False
                    # Exclude moving reflections and the athlete in the centre
                    # strip on this exposure. The identifiable left ice survives.
                    good&=~((p['pts'][p['index']]==round(154.68*12800))&(p['uv'][:,0]>530)&(p['uv'][:,0]<1390))
                packs.append(p);surfaces.append(surface);flags.append(good)
        for p in packs:np.testing.assert_array_equal(p['pts'],original['pts'])
        fields={k:np.concatenate([p[k] for p in packs]) for k in ['index','lag','left_uv','uv','held','world','valid']}
        fields['family']=np.concatenate([np.full(len(p['index']),surface,dtype='i1') for surface,p in zip(surfaces,packs)])
        if extended:fields['spatial_bin']=(fields['left_uv'][:,0]//320).astype('i1')
        saved=WORK/label;save(saved/(name+'.npz'),**fields,pts=original['pts'],source_sha256=original['source_sha256'])
        save(saved/'quality'/(name+'.npz'),keep=np.concatenate(flags),pts=original['pts'])
        old[name]=str(saved)
        sec=original['pts']/12800.
        limits=(122.10,123.96) if name=='jin_s14' else (153.85,155.09)
        save(WORK/'detail'/(name+'.npz'),transient=np.flatnonzero((sec>=limits[0])&(sec<=limits[1])))
    write(WORK/label/'sources.json',old)


def camera(name,stage,parent,stride,flow_gain,texture=None,detail=False,weights=None,rig=()):
    options=['--mode','camera','--shots',name,'--stride',str(stride),'--unit-weights','--huber-delta','2.7',
        '--station-seed-dir',PREVIOUS/'stations','--material-weights',weights or WORK/'precision.json',
        '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
        '--texture-dir',texture or PREVIOUS/'texture','--rebind-texture','--triplet-scale','4','--flow-gain',str(flow_gain)]
    if detail:options+=['--detail-dir',WORK/'detail','--local-detail']
    if rig:options+=['--rig','--rig-roll','--rig-fields',*rig]
    native_run(WORK/stage/name,parent,options)


def field():
    evidence=read(WORK/'lane_source/summary.json')
    old=camera_state(BASE);n=native_for(BASE,'jin')
    start=n.parameter_state.blocks['scene/ankara_2026.field/ice_tolerance']['offset']-39
    ref=old['scene_values'].copy();v=ref.copy()
    if evidence['admitted_rows']:
        native_run(WORK/'lane_fit',BASE,['--mode','scene','--paths',SIDE,'--materials','traced_lane4_upper',
            '--unit-weights','--scene-blocks','scene/ankara_2026.field/ice_tolerance',
            '--ice-coordinates','lane_centre_at_F/3','lane_centre_at_R/3'])
        reference=relative(WORK/'lane_fit')
    else:
        # The full-colour source at the inner transverse shows its thin paint
        # beyond the older lane_motion side as well. The carrier just before
        # the misclassified side solve spans both observed transverse limits.
        reference='review_fixes_1118/free_seed'
    target=np.asarray(native_for(reference,'jin').paths[SIDE]['parameters']).reshape(2,3)
    current=np.asarray(n.paths[SIDE]['parameters']).reshape(2,3);q=ref[start:]
    for c,w,anchor in ((150,90,q[0]+q[3]*q[46]),(154,94,q[1]+q[3]*q[50])):
        def ordinate(line):
            t=(anchor-line[0,0])/(line[1,0]-line[0,0])
            return line[0,1]+t*(line[1,1]-line[0,1])
        d=(ordinate(target)-ordinate(current))/q[9]
        v[start+c]+=.5*d;v[start+w]+=d
    old['scene_values']=v;old['scene_values_sha256']=numeric_sha(v)
    save(WORK/'field/state.npz',**old)
    candidate=native_for(relative(WORK/'field'),'jin')
    changed=[k for k in n.paths if not np.allclose(n.paths[k]['parameters'],candidate.paths[k]['parameters'],rtol=0.,atol=2e-12)]
    if set(changed)!={CAP,SIDE,TRANSVERSE}:raise ValueError(('Unexpected changed model paths',changed))
    after=np.asarray(candidate.paths[SIDE]['parameters']).reshape(2,3)
    slope=np.diff(target[:,1])/np.diff(target[:,0])
    np.testing.assert_allclose(after[:,1],target[0,1]+(after[:,0]-target[0,0])*slope,rtol=0.,atol=2e-12)
    write(WORK/'field/inputs.json',dict(parent=BASE,upper_carrier=reference,direct_new_upper_uv=evidence['admitted_rows'],
        original_pixels=str(WORK/'lane_cap_limit/limit.json'),left_transverse=str(WORK/'lane_cross_limit'),changed=changed,
        previous=n.paths[SIDE]['parameters'],after=candidate.paths[SIDE]['parameters']))


def combine(parent,branches,label='release'):
    b=camera_state(parent);sources={}
    for name,branch in branches.items():
        z=camera_state(branch)
        for k in ('_pts','_times'):np.testing.assert_array_equal(b[name+k],z[name+k])
        for k in ('_controls','_rigs','_active_folds'):b[name+k]=z[name+k]
        sources[name]=dict(candidate=branch,state_sha256=sha(OUT/branch/'state.npz'))
    save(WORK/label/'state.npz',**b);write(WORK/label/'inputs.json',dict(common=parent,cameras=sources))


def supported_field():
    """Extend the visible transverse paint up to its measured lower limit in both camera fits."""
    parent=relative(WORK/'field');b=camera_state(parent);n=native_for(parent,'jin')
    evidence=read(WORK/'lane_source/carrier_endpoint_review.json')
    left=evidence['cases']['left_transverse'];xy=left['x_current'],left['y_current']
    limits=[]
    for fr in left['frames']:
        tail=fr['raw_tail_selection']['0.00']
        for key in ('unchanged_camera_px','alternate_camera_px'):
            gaps=tail[key]
            if not gaps or gaps['current']<=0:continue
            trials=sorted([(xy[1],gaps['current'])]+[(float(k),v) for k,v in gaps.items() if k[:1].isdigit()])
            ordinates,gap=np.array(trials).T
            if np.max(np.diff(gap))>=0 or gap[-1]>0:raise ValueError((fr['pts'],key,gap))
            limits.append(dict(pts=fr['pts'],camera=key,ordinate=float(np.interp(0.,-gap,ordinates))))
    y=max([xy[1]]+[g['ordinate'] for g in limits])
    # Retain the right closure and meet the nearest tenth of a millirink that spans the visible paint.
    y=np.ceil(y*10000)/10000
    endpoints=np.asarray(n.paths[SIDE]['parameters']).reshape(2,3);right=endpoints[1,:2]
    start=n.parameter_state.blocks['scene/ankara_2026.field/ice_tolerance']['offset']-39
    v=b['scene_values'].copy();q=v[start:]
    for c,w,anchor in ((150,90,q[0]+q[3]*q[46]),(154,94,q[1]+q[3]*q[50])):
        delta=(y-xy[1])*(right[0]-anchor)/(right[0]-xy[0])/q[9]
        v[start+c]+=.5*delta;v[start+w]+=delta
    b['scene_values']=v;b['scene_values_sha256']=numeric_sha(v)
    save(WORK/'common_field/state.npz',**b)
    final=native_for(relative(WORK/'common_field'),'jin')
    chosen=np.asarray(final.paths[SIDE]['parameters']).reshape(2,3)
    np.testing.assert_allclose(chosen[1],endpoints[1],rtol=0,atol=2.e-9)
    cross=np.asarray(final.paths[TRANSVERSE]['parameters']).reshape(2,3)
    if abs(cross[:,1].max()-y)>1.e-5:raise ValueError((cross,y))
    initial=native_for(BASE,'jin')
    changed=[k for k in final.paths if not np.allclose(final.paths[k]['parameters'],initial.paths[k]['parameters'],rtol=0,atol=2e-12)]
    assert set(changed)=={CAP,SIDE,TRANSVERSE}
    write(WORK/'common_field/inputs.json',dict(free_seed=read(WORK/'field/inputs.json'),actual_transverse=str(WORK/'lane_cross_limit/limit.json'),
        image_comparisons=str(WORK/'lane_source/carrier_endpoint_review.json'),transverse_bounds=limits,
        selected_transverse_y=float(y),upper_endpoints=chosen.tolist(),changed=changed,
        direct_upper_visibility=read(WORK/'lane_source/summary.json')['admitted_rows']))


def shared_optics(names,parent,stage):
    # A single constant calibration update for each existing physical
    # installation. Only the pan/tilt/zoom splines vary with exposure.
    options=['--mode','camera','--shots',*names,'--stride','5','--unit-weights','--rig','--rig-roll',
        '--rig-fields','R0_z','eta','k10','p1','p2','wide_k1','--huber-delta','2.7',
        '--station-seed-dir',PREVIOUS/'stations','--material-weights',WORK/'surface_precision.json',
        '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
        '--texture-dir',WORK/'text_wide','--rebind-texture','--triplet-scale','4','--flow-gain','5.']
    native_run(WORK/stage,parent,options)


def other_surface_tracks():
    target=WORK/'text_four';origins=read(WORK/'text_wide/sources.json')
    for name,folder,ice in [('okada_s07','okada_s07_extended',False),('chen_s20','chen_s20_full',False),
                             ('chen_s12','programme_probes',True),('nayeon_s06','programme_probes',True)]:
        root=Path(origins[name]);old=arrays(root/(name+'.npz'))
        flags=[arrays(root/'quality'/(name+'.npz'))['keep']]
        packs=[old];surfaces=[np.zeros(len(old['index']),dtype='i1')]
        for code in range(2 if ice else 1):
            src=WORK/'motion_evidence'/folder
            if code:src=src/'ice'
            if not (src/(name+'.npz')).is_file():continue
            a=arrays(src/(name+'.npz'));np.testing.assert_array_equal(a['pts'],old['pts'])
            assert a['source_sha256'].item()==old['source_sha256'].item()
            packs.append(a);surfaces.append(np.full(len(a['index']),code,dtype='i1'));flags.append(arrays(src/'quality'/(name+'.npz'))['keep'])
        vals={k:np.concatenate([p[k] for p in packs]) for k in ['index','lag','left_uv','uv','held','world','valid']}
        save(target/(name+'.npz'),**vals,pts=old['pts'],source_sha256=old['source_sha256'],family=np.concatenate(surfaces),
            spatial_bin=(vals['left_uv'][:,0]//320).astype('i1'))
        save(target/'quality'/(name+'.npz'),keep=np.concatenate(flags),pts=old['pts']);origins[name]=str(target)
    write(target/'sources.json',origins)


def release_inputs():
    """Retain the raw packets, record confirmed material changes and bind the final camera measurements."""
    if PREPARED.resolve()!=(WORK/'admitted').resolve():raise ValueError(PREPARED)
    folder=WORK/'verified';folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json');audit=read(PREPARED/'source_audit.json')
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,shot in setup['shots'].items():
        parent=prepared_packet_path(name);shot['observations']=str(parent);key=folder/(name+'.npz')
        flags=WORK/'appearance_audit/retire'/(name+'.npz')
        if not flags.is_file() and name not in ('chen_s20','nayeon_s03'):continue
        a=arrays(parent);entry=audit.setdefault(name,dict(parent=str(parent),changed_previous_roles=0,new=[]))
        if name=='chen_s20':
            cdir=WORK/'motion_evidence/chen_s20_registration/prepared';extra=arrays(cdir/(name+'.npz'))
            for col in a:
                if col!='fold':np.testing.assert_array_equal(a[col],extra[col][:len(a['pts'])])
            diff=arrays(cdir.parent/'changed_source_roles.npz');ii=diff['rows']
            np.testing.assert_array_equal(ii,np.flatnonzero(a['fold']!=extra['fold'][:len(a['pts'])]))
            np.testing.assert_array_equal(a['fold'][ii],diff['previous_fold']);assert not np.any(a['station_fixed'][ii])
            np.testing.assert_array_equal(extra['fold'][ii],diff['current_fold'])
            entry['new_panel_corners']=dict(packet=str(cdir/(name+'.npz')),extra_rows=len(extra['pts'])-len(a['pts']),
                correlated_border_replacements=str(cdir.parent/'changed_source_roles.npz'),replaced=len(ii))
            a=extra
            shot['source_packets']=read(cdir/'setup.json')['shots'][name]['source_packets']
        if name=='nayeon_s03':
            nd=WORK/'nayeon_guard/native/partial';u=arrays(nd/'admitted_role_updates.npz');ii=u['rows']
            for col in ('pts','uv','path'):np.testing.assert_array_equal(a[col][ii],u[col])
            np.testing.assert_array_equal(a['fold'][ii],u['prior_fold'])
            assert not a['station_fixed'][ii].any()
            a['fold'][ii]=u['fold']
            entry['developed_source_roles']=dict(count=len(ii),row_map=str(nd/'admitted_role_updates.npz'))
            # Whole-print registration is a measured translation. Its representative
            # station is a factor coordinate, not an individually read image feature.
            shot['whole_print_translation_factors']=dict(rows=str(nd/'new_render_uv.npz'),
                registration=str(PREVIOUS/'print_reference/left_end_transfer.npz'),roles=str(nd/'source_roles.json'))
            entry['whole_print_translation_factors']=shot['whole_print_translation_factors']
        if flags.is_file():
            q=arrays(flags);idx=q['row'];assert len(np.unique(idx))==len(idx)
            assert q['source_sha256'].item()==shot['source_sha256'] and q['source_packet_sha256'].item()==sha(parent)
            for col in ('pts','uv','path'):np.testing.assert_array_equal(a[col][idx],q[col])
            if np.any(a['station_fixed'][idx]):raise ValueError(('Fixed mark in retirement evidence',name))
            previous=a['fold'][idx].copy();a['fold'][idx]=9
            for col in ('admission_reason','restored_admission_reason'):
                if col in a:a[col]=np.where(np.isin(np.arange(len(a['pts'])),idx),'confirmed_other_material_in_source',a[col])
            entry['confirmed_material_errors']=dict(rows=len(idx),prior_roles={str(int(k)):int((previous==k).sum()) for k in np.unique(previous)},
                audit=str(flags),audit_sha256=sha(flags))
        save(key,**a);shot['observations']=str(key)
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup);write(folder/'source_audit.json',audit)


def inspection_inputs(label, existing, new_absolute, excluded, replacements=None):
    """Carry the previous source ledger forward and bind reads from the full-shot review."""
    folder=WORK/label
    folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json')
    audit=read(PREPARED/'source_audit.json')
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    state=camera_state(relative(WORK/'release'))
    replacements=replacements or {}
    for name,shot in setup['shots'].items():
        parent=prepared_packet_path(name)
        shot['observations']=str(parent)
        if name not in existing and name not in new_absolute and name not in excluded and name not in replacements:
            continue
        a=arrays(parent)
        changes=[]
        for record in existing.get(name,[]):
            r=arrays(record)
            rows=r['row'] if 'row' in r else r['admitted_row']
            if 'role' in r:
                keep=r['role']=='repair_support'
            else:
                keep=np.ones(len(rows),bool)
            for key in ('pts','path','uv'):
                np.testing.assert_array_equal(a[key][rows],r[key])
            use=rows[keep]
            if np.any(a['fold'][use]>=9):
                raise ValueError(('Excluded observation in new fit',name,record))
            promoted=np.unique(use[a['fold'][use]>=5])
            a['fold'][promoted]=0
            changes.append(dict(existing_fit=str(record),sha256=sha(record),rows=len(use),
                                promoted_rows=promoted.tolist()))
        for record in excluded.get(name,[]):
            r=arrays(record);rows=r['row'] if 'row' in r else r['rows']
            for key in ('pts','path','uv'):
                np.testing.assert_array_equal(a[key][rows],r[key])
            if np.any(a['station_fixed'][rows]):
                raise ValueError(('Fixed observation in source retirement',name,record))
            old=a['fold'][rows].copy()
            a['fold'][rows]=9
            for key in ('admission_reason','restored_admission_reason'):
                if key not in a:continue
                labels=a[key].astype('U100')
                labels[rows]='source_pixel_on_kickplate_or_white_band'
                a[key]=labels
            changes.append(dict(excluded=str(record),sha256=sha(record),rows=rows.tolist(),
                                previous_roles=old.tolist()))
        shot['_pts']=state[name+'_pts']
        for record,reason in replacements.get(name,[]):
            r=arrays(record)
            rows=r['row']
            for key in ('pts','path'):
                np.testing.assert_array_equal(a[key][rows],r[key])
            original=r['old_uv'] if 'old_uv' in r else r['original_uv']
            np.testing.assert_array_equal(a['uv'][rows],original)
            if len(np.unique(rows))!=len(rows) or np.any(a['station_fixed'][rows]):
                raise ValueError(('Unexpected replacement of a fixed or repeated row',name,record))
            roles=r['fold'].astype('i1')
            if (np.any(roles<0) or np.any(roles>=9) or
                    ('role' in r and not np.array_equal(roles<5,r['role']=='repair_support'))):
                raise ValueError(('Source reading has ambiguous roles',name,record))
            old=a['fold'][rows].copy()
            a['fold'][rows]=9
            for key in ('admission_reason','restored_admission_reason'):
                if key not in a:continue
                labels=a[key].astype('U100')
                labels[rows]=reason
                a[key]=labels
            changes.append(dict(replaced_source_pixels=str(record),sha256=sha(record),rows=rows.tolist(),
                                previous_roles=old.tolist(),new_reads=append(a,shot,record,roles,77)))
        for record in new_absolute.get(name,[]):
            r=arrays(record)
            fold=r['fit_fold']
            if np.any(fold>=5):
                raise ValueError(('Unexpected role in actual fit record',name,record))
            changes.append(append(a,shot,record,fold,75))
        shot.pop('_pts')
        output=folder/(name+'.npz')
        save(output,**a)
        shot['observations']=str(output)
        audit.setdefault(name,{}).setdefault('full_timeline_corrections',[]).extend(changes)
    setup['parent']=str(PREPARED)
    write(folder/'setup.json',setup)
    write(folder/'source_audit.json',audit)


def protect_source_materials(label='guard_inputs',verified_only=False):
    """Identify the ink and printed sides confirmed against the original adjacent exposures."""
    folder=WORK/label;folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');report={}
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,s in setup['shots'].items():
        file=prepared_packet_path(name);s['observations']=str(file)
        if name not in ('jin_s08','jin_s14','jin_s15','okada_s07'):continue
        q=arrays(file);t=q['pts']/12800.;m=q['material'];p=q['path']
        if name=='jin_s08':
            in_view=(t>=71.67)&(t<=72.22)
            rows=in_view&(p=='print_instance/Ad3::Ad3.right_vertical')&np.isin(m,['whole_panel_side','panel_edge'])
        elif name=='jin_s14':
            in_view=(t>=122.88)&(t<=123.24)
            rows=in_view&(p=='demo_initial/goal_line_left:0')&np.isin(m,['inherited_qualified_source','reviewed_prior_source'])
        elif name=='jin_s15':
            rows=(t>=124.)&(t<=124.32)&(q['family']=='Ad13')&(m=='red_badge_outer_print_stroke')
        else:
            rows=(t>=113.45)&(t<=114.50)&(p=='board/continuous/top')&(q['uv'][:,0]<350)&(q['uv'][:,1]>700)
        if verified_only:
            source=arrays(WORK/'appearance_audit/repair'/(name+'.npz'))
            ix=source['row']
            for col in ('pts','uv','path'):np.testing.assert_array_equal(q[col][ix],source[col])
            rows=np.zeros(len(m),bool);rows[ix]=True
        rows&=q['fold']<5
        if not rows.any():raise ValueError(('No qualified protection',name))
        q['material']=np.where(rows,'protected_verified_material',m)
        dst=folder/(name+'.npz');save(dst,**q);s['observations']=str(dst)
        report[name]=dict(parent=str(file),rows=int(rows.sum()),pts=np.unique(q['pts'][rows]).tolist(),
            labels=list(map(str,np.unique(m[rows]))))
    write(folder/'setup.json',setup);write(folder/'protected.json',report)


def guard_window(name,parent,stage,lo,hi,weights,gain,features=None,basis=None):
    b=camera_state(parent);t=b[name+'_pts']/12800.
    detail=WORK/stage/'detail';save(detail/(name+'.npz'),transient=np.flatnonzero((t>=lo)&(t<=hi)),
                                     **({'motion_basis':basis} if basis is not None else {}))
    opt=['--mode','camera','--shots',name,'--stride','2','--unit-weights','--huber-delta','2.7',
         '--station-seed-dir',PREVIOUS/'stations','--material-weights',weights,
         '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
         '--texture-dir',features or WORK/'text_four','--rebind-texture','--triplet-scale','4','--flow-gain',str(gain),
         '--detail-dir',detail,'--local-detail']
    native_run(WORK/stage/name,parent,opt)


def prepare_window_anchors(name, sources, folder):
    """Use the verified rows of the current packet in a local camera adjustment."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    setup=read(PREPARED/'setup.json')
    for key,shot in setup['shots'].items():
        shot['observations']=str(prepared_packet_path(key))
    packet=prepared_packet_path(name)
    a=arrays(packet)
    labels=a['material'].astype('U80')
    bindings=[]
    for path,label in sources:
        r=arrays(path)
        keep=r['role']=='repair_support' if 'role' in r else np.ones(len(r['pts']),bool)
        rows=r['row'][keep]
        for field in ('pts','path','uv'):
            np.testing.assert_array_equal(a[field][rows],r[field][keep])
        if np.any(a['fold'][rows]>=9):
            raise ValueError(('Excluded row used as an anchor',path))
        if not len(rows):
            raise ValueError(('Empty source selection',path))
        labels[rows]=label
        promoted=rows[a['fold'][rows]>=5]
        a['fold'][promoted]=0
        bindings.append(dict(file=str(path),sha256=sha(path),label=label,rows=len(rows),
                             promoted=promoted.tolist()))
    a['material']=labels
    dest=folder/(name+'.npz')
    save(dest,**a)
    setup['shots'][name]['observations']=str(dest)
    write(folder/'setup.json',setup)
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    write(folder/'window_anchors.json',dict(parent=str(packet),parent_sha256=sha(packet),sources=bindings))


def reviewed_surface_source(name,record):
    folder=WORK/('source_'+name);folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json')
    for key,shot in setup['shots'].items():shot['observations']=str(prepared_packet_path(key))
    p=prepared_packet_path(name);a=arrays(p);reviewed=arrays(record)
    rows=reviewed['row'][reviewed['role']=='repair_support']
    for key in ('uv','pts','path'):np.testing.assert_array_equal(a[key][rows],reviewed[key][reviewed['role']=='repair_support'])
    assert np.all(a['fold'][rows]<9) and np.all(~a['station_fixed'][rows])
    a['fold'][rows]=0
    a['material']=np.where(np.isin(np.arange(len(a['pts'])),rows),'checked_surface_paint',a['material'])
    out=folder/(name+'.npz');save(out,**a);setup['shots'][name]['observations']=str(out)
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz');write(folder/'setup.json',setup)
    return folder


def soft_paint_window(name,parent,*,stage='broad_paint',paint_weight=None):
    """Refit the original broad ice paint without introducing an isolated single-frame correction."""
    target=WORK/stage/('texture_'+name)
    origins=read(WORK/'text_four/sources.json');orig=Path(origins[name])
    rows=[arrays(orig/(name+'.npz'))];flags=[arrays(orig/'quality'/(name+'.npz'))['keep']]
    families=[rows[0].get('family',np.zeros(len(rows[0]['index']),dtype='i1'))]
    survey=WORK/'motion_evidence'/{'jin_s14':'s14_early','jin_s15':'guard_s15_extended',
                                   'nayeon_s06':'nayeon_s06_late'}[name]
    for src,code in [(survey,0),(survey/'ice',1)]:
        if not (src/(name+'.npz')).is_file():continue
        q=arrays(src/(name+'.npz'));assert q['source_sha256']==rows[0]['source_sha256']
        np.testing.assert_array_equal(q['pts'],rows[0]['pts'])
        rows.append(q);flags.append(arrays(src/'quality'/(name+'.npz'))['keep'])
        families.append(np.full(len(q['index']),code,dtype='i1'))
    out={k:np.concatenate([q[k] for q in rows]) for k in ('index','lag','left_uv','uv','world','held','valid')}
    save(target/(name+'.npz'),**out,pts=rows[0]['pts'],source_sha256=rows[0]['source_sha256'],
         family=np.concatenate(families),spatial_bin=(out['left_uv'][:,0]//320).astype('i1'))
    save(target/'quality'/(name+'.npz'),pts=rows[0]['pts'],keep=np.concatenate(flags))
    origins[name]=str(target);write(target/'sources.json',origins)
    b=camera_state(parent);s=read(PREPARED/'setup.json')['shots'][name]
    t=b[name+'_pts'][0]/12800.+s['first']+s['dt']*np.arange(len(b[name+'_controls']))
    def rise(t):
        x=t.clip(0.,1.)
        return x**3*(10.-15.*x+6.*x*x)
    if name=='jin_s14':
        lo,hi=120.99,122.36
        basis=(rise((t-121.02)/.49)*rise((122.34-t)/.51))[:,None]
    elif name=='jin_s15':
        lo,hi=126.79,127.60
        basis=rise((t-126.81)/.67)[:,None]
    elif name=='nayeon_s06':
        lo,hi=123.65,126.22
        support=rise((t-lo)/.47)*rise((hi-t)/.49)
        centres=np.array([124.08,124.40,124.72,125.04,125.36,125.68])
        basis=support[:,None]*np.maximum(0.,1.-((t[:,None]-centres)/.67)**2)**3
    else:raise ValueError(name)
    weights=WORK/('source_'+name)/'paint.json'
    if paint_weight is not None:
        updated=read(weights);updated['checked_surface_paint']=paint_weight
        weights=target/'paint.json';write(weights,updated)
    guard_window(name,parent,stage,lo,hi,weights,38,target,basis)


def boundary_tracks():
    target=WORK/'text_boundaries';src=WORK/'motion_evidence/guard_boundaries';origins=read(WORK/'text_four/sources.json')
    for name in ['jin_s08','jin_s15','jin_s22']:
        root=Path(origins[name]);original=arrays(root/(name+'.npz'));parts=[original]
        roles=[original.get('family',np.zeros(len(original['index']),dtype='i1'))]
        flags=[arrays(root/'quality'/(name+'.npz'))['keep']]
        for code,folder in [(0,src),(1,src/'ice')]:
            if not (folder/(name+'.npz')).exists():continue
            z=arrays(folder/(name+'.npz'))
            assert z['source_sha256'].item()==original['source_sha256'].item()
            np.testing.assert_array_equal(z['pts'],original['pts'])
            parts.append(z);roles.append(np.full(len(z['index']),code,dtype='i1'));flags.append(arrays(folder/'quality'/(name+'.npz'))['keep'])
        a={k:np.concatenate([p[k] for p in parts]) for k in ('index','lag','left_uv','uv','world','held','valid')}
        save(target/(name+'.npz'),**a,family=np.concatenate(roles),pts=original['pts'],
             source_sha256=original['source_sha256'],spatial_bin=(a['left_uv'][:,0]//320).astype('i1'))
        save(target/'quality'/(name+'.npz'),pts=original['pts'],keep=np.concatenate(flags))
        origins[name]=str(target)
    write(target/'sources.json',origins)


def include_reviewed_tips():
    folder=WORK/'tip_inputs';folder.mkdir(exist_ok=True)
    source=WORK/'verified_guards';setup=read(source/'setup.json')
    shutil.copyfile(source/'baseline.npz',folder/'baseline.npz')
    name='jin_s14';shot=setup['shots'][name];q=arrays(Path(shot['observations']))
    measurements=arrays(WORK/'appearance_audit/repair/jin_s14_tips.npz');used=measurements['role']=='repair_support'
    rows=measurements['row'][used]
    for k in ('family','pts','uv','path'):np.testing.assert_array_equal(q[k][rows],measurements[k][used])
    assert np.all(q['fold'][rows]==6) and np.all(q['material'][rows]=='outline_xmax')
    save(folder/'developed_tips.npz',row=rows,pts=q['pts'][rows],uv=q['uv'][rows],path=q['path'][rows],prior_fold=q['fold'][rows])
    q['fold'][rows]=0;out=folder/(name+'.npz');save(out,**q);shot['observations']=str(out)
    write(folder/'setup.json',setup)


def include_plate_corners():
    folder=WORK/'plate_inputs';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    name='jin_s22';s=setup['shots'][name];s['_pts']=camera_state(BASE)[name+'_pts']
    q=arrays(prepared_packet_path(name));p=WORK/'motion_evidence/s22_right_print/chroma_protection.npz'
    corners=arrays(p);fold=np.where(corners['pts']==round(155.06*12800),6,0).astype('i1')
    entry=append(q,s,p,fold,45);s.pop('_pts');out=folder/(name+'.npz');save(out,**q)
    s['observations']=str(out);write(folder/'setup.json',setup);write(folder/'source_audit.json',entry)


def bind_direct_source(a,shot,source,used,number):
    """Admit distinct source pixels and keep derived texture registrations in their own packet."""
    full=arrays(source); fit=arrays(used); idx=fit['source_row']; real=np.ones(len(idx),bool)
    if 'input_packet' in fit:real&=fit['input_packet']==0
    if 'material' in fit:real&=fit['material']!='whole_print_transfer'
    idx=idx[real]
    for k in ('idx','pts','path','uv'):np.testing.assert_array_equal(fit[k][real],full[k][idx])
    role=np.where(full['fold']<9,6,9).astype('i1')
    if np.any(role[idx]==9):raise ValueError(('Failed source check for declared fit rows',source))
    role[idx]=0
    # A direct reading transferred unchanged from an existing packet has one red pixel.
    common=np.flatnonzero(np.isin(a['pts'],full['pts'])&np.isin(a['path'],full['path']))
    lookup={(a['pts'][i],a['path'][i],*a['uv'][i]):i for i in common}
    reused=0;promoted=[];excluded=[]
    for j,(pt,path,(x,y)) in enumerate(zip(full['pts'],full['path'],full['uv'])):
        i=lookup.get((pt,path,x,y))
        if i is None:continue
        if a['fold'][i]>=9:
            if role[j]==0:raise ValueError(('Re-used a previously disqualified source measurement',shot['shot'],j,i))
            excluded.append(j)
        elif role[j]==0 and a['fold'][i]>=5:
            promoted.append(i);a['fold'][i]=0
        role[j]=9;reused+=1
    report=append(a,shot,source,role,number) if (role<9).any() else dict(packet=str(source),fit=0,check=0)
    report.update(fitted_direct_reads=str(used),matched_prior_pixels=reused,prior_rows_promoted=promoted,
                  already_disqualified=excluded,derived_factors=int((~real).sum()))
    return report


def existing_source_rows(name,factors,reference,which,label):
    used=arrays(factors);src=arrays(reference);select=used['input_packet']==which
    s=read(PREPARED/'setup.json')['shots'][name];a=arrays(prepared_packet_path(name))
    code,= [int(k) for k,b in s['source_packets'].items() if Path(b['path']).resolve()==reference.resolve()]
    lookup={row:i for i,row in zip(np.flatnonzero(a['source_packet']==code),a['source_row'][a['source_packet']==code])}
    rr=used['source_row'][select];rows=np.array([lookup[j] for j in rr],dtype=int)
    for k in ('uv','pts','path'):
        np.testing.assert_array_equal(src[k][rr],used[k][select]);np.testing.assert_array_equal(a[k][rows],used[k][select])
    file=WORK/'nayeon_late'/name/label
    save(file,row=rows,uv=a['uv'][rows],path=a['path'][rows],pts=a['pts'][rows],
         source_sha256=src['source_sha256'],native_factor_file=str(factors))
    return file


def final_inputs(extra_rows=None, *, label='final_inputs', new_reads=None, retire=None, check_sources=None, replaced_nayeon_trajectory=False):
    """Bind only the exposures actually used and preserve separate raw/source-factor roles."""
    if PREPARED.resolve()!=(WORK/'verified').resolve():raise ValueError(PREPARED)
    folder=WORK/label;folder.mkdir(exist_ok=True);setup=read(PREPARED/'setup.json')
    audit=read(PREPARED/'source_audit.json');shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    for name,s in setup['shots'].items():
        parent=prepared_packet_path(name);s['observations']=str(parent)
        if name=='jin_s22':
            src=WORK/'plate_inputs';p=src/(name+'.npz')
            a=arrays(parent);z=arrays(p)
            for k in a:np.testing.assert_array_equal(z[k][:len(a['pts'])],a[k])
            s['source_packets']=read(src/'setup.json')['shots'][name]['source_packets']
            s['observations']=str(p);audit[name]['new_plate_corners']=read(src/'source_audit.json')
        elif name=='jin_s14':
            update=arrays(WORK/'tip_inputs/developed_tips.npz');rows=update['row'];a=arrays(parent)
            for k in ('path','pts','uv'):np.testing.assert_array_equal(a[k][rows],update[k])
            np.testing.assert_array_equal(a['fold'][rows],update['prior_fold'])
            a['fold'][rows]=0;p=folder/(name+'.npz');save(p,**a);s['observations']=str(p)
            audit[name]['measured_ice_ends']=dict(row_map=str(WORK/'tip_inputs/developed_tips.npz'),
                source_evidence=str(WORK/'appearance_audit/repair/jin_s14_tips.npz'),promoted=len(rows))
        updates=(extra_rows or {}).get(name,[])
        bindings=(new_reads or {}).get(name,[])
        retirements=(retire or {}).get(name,[])
        checks=(check_sources or {}).get(name,[])
        if updates or bindings or retirements or checks:
            a=arrays(Path(s['observations']));records=[]
            if name=='nayeon_s03' and replaced_nayeon_trajectory:
                prior=arrays(WORK/'nayeon_guard/native/partial/admitted_role_updates.npz')
                rr=prior['rows'];a['fold'][rr]=prior['prior_fold']
                records.append(dict(replaced_trajectory_only_roles=len(rr),source=audit[name].pop('developed_source_roles')))
            for p in updates:
                r=arrays(p);use=r['role']=='repair_support' if 'role' in r else slice(None)
                rows=(r['row'] if 'row' in r else r['rows'])[use]
                for k in ('pts','path','uv'):np.testing.assert_array_equal(a[k][rows],r[k][use])
                if np.any(a['fold'][rows]>=9):raise ValueError(('Retired row used as a guard',name))
                changed=rows[a['fold'][rows]>=5];a['fold'][changed]=0
                records.append(dict(source=str(p),rows=len(rows),promoted=len(changed)))
            for p in retirements:
                r=arrays(p);rows=r['row']
                for k in ('pts','path','uv'):np.testing.assert_array_equal(a[k][rows],r[k])
                if a['station_fixed'][rows].any():raise ValueError(('Fixed source mark in retirement',name))
                newly=int((a['fold'][rows]<9).sum())
                a['fold'][rows]=9
                for k in ('admission_reason','restored_admission_reason'):
                    if k in a:a[k]=np.where(np.isin(np.arange(len(a['fold'])),rows),'confirmed_other_material_in_source',a[k])
                flags=audit[name].get('confirmed_material_errors')
                if flags and flags['audit']==str(p):
                    previous=Path(str(r['source_packet'].item()));assert sha(previous)==r['source_packet_sha256'].item()
                    old_roles=arrays(previous)['fold'][rows]
                    flags.update(rows=len(rows),audit_sha256=sha(p),prior_roles={str(int(x)):int((old_roles==x).sum()) for x in np.unique(old_roles)})
                single_flank='final_role' in r and np.all(r['final_role']=='single_side_evidence')
                disposition='single_flank_without_paint_centre' if single_flank else 'confirmed_wrong_material'
                records.append(dict(source=str(p),source_sha256=sha(p),additional=newly,**{disposition:len(rows)}))
            if bindings or checks:
                s['_pts']=camera_state(BASE)[name+'_pts']
                for j,(raw,used) in enumerate(bindings):
                    records.append(bind_direct_source(a,s,raw,used,50+j))
                for j,p in enumerate(checks):
                    original=arrays(p);assert np.all(original['role']=='same_exposure_check')
                    records.append(append(a,s,p,np.full(len(original['pts']),6,dtype='i1'),60+j))
                s.pop('_pts')
            p=folder/(name+'.npz');save(p,**a);s['observations']=str(p)
            audit[name]['additional_checked_targets']=records
    setup['parent']=str(PREPARED);write(folder/'setup.json',setup);write(folder/'source_audit.json',audit)
