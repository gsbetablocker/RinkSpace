"""Independent NumPy source UV diagnostics at a genuine physical nuisance state.

This measures all active rows, including failures, without profiling stations or
claiming that a finite Query closes. Curves and fixed material points are split.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from .io_utils import read_json, write_json, file_sha256
from .joint_model import as_event_view, graph_from_joint
from .camera_graph import CameraGraph
from .joint_replay import render_visibility,material_visibility_records
from rinkfit_start.native_scene import NativeScene


def _array(root, descriptor):
    p = Path(descriptor['file'])
    if not p.is_absolute(): p = Path(root) / p
    if file_sha256(p) != descriptor['sha256']:
        raise ValueError('Changed numerical/source dependency: ' + str(p))
    count = int(np.prod(descriptor['shape']))
    a = np.fromfile(p, descriptor['dtype'], count=count,
                    offset=descriptor.get('offset_bytes', 0))
    if a.size != count: raise ValueError('Incomplete array: ' + str(p))
    return a.reshape(descriptor['shape'])


def _stats(v, good):
    v = np.asarray(v, float); good = np.asarray(good, bool) & np.isfinite(v)
    a = v[good]
    return dict(denominator=len(v), finite=int(good.sum()), invalid=int((~good).sum()),
                rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def _source_geometry(model, packet, packet_root, arrays, active, aux, scene):
    """Same finite source domains and local material flanks as native layout."""
    station=aux['station'][active]
    primitive=np.isfinite(station)&(station>=0)&(station<=1)
    material=arrays['material_index'][active]; branch=aux['path'][active]
    world,finite,reason=scene.world_rows(material,branch,np.where(primitive,station,0.))
    finite &= primitive; reason[~primitive]='primitive_station_domain'
    width_block=np.full(len(active),-1,np.int64); width_sign=np.zeros(len(active),np.int8)
    local_index=np.full(int(packet['row_count']),-1,np.int64); local_index[active]=np.arange(len(active))
    for mi,pi in np.unique(np.c_[material,branch],axis=0):
        ix=np.flatnonzero((material==mi)&(branch==pi)); path=scene.path(int(mi),int(pi))
        allowed=np.zeros(len(ix),bool)
        for lo,hi in path.get('allowed_station_intervals',[[0.,1.]]):
            allowed |= (station[ix]>=lo)&(station[ix]<=hi)
        finite[ix] &= allowed; reason[ix[~allowed]]='restricted_material_station_domain'
        if path.get('physical_point') and path.get('kind')==0:
            on_ice=scene.containment(world[ix,:2])<=1e-10
            finite[ix]&=on_ice;reason[ix[~on_ice]]='fixed_ice_point_outside_finite_rink'
        if np.any(arrays['kind'][active[ix]]==1) and not path.get('physical_point'):
            raise ValueError('Fixed source point mapped to sliding curve')
    domains=packet.get('source_station_domains'); source_domain_rows=0
    if domains:
        if domains['schema']!='rinkfit.source_row_station_domains.v1' or not domains['source_refs']:
            raise ValueError('Source finite domains lack explicit identity')
        for ref in domains['source_refs']:
            f=Path(ref['file']); f=f if f.is_absolute() else Path(packet_root)/f
            if file_sha256(f)!=ref['sha256']:raise ValueError('Changed source finite-domain evidence')
        seen=set()
        for d in domains['rows']:
            raw=int(d['row_index']); target=int(d['path_index'])
            if raw in seen or raw<0 or raw>=len(local_index) or arrays['kind'][raw]==1 or d['source_row_id']!=packet['source_row_ids'][raw]:
                raise ValueError('Source finite-domain immutable identity mismatch')
            seen.add(raw); intervals=np.asarray(d['allowed_station_intervals'],float)
            if intervals.ndim!=2 or intervals.shape[1]!=2 or not len(intervals) or not np.isfinite(intervals).all() or np.any(intervals<0) or np.any(intervals>1) or np.any(intervals[:,0]>intervals[:,1]) or np.any(intervals[1:,0]<=intervals[:-1,1]):
                raise ValueError('Invalid finite source interval union')
            ix=int(local_index[raw])
            if ix<0:continue
            source_domain_rows+=1
            allowed=branch[ix]==target and bool(np.any((station[ix]>=intervals[:,0])&(station[ix]<=intervals[:,1])))
            if not allowed:finite[ix]=False;reason[ix]='restricted_source_station_domain'
    scopes={w['width_scope_id']:int(w['half_width_block']) for w in model.get('measurement_widths',[]) if w.get('scope_is_source_confirmed_section_not_uniform_whole_stroke') is True}
    bindings={b['pair_id']:b for b in model.get('stroke_pair_model_bindings',[])}
    pair_seen=set(); pairs=0
    for pair_index,pair in enumerate(packet.get('stroke_pairs',[])):
        rr=list(map(int,pair['rows']))
        if len(rr)!=2 or len(set(rr))!=2 or any(i in pair_seen or i<0 or i>=len(local_index) for i in rr):raise ValueError('Invalid source flank pair membership')
        pair_seen.update(rr)
        if any(int(arrays['stroke_pair_index'][i])!=pair_index or int(arrays['stroke_source_side'][i])!=side for side,i in enumerate(rr)):raise ValueError('Source flank typed order mismatch')
        selected=[int(local_index[i]) for i in rr]
        if all(i<0 for i in selected):continue
        if any(i<0 for i in selected):raise ValueError('One active flank may not substitute for a two-flank source section')
        binding=bindings[pair['id']]; block=int(binding['half_width_block']); signs=binding['source_world_signs']
        if scopes.get(pair['width_scope_id'])!=block or binding['width_scope_id']!=pair['width_scope_id'] or len(signs)!=2 or signs[0]*signs[1]!=-1 or abs(signs[0])!=1 or signs not in pair['allowed_world_side_assignments']:
            raise ValueError('Source flank physical width/side binding mismatch')
        half=np.asarray(model['blocks'][block]['values'],float)
        if half.shape!=(1,) or not np.isfinite(half).all() or half[0]<=0:raise ValueError('Invalid local source paint halfwidth')
        for raw,ix,sign in zip(rr,selected,signs):
            mi=int(material[ix]); pi=int(branch[ix]); path=scene.path(mi,pi)
            if arrays['kind'][raw]==1 or mi!=pair['material_index'] or mi!=binding['material_index']:raise ValueError('Source flank changed fixed/material identity')
            kind=path['kind']; q=scene.values[int(path['geometry_block'])]; shift=float(sign)*half[0]
            if kind==0:
                tangent=q[2:4]-q[:2]; length=np.linalg.norm(tangent)
                if not np.isfinite(length) or length<=0:raise ValueError('Degenerate source stroke carrier')
                world[ix,:2]+=shift*np.array([-tangent[1],tangent[0]])/length
            elif kind==1:
                angle=path.get('start_angle',0.)+station[ix]*(path.get('end_angle',2*np.pi)-path.get('start_angle',0.))
                if q[int(path.get('radius_index',2))]+shift<=0:finite[ix]=False;reason[ix]='nonpositive_source_flank_radius'
                world[ix,:2]+=shift*np.array([np.cos(angle),np.sin(angle)])
            else:raise ValueError('Source flanks require a finite ice stroke')
            width_block[ix]=block; width_sign[ix]=sign
        pairs+=1
    return world,finite,reason,width_block,width_sign,dict(active_source_flank_pairs=pairs,active_source_flank_rows=int((width_block>=0).sum()),active_row_finite_source_domains=source_domain_rows)


def evaluate_source_state(model_file, packet_file, state_manifest, output, *, fixed_only=False):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Use a new immutable diagnostic directory')
    model=read_json(model_file); packet=read_json(packet_file)
    n=int(packet['row_count']); packet_sha=file_sha256(packet_file)
    if fixed_only:
        if state_manifest is not None:raise ValueError('Fixed-entity diagnostic uses the original fixed source stations, no curve-state import')
        state=dict(raw_rows=n,input_packet_sha256=packet_sha,files={key+'.'+suffix:packet['arrays'][key] for key,suffix in [('station','f64'),('path','u32')]})
    else:state=read_json(state_manifest)
    if state.get('input_packet_sha256',state.get('source_packet_sha256'))!=packet_sha:raise ValueError('State belongs to a different source packet')
    if state['raw_rows']!=n:raise ValueError('Incomplete raw nuisance state')
    files=state['files']; aux={}
    for key,dtype,suffix in [('station','<f8','f64'),('path','<u4','u32')]:
        if fixed_only:
            aux[key]=_array(Path(packet_file).parent,packet['arrays'][key]);continue
        name=key+'.'+suffix; d=files[name]; path=Path(d.get('file',name))
        if not path.is_absolute():path=(Path(packet_file).parent if fixed_only else Path(state_manifest).parent)/path
        if file_sha256(path)!=d['sha256']:raise ValueError('Changed nuisance '+name)
        aux[key]=np.fromfile(path,dtype)
        if aux[key].shape!=(n,):raise ValueError('Incomplete nuisance '+name)
    if 'model.json' in files and file_sha256(model_file)!=files['model.json']['sha256']:raise ValueError('Diagnostic model differs from actual state model')
    names=['uv','native_pts','shot_index','kind','material_index','profile_normal','admission','fit_role','E']
    if packet.get('stroke_pairs'):names += ['stroke_pair_index','stroke_source_side']
    a={k:_array(Path(packet_file).parent,packet['arrays'][k]) for k in names}
    active=np.flatnonzero((a['admission']==1)&(a['fit_role']==1))
    input_active_rows=len(active)
    if fixed_only:active=active[a['kind'][active]==1]
    scene=NativeScene(as_event_view(model,'s01')); cg=CameraGraph(graph_from_joint(model))
    world,finite,reason,width_block,width_sign,source_geometry=_source_geometry(model,packet,Path(packet_file).parent,a,active,aux,scene)
    material=a['material_index'][active]; branch=aux['path'][active]
    prediction = np.full((len(active),2),np.nan); row = np.full(len(active),np.nan)
    valid = np.zeros(len(active),bool); rs_valid = valid.copy(); visible = valid.copy()
    root_equality=np.full(len(active),np.nan)
    shot = a['shot_index'][active]; pts = a['native_pts'][active]
    for si,sid in enumerate(packet['shot_ids']):
        ix = np.flatnonzero(shot==si)
        if not len(ix): continue
        camera = cg.native_camera(sid)
        local = cg.local_seconds(sid,pts[ix])
        safe = world[ix].copy(); safe[~finite[ix]] = 0.
        projected = camera.project_rs(safe,local)
        prediction[ix] = projected['uv']; row[ix]=projected['row']
        root_equality[ix]=projected['root_equality_px']
        rs_valid[ix] = projected['valid'] & finite[ix]
        records = material_visibility_records(scene,material[ix],branch[ix])
        visible[ix] = render_visibility(camera,scene,safe,records,local,projected['uv'])
        valid[ix] = rs_valid[ix]&visible[ix]
        print(sid,'active',len(ix),'valid',int(valid[ix].sum()),flush=True)
    error = prediction-a['uv'][active]; distance=np.linalg.norm(error,axis=1)
    normal = a['profile_normal'][active]; normal_length=np.linalg.norm(normal,axis=1)
    normal_error=np.abs(np.einsum('ij,ij->i',error,normal))
    fixed=a['kind'][active]==1; profile=(~fixed)&(normal_length>0)
    metric=np.where(profile,normal_error,distance)
    def summary(ix):
        return dict(all_source_uv_px=_stats(distance[ix],valid[ix]),
                    fixed_entity_uv_px=_stats(distance[ix&fixed],valid[ix&fixed]),
                    curve_normal_px=_stats(normal_error[ix&profile],valid[ix&profile]),
                    source_operator_metric_px=_stats(metric[ix],valid[ix]),
                    true_RS_invalid=int(np.count_nonzero(ix&finite&~rs_valid)),
                    true_RS_not_evaluable=int(np.count_nonzero(ix&~finite)),
                    finite_domain_invalid=int(np.count_nonzero(ix&~finite)),
                    occluded_or_backface=int(np.count_nonzero(ix&rs_valid&~visible)))
    by_shot={sid:summary(shot==si)for si,sid in enumerate(packet['shot_ids'])}
    by_entity={model['materials'][int(mi)]['label']:summary(material==mi)for mi in np.unique(material)}
    # Consecutive source frames only; missing frames explicitly split a segment.
    segments=[]
    for si,sid in enumerate(packet['shot_ids']):
        local_pts=np.unique(pts[shot==si]); cuts=np.r_[0,np.flatnonzero(np.diff(local_pts)!=256)+1,len(local_pts)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            ii=(shot==si)&(pts>=local_pts[lo])&(pts<=local_pts[hi-1])
            q=summary(ii);segments.append(dict(shot_id=sid,first_pts=int(local_pts[lo]),last_pts=int(local_pts[hi-1]),source_frames=int(hi-lo),**q))
    segments.sort(key=lambda s:(s['source_operator_metric_px']['invalid'],s['source_operator_metric_px']['rms'] or 0.),reverse=True)
    out.mkdir(parents=True)
    np.savez_compressed(out/'source_rows.npz',joint_row=active,native_pts=pts,shot_index=shot,
                        prediction_uv=prediction,source_uv=a['uv'][active],error_uv=error,
                        valid=valid,RS_valid=rs_valid,visible=visible,station=aux['station'][active],
                        path=branch,material_index=material,kind=a['kind'][active],row=row,
                        finite=finite,finite_reason=reason,world_xyz=world,root_equality_px=root_equality,
                        profile_normal=normal,normal_length=normal_length,width_block=width_block,width_sign=width_sign)
    report=dict(schema='rinkfit.independent_physical_source_state_diagnostics.v1',
                model=dict(file=str(Path(model_file).resolve()),sha256=file_sha256(model_file)),
                packet=dict(file=str(Path(packet_file).resolve()),sha256=packet_sha),
                nuisance_manifest=None if fixed_only else dict(file=str(Path(state_manifest).resolve()),sha256=file_sha256(state_manifest)),
                raw_rows=n,active_rows=len(active),input_active_rows=input_active_rows,all_active_rows_retained=not fixed_only,
                evaluation_scope='all_active_fixed_entities_only' if fixed_only else 'all_active_source_rows',
                unevaluated_active_curve_rows=input_active_rows-len(active),fixed_station_from_original_packet=bool(fixed_only),
                implementation_sha256=file_sha256(__file__),source_geometry=source_geometry,
                all_source_geometry_operators_consumed=not fixed_only,
                finite_domain_invalid_reasons={str(k):int(v) for k,v in zip(*np.unique(reason[~finite],return_counts=True))},
                nonzero_curve_normal_rows=int(profile.sum()),
                curve_normal_length_min=float(normal_length[profile].min()) if profile.any() else None,
                curve_normal_length_max=float(normal_length[profile].max()) if profile.any() else None,
                source_normal_coefficients_used_without_rescaling=True,
                original_correlated_W_or_engineering_objective_cost_claimed=False,
                maximum_valid_root_equality_px=float(np.max(np.abs(root_equality[valid]),initial=0.)),
                station_or_source_UV_profiled_or_changed=False,finite_Query_closure_claimed=False,
                engineering_E_is_not_measurement_covariance=True,independent_holdout=False,
                all=summary(np.ones(len(active),bool)),by_shot=by_shot,by_entity=by_entity,
                consecutive_source_support_segments=segments,
                rows_file=dict(file=str(out/'source_rows.npz'),sha256=file_sha256(out/'source_rows.npz')))
    write_json(out/'report.json',report)
    return report



def pin_coarse_checkpoint(checkpoint_manifest, output):
    """Freeze one complete accepted primal before reading a mutable journal.

    Only the changing model/station are copied. The other complete numerical
    arrays reference this run's immutable initial-state files, with exact SHA.
    A concurrent slot overwrite is rejected before creating any output.
    """
    import hashlib
    source=Path(checkpoint_manifest).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Pinned coarse checkpoint already exists')
    raw=source.read_bytes();manifest=json.loads(raw)
    if manifest.get('schema')!='rinkfit.joint_source_capture_checkpoint.v1':
        raise ValueError('Expected a genuine accepted source capture checkpoint')
    captured={}
    for name in ('model.json','station.f64','row.f64','alpha.f64','path.u32'):
        d=manifest['files'][name];p=Path(d['file'])
        if not p.is_absolute():p=source.parent/p
        if name in ('model.json','station.f64'):
            data=p.read_bytes()
            if hashlib.sha256(data).hexdigest()!=d['sha256'] or len(data)!=d['size_bytes']:
                raise ValueError('Checkpoint slot advanced during pin; read the current journal again')
            captured[name]=data
        elif file_sha256(p)!=d['sha256'] or p.stat().st_size!=d['size_bytes']:
            raise ValueError('Changed immutable source capture auxiliary '+name)
    n=int(manifest['raw_rows'])
    if len(captured['station.f64'])!=n*8:raise ValueError('Incomplete source capture stations')
    out.mkdir(parents=True)
    (out/'journal_manifest_at_pin.json').write_bytes(raw)
    for name,data in captured.items():
        target=out/name;target.write_bytes(data)
        manifest['files'][name]={**manifest['files'][name],'file':str(target)}
    manifest.update(pinned_journal_manifest=dict(file=str(out/'journal_manifest_at_pin.json'),
                    sha256=hashlib.sha256(raw).hexdigest()),
                    original_mutable_journal_path=str(source),immutable_pinned_primal=True)
    write_json(out/'manifest.json',manifest)
    return out/'manifest.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--packet',required=True)
    p.add_argument('--state-manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    evaluate_source_state(a.model,a.packet,a.state_manifest,a.out)
