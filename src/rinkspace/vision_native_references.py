"""Read current selected native source correspondences as navigation references.

No historical world scene, removed print detail, source pixel approval or new
geometry is imported. Registered source identities remain distinct from human
acceptance; actual context/appearance matching is still required after a cut.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from .vision_native import file_ref

REGISTERED_SOURCE_IDENTITY='current_native_registered_source_correspondence'


def _load(reference):
    path=Path(reference['path']).resolve();raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('Current source reference changed: '+str(path))
    return json.loads(raw.decode('utf-8-sig'))


def _selected_external_rows(assets,asset_path):
    """Authenticate existing current-path source banks; never load world vertices."""
    store=assets.get('observation_store')
    if not store:return []
    refs={}
    for key in ('arrays','source_correspondence'):
        item=store.get(key)
        if not item:return []
        path=Path(item['path']);path=path if path.is_absolute() else asset_path.parent/path
        refs[key]=dict(path=str(path.resolve()),sha256=item['sha256'])
        if file_ref(path)['sha256']!=item['sha256']:raise ValueError('Selected native external source store changed')
    correspondence=_load(refs['source_correspondence'])
    with np.load(refs['arrays']['path'],allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    sources=assets.get('sources',{})
    source=sources.get(store.get('source_id'),{})
    if not source:
        wanted=store.get('time_base',store.get('native_time_base','1/12800'))
        official={}
        for value in sources.values():
            if value.get('time_base')!=wanted:continue
            raw=value.get('original_source_record',value.get('original_record',{}))
            sha=value.get('video_sha256',value.get('sha256'))
            video=value.get('video_path',raw.get('source'))
            if sha and video:official[(sha,video)]=dict(sha256=sha,video_path=video,time_base=wanted)
        if len(official)==1:source=next(iter(official.values()))
    sha=source.get('sha256') or source.get('video_sha256');clock=store.get('time_base',store.get('native_time_base',source.get('time_base')))
    if not sha or not clock:return []
    paths={p['id']:p for t in assets['templates'] for p in t['paths']}
    grouped={}
    if 'curve_index' in arrays:
        registry=store['curve_registry'];instance=store['instance_id']
        records=[]
        for i,(ci,pts,uv) in enumerate(zip(arrays['curve_index'],arrays['pts'],arrays['uv'])):
            if ci<0 or ci>=len(registry) or arrays['exclusion_bits'][i]!=0:continue
            name=registry[int(ci)].get('template_path_id')
            if name in paths:records.append((i,instance,name,int(pts),uv))
    elif 'source_target_labels' in arrays:
        by_row={int(r['original_inventory_row']):r for r in correspondence.get('rows',[])};records=[]
        for i,original in enumerate(arrays['original_inventory_row']):
            row=by_row.get(int(original),{});names=row.get('candidate_feature_ids',[])
            if len(names)!=1 or names[0] not in paths:continue
            records.append((i,row['instance_id'],names[0],int(arrays['native_pts'][i]),arrays['source_uv'][i]))
    elif 'current_feature_id' in arrays and isinstance(correspondence,dict) and correspondence.get('advertisement'):
        # This original restoration delta explicitly names the physical ad.
        instance='print_instance/Ad'+str(correspondence['advertisement']);records=[]
        for i,(name,pts,uv) in enumerate(zip(arrays['current_feature_id'],arrays['pts'],arrays['uv'])):
            if str(name) in paths and arrays['historical_exclusion_bits'][i]==0:records.append((i,instance,str(name),int(pts),uv))
    else:return []
    for i,instance,name,pts,uv in records:
        grouped.setdefault((instance,name,pts),[]).append((i,uv))
    # Reference selection uses source support coverage; every raw row remains
    # in the authenticated original store. It is not a programme frame limit.
    best={}
    for (instance,name,pts),rows in grouped.items():
        if len(rows)<2:continue
        key=(instance,name)
        if key not in best or len(rows)>len(best[key][1]):best[key]=(pts,rows)
    output=[]
    for (instance,name),(pts,rows) in best.items():
        indices=[i for i,_ in rows];points=np.asarray([uv for _,uv in rows],float)
        stable=arrays.get('stable_id',arrays.get('source_point_ids'))
        labels=[str(stable[i].decode() if isinstance(stable[i],bytes) else stable[i]) for i in indices]
        segments=[];lookup={}
        for label in labels:
            part=label.rsplit('/vertex/',1)[0];segments.append(lookup.setdefault(part,len(lookup)))
        ident='external:'+refs['arrays']['sha256']+':'+str(pts)+':'+name
        output.append(dict(id=ident,feature_id=name,instance_id=instance,source_sha256=sha,pts=pts,time_base=clock,
            source_native_uv=points.tolist(),source_segment_ids=segments,closed=False,
            external_source_store=refs,source_video_path=source.get('video_path',source.get('path')),
            original_record=dict(source_array_indices=indices,source_point_ids=labels,
                original_dependence_groups_per_point=[label.rsplit('/vertex/',1)[0] for label in labels],
                source_bank=refs,source_membership_rewritten=False,source_qualification_promoted=False),
            human_pixel_acceptance=False,active_for_fit=False))
    return output


@lru_cache(maxsize=32)
def _video_ref(path,mtime,size):
    return file_ref(path)


def _external_frame(row,cache):
    """Decode the registered video at the unchanged native integer PTS."""
    from fractions import Fraction
    import av
    from PIL import Image
    video=Path(row['source_video_path'])
    stat=video.stat()
    if _video_ref(str(video),stat.st_mtime_ns,stat.st_size)['sha256']!=row['source_sha256']:raise ValueError('Registered original video changed')
    target=cache/'source_frames'/(row['source_sha256'][:12]+'_'+str(row['pts'])+'.png')
    metadata=target.with_suffix('.json')
    if target.exists() and metadata.exists():
        record=json.loads(metadata.read_text(encoding='utf8'))
        if (record['source_sha256']!=row['source_sha256'] or record['source_pts']!=row['pts']
            or record['time_base']!=row['time_base'] or file_ref(target)['sha256']!=record['source_frame_sha256']):
            raise ValueError('Cached exact native source frame changed')
    else:
        found=False
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            if Fraction(stream.time_base)!=Fraction(row['time_base']):raise ValueError('External source clock differs')
            container.seek(row['pts'],stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts<row['pts']:continue
                if frame.pts!=row['pts']:raise ValueError('Exact external source PTS missing')
                target.parent.mkdir(parents=True,exist_ok=True);rgb=frame.to_ndarray(format='rgb24')
                Image.fromarray(rgb).save(target)
                metadata.write_text(json.dumps(dict(source_sha256=row['source_sha256'],source_pts=row['pts'],
                    time_base=row['time_base'],source_frame_sha256=file_ref(target)['sha256'],
                    decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())),encoding='utf8')
                found=True
                break
        if not found:raise ValueError('Exact external source PTS missing at video EOF')
    return dict(row,source_frame_path=str(target),source_frame_sha256=file_ref(target)['sha256'])


@lru_cache(maxsize=256)
def _verified(encoded, stamps):
    authority=json.loads(encoded)
    scene=_load(authority['scene']);assets=_load(authority['reference'])
    target=Path(authority['reference']['path']).resolve()
    refs=scene.get('advertisement_assets',[])
    selected=[r for r in refs if (Path(authority['scene']['path']).parent/r['path']).resolve()==target
        and r['sha256']==authority['reference']['sha256']]
    instance=authority['instance_id']
    if len(selected)!=1 or instance not in selected[0].get('instance_ids',[selected[0]['instance_id']]):return False
    templates={t['id']:t for t in assets['templates']}
    instances={i['id']:i for i in assets['instances']}
    if instance not in instances:return False
    template=templates[instances[instance]['template_id']]
    names={p['id'] for p in template['paths']}
    if not authority['template_path_ids'] or not set(authority['template_path_ids'])<=names:return False
    source=authority['reference_source_binding'];observations={o['id']:o for o in assets.get('observations',[])}
    observations.update({o['id']:o for o in _selected_external_rows(assets,target)})
    for oid in authority['source_observation_ids']:
        row=observations.get(oid)
        if row is None:return False
        original=assets['sources'].get(row.get('source_id'),{})
        sha=row.get('source_sha256') or original.get('sha256') or original.get('video_sha256')
        if (sha!=source['source_sha256'] or row.get('pts')!=source['source_pts'] or row.get('time_base')!=source['time_base']):return False
        if row.get('instance_id',instance)!=instance:return False
    return bool(authority['source_observation_ids'] and authority.get('human_pixel_acceptance_inferred') is False)


def registered_source_identity_verified(authority, binding=None, physical_id=None):
    if not isinstance(authority,dict) or authority.get('kind')!=REGISTERED_SOURCE_IDENTITY:return False
    try:
        if physical_id and authority['instance_id'].rsplit('/',1)[-1]!=physical_id:return False
        if binding and any(authority['reference_source_binding'].get(k)!=binding.get(k)
            for k in ('source_sha256','source_pts','time_base')):return False
        stamps=tuple((r['path'],Path(r['path']).stat().st_mtime_ns,Path(r['path']).stat().st_size)
            for r in (authority['scene'],authority['reference'],*authority.get('external_source_store_refs',[])))
        return _verified(json.dumps(authority,sort_keys=True),stamps)
    except (KeyError,TypeError,FileNotFoundError):return False


def _recover_registered_frames(assets, records, cache, source_sha256):
    """Restore missing frame caches only from their registered original clock."""
    ready=[];remaining=[];recovered=[]
    for record in records:
        if record['status']=='ready':continue
        if set(record['missing'])-{'source_frame_path_missing','source_frame_unavailable'}:
            remaining.append(record);continue
        row=assets['observations'][record['row_index']]
        source=assets.get('sources',{}).get(row.get('source_id'),{})
        sha=row.get('source_sha256') or source.get('video_sha256') or source.get('sha256')
        video=source.get('video_path') or source.get('source_video_path')
        raw=source.get('original_source_record',source.get('original_record',{}))
        video=video or raw.get('source')
        annotation=source.get('annotation') or row.get('source_annotation')
        annotation_ref=None
        if not video and annotation and Path(annotation).is_file():
            packet=json.loads(Path(annotation).read_text(encoding='utf-8-sig'));origin=packet.get('source',{})
            if (origin.get('source_video_sha256')==sha and origin.get('time_base')==row.get('time_base')
                    and packet.get('source_pts')==row.get('pts')):
                video=origin.get('source');annotation_ref=file_ref(annotation)
        if sha!=source_sha256 or not video or type(row.get('pts')) is not int:
            remaining.append(record);continue
        value=_external_frame(dict(row,source_sha256=sha,source_video_path=video),cache)
        ready.append((dict(record,status='ready_from_original_video'),value))
        recovered.append(dict(original_unavailable_record=record,source_frame=file_ref(value['source_frame_path']),
            source_sha256=sha,source_pts=row['pts'],time_base=row['time_base'],annotation=annotation_ref,
            original_UV_changed=False,acceptance_promoted=False))
    return ready,remaining,recovered


def current_native_source_banks(geometry, *, cache_directory, identity_catalog, event_id, source_sha256, programme_scope):
    from PIL import Image
    from .session.ad_review import _source_rows
    cache=Path(cache_directory);cache.mkdir(parents=True,exist_ok=True)
    frames={};missing=[];recovered_sources=[]
    for reference,assets in geometry.scene_assets:
        asset_path=(geometry.scene_path.parent/reference['path']).resolve()
        asset_ref=file_ref(asset_path)
        ready,records=_source_rows(assets,asset_path)
        ready.extend((dict(status='ready'),_external_frame(row,cache)) for row in _selected_external_rows(assets,asset_path)
            if row['source_sha256']==source_sha256)
        restored,unavailable,recovered=_recover_registered_frames(assets,records,cache,source_sha256)
        ready.extend(restored);recovered_sources.extend(dict(asset=asset_ref,**r) for r in recovered)
        missing.extend(dict(asset=asset_ref,record=r) for r in unavailable)
        templates={t['id']:t for t in assets['templates']}
        for record,row in ready:
            if row['source_sha256']!=source_sha256 or type(row.get('pts')) is not int:continue
            instance=row.get('instance_id')
            if instance is None:
                prefix=str(row.get('feature_id','')).split('.',1)[0]
                candidate='print_instance/'+prefix
                if candidate in reference.get('instance_ids',[reference['instance_id']]):instance=candidate
            if instance is None:continue
            inst=next((i for i in assets['instances'] if i['id']==instance),None)
            if inst is None:continue
            paths=templates[inst['template_id']]['paths']
            linked=[p for p in paths if row['id'] in p.get('source_observation_ids',[])
                or row['id']==p.get('source_observation_id') or row.get('feature_id')==p['id']
                or row.get('feature_id') in p.get('logical_labels',[])]
            if len(linked)!=1:continue
            path=linked[0];native_id=instance+'::'+path['id']
            if geometry.lookup(native_id) is None:continue
            uv=np.asarray(row['source_native_uv'],float).reshape(-1,2)
            raw=row.get('original_record',{})
            measured=raw.get('source_native_readout_uv')
            if measured is not None:uv=np.asarray(measured,float).reshape(-1,2)
            if len(uv)<2 or not np.isfinite(uv).all():continue
            key=(row['source_sha256'],row['time_base'],row['pts'],row['source_frame_path'])
            frame=frames.setdefault(key,dict(path=row['source_frame_path'],sha=row.get('source_frame_sha256'),ads={}))
            ad=frame['ads'].setdefault(instance,dict(asset=asset_ref,items=[]))
            ad['items'].append(dict(native_id=native_id,path_id=path['id'],observation_id=row['id'],uv=uv,
                closed=bool(row.get('closed',False)),segments=row.get('source_segment_ids'),original_record=row,
                original_UV_used_for_navigation_only=True))
    packets=[];references={}
    for (sha,clock,pts,image_path),frame in frames.items():
        image_ref=file_ref(image_path)
        if frame['sha'] and frame['sha']!=image_ref['sha256']:raise ValueError('Native source reference image changed')
        rgb=np.asarray(Image.open(image_path).convert('RGB'));h,w=rgb.shape[:2]
        from .source_binding import registered_reference_binding
        binding = registered_reference_binding(dict(source_sha256=sha, time_base=clock, source_pts=pts,
            native_hw=[h,w], event_id=event_id), programme_scope)
        advertisements=[];guides=[]
        for instance,ad in frame['ads'].items():
            items=ad['items'];points=np.concatenate([i['uv'] for i in items])
            low=np.maximum(np.floor(points.min(0)-2),0).astype(int)
            high=np.minimum(np.ceil(points.max(0)+2),[w,h]).astype(int)
            if min(high-low)<=8:continue
            authority=dict(kind=REGISTERED_SOURCE_IDENTITY,reference=ad['asset'],scene=geometry.refs[0],instance_id=instance,
                reference_source_binding=binding,source_observation_ids=sorted({i['observation_id'] for i in items}),
                template_path_ids=sorted({i['path_id'] for i in items}),human_pixel_acceptance_inferred=False,
                external_source_store_refs=list({r['sha256']:r for i in items for r in i['original_record'].get('external_source_store',{}).values()}.values()),
                scope='existing_selected_native_source_correspondence_navigation_only')
            if not registered_source_identity_verified(authority,binding,instance.rsplit('/',1)[-1]):
                raise ValueError('Native reference failed source authentication')
            number=int(instance.rsplit('/Ad',1)[-1])
            advertisements.append(dict(advertisement_number=number,physical_identity_human_confirmed=False,
                registered_source_identity_authority=authority,physical_identity_authority=authority,
                components=[dict(component_id='current_native_retained_paths',observed_kind='printed_pattern',
                    actual_source_navigation_roi=np.r_[low,high].tolist(),source_component_scope='current_native_retained_paths_only')]))
            grouped={}
            for item in items:grouped.setdefault(item['native_id'],[]).append(item)
            for native_id,group in grouped.items():
                points=np.concatenate([i['uv'] for i in group]);segments=[];offset=0
                for item in group:
                    values=np.asarray(item['segments'] if item.get('segments') is not None else np.zeros(len(item['uv']),int))
                    segments.extend((values+offset).tolist());offset+=int(values.max())+1
                guides.append(dict(id=native_id,kind='ad_boundary',points_uv=points,segment_ids=segments,
                    source_binding=binding,origin='source_annotation_search_hint',identity_confirmed=True,
                    identity_authority=authority,search_radius_px=6.,closed=len(group)==1 and group[0]['closed'],
                    source_group_id='native_registered:'+native_id,
                    original_dependence_groups_per_point=[value for i in group for value in
                        i['original_record'].get('original_record',{}).get('original_dependence_groups_per_point',[None]*len(i['uv']))],
                    original_source_observation_ids=[i['observation_id'] for i in group],
                    original_source_records=[i['original_record'] for i in group],
                    registered_source_identity_is_human_pixel_acceptance=False))
        if not advertisements:continue
        packet=dict(event_id=event_id,identity_catalog=identity_catalog,source=dict(binding,
            source_frame_path=image_path,source_frame_sha256=image_ref['sha256'],
            decoded_native_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),advertisements=advertisements,
            image_left_to_right_physical_order=[a['advertisement_number'] for a in sorted(advertisements,
                key=lambda a:a['components'][0]['actual_source_navigation_roi'][0])])
        name=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest()[:24]+'.json'
        target=cache/name;target.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
        ref=file_ref(target);bound,report=geometry.bind(guides)
        references[ref['sha256']]=dict(guides=bound,reference=ref,source_binding=binding)
        packets.append(target)
    return packets,references,dict(reference_count=len(packets),missing_source_records=missing,
        recovered_registered_source_frames=recovered_sources,
        current_native_scene=geometry.refs[0],original_pixels_modified=False,old_geometry_imported=False)
