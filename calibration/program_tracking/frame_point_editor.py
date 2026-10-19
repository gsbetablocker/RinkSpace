"""A loopback point editor for native, model-bound frame review packets.

Writes are explicit user actions, version checked, and stored with rollback
snapshots. This module never extracts observations or changes model geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit
import uuid

import cv2
import numpy as np

from rinkspace.path_policy import guard_project_write
from rinkspace.packed_assets import iter_project_assets, materialize_project_asset

from .extract_selected_frame_review import make_binding
from .framewise_model import evaluate_rows
from .render_opening_frame_batch import page_html, timestamp
from .render_single_frame_source_review import LAYERS, layer_for, native_overlay, source_segments, display_segments
from .validate_opening_frame_batch import assemble, read, require, sha

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT/'src/rinkspace/static'
POINT_ARRAYS = ('uv','point_ids','point_bindings','source_evidence','sigma_px',
                'transition_iqr_px','localization_sigma_px','retained_pre_compaction_vertex_indices')
INDEX_ARRAYS = ('protected_vertex_indices','corner_vertex_indices')


class EditConflict(ValueError):
    pass


def json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def ref(path, data=None):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=digest(data) if data is not None else sha(path))


def write_atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_bytes(data);os.replace(temp,path)
    finally:
        if temp.exists(): temp.unlink()


def file_signature(path):
    """Invalidate display caches without repeatedly parsing unchanged packets."""
    try:
        stat=materialize_project_asset(path, project_root=ROOT).stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def target_layer(descriptor):
    kind=descriptor['kind']
    if kind=='ice_curve': return 'ice_features'
    if kind=='board_edge': return 'board_top' if descriptor['role']=='top' else 'board_contact'
    return 'ad2_print' if kind=='ad_print' else 'ad2_border'


def target_label(tid, d):
    roles={'left_top':'左上角','left_bottom':'左下角','right_top':'右上角','right_bottom':'右下角',
           'left_vertical':'左边','right_vertical':'右边','top_edge':'上边','bottom_edge':'下边'}
    if d['kind']=='ice_curve':
        curve=d['curve_id']
        if d.get('effective_measurement_operator')=='paired_material_boundary_midline':
            label=curve.split(':')[0]+' · 空心描线中线'
        else:
            label=curve.replace(':outer',' · 外圈').replace(':middle',' · 中圈').replace(':inner',' · 内圈').replace(':center',' · 最内小圈')
        label={'lane3:edge2':'lane3 下长边','lane2:edge0':'lane2 上长边'}.get(d['curve_id'],label)
        return label
    if d['kind']=='board_edge':
        side={'upper_long_side':'画面上侧长边','upper_right':'右上弧段','upper_left':'左上弧段',
              'left_end':'左端','lower_left':'左下弧段','lower_long_side':'下侧长边','lower_right':'右下弧段','right_end':'右端'}
        return ('蓝色压顶顶缘' if d['role']=='top' else '黄色带本体下缘')+' · '+side.get(d.get('part'),'整圈')
    if d['kind']=='ad_print': return f"Ad{d['advertisement_id']} 内部轮廓 · "+d['contour_key'].replace('_inner',' 内孔')
    return f"Ad{d['advertisement_id']} "+roles.get(d['role'],d['role'])


def point_inventory(packet):
    result={}
    for f in packet['features']+packet.get('unbound_static_materials',[]):
        for si,s in enumerate(f['segments']):
            for vi,uv in enumerate(s['uv']):
                pid=s.get('point_ids',[f"{f['id']}/segment/{si}/vertex/{j}" for j in range(len(s['uv']))])[vi]
                require(pid not in result,'重复点编号')
                result[pid]=(f,si,vi,uv)
    return result


def apply_operations(original, operations, registry, transaction_id):
    """Pure edit: only explicitly addressed points change; source reads remain evidence."""
    packet=deepcopy(original)
    inventory=point_inventory(packet)
    require(isinstance(operations,list) and operations,'没有待保存的修改')
    deletions=set();adds=[]
    for op in operations:
        require(isinstance(op,dict),'修改格式不正确')
        if op.get('type')=='delete':
            pid=op.get('point_id');require(pid in inventory,'找不到要删除的点，请刷新当前帧')
            require(pid not in deletions,'同一个点重复删除');deletions.add(pid)
        elif op.get('type')=='add':
            tid=op.get('target_id');require(tid in registry['targets'],'请先选择已登记的模型特征')
            uv=op.get('uv');require(isinstance(uv,list) and len(uv)==2 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in uv),'点坐标不正确')
            require(np.isfinite(uv).all() and 0<=uv[0]<=1919 and 0<=uv[1]<=1079,'点必须位于原生画面内')
            adds.append((tid,[round(float(v),4) for v in uv]))
        else: raise ValueError('不支持的修改操作')
    for feature in packet['features']+packet.get('unbound_static_materials',[]):
        segments=[];changed=False
        for si,s in enumerate(feature['segments']):
            ids=s.get('point_ids',[f"{feature['id']}/segment/{si}/vertex/{vi}" for vi in range(len(s['uv']))])
            s['point_ids']=ids
            if not any(pid in deletions for pid in ids):segments.append(s);continue
            changed=True;runs=[];current=[]
            for vi,pid in enumerate(ids):
                if pid in deletions:
                    if current:runs.append(current);current=[]
                else:current.append(vi)
            if current:runs.append(current)
            for run in runs:
                segment=deepcopy(s);mapping={old:new for new,old in enumerate(run)}
                for key in POINT_ARRAYS:
                    if key in s:
                        require(len(s[key])==len(s['uv']),'逐点证据长度不一致：'+key)
                        segment[key]=[s[key][vi] for vi in run]
                for key in INDEX_ARRAYS:
                    if key in s:
                        if isinstance(s[key],dict):
                            segment[key]={role:mapping[v] for role,v in s[key].items() if v in mapping}
                        else:
                            require(isinstance(s[key],list),'点索引列表格式不正确')
                            segment[key]=[mapping[v] for v in s[key] if v in mapping]
                segment.update(vertex_count=len(run),closed=False)
                # Compaction diagnostics describe the original run, not this edited subset.
                segment.pop('draw_current_native_material_readout',None)
                for key in ('pre_compaction_vertex_count','sparse_representation_max_deviation_px',
                            'pre_compaction_reconstruction_max_px','measured_maximum_spacing_px'):
                    segment.pop(key,None)
                segments.append(segment)
        feature['segments']=segments
        if 'point_count' in feature:feature['point_count']=sum(len(s['uv']) for s in segments)
        if changed:
            feature.pop('geometric_interpretation',None)
            feature['manual_deletion_applied']=True
    # New observations have their own human provenance, not a copied profile or shared-line estimate.
    existing_corners=set()
    for f in packet['features']:
        for s in f['segments']:
            for vi in range(len(s['uv'])):
                tid=s['point_bindings'][vi]['physical_target'] if s.get('point_bindings') else f['target_id']
                if registry['targets'][tid]['kind']=='ad_corner':existing_corners.add(tid)
    for tid,uv in adds:
        d=registry['targets'][tid]
        if d['kind']=='ad_corner':
            require(tid not in existing_corners,'这个角点已有标点；请先删掉旧点再加新点')
            existing_corners.add(tid)
        pid='manual:'+uuid.uuid4().hex
        evidence=dict(origin='human_native_image_click',native_pts=packet['source']['native_pts'],
                      source_decoded_BGR_sha256=packet['source'].get('decoded_BGR_sha256'),
                      selected_target_id=tid,uv=uv,measurement_uncertainty_not_estimated=True)
        packet['features'].append(dict(id='manual.'+uuid.uuid4().hex,target_id=tid,
            label_zh=target_label(tid,d),review_layer=target_layer(d),manual_source_observation=True,
            physical_target_descriptor=deepcopy(d),physical_surface='ice' if d['kind']=='ice_curve' else 'board',
            world_curve=d.get('curve_id',tid),source_evidence_group_id=pid,
            segments=[dict(uv=[uv],point_ids=[pid],vertex_count=1,closed=False,
                measurement_origin='user_manual_native_image_click',pixels_measured_in_this_frame=True,
                source_evidence_measured_in_this_frame=True,source_evidence=[evidence],
                user_confirmed=True,hidden_span_filled=False)],point_count=1,
            projected_uv_used_as_observations=False))
    packet['point_count']=sum(len(s['uv']) for f in packet['features'] for s in f['segments'])
    extra_count=sum(len(s['uv']) for f in packet.get('unbound_static_materials',[]) for s in f['segments'])
    packet['source_point_count']=packet['point_count']+extra_count
    require(0<packet['source_point_count']<=500,'每帧需保留 1–500 个点（含身份待确认的实体边界）')
    groups=Counter()
    for f in packet['features']:
        groups[f.get('source_evidence_group_id',f['target_id'])]+=sum(len(s['uv']) for s in f['segments'])
    for group in packet.get('source_correlated_groups',[]):
        key=group.get('group_id',group.get('id',group.get('source_evidence_group_id')))
        if key in groups:group['representative_point_count']=groups[key]
    packet['user_accepted']=False
    packet['status']='manually_edited_awaiting_review'
    packet.pop('template_coordinates_exactly_preserved',None)
    packet['manual_editor']=dict(revision_id=transaction_id,updated_at=datetime.now(timezone.utc).isoformat(),
        authoritative_over_extraction_candidates=True,whole_frame_accepted=False)
    return packet


class PointEditorStore:
    def __init__(self,dataset,media,model,source_media=None):
        self.dataset=Path(dataset).resolve();self.media=Path(media).resolve();self.model=Path(model).resolve()
        # Revisions own their overlays/history but reuse the immutable native PNGs.
        self.source_media=Path(source_media).resolve() if source_media else self.media
        self.lock=threading.RLock();self.editor_url=None
        self.registry=read(self.model/'feature_registry.json');self.scene=read(self.model/'scene.json');self.templates=read(self.model/'ad2_material_template.json')
        try:
            from rinkspace.feedback_outbox import apply_editor_role_labels
            apply_editor_role_labels(self.registry,self.model,ROOT)
        except (OSError,ValueError):pass
        self.model_hashes={name:sha(self.model/name) for name in ('scene.json','feature_registry.json','ad2_material_template.json')}
        self.indices=[int(p.stem) for p in sorted(iter_project_assets(self.dataset/'frames', '[0-9]'*6+'.json', project_root=ROOT))]
        require(bool(self.indices),'没有可编辑的逐帧数据')
        self.current_indices=tuple(self.indices)
        self.current_acceptance_path=self.dataset/'user_acceptance.json'
        self._current_acceptance={};self._current_acceptance_signature=None
        self.feedback_errors=[]
        self.history=self.media/'point_editor_history';self.history_index=self.history/'index.json'
        camera_path=self.media/'camera_seeds.npz'
        if not camera_path.exists():camera_path=self.source_media/'camera_seeds.npz'
        self.cameras=np.load(camera_path,allow_pickle=False)
        self.camera_rows={int(i):camera for i,camera in zip(self.cameras['programme_indices'],self.cameras['camera25'])}
        require(all(i in self.camera_rows for i in self.current_indices),'当前选帧缺少对应相机；不能保存不匹配的模型登记')
        self.read_only_frames={}
        manifest=self.dataset/'batch_manifest.json'
        for batch in (read(manifest) if manifest.exists() else {}).get('accepted_review_batches',[]):
            directory=Path(batch['dataset']).resolve();source=Path(batch['media']).resolve()
            acceptance=read(directory/'user_acceptance.json')
            require(acceptance['accepted'],'关联批次尚未验收')
            for row in acceptance['frames']:
                index=row['index'];require(index not in self.indices,'审阅帧号重复')
                require(sha(directory/f'frames/{index:06d}.json')==row['packet_sha256'],'已验收数据发生变化')
                require(sha(directory/f'bindings/{index:06d}.json')==row['binding_sha256'],'已验收登记发生变化')
                self.read_only_frames[index]=(directory,source)
        self.indices=sorted(self.indices+list(self.read_only_frames))
        self.refresh_current_acceptance()
        self.recover_transactions()
        self.flush_feedback()
        self._review_cache={}

    def refresh_current_acceptance(self):
        signature=file_signature(self.current_acceptance_path)
        if signature is None:
            self._current_acceptance={};self._current_acceptance_signature=None
            return
        if signature==self._current_acceptance_signature:return
        receipt=read(self.current_acceptance_path)
        require(isinstance(receipt,dict) and receipt.get('accepted') is True,'当前批次验收记录未明确验收')
        frames=receipt.get('frames')
        require(isinstance(frames,list),'当前批次验收记录缺少逐帧登记')
        accepted={}
        for row in frames:
            require(isinstance(row,dict) and type(row.get('index')) is int,'当前批次验收帧号不正确')
            index=row['index']
            require(index not in accepted,'当前批次验收帧号重复')
            accepted[index]=row
        require(set(accepted)==set(self.current_indices),'当前批次验收记录未恰好覆盖全部当前帧')
        for index,row in accepted.items():
            require(sha(self.dataset/f'frames/{index:06d}.json')==row.get('packet_sha256'),'已验收数据发生变化')
            require(sha(self.dataset/f'bindings/{index:06d}.json')==row.get('binding_sha256'),'已验收登记发生变化')
        require(file_signature(self.current_acceptance_path)==signature,'当前批次验收记录在核对期间发生变化')
        self._current_acceptance=accepted;self._current_acceptance_signature=signature
        self.flush_feedback()

    def flush_feedback(self,transaction_path=None):
        """Committed journals are durable outbox entries; feedback failure cannot undo a save."""
        paths=[Path(transaction_path)] if transaction_path else sorted(self.history.glob('*/transaction.json'))
        self.feedback_errors=[]
        for transaction_path in paths:
            try:
                from rinkspace.feedback_outbox import deliver_human_edit
                transaction=read(transaction_path)
                if transaction.get('status')!='committed':continue
                index=transaction['index'];folder=transaction_path.parent
                after=folder/'after_packet.json';binding=folder/'after_binding.json'
                # Historical journals own before bytes. Their after version is usable
                # only while its authenticated packet/binding still matches.
                if not after.exists():
                    live=self.path(index)
                    if not live.exists() or sha(live)!=transaction['packet_after']['sha256']:
                        raise ValueError('Historical committed after snapshot unavailable; no current frame substituted')
                    write_atomic(guard_project_write(after),live.read_bytes())
                if not binding.exists():
                    live=self.binding_path(index)
                    if read(live)['source_packet']['sha256']!=sha(after):
                        raise ValueError('Historical committed binding snapshot unavailable')
                    write_atomic(guard_project_write(binding),live.read_bytes())
                acceptance=None
                if self.current_acceptance_path.exists():
                    receipt=read(self.current_acceptance_path)
                    row=next((r for r in receipt.get('frames',[]) if r.get('index')==index),None)
                    if row and row.get('packet_sha256')==sha(after) and row.get('binding_sha256')==sha(binding):
                        acceptance=self.current_acceptance_path
                deliver_human_edit(ROOT,transaction_path,dataset=self.dataset,
                    after_packet_path=after,binding_path=binding,acceptance_path=acceptance,
                    source_image_path=self.source_media/f'frames/{index:06d}.png')
            except Exception as error:
                item=dict(transaction=str(transaction_path),error=type(error).__name__+': '+str(error),
                          committed_edit_preserved=True,retry='reopen editor or call flush_feedback')
                self.feedback_errors.append(item)
                try:write_atomic(guard_project_write(transaction_path.parent/'feedback_retry.json'),json_bytes(item))
                except Exception:pass
        return self.feedback_errors

    def is_read_only(self,index):
        self.refresh_current_acceptance()
        return index in self.read_only_frames or index in self._current_acceptance

    def recover_transactions(self):
        if not self.history.exists():return
        for path in self.history.glob('*/transaction.json'):
            record=read(path)
            if record.get('status') not in ('prepared','applying'):continue
            require(not self.current_acceptance_path.exists(),'已验收批次存在未完成的保存事务；未回退任何已验收数据')
            for item in record['files']:
                target=Path(item['path']);before=path.parent/item['before_file']
                if item['existed']:write_atomic(target,before.read_bytes())
                elif target.exists():target.unlink()
            record['status']='rolled_back_after_interruption';write_atomic(path,json_bytes(record))

    def path(self,index):
        require(type(index) is int and index in self.indices,'帧号不在当前批次内')
        directory=self.read_only_frames.get(index,(self.dataset,None))[0]
        return directory/f'frames/{index:06d}.json'

    def binding_path(self,index):
        return self.path(index).parent.parent/f'bindings/{index:06d}.json'

    def media_path(self,index,kind,packet=None):
        source=self.read_only_frames.get(index,(None,self.source_media))[1]
        if kind=='source':return source/f'frames/{index:06d}.png'
        revised=self.media/f'navigation/{index:06d}.png'
        metadata=revised.with_suffix('.json')
        if revised.exists() and metadata.exists() and read(metadata).get('scene_sha256')==self.model_hashes['scene.json']:
            return revised
        if packet is None:packet=read(self.path(index))
        path=Path(packet['source'].get('navigation_overlay_path',source/f'navigation/{index:06d}.png')).resolve()
        require(path.is_relative_to(self.media) or path.is_relative_to(source),'Navigation must remain inside the selected media directories')
        return path

    def histories(self):
        return read(self.history_index) if self.history_index.exists() else {}

    def bootstrap(self):
        self.refresh_current_acceptance()
        used=Counter()
        for index in self.indices:
            for row in read(self.binding_path(index))['observations']:used[row['target_id']]+=len(row['source_vertex_indices'])
        targets=[dict(id=tid,label=target_label(tid,d),kind=d['kind'],layer=target_layer(d)) for tid,d in self.registry['targets'].items()]
        targets.sort(key=lambda r:(r['id'] not in used,list(LAYERS).index(r['layer']),r['id']))
        contiguous=len(self.indices)==self.indices[-1]-self.indices[0]+1
        title=(f'第 {self.indices[0]}–{self.indices[-1]} 帧 · 点编辑' if contiguous else
               f'选中 {len(self.indices)} 张 · 节目第 {self.indices[0]}–{self.indices[-1]} 帧 · 点编辑')
        if self._current_acceptance:title=title.replace('点编辑','已验收 · 只读审阅')
        return dict(title=title,frame_indices=self.indices,layers=LAYERS,targets=targets,
                    dataset_id=digest(str(self.dataset).encode('utf-8')),native_width=1920,native_height=1080,hard_point_ceiling=500)

    def frame(self,index):
        with self.lock:
            read_only=self.is_read_only(index)
            path=self.path(index);p=read(path);binding=read(self.binding_path(index))
            packet_hash=sha(path)
            require(binding['source_packet']['sha256']==packet_hash,'标点和模型登记版本不一致，请先修复数据')
            if index in self._current_acceptance:
                accepted=self._current_acceptance[index]
                require(packet_hash==accepted['packet_sha256'],'已验收数据发生变化')
                require(sha(self.binding_path(index))==accepted['binding_sha256'],'已验收登记发生变化')
            addresses={(o['source_feature_id'],o['source_segment_index'],vi):(pid,o['target_id']) for o in binding['observations'] for vi,pid in zip(o['source_vertex_indices'],o['source_point_ids'])}
            segments=[]
            for f in p['features']:
                for si,s in enumerate(f['segments']):
                    points=[dict(id=addresses[(f['id'],si,vi)][0],uv=uv,target_id=addresses[(f['id'],si,vi)][1]) for vi,uv in enumerate(s['uv'])]
                    segments.append(dict(key=f"{f['id']}/{si}",feature_id=f['id'],segment_index=si,
                        layer=f.get('review_layer') or layer_for(f['id']),target_id=f.get('target_id') or (points[0]['target_id'] if points else None),closed=bool(s.get('closed')),points=points,
                        geometry=f.get('geometric_interpretation'),
                        source_draw_uv=s.get('source_native_readout_uv') if s.get('draw_current_native_material_readout') and not f.get('manual_deletion_applied') else None))
            for f in p.get('unbound_static_materials',[]):
                for si,s in enumerate(f['segments']):
                    segments.append(dict(key=f"{f['id']}/{si}",feature_id=f['id'],segment_index=si,
                        layer='unbound_static',target_id=None,label=f.get('label_zh',f['id']),closed=bool(s.get('closed')),
                        points=[dict(id=s.get('point_ids',[f"{f['id']}/segment/{si}/vertex/{j}" for j in range(len(s['uv']))])[vi],uv=uv,target_id=None,
                            unbound=True,label=f.get('label_zh',f['id'])) for vi,uv in enumerate(s['uv'])]))
            return dict(index=index,revision=sha(path),point_count=p['point_count'],source_time=timestamp(p['source']),
                source_point_count=p.get('source_point_count',p['point_count']),
                source_limits=p.get('unresolved_source_regions',[])+p.get('unresolved_board_regions',[]),
                pts=p['source']['native_pts'],time_base=p['source']['time_base'],source_url=f'/media/source/{index}',
                navigation_url=f'/media/navigation/{index}',segments=segments,inferred_segments=p.get('inferred_segments',[]),
                history_available=bool(self.histories().get(str(index))),
                status='user_accepted' if index in self._current_acceptance else p.get('status'),read_only=read_only)

    def checked_base(self,index,revision):
        path=self.path(index)
        require(index not in self.read_only_frames and not self.current_acceptance_path.exists(),
                '这一帧已经验收，当前页面只读；未改动任何点')
        if sha(path)!=revision:raise EditConflict('这一帧已在别处修改。请保留当前草稿，刷新后核对；未覆盖任何数据。')
        for name,expected in self.model_hashes.items():
            if sha(self.model/name)!=expected:raise EditConflict('模型版本已改变，请重启编辑器后再保存。')
        return read(path)

    def edit(self,index,revision,operations):
        with self.lock:
            old=self.checked_base(index,revision);tx=uuid.uuid4().hex
            packet=apply_operations(old,operations,self.registry,tx)
            histories=self.histories();histories.setdefault(str(index),[]).append(tx)
            self.commit(index,packet,old,tx,histories,dict(action='edit',operations=operations,base_revision=revision))
            result=self.frame(index);result['message']='已保存到点数据和模型登记';return result

    def undo(self,index,revision):
        with self.lock:
            old=self.checked_base(index,revision);histories=self.histories();stack=histories.get(str(index),[])
            require(bool(stack),'当前帧没有可撤销的已保存修改')
            last=stack.pop();packet=read(self.history/last/'before_packet.json');tx=uuid.uuid4().hex
            packet['manual_editor']=dict(revision_id=tx,undo_of=last,authoritative_over_extraction_candidates=True,
                                        updated_at=datetime.now(timezone.utc).isoformat(),whole_frame_accepted=False)
            packet['user_accepted']=False
            self.commit(index,packet,old,tx,histories,dict(action='undo',undo_of=last,base_revision=revision))
            result=self.frame(index);result['message']='已撤销最近一次保存';return result

    def review_signature(self,index):
        navigation=self.media/f'navigation/{index:06d}.png'
        return (file_signature(self.path(index)),file_signature(navigation),
                file_signature(navigation.with_suffix('.json')),file_signature(self.current_acceptance_path))

    def review_entry(self,index,packet):
        segments=list(source_segments(packet))
        frame_media=self.read_only_frames.get(index,(None,self.media))[1]
        return dict(index=index,source=self.media_path(index,'source').as_uri(),
            navigation=self.media_path(index,'navigation',packet).as_uri(),overlay=(frame_media/f'overlays/{index:06d}.png').as_uri(),
            source_time=timestamp(packet['source']),pts=packet['source']['native_pts'],time_base=packet['source']['time_base'],
            point_count=packet.get('source_point_count',packet['point_count']),counts={key:sum(len(uv) for layer,_,uv,_ in segments if layer==key) for key in LAYERS},
            status='已由用户验收' if self.is_read_only(index) else ('已手动修改，等待验收' if packet.get('manual_editor') else '等待用户验收'),
            notes=[r.get('note_zh',r.get('reason','材料边界暂不能可靠区分')) for r in packet.get('unresolved_source_regions',[])+packet.get('unresolved_board_regions',[])],
            inferred_segments=packet.get('inferred_segments',[]),segments=display_segments(packet))

    def review_candidates(self,index=None,replacement=None):
        candidates={}
        for i in self.indices:
            signature=self.review_signature(i)
            cached=self._review_cache.get(i)
            if i!=index and cached is not None and cached[0]==signature:
                candidates[i]=cached
                continue
            packet=replacement if i==index else read(self.path(i))
            entry=self.review_entry(i,packet)
            fragment=json.dumps(entry,ensure_ascii=False,separators=(',',':'))
            if self.review_signature(i)!=signature:
                raise EditConflict('审阅数据在读取期间发生变化，请稍后重试；未写入修改。')
            candidates[i]=(signature,entry,fragment)
        return candidates

    def warm_review_cache(self):
        with self.lock:
            self._review_cache=self.review_candidates()

    def prepare_review(self,index,replacement):
        candidates=self.review_candidates(index,replacement)
        entries=[candidates[i][1] for i in self.indices]
        fragments=[candidates[i][2] for i in self.indices]
        result=page_html(entries,self.dataset/'batch_manifest.json',frame_json=fragments)
        return self.editor_link(result).encode('utf-8'),candidates

    def review_html(self,index,replacement):
        return self.prepare_review(index,replacement)[0]

    def editor_link(self,page):
        page=re.sub(r'<!-- POINT_EDITOR_LINK_BEGIN -->.*?<!-- POINT_EDITOR_LINK_END -->','',page,flags=re.S)
        if self.editor_url:
            label=('打开已验收只读页面：查看标点与模型归属' if self._current_acceptance else
                   '打开可编辑页面：删点 / 框选 / 加点 / 撤销 / 保存')
            message=f'<!-- POINT_EDITOR_LINK_BEGIN --><p><a href="{self.editor_url}">{label}</a></p><!-- POINT_EDITOR_LINK_END -->'
            page=page.replace('</header>',message+'</header>',1)
        return page

    def commit(self,index,packet,old,tx,histories,action):
        path=self.path(index);packet_data=json_bytes(packet)
        # Current explicit human edits/undo are authoritative. Candidate feedback
        # gates run for automatic admission, never veto this save on an old case.
        binding=make_binding(packet,path,self.registry,apply_feedback_withdrawals=False);binding['source_packet']=ref(path,packet_data)
        rows,counts,corners=assemble(packet,binding,self.registry,self.scene,self.templates,apply_feedback_withdrawals=False)
        residual,jac=evaluate_rows(self.camera_rows[index],rows,jacobian=True)
        require(residual.shape==(2*packet['point_count'],) and np.isfinite(residual).all() and np.isfinite(jac).all(),'模型入口检查失败，未保存')
        raw=cv2.imread(str(self.source_media/f'frames/{index:06d}.png'))
        require(raw is not None and raw.shape==(1080,1920,3),'找不到原生画面')
        require(digest(raw.tobytes())==packet['source']['decoded_BGR_sha256'],'原生画面版本不一致')
        overlay=native_overlay(raw,list(source_segments(packet)),packet.get('inferred_segments',[]),packet=packet)
        ok,png=cv2.imencode('.png',overlay);require(ok,'不能生成审阅图')
        binding_path=self.dataset/f'bindings/{index:06d}.json';binding_data=json_bytes(binding)
        manifest=read(self.dataset/'batch_manifest.json')
        row=next(r for r in manifest['frames'] if r['index']==index)
        row.update(point_count=packet['point_count'],packet=ref(path,packet_data),model_binding=ref(binding_path,binding_data),
                   source_point_count=packet.get('source_point_count',packet['point_count']),
                   unbound_material_point_count=packet.get('source_point_count',packet['point_count'])-packet['point_count'],
                   target_point_counts=dict(counts),user_accepted=False,manual_editor_revision=tx)
        numbers=[r.get('source_point_count',r['point_count']) for r in manifest['frames']]
        manifest['density'].update(total=sum(numbers),minimum=min(numbers),maximum=max(numbers),mean=float(np.mean(numbers)))
        manifest.update(user_accepted=False,status='contains_user_page_edits_awaiting_review',
                        manual_editor_history=str(self.history_index),automated_extraction_is_not_authoritative_for_manually_edited_frames=True)
        # Prior batch audits are historical; do not claim their old hashes validate later human edits.
        for key in ('model_entry_validation','model_semantic_audit'):
            if key in manifest:manifest.setdefault('audits_before_manual_edits',{}).setdefault(key,manifest.pop(key))
        manifest['latest_manual_edit_validation']=dict(frame=index,point_count=packet['point_count'],
            all_bound_points_have_unique_typed_model_targets=True,
            all_points_have_unique_typed_model_targets=not packet.get('unbound_static_materials'),
            finite_residual_entry=True,optimization_performed=False)
        model_path=self.model/'model.json';model_data=None
        if model_path.exists():
            model=read(model_path)
            for batch in model.get('candidate_batch_datasets',[]):
                if Path(batch['path']).resolve()==(self.dataset/'batch_manifest.json').resolve():
                    # The manifest hashes the model. Its backlink must stay a
                    # locator, otherwise every human save creates a hash cycle.
                    batch.pop('sha256',None)
                    batch.update(path=str((self.dataset/'batch_manifest.json').resolve()),
                        point_count=sum(r['point_count'] for r in manifest['frames']))
            model['candidate_source_point_count']=sum(r['point_count'] for r in model.get('candidate_batch_datasets',[])+model.get('candidate_frame_datasets',[]))
            model_data=json_bytes(model)
            manifest['model']=ref(model_path,model_data)
        manifest_data=json_bytes(manifest)
        review_data,review_cache=self.prepare_review(index,packet)
        writes={path:packet_data,binding_path:binding_data,self.dataset/'batch_manifest.json':manifest_data,
            self.media/f'overlays/{index:06d}.png':png.tobytes(),self.dataset/'review.html':review_data,
            self.history_index:json_bytes(histories)}
        if model_data is not None:writes[model_path]=model_data
        release_path=self.media/'release_check.json'
        if release_path.exists():
            release=read(release_path);release.update(status='historical_release_followed_by_user_page_edits',
                last_manual_edit_revision=tx,current_point_count=sum(numbers),requires_new_batch_review=True)
            writes[release_path]=json_bytes(release)
        folder=self.history/tx;folder.mkdir(parents=True)
        write_atomic(folder/'before_packet.json',json_bytes(old))
        write_atomic(guard_project_write(folder/'after_packet.json'),packet_data)
        write_atomic(guard_project_write(folder/'after_binding.json'),binding_data)
        transaction=dict(schema='rinkspace.human-frame-point-edit.v1',id=tx,index=index,status='prepared',
            created_at=datetime.now(timezone.utc).isoformat(),action=action,source_pts=packet['source']['native_pts'],
            model_id=self.registry['model_id'],point_count_before=old['point_count'],point_count_after=packet['point_count'],
            packet_after=ref(path,packet_data),model_geometry_changed=False,
            feedback_outbox=dict(schema='rinkspace.committed-edit-outbox.v1',
                after_packet_snapshot='after_packet.json',after_binding_snapshot='after_binding.json',
                consume_only_status='committed'),files=[])
        for n,(target,data) in enumerate(writes.items()):
            entry=dict(path=str(target),existed=target.exists(),before_file=f'before_{n}.bin',after_sha256=digest(data))
            if target.exists():(folder/entry['before_file']).write_bytes(target.read_bytes())
            transaction['files'].append(entry)
        write_atomic(folder/'transaction.json',json_bytes(transaction))
        try:
            for target,data in writes.items():write_atomic(target,data)
            transaction['status']='committed';write_atomic(folder/'transaction.json',json_bytes(transaction))
        except Exception:
            for entry in transaction['files']:
                target=Path(entry['path'])
                if entry['existed']:write_atomic(target,(folder/entry['before_file']).read_bytes())
                elif target.exists():target.unlink()
            transaction['status']='rolled_back';write_atomic(folder/'transaction.json',json_bytes(transaction));raise
        # Publish display state only after every authoritative file was committed.
        # The edited packet was atomically replaced, so its new file identity differs.
        signature,entry,fragment=review_cache[index]
        try:
            packet_signature=file_signature(path)
            if (packet_signature is None or sha(path)!=binding['source_packet']['sha256']
                    or file_signature(path)!=packet_signature):
                review_cache.pop(index,None)
            else:
                review_cache[index]=((packet_signature,*signature[1:]),entry,fragment)
        except OSError:
            review_cache.pop(index,None)
        self._review_cache=review_cache
        self.flush_feedback(folder/'transaction.json')


def create_server(store,port=0):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):print(fmt%args,flush=True)
        def send_bytes(self,data,kind,status=200):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def json(self,value,status=200):self.send_bytes(json_bytes(value),'application/json; charset=utf-8',status)
        def host_valid(self):return self.headers.get('Host')==f'127.0.0.1:{self.server.server_port}'
        def do_GET(self):
            try:
                if not self.host_valid():return self.json({'error':'仅允许本机编辑器地址'},403)
                path=urlsplit(self.path).path
                if path=='/api/bootstrap':return self.json(dict(store.bootstrap(),csrf_token=token))
                if path=='/api/health':return self.json(dict(status='ok',dataset=str(store.dataset),review_cache_frames=len(store._review_cache)))
                m=re.fullmatch(r'/api/frames/(\d+)',path)
                if m:return self.json(store.frame(int(m[1])))
                m=re.fullmatch(r'/media/(source|navigation)/(\d+)',path)
                if m:
                    index=int(m[2]);store.path(index)
                    file=store.media_path(index,m[1])
                elif path in ('/','/index.html'):file=STATIC/'frame_point_editor.html'
                elif path in ('/static/frame_point_editor.js','/static/frame_point_editor.css'):file=STATIC/path.rsplit('/',1)[1]
                else:return self.json({'error':'找不到页面'},404)
                self.send_bytes(file.read_bytes(),(mimetypes.guess_type(file)[0] or 'application/octet-stream')+('' if file.suffix=='.png' else '; charset=utf-8'))
            except (ValueError,KeyError) as exc:self.json({'error':str(exc)},400)
            except FileNotFoundError:self.json({'error':'找不到文件'},404)
        def do_POST(self):
            origin=f'http://127.0.0.1:{self.server.server_port}'
            if (not self.host_valid() or self.headers.get('Origin')!=origin or self.headers.get('X-Editor-Token')!=token or
                    self.headers.get('Content-Type','').split(';')[0]!='application/json'):
                return self.json({'error':'请从本机编辑页面保存'},403)
            try:
                size=int(self.headers.get('Content-Length','0'));require(0<size<=2_000_000,'修改请求大小不正确')
                body=json.loads(self.rfile.read(size));require(isinstance(body,dict),'修改请求格式不正确')
                match=re.fullmatch(r'/api/frames/(\d+)/(edit|undo)',urlsplit(self.path).path)
                require(match is not None,'未知保存操作')
                index=int(match[1]);revision=body.get('base_revision')
                result=store.edit(index,revision,body.get('operations')) if match[2]=='edit' else store.undo(index,revision)
                self.json(result)
            except EditConflict as exc:self.json({'error':str(exc)},409)
            except (ValueError,KeyError,TypeError) as exc:self.json({'error':str(exc)},400)
            except Exception as exc:
                import traceback;traceback.print_exc();self.json({'error':'保存失败，未完成的写入已回退：'+str(exc)},500)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    store.editor_url=f'http://127.0.0.1:{server.server_port}/'
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--media',type=Path,required=True)
    p.add_argument('--source-media',type=Path,help='Read native images, navigation and cameras from an existing media directory')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--port',type=int,default=48170);p.add_argument('--open',action='store_true')
    args=p.parse_args();store=PointEditorStore(args.dataset,args.media,args.model,args.source_media)
    store.warm_review_cache()
    server=create_server(store,args.port)
    runtime=store.media/'point_editor_runtime.json'
    write_atomic(runtime,json_bytes(dict(url=store.editor_url,pid=os.getpid(),dataset=str(store.dataset),media=str(store.media),source_media=str(store.source_media),model=str(store.model))))
    page=store.dataset/'review.html'
    if page.exists():write_atomic(page,store.editor_link(page.read_text(encoding='utf-8')).encode('utf-8'))
    print(store.editor_url,flush=True)
    if args.open:
        import webbrowser;webbrowser.open(store.editor_url)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
