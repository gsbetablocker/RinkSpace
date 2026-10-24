"""Reversible source-ID selection; immutable observations and complete W stay intact."""
from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import uuid
import numpy as np
from .store import StoreError, atomic_json, checked_path, now_iso, read_object

SCHEMA='rinksolve.observation_edits.v1'
EDIT_REASON_BIT=512

def text(value):
    return bytes(value).decode('utf8') if isinstance(value,(bytes,np.bytes_)) else str(value)

@dataclass(frozen=True)
class SelectionSnapshot:
    revision:int
    disabled_rows:np.ndarray
    requested_rows:np.ndarray
    disabled_supplement_ids:tuple[str,...]
    disabled_group_ids:tuple[str,...]
    explicit_reasons:dict

class ObservationSelection:
    def __init__(self,prepared,path,*,data_root):
        self.prepared=prepared
        self.data=Path(data_root).resolve()
        self.path=checked_path(path)
        try:self.path.relative_to(self.data)
        except ValueError:raise StoreError('取点编辑必须写入Data',422,'selection_path')
        self._lock=threading.RLock()
        self.ids=prepared.observation_ids()
        self._index={text(value):i for i,value in enumerate(self.ids)}
        if len(self._index)!=prepared.row_count:raise StoreError('源ID重复，不能进行取点编辑',503,'selection_source_id')
        self.supplements={str(v['id']):v for v in prepared.manifest.get('new_intersection_observations',[])+prepared.manifest.get('ad7_fixed_observations',[])}
        self._active=np.zeros(prepared.row_count,dtype=bool)
        self._active[prepared.selection_current_active_rows()]=True
        self._independent=np.zeros(prepared.row_count,dtype=bool)
        self._independent[prepared.arrays['independent_source_rows']]=True
        self._groups={}
        for i,g in enumerate(prepared.manifest['groups']):
            if not g.get('active_statistical') or not g.get('rank'):continue
            lo,hi=prepared.arrays['group_member_offsets'][i:i+2]
            self._groups[g['id']]=prepared.arrays['group_member_rows'][lo:hi]
        self._row_group={}
        for gid,rows in self._groups.items():
            for row in rows:self._row_group[int(row)]=(gid,len(rows))
        self._cache_key=None
        self._snapshot=None
    @contextmanager
    def _writer(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as stream:
            stream.seek(0,2)
            if not stream.tell():stream.write(b'0');stream.flush()
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:
                stream.seek(0)
                if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
    def ledger(self):
        if not self.path.exists():return {'schema':SCHEMA,'revision':0,'disabled':{}}
        value=read_object(self.path)
        if value.get('schema')!=SCHEMA or type(value.get('revision')) is not int or not isinstance(value.get('disabled'),dict):
            raise StoreError('取点编辑状态不一致',503,'selection_schema')
        if any(not isinstance(k,str) or not isinstance(v,dict) or not isinstance(v.get('reason'),str) for k,v in value['disabled'].items()):
            raise StoreError('取点编辑原因缺失',503,'selection_reason')
        return value
    def snapshot(self):
        with self._lock:
            key=(self.path.stat().st_mtime_ns,self.path.stat().st_size) if self.path.exists() else None
            if self._snapshot is not None and key==self._cache_key:return self._snapshot
            value=self.ledger()
            requested=np.array(sorted(self._index[k] for k in value['disabled'] if k in self._index),dtype=np.int64)
            disabled=np.zeros(self.prepared.row_count,dtype=bool);disabled[requested]=True
            groups=set()
            # Entire original W factors are removed. No submatrix or independent
            # noise is invented. Closure also handles intersecting source groups.
            changed=True
            while changed:
                changed=False
                for gid,rows in self._groups.items():
                    if gid not in groups and np.any(disabled[rows]):
                        groups.add(gid);disabled[rows]=True;changed=True
            rows=np.flatnonzero(disabled);rows.setflags(write=False);requested.setflags(write=False)
            self._snapshot=SelectionSnapshot(value['revision'],rows,requested,
                tuple(k for k in value['disabled'] if k in self.supplements),tuple(sorted(groups)),copy.deepcopy(value['disabled']))
            self._cache_key=key
            return self._snapshot
    def update(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'expected_revision','observation_ids','enabled','reason'}:
            raise StoreError('提交需包含expected_revision、observation_ids、enabled和reason')
        if type(payload['expected_revision']) is not int or type(payload['enabled']) is not bool:
            raise StoreError('修订必须为整数，enabled必须为布尔值')
        ids=payload['observation_ids'];reason=payload['reason']
        if not isinstance(ids,list) or not ids or any(not isinstance(k,str) for k in ids):raise StoreError('必须指定稳定源ID列表')
        if not isinstance(reason,str) or not reason.strip():raise StoreError('撤点或恢复必须填写具体原因')
        unknown=[k for k in ids if k not in self._index and k not in self.supplements]
        if unknown:raise StoreError('当前准备中不存在源ID：'+unknown[0],422,'selection_unknown_id')
        with self._lock,self._writer():
            before=self.ledger()
            if before['revision']!=payload['expected_revision']:raise StoreError('取点编辑已更新，请重新载入',409,'revision_conflict')
            if payload['enabled']:
                invalid=[k for k in ids if k in self._index and not self._active[self._index[k]]]
                if invalid:raise StoreError('该点原本未获当前拟合资格；恢复编辑不能绕过来源或实体限制',422,'selection_admission')
            after=copy.deepcopy(before);stamp=now_iso()
            for k in set(ids):
                if payload['enabled']:after['disabled'].pop(k,None)
                else:after['disabled'][k]={'reason':reason.strip(),'at':stamp,'actor':'local_selection_editor'}
            if after['disabled']==before['disabled']:return self.summary()
            after['revision']+=1
            directory=self.path.parent/'selection_edits'/uuid.uuid4().hex[:16]
            event={'schema':'rinksolve.observation_edit_event.v1','time':stamp,'revision':after['revision'],
                'prepared_manifest':str(self.prepared.directory/'manifest.json'),'observation_ids':ids,
                'enabled':payload['enabled'],'reason':reason.strip(),'source_pixels_modified':False,
                'message':('恢复主动退出' if payload['enabled'] else '退出优化')+' '+str(len(set(ids)))+' 个稳定源点；原UV/PTS保留','details':reason.strip()}
            atomic_json(directory/'before.json',before);atomic_json(directory/'event.json',event)
            atomic_json(self.path,after);self._cache_key=None;self._snapshot=None
            return self.summary()
    def summary(self):
        s=self.snapshot()
        return {'schema':SCHEMA,'revision':s.revision,'prepared':str(self.prepared.directory),
            'model_id':self.prepared.manifest['model_id'],'model_revision':self.prepared.manifest['prepared_against_model_revision'],
            'explicit_disabled_count':len(s.explicit_reasons),'effective_withdrawn_raw_count':len(s.disabled_rows),
            'collateral_raw_count':len(s.disabled_rows)-len(s.requested_rows),'disabled_group_ids':list(s.disabled_group_ids),
            'disabled_supplement_ids':list(s.disabled_supplement_ids),'source_pixels_modified':False,
            'policy':'Withdraw an entire original W group when any member is disabled. Restore never overrides original admission.'}
    def records(self,*,native_pts=None,query='',curve_id=None,offset=0,limit=64):
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=200:raise StoreError('分页范围无效')
        if native_pts is not None and (type(native_pts) is not int or native_pts<256256 or native_pts>575488 or native_pts%256):raise StoreError('必须使用S01原生整数PTS')
        a=self.prepared.arrays;mask=np.ones(self.prepared.row_count,dtype=bool)
        if native_pts is not None:mask&=a['native_pts']==native_pts
        registry=self.prepared.manifest['curve_registry']
        if curve_id is not None:
            indices=[i for i,c in enumerate(registry) if c['id']==curve_id]
            if not indices:mask[:]=False
            else:mask&=a['curve_index']==indices[0]
        candidates=np.flatnonzero(mask)
        if query:
            word=query.casefold()
            matched=[i for i,c in enumerate(registry) if word in (c['id']+' '+c.get('label','')).casefold()]
            candidates=np.array([int(i) for i in candidates if int(a['curve_index'][i]) in matched or word in text(self.ids[i]).casefold()],dtype=np.int64)
        s=self.snapshot();disabled=set(map(int,s.disabled_rows));explicit=set(map(int,s.requested_rows))
        group_lookup={int(r):gid for gid,rows in self._groups.items() if gid in s.disabled_group_ids for r in rows}
        supplements=[v for v in self.supplements.values() if (native_pts is None or v['native_pts']==native_pts) and (curve_id is None or v['fixed_feature_id']==curve_id) and (not query or query.casefold() in (v['id']+' '+v['fixed_feature_id']).casefold())]
        selected_raw=candidates[offset:offset+limit]
        supplemental_start=max(0,offset-len(candidates))
        selected_supplement=supplements[supplemental_start:supplemental_start+max(0,limit-len(selected_raw))]
        records=[]
        for row in selected_raw:
            i=int(row);identity=text(self.ids[i]);ci=int(a['curve_index'][i]);curve=registry[ci] if ci>=0 else None
            sigma=float(a['source_sigma'][i]);budget=float(a['engineering_budget_px'][i])
            records.append({'observation_id':identity,'row_address':i,'native_pts':int(a['native_pts'][i]),
                'uv_native_pixels':a['immutable_uv'][i].tolist(),'curve':curve,'semantic_kind':int(a['semantic_kind'][i]),
                'baseline_active':bool(self._active[i]),'baseline_exclusion_bits':int(a['exclusion_bits'][i]),
                'baseline_exclusion_reasons':[{'bit':int(bit),'code':code} for bit,code in self.prepared.manifest.get('reason_bits',{}).items() if int(a['exclusion_bits'][i]) & int(bit)],
                'explicit_disabled':i in explicit,'effective_enabled':bool(self._active[i] and i not in disabled),
                'withdrawn_group_id':group_lookup.get(i),'statistical_group_id':self._row_group.get(i,(None,None))[0],
                'statistical_group_size':self._row_group.get(i,(None,None))[1],'edit':s.explicit_reasons.get(identity),
                'sigma_px':sigma if self._independent[i] and np.isfinite(sigma) and sigma>0 else None,
                'statistical_noise_kind':'original_independent_sigma' if self._independent[i] else 'original_correlated_group_W' if i in self._row_group else 'unknown',
                'engineering_budget_px':budget if np.isfinite(budget) and budget>0 else None})
        for item in selected_supplement:
            identity=item['id']
            records.append({'observation_id':identity,'row_address':None,'native_pts':item['native_pts'],
                'uv_native_pixels':item['uv_native_pixels'],'curve':{'id':item['fixed_feature_id'],'label':item['fixed_feature_id'],'kind':'fixed_landmark'},
                'semantic_kind':'fixed_landmark','baseline_active':True,'baseline_exclusion_bits':0,
                'explicit_disabled':identity in s.disabled_supplement_ids,'effective_enabled':identity not in s.disabled_supplement_ids,
                'withdrawn_group_id':None,'edit':s.explicit_reasons.get(identity),'sigma_px':None,
                'engineering_budget_px':item.get('engineering_budget_px',self.prepared.manifest.get('new_fixed_intersections',{}).get('engineering_target_px')),
                'dependencies':item.get('parent_evidence_group_ids'),'raw_crop_path':item.get('raw_crop_path')})
        return {**self.summary(),'offset':offset,'limit':limit,'total':len(candidates)+len(supplements),'records':records}

def apply_selection(prepared,snapshot):
    """Make an in-memory prepared view; preserve source arrays and original W values.

    The numerical owner assembles a new graph for this explicit selection revision.
    An existing graph/checkpoint must never silently keep the previous selection.
    """
    from .preparation import PreparedS01
    n=prepared.row_count;disabled=np.zeros(n,dtype=bool);disabled[snapshot.disabled_rows]=True
    meta=copy.deepcopy(prepared.manifest);arrays=dict(prepared.arrays)
    for key in ('active_independent_rows','budget_ready_rows','unknown_noise_budget_ready_rows','active_query_rows'):
        if key in arrays:
            values=arrays[key];arrays[key]=values[~disabled[values]]
    if 'active_query_indices' in arrays:
        arrays['active_query_indices']=arrays['active_query_indices'][~disabled[prepared.arrays['active_query_rows']]]
    exclusion=np.array(arrays['exclusion_bits'],copy=True);exclusion[disabled]|=EDIT_REASON_BIT;exclusion.setflags(write=False);arrays['exclusion_bits']=exclusion
    kept_factors=[];matrix_rows=[];cursor=0
    for factor in meta['statistical_factors']:
        if np.any(disabled[factor['rows']]):continue
        start,count=factor['residual_start'],factor['residual_count'];matrix_rows.extend(range(start,start+count))
        factor['residual_start']=cursor;cursor+=count;kept_factors.append(factor)
    for group in meta['groups']:
        if group['id'] in snapshot.disabled_group_ids:
            group['active_statistical']=False;group['selection_exclusion_reason']='explicit observation edit; complete original W factor withdrawn'
    meta['statistical_factors']=kept_factors
    whitening=prepared.whitening[matrix_rows,:].tocsr()
    for suffix in ('data','indices','indptr'):getattr(whitening,suffix).setflags(write=False)
    meta['report']['statistical_residual_rows']=cursor
    meta['report']['statistical_nonzero_coefficients']=int(whitening.nnz)
    meta['report']['active_independent_sources']=len(arrays['active_independent_rows'])
    meta['report']['engineering_budget_ready_rows']=len(arrays['budget_ready_rows'])
    meta['report']['unknown_noise_budget_ready_rows']=len(arrays.get('unknown_noise_budget_ready_rows',[]))
    meta['report']['active_original_W_groups']=sum(bool(g.get('active_statistical')) for g in meta['groups'])
    meta['report']['selection_revision']=snapshot.revision
    meta['reason_bits'][str(EDIT_REASON_BIT)]='explicit_selection_or_complete_W_group_withdrawal'
    meta['selection_revision']=snapshot.revision;meta['selection_explicit_reasons']=snapshot.explicit_reasons
    meta['selection_disabled_supplement_ids']=list(snapshot.disabled_supplement_ids)
    if snapshot.disabled_supplement_ids:
        for field,prefix in (('new_intersection_observations','intersection'),('ad7_fixed_observations','ad7_fixed')):
            source=meta.get(field,[])
            keep=np.array([v['id'] not in snapshot.disabled_supplement_ids for v in source],dtype=bool)
            for suffix in ('native_pts','uv','landmark_index','row_auxiliary_seed','engineering_budget_px'):
                key=prefix+'_'+suffix
                if key in arrays:arrays[key]=arrays[key][keep]
            meta[field]=[v for v,k in zip(source,keep) if k]
    return PreparedS01(prepared.directory,meta,arrays,whitening)
