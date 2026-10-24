"""Atomic current-model generations shared by inspection, editing and solving.

Parameters are separate from immutable evidence. A visible revision is one whole
ModelBundle; a new generation is complete before its current pointer changes.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import threading
import uuid
import numpy as np
from .model import load_model_bundle,save_model_bundle
from .store import StoreError,atomic_json,checked_path,read_object

_STORES={}
_STORES_LOCK=threading.Lock()

class CachedModelSnapshotWriter:
    """Write complete existing-schema snapshots with cached constant metadata.

    Each generation retains its full model.json and parameters.npz and loads
    independently. A changed model structure requires a new writer.
    """
    def __init__(self, model, template_directory):
        self.base = model
        template = Path(template_directory)
        save_model_bundle(model, template)
        metadata = json.loads((template/'model.json').read_text(encoding='utf8'))
        state = metadata.pop('state')
        for key in ('id', 'revision', 'sha256'):
            state.pop(key)
        dump = lambda value: json.dumps(value, ensure_ascii=False,
                                       separators=(',', ':'), allow_nan=False)
        self.prefix = dump(metadata)[:-1] + ',"state":{'
        self.suffix = ',' + dump(state)[1:-1] + '}}\n'
        self.array_names = dict(state['arrays'])

    def write(self, model, directory):
        base = self.base
        if (model.id, model.revision) != (base.id, base.revision):
            raise ValueError('snapshot model revision changed; create a new writer')
        for key in ('scene', 'rigs', 'shots', 'artworks', 'installations',
                    'provenance', 'landmarks'):
            if getattr(model, key) is not getattr(base, key):
                raise ValueError('snapshot immutable model member changed: ' + key)
        if (model.state.blocks is not base.state.blocks or
                model.state.coordinate_system_id != base.state.coordinate_system_id):
            raise ValueError('snapshot parameter layout changed')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {self.array_names[b.id]: model.state.get(b.id)
                  for b in model.state.blocks}
        np.savez(directory/'parameters.npz', **arrays)
        dynamic = ('"id":' + json.dumps(model.state.id, ensure_ascii=False) +
                   ',"revision":' + str(int(model.state.revision)) +
                   ',"sha256":' + json.dumps(model.state.digest()))
        (directory/'model.json').write_text(self.prefix + dynamic + self.suffix,
                                            encoding='utf8')
        return directory/'model.json'


def model_store(package,data_root):
    key=(str(Path(package).resolve()),str(Path(data_root).resolve()))
    with _STORES_LOCK:
        if key not in _STORES:_STORES[key]=RuntimeModelStore(*key)
        return _STORES[key]

class RuntimeModelStore:
    def __init__(self,package,data_root):
        self.root=Path(package).resolve();self.data=Path(data_root).resolve()
        try:self.root.relative_to(self.data)
        except ValueError:raise StoreError('模型状态必须在Data内',422,'model_state_location')
        self.pointer=self.root/'current.json';self._lock=threading.RLock()
        self._cache_key=None;self._model=None;self._inspection_cache=None
    @contextmanager
    def _writer(self):
        self.root.mkdir(parents=True,exist_ok=True)
        path=checked_path(self.root/'write.lock')
        with path.open('a+b') as file:
            file.seek(0,2)
            if not file.tell():file.write(b'0');file.flush()
            file.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(file.fileno(),msvcrt.LK_LOCK,1)
            else:
                import fcntl
                fcntl.flock(file.fileno(),fcntl.LOCK_EX)
            try:yield
            finally:
                file.seek(0)
                if os.name=='nt':msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(file.fileno(),fcntl.LOCK_UN)
    def _location(self):
        path=self.root
        self._pointer_info=None
        if self.pointer.is_file():
            p=read_object(self.pointer)
            if p.get('schema')!='rinksolve.current_model_pointer.v1':raise StoreError('模型指针版本错误',503,'model_pointer_schema')
            self._pointer_info=p
            path=(self.root/p['generation']).resolve()
            try:path.relative_to(self.root)
            except ValueError:raise StoreError('模型generation越界',503,'model_pointer_path')
        return path
    def load(self):
        with self._lock:
            location=self._location()
            key=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in (location/'model.json',location/'parameters.npz'))
            if key!=self._cache_key:
                try:self._model=load_model_bundle(location)
                except (ValueError,KeyError,TypeError) as exc:raise StoreError('当前模型状态不一致: '+str(exc),503,'model_state_invalid') from exc
                self._cache_key=key;self._inspection_cache=None
            if self._pointer_info is not None:
                p=self._pointer_info;m=self._model
                if (p.get('model_id'),p.get('model_revision'),p.get('state_id'),p.get('parameter_revision'),p.get('sha256')) != (m.id,m.revision,m.state.id,m.state.revision,m.state.digest()):
                    raise StoreError('当前指针与完整模型状态身份不一致',503,'model_pointer_identity')
            return self._model
    def inspection(self,native_pts=None):
        with self._lock:
            m=self.load()
            if self._inspection_cache is None:self._inspection_cache=m.inspection(samples=128)
            if native_pts is None:return self._inspection_cache
            if type(native_pts) is not int:raise StoreError('须给原生整数PTS',422,'native_pts_type')
            shot=m.shots[0];operator=m.operator(shot.id)
            try:pose=operator.pose_at(operator.frame_time(np.asarray(native_pts,dtype=np.int64)))
            except (ValueError,TypeError) as exc:raise StoreError(str(exc),422,'native_pts_scope') from exc
            result=dict(self._inspection_cache)
            result['selected_exposure']=dict(native_pts=native_pts,source_seconds=native_pts*shot.time_base_num/shot.time_base_den,
                shot_id=shot.id,rig_id=shot.rig_id,R_cw=pose['R_cw'].tolist(),C_w=pose['C_w'].tolist(),K=pose['K'].tolist(),
                parameter_state_id=m.state.id,model_revision=m.revision)
            result['rigs']=[dict(r) for r in self._inspection_cache['rigs']]
            for rig in result['rigs']:
                if rig['id']==shot.rig_id:
                    rig.update(inspection_position_world=pose['C_w'].tolist(),inspection_direction_world=pose['R_cw'][2].tolist(),inspection_source_pts=native_pts)
            return result
    def inspection_record(self,native_pts=None):
        with self._lock:
            model=self.load()
            return self.inspection(native_pts),model,Path(self._cache_key[0][0]).parent
    def parameters(self):
        with self._lock:
            m=self.load()
            blocks=[]
            for spec in m.state.blocks:
                blocks.append({'id':spec.id,'owner':{'kind':spec.owner.kind.value,'id':spec.owner.id},
                    'mode':spec.mode.value,'components':[{'name':c.name,'unit':c.unit.unit.value,'coordinate_system_id':c.unit.reference_id} for c in spec.components],
                    'values':m.state.get(spec.id).tolist()})
            return {'schema':'rinksolve.editable_parameters.v1','model_id':m.id,'model_revision':m.revision,
                'state_id':m.state.id,'revision':m.state.revision,'sha256':m.state.digest(),
                'blocks':blocks,'source_evidence_mutable':False,
                'notes':'编辑同一物理模型参数；数值初值不是用户确认事实。新的修订使旧数值准备失效，原始UV/PTS/ID不变。'}
    def update(self,payload):
        if not isinstance(payload,dict) or set(payload)-{'block_id','values','expected_revision'}:
            raise StoreError('参数请求只允许block_id/values/expected_revision',422,'parameter_payload')
        if type(payload.get('expected_revision')) is not int or not isinstance(payload.get('block_id'),str):
            raise StoreError('必须给block_id和当前整数revision',422,'parameter_payload')
        values=payload.get('values')
        if not isinstance(values,list) or any(type(x) not in (int,float) or not np.isfinite(x) for x in values):
            raise StoreError('参数必须为有限实数数组',422,'parameter_values')
        with self._lock,self._writer():
            m=self.load()
            if payload['expected_revision']!=m.state.revision:
                raise StoreError('当前参数已变化，请刷新后重新核对',409,'parameter_revision_conflict')
            try:
                state=m.state.updated({payload['block_id']:values},expected_revision=payload['expected_revision'])
                changed=m.with_state(state)
                # Only the affected current model is checked. No old source chain.
                changed.inspection(samples=32)
            except (ValueError,KeyError,TypeError) as exc:
                raise StoreError('参数违反当前模型定义: '+str(exc),422,'parameter_model_domain') from exc
            return self._publish(changed,m,{'actor':'user_parameter_editor','block_id':payload['block_id'],
                'before':m.state.get(payload['block_id']).tolist(),'after':values})
    def publish_solver_state(self,model,*,expected_revision,diagnostics,initialization=False):
        if type(initialization) is not bool:raise StoreError('initialization必须为布尔值')
        with self._lock,self._writer():
            previous=self.load()
            if previous.state.revision!=expected_revision:
                raise StoreError('用户或另一接受步已修改状态，求解器须重建当前准备',409,'parameter_revision_conflict')
            if model.id!=previous.id or model.revision!=previous.revision:
                raise StoreError('求解状态不能暗改模型结构',422,'solver_model_revision')
            if model.state.revision<=previous.state.revision:
                raise StoreError('新发布状态必须增加参数revision',422,'solver_parameter_revision')
            model.inspection(samples=32)
            return self._publish(model,previous,{'actor':'solver_initialization' if initialization else 'solver_accepted_step','initialization':initialization,'diagnostics':diagnostics})
    def _publish(self,model,previous,edit):
        stamp=datetime.now(timezone.utc).isoformat()
        name='g%06d_'%model.state.revision+uuid.uuid4().hex[:8]
        target=checked_path(self.root/'generations'/name)
        for name_part in ('model.json','parameters.npz','edit.json'):checked_path(target/name_part)
        save_model_bundle(model,target)
        restored=load_model_bundle(target)
        if restored.state.digest()!=model.state.digest():raise StoreError('新状态落盘后数值不一致',500,'state_publish_failed')
        atomic_json(target/'edit.json',{'schema':'rinksolve.model_edit.v1','timestamp':stamp,
            'previous_state':previous.state.id,'previous_revision':previous.state.revision,
            'previous_sha256':previous.state.digest(),'new_state':model.state.id,'new_revision':model.state.revision,
            'new_sha256':model.state.digest(),**edit})
        atomic_json(self.pointer,{'schema':'rinksolve.current_model_pointer.v1','generation':str(target.relative_to(self.root)),
            'model_id':model.id,'model_revision':model.revision,'state_id':model.state.id,
            'parameter_revision':model.state.revision,'sha256':model.state.digest(),'updated_at':stamp})
        self._cache_key=None;self.load()
        return {'status':'saved','revision':model.state.revision,'state_id':model.state.id,
            'sha256':model.state.digest(),'edit_record':str(target/'edit.json'),
            'requires_reprepare':True,'source_evidence_modified':False}
