"""Read the one current numerical run; never infer progress from historical logs."""
from __future__ import annotations
import copy
import ctypes
import os
from pathlib import Path
from .store import read_object,StoreError

def process_status(pid):
    if type(pid) is not int or pid<=0:return {'alive':None,'reason':'no_valid_pid'}
    if os.name!='nt':
        try:os.kill(pid,0);return {'alive':True}
        except ProcessLookupError:return {'alive':False}
        except PermissionError:return {'alive':None,'reason':'permission'}
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];kernel.GetExitCodeProcess.restype=ctypes.c_int
    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle.restype=ctypes.c_int
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        return {'alive':False,'winerror':error} if error==87 else {'alive':None,'winerror':error}
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):return {'alive':None,'winerror':ctypes.get_last_error()}
        return {'alive':code.value==259,'exit_code':None if code.value==259 else code.value}
    finally:kernel.CloseHandle(handle)


def current_run(programme,data_root):
    path=programme.get('run_status')
    if not path:return None
    path=Path(path).resolve()
    try:path.relative_to(Path(data_root).resolve())
    except ValueError:raise StoreError('当前run状态须在Data内',503,'run_status_location')
    if not path.exists():return None
    value=read_object(path)
    if value.get('schema')!='rinksolve.current_run.v1' or not isinstance(value.get('run'),dict):
        raise StoreError('当前run状态契约不一致',503,'run_status_schema')
    result=copy.deepcopy(value);run=result['run']
    if not isinstance(run.get('status'),str) or not isinstance(run.get('main_log'),str):
        raise StoreError('当前run须声明真实status和主日志',503,'run_status_fields')
    log=Path(run['main_log']).resolve()
    try:log.relative_to(Path(data_root).resolve())
    except ValueError:raise StoreError('主日志须在Data内',503,'run_log_location')
    run['process']=process_status(run.get('pid'))
    if run['status'] in ('running','starting','preflight','preparing','initializing') and run['process']['alive'] is False:
        run['reported_status']=run['status'];run['status']='process_exited'
        result['phase']='process_exited';result['message']='本次数值进程已退出；保留最后实际进度，需检查主日志并自行纠错续算。'
    if programme.get('inspection_model',{}).get('kind')=='runtime_model':
        from .runtime_state import model_store
        model=model_store(programme['inspection_model']['package'],data_root).load()
        identity_matches=run.get('model_id')==model.id and run.get('model_revision')==model.revision
        result['matches_visible_model']=identity_matches
        result['visible_parameter_revision']=model.state.revision
        result['visible_parameter_sha256']=model.state.digest()
        result['matches_visible_parameters']=run.get('parameter_sha256')==model.state.digest()
        if not identity_matches:
            run['reported_status']=run['status'];run['status']='stale_model'
            result['phase']='stale_model';result['message']='当前可见模型结构已更新，本运行属于前一明确模型修订；不能作为当前模型进度。'
    result['engine']={**result.get('engine',{}),'legacy_solver_enabled':False}
    result['advance_allowed']=False
    return result
