"""Read-only readable iteration/cost tail of an actual solver log. No solver/UI."""
from __future__ import annotations
import argparse,json,time,os,ctypes
from datetime import datetime
from pathlib import Path
from rinksolve.atomic_io import atomic_replace


def monitor(source,output,*,pid=None,once=False):
    source=Path(source).resolve();output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if source==output:raise ValueError('progress output must not replace the real main log')
    pending=b'';offset=0;inner=0;begin=None;last=None;total_steps=0;last_native=-1
    snapshot=output.with_name('current_progress.txt')
    handle=None;kernel=None
    if pid and os.name=='nt':
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.GetExitCodeProcess.restype=wintypes.BOOL
        handle=kernel.OpenProcess(0x1000,False,pid)
    with source.open('rb') as raw,output.open('w',encoding='utf8',buffering=1) as pretty:
        pretty.write('实际求解进度：迭代步数 / cost / cost下降量 / 累计用时\n')
        pretty.write('源日志：'+str(source)+'\n')
        pretty.write('时间          内层    步数             cost          cost下降    累计秒    结果\n')
        while True:
            chunk=raw.read(65536)
            if chunk:
                pending+=chunk
                lines=pending.split(b'\n');pending=lines.pop()
                for line in lines:
                    if not line.strip():continue
                    try:item=json.loads(line.decode('utf8'))
                    except (ValueError,UnicodeError):continue
                    kind=item.get('kind');stamp=datetime.fromisoformat(item['time']);local=stamp.astimezone().strftime('%H:%M:%S')
                    if kind=='inner_start':
                        inner+=1;last_native=-1
                        if begin is None:begin=stamp
                        pretty.write(local+'  开始连续求解内层 '+str(inner)+'\n')
                    if kind in ('accepted','trial','inner_start'):
                        event=item.get('event',{});native=event.get('iteration');cost=event.get('cost');change=event.get('cost_change')
                        if cost is None or native is None:continue
                        if kind!='inner_start' and native!=last_native:total_steps+=1
                        last_native=native;elapsed=(stamp-(begin or stamp)).total_seconds();result='接受' if kind=='accepted' else '初始' if kind=='inner_start' else '未接受'
                        pretty.write(f'{local}  {inner:>4}  {int(native):>6}  {float(cost):>17.6f}  {float(change or 0):>14.6g}  {elapsed:>8.1f}  {result}\n')
                        last=dict(time=local,inner=inner,iteration=int(native),total_steps=total_steps,cost=float(cost),cost_change=float(change or 0),elapsed_seconds=elapsed,result=result)
                        text=f'更新时间：{local}\n当前内层：{inner}\n当前步数：{int(native)}\ncost：{float(cost):.6f}\ncost下降：{float(change or 0):.6g}\n累计用时：{elapsed:.1f}秒\n本步：{result}\n原冰场线和相机：固定不动\n'
                        temp=snapshot.with_name(snapshot.name+'.tmp');temp.write_text(text,encoding='utf8')
                        try:atomic_replace(temp,snapshot)
                        except OSError:pass
                    elif kind=='failure':pretty.write(local+'  程序错误：'+str(item.get('error','未提供'))+'\n')
                    elif kind=='numerical_stagnation_detected':pretty.write(local+'  检测到数值停滞，正在保存完整状态。\n')
                    elif kind=='numerical_stagnation_requires_repair':pretty.write(local+'  已停止本次求解并保留完整状态，需要修复后继续；未达到收敛。\n')
                    elif kind=='outer_result':pretty.write(local+'  阶段记录：'+str(item.get('status'))+'\n')
                    elif kind=='phase_transition':pretty.write(local+'  开始细化广告拟合\n')
                    elif kind=='requires_review':pretty.write(local+'  当前数值阶段结束；广告是否贴合仍需原片核对\n')
                continue
            if once:break
            alive=False
            if pid and os.name=='nt':
                code=ctypes.c_ulong();alive=bool(handle and kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259)
            elif pid:
                try:os.kill(pid,0);alive=True
                except (ProcessLookupError,PermissionError):pass
            if pid and not alive:
                status='求解进程已退出；最后进度保留。\n'
                pretty.write(status)
                if last:
                    temp=snapshot.with_name(snapshot.name+'.tmp')
                    temp.write_text(text+'运行状态：'+status,encoding='utf8')
                    try:atomic_replace(temp,snapshot)
                    except OSError:pass
                break
            time.sleep(1)
    if handle:kernel.CloseHandle(ctypes.c_void_p(handle))
    return last


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--pid',type=int);p.add_argument('--once',action='store_true');a=p.parse_args();print(json.dumps(monitor(a.source,a.output,pid=a.pid,once=a.once),ensure_ascii=False))


if __name__=='__main__':main()
