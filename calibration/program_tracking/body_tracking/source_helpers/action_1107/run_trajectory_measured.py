import ctypes as c
import json
from pathlib import Path
import runpy
import sys
import threading
import time

output=Path(sys.argv[1])
sys.argv=[sys.argv[0],*sys.argv[2:]]
stop=threading.Event()
started=time.perf_counter()
class GPUMemory(c.Structure):
    _fields_=[('total',c.c_uint64),('free',c.c_uint64),('used',c.c_uint64)]
def nvml_samples():
    try:
        lib=c.WinDLL('nvml.dll')
        handle=c.c_void_p()
        mem=GPUMemory()
        if lib.nvmlInit_v2()!=0 or lib.nvmlDeviceGetHandleByIndex_v2(0,c.byref(handle))!=0:
            raise RuntimeError('NVML init or device handle unavailable')
        lib.nvmlDeviceGetMemoryInfo.argtypes=[c.c_void_p,c.POINTER(GPUMemory)]
        with output.open('x',encoding='utf8',buffering=1) as f:
            while True:
                if lib.nvmlDeviceGetMemoryInfo(handle,c.byref(mem))!=0: raise RuntimeError('NVML reading unavailable')
                f.write(json.dumps(dict(seconds=time.perf_counter()-started,device_used_bytes=mem.used,device_total_bytes=mem.total))+'\n')
                if stop.wait(.25): break
        lib.nvmlShutdown()
    except (OSError,AttributeError,RuntimeError) as error:
        output.with_suffix('.error.txt').write_text(str(error),encoding='utf8')
worker=threading.Thread(target=nvml_samples,daemon=True)
worker.start()
try:
    runpy.run_module('calibration.program_tracking.body_tracking.trajectory_backend',run_name='__main__')
finally:
    stop.set()
    worker.join(2.)
    print(json.dumps(dict(event='measured_process_wall',seconds=time.perf_counter()-started)),flush=True)
