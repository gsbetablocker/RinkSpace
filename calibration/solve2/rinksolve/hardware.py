"""Background, read-only system telemetry; unavailable metrics remain null."""
from __future__ import annotations

import copy
import ctypes
import csv
from datetime import datetime, timezone
import io
import os
import subprocess
import threading
import time


def _stamp():
    return datetime.now(timezone.utc).isoformat()


class HardwareMonitor:
    def __init__(self, poll_seconds=2.0):
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._previous_cpu = None
        self._thread = None
        self._snapshot = {"cpu_count": os.cpu_count(), "cpu_utilization": None,
                          "memory_total_gb": None, "memory_used_gb": None, "gpus": [],
                          "sampled_at": None, "errors": ["首次真实资源采样待完成"]}

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="solve2-hardware", daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=7)

    def snapshot(self):
        with self._lock:
            result = copy.deepcopy(self._snapshot)
        result["poll_seconds"] = self.poll_seconds
        return result

    def _loop(self):
        while not self._stop.is_set():
            try:
                snapshot = self.sample()
            except Exception as exc:
                snapshot = {"cpu_count": os.cpu_count(), "cpu_utilization": None,
                            "memory_total_gb": None, "memory_used_gb": None, "gpus": [],
                            "sampled_at": _stamp(), "errors": [f"资源采样失败: {type(exc).__name__}: {exc}"]}
            with self._lock:
                self._snapshot = snapshot
            self._stop.wait(self.poll_seconds)

    def _cpu_memory(self):
        if os.name == "nt":
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            idle, system, user = FILETIME(), FILETIME(), FILETIME()
            kernel.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME)] * 3
            kernel.GetSystemTimes.restype = wintypes.BOOL
            if not kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
                raise ctypes.WinError(ctypes.get_last_error())
            as_int = lambda value: value.low + (value.high << 32)
            current = (as_int(idle), as_int(system) + as_int(user))
            memory = MEMORYSTATUSEX()
            memory.length = ctypes.sizeof(memory)
            kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
            kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
            if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
                raise ctypes.WinError(ctypes.get_last_error())
            total, used = memory.total_phys, memory.total_phys - memory.avail_phys
        else:
            with open("/proc/stat", encoding="ascii") as handle:
                ticks = [int(x) for x in handle.readline().split()[1:]]
            current = (ticks[3] + (ticks[4] if len(ticks) > 4 else 0), sum(ticks[:8]))
            with open("/proc/meminfo", encoding="ascii") as handle:
                values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in handle}
            total = values["MemTotal"]
            used = total - values.get("MemAvailable", values["MemFree"])
        utilization = None
        if self._previous_cpu is not None:
            idle_delta = current[0] - self._previous_cpu[0]
            total_delta = current[1] - self._previous_cpu[1]
            if total_delta > 0:
                utilization = round(max(0.0, min(100.0, 100.0 * (1 - idle_delta / total_delta))), 2)
        self._previous_cpu = current
        return utilization, round(total / 1024 ** 3, 3), round(used / 1024 ** 3, 3)

    @staticmethod
    def _cuda_devices():
        if os.name != "nt":
            raise RuntimeError("当前系统无Windows CUDA驱动查询")
        cuda = ctypes.WinDLL("nvcuda.dll")
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cuDeviceGetCount.restype = ctypes.c_int
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGet.restype = ctypes.c_int
        cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cuda.cuDeviceGetName.restype = ctypes.c_int
        total_mem = getattr(cuda, "cuDeviceTotalMem_v2", None) or cuda.cuDeviceTotalMem
        total_mem.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        total_mem.restype = ctypes.c_int

        def check(code, operation):
            if code:
                raise RuntimeError(f"{operation} CUDA error {code}")

        check(cuda.cuInit(0), "cuInit")
        count = ctypes.c_int()
        check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        devices = []
        for index in range(count.value):
            device, memory = ctypes.c_int(), ctypes.c_size_t()
            name = ctypes.create_string_buffer(256)
            check(cuda.cuDeviceGet(ctypes.byref(device), index), "cuDeviceGet")
            check(cuda.cuDeviceGetName(name, len(name), device.value), "cuDeviceGetName")
            check(total_mem(ctypes.byref(memory), device.value), "cuDeviceTotalMem")
            devices.append({"index": index, "name": name.value.decode("utf-8", errors="replace"),
                            "memory_total_gb": round(memory.value / 1024 ** 3, 3), "memory_used_gb": None,
                            "utilization": None, "temperature": None, "source": "cuda_driver",
                            "metrics_status": "只读驱动确认设备及显存总量；NVML动态指标不可用"})
        return devices

    @classmethod
    def _gpus(cls):
        errors = []
        try:
            options = {"capture_output": True, "text": True, "timeout": 5, "check": False}
            if os.name == "nt":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                                     "--format=csv,noheader,nounits"], **options)
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout).strip() or f"exit {result.returncode}")
            rows = list(csv.reader(io.StringIO(result.stdout)))
            devices = []
            for index, row in enumerate(rows):
                if len(row) != 5:
                    raise RuntimeError(result.stdout.strip() or "nvidia-smi返回空数据")
                name, total, used, utilization, temperature = [x.strip() for x in row]

                def number(value):
                    return None if value in {"N/A", "[N/A]", "[Not Supported]"} else float(value)

                mem_total, mem_used = number(total), number(used)
                devices.append({"index": index, "name": name,
                                "memory_total_gb": None if mem_total is None else round(mem_total / 1024, 3),
                                "memory_used_gb": None if mem_used is None else round(mem_used / 1024, 3),
                                "utilization": number(utilization), "temperature": number(temperature),
                                "source": "nvidia_smi", "metrics_status": "live"})
            if devices:
                return devices, errors
            raise RuntimeError("nvidia-smi未返回GPU")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"nvidia-smi: {exc}")
        try:
            return cls._cuda_devices(), errors
        except (OSError, AttributeError, RuntimeError) as exc:
            errors.append(f"CUDA driver: {exc}")
            return [], errors

    def sample(self):
        snapshot = {"cpu_count": os.cpu_count(), "cpu_utilization": None,
                    "memory_total_gb": None, "memory_used_gb": None, "gpus": [],
                    "sampled_at": _stamp(), "errors": []}
        try:
            snapshot["cpu_utilization"], snapshot["memory_total_gb"], snapshot["memory_used_gb"] = self._cpu_memory()
        except (OSError, ValueError, KeyError) as exc:
            snapshot["errors"].append(f"CPU/memory: {exc}")
        snapshot["gpus"], gpu_errors = self._gpus()
        snapshot["errors"].extend(gpu_errors)
        return snapshot
