"""Measured Windows memory headroom for caller-owned solver workers.

No process is stopped here. The caller queues work or closes its own pool at
a safe point. Reserves and peak margins are explicit engineering settings,
not user limits. RSS sums include shared pages and are conservative.
"""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import time
from typing import Iterable

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


@dataclass(frozen=True)
class SystemMemory:
    physical_total_bytes: int
    physical_available_bytes: int
    commit_limit_bytes: int
    commit_available_bytes: int
    sampled_at: float
    backend: str


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_token: int


@dataclass(frozen=True)
class ProcessMemory:
    identity: ProcessIdentity
    parent_pid: int
    name: str
    rss_bytes: int
    private_bytes: int
    peak_rss_bytes: int
    peak_private_bytes: int
    backend: str


@dataclass(frozen=True)
class ProcessTreeMemory:
    roots: tuple[ProcessIdentity, ...]
    processes: tuple[ProcessMemory, ...]
    errors: tuple[str, ...]
    sampled_at: float

    @property
    def rss_bytes(self) -> int:
        return sum(p.rss_bytes for p in self.processes)

    @property
    def private_bytes(self) -> int:
        return sum(p.private_bytes for p in self.processes)

    @property
    def complete(self) -> bool:
        return not self.errors and all(r in {p.identity for p in self.processes} for r in self.roots)


@dataclass(frozen=True)
class WorkerPeak:
    rss_bytes: int
    private_bytes: int
    samples: int


def subtree_from_snapshot(snapshot: ProcessTreeMemory, identity: ProcessIdentity) -> ProcessTreeMemory:
    """Project an authenticated worker subtree from the same fresh OS sample.

    Reuses measured process objects, including descendants and their creation
    tokens. It neither resamples the system nor turns parent totals into a
    worker estimate. Incomplete samples and reused/missing root PIDs fail.
    """
    if not snapshot.complete:
        raise MemoryError('Complete owned snapshot required for a worker subtree')
    by_pid = {process.identity.pid: process for process in snapshot.processes}
    root = by_pid.get(identity.pid)
    if root is None or root.identity != identity:
        raise MemoryError('Worker PID identity missing or changed in owned snapshot')
    selected = {identity.pid}
    while True:
        children = {process.identity.pid for process in snapshot.processes
                    if process.parent_pid in selected}
        added = children - selected
        if not added:
            break
        selected.update(added)
    return ProcessTreeMemory((identity,), tuple(process for process in snapshot.processes
        if process.identity.pid in selected), (), snapshot.sampled_at)


class _MemoryStatus(ct.Structure):
    _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + [
        (n, ct.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys',
        'ullTotalPageFile', 'ullAvailPageFile', 'ullTotalVirtual',
        'ullAvailVirtual', 'ullAvailExtendedVirtual')]


class _PerformanceInfo(ct.Structure):
    _fields_ = [('cb', wt.DWORD)] + [(n, ct.c_size_t) for n in (
        'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
        'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
        'KernelNonpaged', 'PageSize')] + [(n, wt.DWORD) for n in (
        'HandleCount', 'ProcessCount', 'ThreadCount')]


class _MemoryCounters(ct.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD)] + [
        (n, ct.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
        'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


class _ProcessEntry(ct.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD),
        ('th32ProcessID', wt.DWORD), ('th32DefaultHeapID', ct.c_size_t),
        ('th32ModuleID', wt.DWORD), ('cntThreads', wt.DWORD),
        ('th32ParentProcessID', wt.DWORD), ('pcPriClassBase', wt.LONG),
        ('dwFlags', wt.DWORD), ('szExeFile', wt.WCHAR * 260)]


@lru_cache(maxsize=1)
def _windows_api():
    if os.name != 'nt':
        raise RuntimeError('Windows physical/commit counters are required; no guessed fallback')
    kernel = ct.WinDLL('kernel32', use_last_error=True)
    psapi = ct.WinDLL('psapi', use_last_error=True)
    signatures = (
        (kernel.GlobalMemoryStatusEx, [ct.POINTER(_MemoryStatus)], wt.BOOL),
        (psapi.GetPerformanceInfo, [ct.POINTER(_PerformanceInfo), wt.DWORD], wt.BOOL),
        (kernel.OpenProcess, [wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
        (kernel.CloseHandle, [wt.HANDLE], wt.BOOL),
        (kernel.GetProcessTimes, [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4, wt.BOOL),
        (psapi.GetProcessMemoryInfo, [wt.HANDLE, ct.POINTER(_MemoryCounters), wt.DWORD], wt.BOOL),
        (kernel.CreateToolhelp32Snapshot, [wt.DWORD, wt.DWORD], wt.HANDLE),
        (kernel.Process32FirstW, [wt.HANDLE, ct.POINTER(_ProcessEntry)], wt.BOOL),
        (kernel.Process32NextW, [wt.HANDLE, ct.POINTER(_ProcessEntry)], wt.BOOL),
    )
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result
    return kernel, psapi


def read_system_memory() -> SystemMemory:
    """Read available physical RAM and commit, not virtual address space.

    GetPerformanceInfo supplies system commit pages; GlobalMemoryStatusEx
    also clamps the calling process's available commit. No paging-file growth
    is presumed. See Microsoft's MEMORYSTATUSEX/PERFORMANCE_INFORMATION docs.
    """
    kernel, psapi = _windows_api()
    status = _MemoryStatus(); status.dwLength = ct.sizeof(status)
    performance = _PerformanceInfo(); performance.cb = ct.sizeof(performance)
    if not kernel.GlobalMemoryStatusEx(ct.byref(status)):
        raise ct.WinError(ct.get_last_error())
    if not psapi.GetPerformanceInfo(ct.byref(performance), ct.sizeof(performance)):
        raise ct.WinError(ct.get_last_error())
    available = max(0, (performance.CommitLimit - performance.CommitTotal) * performance.PageSize)
    return SystemMemory(int(status.ullTotalPhys), int(status.ullAvailPhys),
        int(performance.CommitLimit * performance.PageSize),
        min(int(status.ullAvailPageFile), int(available)), time.time(),
        'GlobalMemoryStatusEx+GetPerformanceInfo')


def _open_process(pid):
    kernel, _ = _windows_api()
    handle = kernel.OpenProcess(0x1000, False, int(pid))  # query limited information
    if not handle:
        raise ct.WinError(ct.get_last_error())
    return handle


def _identity_from_handle(pid, handle):
    kernel, _ = _windows_api()
    creation, end, cpu_kernel, cpu_user = (wt.FILETIME() for _ in range(4))
    if not kernel.GetProcessTimes(handle, *[ct.byref(t) for t in (creation, end, cpu_kernel, cpu_user)]):
        raise ct.WinError(ct.get_last_error())
    return ProcessIdentity(int(pid), (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime))


def process_identity(pid: int | None = None) -> ProcessIdentity:
    """Pin a caller-owned PID to its creation time to reject PID reuse."""
    pid = os.getpid() if pid is None else int(pid)
    handle = _open_process(pid)
    try:
        return _identity_from_handle(pid, handle)
    finally:
        _windows_api()[0].CloseHandle(handle)


def _process_table():
    if _psutil is not None:
        return {p.info['pid']: (p.info['ppid'], p.info['name'] or '')
            for p in _psutil.process_iter(['pid', 'ppid', 'name'])}
    kernel, _ = _windows_api()
    handle = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if handle == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    entry = _ProcessEntry(); entry.dwSize = ct.sizeof(entry)
    table = {}
    try:
        valid = kernel.Process32FirstW(handle, ct.byref(entry))
        if not valid:
            raise ct.WinError(ct.get_last_error())
        while valid:
            table[int(entry.th32ProcessID)] = (int(entry.th32ParentProcessID), entry.szExeFile)
            valid = kernel.Process32NextW(handle, ct.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return table


def _read_process(pid, parent_pid, name):
    kernel, psapi = _windows_api()
    handle = _open_process(pid)
    try:
        identity = _identity_from_handle(pid, handle)
        if _psutil is not None:
            info = _psutil.Process(pid).memory_info()
            result = ProcessMemory(identity, parent_pid, name, int(info.rss),
                int(info.private), max(int(info.rss), int(info.peak_wset)),
                max(int(info.private), int(info.peak_pagefile)), 'psutil')
            # The separately opened handle must still identify this PID.
            if process_identity(pid) != identity:
                raise RuntimeError('Process exited or PID was reused during sampling')
            return result
        info = _MemoryCounters(); info.cb = ct.sizeof(info)
        if not psapi.GetProcessMemoryInfo(handle, ct.byref(info), ct.sizeof(info)):
            raise ct.WinError(ct.get_last_error())
        return ProcessMemory(identity, parent_pid, name, int(info.WorkingSetSize),
            int(info.PrivateUsage), int(info.PeakWorkingSetSize),
            max(int(info.PrivateUsage), int(info.PeakPagefileUsage)), 'WinAPI')
    finally:
        kernel.CloseHandle(handle)


def sample_owned_tree(roots: Iterable[ProcessIdentity]) -> ProcessTreeMemory:
    """Measure only explicit owned roots and their live descendants.

    Root creation tokens are checked before traversing. Descendants created
    before their root are excluded, guarding against a recycled parent PID.
    Disappearing/inaccessible children are reported; incomplete samples must
    not be used to admit more workers. Nothing is terminated by this helper.
    """
    roots = tuple(dict.fromkeys(roots))
    if not roots or any(not isinstance(r, ProcessIdentity) for r in roots):
        raise ValueError('At least one explicit owned ProcessIdentity is required')
    table = _process_table(); selected = {}; errors = []
    for root in roots:
        try:
            parent, name = table[root.pid]
            process = _read_process(root.pid, parent, name)
            if process.identity != root:
                raise RuntimeError('Owned root PID reused')
            selected[root.pid] = process
        except (OSError, RuntimeError, KeyError) as error:
            errors.append(f'{root.pid}: {error}')
    pending = list(selected)
    while pending:
        parent = pending.pop()
        for pid, (ppid, name) in table.items():
            if ppid != parent or pid in selected:
                continue
            try:
                process = _read_process(pid, ppid, name)
                if process.identity.creation_token < selected[parent].identity.creation_token:
                    continue
                selected[pid] = process; pending.append(pid)
            except (OSError, RuntimeError) as error:
                errors.append(f'{pid}: {error}')
    return ProcessTreeMemory(roots, tuple(selected[p] for p in sorted(selected)), tuple(errors), time.time())


def measured_worker_peak(samples: Iterable[ProcessTreeMemory]) -> WorkerPeak | None:
    """Conservative measured peak of ONE representative worker tree.

    Sample during startup and its largest representative normal evaluation.
    A bootstrap-only measurement does not certify the evaluation's peak.
    Never pass the whole multi-worker driver tree as a per-worker estimate.
    """
    samples = tuple(samples)
    if not samples or any(not s.complete for s in samples):
        return None
    return WorkerPeak(max(sum(p.peak_rss_bytes for p in s.processes) for s in samples),
        max(sum(p.peak_private_bytes for p in s.processes) for s in samples), len(samples))


class OwnedMemoryMonitor:
    """Bounded latest sample and cumulative measured peaks, no polling thread.

    Call sample between allocations/at solver safe points and serialize its
    dataclasses with dataclasses.asdict to a caller-owned Data log.
    """
    def __init__(self, roots: Iterable[ProcessIdentity]):
        self.roots = tuple(roots)
        self.latest: ProcessTreeMemory | None = None
        self.samples = 0
        self.peak_rss_bytes = 0
        self.peak_private_bytes = 0

    def sample(self) -> ProcessTreeMemory:
        self.latest = sample_owned_tree(self.roots)
        self.samples += 1
        self.peak_rss_bytes = max(self.peak_rss_bytes,
            sum(p.peak_rss_bytes for p in self.latest.processes))
        self.peak_private_bytes = max(self.peak_private_bytes,
            sum(p.peak_private_bytes for p in self.latest.processes))
        return self.latest


def _bytes(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer byte count')
    return value


def memory_pressure(system: SystemMemory, *, reserve_physical_bytes: int,
        reserve_commit_bytes: int, inflight_physical_bytes: int = 0,
        inflight_commit_bytes: int = 0) -> bool:
    """True means stop submitting and release OWNED idle workers safely."""
    physical = _bytes(reserve_physical_bytes, 'reserve_physical_bytes') + _bytes(inflight_physical_bytes, 'inflight_physical_bytes')
    commit = _bytes(reserve_commit_bytes, 'reserve_commit_bytes') + _bytes(inflight_commit_bytes, 'inflight_commit_bytes')
    return system.physical_available_bytes <= physical or system.commit_available_bytes <= commit


def choose_worker_count(requested_workers: int, measured_peak: WorkerPeak | None, *,
        reserve_physical_bytes: int, reserve_commit_bytes: int,
        system: SystemMemory | None = None, peak_margin: float = 1.25,
        inflight_physical_bytes: int = 0, inflight_commit_bytes: int = 0) -> int:
    """Number of ADDITIONAL workers affordable NOW; may return zero.

    Existing workers already reduce available system counters; do not subtract
    their RSS again. Inflight bytes reserve shared/parent allocations and any
    workers admitted but not fully allocated yet. All pools must share this
    admission point and remeasure before each spawn. No peak evidence => zero.
    The configurable 1.25 margin is an engineering allowance, not a user rule.
    """
    requested = _bytes(requested_workers, 'requested_workers')
    reserve_phys = _bytes(reserve_physical_bytes, 'reserve_physical_bytes')
    reserve_commit = _bytes(reserve_commit_bytes, 'reserve_commit_bytes')
    inflight_phys = _bytes(inflight_physical_bytes, 'inflight_physical_bytes')
    inflight_commit = _bytes(inflight_commit_bytes, 'inflight_commit_bytes')
    if not math.isfinite(peak_margin) or peak_margin < 1:
        raise ValueError('peak_margin must be finite and at least 1')
    if not requested or measured_peak is None:
        return 0
    if measured_peak.samples < 1 or measured_peak.rss_bytes <= 0 or measured_peak.private_bytes <= 0:
        return 0
    system = read_system_memory() if system is None else system
    rss = math.ceil(measured_peak.rss_bytes * peak_margin)
    private = math.ceil(measured_peak.private_bytes * peak_margin)
    physical_room = max(0, system.physical_available_bytes - reserve_phys - inflight_phys)
    commit_room = max(0, system.commit_available_bytes - reserve_commit - inflight_commit)
    return int(min(requested, physical_room // rss, commit_room // private))
