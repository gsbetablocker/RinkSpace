"""Shared snapshot reads and atomic publication across Windows readers.

No model, ledger, numerical objective, or source admission is defined here.
"""
from __future__ import annotations
import os
import time
from pathlib import Path


def read_text_shared(path):
    """Keep an atomic Windows snapshot readable while its name is replaced.

    The default CRT reader denies FILE_SHARE_DELETE, so a monitoring reader can
    otherwise make os.replace fail in a numerical iteration callback.
    """
    if os.name != "nt":
        return Path(path).read_text(encoding="utf-8-sig")
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.CreateFileW(str(path), 0x80000000, 1 | 2 | 4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        stream = os.fdopen(descriptor, "r", encoding="utf-8-sig")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        return stream.read()


def _replace_once(source, target):
    if os.name != "nt":
        os.replace(source, target)
        return
    # ReplaceFileW permits a shared-delete reader to keep its old snapshot.
    # MoveFileExW (used by os.replace) rejects that case on this filesystem.
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.ReplaceFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                   ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p]
    kernel.ReplaceFileW.restype = ctypes.c_int
    if kernel.ReplaceFileW(str(target), str(source), None, 0, None, None):
        return
    error = ctypes.get_last_error()
    if error == 2 and not Path(target).exists():
        os.replace(source, target)
        return
    raise ctypes.WinError(error)


def atomic_replace(source, target):
    """Retry only transient Windows sharing failures; never rewrite in place."""
    delays = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
    for attempt in range(len(delays) + 1):
        try:
            _replace_once(source, target)
            return
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


