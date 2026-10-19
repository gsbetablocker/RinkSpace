"""Ordered Windows-spawn demo rendering with shared-memory BGR frame slots.

Only small (slot,PTS) tasks and status dictionaries are pickled. Each spawned
worker imports its own Sample and advances through strictly increasing PTS.
Source decoding and encoded output stay in the parent process.

A yielded BGR view is valid only until iteration resumes. Consume/write it
before requesting the next frame, or copy it to retain it. Close the generator
on early cancellation (contextlib.closing is convenient). Source iterators are
closed on completion/cancellation. The Sample must produce deterministic frames
for any increasing subset of indices; v2 uses heat_update_anchor=0 for this.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import cv2
import numpy as np

NATIVE_SHAPE = (1080,1920,3)
DEFAULT_FACTORY = 'calibration.program_tracking.render_demo_last_shot:Sample'
_WORKER_SAMPLE = None
_WORKER_MEMORY = None
_WORKER_NATIVE = None
_WORKER_OUTPUT = None
_WORKER_LAST_PTS = None


def _resolve_factory(factory_path):
    module_name, separator, attribute = str(factory_path).partition(':')
    if not separator or not module_name or not attribute:
        raise ValueError('sample_factory must have module:attribute syntax')
    value = importlib.import_module(module_name)
    for component in attribute.split('.'):
        value = getattr(value,component)
    if not callable(value):
        raise TypeError('sample_factory must identify a callable')
    return value


def _check_small_kwargs(value):
    if isinstance(value,np.ndarray):
        raise TypeError('Pass model paths/options in sample_kwargs, not pixel or model arrays')
    if isinstance(value,dict):
        for item in value.values():
            _check_small_kwargs(item)
    elif isinstance(value,(list,tuple)):
        for item in value:
            _check_small_kwargs(item)


def _initialize_worker(factory_path,sample_kwargs,native_name,output_name,slots,cv_threads,ready_barrier):
    global _WORKER_SAMPLE,_WORKER_MEMORY,_WORKER_NATIVE,_WORKER_OUTPUT,_WORKER_LAST_PTS
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    cv2.setNumThreads(int(cv_threads))
    _WORKER_MEMORY = (shared_memory.SharedMemory(name=native_name),
                      shared_memory.SharedMemory(name=output_name))
    _WORKER_NATIVE = np.ndarray((slots,*NATIVE_SHAPE),np.uint8,buffer=_WORKER_MEMORY[0].buf)
    _WORKER_OUTPUT = np.ndarray((slots,*NATIVE_SHAPE),np.uint8,buffer=_WORKER_MEMORY[1].buf)
    _WORKER_SAMPLE = _resolve_factory(factory_path)(**sample_kwargs)
    _WORKER_LAST_PTS = None
    # Load both complete Samples before the first heavy frame is dispatched.
    # Otherwise one quickly initialized worker can consume a short sample alone.
    ready_barrier.wait()


def _worker_ready():
    return os.getpid()


def _draw_chunk(items):
    global _WORKER_LAST_PTS
    start = time.perf_counter()
    for slot,pts in items:
        if _WORKER_LAST_PTS is not None and pts<=_WORKER_LAST_PTS:
            raise ValueError('A rendering worker received non-increasing source PTS')
        result = _WORKER_SAMPLE.draw(_WORKER_NATIVE[slot],int(pts))
        frame = result[0] if isinstance(result,tuple) else result
        if not isinstance(frame,np.ndarray) or frame.shape!=NATIVE_SHAPE or frame.dtype!=np.uint8:
            raise ValueError('Sample.draw must return uint8 BGR1920x1080 or a tuple starting with it')
        np.copyto(_WORKER_OUTPUT[slot],frame)
        _WORKER_LAST_PTS = int(pts)
    return {'pts':[int(pts) for _,pts in items], 'pid':os.getpid(),
            'draw_seconds':time.perf_counter()-start,
            'mask_counts':dict(getattr(_WORKER_SAMPLE,'mask_counts',{}))}


def ordered_frames(frames,sample_kwargs=None,workers=2,slots=6,*,
                   sample_factory=DEFAULT_FACTORY,chunk_size=1,cv_threads=1,stats=None):
    """Yield (native_pts,rendered_bgr_view) in input order, with bounded memory.

    frames: increasing (PTS,uint8 BGR1920x1080) iterator, decoded in the parent.
    sample_kwargs: small constructor arguments, e.g. {'version':'v2'}.
    workers/slots: defaults use two draw processes and six input/output pairs.
    chunk_size: one or two consecutive frames per small task work well here.
    stats: optional caller-owned dictionary receives timings/counts/mask totals.

    Exceptions propagate with their worker traceback. Finally shuts down only
    this pool and releases/unlinks its own two named shared-memory allocations.
    """
    workers,slots,chunk_size,cv_threads = map(int,(workers,slots,chunk_size,cv_threads))
    if workers<1 or slots<workers or chunk_size<1 or chunk_size>slots or cv_threads<1:
        raise ValueError('Positive workers/chunk/threads and slots>=workers are required')
    kwargs = dict(sample_kwargs or {})
    _check_small_kwargs(kwargs)
    # Validate a malformed import path before allocating the ring; actual import
    # stays in workers so the parent need not load another Sample/model package.
    if ':' not in str(sample_factory):
        raise ValueError('sample_factory must have module:attribute syntax')
    status = stats if stats is not None else {}
    frame_bytes = int(np.prod(NATIVE_SHAPE))
    status.update(mode='ordered_process_pool_shared_memory',workers=workers,
                  slots=slots,chunk_size=chunk_size,cv_threads_per_worker=cv_threads,
                  source_pixel_arrays_pickled=False,frames=0,worker_draw_seconds=0.,
                  shared_memory_bytes=2*slots*frame_bytes,
                  worker_frame_counts={},worker_mask_counts={})
    native_memory=output_memory=pool=native=output=None
    source=iter(frames)
    started=time.perf_counter()
    finished=False
    try:
        native_memory=shared_memory.SharedMemory(create=True,size=slots*frame_bytes)
        output_memory=shared_memory.SharedMemory(create=True,size=slots*frame_bytes)
        native=np.ndarray((slots,*NATIVE_SHAPE),np.uint8,buffer=native_memory.buf)
        output=np.ndarray((slots,*NATIVE_SHAPE),np.uint8,buffer=output_memory.buf)
        context=mp.get_context('spawn')
        ready_barrier=context.Barrier(workers)
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,
            initializer=_initialize_worker,initargs=(str(sample_factory),kwargs,
                native_memory.name,output_memory.name,slots,cv_threads,ready_barrier))
        warmups=[pool.submit(_worker_ready) for _ in range(workers)]
        for future in warmups:
            future.result()
        status['worker_initialization_seconds']=time.perf_counter()-started
        free=deque(range(slots))
        pending=deque()
        exhausted=False
        previous_submitted=previous_yielded=None

        def fill():
            nonlocal exhausted,previous_submitted
            while free and not exhausted:
                items=[]
                for _ in range(min(chunk_size,len(free))):
                    try:
                        pts,image=next(source)
                    except StopIteration:
                        exhausted=True
                        break
                    pts=int(pts)
                    if previous_submitted is not None and pts<=previous_submitted:
                        raise ValueError('Source frames must have strictly increasing native PTS')
                    if not isinstance(image,np.ndarray) or image.shape!=NATIVE_SHAPE or image.dtype!=np.uint8:
                        raise ValueError('Input requires uint8 BGR1920x1080')
                    slot=free.popleft()
                    np.copyto(native[slot],image)
                    items.append((slot,pts))
                    previous_submitted=pts
                if items:
                    pending.append((items,pool.submit(_draw_chunk,items)))
                else:
                    break

        fill()
        while pending:
            items,future=pending.popleft()
            result=future.result()
            if result['pts']!=[pts for _,pts in items]:
                raise ValueError('Worker output PTS differ from submitted native source frames')
            key=str(result['pid'])
            status['worker_draw_seconds']+=float(result['draw_seconds'])
            status['worker_frame_counts'][key]=status['worker_frame_counts'].get(key,0)+len(items)
            status['worker_mask_counts'][key]=result['mask_counts']
            for slot,pts in items:
                if previous_yielded is not None and pts<=previous_yielded:
                    raise ValueError('Ordered rendering output lost native source order')
                previous_yielded=pts
                status['frames']+=1
                # Slot stays owned by this frame throughout the caller's write.
                yield pts,output[slot]
                free.append(slot)
            fill()
        finished=True
    finally:
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=not finished)
        native=output=None
        for memory in (native_memory,output_memory):
            if memory is not None:
                try:
                    memory.close()
                finally:
                    try:
                        memory.unlink()
                    except FileNotFoundError:
                        pass
        close=getattr(source,'close',None)
        if close is not None:
            close()
        status['completed']=finished
        status['pipeline_seconds_including_decode_and_sink']=time.perf_counter()-started
        status['mask_counts']={}
        for counts in status['worker_mask_counts'].values():
            for label,count in counts.items():
                status['mask_counts'][label]=status['mask_counts'].get(label,0)+int(count)
        status['mask_states']=status['mask_counts'].copy()


def benchmark_frames(frames,sample_kwargs=None,*,sample_factory=DEFAULT_FACTORY,
                     workers=2,slots=6,chunk_size=1):
    """Compare a few buffered real-source frames to serial pixel hashes.

    Does not write images or video. Intended for 3-6 frames, so startup overhead
    is reported explicitly and is not mistaken for steady-state throughput.
    """
    frames=list(frames)
    if not frames:
        raise ValueError('benchmark needs at least one frame')
    serial=_resolve_factory(sample_factory)(**dict(sample_kwargs or {}))
    expected=[]
    started=time.perf_counter()
    for pts,image in frames:
        result=serial.draw(image.copy(),int(pts))
        rendered=result[0] if isinstance(result,tuple) else result
        expected.append((int(pts),hashlib.sha256(rendered).hexdigest()))
    serial_seconds=time.perf_counter()-started
    actual=[]
    stats={}
    started=time.perf_counter()
    for pts,image in ordered_frames(frames,sample_kwargs,workers,slots,
            sample_factory=sample_factory,chunk_size=chunk_size,stats=stats):
        actual.append((int(pts),hashlib.sha256(image).hexdigest()))
    parallel_seconds=time.perf_counter()-started
    if actual!=expected:
        mismatch=[a[0] for a,b in zip(actual,expected) if a!=b]
        raise AssertionError('Shared/serial frame hash mismatch at PTS '+repr(mismatch))
    return {'frames':len(frames),'first_pts':int(frames[0][0]),'last_pts':int(frames[-1][0]),
            'exact_pts_order':True,'exact_pixel_hashes':True,
            'serial_draw_and_hash_seconds':serial_seconds,
            'parallel_seconds_including_spawn_and_shutdown':parallel_seconds,
            'no_images_or_video_written':True,'pipeline':stats}
