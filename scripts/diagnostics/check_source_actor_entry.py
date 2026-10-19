"""Exercise the actual Windows actor entry with an empty transport-only block."""
from pathlib import Path
from multiprocessing import get_context, shared_memory
from types import SimpleNamespace
import hashlib
import pickle
import sys
import uuid


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.parallel_source_guard import install_parallel_source_guard
    install_parallel_source_guard()
    from calibration.solver_runtime.worker_entry import run_source_actor
    from calibration.solver_runtime.source_actor_payload_runtime import run_file_source_actor
    root = Path(sys.argv[1]).resolve()
    path = root / (uuid.uuid4().hex + '.pickle')
    payload = pickle.dumps(SimpleNamespace(name='owned_transport_entry_check', bundle=SimpleNamespace()), protocol=5)
    path.write_bytes(payload)
    memory = shared_memory.SharedMemory(create=True, size=8)
    parent, child = get_context('spawn').Pipe()
    descriptor = dict(path=str(path), owned_transport_root=str(root), owned_name=path.name,
                      bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), original_actor_entry=run_source_actor)
    process = get_context('spawn').Process(target=run_file_source_actor,
        args=(child, descriptor, str(Path.cwd()), 1, {}, dict(name=memory.name, arrays={})))
    try:
        process.start()
        child.close()
        ready = parent.recv()
        if ready.get('status') != 'ready' or ready.get('finite_extent_guard_supported') is not True:
            raise RuntimeError('Original guarded entry not retained: '+str(ready))
        parent.send(dict(mode='close'))
        process.join()
        if process.exitcode != 0:
            raise RuntimeError('Owned actor failed')
        print('PASS: original Windows guarded entry retained; no source graph or derivatives evaluated.', flush=True)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        memory.close()
        memory.unlink()
        path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
