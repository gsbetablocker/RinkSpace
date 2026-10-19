"""Light project-owned targets for Windows package-CLI spawned workers.

Multiprocessing skips a package's ``__main__`` when preparing a spawned
child.  Import this qualified module before unpickling any legacy source
target so the same owned finder and native contract exist in both processes.
"""
from . import install

install()
from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract

install_source_worker_backend_contract()


def initialize_pattern_worker(task_root):
    from structured_programme_source_candidate.current_geometry_pattern_pool import _worker_initialize
    return _worker_initialize(task_root)


def evaluate_pattern_worker(*args):
    from structured_programme_source_candidate.current_geometry_pattern_pool import _worker_evaluate
    return _worker_evaluate(*args)


def run_source_actor(*args):
    import z190_process_source_evaluation as source
    return source._actor_worker(*args)
