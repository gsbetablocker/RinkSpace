"""Compatibility bootstrap for existing source-worker and pickle identities."""
from calibration.solver_runtime.config import PROGRAM_SOURCE_ROOT as SNAPSHOT, PROJECT_ROOT as PROJECT, data_root

BASE = data_root()


def install():
    from calibration.solver_runtime import install as install_owned_runtime
    base = install_owned_runtime()
    import z190_secondary_qualified_core_source as source
    source.TEMP_ROOT = base
    return base
