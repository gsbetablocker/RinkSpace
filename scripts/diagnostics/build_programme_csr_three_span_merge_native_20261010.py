"""Build the isolated CSR-copy kernel; never build during solver execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.solver_runtime.programme_csr_three_span_merge_runtime import SOURCE, native_build_directory


def main():
    started = time.perf_counter()
    directory = native_build_directory()
    directory.mkdir(parents=True, exist_ok=True)
    # Reuse the existing reviewed strict-double MSVC/OpenMP toolchain.
    existing = Path(r'D:\CodexData\RinkSpace_Data\events\ankara_2026\calibration\runs\solver_acceleration_20261003\native_scatter\build.json')
    original = json.loads(existing.read_text(encoding='utf8'))
    setup = original['compiler_setup']
    dll, obj = directory/'programme_csr_three_span_merge.dll', directory/'programme_csr_three_span_merge.obj'
    flags = original['flags']
    command = ('@echo off\ncall "'+setup+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n'
        +'cl.exe '+ ' '.join(flags)+' "'+str(SOURCE)+'" /Fo:"'+str(obj)
        +'" /Fe:"'+str(dll)+'" /link /OUT:"'+str(dll)+'"\nexit /b %errorlevel%\n')
    batch = directory/'build.cmd'
    batch.write_text(command, encoding='utf8')
    result = subprocess.run(['cmd.exe', '/d', '/c', str(batch)], cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = directory/'build.log'
    log.write_text(result.stdout, encoding='utf8')
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    manifest = dict(schema='rinkspace.original-programme-CSR-copy-native-build.v1',
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        binary=dll.name, dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
        flags=flags, compiler_setup=setup, build_log=str(log),
        build_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        only_original_numeric_double_bytes_copied=True, seconds=time.perf_counter()-started)
    (directory/'build.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
