"""Explicit strict-double build of the source-addressed human support kernel."""
from pathlib import Path
import hashlib
import json
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[2]
    source = root/'calibration/solver_runtime/legacy/structured_programme_source_candidate/native/normal_scanline_human_query.c'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    directory = Path('D:/CodexData/RinkSpace_Data/native/solver_runtime/normal_scanline_human_query')/sha(source)
    original = json.loads(Path('D:/CodexData/RinkSpace_Data/native/solver_runtime/normal_scanline_human_query/817ca0b1dfe4bb2b009a8567a6498419de7421d7943d46e4e856df016f943019/build.json').read_text(encoding='utf8'))
    directory.mkdir(parents=True, exist_ok=True)
    dll = directory/'normal_scanline_human_query.dll'
    obj = directory/'normal_scanline_human_query.obj'
    flags = original['flags']
    assert flags == ['/nologo', '/O2', '/fp:strict', '/openmp', '/LD', '/TC']
    command = '@echo off\ncall "'+original['compiler_setup']+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n'
    command += 'cl.exe '+' '.join(flags)+' "'+str(source)+'" /Fo:"'+str(obj)+'" /Fe:"'+str(dll)+'" /link /OUT:"'+str(dll)+'"\nexit /b %errorlevel%\n'
    batch = directory/'build.cmd'
    batch.write_text(command, encoding='utf8')
    started = time.perf_counter()
    result = subprocess.run(['cmd.exe', '/d', '/c', str(batch)], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    log = directory/'build.log'
    log.write_text(result.stdout, encoding='utf8')
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    manifest = dict(schema='rinkspace.original-finite-source-rectangle-strict-native-build.v1',
        source=str(source), source_sha256=sha(source), binary=dll.name, dll_sha256=sha(dll),
        flags=flags, compiler_setup=original['compiler_setup'], build_log=str(log),
        build_log_sha256=sha(log), original_normal_scanline_export_retained=True,
        original_support_rectangle_not_expanded=True, seconds=time.perf_counter()-started)
    (directory/'build.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
