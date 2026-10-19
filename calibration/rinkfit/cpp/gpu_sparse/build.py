"""Build the opt-in cuSPARSE wrapper with an existing Windows CUDA/MSVC install."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--vcvars", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, default=Path("C:/Program Files (x86)/Windows Kits/10"))
    args = parser.parse_args()
    source = Path(__file__).with_name("gpu_sparse.cpp").resolve()
    project = source.parents[4]
    output = args.out_dir.resolve()
    if output.is_relative_to(project):
        parser.error("Build products must stay outside the OneDrive project")
    cuda = args.cuda_root.resolve()
    vcvars = args.vcvars.resolve()
    for path in (source, vcvars, cuda / "include/cusparse.h", cuda / "lib/x64/cusparse.lib", cuda / "lib/x64/cudart.lib"):
        if not path.is_file():
            parser.error(f"Missing existing build dependency: {path}")
    output.mkdir(parents=True, exist_ok=True)
    dll = output / "rinkfit_gpu_sparse.dll"
    flags = ["/nologo", "/O2", "/fp:strict", "/EHsc", "/std:c++17", "/MD", "/LD", "/W4", "/utf-8"]
    # Direct compiler invocation avoids shell expansion and a vcvars subprocess.
    msvc_versions = sorted((vcvars.parents[2] / "Tools/MSVC").iterdir())
    sdk_versions = sorted((args.sdk_root / "Include").iterdir())
    if len(msvc_versions) != 1 or len(sdk_versions) != 1:
        parser.error("Ambiguous installed compiler/SDK version; select one installation")
    msvc = msvc_versions[0]
    sdk = sdk_versions[0]
    cl = msvc / "bin/Hostx64/x64/cl.exe"
    includes = [msvc / "include"] + [sdk / name for name in ("ucrt", "shared", "um", "winrt", "cppwinrt")]
    libs = [msvc / "lib/x64"] + [args.sdk_root / "Lib" / sdk.name / name / "x64" for name in ("ucrt", "um")]
    for path in [cl, *includes, *libs]:
        if not path.exists():
            parser.error(f"Missing existing compiler/SDK path: {path}")
    command = [str(cl), *flags, "/I" + str(cuda / "include"), str(source), "/Fo" + str(output / "gpu_sparse.obj"), "/Fe" + str(dll), "/link", "/LIBPATH:" + str(cuda / "lib/x64"), "cudart.lib", "cusparse.lib", "/IMPLIB:" + str(output / "rinkfit_gpu_sparse.lib")]
    env = {key.upper(): value for key, value in os.environ.items()}
    env["PATH"] = str(cl.parent) + ";" + env.get("PATH", "")
    env["INCLUDE"] = ";".join(map(str, includes))
    env["LIB"] = ";".join(map(str, libs))
    start = time.perf_counter()
    result = subprocess.run(command, cwd=output, env=env, capture_output=True, text=True, errors="replace")
    (output / "build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    report = {"schema": "rinkfit.gpu_sparse_build.v1", "source": str(source), "source_sha256": sha(source), "python_source_sha256": sha(project / "calibration/rinkfit/src/rinkfit_start/gpu_sparse.py"), "cuda_root": str(cuda), "cuda_version": json.loads((cuda / "version.json").read_text()), "vcvars": str(vcvars), "flags": flags, "command": command, "seconds": time.perf_counter()-start, "returncode": result.returncode, "dll": str(dll), "dll_sha256": sha(dll) if result.returncode == 0 else None, "no_install": True, "no_solver": True}
    (output / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(result.stdout + result.stderr)
    print(json.dumps(report, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
