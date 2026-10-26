"""Build the implemented native-core slice with an existing toolchain.

No downloads, package installs or optimizer launch. CUDA G1/G2 is opt-in. All build
outputs and logs must be outside the source tree (Data on the user's machine).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cmake", help="Path to an already installed CMake")
    parser.add_argument("--cuda", action="store_true", help="Build the optional G1/G2 CUDA evaluator and linearization")
    parser.add_argument("--cudss", type=Path, help="Existing project-owned cuDSS root; enables one native LM candidate")
    parser.add_argument("--generator", help="Existing CMake generator, e.g. Ninja")
    parser.add_argument("--cuda-architectures", default="native", help="CMake CUDA architectures; default builds for the installed GPU")
    parser.add_argument("--cuda-fast-compile", choices=["0","min","mid","max"], default="0", help="Explicit CUDA compile-time optimizer tradeoff; requires compiler support")
    args = parser.parse_args()
    if args.cudss and not args.cuda: parser.error("--cudss requires --cuda")
    project = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    if output == project or project in output.parents:
        parser.error("build output must be outside the source/OneDrive project")
    output.mkdir(parents=True, exist_ok=True)
    log = output/"build.log"
    print(f"Native build log: {log}", flush=True)
    cmake = args.cmake or shutil.which("cmake")
    commands = []
    if cmake:
        commands = [[cmake, "-S", str(project/"native"), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"],
                    [cmake, "--build", str(output), "--config", "Release"]]
        commands[0].append("-DRINKSPACE_ENABLE_CUDA="+("ON" if args.cuda else "OFF"))
        commands[0].append("-DRINKSPACE_CUDSS_ROOT="+(str(args.cudss.resolve(strict=True)) if args.cudss else ""))
        if args.cuda:commands[0].extend(["-DCMAKE_CUDA_ARCHITECTURES="+args.cuda_architectures,"-DRINKSPACE_CUDA_FAST_COMPILE="+args.cuda_fast_compile])
        if args.generator:
            commands[0].extend(["-G", args.generator])
        expected = ("rinkspace_core.dll" if os.name == "nt" else
                    "librinkspace_core.dylib" if sys.platform == "darwin" else
                    "librinkspace_core.so")
    elif not args.cuda and os.name != "nt" and sys.platform != "darwin" and shutil.which("g++"):
        expected = "librinkspace_core.so"
        commands = [[shutil.which("g++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-fPIC", "-shared",
                     "-DRINKSPACE_CORE_BUILD", "-I"+str(project/"native/include"), str(project/"native/src/core_api.cpp"), "-o", str(output/expected)]]
    else:
        print("No supported existing toolchain found. Pass --cmake with your existing Ceres/CUDA environment's CMake path; nothing was installed.", file=sys.stderr)
        return 2
    with log.open("w", encoding="utf-8") as stream:
        for command in commands:
            stream.write(json.dumps(command, ensure_ascii=False)+"\n")
            stream.flush()
            process = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
            if process.returncode:
                print(f"Native build failed ({process.returncode}); see {log}", file=sys.stderr)
                return process.returncode
    preferred = [output/"Release"/expected, output/expected]
    candidates = next(([path] for path in preferred if path.is_file()), None)
    if candidates is None:
        candidates = list(output.rglob(expected))
    if len(candidates) != 1:
        print(f"Expected exactly one {expected}; found {len(candidates)}. Check {log}", file=sys.stderr)
        return 2
    result = {"library": str(candidates[0].resolve()), "log": str(log), "cpu_core_built": True,
              "cuda_built": args.cuda, "cuda_architectures": args.cuda_architectures if args.cuda else None, "cuda_fast_compile": args.cuda_fast_compile if args.cuda else None, "lm_candidate_implemented": bool(args.cudss), "nonlinear_solver_implemented": bool(args.cudss),
              "native_geometry_plan_implemented": bool(args.cuda), "persistent_resume_implemented": False}
    (output/"build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
