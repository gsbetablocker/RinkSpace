"""Build the implemented CPU native-core slice with an existing toolchain.

No downloads, package installs, CUDA claim or optimizer launch. All build
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
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    if output == project or project in output.parents:
        parser.error("build output must be outside the source/OneDrive project")
    output.mkdir(parents=True, exist_ok=True)
    log = output/"build.log"
    print(f"Native CPU build log: {log}", flush=True)
    cmake = args.cmake or shutil.which("cmake")
    commands = []
    if cmake:
        commands = [[cmake, "-S", str(project/"native"), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"],
                    [cmake, "--build", str(output), "--config", "Release"]]
        expected = "rinkspace_core.dll" if os.name == "nt" else "librinkspace_core.so"
    elif os.name != "nt" and sys.platform != "darwin" and shutil.which("g++"):
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
              "cuda_built": False, "nonlinear_solver_implemented": False}
    (output/"build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
