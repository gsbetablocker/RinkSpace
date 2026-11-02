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
import tempfile


class ToolchainError(RuntimeError):
    """An unusable execution environment, before any CMake build is attempted."""


def windows_shell(body: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    # cmd.exe uses its own quoting grammar, not the MSVC argv grammar used by
    # list2cmdline. Keep the /c body intact; supply paths through environment vars.
    comspec = env.get("COMSPEC") or str(Path(env.get("SYSTEMROOT", r"C:\Windows"))/"System32/cmd.exe")
    command = f'"{comspec}" /d /u /s /c "{body}"'
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                          encoding="utf-16-le", errors="replace")


def windows_preflight(project: Path, output: Path, env: dict[str, str]) -> None:
    # Printing cwd, reading files from Node, or nvcc --version does not prove
    # that native compiler children can traverse the same paths (Codex MXC).
    for label, directory in (("project", project), ("build output", output),
                             ("temporary directory", Path(tempfile.gettempdir()))):
        probe_env = {**env, "RINKSPACE_PROBE_DIR": str(directory)}
        try:
            result = windows_shell('cd /d "%RINKSPACE_PROBE_DIR%" && echo RINKSPACE_CWD_OK',
                                   probe_env, output)
        except OSError as exc:
            raise ToolchainError(f"Native cmd.exe could not start for {label}: {exc}") from exc
        if result.returncode or "RINKSPACE_CWD_OK" not in result.stdout:
            raise ToolchainError(
                f"Native child process cannot enter {label}: {directory}\n"
                f"cmd.exe exit={result.returncode}: {result.stderr.strip() or result.stdout.strip()}\n"
                "This is an execution/sandbox access failure, before CUDA compilation. "
                "In Codex, rerun this same build_core.py command through exec_command "
                "with sandbox_permissions=require_escalated (subject to approval); "
                "do not retry via node_repl child_process or change compiler flags.")
    # Test child reads/writes as well as directory traversal. Preserve previous
    # build files, and remove only this invocation's temporary probe.
    with tempfile.TemporaryDirectory(prefix="rinkspace probe ", dir=output) as scratch:
        marker = Path(scratch)/"child.txt"
        probe_env = {**env, "RINKSPACE_PROBE_SOURCE": str(project/"native/CMakeLists.txt"),
                     "RINKSPACE_PROBE_MARKER": str(marker)}
        result = windows_shell('type "%RINKSPACE_PROBE_SOURCE%" >nul && '
                               'echo RINKSPACE_CHILD_OK>"%RINKSPACE_PROBE_MARKER%"', probe_env, output)
        if result.returncode or not marker.is_file():
            raise ToolchainError("Native child cannot read the source or write the build output: "
                                 + result.stderr.strip())
        if "RINKSPACE_CHILD_OK" not in marker.read_text(encoding="utf-16-le"):
            raise ToolchainError("Native child output probe did not write the expected marker")


def find_vcvars(env: dict[str, str], explicit: Path | None = None) -> Path | None:
    if explicit:
        if not explicit.is_file():
            raise ToolchainError(f"vcvars64.bat not found: {explicit}")
        return explicit.resolve()
    if env.get("VSINSTALLDIR"):
        candidate = Path(env["VSINSTALLDIR"])/"VC/Auxiliary/Build/vcvars64.bat"
        if candidate.is_file():
            return candidate
    vswhere = Path(env.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))/"Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath", "-utf8"],
                                capture_output=True, encoding="utf-8", errors="replace", env=env)
        if result.returncode == 0 and result.stdout.strip():
            candidate = Path(result.stdout.strip())/"VC/Auxiliary/Build/vcvars64.bat"
            if candidate.is_file():
                return candidate
    return None


def prepare_toolchain(args: argparse.Namespace, project: Path, output: Path) -> tuple[dict[str, str], dict]:
    env = {key.upper(): value for key, value in os.environ.items()} if os.name == "nt" else dict(os.environ)
    vcvars = None
    if os.name == "nt":
        windows_preflight(project, output, env)
        needs_msvc = args.cuda or args.vcvars or args.generator in ("Ninja", "Ninja Multi-Config", "NMake Makefiles")
        ready = shutil.which("cl.exe", path=env.get("PATH")) and env.get("INCLUDE") and env.get("LIB")
        if args.vcvars or (needs_msvc and not ready):
            vcvars = find_vcvars(env, args.vcvars)
            if vcvars is None:
                raise ToolchainError("MSVC development environment missing. Pass --vcvars <existing vcvars64.bat>; nothing was installed.")
            result = windows_shell('call "%RINKSPACE_VCVARS%" >nul && set',
                                   {**env, "RINKSPACE_VCVARS": str(vcvars)}, output)
            if result.returncode:
                raise ToolchainError(f"vcvars64.bat failed ({result.returncode}): {result.stderr.strip()}")
            for line in result.stdout.splitlines():
                key, separator, value = line.partition("=")
                if separator and key and not key.startswith("="):
                    env[key.upper()] = value
            env.pop("RINKSPACE_VCVARS", None)
            if not shutil.which("cl.exe", path=env.get("PATH")) or not env.get("INCLUDE") or not env.get("LIB"):
                raise ToolchainError("vcvars64.bat did not provide a usable MSVC environment")
    cmake = args.cmake or shutil.which("cmake", path=env.get("PATH"))
    if not cmake and os.name == "nt":
        vcvars = vcvars or find_vcvars(env)
        if vcvars:
            candidate = vcvars.parents[3]/"Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
            if candidate.is_file():
                cmake = str(candidate)
    nvcc = None
    if args.cuda:
        nvcc = env.get("CUDACXX")
        if not nvcc and env.get("CUDA_PATH"):
            candidate = Path(env["CUDA_PATH"])/"bin"/("nvcc.exe" if os.name == "nt" else "nvcc")
            if candidate.is_file():
                nvcc = str(candidate)
        nvcc = nvcc or shutil.which("nvcc", path=env.get("PATH"))
        if not nvcc:
            raise ToolchainError("CUDA compiler missing. Set CUDA_PATH or CUDACXX to the existing toolkit.")
    info = {"cmake": cmake, "vcvars": str(vcvars) if vcvars else None,
            "cl": shutil.which("cl.exe", path=env.get("PATH")) if os.name == "nt" else None,
            "nvcc": nvcc, "ninja": shutil.which("ninja", path=env.get("PATH")),
            "project": str(project), "output": str(output),
            "native_child_access_checked": os.name == "nt", "versions": {}}
    if args.generator in ("Ninja", "Ninja Multi-Config") and not info["ninja"]:
        raise ToolchainError("Ninja missing from the build environment; use the existing VS CMake tools or pass another generator.")
    for name, executable in (("cmake", cmake), ("nvcc", nvcc)):
        if executable:
            result = subprocess.run([executable, "--version"], cwd=output, env=env, capture_output=True,
                                    encoding="utf-8", errors="replace")
            if result.returncode:
                raise ToolchainError(f"{name} --version failed ({result.returncode}): {result.stderr.strip()}")
            info["versions"][name] = result.stdout.strip()
    return env, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cmake", help="Path to an already installed CMake")
    parser.add_argument("--cuda", action="store_true", help="Build the optional G1/G2 CUDA evaluator and linearization")
    parser.add_argument("--cudss", type=Path, help="Existing project-owned cuDSS root; enables one native LM candidate")
    parser.add_argument("--generator", help="Existing CMake generator, e.g. Ninja")
    parser.add_argument("--cuda-architectures", default="native", help="CMake CUDA architectures; default builds for the installed GPU")
    parser.add_argument("--cuda-fast-compile", choices=["0","min","mid","max"], default="0", help="Explicit CUDA compile-time optimizer tradeoff; requires compiler support")
    parser.add_argument("--vcvars", type=Path, help="Existing vcvars64.bat; otherwise discover MSVC with vswhere when needed")
    parser.add_argument("--check-toolchain", action="store_true", help="Check native child access and resolve the existing toolchain without compiling")
    args = parser.parse_args()
    if args.cudss and not args.cuda: parser.error("--cudss requires --cuda")
    project = Path(__file__).resolve().parents[1]
    if args.cudss:
        sys.path.insert(0, str(project / "src"))
        from rinkspace.storage import AssetStore
        args.cudss = AssetStore(project).source(args.cudss.resolve()).resolve(strict=True)
    output = args.out.resolve()
    if output == project or project in output.parents:
        parser.error("build output must be outside the source/OneDrive project")
    try:
        output.mkdir(parents=True, exist_ok=True)
        build_env, toolchain = prepare_toolchain(args, project, output)
    except (ToolchainError, OSError) as exc:
        print(f"Native build preflight failed: {exc}", file=sys.stderr)
        return 2
    (output/"toolchain.json").write_text(json.dumps(toolchain, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    log = output/"build.log"
    cmake = toolchain["cmake"]
    commands = []
    if cmake:
        commands = [[cmake, "-S", str(project/"native"), "-B", str(output), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF"],
                    [cmake, "--build", str(output), "--config", "Release"]]
        commands[0].append("-DRINKSPACE_ENABLE_CUDA="+("ON" if args.cuda else "OFF"))
        commands[0].append("-DRINKSPACE_CUDSS_ROOT="+(str(args.cudss.resolve(strict=True)) if args.cudss else ""))
        if os.name == "nt":
            commands[0].append("-DCMAKE_TRY_COMPILE_CONFIGURATION=Release")
        if args.cuda:
            commands[0].append("-DCMAKE_CUDA_COMPILER="+toolchain["nvcc"])
            commands[0].extend(["-DCMAKE_CUDA_ARCHITECTURES="+args.cuda_architectures,"-DRINKSPACE_CUDA_FAST_COMPILE="+args.cuda_fast_compile])
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
    if args.check_toolchain:
        print(json.dumps(toolchain, ensure_ascii=False, indent=2))
        return 0
    print(f"Native build log: {log}", flush=True)
    # Keep failed attempts available when rerunning in the correct environment.
    with log.open("a", encoding="utf-8") as stream:
        for command in commands:
            stream.write(json.dumps(command, ensure_ascii=False)+"\n")
            stream.flush()
            process = subprocess.run(command, cwd=output, env=build_env, stdout=stream, stderr=subprocess.STDOUT)
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
              "native_geometry_plan_implemented": bool(args.cuda), "persistent_resume_implemented": False, "toolchain": toolchain, "commands": commands}
    (output/"build.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
