"""Record one independent CPU validation and its downloaded wheel hashes."""
from __future__ import annotations

import argparse
import email
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    environment = args.environment.resolve()
    if not environment.is_relative_to(root):
        raise SystemExit("Environment must stay under the independent Data root.")
    build = Path(str(environment) + "-build")
    destination = Path(__file__).resolve().parent
    smoke = json.loads((build / "cpu-smoke.json").read_text(encoding="utf-8"))
    if smoke["status"] != "passed" or Path(smoke["prefix"]).resolve() != environment:
        raise SystemExit("A successful smoke report from the selected environment is required.")
    wheels = []
    for wheel in sorted((root / "wheelhouse").glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            members = [name for name in archive.namelist()
                       if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            if len(members) != 1:
                raise ValueError(f"Expected one root wheel METADATA: {wheel}")
            metadata = email.message_from_bytes(archive.read(members[0]))
        wheels.append(dict(name=metadata["Name"], version=metadata["Version"], filename=wheel.name,
                           size_bytes=wheel.stat().st_size, sha256=digest(wheel)))
    wheels.sort(key=lambda item: item["name"].lower())
    if not wheels:
        raise SystemExit("No downloaded dependency wheels found.")
    for wheel in wheels:
        installed = {key.lower(): value for key, value in smoke["packages"].items()}
        if installed.get(wheel["name"].lower()) != wheel["version"]:
            raise SystemExit(f"Downloaded wheel does not match installed package: {wheel['name']}")
    lock = (
        "# Dependency wheel-hash lock validated on Windows AMD64 / Python 3.11.0, 2026-10-11.\n"
        "# Only dependency imports/tiny mathematics/CLI smoke were validated.\n"
        "# No event/native/GPU/media acceptance. Project source is built separately.\n"
        "# pip is supplied by the selected base Python venv seed.\n"
    )
    lock += "".join(f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n" for w in wheels)
    (destination / "requirements-cpu-validated-windows-lock.txt").write_text(lock, encoding="utf-8")
    source_manifest = []
    staged_source = build / "source"
    for source in sorted(staged_source.rglob("*")):
        relative = source.relative_to(staged_source).as_posix()
        if (source.is_file() and "__pycache__" not in source.parts and not source.name.endswith(".pyc")
                and ".egg-info" not in relative
                and (relative == "pyproject.toml" or relative.startswith("src/rinkspace/"))):
            source_manifest.append(dict(path=relative, sha256=digest(source)))
    report = {
        "schema": "rinkspace.environment-reconstruction-validation.v1",
        "validation_date": "2026-10-11",
        "overall_status": "cpu_dependency_cli_smoke_passed; full_project_runtime_not_validated",
        "cpu": {
            "status": "independent_environment_dependency_cli_smoke_passed",
            "base_interpreter": "D:/Python/Python311/python.exe",
            "environment_path": str(environment), "python": smoke["python"],
            "installation": "New venv; downloaded public wheels installed offline. Project Python wheel built from a private Data source copy. Existing environments unchanged.",
            "dependency_input": "requirements-cpu-python311-windows.txt",
            "dependency_hash_lock": "requirements-cpu-validated-windows-lock.txt",
            "wheel_count_third_party": len(wheels),
            "wheel_total_size_bytes": sum(w["size_bytes"] for w in wheels),
            "wheels": wheels, "installed_packages": smoke["packages"],
            "checks_passed": ["pip_check", "installer_powershell_syntax", "smoke_python_syntax",
                              "existing_target_rejected", "live_project_venv_target_rejected"] + smoke["checks"],
            "cli_doctor": smoke["cli"]["doctor"], "source_snapshot": source_manifest,
            "observed_differences": [
                dict(name="setuptools", live_observed="65.5.0", validated="84.0.0",
                     reason="Current pyproject requires setuptools>=69."),
                dict(name="iniconfig", live_observed="2.3.0", validated="2.3.1",
                     reason="pytest transitive dependency resolved from public PyPI."),
                dict(name="rinkspace_metadata", live_observed="0.1.0", validated="0.4.0",
                     reason="Built the currently declared project version."),
            ],
            "limitations": [
                "CLI doctor/source __version__ is 0.3.0; installed wheel metadata is 0.4.0. Active source was not changed.",
                "ffmpeg/ffprobe are absent from the validation subprocess PATH; external video tools were not reconstructed.",
                "No event files, source videos, checkpoints or active solver inputs were opened.",
                "No native backend loaded or compiled; certified kernels and complete-state bindings are separate prerequisites.",
                "No geometry accuracy or numerical equivalence to the live runtime is claimed.",
            ],
            "logs_data_root": str(root), "smoke_report": str(build / "cpu-smoke.json"),
            "wheel_manifest_in_data": str(build / "wheel-manifest.json"),
            "download_history": [
                dict(log="download.log", result="Default-permission index lookup returned no distributions; not evidence of nonexistent versions."),
                dict(log="download-pypi.log", result="Explicit public index under default permissions returned no distributions."),
                dict(log="download-pypi-reviewed.log", result="Approved public-index access downloaded several wheels, then a files.pythonhosted.org read timeout during PyAV download."),
                dict(log="download-pypi-longer-timeout.log", result="Approved public-index retry with retained cache and 60-second socket timeout succeeded."),
            ],
        },
        "vision": {
            "status": "candidate_definition_only_not_installed_or_validated",
            "definition": "requirements-vision-torch-cu128.txt",
            "base_layer": "requirements-cpu-validated-windows-lock.txt in a separate newly created environment",
            "note": "Observed CUDA torch/torchvision builds recorded. The candidate uses the project-compatible OpenCV CPU layer instead of the external environment's OpenCV 5. No GPU compatibility acceptance.",
        },
        "gtsam": {
            "status": "candidate_definition_only_not_installed_or_validated",
            "definition": "gtsam-reconstruction-candidate.yml",
            "numpy_boundary": "1.26.4, separate from validated CPU NumPy 2.4.6",
            "note": "Core Python/GTSAM/NumPy values observed in historical conda environment. Added geometry packages are an untested candidate. Conda build strings, native toolchain and vendor dependencies are not locked.",
        },
        "body": {
            "status": "upstream_identity_record_only_not_installed_or_validated",
            "definition": "body-upstream-provenance.json",
            "note": "Local GVHMR and PyTorch3D source identity recorded. Divergent upstream Linux requirements, local compiler flags/modifications and external model prerequisites need separate preservation. No blind freeze replay.",
        },
        "activity_boundary": "Only environments/* in OneDrive and this new Data validation directory were modified. No active runtime source/config, existing .venv, model/input, source material, native build, GPU workload, Git commit or push.",
    }
    (destination / "reconstruction-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(third_party_wheels=len(wheels), source_assets=len(source_manifest),
                          cpu_status=report["cpu"]["status"]), indent=2))


if __name__ == "__main__":
    main()
