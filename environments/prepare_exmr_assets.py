"""Verify or fetch only the pinned assets for the ExMR camera-frame comparison.

Uses the existing Hugging Face client; never installs packages or loads models.
Default is verification only. Pass --fetch to obtain missing files. Existing
different bytes are preserved. HTTP ranges retain their candidate and chunk
ledger for interrupted transfers; HF controls its own temporary-file cleanup.
SMPL-X is referenced in place; MANO, YOLO, Sapiens2 and DPVO are not downloaded.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen
import zipfile


EXMR_COMMIT = "eac72e47339b404c0443f5fb323412e2ce432410"
TABLE_SHA256 = "af1d5c02a044e8c909c4463f9a254ef8ee7eb85725936dab18336c26cbb9a03e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def literals(path: Path, names: dict | None = None) -> dict:
    """Read upstream constants without importing its GPU/model dependencies."""
    values = dict(names or {})
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        try:
            if isinstance(node.value, ast.Dict):
                value = {}
                for key, item in zip(node.value.keys, node.value.values):
                    value[ast.literal_eval(key)] = (
                        values[item.id] if isinstance(item, ast.Name) else ast.literal_eval(item)
                    )
            else:
                value = ast.literal_eval(node.value)
        except (ValueError, TypeError, KeyError):
            continue
        if isinstance(target, ast.Name):
            values[target.id] = value
        elif isinstance(target, ast.Tuple):
            values.update({part.id: item for part, item in zip(target.elts, value)
                           if isinstance(part, ast.Name)})
    return values


def plan(source: Path) -> tuple[list[dict], dict]:
    table = source / "tools/download_models.py"
    if sha256_file(table) != TABLE_SHA256:
        raise ValueError("Pinned ExMR download table differs; existing source preserved")
    constants = literals(table, literals(source / "exmr/assets.py"))
    rows = []
    groups = [
        ("exmr", "EXMR_REPO", "EXMR_REVISION", "model",
         ["exmr.safetensors", "config.json"]),
        ("sam-3d-body-dinov3", "SAM3D_REPO", "SAM3D_REVISION", "model",
         ["model.ckpt", "model_config.yaml", "assets/mhr_model.pt"]),
        ("hamer", "HAMER_SPACE", "HAMER_SPACE_REVISION", "space",
         ["hamer_ckpts/checkpoints/hamer.ckpt", "hamer_ckpts/model_config.yaml",
          "hamer_ckpts/dataset_config.yaml", "data/mano_mean_params.npz"]),
        ("rtmw", "EXMR_REPO", "EXMR_REVISION", "model",
         ["rtmw-x_384x288_fp16.onnx"]),
    ]
    for component, repo, revision, repo_type, filenames in groups:
        for filename in filenames:
            relative = f"{component}/{filename}"
            expected = constants["SHA256"][relative]
            rows.append(dict(component=component, repo_id=constants[repo],
                             revision=constants[revision], repo_type=repo_type,
                             filename=f"_DATA/{filename}" if component == "hamer" else filename,
                             relative_path=relative,
                             expected_sha256=list(expected) if isinstance(expected, tuple) else [expected]))
    return rows, constants


class Recorder:
    def __init__(self, directory: Path, manifest: dict, *, prefetch: bool = False):
        directory.mkdir(parents=True, exist_ok=True)
        self.log = directory / "assets_progress.jsonl"
        self.path = directory / ("assets_prefetch_manifest.json" if prefetch else "assets_manifest.json")
        self.manifest = manifest
        self.lock = threading.Lock()

    def event(self, event: str, **fields) -> None:
        row = dict(utc=utc(), run_id=self.manifest["run_id"], event=event)
        row.update(fields)
        raw = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self.lock:
            with self.log.open("a", encoding="utf-8") as stream:
                stream.write(raw + "\n")
            print(raw, flush=True)

    def save(self) -> None:
        self.manifest["updated_utc"] = utc()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def safe_reason(error: Exception, token: str | None) -> str:
    reason = str(error).replace(token, "[REDACTED]") if token else str(error)
    reason = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", reason)
    reason = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", reason)
    return reason.splitlines()[0][:1600] if reason else type(error).__name__


def verify(path: Path, expected: list[str]) -> str:
    actual = sha256_file(path)
    if actual not in expected:
        raise ValueError(f"SHA256 mismatch; preserved {path}: {actual}")
    return actual


def heartbeat(stop: threading.Event, recorder: Recorder, directory: Path, relative: str) -> None:
    while not stop.wait(15):
        partial = [dict(path=str(p), bytes=p.stat().st_size)
                   for p in directory.rglob("*.incomplete") if p.is_file()]
        recorder.event("download_progress", asset=relative, incomplete_files=partial)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    # Match the upstream SHA's POSIX order explicitly. WindowsPath sorts
    # case-insensitively, which otherwise hashes the same 221 files differently.
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def prepare_dinov3(data: Path, constants: dict, recorder: Recorder, fetch: bool) -> None:
    commit = constants["DINOV3_COMMIT"]
    expected = constants["DINOV3_TREE_SHA256"]
    destination = data / "cache/torch/hub/facebookresearch_dinov3_main"
    record = dict(component="dinov3_code", commit=commit, destination=str(destination),
                  expected_tree_sha256=expected, weights_downloaded=False)
    recorder.manifest["dinov3"] = record
    recorder.save()
    if destination.exists():
        actual = tree_sha256(destination)
        if actual != expected:
            raise ValueError("Existing DINOv3 code differs from pinned tree; preserved")
    else:
        if not fetch:
            raise FileNotFoundError(destination)
        url = f"https://github.com/facebookresearch/dinov3/archive/{commit}.zip"
        packed = data / "cache/packed_assets" / f"dinov3_{commit}.zip"
        packed.parent.mkdir(parents=True, exist_ok=True)
        recorder.event("download_started", asset="dinov3_code", url=url)
        if not packed.exists():
            partial = packed.with_suffix(".zip.incomplete")
            # A complete zip is promoted only after the network response finishes.
            # Failed response bytes remain in .incomplete for diagnosis.
            if partial.exists() and partial.stat().st_size:
                retained = partial.with_name(partial.name + f".{time.time_ns()}.failed")
                os.replace(partial, retained)
                recorder.event("incomplete_preserved", asset="dinov3_code", path=str(retained), bytes=retained.stat().st_size)
            with urlopen(Request(url, headers={"User-Agent": "RinkSpace pinned assets"}), timeout=60) as response:
                with partial.open("wb") as stream:
                    for chunk in iter(lambda: response.read(1 << 20), b""):
                        stream.write(chunk)
            os.replace(partial, packed)
        staging = data / "cache/torch/hub" / f".dinov3_{commit}.staging"
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packed) as archive:
            for member in archive.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise ValueError("DINOv3 archive member escapes staging directory")
            archive.extractall(staging)
        extracted = staging / f"dinov3-{commit}"
        actual = tree_sha256(extracted)
        if actual != expected:
            raise ValueError(f"DINOv3 pinned tree SHA256 mismatch: {actual}")
        os.replace(extracted, destination)
        record.update(source_url=url, archive=str(packed), archive_sha256=sha256_file(packed))
    trust = destination.parent / "trusted_list"
    existing = trust.read_text(encoding="utf-8") if trust.exists() else ""
    if "facebookresearch_dinov3" not in existing.splitlines():
        with trust.open("a", encoding="utf-8") as stream:
            if existing and not existing.endswith("\n"):
                stream.write("\n")
            stream.write("facebookresearch_dinov3\n")
    record.update(status="verified", actual_tree_sha256=actual,
                  torch_home=str(data / "cache/torch"), trust_cache=str(trust),
                  trusted_entry="facebookresearch_dinov3")
    recorder.save()
    recorder.event("asset_verified", asset="dinov3_code", destination=str(destination), tree_sha256=actual)


def run(args: argparse.Namespace) -> int:
    data = args.data_root.resolve()
    project = Path(__file__).resolve().parents[1]
    if data.is_relative_to(project) or project.is_relative_to(data):
        raise ValueError("Data must be separate from the source worktree")
    source = data / "runtime_tools/body/ExMR"
    root = data / "cache/pretrained/exmr"
    record_dir = args.record_dir.resolve()
    if not record_dir.is_relative_to(data):
        raise ValueError("Asset records must stay inside Data")
    rows, constants = plan(source)
    choices = args.components or ["exmr", "sam3d", "hamer", "rtmw"]
    rows = [row for component in choices for row in rows
            if row["component"] == ("sam-3d-body-dinov3" if component == "sam3d" else component)]
    previous = record_dir / ("assets_prefetch_manifest.json" if args.prefetch else "assets_manifest.json")
    history = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    event_history = []
    progress = record_dir / "assets_progress.jsonl"
    if progress.exists():
        retained_events = {"asset_failed", "dinov3_failed", "transport_switch",
                           "auxiliary_superseded", "http_probe_complete"}
        for line in progress.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted final log line stays in the original JSONL.
            if event.get("event") in retained_events:
                event_history.append(event)
    manifest = dict(schema="rinkspace.exmr-assets.v1", run_id=utc(), status="running",
                    mode="fetch" if args.fetch else "verify", data_root=str(data),
                    checkpoint_root=str(root), source_root=str(source),
                    exmr_source_commit=EXMR_COMMIT, upstream_download_table_sha256=TABLE_SHA256,
                    preparation_code={str(path): sha256_file(path) for path in
                                      [Path(__file__), Path(__file__).with_name("exmr_http_ranges.py")]
                                      if path.is_file()},
                    torch_home=str(data / "cache/torch"),
                    token_source="explicit local file; value never recorded",
                    selected_components=list(dict.fromkeys(choices + ([] if args.prefetch else ["dinov3_code"]))),
                    prefetch_only=args.prefetch,
                    transport=args.transport,
                    excluded_components=["mano", "yolo", "sapiens2", "dpvo"],
                    licenses={key: constants["LICENSES"][key] for key in ("exmr", "smplx", "sam3d", "hamer", "rtmw")},
                    mano=dict(status="not_downloaded", reason="Feature-only HaMeR loader uses backbone/mano_head tokens; MANO mesh is not needed by that path"),
                    no_inference=True, no_fitting=True, no_environment_installation=True,
                    prior_runs=history.get("prior_runs", []) + ([dict(run_id=history.get("run_id"), status=history.get("status"))] if history else []),
                    historical_events=event_history,
                    files=rows, failures=[])
    recorder = Recorder(record_dir, manifest, prefetch=args.prefetch)
    recorder.save()
    recorder.event("preparation_started", checkpoint_root=str(root), files=len(rows), fetch=args.fetch)
    # Set before importing huggingface_hub so its fallback caches stay in Data.
    os.environ["HF_HOME"] = str(data / "cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(data / "cache/huggingface/hub")
    os.environ["HF_XET_CACHE"] = str(data / "cache/huggingface/xet")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    token = None
    if args.fetch and rows:
        try:
            token = args.token_file.read_text(encoding="utf-8").strip()
            if not token:
                raise ValueError("Explicit local Hugging Face token file is empty")
            from huggingface_hub import hf_hub_download
            manifest["huggingface_hub_version"] = __import__("huggingface_hub").__version__
        except Exception as error:
            failure = dict(asset="client_initialization", error_type=type(error).__name__, reason=safe_reason(error, token), utc=utc())
            manifest.update(status="failed", failures=[failure], completed_utc=utc())
            recorder.save()
            recorder.event("preparation_failed", **failure)
            return 1
    for row in rows:
        target = root / row["relative_path"]
        row.update(destination=str(target), status="checking")
        recorder.save()
        try:
            if not target.exists():
                if args.prefetch and (record_dir / "assets_manifest.json").exists():
                    primary = json.loads((record_dir / "assets_manifest.json").read_text(encoding="utf-8"))
                    active = next((item for item in primary.get("files", [])
                                   if item["relative_path"] == row["relative_path"]), {})
                    if primary.get("status") == "running" and active.get("status") == "downloading":
                        row.update(status="deferred_to_primary")
                        recorder.save()
                        recorder.event("asset_deferred", asset=row["relative_path"], reason="Primary is already downloading this asset; avoid redundant transfer after layout promotion")
                        continue
                if not args.fetch:
                    raise FileNotFoundError(target)
                local_dir = root / row["component"]
                local_dir.mkdir(parents=True, exist_ok=True)
                row["status"] = "downloading"
                recorder.save()
                recorder.event("download_started", asset=row["relative_path"], repo_id=row["repo_id"], revision=row["revision"], filename=row["filename"])
                if args.transport == "http":
                    from exmr_http_ranges import download_ranges
                    result = download_ranges(row, target, recorder, token,
                                             workers=args.http_workers,
                                             probe_chunks=args.http_probe_chunks,
                                             attempts=args.http_attempts)
                    if not result["complete"]:
                        row.update(status="partial_probe", http_probe=result)
                        manifest.update(status="probe_complete", completed_utc=utc())
                        recorder.save()
                        return 0
                    actual = result["sha256"]
                    verified_path = target
                else:
                    stop = threading.Event()
                    worker = threading.Thread(target=heartbeat, args=(stop, recorder, local_dir, row["relative_path"]), daemon=True)
                    worker.start()
                    try:
                        fetched = Path(hf_hub_download(repo_id=row["repo_id"], filename=row["filename"], revision=row["revision"], repo_type=row["repo_type"], local_dir=local_dir, token=token))
                    finally:
                        stop.set()
                        worker.join()
                    actual = verify(fetched, row["expected_sha256"])
                    if fetched.resolve() != target.resolve() and not args.prefetch:
                        # HaMeR Space has a _DATA prefix. Move on the same Data
                        # volume into the official checkpoint layout; never copy.
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            raise FileExistsError(target)
                        os.replace(fetched, target)
                    verified_path = fetched if args.prefetch else target
                row["downloaded_this_run"] = True
            else:
                actual = verify(target, row["expected_sha256"])
                verified_path = target
                row["downloaded_this_run"] = False
            row.update(status="verified", sha256=actual, bytes=verified_path.stat().st_size,
                       verified_path=str(verified_path), pending_layout_move=verified_path.resolve() != target.resolve())
            recorder.save()
            recorder.event("asset_verified", asset=row["relative_path"], destination=str(verified_path), bytes=row["bytes"], sha256=actual, downloaded_this_run=row["downloaded_this_run"], pending_layout_move=row["pending_layout_move"])
        except Exception as error:
            failure = dict(asset=row["relative_path"], error_type=type(error).__name__, reason=safe_reason(error, token), utc=utc())
            row.update(status="failed", failure=failure)
            manifest["failures"].append(failure)
            recorder.save()
            recorder.event("asset_failed", **failure)
    if args.prefetch:
        manifest["status"] = "failed" if manifest["failures"] else "complete"
        manifest["completed_utc"] = utc()
        recorder.save()
        recorder.event("prefetch_complete", status=manifest["status"], verified_files=sum(row["status"] == "verified" for row in rows), failed_assets=len(manifest["failures"]))
        return 1 if manifest["failures"] else 0
    smplx = data / "runtime_tools/body/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    try:
        actual = verify(smplx, [constants["SHA256"]["body_models/smplx/SMPLX_NEUTRAL.npz"]])
        manifest["smplx_reference"] = dict(path=str(smplx), sha256=actual, bytes=smplx.stat().st_size, copied=False, status="verified")
        recorder.event("reference_verified", asset="smplx", path=str(smplx), sha256=actual, copied=False)
    except Exception as error:
        failure = dict(asset="smplx_reference", error_type=type(error).__name__, reason=safe_reason(error, token), utc=utc())
        manifest["failures"].append(failure)
        recorder.event("asset_failed", **failure)
    try:
        prepare_dinov3(data, constants, recorder, args.fetch)
    except Exception as error:
        failure = dict(asset="dinov3_code", error_type=type(error).__name__, reason=safe_reason(error, token), utc=utc())
        manifest["failures"].append(failure)
        recorder.event("asset_failed", **failure)
    manifest["status"] = "failed" if manifest["failures"] else "complete"
    manifest["completed_utc"] = utc()
    recorder.save()
    recorder.event("preparation_complete", status=manifest["status"], verified_files=sum(row["status"] == "verified" for row in rows), failed_assets=len(manifest["failures"]))
    return 1 if manifest["failures"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".cache/huggingface/token")
    parser.add_argument("--fetch", action="store_true", help="Fetch missing pinned assets; preserve different existing bytes")
    parser.add_argument("--components", nargs="+", choices=["exmr", "sam3d", "hamer", "rtmw", "dinov3"], help="Selected components in download order; dinov3 alone prepares code and trust cache without weights")
    parser.add_argument("--prefetch", action="store_true", help="Use separate receipt; leave HaMeR Space files in HF's _DATA path for a primary run to promote without copying")
    parser.add_argument("--transport", choices=["hf", "http"], default="hf", help="HF local_dir or resumable official HTTP ranges")
    parser.add_argument("--http-workers", type=int, default=8, help="HTTP connections, at most 8; one disk writer")
    parser.add_argument("--http-probe-chunks", type=int, default=0, help="Fetch only this many 16MiB chunks, preserving candidate and ledger for continuation")
    parser.add_argument("--http-attempts", type=int, default=4, help="Bounded attempts per HTTP chunk")
    args = parser.parse_args()
    if args.prefetch and not args.fetch:
        parser.error("--prefetch requires --fetch")
    if args.components and len(set(args.components)) != len(args.components):
        parser.error("--components must not contain duplicates")
    if not 1 <= args.http_workers <= 8 or args.http_probe_chunks < 0 or args.http_attempts < 1:
        parser.error("Invalid HTTP worker, probe or attempt options")
    if args.http_probe_chunks and (args.transport != "http" or not args.fetch):
        parser.error("HTTP probe requires --transport http --fetch")
    if args.prefetch and args.transport == "http":
        parser.error("HTTP ranges write the final-layout candidate directly; do not combine with --prefetch")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
