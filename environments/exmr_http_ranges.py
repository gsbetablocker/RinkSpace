"""Resumable pinned Hugging Face HTTP ranges, with one disk writer.

Signed locations and credentials remain in memory. A per-chunk SHA ledger is
written atomically only after the candidate file is flushed. The candidate is
promoted on the same volume only after its complete official SHA256 matches.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def download_ranges(row: dict, target: Path, recorder, token: str, *, workers: int = 8,
                    chunk_bytes: int = 16 << 20, probe_chunks: int = 0,
                    attempts: int = 4, socket_timeout: float = 30.0) -> dict:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    if not 1 <= workers <= 8 or chunk_bytes <= 0 or attempts < 1:
        raise ValueError("Invalid HTTP worker, chunk size or retry options")
    source_url = hf_hub_url(row["repo_id"], row["filename"], repo_type=row["repo_type"],
                           revision=row["revision"])
    location_lock = threading.Lock()
    metadata = get_hf_file_metadata(source_url, token=token)
    location = metadata.location  # Never log this signed URL.
    total = metadata.size
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Pinned HTTP file has no usable content length")
    if metadata.commit_hash != row["revision"]:
        raise ValueError("HTTP metadata differs from pinned repository revision")
    # LFS ETags are SHA256. Small repository files use a 40-character Git
    # blob ID instead; the pinned revision and final SHA still bind them.
    if len(metadata.etag or "") == 64 and metadata.etag not in row["expected_sha256"]:
        raise ValueError("HTTP metadata ETag differs from the official file SHA256")
    candidate = target.with_name(target.name + ".pending")
    bitmap = target.with_name(target.name + ".pending.bitmap.json")
    identity = dict(repo_id=row["repo_id"], revision=row["revision"], filename=row["filename"],
                    expected_sha256=row["expected_sha256"], bytes=total, chunk_bytes=chunk_bytes)
    count = (total + chunk_bytes - 1) // chunk_bytes
    target.parent.mkdir(parents=True, exist_ok=True)
    if candidate.exists() or bitmap.exists():
        if not candidate.exists() or not bitmap.exists():
            raise ValueError("HTTP candidate or ledger missing; existing bytes preserved")
        state = json.loads(bitmap.read_text(encoding="utf-8"))
        if state.get("identity") != identity or candidate.stat().st_size != total:
            raise ValueError("HTTP candidate identity differs; existing bytes preserved")
        if len(state.get("chunks", [])) != count:
            raise ValueError("HTTP chunk ledger length differs; existing bytes preserved")
        # A ledger bit alone is not trusted on a resumed run.
        with candidate.open("rb") as stream:
            for index, digest in enumerate(state["chunks"]):
                if digest is None:
                    continue
                stream.seek(index * chunk_bytes)
                raw = stream.read(min(chunk_bytes, total - index * chunk_bytes))
                if hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError(f"Resumed HTTP chunk {index} no longer matches its SHA256")
    else:
        with candidate.open("xb") as stream:
            stream.truncate(total)
            stream.flush()
            os.fsync(stream.fileno())
        state = dict(schema="rinkspace.http-range-ledger.v1", identity=identity,
                     chunks=[None] * count)
        _atomic_json(bitmap, state)

    def completed_bytes() -> int:
        return sum(min(chunk_bytes, total - index * chunk_bytes)
                   for index, digest in enumerate(state["chunks"]) if digest is not None)

    def read_chunk(index: int) -> tuple[int, bytes, str]:
        nonlocal location
        begin = index * chunk_bytes
        end = min(begin + chunk_bytes, total) - 1
        expected = end - begin + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                # Only the ephemeral signed location is requested here; no
                # bearer token is forwarded to the CDN host.
                request = Request(location, headers={"Range": f"bytes={begin}-{end}",
                                                     "User-Agent": "RinkSpace pinned HTTP ranges"})
                with urlopen(request, timeout=socket_timeout) as response:
                    whole_file_200 = response.status == 200 and begin == 0 and end == total - 1
                    if response.status != 206 and not whole_file_200:
                        raise ValueError(f"Range response is HTTP {response.status}, expected 206")
                    if not whole_file_200 and response.headers.get("Content-Range") != f"bytes {begin}-{end}/{total}":
                        raise ValueError("Range response Content-Range does not match requested bytes")
                    data = bytearray()
                    while len(data) < expected:
                        piece = response.read(min(1 << 20, expected - len(data)))
                        if not piece:
                            raise ValueError("Range response ended before its expected byte count")
                        data.extend(piece)
                    if response.read(1):
                        raise ValueError("Range response exceeded its expected byte count")
                raw = bytes(data)
                return index, raw, hashlib.sha256(raw).hexdigest()
            except Exception as error:
                last_error = error
                recorder.event("http_chunk_retry", asset=row["relative_path"], chunk=index,
                               attempt=attempt, error_type=type(error).__name__,
                               http_status=error.code if isinstance(error, HTTPError) else None)
                if isinstance(error, HTTPError) and error.code in (401, 403):
                    with location_lock:
                        metadata_new = get_hf_file_metadata(source_url, token=token)
                        if metadata_new.size != total or metadata_new.etag != metadata.etag:
                            raise ValueError("Refreshed signed location changed file identity")
                        location = metadata_new.location
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"HTTP chunk {index} failed after {attempts} attempts ({type(last_error).__name__})")

    pending = [index for index, digest in enumerate(state["chunks"]) if digest is None]
    if probe_chunks:
        pending = pending[:probe_chunks]
    before = completed_bytes()
    started = time.perf_counter()
    recorder.event("http_ranges_started", asset=row["relative_path"], candidate=str(candidate),
                   bitmap=str(bitmap), total_bytes=total, completed_bytes=before,
                   workers=workers, chunk_bytes=chunk_bytes, chunks_to_fetch=len(pending),
                   max_attempts=attempts, socket_timeout_s=socket_timeout)
    next_index = iter(pending)
    with candidate.open("r+b") as writer, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for _ in range(min(workers, len(pending))):
            index = next(next_index)
            futures[pool.submit(read_chunk, index)] = index
        try:
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in ready:
                    del futures[future]
                    index, raw, digest = future.result()
                    # This is the only write path. Worker threads only return
                    # in-memory byte buffers; they never open the candidate.
                    writer.seek(index * chunk_bytes)
                    writer.write(raw)
                    writer.flush()
                    os.fsync(writer.fileno())
                    state["chunks"][index] = digest
                    _atomic_json(bitmap, state)
                    done = completed_bytes()
                    elapsed = time.perf_counter() - started
                    row.update(http_completed_bytes=done, http_total_bytes=total,
                               http_bitmap=str(bitmap), http_candidate=str(candidate))
                    recorder.save()
                    recorder.event("http_chunk_complete", asset=row["relative_path"], chunk=index,
                                   chunk_sha256=digest, completed_bytes=done, total_bytes=total,
                                   elapsed_s=elapsed, transfer_MiB_per_s=(done - before) / (1 << 20) / elapsed)
                    try:
                        next_chunk = next(next_index)
                    except StopIteration:
                        continue
                    futures[pool.submit(read_chunk, next_chunk)] = next_chunk
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    elapsed = time.perf_counter() - started
    result = dict(complete=all(state["chunks"]), completed_bytes=completed_bytes(), total_bytes=total,
                  elapsed_s=elapsed, transfer_MiB_per_s=(completed_bytes() - before) / (1 << 20) / max(elapsed, 1e-9),
                  candidate=str(candidate), bitmap=str(bitmap))
    if not result["complete"]:
        recorder.event("http_probe_complete", asset=row["relative_path"], **result)
        return result
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual not in row["expected_sha256"]:
        state.update(status="whole_file_sha256_failed", actual_sha256=actual)
        _atomic_json(bitmap, state)
        raise ValueError(f"Complete HTTP file SHA256 mismatch; candidate preserved: {actual}")
    if target.exists():
        raise FileExistsError(f"Final target appeared during HTTP transfer; candidate preserved: {target}")
    os.replace(candidate, target)
    state.update(status="verified_and_promoted", actual_sha256=actual)
    _atomic_json(bitmap, state)
    result.update(sha256=actual, destination=str(target))
    recorder.event("http_file_verified", asset=row["relative_path"], **result)
    return result
