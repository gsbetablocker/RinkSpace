"""Read-only, loopback camera catalog browser with seekable local source media.

The viewer presents catalog decisions verbatim. Browser time seeks are for
navigation; the catalog's source PTS and boundary observations remain evidence.
"""
from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .web import byte_range


STATIC = Path(__file__).parent / "static"
STATIC_FILES = {
    "camera_classification.html", "camera_classification.css", "camera_classification.js"
}


def _as_list(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def _has_boundary_issue(shot):
    if (shot.get("interval_type") == "boundary_unresolved"
            or str(shot.get("boundary_status", "")).startswith("unresolved")):
        return True
    for key in ("boundary_start", "boundary_end"):
        boundary = shot.get(key) or {}
        if isinstance(boundary, dict) and str(boundary.get("status", "")).startswith("unresolved"):
            return True
    return False


class CameraCatalog:
    def __init__(self, root, catalog_path="outputs/camera_classification/catalog.json"):
        self.root = Path(root).resolve()
        self.path = Path(catalog_path)
        if not self.path.is_absolute():
            self.path = self.root / self.path
        self.path = self.path.resolve()
        if not self.path.is_relative_to(self.root):
            raise ValueError("Catalog must be inside the project")
        self.lock = threading.RLock()
        self._mtime = None
        self.reload()

    def _resolve(self, value):
        if not isinstance(value, str) or len(value) > 2048 or "\n" in value:
            return None
        try:
            candidate = Path(value.replace("\\", "/"))
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = candidate.resolve()
            if candidate.is_relative_to(self.root) and candidate.is_file():
                return candidate
        except (OSError, ValueError):
            pass
        return None

    def _reference_path(self, value):
        """Normalize declared paths without stat-ing every evidence image.

        Real path and symlink containment are checked when a file is requested.
        A catalog can reference more than 100,000 images on a synced directory.
        """
        if not isinstance(value, str) or len(value) > 2048 or "\n" in value:
            return None
        try:
            path = Path(value.replace("\\", "/"))
            if not path.is_absolute():
                path = self.root / path
            path = Path(os.path.abspath(path))
            return path if path.is_relative_to(self.root) else None
        except (OSError, ValueError):
            return None

    def reload(self):
        """Pick up atomically replaced classification results without a restart."""
        with self.lock:
            related = [self.path.with_name("validation.json"), self.path.parent / "checks/index.json",
                       self.path.with_name("human_evidence.json"), self.path.with_name("reference_evidence.json"),
                       self.path.parent / "evidence/context_references.json"]
            mtime = (self.path.stat().st_mtime_ns,) + tuple(p.stat().st_mtime_ns if p.exists() else None for p in related)
            if mtime == self._mtime:
                return
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            shots = data.get("shots", [])
            by_id = {shot["shot_id"]: shot for shot in shots}
            if len(by_id) != len(shots):
                raise ValueError("Catalog has duplicate shot_id values")
            files = {self.path}
            resolved_cache = {}

            def collect(item):
                if isinstance(item, dict):
                    for value in item.values():
                        collect(value)
                elif isinstance(item, list):
                    for value in item:
                        collect(value)
                elif isinstance(item, str) and (
                    "/" in item or "\\" in item or item.endswith((".json", ".jpg", ".png", ".mp4", ".webm"))
                ):
                    if item not in resolved_cache:
                        resolved_cache[item] = self._reference_path(item)
                    resolved = resolved_cache[item]
                    if resolved:
                        files.add(resolved)

            collect(data)
            validation_path = self._resolve(data.get("validation_path")) or self.path.with_name("validation.json")
            validation = None
            if validation_path.is_file() and validation_path.is_relative_to(self.root):
                validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
                files.add(validation_path)
                collect(validation)
            checks_path = self.path.parent / "checks/index.json"
            checks = None
            if checks_path.is_file():
                checks = json.loads(checks_path.read_text(encoding="utf-8-sig"))
                files.add(checks_path)
                collect(checks)
            self.data, self.shots, self.by_id = data, shots, by_id
            self.allowed_files, self.validation = files, validation
            self.checks = checks
            human_path = self.path.with_name("human_evidence.json")
            self.human_evidence = None
            if human_path.is_file():
                self.human_evidence = json.loads(human_path.read_text(encoding="utf-8-sig"))
                files.add(human_path)
                collect(self.human_evidence)
            self.references = []
            for reference_path in (self.path.with_name("reference_evidence.json"),
                                   self.path.parent / "evidence/context_references.json"):
                if reference_path.is_file():
                    manifest = json.loads(reference_path.read_text(encoding="utf-8-sig"))
                    self.references.extend(manifest.get("references", []))
                    files.add(reference_path)
                    collect(manifest)
            self._mtime = mtime

    def file(self, value):
        declared = self._reference_path(value)
        if declared not in self.allowed_files:
            raise FileNotFoundError("File is not referenced by this catalog")
        resolved = self._resolve(value)
        if resolved is None:
            raise FileNotFoundError("Referenced file is missing or outside the project")
        return resolved

    def metadata(self):
        result = {key: value for key, value in self.data.items() if key not in {"shots", "appearance_groups"}}
        statuses, content_kinds = {}, set()
        for shot in self.shots:
            status = (shot.get("classification") or {}).get("status", "unprocessed")
            row = statuses.setdefault(status, {"shot_count": 0, "duration_seconds": 0})
            row["shot_count"] += 1
            row["duration_seconds"] += max(0, shot.get("end_seconds", 0) - shot.get("start_seconds", 0))
            kind = (shot.get("content") or {}).get("kind")
            if kind:
                content_kinds.add(kind)
        result["viewer"] = {
            "shot_count": len(self.shots), "read_only": True,
            "browser_seek_is_exact_pts": False,
            "catalog_path": self.path.relative_to(self.root).as_posix(),
            "status_counts": statuses, "content_kinds": sorted(content_kinds),
            "boundary_issue_count": sum(_has_boundary_issue(shot) for shot in self.shots),
            "appearance_group_count": len(self.data.get("appearance_groups", [])),
        }
        return result

    def appearance_groups(self, params):
        rows = self.data.get("appearance_groups", [])
        if params.get("source_id"):
            rows = [r for r in rows if r.get("source_id") == params["source_id"]]
        if params.get("q"):
            needle = params["q"].casefold()
            rows = [r for r in rows if needle in r.get("appearance_group_id", "").casefold()]
        offset = max(0, int(params.get("offset", 0)))
        limit = min(100, max(1, int(params.get("limit", 24))))
        return {"groups": rows[offset:offset+limit], "total": len(rows), "offset": offset, "limit": limit}

    def query(self, params):
        rows = self.shots
        for key in ("source_id", "physical_camera_id", "installation_id", "status", "appearance_group_id"):
            value = params.get(key)
            if not value:
                continue
            if key == "source_id":
                rows = [r for r in rows if r.get(key) == value]
            elif key == "appearance_group_id":
                rows = [r for r in rows if value in _as_list((r.get("classification") or {}).get("appearance_group_ids"))]
            elif key == "installation_id":
                rows = [r for r in rows if (r.get("classification") or {}).get(key) == value
                        or value in _as_list((r.get("classification") or {}).get("candidate_installation_ids"))]
            elif key == "physical_camera_id":
                installations = {item.get("installation_id") for item in self.data.get("installations", [])
                                 if item.get("physical_camera_id") == value and item.get("installation_id")}
                rows = [r for r in rows if (r.get("classification") or {}).get(key) == value
                        or (r.get("classification") or {}).get("installation_id") in installations
                        or installations.intersection(_as_list((r.get("classification") or {}).get("candidate_installation_ids")))]
            else:
                rows = [r for r in rows if (r.get("classification") or {}).get(key) == value]
        if params.get("content_kind"):
            rows = [r for r in rows if (r.get("content") or {}).get("kind") == params["content_kind"]]
        special = params.get("special")
        if special == "boundary":
            rows = [r for r in rows if _has_boundary_issue(r)]
        elif special == "duplicate":
            rows = [r for r in rows if r.get("duplicate_links")]
        elif special == "replay":
            rows = [r for r in rows if (r.get("content") or {}).get("replay_status") not in
                    (None, "", "unknown", "unresolved", "not_replay", "live", "not_detected", "unassessed")]
        elif special == "replay_unknown":
            rows = [r for r in rows if (r.get("content") or {}).get("replay_status") in
                    (None, "", "unknown", "unresolved", "unassessed")]
        if params.get("q"):
            needle = params["q"].casefold()
            rows = [r for r in rows if needle in json.dumps(r, ensure_ascii=False).casefold()]
        if params.get("at_seconds"):
            time = float(params["at_seconds"])
            rows = [r for r in rows if r.get("start_seconds", 0) <= time < r.get("end_seconds", 0)]
        total = len(rows)
        offset = max(0, int(params.get("offset", 0)))
        limit = min(200, max(1, int(params.get("limit", 48))))
        return {"shots": rows[offset:offset + limit], "total": total, "offset": offset,
                "limit": limit, "duration_seconds": sum(max(0, r.get("end_seconds", 0) - r.get("start_seconds", 0)) for r in rows)}

    def timeline(self, source_id):
        return [{"shot_id": r["shot_id"], "start_seconds": r.get("start_seconds"),
                 "end_seconds": r.get("end_seconds"),
                 "status": (r.get("classification") or {}).get("status", "unprocessed"),
                 "interval_type": r.get("interval_type", "shot"),
                 "boundary_issue": _has_boundary_issue(r)}
                for r in self.shots if r.get("source_id") == source_id]


def make_server(root, catalog_path="outputs/camera_classification/catalog.json", port=8767):
    catalog = CameraCatalog(root, catalog_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def _file(self, path):
            size = path.stat().st_size
            range_header = self.headers.get("Range")
            try:
                start, end = byte_range(range_header, size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206 if range_header else 200)
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            if path.suffix.lower() in {".html", ".css", ".js"}:
                content_type += "; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            parsed = urlsplit(self.path)
            params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                if parsed.path in ("/", "/index.html"):
                    return self._file(STATIC / "camera_classification.html")
                if parsed.path.startswith("/static/"):
                    filename = parsed.path[len("/static/"):]
                    if filename not in STATIC_FILES:
                        raise FileNotFoundError()
                    return self._file(STATIC / filename)
                catalog.reload()
                with catalog.lock:
                    if parsed.path == "/api/catalog":
                        return self._json(catalog.metadata())
                    if parsed.path == "/api/shots":
                        return self._json(catalog.query(params))
                    if parsed.path == "/api/appearance-groups":
                        return self._json(catalog.appearance_groups(params))
                    if parsed.path == "/api/references":
                        return self._json({"references": catalog.references})
                    if parsed.path == "/api/shot":
                        shot = catalog.by_id.get(params.get("shot_id"))
                        if shot is None:
                            raise FileNotFoundError("Shot not found")
                        return self._json(shot)
                    if parsed.path == "/api/timeline":
                        return self._json(catalog.timeline(params.get("source_id")))
                    if parsed.path == "/api/validation":
                        return self._json(catalog.validation or {"status": "missing"})
                    if parsed.path == "/api/checks":
                        return self._json(catalog.checks or {"cases": []})
                    if parsed.path == "/api/human-evidence":
                        return self._json(catalog.human_evidence or {"cases": []})
                    if parsed.path == "/api/file":
                        path = catalog.file(params.get("path"))
                    else:
                        raise FileNotFoundError()
                return self._file(path)
            except (ValueError, TypeError) as error:
                return self._json({"error": str(error)}, 400)
            except (FileNotFoundError, KeyError) as error:
                return self._json({"error": str(error) or "Not found"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.catalog = catalog
    return server


def write_index(root, output="outputs/camera_classification/index.html", port=8767):
    """A directly openable entry, with an explicit link to the local video server."""
    target = Path(root) / output
    target.parent.mkdir(parents=True, exist_ok=True)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>RinkSpace · 整场机位目录</title>
<style>body{{font:16px/1.8 system-ui,sans-serif;margin:8vh auto;max-width:850px;padding:24px;color:#17353e;background:#eff4f5}}a{{color:#056c68}}pre{{padding:20px;background:white;border-radius:8px;white-space:pre-wrap}}</style>
<h1>RinkSpace · 整场机位目录</h1>
<p><a href="http://127.0.0.1:{port}/">打开完整时间轴、机位证据和源视频</a></p>
<p>此查看页通过本地只读服务按需读取原始视频，保留源时间、PTS、邻帧与不确定项。全部编目与物理机位确认分别展示。</p>
<p>若本地服务尚未启动，在项目目录运行：</p>
<pre>.venv\\Scripts\\python.exe scripts/serve_camera_classification.py --port {port}</pre>
<p><a href="catalog.json">完整分类数据</a> · <a href="validation.json">检查结果</a></p></html>'''
    target.write_text(html, encoding="utf-8")
    return target
