"""One-time privacy inspection export; daily work uses the original single checkout.

The OneDrive working tree remains the source of project assets. This command never
creates a Git repository, commits, pushes, alters original files, or executes
packaged code. Array/pickle payloads and image pixels are not decoded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_PARTS = {".git", ".codex", ".agents", ".venv", "__pycache__", ".pytest_cache",
                 ".mypy_cache", ".ruff_cache", "node_modules", ".cache"}
COMPILED = {".obj", ".o", ".pdb", ".exp", ".dll", ".lib", ".exe", ".pyd",
            ".so", ".dylib", ".a", ".pyc", ".pyo", ".cubin", ".fatbin", ".ptx"}
TEXT = {".py", ".json", ".jsonl", ".md", ".txt", ".html", ".htm", ".js", ".ts",
        ".css", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".csv", ".tsv", ".log",
        ".ps1", ".cmd", ".bat", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu",
        ".cuh", ".cmake", ".xml", ".patch", ".diff", ".svg", ".sha256", ".rst"}
HOME = re.compile(r"(?P<prefix>[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+|/(?:Users|home)/)(?P<name>[^\\/\"'\s<>]+)", re.I)
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
CODEX_UUID = re.compile(r"\b01[0-9a-f]{6}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
ID_CONTEXT = re.compile(r"(?P<prefix>thread://|codex://threads/|codex-clipboard-|[/\\]+visualizations[/\\]+\d{4}[/\\]+\d{2}[/\\]+\d{2}[/\\]+)(?P<id>"+UUID+r")", re.I)
ID_FIELD = re.compile(r"""(?P<prefix>["']?(?P<key>(?:source[_-]?)?(?:thread|conversation|chat|turn|message|host)[_-]?ids?)["']?\s*[:=]\s*)(?P<q>["'])(?P<value>[^"'\r\n]{1,180})(?P=q)""", re.I)
HOST_QUERY = re.compile(r"(?P<prefix>[?&]hostId=)(?P<id>[A-Za-z0-9_.:-]+)", re.I)
PRIVATE_KEY_TOKEN = re.compile(r"""["']?(?:source[_-]?)?(?:thread|conversation|chat|turn|message|host)[_-]?ids?["']?\s*[:=]""", re.I)
QUOTE_KEY_TOKEN = re.compile(r"""["'][A-Za-z0-9_-]{0,100}(?:quote|verbatim)[A-Za-z0-9_-]{0,100}["']\s*:""", re.I)
QUOTE_FIELD = re.compile(r"""(?P<prefix>["'](?P<key>(?:source_user_quote|current_user_quote|user_clarification_quote|user_instruction_quote|private_chat_excerpt|verbatim_current_user_messages))["']\s*:\s*)(?P<q>["'])(?P<value>(?:\\.|[^"'\\\r\n])*)(?P=q)""", re.I)
PRIVACY_HINTS = ("users", "/home/", "thread://", "codex://threads/", "source_thread", "source_turn", "source_message", "source_host", "thread_id", "message_id", "host_id", "visualizations", "codex-clipboard", "user_quote", "clarification_quote", "instruction_quote", "verbatim", "private_chat", "pxl_")

REDACTED_QUOTE = "[private conversation text omitted]"
SAFE_ID = re.compile(r"(?:private[-_]|redact|withheld|not[_ -]?supplied|not[_ -]?available|unknown|none|null|^$)", re.I)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_kind(key):
    normalized = re.sub(r"[_-]", "", key).casefold()
    if normalized.startswith("source"):
        normalized = normalized[6:]
    if normalized.endswith("ids"):
        normalized = normalized[:-1]
    return normalized[:-2] if normalized in {x+"id" for x in ("thread", "conversation", "chat", "turn", "message", "host")} else None


def quote_key(key):
    normalized = re.sub(r"[_-]", "", key).casefold()
    return ("user" in normalized and ("quote" in normalized or "verbatim" in normalized)
            or normalized in {"privatechatexcerpt", "privateconversationtext"})


def text_decode(value, suffix):
    if suffix.casefold() not in TEXT and suffix not in {".gitignore", ".gitattributes"}:
        return None
    try:
        return value.decode("utf-8-sig") if value.startswith(b"\xef\xbb\xbf") else value.decode("utf-8")
    except UnicodeError:
        return None


class Export:
    def __init__(self, source, output, plan):
        self.source, self.output = source, output
        self.plan = plan
        self.aliases, self.counts = {}, Counter()
        self.quotes = set()
        self.overrides = {x["project_path"]: x for x in plan.get("overrides", [])}
        self.media = {x["project_path"]: x for x in plan.get("media_exclusions", [])}
        self.photo_names = {Path(p).name: x.get("public_alias", "private_reference") + ".withheld"
                            for p, x in self.media.items()}
        self.files, self.excluded, self.archive_changes = [], [], {}
        self.archive_targets = {}
        for entry in plan.get("archive_text_overrides", []):
            self.archive_targets.setdefault(entry["archive_project_path"], set()).add(entry["member"])
        self.private_long, self.private_re = {}, None
        self.discovery_bytes = 0
        self.scan_bytes = 0

    def alias(self, kind, value):
        value = str(value)
        if SAFE_ID.search(value):
            return value
        canonical = value.casefold()
        key = (kind, canonical)
        if key not in self.aliases:
            self.counts["alias_"+kind] += 1
            self.aliases[key] = f"private-{kind}-{self.counts['alias_'+kind]:05d}"
        return self.aliases[key]

    def redact_quote_value(self, value, discover=False):
        """Remove actual conversation text without changing technical data types.

        Some provenance keys containing verbatim are boolean qualifications;
        others contain structured message records with ordering, source clock,
        and confirmation facts. Only text bodies in those records are private.
        """
        if isinstance(value, str):
            if discover and len(value) >= 8:
                self.quotes.add(value)
            return REDACTED_QUOTE
        if isinstance(value, list):
            return [self.redact_quote_value(item, discover) for item in value]
        if isinstance(value, dict):
            return self.walk_json(value, discover, quoted_context=True)
        return value

    def walk_json(self, value, discover=False, quoted_context=False):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                kind = key_kind(key)
                is_body = quoted_context and key.casefold() in {
                    "text", "content", "message", "body", "quote", "utterance"
                }
                if quote_key(key) or is_body:
                    out[key] = self.redact_quote_value(item, discover)
                elif kind and item is not None and not isinstance(item, bool):
                    if isinstance(item, list):
                        out[key] = [self.alias(kind, x) for x in item]
                    elif isinstance(item, (str, int)):
                        out[key] = self.alias(kind, item)
                    else:
                        out[key] = self.walk_json(item, discover)
                else:
                    out[key] = self.walk_json(item, discover,
                                              quoted_context=quoted_context)
            return out
        if isinstance(value, list):
            return [self.walk_json(x, discover, quoted_context=quoted_context)
                    for x in value]
        return value

    def discover(self, text):
        lowered = text.casefold()
        if not any(hint in lowered for hint in PRIVACY_HINTS) and not CODEX_UUID.search(text):
            return
        self.discovery_bytes += len(text.encode("utf-8"))
        for match in CODEX_UUID.finditer(text):
            self.alias("thread", match.group())
        for match in ID_CONTEXT.finditer(text):
            self.alias("attachment" if "clipboard" in match["prefix"].casefold() else "thread", match["id"])
        for match in ID_FIELD.finditer(text):
            self.alias(key_kind(match["key"]) or "source", match["value"])
        for match in HOST_QUERY.finditer(text):
            self.alias("host", match["id"])
        if PRIVATE_KEY_TOKEN.search(text) or QUOTE_KEY_TOKEN.search(text):
            try:
                self.walk_json(json.loads(text), discover=True)
            except (ValueError, TypeError):
                pass

    def clean_string(self, text):
        text = HOME.sub(lambda m: m["prefix"] + "REDACTED", text)
        text = ID_FIELD.sub(lambda m: m["prefix"] + m["q"] +
                            self.alias(key_kind(m["key"]) or "source", m["value"]) + m["q"], text)
        text = HOST_QUERY.sub(lambda m: m["prefix"] + self.alias("host", m["id"]), text)
        text = ID_CONTEXT.sub(lambda m: m["prefix"] +
                              self.alias("attachment" if "clipboard" in m["prefix"].casefold() else "thread", m["id"]), text)
        text = CODEX_UUID.sub(lambda m: self.alias("thread", m.group()), text)
        # One compiled lookup avoids rescanning huge numeric documents per identifier.
        if self.private_re is not None:
            text = self.private_re.sub(lambda m: self.private_long[m.group().casefold()], text)
        for original, alias in self.photo_names.items():
            text = text.replace(original, alias)
        text = QUOTE_FIELD.sub(lambda m: m["prefix"] + m["q"] + REDACTED_QUOTE + m["q"], text)
        return text

    def sanitize(self, value, suffix):
        text = text_decode(value, suffix)
        if text is None:
            return value
        original = text
        lowered = text.casefold()
        if not any(hint in lowered for hint in PRIVACY_HINTS) and not CODEX_UUID.search(text):
            return value
        if PRIVATE_KEY_TOKEN.search(text) or QUOTE_KEY_TOKEN.search(text):
            try:
                parsed = json.loads(text)
                changed = self.walk_json(parsed)
                if parsed != changed:
                    text = json.dumps(changed, ensure_ascii=False, indent=2) + "\n"
            except (ValueError, TypeError):
                pass
        text = self.clean_string(text)
        return value if text == original else text.encode("utf-8")

    def payloads(self, path):
        suffix = path.suffix.casefold()
        if suffix == ".zip":
            relative = path.relative_to(self.output if path.is_relative_to(self.output) else self.source).as_posix()
            allowed = set(self.archive_targets.get(relative, ()))
            allowed.update(x["member"] for x in self.archive_changes.get(relative, {}).get("changed_members", {}).values())
            if not allowed:
                return
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.filename not in allowed:
                        continue
                    if not info.is_dir() and Path(info.filename).suffix.casefold() in TEXT:
                        yield info.filename, archive.read(info)
                    elif not info.is_dir() and info.filename.casefold().endswith(".gz"):
                        value = gzip.decompress(archive.read(info))
                        yield info.filename + "!/gzip-text", value
        elif suffix == ".gz":
            yield path.name + "!/gzip-text", gzip.decompress(path.read_bytes())
        elif suffix in TEXT or path.name in {".gitignore", ".gitattributes"}:
            yield path.name, path.read_bytes()

    def candidates(self):
        raw = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=self.source)
        paths = sorted(set(x.decode("utf-8") for x in raw.split(b"\0") if x))
        result = []
        for relative in paths:
            posix = PurePosixPath(relative.replace("\\", "/"))
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError("Unsafe Git candidate path")
            path = self.source.joinpath(*posix.parts)
            if not path.exists():
                self.excluded.append({"path": self.clean_string(posix.as_posix()), "reason": "missing_current_working_tree_file"})
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Only ordinary current working-tree files may be exported")
            reason = ("local_configuration_or_cache" if set(posix.parts) & BLOCKED_PARTS else
                      "compiled_artifact" if path.suffix.casefold() in COMPILED else
                      "private_reference_image_withheld" if posix.as_posix() in self.media else None)
            if reason:
                public_path = self.media[posix.as_posix()].get("public_alias") if posix.as_posix() in self.media else self.clean_string(posix.as_posix())
                self.excluded.append({"path": public_path, "reason": reason})
                continue
            result.append((posix.as_posix(), path))
        return result

    def export_zip(self, relative, source, target):
        changed, dropped = {}, []
        if relative not in self.archive_targets:
            shutil.copyfile(source, target)
            return
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                parts = PurePosixPath(info.filename).parts
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (info.filename.startswith("/") or "\\" in info.filename or ".." in parts
                        or kind not in (0, stat.S_IFREG, stat.S_IFDIR) or info.flag_bits & 1):
                    raise ValueError("Unsafe archive member")
                suffix = Path(info.filename).suffix.casefold()
                if suffix in COMPILED or set(parts) & BLOCKED_PARTS:
                    dropped.append(info.filename)
                    continue
                if info.filename not in self.archive_targets[relative]:
                    continue
                if suffix not in TEXT and suffix != ".gz":
                    continue
                original = archive.read(info)
                if suffix == ".gz":
                    inner = gzip.decompress(original)
                    public = self.sanitize(inner, ".json")
                    value = original if public == inner else gzip.compress(public, mtime=0)
                else:
                    value = self.sanitize(original, suffix)
                if value != original:
                    public_sha = hashlib.sha256(value).hexdigest()
                    name = ("files/" + public_sha + Path(info.filename).suffix
                            if re.fullmatch(r"files/[0-9a-f]{64}(?:\.[^/]+)?", info.filename)
                            else self.clean_string(info.filename))
                    changed[info.filename] = {"member": name, "sha256": public_sha, "bytes": len(value),
                                              "original_sha256": hashlib.sha256(original).hexdigest(),
                                              "value": value}
            if not changed and not dropped:
                shutil.copyfile(source, target)
                return
            with zipfile.ZipFile(target, "x") as public_zip:
                for info in archive.infolist():
                    if info.filename in dropped:
                        continue
                    if info.filename in changed:
                        replacement = changed[info.filename]
                        # Only the small archive containing changed text is rewritten.
                        new_info = zipfile.ZipInfo(replacement["member"], info.date_time)
                        new_info.external_attr = info.external_attr
                        new_info.compress_type = info.compress_type
                        public_zip.writestr(new_info, replacement["value"])
                    else:
                        with archive.open(info) as original_stream, public_zip.open(info, "w") as public_stream:
                            shutil.copyfileobj(original_stream, public_stream, 1024 * 1024)
        self.archive_changes[relative] = {
            "changed_members": {k: {a: b for a, b in v.items() if a != "value"} for k, v in changed.items()},
            "excluded_members": dropped, "bytes": target.stat().st_size,
            "members": len(zipfile.ZipFile(target).infolist()),
            "original_sha256": sha(source), "sha256": sha(target)}

    def patch_indexes(self):
        for path in sorted(self.output.rglob("*.json")):
            if path.name.startswith("PUBLIC_EXPORT"):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, UnicodeError):
                continue
            if not isinstance(doc, dict) or not isinstance(doc.get("assets"), list):
                continue
            schema = doc.get("schema", "")
            is_review = schema.startswith("rinkspace.offline_review")
            is_packed = schema.startswith("rinkspace.packed-assets")
            if not (is_review or is_packed):
                continue
            touched, assets = False, []
            for asset in doc["assets"]:
                if not isinstance(asset, dict):
                    assets.append(asset)
                    continue
                archive_relative = ((path.parent / asset["artifact"]).relative_to(self.output).as_posix()
                                    if is_review else doc.get("archive", ""))
                change = self.archive_changes.get(archive_relative)
                if change and asset.get("member") in change["excluded_members"]:
                    self.excluded.append({"path": self.clean_string(asset.get("logical_path", asset.get("path", ""))),
                                          "reason": "compiled_archive_member", "archive": archive_relative})
                    touched = True
                    continue
                replacement = change["changed_members"].get(asset.get("member")) if change else None
                if replacement:
                    asset.update({k: replacement[k] for k in ("member", "bytes", "sha256", "original_sha256")})
                    asset["privacy_derivative"] = True
                    touched = True
                assets.append(asset)
            if touched:
                doc["assets"] = assets
                doc["public_privacy_derivative"] = True
                if is_packed:
                    doc["source_bytes_unchanged"] = False
                for entry in doc.get("archives", []):
                    relative = (path.parent / entry["file"]).relative_to(self.output).as_posix()
                    if relative in self.archive_changes:
                        change = self.archive_changes[relative]
                        entry.update({k: change[k] for k in ("bytes", "members", "original_sha256", "sha256")})
                path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def scan(self):
        failures = []
        scanned = 0
        for path in sorted(self.output.rglob("*")):
            if not path.is_file():
                continue
            if set(path.relative_to(self.output).parts) & BLOCKED_PARTS:
                failures.append({"path": path.relative_to(self.output).as_posix(), "category": "forbidden_local_content"})
            for member, value in self.payloads(path):
                try:
                    text = value.decode("utf-8-sig")
                except UnicodeError:
                    continue
                scanned += 1
                self.scan_bytes += len(value)
                lowered = text.casefold()
                if not any(hint in lowered for hint in PRIVACY_HINTS) and not CODEX_UUID.search(text):
                    continue
                categories = set()
                if any(m["name"] != "REDACTED" for m in HOME.finditer(text)):
                    categories.add("unredacted_user_home")
                if CODEX_UUID.search(text):
                    categories.add("private_codex_identifier")
                if any(not SAFE_ID.search(m["value"]) for m in ID_FIELD.finditer(text)):
                    categories.add("private_identifier_field")
                if any(m["value"] != REDACTED_QUOTE for m in QUOTE_FIELD.finditer(text)):
                    categories.add("verbatim_private_quote_field")
                if self.private_re is not None and self.private_re.search(text):
                    categories.add("known_private_identifier")
                for original in self.photo_names:
                    if original in text:
                        categories.add("withheld_private_photo_reference")
                for category in categories:
                    failures.append({"path": self.clean_string(path.relative_to(self.output).as_posix()),
                                     "member": self.clean_string(member), "category": category})
        if (self.output / ".git").exists():
            failures.append({"path": ".git", "category": "git_database_or_pointer"})
        return {"archive_scope": "Only planned redacted ZIP text members re-scanned; other archive payload privacy review inherited from the prior complete source/archive audit. Unchanged ZIP files are byte copies; unchanged members of the two rewritten ZIPs are streamed without array/pickle interpretation.",
                "text_payloads": scanned, "text_bytes": self.scan_bytes,
                "remaining_privacy_findings": failures,
                "binary_scope": "Array/pickle payloads and image pixels were not decoded; unchanged binary/media assets retain prior audit limits."}

    def run(self):
        candidates = self.candidates()
        for relative, source in candidates:
            for member, value in self.payloads(source):
                try:
                    self.discover(value.decode("utf-8-sig"))
                except UnicodeError:
                    pass
            if relative in self.overrides:
                # Discover original private identifiers too; the mapping remains only in memory.
                self.discover(source.read_text(encoding="utf-8-sig"))
        for (kind, original), alias in self.aliases.items():
            if len(original) >= 8:
                self.private_long.setdefault(original, alias)
        if self.private_long:
            self.private_re = re.compile("|".join(re.escape(x) for x in sorted(self.private_long, key=len, reverse=True)), re.I)
        print(json.dumps({"stage": "discovered", "candidate_files": len(candidates), "private_aliases": len(self.aliases)}), flush=True)
        if self.output.exists():
            raise FileExistsError("Export output must be a new directory")
        self.output.mkdir(parents=True)
        for relative, source in sorted(candidates, key=lambda x: (x[1].suffix.casefold() != ".zip", x[0])):
            public_relative = self.clean_string(relative)
            target = self.output.joinpath(*PurePosixPath(public_relative).parts)
            if target.exists():
                raise ValueError("Public alias path collision")
            target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            original_sha = sha(source)
            if source.suffix.casefold() == ".zip":
                self.export_zip(relative, source, target)
            else:
                selected = source
                if relative in self.overrides:
                    override = self.overrides[relative]
                    if original_sha != override["original_sha256"]:
                        raise ValueError("Privacy override original source identity changed")
                    selected = Path(override["override_file"])
                    if sha(selected) != override["public_sha256"]:
                        raise ValueError("Privacy override identity changed")
                value = selected.read_bytes()
                if source.suffix.casefold() == ".gz":
                    inner = gzip.decompress(value)
                    public = self.sanitize(inner, ".json")
                    value = value if inner == public else gzip.compress(public, mtime=0)
                else:
                    value = self.sanitize(value, source.suffix if source.suffix else source.name)
                target.write_bytes(value)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("A source changed during export; candidate is incomplete")
            self.files.append({"path": public_relative, "original_sha256": original_sha,
                               "original_bytes": before.st_size})
        print(json.dumps({"stage": "copied", "files": len(self.files), "rewritten_archives": len(self.archive_changes)}), flush=True)
        self.patch_indexes()
        notice = ("# 脱敏公开审阅候选\n\n本目录是当前 OneDrive 核心资产的一次发布副本；OneDrive 原项目仍是资产来源，Data 中本目录不成为新的权威项目。\n\n"
                  "候选不含 Git 历史、Git 配置、本机会话配置或已安装环境，未提交、未推送。个人用户目录、私人会话标识和明确的私聊引句已脱敏；私人机械参考图和编译产物未包含。\n\n"
                  "脱敏文本、ZIP 成员和相应清单使用公开副本 SHA，并另存 original_sha256。其他来源记录中的历史 SHA 仍指原始实物，不能据此宣称本副本与本机运行身份完全相同。\n\n"
                  "关键代码、模型、统计输入与原始失败说明可离线审阅；原视频、外部模型/专用环境和部分原生后端仍有明确缺口。本候选不保证完整赛事拟合可以运行，也没有恢复已暂停的求解或生成视频。\n\n"
                  "从 [README.md](README.md) 和 [GitHub审阅说明](docs/GITHUB_REVIEW.md) 阅读。"
                  "详细变更、排除项及检查范围见 [PUBLIC_EXPORT_PROVENANCE.json](PUBLIC_EXPORT_PROVENANCE.json)。\n")
        (self.output / "PUBLIC_EXPORT_README.md").write_text(notice, encoding="utf-8")
        main_readme = self.output / "README.md"
        if main_readme.exists():
            main_readme.write_text("> 此目录是脱敏公开审阅候选。先读 [公开副本边界](PUBLIC_EXPORT_README.md)；"
                                   "原始 OneDrive 项目与历史运行身份保持独立。\n\n" +
                                   main_readme.read_text(encoding="utf-8-sig"), encoding="utf-8")
        for row in self.files:
            target = self.output / row["path"]
            row["public_sha256"], row["public_bytes"] = sha(target), target.stat().st_size
            row["changed"] = row["public_sha256"] != row["original_sha256"]
        scan = self.scan()
        report = {"schema": "rinkspace.public-export-provenance.v1",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_authority": "Current OneDrive project working-tree assets; originals untouched.",
                  "candidate_selection": "git ls-files -co --exclude-standard; only currently existing ordinary files; no index restoration",
                  "git_history_included": False, "git_created_or_committed_or_pushed": False,
                  "alias_policy": "Stable ordinal aliases within this export; private identifier map is memory-only and not distributed.",
                  "no_full_runtime_equivalence_claim": True, "no_fitting_resumed": True,
                  "original_sha256_meaning": "Original-source identity only; public_sha256 is the exported-file identity.",
                  "files": self.files, "exclusions": self.excluded,
                  "archive_changes": self.archive_changes,
                  "scan": scan,
                  "limitations": ["No original video, complete installed environment, or external licensed body weights added.",
                                  "Compiled artifacts removed, including native backend inside review code ZIP; rebuilding/acceptance still required.",
                                  "Unchanged scientific arrays are copied without array/pickle loading.",
                                  "Images retain prior metadata audit scope; no pixel-level privacy re-review performed.",
                                  "Historical source hashes in technical provenance remain historical; do not treat them as a public runtime certification."]}
        (self.output / "PUBLIC_EXPORT_PROVENANCE.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if scan["remaining_privacy_findings"]:
            raise ValueError("Public candidate retains privacy findings; inspect category/location-only provenance report")
        print(json.dumps({"ok": True, "files": len(self.files) + 2,
                          "changed_source_files": sum(x["changed"] for x in self.files),
                          "exclusions": len(self.excluded), "rewritten_archives": len(self.archive_changes),
                          "privacy_findings": 0, "output": str(self.output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides-plan", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or output == source or output.is_relative_to(source):
        raise ValueError("Use an existing source checkout and a new output directory outside it")
    plan = json.loads(args.overrides_plan.read_text(encoding="utf-8-sig"))
    Export(source, output, plan).run()


if __name__ == "__main__":
    main()
