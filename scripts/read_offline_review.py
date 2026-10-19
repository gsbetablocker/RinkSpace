"""Read archived review evidence without opening Data or running packaged code.

The default check reads only ZIP central directories. Use --get to inspect an
asset, --get QUERY --output ABSOLUTE_FILE to write one verified asset, or
--extract ABSOLUTE_NEW_DIRECTORY to unpack evidence outside this repository.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "events/ankara_2026/programmes/okada_mei_sp/review/manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SUFFIX = re.compile(r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?)?\Z")


class ReviewError(ValueError):
    """Invalid, ambiguous, missing, or unsafe offline review evidence."""


def _size(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewError(f"{label} must be a non-negative integer")
    return value


def _key(value):
    return value.replace("\\", "/").casefold()


def _safe_member(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/") or PureWindowsPath(name).drive
            or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))):
        raise ReviewError(f"Unsafe ZIP member: {name!r}")


class Review:
    """Manifest-only lookup plus explicit, verified access to archived bytes.

    find(query) returns asset metadata; read(query) returns verified bytes.
    check() reads metadata only, without decompressing or hashing payloads.
    write(query, absolute_file) never overwrites an existing file. extract()
    requires a new absolute directory outside this repository and writes a
    resource_map.json mapping original and logical paths to extracted files.
    """

    def __init__(self, manifest=DEFAULT_MANIFEST):
        self.manifest_path = Path(manifest).resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(self.manifest, dict) or not isinstance(self.manifest.get("schema"), str) or not self.manifest["schema"]:
            raise ReviewError("Manifest must have a non-empty schema string")
        archives = self.manifest.get("archives")
        assets = self.manifest.get("assets")
        if not isinstance(archives, list) or not isinstance(assets, list):
            raise ReviewError("Manifest archives and assets must be lists")
        self.archives = {}
        for entry in archives:
            if not isinstance(entry, dict):
                raise ReviewError("Archive entry must be an object")
            name = entry.get("file")
            if (not isinstance(name, str) or not name or name in (".", "..")
                    or any(c in name for c in "/\\:\x00") or PureWindowsPath(name).drive):
                raise ReviewError(f"Archive must be a file beside the manifest: {name!r}")
            if name in self.archives:
                raise ReviewError(f"Duplicate archive: {name}")
            _size(entry.get("bytes"), f"{name} bytes")
            members = entry.get("members")
            if not (isinstance(members, list) and all(isinstance(x, str) for x in members)):
                _size(members, f"{name} members")
            self.archives[name] = dict(entry)
        self.assets = []
        self._paths = {}
        for entry in assets:
            if not isinstance(entry, dict):
                raise ReviewError("Asset entry must be an object")
            asset = dict(entry)
            for key in ("logical_path", "original_path", "sha256", "artifact", "member", "role"):
                if not isinstance(asset.get(key), str) or not asset[key]:
                    raise ReviewError(f"Asset must have a non-empty {key}")
            _size(asset.get("bytes"), f"{asset['logical_path']} bytes")
            if not SHA256.fullmatch(asset["sha256"]):
                raise ReviewError(f"Invalid SHA-256: {asset['logical_path']}")
            if asset["artifact"] not in self.archives:
                raise ReviewError(f"Unknown archive: {asset['artifact']}")
            _safe_member(asset["member"])
            prefix = "files/" + asset["sha256"]
            if not asset["member"].startswith(prefix) or not SUFFIX.fullmatch(asset["member"][len(prefix):]):
                raise ReviewError(f"Asset member must be files/<sha256><suffix>: {asset['member']}")
            self.assets.append(asset)
            for key in ("logical_path", "original_path"):
                self._paths.setdefault(_key(asset[key]), []).append(asset)
        self._payloads = {}
        for asset in self.assets:
            key = (asset["artifact"], asset["member"])
            previous = self._payloads.setdefault(key, asset)
            if (previous["sha256"], previous["bytes"]) != (asset["sha256"], asset["bytes"]):
                raise ReviewError(f"Conflicting metadata for {key}")

    def find(self, query):
        """Resolve a logical/original path or a unique SHA-256 prefix."""
        if not isinstance(query, str) or not query:
            raise ReviewError("Query must be a non-empty path or SHA-256 prefix")
        matches = self._paths.get(_key(query), [])
        if not matches and re.fullmatch(r"[0-9a-fA-F]{1,64}", query):
            matches = [a for a in self.assets if a["sha256"].startswith(query.lower())]
        if not matches:
            raise ReviewError(f"Asset not found in manifest: {query}")
        if len({a["sha256"] for a in matches}) != 1:
            raise ReviewError(f"Ambiguous asset query: {query}")
        return dict(matches[0])

    @contextmanager
    def _zip(self, name):
        entry = self.archives[name]
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ReviewError(f"Archive leaves the manifest directory: {name}")
        if path.stat().st_size != entry["bytes"]:
            raise ReviewError(f"Archive byte size changed: {name}")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = set()
            for info in infos:
                _safe_member(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ReviewError(f"ZIP links or special files are forbidden: {info.filename}")
                if info.flag_bits & 1:
                    raise ReviewError(f"Encrypted ZIP member is unsupported: {info.filename}")
                if info.filename in names:
                    raise ReviewError(f"Duplicate ZIP member: {info.filename}")
                names.add(info.filename)
            expected = entry["members"]
            if isinstance(expected, list):
                if len(expected) != len(infos) or set(expected) != names:
                    raise ReviewError(f"ZIP central directory changed: {name}")
            elif expected != len(infos):
                raise ReviewError(f"ZIP member count changed: {name}")
            yield archive

    @staticmethod
    def _info(archive, asset):
        try:
            info = archive.getinfo(asset["member"])
        except KeyError as exc:
            raise ReviewError(f"Missing asset member: {asset['member']}") from exc
        if info.is_dir() or info.file_size != asset["bytes"]:
            raise ReviewError(f"Asset uncompressed size changed: {asset['member']}")
        return info

    def check(self):
        """Check archive sizes and central directories; do not read payloads."""
        for name in self.archives:
            with self._zip(name) as archive:
                for asset in self._payloads.values():
                    if asset["artifact"] == name:
                        self._info(archive, asset)
        return dict(ok=True, archives=len(self.archives), assets=len(self.assets),
                    unique_members=len(self._payloads),
                    archive_bytes=sum(a["bytes"] for a in self.archives.values()),
                    uncompressed_bytes=sum(a["bytes"] for a in self._payloads.values()),
                    content_hashed=False)

    @staticmethod
    def _verify(asset, count, digest):
        if count != asset["bytes"] or digest != asset["sha256"]:
            raise ReviewError(f"Asset content hash changed: {asset['logical_path']}")

    def read(self, query):
        """Return bytes after ZIP CRC, byte size, and SHA-256 verification."""
        asset = self.find(query)
        with self._zip(asset["artifact"]) as archive:
            data = archive.read(self._info(archive, asset))
        self._verify(asset, len(data), hashlib.sha256(data).hexdigest())
        return data

    @staticmethod
    def _absolute_output(output, bulk=False):
        path = Path(output)
        if not path.is_absolute():
            raise ReviewError("Output must be an absolute path")
        if path.is_symlink():
            raise ReviewError("Output must not be a symbolic link")
        path = path.resolve()
        if path.is_relative_to(PROJECT):
            if bulk:
                raise ReviewError("Bulk extraction inside this repository is forbidden; use a Data directory")
            units = lambda s: len(str(s).encode("utf-16-le")) // 2
            if units(path) >= 260 or any(units(part) > 255 for part in path.parts):
                raise ReviewError("Output exceeds the OneDrive Windows path limits")
        return path

    def _write_asset(self, asset, output):
        created = False
        try:
            with self._zip(asset["artifact"]) as archive:
                info = self._info(archive, asset)
                with archive.open(info) as source, output.open("xb") as target:
                    created = True
                    digest = hashlib.sha256()
                    count = 0
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                self._verify(asset, count, digest.hexdigest())
        except BaseException:
            if created:
                output.unlink()
            raise

    def write(self, query, output):
        """Write one verified asset to an absolute file without overwriting."""
        asset = self.find(query)
        path = self._absolute_output(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_asset(asset, path)
        return path

    def extract(self, output):
        """Extract unique payloads flat, outside the checkout, with a path map."""
        path = self._absolute_output(output, bulk=True)
        if path.exists():
            raise ReviewError(f"Extraction requires a new directory: {path}")
        self.check()
        resources = {}
        targets = {}
        for asset in self.assets:
            target = path / asset["member"]
            targets.setdefault(target, asset)
            for key in ("logical_path", "original_path"):
                old = resources.setdefault(asset[key], str(target))
                if old != str(target):
                    raise ReviewError(f"Conflicting resource map path: {asset[key]}")
        path.mkdir(parents=True)
        (path / "files").mkdir()
        for target, asset in targets.items():
            self._write_asset(asset, target)
        map_path = path / "resource_map.json"
        with map_path.open("x", encoding="utf-8") as target:
            json.dump(resources, target, ensure_ascii=False, indent=2)
            target.write("\n")
        return dict(directory=str(path), unique_files=len(targets), resource_map=str(map_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Metadata-only check (default); no decompression or hashes")
    action.add_argument("--get", metavar="PATH_OR_SHA_PREFIX", help="Show matching asset metadata; add --output to write its bytes")
    action.add_argument("--extract", type=Path, metavar="ABSOLUTE_NEW_DIRECTORY", help="Unpack unique files outside this repository and write resource_map.json")
    parser.add_argument("--output", type=Path, metavar="ABSOLUTE_FILE", help="Write the --get asset without overwriting")
    args = parser.parse_args(argv)
    if args.output is not None and args.get is None:
        parser.error("--output requires --get")
    try:
        review = Review(args.manifest)
        if args.get is not None:
            result = review.find(args.get)
            if args.output is not None:
                result["written_to"] = str(review.write(args.get, args.output))
        elif args.extract is not None:
            result = review.extract(args.extract)
        else:
            result = review.check()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        print(f"Offline review error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
