"""Prepare the pinned FootMR/CoTracker source subsets in external Data.

Default: verify existing files only. --fetch downloads missing pinned source
files and restores the recorded namespace glue. Existing bytes are never
overwritten. This does not install packages, fetch weights or run inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("body-runtime-sources.json")


def prepare(component: str, data_root: Path, *, fetch: bool = False) -> dict:
    data = Path(data_root).resolve()
    project = PROJECT.resolve()
    if data.is_relative_to(project) or project.is_relative_to(data):
        raise ValueError("Data must be separate from the source worktree")
    entry = json.loads(MANIFEST.read_text(encoding="utf8"))["components"][component]
    destination = (data / entry["data_path"]).resolve()
    if not destination.is_relative_to(data):
        raise ValueError("Source destination escapes Data")
    installed = 0
    rows = entry["files"] + entry.get("integration_files", [])
    for row in rows:
        target = (destination / row["file"]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Source member escapes its component directory")
        if target.exists():
            raw = target.read_bytes()
        elif not fetch:
            raise FileNotFoundError(f"{target}; use --fetch to obtain missing pinned sources")
        elif "content_utf8" in row:
            raw = row["content_utf8"].encode("utf8")
        else:
            repository = entry["repository"].removeprefix("https://github.com/")
            url = f"https://raw.githubusercontent.com/{repository}/{entry['commit']}/{row['file']}"
            request = Request(url, headers={"User-Agent": "RinkSpace pinned runtime preparation"})
            with urlopen(request, timeout=60) as response:
                raw = response.read()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Source identity mismatch; preserved without overwrite: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            installed += 1
    return dict(component=component, commit=entry["commit"], source_directory=str(destination),
                verified_files=len(rows), created_files=installed, weights_downloaded=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--component", choices=["footmr", "cotracker"], action="append")
    parser.add_argument("--fetch", action="store_true", help="Obtain missing pinned files; never replace existing files")
    args = parser.parse_args()
    for component in args.component or ["footmr", "cotracker"]:
        print(json.dumps(prepare(component, args.data_root, fetch=args.fetch)))


if __name__ == "__main__":
    main()
