"""Reproduce this delivery's two explicitly authorized programme inputs.

python scripts/acquire_sources.py --download --prepare
Original downloads are never transcoded or overwritten. Format 399 is the
1080p50 AV1 video-only source representation; audio is outside delivery A.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from rinkspace.io import dump_json, sha256
from rinkspace.workspace import import_source, prepare_source

SOURCES = [
    ("jin", "PVXWd7nLt8Q", "Shuxian JIN / Ankara 2026"),
    ("okada", "HZQx6HK8zR8", "Mei OKADA / Ankara 2026"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download", action="store_true", help="Download only the two authorized official videos")
    parser.add_argument("--prepare", action="store_true", help="Index complete decoded frames and make mapped proxies")
    args = parser.parse_args()
    args.root = args.root.resolve()
    inputs = args.root / "data/input"
    inputs.mkdir(parents=True, exist_ok=True)
    facts = []
    for key, video_id, title in SOURCES:
        url = f"https://www.youtube.com/watch?v={video_id}"
        path = inputs / f"{video_id}.mp4"
        info = None
        if not path.exists() and args.download:
            try:
                import yt_dlp
            except ImportError as exc:
                raise RuntimeError("Acquisition needs the optional yt-dlp package: pip install yt-dlp") from exc
            options = {"format": "399", "noplaylist": True, "continuedl": True,
                       "retries": 4, "fragment_retries": 4,
                       "outtmpl": str(inputs / "%(id)s.%(ext)s")}
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
        if not path.exists():
            print(f"Missing source: {path}. Use --download to acquire the authorized video.", file=sys.stderr)
            continue
        import_source(args.root, path, key=key, url=url, title=title)
        fact = {"key": key, "youtube_id": video_id, "source_url": url, "sha256": sha256(path),
                "size_bytes": path.stat().st_size, "format_id": "399", "audio": "video-only representation"}
        if info:
            fact.update({field: info.get(field) for field in ("title", "channel", "channel_id", "upload_date", "duration")})
        facts.append(fact)
        if args.prepare:
            prepare_source(args.root, key, progress=lambda record: print(record, flush=True))
    dump_json(args.root / "outputs/workspace/acquisition.json", {"sources": facts})
    return 0 if len(facts) == len(SOURCES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
