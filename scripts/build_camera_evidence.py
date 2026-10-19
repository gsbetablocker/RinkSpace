"""Build classification evidence without updating historical survey/model files."""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse
from pathlib import Path
from rinkspace.camera_evidence import legacy_inventory, high_corner_correspondence, write_json, build_programme_mappings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--duplicates", action="store_true", help="Find local programme/broadcast pixel mappings, reusing evidence caches")
    args = parser.parse_args()
    root = args.root.resolve()
    out = _resolve_asset(root, root / "outputs/camera_classification/evidence")
    write_json(out / "legacy_links.json", legacy_inventory(root))
    write_json(out / "high_corner_multisession.json", high_corner_correspondence(root))
    if args.duplicates:
        build_programme_mappings(root, out)
    print(out)


if __name__ == "__main__":
    main()
