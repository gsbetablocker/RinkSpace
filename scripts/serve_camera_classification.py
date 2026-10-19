"""Serve the read-only full-event camera catalog and original local video tracks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rinkspace.camera_viewer import make_server, write_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--catalog", default="outputs/camera_classification/catalog.json")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--write-index", action="store_true")
    args = parser.parse_args()
    if args.write_index:
        print(write_index(args.root, port=args.port))
    server = make_server(args.root, args.catalog, args.port)
    print(f"Camera classification: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
