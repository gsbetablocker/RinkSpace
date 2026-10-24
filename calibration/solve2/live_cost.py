"""Display only accepted step numbers and costs from the running solver."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def accepted_rows(data: bytes):
    for line in data.splitlines():
        try:
            item = json.loads(line)
            if item.get("kind") != "accepted":
                continue
            yield int(item["accepted_steps_this_process"]), float(item["event"]["cost"])
        except (ValueError, KeyError, TypeError):
            continue


def follow(control: Path, *, once: bool = False):
    source = None
    reader = None
    pending = b""
    try:
        while True:
            try:
                selected = Path(json.loads(control.read_text(encoding="utf-8"))["source"])
            except (OSError, ValueError, KeyError, TypeError):
                selected = source
            if selected is not None and selected != source:
                try:
                    new_reader = selected.open("rb")
                except OSError:
                    new_reader = None
                if new_reader is not None:
                    if reader is not None:
                        reader.close()
                    source, reader = selected, new_reader
                    content = reader.read()
                    complete, separator, pending = content.rpartition(b"\n")
                    if not separator:
                        complete, pending = b"", content
                    for step, cost in list(accepted_rows(complete))[-20:]:
                        print(f"第 {step} 步    cost {cost:.12g}", flush=True)
            if reader is not None:
                chunk = reader.read(1 << 20)
                if chunk:
                    complete, separator, tail = (pending + chunk).rpartition(b"\n")
                    if separator:
                        pending = tail
                        for step, cost in accepted_rows(complete):
                            print(f"第 {step} 步    cost {cost:.12g}", flush=True)
                    else:
                        pending += chunk
                    continue
            if once:
                return
            time.sleep(0.25)
    finally:
        if reader is not None:
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        follow(args.control, once=args.once)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
