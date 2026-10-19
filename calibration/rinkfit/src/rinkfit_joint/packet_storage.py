"""Compact immutable packet metadata; numerical artifacts stay referenced."""
from pathlib import Path
import json

def write_packet(path, value):
    path=Path(path)
    with path.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(",",":"))
        stream.write("\n")
