"""Compact immutable packet metadata; numerical artifacts stay referenced."""
from pathlib import Path
import json

def write_packet(path, value):
    path=Path(path)
    if isinstance(value,dict) and "dense_source_binding" in value:
        from .dense_joint_packet import compact_source_registry
        value=compact_source_registry(value,path)
    with path.open("x",encoding="utf8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(",",":"))
        stream.write("\n")
