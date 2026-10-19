"""Small deterministic file helpers; source videos are never modified."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from .path_policy import guard_project_write
from .packed_assets import materialize_project_asset, read_project_json


def dump_json(path: str | Path, data: Any) -> None:
    p = guard_project_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    local = Path(path)
    physical = local if local.is_file() else materialize_project_asset(path)
    with physical.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    local = Path(path)
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8-sig"))
    return read_project_json(path)
