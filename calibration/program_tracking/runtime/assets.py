"""Resolve legacy records at read time without rewriting frozen inputs."""
from __future__ import annotations

import json
from pathlib import Path

from rinkspace.storage import AssetStore, resolve_asset

ROOT = Path(__file__).resolve().parents[3]
STORAGE = AssetStore(ROOT)


def asset_path(value):
    return resolve_asset(ROOT, value)


def read_json(value):
    return json.loads(asset_path(value).read_text(encoding="utf-8-sig"))


def source_record(record, field="source_path"):
    """Return a copy with an operational path and the untouched recorded path."""
    result = dict(record)
    if field in result:
        result.setdefault(field + "_original", result[field])
        result[field] = str(STORAGE.source(result[field]))
    return result


def same_source(first, second):
    return STORAGE.source(first) == STORAGE.source(second)


def data_output(value):
    """Keep generated products in registered data, including legacy callers."""
    path = asset_path(value)
    if STORAGE.configured and not path.is_relative_to(STORAGE.data):
        raise ValueError("Generated products must use the registered data root")
    return path
