"""Explicit same-byte archive resolution for historical Python source references.

Only an entered, SHA-bound registry authorizes aliases. Data, UV, masks, JSON
and numeric artifacts never use this compatibility path; file SHA stays physical.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path

_STATE = ContextVar("rinkfit_exact_source_code_aliases", default=None)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def resolve_dependency(ref, base=None):
    p=Path(ref.get("file",ref.get("path")))
    p=(p if p.is_absolute() else Path(base or ".")/p).resolve()
    expected=ref["sha256"]
    if p.is_file() and _sha(p)==expected:
        return p
    state=_STATE.get()
    key=(str(p).casefold(),expected)
    item=None if state is None else state["aliases"].get(key)
    if p.suffix.lower()!=".py" or item is None:
        raise ValueError("Changed exact source dependency: "+str(p))
    actual=Path(item["actual_file"]).resolve()
    if actual.suffix.lower()!=".py" or not actual.is_file() or _sha(actual)!=expected:
        raise ValueError("Archived original Python source bytes differ: "+str(actual))
    used=dict(legacy_path=str(p),sha256=expected,actual_file=str(actual),reason=item.get("reason"),
              actual_archive_bytes_verified=True,source_data_alias_allowed=False)
    if used not in state["used"]:state["used"].append(used)
    return actual


@contextmanager
def dependency_alias_context(registry_ref):
    if registry_ref is None:
        yield {"registry":None,"used":[]}
        return
    rp=Path(registry_ref.get("file",registry_ref.get("path"))).resolve()
    if _sha(rp)!=registry_ref["sha256"]:
        raise ValueError("Exact source-code alias registry changed")
    value=json.loads(rp.read_text(encoding="utf-8-sig"))
    if value.get("schema")!="rinkfit.exact_dependency_aliases.v1":
        raise ValueError("Unsupported source-code archive resolver schema")
    aliases={}
    for r in value["aliases"]:
        old=Path(r["legacy_path"]).resolve();actual=Path(r["actual_file"]).resolve()
        key=(str(old).casefold(),r["sha256"])
        if old.suffix.lower()!=".py" or actual.suffix.lower()!=".py" or key in aliases:
            raise ValueError("Only unambiguous Python-source archive aliases are supported")
        if not actual.is_file() or _sha(actual)!=r["sha256"]:
            raise ValueError("Source-code archive is not exact original bytes")
        aliases[key]=r
    state={"registry":dict(registry_ref),"aliases":aliases,"used":[]}
    token=_STATE.set(state)
    try:
        yield state
        if _sha(rp)!=registry_ref["sha256"]:
            raise ValueError("Source-code alias registry changed during verification")
    finally:_STATE.reset(token)


def current_alias_usage():
    state=_STATE.get()
    return [] if state is None else list(state["used"])
