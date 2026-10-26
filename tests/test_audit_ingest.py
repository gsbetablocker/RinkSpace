"""Exercise both actual normalizers with named and unbound source materials."""
import csv
import gzip
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("module_name", ["data_ingest", "source_generation_v1"])
@pytest.mark.parametrize("explicit_surface", [False, True])
def test_source_surface_is_derived_only_from_this_rows_collection(tmp_path, monkeypatch,
                                                                module_name, explicit_surface):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "calibration/rinkfit/src"))
    module = importlib.import_module("rinkfit_joint." + module_name)
    source = tmp_path / "source"
    evidence = source / "evidence"
    evidence.mkdir(parents=True)
    named = dict(id="named", target_id="ice.C0", segments=[dict(uv=[[20., 30.]])])
    if explicit_surface:
        named["physical_surface"] = "ice"
    packet = dict(source=dict(native_pts=256256, time_base="1/12800",
                              programme_frame_index_1based=1, native_hw=[100, 100]),
                  features=[named], unbound_static_materials=[dict(
                      id="unidentified", target_id="ice.C0", segments=[dict(uv=[[40., 50.]])])])
    entries = []
    for name, value in {"s02/model/feature_registry.json": {"targets": {}},
                        "s02/packet.json": packet}.items():
        path = evidence / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        entries.append(dict(archive_path=name, path=str(path),
                            sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
    manifest = {"files": entries}
    for name in ("MANIFEST.json", "manifest.json"):
        (source / name).write_text(json.dumps(manifest))
    row = dict(shot=2, programme_index=1, native_pts=256256, time_base="1/12800",
               source_seconds=20.02, packets="s02/packet.json", stage="candidate", binding="",
               mask="", bound_representatives=1, unbound_representatives=1)
    with (source / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "normalized"
    if module_name == "data_ingest":
        module.normalize_dataset(source, output)
    else:
        generation = tmp_path / "generation.json"
        generation.write_text(json.dumps({"references": {}}))
        module.normalize_source_generation(source, output, source_generation=generation)
    with gzip.open(output / "observations.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert [r["physical_surface"] for r in rows] == ["ice", None]
    assert [r["source_uv"] for r in rows] == [[20., 30.], [40., 50.]]
    assert all(r["solver_eligible"] is False for r in rows)
    assert rows[1]["target_id"] is None
