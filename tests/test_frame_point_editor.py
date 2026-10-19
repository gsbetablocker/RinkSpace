"""Exercise real point packets only through isolated Data test copies."""
from copy import deepcopy
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import re
import shutil
import threading

import numpy as np
import pytest

from calibration.program_tracking import frame_point_editor as editor
from calibration.program_tracking.extract_selected_frame_review import make_binding
from calibration.program_tracking.validate_opening_frame_batch import assemble, read, sha


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/CodexData/RinkSpace_Data")
RELATIVE_BATCH = Path(
    "events/ankara_2026/programmes/okada_mei_sp/"
    "source_datasets/frame_by_frame_review_20261013/continuation_101_200_50fps"
)
MODEL = ROOT / "events/ankara_2026/models/framewise_8a28_20261013"


def test_circle_display_preserves_source_points_and_missing_arc():
    from calibration.program_tracking.render_single_frame_source_review import display_segments
    g = dict(kind='projected_physical_circle_ellipse', center_uv=[100, 100],
             axes_px=[100, 40], angle_degrees=0)
    def arc(degrees):
        t = np.deg2rad(degrees)
        return np.c_[100+50*np.cos(t), 100+20*np.sin(t)].tolist()
    packet = {'features': [dict(id='ice.C0.outer', geometric_interpretation=g,
                  segments=[dict(uv=arc([20, 70]), closed=False),
                            dict(uv=arc([160, 200]), closed=False)])]}
    before = deepcopy(packet)
    shown = display_segments(packet)
    assert packet == before
    assert len(shown) == 2 and sum(len(s['uv']) for s in shown) == 4
    for segment, bounds in zip(shown, [(20, 70), (160, 200)]):
        drawn = np.array(segment['draw_uv'])
        assert len(drawn) > len(segment['uv'])
        ellipse = ((drawn[:, 0]-100)/50)**2 + ((drawn[:, 1]-100)/20)**2
        assert np.allclose(ellipse, 1, atol=1e-5)
        angle = np.rad2deg(np.unwrap(np.arctan2((drawn[:, 1]-100)/20, (drawn[:, 0]-100)/50)))
        assert angle.min() >= bounds[0]-1e-3 and angle.max() <= bounds[1]+1e-3


def test_native_material_trace_display_does_not_add_observations():
    from calibration.program_tracking.render_single_frame_source_review import display_segments
    trace=[[100,100],[102,98],[105,100],[108,105],[100,110]]
    packet={'features':[dict(id='Ad1.print.O',segments=[dict(uv=[trace[0],trace[2],trace[-1]],
        closed=True,draw_current_native_material_readout=True,source_native_readout_uv=trace,
        pixels_measured_in_this_frame=True,source_evidence_measured_in_this_frame=True)])]}
    before=deepcopy(packet)
    shown=display_segments(packet)
    assert shown[0]['draw_uv']==trace
    assert len(shown[0]['uv'])==3
    assert packet==before
    packet['features'][0]['manual_deletion_applied']=True
    assert display_segments(packet)[0]['draw_uv']==packet['features'][0]['segments'][0]['uv']


@pytest.fixture(scope="module")
def real_inputs():
    batch = ROOT / RELATIVE_BATCH
    if not (batch / "frames/000113.json").exists():
        pytest.skip("Requires the native 101–200 review fixture")
    packets = {i: read(batch / f"frames/{i:06d}.json") for i in (113, 139)}
    registry, scene, templates = [
        read(MODEL / n)
        for n in ("feature_registry.json", "scene.json", "ad2_material_template.json")
    ]
    return batch, packets, registry, scene, templates


def packet_vertices(packet):
    return {pid: tuple(item[3]) for pid, item in editor.point_inventory(packet).items()}


def corner_point(packet, role="left_top"):
    for pid, (feature, si, vi, uv) in editor.point_inventory(packet).items():
        segment = feature["segments"][si]
        binding = segment.get("point_bindings", [{}] * len(segment["uv"]))[vi]
        if binding.get("physical_target") == "Ad2." + role:
            return pid, feature, si, vi, uv
    raise AssertionError("Expected a native semantic corner fixture")


def ice_point_id(packet):
    return next(pid for pid, (f, _, _, _) in editor.point_inventory(packet).items()
                if f["id"].startswith("ice.") and sum(len(s["uv"]) for s in f["segments"]) > 2)


def bind_and_assemble(packet, registry, scene, templates, path):
    binding = make_binding(packet, path, registry)
    rows, counts, corners = assemble(packet, binding, registry, scene, templates)
    return binding, rows, counts, corners


def test_delete_ice_point_preserves_other_coordinates_and_stable_ids(real_inputs, tmp_path):
    _, packets, registry, scene, templates = real_inputs
    original = deepcopy(packets[113]); before = packet_vertices(original)
    pid = ice_point_id(original)
    edited = editor.apply_operations(original, [{"type": "delete", "point_id": pid}], registry, "delete-ice")
    assert packet_vertices(original) == before
    assert packet_vertices(edited) == {p: uv for p, uv in before.items() if p != pid}
    assert edited["point_count"] == original["point_count"] - 1
    path = tmp_path / "packet.json"; path.write_bytes(editor.json_bytes(edited))
    binding, _, counts, _ = bind_and_assemble(edited, registry, scene, templates, path)
    assert sum(counts.values()) == edited["point_count"]
    assert set(p for o in binding["observations"] for p in o["source_point_ids"]) == set(before) - {pid}


def test_unbound_material_can_be_reviewed_and_deleted_without_entering_fit(real_inputs, tmp_path):
    _, packets, registry, scene, templates = real_inputs
    original = deepcopy(packets[113])
    original['unbound_static_materials'] = [dict(id='red_material', allowed_for_fit=False,
        binding_unresolved=True, segments=[dict(uv=[[100,100],[110,100],[110,110],[100,110]],
        point_ids=['unbound:0','unbound:1','unbound:2','unbound:3'],closed=True,vertex_count=4)])]
    edited = editor.apply_operations(original, [{'type':'delete','point_id':'unbound:1'}], registry, 'unbound-delete')
    assert edited['point_count'] == original['point_count']
    assert edited['source_point_count'] == original['point_count'] + 3
    assert 'unbound:1' not in editor.point_inventory(edited)
    path = tmp_path / 'unbound.json'; path.write_bytes(editor.json_bytes(edited))
    binding, _, counts, _ = bind_and_assemble(edited, registry, scene, templates, path)
    assert sum(counts.values()) == original['point_count']
    assert not any(pid.startswith('unbound:') for o in binding['observations'] for pid in o['source_point_ids'])
    extra = 500 - edited['source_point_count']
    adds = [{'type':'add','target_id':'ice.R4:outer','uv':[100+i/10,100]} for i in range(extra+1)]
    assert editor.apply_operations(edited,adds[:-1],registry,'up-to-cap')['source_point_count']==500
    with pytest.raises(ValueError,match='500'):
        editor.apply_operations(edited,adds,registry,'over-cap')


def test_delete_corner_and_adjacent_edge_reindexes_role_and_protected_metadata(real_inputs, tmp_path):
    _, packets, registry, scene, templates = real_inputs
    original = deepcopy(packets[139]); pid, feature, si, vi, _ = corner_point(original)
    old_segment = feature["segments"][si]
    assert isinstance(old_segment["corner_vertex_indices"], dict)
    # Removing an interior edge point must preserve the named corner and its index.
    interior_vi = next(i for i in range(len(old_segment["uv"])) if i != vi)
    interior_pid = old_segment.get("point_ids", [f"{feature['id']}/segment/{si}/vertex/{i}" for i in range(len(old_segment["uv"]))])[interior_vi]
    edge_edited = editor.apply_operations(original, [{"type": "delete", "point_id": interior_pid}], registry, "delete-edge")
    corner_after = corner_point(edge_edited)
    assert corner_after[0] == pid
    for f in edge_edited["features"]:
        for s in f["segments"]:
            for role, index in s.get("corner_vertex_indices", {}).items():
                assert s["point_bindings"][index]["physical_target"] == "Ad2." + role
            assert all(0 <= i < len(s["uv"]) for i in s.get("protected_vertex_indices", []))
    edited = editor.apply_operations(original, [{"type": "delete", "point_id": pid}], registry, "delete-corner")
    assert pid not in editor.point_inventory(edited)
    assert all("left_top" not in s.get("corner_vertex_indices", {})
               for f in edited["features"] for s in f["segments"])
    path = tmp_path / "packet.json"; path.write_bytes(editor.json_bytes(edited))
    _, _, _, corners = bind_and_assemble(edited, registry, scene, templates, path)
    assert all(c["target_id"] != "Ad2.left_top" for c in corners)


@pytest.mark.parametrize("target", ["ice.R4:outer", "Ad2.print.U", "Ad2.top_edge", "board.top.upper_long_side", "Ad2.right_top"])
def test_manual_points_have_human_provenance_and_exact_typed_targets(real_inputs, tmp_path, target):
    _, packets, registry, scene, templates = real_inputs
    original = packets[113]
    edited = editor.apply_operations(original, [{"type": "add", "target_id": target, "uv": [600.123456, 650.987654]}], registry, "add-point")
    manual = edited["features"][-1]; segment = manual["segments"][0]
    assert manual["manual_source_observation"] is True
    assert manual["physical_target_descriptor"] == registry["targets"][target]
    assert segment["measurement_origin"] == "user_manual_native_image_click"
    assert segment["source_evidence"][0]["origin"] == "human_native_image_click"
    assert segment["source_evidence"][0]["native_pts"] == original["source"]["native_pts"]
    assert segment["source_evidence"][0]["measurement_uncertainty_not_estimated"] is True
    assert not any(k in segment for k in ("sigma_px", "transition_iqr_px", "localization_sigma_px"))
    assert segment["uv"] == [[600.1235, 650.9877]]
    assert edited["user_accepted"] is False
    path = tmp_path / "packet.json"; path.write_bytes(editor.json_bytes(edited))
    binding, _, _, _ = bind_and_assemble(edited, registry, scene, templates, path)
    manual_binding = next(o for o in binding["observations"] if o["source_feature_id"] == manual["id"])
    assert manual_binding["source_point_ids"] == segment["point_ids"]
    assert manual_binding["target_id"] == target
    assert manual_binding["point_correspondence"] == (registry["targets"][target]["kind"] == "ad_corner")


def test_duplicate_corner_refused_and_explicit_replacement_allowed(real_inputs):
    _, packets, registry, _, _ = real_inputs
    original = packets[113]; pid, _, _, _, uv = corner_point(original)
    add = {"type": "add", "target_id": "Ad2.left_top", "uv": uv}
    with pytest.raises(ValueError, match="角点"):
        editor.apply_operations(original, [add], registry, "duplicate-corner")
    edited = editor.apply_operations(original, [{"type": "delete", "point_id": pid}, add], registry, "replace-corner")
    assert edited["point_count"] == original["point_count"]
    assert pid not in editor.point_inventory(edited)
    assert edited["features"][-1]["target_id"] == "Ad2.left_top"


@pytest.mark.parametrize("operation", [
    {"type": "add", "target_id": "unknown", "uv": [10, 10]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [-.1, 20]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [1920, 20]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [20, 1080]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [float("nan"), 20]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [float("inf"), 20]},
    {"type": "add", "target_id": "ice.R4:outer", "uv": [True, 20]},
    {"type": "delete", "point_id": "absent"},
    {"type": "teleport", "uv": [10, 20]},
])
def test_bad_operations_do_not_mutate_source(real_inputs, operation):
    _, packets, registry, _, _ = real_inputs
    original = deepcopy(packets[113]); before = deepcopy(original)
    with pytest.raises(ValueError):
        editor.apply_operations(original, [operation], registry, "bad-input")
    assert original == before


def test_hard_ceiling_refuses_501_but_allows_500(real_inputs):
    _, packets, registry, _, _ = real_inputs
    original = packets[113]
    count = 500 - original["point_count"]
    operations = [{"type": "add", "target_id": "ice.R4:outer", "uv": [100 + i / 10, 100]} for i in range(count + 1)]
    assert editor.apply_operations(original, operations[:-1], registry, "at-ceiling")["point_count"] == 500
    with pytest.raises(ValueError, match="500"):
        editor.apply_operations(original, operations, registry, "over-ceiling")


@pytest.fixture
def store(real_inputs, tmp_path):
    assert tmp_path.resolve().is_relative_to(DATA.resolve()), "Run these tests with --basetemp under RinkSpace_Data"
    batch, packets, registry, _, _ = real_inputs
    dataset = tmp_path / "dataset"; media = tmp_path / "media"; model = tmp_path / "model"
    for path in (dataset / "frames", dataset / "bindings", media / "frames", media / "navigation", media / "overlays", model):
        path.mkdir(parents=True, exist_ok=True)
    source_media = Path(packets[113]["source"]["image_path"]).parent.parent
    for index, packet in packets.items():
        destination = dataset / f"frames/{index:06d}.json"
        destination.write_bytes(editor.json_bytes(packet))
        (dataset / f"bindings/{index:06d}.json").write_bytes(editor.json_bytes(make_binding(packet, destination, registry)))
        # Copy only the two source images needed by actual native overlay persistence.
        shutil.copyfile(Path(packet["source"]["image_path"]), media / f"frames/{index:06d}.png")
        (media / f"overlays/{index:06d}.png").write_bytes(b"previous-overlay")
    cameras = np.load(source_media / "camera_seeds.npz", allow_pickle=False)
    wanted = np.isin(cameras["programme_indices"], list(packets))
    np.savez(media / "camera_seeds.npz", programme_indices=cameras["programme_indices"][wanted], camera25=cameras["camera25"][wanted])
    manifest = read(batch / "batch_manifest.json")
    manifest["frames"] = [r for r in manifest["frames"] if r["index"] in packets]
    numbers = [r["point_count"] for r in manifest["frames"]]
    manifest["density"].update(total=sum(numbers), minimum=min(numbers), maximum=max(numbers), mean=float(np.mean(numbers)))
    (dataset / "batch_manifest.json").write_bytes(editor.json_bytes(manifest))
    (dataset / "review.html").write_text("<header>old review</header>", encoding="utf-8")
    (media / "release_check.json").write_bytes(editor.json_bytes({"status": "fixture_before_edit"}))
    for name in ("scene.json", "feature_registry.json", "ad2_material_template.json"):
        shutil.copyfile(MODEL / name, model / name)
    (model / "model.json").write_bytes(editor.json_bytes({"candidate_batch_datasets": [{"path": str(dataset / "batch_manifest.json"), "point_count": sum(numbers)}]}))
    return editor.PointEditorStore(dataset, media, model)


def test_store_save_rebinds_persists_counts_and_undo_restores_points(store):
    old = read(store.path(113)); before = packet_vertices(old)
    revision = sha(store.path(113)); pid = ice_point_id(old)
    result = store.edit(113, revision, [{"type": "delete", "point_id": pid}, {"type": "add", "target_id": "Ad2.print.U", "uv": [1400, 500]}])
    assert result["point_count"] == old["point_count"]
    assert result["revision"] != revision and result["history_available"]
    current = read(store.path(113)); binding = read(store.dataset / "bindings/000113.json")
    assert binding["source_packet"]["sha256"] == sha(store.path(113))
    assert pid not in packet_vertices(current)
    assert "manual." in (store.dataset / "review.html").read_text(encoding="utf-8")
    manifest = read(store.dataset / "batch_manifest.json")
    assert manifest["latest_manual_edit_validation"]["all_points_have_unique_typed_model_targets"] is True
    assert read(store.model / "model.json")["candidate_source_point_count"] == manifest["density"]["total"]
    assert (store.media / "overlays/000113.png").read_bytes().startswith(b"\x89PNG")
    undone = store.undo(113, result["revision"])
    assert packet_vertices(read(store.path(113))) == before
    assert undone["point_count"] == old["point_count"] and not undone["history_available"]
    assert read(store.path(113))["manual_editor"]["authoritative_over_extraction_candidates"] is True


def test_save_and_undo_keep_current_model_reference_without_hash_cycle(store):
    model_path=store.model/'model.json'
    model=read(model_path)
    model['candidate_batch_datasets'][0]['sha256']='historical-manifest-hash'
    model_path.write_bytes(editor.json_bytes(model))
    before=packet_vertices(read(store.path(113)))
    result=store.edit(113,sha(store.path(113)),[
        {'type':'delete','point_id':ice_point_id(read(store.path(113)))}])
    for state in ('saved','undone'):
        if state=='undone':store.undo(113,result['revision'])
        manifest=read(store.dataset/'batch_manifest.json')
        assert manifest['model']==editor.ref(model_path)
        current=read(model_path)
        assert 'sha256' not in current['candidate_batch_datasets'][0]
        assert current['candidate_source_point_count']==sum(r['point_count']for r in manifest['frames'])
    assert packet_vertices(read(store.path(113)))==before


def test_save_unbound_deletion_updates_manifest_inventory(store):
    path=store.path(113);packet=read(path)
    packet['unbound_static_materials']=[dict(id='review-only',allowed_for_fit=False,
        segments=[dict(uv=[[100,100],[110,100],[110,110]],closed=True,
            vertex_count=3,point_ids=['review-only:0','review-only:1','review-only:2'])])]
    packet['source_point_count']=packet['point_count']+3
    path.write_bytes(editor.json_bytes(packet))
    manifest_path=store.dataset/'batch_manifest.json';manifest=read(manifest_path)
    row=next(r for r in manifest['frames']if r['index']==113)
    row.update(source_point_count=packet['source_point_count'],unbound_material_point_count=3)
    manifest_path.write_bytes(editor.json_bytes(manifest))
    store.edit(113,sha(path),[{'type':'delete','point_id':'review-only:1'}])
    updated=read(path);row=next(r for r in read(manifest_path)['frames']if r['index']==113)
    assert updated['point_count']==packet['point_count']
    assert row['unbound_material_point_count']==2
    assert row['source_point_count']==row['point_count']+2


def test_revision_reuses_source_media_without_overwriting_previous_overlays(store, tmp_path):
    previous_media = store.media
    previous_files = {p: p.read_bytes() for p in previous_media.rglob("*") if p.is_file()}
    revision_media = tmp_path / "revision_media"
    revision_media.mkdir()
    revision = editor.PointEditorStore(store.dataset, revision_media, store.model, previous_media)
    packet = read(revision.path(113))
    revision.edit(113, sha(revision.path(113)), [{"type": "delete", "point_id": ice_point_id(packet)}])
    assert (revision_media / "overlays/000113.png").read_bytes().startswith(b"\x89PNG")
    assert revision.history_index.exists()
    assert not (revision_media / "frames").exists()
    assert not (revision_media / "camera_seeds.npz").exists()
    assert all(path.read_bytes() == content for path, content in previous_files.items())
    page = (revision.dataset / "review.html").read_text(encoding="utf-8")
    assert (previous_media / "frames/000113.png").as_uri() in page


def test_accepted_frames_share_review_navigation_but_reject_all_writes(store,tmp_path):
    accepted=tmp_path/'accepted';(accepted/'frames').mkdir(parents=True);(accepted/'bindings').mkdir()
    packet=read(store.path(139));path=accepted/'frames/000139.json'
    path.write_bytes(editor.json_bytes(packet))
    binding=accepted/'bindings/000139.json'
    binding.write_bytes(editor.json_bytes(make_binding(packet,path,store.registry)))
    receipt={'accepted':True,'frames':[{'index':139,'packet_sha256':sha(path),'binding_sha256':sha(binding)}]}
    (accepted/'user_acceptance.json').write_bytes(editor.json_bytes(receipt))
    # This isolated fixture's second frame is now a linked accepted batch.
    store.path(139).unlink();(store.dataset/'bindings/000139.json').unlink()
    manifest=read(store.dataset/'batch_manifest.json')
    manifest['frames']=[r for r in manifest['frames'] if r['index']==113]
    manifest['accepted_review_batches']=[{'dataset':str(accepted),'media':str(store.media)}]
    (store.dataset/'batch_manifest.json').write_bytes(editor.json_bytes(manifest))
    revised=editor.PointEditorStore(store.dataset,store.media,store.model)
    before={p:p.read_bytes() for p in accepted.rglob('*') if p.is_file()}
    assert revised.bootstrap()['frame_indices']==[113,139]
    assert revised.frame(139)['read_only'] and not revised.frame(113)['read_only']
    assert revised.media_path(139,'source')==store.media/'frames/000139.png'
    with pytest.raises(ValueError,match='已经验收'):
        revised.edit(139,sha(path),[{'type':'delete','point_id':ice_point_id(packet)}])
    with pytest.raises(ValueError,match='已经验收'):revised.undo(139,sha(path))
    assert all(p.read_bytes()==data for p,data in before.items())
    revised.edit(113,sha(revised.path(113)),[{'type':'delete','point_id':ice_point_id(read(revised.path(113)))}])
    assert all(p.read_bytes()==data for p,data in before.items())


def current_batch_acceptance(store):
    return {"accepted": True, "frames": [
        {"index": index, "packet_sha256": sha(store.path(index)),
         "binding_sha256": sha(store.binding_path(index))}
        for index in store.current_indices
    ]}


def fixture_file_snapshot(store):
    return {path: path.read_bytes() for directory in (store.dataset, store.media, store.model)
            for path in directory.rglob("*") if path.is_file()}


def test_current_accepted_batch_starts_read_only_and_serves_frames(store):
    store.current_acceptance_path.write_bytes(editor.json_bytes(current_batch_acceptance(store)))
    before = fixture_file_snapshot(store)
    accepted = editor.PointEditorStore(store.dataset, store.media, store.model, store.source_media)
    accepted.warm_review_cache()
    instance = editor.create_server(accepted)
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    try:
        status, bootstrap = http(instance, "GET", "/api/bootstrap")
        assert status == 200 and bootstrap["frame_indices"] == [113, 139]
        assert "已验收" in bootstrap["title"]
        for index in (113, 139):
            status, frame = http(instance, "GET", f"/api/frames/{index}")
            assert status == 200 and frame["read_only"]
            assert frame["status"] == "user_accepted"
            packet = read(accepted.path(index))
            with pytest.raises(ValueError, match="已经验收"):
                accepted.edit(index, sha(accepted.path(index)), [
                    {"type": "delete", "point_id": ice_point_id(packet)}
                ])
            with pytest.raises(ValueError, match="已经验收"):
                accepted.undo(index, sha(accepted.path(index)))
        payload, _ = review_payload(accepted.review_html(113, read(accepted.path(113))))
        assert all(frame["status"] == "已由用户验收" for frame in payload["frames"])
    finally:
        instance.shutdown(); instance.server_close(); worker.join(timeout=3)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("defect", ["accepted_false", "packet_hash", "binding_hash", "missing", "extra", "duplicate"])
def test_current_acceptance_invalid_receipt_refuses_start_without_writes(store, defect):
    receipt = current_batch_acceptance(store)
    if defect == "accepted_false":
        receipt["accepted"] = False
    elif defect in ("packet_hash", "binding_hash"):
        receipt["frames"][0]["packet_sha256" if defect == "packet_hash" else "binding_sha256"] = "0" * 64
    elif defect == "missing":
        receipt["frames"].pop()
    elif defect == "extra":
        receipt["frames"].append({**receipt["frames"][0], "index": 99999})
    elif defect == "duplicate":
        receipt["frames"].append(deepcopy(receipt["frames"][0]))
    store.current_acceptance_path.write_bytes(editor.json_bytes(receipt))
    before = fixture_file_snapshot(store)
    with pytest.raises(ValueError):
        editor.PointEditorStore(store.dataset, store.media, store.model, store.source_media)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("valid_receipt", [True, False])
def test_current_acceptance_appearing_during_run_blocks_edit_and_undo(store, valid_receipt):
    delete_fixture_ice_point(store, 113)
    packet = read(store.path(113))
    receipt = current_batch_acceptance(store)
    if not valid_receipt:
        receipt["accepted"] = False
    store.current_acceptance_path.write_bytes(editor.json_bytes(receipt))
    before = fixture_file_snapshot(store)
    with pytest.raises(ValueError, match="已经验收"):
        store.edit(113, sha(store.path(113)), [{"type": "delete", "point_id": ice_point_id(packet)}])
    with pytest.raises(ValueError, match="已经验收"):
        store.undo(113, sha(store.path(113)))
    if valid_receipt:
        assert all(store.frame(index)["read_only"] for index in store.current_indices)
        payload, _ = review_payload(store.review_html(113, packet))
        assert all(frame["status"] == "已由用户验收" for frame in payload["frames"])
    else:
        with pytest.raises(ValueError, match="未明确验收"):
            store.frame(113)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("changed_file", ["packet", "binding"])
def test_current_accepted_frame_detects_changes_after_receipt_validation(store, changed_file):
    store.current_acceptance_path.write_bytes(editor.json_bytes(current_batch_acceptance(store)))
    accepted = editor.PointEditorStore(store.dataset, store.media, store.model, store.source_media)
    if changed_file == "packet":
        packet = read(accepted.path(113))
        packet["status"] = "external_change_after_acceptance"
        replace_fixture_packet(accepted, 113, packet)
    else:
        path = accepted.binding_path(113)
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="已验收.*发生变化"):
        accepted.frame(113)


def test_current_accepted_pending_recovery_does_not_overwrite_accepted_bytes(store):
    original = store.path(113).read_bytes()
    delete_fixture_ice_point(store, 113)
    store.current_acceptance_path.write_bytes(editor.json_bytes(current_batch_acceptance(store)))
    folder = store.history / "pending-before-acceptance"
    folder.mkdir()
    (folder / "before_0.bin").write_bytes(original)
    record = {"status": "prepared", "files": [
        {"path": str(store.path(113)), "existed": True, "before_file": "before_0.bin"}
    ]}
    (folder / "transaction.json").write_bytes(editor.json_bytes(record))
    before = fixture_file_snapshot(store)
    with pytest.raises(ValueError, match="未完成的保存事务"):
        editor.PointEditorStore(store.dataset, store.media, store.model, store.source_media)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_store_conflict_and_model_change_reject_before_write(store):
    old = read(store.path(113)); revision = sha(store.path(113)); original_bytes = store.path(113).read_bytes()
    operation = [{"type": "delete", "point_id": ice_point_id(old)}]
    with pytest.raises(editor.EditConflict):
        store.edit(113, "stale-revision", operation)
    assert store.path(113).read_bytes() == original_bytes
    model_path = store.model / "scene.json"; model_path.write_bytes(model_path.read_bytes() + b"\n")
    with pytest.raises(editor.EditConflict, match="模型"):
        store.edit(113, revision, operation)
    assert store.path(113).read_bytes() == original_bytes and not store.history_index.exists()


def test_legacy_feature_without_target_uses_registered_point_binding(store):
    # The accepted opening batch stores targets in binding observations, while
    # newer batches also duplicate the target at feature level.
    path = store.path(113)
    packet = read(path)
    for feature in packet['features']:
        feature.pop('target_id', None)
    path.write_bytes(editor.json_bytes(packet))
    binding_path = store.binding_path(113)
    binding = read(binding_path)
    binding['source_packet']['sha256'] = sha(path)
    binding_path.write_bytes(editor.json_bytes(binding))
    before = path.read_bytes()
    result = store.frame(113)
    assert sum(len(s['points']) for s in result['segments']) == packet['point_count']
    assert all(s['target_id'] in store.registry['targets'] for s in result['segments'])
    assert path.read_bytes() == before


def test_transaction_write_failure_rolls_back_all_authoritative_files(store, monkeypatch):
    files = [store.path(113), store.dataset / "bindings/000113.json", store.dataset / "batch_manifest.json", store.dataset / "review.html",
             store.media / "overlays/000113.png", store.media / "release_check.json", store.model / "model.json"]
    before = {path: path.read_bytes() for path in files}; original_atomic = editor.write_atomic
    failed = False
    def fail_once(path, data):
        nonlocal failed
        if Path(path) == store.dataset / "bindings/000113.json" and not failed:
            failed = True
            raise OSError("simulated save failure")
        return original_atomic(path, data)
    monkeypatch.setattr(editor, "write_atomic", fail_once)
    with pytest.raises(OSError, match="simulated"):
        store.edit(113, sha(store.path(113)), [{"type": "delete", "point_id": ice_point_id(read(store.path(113)))}])
    assert failed and not store.history_index.exists()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert [read(p)["status"] for p in store.history.glob("*/transaction.json")] == ["rolled_back"]


def test_interrupted_transaction_recovery_restores_snapshot(store):
    packet_path = store.path(113); before = packet_path.read_bytes(); folder = store.history / "interrupted"
    folder.mkdir(parents=True); (folder / "before_0.bin").write_bytes(before)
    record = {"status": "applying", "files": [{"path": str(packet_path), "existed": True, "before_file": "before_0.bin"}]}
    (folder / "transaction.json").write_bytes(editor.json_bytes(record)); packet_path.write_bytes(b"partial save")
    store.recover_transactions()
    assert packet_path.read_bytes() == before
    assert read(folder / "transaction.json")["status"] == "rolled_back_after_interruption"


def review_payload(page):
    if isinstance(page, bytes):
        page = page.decode("utf-8")
    match = re.search(r'<script type="application/json" id="batch-data">(.*?)</script>', page, re.S)
    assert match, "The saved review must retain its standalone frame data"
    return json.loads(match.group(1)), match.group(1)


def saved_review_frames(store):
    payload, _ = review_payload((store.dataset / "review.html").read_bytes())
    return {frame["index"]: frame for frame in payload["frames"]}


def assert_review_matches_packet(frame, packet):
    assert frame["point_count"] == packet["point_count"]
    assert sum(frame["counts"].values()) == packet["point_count"]
    assert frame["source_time"] == editor.timestamp(packet["source"])
    assert frame["segments"] == editor.display_segments(packet)
    assert frame["inferred_segments"] == packet.get("inferred_segments", [])


def replace_fixture_packet(store, index, packet):
    """Keep both isolated fixture files coherent after an external edit."""
    path = store.path(index)
    editor.write_atomic(path, editor.json_bytes(packet))
    binding = make_binding(packet, path, store.registry)
    editor.write_atomic(store.binding_path(index), editor.json_bytes(binding))


def delete_fixture_ice_point(store, index):
    packet = read(store.path(index))
    return store.edit(index, sha(store.path(index)), [
        {"type": "delete", "point_id": ice_point_id(packet)}
    ])


def test_review_cache_second_save_reuses_unchanged_frame(store, monkeypatch):
    delete_fixture_ice_point(store, 113)
    unchanged = read(store.path(139))
    packet_reads = []
    display_calls = []
    original_read, original_display = editor.read, editor.display_segments

    def counted_read(path):
        if Path(path) == store.path(139):
            packet_reads.append(Path(path))
        return original_read(path)

    def counted_display(packet):
        if packet["source"]["native_pts"] == unchanged["source"]["native_pts"]:
            display_calls.append(packet["source"]["native_pts"])
        return original_display(packet)

    monkeypatch.setattr(editor, "read", counted_read)
    monkeypatch.setattr(editor, "display_segments", counted_display)
    delete_fixture_ice_point(store, 113)
    assert packet_reads == []
    assert display_calls == []
    frames = saved_review_frames(store)
    assert_review_matches_packet(frames[113], read(store.path(113)))
    assert_review_matches_packet(frames[139], unchanged)


def test_review_cache_external_atomic_replace_with_preserved_size_and_mtime(store):
    delete_fixture_ice_point(store, 113)
    path = store.path(139)
    before = path.stat()
    packet = read(path)
    old_time = packet["source"]["source_time"]
    packet["source"]["source_time"] = old_time[:-1] + ("1" if old_time[-1] != "1" else "2")
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(editor.json_bytes(packet))
    assert replacement.stat().st_size == before.st_size
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    store.binding_path(139).write_bytes(editor.json_bytes(make_binding(packet, path, store.registry)))
    delete_fixture_ice_point(store, 113)
    frame = saved_review_frames(store)[139]
    assert frame["source_time"] != old_time
    assert_review_matches_packet(frame, packet)


def test_review_cache_external_replace_after_commit_keeps_actual_packet(store, monkeypatch):
    delete_fixture_ice_point(store, 113)
    original_atomic = editor.write_atomic
    external_packet = None

    def replace_after_commit(path, data):
        nonlocal external_packet
        original_atomic(path, data)
        if (external_packet is None and Path(path).name == "transaction.json"
                and json.loads(data)["status"] == "committed"):
            external_packet = read(store.path(113))
            pid = ice_point_id(external_packet)
            feature, segment_index, vertex_index, uv = editor.point_inventory(external_packet)[pid]
            changed_uv = [round(uv[0] + (0.125 if uv[0] < 1918 else -0.125), 4), uv[1]]
            feature["segments"][segment_index]["uv"][vertex_index] = changed_uv
            replace_fixture_packet(store, 113, external_packet)

    monkeypatch.setattr(editor, "write_atomic", replace_after_commit)
    result = delete_fixture_ice_point(store, 113)
    assert external_packet is not None
    assert result["revision"] == sha(store.path(113))
    assert packet_vertices(read(store.path(113))) == packet_vertices(external_packet)
    delete_fixture_ice_point(store, 139)
    assert_review_matches_packet(saved_review_frames(store)[113], external_packet)


def test_review_cache_navigation_appearing_metadata_change_and_deletion(store):
    packet = read(store.path(139))
    fallback = store.media / "fallback_navigation/000139.png"
    fallback.parent.mkdir()
    shutil.copyfile(store.media / "frames/000139.png", fallback)
    packet["source"]["navigation_overlay_path"] = str(fallback)
    replace_fixture_packet(store, 139, packet)
    delete_fixture_ice_point(store, 113)
    assert saved_review_frames(store)[139]["navigation"] == fallback.as_uri()

    revised = store.media / "navigation/000139.png"
    metadata = revised.with_suffix(".json")
    shutil.copyfile(store.media / "frames/000139.png", revised)
    metadata.write_bytes(editor.json_bytes({"scene_sha256": store.model_hashes["scene.json"]}))
    delete_fixture_ice_point(store, 113)
    assert saved_review_frames(store)[139]["navigation"] == revised.as_uri()

    editor.write_atomic(metadata, editor.json_bytes({"scene_sha256": "0" * 64}))
    delete_fixture_ice_point(store, 113)
    assert saved_review_frames(store)[139]["navigation"] == fallback.as_uri()

    editor.write_atomic(metadata, editor.json_bytes({"scene_sha256": store.model_hashes["scene.json"]}))
    delete_fixture_ice_point(store, 113)
    assert saved_review_frames(store)[139]["navigation"] == revised.as_uri()
    revised.unlink()
    delete_fixture_ice_point(store, 113)
    assert saved_review_frames(store)[139]["navigation"] == fallback.as_uri()


@pytest.mark.parametrize("failure_stage", ["history", "commit_record"])
def test_review_cache_late_failure_does_not_reappear_in_next_save(store, monkeypatch, failure_stage):
    delete_fixture_ice_point(store, 113)
    packet = read(store.path(113))
    files = [store.path(113), store.binding_path(113), store.dataset / "batch_manifest.json",
             store.dataset / "review.html", store.media / "overlays/000113.png", store.history_index,
             store.media / "release_check.json", store.model / "model.json"]
    before = {path: path.read_bytes() for path in files}
    original_atomic = editor.write_atomic
    failed = False

    def fail_once(path, data):
        nonlocal failed
        path = Path(path)
        requested_stage = (failure_stage == "history" and path == store.history_index) or (
            failure_stage == "commit_record" and path.name == "transaction.json"
            and json.loads(data)["status"] == "committed")
        if requested_stage and not failed:
            failed = True
            raise OSError("simulated late save failure")
        return original_atomic(path, data)

    monkeypatch.setattr(editor, "write_atomic", fail_once)
    operations = [
        {"type": "delete", "point_id": ice_point_id(packet)},
        {"type": "add", "target_id": "Ad2.print.U", "uv": [1188.321, 777.654]},
    ]
    with pytest.raises(OSError, match="simulated late"):
        store.edit(113, sha(store.path(113)), operations)
    assert failed
    assert all(path.read_bytes() == content for path, content in before.items())
    delete_fixture_ice_point(store, 139)
    frames = saved_review_frames(store)
    assert_review_matches_packet(frames[113], packet)
    assert_review_matches_packet(frames[139], read(store.path(139)))
    assert read(store.history_index)["113"] == json.loads(before[store.history_index])["113"]


def test_review_cache_continuous_edits_and_undo_survive_other_frame_save(store):
    first = delete_fixture_ice_point(store, 113)
    first_packet = read(store.path(113))
    second = store.edit(113, first["revision"], [
        {"type": "delete", "point_id": ice_point_id(first_packet)},
        {"type": "add", "target_id": "Ad2.print.U", "uv": [1188.321, 777.654]},
    ])
    assert_review_matches_packet(saved_review_frames(store)[113], read(store.path(113)))
    store.undo(113, second["revision"])
    assert packet_vertices(read(store.path(113))) == packet_vertices(first_packet)
    delete_fixture_ice_point(store, 139)
    assert_review_matches_packet(saved_review_frames(store)[113], read(store.path(113)))


def test_review_cache_restart_recovers_prepared_files_before_building_review(store):
    delete_fixture_ice_point(store, 113)
    packet = read(store.path(113))
    paths = [store.path(113), store.binding_path(113), store.dataset / "review.html", store.history_index]
    before = {path: path.read_bytes() for path in paths}
    folder = store.history / "interrupted-review-cache"
    folder.mkdir()
    entries = []
    for number, path in enumerate(paths):
        snapshot = f"before_{number}.bin"
        (folder / snapshot).write_bytes(before[path])
        entries.append({"path": str(path), "existed": True, "before_file": snapshot})
    (folder / "transaction.json").write_bytes(editor.json_bytes({"status": "prepared", "files": entries}))
    for path in paths:
        path.write_bytes(b"partial interrupted write")
    restarted = editor.PointEditorStore(store.dataset, store.media, store.model, store.source_media)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert read(folder / "transaction.json")["status"] == "rolled_back_after_interruption"
    delete_fixture_ice_point(restarted, 139)
    assert_review_matches_packet(saved_review_frames(restarted)[113], packet)


def test_review_cache_serialization_preserves_text_without_executable_markup(store):
    packet = read(store.path(139))
    feature = next(feature for feature in packet["features"] if feature["id"].startswith("ice."))
    marker = feature["id"] + '</script><script>window.untrusted = 1</script> & > <'
    feature["id"] = marker
    replace_fixture_packet(store, 139, packet)
    for _ in range(2):
        delete_fixture_ice_point(store, 113)
        page = (store.dataset / "review.html").read_text(encoding="utf-8")
        payload, serialized = review_payload(page)
        assert marker not in page
        assert "</script>" not in serialized and "<" not in serialized and "&" not in serialized
        assert "\\u003c" in serialized and "\\u0026" in serialized and "\\u003e" in serialized
        frame = next(frame for frame in payload["frames"] if frame["index"] == 139)
        assert any(segment["id"] == marker for segment in frame["segments"])
        assert_review_matches_packet(frame, packet)


@pytest.fixture
def server(store):
    instance = editor.create_server(store); worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    try:
        yield instance
    finally:
        instance.shutdown(); instance.server_close(); worker.join(timeout=3)


def http(server, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse(); data = response.read()
        return response.status, json.loads(data)
    finally:
        connection.close()


def test_http_origin_token_host_protection_and_authorized_save(server, store):
    status, bootstrap = http(server, "GET", "/api/bootstrap")
    assert status == 200
    token = bootstrap["csrf_token"]; url = f"http://127.0.0.1:{server.server_port}"
    old_revision = sha(store.path(113))
    body = json.dumps({"base_revision": old_revision, "operations": [{"type": "delete", "point_id": ice_point_id(read(store.path(113)))}]})
    headers = {"Origin": url, "X-Editor-Token": token, "Content-Type": "application/json"}
    for changes in ({"Origin": "http://external.example"}, {"X-Editor-Token": "wrong"}, {"Host": "external.example"}, {"Content-Type": "text/plain"}):
        status, _ = http(server, "POST", "/api/frames/113/edit", body, dict(headers, **changes))
        assert status == 403
        assert sha(store.path(113)) == old_revision
    assert http(server, "POST", "/api/frames/113/edit", body, {k: v for k, v in headers.items() if k != "Origin"})[0] == 403
    assert http(server, "GET", "/api/bootstrap", headers={"Host": "external.example"})[0] == 403
    status, result = http(server, "POST", "/api/frames/113/edit", body, headers)
    assert status == 200 and result["revision"] != old_revision
    assert http(server, "POST", "/api/frames/113/edit", body, headers)[0] == 409


def test_feedback_delivery_failure_never_rolls_back_saved_observations(store,monkeypatch):
    from rinkspace import feedback_outbox
    before=read(store.path(113));pid=ice_point_id(before)
    def unavailable(*args,**kwargs):raise RuntimeError('feedback store temporarily unavailable')
    monkeypatch.setattr(feedback_outbox,'deliver_human_edit',unavailable)
    result=store.edit(113,sha(store.path(113)),[{'type':'delete','point_id':pid}])
    assert pid not in packet_vertices(read(store.path(113)))
    tx=read(store.history/read(store.path(113))['manual_editor']['revision_id']/'transaction.json')
    assert tx['status']=='committed' and result['revision']==sha(store.path(113))
    assert store.feedback_errors and store.feedback_errors[0]['committed_edit_preserved']
    assert (store.history/tx['id']/'after_packet.json').exists()
