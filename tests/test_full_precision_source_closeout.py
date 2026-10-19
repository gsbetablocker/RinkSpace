import json
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.strict_model_scope import file_sha
from scripts.diagnostics import consolidate_full_precision_source_continuation_20261012 as closeout


def _reference(path):
    return dict(path=str(path),sha256=file_sha(path))


def _nine_cut_fixture(tmp_path):
    source = dict(video_sha256='a'*64,time_base='1/15360',pixel_domain='native1920x1080')
    canonical = np.arange(100,1000,100,dtype=np.int64)
    native = np.asarray([p+offset for p in canonical for offset in (0,10)],np.int64)
    camera = tmp_path/'camera.npz'
    np.savez(camera,pts=native,exposure_representative_pts=np.repeat(canonical,2),
        metadata_json=json.dumps(dict(source_sha256='a'*64,source_time_base='1/15360')))
    identity = tmp_path/'identity.json'
    identity.write_text(json.dumps(dict(camera_path=str(camera),camera_sha256=file_sha(camera))),encoding='utf8')
    batch_dir = tmp_path/'batch'
    batch_dir.mkdir()
    retained, pending, terminals = [], [], []
    for number,pts in enumerate(canonical):
        cut = dict(shot_id='cut_'+str(number),start_pts=int(pts),stop_pts=int(pts+11),
            native_count=2,canonical_count=1)
        path = batch_dir/(cut['shot_id']+'.npz')
        originals = []
        if number >= 4:
            graph_dir = path.with_name(path.stem+'_pending_material_graphs')
            graph_dir.mkdir()
            sidecar = graph_dir/(path.stem+'_source_precision_pending.npz')
            precision_receipt = dict(schema='rinkspace.final-q50-source-precision-pending.v1',
                missing_nodes_forged=False,source_points_not_admitted=True,
                source_precision_receipts=[dict(pending_actual_source_candidates=[dict(candidate=0)])])
            np.savez(sidecar,pending_actual_source_receipts_utf8=np.frombuffer(
                json.dumps(precision_receipt).encode('utf8'),np.uint8))
            originals.append(_reference(sidecar))
        bindings = np.full(len(canonical),-1,np.int64)
        bindings[number] = pts
        hashes = np.full(len(canonical),b'',dtype='S64')
        hashes[number] = b'b'*64
        metadata = dict(source=source,legacy_family_fallback='forbidden',
            physically_compacted_approved_pixels_only=True,expected_actual_pts=canonical.tolist(),
            processed_actual_pts=[int(pts)],original_references=originals,
            source_uncertainty_ready_for_fit=False)
        np.savez(path,source_frame_actual_pts=canonical,source_frame_native_pts=bindings,
            source_frame_BGR_sha256=hashes,metadata_json=json.dumps(metadata))
        alias = path.with_name(path.stem+'_native_alias_review.npz')
        np.savez(alias,native_pts=[pts,pts+10],actual_pts=[pts,pts],representative=[True,False],
            alias_native_pts=[pts+10],alias_actual_pts=[pts],canonical_candidate_status=np.asarray([0],np.uint8),
            metadata_json=json.dumps(dict(source=source,new_FIT_observations_created=False)))
        retirement = path.with_name(path.stem+'_retired_old_source_indices.npz')
        np.savez(retirement,disposition=np.zeros(1,np.uint8))
        if number < 4:
            retained.append(dict(cut,absolute_inputs=_reference(path),native_alias_review=_reference(alias),
                old_retirement_indices=_reference(retirement)))
        else:
            pending.append(cut)
            terminals.append(dict(cut,exit_code=0,output_path=str(path),sha256=file_sha(path)))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='rinkspace.exact-source-cut-batch.v1',
        references=[_reference(identity)],common_arguments=['--identity',str(identity)],
        cuts=pending,preserved_completed_source_cuts=retained)),encoding='utf8')
    receipt = batch_dir/'batch_receipt.json'
    receipt.write_text(json.dumps(dict(schema='rinkspace.exact-source-cut-batch-receipt.v1',
        manifest_reference=_reference(manifest),cuts=terminals)),encoding='utf8')
    return manifest,batch_dir,receipt


def test_nine_cut_closeout_keeps_unknown_aliases_pending(tmp_path):
    manifest,batch_dir,_ = _nine_cut_fixture(tmp_path)
    plan = closeout.source_closeout_plan(manifest,[batch_dir])
    assert plan['ready'] is True
    assert plan['expected_native_frames'] == 18
    assert plan['expected_canonical_exposures'] == 9
    assert sum(entry['alias_identity_pending_candidates'] for entry in plan['entries']) == 9
    assert sum(entry['precision_pending_candidates'] for entry in plan['entries']) == 5
    assert plan['whole_programme_source_fit_ready'] is False
    assert all(not entry['source_uncertainty_ready_for_fit'] for entry in plan['entries'])
    assert plan['old_detector_pixels_are_not_merge_inputs'] is True
    assert plan['source_processing_is_not_manual_visual_completion'] is True


def test_missing_success_receipt_never_starts_a_merge(tmp_path,monkeypatch):
    manifest,batch_dir,receipt = _nine_cut_fixture(tmp_path)
    batch = json.loads(receipt.read_text())
    batch['cuts'].pop()
    receipt.write_text(json.dumps(batch),encoding='utf8')
    plan = closeout.source_closeout_plan(manifest,[batch_dir])
    assert plan['ready'] is False
    assert plan['pending_cuts'][0]['shot_id'] == 'cut_8'
    monkeypatch.setattr(closeout,'merge_strict_absolute_source_packs',
        lambda *args:pytest.fail('An incomplete source clock must not be merged'))
    with pytest.raises(ValueError,match='complete successful'):
        closeout.consolidate(plan,tmp_path/'outputs')
    assert not (tmp_path/'outputs').exists()


def test_success_receipt_cannot_hide_a_missing_native_alias(tmp_path):
    manifest,batch_dir,receipt = _nine_cut_fixture(tmp_path)
    alias = batch_dir/'cut_8_native_alias_review.npz'
    with np.load(alias,allow_pickle=False) as original:
        values = {key:original[key].copy() for key in original.files}
    values['alias_native_pts'] = np.empty(0,np.int64)
    np.savez(alias,**values)
    with pytest.raises(ValueError,match='native alias clock coverage'):
        closeout.source_closeout_plan(manifest,[batch_dir])
