"""Verify the frozen native source, current exclusion and navigation receipts."""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

from prepare_continuation_frame_sources import DATA, OUT, file_sha, read, write


def main():
    started=time.perf_counter()
    manifest=read(DATA/'frame_manifest.json')
    mapping=read(OUT/'navigation_mapping.json')
    masks=read(DATA/'masks_current/current_mask_manifest.json')
    rows=manifest['frames']
    indices=list(range(101,201))
    assert [r['index'] for r in rows]==indices
    assert [r['programme_frame_index_1based'] for r in mapping['frames']]==indices
    receipts={r['index']:r for r in masks['rows']}
    assert sorted(receipts)==indices
    assert masks['source_manifest']['sha256']==file_sha(DATA/'frame_manifest.json')
    with np.load(DATA/'camera_seeds.npz',allow_pickle=False) as camera:
        assert camera['camera25'].shape==(100,25)
        assert np.array_equal(camera['programme_indices'],indices)
        assert np.array_equal(camera['pts'],[256256+256*(i-1) for i in indices])
        for j,(row,nav) in enumerate(zip(rows,mapping['frames'])):
            source=row['source']; index=row['index']
            pts=256256+256*(index-1)
            assert row['native_pts']==row['source_pts']==source['native_pts']==source['actual_pts']==pts
            assert row['native_frame_index_0based']==source['native_frame_index_0based']==pts//256
            assert row['source_time_base']==source['time_base']=='1/12800'
            assert source['native_hw']==row['native_hw']==[1080,1920]
            assert source['programme_frame_index_1based']==index
            raw=cv2.imread(row['image_path'])
            mask=cv2.imread(row['mask_path'],cv2.IMREAD_GRAYSCALE)
            assert raw.shape==(1080,1920,3) and mask.shape==(1080,1920)
            assert set(np.unique(mask)).issubset({0,255})
            assert file_sha(row['image_path'])==row['png_sha256']==source['png_sha256']
            assert hashlib.sha256(raw.tobytes()).hexdigest()==row['decoded_native_BGR_sha256']==source['decoded_BGR_sha256']
            assert hashlib.sha256(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==row['decoded_native_RGB_sha256']==source['decoded_RGB_sha256']
            mask_sha=file_sha(row['mask_path'])
            assert mask_sha==row['mask_sha256']==source['mask_sha256']==receipts[index]['mask_sha256']
            assert source['mask_path']==row['mask_path']
            assert source['original_PTS_verified'] and source['PNG_roundtrip_lossless_verified']
            assert nav['official_pts']==pts and nav['official_time_base']=='1/12800'
            assert nav['ledger_evidence']['match_status']=='unique_pixel_match'
            assert nav['mapping_status']=='confirmed_pixel_match'
            assert hashlib.sha256(camera['camera25'][j].tobytes()).hexdigest()==nav['camera25_float64_sha256']
            assert int(camera['legacy60_pts'][j])==nav['legacy60_pts']
            overlay=nav['navigation_overlay']
            assert overlay['current_mask_sha256']==mask_sha
            assert overlay['sha256']==file_sha(overlay['path'])
            drawn=cv2.imread(overlay['path'])
            assert drawn.shape==raw.shape and np.array_equal(drawn[mask>0],raw[mask>0])
    canonical=DATA.parent/'frame_000200_50fps'
    assert file_sha(DATA/'frames/000200.png')==file_sha(canonical/'source.png')
    assert file_sha(DATA/'masks_current/000200.png')==file_sha(canonical/'person_mask.png')
    evidence=dict(schema='rinkspace.frozen-continuation-source-preparation-verification.v1',
        frame_count=100,indices=indices,first_pts=281856,last_pts=307200,time_base='1/12800',
        first_time='00:00:22.020',last_time='00:00:24.000',all_image_and_mask_native_hw=[1080,1920],
        native_PTS_frame_index_SHA_checks_passed=True,PNG_lossless_roundtrip_verified=True,
        canonical200_source_and_mask_byte_identical=True,camera25_shape=[100,25],
        all_current_mask_source_sha_bound=True,mask_candidate_not_human_pixel_truth=True,
        no_optimization=True,navigation_frame_count=100,
        all_navigation_current_mask_pixels_preserved=True,all_navigation_mappings_unique_pixel_match=True,
        source_manifest_sha256=file_sha(DATA/'frame_manifest.json'),
        current_mask_manifest_sha256=file_sha(DATA/'masks_current/current_mask_manifest.json'),
        navigation_mapping_sha256=file_sha(OUT/'navigation_mapping.json'),
        camera_archive_sha256=file_sha(DATA/'camera_seeds.npz'),
        final_local_mask_repair_indices=list(range(178,194)),
        verification_elapsed_seconds=time.perf_counter()-started)
    write(DATA/'preparation_validation.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False))


if __name__=='__main__':
    main()
