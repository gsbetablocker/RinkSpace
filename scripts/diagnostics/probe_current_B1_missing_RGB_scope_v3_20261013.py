"""Re-read witnessed B1 gaps from original RGB; never edit the frozen dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import cv2
import numpy as np
from calibration.solver_runtime import install
install()
from calibration.program_tracking.strict_model_scope import file_sha
from calibration.program_tracking.strict_source_pack_merge import _slice_row
from calibration.program_tracking.strict_absolute_source import normalize_selected_ice_rows
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE, SceneRenderer
from calibration.program_tracking.far_rail_source_components import source_exclusion
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from rinkspace.programme_source_stream import FrozenProgrammeSource
from rinkspace.programme_source_measurement_cache import (
    ProgrammeSourceMeasurementCache, restore_exact_navigation_queries, recheck_source_candidate_exclusions)
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.vision_programme_board_review import conservative_ice_exclusion
from rinkspace.vision_blue_paint_RGB_vector_scope_delta import append_blue_paint_scope_RGB_vector_readouts as append_blue_paint_RGB_vector_readouts
from rinkspace.vision_ice_model_context_identity import SameStateIceModelContextIdentity, source_paint_pixel_accuracy_check
from rinkspace.vision_ice_temporal_identity import TemporalIceIdentitySelector

WORK = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
DATASET = ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/current_review_20261013/fit_dataset.json'
AUTHORITY = ROOT/'events/ankara_2026/model_constraints/user_authorized_same_state_model_navigation_20261012_d7d4998380d40a72.json'
SOURCE = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/media/originals/oxXu1avhrvs.mp4')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def literal(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--start-pts',type=int,default=229490944)
    p.add_argument('--stop-pts',type=int,default=229491713)
    args=p.parse_args()
    out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Keep earlier source diagnostics unchanged')
    out.mkdir(parents=True)
    cv2.setNumThreads(1)
    dataset=read(DATASET)
    programme=FrozenProgrammeSource(dataset['frozen_model_reference']['path'],SOURCE,
        ROOT/'calibration/program_tracking/source_shots.json')
    cache=ProgrammeSourceMeasurementCache(WORK/'source_pixels/source_measurements.json',
        source_sha256=programme.source_sha256,time_base=programme.time_base,
        expected_actual_pts=programme.actual_inventory)
    navigator=IceProgrammeSourceReader(programme.scene,source_sha256=programme.source_sha256,
        time_base=programme.time_base)
    model_context=SameStateIceModelContextIdentity(cache.inventory,
        state_identity_path=dataset['frozen_model_reference']['path'],navigation_authority=reference(AUTHORITY),
        source_sha256=programme.source_sha256,time_base=programme.time_base)
    selector=TemporalIceIdentitySelector(cache.inventory,source_sha256=programme.source_sha256,
        time_base=programme.time_base,model_context_identity=model_context)
    definition=next(r for r in cache.inventory if r['curve_id']=='B1:segment0')
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    renderer=SceneRenderer(programme.scene,draw_grid=True,draw_advertisements=True)
    implementation=[reference(ROOT/name)for name in (
        'scripts/diagnostics/probe_current_B1_missing_RGB_scope_v3_20261013.py',
        'src/rinkspace/vision_blue_paint_RGB_vector_scope_delta.py',
        'src/rinkspace/vision_blue_paint_RGB_vector_delta.py',
        'src/rinkspace/vision_ice_model_context_identity.py',
        'src/rinkspace/vision_ice_temporal_identity.py',
        'calibration/program_tracking/strict_absolute_source.py')]
    records=[];payload={}
    for frame in programme.frames(start_pts=args.start_pts,stop_pts=args.stop_pts,cpu_threads=1):
        if not frame['representative']:continue
        pts=int(frame['actual_pts']);native=int(frame['native_pts'])
        binding=dict(frame['source_binding'],native_pts=native)
        raw=frame['raw_bgr'];rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        union,_=masks.static_scene_exclusion(native)
        exclusion=source_exclusion(raw,union)
        board=measure_board_source_frame(raw,exclusion,binding)
        boundaries=board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
        ice_exclusion=conservative_ice_exclusion(exclusion,boundaries)
        ice_scope=~conservative_ice_exclusion(np.zeros_like(exclusion),boundaries)
        measurement=cache.frame(pts,source_binding=binding)
        restore_exact_navigation_queries(measurement,navigator,frame['camera25'])
        recheck_source_candidate_exclusions(measurement,ice_exclusion)
        old_candidates=len(measurement['candidate_uv'])
        if 'candidate_side_quantiles_uv'not in measurement:
            measurement['candidate_side_quantiles_uv']=np.full((old_candidates,2,3,2),np.nan)
        delta=append_blue_paint_RGB_vector_readouts(rgb,measurement,ice_exclusion,cache.inventory,
            source_ice_scope_mask=ice_scope,curve_ids=('B1:segment0',),profile_backend='cpu',keep_profiles=True)
        current=delta['measurement'];new=np.arange(old_candidates,len(current['candidate_uv']))
        clear=new[(current['candidate_flags'][new]&3)==0]
        checked=source_paint_pixel_accuracy_check(rgb,current,clear,definition,check_path=True)
        pixel_supported=np.asarray(checked['accepted_candidate_indices'],np.int64)
        count=Counter(map(int,current['candidate_query'][pixel_supported]))
        ambiguous_queries=sorted(q for q,n in count.items()if n>1)
        ambiguous=pixel_supported[np.isin(current['candidate_query'][pixel_supported],ambiguous_queries)]
        selected=selector.consume(rgb,current,ice_exclusion,source_ice_scope_mask=ice_scope)
        b1=[r for r in selected['rows']if r['curve_id']=='B1:segment0']
        source_rows=[]
        eligible=set(map(int,pixel_supported))-set(map(int,ambiguous))
        for row in b1:
            ids=np.asarray(row['original_candidate_indices'],np.int64)
            local=np.asarray([i for i,c in enumerate(ids)if int(c)in eligible and c>=old_candidates],np.int64)
            if len(local):source_rows.append(_slice_row(row,local))
        normalized=normalize_selected_ice_rows(source_rows,current,source_binding=binding,curve_inventory=cache.inventory)
        candidates=[]
        for row in normalized['rows']:
            ids=np.asarray(row['original_point_indices'],np.int64)
            allowed=np.isin(ids,pixel_supported)&~np.isin(ids,ambiguous)
            for local in np.flatnonzero(allowed):
                candidates.append(dict(candidate_index=int(ids[local]),uv=np.asarray(row['uv'])[local],
                    sides=np.asarray(row['source_sides_uv'])[local],
                    quantiles=np.asarray(row['source_side_quantiles_uv'])[local],
                    sigma_px=float(np.asarray(row['sigma_px'])[local])))
        raw_path=out/f'{pts}_EXACT_RAW_NATIVE.png';cv2.imwrite(str(raw_path),raw)
        canvas,_=renderer.draw(raw,frame['camera25'],title=f'8a28 ALL GRID; B1 actual RGB q50 review only; PTS {pts}')
        for c in candidates:
            for side in c['sides']:cv2.circle(canvas,tuple(np.rint(side).astype(int)),2,(0,255,255),-1,cv2.LINE_AA)
            cv2.circle(canvas,tuple(np.rint(c['uv']).astype(int)),3,(0,0,255),1,cv2.LINE_AA)
        grid_path=out/f'{pts}_same8a28_ALL_GRID_B1_own_q50.png';cv2.imwrite(str(grid_path),canvas)
        crop_path=out/f'{pts}_B1_LEFT_EXACT_RAW.png';cv2.imwrite(str(crop_path),raw[400:600,:420])
        record=dict(source_binding=binding,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            original_cache_measurement=measurement['source_cache_reference'],
            original_chroma_candidates=old_candidates,actual_B1_RGB_vector_candidates=len(new),
            actual_B1_scope_clear_candidates=len(clear),actual_B1_pixel_supported_candidates=len(pixel_supported),
            multiple_complete_actual_blue_bands_at_same_query=ambiguous_queries,
            B1_source_model_context_decisions=[r for r in selected['decisions']if r.get('curve_id')=='B1:segment0'],
            normalized_rows=normalized['rows'],actual_source_supported_for_point_review=candidates,
            point_precision_pending=checked['pending'],source_vector_receipt=delta['source_blue_RGB_vector_queries'],
            raw_reference=reference(raw_path),grid_reference=reference(grid_path),raw_crop_reference=reference(crop_path),
            source_FIT_admitted=False,current_frozen_dataset_changed=False)
        records.append(record)
        for prefix,values in [('current',current),('profiles',delta['actual_source_profiles'])]:
            for name,values_ in values.items():
                if isinstance(values_,np.ndarray):payload[f'{pts}_{prefix}_{name}']=values_
        print(json.dumps(dict(pts=pts,new=len(new),scope_clear=len(clear),pixel_supported=len(pixel_supported),
            temporal_rows=len(b1),point_review=len(candidates),multi_queries=len(ambiguous_queries))),flush=True)
    result=dict(schema='rinkspace.current-B1-missing-original-RGB-reentry-review.v1',
        frozen_dataset_reference=reference(DATASET),model_reference=dataset['frozen_model_reference'],
        source_cache_reference=reference(cache.path),implementation_references=implementation,
        records=records,source_FIT_admitted=False,independent_accuracy_accepted=False,
        same_query_competing_actual_bands_not_selected_by_model_distance=True)
    for ref in implementation:
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Implementation changed during original-source replay')
    path=out/'actual_B1_gap_review.json'
    path.write_text(json.dumps(result,ensure_ascii=False,default=literal,separators=(',',':')),encoding='utf8')
    np.savez_compressed(out/'actual_B1_gap_profiles.npz',**payload)
    print(json.dumps(dict(receipt=reference(path),frames=len(records))),flush=True)


if __name__=='__main__':main()


