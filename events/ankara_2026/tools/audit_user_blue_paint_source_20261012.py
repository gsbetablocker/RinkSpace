"""Inspect the user's two first-shot blue-line examples in their own source RGB."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from rinkspace.vision_programme_ice_review import IceProgrammeSourceReader
from rinkspace.vision_ice_model_context_identity import (SameStateIceModelContextIdentity,
    _point_polyline_distance,source_paint_pixel_accuracy_check)
from calibration.program_tracking.render_result import PersonMasks,DEFAULT_MASKS,DEFAULT_PEOPLE
from calibration.program_tracking.source_board_measurements import measure_board_source_frame
from calibration.program_tracking.far_rail_source_components import source_exclusion
from scripts.review_programme_source_inputs import _observed_ice_region,_ice_exclusion

WORK=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')


def main():
    cv2.setNumThreads(4)
    manifest=json.loads((WORK/'source_pixels/source_measurements.json').read_text())
    identity_path=Path(manifest['camera_identity'])
    identity=json.loads(identity_path.read_text())
    scene=json.loads((identity_path.parent/'scene_UNACCEPTED.json').read_text())
    roles=json.loads((WORK.parents[3]/'observations/vision_training/paint_role_contract_current.json').read_text())
    reader=IceProgrammeSourceReader(scene,source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],role_contract=roles)
    authority=ROOT/'events/ankara_2026/model_constraints/user_authorized_same_state_model_navigation_20261012_d7d4998380d40a72.json'
    context=SameStateIceModelContextIdentity(manifest['inventory'],state_identity_path=identity_path,
        navigation_authority=dict(path=str(authority),sha256=hashlib.sha256(authority.read_bytes()).hexdigest()),
        source_sha256=manifest['source_sha256'],time_base=manifest['time_base'])
    with np.load(identity['camera_path']) as camera:
        cameras={int(p):q for p,q in zip(camera['pts'],camera['camera25'])}
        representatives={int(p):int(q) for p,q in zip(camera['pts'],camera['exposure_representative_pts'])}
    masks=PersonMasks(DEFAULT_MASKS,DEFAULT_PEOPLE)
    arrays={};frames=[];wanted={229208320,229257216}
    with av.open(manifest['source_path']) as container:
        stream=container.streams.video[0];stream.thread_count=4
        container.seek(min(wanted),stream=stream,backward=True)
        for native in container.decode(stream):
            if native.pts is None or native.pts<min(wanted):continue
            if native.pts>max(wanted):break
            if native.pts not in wanted:continue
            pts=int(native.pts);bgr=native.to_ndarray(format='bgr24');rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            binding=dict(source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],
                actual_pts=representatives[pts],native_pts=pts,source_pts=pts,shot_id='223403520:R0402_034',
                native_hw=[1080,1920],decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest())
            raw_path=WORK/'source_feedback'/f'user_B1_B2_actual_native_{pts}.png'
            if raw_path.exists():
                if not np.array_equal(cv2.imread(str(raw_path)),bgr):raise ValueError('Existing source PNG differs')
            else:cv2.imwrite(str(raw_path),bgr)
            excluded=source_exclusion(bgr,masks.static_scene_exclusion(pts)[0])
            board=measure_board_source_frame(bgr,excluded,binding)
            boundaries=board.get('ice_scope_boundary_candidates',board['boundary_candidates'])
            mask=_ice_exclusion(excluded,boundaries);scope=_observed_ice_region(boundaries)
            measurement=reader.read(rgb,pts,cameras[pts],mask,shot_id=binding['shot_id'])
            measurement['source_binding']=binding
            navigation=context.navigation(measurement)
            query=np.asarray(measurement['candidate_query']);curves=np.asarray(measurement['query_curve'])
            flags=np.asarray(measurement['candidate_flags'])
            info=dict(source_binding=binding,decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                actual_original_PNG=dict(path=str(raw_path),sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest()),curves=[])
            for name in ('B1:segment0','B2:segment0'):
                ci=next(i for i,v in enumerate(manifest['inventory']) if v['curve_id']==name)
                indices=np.flatnonzero(curves[query]==ci);query_indices=np.flatnonzero(curves==ci)
                uv=np.asarray(measurement['candidate_uv'],float)[indices]
                paths=navigation[ci]
                distance=(np.min(np.stack([_point_polyline_distance(uv,path) for path in paths]),axis=0)
                    if len(indices) and paths else np.empty(0))
                near=indices[distance<=24.] if len(indices) else indices
                precision=(source_paint_pixel_accuracy_check(rgb,measurement,near,manifest['inventory'][ci])
                    if len(near) else dict(accepted_candidate_indices=[],pending=[]))
                prefix=f'{pts}_{name.split(":")[0]}'
                for key in ('uv','sides','query','flags','side_quantiles_uv'):
                    arrays[prefix+'_candidate_'+key]=measurement['candidate_'+key][indices]
                arrays[prefix+'_candidate_ids']=indices
                for key in ('uv','status'):arrays[prefix+'_query_'+key]=measurement['query_'+key][query_indices]
                model_uv=np.concatenate(paths) if paths else np.empty((0,2))
                record=dict(curve_id=name,query_count=len(query_indices),
                    query_status_histogram={str(int(v)):int(np.sum(measurement['query_status'][query_indices]==v))
                        for v in np.unique(measurement['query_status'][query_indices])},
                    navigation_bounds=(model_uv.min(0).tolist()+model_uv.max(0).tolist()) if len(model_uv) else None,
                    actual_candidates=len(indices),
                    candidate_flags_histogram={str(int(v)):int(np.sum(flags[indices]==v)) for v in np.unique(flags[indices])},
                    current_rgb_nearby_actual_candidates=len(near),
                    actual_paint_precision_accepted_candidates=len(precision['accepted_candidate_indices']),
                    source_pixel_precision_pending=precision['pending'])
                info['curves'].append(record)
            frames.append(info)
            print(json.dumps(dict(native_pts=pts,actual_pts=representatives[pts],curves=info['curves']),ensure_ascii=False))
    arrays['metadata_json']=json.dumps(dict(schema='rinkspace.user-screenshot-actual-blue-paint-audit.v1',frames=frames,
        source_pixels_actually_measured=True,UV_from_RGB_two_sides=True,no_new_FIT_observations=True,
        full_programme_accuracy_review_complete=False),ensure_ascii=False,separators=(',',':'))
    output=WORK/'user_B1_B2_screenshot_ownnative_pixel_readout_229208320_229257216.npz'
    if output.exists():raise ValueError('No evidence overwrite')
    np.savez_compressed(output,**arrays)


if __name__=='__main__':main()
