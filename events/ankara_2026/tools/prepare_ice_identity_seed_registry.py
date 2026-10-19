"""Preserve actual confirmed source identities; keep old candidates as hints."""
from pathlib import Path
import argparse
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
from rinkspace.vision_ice_seed_registry import make_source_identity_seed_registry,_shot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workroot',type=Path,required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    manifest=json.loads((args.workroot/'source_pixels/source_measurements.json').read_text(encoding='utf8'))
    shots=json.loads((ROOT/'calibration/program_tracking/source_shots.json').read_text(encoding='utf8'))['shots']
    state=json.loads(Path(manifest['camera_identity']).read_text(encoding='utf8'))
    with np.load(state['camera_path'],allow_pickle=False) as saved:
        aliases={int(p):int(a) for p,a in zip(saved['pts'],saved['exposure_representative_pts'])}
    first=[]
    current_png=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/current_video24_feedback_original_RGB_review/229257216_independently_decoded_RAW_NATIVE.png')
    for name,pts,uv,note in [
        ('B2',229208320,[[150,564],[350,612],[550,662],[750,712],[1250,834],[1450,884],[1600,922]],
         'Original first-shot wide blue paint on the R side of the independently user-identified C red center stripe; source ANKARA/ad3 order and continuous pan establish side identity, not camera projection.'),
        ('B1',229257216,[[160,750],[360,705],[560,661],[700,630]],
         'Original first-shot wide blue paint on the F side after the independently user-identified C red center stripe passes across this same shot; both physical blue band edges and adjoining ice are visible.')]:
        reviewed_native_pts=pts
        pts=aliases[pts]
        with av.open(manifest['source_path']) as video:
            stream=video.streams.video[0];stream.thread_count=1
            if str(stream.time_base)!=manifest['time_base']:raise ValueError('Source clock changed')
            video.seek(pts,stream=stream,backward=True)
            bgr=next(frame.to_ndarray(format='bgr24') for frame in video.decode(stream) if frame.pts==pts)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        first.append(dict(curve_id=name+':segment0',physical_id=name,actual_pts=pts,native_pts=pts,
            source_sha256=manifest['source_sha256'],time_base=manifest['time_base'],shot_id=_shot(shots,pts),
            decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            decoded_native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
            navigation_uv=uv,source_material_class='static_painted_ice',
            original_reviewed_native_pts=reviewed_native_pts,source_exposure_alias_clock_authority=manifest['camera_identity'],
            surface_and_physical_identity_confirmed=True,physical_identity_basis=note,
            original_source_image_actually_viewed=True,model_projection_or_fit_residual_used_for_identity=False,
            source_coordinate_semantics='search_navigation_only; actual two material sides must be measured per exposure',
            is_exact_centerline_pixel_ground_truth=False))
    authority=args.workroot/'ice_source_identity_firstshot_B1_B2.json'
    authority.write_text(json.dumps(dict(schema='rinkspace.actual-original-RGB-physical-ice-identities.v1',
        source_video_path=manifest['source_path'],records=first,
        observed_band_width_is_not_centerline_tolerance=True,solver_started=False),ensure_ascii=False,separators=(',',':')),encoding='utf8')
    reference=dict(kind='assistant_actual_original_RGB_physical_material_identity_review',
        path=str(authority.resolve()),sha256=hashlib.sha256(authority.read_bytes()).hexdigest())
    first=[{**row,'authority':reference} for row in first]
    user_stroke=ROOT/'events/ankara_2026/programmes/okada_mei_sp/observations/evidence/user_video_review_20261009/a13_complete_programme/centre_C_04_08_44_933/registered_user_yellow_C_centreline_source_trace_v1.json'
    packet=ROOT/'events/ankara_2026/observations/source_packets/ice108_Lane2_RGB_paired_centres_qualified_20261012/packet.json'
    registry=make_source_identity_seed_registry(manifest['inventory'],source_sha256=manifest['source_sha256'],
        time_base=manifest['time_base'],source_shots=shots,
        original_absolute_inventory=args.workroot/'source_inventory/original_absolute_source_review_inventory.npz',
        native_ice_packets=[packet],user_strokes=[user_stroke],additional_confirmed_seeds=first)
    output=args.workroot/'ice_source_identity_seeds.json'
    output.write_text(json.dumps(registry,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(confirmed_source_anchors=len(registry['records']),
        confirmed_curve_identities=len({r['curve_id'] for r in registry['records']}),
        curve_shot_anchors=len(registry['earliest_curve_shot_anchors']),
        navigation_only_hints=len(registry['navigation_only_seed_hints']),output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
