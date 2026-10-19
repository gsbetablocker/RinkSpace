"""Assemble measured detail tracks on any identified board side or corner."""
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse
from pathlib import Path

from prepare_joint_scene_evidence import ROOT, OUT, load, save
from rinkspace.calibration_companion.advertisement_tracks import assemble_tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, default=OUT/'advertisements/fit_landmark_tracks.json')
    parser.add_argument('--scene', type=Path, default=_resolve_asset(ROOT, ROOT/'outputs/rink_model/ankara_2026_shared/v0.14-scene3d/scene.json'))
    parser.add_argument('--initial-landmarks', type=Path, default=OUT/'advertisements/fit_problem_observations.json')
    parser.add_argument('--identities', type=Path, default=OUT/'advertisements/user_correspondence_20260922.json')
    parser.add_argument('--out', type=Path, default=OUT/'perimeter_camera_support_20260922/assembled_existing_advert_tracks.json')
    args = parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in (args.tracks, args.scene, args.initial_landmarks, args.identities)}:
        parser.error('output must not overwrite source evidence or the model')
    previous = load(args.initial_landmarks).get('board_landmarks', []) if args.initial_landmarks.exists() else []
    identities = load(args.identities).get('correspondences', []) if args.identities.exists() else []
    result = assemble_tracks(load(args.tracks), load(args.scene)['parameters'],
                             initial_landmarks=previous, identity_records=identities)
    result['source_track_file'] = str(args.tracks.resolve())
    save(args.out, result)
    print({'frames': len(result['frames']), 'landmarks': len(result['board_landmarks']),
           'fit_observations': result['fit_observation_count'], 'check_observations': result['check_observation_count']})


if __name__ == '__main__':
    main()
