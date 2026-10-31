"""Complete Ad3's registration reference; create no current-frame observations.

The official source supplies artwork topology. Both references only initialize
the public RGB reader, whose output must still be measured in each exposure.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(composition, old_atlases, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(composition).read_text(encoding='utf-8'))
    old = json.loads(Path(old_atlases).read_text(encoding='utf-8'))['atlases'][70]
    contours = source['contours']
    lookup = {c['id']: c for c in contours}
    if len(lookup) != 36 or sha(source['source_svg']) != source['source_sha256']:
        raise ValueError('Official complete source identity differs')
    if sha(old['image']) != old['image_sha256']:
        raise ValueError('Actual photographic reference changed')
    xx, yy = [], []
    for path in old['paths']:
        xy = lookup[path['path_id'].removeprefix('Ad3.print.')]['xy_composition']
        if len(xy) != len(path['pixels']):
            raise ValueError('Existing reference correspondence differs')
        xx.extend(xy)
        yy.extend(path['pixels'])
    xx, yy = np.asarray(xx), np.asarray(yy)
    photo_H, _ = cv2.findHomography(xx, yy, 0)
    photo_error = np.linalg.norm(cv2.perspectiveTransform(xx[None], photo_H)[0] - yy, axis=1)
    if photo_error.max() > 1e-4:
        raise ValueError('Common paths do not share the existing photographic chart')

    all_xy = np.concatenate([c['xy_composition'] for c in contours])
    origin = np.floor(all_xy.min(0)) - 5
    scale = 2.
    shape = np.ceil((all_xy.max(0) - origin + 5) * scale).astype(int)
    artwork = np.zeros((shape[1], shape[0], 3), np.uint8)
    paths, photograph_paths = [], []
    # The source declares outer contours and their holes explicitly. Render
    # artwork as a registration reference; these pixels are never fit UVs.
    for role, color in (('outer', 255), ('hole', 0)):
        for c in contours:
            if c['role'] != role:
                continue
            pixels = (np.asarray(c['xy_composition']) - origin) * scale
            cv2.fillPoly(artwork, [np.rint(pixels).astype(np.int32)], (color,) * 3)
    for c in contours:
        path_id = 'Ad3.print.' + c['id']
        xy = np.asarray(c['xy_composition'])
        proof = dict(kind='official_complete_artwork_registration_reference',
                     composition=str(composition), composition_sha256=sha(composition),
                     source_svg_sha256=source['source_sha256'],
                     role=c['role'], parent_id=c['parent_id'],
                     source_UV_observation=False, reference_is_search_only=True)
        pixels = (xy - origin) * scale
        topology = dict(source_closed=True,
            role='letter_hole_boundary' if c['role']=='hole' else 'printed_ink_external_boundary',
            parent_contour_key=('Ad3.print.'+c['parent_id']) if c['parent_id'] else None)
        paths.append(dict(path_id=path_id, pixels=pixels.tolist(), proof=proof, **topology))
        photograph_paths.append(dict(path_id=path_id,
            pixels=cv2.perspectiveTransform(xy[None], photo_H)[0].tolist(),
            proof=dict(proof, kind='official_artwork_in_existing_actual_source_chart',
                       shared_chart_maximum_common_path_error_px=float(photo_error.max())), **topology))
    image_path = output / 'ad3_complete_official_reference.png'
    if not cv2.imwrite(str(image_path), artwork):
        raise OSError(image_path)
    atlases = [dict(id=0, ad=3, image=str(image_path), image_sha256=sha(image_path), paths=paths,
                   reference_is_official_artwork_not_source_observation=True),
               dict(id=1, ad=3, image=old['image'], image_sha256=old['image_sha256'],
                    paths=photograph_paths, reference_is_search_only=True)]
    result = dict(schema='rinkspace.complete_print_reference_library.v1', atlases=atlases,
                  current_source_observations_created=False,
                  old_source_observations_reactivated=False,
                  shared_photographic_chart=photo_H.tolist(),
                  common_path_maximum_error_px=float(photo_error.max()),
                  official_to_raster_scale=scale, official_to_raster_origin=origin.tolist())
    (output / 'atlases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), paths_per_atlas=len(paths),
                          new_daikichi_paths=sum('path9.' in p['path_id'] for p in paths),
                          common_path_maximum_error_px=float(photo_error.max()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--composition', required=True, type=Path)
    parser.add_argument('--old-atlases', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.composition, args.old_atlases, args.output)
