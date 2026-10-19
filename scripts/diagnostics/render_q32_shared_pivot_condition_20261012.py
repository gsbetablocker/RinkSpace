"""Render the saved four-exposure comparison with the unchanged complete scene."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import cv2
import numpy as np
from render_result import SceneRenderer

DATA = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
REPORT = DATA / 'q32_four_existing_frames_fixed_pivot_shared_pivot_condition_comparison_20261012.json'


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    report = json.loads(REPORT.read_text(encoding='utf8'))
    source_receipt = report['source_RGB_receipt']
    assert ref(source_receipt['path'])['sha256'] == source_receipt['sha256']
    sources = json.loads(Path(source_receipt['path']).read_text(encoding='utf8'))
    identity_path = Path(report['state_identity']['path'])
    assert ref(identity_path)['sha256'] == report['state_identity']['sha256']
    identity = json.loads(identity_path.read_text(encoding='utf8'))
    scene_path = identity_path.parent / 'scene_UNACCEPTED.json'
    assert ref(scene_path)['sha256'] == identity['scene_sha256']
    renderer = SceneRenderer(json.loads(scene_path.read_text(encoding='utf8')), draw_grid=True,
        draw_advertisements=True, graphics_regions=((28, 24, 554, 253), (1690, 944, 1865, 1000)))
    modes = [('q32 baseline', report['baseline'])]
    for name, mode in report['modes'].items():
        if mode.get('cameras'):
            modes.append((name, mode))
    cameras = [(name, {int(c['actual_pts']): np.asarray(c['camera25']) for c in mode['cameras']})
               for name, mode in modes]
    output = []
    for source in sources['frames']:
        assert ref(source['path'])['sha256'] == source['sha256']
        raw = cv2.imread(source['path'])
        assert raw is not None and raw.shape == (1080, 1920, 3)
        pts = int(source['pts'])
        panels = []
        for name, bypts in cameras:
            panel, metrics = renderer.draw(raw.copy(), bypts[pts], title=f'{name} | PTS {pts} | all scene geometry | FIT diagnostic')
            panels.append(panel)
        target = DATA / f'q32_shared_pivot_{pts}_all_geometry.png'
        assert cv2.imwrite(str(target), np.vstack(panels))
        output.append(dict(actual_pts=pts, source=source, image=ref(target), panel_order=[v[0] for v in modes]))
    receipt = dict(schema='rinkspace.actual-scene-four-exposure-camera-condition-render.v1',
        report=ref(REPORT), scene=ref(scene_path), path_counts=renderer.path_counts,
        advertisement_path_count=len(renderer.advertisement_paths),
        advertisement_ids=sorted({p['number'] for p in renderer.advertisement_paths}),
        frames=output, original_source_images_reused=True, scene_changed=False,
        all_modeled_geometry_enabled=True, full_shot_review=False, independent_validation=False,
        accuracy_accepted=False, actual_visual_review_pending=True)
    target = DATA / 'q32_shared_pivot_four_all_geometry_review_20261012.json'
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(receipt=ref(target), images=[r['image']['path'] for r in output])))


if __name__ == '__main__':
    main()
