"""Programme current ice extraction; historical readers are retained unchanged."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from rinkspace.storage import AssetStore
from calibration.program_tracking.native_circle_source_admission import ReviewedCircleAdmission
from calibration.program_tracking.admitted_first_shot_ice import AdmittedFirstShotExtractor

POLICY = dict(path=str(Path(__file__).parent / 'source_datasets/s1_wide/c0_gate_1017.json'),
    sha256='f6f700ce74e6b912c4f6acd3c393ab612ba5c2a149e6731c38f5dcc7b9f9045f')
SOURCE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--indices', required=True, help='Comma-separated native programme indices')
    parser.add_argument('--kind', choices=('circles', 'all'), default='circles')
    parser.add_argument('--fresh', action='store_true', help='Read current source pixels instead of circle cache')
    args = parser.parse_args()
    store = AssetStore(ROOT)
    admission = ReviewedCircleAdmission(POLICY, resolve=store.resolve)
    data_root = store.data.resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(data_root):
        raise ValueError('Extraction diagnostics and source packets belong in the registered Data root')
    extractor = AdmittedFirstShotExtractor(admission, source_root=args.source_root,
        output_root=output_root, model_root=args.model_root)
    for index in map(int, args.indices.split(',')):
        destination = output_root / f'{"circle_candidates" if args.kind == "circles" else "ice_candidates"}/{index:06d}.json'
        if destination.exists():
            raise FileExistsError('Preserve prior source evidence; choose another output directory: ' + str(destination))
        packet = (extractor.circle_packet(index, fresh=args.fresh) if args.kind == 'circles'
            else extractor.extract(index, fresh=args.fresh))
        if args.kind == 'circles':
            extractor.write_circle(packet)
        print(json.dumps(dict(frame=index, native_pts=packet['source']['native_pts'],
            points=packet['point_count'], complete_estimator_archives=len(packet['inactive_source_estimators']),
            output=str(destination), source_revision=packet['source_admission_revision']['policy']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
