"""Apply an explicit Ad6 source-identity review using the shared repair path."""
from pathlib import Path
import argparse,importlib.util

spec=importlib.util.spec_from_file_location('repair',Path(__file__).with_name('repair_native_s01_ad4_bindings.py'))
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','install'])
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--scene-path',type=Path,required=True)
    parser.add_argument('--review',type=Path,help='Explicit point-by-point source review; defaults to the original RU review')
    parser.add_argument('--stage',type=Path,help='New repair stage below the selected run; never reuse a previous installed repair')
    args=parser.parse_args()
    repair.prep.RUN=args.run_dir.resolve(strict=True);repair.prep.SCENE=args.scene_path.resolve(strict=True)
    repair.STAGE=args.stage.resolve() if args.stage is not None else repair.prep.RUN/'ad6_binding_review/input'
    repair.STAGE.relative_to(repair.prep.RUN)
    repair.REVIEW=args.review.resolve(strict=True) if args.review is not None else Path(__file__).with_name('ad6_disconnected_contour_review_20261026.json')
    repair.replacement.STAGE=repair.STAGE
    repair.prepare() if args.action=='prepare' else repair.replacement.install()
