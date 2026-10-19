"""Review one existing native board task and write scoped feedback evidence."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'calibration/rinkfit/src'),
               str(ROOT / 'calibration/program_tracking')]
from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
from rinkspace.feedback_review import review_existing_board_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--pts', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--predecessor-finding', action='append', default=[])
    args = parser.parse_args()
    result = review_existing_board_frame(args.run, args.pts, args.output, device=args.device,
        predecessor_findings=args.predecessor_finding)
    print(json.dumps(dict(source=result['source'], counts=result['counts'],
        receipt=str(Path(args.output) / 'receipt.json'), resolutions=len(result['resolutions'])), ensure_ascii=False))
