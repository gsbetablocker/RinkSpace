from pathlib import Path
import sys,json
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import render_point_videos
m=render_point_videos(sys.argv[1],sys.argv[2]);print(json.dumps(dict(frames=m['frames'],videos=m['videos']),ensure_ascii=False))
