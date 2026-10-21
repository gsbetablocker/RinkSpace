from pathlib import Path
import sys,json
P=Path(r'D:/OneDrive/codex_projs/RinkSpace');O=Path(r'D:/CodexData/RinkSpace_Data/runs/okada_rec_1019/recover_v2/dense234_v1')
sys.path.insert(0,str(P/'calibration/rinkfit/src'))
from rinkfit_joint.dense_source_review import render_point_videos
m=render_point_videos(O/'generation_v2/video_config.json',O/'videos_v1')
print(json.dumps(dict(frames=m['frames'],videos=m['videos']),ensure_ascii=False))
