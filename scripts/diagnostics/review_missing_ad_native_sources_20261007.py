"""Revisit existing irregular-print evidence in exact programme RGB."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'calibration/program_tracking')]
from calibration.program_tracking.render_result import decode_selected
import cv2

def main():
    data=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
    out=data/'diagnostics/irregular_six_ads_source_revisit_20261007'
    out.mkdir(parents=True,exist_ok=True)
    source=data/'media/originals/oxXu1avhrvs.mp4'
    stamps=[229387520,230022144,230066432,230123008,230379776,231013120]
    rows=[]
    for pts,raw in decode_selected(source,stamps):
        target=out/f'raw_{pts}.png'
        if not target.exists() and not cv2.imwrite(str(target),raw):raise OSError(target)
        saved=cv2.imread(str(target))
        if saved.shape!=raw.shape or not (saved==raw).all():raise ValueError('Existing source differs')
        rows.append(dict(pts=pts,time_base='1/15360',path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            decoded_BGR_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    (out/'raw_source_bindings.json').write_text(json.dumps(dict(source=str(source),rows=rows),indent=2),encoding='utf8')
    print(out)

if __name__=='__main__':main()
