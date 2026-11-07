"""Decode exact source exposures for action review, with bbox for navigation."""
from pathlib import Path
import argparse
import hashlib
import json
import av
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--range', nargs=2, type=float, required=True, dest='seconds')
    parser.add_argument('--step', type=int, default=20, help='Native exposures between adjacent review images')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--columns', type=int, default=7)
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--tile', type=int, default=216)
    args = parser.parse_args()
    prepared = Path(r'D:/CodexData/RinkSpace_Data/runs/body_s01_20261029/trajectory_continue_1107/programme_source_temporal')
    meta = json.loads((prepared/'preparation.json').read_text(encoding='utf8'))
    video = Path(meta['source_video'])
    with video.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['source_sha256']
    with np.load(prepared/'input.npz', allow_pickle=False) as p:
        pts = p['pts']
        bbox = p['source_bbox'].copy()
        missing = ~np.isfinite(bbox).all(1) | (bbox[:,2]<=bbox[:,0]) | (bbox[:,3]<=bbox[:,1])
        for i in np.flatnonzero(missing):
            nearby = np.flatnonzero(~missing)
            bbox[i] = bbox[nearby[np.argmin(abs(nearby-i))]]
    want = np.flatnonzero((pts>=round(args.seconds[0]*12800)) & (pts<=round(args.seconds[1]*12800)))[::args.step]
    index = {int(pts[i]): int(i) for i in want}
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    tile = args.tile
    cols, rows = args.columns, args.rows
    size = cols * rows
    board = np.full((rows*tile, cols*tile, 3), 27, np.uint8)
    decoded, selected = [], []
    with av.open(str(video)) as source:
        stream = source.streams.video[0]
        assert (stream.time_base.numerator, stream.time_base.denominator)==(1,12800)
        stream.codec_context.thread_count=3
        source.seek(int(pts[want[0]]), stream=stream, backward=True)
        for frame in source.decode(stream):
            if frame.pts<pts[want[0]]:
                continue
            if frame.pts>pts[want[-1]]:
                break
            decoded.append(int(frame.pts))
            if frame.pts not in index:
                continue
            i = index[frame.pts]
            image = frame.to_ndarray(format='bgr24')
            lo = bbox[i,:2]; hi = bbox[i,2:4]
            mid = .5*(lo+hi)
            r = max(16., np.max(hi-lo)*.69)
            x0,y0 = np.maximum((mid-r).astype(int),0)
            x1,y1 = np.minimum((mid+r).astype(int),[image.shape[1],image.shape[0]])
            crop = image[y0:y1, x0:x1]
            if not crop.size:
                raise ValueError(f'Empty crop {frame.pts} {bbox[i]}')
            avail = tile-21
            ratio = min(avail/crop.shape[0], tile/crop.shape[1])
            im = cv2.resize(crop, (max(1,round(crop.shape[1]*ratio)),max(1,round(crop.shape[0]*ratio))), interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
            j = len(selected)
            page = 1 + j//size
            y,x = divmod(j%size,cols)
            x,y = x*tile,y*tile
            dx=(tile-im.shape[1])//2
            board[y:y+im.shape[0], x+dx:x+dx+im.shape[1]]=im
            cv2.putText(board,f'{frame.pts/12800:.2f} | {frame.pts}',(x+3,y+tile-6),cv2.FONT_HERSHEY_SIMPLEX,.47,(248,248,248),1,cv2.LINE_AA)
            selected.append(dict(pts=int(frame.pts), programme_row=i, crop_xyxy=[int(x0),int(y0),int(x1),int(y1)], page=page))
            if j%size==size-1 or j+1==len(want):
                cv2.imwrite(str(args.out/f'sheet_{page:02d}.jpg'), board, [cv2.IMWRITE_JPEG_QUALITY,94])
                board.fill(27)
    assert decoded==list(pts[want[0]:want[-1]+1]), 'Source clock mismatch'
    assert len(selected)==len(want)
    receipt = dict(source=str(video), source_sha256=meta['source_sha256'], time_base=[1,12800], selected=selected,
                   decoded_pts_count=len(decoded), first_decoded_pts=decoded[0], last_decoded_pts=decoded[-1],
                   crop_semantics='existing source box for navigation, no fitted overlay; boundaries read from source')
    (args.out/'review.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(args.out, len(decoded), len(selected), flush=True)
if __name__=='__main__':
    main()
