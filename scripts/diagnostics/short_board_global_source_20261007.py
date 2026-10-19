"""Native multi-time short-board evidence: no projected geometry selection."""
from pathlib import Path
import json
import hashlib
import av
import cv2
import numpy as np

DATA=Path(r"D:/CodexData/RinkSpace_Data/events/ankara_2026")
SOURCE=DATA/"media/originals/oxXu1avhrvs.mp4"
OUT=DATA/"diagnostics/external_expert_20261007/short_board_global_source_20261007"
STAMPS=[230043904,230092800,230146560,230200320,230254336]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frames={};receipts=[]
    with av.open(str(SOURCE)) as c:
        stream=c.streams.video[0];stream.thread_count=1;c.seek(STAMPS[0]-15360,stream=stream,backward=True,any_frame=False)
        for f in c.decode(stream):
            if f.pts>STAMPS[-1]:break
            if f.pts not in STAMPS:continue
            bgr=f.to_ndarray(format="bgr24");p=int(f.pts);frames[p]=bgr
            file=OUT/f"native_{p}.png";cv2.imwrite(str(file),bgr)
            receipts.append(dict(pts=p,time_base=str(f.time_base),source_time_seconds=p/15360,
                native_hw=list(bgr.shape[:2]),native_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                native_png_path=str(file),native_png_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    if set(frames)!=set(STAMPS):raise ValueError("Missing native requested source PTS")
    panels=[]
    for p in STAMPS:
        head=np.full((35,1920,3),245,np.uint8)
        cv2.putText(head,f"PTS{p} time{p/15360:.6f}s | untouched native source board crop y=230..640",(10,25),0,.65,(20,20,20),1)
        panels.append(np.vstack((head,frames[p][230:640])))
    image=OUT/"short_board_five_native_times.png";cv2.imwrite(str(image),np.vstack(panels))
    report=dict(schema="rinkspace.short-board-multitime-native-source-20261007.v1",source_video=str(SOURCE),
        source_video_sha256_from_original_ledger="f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891",
        source_time_base="1/15360",requested_user_times=["04:09:36.817","04:09:50.517"],
        endpoints_nearest_actual_display_pts=[STAMPS[0],STAMPS[-1]],native_receipts=receipts,
        source_image_board_montage=str(image),crop_xyxy=[0,230,1920,640],
        source_model_camera_or_network_not_imported=True,solver_or_observations_unchanged=True)
    path=OUT/"short_board_five_native_source_receipts.json";path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(report=str(path),image=str(image))))


if __name__=="__main__":main()
