"""Native 1:1 source crops of every newly recovered clear print component."""
from .repair_first_shot_board_patterns import *

def render():
    reports=read(REPAIR/'clear_print_component_recovery.json')['rows']
    destination=REPAIR/'clear_print_sequential_source_pages';destination.mkdir(exist_ok=True)
    pages=[];canvas=np.zeros((960,1920,3),np.uint8)
    for j,row in enumerate(reports):
        index=row['index'];file=REPAIR/f'board_candidates/{index:06d}.json';packet=read(file)
        ids={r['target_id']for r in row['changes']};features=[f for f in packet['features']if f['target_id']in ids]
        actual=np.asarray([uv for f in features for s in f['segments']for uv in s['uv']])
        center=(actual.min(0)+actual.max(0))/2
        x0=int(np.clip(center[0]-480,0,960));y0=int(np.clip(center[1]-105,0,860))
        raw=cv2.imread(str(DATA/f'frames/{index:06d}.png'))
        overlay=native_overlay(raw,list(source_segments(dict(features=features))))
        tile=np.zeros((240,960,3),np.uint8);tile[20:240]=overlay[y0:y0+220,x0:x0+960]
        cv2.putText(tile,f'{index:04d} current native clear components={len(features)}',(7,16),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        rr,cc=divmod(j%8,2);canvas[rr*240:(rr+1)*240,cc*960:(cc+1)*960]=tile
        if j%8==7 or j==len(reports)-1:
            number=j//8+1;path=destination/f'page_{number:02d}.jpg';cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,97])
            rows=reports[j-j%8:j+1]
            pages.append(dict(page=number,path=str(path),sha256=sha(path),indices=[r['index']for r in rows],
                candidates=[dict(index=r['index'],sha256=sha(REPAIR/f"board_candidates/{r['index']:06d}.json"))for r in rows]))
            canvas.fill(0)
    write(destination/'manifest.json',dict(native_source_crop_scale=1.,pages=pages,frames=len(reports),mask_corrections_sha256=sha(REPAIR/'mask_corrections.json')))
    print('Native clear print review pages',len(pages),flush=True)

if __name__=='__main__':render()
