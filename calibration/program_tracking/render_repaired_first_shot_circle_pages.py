from calibration.program_tracking.audit_repaired_first_shot_circles import *
import argparse,datetime
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--stop',type=int,required=True);args=ap.parse_args()
root=DATA/'repair_20261014';cache=root/'circle_candidates';folder=root/'circle_diagnostics/contact_pages';folder.mkdir(exist_ok=True);pages=[]
for start in range(args.start,args.stop+1,16):
 stop=min(start+15,args.stop);canvas=np.zeros((4*288,4*480,3),np.uint8)
 for offset,index in enumerate(range(start,stop+1)):
  image=cv2.resize(circle_overlay(index,cache),(480,270),interpolation=cv2.INTER_AREA);y,x=divmod(offset,4);canvas[y*288:y*288+270,x*480:x*480+480]=image;cv2.putText(canvas,str(index),(x*480+10,y*288+286),0,.45,(230,230,230),1)
 file=folder/f'{start:06d}_{stop:06d}.jpg';temporary=file.with_suffix('.tmp.jpg');cv2.imwrite(str(temporary),canvas,[cv2.IMWRITE_JPEG_QUALITY,92]);temporary.replace(file);pages.append(dict(path=str(file),frame_range=[start,stop],sha256=sha(file)))
write(root/f'circle_diagnostics/contact_pages_manifest_{args.start}_{args.stop}.json',dict(status='rendered_final_current_source_candidates_ready_for_actual_visual_review',algorithm_sha256=sha(ROOT/'calibration/program_tracking/native_ice_material_repair.py'),pages=pages,
 candidates=[dict(frame=i,sha256=sha(cache/f'{i:06d}.json')) for i in range(args.start,args.stop+1)],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print(len(pages),str(folder),flush=True)
