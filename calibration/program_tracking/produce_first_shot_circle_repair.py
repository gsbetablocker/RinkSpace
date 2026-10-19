"""Produce all current-native repaired circle families without waiting on thin lines."""
from calibration.program_tracking.extract_first_shot_ice import *
import datetime

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-root',required=True);ap.add_argument('--start',type=int,default=201);ap.add_argument('--stop',type=int,default=1248);args=ap.parse_args()
 ex=Extractor(args.output_root);ex.use_circle_cache=False
 folder=ex.output_root/'circle_candidates';folder.mkdir(exist_ok=True)
 algorithm=sha(ROOT/'calibration/program_tracking/native_ice_material_repair.py');started=datetime.datetime.now(datetime.timezone.utc).isoformat();counts=[]
 write(ex.output_root/'circle_production_status.json',dict(status='running',started_at_utc=started,algorithm_sha256=algorithm,frame_range=[args.start,args.stop]))
 for index in range(args.start,args.stop+1):
  start=time.perf_counter();source=deepcopy(ex.rows[index]['source']);bgr=cv2.imread(str(DATA/f'frames/{index:06d}.png'));person=cv2.imread(source['mask_path'],0)
  broadcast,bcast=current_broadcast_exclusion(bgr);outside_ice,domain=current_ice_visible_domain(index)
  native=NativeProfiles(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),source,(person>0)|broadcast|outside_ice)
  features,diagnostics=ex.nested_house_features(index,native);source.update(mask_sha256=sha(source['mask_path']),broadcast_timer_exclusion=bcast,ice_visible_domain=domain)
  packet=dict(schema='rinkspace.repaired.current-native-circle-cache.v1',programme_frame_index_1based=index,source=source,algorithm_sha256=algorithm,
   model_scene_sha256=sha(ex.model_root/'scene.json'),circle_geometry_sha256=ex.circle_geometry_sha256,features_by_curve=features,diagnostics=diagnostics,current_native_only=True,model_and_temporal_pixels_not_observations=True,
   elapsed_seconds=time.perf_counter()-start)
  write(folder/f'{index:06d}.json',packet);count=sum(f['point_count'] for f in features.values());counts.append(count)
  if index%20==0 or index==args.start or index==args.stop:
   write(ex.output_root/'circle_production_status.json',dict(status='running',started_at_utc=started,last_completed_frame=index,algorithm_sha256=algorithm,frame_range=[args.start,args.stop],completed=len(counts)))
   print(json.dumps(dict(frame=index,circle_points=count,seconds=packet['elapsed_seconds']),ensure_ascii=False),flush=True)
 write(ex.output_root/'circle_production_status.json',dict(status='source_measurement_complete_pending_independent_QA',started_at_utc=started,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),algorithm_sha256=algorithm,frame_range=[args.start,args.stop],completed=len(counts),point_count_range=[min(counts),max(counts)]))
if __name__=='__main__':main()
