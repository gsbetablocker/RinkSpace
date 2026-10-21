"""Native source video plus immutable source UV points, without a physical model."""
from pathlib import Path
from fractions import Fraction
import gzip,hashlib,json,subprocess,time
from contextlib import closing
import av,cv2,numpy as np
from .io_utils import file_sha256,read_json,write_json
from .source_record_io import canonical_record as _canonical_record,pin_reference as _pin_point_reference,checked_records

COLORS={'board_top':(60,240,60),'board_top_inferred':(30,140,30),'board_contact':(0,165,255),'board_contact_inferred':(0,100,155),'ice':(255,220,0),'ice_inferred':(155,135,0),'print':(255,60,255),'print_inferred':(155,35,155),'manual':(255,255,255)}



def _native_review_spec(cfg):
 if cfg.get('purpose')!='user_requested_source_points_review':raise ValueError('Explicit source-point review purpose required')
 source=cfg['source'];fps=Fraction(source['native_fps']);tb=Fraction(source['time_base']);step=int(source['native_frame_step_pts']);height,width=map(int,source['image_size_hw'])
 if fps!=50 or tb!=Fraction(1,12800)or step!=256:raise ValueError('Official native-clock point review only')
 shots=cfg['shots'];wanted=[(p,s['id'])for s in shots for p in range(s['first_pts'],s['last_pts']+1,step)]
 if not wanted:raise ValueError('No selected native exposures')
 if len({p for p,s in wanted})!=len(wanted):raise ValueError('Duplicate selected native exposures')
 if any(b[0]-a[0]!=step for a,b in zip(wanted,wanted[1:])):raise ValueError('Selected shots must retain their complete contiguous native interval')
 if any(s['first_pts']>s['last_pts']or(s['last_pts']-s['first_pts'])%step for s in shots):raise ValueError('Shot endpoints differ from the native exposure clock')
 return source,fps,tb,step,height,width,shots,wanted



def _point_frame_records(ref,pins):
 """Use the same checked source-record reader as non-visual production consumers."""
 yield from checked_records(ref,pins)


def _validated_point_frames(cfg,pins,expected_frame_ref=None):
 """The same ordered frame/point contract is consumed by export and validation."""
 source,fps,tb,step,height,width,shots,wanted=_native_review_spec(cfg)
 records=_point_frame_records(cfg['selected_frame_file'],pins)
 expected=_point_frame_records(expected_frame_ref,pins)if expected_frame_ref is not None else None
 count=0
 try:
  for row in records:
   if count>=len(wanted)or(row['native_pts'],row['shot_id'])!=wanted[count]:raise ValueError('Missing, duplicate or reordered selected native exposure')
   digest=row['decoded_native_BGR_sha256']
   if not isinstance(digest,str)or len(digest)!=64 or any(c not in'0123456789abcdef'for c in digest):raise ValueError('Invalid decoded native BGR binding')
   if len(row['points'])>500:raise ValueError('Source annotation exceeds existing 500-point ceiling')
   ids=[]
   for point in row['points']:
    xy=np.asarray(point['uv'],float)
    if xy.shape!=(2,)or not np.isfinite(xy).all()or(xy<0).any()or(xy>[width-1,height-1]).any():raise ValueError('Source point outside native pixel domain')
    if point['display_class']not in COLORS:raise ValueError('Unknown source point display class')
    identity=point['selected_row_id']
    if not isinstance(identity,str)or not identity:raise ValueError('Missing selected source row identity')
    ids.append(identity)
   if len(set(ids))!=len(ids):raise ValueError('Duplicated selected source row within exposure')
   if expected is not None:
    prior=next(expected,None)
    if prior is None or row!=prior or _canonical_record(row)!=_canonical_record(prior):raise ValueError('Logical source frame differs from actual legacy frame, point order, UV, class, JSON numeric type or decoded BGR binding')
   count+=1;yield row
  if count!=len(wanted):raise ValueError('Incomplete selected native source exposures')
  if expected is not None and next(expected,None)is not None:raise ValueError('Extra actual legacy selected source exposure')
 finally:
  records.close()
  if expected is not None:expected.close()


def consume_point_frames(config_file,*,expected_frame_ref=None):
 """Exercise the production point consumer without opening or decoding the video.

 With expected_frame_ref every recovered frame dictionary is compared against
 its real legacy stream, including decoded-BGR metadata and complete point order.
 This verifies storage/consumer equivalence, not source-image accuracy.
 """
 config_file=Path(config_file).resolve();cfg=read_json(config_file)
 source,fps,tb,step,height,width,shots,wanted=_native_review_spec(cfg)
 pins={str(config_file):file_sha256(config_file)};frames=0;rows=0;byshot={};classes={};semantic=hashlib.sha256()
 for frame in _validated_point_frames(cfg,pins,expected_frame_ref):
  sid=frame['shot_id'];frames+=1;rows+=len(frame['points']);byshot[sid]=byshot.get(sid,0)+1
  for point in frame['points']:
   key=point['display_class'];classes[key]=classes.get(key,0)+1
  semantic.update(_canonical_record(frame))
 for path,digest in pins.items():
  if file_sha256(path)!=digest:raise ValueError('Point-consumer inputs changed during validation')
 return dict(schema='rinkfit.native_source_point_consumer_validation.v1',complete=True,frames=frames,by_shot=byshot,point_rows=rows,display_classes=classes,display_colors_BGR={k:list(v)for k,v in COLORS.items()},selected_frame_file=cfg['selected_frame_file'],expected_legacy_selected_frame_file=expected_frame_ref,actual_legacy_frame_records_exactly_equal=expected_frame_ref is not None,decoded_BGR_bindings_match_actual_legacy=expected_frame_ref is not None,point_ID_UV_class_and_order_match_actual_legacy=expected_frame_ref is not None,ordered_frame_semantic_sha256=semantic.hexdigest(),physical_input_pins=pins,source_video_file_rehashed=False,source_pixels_decoded=False,video_encoded=False,source_image_quality_or_physical_fit_validated=False)


def render_point_videos(config_file,output=None,*,ffmpeg='ffmpeg',validate_only=False,expected_frame_ref=None):
 if validate_only:return consume_point_frames(config_file,expected_frame_ref=expected_frame_ref)
 config_file=Path(config_file).resolve();out=Path(output).resolve()
 if out.exists():raise FileExistsError('Use a new immutable point-review export')
 cfg=read_json(config_file)
 source,fps,tb,step,height,width,shots,wanted=_native_review_spec(cfg)
 if file_sha256(source['file'])!=source['sha256']:raise ValueError('Source video changed')
 pins={str(config_file):file_sha256(config_file)};_pin_point_reference(cfg['selected_frame_file'],pins)
 if expected_frame_ref is not None:_pin_point_reference(expected_frame_ref,pins)
 out.mkdir(parents=True);started=time.perf_counter();done=0;byshot={};pointtotals={};exports=[];sink=None;enc=None;current=None;ledger=gzip.open(out/'frame_ledger.jsonl.gz','wt',encoding='utf8')
 def close_video():
  nonlocal sink,enc
  if sink is not None:
   for pkt in enc.encode(None):sink.mux(pkt)
   sink.close();sink=None;enc=None
 try:
  with closing(_validated_point_frames(cfg,pins,expected_frame_ref))as points,av.open(source['file'])as original:
   stream=original.streams.video[0]
   if Fraction(stream.time_base)!=tb:raise ValueError('Source stream time base differs')
   original.seek(wanted[0][0],stream=stream,backward=True)
   for frame in original.decode(stream):
    pts=int(frame.pts)
    if pts<wanted[0][0]:continue
    if pts>wanted[-1][0]:break
    if done>=len(wanted)or pts!=wanted[done][0]:raise ValueError('Missing, duplicate or reordered source exposure')
    sid=wanted[done][1];row=next(points)
    if row['native_pts']!=pts or row['shot_id']!=sid:raise ValueError('Source UV frame clock differs from original video')
    if len(row['points'])>500:raise ValueError('Source annotation exceeds existing 500-point ceiling')
    raw=frame.to_ndarray(format='bgr24');rawsha=hashlib.sha256(raw.tobytes()).hexdigest()
    if raw.shape!=(height,width,3)or rawsha!=row['decoded_native_BGR_sha256']:raise ValueError('Selected UV belongs to different original pixels')
    if sid!=current:
     close_video();spec=next(s for s in shots if s['id']==sid);silent=out/(sid+'_points_silent.mp4');sink=av.open(str(silent),'w');enc=sink.add_stream('libx264',rate=fps);enc.width=width;enc.height=height;enc.pix_fmt='yuv420p';enc.time_base=tb;enc.options={'crf':'16','preset':'fast'};current=sid;byshot[sid]=0;pointtotals[sid]=0
    image=raw.copy();ids=[]
    for point in row['points']:
     xy=np.asarray(point['uv'],float)
     if xy.shape!=(2,)or not np.isfinite(xy).all()or(xy<0).any()or(xy>[width-1,height-1]).any():raise ValueError('Source point outside native pixel domain')
     cv2.circle(image,tuple(np.rint(xy).astype(int)),2,COLORS[point['display_class']],1,cv2.LINE_AA);ids.append(point['selected_row_id'])
    if len(set(ids))!=len(ids):raise ValueError('Duplicated selected source row within exposure')
    result=av.VideoFrame.from_ndarray(image,format='bgr24');result.pts=pts-spec['first_pts'];result.time_base=tb
    for pkt in enc.encode(result):sink.mux(pkt)
    ledger.write(json.dumps(dict(shot_id=sid,native_pts=pts,time_base=str(tb),source_seconds=float(pts*tb),output_frame_index=byshot[sid],source_BGR_sha256=rawsha,source_UV_count=len(ids),selected_row_ids=ids,source_UV_from_model=False,physical_model_rendered=False),separators=(',',':'))+'\n')
    byshot[sid]+=1;pointtotals[sid]+=len(ids);done+=1
    if byshot[sid]%50==0:write_json(out/'progress.json',dict(frames=done,total=len(wanted),shot=sid,native_pts=pts,elapsed_seconds=time.perf_counter()-started))
   try:next(points);raise ValueError('Extra selected point exposure outside requested video')
   except StopIteration:pass
  close_video()
 finally:
  close_video();ledger.close()
 if done!=len(wanted):raise ValueError('Incomplete source-point replay')
 for s in shots:
  sid=s['id'];n=(s['last_pts']-s['first_pts'])//step+1;silent=out/(sid+'_points_silent.mp4');final=out/(sid+'_original_ALL_source_points.mp4')
  cmd=[str(ffmpeg),'-hide_banner','-nostdin','-y','-ss',f'{float(s["first_pts"]*tb):.8f}','-i',source['file'],'-i',str(silent),'-map','1:v:0','-map','0:a:0?','-c:v','copy','-c:a','aac','-t',f'{n/float(fps):.8f}','-movflags','+faststart',str(final)]
  with(out/(sid+'_audio_mux.log')).open('w',encoding='utf8')as log:subprocess.run(cmd,stdout=log,stderr=log,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  clocks=[]
  with av.open(str(final))as check:
   st=check.streams.video[0]
   for f in check.decode(st):clocks.append(Fraction(f.pts)*st.time_base)
  if len(clocks)!=n or any(t!=Fraction(i,1)/fps for i,t in enumerate(clocks)):raise ValueError('Encoded point video shifted or lost native exposures')
  silent.unlink()
  exports.append(dict(shot_id=sid,file=str(final),sha256=file_sha256(final),frames=n,source_seconds=[float(s['first_pts']*tb),float((s['last_pts']+step)*tb)],source_pts=[s['first_pts'],s['last_pts']],output_frame_times_verified=True,drawn_point_rows=pointtotals[sid]))
 for p,h in pins.items():
  if file_sha256(p)!=h:raise ValueError('Point-review inputs changed while exporting')
 manifest=dict(schema='rinkfit.native_source_points_only_video.v1',source=source,selected_frame_file=cfg['selected_frame_file'],videos=exports,frames=done,by_shot=byshot,only_original_pixels_and_source_points=True,original_image_rescaled=False,physical_model_used=False,point_UV_interpolated_by_renderer=False,source_annotation_origin_preserved_in_selected_generation=True,pixel_accuracy_or_complete_physical_fit_claimed=False,elapsed_seconds=time.perf_counter()-started)
 write_json(out/'manifest.json',manifest);return manifest
