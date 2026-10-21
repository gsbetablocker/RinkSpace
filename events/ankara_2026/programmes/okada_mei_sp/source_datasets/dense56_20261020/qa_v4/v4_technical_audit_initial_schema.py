"""Read-only audit of final v4 renderer ledgers and encoded packet clocks. No video decoding."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import sys,json,gzip,hashlib,subprocess,math,datetime
SPECS={'s05':(948992,1300992),'s06':(1301248,1425408)}
root=Path(sys.argv[1]);gp=Path(sys.argv[2]);report=Path(sys.argv[3]);ffprobe=sys.argv[4]if len(sys.argv)>4 else'ffprobe'
if report.exists():raise FileExistsError('Preserve prior technical receipts: '+str(report))
errors=[];warnings=[];pins={}
def check(ok,msg):
 if not ok:errors.append(msg)
def sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p,expected=None):
 p=Path(p);h=sha(p);pins[str(p)]=h
 if expected is not None:check(h==expected,'SHA mismatch '+str(p))
 return h
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stream(p):
 with gzip.open(p,'rt',encoding='utf8')as f:
  for line in f:
   if line.strip():yield json.loads(line)
def probe(p,packets=True):
 cmd=[ffprobe,'-v','error','-show_streams','-show_format']
 if packets:cmd+=['-show_packets']
 cmd+=['-show_entries','packet=stream_index,pts,dts,duration,pts_time,duration_time,flags:packet_side_data=side_data_type,skip_samples,discard_padding:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration_ts,duration,nb_frames,sample_rate,channels:format=duration,start_time,format_name','-of','json',str(p)]
 proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));return json.loads(proc.stdout)
result=dict(schema='rinkspace.v4_independent_video_clock_audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),generation_manifest=str(gp),renderer_assembly=str(root),method='Read only the final explicit video assembly child refs and gzip frame ledgers; verify all five merged gzip byte members, frozen selected IDs/UV and actual MP4 packet clocks via ffprobe. Public renderer source-pixel/full-output decode checks are pinned production evidence, not independently repeated here. No source or output video decoding.',image_quality_accepted=False,physical_model_accepted=False,all_frames_manually_viewed=False,no_video_decoder_used=True,errors=errors,warnings=warnings)
try:
 g=read(gp);pin(gp);check(g.get('complete')and g.get('generation_id')=='dense56_v4','Only complete dense56_v4 generation is in scope');check(g.get('frames')==1862,'Generation must cover 1862 native exposures');check(g.get('source_UV_moved')is False,'Generation source UV moved flag');check(g.get('physical_solver_or_live_packet_changed')is False,'Generation physical solver scope flag')
 base=gp.parent.parent;check(gp.resolve()==(base/'generation_v4/manifest.json').resolve(),'Unexpected final generation location');expected_names={'consumer_bindings.jsonl.gz','coverage.jsonl.gz','excluded.jsonl.gz','selected_frames.jsonl.gz','selected_rows.jsonl.gz'};check(set(g['files'])==expected_names,'Final generation must bind exactly five source gzip products')
 qa6p=base/'qa_independent/v2_s06_image_review.json';pin(qa6p,'4a438490a756e08577391635261bdb843193411e291bbe631a4ea9694d133c01');qa6=read(qa6p);check(qa6.get('blocking_errors')==[]and qa6.get('eligible_for_requested_full_source_point_video_render')is True,'S06 image QA is not accepted');sp=base/'qa_independent/v2_s06_structural.json';pin(sp,'ccba53a46a87e74e0e878783c7422cfe9aaa06ebd59a633a7c64bc79cb62517c');sqa=read(sp);check(sqa.get('errors')==[],'S06 structural receipt has errors');result['accepted_S06_image_QA_binding']=dict(file=str(qa6p),sha256=pins[str(qa6p)],structural_file=str(sp),structural_sha256=pins[str(sp)],images_loaded=False,scope='Existing source image review reused only for its exact immutable S06 generation; does not accept S05 or certify all frames')
 producer=Path(r'D:/OneDrive/codex_projs/RinkSpace/calibration/rinkfit/src/rinkfit_joint/dense_source_review.py');pin(producer);result['public_renderer_production_evidence']=dict(producer_file=str(producer),producer_sha256=pins[str(producer)],per_frame_source_BGR_hash_checked_in_producer=True,full_encoded_native_clock_decode_checked_in_producer=True,independent_video_decode_repeated=False)
 parents=g.get('parent_shot_generations',[]);check(len(parents)==2,'Final merge must bind both frozen shot manifests');childfacts=[];children={}
 for pref in parents:
  pin(pref['file'],pref['sha256']);cm=read(pref['file']);check(cm.get('complete')is True and len(cm.get('shot_ids',[]))==1,'Invalid merged child generation');sid=cm['shot_ids'][0];check(sid not in children,'Duplicate child generation '+sid);children[sid]=cm;check(set(cm['files'])==expected_names,'Child must bind same five source gzip products '+sid);check(cm['frames']==(1376 if sid=='s05' else 486),'Child native frame count differs '+sid);childfacts.append(dict(shot_id=sid,manifest=pref,selected_frames=cm['files']['selected_frames.jsonl.gz'],selected_rows=cm['files']['selected_rows.jsonl.gz']))
  if sid=='s06':check(pref['sha256']=='0db0b91dbb668c895e65cd5cb07563711e1241b9e1d94304a9bf3475061a181b'and cm['files']['selected_frames.jsonl.gz']['sha256']=='7c33c9862fa4ab6d89404607e46bdd9e667f6816d0adc7c263cc22c540a4a47e'and cm['files']['selected_rows.jsonl.gz']['sha256']=='e38e80af6776e9647a9d78b41b1f83488cdb7c3c1602a8c366cd20d966b681cd','S06 source differs from actual image-reviewed generation')
 check(set(children)==set(SPECS),'Missing or extra frozen shot generations');check([x['shot_id']for x in childfacts]==['s05','s06'],'Child source generations not in native shot order');check(Path(qa6['generation_manifest']['file']).resolve()==Path(parents[1]['file']).resolve()and qa6['generation_manifest']['sha256']==parents[1]['sha256']and qa6['selected_frames']['sha256']==children['s06']['files']['selected_frames.jsonl.gz']['sha256']and qa6['selected_rows']['sha256']==children['s06']['files']['selected_rows.jsonl.gz']['sha256'],'S06 image QA is bound to different source files')
 result['frozen_child_generations']=childfacts;result['merge_gzip_byte_members']=[]
 for name in sorted(expected_names):
  merged=Path(g['files'][name]['file']);members=g.get('gzip_member_concatenation_without_UV_or_row_ID_change',{}).get(name,[]);check(len(members)==2,'Final selected gzip member list differs '+name);pos=0
  with merged.open('rb')as mf:
   for mi,member in enumerate(members):
    childref=children[['s05','s06'][mi]]['files'][name];check(Path(member['file']).resolve()==Path(childref['file']).resolve()and member['sha256']==childref['sha256'],'Merged byte member differs from frozen child '+name);check(member['combined_file_offset']==pos,'Merged selected member offset differs '+name);n=int(member['member_bytes']);check(Path(member['file']).stat().st_size==n,'Merged selected parent byte length differs '+name);pin(member['file'],member['sha256']);h=hashlib.sha256();left=n
    while left:
     b=mf.read(min(4*1024*1024,left))
     if not b:break
     h.update(b);left-=len(b)
    check(left==0 and h.hexdigest()==member['sha256'],'Merged selected member byte identity differs '+name);result['merge_gzip_byte_members'].append(dict(merged_file=str(merged),member_source=member['file'],offset=pos,length=n,expected_sha256=member['sha256'],actual_merged_segment_sha256=h.hexdigest()));pos+=n
   check(pos==merged.stat().st_size and mf.read(1)==b'','Extra or missing merged selected bytes '+name)
  pin(merged,g['files'][name]['sha256'])
 sfref=g['files']['selected_frames.jsonl.gz'];srref=g['files']['selected_rows.jsonl.gz'];pin(sfref['file'],sfref['sha256']);pin(srref['file'],srref['sha256']);frames={};frameorder=[];globalpoints={};counts=Counter()
 for r in stream(sfref['file']):
  sid=r['shot_id'];p=r['native_pts'];key=(sid,p);check(key not in frames,'Duplicate selected frame '+str(key));frames[key]=r;frameorder.append(key);check(len(r['points'])<=500,'Density exceeds500 '+str(key));ids=[pt['selected_row_id']for pt in r['points']];check(len(ids)==len(set(ids)),'Repeated source row '+str(key))
  for pt in r['points']:
   check(len(pt['uv'])==2 and all(math.isfinite(float(v))for v in pt['uv'])and 0<=pt['uv'][0]<=1919 and 0<=pt['uv'][1]<=1079,'Invalid source UV '+pt['selected_row_id']);check(pt['selected_row_id']not in globalpoints,'Source row ID reused');globalpoints[pt['selected_row_id']]=(sid,p,pt)
  counts[sid]+=len(r['points'])
 expected=[(s,p)for s,(a,b)in SPECS.items()for p in range(a,b+1,256)];check(frameorder==expected,'Selected frames differ from full actual native shot clock');rowids=set()
 for r in stream(srref['file']):
  rid=r['selected_row_id'];check(rid in globalpoints,'Selected row absent from frame '+rid)
  if rid in globalpoints:
   sid,p,pt=globalpoints[rid];check((r['shot_id'],r['native_pts'])==(sid,p)and r['uv']==pt['uv']and r['display_class']==pt['display_class'],'Selected row/frame identity differs '+rid)
  check(rid not in rowids,'Selected row duplicate '+rid);rowids.add(rid);check(r.get('physical_world_correspondence_issued')is False and r.get('independent_fit_weight_issued')is False,'Unexpected physical correspondence/fit weight '+rid)
 check(rowids==set(globalpoints),'Selected rows do not exactly match drawn frame IDs');check(len(rowids)==g['point_rows'],'Generation point count does not match actual selected rows');check({s:sum(1 for k in frames if k[0]==s)for s in SPECS}=={'s05':1376,'s06':486},'Selected shot exposure counts differ');result['selected_frames']=len(frames);result['selected_rows']=len(rowids);result['point_count_by_shot']=dict(counts);result['point_count_per_native_frame']=dict(min=min(len(f['points'])for f in frames.values()),max=max(len(f['points'])for f in frames.values()))
 result['auxiliary_native_clock_checks']={}
 for name in ('consumer_bindings.jsonl.gz','coverage.jsonl.gz'):
  seen=[]
  for r in stream(g['files'][name]['file']):
   key=(r['shot_id'],r['native_pts']);seen.append(key);check(key in frames,'Auxiliary source row outside native interval '+name+' '+str(key))
   if key in frames:
    fr=frames[key]
    if name=='consumer_bindings.jsonl.gz':check(r['source_BGR_sha256']==fr['decoded_native_BGR_sha256'],'Consumer source pixel binding differs '+str(key))
    else:
     actual=Counter('board'if p['display_class'].startswith('board_')else p['display_class'].split('_')[0]for p in fr['points']);check(dict(actual)==r['selected_rows_by_part'],'Coverage point counts differ from actual displayed source rows '+str(key))
  check(seen==expected,'Auxiliary frames differ from complete native order '+name);result['auxiliary_native_clock_checks'][name]=dict(frames=len(seen),native_clock_exact=seen==expected)
 excluded_counts=Counter();excluded_reasons=Counter();previous=None;backward=0;outside=0
 for r in stream(g['files']['excluded.jsonl.gz']['file']):
  key=(r['shot_id'],r['native_pts'])
  if key not in frames:outside+=1
  if previous is not None and key<previous:backward+=1
  previous=key;excluded_counts[r['shot_id']]+=1;excluded_reasons[r.get('reason','missing_reason')]+=1
 check(backward==0,'Excluded source record native PTS moved backwards');check(outside==0,'Excluded source record outside final native interval');result['excluded_source_rows_actual']=dict(total=sum(excluded_counts.values()),by_shot=dict(excluded_counts),by_reason=dict(excluded_reasons),native_clock_backsteps=backward,outside_native_clock=outside,prior_rejections_and_new_corrections_preserved_by_frozen_child_byte_binding=True)
 assemblyp=root if root.is_file()else root/'manifest.json';assembly=read(assemblyp);pin(assemblyp);check(assemblyp.resolve()==(base/'videos_v4/manifest.json').resolve(),'Unexpected final assembly path');check(assembly.get('schema')=='rinkspace.native_source_points_only_video_assembly.v4'and assembly.get('complete')is True and assembly.get('frames')==1862 and assembly.get('by_shot')=={'s05':1376,'s06':486},'Incomplete final explicit video assembly');check(Path(assembly['source_generation']['file']).resolve()==gp.resolve()and assembly['source_generation']['sha256']==pins[str(gp)],'Final video assembly uses a different generation');check(assembly.get('drawn_source_point_rows')==len(rowids),'Assembly point count differs');check(assembly.get('source_frames_missing_or_duplicated')==0 and assembly.get('output_frame_times_verified')is True and assembly.get('merged_selected_frames_preserve_child_gzip_members_bytewise')is True,'Assembly native clock/byte identity flags differ');check(assembly.get('only_original_pixels_and_source_points')is True and assembly.get('physical_model_used')is False and assembly.get('original_image_rescaled')is False and assembly.get('point_UV_interpolated_by_renderer')is False,'Assembly scope flags differ');check(assembly['selected_frame_file']['sha256']==sfref['sha256']and Path(assembly['selected_frame_file']['file']).resolve()==Path(sfref['file']).resolve(),'Assembly selected frames differs');renderers=[];childrefs=assembly.get('child_video_exports',[]);check(len(childrefs)==2,'Final video assembly must explicitly bind both actual child exports');childledgerpins={};expected_renderers=[base/'videos_v4/s05/manifest.json',base/'videos_v2/s06/manifest.json']
 for i,ref in enumerate(childrefs):
  mp=Path(ref['file']);check(mp.resolve()==expected_renderers[i].resolve(),'Explicit renderer export path differs from final shot scope');pin(mp,ref['sha256']);m=read(mp);check(m.get('schema')=='rinkfit.native_source_points_only_video.v1','Unexpected child renderer schema');sid=['s05','s06'][i];check(m.get('frames')==(1376 if sid=='s05' else 486)and m.get('by_shot')=={sid:(1376 if sid=='s05' else 486)},'Child renderer shot scope differs');check(ref['selected_frame_file']==m['selected_frame_file'],'Assembly child selected frame ref differs');check(m['selected_frame_file']['sha256']==children[sid]['files']['selected_frames.jsonl.gz']['sha256']and Path(m['selected_frame_file']['file']).resolve()==Path(children[sid]['files']['selected_frames.jsonl.gz']['file']).resolve(),'Child renderer selected source differs from frozen shot generation');lr=ref['frame_ledger_ref'];check(Path(lr['file']).resolve()==(mp.parent/'frame_ledger.jsonl.gz').resolve(),'Assembly ledger path differs');pin(lr['file'],lr['sha256']);childledgerpins[str(mp.resolve())]=lr
  if sid=='s06':check(ref['sha256']=='c0404386849a6d227f8e7e30b2181f8cd7aff2e5b709a86f56e11cd96a9d3493','S06 video renderer differs from accepted unchanged export')
  renderers.append((mp,m))
 result['explicit_video_assembly']=dict(file=str(assemblyp),sha256=pins[str(assemblyp)],child_video_exports=childrefs,no_recursive_renderer_search=True)
 check(bool(renderers),'No new native source renderer manifests found');videos={};ledgers=set();ledgerkeys=[];sourcefacts={};result['renderer_manifests']=[]
 for mp,m in renderers:
  pin(mp);check(m.get('only_original_pixels_and_source_points')is True and m.get('physical_model_used')is False and m.get('original_image_rescaled')is False and m.get('point_UV_interpolated_by_renderer')is False,'Renderer scope flags '+str(mp));src=m['source'];pin(src['file'],src['sha256']);check(src['sha256']=='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d','Wrong official source identity');check(Fraction(src['native_fps'])==50 and Fraction(src['time_base'])==Fraction(1,12800)and src['native_frame_step_pts']==256 and src['image_size_hw']==[1080,1920],'Official source clock/dimensions differ');pref=m['selected_frame_file'];pin(pref['file'],pref['sha256']);check(src==assembly['source'],'Renderer official source differs from final assembly');sourcefacts[src['sha256']]=src
  lp=mp.parent/'frame_ledger.jsonl.gz';check(lp.exists(),'Missing ledger '+str(lp));ledger_count=0;byshot=Counter();localkeys=[]
  if lp.exists()and str(lp.resolve())not in ledgers:
   ledgers.add(str(lp.resolve()));pin(lp,childledgerpins[str(mp.resolve())]['sha256'])
   for l in stream(lp):
    key=(l['shot_id'],l['native_pts']);localkeys.append(key);ledgerkeys.append(key);check(key in frames,'Ledger outside frozen native source '+str(key));check(l['time_base']=='1/12800'and Fraction(str(l['source_seconds']))==Fraction(l['native_pts'],12800),'Ledger source timestamp differs '+str(key));check(l['output_frame_index']==byshot[l['shot_id']],'Ledger output index differs '+str(key));check(l.get('source_UV_from_model')is False and l.get('physical_model_rendered')is False,'Ledger physical model scope '+str(key))
    if key in frames:
     fr=frames[key];check(l['source_BGR_sha256']==fr['decoded_native_BGR_sha256'],'Ledger source RGB binding differs '+str(key));check(l['selected_row_ids']==[p['selected_row_id']for p in fr['points']]and l['source_UV_count']==len(fr['points']),'Ledger drawn source IDs differ '+str(key))
    byshot[l['shot_id']]+=1;ledger_count+=1
   check(ledger_count==m['frames'],'Renderer/ledger frame count differs '+str(mp));check(dict(byshot)==m['by_shot'],'Renderer/ledger by-shot count differs '+str(mp))
  result['renderer_manifests'].append(dict(file=str(mp),sha256=pins[str(mp)],ledger=str(lp),frames=m['frames'],local_clock_rows=ledger_count))
  for v in m['videos']:
   sid=v['shot_id'];check(sid in SPECS,'Unexpected video shot '+sid)
   if sid in videos:check(videos[sid]['file']==v['file']and videos[sid]['sha256']==v['sha256'],'Duplicate video definitions differ '+sid)
   else:videos[sid]=v
 check(sorted(ledgerkeys)==sorted(expected)and len(ledgerkeys)==len(expected),'Ledgers do not cover each native frame exactly once');check(set(videos)==set(SPECS),'Final two videos missing or extra');check(assembly['videos']==[videos[s]for s in SPECS],'Assembly videos differ from exact child video definitions');result['videos']=[]
 for sid,v in videos.items():
  a,b=SPECS[sid];n=(b-a)//256+1;expected_duration=Fraction(n,50);p=Path(v['file']);pin(p,v['sha256']);check(v.get('output_frame_times_verified')is True,'Public renderer full encoded clock check absent '+sid);check(v['frames']==n and v['source_pts']==[a,b]and v['drawn_point_rows']==counts[sid],'Video export source count/PTS differs '+sid);check(v['source_seconds']==[float(Fraction(a,12800)),float(Fraction(b+256,12800))],'Video export source seconds differ '+sid);q=probe(p);qp=report.with_name(report.stem+'_'+sid+'_ffprobe.json');
  if qp.exists():raise FileExistsError('Preserve prior ffprobe receipt: '+str(qp))
  qp.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8');pin(qp);vs=[s for s in q['streams']if s['codec_type']=='video'];aud=[s for s in q['streams']if s['codec_type']=='audio'];check(len(vs)==1 and len(aud)==1,'Expected one video and one audio stream '+sid);rec=dict(shot_id=sid,file=str(p),sha256=v['sha256'],frames_expected=n,source_pts=[a,b],source_seconds=v['source_seconds'],ffprobe_file=str(qp))
  if vs:
   st=vs[0];tb=Fraction(st['time_base']);pk=[r for r in q.get('packets',[])if r['stream_index']==st['index']];pts=sorted(Fraction(int(r['pts']))*tb for r in pk);check(pts==[Fraction(i,50)for i in range(n)],'Encoded video packet exposure clocks differ '+sid);check(st['width']==1920 and st['height']==1080 and Fraction(st['avg_frame_rate'])==50 and Fraction(st['r_frame_rate'])==50 and tb==Fraction(1,12800)and int(st['nb_frames'])==n and Fraction(st.get('start_time','0'))==0,'Encoded video geometry/FPS/timebase/start/count differs '+sid);check(Fraction(int(st['duration_ts']))*tb==expected_duration,'Encoded video duration differs '+sid);rec['video']=dict(codec=st['codec_name'],time_base=str(tb),packets=len(pk),first_packet_time=float(pts[0])if pts else None,last_packet_time=float(pts[-1])if pts else None,duration=float(Fraction(int(st['duration_ts']))*tb),frame_clock_exact=pts==[Fraction(i,50)for i in range(n)])
  if aud:
   st=aud[0];tb=Fraction(st['time_base']);rate=int(st['sample_rate']);pk=[r for r in q.get('packets',[])if r['stream_index']==st['index']];start=Fraction(int(st.get('start_pts',0)))*tb;duration=Fraction(int(st['duration_ts']))*tb;clock=abs(start)<=Fraction(1,rate)and abs(start+duration-expected_duration)<=Fraction(1,rate);check(clock,'Audio playable stream clock differs from selected native segment '+sid);rec['audio']=dict(codec=st['codec_name'],sample_rate=rate,time_base=str(tb),playable_stream_start=float(start),duration=float(duration),expected_segment_duration=float(expected_duration),duration_error_seconds=float(start+duration-expected_duration),packets=len(pk),first_packet_pts=min(int(x['pts'])for x in pk)if pk else None,last_packet_end=max(int(x['pts'])+int(x.get('duration',0))for x in pk)if pk else None,first_packet_side_data=pk[0].get('side_data_list')if pk else None,playable_stream_clock_matches_native_segment=clock,source_audio_waveform_content_correlation_run=False)
  result['videos'].append(rec)
 for h,src in sourcefacts.items():
  q=probe(src['file'],False);result['official_source_streams']=q['streams'];check(any(s['codec_type']=='audio'for s in q['streams']),'Official source has no source audio stream');sv=[s for s in q['streams']if s['codec_type']=='video'];check(len(sv)==1 and sv[0]['width']==1920 and sv[0]['height']==1080 and Fraction(sv[0]['avg_frame_rate'])==50 and Fraction(sv[0]['time_base'])==Fraction(1,12800),'Actual official source stream differs')
except Exception as e:
 errors.append(type(e).__name__+': '+str(e))
result['pins']=pins;result['technical_checks_passed']=not errors;result['conclusion']='technical_clock_and_binding_passed_image_quality_separate'if not errors else'technical_check_failed';result['limitations']=['Packet clocks and renderer source bindings do not certify the drawn material ownership, pixel accuracy or full-frame manual viewing.','Audio source identity and playable segment clock are checked; waveform content alignment is not independently measured.','Source RGB and full encoded decoder checks are reused from the pinned public renderer manifest and production code; this metadata-only audit does not independently decode pixels.'];report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(report=str(report),technical_checks_passed=not errors,errors=errors,frames=result.get('selected_frames'),videos=[dict(shot_id=v['shot_id'],file=v['file'],video=v.get('video'),audio=v.get('audio'))for v in result.get('videos',[])]),ensure_ascii=False));sys.exit(0 if not errors else 1)
