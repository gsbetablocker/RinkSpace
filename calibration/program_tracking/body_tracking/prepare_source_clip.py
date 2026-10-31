"""Prepare the fixed S01 private review clip while preserving packets and PTS.

Recovered from the 2026-10-29 driver; this creates local media only when invoked.
Original SHA, copied packet identity and all 1143 decoded exposures are checked.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import av,json,hashlib,time,subprocess,imageio_ffmpeg
from rinkspace.storage import AssetStore


def prepare(review_path, output):
    project = Path(__file__).resolve().parents[3]
    out = Path(output).resolve()
    if out.is_relative_to(project):
        raise ValueError("Source media outputs must stay outside the repository")
    review = json.loads(Path(review_path).read_text(encoding="utf8"))
    source = AssetStore(project).source(review["source"]["registered_path"])
    out.mkdir(parents=True, exist_ok=True)
    target, receipt = out / "source_s01.mp4", out / "source_clip_receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    first, last, start = 283136, 575488, time.time()
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if target.exists():raise FileExistsError(target)
    packets=[]
    with av.open(str(source)) as original:
     s=original.streams.video[0];original.seek(first,stream=s,backward=True,any_frame=False)
     for packet in original.demux(s):
      if packet.pts is None:continue
      if packet.pts>last:break
      packets.append(dict(pts=packet.pts,dts=packet.dts,bytes=packet.size,sha256=hashlib.sha256(bytes(packet)).hexdigest(),keyframe=packet.is_keyframe))
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-copyts','-ss',str(packets[0]['pts']/12800),'-i',str(source),'-map','0:v:0','-to',str((last+256)/12800),'-c:v','copy','-an','-sn','-dn','-avoid_negative_ts','disabled','-video_track_timescale','12800',str(target)]
    subprocess.run(command,check=True)
    print(json.dumps({'phase':'slice_muxed','frames_with_preroll':len(packets),'bytes':target.stat().st_size}),flush=True)
    with av.open(str(target)) as c:
     s=c.streams.video[0];copied=[dict(pts=p.pts,dts=p.dts,bytes=p.size,sha256=hashlib.sha256(bytes(p)).hexdigest(),keyframe=p.is_keyframe) for p in c.demux(s) if p.pts is not None];tb=str(s.time_base)
    assert packets==copied, 'remuxed packet identity or source PTS changed'
    def pixels(file):
     results={}
     with av.open(str(file)) as c:
      s=c.streams.video[0];s.codec_context.thread_count=2;c.seek(first,stream=s,backward=True,any_frame=False)
      for frame in c.decode(s):
       if frame.pts>last:break
       if frame.pts>=first:results[frame.pts]=hashlib.sha256(frame.to_ndarray(format='bgr24').tobytes()).hexdigest()
     return results
    original_pixels=pixels(source);copied_pixels=pixels(target)
    assert list(original_pixels)==list(range(first,last+256,256));assert original_pixels==copied_pixels
    record=dict(schema='rinkspace.lossless-source-bitstream-slice.v1',original_source=review['source'],original_sha256_verified=sha(source)==review['source']['sha256'],slice_name=target.name,slice_sha256=sha(target),slice_bytes=target.stat().st_size,time_base=tb,target_first_pts=first,target_last_pts_inclusive=last,target_exposures=len(original_pixels),decode_preroll_first_pts=packets[0]['pts'],slice_packets_including_preroll=len(packets),video_codec='AV1 copied packets',reencoded=False,audio_included=False,all_packet_bytes_and_pts_identical=True,all_target_decoded_bgr_pixels_identical=True,pixel_hashes_by_pts=original_pixels,packet_identity=packets,elapsed_seconds=time.time()-start)
    assert record['original_sha256_verified'];receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in record.items() if k not in ['pixel_hashes_by_pts','packet_identity','original_source']},ensure_ascii=False),flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.review, args.out)


if __name__ == "__main__":
    main()
