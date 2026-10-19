"""Replace an accepted demo ending while copying complete closed GOPs and AAC.

Only the final incomplete GOP before the replacement boundary is decoded from
the accepted finished video and encoded again. No scene/trajectory renderer runs.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'calibration/program_tracking')]
import av
import numpy as np
from calibration.program_tracking.assemble_demo_full import file_info, write_json
from calibration.program_tracking.demo_source import check_output_path
from calibration.program_tracking.demo_particle_pass import _encoder_quality
from calibration.program_tracking.render_result import _available_encoder, _ffmpeg_executable
from calibration.program_tracking.render_demo_intro_v2 import (
    _check_video, _contains_idr, _sequence_hash, _video_packets,
    FPS, TIME_BASE, TICK_STEP,
)

DATA = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp')
OLD = DATA/'results/demo_full_20261016/v3/RinkSpace_full_demo_v3.mp4'
OUTRO = DATA/'results/demo_outro_stats_20261016/RinkSpace_stats_outro_only.mp4'
OUT = DATA/'results/demo_full_20261016/final'
KEEP_FRAMES = 8844
EXPECTED_OLD_FRAMES = 9054
EXPECTED_OUTRO_FRAMES = 650


def _status(output, stage, **extra):
    value = dict(status='running', stage=stage, **extra)
    write_json(output/'assembly_status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def _encode_tail(old, target, output, start, end):
    """Decode only the old accepted video frames in the final incomplete GOP."""
    ffmpeg = _ffmpeg_executable()
    args, encoder = _available_encoder(ffmpeg)
    args = _encoder_quality(args, encoder)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1920x1080',
               '-framerate', str(FPS), '-i', 'pipe:0', *args, '-pix_fmt', 'yuv420p',
               '-an', '-frames:v', str(end-start), '-video_track_timescale', '12800',
               '-movflags', '+faststart', str(target)]
    write_json(output/'tail_encode_command.json', command)
    count = 0
    with (output/'tail_encode.stderr.log').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=log, creationflags=0x08000000)
        try:
            with av.open(str(old)) as source:
                stream = source.streams.video[0]
                source.seek(start*TICK_STEP, stream=stream, backward=True, any_frame=False)
                for frame in source.decode(stream):
                    exact = Fraction(frame.pts)*frame.time_base/TIME_BASE
                    if exact.denominator != 1:
                        raise ValueError('Old decoded frame leaves native tick grid')
                    pts = int(exact)
                    if pts < start*TICK_STEP:
                        continue
                    if pts >= end*TICK_STEP:
                        break
                    if pts != (start+count)*TICK_STEP:
                        raise ValueError('Old tail frame PTS is incomplete')
                    process.stdin.write(frame.to_ndarray(format='bgr24').tobytes())
                    count += 1
        finally:
            process.stdin.close()
            result = process.wait()
        if result:
            raise RuntimeError((output/'tail_encode.stderr.log').read_text(errors='replace')[-3000:])
    if count != end-start:
        raise ValueError('Reencoded tail decode frame count differs')
    return dict(frames=count, start_frame=start, end_frame_exclusive=end,
                source='decoded accepted completed video; no effect renderer',
                encoder=encoder, encoder_args=args, command=str(output/'tail_encode_command.json'))


def _join_video(old, tail, outro, target, prefix_frames, keep_frames):
    old_extra, old_packets = _video_packets(old)
    tail_extra, tail_packets = _video_packets(tail)
    outro_extra, outro_packets = _video_packets(outro)
    if old_extra != tail_extra or old_extra != outro_extra:
        raise ValueError('Copied and reencoded segments must have identical AVC extradata')
    segments = ((old, prefix_frames, 0),
                (tail, keep_frames-prefix_frames, prefix_frames*TICK_STEP),
                (outro, len(outro_packets), keep_frames*TICK_STEP))
    count = 0
    previous = None
    with av.open(str(old)) as first, av.open(str(target), 'w', format='mp4',
            options={'movflags':'+faststart', 'video_track_timescale':'12800'}) as joined:
        output_stream = joined.add_stream_from_template(first.streams.video[0])
        output_stream.time_base = TIME_BASE
        for path, expected, offset in segments:
            local = 0
            with av.open(str(path)) as source:
                stream = source.streams.video[0]
                for packet in source.demux(stream):
                    if packet.size <= 0:
                        continue
                    if local >= expected:
                        break
                    packet.pts = int(packet.pts)+offset
                    packet.dts = int(packet.dts)+offset
                    packet.time_base = TIME_BASE
                    if previous is not None and packet.dts-previous != TICK_STEP:
                        raise ValueError('Join has a decode-clock gap')
                    previous = int(packet.dts)
                    packet.stream = output_stream
                    joined.mux(packet)
                    local += 1
                    count += 1
            if local != expected:
                raise ValueError('Join segment has an incorrect packet count')
    _, actual = _video_packets(target)
    expected_total = keep_frames+len(outro_packets)
    if count != expected_total or len(actual) != expected_total:
        raise ValueError('Join frame count differs')
    if actual[:prefix_frames] != old_packets[:prefix_frames]:
        raise ValueError('Accepted prefix payload/PTS/DTS changed')
    tail_normalized = [(p[0]-prefix_frames*TICK_STEP, p[1]-prefix_frames*TICK_STEP,
                        p[2], p[3], p[4]) for p in actual[prefix_frames:keep_frames]]
    outro_normalized = [(p[0]-keep_frames*TICK_STEP, p[1]-keep_frames*TICK_STEP,
                         p[2], p[3], p[4]) for p in actual[keep_frames:]]
    if tail_normalized != tail_packets or outro_normalized != outro_packets:
        raise ValueError('Tail or approved ending compressed payload changed in join')
    if sorted(p[0] for p in actual) != list(range(0, expected_total*TICK_STEP, TICK_STEP)):
        raise ValueError('Join display PTS grid is incomplete')
    return dict(frames=expected_total, extradata_identical=True,
                extradata_sha256=hashlib.sha256(old_extra).hexdigest(),
                retained_prefix_frames=prefix_frames,
                retained_prefix_seconds=prefix_frames/FPS,
                retained_prefix_payload_pts_dts_exact=True,
                selected_splice_anchor_actual_idr=True,
                retained_prefix_payload_sequence_sha256=_sequence_hash(actual[:prefix_frames]),
                tail_reencoded_frames=keep_frames-prefix_frames,
                approved_outro_frames=len(outro_packets),
                approved_outro_output_start_frame=keep_frames,
                approved_outro_payload_shifted_pts_dts_exact=True,
                approved_outro_payload_sequence_sha256=_sequence_hash(outro_normalized))


def _audio_identities(path):
    with av.open(str(path)) as source:
        stream = source.streams.audio[0]
        if stream.codec_context.name != 'aac' or stream.rate != 48000:
            raise ValueError('Expected the accepted 48kHz AAC track')
        extra = bytes(stream.codec_context.extradata)
        rate = int(stream.rate)
        rows = [(int(p.pts), int(p.dts), int(p.duration),
                 hashlib.sha256(bytes(p)).hexdigest())
                for p in source.demux(stream) if p.size > 0]
    return extra, rate, rows


def _mux_extend_original_audio(video, old, target, total_frames):
    """Keep original AAC payloads and append copies of its actual silent AAC."""
    old_extra, rate, original = _audio_identities(old)
    if original[0][:3] != (-1024, -1024, 1024):
        raise ValueError('Unexpected AAC priming convention')
    for i, row in enumerate(original):
        if row[0] != (i-1)*1024 or row[1] != row[0]:
            raise ValueError('Accepted AAC clock is not continuous')
    if len({row[3] for row in original[-32:]}) != 1:
        raise ValueError('Last original AAC packets are not a repeated silent payload')
    # Verify silence from the real accepted file, rather than assuming packet size.
    silent_peak = 0.
    silent_values = 0
    with av.open(str(old)) as source:
        stream = source.streams.audio[0]
        source.seek(original[-32][0], stream=stream, backward=True, any_frame=False)
        for frame in source.decode(stream):
            start = Fraction(frame.pts)*frame.time_base*rate
            if start >= original[-31][0]:
                values = frame.to_ndarray().astype(np.float64)
                silent_peak = max(silent_peak, float(np.abs(values).max()))
                silent_values += values.size
    if silent_values == 0 or silent_peak > 1e-8:
        raise ValueError('Accepted AAC ending must decode to exact silence')
    endpoint = total_frames*rate//FPS
    if endpoint != Fraction(total_frames, FPS)*rate:
        raise ValueError('Final duration is not an integral audio sample count')
    old_video = _video_packets(video)[1]
    copied = 0
    appended = 0
    final_duration = None
    with av.open(str(video)) as vsource, av.open(str(old)) as asource, av.open(
            str(target), 'w', format='mp4',
            options={'movflags':'+faststart', 'video_track_timescale':'12800'}) as output:
        outv = output.add_stream_from_template(vsource.streams.video[0])
        outa = output.add_stream_from_template(asource.streams.audio[0])
        outv.time_base = TIME_BASE
        outa.time_base = Fraction(1, rate)

        def video_packets():
            for packet in vsource.demux(vsource.streams.video[0]):
                if packet.size > 0:
                    packet.stream = outv
                    yield packet

        def audio_packets():
            nonlocal copied, appended, final_duration
            last_payload = None
            next_pts = -1024
            for packet in asource.demux(asource.streams.audio[0]):
                if packet.size <= 0:
                    continue
                if int(packet.pts) != next_pts or next_pts >= endpoint:
                    raise ValueError('Original audio does not fit extended endpoint')
                last_payload = bytes(packet)
                packet.duration = min(1024, endpoint-next_pts)
                packet.time_base = Fraction(1, rate)
                packet.stream = outa
                final_duration = int(packet.duration)
                copied += 1
                next_pts += 1024
                yield packet
            if last_payload is None:
                raise ValueError('Original AAC is empty')
            while next_pts < endpoint:
                packet = av.Packet(last_payload)
                packet.pts = packet.dts = next_pts
                packet.duration = min(1024, endpoint-next_pts)
                packet.time_base = Fraction(1, rate)
                packet.stream = outa
                packet.is_keyframe = True
                appended += 1
                next_pts += 1024
                final_duration = int(packet.duration)
                yield packet

        video_iter, audio_iter = iter(video_packets()), iter(audio_packets())
        vp, ap = next(video_iter, None), next(audio_iter, None)
        while vp is not None or ap is not None:
            if ap is None or (vp is not None and Fraction(vp.dts)*vp.time_base <= Fraction(ap.dts)*ap.time_base):
                output.mux(vp)
                vp = next(video_iter, None)
            else:
                output.mux(ap)
                ap = next(audio_iter, None)
    final_extra, final_rate, actual = _audio_identities(target)
    if old_extra != final_extra or rate != final_rate:
        raise ValueError('AAC codec configuration changed')
    if [r[3] for r in actual[:len(original)]] != [r[3] for r in original]:
        raise ValueError('Original AAC compressed payload changed')
    if actual[-1][0]+actual[-1][2] != endpoint:
        raise ValueError('Final AAC endpoint differs from video')
    for i, row in enumerate(actual):
        if row[0] != (i-1)*1024 or row[1] != row[0]:
            raise ValueError('Extended AAC has a sparse timestamp')
        if i < len(actual)-1 and row[2] != 1024:
            raise ValueError('Internal AAC packet duration is not 1024 samples')
    if _video_packets(target)[1] != old_video:
        raise ValueError('Audio mux changed video payload or timestamps')
    return dict(sample_rate=rate, original_aac_packets_copied=copied,
                appended_silent_aac_packets=appended,
                original_audio_payload_exact=True, original_audio_reencoded=False,
                old_final_duration_samples=original[-1][2],
                old_final_silent_packet_extended_to_samples=1024,
                silent_aac_payload_sha256=original[-1][3],
                decoded_source_tail_silence_peak=silent_peak,
                internal_sample_step=1024, final_packet_samples=final_duration,
                final_sample_endpoint=endpoint, total_packets=len(actual),
                audio_clock_continuous=True, video_payload_pts_dts_exact=True)


def replace(*, old=OLD, outro=OUTRO, output=OUT, keep_frames=KEEP_FRAMES):
    started = time.perf_counter()
    old, outro, output = Path(old), Path(outro), Path(output)
    for path in (output/'tail_reencoded.mp4', output/'joined_video.mp4',
                 output/'RinkSpace_full_demo_final.mp4', output/'assembly_report.json'):
        check_output_path(path)
    output.mkdir(parents=True, exist_ok=True)
    old_extra, old_packets = _video_packets(old)
    outro_extra, outro_packets = _video_packets(outro)
    if len(old_packets) != EXPECTED_OLD_FRAMES or len(outro_packets) != EXPECTED_OUTRO_FRAMES:
        raise ValueError('Accepted source or ending frame count differs')
    if old_extra != outro_extra:
        raise ValueError('Accepted full video and approved ending codec configurations differ')
    candidates = [i for i, row in enumerate(old_packets)
                  if row[4] and row[0] < keep_frames*TICK_STEP]
    prefix = candidates[-1]
    if old_packets[prefix][0] != prefix*TICK_STEP:
        raise ValueError('Last IDR packet does not align to its closed GOP display index')
    if sorted(row[0] for row in old_packets[:prefix]) != list(range(0, prefix*TICK_STEP, TICK_STEP)):
        raise ValueError('Retained prefix contains a cross-GOP display frame dependency')
    with av.open(str(old)) as source:
        stream = source.streams.video[0]
        for index, packet in enumerate(p for p in source.demux(stream) if p.size > 0):
            if index == prefix:
                if not _contains_idr(bytes(packet), old_extra):
                    raise ValueError('Retained prefix must end immediately before an actual IDR')
                break
    _status(output, 'tail_encode', retained_prefix_frames=prefix,
            tail_reencode_frames=keep_frames-prefix)
    tail = output/'tail_reencoded.mp4'
    encoded = _encode_tail(old, tail, output, prefix, keep_frames)
    _status(output, 'copy_video_segments')
    joined = output/'joined_video.mp4'
    join = _join_video(old, tail, outro, joined, prefix, keep_frames)
    _status(output, 'copy_and_extend_audio')
    final = output/'RinkSpace_full_demo_final.mp4'
    audio = _mux_extend_original_audio(joined, old, final, join['frames'])
    report = dict(schema='rinkspace.demo-ending-replacement.v1', status='complete',
                  video=file_info(final), source_full=file_info(old),
                  approved_outro=file_info(outro),
                  frames=join['frames'], fps=FPS, width=1920, height=1080,
                  duration_seconds=join['frames']/FPS, intro_frames=410,
                  programme_frames=8434, previous_outro_removed_frames=210,
                  new_outro_frames=len(outro_packets),
                  whole_programme_rerendered=False, effects_renderer_run=False,
                  short_tail_transcode=encoded, video_join=join, audio=audio,
                  source_time_base='1/12800', source_first_pts=256256,
                  source_last_pts_inclusive=2415104,
                  source_seconds_inclusive=[20.02,188.68],
                  wall_seconds=time.perf_counter()-started)
    write_json(output/'assembly_report.json', report)
    write_json(output/'assembly_status.json', report)
    print(json.dumps(dict(status='complete', video=str(final), frames=join['frames'],
                          retained_prefix_frames=prefix, tail_reencoded_frames=keep_frames-prefix,
                          duration_seconds=join['frames']/FPS)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=OLD)
    parser.add_argument('--outro', type=Path, default=OUTRO)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    replace(old=args.old, outro=args.outro, output=args.output)
