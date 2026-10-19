"""Bounded-memory provenance repair for immutable rank-zero source groups.

Copy the raw registry, all scalar descriptors and the rest of the packet byte
for byte. Decode one statistical group at a time, and only add missing source
references to an already explicit unknown-noise, zero-rank, empty-W group.
"""
from pathlib import Path
import json
from .io_utils import file_sha256, write_json


def repair_rankzero_source_provenance(packet_path, output):
    source=Path(packet_path).resolve();folder=Path(output).resolve()
    if folder.exists():raise ValueError('Immutable provenance repair output exists')
    folder.mkdir(parents=True)
    target=folder/'packet.json';marker='"groups":[';decoder=json.JSONDecoder();changed=[];count=0
    ref=dict(file=str(source),sha256=file_sha256(source),role='immutable_parent_raw_source_ledger_not_source_qualification')
    with source.open('r',encoding='utf8',newline='')as stream,target.open('x',encoding='utf8',newline='')as sink:
        buffer=''
        while True:
            text=stream.read(1024*1024)
            if not text:raise ValueError('Packet has no compact immutable group array')
            buffer+=text;index=buffer.find(marker)
            if index>=0:
                sink.write(buffer[:index+len(marker)]);buffer=buffer[index+len(marker):];break
            keep=len(marker)-1
            if len(buffer)>keep:sink.write(buffer[:-keep]);buffer=buffer[-keep:]
        position=0;first=True
        while True:
            while True:
                while position<len(buffer)and buffer[position].isspace():position+=1
                if position<len(buffer):break
                buffer+=stream.read(1024*1024)
                if position==len(buffer):raise ValueError('Truncated group array')
            if buffer[position]==']':
                sink.write(buffer[position:]);break
            if not first:
                if buffer[position]!=',':raise ValueError('Malformed group separator')
                position+=1
                while position<len(buffer)and buffer[position].isspace():position+=1
            while True:
                try:group,end=decoder.raw_decode(buffer,position);break
                except json.JSONDecodeError:
                    text=stream.read(1024*1024)
                    if not text:raise ValueError('Truncated statistical group')
                    buffer+=text
            if not first:sink.write(',')
            count+=1
            if group.get('statistic_kind')=='constraint_only_unknown_noise'and not group.get('unknown_noise_source_refs'):
                if group.get('rank')!=0 or group.get('W')!=[]:raise ValueError('Unknown noise cannot carry a precision factor')
                group['unknown_noise_source_refs']=[ref];changed.append(group['id'])
                sink.write(json.dumps(group,ensure_ascii=False,separators=(',',':'),allow_nan=False))
            else:sink.write(buffer[position:end])
            buffer=buffer[end:];position=0;first=False
        while True:
            text=stream.read(1024*1024)
            if not text:break
            sink.write(text)
    report=dict(schema='rinkfit.streaming_rankzero_source_provenance_repair.v1',parent_packet=ref,
        packet=dict(file=str(target),sha256=file_sha256(target)),groups=count,repaired_groups=len(changed),group_ids=changed,
        source_arrays_IDs_UV_PTS_operator_membership_certificate_and_W_unchanged=True,
        all_content_outside_groups_exact_bytes_copied=True,
        unmodified_group_serialization_bytes_preserved=True,
        max_read_chunk_bytes=1024*1024,one_group_decoded_at_a_time=True,
        native_qualification_check_required=True,new_source_qualification_created=False,new_precision_created=False)
    write_json(folder/'report.json',report);return report
