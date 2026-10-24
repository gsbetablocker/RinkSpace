"""Compare the numerical contents of two self-contained current inputs.

Native frame labels and redundant summary metadata do not change the source
measurements, factor support, W, E, Query or physical model.
"""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def same_numerical_input(previous_packet, current_packet):
    old=read(previous_packet);new=read(current_packet)
    def signature(d):return {k:d[k]for k in ('sha256','dtype','shape')}
    if old['observations']!=new['observations']or old['immutable_source_hashes']!=new['immutable_source_hashes']:return False
    if set(old['arrays'])!=set(new['arrays'])or any(signature(old['arrays'][k])!=signature(new['arrays'][k])for k in old['arrays']):return False
    a=read(old['semantic_packet']['manifest']);b=read(new['semantic_packet']['manifest'])
    if set(a['arrays'])!=set(b['arrays'])or any(signature(a['arrays'][k])!=signature(b['arrays'][k])for k in a['arrays']):return False
    for key in ('resolved_material_index','resolved_variant_index','active_source_mask','statistical_active_mask'):
        if signature(a[key])!=signature(b[key]):return False
    qa=read(old['query_packet']['manifest']);qb=read(new['query_packet']['manifest'])
    if set(qa['arrays'])!=set(qb['arrays'])or any(signature(qa['arrays'][k])!=signature(qb['arrays'][k])for k in qa['arrays'])or qa['certificates']['sha256']!=qb['certificates']['sha256']:return False
    if a['assets']['groups.jsonl']['sha256']!=b['assets']['groups.jsonl']['sha256']:
        def groups(s):
            return [{k:v for k,v in json.loads(line).items()if k not in ('native_pts','frame')}for line in Path(s['assets']['groups.jsonl']['file']).read_text(encoding='utf8').splitlines()]
        if groups(a)!=groups(b):return False
    return a['current_input_scope']==b['current_input_scope']and old['event_model']['sha256']==new['event_model']['sha256']
