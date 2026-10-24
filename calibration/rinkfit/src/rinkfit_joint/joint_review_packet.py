"""Read current review columns and requested raw identities without source replay.

This module is for offline fixed-frame review only. It consumes a self-contained
current packet and its optional current manifest, not a producer/history chain.
Geometry and drawing stay in the existing joint renderer.
"""
from pathlib import Path
from contextlib import contextmanager
import codecs, json, mmap, re
import numpy as np
from .io_utils import read_json, file_sha256
from .baseline_auxiliary import array_items, checked_array


def packet_field(path, key, default=None):
    with Path(path).open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            found = re.search(rb'"'+re.escape(key.encode())+rb'"\s*:', mm)
            if found is None: return default
            offset = found.end()
        stream.seek(offset); utf = codecs.getincrementaldecoder('utf8')(); text = ''; decoder = json.JSONDecoder()
        while True:
            chunk = stream.read(65536)
            if not chunk: raise ValueError('Truncated current review field: '+key)
            text += utf.decode(chunk)
            try: return decoder.raw_decode(text.lstrip())[0]
            except json.JSONDecodeError: pass


class CurrentReviewPacket:
    def __init__(self, packet, *, current_input=None, required_rows=()):
        self.path = Path(packet).resolve(); self.root = self.path.parent; self.checked = {}; self.arrays = {}; self._ids = {}
        self.required_rows = set(map(int, required_rows)); self.packet_sha = file_sha256(self.path)
        self.current = None; self.current_input_ref = None
        if current_input is not None:
            p = Path(current_input['file']).resolve()
            if file_sha256(p) != current_input['sha256']: raise ValueError('Changed current review manifest')
            self.current = read_json(p); self.current_root = p.parent; self.current_input_ref = current_input
            actual = Path(self.current['packet']['file']); actual = actual if actual.is_absolute() else p.parent/actual
            if (self.current.get('schema') != 'rinkfit.current_joint_input.v1' or actual.resolve() != self.path
                    or self.current['packet']['sha256'] != self.packet_sha):
                raise ValueError('Review current manifest/actual packet binding differs')
        metadata = {key:packet_field(self.path,key) for key in ('row_count','shot_ids','arrays')}
        materials = packet_field(self.path,'material_binding',{}).get('materials',[])
        self.material_labels = {int(m['index']):m['label'] for m in materials}
        if metadata['row_count'] is None or metadata['shot_ids'] is None or metadata['arrays'] is None:
            raise ValueError('Fixed current review requires self-contained packet columns')
        if self.current is not None and metadata['row_count'] != self.current['raw_rows']:
            raise ValueError('Current review raw denominator differs')
        self.registries = ([packet_field(self.path,'dense_source_binding',{})]
            + packet_field(self.path,'native_ad_source_bindings',[])
            + [packet_field(self.path,'native_ad_source_binding',{})])
        self.registries = [r for r in self.registries if r]
        self.prefix_count = min((int(r['row_start']) for r in self.registries),default=int(metadata['row_count']))
        owner = self
        class RawIDs:
            def __len__(self): return int(metadata['row_count'])
            def __getitem__(self, row):
                if isinstance(row,slice): raise TypeError('No full-domain ID hydration in fixed review')
                return owner.source_id(int(row))
        self.packet = dict(metadata,source_row_ids=RawIDs())
        if self.current is not None:
            q = self.current['qualification']
            self.current_source_membership_v1 = dict(schema='rinkfit.current_source_point_membership.v1',
                logical_source_id=self.packet_sha, source_logical_id=self.current['logical_source_id'],
                raw_rows=self.current['raw_rows'], current_input=current_input,
                physical=self._column(q['physical_mask'],self.current_root),
                training=self._column(q['training_mask'],self.current_root),
                fit_rows=self._column(q['fit_rows'],self.current_root),
                inspection_rows=self._column(q['inspection_rows'],self.current_root))

    def _column(self, desc, base):
        key = (str(Path(base).resolve()),desc['sha256'])
        if key not in self.checked: self.checked[key] = checked_array(desc,base)
        return self.checked[key]

    def load(self, name):
        if name not in self.arrays: self.arrays[name] = self._column(self.packet['arrays'][name],self.root)
        return self.arrays[name]

    def source_id(self, row):
        if not 0 <= row < int(self.packet['row_count']): raise ValueError('Review raw ID outside source domain')
        if row in self._ids: return self._ids[row]
        if row < self.prefix_count:
            wanted = {row} | {i for i in self.required_rows if 0 <= i < self.prefix_count}
            last_wanted = max(wanted)
            for i,value in enumerate(array_items(self.path,'source_row_ids')):
                if i in wanted: self._ids[i] = value
                if i >= last_wanted: break
            if row not in self._ids: raise ValueError('Missing current raw ID prefix entry')
            return self._ids[row]
        for reg in self.registries:
            local = row-int(reg['row_start'])
            if 0 <= local < int(reg['count']):
                offsets = self._column(reg['source_id_offsets'],self.root); blob = self._column(reg['source_id_utf8'],self.root)
                lo,hi = int(offsets[local]),int(offsets[local+1])
                if not 0 <= lo < hi <= len(blob): raise ValueError('Current raw ID byte range differs')
                self._ids[row] = bytes(blob[lo:hi]).decode('utf8'); return self._ids[row]
        raise ValueError('Missing current compact source ID')

    def source_binding(self, row):
        row = int(row); material = int(self.load('material_index')[row]); result = dict(
            source_row_id=self.source_id(row), native_pts=int(self.load('native_pts')[row]),
            shot_id=self.packet['shot_ids'][int(self.load('shot_index')[row])],
            source_uv=self.load('uv')[row].tolist(), material_index=material,
            path=int(self.load('path')[row]), material_label=self.material_labels.get(material), retired_material=material==int(np.iinfo(np.uint32).max),
            packet=dict(file=str(self.path),sha256=self.packet_sha))
        if self.current is not None:
            m = self.current_source_membership_v1
            contains = lambda a:bool(np.searchsorted(a,row)<len(a) and a[np.searchsorted(a,row)]==row)
            result['qualification'] = dict(physical=bool(m['physical'][row]),training=bool(m['training'][row]),
                representative_fit=contains(m['fit_rows']),inspection=contains(m['inspection_rows']),
                inspection_is_independent_validation=False)
            if result['retired_material'] and any(result['qualification'][k] for k in ('physical','training','representative_fit','inspection')):
                raise ValueError('Retired fixed source row still has current qualification')
            result['current_input_v1'] = self.current_input_ref
        return result


@contextmanager
def source_point_reader(reader):
    """Install only the source-column reader for this sequential render call."""
    from . import joint_display
    previous = joint_display.PacketArrays
    def exact_reader(path):
        if Path(path).resolve() != reader.path: raise ValueError("Renderer requested another side's source packet")
        return reader
    joint_display.PacketArrays = exact_reader
    try: yield
    finally: joint_display.PacketArrays = previous
