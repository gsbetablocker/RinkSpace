"""Read existing qualified fixed corners by their exact packet measurement.

This reader supplies real tracking anchors. It does not issue qualification for
new exposures, overwrite historical feedback caches or obtain UV from a model.
"""
from pathlib import Path
import json
import numpy as np
from .io_utils import file_sha256, object_sha256, read_jsonl
from .board_source_qualification_guard import _load_ref

VERSION='exact_packet_qualified_fixed_corner_journal_v1'


def reference(path):
    path=Path(path).resolve()
    return dict(file=str(path),sha256=file_sha256(path))


class PacketBoundSourceReview:
    def __init__(self,packet_path,qualification_gate_refs,*,requested_row_indices):
        self.packet_path=Path(packet_path).resolve()
        self.packet=json.loads(self.packet_path.read_text(encoding='utf8'))
        self.seen=set();self.arrays={};self.rows={};self.pending={}
        self.requested=sorted(set(map(int,requested_row_indices)))
        n=int(self.packet['row_count'])
        if not self.requested or self.requested[0]<0 or self.requested[-1]>=n:raise ValueError('Exact packet row indices are required')
        for name in ('native_pts','uv','kind','material_index','shot_index','admission','fit_role'):
            d=self.packet['arrays'][name];p=_load_ref(d,self.packet_path.parent,self.seen)
            self.arrays[name]=np.memmap(p,dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))
        by_id={}
        for i in self.requested:
            if self.arrays['kind'][i]!=1 or self.arrays['admission'][i]!=1 or self.arrays['fit_role'][i]!=1:
                raise ValueError('Only existing active fixed-entity source rows may seed this reader')
            by_id.setdefault(self.packet['source_row_ids'][i],[]).append(i)
        for gate_ref in qualification_gate_refs:
            gatepath=_load_ref(gate_ref,self.packet_path.parent,self.seen)
            gate=json.loads(gatepath.read_text(encoding='utf8'))
            if gate.get('schema')!='rinkfit.new_actual_type1_source_qualification_gate.v1':raise ValueError('An actual fixed entity source gate is required')
            if gate.get('old_raw_source_flags_false_are_not_overwritten')is not True or gate.get('new_measurement_version_geometry_qualification_only')is not True:raise ValueError('Source gate rewrites original source history')
            receiptpath=_load_ref(gate['qualification_receipts'],gatepath.parent,self.seen)
            journalpath=_load_ref(gate['input_cumulative_journal'],gatepath.parent,self.seen)
            _load_ref(gate['source_mean_and_floating_estimator_replay_QA'],gatepath.parent,self.seen)
            receipts={}
            for r in read_jsonl(receiptpath):
                rid=r['source_row_id']
                if rid not in by_id:continue
                for i in by_id[rid]:
                    expected=object_sha256(dict(source_row_id=rid,native_pts=int(self.arrays['native_pts'][i]),current_2D_UV=self.arrays['uv'][i].tolist(),estimator='native_two_edge_quadratic_intersection_v1'))
                    if (r.get('current_2D_source_mean_sha256')==expected and int(r['native_pts'])==int(self.arrays['native_pts'][i])
                        and r.get('existing_material_index')==int(self.arrays['material_index'][i]) and r.get('new_actual_source_geometry_qualified')is True
                        and r.get('entity_kind')==1 and r.get('physical_point')is True and r.get('all_source_support_pixels_unmasked')is True):
                        receipts.setdefault(rid,[]).append((i,r))
            if not receipts:continue
            for j in read_jsonl(journalpath):
                rid=j['source_row_id']
                for i,r in receipts.get(rid,[]):
                    # A locator may survive several measurement generations.
                    # Only this exact frozen mean and original UV may own it.
                    if i in self.rows:continue
                    if (j.get('current_uv')!=self.arrays['uv'][i].tolist() or int(j.get('native_pts',-1))!=int(self.arrays['native_pts'][i])
                        or j.get('source_uv')!=r['original_UV_preserved'] or j.get('target_id')!=r['target_id']):continue
                    op=j.get('current_corner_operator',{});audit=j.get('source_measurement_audit',{}).get('current_two_material_edges',{})
                    if not audit.get('valid') or audit.get('current_intersection_uv')!=j['current_uv'] or op.get('all_support_pixels_unmasked')is not True:continue
                    identity=next((v for k,v in j.items()if k.startswith('same_actual_physical_Ad')and k.endswith('_source_review')),r.get('same_actual_physical_source_review'))
                    if identity is None:continue
                    _load_ref(identity,journalpath.parent,self.seen)
                    frame=j.get('current_native_frame_source',{})
                    if (r.get('current_native_RGB_sha256')!=frame.get('RGB_sha256') or r.get('current_native_mask_sha256')!=frame.get('mask_sha256')):continue
                    _load_ref(dict(file=frame['file'],sha256=frame['sha256']),journalpath.parent,self.seen)
                    _load_ref(dict(file=frame['mask_file'],sha256=frame['mask_sha256']),journalpath.parent,self.seen)
                    self.rows[i]=dict(raw_index=i,source_row_id=rid,source_dataset_id=self.packet['source_dataset_ids'][i],
                        native_pts=int(self.arrays['native_pts'][i]),shot_id=self.packet['shot_ids'][int(self.arrays['shot_index'][i])],
                        packet_uv=self.arrays['uv'][i].tolist(),journal=j,journal_ref=gate['input_cumulative_journal'],
                        qualification_gate=gate_ref,qualification_receipt=dict(ledger=gate['qualification_receipts'],source_row_id=rid,receipt_sha256=object_sha256(r)),
                        receipt=r,physical_instance_review=identity,packet_mean_exact=True,original_source_uv_exact=True,
                        new_exposure_qualification_issued=False)
        for i in self.requested:
            if i not in self.rows:self.pending[i]='no_exact_packet_mean_original_UV_clock_ID_and_qualified_finite_source_journal_match'
        self.reader_ref=reference(Path(__file__))

    def provenance(self):
        return dict(version=VERSION,reader_module=self.reader_ref,source_packet=reference(self.packet_path),
            requested_rows=len(self.requested),exact_qualified_existing_rows=len(self.rows),pending_rows=len(self.pending),
            current_packet_uv_or_membership_changed=False,new_exposure_qualification_issued=False,
            caches_or_historical_source_generations_changed=False,model_predictions_used=False)
