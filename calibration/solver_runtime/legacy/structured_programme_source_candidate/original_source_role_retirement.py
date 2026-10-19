"""Retire one source-proven false print role without rewriting its history.

The original raw observation, sigma, row slots, masks and profile buffers stay
intact. Only the two active residual/J rows become structural zeros. They are
explicitly excluded from print FIT accuracy statistics, not perfect matches.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json,pickle
import numpy as np
from z190_streaming_source_normal import SourceRowBlock

CORRECTION_SHA256='427893fe8a865ca1fe8afff44c12f49f76ace9c049a99572d9fdba558d022d1e'
PACKET_SHA256='ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78'
SOURCE_SHA256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
GLOBAL_ROWS=np.array([1478186,1478187],dtype=np.int64)
SCHEMA='rinkspace.source-proven-original-print-role-retirement.v1'

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))

def _locator(item):
    return (item['block'],item['collection_role'],int(item['group_actual_pts']),int(item['group_order']),int(item['row_order']))

class OriginalSourceRoleRetirement:
    def __init__(self,packet_path):
        path=Path(packet_path).resolve()
        if _sha(path)!=CORRECTION_SHA256:raise ValueError('Reviewed original print-role correction authority changed')
        data=json.loads(path.read_text(encoding='utf8'))
        if(data['schema']!='rinkspace.original-source-role-correction-candidate.v1'or
                data['source_video_sha256']!=SOURCE_SHA256 or data['time_base']!='1/15360'):
            raise ValueError('The exact same-event native source correction is required')
        parent=data['original_absolute_packet'];original_path=Path(parent['path'])
        if parent['sha256']!=PACKET_SHA256 or _sha(original_path)!=PACKET_SHA256:
            raise ValueError('Immutable original programme source authority changed')
        original=pickle.loads(original_path.read_bytes())['programme_absolute_rows']
        bad=[x for x in original if _locator(x)==('secondary','FIT',230445824,131,2)]
        good=[x for x in original if _locator(x)==('secondary','FIT',230445824,131,6)]
        if len(bad)!=1 or len(good)!=1:raise ValueError('Ambiguous original source instances')
        self.original=bad[0]['source_row'];self.retained=good[0]['source_row']
        if _plain(self.original)!=data['original_source_row_preserved']:
            raise ValueError('Correction changed original source UV, sigma, operator or history')
        for row,uv,sigma in ((self.original,[[1599.74,456.08]],[15.77]),(self.retained,[[1594.,466.]],[10.])):
            if(row['id']!='ad7:right_top'or not np.array_equal(row['uv'],uv)or not np.array_equal(row['sigma'],sigma)):
                raise ValueError('The false rail sample and retained print corner must stay distinct')
        for ref in [data['original_native_raw'],data['evidence'],*data['evidence']['review_images']]:
            if _sha(ref['path'])!=ref['sha256']:raise ValueError('Source-only identity evidence changed')
        self.contract=dict(schema=SCHEMA,packet_path=str(path),packet_sha256=CORRECTION_SHA256,
            original_absolute_packet=parent,source_row_locator=data['source_row_locator'],
            source_video_sha256=SOURCE_SHA256,time_base='1/15360',global_scalar_slots=GLOBAL_ROWS.tolist(),
            active_FIT_scalars_retired=2,source_scalar_slots_removed=0,new_FIT_scalars_added=0,
            observed_identity=data['observed_identity'],pending_source_id=data['proposed_revision']['pending_source_id'],
            original_native_raw=data['original_native_raw'],confirmed_print_corner_retained=data['confirmed_print_corner_retained'],
            source_UV_sigma_origin_operator_and_historical_FIT_exact=True,
            quadratic_scales_profile_weights_and_CSR_slot_layout_preserved=True,
            unverified_rail_height_or_station_fitted=False,replacement_print_corner_added=False,
            inactive_zero_residual_is_accuracy_measurement=False,exclude_from_active_print_FIT_accuracy_statistics=True,
            accuracy_accepted=False)

    def qualify_state(self,state):
        """Locate from complete original block/group order, not a guessed offset."""
        remap=getattr(state,'original_motion_compaction_scalar_remap',None)
        active_rows=GLOBAL_ROWS
        if remap is not None:
            if not isinstance(remap,dict) or any(int(row)not in remap for row in GLOBAL_ROWS):
                raise ValueError('Original retired scalar slots require an explicit compact row mapping')
            values=[remap[int(row)]for row in GLOBAL_ROWS]
            if any(isinstance(v,bool)or not isinstance(v,(int,np.integer))or v < -1 for v in values):
                raise ValueError('Compact retired scalar slots must be integer rows or explicit -1')
            active_rows=np.asarray([v for v in values if v>=0],np.int64)
            if len(active_rows)not in(0,2)or(len(active_rows)==2 and active_rows[1]!=active_rows[0]+1):
                raise ValueError('Both original print scalar components must share their compact status')
        cursor=0;found=[]
        for block in state.blocks:
            if block.name=='secondary':
                if len(block.groups)<=131:raise ValueError('Original secondary group order absent')
                pts,rows=block.groups[131]
                if int(pts)!=230445824 or len(rows)<=6:raise ValueError('Original source exposure/order changed')
                for runtime,original in ((rows[2],self.original),(rows[6],self.retained)):
                    for key in ('id','ad_id','role','uv','sigma','source_uv','sigma_px','pts','split','origin',
                            'point_correspondence','curve_correspondence','curve_observation_kind',
                            'normal','normal_interval','point_support_bounds_xy_px',
                            'scanline_fixed_axis','scanline_branch','extent_side'):
                        if _plain(runtime.get(key))!=_plain(original.get(key)):
                            raise ValueError('Original source identity/support changed: '+key)
                before=sum(2*len(row['uv'])for _,group in block.groups[:131]for row in group)
                before+=sum(2*len(row['uv'])for row in rows[:2])
                start=cursor+int(block.bundle.motion_rows)+before
                if remap is None:
                    if start!=int(GLOBAL_ROWS[0]):raise ValueError('Original global source row slots changed')
                elif len(active_rows) and not np.array_equal(active_rows,[start,start+1]):
                    raise ValueError('Original print source row compact mapping disagrees with complete group order')
                found.append((cursor,cursor+len(block.quadratic)))
            cursor+=len(block.quadratic)
        if len(found)!=1:raise ValueError('One complete original secondary block is required')
        self.block_start,self.block_stop=found[0]
        self.active_global_rows=np.asarray(active_rows,np.int64)
        if np.any(self.active_global_rows<self.block_start)or np.any(self.active_global_rows>=self.block_stop):
            raise ValueError('Retired print compact scalar slots lie outside the secondary block')
        self.contract.update(programme_block_start=self.block_start,programme_block_stop=self.block_stop,
            source_slot_location_derived_from_complete_original_group_order=True)
        if remap is not None:
            self.contract.update(original_global_scalar_slots=GLOBAL_ROWS.tolist(),
                active_runtime_global_scalar_slots=self.active_global_rows.tolist(),
                active_runtime_retired_scalar_count=len(self.active_global_rows),
                original_motion_compaction_row_mapping_applied=True,
                no_active_runtime_rows_if_explicitly_compacted=not len(self.active_global_rows))

    def history_review(self):
        return dict(contract=deepcopy(self.contract),rows=[dict(
            **deepcopy(self.contract['source_row_locator']),original_collection_role='FIT',
            active_source_role='unfitted_rail_identity_pending',evaluation_status='retired_false_print_corner',
            native_source_UV=_plain(self.original['uv']),original_sigma_px=_plain(self.original['sigma']),
            original_source_record=_plain(self.original),global_scalar_slots=GLOBAL_ROWS.tolist(),
            included_in_active_print_FIT_accuracy_statistics=False,
            inactive_zero_residual_is_pixel_alignment=False,accuracy_metric=None)],accuracy_accepted=False)

    def annotate_original_review(self,report):
        report=dict(report)
        report['original_source_role_retirement']=self.history_review()
        report['original225_HELD_inventory_and_metrics_unchanged']=True
        return report

    def block(self,block,jacobian):
        ledger=block.row_ledger
        if(ledger.get('family')!='programme:secondary'or ledger.get('original_start')!=self.block_start
                or ledger.get('original_stop')!=self.block_stop):
            raise ValueError('Retirement must consume the exact original secondary source block')
        local=self.active_global_rows-self.block_start
        r=np.asarray(block.r)
        if len(r)!=self.block_stop-self.block_start:
            raise ValueError('Retired original secondary residual layout changed')
        if len(local):r=r.copy();r[local]=0.
        matrix=block.J if jacobian and not len(local)else None
        if jacobian:
            if block.J is None or block.J.shape[0]!=len(r):raise ValueError('Retired print Jacobian rows changed')
            if len(local):
                matrix=block.J.tocsr(copy=True)
                for row in local:matrix.data[matrix.indptr[row]:matrix.indptr[row+1]]=0.
        inactive=(GLOBAL_ROWS if 'original_motion_compaction_row_mapping_applied'not in self.contract else self.active_global_rows)
        revised=dict(ledger,original_source_role_retirement=deepcopy(self.contract),
            inactive_scalar_slots_excluded_from_source_accuracy_statistics=inactive.tolist(),
            # The ledger is source identity, not q-dependent numerical output.
            # Trial costs must carry the same inventory as the linearization.
            # Original UV/sigma/roles remain in history_review(), with no
            # accuracy metric attached to these explicitly retired rows.
            historical_false_role_residual_is_not_print_accuracy=True,
            original_retired_source_history_preserved_separately=True)
        return SourceRowBlock(r,matrix,block.quadratic,block.scales,revised,getattr(block,'loss_weights',None))

class SourceWithOriginalRoleRetirement:
    def __init__(self,base,state,packet_path):
        self.base=base;self.original_role_retirement=OriginalSourceRoleRetirement(packet_path)
        self.original_role_retirement.qualify_state(state)
        self.count,self.updater=base.count,base.updater
        self.quadratic,self.scales=base.quadratic,base.scales
        self.contract=dict(schema=SCHEMA+'.complete-source',original_source_contract=deepcopy(base.contract),
            original_source_role_retirement=deepcopy(self.original_role_retirement.contract),total_rows=len(self.scales),
            complete_parameter_source_slot_mask_scale_profile_and_geometry_layout_preserved=True,accuracy_accepted=False)

    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)

    def blocks(self,q,jacobian=True):
        found=0
        for block in self.base.blocks(q,jacobian):
            if block.row_ledger.get('family')=='programme:secondary':
                block=self.original_role_retirement.block(block,jacobian);found+=1
            yield block
        if found!=1:raise ValueError('Original secondary source block omitted or duplicated')

def wrap_original_source_role_retirement(base,state,packet_path):
    return SourceWithOriginalRoleRetirement(base,state,packet_path)
