"""Explicit original ICE curve/exposure budgets on the current source stream.

Stored pixels, sigmas, R/J and hard supports remain unchanged. Sampling
integration is outside Huber; only an original producer's explicit sqrt(N)
working normalization is removed. Missing visible paths use isolated anchors,
never an invented arc or sampling window.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from rinkspace.source_sampling import compile_curve_sampling
from .source_loss import SourceLossWeights
from .original_board_curve_integration import (file_reference,isolated_anchor_masses,
    compose_disjoint_loss_weights,_verified_observation_subset,_plain,_reference_identity)

SCHEMA='rinkspace.original-ice-curve-source-integration.v1'
SCAN_DECLARATIONS=(
    '2px outside explicit blur intervals, physical group scan normalization sqrt(N);',
    '2px engineering local source read sigma outside explicit blur interval; sqrt(group scan count) correlation normalization,')

def physical_ice_curve(geometry,row):
    """Use the existing WORLD registrar, including each original ring layer."""
    _,binding=geometry._binding(row)
    identity=_plain(binding.get('contract',{}).get('physical_identity'))
    if (binding.get('kind') not in ('circle','line','carrier','lane','crease')
            or not isinstance(identity,list) or len(identity)<2
            or identity[0] not in ('circle','fixed_circle','line','carrier','lane','crease')
            or row.get('point_correspondence') is True):
        raise ValueError('An authenticated original ice physical curve, not a material corner, is required')
    return identity

def original_working_sigma(row,stored,original_group_scan_count):
    """The original full group count certifies a declared factor, not active N."""
    text=str(row.get('sigma_modelling_decision',''))
    if text.startswith(SCAN_DECLARATIONS):
        if type(original_group_scan_count)is not int or original_group_scan_count<=0:
            raise ValueError('Declared original complete scan group count required')
        prior=np.full(len(stored),np.sqrt(original_group_scan_count))
        working=np.asarray(stored,float)/prior
        if not np.allclose(working,2.,rtol=1e-12,atol=1e-12):
            raise ValueError('Original declared sigma2 times sqrt(group scan count) changed')
        return working,prior,'original_explicit_sigma2_times_sqrt_complete_physical_scan_group'
    if 'sqrt(' in text or 'sqrtN' in text:
        raise ValueError('Unrecognized original ICE sampling normalization declaration')
    return np.asarray(stored,float).copy(),np.ones(len(stored)),'original_pointwise_working_sigma_no_sampling_factor_declared'

class InstalledIcePixels:
    """Reuse the already verified correction installation, with exact lineage."""
    def __init__(self,state,parent,video):
        self.state=state;self.parent=_reference_identity(parent);self.video=video
        self.packet=None;self.reference=None
    def subset(self,row,entry,index):
        old=entry['source_row'];metadata=row.get('active_source_pixel_correction')
        if metadata is None:
            keep,stored=_verified_observation_subset(row,old,np.asarray(old['uv'],float))
            self._operator(row,old,keep,None)
            return keep,stored
        if self.packet is None:
            contract=getattr(self.state,'original_ice_pixel_correction',None)
            if (not isinstance(contract,dict) or contract.get('schema')!='rinkspace.original-native-ice-pixel-correction.v1'
                    or _reference_identity(contract.get('original_absolute_packet',{}))!=self.parent
                    or contract.get('source_video_sha256')!=self.video
                    or contract.get('original_world_sigma_roles_source_order_masks_and_scalar_slots_unchanged')is not True):
                raise ValueError('Original ICE integration requires the installed exact pixel correction contract')
            self.reference=dict(path=str(Path(contract['packet_path']).resolve()),sha256=contract['packet_sha256'])
            if file_reference(self.reference['path'])!=self.reference:raise ValueError('Installed original ICE correction packet changed')
            self.packet=json.loads(Path(self.reference['path']).read_text(encoding='utf8'));self.contract=contract
        if (dict(path=str(Path(metadata['packet_path']).resolve()),sha256=metadata['packet_sha256'])!=self.reference
                or metadata.get('original_packet_row_index')!=index):
            raise ValueError('Active original ICE correction identity changed')
        records=[r for r in self.packet['records'] if r['original_packet_row_index']==index]
        histories=[r for r in self.contract['corrected_rows'] if r['original_packet_row_index']==index]
        locator={k:entry[k] for k in ('block','collection_role','group_actual_pts','group_order','row_order')}
        if len(records)!=1 or len(histories)!=1:raise ValueError('One installed original ICE correction per row required')
        record=records[0];history=histories[0]
        if (history['locator']!=locator or _plain(history['original_source_row'])!=_plain(old)
                or history['point_indices']!=record['point_indices'] or history['corrected_uv']!=record['corrected_uv']
                or metadata['point_indices']!=record['point_indices']):
            raise ValueError('Installed original ICE correction history changed')
        from structured_programme_source_candidate.original_ice_pixel_correction import _verify_record_original
        _verify_record_original(record,entry,old,record['point_indices'])
        full=np.asarray(old['uv'],float).copy();full[record['point_indices']]=np.asarray(record['corrected_uv'],float)
        keep,stored=_verified_observation_subset(row,old,full)
        self._operator(row,old,keep,history)
        return keep,stored
    @staticmethod
    def _operator(row,old,keep,history):
        for key in ('world','upper_world','id','pts','split','origin','point_correspondence',
                    'curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side'):
            if _plain(row.get(key))!=_plain(old.get(key)):
                raise ValueError('Original ICE operator, finite WORLD or source role changed: '+key)
        for key in ('normal','normal_interval','normal_interval_px','point_support_bounds_xy_px'):
            expected=old.get(key)
            if history is not None and key in ('normal_interval','normal_interval_px'):
                if history.get('original_normal_interval_removed_from_operator'):expected=None
                elif key=='normal_interval' and history.get('optical_support_updated'):
                    expected=np.asarray(expected,float).copy()
                    expected[history['point_indices']]=history['corrected_normal_interval_px']
            if expected is not None:
                expected=np.asarray(expected)
                if expected.ndim and expected.shape[0]==len(old['uv']):expected=expected[keep]
            if _plain(row.get(key))!=_plain(expected):
                raise ValueError('Original ICE normal or support differs from its installed source revision: '+key)

def _existing_payloads(base,state):
    owner=base
    while owner is not None:
        attrs=vars(owner)
        if 'original_board_curve_integration' in attrs:
            return {family:(payload,owner._new_slots[family]|owner._old_slots[family])
                    for family,payload in owner._payloads.items()}
        owner=attrs.get('base',attrs.get('parent'))
    result={}
    for family,payload in getattr(state,'original_absolute_input_compaction_loss_weights',{}).items():
        from .source_loss import loss_weight_arrays
        w,c=loss_weight_arrays(payload,len(payload.integration_weights))
        result[family]=(payload,(w!=1.)|((c!=1.) if c is not None else False))
    return result

class SourceWithOriginalIceCurveIntegration:
    def __init__(self,base,state,policy_path):
        began=time.perf_counter();self.base=base
        path=Path(policy_path).resolve();policy=json.loads(path.read_text(encoding='utf8'))
        if (policy.get('schema')!=SCHEMA or policy.get('curve_budget')!=1.
                or policy.get('enable_point_compression')is not False
                or policy.get('support_basis')!='isolated_authenticated_source_anchors'):
            raise ValueError('Explicit original ICE budget1 and authenticated anchor support policy required')
        video=policy['source_video_sha256'];parent=_reference_identity(policy['original_packet'])
        if len(video)!=64 or any(c not in '0123456789abcdef' for c in video):raise ValueError('Exact source video SHA required')
        if file_reference(parent['path'])!=parent:raise ValueError('Original ICE source packet changed')
        entries=pickle.loads(Path(parent['path']).read_bytes())['programme_absolute_rows']
        selected_indices=policy['original_packet_row_indices']
        if (not isinstance(selected_indices,list) or not selected_indices or any(type(i)is not int or not 0<=i<len(entries) for i in selected_indices)
                or len(set(selected_indices))!=len(selected_indices)):
            raise ValueError('Exact original ICE FIT row indices required')
        wanted=set(selected_indices);lookup={};scan_counts=defaultdict(int);bindings={}
        for i in selected_indices:
            entry=entries[i];row=entry['source_row']
            if entry['collection_role']!='FIT' or row.get('split') not in ('fit','development'):
                raise ValueError('Original ICE integration may not admit HELD sources')
            identity=physical_ice_curve(state.geometry,row);bindings[i]=identity
            key=(entry['block'],int(entry['group_order']),int(entry['row_order']))
            if key in lookup:raise ValueError('Duplicate original ICE source locator')
            lookup[key]=(i,entry)
            if str(row.get('sigma_modelling_decision','')).startswith(SCAN_DECLARATIONS):
                scan_counts[(int(entry['group_actual_pts']),json.dumps(identity),str(row.get('origin')),
                             str(row['sigma_modelling_decision']))]+=len(row['uv'])
        previous=_existing_payloads(base,state);pixels=InstalledIcePixels(state,parent,video)
        self.count=base.count;self.scales=base.scales;self.quadratic=base.quadratic
        self._payloads={};self._previous={};self._rows={};records=[];families=[];summary=[];seen=set()
        for block in state.blocks:
            family='programme:'+block.name;nrows=len(block.quadratic);offset=int(block.bundle.motion_rows)
            rows=[];selected=np.zeros(nrows,bool)
            for gi,(actual,source_rows) in enumerate(block.groups):
                for ri,row in enumerate(source_rows):
                    count=len(row['uv']);slots=np.arange(offset,offset+2*count);offset+=2*count
                    found=lookup.get((block.name,gi,ri))
                    if found is None:continue
                    i,entry=found;seen.add(i);old=entry['source_row']
                    if int(actual)!=int(entry['group_actual_pts']) or physical_ice_curve(state.geometry,row)!=bindings[i]:
                        raise ValueError('Original ICE exposure or registered physical identity changed')
                    keep,full_stored=pixels.subset(row,entry,i)
                    if not count:continue
                    identity=bindings[i];group_count=scan_counts.get((int(actual),json.dumps(identity),str(old.get('origin')),
                                                                   str(old.get('sigma_modelling_decision'))),0)
                    full_working,full_prior,basis=original_working_sigma(old,full_stored,group_count)
                    uv=np.asarray(row['uv'],float);name=f'{family}:{gi}:{ri}'
                    operator=json.dumps({k:row.get(k) for k in ('curve_observation_kind','scanline_fixed_axis','scanline_branch','extent_side')},sort_keys=True)
                    records.append(dict(id=name,split='fit',actual_pts=int(actual),source_video_sha256=video,
                        physical_curve_id=json.dumps(identity,separators=(',',':')),operator_id=operator,
                        uv=uv,stored_sigma_px=full_stored[keep],working_sigma_px=full_working[keep],
                        prior_sampling_multiplier=full_prior[keep],point_integration_mass=isolated_anchor_masses(uv),
                        sampling_kind=['measured']*count,source_anchor_identity_authority=parent))
                    selected[slots]=True;rows.append((name,slots))
                    summary.append(dict(original_packet_row_index=i,block=block.name,group_order=gi,row_order=ri,
                        actual_pts=int(actual),id=row['id'],physical_identity=identity,points=count,
                        original_points=len(old['uv']),original_point_indices=keep.tolist(),sigma_decomposition=basis,
                        original_declared_scan_group_count=group_count if group_count else None,
                        old_inverse_stored_sigma_squared_sum=float(np.sum(1/full_stored[keep]**2)),
                        current_active_UV_sha256=hashlib.sha256(uv.tobytes()).hexdigest()))
            if offset!=int(block.bundle.source_residual_rows):raise ValueError('Original ICE active source scalar boundary changed')
            if rows:families.append((family,nrows,rows,selected))
        if seen!=wanted:raise ValueError('Original ICE source stream omitted an authenticated FIT row')
        duplicate=dict(kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',source_anchor_identity_authority=parent)
        plan=compile_curve_sampling(records,duplicate_anchor_policy=duplicate)
        for family,nrows,rows,selected in families:
            w=np.ones(nrows);c=np.ones(nrows)
            for name,slots in rows:
                w[slots]=np.repeat(plan['rows'][name]['point_integration_weights'],2)
                c[slots]=np.repeat(plan['rows'][name]['working_normalization_multiplier'],2)
            old,owned=previous.get(family,(None,np.zeros(nrows,bool)))
            payload=SourceLossWeights(w,c if np.any(c!=1.) else None)
            self._payloads[family]=compose_disjoint_loss_weights(old,payload,existing_slots=owned,additional_slots=selected,rows=nrows)
            self._previous[family]=old;self._rows[family]=nrows
        self.original_ice_curve_integration=dict(schema=SCHEMA,policy_reference=file_reference(path),original_packet=parent,
            source_video_sha256=video,source_rows=summary,groups=plan['groups'],physical_curve_budget=1.,
            support_basis=policy['support_basis'],authenticated_source_correction_packet_reference=pixels.reference,
            duplicate_anchor_policy=duplicate,all_current_active_UV_R_J_scales_roles_truth_and_scalar_slots_preserved=True,
            ring_layers_and_lane_edges_and_actual_exposures_not_merged=True,existing_board_and_near49_payload_preserved=True,
            stored_sigma_preserved_and_declared_sampling_normalization_separated=True,
            sampling_count_removed_only_from_explicit_original_producer_declaration=True,
            no_inferred_visible_arcs_or_point_pruning=True,fresh_source_J_and_profiles_required=True,
            loss='weight * original Huber2(native pixel residual / explicitly original working sigma)',
            statistical_covariance_or_measured_precision_claim=False,accuracy_accepted=False,
            implementation_sources=[file_reference(__file__),file_reference(Path(__file__).with_name('original_board_curve_integration.py')),
                                    file_reference(Path(__file__).with_name('source_loss.py'))])
        import rinkspace.source_sampling as sampling
        self.original_ice_curve_integration['implementation_sources'].append(file_reference(sampling.__file__))
        self.original_ice_curve_integration_preparation_seconds=time.perf_counter()-began
        self.contract=dict(base.contract,original_ice_curve_integration=self.original_ice_curve_integration)
    def __getattr__(self,name):return getattr(self.base,name)
    def apply_block(self,block):
        family=block.row_ledger.get('family')
        if family not in self._payloads:return block
        if len(block.r)!=self._rows[family]:raise ValueError('Original ICE source scalar family changed')
        if block.row_ledger.get('source_sha256',self.original_ice_curve_integration['source_video_sha256'])!=self.original_ice_curve_integration['source_video_sha256']:
            raise ValueError('Original ICE source stream belongs to a different video')
        if 'original_ice_curve_integration' in block.row_ledger:raise ValueError('Original ICE curve budget already applied')
        expected=self._previous[family];actual=block.loss_weights
        if ((expected is None and actual is not None) or (expected is not None and
                (not isinstance(actual,SourceLossWeights) or actual.identity_digest!=expected.identity_digest))):
            raise ValueError('Original ICE integration existing disjoint loss payload changed')
        from z190_streaming_source_normal import SourceRowBlock
        return SourceRowBlock(block.r,block.J,block.quadratic,block.scales,
            dict(block.row_ledger,original_ice_curve_integration=dict(schema=SCHEMA,
                policy_reference=self.original_ice_curve_integration['policy_reference'])),self._payloads[family])
    def blocks(self,q,jacobian=True):
        found=set()
        for block in self.base.blocks(q,jacobian):
            family=block.row_ledger.get('family')
            if family in self._payloads:
                if family in found:raise ValueError('Original ICE programme family duplicated')
                found.add(family)
            yield self.apply_block(block)
        if found!=set(self._payloads):raise ValueError('Original ICE source family omitted')

def wrap_original_ice_curve_integration(source,state,policy_path=None):
    if policy_path is None:return source
    if hasattr(source,'original_ice_curve_integration'):raise ValueError('Original ICE curve integration already installed')
    return SourceWithOriginalIceCurveIntegration(source,state,policy_path)
