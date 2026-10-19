"""Retire source-proven broadcast-graphic motion roles without deleting history.

Only the reviewed motion residual/J scalar slots become structural zeros.
Original LK UV, sigma, masks, profiles, finite materials, source order and all
physical geometry stay intact. Zero retired rows are never alignment evidence.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np
from z190_streaming_source_normal import SourceRowBlock

PACKET_SCHEMA='rinkspace.source-proven-native-motion-role-retirement.v1'
SCHEMA='rinkspace.static-motion-source-role-retirement.v1'
SOURCE_SHA256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'
ROLE='screen_fixed_broadcast_graphic_overlay'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(ref,label):
    if not isinstance(ref,dict):raise ValueError(label+' source reference required')
    path=Path(ref.get('path','')).resolve();expected=ref.get('sha256')
    if not path.is_file() or not isinstance(expected,str) or len(expected)!=64 or _sha(path)!=expected:
        raise ValueError(label+' source bytes changed')
    return path


def _integer(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
        raise ValueError(label+' must be a nonnegative integer source index')
    return int(value)


def _plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))



DYNAMIC_VALIDITY_SCHEMA='rinkspace.source-reviewed-moving-human-motion-validity-append.v1'
COMPOUND_VALIDITY_SCHEMA='rinkspace.original-UI-retirement-plus-source-reviewed-dynamic-validity.v1'

def _read_dynamic_validity_append(reference):
    path=_reference(reference,'source-reviewed moving-human validity append')
    packet=json.loads(path.read_text(encoding='utf8'))
    if(packet.get('schema')!=DYNAMIC_VALIDITY_SCHEMA or packet.get('status')!='explicit_source_material_validity_revision_authorized'
            or packet.get('source_video_sha256')!=SOURCE_SHA256 or packet.get('time_base')!='1/15360'
            or not isinstance(packet.get('authority'),str)or not packet['authority'].strip()
            or packet.get('new_scalar_rows')!=0 or packet.get('source_scalar_slots_removed')!=0):
        raise ValueError('Explicit source-only moving-human validity authority required')
    paths={k:_reference(packet[k],k)for k in('original_raw_motion','original_selected_rows','original_source_ledger','original_clock_alias_authority','original_compaction_map','source_only_material_review')}
    records=packet.get('records');seen=set()
    if not isinstance(records,list)or not records or len(records)!=packet.get('retired_dynamic_point_count'):
        raise ValueError('Literal source-reviewed moving-human records required')
    with np.load(paths['original_raw_motion'],allow_pickle=False)as raw,np.load(paths['original_selected_rows'],allow_pickle=False)as selected,np.load(paths['original_source_ledger'],allow_pickle=False)as ledger,np.load(paths['original_compaction_map'],allow_pickle=False)as compact,np.load(paths['original_clock_alias_authority'],allow_pickle=False)as clock:
        aliases=dict(zip(map(int,clock['pts']),map(int,clock['exposure_representative_pts'])))
        active=ledger['active_original_global_index'];ids=selected['original_global_row_indices'];mapping=compact['old_point_to_compact_point']
        for row in records:
            if(row.get('source_role')!='moving_human_foreground' or row.get('split')!='fit'
                    or row.get('both_actual_RGB_endpoints_manually_reviewed')is not True
                    or row.get('either_endpoint_source_visible_moving_human')is not True
                    or row.get('source_qualified_for_static_scene_motion')is not False
                    or row.get('source_camera_or_projection_or_residual_used_for_validity')is not False
                    or row.get('original_UV_sigma_weights_clock_split_preserved')is not True):
                raise ValueError('Only actually reviewed human foreground may leave static motion')
            pair=_integer(row['original_raw_pair_index'],'raw pair');g=_integer(row['original_raw_cut5_point_index'],'raw source row');s=_integer(row['original_selected_cut5_row'],'selected source row');full=_integer(row['original_full_low_bundle_row'],'historical bundle row');prefix=_integer(row['current_original_prefix_bundle_index'],'original prefix row');current=_integer(row['active_compacted_motion_point_index'],'current compact row')
            if prefix in seen:raise ValueError('Repeated moving-human validity row')
            seen.add(prefix)
            if not(s<len(ids)and prefix<len(active)and prefix<len(mapping)and full<len(ledger['static_factor_active'])and pair<len(raw['pair_pts'])):raise ValueError('Moving-human source index outside original inventory')
            if(int(ids[s])!=g or int(active[prefix])!=full or int(mapping[prefix])!=current
                    or not bool(ledger['static_factor_active'][full])or int(raw['offsets'][pair])+int(row['original_raw_pair_local_index'])!=g
                    or not int(raw['offsets'][pair])<=g<int(raw['offsets'][pair+1])):raise ValueError('Moving-human raw/selected/active source identity changed')
            if not np.array_equal(raw['pair_pts'][pair],row['source_native_pair_pts']):raise ValueError('Original source endpoints changed')
            for key,expected in(('source_uv',raw['xy0'][g]),('target_uv',raw['xy1'][g]),('source_sigma_px',ledger['original_sigma_xy'][full])):
                if not np.array_equal(row[key],expected):raise ValueError('Moving-human original UV or sigma changed')
            if not np.array_equal(ledger['source_uv_a'][full],row['source_uv'])or not np.array_equal(ledger['source_uv_b'][full],row['target_uv']):raise ValueError('Original foreground ledger source UV differs')
            if [aliases[int(p)]for p in row['source_native_pair_pts']]!=[row['source_pts'],row['target_pts']]:raise ValueError('Original source clock differs')
            if not np.array_equal(row['expected_original_finite_plane'],np.zeros(4)):raise ValueError('Finite geometry source may not retire as unknown human')
            refs=row['source_raw_references']
            if len(refs)!=2:raise ValueError('Both actual source RGB references required')
            for p,reference in zip(row['source_native_pair_pts'],refs):
                _reference(reference,'source-reviewed human RGB endpoint')
                if(reference.get('native_pts')!=int(p)or reference.get('source_video_sha256')!=SOURCE_SHA256 or reference.get('time_base')!='1/15360' or len(reference.get('decoded_BGR_sha256',''))!=64):raise ValueError('Human original RGB source identity differs')
    return dict(path=str(path),sha256=_sha(path),packet=packet,records=deepcopy(records))

def _qualify_dynamic_validity_after_original_compaction(retirement,state):
    append=retirement._dynamic_validity_append
    if append is None:return
    # Original overlay qualification/compaction runs first, without any new row removal.
    if getattr(state,'original_motion_compaction_retirement_prequalification',None)is None:return
    found=[];cursor=0;qualified=[]
    for block in state.blocks:
        stop=cursor+len(block.quadratic)
        if block.name=='low':
            bundle=block.bundle;n=len(bundle.a);mapping=np.asarray(getattr(bundle,'original_motion_compaction_old_to_active',None))
            with np.load(_reference(append['packet']['original_compaction_map'],'dynamic original compaction map'),allow_pickle=False)as original:
                if not np.array_equal(mapping,original['old_point_to_compact_point']):raise ValueError('Actual dynamic source compaction map changed')
            if(int(bundle.motion_rows)!=2*n or len(bundle.b)!=n or stop-cursor<2*n):raise ValueError('Actual low source motion layout changed')
            for row in append['records']:
                j=int(row['active_compacted_motion_point_index']);prefix=int(row['current_original_prefix_bundle_index'])
                if j>=n or int(mapping[prefix])!=j:raise ValueError('Moving-human source must still be genuinely active before filtering')
                expected=[int(state.original.aliases[p])for p in row['source_native_pair_pts']]
                if [int(bundle.stamps[bundle.ia[j]]),int(bundle.stamps[bundle.ib[j]])]!=expected:raise ValueError('Actual dynamic source exposure alias differs')
                for value,expected_value in((bundle.a[j],row['source_uv']),(bundle.b[j],row['target_uv']),(bundle.motion_sigma[2*j:2*j+2],row['source_sigma_px']),(bundle.finite_planes[j],row['expected_original_finite_plane'])):
                    if not np.array_equal(value,expected_value):raise ValueError('Actual dynamic source UV/sigma/plane differs')
                qualified.append(dict(source_record=deepcopy(row),programme_family='programme:low',local_scalar_slots=[2*j,2*j+1],global_scalar_slots=[cursor+2*j,cursor+2*j+1],included_in_active_motion_FIT_accuracy_statistics=False,inactive_zero_residual_is_pixel_alignment=False))
            found.append(dict(family='programme:low',original_start=cursor,original_stop=stop,motion_point_count=n))
        cursor=stop
    if len(found)!=1 or found[0]!=retirement.block_binding:raise ValueError('Exact complete original low programme family required')
    legacy=deepcopy(retirement.contract)
    newrows=np.asarray(sorted(s for r in qualified for s in r['global_scalar_slots']),np.int64)
    if set(newrows).intersection(retirement.global_rows.tolist()):raise ValueError('Human source already retired as another material')
    retirement.global_rows=np.unique(np.r_[retirement.global_rows,newrows]);retirement.local_rows=retirement.global_rows-retirement.block_binding['original_start']
    retirement.dynamic_qualified=qualified
    retirement.contract=dict(schema=SCHEMA+'.source-reviewed-dynamic-append',original_UI_retirement_contract=legacy,source_reviewed_dynamic_validity_append=dict(path=append['path'],sha256=append['sha256']),programme_block=deepcopy(retirement.block_binding),global_scalar_slots=retirement.global_rows.tolist(),active_original_motion_FIT_scalars_retired=len(retirement.global_rows),new_scalar_rows=0,source_scalar_slots_removed=0,source_UV_sigma_weights_clock_and_all_prior_retirement_exact=True,dynamic_validity_is_source_identity_not_robust_weighting=True,exclude_from_active_motion_FIT_accuracy_statistics=True,inactive_zero_residual_is_accuracy_measurement=False,accuracy_accepted=False)
    retirement._dynamic_validity_applied=True


class StaticMotionSourceRoleRetirement:
    def __init__(self,packet_path):
        self.packet_path=Path(packet_path).resolve()
        data=json.loads(self.packet_path.read_text(encoding='utf8'))
        self._dynamic_validity_append=None;self._dynamic_validity_applied=False
        if data.get('schema')==COMPOUND_VALIDITY_SCHEMA:
            if(data.get('source_video_sha256')!=SOURCE_SHA256 or data.get('time_base')!='1/15360' or data.get('original_UI_retirement_packet_and_compaction_exact')is not True):raise ValueError('Exact original UI retirement authority must be preserved')
            self._dynamic_validity_append=_read_dynamic_validity_append(data['source_reviewed_dynamic_validity_append'])
            self.packet_path=_reference(data['original_UI_retirement'],'original immutable UI retirement')
            data=json.loads(self.packet_path.read_text(encoding='utf8'))
        if(data.get('schema')!=PACKET_SCHEMA or data.get('source_video_sha256')!=SOURCE_SHA256
                or data.get('time_base')!='1/15360'
                or data.get('model_projection_used_for_selection') is not False
                or data.get('frozen_before_target_switch') is not True
                or data.get('source_visible_UI_evidence',{}).get('no_model_or_residual_was_used_to_classify_UI') is not True):
            raise ValueError('Frozen same-event source-only motion-role authority required')
        self.records=deepcopy(data.get('records',[]))
        if not self.records:raise ValueError('At least one source-qualified overlay record required')
        self.packet=deepcopy(data);self._qualified=False
        self.raw_motion_path=_reference(data['original_motion_npz'],'original motion')
        self.selected_path=_reference(data['original_selected_rows_npz'],'retained original cut5 row order')
        self.admission_path=_reference(data['original_foreground_admission_npz'],'original low foreground admission')
        self.clock_path=_reference(data['source_clock_authority'],'original source clock')
        for key in ('original_pair_source_mask_bindings','original_low_source_factory_manifest','existing_frozen_actual_source_provenance'):
            _reference(data[key],key)
        evidence=data['source_visible_UI_evidence']
        _reference(evidence['source_patch_page'],'source-only reviewed UI patches')
        sites={(tuple(site['xy0']),site['source_element_role'])for site in evidence['source_qualified_exact_site_catalog']}
        core=np.asarray(evidence['conservative_both_endpoint_core_xyxy'],float)
        if core.shape!=(4,) or not np.isfinite(core).all() or np.any(core[2:]<core[:2]):
            raise ValueError('Reviewed source UI endpoint support required')
        self.original_prefix_count=_integer(data['original_low_motion_prefix_point_count'],'original low motion prefix')
        self.cut3_retained_count=_integer(data['cut3_retained_motion_point_count'],'retained cut3 prefix')
        clock={}
        for line in self.clock_path.open(encoding='utf8'):
            row=json.loads(line);clock[int(row['full_pts'])]=row
        self.clock=clock;seen=set()
        with np.load(self.raw_motion_path,allow_pickle=False)as raw, np.load(self.selected_path,allow_pickle=False)as selected, np.load(self.admission_path,allow_pickle=False)as admission:
            ids=selected['original_global_row_indices'];active=admission['active_original_global_index']
            if len(active)!=self.original_prefix_count:raise ValueError('Original low motion admission prefix changed')
            for record in self.records:
                if(record.get('block')!='low' or record.get('family')!='programme:low'
                        or record.get('source_role')!=ROLE
                        or record.get('both_end_source_only_manually_reviewed') is not True
                        or record.get('source_qualified_for_static_scene_motion') is not False
                        or record.get('source_geometry_or_camera_used_for_selection') is not False
                        or record.get('model_projection_used_for_selection') is not False):
                    raise ValueError('Only source-reviewed screen-fixed broadcast graphics may retire')
                j=_integer(record['current_bundle_motion_row'],'current bundle motion row')
                pair=_integer(record['original_cut5_pair_index'],'original cut5 pair')
                g=_integer(record['original_raw_global_row'],'original raw motion row')
                local=_integer(record['original_pair_local_row'],'original pair local motion row')
                selected_row=_integer(record['selected_cut5_row'],'retained cut5 source row')
                original=_integer(record['original_full_low_bundle_row'],'original full low motion row')
                if j in seen:raise ValueError('Duplicate retired source motion row')
                seen.add(j)
                if not(j<len(active) and selected_row<len(ids) and pair<len(raw['pair_pts'])):
                    raise ValueError('Retired source row is outside the original source inventory')
                if int(ids[selected_row])!=g or original!=self.cut3_retained_count+selected_row or int(active[j])!=original:
                    raise ValueError('Original raw/retained/full-low/current motion row mapping changed')
                if not int(raw['offsets'][pair])<=g<int(raw['offsets'][pair+1]) or g!=int(raw['offsets'][pair])+local:
                    raise ValueError('Original motion row lies outside its certified source pair')
                if np.asarray(record['native_pair_pts']).shape!=(2,) or not np.array_equal(raw['pair_pts'][pair],record['native_pair_pts']):
                    raise ValueError('Original native motion endpoint time changed')
                for key,expected in(('xy0',raw['xy0'][g]),('xy1',raw['xy1'][g])):
                    value=np.asarray(record[key],float)
                    if value.shape!=(2,) or not np.isfinite(value).all() or not np.array_equal(value,expected):
                        raise ValueError('Original motion source UV changed: '+key)
                    if np.any(value<core[:2]) or np.any(value>core[2:]):
                        raise ValueError('Original source endpoint is outside the reviewed UI support')
                if (tuple(record['xy0']),record['source_element_role'])not in sites:
                    raise ValueError('Original LK site lacks exact source-only UI identity')
                sigma=np.asarray(record['original_sigma_xy_px'],float)
                if sigma.shape!=(2,) or not np.isfinite(sigma).all() or np.any(sigma<=0):
                    raise ValueError('Original per-component motion sigma required')
                for key,value in(('source_uv_a',record['xy0']),('source_uv_b',record['xy1']),('original_sigma_xy',sigma),
                                  ('source_endpoint_index_a',record['original_full_low_endpoint_index_a']),
                                  ('source_endpoint_index_b',record['original_full_low_endpoint_index_b'])):
                    if not np.array_equal(admission[key][original],value):
                        raise ValueError('Original foreground-admission source identity changed: '+key)
                if not bool(admission['static_factor_active'][original]):
                    raise ValueError('A historical inactive source row cannot retire again')
                if record['proposed_local_source_scalar_slots']!=[2*j,2*j+1]:
                    raise ValueError('Original proposed local motion scalar slots changed')
                if not np.array_equal(record['finite_plane_expected'],np.zeros(4)):
                    raise ValueError('A physical finite-plane source cannot be retired as an untyped overlay')
                if np.asarray(record['actual_pair_pts']).shape!=(2,):raise ValueError('Two original exposure aliases required')
                refs=record['source_raw_references']
                if len(refs)!=2:raise ValueError('Both original native endpoint images required')
                for native,actual,ref in zip(record['native_pair_pts'],record['actual_pair_pts'],refs):
                    native=int(native);actual=int(actual);raw_native=int(ref['native_pts'])
                    if(raw_native!=actual or int(ref['actual_pts'])!=actual
                            or int(ref['original_correspondence_native_pts'])!=native
                            or int(ref['source_raw_requested_native_pts'])!=native):
                        raise ValueError('Endpoint raw native/representative source binding changed')
                    _reference(ref,'endpoint native raw')
                    row=clock.get(native);raw_clock=clock.get(raw_native)
                    if(row is None or raw_clock is None or row.get('match_status')!='unique_pixel_match'
                            or raw_clock.get('match_status')!='unique_pixel_match'
                            or row.get('accepted_official_pts')!=raw_clock.get('accepted_official_pts')
                            or ref.get('official_pts')!=raw_clock.get('accepted_official_pts')):
                        raise ValueError('Retired source endpoint lacks the original unique clock authority')
                if(record.get('official_pair_pts') is not None
                        and [clock[int(p)]['accepted_official_pts']for p in record['native_pair_pts']]!=record['official_pair_pts']):
                    raise ValueError('Retired native endpoints changed their official exposure identity')
        self.contract=dict(schema=SCHEMA,packet_path=str(self.packet_path),packet_sha256=_sha(self.packet_path),
            source_video_sha256=SOURCE_SHA256,time_base='1/15360',source_role=ROLE,
            original_motion_npz=deepcopy(data['original_motion_npz']),original_selected_rows_npz=deepcopy(data['original_selected_rows_npz']),
            original_foreground_admission_npz=deepcopy(data['original_foreground_admission_npz']),source_clock_authority=deepcopy(data['source_clock_authority']),
            original_low_motion_prefix_point_count=self.original_prefix_count,retired_motion_point_count=len(self.records),
            source_scalar_slots_removed=0,new_source_scalar_slots_added=0,
            original_UV_sigma_motion_order_finite_planes_masks_scales_and_profile_buffers_preserved=True,
            source_only_overlay_identity_is_retirement_authority=True,unknown_reflection_rows_not_retired=True,
            inactive_zero_residual_is_accuracy_measurement=False,exclude_from_active_motion_FIT_accuracy_statistics=True,
            all_physical_world_entities_and_render_paths_preserved=True,accuracy_accepted=False)

    def qualify_state(self,state):
        """Derive global double-scalar slots from the live original block order."""
        prequalified=getattr(state,'original_motion_compaction_retirement_prequalification',None)
        if prequalified is not None:
            self._qualify_compacted_state(state,prequalified)
            _qualify_dynamic_validity_after_original_compaction(self,state)
            return
        cursor=0;found=[];qualified=[];aliases=state.original.aliases
        for block in state.blocks:
            stop=cursor+len(block.quadratic)
            if block.name=='low':
                bundle=block.bundle;n=len(bundle.a)
                if(len(bundle.b)!=n or len(bundle.ia)!=n or len(bundle.ib)!=n or n<self.original_prefix_count
                        or int(bundle.motion_rows)!=2*n or np.asarray(bundle.motion_sigma).shape!=(2*n,)
                        or np.asarray(bundle.finite_planes).shape!=(n,4) or stop-cursor<2*n):
                    raise ValueError('Original low bundle motion-prefix layout changed')
                for record in self.records:
                    j=int(record['current_bundle_motion_row']);native=list(map(int,record['native_pair_pts']))
                    actual=[int(aliases[p])for p in native]
                    if actual!=list(map(int,record['actual_pair_pts'])):
                        raise ValueError('Retired motion original actual exposure alias changed')
                    runtime=[int(bundle.stamps[int(bundle.ia[j])]),int(bundle.stamps[int(bundle.ib[j])])]
                    if runtime!=actual:raise ValueError('Retired motion bundle endpoints changed')
                    for key,value in(('xy0',bundle.a[j]),('xy1',bundle.b[j]),('original_sigma_xy_px',bundle.motion_sigma[2*j:2*j+2]),('finite_plane_expected',bundle.finite_planes[j])):
                        if not np.array_equal(value,record[key]):raise ValueError('Retired original bundle field changed: '+key)
                    for p,a in zip(native,actual):
                        target=self.clock.get(a)
                        if target is None or target.get('accepted_official_pts')!=self.clock[p].get('accepted_official_pts'):
                            raise ValueError('Retired actual alias does not match the original official source clock')
                    slots=[cursor+2*j,cursor+2*j+1]
                    if 'global_scalar_slots' in record and slots!=record['global_scalar_slots']:
                        raise ValueError('Frozen expected motion scalar slots disagree with original block order')
                    qualified.append(dict(source_record=deepcopy(record),programme_family='programme:low',
                        local_scalar_slots=[2*j,2*j+1],global_scalar_slots=slots))
                found.append(dict(family='programme:low',original_start=cursor,original_stop=stop,motion_point_count=n))
            cursor=stop
        if len(found)!=1:raise ValueError('One complete original low programme block required')
        self.block_binding=found[0];self.qualified=qualified
        self.global_rows=np.asarray(sorted(s for r in qualified for s in r['global_scalar_slots']),np.int64)
        self.local_rows=self.global_rows-self.block_binding['original_start']
        self.contract.update(programme_block=deepcopy(self.block_binding),global_scalar_slots=self.global_rows.tolist(),
            active_original_motion_FIT_scalars_retired=len(self.global_rows),
            source_slot_location_derived_from_current_original_bundle_order=True)
        self._qualified=True

    def _qualify_compacted_state(self,state,prequalified):
        """Retain source-qualified original history after its active rows are removed."""
        if(not isinstance(prequalified,dict)
                or prequalified.get('schema')!='rinkspace.actual-original-retirement-qualified-before-compaction.v1'
                or prequalified.get('original_qualification_passed')is not True
                or Path(prequalified.get('packet_path','')).resolve()!=self.packet_path
                or prequalified.get('packet_sha256')!=self.contract['packet_sha256']):
            raise ValueError('Exact original source retirement qualification is required before compaction')
        qualified=prequalified.get('qualified_records')
        if(not isinstance(qualified,list)or len(qualified)!=len(self.records)
                or [_plain(r.get('source_record'))for r in qualified]!=[_plain(r)for r in self.records]):
            raise ValueError('Prequalified original retirement UV, sigma, clock or source record changed')
        ids=np.asarray(prequalified.get('original_point_indices'))
        expected=np.asarray([int(r['current_bundle_motion_row'])for r in self.records],np.int64)
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)or not np.array_equal(ids,expected):
            raise ValueError('Prequalified original retirement point order changed')
        old_binding=prequalified.get('original_block_binding')
        if(not isinstance(old_binding,dict)or old_binding.get('family')!='programme:low'
                or _integer(old_binding.get('motion_point_count'),'original low motion count')<self.original_prefix_count):
            raise ValueError('Original low retirement block binding required')
        old_start=_integer(old_binding.get('original_start'),'original low block start')
        old_stop=_integer(old_binding.get('original_stop'),'original low block stop')
        if old_stop-old_start<2*int(old_binding['motion_point_count']):
            raise ValueError('Original low retirement block is smaller than its motion prefix')
        aliases=state.original.aliases
        for record,row,j in zip(self.records,qualified,ids):
            local=[2*int(j),2*int(j)+1];slots=[old_start+x for x in local]
            if(row.get('programme_family')!='programme:low'or row.get('local_scalar_slots')!=local
                    or row.get('global_scalar_slots')!=slots
                    or ('global_scalar_slots'in record and record['global_scalar_slots']!=slots)):
                raise ValueError('Prequalified original retirement scalar location changed')
            native=list(map(int,record['native_pair_pts']));actual=[int(aliases[p])for p in native]
            if actual!=list(map(int,record['actual_pair_pts'])):
                raise ValueError('Compacted retirement original exposure alias changed')
            for p,a in zip(native,actual):
                target=self.clock.get(a)
                if target is None or target.get('accepted_official_pts')!=self.clock[p].get('accepted_official_pts'):
                    raise ValueError('Compacted retirement original official source clock changed')
        map_path=_reference(prequalified.get('candidate_map_reference'),'retirement compaction map')
        cursor=0;found=[]
        for block in state.blocks:
            stop=cursor+len(block.quadratic)
            if block.name=='low':
                bundle=block.bundle;n=len(bundle.a)
                binding=dict(family='programme:low',original_start=cursor,original_stop=stop,motion_point_count=n)
                if binding!=prequalified.get('compact_block_binding'):
                    raise ValueError('Compacted low retirement block location changed')
                if(len(bundle.b)!=n or len(bundle.ia)!=n or len(bundle.ib)!=n
                        or int(bundle.motion_rows)!=2*n or np.asarray(bundle.motion_sigma).shape!=(2*n,)
                        or np.asarray(bundle.finite_planes).shape!=(n,4)or stop-cursor<2*n):
                    raise ValueError('Compacted low retirement motion layout changed')
                mapping=np.asarray(getattr(bundle,'original_motion_compaction_old_to_active',None))
                if(mapping.ndim!=1 or not np.issubdtype(mapping.dtype,np.integer)
                        or len(mapping)!=int(old_binding['motion_point_count'])or np.any(ids>=len(mapping))):
                    raise ValueError('Actual original-to-active low point mapping required')
                with np.load(map_path,allow_pickle=False)as candidate:
                    if not np.array_equal(mapping,candidate['old_point_to_compact_point']):
                        raise ValueError('Actual low retirement mapping differs from its authenticated source map')
                mapped=np.asarray(prequalified.get('mapped_active_point_indices'))
                if(not np.array_equal(mapped,mapping[ids])or np.any(mapping[ids]!=-1)):
                    raise ValueError('Every prequalified source retirement point must be explicitly compacted')
                found.append(binding)
            cursor=stop
        if len(found)!=1:raise ValueError('One complete compacted low programme block required')
        self.block_binding=found[0];self.qualified=deepcopy(qualified)
        self.global_rows=np.empty(0,np.int64);self.local_rows=np.empty(0,np.int64)
        historical=sorted(s for row in qualified for s in row['global_scalar_slots'])
        self.contract.update(programme_block=deepcopy(self.block_binding),global_scalar_slots=[],
            active_original_motion_FIT_scalars_retired=0,historical_global_scalar_slots=historical,
            original_precompaction_programme_block=deepcopy(old_binding),
            original_source_qualification_before_compaction_preserved=True,
            source_slot_location_derived_from_current_original_bundle_order=True,
            all_prequalified_retirement_points_explicitly_compacted=True,
            runtime_retired_source_scalar_slots_compacted=len(historical),
            candidate_map_reference=deepcopy(prequalified['candidate_map_reference']),
            active_retirement_has_no_residual_or_J_rows=True)
        self._qualified=True

    def history_review(self):
        if not self._qualified:raise ValueError('Retired motion source has not qualified the complete state')
        if self._dynamic_validity_applied:
            legacy=object.__new__(type(self));legacy.__dict__=self.__dict__.copy();legacy._dynamic_validity_applied=False;legacy.contract=deepcopy(self.contract['original_UI_retirement_contract'])
            return dict(contract=deepcopy(self.contract),original_UI_history=legacy.history_review(),source_reviewed_dynamic_foreground_rows=deepcopy(self.dynamic_qualified),all_retired_rows_excluded_from_active_accuracy=True,inactive_zero_residual_is_pixel_alignment=False,accuracy_accepted=False)
        return dict(contract=deepcopy(self.contract),rows=[dict(**deepcopy(r),
            original_source_role='original_untyped_LK_motion_FIT',active_source_role='retired_screen_fixed_graphic_not_world_motion',
            included_in_active_motion_FIT_accuracy_statistics=False,inactive_zero_residual_is_pixel_alignment=False,
            accuracy_metric=None)for r in self.qualified],accuracy_accepted=False)

    def annotate_original_review(self,report):
        out=dict(report);out['static_motion_source_role_retirement']=self.history_review();return out

    def block(self,block,jacobian):
        if not self._qualified:raise ValueError('Retired motion source must qualify the complete state first')
        ledger=block.row_ledger
        if any(ledger.get(k)!=v for k,v in self.block_binding.items()if k in('family','original_start','original_stop')):
            raise ValueError('Motion retirement must consume the exact original programme family block')
        r=np.asarray(block.r)
        if len(r)!=self.block_binding['original_stop']-self.block_binding['original_start']:
            raise ValueError('Retired original programme residual layout changed')
        if len(self.local_rows):r=r.copy();r[self.local_rows]=0.
        matrix=block.J if jacobian and not len(self.local_rows)else None
        if jacobian:
            if block.J is None or block.J.shape[0]!=len(r):raise ValueError('Retired motion Jacobian rows changed')
            if len(self.local_rows):
                matrix=block.J.tocsr(copy=True)
                for row in self.local_rows:matrix.data[matrix.indptr[row]:matrix.indptr[row+1]]=0.
        inactive=sorted(set(ledger.get('inactive_scalar_slots_excluded_from_source_accuracy_statistics',[]))|set(self.global_rows.tolist()))
        revised=dict(ledger,static_motion_source_role_retirement=deepcopy(self.contract),
            inactive_scalar_slots_excluded_from_source_accuracy_statistics=inactive,
            original_retired_motion_source_history_preserved_separately=True,
            retired_graphic_structural_zero_is_not_world_motion_accuracy=True)
        return SourceRowBlock(r,matrix,block.quadratic,block.scales,revised,getattr(block,'loss_weights',None))


class SourceWithStaticMotionRoleRetirement:
    def __init__(self,base,state,packet_path):
        self.base=base;self.static_motion_role_retirement=StaticMotionSourceRoleRetirement(packet_path)
        self.static_motion_role_retirement.qualify_state(state)
        if self.static_motion_role_retirement._dynamic_validity_append is not None and not self.static_motion_role_retirement._dynamic_validity_applied:raise ValueError('Dynamic source validity requires the exact completed original compaction')
        self.count,self.updater=base.count,base.updater
        self.quadratic,self.scales=base.quadratic,base.scales
        self.contract=dict(schema=SCHEMA+'.complete-source',original_source_contract=deepcopy(base.contract),
            static_motion_source_role_retirement=deepcopy(self.static_motion_role_retirement.contract),
            total_rows=len(self.scales),complete_parameter_source_slot_mask_scale_profile_and_geometry_layout_preserved=True,
            accuracy_accepted=False)

    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)

    def blocks(self,q,jacobian=True):
        found=0;family=self.static_motion_role_retirement.block_binding['family']
        for block in self.base.blocks(q,jacobian):
            if block.row_ledger.get('family')==family:
                block=self.static_motion_role_retirement.block(block,jacobian);found+=1
            yield block
        if found!=1:raise ValueError('Retired original programme motion block omitted or duplicated')


def wrap_static_motion_source_role_retirement(base,state,packet_path):
    return SourceWithStaticMotionRoleRetirement(base,state,packet_path)
