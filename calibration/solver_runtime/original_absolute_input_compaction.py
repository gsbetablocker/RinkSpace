"""Physically compact source-certified absolute FIT inputs before actors start.

The original rows stay available for review and hard truth. Only source-only
rejected pixels and explicitly certified RGB-polyline representatives leave
the active native input/CSR arrays. No camera or model residual selects points.
"""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import numpy as np

from .source_point_qualification import file_sha256, original_uv_sha256, _direct_user_annotation
from .original_source_point_qualification import OriginalSourcePointQualification, _plain
from .source_loss import SourceLossWeights

SCHEMA = 'rinkspace.original-absolute-active-input-compaction.v1'


def _ref(reference):
    path = Path(reference['path']).resolve()
    if file_sha256(path) != reference['sha256']:
        raise ValueError('Absolute compaction source proof bytes changed')
    return path


def slice_observed_points(row, keep):
    """Keep the complete physical primitive; subset only observed pixel fields."""
    n = len(row['uv']); keep = np.asarray(keep, np.int64)
    if (keep.ndim != 1 or np.any(keep < 0) or np.any(keep >= n)
            or len(keep) > 1 and np.any(np.diff(keep) <= 0)):
        raise ValueError('Ordered original observed point indices required')
    result = dict(row)
    fields = ('uv', 'source_uv', 'sigma', 'sigma_px', 'normal', 'normal_interval',
        'point_support_bounds_xy_px', 'source_tangent_interval_px')
    for key in fields:
        if key not in row: continue
        value = np.asarray(row[key])
        if value.ndim == 0: continue
        if value.shape[0] != n:
            raise ValueError('Observed point field has another original layout: ' + key)
        selected = value[keep].copy()
        result[key] = selected if isinstance(row[key], np.ndarray) else selected.tolist()
    # WORLD is the complete finite curve, not an observation-point count.
    # Point-to-point absolute sources are not admitted by this curve policy.
    if row.get('point_correspondence') and len(keep) != n:
        raise ValueError('Point correspondence needs a separate world-index certificate')
    result['active_original_absolute_input_compaction']=dict(original_point_count=n,
        retained_original_point_indices=keep.tolist(),original_uv_float64_sha256=original_uv_sha256(row),
        retained_uv_float64_sha256=original_uv_sha256(result))
    return result


def original_points_to_active(row,points):
    """Resolve unchanged source point IDs through a verified active UV view."""
    points=np.asarray(points,np.int64);mapping=row.get('active_original_absolute_input_compaction')
    if mapping is None:
        if points.ndim!=1 or np.any(points<0)or np.any(points>=len(row['uv'])):
            raise ValueError('Original point IDs outside unchanged active UV layout')
        return points
    keep=np.asarray(mapping.get('retained_original_point_indices'),np.int64)
    n=mapping.get('original_point_count')
    if(type(n)is not int or points.ndim!=1 or np.any(points<0)or np.any(points>=n)
        or keep.shape!=(len(row['uv']),)or np.any(keep<0)or np.any(keep>=n)
        or len(keep)>1 and np.any(np.diff(keep)<=0)
        or original_uv_sha256(row)!=mapping.get('retained_uv_float64_sha256')):
        raise ValueError('Original-to-active absolute UV map or retained source bytes changed')
    positions=np.searchsorted(keep,points)
    if np.any(positions>=len(keep))or not np.array_equal(keep[positions],points):
        raise ValueError('A protected corrected source point was physically removed')
    return positions


def compose_selected_scalar_positions(previous, block_start, compact_start, scalar_keep, old_rows):
    """Map only explicitly needed retirement slots, never all programme rows."""
    result={}
    for key,value in previous.items():
        if value<0:result[key]=-1;continue
        if not block_start<=value<block_start+old_rows:continue
        local=int(value-block_start);position=int(np.searchsorted(scalar_keep,local))
        result[key]=compact_start+position if position<len(scalar_keep) and int(scalar_keep[position])==local else -1
    return result


def reindex_retirement_block_binding(state,reports):
    """Compose the active low block range; original source qualification stays."""
    prequalified=getattr(state,'original_motion_compaction_retirement_prequalification',None)
    if prequalified is None:return None
    rows=[r for r in reports if r['block']=='low']
    if len(rows)!=1:raise ValueError('One complete low source range required for retirement composition')
    row=rows[0]
    before=dict(family='programme:low',original_start=row['original_start'],
        original_stop=row['original_start']+row['original_rows'],motion_point_count=row['motion_point_count'])
    if prequalified.get('compact_block_binding')!=before:
        raise ValueError('Original motion compaction retirement binding differs before absolute input compaction')
    after=dict(family='programme:low',original_start=row['compact_start'],
        original_stop=row['compact_start']+row['compact_rows'],motion_point_count=row['motion_point_count'])
    prequalified['compact_block_binding']=after
    prequalified['absolute_compaction_index_only_binding']=dict(before=before,after=after,
        original_source_qualification_records_clock_and_motion_map_unchanged=True)
    return dict(before=before,after=after)


def compact_original_absolute_inputs(state, capold, template, evidence_path, *, qualification_path):
    if getattr(state, 'source_actors', None) is not None:
        raise ValueError('Absolute input compaction must precede source actors')
    path = Path(evidence_path).resolve(); evidence = json.loads(path.read_text(encoding='utf8'))
    if evidence.get('schema') != 'rinkspace.nearboard-source-arclength-mass-coarsening-evidence.v1':
        raise ValueError('Explicit original RGB-polyline compaction evidence required')
    parent = _ref(evidence['original_packet'])
    original = pickle.loads(parent.read_bytes())['programme_absolute_rows']
    _ref(evidence['original_source_material_npz']); _ref(evidence['source_shape_proposals'])
    qualification = OriginalSourcePointQualification(qualification_path)
    qualification.qualify_state(state)
    if qualification.contract['original_packet']['sha256'] != evidence['original_packet']['sha256']:
        raise ValueError('Source qualification and compaction have different original packets')
    exclusions = {tuple(entry['item'][k] for k in ('block','group_order','row_order')):
        entry['record']['excluded_point_indices'] for entry in qualification.records}
    policies = {}; ids = set()
    for record in evidence['records']:
        i = record['original_entry_index']; item = original[i]; row = item['source_row']
        locator = record['original_locator']; key = (locator['block'], locator['group_order'], locator['row_order'])
        if i in ids or key in policies: raise ValueError('Duplicate absolute compaction record')
        ids.add(i)
        if (_plain(locator) != _plain({k:item[k] for k in locator})
                or item['collection_role'] != 'FIT' or row.get('split') not in ('fit','development')
                or _direct_user_annotation(row) or row['id'] != record['source_row_id']
                or original_uv_sha256(row) != record['original_uv_float64_sha256']
                or not np.array_equal(row['uv'], record['original_uv'])
                or not np.array_equal(row['sigma'], record['original_stored_sigma_px'])
                or 'full_main_bottom_qualified_packet.npz' not in str(row.get('origin'))):
            raise ValueError('Only the exact reviewed original nearboard FIT curve may compact')
        _ref(record['working_sigma_authority']); _ref(record['physical_identity_authority'])
        keep = np.asarray(record['kept_original_point_indices'], np.int64)
        mass = np.asarray(record['retained_point_integration_mass'], float)
        before_mass = np.asarray(record['original_point_integration_mass'], float)
        if (mass.shape != keep.shape or not np.isfinite(mass).all() or np.any(mass <= 0.)
                or not np.isclose(mass.sum(), before_mass.sum(), rtol=1e-12, atol=1e-10)
                or record['source_polyline_max_chord_deviation_px'] > .5
                or keep[0] != 0 or keep[-1] != len(row['uv'])-1):
            raise ValueError('Visible RGB support, endpoints and .5px curve shape must be conserved')
        if not np.array_equal(np.asarray(row['uv'])[keep], record['retained_uv']):
            raise ValueError('Compaction can only retain original measured UV')
        stored = np.asarray(row['sigma'],float); prior = np.asarray(record['original_prior_sampling_multiplier'],float)
        working = float(record['original_working_sigma_px'])
        if (working != 4. or not np.array_equal(stored/working, prior)
                or not np.allclose(prior**2, len(stored), rtol=1e-14,atol=1e-14)):
            raise ValueError('Explicit original working sigma and original sqrt(N) sampling correction required')
        policies[key] = (record, keep, mass, prior[keep])
    # This packet is deliberately a finite current-source policy, not a
    # source_edges-wide density filter. Do not silently extend its inventory.
    if len(policies) != 49 or sum(len(original[i]['source_row']['uv']) for i in ids) != 1850:
        raise ValueError('Reviewed 49-row nearboard compaction inventory differs')
    budgets = {}
    for key, (record, keep, mass, prior) in policies.items():
        group = (key[0], int(record['original_locator']['group_actual_pts']), record['source_identity'])
        budgets[group] = budgets.get(group, 0.) + float(mass.sum())
    old_q = np.asarray(state.q).copy(); old_cursor = new_cursor = 0
    all_keep = []; reports = []; weights_by_family = {}
    previous_remap=getattr(state,'original_motion_compaction_scalar_remap',None)
    composed_remap={} if previous_remap is not None else None
    for block in state.blocks:
        old_rows = len(block.quadratic); bundle = block.bundle
        block.original_absolute_groups = block.groups
        groups = []; absolute_keep = []; absolute_weights = []; absolute_normalizations = []; offset = 0; changed = []
        for gi, (pts, rows) in enumerate(block.groups):
            selected_rows = []
            for ri, row in enumerate(rows):
                key = (block.name, gi, ri); n = len(row['uv']); keep = np.arange(n,dtype=np.int64)
                loss = np.ones(n); normalization = np.ones(n)
                if key in policies:
                    record, keep, mass, prior = policies[key]
                    # Pixel corrections must not be folded into this old RGB
                    # shape certificate. The current corrected ice rows are
                    # a different, explicitly protected source inventory.
                    if original_uv_sha256(row) != record['original_uv_float64_sha256']:
                        raise ValueError('Runtime nearboard UV differs from source certificate')
                    group = (block.name, int(pts), record['source_identity'])
                    # Stored sigma contains the historical sqrt(N) sampling
                    # correction. Remove it inside the robust argument, not
                    # outside Huber: w*Huber(e/4), including its 8px cutoff.
                    loss = mass / budgets[group]; normalization = prior
                if key in exclusions:
                    if key in policies: raise ValueError('Qualification and representative sampling overlap')
                    keep = np.setdiff1d(keep, exclusions[key], assume_unique=True)
                    loss = np.ones(len(keep)); normalization = np.ones(len(keep))
                if len(keep) != n:
                    selected_rows.append(slice_observed_points(row, keep))
                    changed.append(dict(group_order=gi,row_order=ri,actual_pts=int(pts),id=row['id'],
                        old_points=n,kept_original_point_indices=keep.tolist(),
                        retired_source_identity_point_indices=exclusions.get(key,[])))
                else: selected_rows.append(row)
                absolute_keep.extend((offset+2*keep[:,None]+np.arange(2)).ravel().tolist())
                absolute_weights.extend(np.repeat(loss,2).tolist())
                absolute_normalizations.extend(np.repeat(normalization,2).tolist()); offset += 2*n
            groups.append((pts, selected_rows))
        scalar_keep = np.r_[np.arange(bundle.motion_rows), bundle.motion_rows+np.asarray(absolute_keep,np.int64),
            np.arange(bundle.motion_rows+offset,old_rows)]
        native_keep = scalar_keep[scalar_keep < bundle.source_residual_rows]
        all_keep.append(old_cursor+scalar_keep)
        if previous_remap is not None:
            composed_remap.update(compose_selected_scalar_positions(previous_remap,old_cursor,new_cursor,scalar_keep,old_rows))
        if changed:
            block.groups=groups; bundle.abs_groups=groups
            bundle.native_residual_sigma=bundle.native_residual_sigma[native_keep].copy()
            bundle.source_residual_rows=len(native_keep)
            block.quadratic=block.quadratic[scalar_keep].copy()
            from z190_moving_pivot_source_bundle import motion_source_csr_layout
            bundle.indices,bundle.pointer,bundle.swap_rows=motion_source_csr_layout(bundle.ia,bundle.ib,bundle.index,groups)
            if len(bundle.pointer)-1 != bundle.source_residual_rows:
                raise ValueError('Compacted native absolute CSR row inventory differs')
            block._inactive_material_certificates=set()
            if hasattr(block,'actor_output_capacity'):
                block.actor_output_capacity=dict(block.actor_output_capacity,rows=len(block.quadratic),source_rows=bundle.source_residual_rows)
        weights=np.r_[np.ones(bundle.motion_rows),absolute_weights,np.ones(old_rows-bundle.motion_rows-offset)]
        normalization=np.r_[np.ones(bundle.motion_rows),absolute_normalizations,np.ones(old_rows-bundle.motion_rows-offset)]
        if np.any(weights!=1.)or np.any(normalization!=1.):
            weights_by_family['programme:'+block.name]=SourceLossWeights(weights,normalization)
        reports.append(dict(block=block.name,original_start=old_cursor,original_rows=old_rows,
            compact_start=new_cursor,compact_rows=len(scalar_keep),motion_point_count=len(bundle.a),row_maps=changed))
        old_cursor+=old_rows;new_cursor+=len(scalar_keep)
    for buffer in (capold,template):
        for key in ('scales','quadratic'):
            values=np.asarray(buffer[key]); keep=np.r_[*all_keep,np.arange(old_cursor,len(values))]
            buffer[key]=values[keep].copy()
    # The historical Ad7 retirement already carries a motion-compaction map.
    # Compose the actual current global row change; do not guess an offset.
    if previous_remap is not None:
        if set(composed_remap)!=set(previous_remap):raise ValueError('Every declared original retirement scalar needs an exact compact position')
        state.original_motion_compaction_scalar_remap=composed_remap
    retirement_binding=reindex_retirement_block_binding(state,reports)
    if not np.array_equal(state.q,old_q): raise ValueError('Absolute input compaction changed q')
    contract=dict(schema=SCHEMA,packet_path=str(path),packet_sha256=file_sha256(path),
        original_packet=evidence['original_packet'],blocks=reports,
        removed_dense_nearboard_points=1455,
        removed_unqualified_absolute_points=sum(len(values) for values in exclusions.values()),
        removed_active_scalar_rows=old_cursor-new_cursor,source_history_and_HELD_preserved=True,
        selected_original_retirement_scalar_remap=None if composed_remap is None else {str(k):v for k,v in composed_remap.items()},
        original_retirement_low_block_binding_composition=retirement_binding,
        original_stored_sigma_unchanged=True,original_complete_physical_primitives_unchanged=True,
        original_Huber_threshold_in_stored_sigma_units_unchanged=False,
        nearboard_Huber_threshold_in_explicit_working_sigma_units=2.,nearboard_component_Huber_threshold_px=8.,
        nearboard_source_loss='visible_support_weight * Huber2(pixel_error / original_working_sigma4)',
        nearboard_original_sqrtN_sampling_multiplier_removed_inside_robust_argument=True,
        non_nearboard_loss_sigmas_and_Huber_cutoffs_unchanged=True,
        nearboard_working_sigma_px=4.,source_budget_rule='one authenticated physical curve per actual exposure; visible RGB support mass',
        implementation_sources=[dict(path=str(Path(p).resolve()),sha256=file_sha256(p))
            for p in (Path(__file__),Path(__file__).with_name('source_loss.py'))],
        fresh_complete_source_profiles_and_J_required=True,accuracy_accepted=False)
    state.original_absolute_input_compaction_contract=contract
    state.original_absolute_input_compaction_loss_weights=weights_by_family
    state.original_absolute_point_qualification_prequalification=qualification
    return contract


class SourceWithAbsoluteInputCompaction:
    def __init__(self,base,state):
        self.base=base;self.weights=state.original_absolute_input_compaction_loss_weights
        self.absolute_input_compaction=state.original_absolute_input_compaction_contract
        self.contract=dict(base.contract,original_absolute_input_compaction=deepcopy(self.absolute_input_compaction))
    def __getattr__(self,name): return getattr(self.base,name)
    def apply_block(self,block):
        from z190_streaming_source_normal import SourceRowBlock
        weights=self.weights.get(block.row_ledger.get('family'))
        if weights is None:return block
        if block.loss_weights is not None: raise ValueError('Absolute sampling integration already applied')
        return SourceRowBlock(block.r,block.J,block.quadratic,block.scales,
            dict(block.row_ledger,original_absolute_input_compaction=deepcopy(self.absolute_input_compaction)),weights)
    def blocks(self,q,jacobian=True):
        for block in self.base.blocks(q,jacobian):yield self.apply_block(block)


def wrap_original_absolute_input_compaction(base,state):
    return SourceWithAbsoluteInputCompaction(base,state)
