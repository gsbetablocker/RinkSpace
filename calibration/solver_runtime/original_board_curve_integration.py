"""Opt-in per-exposure original board-curve budgets outside original Huber2.

This adapter retains every current active UV, sigma, R/J and hard constraint.
It neither compresses sparse curves nor changes source-identity qualifications.
Original dense detector traces supply support at already retained anchors;
sources without a declared continuous trace use isolated measured anchors.
Explicitly installed RGB pixel corrections authenticate their revised full
observation and independent singleton support; no old detector arc is reused.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from .source_loss import SourceLossWeights,loss_weight_arrays
from rinkspace.source_sampling import compile_curve_sampling,source_support_masses

SCHEMA='rinkspace.original-board-curve-source-integration.v1'
CORRECTED_RGB_SCHEMA='rinkspace.original-corrected-RGB-anchor-support-receipt.v1'
CORRECTED_RGB_BASIS='isolated_measured_corrected_RGB_anchors'

def file_reference(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def compose_disjoint_loss_weights(existing,additional,*,existing_slots,additional_slots,rows):
    """Combine declared disjoint policies without multiplying a budget twice.

    Slot ownership is explicit, including slots whose policy value happens
    to be one. Untouched residual components keep their original loss.
    """
    old=np.asarray(existing_slots);new=np.asarray(additional_slots)
    if old.dtype!=bool or new.dtype!=bool or old.shape!=(rows,) or new.shape!=(rows,):
        raise ValueError('Exact boolean scalar policy ownership required')
    if np.any(old&new):raise ValueError('Source loss policies overlap scalar slots')
    def arrays(value,ownership):
        w,c=loss_weight_arrays(value,rows)
        w=np.ones(rows) if w is None else w
        c=np.ones(rows) if c is None else c
        if np.any((~ownership)&((w!=1.)|(c!=1.))):
            raise ValueError('Nonidentity source loss outside declared policy slots')
        return w,c
    ow,oc=arrays(existing,old);nw,nc=arrays(additional,new)
    w=np.asarray(ow).copy();c=np.asarray(oc).copy()
    w[new]=nw[new];c[new]=nc[new]
    return SourceLossWeights(w,c if np.any(c!=1.) else None)

def isolated_anchor_masses(uv):
    """No inferred continuity, occlusion coverage or inter-anchor edge."""
    uv=np.asarray(uv,float)
    return source_support_masses(uv,[[i] for i in range(len(uv))])

def verified_original_observation_subset(row,original):
    """Exact source-to-active lineage, reusable for any measured curve role."""
    return _verified_observation_subset(row,original,np.asarray(original['uv'],float))

def _verified_observation_subset(row,original,expected_full_uv):
    """The caller must separately authenticate any revised full observation."""
    from .source_point_qualification import original_uv_sha256
    before=np.asarray(expected_full_uv,float);current=np.asarray(row['uv'],float)
    if before.shape!=np.asarray(original['uv']).shape:
        raise ValueError('Certified observation revision changed the original point layout')
    mapping=row.get('active_original_absolute_input_compaction')
    if mapping is None:keep=np.arange(len(before),dtype=int)
    else:
        keep=np.asarray(mapping.get('retained_original_point_indices'),int)
        if (mapping.get('original_point_count')!=len(before)
                or mapping.get('original_uv_float64_sha256')!=original_uv_sha256(dict(uv=before))
                or mapping.get('retained_uv_float64_sha256')!=original_uv_sha256(row)
                or keep.ndim!=1 or np.any(keep<0) or np.any(keep>=len(before))
                or len(keep)>1 and np.any(np.diff(keep)<=0)):
            raise ValueError('Original-to-active source observation lineage changed')
    old_sigma=np.broadcast_to(np.asarray(original['sigma'],float),(len(before),))
    sigma=np.broadcast_to(np.asarray(row['sigma'],float),(len(current),))
    if (row['id']!=original['id'] or keep.shape!=(len(current),)
            or not np.array_equal(current,before[keep]) or not np.array_equal(sigma,old_sigma[keep])):
        raise ValueError('Original source UV or pointwise sigma is not its certified active subset')
    return keep,old_sigma

def _plain(value):
    return json.loads(json.dumps(value,default=lambda v:v.tolist() if isinstance(v,np.ndarray)
        else v.item() if isinstance(v,np.generic) else str(v)))

def _reference_identity(reference):
    if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
            or not reference['path'] or not isinstance(reference.get('sha256'),str)
            or len(reference['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in reference['sha256'])):
        raise ValueError('Exact source evidence path and SHA256 required')
    return dict(path=str(Path(reference['path']).resolve()),sha256=reference['sha256'])

class AuthenticatedCorrectedRGBAnchors:
    """Authenticate installed source revisions, never arbitrary row metadata.

    These are isolated actual RGB measurements. Their numerical singleton mass
    claims neither a dense trace nor visible arclength between measurements.
    """
    def __init__(self,state,original_packet,source_video_sha256):
        self.state=state;self.original_packet=_reference_identity(original_packet)
        self.source_video_sha256=source_video_sha256
        self.packet=None;self.packet_reference=None;self.receipts={};self.raw_frames={}
        self.used_references={}

    def _load_installed_packet(self):
        if self.packet is not None:return
        contract=getattr(self.state,'original_ice_pixel_correction',None)
        if (not isinstance(contract,dict)
                or contract.get('schema')!='rinkspace.original-native-ice-pixel-correction.v1'
                or contract.get('source_video_sha256')!=self.source_video_sha256
                or contract.get('time_base')!='1/15360'
                or _reference_identity(contract.get('original_absolute_packet',{}))!=self.original_packet
                or contract.get('original_world_sigma_roles_source_order_masks_and_scalar_slots_unchanged')is not True):
            raise ValueError('Corrected board anchors require the installed immutable source correction contract')
        reference=dict(path=contract['packet_path'],sha256=contract['packet_sha256'])
        path=Path(reference['path']).resolve()
        if file_reference(path)!=_reference_identity(reference):
            raise ValueError('Installed board source correction packet changed')
        packet=json.loads(path.read_text(encoding='utf8'))
        if (packet.get('schema')!=contract['schema']
                or packet.get('source_video_sha256')!=self.source_video_sha256
                or packet.get('time_base')!='1/15360'
                or _reference_identity(packet.get('original_absolute_packet',{}))!=self.original_packet):
            raise ValueError('Installed source correction and original packet identity differ')
        self.packet=packet;self.packet_reference=_reference_identity(reference);self.contract=contract

    def row(self,row,entry,original_index):
        metadata=row.get('active_source_pixel_correction')
        if metadata is None:return None
        self._load_installed_packet()
        if (dict(path=str(Path(metadata['packet_path']).resolve()),sha256=metadata['packet_sha256'])!=self.packet_reference
                or metadata.get('original_packet_row_index')!=original_index):
            raise ValueError('Active board correction differs from its installed packet')
        records=[r for r in self.packet['records'] if r.get('original_packet_row_index')==original_index]
        histories=[r for r in self.contract['corrected_rows'] if r.get('original_packet_row_index')==original_index]
        if len(records)!=1 or len(histories)!=1:raise ValueError('One installed original board correction per source row required')
        record=records[0];history=histories[0];old=entry['source_row']
        locator={k:entry[k] for k in ('block','collection_role','group_actual_pts','group_order','row_order')}
        if (history['locator']!=locator or _plain(history['original_source_row'])!=_plain(old)
                or not (history.get('original_top_material_preserved')is True
                        or history.get('original_contact_material_preserved')is True)
                or history.get('point_indices')!=record.get('point_indices')
                or history.get('corrected_uv')!=record.get('corrected_uv')
                or metadata.get('point_indices')!=record.get('point_indices')):
            raise ValueError('Corrected board pixels lack exact original finite-material installation history')
        from structured_programme_source_candidate.original_ice_pixel_correction import _verify_record_original
        points=record['point_indices'];_verify_record_original(record,entry,old,points)
        full=np.asarray(old['uv'],float).copy();full[points]=np.asarray(record['corrected_uv'],float)
        keep,stored=_verified_observation_subset(row,old,full)
        # A source revision can change only its admitted pixels/support. The
        # existing finite material and pointwise working errors remain exact.
        for key in ('id','pts','split','origin','point_correspondence','curve_observation_kind',
                    'scanline_branch','scanline_fixed_axis','extent_side'):
            if _plain(row.get(key))!=_plain(old.get(key)):
                raise ValueError('Corrected board source operator or role changed: '+key)
        for key in ('world','upper_world'):
            if _plain(row.get(key))!=_plain(old.get(key)):
                raise ValueError('Corrected board finite WORLD changed: '+key)
        normal=old.get('normal')
        if normal is None:
            if row.get('normal')is not None:raise ValueError('Corrected board source normal changed')
        elif not np.array_equal(np.asarray(row.get('normal')),np.asarray(normal)[keep]):
            raise ValueError('Corrected board source normal changed')
        expected_support=old.get('normal_interval')
        if expected_support is not None:
            expected_support=np.asarray(expected_support,float).copy()
            if record.get('corrected_normal_interval_px')is not None:
                expected_support[points]=np.asarray(record['corrected_normal_interval_px'],float)
        if ((expected_support is None and row.get('normal_interval')is not None)
                or (expected_support is not None and not np.array_equal(np.asarray(row.get('normal_interval')),expected_support[keep]))):
            raise ValueError('Corrected board optical support differs from its admitted revision')
        reference=record.get('source_boundary_measurement')
        if not isinstance(reference,dict) or type(reference.get('record_index'))is not int:
            raise ValueError('Corrected board support requires an independent authenticated RGB receipt')
        ref=_reference_identity(reference);key=(ref['path'],ref['sha256'])
        if key not in self.receipts:
            if file_reference(ref['path'])!=ref:raise ValueError('Corrected board RGB receipt changed')
            self.receipts[key]=json.loads(Path(ref['path']).read_text(encoding='utf8'))
        receipt=self.receipts[key];i=reference['record_index']
        if (receipt.get('schema')!=CORRECTED_RGB_SCHEMA
                or receipt.get('source_video_sha256')!=self.source_video_sha256
                or receipt.get('time_base')!='1/15360'
                or _reference_identity(receipt.get('original_absolute_packet',{}))!=self.original_packet
                or not 0<=i<len(receipt.get('records',[]))):
            raise ValueError('Corrected board RGB receipt source identity differs')
        measured=receipt['records'][i]
        from .source_point_qualification import original_uv_sha256
        if (measured.get('original_packet_row_index')!=original_index
                or measured.get('locator')!=locator
                or measured.get('original_full_uv_float64_sha256')!=original_uv_sha256(old)
                or not np.array_equal(np.asarray(measured.get('full_corrected_uv')),full)
                or measured.get('source_support_basis')!=CORRECTED_RGB_BASIS
                or measured.get('raw_source_pixels_read')is not True
                or measured.get('observed_UV_generated_from_model')is not False):
            raise ValueError('RGB receipt must authenticate this exact corrected original board observation')
        mask=np.asarray(measured.get('measured_mask',[True]*len(full)))
        paths=measured.get('source_visible_paths')
        if (mask.dtype!=bool or mask.shape!=(len(full),) or not mask[keep].all()
                or paths!=[[int(j)] for j in np.flatnonzero(mask)]):
            raise ValueError('Corrected board support must be isolated measured active anchors without gap coverage')
        raw=measured.get('source_raw_frame')
        if not isinstance(raw,dict) or type(raw.get('pts'))is not int:
            raise ValueError('Actual source RGB frame and native PTS required')
        raw_ref=_reference_identity(raw);native=raw['pts']
        if (native not in self.state.original.aliases
                or int(self.state.original.aliases[native])!=int(entry['group_actual_pts'])
                or not any(_reference_identity(e)==raw_ref and e.get('pts')==native
                           for e in self.packet.get('source_evidence',[]))):
            raise ValueError('Corrected board RGB must be the installed same-exposure source evidence')
        raw_key=(raw_ref['path'],raw_ref['sha256'],raw.get('native_bgr24_sha256'))
        if raw_key not in self.raw_frames:
            if file_reference(raw_ref['path'])!=raw_ref:raise ValueError('Corrected board original RGB bytes changed')
            import cv2
            image=cv2.imread(raw_ref['path'],cv2.IMREAD_COLOR)
            if (image is None or image.shape!=(1080,1920,3) or image.dtype!=np.uint8
                    or hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()!=raw_key[2]):
                raise ValueError('Corrected board native BGR pixels differ from their actual source receipt')
            self.raw_frames[raw_key]=True
        mass=source_support_masses(full,paths,measured_mask=mask)[keep]
        self.used_references[key]=ref
        proof=dict(kind=CORRECTED_RGB_BASIS,measurement_reference=dict(ref,record_index=i),
            correction_packet_reference=self.packet_reference,native_source_pts=native,
            source_raw_frame=raw,original_active_point_indices=keep.tolist(),
            continuous_support_claim=False,dense_original_detector_support_reused=False,
            excluded_anchor_cells_not_redistributed=True,qualified_active_source_support_mass=float(mass.sum()))
        return keep,stored,mass,proof

def dense_trace_anchor_masses(trace,selected,*,gap_policy):
    """Original trace arclength cells on an exact measured-anchor subset.

    An explicitly declared station-gap policy separates source support. A fragment without any
    retained original anchor contributes no invented measurement or mass.
    """
    trace=np.asarray(trace,float);selected=np.asarray(selected,float)
    if trace.ndim!=2 or trace.shape[1:]!=(2,) or not len(trace):
        raise ValueError('Original finite measured source trace required')
    if not np.isfinite(trace).all() or not np.isfinite(selected).all():
        raise ValueError('Finite original source pixels required')
    lookup=defaultdict(list)
    for i,p in enumerate(trace):lookup[p.tobytes()].append(i)
    indices=[]
    for p in selected:
        hits=lookup.get(p.tobytes(),[])
        if len(hits)!=1:raise ValueError('Retained UV must occur exactly once in original detector trace')
        indices.append(hits[0])
    if len(set(indices))!=len(indices):raise ValueError('Repeated anchor needs explicit source lineage')
    if np.any(np.diff(indices)<=0):raise ValueError('Original measured trace order changed')
    # Source detector stations advance in x; vertical jitter is not a missing
    # station. Refuse reversed stations rather than sorting pixels into a path.
    if not isinstance(gap_policy,dict):raise ValueError('Explicit source station-gap policy required')
    method=gap_policy.get('method')
    dx=np.diff(trace[:,0]);positive=dx[dx>0.]
    if np.any(dx<=0.) or not len(positive):
        if len(trace)!=1:raise ValueError('Original detector grid must be ordered in x')
        bounds=np.array([0,1])
        step=None;threshold=None
    elif method=='declared_regular_source_x_grid':
        step=float(gap_policy['step_native_px'])
        if not np.isfinite(step) or step<=0. or not np.allclose(dx/step,np.rint(dx/step),rtol=0.,atol=1e-8):
            raise ValueError('Original measured stations disagree with declared source grid')
        threshold=step*(1.+1e-8)
        bounds=np.r_[0,np.flatnonzero(dx>threshold)+1,len(trace)]
    elif method=='inferred_median_station_spacing':
        factor=float(gap_policy['gap_factor']);step=float(np.median(positive))
        if not np.isfinite(factor) or factor<=1. or gap_policy.get('source_gap_visibility_certified')is not False:
            raise ValueError('Explicit engineering gap proposal, not visibility certification, required')
        threshold=factor*step;bounds=np.r_[0,np.flatnonzero(dx>threshold)+1,len(trace)]
    else:raise ValueError('Explicit declared-grid or disclosed inferred-gap method required')
    mass=np.zeros(len(selected));cells=[];covered=0.
    for a,b in zip(bounds[:-1],bounds[1:]):
        slots=[j for j,i in enumerate(indices) if a<=i<b]
        if not slots:continue
        if b-a==1:
            mass[slots[0]]=1.;covered+=1.
            cells.append(dict(raw_trace_indices=[int(a),int(b)],anchor_indices=slots,source_arclength_px=[0.,1.]))
            continue
        segment=trace[a:b];arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(segment,axis=0),axis=1))]
        at=arc[np.asarray([indices[j]-a for j in slots])]
        boundaries=np.r_[0.,.5*(at[:-1]+at[1:]),arc[-1]]
        for j,value in zip(slots,np.diff(boundaries)):mass[j]+=float(value)
        covered+=float(arc[-1])
        cells.append(dict(raw_trace_indices=[int(a),int(b)],anchor_indices=slots,source_arclength_px=boundaries.tolist()))
    if np.any(mass<=0.) or not np.isclose(mass.sum(),covered,rtol=1e-12,atol=1e-9):
        raise ValueError('Original support mass was lost or bridged')
    return mass,dict(source_station_gap_policy=dict(gap_policy),actual_station_step_px=step,actual_break_threshold_px=threshold,
        station_break_is_not_independent_visibility_or_occlusion_validation=True,
        raw_measured_points=len(trace),retained_original_trace_indices=indices,
        visible_source_fragments=len(bounds)-1,represented_source_fragments=len(cells),source_support_mass=float(covered),
        unrepresented_fragments_add_no_measurements=True,source_cells=cells)

class OriginalDetectorSupport:
    def __init__(self,source_video_sha256,gap_policy):
        self.source_video_sha256=source_video_sha256;self.gap_policy=gap_policy
        self.files={};self.references={};self._frame_cache={}
    def masses(self,original,uv,actual,*,original_point_indices):
        full_uv=np.asarray(original['uv'],float);keep=np.asarray(original_point_indices,int)
        if not np.array_equal(full_uv[keep],np.asarray(uv,float)):
            raise ValueError('Support requested for a nonoriginal source UV')
        origin=Path(str(original.get('origin','')))
        if 'source_edges' not in origin.parts or origin.suffix.lower()!='.npz':
            return isolated_anchor_masses(full_uv)[keep],dict(kind='isolated_original_measured_anchors',continuous_support_claim=False)
        key=str(origin.resolve())
        if key not in self.files:
            reference=file_reference(origin)
            with np.load(origin,allow_pickle=False) as z:
                data={name:z[name] for name in ('pts','offsets','observed_uv','edge_id','edge_names','source_sha256')}
            if str(data['source_sha256'].item())!=self.source_video_sha256:raise ValueError('Original detector source video changed')
            data['index']={int(p):i for i,p in enumerate(data['pts'])}
            self.files[key]=data;self.references[key]=reference
        data=self.files[key];native=int(original.get('source_native_pts',original.get('pts',actual)))
        if native not in data['index']:raise ValueError('Original source detector native PTS absent')
        role='board_top' if str(original['id']) in ('far_top','board_top','board_top_near_visible') else 'ice_board_contact'
        name=np.flatnonzero(data['edge_names']==role)
        if len(name)!=1:raise ValueError('Exact original detector material role absent')
        frame_key=(key,native,int(name[0]))
        if frame_key not in self._frame_cache:
            i=data['index'][native];a,b=map(int,data['offsets'][i:i+2]);mask=data['edge_id'][a:b]==int(name[0])
            self._frame_cache[frame_key]=np.asarray(data['observed_uv'][a:b][mask],float)
        trace=self._frame_cache[frame_key]
        # Qualifying out a wrong/occluded original anchor also removes its
        # original source-support cell. Do not spread invalid support to its
        # neighbours by recomputing cells from only the remaining anchors.
        mass,receipt=dense_trace_anchor_masses(trace,full_uv,gap_policy=self.gap_policy)
        return mass[keep],dict(receipt,kind='original_dense_source_detector_trace',source_reference=self.references[key],native_source_pts=native,
            original_active_point_indices=keep.tolist(),excluded_anchor_cells_not_redistributed=True,
            qualified_active_source_support_mass=float(mass[keep].sum()),
            detector_role=role,source_UV_is_not_independently_verified_by_this_adapter=True)

def declared_working_sigma(row,stored):
    """Remove only an explicit historical sqrt(N) policy, never infer one."""
    text=str(row.get('sigma_modelling_decision',''))
    if row.get('curve_observation_kind')=='confirmed_lower_long_side_top_horizontal_support':
        if 'times sqrt(unique exposure scan rows)' in text:
            basis='explicit_original_outside_support_sigma4_times_sqrt_unique_scan_rows'
        elif text=='one correlated physical cap edge per actual exposure; original engineering sigma4 times sqrt(combined upper+lower row count); original lower sigma separately retained':
            # The original combined-cap producer records both the old lower
            # observations and the separately qualified upper source.  Its
            # sqrt(N) uses the original combined row, never the active subset.
            lower=np.asarray(row.get('original_lower_source_uv'),float)
            lower_sigma=np.asarray(row.get('original_lower_source_sigma'),float)
            if (lower.ndim!=2 or lower.shape[1:]!=(2,) or not len(lower)
                    or lower_sigma.shape!=(len(lower),)
                    or not np.allclose(lower_sigma,4.*np.sqrt(len(lower)),rtol=1e-12,atol=1e-12)
                    or not isinstance(row.get('upper_source_qualification'),dict)
                    or len({tuple(p) for p in lower}&{tuple(p) for p in np.asarray(row['uv'],float)})!=len(lower)):
                raise ValueError('Original combined-cap sampling authority or lower lineage changed')
            basis='explicit_original_outside_support_sigma4_times_sqrt_combined_upper_lower_rows'
        else:
            raise ValueError('Original lower-side sampling correction lacks explicit authority')
        n=len(row['uv']);prior=np.full(n,np.sqrt(n));working=np.asarray(stored)/prior
        if not np.allclose(working,4.,rtol=1e-12,atol=1e-12):
            raise ValueError('Declared original engineering sigma4/sqrtN decomposition changed')
        return working,prior,basis
    return np.asarray(stored).copy(),np.ones(len(stored)),'original_pointwise_working_sigma_no_count_factor_declared'

def physical_board_curve(declaration):
    identity,level=declaration
    if identity in ('qualified_lower_long_side_top','original_reviewed_lower_long_side_top'):
        return 'board_top_original_finite_lower_long_side'
    return 'board_top_original_perimeter' if level=='top' else 'board_ice_contact_original_perimeter'

def reconcile_isolated_anchor_with_declared_trace(records):
    """An isolated numerical cell adds no support to an existing trace cell.

    Original UV and working sigmas stay in their own loss operators.  Only
    the shared physical anchor's integration mass is represented once. Two
    incompatible measured trace-cell claims remain an error, not a choice.
    """
    anchors=defaultdict(list);receipt=[]
    for record in records:
        group=(record['source_video_sha256'],record['actual_pts'],record['physical_curve_id'],record['operator_id'])
        for i,uv in enumerate(record['uv']):anchors[(group,uv.tobytes())].append((record,i))
    for (group,_),copies in anchors.items():
        if len(copies)<2:continue
        trace=[(r,i) for r,i in copies if r['source_support_basis']=='original_dense_source_detector_trace']
        isolated=[(r,i) for r,i in copies if r['source_support_basis']in('isolated_original_measured_anchors',CORRECTED_RGB_BASIS)]
        if len(trace)+len(isolated)!=len(copies):raise ValueError('Unrecognized original support representation')
        if not trace:continue
        mass=float(trace[0][0]['point_integration_mass'][trace[0][1]])
        if any(not np.isclose(float(r['point_integration_mass'][i]),mass,rtol=1e-12,atol=1e-12) for r,i in trace):
            raise ValueError('Duplicate measured trace support cells conflict')
        if any(float(r['point_integration_mass'][i])!=1. for r,i in isolated):
            raise ValueError('Isolated original anchor must have only its numerical singleton support')
        if isolated:
            receipt.append(dict(source_video_sha256=group[0],actual_pts=group[1],physical_curve_id=group[2],operator_id=group[3],
                uv=copies[0][0]['uv'][copies[0][1]].tolist(),shared_source_anchor_mass=mass,
                original_copies=[dict(row_id=r['id'],point_index=i,original_support_mass=float(r['point_integration_mass'][i]),working_sigma_px=float(r['working_sigma_px'][i])) for r,i in copies],
                numerical_singleton_support_does_not_add_to_measured_trace_cell=True,original_UV_and_pointwise_sigma_preserved=True))
            for r,i in copies:r['point_integration_mass'][i]=mass
    return receipt

class SourceWithOriginalBoardCurveIntegration:
    def __init__(self,base,state,policy_path):
        started=time.perf_counter();self.base=base
        path=Path(policy_path).resolve();policy=json.loads(path.read_text(encoding='utf8'))
        if policy.get('schema')!=SCHEMA or policy.get('curve_budget')!=1. or policy.get('enable_point_compression')is not False:
            raise ValueError('Explicit budget1 original-board policy without point compression required')
        source_video_sha256=policy.get('source_video_sha256')
        if not isinstance(source_video_sha256,str) or len(source_video_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_video_sha256):
            raise ValueError('Explicit event source-video SHA256 required')
        gap_policy=policy.get('source_station_gap_policy')
        if not isinstance(gap_policy,dict):raise ValueError('Explicit source station-gap policy required')
        if gap_policy.get('method')=='declared_regular_source_x_grid':
            authority=gap_policy.get('source_grid_authority')
            if not isinstance(authority,dict) or file_reference(authority['path'])!=authority:
                raise ValueError('Declared source station grid needs exact producer authority')
        self._source_video_sha256=source_video_sha256
        packet_ref=policy['original_packet'];packet_path=Path(packet_ref['path']).resolve()
        if file_reference(packet_path)!=dict(path=str(packet_path),sha256=packet_ref['sha256']):
            raise ValueError('Original absolute pixel packet identity changed')
        entries=pickle.loads(packet_path.read_bytes())['programme_absolute_rows']
        original={(e['block'],e['group_order'],e['row_order']):(i,e) for i,e in enumerate(entries) if e['collection_role']=='FIT'}
        from shared_board_course_candidate.course_absolute_material import declared_absolute_course_identity
        self.count=base.count;self.scales=base.scales;self.quadratic=base.quadratic
        self._payloads={};self._new_slots={};self._old_slots={};self._rows={}
        records=[];support=OriginalDetectorSupport(source_video_sha256,gap_policy);families=[];summary=[];skipped=[]
        corrected_support=AuthenticatedCorrectedRGBAnchors(state,packet_ref,source_video_sha256)
        existing=getattr(state,'original_absolute_input_compaction_loss_weights',{})
        for block in state.blocks:
            family='programme:'+block.name;nrows=len(block.quadratic);offset=int(block.bundle.motion_rows)
            selected=np.zeros(nrows,bool);family_records=[];source_rows=[]
            previous=existing.get(family)
            if previous is None:occupied=np.zeros(nrows,bool)
            else:
                w,c=loss_weight_arrays(previous,nrows)
                occupied=(w!=1.)|((c!=1.) if c is not None else False)
            for gi,(actual,rows) in enumerate(block.groups):
                for ri,row in enumerate(rows):
                    count=len(row['uv']);slots=np.arange(offset,offset+2*count);offset+=2*count
                    declaration=declared_absolute_course_identity(row)
                    if declaration is None:continue
                    if row.get('point_correspondence')is True:raise ValueError('Independent material corners cannot be curve-normalized')
                    found=original.get((block.name,gi,ri))
                    if found is None:raise ValueError('Actual source row locator changed')
                    original_index,entry=found
                    if int(entry['group_actual_pts'])!=int(actual):raise ValueError('Actual source row locator changed')
                    old=entry['source_row'];corrected=corrected_support.row(row,entry,original_index)
                    if corrected is None:keep,full_stored=verified_original_observation_subset(row,old)
                    else:keep,full_stored,corrected_mass,corrected_proof=corrected
                    uv=np.asarray(row['uv'],float);stored=full_stored[keep]
                    if count==0:
                        skipped.append(dict(block=block.name,group_order=gi,row_order=ri,actual_pts=int(actual),id=row['id'],reason='all_original_anchors_source_qualified_out'))
                        continue
                    if np.any(occupied[slots]):
                        if not np.all(occupied[slots]):raise ValueError('Existing board policy partially overlaps an original row')
                        skipped.append(dict(block=block.name,group_order=gi,row_order=ri,actual_pts=int(actual),id=row['id'],reason='already_lossweighted_original_curve'))
                        continue
                    full_working,full_prior,basis=declared_working_sigma(old,full_stored)
                    working=full_working[keep];prior=full_prior[keep]
                    mass,proof=(support.masses(old,uv,int(actual),original_point_indices=keep)
                                if corrected is None else (corrected_mass,corrected_proof))
                    name=f'{family}:{gi}:{ri}'
                    record=dict(id=name,split='fit',actual_pts=int(actual),source_video_sha256=source_video_sha256,
                        physical_curve_id=physical_board_curve(declaration),operator_id=str(row.get('curve_observation_kind','finite_curve_vector_distance')),
                        uv=uv,stored_sigma_px=stored,working_sigma_px=working,prior_sampling_multiplier=prior,
                        point_integration_mass=mass,sampling_kind=['measured']*count,
                        source_support_basis=proof['kind'],source_anchor_identity_authority=packet_ref)
                    records.append(record);family_records.append((name,slots));selected[slots]=True
                    source_rows.append(dict(group_order=gi,row_order=ri,actual_pts=int(actual),id=row['id'],points=count,
                        physical_curve_id=record['physical_curve_id'],declared_identity=declaration[0],sigma_decomposition=basis,
                        original_point_indices=keep.tolist(),original_point_count=len(old['uv']),
                        original_UV_sha256=hashlib.sha256(np.asarray(old['uv'],float)[keep].tobytes()).hexdigest(),
                        current_active_UV_sha256=hashlib.sha256(uv.tobytes()).hexdigest(),support_basis=proof['kind'],
                        corrected_RGB_source_proof=proof if corrected is not None else None,
                        original_inverse_sigma_squared_sum=float(np.sum(1/stored**2)),source_support_mass=float(mass.sum())))
            if offset!=int(block.bundle.source_residual_rows):raise ValueError('Exact motion/absolute source scalar boundary changed')
            if family_records:
                families.append((family,nrows,family_records,selected,occupied,previous));summary.extend(source_rows)
        if not records:raise ValueError('No active original board curves for explicit integration')
        shared_anchor_receipt=reconcile_isolated_anchor_with_declared_trace(records)
        anchor_authority=corrected_support.packet_reference if corrected_support.used_references else packet_ref
        for record in records:record['source_anchor_identity_authority']=anchor_authority
        duplicate_policy=dict(kind='same_source_actual_physical_curve_operator_UV_share_budget_preserve_pointwise_sigma',
            source_anchor_identity_authority=anchor_authority)
        plan=compile_curve_sampling(records,duplicate_anchor_policy=duplicate_policy)
        for family,nrows,rows,selected,occupied,previous in families:
            weights=np.ones(nrows);normalization=np.ones(nrows)
            for name,slots in rows:
                weights[slots]=np.repeat(plan['rows'][name]['point_integration_weights'],2)
                normalization[slots]=np.repeat(plan['rows'][name]['working_normalization_multiplier'],2)
            payload=SourceLossWeights(weights,normalization if np.any(normalization!=1.) else None)
            self._payloads[family]=compose_disjoint_loss_weights(previous,payload,existing_slots=occupied,additional_slots=selected,rows=nrows)
            self._new_slots[family]=selected;self._old_slots[family]=occupied;self._rows[family]=nrows
        self.original_board_curve_integration=dict(schema=SCHEMA,policy_reference=file_reference(path),original_packet=packet_ref,
            source_video_sha256=source_video_sha256,source_station_gap_policy=gap_policy,
            source_detector_references=list(support.references.values()),physical_curve_budget=1.,families=list(self._payloads),
            authenticated_corrected_RGB_measurement_references=list(corrected_support.used_references.values()),
            authenticated_source_correction_packet_reference=corrected_support.packet_reference,
            source_rows=summary,groups=plan['groups'],already_weighted_rows_preserved=skipped,
            duplicate_anchor_policy=duplicate_policy,shared_source_anchor_support_reconciliation=shared_anchor_receipt,
            all_current_active_UV_R_J_scales_roles_truth_and_scalar_slots_preserved=True,
            original_dense_traces_read_once_per_file=True,source_identity_qualifications_inherited_not_newly_verified=True,
            native_quadratic_and_independent_material_corners_not_selected=True,
            loss='mass * original Huber2(native pixel residual / explicitly original working sigma)',
            historical_N_removed_only_when_original_producer_explicitly_declares_it=True,
            sparse_original_points_not_deleted=True,compression_enabled=False,
            no_existing_policy_multiplied_or_normalized_twice=True,fresh_source_J_and_profiles_required=True,
            statistical_covariance_or_measured_precision_claim=False,accuracy_accepted=False,
            implementation_sources=[file_reference(__file__),file_reference(Path(__file__).with_name('source_loss.py'))])
        import rinkspace.source_sampling as sampling
        self.original_board_curve_integration['implementation_sources'].append(file_reference(sampling.__file__))
        # Timing is execution evidence, not source-objective identity.
        self.original_board_curve_integration_preparation_seconds=time.perf_counter()-started
        self.contract=dict(base.contract,original_board_curve_integration=self.original_board_curve_integration)

    def __getattr__(self,name):return getattr(self.base,name)
    def apply_block(self,block):
        family=block.row_ledger.get('family')
        if family not in self._payloads:return block
        if len(block.r)!=self._rows[family]:raise ValueError('Original board source scalar family changed')
        if block.row_ledger.get('source_sha256',self._source_video_sha256)!=self._source_video_sha256:
            raise ValueError('Source stream belongs to a different event video')
        if 'original_board_curve_integration' in block.row_ledger:raise ValueError('Original board curve budget already applied')
        existing=block.loss_weights
        old_slots=self._old_slots[family]
        if existing is None:
            if np.any(old_slots):raise ValueError('Existing qualified curve payload missing')
        else:
            w,c=loss_weight_arrays(existing,len(block.r));expected=self._payloads[family]
            # Compare only existing policy slots; new slots intentionally get
            # their separate budget. Any third policy is rejected explicitly.
            ew,ec=loss_weight_arrays(expected,len(block.r));c=np.ones(len(w)) if c is None else c;ec=np.ones(len(w)) if ec is None else ec
            if np.any((~old_slots)&((w!=1.)|(c!=1.))) or not np.array_equal(w[old_slots],ew[old_slots]) or not np.array_equal(c[old_slots],ec[old_slots]):
                raise ValueError('Unexpected or changed existing source loss policy')
        from z190_streaming_source_normal import SourceRowBlock
        return SourceRowBlock(block.r,block.J,block.quadratic,block.scales,
            dict(block.row_ledger,original_board_curve_integration=dict(schema=SCHEMA,policy_reference=self.original_board_curve_integration['policy_reference'],
                original_R_J_sigma_and_roles_unchanged=True,disjoint_existing_loss_payload_preserved=True)),self._payloads[family])
    def blocks(self,q,jacobian=True):
        found=set()
        for block in self.base.blocks(q,jacobian):
            family=block.row_ledger.get('family')
            if family in self._payloads:
                if family in found:raise ValueError('Original programme family duplicated')
                found.add(family)
            yield self.apply_block(block)
        if found!=set(self._payloads):raise ValueError('Original board family omitted by source stream')

def wrap_original_board_curve_integration(source,state,policy_path=None):
    if policy_path is None:return source
    if hasattr(source,'original_board_curve_integration'):raise ValueError('Original board integration already installed')
    return SourceWithOriginalBoardCurveIntegration(source,state,policy_path)
