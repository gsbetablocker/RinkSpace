"""Value-only source tails for a shot-private camera counterfactual.

This reconstructs admitted source operators, not a source graph or Jacobian.
The caller owns the immutable q8 baseline and the shot RF4 counterfactual.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import hashlib
import json

import cv2
import joint_absolute_source_context_20261011 as ctx
import joint_absolute_global_tail_context_20261011 as global_tail
from calibration.solver_runtime.source_loss import weighted_source_cost
from structured_programme_source_candidate import native_board_track_source as track_module
from structured_programme_source_candidate.persistent_ice_long_baseline_source import PersistentIceLongBaselineFactor
from z190_fixed_target_cross_view_factor import fixed_target_source_factor

a=ctx.a
np=ctx.np
BOOT=Path(str(a.RUN)+'.bootstrap')
AUTH=BOOT/'startup_source_auth_execution/actual_startup_source_auth_reuse.json'
AUTH_SHA='a9997db395424de4bf1e19c7dedb26cf18c8a92c9129802065c693130724cf1f'
N=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp')
CROSS_REFS=(
 (N/'ice_alignment_review/long_side_real_camera_source_v1/low5_to_main_B_whole_pan_tilt_head_support_QUALIFIED_FROZEN_v2.json',
  'cffb67e5774c177e1bc27928d41936da8dc3af3b8ce400dfd347dc87d96e53d2'),
 (N/'source_solver_audit/low_camera_mutual_native_source_v1/secondary6_to_low_whole_pan_tilt_head_support_QUALIFIED_FROZEN_v3.json',
  '95a2e7e37f98d55666772e8063f6fc2d131289cc730c493e1bf49183e5ab46c8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


class SourceBytes:
    """Fresh stream SHA; missing PNGs reuse an authenticated decode receipt.

    The original video and manifest bytes are freshly hashed, while the prior
    exact-PTS BGR/PNG reproduction is explicitly historical, not a new decode.
    """
    def __init__(self):
        if sha(AUTH)!=AUTH_SHA:raise ValueError('Saved native-byte authentication changed')
        receipt=a.read(AUTH)
        if receipt['succeeded'] is not True:raise ValueError('Incomplete saved source authentication')
        self.missing={};self.cache={};self.receipts=[]
        for r in receipt['missing_native_png_authentications']:
            key=Path(r['requested_path']).resolve()
            if key in self.missing and self.missing[key]!=r:raise ValueError('Conflicting saved PNG authentication')
            self.missing[key]=r

    def __call__(self,value):
        path=Path(value).resolve()
        if path.exists():
            stamp=identity(path)
            if path not in self.cache:
                digest=sha(path)
                if identity(path)!=stamp:raise ValueError('Source changed during stream SHA')
                self.cache[path]=(stamp,digest)
            elif self.cache[path][0]!=stamp:raise ValueError('Authenticated source changed')
            return self.cache[path][1]
        r=self.missing.get(path)
        if r is None:raise FileNotFoundError(path)
        for prefix in ('manifest','source'):
            dependency=Path(r[prefix+'_path'])
            if identity(dependency)!=r[prefix+'_identity'] or self(dependency)!=r[prefix+'_sha256']:
                raise ValueError('Saved missing-PNG authority changed')
        cache=a.read(r['manifest_path'])
        rows=[x for x in cache['frames'] if int(x['native_pts'])==int(r['native_pts'])]
        # The frozen receipt itself binds the original literal path and exact
        # native presentation; the current manifest SHA binds all its fields.
        if len(rows)!=1:raise ValueError('Ambiguous saved native presentation')
        self.receipts.append(dict(requested_path=str(path),native_pts=r['native_pts'],actual_pts=r['actual_pts'],
            reproduced_original_PNG_sha256=r['reproduced_original_PNG_sha256'],
            original_BGR_SHA=r['native_BGR_sha256'],saved_decode_receipt_path=str(AUTH),saved_decode_receipt_sha256=AUTH_SHA,
            source_and_manifest_fresh_stream_SHA=True,source_images_decoded_now=0,
            historical_exact_native_BGR_PNG_reproduction_reused=True))
        return r['reproduced_original_PNG_sha256']

    def finish(self):
        for path,(stamp,digest) in self.cache.items():
            if identity(path)!=stamp:raise ValueError('Source identity changed before tail construction finished')


@contextmanager
def authenticated_track_constructor(auth):
    old=track_module._sha
    track_module._sha=auth
    try:yield
    finally:track_module._sha=old


def _saved(c,family):
    rows=c.ledgers[family]
    if len(rows)!=1:raise ValueError('Exactly one saved family ledger required: '+family)
    row=rows[0];ledger=row['original_row_ledger'];lo,hi=ledger['original_start'],ledger['original_stop']
    return row,c.arrays['quadratic'][lo:hi].copy(),c.arrays['scales'][lo:hi].copy()


def _add(owner,family,evaluate,labels,dependency_pts,*,saved_family=None,basis=None,saved_cost=None):
    c=owner.c;baseline=np.asarray(evaluate(c.q),float)
    saved,quad,scales=_saved(c,saved_family or family)
    if saved_family is not None:raise ValueError('Use an explicit prior partition for partial programme tails')
    if baseline.shape!=quad.shape or baseline.shape!=scales.shape:raise ValueError('Original scalar order/count changed: '+family)
    cost=weighted_source_cost(baseline,quad,None)
    target=saved['normalized_cost'] if saved_cost is None else saved_cost
    if abs(cost-target)>2e-8:raise ValueError('Saved baseline cost mismatch: '+str((family,cost,target)))
    owner.families.append(dict(family=family,evaluate=evaluate,baseline=baseline,quadratic=quad,scales=scales,
        labels=labels,dependency_pts=sorted(set(map(int,dependency_pts))),basis=basis,
        baseline_receipt=dict(scalar_rows=len(baseline),cost=cost,saved_cost=target,cost_difference=cost-target,
            same_q8_complete_source_family=True)))


def _persistent(c,auth):
    contract=c.manifest['additional_real_native_persistent_ice_long_baseline']
    path=Path(contract['packet_path'])
    if auth(path)!=contract['packet_sha256']:raise ValueError('Persistent source packet changed')
    packet=a.read(path);clock=contract['source_clock_authority']
    if auth(clock['path'])!=clock['sha256']:raise ValueError('Persistent source clock changed')
    native_endpoints={int(r[k]['source_native_pts'])for r in packet['new_rows']for k in ('anchor','target')}
    clock_rows={}
    with Path(clock['path']).open(encoding='utf8')as stream:
        for line in stream:
            record=json.loads(line)
            if int(record['full_pts'])in native_endpoints:clock_rows[int(record['full_pts'])]=record
    prior=packet.get('parent_source_packet',packet.get('parent_v1_source_packet'))
    if prior is not None and auth(prior['path'])!=prior['sha256']:raise ValueError('Persistent source append parent changed')
    factor=PersistentIceLongBaselineFactor.__new__(PersistentIceLongBaselineFactor)
    factor.geometry=c.state.geometry.current;factor.world_indices=c.world_indices;factor._border_cache=OrderedDict()
    factor.tracks={};factor.rows=[];factor.held=[];aliases=c.state.original.aliases;frames={}
    def endpoint(ref):
        p=int(ref['source_native_pts']);actual=int(aliases[p])
        if 'source_actual_pts'in ref and actual!=int(ref['source_actual_pts']):raise ValueError('Persistent actual source clock changed')
        if ref['source_clock_authority_sha256']!=clock['sha256'] or Path(ref['source_clock_authority_path']).resolve()!=Path(clock['path']).resolve():
            raise ValueError('Persistent endpoint clock authority changed')
        row=clock_rows.get(p)
        if row is None or row.get('accepted_official_pts') is None or int(row['accepted_official_pts'])!=int(ref['official_pts']):
            raise ValueError('Persistent endpoint lacks its authenticated accepted official exposure')
        png=ref['raw_png'];expected=ref['raw_png_sha256']
        if png not in frames:
            if auth(png)!=expected:raise ValueError('Persistent original PNG changed')
            frames[png]=expected
        elif frames[png]!=expected:raise ValueError('Conflicting persistent PNG SHA')
        return dict(native_pts=p,actual_pts=actual,raw_png=png,raw_png_sha256=expected,native_bgr_sha256=ref['native_bgr_sha256'])
    for t in packet['tracks']:
        factor.tracks[t['track_id']]=dict(anchor=endpoint(t['anchor']),uv=np.asarray(t['anchor']['uv'],float))
    for literal in packet['new_rows']:
        r=deepcopy(literal);r.update(anchor_endpoint=endpoint(r['anchor']),target_endpoint=endpoint(r['target']),
            target_uv=np.asarray(r['xy1'],float),sigma=np.asarray(r['sigma_native_px'],float))
        if not np.array_equal(r['sigma'],[6.,6.]):raise ValueError('Persistent original sigma changed')
        t=factor.tracks[r['track_id']]
        if r['anchor_endpoint']!=t['anchor'] or not np.array_equal(r['xy0'],t['uv']):raise ValueError('Persistent anchor association changed')
        (factor.rows if r['split']=='fit' else factor.held).append(r)
    if [r['id']for r in factor.rows]!=contract['fit_ids'] or [r['id']for r in factor.held]!=contract['held_ids']:
        raise ValueError('Persistent admitted FIT/HELD order changed')
    def evaluate(value):
        pts={r[k]['actual_pts']for r in factor.rows for k in ('anchor_endpoint','target_endpoint')}
        cameras={p:ctx.camera(c,value,p)for p in pts}
        return factor._evaluate(value,factor.rows,cameras=cameras)[0]
    def guard(value):
        all_rows=factor.rows+factor.held
        pts={r[k]['actual_pts']for r in all_rows for k in ('anchor_endpoint','target_endpoint')}
        cameras={p:ctx.camera(c,value,p)for p in pts}
        factor._evaluate(value,all_rows,cameras=cameras)
    labels=[dict(id=r['id'],scalar_rows=2,anchor_actual_pts=r['anchor_endpoint']['actual_pts'],
        target_actual_pts=r['target_endpoint']['actual_pts'],basis='original_current_ray_finite_ice_z0')for r in factor.rows]
    return factor,evaluate,labels,guard


def prepare_tail_context(c,shot_pts):
    shot=set(map(int,shot_pts));columns=np.concatenate([c.state.bypts[p].start+4*c.state.bypts[p].bundle.index[p]+np.arange(4)for p in sorted(shot)])
    owner=SimpleNamespace(c=c,shot_pts=shot,shot_columns=columns,families=[],gaps=[],proof=[])
    # Install the same public calibration hooks before any original tail call.
    c._global_tail_roll=global_tail._install_camera_hooks(c)
    auth=SourceBytes()
    with authenticated_track_constructor(auth):
        tracks=track_module.NativeBoardTrackFactor(c.state,c.options['--additional-native-board-track-source'],len(c.q))
    labels=[];dependency=set()
    for r in tracks.fit_evaluation_rows:
        if r.get('_anchor'):
            pts=[int(r['actual_pts'])];kind='frozen_initialized_print_anchor'
        elif r.get('initialization'):
            pts=[int(r['target_actual_pts'])];kind='frozen_initialized_print_target'
        else:
            pts=[int(r['source_actual_pts']),int(r['target_actual_pts'])];kind='original_current_ray_finite_board_transfer'
            pair=tracks.shared_print_templates.get(str(r.get('shared_print_pair_id')))
            if pair is not None:pts+= [int(c.state.original.aliases[int(pair[k+'_anchor_pts'])])for k in ('canonical','counterpart')]
        dependency.update(pts)
        labels.append(dict(id=r['id'],scalar_rows=r['sigma'].size,source_actual_pts=r.get('source_actual_pts',r.get('actual_pts')),
            target_actual_pts=r.get('target_actual_pts',r.get('actual_pts')),operator_camera_actual_pts=pts,
            source_native_pts=r.get('source_pts',r.get('native_pts')),target_native_pts=r.get('target_pts',r.get('native_pts')),
            frozen_template_id=r.get('initialization',{}).get('template_id'),shared_print_pair_id=r.get('shared_print_pair_id'),
            touches_shot=bool(shot.intersection(pts)),basis=kind,initialization=bool(r.get('initialization')),
            actual_source_anchor=bool(r.get('_anchor')),frozen_template_is_independent_measured_XYZ=False))
    _add(owner,'additional_native_static_board_tracks',lambda value:tracks.block(value,False).r,labels,dependency,
        basis='per-row current-ray or frozen-initialized-print; frozen normalized print is not measured XYZ')
    factor,evaluate,labels,persistent_guard=_persistent(c,auth)
    dependency={r[k]['actual_pts']for r in factor.rows for k in ('anchor_endpoint','target_endpoint')}
    _add(owner,'additional_real_native_persistent_ice_long_baseline',evaluate,labels,dependency,
        basis='admitted original current anchor ray -> finite ice z0 -> target Brown; frozen source admission reused without BGR decode')
    owner.track_factor=tracks;owner.persistent_factor=factor;owner.persistent_guard=persistent_guard
    _add_cross(owner,auth)
    _add_roll(owner)
    _add_height_invariance(owner,auth)
    for tail in global_tail.prepare_tails(c):
        # Public actual dependency columns, rather than the family name,
        # establish whether this family can change under shot-private RF4.
        if np.intersect1d(tail['columns'],columns).size:raise ValueError('Global tail touches unhandled private shot columns')
        _add(owner,tail['family'],tail['evaluate'],[dict(label,scalar_rows=1)for label in tail['row_labels']],[],basis='public original owner dependency columns disjoint from shot RF4')
        owner.proof.append(dict(family=tail['family'],private_shot_column_intersection=[],columns=tail['columns'].tolist()))
    auth.finish()
    owner.source_authentication=dict(saved_missing_PNG_receipt_path=str(AUTH),saved_missing_PNG_receipt_sha256=AUTH_SHA,
        current_distinct_files_stream_SHA_checked=len(auth.cache),missing_PNG_receipts=auth.receipts,
        source_images_decoded=0,source_images_written=0,persistent_source_BGR_validation_is_prior_admission=True)
    owner.baseline_q=c.q.copy()
    return owner


def _add_cross(owner,auth):
    c=owner.c;roles={'long_side_main':0,'long_side_secondary':1,'low_end':2}
    for path,expected in CROSS_REFS:
        if auth(path)!=expected:raise ValueError('Original physical target packet changed')
        packet=a.read(path)
        for r in packet['rows']:
            if r['split']!='fit':continue
            p=int(c.state.original.aliases[int(r['source_native_pts'])])
            if p!=int(r['source_actual_pts']):raise ValueError('Physical target source exposure changed')
            ref=r['source_native_reference']
            if auth(ref['raw_png'])!=ref['raw_png_sha256']:raise ValueError('Physical target source PNG changed')
            bounds=np.array([[float('-inf')if lo is None else float(lo),float('inf')if hi is None else float(hi)]
                for lo,hi in zip(r['physical_axis_broad_support_xy_min'],r['physical_axis_broad_support_xy_max'])])
            fs=dict(source_visible=True,source_support_manually_qualified=True,model_projection_used_to_generate_or_select_uv=False,
                physical_target_kind='fixed_pan_head_pivot',fixed_pivot_source_qualified=True,
                physical_target_authority=dict(whole_head_mechanical_origin_support=True),
                native_support_bounds_xy_px=bounds.tolist(),sigma_native_px=np.broadcast_to(r['source_sigma_engineering_native_px'],2).tolist(),
                sigma_semantics=r['sigma_semantics'])
            observer=c.state.ids[roles[r['observer_installation']]];target=c.state.ids[roles[r['target_installation']]]
            def evaluate(value,p=p,fs=fs,observer=observer,target=target):
                b=c.state.bypts[p];cal=b.bundle.current_calibration(value);start=b.start+4*b.bundle.index[p];view=value[start:start+4]
                return fixed_target_source_factor(fs,value[c.state.slots[observer]],value[c.state.slots[target]],view[:3],view[3],
                    np.asarray(cal.lever_xyz)*c.provider.scene_unit_m,np.array([1920*cal.principal_x_fraction[0],1080*cal.principal_y_fraction[0]]),cal.k1[0]).residual
            family='physical_target:'+r['id']
            _add(owner,family,evaluate,[dict(id=r['id'],actual_pts=p,scalar_rows=2,observer_installation=observer,
                target_installation=target,mechanical_origin_is_broad_support_not_exact_pixel=True)], [p],
                basis='original fixed panhead support; original normalizer scalar fields reconstructed from authenticated admitted packet, no new BGR decode')


def _add_roll(owner):
    c=owner.c;roll=c._global_tail_roll;bias=np.asarray(c.state.structured_programme_coordinate_chart.bias_indices,int)
    sigma=float(roll['component_sigma_radians'])
    saved=a.read(a.H/'current_all_programme_roll_prior_q276_q8a15_readonly_20261011.json')
    costs={r['block']:r['prior_cost_after']for r in saved['per_block']}
    for b in c.state.blocks:
        inst=c.state.ids.index(b.bundle.calibration.installation_id);pts=list(map(int,b.bundle.stamps));positions=b.start+4*np.arange(len(pts))
        def evaluate(value,positions=positions,inst=inst):
            angles=[]
            for p in positions:
                R=cv2.Rodrigues(np.asarray(value[p:p+3]))[0];x,y=R[0,2],-R[1,2]
                if x*x+y*y<1e-8:raise ValueError('Original roll domain undefined')
                angles.append(np.arctan2(x,y))
            return (np.asarray(angles)-value[bias[inst]])/sigma
        baseline=evaluate(c.q);cost=weighted_source_cost(baseline,np.ones(len(pts),bool),None)
        if abs(cost-costs[b.name])>2e-8:raise ValueError('Saved roll-prior partition differs')
        owner.families.append(dict(family='programme_roll_prior:'+b.name,evaluate=evaluate,baseline=baseline,
            quadratic=np.ones(len(pts),bool),scales=np.ones(len(pts)),dependency_pts=pts,
            labels=[dict(actual_pts=p,scalar_rows=1,basis='engineering_installation_correlated_roll')for p in pts],
            basis='exact original explicit correlated-roll partition, not an extra objective term',
            baseline_receipt=dict(scalar_rows=len(pts),cost=cost,saved_cost=costs[b.name],cost_difference=cost-costs[b.name],
                same_q8_saved_original_partition=True)))


def _add_height_invariance(owner,auth):
    c=owner.c;low=next(b for b in c.state.blocks if b.name=='low')
    folder=a.R/'prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory'
    path=folder/'initial_current_state4rf.npz';expected='5109414894b74593aa16ab6bf074902f25137dfaac8119921070fb8745c7c5ac'
    if auth(path)!=expected:raise ValueError('Original LOW reference exposure inventory changed')
    with np.load(path,allow_pickle=False)as z:
        stamps=z['canonical_pts'].copy()
    if not np.array_equal(stamps,low.bundle.stamps):raise ValueError('Original LOW actual view inventory changed')
    reference=int(stamps[0])
    producer=a.ROOT/'calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking/z190_low_shared_F_scene_rf_recovery.py'
    producer_sha='ae93446d2d74b017984ce1ac32273d55886ae3353fc92cb79b6ad84735ab9a09'
    if auth(producer)!=producer_sha:raise ValueError('Original LOW reference producer changed')
    if reference in owner.shot_pts:raise ValueError('LOW reference height is not invariant under this shot counterfactual')
    hypothesis=low.bundle.calibration.field_settings['working_hypothesis']
    target=float(hypothesis['lens_height_above_ground_m']+hypothesis['ground_minus_ice_m'])
    sigma=float(hypothesis['height_sigma_m'])
    if (target,sigma)!=(1.6,.18):raise ValueError('Original LOW engineering height definition changed')
    def evaluate(value):return np.array([(ctx.camera(c,value,reference)[2]*c.provider.scene_unit_m-target)/sigma])
    cal_ref=low.bundle.calibration.field_settings['reference_optical_height_prior']
    if int(cal_ref['reference_exposure_pts'])!=reference or (float(cal_ref['target_above_ice_working_m']),float(cal_ref['sigma_m']))!=(target,sigma):
        raise ValueError('Saved camera reference-height authority differs from original source reference')
    family='optical_height:'+low.bundle.calibration.installation_id
    _add(owner,family,evaluate,[dict(actual_pts=reference,scalar_rows=1,basis='engineering_reference_lens_height')],[reference],
        basis='CurrentProgrammeSource extra height at installation first block reference; exact original LOW reference is outside shot5')
    owner.proof.append(dict(family='programme_original_reference_height:low',reference_actual_pts=reference,
        scalar_rows=1,source_inventory_path=str(path),source_inventory_sha256=expected,
        public_reference_producer_path=str(producer),public_reference_producer_sha256=producer_sha,
        producer_definition='MovingPivotSourceBundle(reference_pts=stamps[0]); original prior uses camera(ref).Cz',
        engineering_target_m=target,engineering_sigma_m=sigma,private_shot_column_intersection=[],
        same_value_as_extra_reference_height=True,baseline_cost=owner.families[-1]['baseline_receipt']['cost'],
        individual_programme_height_residual_not_archived=True,
        explicit_extra_height_and_programme_original_height_are_distinct_existing_scalar_rows=True))
    for b in c.state.blocks:
        if b is low:continue
        cols=np.arange(b.start,b.start+4*len(b.bundle.stamps))
        if np.intersect1d(cols,owner.shot_columns).size:raise ValueError('Other programme RF4 owner overlaps shot')
        owner.proof.append(dict(family='programme_original_reference_height:'+b.name,
            basis='All owner view RF4 and installation parameters unchanged, independent of the reference choice',
            private_shot_column_intersection=[],numeric_value_not_recomputed=True))
    for inst in c.state.ids:
        if inst==low.bundle.calibration.installation_id:continue
        b=next(b for b in c.state.blocks if b.bundle.calibration.installation_id==inst)
        ref=b.bundle.calibration.field_settings['reference_optical_height_prior'];p=int(ref['reference_exposure_pts'])
        if p in owner.shot_pts:raise ValueError('Other installation reference entered shot')
        def evaluate(value,p=p):return np.array([(ctx.camera(c,value,p)[2]*c.provider.scene_unit_m-1.6)/.18])
        if (float(ref['target_above_ice_working_m']),float(ref['sigma_m']))!=(1.6,.18):raise ValueError('Saved original height scale changed')
        _add(owner,'optical_height:'+inst,evaluate,[dict(actual_pts=p,scalar_rows=1,basis='engineering_reference_lens_height')],[p],
            basis='Original extra optical-height tail at saved installation reference; observer RF4 outside shot5')
    for family in ('strong_high_wide_source','soft_nominal_geometry_engineering_references',
                   'explicit_optics_engineering_working_prior','common_row_midpoint'):
        owner.proof.append(dict(family=family,
            basis='Original world, private wide, installation optics/mount/pivots and pattern/placement remain exactly fixed; no programme RF4 dependency',
            private_shot_column_intersection=[],numeric_value_not_recomputed=True))
    for inst in c.state.ids:
        owner.proof.append(dict(family='fixed_pivot_board_side:'+inst,
            basis='Original finite world perimeter and installation pivot inputs both remain exactly fixed',
            private_shot_column_intersection=[],numeric_value_not_recomputed=True))


def evaluate_tail_values(owner,value):
    c=owner.c;value=np.asarray(value,float)
    if not np.array_equal(c.q,owner.baseline_q):raise ValueError('The baseline q was mutated')
    if value.shape!=c.q.shape or not np.isfinite(value).all():raise ValueError('Complete finite counterfactual q required')
    unchanged=np.ones(len(value),bool);unchanged[owner.shot_columns]=False
    if not np.array_equal(value[unchanged],c.q[unchanged]):raise ValueError('Only declared private shot RF4 may change')
    families=[]
    for f in owner.families:
        try:r=np.asarray(f['evaluate'](value),float)
        except ValueError as error:
            families.append(dict(family=f['family'],valid=False,domain_error=str(error),basis=f['basis'],
                baseline_receipt=f['baseline_receipt']));continue
        cost=weighted_source_cost(r,f['quadratic'],None);rows=[];cursor=0
        for label in f['labels']:
            label=deepcopy(label);n=int(label['scalar_rows']);sl=slice(cursor,cursor+n);cursor+=n
            label.update(baseline_cost=weighted_source_cost(f['baseline'][sl],f['quadratic'][sl],None),
                counterfactual_cost=weighted_source_cost(r[sl],f['quadratic'][sl],None),
                normalized_residual_change_max=float(np.max(np.abs(r[sl]-f['baseline'][sl]),initial=0)))
            rows.append(label)
        if cursor!=len(r):raise ValueError('Tail row provenance omitted scalar rows')
        touches=bool(owner.shot_pts.intersection(f['dependency_pts']))
        if not touches and not np.array_equal(r,f['baseline']):raise ValueError('A declared invariant original family changed')
        kinds={}
        for row in rows:
            kind=row.get('basis','original_source');group=kinds.setdefault(kind,dict(rows=0,scalar_rows=0,baseline_cost=0.,counterfactual_cost=0.))
            group['rows']+=1;group['scalar_rows']+=row['scalar_rows'];group['baseline_cost']+=row['baseline_cost'];group['counterfactual_cost']+=row['counterfactual_cost']
        families.append(dict(family=f['family'],valid=True,scalar_rows=len(r),baseline_receipt=f['baseline_receipt'],by_operator_basis=kinds,
            counterfactual_cost=cost,cost_change=cost-f['baseline_receipt']['cost'],depends_on_shot_RF4=touches,
            normalized_residual_change_max=float(np.max(abs(r-f['baseline']),initial=0)),rows=rows,basis=f['basis']))
    try:
        owner.persistent_guard(value)
        persistent_domain=dict(valid=True,FIT_rows=78,HELD_rows=3,HELD_not_in_cost=True)
    except ValueError as error:persistent_domain=dict(valid=False,domain_error=str(error),HELD_not_in_cost=True)
    return dict(families=families,persistent_full_original_domain=persistent_domain,
        source_authentication=owner.source_authentication,dependency_proofs=owner.proof,gaps=owner.gaps,
        all_families_valid=all(f['valid']for f in families),source_images_decoded=0,source_images_written=0,
        fitted_or_saved_new_q=False,complete_QP_objective_claimed=False)


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    c=ctx.load_context();shot_pts=sorted(p for p in c.state.bypts if 229945856<=p<230369024)
    owner=prepare_tail_context(c,shot_pts);result=evaluate_tail_values(owner,c.q)
    if not result['all_families_valid'] or not result['persistent_full_original_domain']['valid']:
        raise ValueError('Baseline original tail domain invalid')
    for f in result['families']:
        if f['cost_change']!=0 or f['normalized_residual_change_max']!=0:raise ValueError('Baseline replay changed')
        if f['family'].startswith('programme_roll_prior:'):f.pop('rows')
    result.update(schema='rinkspace.low-shot-private-camera-original-tail-value-baseline.v1',
        helper_path=str(Path(__file__).resolve()),helper_sha256=sha(__file__),
        complete_q8_sha256=c.identity['q_sha256'],state_npz_path=str(c.path),state_npz_sha256=c.identity['state_npz_sha256'],
        shot_actual_views=len(shot_pts),shot_range_actual_pts=[229945856,230369024],
        native_board_tracks_complete_fit_scalars=2532,persistent_complete_fit_scalars=156,
        counterfactual_evaluated=False,complete_source_J_generated=False,
        static_world_and_installation_private_tail_proofs_do_not_certify_complete_motion_or_all_absolute_or_all86_guards=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=a.plain)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),families=len(result['families']),
        baseline_complete=True,source_images_decoded=0,source_images_written=0)))


if __name__=='__main__':main()
