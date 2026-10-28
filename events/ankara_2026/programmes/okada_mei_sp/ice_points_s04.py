"""S04 source-only ice recovery and current-native point delivery.

The full canonical source pool, branch corrections and later withdrawals are
inputs. Neither fitted world coordinates nor inferred screen continuations are
observations. Native files and original observations are read-only authorities.
"""
from pathlib import Path
import argparse
import collections
import gzip
import functools
import hashlib
import itertools
import json
import sqlite3
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROG = Path(__file__).resolve().parent
DATA = Path('D:/CodexData/RinkSpace_Data')
REC = DATA / 'runs/okada_rec_1019/recover_v2'
ADS = DATA / 'runs/native_ad_points_s04_20261026'
OUT = DATA / 'runs/native_ice_points_s04_20261027'
SCENE = ROOT / 'events/ankara_2026/scene/field/model.json'
LIBRARY = DATA / 'build/native_unclipped_20261027/rinkspace_core.dll'
FIRST, LAST, STEP = 745984, 948736, 256
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(PROG))
from rinkspace.vision_native import NativeSourceGeometry, file_ref
from rinkspace.vision_near_board_qualified import current_graphics
from rinkspace.review.render import source_frames, validate_video


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def rows(p):
    with gzip.open(p, 'rt', encoding='utf8') as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def dump(f, r):
    f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def authority():
    geo = NativeSourceGeometry(SCENE, LIBRARY)
    refs = {r['path']: r for r in geo.refs}
    for p in (ROOT / 'native').rglob('*'):
        if p.is_file():
            refs[str(p.resolve())] = file_ref(p)
    camera = PROG / 'ad_cameras/s04.json'
    for p in [camera, camera.parent / read(camera)['controls_ref']]:
        refs[str(p.resolve())] = file_ref(p)
    path = OUT / 'evidence/native_authority.json'
    now = dict(files=list(refs.values()), current_native_path_count=len(geo.current_paths))
    if path.exists():
        if read(path) != now:
            raise ValueError('Selected native authority changed; do not silently mix generations')
    else:
        write(path, now)
    return geo


def verify_authority():
    for r in read(OUT / 'evidence/native_authority.json')['files']:
        if file_ref(r['path']) != r:
            raise ValueError('Protected authority changed: ' + r['path'])


def corrections():
    excluded, extra, pins = {}, collections.defaultdict(list), []
    for rel in ['dense_ice_v1/branch_delta_v2/manifest.json',
                'source_overlay_check_v1/exclusions_manifest.json']:
        p = REC / rel
        doc = read(p)
        pins.append(file_ref(p))
        ep = Path(doc.get('exclusions_file', doc.get('exclusions', {}).get('file', '')))
        expected = doc.get('exclusions_sha256', doc.get('exclusions', {}).get('sha256'))
        ref = file_ref(ep)
        if ref['sha256'] != expected:
            raise ValueError('Historical correction changed')
        pins.append(ref)
        for n, r in rows(ep):
            excluded[r['source_row_id']] = dict(reason=r.get('reason', r.get('reason_code')),
                file=str(ep), jsonl_line_0based=n)
        if 'rows_file' in doc:
            dp = Path(doc['rows_file'])
            ref = file_ref(dp)
            if ref['sha256'] != doc['rows_sha256']:
                raise ValueError('Branch reread changed')
            pins.append(ref)
            for n, r in rows(dp):
                if r['shot_id'] == 's04':
                    extra[r['native_pts']].append((n, r, str(dp)))
    # The authenticated source index stores original source IDs, not display IDs.
    index = DATA / 'runs/okada_s02_s09_observations_20261026/source_index.sqlite'
    withdrawn = {}
    with sqlite3.connect(index.as_uri() + '?mode=ro', uri=True) as db:
        for key, payload in db.execute("select id,payload from qualification where shot='s04'"):
            withdrawn[key] = json.loads(payload)
    write(OUT / 'evidence/correction_sources.json', dict(files=pins,
        withdrawal_index=str(index), exact_withdrawn_source_ids=len(withdrawn),
        withdrawals=withdrawn, superseded_zero_delta_not_consumed=True))
    return excluded, extra, withdrawn, pins


def material_path(geo, r):
    name = r['feature_id']
    row = geo.lookup(name)
    # These historical suffixes name a material curve, not its measurement
    # operator. Only an actual two-sided midpoint can bind the current centreline.
    if row is None and name in ['ice.C0:outer', 'ice.A1:outer', 'ice.A2:outer',
                                'ice.A3:outer', 'ice.A4:outer']:
        if r.get('definition') == 'width_midpoint_between_both_material_edges':
            row = geo.lookup(name.replace(':outer', ':centreline'))
    return row


def source_domain(im, mask, board):
    graphics, receipt = current_graphics(im)
    q = np.asarray(board['roles']['board.contact']['trace_uv'], float)
    q = q[np.isfinite(q).all(1)]
    q = q[np.argsort(q[:, 0])]
    if len(q) < 2:
        raise ValueError('Missing actual source board-contact support')
    axis = np.arange(1920)
    border = np.interp(axis, q[:, 0], q[:, 1])
    supported = np.zeros(1920, bool)
    for span in board['roles']['board.contact']['spans']:
        a, b = span['x_range']
        supported |= (axis >= a) & (axis <= b)
    missing = np.flatnonzero(~supported)
    if len(missing):
        for part in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
            a, b = part[[0, -1]]
            flank = [border[k] for k in (a-1, b+1) if 0 <= k < 1920 and supported[k]]
            if flank:
                border[a:b+1] = max(flank)
    distance = cv2.distanceTransform((mask == 0).astype('uint8'), cv2.DIST_L2, 5)
    return graphics, border, distance, receipt


def support_positions(r):
    q = [r['uv']]
    if r.get('paired_edge_uv') is not None:
        q.extend(np.asarray(r['paired_edge_uv']).reshape(-1, 2).tolist())
    quartiles = r.get('pixel_support', {}).get('side_interface_quartiles_uv')
    if quartiles is not None:
        q.extend(np.asarray(quartiles).reshape(-1, 2).tolist())
    return np.asarray(q, float)


def navigation(*, output_dir=None, visibility_path=None, authority_init=authority, authority_verify=verify_authority, board_dir=None):
    """Reuse adjacent image registration solely to transport finite source spans."""
    from rinkspace.registration import adjacent_lk
    cv2.setNumThreads(2)
    authority_init()
    output_dir=Path(output_dir) if output_dir is not None else OUT
    frames=read(visibility_path or ADS/'evidence/visibility.json')['frames']
    chain=[np.eye(3)];components=[0];receipts=[];previous=None;start=time.monotonic()
    scale=np.diag([.5,.5,1.])
    for i,f in enumerate(frames):
        im=cv2.imread(f['image']);pt=f['pts']
        if hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:
            raise ValueError('Temporal navigation source pixels changed')
        mask=cv2.imread(f['mask_path'],0)
        board=read((Path(board_dir)if board_dir is not None else REC/'dense_board_v1/qualified_frames')/f'{pt}.json')
        if f.get('near_outside_ref'):
            near=cv2.imread(f['near_outside_ref']['file'],0)
            if near is None or near.shape!=mask.shape:raise ValueError('Missing near-board visibility source')
            mask=np.maximum(mask,near)
        graphics,border,distance,_=source_domain(im,mask,board)
        good=(~graphics)&(distance>8)&(np.arange(1080)[:,None]>border[None]+8)
        current=(cv2.resize(im,(960,540)),cv2.resize(good.astype('uint8')*255,(960,540),interpolation=cv2.INTER_NEAREST))
        if previous is not None:
            r=adjacent_lk(previous[0],current[0],previous[1],current[1])
            r.update(source_pts=pt,previous_pts=pt-STEP,source_BGR_sha256=f['source_BGR_sha256'],navigation_only=True)
            if r['status']=='candidate':
                pair=np.linalg.inv(scale)@np.linalg.inv(np.array(r['H_reference_from_current']))@scale
                h=pair@chain[-1];h/=h[2,2];chain.append(h);components.append(components[-1])
            else:
                chain.append(np.eye(3));components.append(components[-1]+1)
            receipts.append(r)
        previous=current
        if i%100==0:emit('source_temporal_navigation',frames=i+1,components=components[-1]+1,seconds=round(time.monotonic()-start,1))
    np.savez_compressed(output_dir/'evidence/source_navigation.npz',pts=np.array([f['pts']for f in frames],np.int64),H=np.asarray(chain),component=np.array(components,np.int32))
    write(output_dir/'evidence/source_navigation.json',dict(method='existing_registration.adjacent_lk',
        source_images_only=True,native_geometry_used=False,coordinates_emitted_as_observations=False,
        pairs=receipts,seconds=time.monotonic()-start))
    authority_verify()


class FiniteSourceIdentity:
    def __init__(self, *, output_dir=None, visibility_path=None, packet_dir=None, first_pts=FIRST):
        output_dir=Path(output_dir) if output_dir is not None else OUT
        packet_dir=Path(packet_dir) if packet_dir is not None else PROG/'source_datasets/shot4_20261014/frames'
        sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
        from rinkfit_joint.finite_source_support import finite_support
        self.support=finite_support
        self.source_sha256=read(visibility_path or ADS/'evidence/visibility.json')['source_video_sha256']
        pairs=read(output_dir/'evidence/source_navigation.json')['pairs']
        h=np.eye(3);component=0;self.transforms={first_pts:(h.copy(),component)};unusable=[]
        scale=np.diag([.5,.5,1.])
        for r in pairs:
            # The existing matcher's 2px registration scale also bounds its
            # held-out median. A failed diagnostic is a break in search support,
            # not permission to propagate an inaccurate identity across it.
            if r['status']!='candidate' or r['diagnostic_median_reference_px']>2.:
                h=np.eye(3);component+=1;unusable.append(r['source_pts'])
            else:
                pair=np.linalg.inv(scale)@np.linalg.inv(np.asarray(r['H_reference_from_current']))@scale
                h=pair@h;h/=h[2,2]
            self.transforms[int(r['source_pts'])]=(h.copy(),component)
        write(output_dir/'evidence/temporal_support_gaps.json',dict(source_pts=unusable,
            search_components=component+1,diagnostic_limit_half_resolution_px=2.,
            failed_registration_never_emits_or_moves_a_point=True))
        self.guides=collections.defaultdict(list)
        for path in sorted(packet_dir.glob('*.json')):
            frame=read(path);pt=frame['source_pts']
            if pt not in self.transforms:continue
            for feature in frame['features']:
                key=feature.get('target_id',feature['id'])
                if not key.startswith('ice.'):continue
                spans=[]
                direct=feature.get('source_line_fit',{}).get('direct_readouts',[])
                if len(direct)>=2:
                    spans=[np.asarray([r['uv']for r in direct],float)]
                else:
                    for seg in feature.get('segments',[]):
                        ev=seg.get('source_evidence',[])
                        uv=[r['actual_native_readout_uv']for r in ev if r.get('actual_native_readout_uv')is not None]
                        if len(uv)<2:uv=seg.get('source_native_readout_uv',[])
                        if len(uv)>=2:spans.append(np.asarray(uv,float))
                if spans:
                    self.guides[key].append(dict(pts=pt,path=str(path),spans=spans,operator=feature.get('measurement_operator'),
                        signal=feature.get('material_signal'),transition=feature.get('material_transition'),
                        centre=feature.get('geometric_interpretation',{}).get('center_uv'),
                        width=feature.get('current_thin_paint_material_width_reference',{}).get('maximum_width_px')))

    @functools.lru_cache(maxsize=4096)
    def at(self, feature, pt):
        dest,component=self.transforms[pt]
        available=[g for g in self.guides.get(feature,[])if self.transforms[g['pts']][1]==component]
        if not available:return None
        g=min(available,key=lambda g:abs(g['pts']-pt))
        return g,dest@np.linalg.inv(self.transforms[g['pts']][0])

    def check(self,r):
        # Already RGB-qualified original identities outside the affected
        # navigation families keep their original source authority.
        nav=r.get('navigation',{})
        needs_finite=r['feature_id'] in ['ice.lane3:edge1','ice.lane4:edge1']
        template=nav.get('source_file','').startswith('8a28afed scene')
        donor=nav.get('source_frame_pts',r['native_pts'])
        borrowed=not FIRST<=donor<=LAST
        if not needs_finite and not template and not borrowed:return None
        available=self.at(r['feature_id'],r['native_pts'])
        if available is None:return 'no_actual_source_instance_for_template_navigation'
        g,h=available
        # The historical reader's normal search extent is retained. Longitudinal
        # support is strictly finite; no +/-1600px extrapolation is inherited.
        if not self.support([r['uv']],g['spans'],h,normal_tolerance_px=28.)[0]:
            return 'outside_photographic_finite_source_instance'
        r['finite_source_identity']=dict(source_file=g['path'],source_pts=g['pts'],
            actual_source_readout_spans=True,temporal_navigation=str(OUT/'evidence/source_navigation.npz'),
            current_UV_still_from_original_RGB_profile=True,normal_search_extent_px=28.,endpoint_extension_px=0.)
        return None

    def reread(self,im,pt,geo,present,graphics,border,distance):
        """Fill absent material paths from finite photographic search support.

        Existing raw readers do the actual two-sided/contour measurement. Optical
        flow contributes a search transform only and never an output coordinate.
        """
        from rinkspace.vision_programme_ice_review import _source_queries,_batched_actual_profiles,_actual_profile_candidates
        from rinkspace.vision_source_shapes import _MaterialProfileFrame
        rgb=im[:,:,::-1].copy();frame=_MaterialProfileFrame(rgb,dict(source_pts=pt,source_sha256=self.source_sha256,time_base='1/12800',native_hw=[1080,1920],shot_id='s04'))
        output=[];audit=[]
        for name in self.guides:
            available=self.at(name,pt)
            if available is None:continue
            g,h=available
            prototype=dict(feature_id=name,definition='width_midpoint_between_both_material_edges')
            native=material_path(geo,prototype)
            if native is None or (native['id'] in present and name not in ['ice.lane3:edge1','ice.lane4:edge1']):continue
            operator='paint_center' if native['measurement']=='centreline' else 'material_interface'
            if g['operator']!=operator:continue
            if operator=='material_interface' and g['centre'] is None:continue
            config={}
            if g['centre']is not None:
                config['_projected_center']=cv2.perspectiveTransform(np.array(g['centre'],float)[None,None],h)[0,0]
            allq=[];alln=[];alli=[];spanidx=[]
            for si,span in enumerate(g['spans']):
                hom=np.c_[span,np.ones(len(span))]@h.T
                # Do not create a chord across a projective pole.
                safe=np.isfinite(hom).all(1)&(np.abs(hom[:,2])>1e-9)
                if not safe.all() or np.any(hom[:-1,2]*hom[1:,2]<=0):continue
                uv=hom[:,:2]/hom[:,2,None]
                q,n,interval=_source_queries(config,uv,np.ones(len(uv)),[1080,1920],6.,28.)
                allq.extend(q);alln.extend(n);alli.extend(interval);spanidx.extend([si]*len(q))
            q=np.asarray(allq).reshape(-1,2);normal=np.asarray(alln).reshape(-1,2);interval=np.asarray(alli).reshape(-1,2)
            if not len(q):continue
            off,profile,smooth,valid=_batched_actual_profiles(frame,q,normal,interval,np.array([g['signal']or'dark']*len(q)))
            reasons=collections.Counter();kept=0
            for j in range(len(q)):
                count=int(valid[j].sum())
                choices=_actual_profile_candidates(q[j],normal[j],off[j,:count],profile[j,:count],smooth[j,:count],operator=operator,signal=g['signal']or'dark',transition=g['transition'],with_quantiles=True)
                choices=[x for x in choices if not x[2]]
                if operator=='paint_center' and g['width']is not None:
                    choices=[x for x in choices if np.linalg.norm(x[1][1]-x[1][0])<=g['width']]
                if len(choices)!=1:
                    reasons['no_unique_complete_material_profile']+=1;continue
                uv,pair,_,quartiles=choices[0]
                rr=dict(shot_id='s04',native_pts=pt,feature_id=name,curve_id=name.removeprefix('ice.'),
                    definition='width_midpoint_between_both_material_edges' if operator=='paint_center' else 'declared_target_ring_material_boundary',
                    uv=uv.tolist(),paired_edge_uv=pair.tolist() if operator=='paint_center' else None,
                    source_kind='actual_RGB_reread_from_finite_temporal_photographic_support',source_uv_from_model=False,
                    source_UV_is_direct_measurement=True,source_row_id=f's04_finite_RGB/{pt}/{name}/{j}',
                    visible_span_id=f's04/{pt}/{name}/finite_photograph/{spanidx[j]}',
                    pixel_support=dict(query_index=j,normal=normal[j].tolist(),query_uv=q[j].tolist(),
                        native_shift_px=float((uv-q[j])@normal[j]),side_interface_quartiles_uv=quartiles.tolist(),
                        paint_width_px=float(np.linalg.norm(pair[1]-pair[0])),material_signal=g['signal'],material_transition=g['transition']),
                    navigation=dict(source_file=g['path'],source_frame_pts=g['pts'],same_shot=True,
                        image_registration_only=True,source_reference_is_navigation_not_measured_coordinate=True),
                    finite_source_identity=dict(source_file=g['path'],source_pts=g['pts'],endpoint_extension_px=0.,
                        current_RGB_unique_complete_material_profile=True,temporal_navigation=str(OUT/'evidence/source_navigation.npz')))
                reason=qualify(rr,native,{}, {},graphics,border,distance)
                if reason:reasons[reason]+=1;continue
                output.append(rr);kept+=1
            audit.append(dict(pts=pt,feature=name,queries=len(q),actual_reads=kept,search_reference=g['path'],reference_pts=g['pts'],reasons=dict(reasons)))
        return output,audit


def qualify(r, native, excluded, withdrawn, graphics, border, distance):
    sid = r['source_row_id']
    if sid in excluded:
        return 'historical_correction/' + str(excluded[sid]['reason'])
    if sid in withdrawn or r.get('source_parent_row_id') in withdrawn:
        return 'later_identity_withdrawal'
    if r.get('source_uv_from_model') or r.get('source_UV_is_direct_measurement') is False or 'inferred' in r.get('source_kind', ''):
        return 'inferred_search_continuation_not_a_measurement'
    if native is None:
        return 'no_proven_current_native_correspondence'
    if native['measurement'] == 'centreline':
        pair = r.get('paired_edge_uv')
        if r.get('definition') != 'width_midpoint_between_both_material_edges' or pair is None:
            return 'current_native_requires_two_sided_material_midpoint'
        if not np.allclose(np.mean(pair, axis=0), r['uv'], atol=1e-6, rtol=0):
            return 'stored_uv_is_not_measured_pair_midpoint'
    elif r.get('definition') not in ['declared_target_ring_material_boundary', 'declared_goal_crease_material_boundary']:
        blue_half=(r.get('definition')=='filled_blue_goal_area_material_boundary'
            and native.get('role')=='physical_half_ellipse_boundary'
            and r.get('pixel_support',{}).get('material_signal')=='blue')
        if not blue_half:return 'current_native_contour_semantics_not_proven'
    q = support_positions(r)
    if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > [1919, 1079]):
        return 'material_support_outside_source_image'
    xy = np.rint(q).astype(int)
    if graphics[xy[:, 1], xy[:, 0]].any():
        return 'actual_current_broadcast_graphics_occlusion'
    if (q[:, 1] <= np.interp(q[:, 0], np.arange(1920), border)).any():
        return 'above_actual_source_board_contact'
    # Existing generation_v2 actor blur domain; never the detector rectangle.
    if (distance[xy[:, 1], xy[:, 0]] <= 8.).any():
        return 'existing_actor_silhouette_and_blur_boundary_exclusion'
    return None


def select_representatives(records, native, *, source_curve_guide=None, source_curve_order=None):
    """Use the ad package's geometric reduction on actual source-supported runs."""
    from ad_entity_select_s01 import reduce_run
    by = collections.defaultdict(list)
    curve_geometry=None
    if source_curve_guide is not None:
        curve_geometry=np.asarray(source_curve_guide,float)
        theta=np.asarray(source_curve_order,float)
        if curve_geometry.shape!=(len(records),2)or theta.shape!=(len(records),):raise ValueError('Curve guide does not match original rows')
        order=np.argsort(theta);gaps=np.diff(np.r_[theta[order],theta[order[0]]+2*np.pi])
        order=np.roll(order,-int(gaps.argmax())-1)
        by['complete_source_curve']=list(map(int,order))
    elif native['kind']=='ice_segment':
        # One material line has one common source ordering across all original
        # query batches. Fragment endpoints must not create extra local weight.
        uv=np.asarray([r['uv']for r in records]);centre=np.median(uv,axis=0)
        if len(uv)>=2:
            _,_,vt=np.linalg.svd(uv-centre,full_matrices=False)
            order=np.argsort((uv-centre)@vt[0])
        else:order=np.arange(len(uv))
        by['complete_source_line']=list(map(int,order))
    elif native['kind']in ['ice_circle','ice_ellipse']and len(records)>=5:
        uv=np.asarray([r['uv']for r in records],np.float32)
        centre,axes,angle=cv2.fitEllipse(uv)
        if min(axes)>0 and np.isfinite([*centre,*axes,angle]).all():
            a=np.deg2rad(angle);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            local=(uv.astype(float)-centre)@rotation
            theta=np.arctan2(local[:,1]/axes[1],local[:,0]/axes[0])
            order=np.argsort(theta);gaps=np.diff(np.r_[theta[order],theta[order[0]]+2*np.pi]);order=np.roll(order,-int(gaps.argmax())-1)
            curve_geometry=np.asarray(centre)+(np.c_[np.cos(theta),np.sin(theta)]*(np.asarray(axes)/2))@rotation.T
            by['complete_source_curve']=list(map(int,order))
        else:
            for i,r in enumerate(records):by[r.get('visible_span_id',r['feature_id'])].append(i)
    else:
        for i, r in enumerate(records):
            by[r.get('visible_span_id', r['feature_id'])].append(i)
    selected, protect, why = set(), set(), collections.defaultdict(set)
    for indices in by.values():
        if native['kind']!='ice_segment'and curve_geometry is None:
            indices.sort(key=lambda i: records[i].get('pixel_support', {}).get('query_index', i))
        ii = np.asarray(indices)
        uv = np.array([records[i]['uv'] for i in ii])
        guide=curve_geometry[ii]if curve_geometry is not None else uv
        delta = np.linalg.norm(np.diff(guide, axis=0), axis=1)
        positive = delta[delta > .25]
        # Small profile holes do not need a pair of new nearby representatives.
        # This joins selection support only; it emits no gap-filling coordinate.
        gap = max(12., .04*float(np.linalg.norm(np.ptp(uv,axis=0))), 3 * float(np.median(positive))) if len(positive) else 12.
        cuts = np.r_[0, np.flatnonzero(delta > gap) + 1, len(ii)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            q = uv[a:b]
            # Smooth only the selection guide, never any delivered coordinate.
            smooth = (guide[a:b] if curve_geometry is not None else
                np.array([np.median(q[max(0,j-2):min(len(q),j+3)],axis=0)for j in range(len(q))]))
            if native['kind'] == 'ice_segment' and len(q) >= 2:
                centre = np.median(q, axis=0)
                _, _, vt = np.linalg.svd(q-centre, full_matrices=False)
                u = (q-centre) @ vt[0]
                normal = (q-centre) @ vt[1]
                # Match the ad selector: smooth only normal noise. Median
                # smoothing both UV coordinates collapses a three-readout
                # line to its middle point and loses its measured endpoints.
                normal = np.array([np.median(normal[max(0,j-2):min(len(q),j+3)])for j in range(len(q))])
                smooth = centre + np.outer(u,vt[0]) + np.outer(normal,vt[1])
                if len(q)>=5 and np.ptp(u) > 1:
                    poly = np.polynomial.Polynomial.fit(u, (q-centre)@vt[1], min(3, len(q)//5))
                    smooth = centre + np.outer(u,vt[0]) + np.outer(poly(u),vt[1])
            selected |= reduce_run(records, ii[a:b], smooth, protect, why)
    kept = []
    separation=max(2.5,.01*float(np.linalg.norm(np.ptp(np.asarray([r['uv']for r in records]),axis=0))))
    for i in sorted(selected):
        q = np.asarray(records[i]['uv'])
        n = np.asarray(records[i].get('pixel_support',{}).get('normal',[0.,0.]))
        if any(np.linalg.norm(q-records[j]['uv']) < separation and
               np.dot(n,records[j].get('pixel_support',{}).get('normal',[0.,0.])) > .9 for j in kept):
            continue
        kept.append(i)
    return kept, why


def observation_key(r):
    # Historical readers reused query counters. Preserve that original ID,
    # while the numerical input also identifies its exact immutable source row.
    ref=r['source_ref']
    source=hashlib.sha256(str(Path(ref['path']).resolve()).casefold().encode('utf8')).hexdigest()[:16]
    return 'ice:'+r['source_row_id']+'@'+source+':'+str(ref['jsonl_line_0based'])


def competing_source_ids(records):
    """Do not count two material branches from the same photographic query."""
    by=collections.defaultdict(list)
    for r in records:by[r['source_row_id']].append(r)
    unresolved=set()
    for key,rr in by.items():
        for a,b in itertools.combinations(rr,2):
            pa,pb=a.get('pixel_support',{}),b.get('pixel_support',{})
            if not all('native_shift_px'in p and 'normal'in p for p in (pa,pb)):continue
            qa=np.asarray(a['uv'])-pa['native_shift_px']*np.asarray(pa['normal'])
            qb=np.asarray(b['uv'])-pb['native_shift_px']*np.asarray(pb['normal'])
            if np.linalg.norm(qa-qb)<.01 and np.linalg.norm(np.asarray(a['uv'])-b['uv'])>2.5:
                unresolved.add(key)
    return unresolved


def build():
    cv2.setNumThreads(2)
    geo = authority()
    finite=FiniteSourceIdentity()
    excluded, extra, withdrawn, pins = corrections()
    visibility = read(ADS / 'evidence/visibility.json')
    vis = {r['pts']: r for r in visibility['frames']}
    cp = REC / 'dense_ice_v1/canonical_v3/s04/rows.jsonl.gz'
    cm = read(cp.parent / 'manifest.json')
    ref = file_ref(cp)
    if ref['sha256'] != cm['rows_sha256']:
        raise ValueError('Full ice source pool changed')
    pins += [ref, file_ref(cp.parent / 'manifest.json'), file_ref(ADS / 'ads_only/point_observations.npz')]
    coverage = {r['native_pts']: r for _, r in rows(cp.parent / 'coverage.jsonl.gz')}
    selected, stats, totals = [], [], collections.Counter()
    reread_audit=[]
    rawcounts, selectedcounts, missing = collections.Counter(), collections.Counter(), collections.Counter()
    start = time.monotonic()
    grouped = iter(itertools.groupby(rows(cp), key=lambda x: x[1]['native_pts']))
    current=next(grouped,None)
    with gzip.open(OUT / 'evidence/ice_dispositions.jsonl.gz','wt',encoding='utf8',compresslevel=3) as ledger, gzip.open(OUT / 'evidence/selected_ice_records.jsonl.gz','wt',encoding='utf8',compresslevel=3) as evidence, gzip.open(OUT/'evidence/finite_RGB_readings.jsonl.gz','wt',encoding='utf8',compresslevel=3) as freshfile:
        freshline=0
        for frame_index, pt in enumerate(range(FIRST,LAST+1,STEP)):
            if current is not None and current[0]<pt:
                raise ValueError('Canonical source exposure ordering changed')
            original=list(current[1]) if current is not None and current[0]==pt else []
            if current is not None and current[0]==pt:current=next(grouped,None)
            f = vis[pt]
            im = cv2.imread(f['image'])
            if hashlib.sha256(im.tobytes()).hexdigest() != f['source_BGR_sha256']:
                raise ValueError('Source RGB cache changed')
            if hashlib.sha256(im[:,:,::-1].tobytes()).hexdigest() != coverage[pt]['native_RGB_sha256']:
                raise ValueError('Historical ice and current source image disagree')
            if file_ref(f['mask_path'])['sha256'] != f['mask_sha256']:
                raise ValueError('Historical SAM silhouette changed')
            mask = cv2.imread(f['mask_path'], 0)
            bp = REC / 'dense_board_v1/qualified_frames' / f'{pt}.json'
            board = read(bp)
            if board['decoded_native_BGR_sha256'] != f['source_BGR_sha256']:
                raise ValueError('Source domain belongs to another exposure')
            graphics, border, distance, domain = source_domain(im, mask, board)
            candidates = [(n, r, str(cp)) for n, r in original] + extra.get(pt, [])
            by, dispositions = collections.defaultdict(list), []
            for n, r, path in candidates:
                rawcounts[r['feature_id']] += 1
                native = material_path(geo, r)
                reason = qualify(r, native, excluded, withdrawn, graphics, border, distance)
                if reason is None:reason=finite.check(r)
                record = dict(source_id=r['source_row_id'], pts=pt, feature=r['feature_id'],
                    source_ref=dict(path=path,jsonl_line_0based=n), reason=reason)
                if reason:
                    totals[reason] += 1
                    missing[(r['feature_id'],reason)] += 1
                    dump(ledger, record)
                else:
                    rr = dict(r, native_path_id=native['id'], source_ref=record['source_ref'],
                              source_BGR_sha256=f['source_BGR_sha256'])
                    by[native['id']].append(rr)
            fresh,fa=finite.reread(im,pt,geo,by,graphics,border,distance)
            reread_audit.extend(fa)
            for r in fresh:
                native=material_path(geo,r)
                r.update(native_path_id=native['id'],source_ref=dict(path=str(OUT/'evidence/finite_RGB_readings.jsonl.gz'),jsonl_line_0based=freshline),source_BGR_sha256=f['source_BGR_sha256'])
                dump(freshfile,r);freshline+=1
                by[native['id']].append(r)
            for pid, records in by.items():
                conflicts=competing_source_ids(records)
                for r in records:
                    if r['source_row_id']in conflicts:
                        dump(ledger,dict(source_id=r['source_row_id'],pts=pt,feature=r['feature_id'],
                            native_path_id=pid,source_ref=r['source_ref'],
                            reason='unresolved_competing_actual_material_profiles_same_query'))
                records=[r for r in records if r['source_row_id']not in conflicts]
                if not records:continue
                keep, why = select_representatives(records, geo.current_paths[pid])
                keep = set(keep)
                for i, r in enumerate(records):
                    if i in keep:
                        r['selection'] = dict(reasons=sorted(why[i]), actual_UV_unchanged=True,
                            geometry_guide_used_only_to_select_original_rows=True)
                        r['human_acceptance'] = 'not_issued_by_this_machine_selection'
                        selected.append(dict(uv=r['uv'], pts=pt, id=observation_key(r), path=pid))
                        dump(evidence,r)
                        selectedcounts[r['feature_id']] += 1
                    dump(ledger,dict(source_id=r['source_row_id'],pts=pt,feature=r['feature_id'],
                        native_path_id=pid,source_ref=r['source_ref'],
                        reason='selected_original_actual_material_reading' if i in keep else 'redundant_same_visible_material_support'))
            stat=dict(pts=pt,source_seconds=pt/12800,source_BGR_sha256=f['source_BGR_sha256'],
                original_candidates=len(candidates),eligible_actual=sum(map(len,by.values())),
                selected=sum(x['pts']==pt for x in selected),features={pid:len(rs) for pid,rs in by.items()},
                source_graphics=domain,board_source=file_ref(bp),mask_source=dict(path=f['mask_path'],sha256=f['mask_sha256']))
            stats.append(stat)
            if frame_index % 50 == 0:
                emit('recover_full_s04_ice',frames=frame_index+1,total=793,selected=len(selected),seconds=round(time.monotonic()-start,1))
        if len(stats) != 793:
            raise ValueError('S04 source frame range incomplete')
    # Merge paths, never source-row integers from two different namespaces.
    amap=read(ADS/'ads_only/mapping.json')
    ad=dict(np.load(ADS/'ads_only/point_observations.npz',allow_pickle=False))
    if set(amap['path_ids'])-geo.current_paths.keys():
        raise ValueError('Advertisement references no longer retained native paths')
    paths=list(amap['path_ids'])+list(dict.fromkeys(x['path'] for x in selected))
    lookup={p:i for i,p in enumerate(paths)}
    n=len(ad['pts'])
    points=dict(source_rows=np.arange(n+len(selected),dtype=np.int64),
        observation_ids=np.r_[ad['observation_ids'],np.array([x['id'] for x in selected])],
        uv=np.r_[ad['uv'],np.array([x['uv'] for x in selected])],
        pts=np.r_[ad['pts'],np.array([x['pts'] for x in selected],np.int64)],
        kind=np.r_[ad['kind'],np.ones(len(selected),np.uint8)],
        stations=np.r_[ad['stations'],np.full(len(selected),.5)],
        path_indices=np.r_[ad['path_indices'],np.array([lookup[x['path']] for x in selected],np.int32)])
    order=np.argsort(points['pts'],kind='stable')
    points={k:v[order] for k,v in points.items()}
    target=OUT/'points';target.mkdir(exist_ok=True)
    np.savez_compressed(target/'point_observations.npz',**points)
    from rinkspace.domain.scene_assets import scene_content_identity
    mapping=dict(schema='rinkspace.native_s04_source_points.v1',path_ids=paths,
        scene_revision=scene_content_identity(geo.scene,SCENE),scene_path=str(SCENE),scene_sha256=geo.refs[0]['sha256'],
        camera_path=str(PROG/'ad_cameras/s04.json'),source_video=visibility['source_video'],
        source_video_sha256=visibility['source_video_sha256'],point_input_sha256=file_ref(target/'point_observations.npz')['sha256'],
        source_scope=amap['source_scope'],pixel_domain='native_distorted',source_dimensions=[1920,1080],time_base='1/12800',
        source_row_namespace=dict(advertisement=[0,n],ice=[n,n+len(selected)]),
        advertisements_original_package=str(ADS/'ads_only'),stations_status='curve_values_uninitialized_fixed_corners_preserved',
        source_geometry_used_as_native_geometry=False,human_acceptance=False)
    write(target/'mapping.json',mapping)
    write(OUT/'evidence/source_inventory.json',dict(files=pins,full_canonical_rows=cm['rows'],
        historical_display_sample_not_used_as_full_source=True,source_UV_PTS_immutable=True))
    write(OUT/'evidence/ice_coverage.json',dict(frames=stats,raw_by_feature=dict(rawcounts),
        selected_by_feature=dict(selectedcounts),not_selected_reasons=dict(totals),
        feature_reasons=[dict(feature=k[0],reason=k[1],count=v) for k,v in missing.items()],
        all_visible_materials_claimed=False))
    write(OUT/'evidence/finite_RGB_reread.json',dict(records=reread_audit,total_measured_rows=freshline,
        flow_or_model_UV_emitted=False,current_RGB_actually_read=True))
    write(OUT/'result.json',dict(shot='s04',frames=len(stats),ad_points=n,ice_points=len(selected),
        total_points=len(points['pts']),frames_with_ice=sum(x['selected']>0 for x in stats),
        eligible_dense_ice=sum(x['eligible_actual'] for x in stats),
        original_pixels_and_PTS_preserved=True,current_native_unchanged=True,
        user_visual_acceptance='pending',elapsed_seconds=time.monotonic()-start))
    verify_authority()
    emit('full_s04_ice_selected',**read(OUT/'result.json'))


def boards():
    """Retain sparse actual blue-upper/yellow-lower measurements for full S04."""
    verify_authority();geo=authority();cv2.setNumThreads(2)
    folder=OUT/'points';m=read(folder/'mapping.json')
    if 'board' in m['source_row_namespace']:raise ValueError('Board observations already merged')
    z=dict(np.load(folder/'point_observations.npz'));begin=len(z['pts'])
    vis={r['pts']:r for r in read(ADS/'evidence/visibility.json')['frames']}
    chosen=[];counts=[];reasons=collections.Counter()
    with gzip.open(OUT/'evidence/selected_board_records.jsonl.gz','wt',encoding='utf8')as evidence:
        for pt in range(FIRST,LAST+1,STEP):
            bp=REC/'dense_board_v1/qualified_frames'/f'{pt}.json';b=read(bp);f=vis[pt];bref=file_ref(bp)
            if b['decoded_native_BGR_sha256']!=f['source_BGR_sha256']:raise ValueError('Board source differs')
            if not b['paired_blue_cap_and_white_board_body_semantics']or not b['yellow_contact_from_current_global_band_not_accumulated_extrapolation']:
                raise ValueError('Board edge source semantics unavailable')
            mask=cv2.imread(f['mask_path'],0);im=cv2.imread(f['image']);graphics,_=current_graphics(im)
            by=collections.defaultdict(list)
            for j,r in enumerate(b['rows']):
                if not r.get('pixels_measured_in_this_frame')or not r.get('pixel_support',{}).get('current_pixel_profile_reread'):
                    reasons['not_a_direct_current_material_profile']+=1;continue
                x,y=np.rint(r['uv']).astype(int)
                if not (0<=x<1920 and 0<=y<1080)or mask[y,x]or graphics[y,x]or r.get('occlusion'):
                    reasons['current_visibility_exclusion']+=1;continue
                native=geo.lookup('board.top' if r['role']=='board.top' else 'board.bottom')
                item=dict(r,native_path_id=native['id'],source_ref=dict(path=str(bp),sha256=bref['sha256'],row_index=j),
                    visible_span_id=r['segment_id'],source_BGR_sha256=f['source_BGR_sha256'])
                item['pixel_support']=dict(item['pixel_support'],query_index=j,normal=[0.,1.])
                by[native['id']].append(item)
            count=0
            for pid,rr in by.items():
                # The same source-line selector handles a visible projected board
                # edge without assigning it an ice geometry or changing its UV.
                keep,why=select_representatives(rr,dict(kind='ice_segment'))
                for j in keep:
                    r=rr[j];r['selection']=dict(reasons=sorted(why[j]),UV_unchanged=True)
                    chosen.append(r);dump(evidence,r);count+=1
            counts.append(dict(pts=pt,raw_rows=len(b['rows']),selected=count))
    pathids=m['path_ids']+list(dict.fromkeys(r['native_path_id']for r in chosen));lookup={p:i for i,p in enumerate(pathids)}
    new=dict(source_rows=np.arange(begin,begin+len(chosen),dtype=np.int64),
        observation_ids=np.array(['board:'+r['source_row_id']for r in chosen]),
        uv=np.array([r['uv']for r in chosen]),pts=np.array([r['native_pts']for r in chosen],np.int64),
        kind=np.ones(len(chosen),np.uint8),stations=np.full(len(chosen),.5),
        path_indices=np.array([lookup[r['native_path_id']]for r in chosen],np.int32))
    merged={k:np.concatenate([z[k],new[k]])for k in z};order=np.argsort(merged['pts'],kind='stable')
    np.savez_compressed(folder/'point_observations.npz',**{k:v[order]for k,v in merged.items()})
    m.update(path_ids=pathids,point_input_sha256=file_ref(folder/'point_observations.npz')['sha256']);m['source_row_namespace']['board']=[begin,begin+len(chosen)];write(folder/'mapping.json',m)
    result=read(OUT/'result.json');result.update(board_points=len(chosen),total_points=begin+len(chosen));write(OUT/'result.json',result)
    write(OUT/'evidence/board_coverage.json',dict(frames=counts,reasons=dict(reasons),
        top_definition='visible_blue_band_upper_edge',bottom_definition='actual_yellow_kickplate_lower_edge',
        source_domain='visible_vertical_board_face',native_geometry_changed=False))
    verify_authority();emit('board_source_merged',points=len(chosen),frames=793)


def reduce_curves():
    """Consolidate historical query fragments into one source-curve ordering."""
    geo=authority();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));m=read(folder/'mapping.json')
    source=OUT/'evidence/selected_ice_records.jsonl.gz'
    records=[r for _,r in rows(source)];by=collections.defaultdict(list)
    for r in records:by[r['native_pts'],r['native_path_id']].append(r)
    kept=set();reasons={};groups=[]
    for (pt,pid),rr in by.items():
        if geo.current_paths[pid]['kind']in ['ice_circle','ice_ellipse']:
            ii,why=select_representatives(rr,geo.current_paths[pid])
        else:ii=list(range(len(rr)));why={i:set(r['selection']['reasons'])for i,r in enumerate(rr)}
        for i in ii:kept.add(observation_key(rr[i]));reasons[observation_key(rr[i])]=sorted(why[i])
        groups.append(dict(pts=pt,native_path_id=pid,before=len(rr),after=len(ii)))
    keep=np.array([not str(k).startswith('ice:')or str(k)in kept for k in z['observation_ids']])
    np.savez_compressed(folder/'point_observations.npz',**{k:v[keep]for k,v in z.items()})
    with gzip.open(source,'wt',encoding='utf8',compresslevel=3)as out,gzip.open(OUT/'evidence/curve_reduction_delta.jsonl.gz','wt',encoding='utf8')as delta:
        for r in records:
            if observation_key(r)in kept:
                r['selection'].update(reasons=reasons[observation_key(r)],common_source_curve_ordering=True)
                dump(out,r)
            else:dump(delta,dict(source_id=r['source_row_id'],pts=r['native_pts'],reason='redundant_same_geometric_arc_support',source_ref=r['source_ref'],original_UV_retained_at_source=True))
    m['point_input_sha256']=file_ref(folder/'point_observations.npz')['sha256'];write(folder/'mapping.json',m)
    coverage=read(OUT/'evidence/ice_coverage.json');features=collections.Counter();frames=collections.Counter()
    for r in records:
        if observation_key(r)in kept:features[r['feature_id']]+=1;frames[r['native_pts']]+=1
    for r in coverage['frames']:r['selected']=frames[r['pts']]
    coverage['selected_by_feature']=dict(features);write(OUT/'evidence/ice_coverage.json',coverage)
    result=read(OUT/'result.json');result.update(ice_points=len(kept),total_points=int(keep.sum()),frames_with_ice=len(frames));write(OUT/'result.json',result)
    report=dict(dense_eligible_ice=result['eligible_dense_ice'],selected_ice=len(kept),
        reduction_fraction=1-len(kept)/result['eligible_dense_ice'],prior_fragment_representatives=len(records),
        visible_native_path_frame_pairs_before=len(groups),visible_native_path_frame_pairs_after=sum(g['after']>0 for g in groups),
        original_UV_PTS_ID_and_material_membership_unchanged=True,new_weights_sigma_or_loss_added=False,
        source_shape_tolerance='same ad selector: max(0.75px, 1% visible run diagonal), only for representative choice',groups=groups)
    write(OUT/'evidence/sampling_report.json',report);verify_authority();emit('curve_redundancy_removed',before=len(records),after=len(kept),**result)


def resolve_sources():
    """Incremental reconciliation of already selected historical C0 readers."""
    geo=authority();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));m=read(folder/'mapping.json')
    source=OUT/'evidence/selected_ice_records.jsonl.gz'
    records=[r for _,r in rows(source)]
    alternate_path=OUT/'evidence/C0_profile_alternatives.jsonl.gz'
    if not alternate_path.exists():
        wanted={r['source_row_id']for r in records if r['feature_id']=='ice.C0:outer'}
        pool=collections.defaultdict(list)
        cp=REC/'dense_ice_v1/canonical_v3/s04/rows.jsonl.gz'
        for line,r in rows(cp):
            if r['source_row_id']in wanted and 'inferred'not in r['source_kind']:
                pool[r['source_row_id']].append(dict(r,source_ref=dict(path=str(cp),jsonl_line_0based=line)))
        with gzip.open(alternate_path,'wt',encoding='utf8')as f:
            for rr in pool.values():
                if len(rr)>1:dump(f,rr)
    alternatives=[rr for _,rr in rows(alternate_path)]
    finite=FiniteSourceIdentity();excluded,_,withdrawn,_=corrections()
    visibility={r['pts']:r for r in read(ADS/'evidence/visibility.json')['frames']}
    by=collections.defaultdict(list)
    for rr in alternatives:by[rr[0]['native_pts']].extend(rr)
    conflicts=set();audit=[]
    for pt,rr in by.items():
        f=visibility[pt];im=cv2.imread(f['image']);mask=cv2.imread(f['mask_path'],0)
        board=read(REC/'dense_board_v1/qualified_frames'/f'{pt}.json')
        graphics,border,distance,_=source_domain(im,mask,board)
        eligible=[]
        for r in rr:
            reason=qualify(r,material_path(geo,r),excluded,withdrawn,graphics,border,distance)
            if reason is None:reason=finite.check(r)
            if reason is None:eligible.append(r)
        bad=competing_source_ids(eligible);conflicts.update(bad)
        audit.extend(dict(source_row_id=k,pts=pt,source_seconds=pt/12800,
            reason='unresolved_competing_actual_material_profiles_same_query',
            alternatives=[r for r in eligible if r['source_row_id']==k])for k in sorted(bad))
    ice_indices=np.flatnonzero(np.char.startswith(z['observation_ids'].astype(str),'ice:'))
    ice_indices=ice_indices[np.argsort(z['source_rows'][ice_indices])]
    if len(ice_indices)!=len(records):raise ValueError('Selected evidence count differs')
    ids=z['observation_ids'].astype(object);keep=np.ones(len(ids),bool);kept=[]
    with gzip.open(OUT/'evidence/source_conflicts.jsonl.gz','wt',encoding='utf8')as ledger:
        for item in audit:dump(ledger,item)
    for i,r in zip(ice_indices,records):
        if not np.array_equal(z['uv'][i],r['uv'])or z['pts'][i]!=r['native_pts']:
            raise ValueError('Source row ordering mismatch')
        ids[i]=observation_key(r)
        if r['source_row_id']in conflicts:keep[i]=False
        else:
            r['observation_key']=str(ids[i]);kept.append(r)
    z['observation_ids']=np.asarray(ids,dtype=str)
    if len(set(z['observation_ids'][keep]))!=int(keep.sum()):raise ValueError('Source keys remain duplicated')
    np.savez_compressed(folder/'point_observations.npz',**{k:v[keep]for k,v in z.items()})
    with gzip.open(source,'wt',encoding='utf8',compresslevel=3)as f:
        for r in kept:dump(f,r)
    m.update(point_input_sha256=file_ref(folder/'point_observations.npz')['sha256'],
        ice_observation_key='original source_row_id + immutable source file identity + source line',
        source_conflicts=str(OUT/'evidence/source_conflicts.jsonl.gz'))
    write(folder/'mapping.json',m)
    coverage=read(OUT/'evidence/ice_coverage.json');frames=collections.Counter(r['native_pts']for r in kept)
    for r in coverage['frames']:r['selected']=frames[r['pts']]
    coverage['selected_by_feature']=dict(collections.Counter(r['feature_id']for r in kept))
    coverage['unresolved_profile_queries']=len(conflicts)
    write(OUT/'evidence/ice_coverage.json',coverage)
    result=read(OUT/'result.json');result.update(ice_points=len(kept),total_points=int(keep.sum()),
        frames_with_ice=len(frames),unresolved_profile_queries=len(conflicts))
    write(OUT/'result.json',result)
    report=read(OUT/'evidence/sampling_report.json')
    report.update(selected_ice=len(kept),reduction_fraction=1-len(kept)/result['eligible_dense_ice'],
        unresolved_profile_queries=len(conflicts),selected_rows_withheld_for_competition=len(records)-len(kept),
        source_row_id_collisions_do_not_merge_distinct_original_readings=True)
    actual=collections.Counter((r['native_pts'],r['native_path_id'])for r in kept)
    for g in report['groups']:g['after']=actual[g['pts'],g['native_path_id']]
    report['visible_native_path_frame_pairs_after']=len(actual)
    write(OUT/'evidence/sampling_report.json',report)
    verify_authority();emit('source_identity_reconciled',**result)


def thin_spatial():
    """Remove close representatives with the same material normal, not turns."""
    verify_authority();folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));m=read(folder/'mapping.json')
    source=OUT/'evidence/selected_ice_records.jsonl.gz';records=[r for _,r in rows(source)]
    by=collections.defaultdict(list)
    for r in records:by[r['native_pts'],r['native_path_id']].append(r)
    retained=set();removed=[]
    for (pt,pid),rr in by.items():
        separation=max(2.5,.01*float(np.linalg.norm(np.ptp(np.asarray([r['uv']for r in rr]),axis=0))))
        local=[]
        for r in rr:
            q=np.asarray(r['uv']);n=np.asarray(r['pixel_support'].get('normal',[0.,0.]))
            prior=next((a for a in local if np.linalg.norm(q-a['uv'])<separation and
                np.dot(n,a['pixel_support'].get('normal',[0.,0.]))>.9),None)
            if prior is None:local.append(r);retained.add(observation_key(r))
            else:removed.append(dict(observation_key=observation_key(r),source_ref=r['source_ref'],
                pts=pt,native_path_id=pid,kept_neighbor=observation_key(prior),
                original_UV=r['uv'],reason='nearby_same_material_normal_redundant_representative',
                separation_px=separation,source_reading_not_deleted=True))
    keep=np.asarray([not str(k).startswith('ice:')or str(k)in retained for k in z['observation_ids']])
    np.savez_compressed(folder/'point_observations.npz',**{k:v[keep]for k,v in z.items()})
    kept=[r for r in records if observation_key(r)in retained]
    with gzip.open(source,'wt',encoding='utf8',compresslevel=3)as f:
        for r in kept:dump(f,r)
    with gzip.open(OUT/'evidence/spatial_reduction_delta.jsonl.gz','wt',encoding='utf8')as f:
        for r in removed:dump(f,r)
    m['point_input_sha256']=file_ref(folder/'point_observations.npz')['sha256'];write(folder/'mapping.json',m)
    counts=collections.Counter((r['native_pts'],r['native_path_id'])for r in kept)
    frames=collections.Counter(r['native_pts']for r in kept)
    if set(counts)!=set(by):raise ValueError('Density reduction removed a material/frame group')
    coverage=read(OUT/'evidence/ice_coverage.json')
    for r in coverage['frames']:r['selected']=frames[r['pts']]
    coverage['selected_by_feature']=dict(collections.Counter(r['feature_id']for r in kept));write(OUT/'evidence/ice_coverage.json',coverage)
    result=read(OUT/'result.json');result.update(ice_points=len(kept),total_points=int(keep.sum()));write(OUT/'result.json',result)
    report=read(OUT/'evidence/sampling_report.json')
    report.update(selected_ice=len(kept),reduction_fraction=1-len(kept)/result['eligible_dense_ice'],
        same_normal_spatial_redundancies_removed=len(removed),
        nearby_representative_rule='existing same-normal dot > 0.9; separation max(2.5px, existing 1% visible support diagonal)')
    for g in report['groups']:g['after']=counts[g['pts'],g['native_path_id']]
    write(OUT/'evidence/sampling_report.json',report)
    verify_authority();emit('same_normal_redundancy_removed',removed=len(removed),**result)


def render():
    """Render original points, their source-run connections and entity labels."""
    from apply_reviewed_s04_sources import apply as apply_reviewed_sources
    from review_points_s04 import main as connected_review
    directory,_=apply_reviewed_sources()
    connected_review(['--input-dir',str(directory)])


def initialize():
    """Use the current public initializer without any camera/geometry update."""
    import initialize_native_s01 as existing
    from rinkspace.core.native import NativeCore
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    verify_authority()
    folder=OUT/'points';before=file_ref(folder/'point_observations.npz')
    existing.prep.RUN=folder;existing.prep.CAMERA=PROG/'ad_cameras/s04.json'
    existing.prep.SCENE=SCENE;existing.prep.LIBRARY=LIBRARY
    existing.initialize()
    raw=dict(np.load(folder/'initialized_points.npz'))
    # The current point consumer accepts a station-only initializer. Preserve
    # optional search diagnostics separately rather than passing extra fields.
    np.save(folder/'seed_screen_distance.npy',raw['seed_screen_distance'])
    np.savez_compressed(folder/'initialized_points.npz',stations=raw['stations'])
    z,m,station=existing.prep.load_pixel_input(folder)
    core=NativeCore(LIBRARY);native=NativeSceneGPU(SCENE,core)
    camera=read(existing.prep.CAMERA);controls=np.load(existing.prep.CAMERA.parent/camera['controls_ref'])
    with native.resident(camera=camera,controls=controls,path_ids=[m['path_ids'][i]for i in z['path_indices']],stations=station,pts=z['pts'],observed=z['uv'])as resident:
        gpu=resident.evaluate('cuda');cpu=resident.evaluate('cpu')
        good=(gpu['status']==0)&(cpu['status']==0)
        diff=np.abs(gpu['values'][good,3:5]-cpu['values'][good,3:5])
        report=dict(rows=len(z['pts']),frames=len(np.unique(z['pts'])),
            point_input_sha256=before['sha256'],
            gpu_invalid=int(np.count_nonzero(gpu['status'])),cpu_invalid=int(np.count_nonzero(cpu['status'])),
            statuses_identical=bool(np.array_equal(gpu['status'],cpu['status'])),
            max_CPU_GPU_projection_difference_px=float(diff.max())if diff.size else None,
            GPU_timings=gpu['timings'],CPU_timings=cpu['timings'],allocation=resident.creation,
            source_input_unchanged=file_ref(folder/'point_observations.npz')==before,
            native_files_unchanged=True,camera_and_geometry_optimized=False,
            initialization_does_not_certify_observation_accuracy=True)
        failed=collections.defaultdict(list)
        with gzip.open(OUT/'evidence/native_integration_failures.jsonl.gz','wt',encoding='utf8')as f:
            for i in np.flatnonzero(~good):
                pid=m['path_ids'][z['path_indices'][i]];failed[pid].append(int(z['pts'][i]))
                dump(f,dict(observation_key=str(z['observation_ids'][i]),source_row=int(z['source_rows'][i]),
                    native_path_id=pid,source_pts=int(z['pts'][i]),original_UV=z['uv'][i].tolist(),
                    gpu_status=int(gpu['status'][i]),cpu_status=int(cpu['status'][i]),
                    input_row_retained=True))
        report['failed_paths']=[dict(native_path_id=pid,rows=len(pp),first_pts=min(pp),last_pts=max(pp),
            source_pts=sorted(set(pp)))for pid,pp in failed.items()]
    if not report['source_input_unchanged']:raise ValueError('Source input modified by initializer')
    if not np.array_equal(station[z['kind']==0],z['stations'][z['kind']==0]):raise ValueError('Fixed material corner moved')
    write(OUT/'evidence/native_integration.json',report)
    verify_authority();emit('native_integration',**report)


def verify():
    verify_authority()
    z=dict(np.load(OUT/'points/point_observations.npz',allow_pickle=False));m=read(OUT/'points/mapping.json')
    ad=dict(np.load(ADS/'ads_only/point_observations.npz',allow_pickle=False));n=len(ad['pts'])
    if set(z)!={'source_rows','observation_ids','uv','pts','kind','stations','path_indices'}:raise ValueError('Current input field set differs')
    if len(np.unique(z['observation_ids']))!=len(z['pts']):raise ValueError('Duplicate stable source identity')
    if len(np.unique(z['source_rows']))!=len(z['pts']):raise ValueError('Duplicate merged source row')
    if not np.array_equal(np.unique(z['pts']),np.arange(FIRST,LAST+1,STEP)):raise ValueError('Missing source exposure')
    order=np.argsort(z['source_rows']);ordered={k:v[order]for k,v in z.items()}
    for key in ['uv','pts','kind','stations','observation_ids','path_indices']:
        if not np.array_equal(ordered[key][:n],ad[key]):raise ValueError('Supplied advertisement altered: '+key)
    selected={};ice_lookup={str(key):i for i,key in enumerate(z['observation_ids'])if str(key).startswith('ice:')}
    selected_sources=collections.defaultdict(dict)
    for _,r in rows(OUT/'evidence/selected_ice_records.jsonl.gz'):
        key=observation_key(r);i=ice_lookup[key]
        if not np.array_equal(z['uv'][i],r['uv'])or z['pts'][i]!=r['native_pts']or m['path_ids'][z['path_indices'][i]]!=r['native_path_id']:
            raise ValueError('Ice source record and native input differ')
        selected_sources[r['source_ref']['path']][r['source_ref']['jsonl_line_0based']]=r
        selected[key]=r
    if len(selected)!=len(ice_lookup):raise ValueError('Ice evidence is incomplete')
    checked=0
    for path,wanted in selected_sources.items():
        for line,r in rows(path):
            if line not in wanted:continue
            q=wanted.pop(line)
            if r['source_row_id']!=q['source_row_id']or r['native_pts']!=q['native_pts']or r['uv']!=q['uv']:
                raise ValueError('Original source ID/UV/PTS changed in selected pool')
            checked+=1
        if wanted:raise ValueError('Selected original source line is missing')
    board_count=0;board_lookup={str(key):i for i,key in enumerate(z['observation_ids'])if str(key).startswith('board:')}
    cache={}
    for _,r in rows(OUT/'evidence/selected_board_records.jsonl.gz'):
        ref=r['source_ref'];path=ref['path']
        if path not in cache:cache[path]=read(path)
        original=cache[path]['rows'][ref['row_index']];i=board_lookup['board:'+r['source_row_id']]
        if original['uv']!=r['uv']or not np.array_equal(z['uv'][i],r['uv'])or z['pts'][i]!=r['native_pts']:
            raise ValueError('Board original source coordinates changed')
        board_count+=1
    if board_count!=len(board_lookup):raise ValueError('Board source evidence incomplete')
    report=dict(point_input_sha256=file_ref(OUT/'points/point_observations.npz')['sha256'],
        advertisement_rows_unchanged=n,ice_rows_source_verified=checked,board_rows_source_verified=board_count,
        frames=793,all_current_input_fields_valid=True,all_raw_UV_PTS_IDs_unchanged=True,
        exact_withdrawn_source_ids_admitted=0,only_current_native_paths=True,
        human_acceptance=False,native_authority_unchanged=True)
    withdrawn=read(OUT/'evidence/correction_sources.json')['withdrawals']
    if any(r['source_row_id']in withdrawn or r.get('source_parent_row_id')in withdrawn for r in selected.values()):
        raise ValueError('Withdrawn source identity reentered')
    geo=authority()
    if set(m['path_ids'])-geo.current_paths.keys():raise ValueError('Non-native path')
    write(OUT/'evidence/source_consistency.json',report);emit('source_verified',**report)


def deliver():
    """Package only current input and its traceability, without another model."""
    import zipfile
    verify_authority();geo=authority();folder=OUT/'points'
    z=dict(np.load(folder/'point_observations.npz'));m=read(folder/'mapping.json')
    point_hash=file_ref(folder/'point_observations.npz')['sha256']
    for p in [OUT/'review/video.json',OUT/'evidence/native_integration.json',OUT/'evidence/source_consistency.json']:
        if read(p)['point_input_sha256']!=point_hash:raise ValueError('Delivery contains a stale result: '+str(p))
    native=read(OUT/'evidence/native_integration.json');video=read(OUT/'review/video.json')
    if video['decoded_frames']!=793 or not video['native_PTS_equal']:raise ValueError('Incomplete video')
    entities=[]
    for i,pid in enumerate(m['path_ids']):
        r=geo.current_paths[pid];ix=z['path_indices']==i
        entities.append(dict(native_path_id=pid,label=r['label'],kind=r['kind'],measurement=r.get('measurement'),
            selected_rows=int(ix.sum()),source_frames=int(len(np.unique(z['pts'][ix])))))
    write(OUT/'evidence/native_entities.json',dict(native_authority=str(OUT/'evidence/native_authority.json'),entities=entities))
    coverage=read(OUT/'evidence/ice_coverage.json');sampling=read(OUT/'evidence/sampling_report.json')
    conflicts=[r for _,r in rows(OUT/'evidence/source_conflicts.jsonl.gz')]
    missing=[dict(source_pts=r['pts'],source_seconds=r['source_seconds'])for r in coverage['frames']if r['selected']==0]
    delivery=dict(schema='rinkspace.s04_source_delivery.v1',**read(OUT/'result.json'),
        source_scope=dict(first_pts=FIRST,last_pts=LAST,stop_pts_exclusive=LAST+STEP,step_pts=STEP,
            time_base='1/12800',source_seconds=[FIRST/12800,LAST/12800],independent_exposures=793),
        current_input=dict(directory=str(folder),loader='prepare_native_s01.load_pixel_input(directory)',
            fields=sorted(z),raw=file_ref(folder/'point_observations.npz'),mapping=file_ref(folder/'mapping.json'),
            separate_station_seed=file_ref(folder/'initialized_points.npz'),
            station_initialization='Only curve station seeds; fixed material corners unchanged. No camera or geometry update.'),
        video=dict(video,player='PotPlayer',colors='stable per current native entity; labels shown in frame and legend'),
        original_sources_modified=False,native_authority=file_ref(OUT/'evidence/native_authority.json'),
        sampling=dict(dense_eligible=sampling['dense_eligible_ice'],selected=sampling['selected_ice'],
            redundancy_reduction_fraction=sampling['reduction_fraction'],
            retained_material_frame_groups=sampling['visible_native_path_frame_pairs_after'],
            likelihood_weights_or_sigma_added=False),
        limits=dict(ice_frames_without_qualified_reads=missing,
            C0_ambiguous_queries=[dict(source_pts=r['pts'],source_row_id=r['source_row_id'])for r in conflicts],
            C0_note='同一原始搜索位置出现两个材料分支，未能唯一消歧；保留双方原始证据，不重复纳入输入。',
            initial_native_evaluation_failures=native['failed_paths'],
            native_status_8='当前相机初值下的 optical domain 失败；行全部保留，未通过删点或改 native 消除。',
            visible_ice_completeness_certified=False,user_visual_acceptance='pending'),
        validation=dict(source_consistency=file_ref(OUT/'evidence/source_consistency.json'),
            native_integration=file_ref(OUT/'evidence/native_integration.json'),
            representative_source_frames_inspected=True,all_frames_visually_accepted=False),
        source_implementation=[file_ref(Path(__file__)),file_ref(PROG/'initialize_native_s01.py')],
        external_dependencies=dict(native_scene=str(SCENE),native_library=str(LIBRARY),
            current_camera=m['camera_path'],source_video=m['source_video'],
            full_historical_evidence=str(REC),advertisement_original_package=str(ADS)))
    write(OUT/'delivery.json',delivery)
    files=[OUT/'delivery.json']
    files += [folder/name for name in ['point_observations.npz','mapping.json','initialized_points.npz','station_initialization.json']]
    files += [OUT/'evidence'/name for name in ['selected_ice_records.jsonl.gz','selected_board_records.jsonl.gz',
        'source_inventory.json','correction_sources.json','source_consistency.json','native_entities.json',
        'native_authority.json','native_integration.json','native_integration_failures.jsonl.gz',
        'ice_coverage.json','board_coverage.json','sampling_report.json','source_conflicts.jsonl.gz',
        'spatial_reduction_delta.jsonl.gz','finite_RGB_reread.json','temporal_support_gaps.json']]
    files += [OUT/'review'/name for name in ['video.json','frame_index.jsonl','connected_video.json','connected_frame_index.jsonl.gz']]
    archive=OUT/'S04_native_points_20261027.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5)as f:
        for path in files:f.write(path,str(path.relative_to(OUT)),compress_type=zipfile.ZIP_STORED if path.suffix in ['.gz','.npz']else zipfile.ZIP_DEFLATED)
        for name in ['HANDOFF.md','ads_only/mapping.json','evidence/selected_point_records.jsonl.gz',
                     'evidence/source_inventory.json','evidence/source_consistency.json','evidence/coverage_summary.json']:
            f.write(ADS/name,'supplied_ads/'+name)
        for path in [Path(__file__),PROG/'initialize_native_s01.py',PROG/'review_points_s04.py',ROOT/'src/rinkspace/review/point_connections.py']:
            f.write(path,'implementation/'+path.name)
    with zipfile.ZipFile(archive)as f:
        if f.testzip() is not None:raise ValueError('Archive verification failed')
    write(OUT/'package.json',dict(file_ref(archive),bytes=archive.stat().st_size,
        point_input_sha256=point_hash,video_external_path=video['path'],original_video_not_copied=True))
    verify_authority();emit('s04_delivery_ready',archive=str(archive),video=video['path'],points=len(z['pts']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['navigation','build','boards','reduce_curves','resolve_sources','thin_spatial','render','initialize','verify','deliver'])
    args=parser.parse_args()
    {'navigation':navigation,'build':build,'boards':boards,'reduce_curves':reduce_curves,'resolve_sources':resolve_sources,'thin_spatial':thin_spatial,'render':render,'initialize':initialize,'verify':verify,'deliver':deliver}[args.stage]()
