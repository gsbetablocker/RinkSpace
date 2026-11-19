"""Native programme skate evidence: roles and shared source dependencies."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from .absolute_motion_state import chain_exposure
from .compact_body_derivatives import CompactBodyLinearization
from .fullbody_fit import right_jacobian, skew
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import digest
from .skate_physics import source_support_weight_s
from .spin_grasp_factors import region_terms
from .weighted_blocks import block, signature


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def native_row(row, raw_id, track, event, foot, semantic, source_sha256):
    return dict(row, raw_group_id=raw_id, track_id=track, event_id=event, foot=foot,
                semantic=semantic, observation_status='observed' if row['role'] != 'unknown' else 'ambiguous',
                visibility=row.get('visibility', 'partial'), pixel_domain='native Brown 1920x1080',
                source_sha256=source_sha256)


def merge_document(source, landmarks, plane, contours, *, camera_sha256, programme_pts, time_base_value):
    """Append a source revision; preserve every prior source row and detector UV."""
    doc = load_json(source)
    if 'programme_skate_evidence' in doc:
        raise ValueError('Expected unmodified parent source document')
    raw = load_json(landmarks)
    if list(raw['time_base']) != list(time_base_value):
        raise ValueError('Manual evidence uses a different source clock')
    mark_sha = digest(landmarks)
    manual = []
    for r in raw['manual_leg_landmarks']:
        foot = int(r['joint'] in (14, 16))
        event = 'steps_extension' if r['group']=='step' else 'lsp'
        manual.append(native_row(r, f"manual:{r['pts']}:{r['joint']}",
            f"{event}:leg:{foot}", event, foot, 'anatomical_landmark_region', mark_sha))
    bands = []
    for key, phase in [('entry_runner_bands', 'entry'), ('left_runner_ice_interface_bands', 'rotation')]:
        for r in raw[key]:
            bands.append(native_row(r, f"support:{r['pts']}:{r['foot']}", 'lsp:left:support', 'lsp',
                r['foot'] if r['role'] != 'unknown' else None, 'support_neighborhood', mark_sha)
                | dict(phase=phase, foot_candidates=[0, 1] if r['role']=='unknown' else [r['foot']]))
    raw_plane = load_json(plane)
    depth = {r['pts']:r for r in bands if r['phase']=='rotation'}
    if len(depth) != len(raw_plane) or {r['pts'] for r in raw_plane} != set(depth):
        raise ValueError('Conditional plane rows do not share the raw band clock')
    derived = []
    for r in raw_plane:
        parent = depth[r['pts']]
        if parent['role'] != r['role']:
            raise ValueError('Conditional coordinates changed raw role')
        # Unknown row may have only the source stamp and a diagnostic note.
        derived.append(dict(r, raw_group_id=parent['raw_group_id'],
            depends_on=[parent['raw_group_id']], origin='inferred',
            camera_sha256=camera_sha256, ice_plane='Z=0 conditional mapping',
            role=parent['role'], optimization_enabled=False))
    observation_doc = load_json(contours)
    if (observation_doc['time_base'] != list(time_base_value)
            or Path(observation_doc['source_video']) != Path(raw['source_video'])):
        raise ValueError('New skate contours changed original source/clock')
    ordered = [native_row(r, r['raw_group_id'], r['track_id'], r['event_id'], r['foot'],
                          r['semantic'], digest(contours)) for r in observation_doc['rows']]
    evidence = dict(schema='rinkspace.programme-skate-evidence.v1',
        parent_source=binding(source), manual_source=binding(landmarks), plane_source=binding(plane),
        ordered_source=binding(contours), camera_sha256=camera_sha256, time_base=list(time_base_value),
        source_video=raw['source_video'], source_video_sha256=observation_doc['source_video_sha256'],
        manual_landmarks=manual, native_support_bands=bands, plane_xy_diagnostics=derived,
        ordered_contours=ordered, events=observation_doc['events'],
        exposure_review=observation_doc['exposure_review'],
        support_height_policy='Unchanged finite ice and source episode contact modes; derived XY has no independent residual.',
        checks='Development checks in the same action; not independent accuracy validation.',
        weights=dict(manual_landmark=8., ordered_contour=8., native_support=2.))
    validate_evidence(evidence, programme_pts, time_base_value, camera_sha256)
    doc['programme_skate_evidence'] = evidence
    return doc


def extend_document(source, review, *, camera_sha256, programme_pts, time_base_value):
    """Add a source review to an already merged programme ledger."""
    document = load_json(source)
    evidence = document['programme_skate_evidence']
    validate_evidence(evidence, programme_pts, time_base_value, camera_sha256)
    amendment = load_json(review)
    if (amendment['schema'] != 'rinkspace.programme-skate-amendment.v1'
            or amendment['time_base'] != evidence['time_base']
            or amendment['source_video_sha256'] != evidence['source_video_sha256']):
        raise ValueError('Source amendment refers to a different video or native clock')
    events = {event['id']: event for event in amendment['events']}
    current = {event['id'] for event in evidence['events']}
    if not events or len(events) != len(amendment['events']) or current.intersection(events):
        raise ValueError('Source amendment must declare distinct new event identities')
    stamps = set(map(int, programme_pts))
    for event in events.values():
        if (event['start_pts'] not in stamps or event['end_pts'] not in stamps
                or event['start_pts'] >= event['end_pts']
                or not event['active_feet'] or not set(event['active_feet']) <= {0, 1}):
            raise ValueError('Invalid source amendment event: '+event['id'])
    review_binding = binding(review)
    for key in ('manual_landmarks', 'ordered_contours'):
        for row in amendment[key]:
            event = events[row['event_id']]
            if not event['start_pts'] <= row['pts'] <= event['end_pts']:
                raise ValueError('Source row outside its reviewed event: '+row['raw_group_id'])
            semantic = row.get('semantic', 'anatomical_landmark_region')
            evidence[key].append(native_row(row, row['raw_group_id'], row['track_id'],
                row['event_id'], row['foot'], semantic, review_binding['sha256']))
    evidence['events'].extend(amendment['events'])
    evidence['exposure_review'].extend(amendment['exposure_review'])
    evidence.setdefault('amendments', []).append(dict(review=review_binding, previous_ledger=binding(source),
        event_ids=list(events), method_zh=amendment['method_zh']))
    tracked=amendment.get('tracked_sequence_policy')
    if tracked is not None:
        if tracked != dict(schema='rinkspace.skate-tracked-sequence.v1', max_anchor_offset_s=.42,
                           boot_axes_enabled=True, legacy_source_guard=True):
            raise ValueError('Invalid tracked skate policy')
        evidence['tracked_sequence_policy']=tracked
        validate_tracked_anchors(document,amendment['ordered_contours'],events)
    validate_evidence(evidence, programme_pts, time_base_value, camera_sha256)
    return document


def validate_tracked_anchors(document,rows,events):
    feature={'boot_sole':'outsole','front_stanchion':'front_mount','rear_stanchion':'rear_mount',
             'runner_lower_edge':'runner_lower_edge','runner_upper_edge':'runner_upper_edge'}
    current=document['programme_skate_evidence']
    archive={r['id']:r for r in document.get('full_source_skate_structures',{}).get('rows',[])
             if r['role']=='fit' and r.get('optimization_enabled',True)}
    ordered={r['raw_group_id']:r for r in current['ordered_contours']}
    manual={r['raw_group_id']:r for r in rows}
    for row in rows:
        proof=row.get('provenance',{})
        seed_id=proof.get('track_source_anchor_id')
        if seed_id is None:
            continue
        kind=proof['track_source_anchor_kind']
        seed={'legacy':archive,'ordered':ordered,'manual':manual}.get(kind,{}).get(seed_id)
        event=events[row['event_id']]
        luma=proof.get('source_luma_sha256','')
        if (seed is None or not event['start_pts']<=seed['pts']<=event['end_pts']
                or not 0<abs(row['pts']-seed['pts'])*current['time_base'][0]/current['time_base'][1]<=.420000001
                or any(row[k]!=seed[k] for k in ('foot','role'))
                or row['semantic']!=(feature.get(seed.get('structure_feature')) if kind=='legacy' else seed['semantic'])
                or seed.get('provenance',{}).get('track_source_anchor_id') or row['complete_visible']
                or len(luma)!=64 or any(c not in '0123456789abcdef' for c in luma)):
            raise ValueError('Invalid tracked skate source dependency: '+row['raw_group_id'])


def programme_observation_path(prepared, source):
    """Resolve a current reviewed detector correspondence without replacing its parent."""
    original = Path(prepared) / 'observations.npz'
    document = load_json(source) if not isinstance(source, dict) else source
    revision = document.get('programme_body_observation_revision')
    if revision is None:
        return original
    current = Path(revision['path'])
    original_sha = digest(original)
    visited = set()
    while revision is not None:
        path = Path(revision['path'])
        resolved = path.resolve()
        if resolved in visited or digest(path) != revision['sha256']:
            break
        visited.add(resolved)
        # Some prepared bundles already include an earlier revision. Bind the
        # active chain up to that exact ledger, then leave its archive intact.
        if resolved == original.resolve() or revision['parent_observations_sha256'] == original_sha:
            return current
        parent = revision.get('previous_revision')
        if parent is None or parent['sha256'] != revision['parent_observations_sha256']:
            break
        revision = parent
    raise ValueError('Body source observation revision differs from its bound ledger')


def validate_evidence(evidence, pts, tb, camera_sha256):
    if evidence['camera_sha256'] != camera_sha256 or list(tb) != evidence['time_base']:
        raise ValueError('Skate evidence stale camera or clock; regenerate plane diagnostics from native pixels')
    stamps = set(map(int, pts))
    groups = {}
    tracks = {}
    event_ids = {e['id'] for e in evidence['events']}
    for field in ('manual_landmarks', 'native_support_bands', 'ordered_contours'):
        for r in evidence[field]:
            gid, tid, role = (r[k] for k in ('raw_group_id','track_id','role'))
            if r['pts'] not in stamps or role not in ('fit','check','unknown') or r['event_id'] not in event_ids:
                raise ValueError('Invalid native observation identity: '+gid)
            if gid in groups:
                raise ValueError('Repeated raw evidence: '+gid)
            groups[gid] = r
            if r['foot'] not in (0,1) and role != 'unknown':
                raise ValueError('Unassigned skate cannot enter an anatomical fit')
            if r['foot'] is not None:
                identity = r['foot'], r['event_id']
                if tracks.setdefault(tid, identity) != identity:
                    raise ValueError('Physical foot identity changes within track: '+tid)
            if field=='ordered_contours':
                uv = np.asarray(r['polyline_uv'],float)
                if (uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or not np.isfinite(uv).all()
                        or np.any(np.linalg.norm(np.diff(uv,axis=0),axis=-1)<1)
                        or r['radius_px']<=0 or not r['visible_range']):
                    raise ValueError('Invalid observed contour: '+gid)
                strength=r.get('relative_weight',1.)
                if (isinstance(strength,bool) or not isinstance(strength,(int,float))
                        or not np.isfinite(strength) or not 0<strength<=1.):
                    raise ValueError('Invalid declared relative source weight: '+gid)
                if r['source_order']=='image_trace' and (r['complete_visible'] or r.get('structural_endpoints')):
                    raise ValueError('Image trace needs a partial visible arc with no anatomical endpoints: '+gid)
                if r['complete_visible'] and r['visibility']!='complete':
                    raise ValueError('Cannot enable reverse extent for unobserved span: '+gid)
    for r in evidence['ordered_contours']:
        dependency=r.get('provenance',{}).get('depends_on_track_anchor')
        if dependency is None:
            continue
        anchor=groups.get(dependency)
        if (anchor is None or anchor.get('provenance',{}).get('depends_on_track_anchor')
                or any(anchor.get(k)!=r[k] for k in ('track_id','event_id','foot','semantic','role'))
                or not 0<abs(anchor['pts']-r['pts'])*tb[0]/tb[1]<=.100000001):
            raise ValueError('Invalid source-tracking dependency: '+r['raw_group_id'])
    for r in evidence['plane_xy_diagnostics']:
        gid = r['raw_group_id']
        if (gid not in groups or r['depends_on']!=[gid] or r['camera_sha256']!=camera_sha256
                or r['role']!=groups[gid]['role'] or r['pts']!=groups[gid]['pts'] or r['optimization_enabled']):
            raise ValueError('Conditional plane cache has an invalid dependency or is double weighted')
    return groups


def exposure_times(problem, frames):
    frames = np.asarray(frames, int)
    camera = problem.camera
    tau = camera.readout_times(frames) if hasattr(camera,'readout_times') else np.full(len(frames), problem.body.tau)
    return (problem.reference.times[frames,None] + np.linspace(-.5,.5,5)*tau[:,None]).ravel()


def native_point_response(error, radius, weights, *, cauchy=None):
    """Pixel residual and derivative; detector outliers may use Cauchy loss.

    The default retains the existing reviewed-point response. Cauchy has the
    same small-error scale but reduces the influence of gross detector errors,
    including joints hallucinated on clothing during source occlusion.
    """
    distance2=np.sum(error*error,axis=1);scale2=9*radius*radius
    t=distance2/scale2
    gain=(1+t)**-.25;radial=-1/(2*(scale2+distance2))
    if cauchy is not None and np.any(cauchy):
        q=t[cauchy];small=q<1.e-6
        g=np.empty_like(q);v=np.empty_like(q)
        g[small]=1-q[small]/4+13*q[small]**2/96
        v[small]=(-.5+5*q[small]/12)/scale2[cauchy][small]
        qn=q[~small];log=np.log1p(qn)
        g[~small]=np.sqrt(log/qn)
        v[~small]=(qn/((1+qn)*log)-1)/distance2[cauchy][~small]
        gain[cauchy]=g;radial[cauchy]=v
    w=weights/radius*gain
    derivative=w[:,None,None]*(np.eye(2)+radial[:,None,None]*error[:,:,None]*error[:,None,:])
    return w[:,None]*error,derivative


class ManualNativeBodyFactors:
    """Separate reviewed source marks on the same native articulated body."""
    name = 'manual_native_body'
    allowed_joints = tuple(range(17))
    def __init__(self, problem, rows):
        self.problem, self.rows = problem, rows
        ref = problem.reference
        stamps = np.array([r['pts'] for r in rows],dtype=np.int64)
        unique, self.owner = np.unique(stamps, return_inverse=True)
        self.frames = np.searchsorted(ref.p['pts'], unique)
        np.testing.assert_array_equal(ref.p['pts'][self.frames], unique)
        self.joints = np.array([r['joint'] for r in rows],int)
        if not np.isin(self.joints, self.allowed_joints).all():
            raise ValueError('Unsupported anatomical COCO joint in '+self.name)
        self.times = exposure_times(problem, self.frames)
        self.base = ref.at(self.times) if len(rows) else None
        self.uv = np.array([r['uv'] for r in rows],float).reshape(-1,2)
        self.is_region = np.array(['pixel_box' in r for r in rows],bool)
        self.pixel_boxes = np.array([r.get('pixel_box',[[-np.inf,-np.inf],[np.inf,np.inf]])
                                    for r in rows],float).reshape(-1,2,2)
        if np.any(self.pixel_boxes[:,0]>self.pixel_boxes[:,1]):
            raise ValueError('Invalid source image region')
        self.radius = np.array([r['radius_px'] for r in rows],float)
        self.weights = np.sqrt(np.array([source_support_weight_s(r) for r in rows]))
        self.cauchy = np.array([r.get('loss')=='cauchy' for r in rows],bool)
        self.identity = signature(stamps,self.joints,self.uv,self.radius,self.is_region,self.pixel_boxes)
        if self.cauchy.any():self.identity=signature(self.identity,self.cauchy)

    def evaluate(self, x, jacobian=False):
        p = self.problem
        if not self.rows:
            J = csr_matrix((0,p.size)) if jacobian else None
            return dict(residual=np.empty(0),jac=J,parts={self.name:0.},
                        metrics={self.name:[]},weighted_blocks=[block(self.name,[],J)])
        d = p.body.state.increments(x,self.times)
        local = np.asarray(d[:,6:]).reshape(-1,21,3)
        linear = CompactBodyLinearization(p.reference.body,self.base['local_R'],local,
                                          hand_local_R=self.base.get('hand_local_R'))
        from .fullbody_state import matrices
        Q = self.base['Q'] @ matrices(d[:,3:6])
        rotated = np.einsum('nij,nkj->nki',Q,linear.coco)
        world = rotated + (self.base['P'] + d[:,:3])[:,None]
        pr = moving_projection(p.camera,world.reshape(len(self.frames),5,17,3),self.frames,jacobian=jacobian)
        uv = pr['uv'][self.owner,self.joints]
        err = (uv-self.uv)
        # A reviewed occlusion region supplies exclusion evidence only. Its
        # centre is not an invented observation of the hidden joint.
        response_axes=np.ones_like(err)
        if self.is_region.any():
            m=self.is_region;lo=self.pixel_boxes[m,0];hi=self.pixel_boxes[m,1]
            err[m]=uv[m]-np.clip(uv[m],lo,hi)
            response_axes[m]=(uv[m]<lo)|(uv[m]>hi)
        distance = np.linalg.norm(err,axis=1)
        # A smooth scale for a reviewed joint region; weak (not zero)
        # interior attraction, bounded influence outside ~3 click radii.
        response,S=native_point_response(err,self.radius,self.weights,cauchy=self.cauchy)
        residual = response.ravel()
        J = None
        if jacobian:
            raw = pr['pixel_jac_nodes'][self.owner,self.joints]
            sample = self.owner[:,None]*5+np.arange(5)
            j = self.joints[:,None]
            X = np.zeros((len(self.rows),5,3,69))
            X[...,:3] = np.eye(3)
            X[...,3:6] = -skew(rotated[sample,j]) @ (Q@right_jacobian(d[:,3:6]))[sample]
            axes=getattr(p,'active_body_pose_channels',range(6,69))
            for axis in axes:
                if axis<6:continue
                der = linear.derivative(int(axis),kinematics=False)['coco']
                X[...,axis] = np.einsum('nmij,nmj->nmi',Q[sample],der[sample,j])
            raw_local = np.einsum('nplc,nlck->nplk',raw,X)
            S*=response_axes[:,None,:]
            J = chain_exposure(np.einsum('nij,njlk->nilk',S,raw_local),d,sample_indices=sample)
            J = hstack((csr_matrix((len(residual),17)),J),format='csr')
        metrics = [dict(id=r['raw_group_id'],pts=r['pts'],foot=r['foot'],event_id=r['event_id'],
                         joint=r['joint'],role=r['role'],radius_px=r['radius_px'],
                         distance_px=float(e),prediction=u.tolist(),
                         source_uv=None if 'pixel_box' in r else r['uv'],
                         pixel_box=r.get('pixel_box'))
                   for r,e,u in zip(self.rows,distance,uv)]
        b = block(self.name,residual,J,dict(source_pts=[r['pts'] for r in self.rows],
                   raw_group_ids=[r['raw_group_id'] for r in self.rows],unit='px',role='fit'),self.identity)
        return dict(residual=residual,jac=J,parts={b.name:float(residual@residual/2)},
                    metrics={b.name:metrics},weighted_blocks=[b])


class ManualNativeLegFactors(ManualNativeBodyFactors):
    """Backward-compatible knee/ankle evidence and objective identity."""
    name = 'manual_native_leg'
    allowed_joints = (13,14,15,16)


class ProgrammeSkateSourceFactors:
    """Fit and development checks share the projector, never the fit registry."""
    def __init__(self, problem, evidence, *, role='fit'):
        from .body_clock import time_base
        from .source_contact_image import NativeSupportContactFactors
        from .ordered_native_skate import OrderedNativeSkateFactors
        validate_evidence(evidence,problem.reference.p['pts'],time_base(problem.reference.p),
                          problem.camera.info['state_sha256'])
        self.problem,self.role=problem,role
        self.manual=ManualNativeLegFactors(problem,[r for r in evidence['manual_landmarks'] if r['role']==role])
        self.ordered=OrderedNativeSkateFactors(problem,[r for r in evidence['ordered_contours'] if r['role']==role])
        self.bands=[r for r in evidence['native_support_bands'] if r['role']==role]
        self.support=None
        if self.bands and problem.reference.ice_plane_known:
            self.support=NativeSupportContactFactors(problem,native_bands=[
                dict(r,objective_weight=source_support_weight_s(r)) for r in self.bands])

    def evaluate(self,x,jacobian=False):
        from scipy.sparse import vstack
        values=[self.manual.evaluate(x,jacobian),self.ordered.evaluate(x,jacobian)]
        if self.support is not None:
            v=self.support.evaluate(x,jacobian)
            rv=v['residual'];J=v['jac']; name='reviewed_native_support'
            rows=[dict(m,id=r['raw_group_id'],event_id=r['event_id'],phase=r['phase'],role=r['role'])
                  for m,r in zip(v['metrics']['native_source_support_contacts'],self.bands)]
            b=block(name,rv,J,dict(source_pts=[r['pts'] for r in self.bands],
                                   raw_group_ids=[r['raw_group_id'] for r in self.bands],unit='px'),
                    self.support.association_signature)
            values.append(dict(residual=rv,jac=J,parts={name:float(rv@rv/2)},
                               metrics={name:rows},weighted_blocks=[b]))
        return dict(residual=np.concatenate([v['residual'] for v in values]),
                    jac=vstack([v['jac'] for v in values],format='csr') if jacobian else None,
                    parts={k:c for v in values for k,c in v['parts'].items()},
                    metrics={k:c for v in values for k,c in v['metrics'].items()},
                    weighted_blocks=[b for v in values for b in v['weighted_blocks']])
