"""Replay independent native finite-material identity before any ice restoration.

Annotations identify physical entities; pixels, finite bounds and held-out
registration replay that identity at the immutable observation. A local gray
edge, a scene projection and a JSON confirmation flag cannot grant membership.
"""
from pathlib import Path
import cv2
import numpy as np
from .io_utils import read_json
from .dense_joint_proofs import _checked, _ref


def physical_identity_gate_sources():
    from . import native_print_track_readout,source_worker,source_seed_correspondence_v1
    return [_ref(p)for p in (__file__,native_print_track_readout.__file__,source_worker.__file__,source_seed_correspondence_v1.__file__)]


def preserved_negative_rows(qualification, explicit=()):
    """Retain every individual negative revision in the qualification ancestry."""
    negatives=set(map(int,explicit));seen=set();todo=[Path(qualification).resolve()]
    while todo:
        path=todo.pop()
        if path in seen:continue
        seen.add(path);doc=read_json(path)
        if doc.get('revision'):
            revpath=_checked(doc['revision'],path.parent);rev=read_json(revpath)
            negatives.update(map(int,rev.get('withheld_rows',())))
        for key in ('parent_qualification','parent_source_qualification'):
            if doc.get(key):todo.append(_checked(doc[key],path.parent))
        for key in ('qualification_delta','recovery'):
            if doc.get(key):
                delta=_checked(doc[key],path.parent);value=read_json(delta)
                if value.get('parent_source_qualification'):
                    todo.append(_checked(value['parent_source_qualification'],delta.parent))
    return sorted(negatives)


def _checked_cached(ref,root,cache):
    key=(str(ref['file']),ref['sha256'])
    if key not in cache:cache[key]=_checked(ref,root)
    return cache[key]


def _source_feature(annotation,entity,image_ref,pts,checked):
    source=read_json(checked(annotation));src=source.get('source',{})
    features=[f for f in source.get('features',[])if f.get('target_id',f.get('id'))==entity]
    if len(features)!=1 or features[0].get('allowed_for_fit')is not True:
        raise ValueError('Physical identity anchor is absent or explicitly unqualified')
    feature=features[0]
    state=' '.join(str(feature.get(k,''))for k in ('status','qualification','physical_instance_qualification')).lower()
    if any(s in state for s in ('waiting','pending','unknown','rejected','withheld','non_material','candidate')):
        raise ValueError('Unresolved or negative physical source annotation cannot be restored')
    if int(src.get('native_pts',-1))!=int(pts)or src.get('png_sha256')!=image_ref['sha256']:
        raise ValueError('Physical identity source image/PTS mismatch')
    if src.get('image_path')and Path(src['image_path']).resolve()!=checked(image_ref).resolve():
        raise ValueError('Physical source annotation points to a different image')
    checked(image_ref)
    if feature.get('projected_uv_used_as_observations')is True:
        raise ValueError('Projected source pixels cannot certify physical identity')
    return source,feature


def _literal_uv(feature,uv):
    return any(np.array_equal(np.asarray(p,float),np.asarray(uv,float))for s in feature.get('segments',[])for p in s.get('uv',[]))


def _image(ref,checked):
    image=cv2.imread(str(checked(ref)),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Actual native RGB image is unavailable')
    return image


def _finite_material_replay(proof,feature,image,entity,annotation,image_ref,pts):
    """Check actual finite boundary and signed material-side samples, not flags.

    The independent source review supplies the named identity and material order.
    This consumer replays its literal finite contour and original pixel samples;
    it cannot turn generic contrast or a navigation ellipse into that review.
    """
    if(proof.get('schema')!='rinkfit.native_finite_material_source_identity.v1'
            or proof.get('source_annotation')!=annotation or proof.get('source_image')!=image_ref
            or proof.get('physical_entity')!=entity or proof.get('native_pts')!=pts
            or proof.get('identity_uses_model_projection')is not False
            or proof.get('identity_uses_model_residual')is not False
            or proof.get('physical_identity_authority')!='independent_original_native_finite_material_review'
            or not proof.get('named_entity_source_basis')or not proof.get('observed_material_order')):
        raise ValueError('Independent named finite-material source review is missing')
    boundary=proof.get('finite_boundary_uv',[])
    if len(boundary)<2 or not all(_literal_uv(feature,p)for p in boundary):
        raise ValueError('Positive identity needs the literal finite native material boundary')
    samples=proof.get('actual_signed_material_samples',[])
    if not samples:raise ValueError('Independent physical source review has no replayable native samples')
    # Reuse the actual production signed-interface reader, including its finite
    # profile validity conditions. Stored RGB values alone do not prove a boundary.
    from .native_print_track_readout import batch_interfaces
    from .source_worker import _channels
    queries=np.asarray([q['query_uv']for q in samples],float)
    normals=np.asarray([q['profile_normal']for q in samples],float)
    boundary_points=np.asarray([q['boundary_uv']for q in samples],float)
    profiles=np.asarray([q['source_profile']for q in samples],float)
    if(queries.shape!=normals.shape or queries.shape!=boundary_points.shape or queries.shape[1:]!=(2,)
            or not np.isfinite(queries).all()or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals,axis=1),1.,atol=1e-6,rtol=0)):
        raise ValueError('Invalid source signed-interface replay coordinates')
    actual_profile=batch_interfaces(queries,normals,_channels(image[:,:,::-1]))
    if(not np.all(actual_profile['valid'])or profiles.shape!=actual_profile['profile'].shape
            or not np.allclose(actual_profile['uv'],boundary_points,atol=1e-6,rtol=0)
            or not np.allclose(actual_profile['profile'],profiles,atol=1e-7,rtol=0)):
        raise ValueError('Actual finite source signed-interface measurement cannot be reproduced')
    h,w=image.shape[:2]
    for sample in samples:
        if not _literal_uv(feature,sample['boundary_uv']):raise ValueError('Material sample is outside the confirmed finite entity')
        points=np.asarray([sample['inside_uv'],sample['outside_uv']],float)
        if points.shape!=(2,2)or not np.isfinite(points).all():raise ValueError('Invalid finite source material samples')
        pixels=np.rint(points).astype(int)
        if np.any(pixels<0)or np.any(pixels[:,0]>=w)or np.any(pixels[:,1]>=h):raise ValueError('Material sample is outside source image')
        actual=image[pixels[:,1],pixels[:,0]]
        expected=np.asarray(sample['actual_BGR'])
        if(expected.shape!=(2,3)or not np.isfinite(expected).all()or np.any(expected<0)or np.any(expected>255)
                or not np.array_equal(actual,expected)or np.array_equal(actual[0],actual[1])):
            raise ValueError('Native material-side RGB replay differs or has no material transition')


def _registration_replay(support,anchor_image,current_image,source_uv,current_uv):
    """Recompute H from other actual source points, with held-out finite support."""
    from .source_seed_correspondence_v1 import correlation
    pairs=support.get('other_source_correspondences',[])
    if len(pairs)<8:raise ValueError('Held-out source identity needs independent 2D correspondence support')
    src=np.asarray([q['source_uv']for q in pairs],float);dst=np.asarray([q['current_uv']for q in pairs],float)
    if (src.shape!=dst.shape or src.shape[1:]!=(2,)or not np.isfinite(src).all()or not np.isfinite(dst).all()
            or any(np.array_equal(p,source_uv)for p in src)or len({tuple(p)for p in src})!=len(src)):
        raise ValueError('Registration identity support is duplicate or includes held-out source point')
    H,inside=cv2.findHomography(src,dst,cv2.RANSAC,3.,maxIters=5000,confidence=.999)
    if H is None or inside is None or int(inside.sum())<8:raise ValueError('Actual source registration failed')
    good=inside.ravel().astype(bool);hull=cv2.convexHull(src[good].astype(np.float32)).reshape(-1,2)
    points=src[good];den=np.c_[hull,np.ones(len(hull))]@H[2]
    if(np.ptp(points,axis=0)[0]<40 or np.ptp(points,axis=0)[1]<10 or cv2.contourArea(hull)<150
            or min(den)*max(den)<=0 or np.linalg.det(H)/den.mean()**3<=0):
        raise ValueError('Source identity registration has no finite supporting footprint')
    mapped=cv2.perspectiveTransform(src[good][None],H)[0]
    if np.quantile(np.linalg.norm(mapped-dst[good],axis=1),.95)>3.:raise ValueError('Source registration replay p95 exceeds original limit')
    aq=np.asarray(source_uv,float);uv=np.asarray(current_uv,float)
    prediction=cv2.perspectiveTransform(aq[None,None],H)[0,0]
    inverse=cv2.perspectiveTransform(uv[None,None],np.linalg.inv(H))[0,0]
    if(min(cv2.pointPolygonTest(hull,tuple(map(float,aq)),True),cv2.pointPolygonTest(hull,tuple(map(float,inverse)),True))<0
            or np.linalg.norm(prediction-uv)>3.):raise ValueError('Held-out source point has no current finite registration support')
    ag=cv2.cvtColor(anchor_image,cv2.COLOR_BGR2GRAY);cg=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
    checks=[aq,*src[good]]
    scores=[correlation(ag,cg,H,q)for q in checks]
    if any(not np.isfinite(c)or c<.65 for c in scores):raise ValueError('Actual RGB does not reproduce the source registration')


def _positive_native_identity(row,local_proof,root,cache):
    checked=lambda ref:_checked_cached(ref,root,cache)
    entity='ice.'+local_proof['source_row_id'].rsplit('/',2)[-2]
    if(row['physical_entity']!=entity or row.get('measurement_definition')!=local_proof['definition']
            or row.get('physical_surface')!='ice'or row.get('physical_identity_confirmed')is not True
            or row.get('identity_uses_model_projection')is not False or row.get('identity_uses_model_residual')is not False):
        raise ValueError('Physical identity, original measurement definition or source authority changed')
    from .circle_source_identity import circle_name, require_c0_midline_reading
    circular = circle_name(entity)
    if circular in {'A1','A2','A3','A4'}:
        raise ValueError('A1-A4 are removed from current fitting; physical recovery cannot re-admit them')
    if circular == 'C0':
        require_c0_midline_reading(row, label=entity, uv=row['uv'])
    annotation=row['positive_source_annotation'];imref=row['positive_source_image'];pts=row['positive_source_native_pts']
    source,feature=_source_feature(annotation,entity,imref,pts,checked)
    if feature.get('user_confirmed')is not True:
        if(feature.get('source_entity_identity_confirmed')is not True
                or feature.get('physical_identity_authority')not in ('independent_original_native_finite_material_review','original_native_finite_material_review','source_registered_confirmed_finite_material')
                or feature.get('identity_navigation_origin')or feature.get('joint_group_navigation')
                or not row.get('positive_finite_material_review')):
            raise ValueError('Named physical source identity requires positive finite-material evidence')
        proof=read_json(checked(row['positive_finite_material_review']))
        _finite_material_replay(proof,feature,_image(imref,checked),entity,annotation,imref,pts)
    support=read_json(checked(row['current_source_correspondence']))
    if(support.get('schema')!='rinkfit.current_native_finite_entity_correspondence.v1'
            or support.get('identity_uses_model_projection')is not False or support.get('identity_uses_model_residual')is not False
            or support.get('current_static_material_visible')is not True):raise ValueError('Independent actual-current finite source correspondence required')
    for key in ('raw_row','source_row_id','uv','native_pts','physical_entity','material_index','path'):
        if support.get(key)!=row[key]:raise ValueError('Current source correspondence differs from immutable row '+key)
    evidence=support.get('source_evidence',[])
    if not evidence:raise ValueError('Actual current source evidence is missing')
    for ref in evidence:checked(ref)
    method=support.get('source_identity_method')
    if method=='same_actual_exposure_finite_material_review':
        if pts!=row['native_pts']or not _literal_uv(feature,row['uv']):
            raise ValueError('Same-exposure identity needs the literal confirmed finite source point')
    elif method=='held_out_source_image_registration_on_confirmed_finite_entity':
        if not _literal_uv(feature,support['positive_source_uv']):raise ValueError('Registration anchor is not a literal confirmed finite material point')
        current_ref=support['current_source_image'];current_annotation=read_json(checked(support['current_source_frame']))
        current_source=current_annotation.get('source',current_annotation)
        if(int(current_source.get('native_pts',-1))!=row['native_pts']or current_source.get('png_sha256')!=current_ref['sha256']):
            raise ValueError('Registration current native image/PTS mismatch')
        _registration_replay(support,_image(imref,checked),_image(current_ref,checked),support['positive_source_uv'],row['uv'])
    else:raise ValueError('Navigation and temporal continuity cannot grant a physical identity')



def selected_packet_source_ids(doc, packet_root, wanted):
    """Read only requested compact raw IDs; never hydrate all suffix strings.

    This is an offline source-qualification consumer. Historical parents are
    consulted only if this producer's input is a delta lacking the requested
    identity; the normal solver never calls this source-review code.
    """
    requested = set(map(int, wanted)); pending = set(requested); result = {}
    if any(i < 0 or i >= int(doc['row_count']) for i in requested): raise ValueError('Requested source ID outside raw row domain')
    root = Path(packet_root); cache = {}; visited = set()
    while pending:
        prefix = doc.get('source_row_ids', ())
        for i in list(pending):
            if i < len(prefix):
                result[i] = prefix[i]; pending.remove(i)
        registries = ([doc.get('dense_source_binding', {})]
                      + list(doc.get('native_ad_source_bindings', []))
                      + [doc.get('native_ad_source_binding', {})])
        for reg in registries:
            if not reg: continue
            start = int(reg['row_start']); count = int(reg['count'])
            selected = sorted(i for i in pending if start <= i < start+count)
            if not selected: continue
            off = reg['source_id_offsets']; blob = reg['source_id_utf8']
            op = _checked_cached(off, root, cache); bp = _checked_cached(blob, root, cache)
            if off.get('dtype') != '<u8' or off.get('shape') != [count+1] or op.stat().st_size != 8*(count+1):
                raise ValueError('Compact source-ID offsets layout differs')
            offsets = np.memmap(op, dtype='<u8', mode='r', shape=(count+1,))
            size = bp.stat().st_size
            with bp.open('rb') as stream:
                for i in selected:
                    j = i-start; lo, hi = int(offsets[j]), int(offsets[j+1])
                    if not 0 <= lo < hi <= size: raise ValueError('Compact source-ID byte range differs')
                    stream.seek(lo); result[i] = stream.read(hi-lo).decode('utf8'); pending.remove(i)
        if not pending: break
        if not doc.get('dense_parent_packet'): raise ValueError('Missing requested immutable source IDs')
        parent = doc['dense_parent_packet']; path = _checked_cached(parent, root, cache)
        key = (str(path.resolve()), parent['sha256'])
        if key in visited: raise ValueError('Cyclic offline source-ID lineage')
        visited.add(key); doc = read_json(path); root = path.parent
    return result

def require_physical_recovery_review(review,packet_ref,wanted,doc,arrays,local_proofs,*,source_ids=None):
    if review is None:raise ValueError('Separate positive physical source-identity review is required for ice recovery')
    root=Path(review).parent;value=read_json(review)
    if(value.get('schema')!='rinkfit.native_static_material_identity_review.v1'or value.get('source_packet')!=packet_ref
            or value.get('identity_uses_model_projection')is not False or value.get('identity_uses_model_residual')is not False):
        raise ValueError('Physical identity review must bind this exact source packet independently of model')
    expected=set(map(int,wanted));seen=set();cache={}
    for row in value.get('rows',[]):
        i=int(row['raw_row'])
        if i not in expected or i in seen:raise ValueError('Physical recovery identity membership differs or duplicates rows')
        if(row['source_row_id']!=(source_ids[i] if source_ids is not None else doc['source_row_ids'][i])or row['uv']!=arrays['uv'][i].tolist()
                or row['native_pts']!=int(arrays['native_pts'][i])or row['material_index']!=int(arrays['material_index'][i])
                or row['path']!=int(arrays['path'][i])):raise ValueError('Physical recovery changed original UV/ID/PTS/material/path')
        _positive_native_identity(row,local_proofs[i],root,cache);seen.add(i)
    if seen!=expected:raise ValueError('Every restored row needs independent positive physical identity')
    return _ref(review)
