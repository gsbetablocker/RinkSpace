"""Actual grey-to-lighter near-board q50 at explicit source key positions.

Source y keys and x domains navigate the original material profiles only. All
profile/quantile/contrast/guard/direct/weak decisions retain the original reader.
Returned points still need independent source association and fit eligibility.
The original source-qualified reader remains unchanged for historical evidence.
"""
from .vision_near_board_qualified import cv2, np


def profiles(b,m,bb,guard,*,query_ys,query_x_bounds=(0.,1919.),actual_x_bounds=(0.,1919.)):
 query_ys=np.asarray(query_ys)
 if query_ys.ndim!=1 or query_ys.dtype.kind not in 'iuf' or not np.isfinite(query_ys).all():
  raise ValueError('query_ys must be explicit finite one-dimensional real source y key positions')
 query_ys=query_ys.astype(np.float64,copy=False)
 query_x_bounds=np.asarray(query_x_bounds,dtype=np.float64);actual_x_bounds=np.asarray(actual_x_bounds,dtype=np.float64)
 if (query_x_bounds.shape!=(2,) or actual_x_bounds.shape!=(2,) or not np.isfinite(query_x_bounds).all() or not np.isfinite(actual_x_bounds).all()
     or query_x_bounds[0]>query_x_bounds[1] or actual_x_bounds[0]>actual_x_bounds[1]):
  raise ValueError('Source query/candidate x bounds must be finite ordered pairs')
 g=cv2.GaussianBlur(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.8);n=np.array([1.,-m]);n/=np.linalg.norm(n);ds=np.arange(-70.,70.001,.5);pts=[];unknown=[]
 for y in query_ys:
  x=m*y+bb
  if x < query_x_bounds[0] or x>query_x_bounds[1]:continue # observer anchor: if x < -45 or x>680:continue
  xy=np.array([x,y])+ds[:,None]*n;xx=np.broadcast_to(xy[:,0][None,:],(5,len(ds))).astype(np.float32);yy=(xy[:,1][None,:]+np.arange(-2,3)[:,None]).astype(np.float32)
  v=cv2.remap(g,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(0);gradient=np.gradient(v);pk=np.flatnonzero((gradient[1:-1]>=gradient[:-2])&(gradient[1:-1]>=gradient[2:]))+1;opts=[]
  for j in pk:
   if abs(ds[j])>52 or j<56 or j+56>=len(ds)or gradient[j]<.25:continue
   uv0=np.array([x,y])+ds[j]*n
   left_extent=min(25.,(uv0[0]-1)/n[0],(1076-uv0[1])/max(1e-6,-n[1]));
   if left_extent<5:continue
   plate=np.array([uv0-left_extent*n,uv0+25*n])
   if np.any(plate[:,0]<0)or np.any(plate[:,0]>1919)or np.any(plate[:,1]<3)or np.any(plate[:,1]>1076):continue
   if uv0[0]<actual_x_bounds[0] or uv0[0]>actual_x_bounds[1] or uv0[1]<270 or uv0[1]>1076:continue
   lc=int(left_extent*2);lo=float(np.median(v[j-lc:j-max(5,lc//2)]));hi=float(np.median(v[j+22:j+50]));contrast=hi-lo
   if not(6<contrast and 50<lo<155 and hi>135):continue
   if lo>=145 and contrast<25:continue # weak near-white ice gradient does not prove gray cap material
   cs=[]
   for q in(.1,.5,.9):
    target=lo+q*contrast;cross=[k for k in range(j-50,j+50)if v[k]<=target<v[k+1]]
    if not cross:break
    k=min(cross,key=lambda k:abs(k-j));cs.append(float(ds[k]+(target-v[k])/(v[k+1]-v[k])*.5))
   if len(cs)!=3 or not(cs[0]<=cs[1]<=cs[2])or cs[2]-cs[0]>40:continue
   uv=np.array([x,y])+cs[1]*n;a=np.array([x,y])+cs[0]*n;z=np.array([x,y])+cs[2]*n
   # Complete source transition support must lie in the actual source frame, and outside persons/UI/unknown.
   sam=np.array([a+(z-a)*tt for tt in np.linspace(0,1,max(3,int(np.linalg.norm(z-a))*2+1))]);iv=np.round(sam).astype(int)
   if np.any(iv[:,0]<0)or np.any(iv[:,0]>=1920)or np.any(iv[:,1]<0)or np.any(iv[:,1]>=1080):continue
   if np.any(guard[iv[:,1],iv[:,0]]):continue
   plate_support=np.stack([uv0+tt*n for tt in np.arange(-left_extent,25.01,.5)]);pi=np.round(plate_support).astype(int)
   if np.any(guard[pi[:,1],pi[:,0]]):continue
   quality=abs(cs[1])/14 + max(0,14-contrast)/8
   opts.append((quality,dict(uv=uv.tolist(),normal=n.tolist(),interval=[a.tolist(),z.tolist()],contrast=contrast,gray_plateaus=[lo,hi],normal_transition_width=float(cs[2]-cs[0]),direct=contrast>=12 and cs[2]-cs[0]<=24 and left_extent>=12,material_plateau_support_endpoints_uv=plate.tolist(),left_plateau_extent=left_extent)))
  if opts:pts.append(min(opts,key=lambda x:x[0])[1])
  else:unknown.append(dict(navigation_y=y,navigation_x=x,reason='no_complete_current_gray_to_lighter_material_transition_or_occluded'))
 return pts,unknown


def _source_key_parameter_interval(point, normal, tangent, hw, filter_radius):
    """Intersect the normal line with all four borders of its full filter support."""
    height, width = hw
    inset = filter_radius + 2*np.abs(tangent)
    low = inset
    high = np.array([width-1., height-1.])-inset
    begin, end = -np.inf, np.inf
    for position, direction, lower, upper in zip(point, normal, low, high):
        if lower > upper:
            return None
        if direction == 0:
            if not lower <= position <= upper:
                return None
        else:
            a, b = (lower-position)/direction, (upper-position)/direction
            begin, end = max(begin, min(a, b)), min(end, max(a, b))
    return (begin, end) if begin <= end else None


def profiles_at_source_keys(bgr, guard, *, query_uv, iceward_normals):
    """Read gray-to-lighter q50 candidates along explicit source-space normals.

    query_uv is an N x 2 search array. iceward_normals is N x 2 or one shared
    unit vector, pointing from gray material toward lighter ice. These inputs
    navigate sampling and never confer physical identity. The return is
    (points, unknown), like profiles; every new point remains pending.

    Material, quantile, contrast, direct/weak and candidate-ranking rules retain
    the original reader's values. The five taps follow the true unit tangent.
    No source padding is admitted: out-of-frame or guarded search samples stay
    unavailable, and every used plateau/transition sample must have all five
    taps, bilinear neighbours and the original sigma=.8 filter footprint clear.
    Shorter visible gray plateaus use analytic four-border intersections and
    retain the original minimum extent and direct/weak rules.
    """
    bgr = np.asarray(bgr)
    guard = np.asarray(guard)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8 or not all(bgr.shape[:2]):
        raise ValueError('bgr must be a nonempty H x W x 3 original uint8 source image')
    if guard.shape != bgr.shape[:2] or guard.dtype.kind not in 'biu':
        raise ValueError('guard must be a boolean or integer mask of the exact source shape')
    queries = np.asarray(query_uv)
    normals = np.asarray(iceward_normals)
    if queries.size == 0 and queries.ndim == 1:
        queries = queries.reshape(0, 2)
    if queries.ndim != 2 or queries.shape[1] != 2 or queries.dtype.kind not in 'iuf' or not np.isfinite(queries).all():
        raise ValueError('query_uv must be finite real source coordinates with shape N x 2')
    queries = queries.astype(np.float64, copy=False)
    if normals.shape == (2,):
        normals = np.broadcast_to(normals, queries.shape)
    elif not len(queries) and normals.size == 0 and normals.ndim == 1:
        normals = normals.reshape(0, 2)
    if normals.shape != queries.shape or normals.dtype.kind not in 'iuf' or not np.isfinite(normals).all():
        raise ValueError('iceward_normals must be finite real unit vectors with shape N x 2 or 2')
    normals = normals.astype(np.float64, copy=False)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isclose(lengths, 1., rtol=1e-6, atol=0.).all():
        raise ValueError('iceward_normals must be unit vectors')
    normals = normals/lengths[:, None]  # Remove unit-vector roundoff only.
    if not len(queries):
        return [], []

    # The original float-gray sigma=.8 kernel, now with explicit finite support.
    filter_radius = 3
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (7, 7), .8)
    excluded = cv2.dilate((guard != 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    height, width = guard.shape

    def clear_samples(coordinates):
        # Check the actual float32 OpenCV locations and both bilinear neighbours.
        with np.errstate(over='ignore', invalid='ignore'):
            coordinates = np.asarray(coordinates, dtype=np.float32)
        valid = (np.isfinite(coordinates).all(-1)
                 & (coordinates[..., 0] >= filter_radius)
                 & (coordinates[..., 0] <= width-1-filter_radius)
                 & (coordinates[..., 1] >= filter_radius)
                 & (coordinates[..., 1] <= height-1-filter_radius))
        clear = np.zeros(valid.shape, bool)
        xy = coordinates[valid]
        if len(xy):
            low, high = np.floor(xy).astype(int), np.ceil(xy).astype(int)
            clear[valid] = ~(excluded[low[:, 1], low[:, 0]]
                             | excluded[low[:, 1], high[:, 0]]
                             | excluded[high[:, 1], low[:, 0]]
                             | excluded[high[:, 1], high[:, 0]])
        return clear, coordinates

    ds = np.arange(-70., 70.001, .5)
    taps = np.arange(-2., 3.)
    points, unknown = [], []
    for query_index, (query, n) in enumerate(zip(queries, normals)):
        tangent = np.array([-n[1], n[0]])
        coordinates = query + ds[None, :, None]*n + taps[:, None, None]*tangent
        clear, coordinates = clear_samples(coordinates)
        samples = np.full(clear.shape, np.nan, np.float32)
        if clear.any():
            samples[clear] = cv2.remap(g, coordinates[..., 0][clear][None],
                                      coordinates[..., 1][clear][None], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)[0]
        v = samples.mean(0)
        gradient = np.gradient(v)
        pk = np.flatnonzero((gradient[1:-1] >= gradient[:-2]) & (gradient[1:-1] >= gradient[2:]))+1
        options = []
        for j in pk:
            if abs(ds[j]) > 52 or j < 56 or j+56 >= len(ds) or gradient[j] < .25:
                continue
            uv0 = query + ds[j]*n
            domain = _source_key_parameter_interval(uv0, n, tangent, guard.shape, filter_radius)
            if domain is None or not domain[0] <= 0 <= domain[1] or domain[1] < 25:
                continue
            left_extent = float(min(25., -domain[0]))
            if left_extent < 5:
                continue
            plate = np.array([uv0-left_extent*n, uv0+25*n])
            lc = int(left_extent*2)
            lo = float(np.median(v[j-lc:j-max(5, lc//2)]))
            hi = float(np.median(v[j+22:j+50]))
            contrast = hi-lo
            if not (6 < contrast and 50 < lo < 155 and hi > 135):
                continue
            if lo >= 145 and contrast < 25:
                continue
            cs = []
            for q in (.1, .5, .9):
                target = lo+q*contrast
                cross = [k for k in range(j-50, j+50) if v[k] <= target < v[k+1]]
                if not cross:
                    break
                k = min(cross, key=lambda k: abs(k-j))
                cs.append(float(ds[k]+(target-v[k])/(v[k+1]-v[k])*.5))
            if len(cs) != 3 or not (cs[0] <= cs[1] <= cs[2]) or cs[2]-cs[0] > 40:
                continue
            uv = query+cs[1]*n
            a, z = query+cs[0]*n, query+cs[2]*n
            transition = np.array([a+(z-a)*tt for tt in np.linspace(0, 1, max(3, int(np.linalg.norm(z-a))*2+1))])
            plate_support = uv0 + np.r_[np.arange(-left_extent, 25.01, .5), 25.][:, None]*n
            footprint = np.concatenate([transition, plate_support])[None]+taps[:, None, None]*tangent
            if not clear_samples(footprint)[0].all():
                continue
            quality = abs(cs[1])/14 + max(0, 14-contrast)/8
            options.append((quality, dict(uv=uv.tolist(), normal=n.tolist(), interval=[a.tolist(), z.tolist()],
                contrast=contrast, gray_plateaus=[lo, hi], normal_transition_width=float(cs[2]-cs[0]),
                direct=contrast >= 12 and cs[2]-cs[0] <= 24 and left_extent >= 12,
                material_plateau_support_endpoints_uv=plate.tolist(), left_plateau_extent=left_extent,
                source_query_index=query_index, query_uv=query.tolist(), tangent=tangent.tolist(),
                physical_identity_qualified=False, fit_eligible=False,
                source_identity_status='pending_independent_source_association')))
        if options:
            points.append(min(options, key=lambda option: option[0])[1])
        else:
            unknown.append(dict(source_query_index=query_index, query_uv=query.tolist(),
                iceward_normal=n.tolist(), reason='no_complete_current_gray_to_lighter_material_transition_or_occluded'))
    return points, unknown
