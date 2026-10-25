"""Replay the existing official-source near-board v6 material reader.

The retained generation binds navigation, masks and identities to exact source
pixels. Its fitted/interpolated curves are navigation or negative exclusions;
all positive positions still come from this RGB's material profiles.
Original v6 SHA256: 924394cdaf9136f438f87b7620f3043ccce2c4562aacadeb998413057f5eccc2.
"""
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import json
import cv2
import numpy as np


def verified_bytes(reference):
    path = Path(reference.get('file', reference.get('path')))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Changed source-qualified near-board evidence: '+str(path))
    return raw


@lru_cache(maxsize=4)
def qualified_index(path, sha256):
    manifest = json.loads(verified_bytes(dict(path=path, sha256=sha256)))
    if not manifest.get('accepted_for_final_source_qualification'):
        raise ValueError('Near-board generation is not source-qualified')
    rows = [json.loads(s) for s in gzip.decompress(verified_bytes(manifest['index'])).splitlines()]
    if len(rows) != manifest['frames'] or len({r['native_pts'] for r in rows}) != len(rows):
        raise ValueError('Near-board native exposure inventory differs')
    return manifest, {r['native_pts']:r for r in rows}


def mask_image(reference, hw):
    mask = cv2.imdecode(np.frombuffer(verified_bytes(reference), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != hw:
        raise ValueError('Near-board native mask shape differs')
    return mask > 0


# Source-only functions retained verbatim from the existing first-shot reader.
def broadcast_exclusion(bgr):
    """Read this exposure's actual timer rectangle; font holes are excluded."""
    out=np.zeros(bgr.shape[:2],bool)
    out[25:105,25:557]=True
    white=(bgr[920:1020,1650:1900].min(2)>216).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(white,8)
    valid=[]
    for st in stats[1:]:
        x,y,w,h,area=map(int,st)
        if area<600 or not 55<=w<=230 or not 10<=h<=70:
            continue
        rect=bgr[920+y:920+y+h,1650+x:1650+x+w]
        if int((rect.max(2)<110).sum())<40:
            continue
        valid.append(st)
    rectangle=None
    if valid:
        st=max(valid,key=lambda s:int(s[4]))
        x,y,w,h=map(int,st[:4]);x+=1650;y+=920
        rectangle=[x,y,x+w-1,y+h-1]
        timer=np.zeros(out.shape,np.uint8)
        timer[y:y+h,x:x+w]=1
        out|=cv2.dilate(timer,np.ones((9,9),np.uint8))>0
    return out,dict(actual_native_rectangle_xyxy_inclusive=rectangle,edge_halo_px=4,
        method="current-native white timer component rectangle with dark text, no historical timer mask",
        current_source_timer_present=rectangle is not None,
        applied_to_source_readouts_and_line_visibility=True)



def current_broadcast_exclusion(bgr):
 mask,legacy=broadcast_exclusion(bgr)
 # The official broadcast timer stays in this source rectangle throughout
 # this shot. Its translucent grey-white background and grey digits can fail
 # both of the older absolute white/black component thresholds.
 rectangle=[1692,944,1864,1002]
 gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(float)
 body=gray[948:997,1697:1859];glyph=gray[956:994,1720:1848]
 background=float(np.quantile(body,.85));dark=glyph<min(190.,background-22)
 dark_count=int(dark.sum());yy,xx=np.nonzero(dark)
 span=[int(np.ptp(xx))+1,int(np.ptp(yy))+1] if len(xx) else [0,0]
 present=dark_count>=100 and span[0]>=55 and span[1]>=16
 if present:
  x0,y0,x1,y1=rectangle;mask[y0-4:y1+5,x0-4:x1+5]=True
 return mask,dict(actual_native_rectangle_xyxy_inclusive=rectangle if present else legacy['actual_native_rectangle_xyxy_inclusive'],
   current_source_timer_present=present or legacy['current_source_timer_present'],edge_halo_px=4,
   method='same-shot fixed source timer region confirmed by current grey digit mass and rectangle background; union with legacy components',
   current_source_glyph_dark_pixel_count=dark_count,current_source_glyph_extent_px=span,current_source_timer_background_q85=background,
   legacy_component_detector=legacy,applied_to_source_readouts_and_line_visibility=True)


def current_graphics(bgr):
 graphics,timer=current_broadcast_exclusion(bgr)
 # This source's technical-score panel below the two permanent score rows.
 # The source coordinates only locate the layout; current colored header,
 # green execution field and neutral body must all be present.
 header=bgr[94:124,72:395].astype(float);sat=header.max(2)-header.min(2)
 header_fraction=float(((sat>55)&(header.max(2)>100)).mean())
 execution=bgr[187:220,234:396].astype(float)
 green_fraction=float(((execution[:,:,1]>execution[:,:,0]+30)&(execution[:,:,1]>execution[:,:,2]+30)&(execution[:,:,1]>120)).mean())
 body=bgr[125:185,72:395].astype(float)
 neutral_fraction=float(((body.min(2)>155)&(body.max(2)-body.min(2)<55)).mean())
 panel=header_fraction>.25 and green_fraction>.15 and neutral_fraction>.35
 if panel:graphics[90:257,60:400]=True
 return graphics,dict(timer=timer,technical_panel=dict(present=bool(panel),header_saturated_fraction=header_fraction,execution_green_fraction=green_fraction,body_neutral_fraction=neutral_fraction,source_layout_roi_only=True,rectangle_xyxy=[60,90,400,257]if panel else None))


def profiles(b,m,bb,guard):
 g=cv2.GaussianBlur(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.8);n=np.array([1.,-m]);n/=np.linalg.norm(n);ds=np.arange(-70.,70.001,.5);pts=[];unknown=[]
 for y in range(270,1079,6):
  x=m*y+bb
  if x < -45 or x>680:continue
  xy=np.array([x,y])+ds[:,None]*n;xx=np.broadcast_to(xy[:,0][None,:],(5,len(ds))).astype(np.float32);yy=(xy[:,1][None,:]+np.arange(-2,3)[:,None]).astype(np.float32)
  v=cv2.remap(g,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean(0);gradient=np.gradient(v);pk=np.flatnonzero((gradient[1:-1]>=gradient[:-2])&(gradient[1:-1]>=gradient[2:]))+1;opts=[]
  for j in pk:
   if abs(ds[j])>52 or j<56 or j+56>=len(ds)or gradient[j]<.25:continue
   uv0=np.array([x,y])+ds[j]*n
   left_extent=min(25.,(uv0[0]-1)/n[0],(1076-uv0[1])/max(1e-6,-n[1]));
   if left_extent<5:continue
   plate=np.array([uv0-left_extent*n,uv0+25*n])
   if np.any(plate[:,0]<0)or np.any(plate[:,0]>1919)or np.any(plate[:,1]<3)or np.any(plate[:,1]>1076):continue
   if uv0[0]<4 or uv0[0]>650 or uv0[1]<270 or uv0[1]>1076:continue
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

def reread_qualified_near_board(bgr, binding, reference, *, person_mask=None):
    if reference is None:
        return None
    manifest, index = qualified_index(reference['path'], reference['sha256'])
    source = manifest['source']
    if (binding['source_sha256'] != source['sha256'] or binding['time_base'] != source['time_base']
            or binding['source_pts'] not in index):
        return None
    if (not isinstance(binding['source_pts'], int) or bgr.shape != (1080,1920,3)
            or bgr.dtype != np.uint8):
        raise ValueError('Exact native near-board RGB/PTS required')
    entry=index[binding['source_pts']]
    row=json.loads(verified_bytes(dict(file=entry['frame_file'],sha256=entry['frame_sha256'])))
    if (row['native_pts']!=binding['source_pts'] or row['time_base']!=binding['time_base']
            or str(binding.get('shot_id',row['shot_id'])).lower()!=row['shot_id'].lower()
            or hashlib.sha256(bgr.tobytes()).hexdigest()!=row['source_BGR_sha256']
            or hashlib.sha256(np.ascontiguousarray(bgr[...,::-1]).tobytes()).hexdigest()!=row['source_RGB_sha256']):
        raise ValueError('Qualified near-board evidence belongs to different native source pixels')
    hw=bgr.shape[:2]
    person=mask_image(row['current_person_mask_ref'],hw)
    if person_mask is not None:
        extra=np.asarray(person_mask,dtype=bool)
        if extra.shape!=hw:raise ValueError('Current person mask is not native-sized')
        person |= extra
    guard=cv2.dilate(person.astype('uint8'),np.ones((17,17),np.uint8))>0
    unknown_ref=row.get('current_source_unknown_exclusion_mask_ref')
    if unknown_ref:
        guard |= cv2.dilate(mask_image(unknown_ref,hw).astype('uint8'),np.ones((17,17),np.uint8))>0
    graphics,graphics_receipt=current_graphics(bgr)
    guard |= graphics
    nav=row['source_navigation'];m,bb=nav['m'],nav['b']
    pts,unknown=profiles(bgr,m,bb,guard) if nav['bottom']>-45 else ([],[])
    # Same current-pixel re-navigation and rejection steps as retained v6.
    if len(pts)>=8:
        v=np.array([q['uv'] for q in pts]);fit=np.polyfit(v[:,1],v[:,0],1)
        res=v[:,0]-np.polyval(fit,v[:,1])
        keep=np.abs(res-np.median(res))<max(4,2*np.median(np.abs(res-np.median(res))))
        if keep.sum()>=6:
            fit=np.polyfit(v[keep,1],v[keep,0],1)
            if .2<fit[0]<.85:pts,unknown=profiles(bgr,float(fit[0]),float(fit[1]),guard)
    if len(pts)>=4:
        v=np.array([q['uv'] for q in pts]);fit=np.polyfit(v[:,1],v[:,0],1)
        res=v[:,0]-np.polyval(fit,v[:,1]);mad=float(np.median(np.abs(res-np.median(res))))
        keep=np.abs(res-np.median(res))<max(5,3*mad)
        pts=[q for q,k in zip(pts,keep) if k]
    near=mask_image(row['near_outside_exclusion_mask_ref'],hw)
    face=mask_image(row['board_body_visible_mask_ref'],hw) & ~guard & ~near
    ice=mask_image(row['ice_visible_mask_ref'],hw) & ~guard & ~near
    # Exact source evidence already exists for these same decoded bytes. Preserve
    # its UV and original row identity; a different OpenCV build must not replace it.
    fresh_pts=pts
    preserved=[];row_ids=[]
    for original in row['new_source_rows']:
        support=original['pixel_support'];clear=True
        for endpoints in (support['position_interval_endpoints_uv'],support['material_plateau_support_endpoints_uv']):
            a,z=np.asarray(endpoints,float)
            ij=np.round(a[None,:]+np.linspace(0,1,max(3,int(np.linalg.norm(z-a))*2+1))[:,None]*(z-a)).astype(int)
            if guard[ij[:,1],ij[:,0]].any():clear=False;break
        if not clear:continue
        preserved.append(dict(uv=original['uv'],normal=support['normal'],
            interval=support['position_interval_endpoints_uv'],contrast=support['material_contrast'],
            gray_plateaus=support['gray_plateaus'],direct=not original['weak_source'],
            material_plateau_support_endpoints_uv=support['material_plateau_support_endpoints_uv'],
            left_plateau_extent=support['left_plateau_extent_native_px']))
        row_ids.append(original['source_row_id'])
    pts=preserved
    uv=np.array([q['uv'] for q in pts],float).reshape(-1,2)
    normals=np.array([q['normal'] for q in pts],float).reshape(-1,2)
    endpoints=np.array([q['interval'] for q in pts],float).reshape(-1,2,2)
    support=np.einsum('nki,ni->nk',endpoints-uv[:,None,:],normals)
    spans=[];start=0
    for stop in range(1,len(uv)+1):
        if stop==len(uv) or uv[stop,1]-uv[stop-1,1]>9:
            spans.append(dict(actual_source_uv=uv[start:stop],source_indices=np.arange(start,stop),segment_id=len(spans)))
            start=stop
    reading=dict(kind='board_top',source_board_view='top_face_only',material='grey_to_ice',
        actual_source_uv=uv,iceward_normals=normals,normal_interval_px=support,
        actual_source_material_profiles=pts,unknown_rows=unknown,measured_runs=spans,
        source_row_ids=row_ids,source_uv_origin='preserved_exact_native_source_generation',
        fresh_current_RGB_profile_diagnostic=fresh_pts,fresh_read_replaces_existing_UV=False,
        profile_runtime=dict(opencv=cv2.__version__,numpy=np.__version__),
        source_point_count=len(uv),source_weak_point_count=sum(not q['direct'] for q in pts),
        native_path_id='board.top',physical_side_identity='lower_long_side',
        source_branch_id='near_lower_long_side',source_group_id='near_v6/'+binding['source_sha256']+'/'+str(binding['source_pts']),
        temporal_dependence_group='near_v6/'+binding['source_sha256']+'/'+row['shot_id']+'/board.top',
        samples_independent=False,interval_is_not_sigma=True,source_physical_boundary_mean_exact=False,
        human_confirmed_current_pixels=False,source_reference=dict(file=entry['frame_file'],sha256=entry['frame_sha256']),
        source_reader_reference=manifest['source_reader'],existing_human_acceptance_scope_extended=False)
    return dict(readings=[reading] if pts else [],ad_exclusion_mask=near,
        qualified_vertical_face_mask=face,qualified_ice_visible_mask=ice,
        source_reference=reading['source_reference'],source_generation_reference=reference,
        current_source_profile_reread=True,source_graphics_receipt=graphics_receipt,
        state=row['state'],current_source_state='visible' if pts else 'no_current_complete_material_profile',
        negative_exclusion_extrapolation_only=True,source_geometry_unchanged=True,
        source_reader='retained_official_v6_gray_cap_current_RGB')
