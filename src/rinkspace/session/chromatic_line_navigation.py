"""Whole-image chromatic line proposals for locating source profile searches.

Returned polylines are navigation, never observations or physical identities.
"""
import cv2
import numpy as np


def source_board_floor_limit(rgb, exclusion):
    """Locate a long yellow kick plate in source pixels for ice-only searches.

    The conservative top of the material excludes walls and advertisements;
    it neither measures the board/ice contact nor creates an ice observation.
    """
    rgb=np.asarray(rgb,np.uint8);mask=np.asarray(exclusion,bool)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    yellow=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=25)&(hsv[:,:,2]>=70)&~mask
    # Closing joins small broadcast gaps only for locating a search domain.
    seed=cv2.morphologyEx(yellow.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,11),np.uint8));seed[mask]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed,8)
    supported=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if w>=300 and w>=4*h and area>=w:
            supported.append(dict(component=i,bbox=[x,y,x+w,y+h],source_pixels=int(np.count_nonzero(yellow&(labels==i)))))
    if not supported:return None,dict(status='source_kick_plate_unresolved',navigation_only=True)
    y=float(min(q['bbox'][1]for q in supported))
    return y,dict(status='long_source_kick_plate_supported',minimum_y=y,components=supported,navigation_only=True,
        boundary_pixels_used_as_observations=False)



def source_board_foot_component_support(rgb, exclusion):
    """Actual yellow pixels in long thin 2D source bands, never UV output.

    Reuse the floor navigator's material predicate and component width/order
    definition. The support only authenticates a candidate's material body;
    source RGB still supplies its entry and q50 lower transition.
    """
    rgb=np.asarray(rgb,np.uint8);mask=np.asarray(exclusion,bool)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or mask.shape!=rgb.shape[:2]:
        raise ValueError('RGB/mask shape mismatch')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    yellow=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=25)&(hsv[:,:,2]>=70)&~mask
    seed=cv2.morphologyEx(yellow.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,11),np.uint8));seed[mask]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed,8)
    support=np.zeros(mask.shape,bool);components=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        if w>=300 and w>=4*h and area>=w:
            actual=yellow&(labels==i);support|=actual
            components.append(dict(component=i,bbox=[x,y,x+w,y+h],source_pixels=int(actual.sum())))
    return support,dict(components=components,source_2D_material_support_only=True,
        morphology_fill_pixels_are_material_support=False,boundary_pixels_used_as_observations=False,
        source_material_predicate='existing_floor_navigation_HSV_yellow',
        component_rule='existing_floor_navigation_long_thin_component')


def _same_source_band(active, exclusion, first, second):
    """Check actual paint between two perspective sides, not their extrapolations."""
    axis=first[1]-first[0];axis=axis/np.linalg.norm(axis)
    a=first@axis;b=second@axis
    low=max(a.min(),b.min());high=min(a.max(),b.max())
    if high-low<80:return False
    station=np.linspace(low,high,31)
    def at(line,s):
        along=line@axis
        return line[0]+((s-along[0])/(along[1]-along[0]))[:,None]*(line[1]-line[0])
    left=at(first,station);right=at(second,station)
    xy=left[:,None]+np.linspace(0,1,33)[None,:,None]*(right-left)[:,None]
    ij=np.rint(xy).astype(int)
    inside=(ij[:,:,0]>=0)&(ij[:,:,0]<active.shape[1])&(ij[:,:,1]>=0)&(ij[:,:,1]<active.shape[0])
    ij[:,:,0]=np.clip(ij[:,:,0],0,active.shape[1]-1);ij[:,:,1]=np.clip(ij[:,:,1],0,active.shape[0]-1)
    clear=inside&~exclusion[ij[:,:,1],ij[:,:,0]]
    visible=clear.sum(1)>=25
    paint=(active[ij[:,:,1],ij[:,:,0]]&clear).sum(1)/np.maximum(clear.sum(1),1)
    return bool(visible.sum()>=8 and np.mean(paint[visible]>.5)>.65)


def source_chromatic_line_navigation(rgb,exclusion,native_tangent,*,signal='red',minimum_y=0.,radius_px=128.,allow_competing_search=False):
    rgb=np.asarray(rgb);mask=np.asarray(exclusion,bool)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or mask.shape!=rgb.shape[:2]:raise ValueError('RGB/mask shape mismatch')
    tangent=np.asarray(native_tangent,float);tangent/=np.linalg.norm(tangent)
    channel=rgb.astype(np.float32)
    strength=(channel[:,:,0]-(channel[:,:,1]+channel[:,:,2])/2 if signal=='red' else channel[:,:,2]-(channel[:,:,0]+channel[:,:,1])/2)
    active=(strength>6)&~mask
    if signal=='red':
        # Yellow kick-plates and purple print fields have positive average
        # red-minus-background too. Their hue cannot identify a red stripe.
        hsv=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV)
        active&=((hsv[:,:,0]<=12)|(hsv[:,:,0]>=174))&(hsv[:,:,1]>=10)
    active[:max(0,int(np.ceil(minimum_y)))]=False
    lines=cv2.HoughLinesP(np.uint8(active)*255,1,np.pi/720,80,minLineLength=250,maxLineGap=30)
    receipt=dict(navigation_only=True,physical_identity_established=False,model_UV_used_as_measurements=False)
    if lines is None:return None,dict(receipt,status='no_long_chromatic_source_branch')
    lines=lines.reshape(-1,2,2).astype(float);delta=lines[:,1]-lines[:,0];length=np.linalg.norm(delta,axis=1)
    directions=delta/np.maximum(length[:,None],1e-12)
    eligible=np.abs(directions@tangent)>np.cos(np.deg2rad(25.))
    lines,length,directions=lines[eligible],length[eligible],directions[eligible]
    if not len(lines):return None,dict(receipt,status='no_source_branch_in_expected_orientation')
    best=int(np.argmax(length));direction=directions[best];normal=np.array([-direction[1],direction[0]]);center=lines[best].mean(0)
    separation=np.max(abs((lines-center)@normal),axis=1)
    # Hough segments on opposite sides of a broad perspective stripe are
    # not competing materials. Keep their actual connected-colour support:
    # shared source components join proposals, without generating any UV.
    closed=cv2.morphologyEx(np.uint8(active),cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    closed[mask]=0
    _,labels=cv2.connectedComponents(closed,connectivity=8)
    supports=[]
    for line in lines:
        xy=np.rint(line[0]+np.linspace(0,1,101)[:,None]*(line[1]-line[0])).astype(int)
        ids,count=np.unique(labels[xy[:,1],xy[:,0]],return_counts=True)
        supports.append(set(map(int,ids[(ids>0)&(count>=5)])))
    same=np.array([bool(group&supports[best])for group in supports])
    possible=(separation>radius_px)&(length>.7*length[best])&~same
    for i in np.flatnonzero(possible):
        if _same_source_band(active,mask,lines[best],lines[i]):same[i]=True
    competing=possible&~same
    if competing.any() and not allow_competing_search:return None,dict(receipt,status='competing_long_chromatic_source_branches',hough_line=lines[best].tolist(),alternatives=lines[competing].tolist())
    yy,xx=np.where(active);xy=np.c_[xx,yy];support=xy[abs((xy-center)@normal)<radius_px]
    if len(support)<100:return None,dict(receipt,status='insufficient_chromatic_source_support')
    fitted=cv2.fitLine(support.astype(np.float32),cv2.DIST_L1,0,.01,.01).ravel();direction=fitted[:2];center=fitted[2:]
    station=(support-center)@direction;lo,hi=np.percentile(station,[2,98])
    polyline=center+np.linspace(lo,hi,129)[:,None]*direction
    return polyline,dict(receipt,status='competing_source_search_requires_whole_material_organization' if competing.any() else 'unique_long_chromatic_source_search',support_pixels=len(support),
        competing_source_branches=lines[competing].tolist(),
        hough_line=lines[best].tolist(),source_search_span_px=float(hi-lo),minimum_y=float(minimum_y),search_radius_px=float(radius_px))
