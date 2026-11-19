"""Score independent original ink landmarks and the two physical dasher boundaries."""
import numpy as np

from board_face_refine import FAR, NEAR, OUTPUT, PARENT, RAIL
from review import camera_state, ji, native_for
from shared import OUT, PREPARED, arrays, read, save, sha, write
from side_repair import native_table


def summary(a):
    a = np.asarray(a)
    good = a[np.isfinite(a)]
    if not len(good):
        return dict(n=len(a),finite=0)
    return dict(n=len(a),finite=len(good),median=float(np.median(good)),
                p50=float(np.median(abs(good))),p90=float(np.percentile(abs(good),90)),
                max=float(abs(good).max()))


def score(candidate, file, kind):
    name = 'jin_s14'
    q = arrays(file)
    b = camera_state(candidate)
    n = native_for(candidate,'jin')
    s = read(PREPARED/'setup.json')['shots'][name]
    keys,pi = np.unique(q['path'],return_inverse=True)
    tab = native_table(n,keys,count=4097 if kind=='rail' else 513)
    world = {k:v[0] for k,v in tab.items()}
    owner = q['idx']*len(keys)+pi
    order = np.argsort(owner,kind='stable')
    groups = np.split(order,np.flatnonzero(np.diff(owner[order]))+1)
    signed = np.full(len(order),np.nan)
    dxdy = np.full((len(order),2),np.nan)
    for lo in range(0,len(groups),64):
        pieces = groups[lo:lo+64]
        jobs = [(q['idx'][rr[0]],q['path'][rr[0]],b[name+'_times'][q['idx'][rr[0]]]) for rr in pieces]
        preds = ji.project_groups(n,b[name+'_rigs'],b[name+'_controls'],s['first'],s['dt'],jobs,world)
        for rr,(_,key,_),(curve,status) in zip(pieces,jobs,preds,strict=True):
            tt = tab[key][1]
            for j in rr:
                source = q['uv'][j]
                if kind=='far':
                    side = q['direction'][j]
                    axis = int(side in ('top','bottom'))
                    v = curve[status==0,axis]
                    signed[j] = (v.max() if side in ('bottom','right') else v.min())-source[axis]
                    continue
                t = q['station'][j]
                k = np.searchsorted(tt,t,'right').clip(1,len(tt)-1)
                if np.any(status[k-1:k+1]!=0):
                    continue
                alpha = (t-tt[k-1])/(tt[k]-tt[k-1])
                uv = (1-alpha)*curve[k-1]+alpha*curve[k]
                dxdy[j] = uv-source
                if q['station_fixed'][j]:
                    signed[j] = np.linalg.norm(dxdy[j])
                    continue
                # Re-intersect the local physical curve near the recorded
                # station with a normal through the actual source read.
                tangent = curve[k]-curve[k-1]
                normal = np.array([-tangent[1],tangent[0]])/max(1.e-10,np.linalg.norm(tangent))
                if kind=='rail':
                    if np.dot(normal,q['normal'][j])<0: normal *= -1.
                    signed[j] = dxdy[j]@normal
                else:
                    axis = int(q['side'][j] in ('top','bottom'))
                    if abs(normal[axis])>.25: signed[j] = (dxdy[j]@normal)/normal[axis]
    fold = q['fold'].copy()
    if kind=='near':fold[(q['pts']==1583360)&(fold<9)]=6
    if kind=='far':fold[(q['family']=='Ad24')&(q['direction']=='bottom')]=9
    return q,fold,signed,dxdy


def check(candidate):
    root = OUTPUT/'scores'/candidate.replace('/','_')
    result = {}
    for kind,file in [('far',FAR),('near',NEAR),('rail',RAIL)]:
        q,fold,err,delta = score(candidate,file,kind)
        names = q['family'] if kind!='rail' else q['material']
        result[kind] = {}
        for moment,at in [('all',np.ones(len(err),bool)),('122.500',q['pts']==1568000),
                          ('122.980',q['pts']==1574144),('123.060',q['pts']==1575168),('123.700',q['pts']==1583360)]:
            activity = at&(fold<7)
            groups = {}
            for name in np.unique(names[activity]):
                active = activity&(names==name)
                if kind=='near':
                    subsets = np.where(q['station_fixed'],'corner',q['side'])
                elif kind=='far':
                    subsets = q['direction']
                else: subsets = np.where(q['uv'][:,0]>=1280,'right','left_centre')
                groups[str(name)] = {str(group):summary(err[active&(subsets==group)]) for group in np.unique(subsets[active])}
            result[kind][moment] = dict(groups=groups,fit=summary(err[at&(fold<5)]),
                                        check=summary(err[at&np.isin(fold,[5,6])]))
        save(root/(kind+'.npz'),record=np.arange(len(err)),source_sha256=sha(file),fold=fold,
             error=err,dxdy=delta,pts=q['pts'],state_sha256=sha(OUT/candidate/'state.npz'))
    write(root/'source.json',result)
    print(candidate)
    for k,m in [('far','122.500'),('near','123.700'),('rail','all')]:
        print(k,m,{f:{x:round(v['median'],1) for x,v in d.items() if v['finite']} for f,d in result[k][m]['groups'].items()})
    return result


def photo_strips(candidate):
    import cv2
    from board_face_refine import PRIOR
    from review import source_frames
    from shared import prepared_packet_path

    name='jin_s14'
    shot=read(PREPARED/'setup.json')['shots'][name]
    previous=arrays(read(PRIOR/'source_bindings.json')[name]['packet']['path'])
    current=arrays(prepared_packet_path(name))
    folder=OUT/candidate/'inspection'
    folder.mkdir(parents=True,exist_ok=True)
    configs=((122.50,(1290,386,1920,475),'far_prints'),(122.50,(0,374,1919,501),'wide_rail'),
             (122.98,(350,328,1919,463),'before_pan'),(123.06,(350,316,1919,458),'pan'),
             (123.70,(1220,333,1920,650),'near_prints'),(123.82,(1000,300,1920,720),'exit'))
    sides=[]
    for sec,box,title in configs:
        pt=round(sec*12800);x,y,X,Y=box
        _,raw=next(source_frames(shot['source'],[pt]))
        strips=[]
        for label,branch,a in [('Original',None,None),('Previous',PARENT,previous),('Revised',candidate,current)]:
            layer=raw.copy()
            if branch is not None:
                state=camera_state(branch);native=native_for(branch,'jin')
                keys=[k for k,v in native.paths.items() if v['label'].startswith('Ad') or k in ('board/continuous/top','board/continuous/bottom')]
                table=native_table(native,keys,count=321)
                i=int(np.searchsorted(state[name+'_pts'],pt))
                jobs=[(i,k,state[name+'_times'][i]) for k in keys]
                curves=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],shot['first'],shot['dt'],
                                         jobs,{k:g[0] for k,g in table.items()})
                for k,(curve,status) in zip(keys,curves,strict=True):
                    good=np.flatnonzero((status==0)&np.isfinite(curve).all(1)&(abs(curve)<8000).all(1))
                    colour=(255,176,35) if k.startswith('board/') else (70,207,70)
                    for run in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
                        if len(run)>1:cv2.polylines(layer,[np.rint(curve[run]*16).astype('i4')],False,colour,1,cv2.LINE_AA,4)
                # Show the current admitted source UV on each rail. Ink remains
                # unobstructed for inspection of the letter/plate shape.
                rows=(a['pts']==pt)&(a['fold']<5)&np.isin(a['family'],['board.top','board.bottom'])
                for px in a['uv'][rows]:
                    cv2.circle(layer,tuple(np.rint(px*16).astype(int)),17,(25,30,235),-1,cv2.LINE_AA,4)
            crop=layer[y:Y,x:X].copy()
            cap=np.full((23,crop.shape[1],3),31,'u1')
            cv2.putText(cap,f'{label}  /  {sec:.3f} s  /  source PTS {pt}',(8,16),0,.48,(240,240,240),1,cv2.LINE_AA)
            strips.extend([cap,crop])
        path=folder/(title+'.png')
        cv2.imwrite(str(path),np.vstack(strips))
        sides.append(dict(pts=pt,source_seconds=sec,box_xyxy=box,image=str(path),sha256=sha(path)))
    write(folder/'shots.json',sides)


def encoded_windows(candidate, programme):
    """Compare the delivered raster with the same original exposure at registered review sites."""
    from fractions import Fraction
    from itertools import groupby
    from pathlib import Path

    import av
    import cv2
    from board_face_refine import OLD
    from review import source_frames
    from side_release_1118 import case_definitions

    gallery = read(OLD/'appearance_audit/verified/index.json')
    folder = OUT/candidate
    media = read(folder/'review'/programme/'coverage.json')
    sites = [(name, case) for name, shot in gallery['shots'].items() if shot['programme']==programme
             for case in shot['cases']]
    reported = [(name, round(sec*12800), crop, '+'.join(groups))
                for name, sec, groups, crop, _ in case_definitions()
                if gallery['shots'][name]['programme']==programme]
    latest = folder/'inspection/shots.json'
    if programme=='jin' and latest.is_file():
        reported.extend(('jin_s14', case['pts'], case['box_xyxy'], Path(case['image']).stem)
                        for case in read(latest))
    present = {(name, case['pts'], tuple(case['crop'])) for name, case in sites}
    for name, pts, crop, family in reported:
        key = (name, pts, tuple(crop))
        if key not in present:
            sites.append((name, dict(pts=pts, crop=crop, kind='reported', family=family)))
            present.add(key)
    sites.sort(key=lambda item: (item[1]['pts'], item[1]['kind']))
    source = gallery['shots'][sites[0][0]]['source']
    origin = round(media['source_seconds'][0]*12800)
    directory = folder/'inspection'/'encoded'
    directory.mkdir(parents=True,exist_ok=True)
    rows, index = [], []
    with av.open(media['video']) as video:
        stream = video.streams.video[0]
        assert (stream.time_base,stream.width,stream.height)==(Fraction(1,12800),2560,1144)
        stream.thread_count = 3
        for pts, block in groupby(sites,key=lambda site: site[1]['pts']):
            _, original = next(source_frames(source,[pts]))
            relative = pts-origin
            video.seek(relative,stream=stream,backward=True)
            encoded = next(f for f in video.decode(stream) if f.pts>=relative)
            assert encoded.pts==relative
            rendered = encoded.to_ndarray(format='bgr24')[64:1144,:1920]
            for name, case in block:
                x,y,X,Y = case['crop']
                scale = min(1.,620/(X-x),145/(Y-y))
                size = (max(1,round((X-x)*scale)),max(1,round((Y-y)*scale)))
                row = np.full((171,1248,3),29,'u1')
                for left,frame,title in [(0,original,'source'),(624,rendered,'encoded overlay')]:
                    crop = cv2.resize(frame[y:Y,x:X],size,interpolation=cv2.INTER_AREA)
                    row[23:23+size[1],left:left+size[0]] = crop
                    caption = f"{name.upper()}  {pts/12800:.3f}s   {case['family']}   {title}"
                    width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, .47, 1)[0][0]
                    font_scale = .47 * min(1., 609 / width)
                    cv2.putText(row,caption,(left+5,16),0,font_scale,(235,235,235),1,cv2.LINE_AA)
                page = len(index)//6+1
                target = directory/f'{programme}_{page:02}.jpg'
                index.append(dict(shot=name,pts=pts,review_pts=relative,kind=case['kind'],
                                  family=case['family'],crop=case['crop'],page=target.name,row=len(rows)))
                rows.append(row)
                if len(rows)==6 or len(index)==len(sites):
                    if not cv2.imwrite(str(target),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise OSError(target)
                    rows.clear()
    write(directory/(programme+'.json'),dict(candidate=candidate,source=source,
          source_sha256=gallery['shots'][sites[0][0]]['source_sha256'],video=media['video'],
          selection=gallery['check_scope'],windows=index))
    print(programme,len(index),'source and encoded comparison windows',flush=True)


def material_comparison(candidate):
    """Compare identical source rows and admissible material on both camera states."""
    area = OUTPUT/'scores'
    current_camera = candidate.replace('/','_')
    result = {}
    for kind,file in [('far',FAR),('near',NEAR),('rail',RAIL),('red',OUTPUT/'pan_paint/reads.npz')]:
        if kind=='red':
            q,fold,current,_ = score(candidate,file,'rail')
            _,_,previous,_ = score(PARENT,file,'rail')
        else:
            q = arrays(file)
            new = arrays(area/current_camera/(kind+'.npz'))
            old = arrays(area/PARENT.replace('/','_')/(kind+'.npz'))
            np.testing.assert_array_equal(new['record'],old['record'])
            assert new['source_sha256']==old['source_sha256']==sha(file)
            previous,current,fold = old['error'],new['error'],new['fold']
        scope = {}
        for label,mask in [('all',np.ones(len(fold),bool))]+[
                (f'{t:.3f}',q['pts']==round(t*12800)) for t in (122.50,122.98,123.06,123.70)]:
            use = mask&np.isin(fold,[5,6])
            if use.any():
                scope[label]=dict(previous=summary(previous[use]),current=summary(current[use]))
        result[kind] = dict(source=str(file),source_sha256=sha(file),held=scope)
    write(area/current_camera/'comparison.json',result)
    return result


def sportoto_word(candidate):
    """Read the two exterior letter strokes in the sharp S14 far-board views."""
    import cv2
    from review import source_frames

    name = 'jin_s14'
    setup = read(PREPARED/'setup.json')['shots'][name]
    native = native_for(candidate,'jin')
    ends = ['print_instance/Ad23::Ad10.print.glyph_1_S.boundary.0000',
            'print_instance/Ad23::Ad10.print.glyph_8_O.boundary.0000']
    world = {k:p[0] for k,p in ji.native_paths(native,ends,count=257).items()}
    states = [camera_state(PARENT),camera_state(candidate)]
    # Source scanlines pass through the word, between its two decorative arcs.
    scans = [(122.46,437),(122.48,436),(122.50,435),(122.52,434),(122.54,433)]
    rows, photos = [], []
    for sec,first_row in scans:
        pt = round(sec*12800)
        _, frame = next(source_frames(setup['source'],[pt]))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        paths = []
        for state in states:
            i = np.searchsorted(state[name+'_pts'],pt)
            assert state[name+'_pts'][i]==pt
            jobs = [(i,k,state[name+'_times'][i]) for k in ends]
            projected = ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],
                                          setup['first'],setup['dt'],jobs,world)
            assert all((status==0).all() for _,status in projected)
            paths.append([uv for uv,_ in projected])
        x0 = int(paths[0][0][:,0].min())
        x1 = int(np.ceil(paths[0][1][:,0].max()))
        label = f'Source {sec:.3f}s  / S, O'
        x_start = x0-33
        y_start = first_row-12
        originals = frame[y_start:first_row+17,x_start:x1+22].copy()
        marks = originals.copy()
        for y in range(first_row,first_row+3):
            white = np.median(np.r_[gray[y,x0-33:x0-24],gray[y,x1+12:x1+22]])
            for side in (0,1):
                # The first S and last O stand clear of other adverts. Read
                # from the white fascia inwards, with three contrast levels.
                indices = np.arange(x0-24,x0+5) if side==0 else np.arange(x1+11,x1-22,-1)
                darkness = white-gray[y,indices].astype(float)
                contrast = np.mean(np.sort(darkness)[-3:])
                if contrast<22:continue
                cuts = []
                for level in contrast*np.array([.35,.45,.55]):
                    inside = np.flatnonzero(darkness>=level)
                    if len(inside)==0 or inside[0]==0:break
                    j=inside[0]
                    frac=(level-darkness[j-1])/(darkness[j]-darkness[j-1])
                    cuts.append(float(indices[j-1]+frac*(indices[j]-indices[j-1])))
                if len(cuts)!=3:continue
                xs=[]
                for pair in paths:
                    uv=pair[side]; nxt=np.roll(uv,-1,axis=0)
                    cross=(uv[:,1]>y)!=(nxt[:,1]>y)
                    hits=uv[cross,0]+(y-uv[cross,1])*(nxt[cross,0]-uv[cross,0])/(nxt[cross,1]-uv[cross,1])
                    if not len(hits):break
                    xs.append(float(hits.min() if side==0 else hits.max()))
                if len(xs)!=2:continue
                rows.append(dict(pts=pt,y=y,glyph=('S','O')[side],source_x=cuts[1],
                                 threshold_x=cuts,previous_x=xs[0],current_x=xs[1]))
                for x,col in [(cuts[1],(30,220,240)),(xs[1],(50,230,70))]:
                    marks[y-y_start,round(x)-x_start]=col
        cells=[cv2.resize(pic,(pic.shape[1]*4,pic.shape[0]*4),interpolation=cv2.INTER_NEAREST)
               for pic in (originals,marks)]
        strip=np.zeros((147,1100,3),'u1')
        cv2.putText(strip,label,(6,18),0,.49,(238,238,238),1,cv2.LINE_AA)
        for left,cell in zip((0,550),cells):
            strip[27:27+cell.shape[0],left:left+cell.shape[1]]=cell
        photos.append(strip)
    folder=OUT/candidate/'inspection'
    cv2.imwrite(str(folder/'sportoto_word.png'),np.vstack(photos))
    evidence=dict(candidate=candidate,source=setup['source'],source_sha256=setup['source_sha256'],
                  fitted=False,method='contrast crossings on the first S and last O, between separate arcs',
                  rows=rows,source_pixel_uncertainty='finite print stroke, motion blur, and 35–55 percent contrast crossing')
    write(folder/'sportoto_word.json',evidence)
    return rows


def early_paint_check(candidate):
    """Score the independent C0 midpoints and the other materials in the same views."""
    from lane_review import pixel_errors
    from shared import prepared_packet_path

    name='jin_s14'
    state=camera_state(candidate);native=native_for(candidate,'jin')
    setup=read(PREPARED/'setup.json')['shots'][name]
    source=OUTPUT/'motion_check/circle_c0_source/c0_paired_source.npz'
    measured=arrays(source)
    assert measured['source_sha256'].item()==setup['source_sha256']
    pts,uv,outward=measured['pts'],measured['uv'],measured['normal']
    stamps,inv=np.unique(pts,return_inverse=True)
    frame=state[name+'_pts'];ii=np.searchsorted(frame,stamps)
    np.testing.assert_array_equal(frame[ii],stamps)
    paths=np.unique(measured['path'])
    assert len(paths)==1
    path=paths[0]
    table=ji.native_paths(native,[path],count=2049)
    jobs=[(i,path,state[name+'_times'][i]) for i in ii]
    predictions=ji.project_groups(native,state[name+'_rigs'],state[name+'_controls'],
                                  setup['first'],setup['dt'],jobs,{path:table[path][0]})
    err=np.full(len(uv),np.nan)
    def cross2(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for j,(curve,status) in enumerate(predictions):
        start=curve[:-1];vec=np.diff(curve,axis=0)
        valid=(status[:-1]==0)&(status[1:]==0)
        for k in np.flatnonzero(inv==j):
            normal=outward[k];den=cross2(vec,normal)
            u=np.divide(cross2(uv[k]-start,normal),den,out=np.full_like(den,np.nan),where=abs(den)>1.e-10)
            hits=valid&(u>=0)&(u<=1)
            along=(uv[k]-start[hits]-u[hits,None]*vec[hits])@normal
            if len(along):err[k]=along[np.argmin(abs(along))]
    assert np.isfinite(err).all()
    packet=arrays(prepared_packet_path(name))
    early=packet['pts']<round(121.9*12800)
    eligible=packet['fold']<9
    masks=dict(R4=eligible&(packet['family']=='R4')&(packet['material']=='surface_verified_boundary'),
               paired_coloured_lines=eligible&np.isin(packet['material'],['original_two_sided_painted_line','direct_coloured_paint_flanks']),
               early_grey=eligible&(packet['material']=='direct_early_lane1_midpoint'),
               top=eligible&(packet['material']=='vertical_fascia_upper_junction'),
               contact=eligible&(packet['material']=='kickplate_ice_contact'))
    e,_=pixel_errors(native,state,setup,name,packet,early&np.logical_or.reduce(list(masks.values())))
    result=dict(candidate=candidate,source_read_sha256=sha(source),c0=summary(err),family={
        family:summary(e[mask&early]) for family,mask in masks.items()},
        c0_exposures={str(t):summary(err[pts==t]) for t in stamps})
    where=OUTPUT/'scores'/candidate.replace('/','_')
    save(where/'early_circle.npz',pts=pts,observed=uv,outward=outward,signed_source_minus_model=err)
    write(where/'early_ice.json',result)
    return result
