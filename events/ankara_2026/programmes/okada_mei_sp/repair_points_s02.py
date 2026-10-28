"""Whole-shot S02 source repair; immutable source/native, RGB-only new measurements."""
from pathlib import Path
import argparse,collections,functools,gzip,hashlib,io,json,time
import cv2,numpy as np
import points_s02 as old
import fit_camera_s02 as fit
from rinkspace.review.point_connections import polyline_order
OUT=old.OUT/'source_repair'

class GuideGrid(old.NativeGrid):
    """Existing native projection with a frozen, image-aligned camera candidate."""
    def __init__(self):
        from rinkspace.compat.d78_entities import sample_entities
        path=prepare();self.manifest,_,_,geometry,self.core,native=fit.load()
        self.parameter_state=native.parameter_state
        self.scene=self.parameter_state.apply_scene(read(self.manifest['scene_path']))
        self.state=dict(np.load(path,allow_pickle=False))
        if not np.array_equal(geometry['scene_values'],self.state['scene_values']):
            raise ValueError('Auxiliary camera changed the frozen native geometry')
        camera=read(fit.OUT/'camera_seed.json')
        self.first=camera['first_control_time'];self.dt=camera['control_dt'];self.reference=camera['reference_pts']
        self.samples=sample_entities(self.scene,self.core,samples=257)
        self.cuts={p['id']:p for p in self.samples['cuts']}
        self.receipt=dict(native_geometry=fit.file_ref(fit.OUT/'geometry_seed.npz'),camera=fit.file_ref(path),
            projected_coordinates_are_search_guides_only=True)

    @functools.lru_cache(maxsize=24)
    def project(self,pt):return super().project(pt)

    def unchanged(self):fit.load()


class SourceGuides:
    """Reuse source-motion identity, checking the two native line directions.

    A historical horizontal stripe must not seed an end-line track. This is an
    identity comparison between native direction families, not a residual cut.
    """
    def __init__(self,grid):
        from ice_points_s04 import FiniteSourceIdentity
        from rinkspace.compat.d78_entities import native_definition
        folder=OUT/'evidence';folder.mkdir(exist_ok=True)
        p=folder/'source_navigation.json'
        if not p.exists():p.write_bytes((old.OUT/'evidence/source_navigation.json').read_bytes())
        self.base=FiniteSourceIdentity(output_dir=OUT,visibility_path=old.OUT/'evidence/visibility.json',packet_dir=old.PACKETS,first_pts=old.FIRST)
        self.transforms=self.base.transforms;self.grid=grid;self.paths={};self.axes={};self.rejected=[];self.endpoints=[];self.endpoint_axes=[]
        for p in grid.scene['paths']:
            if p['layer']!='ice':continue
            kind,params=native_definition(grid.scene,p,grid.core)
            if kind!=0:continue
            xyz=np.asarray(grid.core.sample_path(grid.scene['board'],kind,params,[],[0.,1.]))
            axis=int(np.argmax(abs(xyz[1,:2]-xyz[0,:2])))
            self.paths[p['label']]=p;self.axes[p['label']]=axis
            self.endpoints.extend(xyz);self.endpoint_axes.append(axis)
        self.endpoints=np.asarray(self.endpoints)
        for file in sorted((old.OUT/'evidence/baseline_ice').glob('*.json')):
            packet=read(file)
            for name,f in packet['features'].items():
                if name not in self.paths or len(f['rows'])<2:continue
                self.base.guides[name].insert(0,dict(pts=packet['source_pts'],path=str(file),spans=[np.asarray([r['uv']for r in f['rows']])],operator='paint_center',signal=None,transition=None,centre=None,width=None))
        for name,gg in self.base.guides.items():
            if name not in self.paths:continue
            good=[]
            for g in gg:
                q=np.vstack(g['spans']);axis=self.direction_family(name,g['pts'],q)
                if axis is not None and axis!=self.axes[name]:
                    self.rejected.append(dict(feature_id=name,pts=g['pts'],source=g['path'],reason='historical_source_stroke_matches_other_native_direction_family',actual_direction_family=axis,expected_direction_family=self.axes[name]))
                else:good.append(g)
            self.base.guides[name]=good
        self.annotation=read(Path(__file__).with_name('s02_lane3_right_identity_20261027.json'))

    @functools.lru_cache(maxsize=600)
    def projected_directions(self,pt):
        result=self.grid.project_xyz(pt,self.endpoints);uv=result['uv'].reshape(-1,2,2);valid=(result['status'].reshape(-1,2)==0).all(1)
        output=[]
        for axis,v,ok in zip(self.endpoint_axes,uv,valid):
            d=v[-1]-v[0];n=np.linalg.norm(d)
            if ok and np.isfinite(d).all() and n>0:output.append((axis,d/n))
        return output

    def direction_family(self,name,pt,q):
        if len(q)<2:return None
        direction=np.linalg.svd(q-q.mean(0),full_matrices=False)[2][0];scores={}
        for axis,d in self.projected_directions(pt):
            scores.setdefault(axis,[]).append(float(abs(direction@d)))
        if len(scores)<2:return None
        # The median direction of each native parallel family is stable even
        # when one finite segment is clipped or an individual length is initial.
        return max(scores,key=lambda k:float(np.median(scores[k])))

    def at(self,name,pt):
        a=self.annotation
        if name==a['feature_id'] and self.transforms[pt][1]==self.transforms[a['source_pts']][1]:
            h=self.transforms[pt][0]@np.linalg.inv(self.transforms[a['source_pts']][0])
            return dict(pts=a['source_pts'],path=str(Path(__file__).with_name('s02_lane3_right_identity_20261027.json')),
                spans=[np.asarray(a['approximate_source_search_uv'])],operator='paint_center'),h
        return self.base.at(name,pt)


def reread(requested=None,resume=False):
    from types import SimpleNamespace
    from ice_points_s04 import source_domain
    from reread_s04_lines import NativeLineReader
    from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
    cv2.setNumThreads(2);grid=GuideGrid();photo=SourceGuides(grid);motion=old.OUT/'evidence/source_navigation.json'
    lines=NativeLineReader(grid,photographic=photo,motion_path=motion,source_prefix='s02_repaired',source_cross_line_guides=True)
    for path in lines.paths:
        path['_source_query_spacing_px']=35
        path['_separate_profile_support']=True
    a=photo.annotation
    lines.identity.guides.append(dict(kind='line',frame=a['source_pts'],curve_id=a['feature_id'].removeprefix('ice.'),uv=a['approximate_source_search_uv'],authority='user_identity_only_source_search',review_path=str(Path(__file__).with_name('s02_lane3_right_identity_20261027.json'))))
    house=old.SourceHouseReader(grid,motion_path=motion)
    vis=read(old.OUT/'evidence/visibility.json');lookup={v['pts']:v for v in vis['frames']}
    pts=sorted(lookup)if requested is None else sorted(requested)
    dest=OUT/'ice';dest.mkdir(exist_ok=True);fit.write(OUT/'source_guide_identity_audit.json',dict(rejected_navigation=photo.rejected,original_points_unchanged=True))
    prefix=old.FIRST-256
    if resume:
        while (dest/f'{prefix+256}.json').exists() and prefix<old.LAST:prefix+=256
        print(json.dumps(dict(stage='resume_actual_readouts',through_pts=prefix)),flush=True)
    for number,pt in enumerate(pts):
        if pt<=prefix:
            saved=read(dest/f'{pt}.json');line_features=[];circle_results=collections.defaultdict(dict)
            for name,f in saved['features'].items():
                nav=f.get('navigation',{})
                if 'source_line_fit'in nav:
                    line_features.append(dict(target_id=name,point_count=len(f['rows']),source_line_fit=dict(nav['source_line_fit'],direct_readouts=f['rows'])))
                for (fid,role),(_,_,path) in house.native_paths.items():
                    if path['label']==name and 'shared_source_affine_parameters'in nav:
                        house.prepare_family(fid,pt);circle_results[fid][role]=dict(rows=f['rows'],joint_group_navigation=nav)
            lines.identity.record(pt,line_features)
            for fid,result in circle_results.items():house.record_source_family(fid,pt,result)
            continue
        start=time.monotonic();v=lookup[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Wrong source exposure')
        graphics,border,_,_=source_domain(im,mask,read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        features,diagnostics=lines.read(pt,im,excluded,vis['source_video_sha256'])
        rejected={}
        for name,f in list(features.items()):
            direction=photo.direction_family(name,pt,np.asarray([r['uv']for r in f['rows']]))
            if direction is not None and direction!=photo.axes[name]:
                rejected[name]=dict(reason='actual_stroke_matches_other_native_direction_family',feature=features.pop(name))
                lines.identity.states.pop(name.removeprefix('ice.'),None)
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=excluded)
        circle_diag={};shape_features=[];results={}
        for fid in house.groups:
            house.prepare_family(fid,pt)
            curves=[house.projection(house.groups[fid],pt,role,np.linspace(0,2*np.pi,513))[0]for role in house.groups[fid]['radii']]
            cloud=np.vstack(curves)
            if not np.isfinite(cloud).all() or (cloud[:,0].max()<-75 or cloud[:,0].min()>1995 or cloud[:,1].max()<-75 or cloud[:,1].min()>1155):
                circle_diag[fid]=dict(status='outside_source');continue
            result,diag=house.read(fid,pt,native);circle_diag[fid]=diag;results[fid]=result
            for role,data in result.items():
                path=house.native_paths[(fid,role)][2];observed=[]
                for i,row in enumerate(data['rows']):
                    r={k:v for k,v in row.items()if k not in ('representative_uv','representative_to_actual_material_px')}
                    r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,source_row_id=f's02_repaired_house_RGB/{pt}/{fid}/{role}/{i}')
                    observed.append(r)
                features[path['label']]=dict(native_path_id=path['id'],rows=observed,navigation=data['joint_group_navigation'])
                ellipse=cv2.fitEllipse(np.asarray(data['curve'],np.float32))
                shape_features.append(dict(target_id=path['label'],measurement_operator='paint_center'if house.groups[fid]['outline']else'material_interface',segments=[dict(uv=[r['uv']for r in observed],source_evidence=[r['source_evidence']for r in observed])],geometric_interpretation=dict(kind='projected_physical_circle_ellipse',center_uv=ellipse[0],axes_px=ellipse[1],angle_degrees=ellipse[2])))
        for failure in conflicts(shape_features):
            name=failure['target_id'];rejected[name]=dict(reason='circle_material_identity_conflict',diagnostic=failure,feature=features.pop(name,None))
        for fid,result in results.items():
            if not any(name.startswith('ice.'+fid+':')for name in rejected):house.record_source_family(fid,pt,result)
        features,ownership=material_ownership(features,{p['id']:p for p in grid.scene['paths']})
        for failure in ownership:lines.identity.states.pop(failure['target_id'].removeprefix('ice.'),None)
        packet=dict(pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,line_diagnostics=diagnostics,circle_diagnostics=circle_diag,rejected_source_candidates=rejected,material_ownership_rejections=ownership,native_authority=grid.receipt,human_acceptance=False,model_or_flow_coordinates_emitted=False,seconds=time.monotonic()-start)
        fit.write(dest/f'{pt}.json',packet)
        if requested is not None or number%25==0 or number==len(pts)-1:
            print(json.dumps(dict(stage='ice_reread',frame=number+1,total=len(pts),pts=pt,counts={k:len(f['rows'])for k,f in features.items()},seconds=packet['seconds']),ensure_ascii=False),flush=True)
    grid.unchanged()


def material_ownership(features,entities):
    from calibration.program_tracking.audit_first_shot_circle_material_identity import centreline_house_conflicts
    failures,_,_=centreline_house_conflicts(features,entities)
    bad={r['source_row_id']for r in failures};output={}
    for name,feature in features.items():
        rr=[r for r in feature['rows']if r['source_row_id']not in bad]
        if rr:output[name]=dict(feature,rows=rr)
    return output,failures


def nonice(requested=None, output_dir=None, *, source_records_path=None, board_dir=None,
           source_prefix='s02', prepare_context=None, verify_context=None):
    """Recheck every existing board/ad row and reread complete panel edges.

    Original measurements remain in the old package. A stored, authenticated
    current-RGB reread is a separate measurement with a new ID, never an edit
    to the original click. Actual photo traces navigate outer-border queries.
    """
    from ice_points_s04 import current_graphics
    from rinkspace.vision_board_reader import source_front_face_top
    r,t,m,o,s,atlases,_=old.ad_configure();cv2.setNumThreads(2)
    (prepare_context or prepare)()
    by=collections.defaultdict(list)
    for row in rows(source_records_path or old.OUT/'points/point_records.jsonl.gz'):
        if '::'in row['native_path_id']or row['native_path_id'].startswith('board/'):
            by[row['pts']].append(row)
    charts=read(old.OUT/'ads/source_review/source_identity_charts.json')['records']
    atlases+=read(old.OUT/'ads/source_review/additional_source_atlases.json')['atlases']
    entities={e['native_path_id']:e for e in read(old.OUT/'evidence/native_entities.json')['entities']}
    selected=sorted(by)if requested is None else sorted(requested);dest=(Path(output_dir)if output_dir is not None else OUT)/'nonice';dest.mkdir(parents=True,exist_ok=True)
    for number,pt in enumerate(selected):
        im,gray,mask=r.current(pt);graphics,_=current_graphics(im);mask=np.maximum(mask,graphics.astype('uint8')*255)
        board=read((Path(board_dir)if board_dir is not None else old.REC/'dense_board_v1/qualified_frames')/f'{pt}.json')
        top=np.asarray([v['uv']for v in board['rows']if v['role']=='board.top'and v.get('pixels_measured_in_this_frame')and not v.get('occlusion')],float)
        contact=np.asarray([v['uv']for v in board['rows']if v['role']=='board.contact'and v.get('pixels_measured_in_this_frame')and not v.get('occlusion')],float)
        contact=contact[np.argsort(contact[:,0])]if len(contact)else contact
        cap,cap_proof=source_front_face_top(np.ascontiguousarray(im[:,:,::-1]),
            dict(source_sha256=r.VIS[pt]['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id=source_prefix),top,mask>0)
        cap=cap[np.argsort(cap[:,0])]if len(cap)else cap
        grouped=collections.defaultdict(list);kept=[];dispositions=[];features={}
        for row in by[pt]:grouped[row['native_path_id']].append(row)
        for pid,group in grouped.items():
            role=pid.split('::')[-1].split('/')[0].rsplit('.',1)[-1]
            border=role in ('top_edge','bottom_edge','right_vertical','left_vertical')
            if not border:
                for row in group:
                    out=row
                    profile=row.get('source_record',{}).get('source_profile',{})
                    measured=np.asarray(profile.get('actual_readout_uv',[]))
                    if ('::'in pid and row['kind']==1 and measured.shape==(2,) and np.isfinite(measured).all()
                        and not any(profile.get('flags',{}).values()) and not row['human_acceptance']):
                        if not np.array_equal(measured,row['uv']):
                            out=dict(row,observation_id=row['observation_id']+'@RGB_profile',uv=measured.tolist(),
                                origin='reuse_authenticated_current_RGB_profile_measurement',
                                original_observation_id=row['observation_id'],original_uv=row['uv'],
                                source_record=dict(row['source_record'],actual_readout_reused=True))
                    x,y=np.rint(out['uv']).astype(int)
                    if not(0<=x<1920 and 0<=y<1080)or mask[y,x]:
                        dispositions.append(dict(observation_id=row['observation_id'],original_uv=row['uv'],pts=pt,path_id=pid,reason='current_source_mask_intersection',preserved_in_original_package=True));continue
                    kept.append(out)
                    if out is not row:dispositions.append(dict(observation_id=row['observation_id'],original_uv=row['uv'],pts=pt,path_id=pid,replacement_observation_id=out['observation_id'],reason='use_existing_actual_current_RGB_profile_instead_of_old_query_coordinate',preserved_in_original_package=True))
                continue
            # One whole source contour supplies finite extent and ordering;
            # never sort scattered edge responses into a new rectangle.
            refs=[];uv=np.asarray([v['uv']for v in group])
            for ai,pi in {(v['review_order'].get('atlas'),v['review_order'].get('source_path'))for v in group if v['review_order'].get('family')=='ad'}:
                chart=charts.get(str(ai),{}).get(str(pt))
                if chart is None:continue
                contour=np.asarray(atlases[ai]['paths'][pi]['pixels'],float)
                q=cv2.perspectiveTransform(contour[None],np.asarray(chart['H'],float))[0]
                order,distance=polyline_order(uv,q)
                refs.append((float(np.median(distance)),q,dict(atlas_id=ai,source_path=pi,chart_pts=pt,source_contour_sha256=atlases[ai]['paths'][pi]['contour_sha256'])))
            if refs:_,q,proof=min(refs,key=lambda x:x[0])
            else:q=uv[np.argsort(uv[:,0])];proof=dict(actual_source_ids=[v['observation_id']for v in group])
            if len(q)<2:
                kept.extend(group);continue
            vertical=role in ('left_vertical','right_vertical')
            # One atlas can contain only part of a visible panel edge. Its
            # finite contour must not truncate already measured source extent.
            # Extend search support with the same entity's original pixels;
            # the colour transition below still measures every output afresh.
            source_extent_axis=1 if vertical else 0
            outside=(uv[:,source_extent_axis]<q[:,source_extent_axis].min())|(uv[:,source_extent_axis]>q[:,source_extent_axis].max())
            if outside.any():
                q=np.vstack([q,uv[outside]])
                proof['extended_by_original_source_ids']=[v['observation_id']for v,keep in zip(group,outside)if keep]
            # Panel verticals are physical vertical straight borders. The
            # horizontal edges may wrap a board corner and retain their curve.
            if vertical:
                v=cv2.fitLine(q.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel();d=v[:2];center=v[2:];ss=(q-center)@d
                q=center+np.linspace(ss.min(),ss.max(),max(2,int(np.ptp(ss)/8)))[:,None]*d
            else:
                q=q[np.argsort(q[:,0])]
            interpolated=m.interp_path(q,35.)
            if interpolated is None:kept.extend(group);continue
            query,normals,order=interpolated
            outward=np.array([-1.,0.]if role=='left_vertical'else [1.,0.]if vertical else [0.,-1.]if role=='top_edge'else [0.,1.])
            normals[np.einsum('ij,j->i',normals,outward)<0]*=-1
            inside=(query>=[40,40]).all(1)&(query<=[1879,1039]).all(1)
            query,normals,order=query[inside],normals[inside],order[inside]
            if not len(query):
                kept.extend(group);continue
            offsets=np.arange(-32.,32.5,.5);pixels=query[:,None]+offsets[None,:,None]*normals[:,None]
            values=cv2.GaussianBlur(r.sample(gray,pixels).astype('float32'),(5,1),.8);gradient=np.diff(values,axis=1)
            # The panel exterior is white board, not a bright phone/letter
            # inside the panel. Evaluate colour on both sides of each edge.
            mid=(offsets[:-1]+offsets[1:])*.5
            left=r.sample(im,query[:,None]+(mid-6)[None,:,None]*normals[:,None]);right=r.sample(im,query[:,None]+(mid+6)[None,:,None]*normals[:,None])
            inner_chroma=np.ptp(left,axis=2);outer_chroma=np.ptp(right,axis=2)
            eligible=(inner_chroma>20)&(outer_chroma<np.maximum(15.,inner_chroma*.60))
            # The native asset says whether this is a panel, logo rectangle,
            # or another retained material. The name suffix is not that role.
            native_role=entities[pid].get('role')
            proof['current_native_material_role']=native_role
            if 'atlas_id'in proof:
                ai=proof['atlas_id'];ref=r.reference(atlases[ai]);H=np.asarray(charts[str(ai)][str(pt)]['H'],float)
                source=cv2.perspectiveTransform((query-8*normals)[None],np.linalg.inv(H))[0]
                expected=r.sample(ref['color'],source[:,None]).reshape(-1,3)
                ec=expected-expected.mean(1,keepdims=True);cc=left-left.mean(2,keepdims=True)
                den=np.linalg.norm(ec,axis=1)[:,None]*np.linalg.norm(cc,axis=2)
                cosine=np.divide(np.einsum('nci,ni->nc',cc,ec),den,out=np.zeros_like(den),where=den>1)
                named_color=np.ptp(expected,axis=1)>20
                eligible[named_color]&=cosine[named_color]>.7
                proof['actual_source_reference_color_used']=True
            colored_query=eligible.any(1)
            # White artwork borders do not have a coloured interior. Keep
            # their original narrow photographic search; do not seek a dark
            # letter up to 32px away merely to obtain a transition.
            eligible[~colored_query]=np.abs(mid)<=4.
            # Evaluate every proposed edge against the actual cap exit. A
            # rejected blue cap must not become the fallback white-artwork edge.
            if len(cap)>=2:
                proposed=query[:,None]+mid[None,:,None]*normals[:,None]
                lower=np.interp(proposed[:,:,0],cap[:,0],cap[:,1],left=np.nan,right=np.nan)
                eligible&=(proposed[:,:,1]>lower+2.)
            if len(contact)>=2:
                proposed=query[:,None]+mid[None,:,None]*normals[:,None]
                bottom=np.interp(proposed[:,:,0],contact[:,0],contact[:,1],left=np.nan,right=np.nan)
                eligible&=(proposed[:,:,1]<bottom-2.)
            gradient[~eligible]=-np.inf;choice=gradient.argmax(1);possible=np.isfinite(gradient[np.arange(len(query)),choice])
            query=query+mid[choice,None]*normals
            indices=np.flatnonzero(possible);actual=[]
            profiles=m.read_profiles(im,r.VIS[pt],query[indices],normals[indices])if len(indices)else []
            for i,profile in zip(indices,profiles):
                if m.profile_reason(profile):continue
                point=np.asarray(profile['actual_source_uv']);x,y=np.rint(point).astype(int)
                if not(0<=x<1920 and 0<=y<1080)or mask[y,x]:continue
                quartiles=np.asarray(profile['source_q10_q50_q90_normal_offsets_px'])
                ss=r.sample(im,(query[i]+np.array([quartiles[0]-3,quartiles[2]+3])[:,None]*normals[i])[None])[0]
                chroma=np.ptp(ss,axis=1)
                if colored_query[i] and not(chroma[0]>20 and chroma[1]<max(15.,chroma[0]*.6)):continue
                if len(cap)>=2 and not point[1]>np.interp(point[0],cap[:,0],cap[:,1],left=np.nan,right=np.nan)+2.:continue
                if len(contact)>=2 and not point[1]<np.interp(point[0],contact[:,0],contact[:,1],left=np.nan,right=np.nan)-2.:continue
                actual.append(dict(uv=point.tolist(),source_row_id=f'{source_prefix}_identity_border_RGB/{pt}/{pid}/{i}',
                    native_path_id=pid,pts=pt,order=float(order[i]),source_evidence=dict(query_uv=query[i].tolist(),profile_normal=normals[i].tolist(),query_index=int(i),source_q10_q50_q90_normal_offsets_px=quartiles.tolist(),source_flags=profile['source_flags'],outer_white_board_inner_colored_panel=bool(colored_query[i]),source_search_radius_px=profile['source_search_radius_px']),pixel_support=dict(normal=normals[i].tolist(),query_index=int(i))))
            # Keep human clicks verbatim. Other prior rows get a complete,
            # traceable replacement disposition even when this exposure blurs.
            kept.extend(v for v in group if v['human_acceptance'])
            rejected_shape=[]
            if len(actual)>=5:
                points=np.asarray([v['uv']for v in actual]);axis=1 if vertical else 0;normal_axis=1-axis
                domain=points[:,axis];coordinate=points[:,normal_axis];good=np.ones(len(points),bool)
                for _ in range(3):
                    curve=np.polynomial.Polynomial.fit(domain[good],coordinate[good],1 if vertical else min(3,len(points[good])//5))
                    error=abs(coordinate-curve(domain))/np.sqrt(1+curve.deriv()(domain)**2)
                    updated=error<=3.
                    if updated.sum()<4 or np.array_equal(updated,good):break
                    good=updated
                rejected_shape=[dict(v,whole_border_error_px=float(error[j]),reason='inconsistent_with_whole_actual_panel_border')for j,v in enumerate(actual)if error[j]>3.]
                actual=[v for j,v in enumerate(actual)if error[j]<=3.]
            features[pid]=dict(rows=actual,navigation=proof,original_source_ids=[v['observation_id']for v in group],native_material_role=native_role,whole_panel_role_required=native_role!='logo_outer_boundary',rejected_actual_shape_candidates=rejected_shape)
            for v in group:
                if not v['human_acceptance']:dispositions.append(dict(observation_id=v['observation_id'],original_uv=v['uv'],pts=pt,path_id=pid,reason='whole_panel_outer_border_RGB_reread',replacement_source_ids=[x['source_row_id']for x in actual],preserved_in_original_package=True))
        fit.write(dest/f'{pt}.json',dict(pts=pt,retained_or_remeasured=kept,borders=features,dispositions=dispositions,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],board_cap_domain=cap_proof,human_acceptance_not_expanded=True))
        if requested is not None or number%50==0 or number==len(selected)-1:
            print(json.dumps(dict(stage='nonice',frame=number+1,total=len(selected),pts=pt,preserved_or_remeasured=len(kept),border_readouts=sum(len(x['rows'])for x in features.values()))),flush=True)
    (verify_context or fit.load)()


def entry_repair():
    """Reread only the identified skipped entries, retaining completed work."""
    from ice_points_s04 import source_domain
    from reread_s04_lines import NativeLineReader
    cv2.setNumThreads(2);grid=GuideGrid();photo=SourceGuides(grid)
    lines=NativeLineReader(grid,photographic=photo,motion_path=old.OUT/'evidence/source_navigation.json',source_prefix='s02_entry_repair',source_cross_line_guides=True)
    for p in lines.paths:p.update(_source_query_spacing_px=35,_separate_profile_support=True)
    affected=read(OUT/'missed_source_entry.json')['affected'];vis=read(old.OUT/'evidence/visibility.json');log=[]
    for index,v in enumerate(vis['frames']):
        pt=v['pts'];file=OUT/f'ice/{pt}.json';saved=read(file)
        if str(pt)in affected and 'entry_repair_attempt'in saved:
            log.append(dict(pts=pt,**{k:saved['entry_repair_attempt'][k]for k in ('attempted','recovered')}))
        elif str(pt)in affected:
            im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Source changed')
            graphics,border,_,_=source_domain(im,mask,read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            features,diag=lines.read(pt,im,(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]),vis['source_video_sha256'])
            added=[]
            for name in affected[str(pt)]:
                f=features.get(name)
                if f is None:continue
                direction=photo.direction_family(name,pt,np.asarray([r['uv']for r in f['rows']]))
                if direction is not None and direction!=photo.axes[name]:continue
                if name in saved['features']:raise ValueError('Entry repair would overwrite a completed feature')
                saved['features'][name]=f;saved['line_diagnostics'][name]=diag[name];added.append(name)
            saved['entry_repair_attempt']=dict(attempted=affected[str(pt)],recovered=added,diagnostics={name:diag.get(name)for name in affected[str(pt)]})
            fit.write(file,saved);log.append(dict(pts=pt,attempted=affected[str(pt)],recovered=added))
        restore=[]
        for name,f in saved['features'].items():
            nav=f.get('navigation',{})
            if 'source_line_fit'in nav:restore.append(dict(target_id=name,point_count=len(f['rows']),source_line_fit=dict(nav['source_line_fit'],direct_readouts=f['rows'])))
        lines.identity.record(pt,restore)
        if index%100==0:print(json.dumps(dict(stage='source_entry_repair',frames=index+1,recovered_groups=sum(len(x['recovered'])for x in log))),flush=True)
    fit.write(OUT/'source_entry_repair.json',dict(frames=567,attempts=log,recovered_groups=sum(len(x['recovered'])for x in log)));grid.unchanged()


def recover_blue_sources():
    """Restore the historical blue contour's actual pixels with old corrections."""
    from ice_points_s04 import source_domain,qualify
    from rinkspace.vision_native import NativeSourceGeometry
    cv2.setNumThreads(2);grid=GuideGrid();geo=NativeSourceGeometry(grid.manifest['scene_path'],grid.manifest['native_library'])
    inventory=read(old.OUT/'evidence/source_inventory.json');source=inventory['source_files'][0]
    if fit.file_ref(Path(source['path']))['sha256']!=source['sha256']:raise ValueError('Historical ice source changed')
    by=collections.defaultdict(list);total=0
    for line,r in enumerate(rows(Path(source['path']))):
        if r['feature_id'] not in ('ice.goal_crease_left:boundary','ice.goal_crease_right:boundary'):continue
        r=dict(r,original_source_ref=dict(**source,jsonl_line_0based=line));by[r['native_pts']].append(r);total+=1
    rejected=collections.Counter();kept=0;frames=[];dest=OUT/'blue_sources';dest.mkdir(exist_ok=True)
    with gzip.open(OUT/'blue_source_qualification.jsonl.gz','wt',encoding='utf8',compresslevel=3)as history:
        for index,v in enumerate(read(old.OUT/'evidence/visibility.json')['frames']):
            pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Source exposure changed')
            graphics,border,distance,_=source_domain(im,mask,read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            features=collections.defaultdict(list)
            for r in by[pt]:
                path=geo.lookup(r['feature_id'])
                reason=qualify(r,path,inventory['historical_corrections'],inventory['withdrawals'],graphics,border,distance)
                if reason:
                    rejected[reason]+=1;history.write(json.dumps(dict(source_row_id=r['source_row_id'],pts=pt,reason=reason))+'\n');continue
                p=r['pixel_support'];normal=np.asarray(p['normal']);uv=np.asarray(r['uv']);quartiles=np.asarray(p['side_interface_quartiles_uv'])
                evidence=dict(query_uv=(uv-float(p.get('native_shift_px',0))*normal).tolist(),profile_normal=normal.tolist(),
                    query_index=p['query_index'],actual_source_sides_uv=[r['uv'],r['uv']],actual_source_side_quantiles_uv=quartiles.tolist(),
                    signed_interface_iqr_contrast=p.get('signed_interface_IQR_contrast'),transition_sigma_px=float(p.get('interface_IQR_px',0))/1.349,
                    material_signal='blue',original_source_row_id=r['source_row_id'],original_source_ref=r['original_source_ref'],
                    actual_source_UV_retained=True,not_model_or_flow_position=True)
                fid=path['label'].removeprefix('ice.')
                features[fid].append(dict(uv=r['uv'],source_row_id=r['source_row_id'],source_evidence=evidence,original_source_record=r))
                kept+=1
            fit.write(dest/f'{pt}.json',dict(pts=pt,features=features,source_BGR_sha256=v['source_BGR_sha256']))
            frames.append(dict(pts=pt,qualified=sum(map(len,features.values()))))
            if index%100==0:print(json.dumps(dict(stage='historical_blue_actual_support',frames=index+1,qualified_rows=kept)),flush=True)
    fit.write(OUT/'blue_source_pool.json',dict(original_rows=total,qualified_rows=kept,rejected=dict(rejected),frames=frames,source=source,
        original_uv_unchanged=True,historical_withdrawals_preserved=True,qualification_is_not_user_acceptance=True));grid.unchanged()


def register_blue_source_navigation(house,photo,fid,pt):
    """Image-only registration from an existing photographed half-ellipse.

    A partially optimized camera is not an accuracy gate for source pixels.
    Keep the current native arc, using a source affine chart for RGB searches.
    """
    available=photo.at('ice.'+fid+':boundary',pt)
    if available is None:return None
    guide,motion=available
    if 'source_ellipse' in guide:e=guide['source_ellipse']
    else:
        packet=read(guide['path'])
        feature=next((f for f in packet['features']if f.get('target_id')=='ice.'+fid+':boundary'),None)
        if feature is None or feature.get('material_signal')!='blue' or feature.get('measurement_operator')!='material_interface':return None
        e=feature.get('geometric_interpretation',{})
    if not all(k in e for k in ('center_uv','axes_px','angle_degrees')):return None
    base=house.projection(house.groups[fid],pt,'outer',np.linspace(0,2*np.pi,513),None)[0]
    if not np.isfinite(base).all() or np.linalg.norm(np.diff(base,axis=0),axis=1).sum()<1e-8:return None
    original=cv2.fitEllipse(base.astype(np.float32))
    def transform(center,axes,angle):
        a=np.deg2rad(angle);t=np.eye(3);t[:2,:2]=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@np.diag(np.asarray(axes)*.5);t[:2,2]=center;return t
    if min(original[1])<1e-6 or min(e['axes_px'])<1e-6:return None
    source=transform(e['center_uv'],e['axes_px'],e['angle_degrees']);dest=np.linalg.inv(transform(*original))
    support=cv2.perspectiveTransform(np.vstack(guide['spans'])[None],motion)[0];candidates=[]
    for sign in (1.,-1.):
        target=source.copy();target[:2,:2]*=sign;H=motion@target@dest
        homogeneous=np.c_[base,np.ones(len(base))]@H.T
        if not np.isfinite(homogeneous).all() or np.any(homogeneous[:-1,2]*homogeneous[1:,2]<=0):continue
        mapped=homogeneous[:,:2]/homogeneous[:,2,None]
        _,distance=polyline_order(support,mapped);candidates.append((float(np.median(distance)),H))
    if not candidates:return None
    error,H=min(candidates,key=lambda x:x[0])
    if error>48:return None
    house.registration[(pt,fid)]=H;house.cache.clear()
    return dict(source_ref=fit.file_ref(Path(guide['path'])),source_pts=guide['pts'],H_source_from_native_search=H.tolist(),
        finite_actual_half_arc_navigation=True,median_guide_alignment_px=error,source_motion_only_within_qualified_component=True,
        current_native_geometry_unchanged=True,projected_or_flow_pixels_are_not_observations=True)


def add_actual_blue_source_guides(photo):
    """Use sufficiently extended actual blue arcs as image search charts."""
    for file in sorted((OUT/'blue_sources').glob('*.json')):
        packet=read(file);pt=packet['pts']
        for fid,rows0 in packet['features'].items():
            if len(rows0)<12:continue
            uv=np.asarray([r['uv']for r in rows0],np.float32);q=uv
            # This robust ellipse supplies query shape only. No source UV is
            # moved or accepted by this fit; the existing RGB operator and
            # common curve check still determine every delivered readout.
            for _ in range(3):
                e=cv2.fitEllipse(q)
                if min(e[1])<5 or max(e[1])>50000:break
                nearest=cv2.getClosestEllipsePoints(e,uv).reshape(-1,2);error=np.linalg.norm(uv-nearest,axis=1)
                q=uv[error<=max(3.,float(np.quantile(error,.75)))]
            else:
                e=cv2.fitEllipse(q);a=np.deg2rad(e[2]);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                extent=np.ptp((q-e[0])@rotation,axis=0)/e[1]
                # A physical half arc can cover only half one ellipse axis.
                # A tiny nearly straight fragment cannot invent its ellipse.
                if extent.min()<.45 or extent.max()<.8:continue
                photo.guides['ice.'+fid+':boundary'].append(dict(pts=pt,path=str(file),spans=[q],operator='material_interface',signal='blue',
                    source_ellipse=dict(center_uv=e[0],axes_px=e[1],angle_degrees=e[2]),observed_ellipse_axis_coverage=extent.tolist()))
    photo.at.cache_clear()


def circle_continuity(blue_only=False,source_pool=False,source_navigation=False):
    from types import SimpleNamespace
    from ice_points_s04 import source_domain
    from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
    cv2.setNumThreads(2);grid=GuideGrid();house=old.SourceHouseReader(grid,motion_path=old.OUT/'evidence/source_navigation.json')
    if blue_only:house.groups={fid:g for fid,g in house.groups.items()if house.native_paths[(fid,'outer')][0]==6}
    labels={p['label'] for (fid,role),(_,_,p)in house.native_paths.items()if fid in house.groups};vis=read(old.OUT/'evidence/visibility.json');report=[];tag='blue_source_navigation'if source_navigation else'blue_source_recovery'if source_pool else'blue_contour_repair'if blue_only else'circle_continuity_repair'
    if source_navigation:
        from ice_points_s04 import FiniteSourceIdentity
        photo=FiniteSourceIdentity(output_dir=OUT,visibility_path=old.OUT/'evidence/visibility.json',packet_dir=old.PACKETS,first_pts=old.FIRST)
        add_actual_blue_source_guides(photo)
    with gzip.open(OUT/(tag+'_history.jsonl.gz'),'at',encoding='utf8',compresslevel=3)as history:
        for index,v in enumerate(vis['frames']):
            pt=v['pts'];file=OUT/f'ice/{pt}.json';saved=read(file)
            photographic={}
            if source_pool:house.source_material_pool={pt:read(OUT/f'blue_sources/{pt}.json')['features']}
            if source_navigation:
                for fid in house.groups:
                    house.prepare_family(fid,pt);photographic[fid]=register_blue_source_navigation(house,photo,fid,pt)
            if saved.get(tag) and (not source_navigation or saved[tag].get('version')==2):
                results=collections.defaultdict(dict)
                for (fid,role),(_,_,path)in house.native_paths.items():
                    if fid not in house.groups:continue
                    f=saved['features'].get(path['label'])
                    if f:
                        house.prepare_family(fid,pt);results[fid][role]=dict(rows=f['rows'],joint_group_navigation=f['navigation'])
                for fid,result in results.items():
                    if all('shared_source_affine_parameters'in v['joint_group_navigation']for v in result.values()):house.record_source_family(fid,pt,result)
                report.append(saved[tag]);continue
            im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Source exposure changed')
            graphics,border,_,_=source_domain(im,mask,read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]))
            before={name:f for name,f in saved['features'].items()if name in labels};features={};diagnostics={};shapes=[];results={}
            for fid in house.groups:
                house.prepare_family(fid,pt)
                curves=[house.projection(house.groups[fid],pt,role,np.linspace(0,2*np.pi,513))[0]for role in house.groups[fid]['radii']];cloud=np.vstack(curves)
                if not np.isfinite(cloud).all()or cloud[:,0].max()<-75 or cloud[:,0].min()>1995 or cloud[:,1].max()<-75 or cloud[:,1].min()>1155:continue
                result,diag=house.read(fid,pt,native);diagnostics[fid]=diag;results[fid]=result
                if photographic.get(fid):
                    for data in result.values():data['joint_group_navigation']['photographic_half_arc_search']=photographic[fid]
                for role,data in result.items():
                    path=house.native_paths[(fid,role)][2];observed=[]
                    for i,row in enumerate(data['rows']):
                        r={k:v for k,v in row.items()if k not in ('representative_uv','representative_to_actual_material_px')}
                        r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,source_row_id=f's02_continuous_house_RGB/{pt}/{fid}/{role}/{i}');observed.append(r)
                    features[path['label']]=dict(native_path_id=path['id'],rows=observed,navigation=data['joint_group_navigation'])
                    ellipse=cv2.fitEllipse(np.asarray(data['curve'],np.float32))
                    shapes.append(dict(target_id=path['label'],measurement_operator='paint_center'if house.groups[fid]['outline']else'material_interface',segments=[dict(uv=[r['uv']for r in observed],source_evidence=[r['source_evidence']for r in observed])],geometric_interpretation=dict(kind='projected_physical_circle_ellipse',center_uv=ellipse[0],axes_px=ellipse[1],angle_degrees=ellipse[2])))
            bad={r['target_id']:r for r in conflicts(shapes)}
            for name,why in bad.items():saved['rejected_source_candidates'][name]=dict(reason='circle_material_identity_conflict',diagnostic=why,feature=features.pop(name,None))
            for fid,result in results.items():
                if not any(name.startswith('ice.'+fid+':')for name in bad):house.record_source_family(fid,pt,result)
            history.write(json.dumps(fit.safe(dict(pts=pt,features=before)),ensure_ascii=False)+'\n')
            saved['features']={name:f for name,f in saved['features'].items()if name not in labels};saved['features'].update(features);saved['circle_diagnostics'].update(diagnostics)
            info=dict(pts=pt,version=2 if source_navigation else 1,before_counts={k:len(f['rows'])for k,f in before.items()},after_counts={k:len(f['rows'])for k,f in features.items()},temporal_source_navigation={k:house.navigation_record.get(k)for k in results})
            saved[tag]=info;fit.write(file,saved);report.append(info)
            if index%50==0 or index==len(vis['frames'])-1:print(json.dumps(dict(stage='circle_continuity',frames=index+1,counts=info['after_counts'])),flush=True)
    fit.write(OUT/(tag+'.json'),dict(frames=567,records=report));grid.unchanged()

def blue_contour():circle_continuity(True)
def blue_source_recovery():circle_continuity(True,True)
def blue_source_navigation():circle_continuity(True,True,True)


def temporal_gap_reread():
    """Reread every missing same-shot material from finite real support."""
    from ice_points_s04 import source_domain
    from rinkspace.vision_native import NativeSourceGeometry
    cv2.setNumThreads(2);grid=GuideGrid();photo=SourceGuides(grid);finite=photo.base
    geo=NativeSourceGeometry(grid.manifest['scene_path'],grid.manifest['native_library']);finite.guides=collections.defaultdict(list)
    for file in sorted((OUT/'ice').glob('*.json')):
        packet=read(file)
        for name,f in packet['features'].items():
            rr=f['rows'];path=geo.current_paths[f['native_path_id']]
            if len(rr)<2:continue
            uv=np.asarray([r['uv']for r in rr]);circular=path['kind']in ('ice_circle','ice_ellipse');centre=None
            if circular:
                order=np.argsort([r.get('source_order_arc_px',r.get('theta',i))for i,r in enumerate(rr)])
                if len(uv)>=5:
                    ellipse=cv2.fitEllipse(uv.astype(np.float32))
                    if min(ellipse[1])>0:centre=ellipse[0]
            else:order=np.argsort((uv-uv.mean(0))@np.linalg.svd(uv-uv.mean(0),full_matrices=False)[2][0])
            q=uv[order];cuts=np.r_[0,np.flatnonzero(np.linalg.norm(np.diff(q,axis=0),axis=1)>(105 if circular else 260))+1,len(q)]
            spans=[q[a:b]for a,b in zip(cuts[:-1],cuts[1:])if b-a>=2]
            if not spans:continue
            finite.guides[name].append(dict(pts=packet['pts'],path=str(file),spans=spans,operator='paint_center'if path['measurement']=='centreline'else'material_interface',
                signal=rr[0]['source_evidence'].get('material_signal')or'dark',transition='light_to_dark'if path.get('role')=='physical_half_ellipse_boundary'else None,
                centre=centre,width=None))
    finite.at.cache_clear();summary=[]
    for index,v in enumerate(read(old.OUT/'evidence/visibility.json')['frames']):
        pt=v['pts'];file=OUT/f'ice/{pt}.json';saved=read(file)
        if 'temporal_gap_reread'in saved:summary.append(saved['temporal_gap_reread']);continue
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Source exposure changed')
        graphics,border,distance,_=source_domain(im,mask,read(old.REC/f'dense_board_v1/qualified_frames/{pt}.json'))
        present={f['native_path_id']for f in saved['features'].values()};candidates,diagnostic=finite.reread(im,pt,geo,present,graphics,border,distance)
        grouped=collections.defaultdict(list)
        for r in candidates:
            if r['feature_id']not in saved['features']:grouped[r['feature_id']].append(r)
        additions={};reasons={}
        for name,rr in grouped.items():
            path=geo.lookup(name)
            if path is None:continue
            circular=path['kind']in ('ice_circle','ice_ellipse');uv=np.asarray([r['uv']for r in rr]);g,h=finite.at(name,pt)
            if len(rr)<2:continue
            if not circular:
                if name not in photo.axes or photo.direction_family(name,pt,uv)!=photo.axes[name]:reasons[name]='other_native_line_direction';continue
                centre=uv.mean(0);normal=np.linalg.svd(uv-centre,full_matrices=False)[2][1];error=abs((uv-centre)@normal)
                order=(uv-centre)@np.array([-normal[1],normal[0]])
            else:
                query=np.asarray([r['pixel_support']['query_uv']for r in rr]);normal=np.asarray([r['pixel_support']['normal']for r in rr]);shift=np.einsum('ij,ij->i',uv-query,normal)
                translation=np.linalg.lstsq(normal,shift,rcond=None)[0];error=abs(shift-normal@translation)
                # One common image displacement; no independently moved rows.
                ref=cv2.perspectiveTransform(np.vstack(g['spans'])[None],h)[0];order,_=polyline_order(uv,ref)
            accepted=[]
            for i,r in enumerate(rr):
                if error[i]>3:continue
                support=r['pixel_support'];pair=np.asarray(r.get('paired_edge_uv')or[r['uv'],r['uv']]);quarters=np.asarray(support['side_interface_quartiles_uv'])
                ev=dict(query_uv=support['query_uv'],profile_normal=support['normal'],query_index=i,actual_source_sides_uv=pair.tolist(),actual_source_side_quantiles_uv=quarters.tolist(),
                    material_signal=g['signal'],actual_source_support_outside_person_and_broadcast_masks=True,measurement_origin='existing_finite_temporal_source_RGB_reread',
                    navigation_coordinates_are_not_observations=True,whole_material_normal_residual_px=float(error[i]))
                accepted.append(dict(uv=r['uv'],source_order_arc_px=float(order[i]),native_path_id=path['id'],feature_id=name,native_pts=pt,
                    source_row_id=f's02_gap_RGB/{pt}/{name}/{i}',source_evidence=ev,original_reread_record=dict(r,shot_id='s02')))
            if len(accepted)<2:reasons[name]='insufficient_coherent_current_material';continue
            additions[name]=dict(native_path_id=path['id'],rows=accepted,navigation=dict(source_pts=g['pts'],source_file=g['path'],
                finite_actual_source_support_only=True,shared_source_shape_test=True,projected_or_flow_UV_emitted=False))
        saved['features'].update(additions);info=dict(pts=pt,recovered={k:len(v['rows'])for k,v in additions.items()},rejected=reasons,diagnostic=diagnostic)
        saved['temporal_gap_reread']=info;fit.write(file,saved);summary.append(info)
        if index%100==0:print(json.dumps(dict(stage='whole_shot_temporal_gap_RGB',frames=index+1,recovered_groups=sum(len(v['recovered'])for v in summary))),flush=True)
    fit.write(OUT/'temporal_gap_reread.json',dict(frames=567,records=summary));grid.unchanged()


def resolve_competing_ice_labels():
    """Disambiguate an impossible trapezoid label on the measured C strip."""
    z=dict(np.load(OUT/'points/point_observations.npz'));evaluation=np.load(OUT/'points/native_evaluation.npz');mapping=read(OUT/'points/mapping.json')
    entities={v['native_path_id']:v for v in read(old.OUT/'evidence/native_entities.json')['entities']}
    groups=sorted({(int(z['pts'][i]),mapping['path_ids'][z['path_indices'][i]])for i in np.flatnonzero(evaluation['status'])})
    records=[]
    for pt,pid in groups:
        name=entities[pid]['label']
        if not name.startswith('ice.goal_trapezoid_'):continue
        file=OUT/f'ice/{pt}.json';packet=read(file);feature=packet['features'].get(name);centre_line=packet['features'].get('ice.C:segment0')
        if feature is None or centre_line is None or len(centre_line['rows'])<3:continue
        reference=centre_line['rows'];q=np.asarray([r['uv']for r in reference]);uv=np.asarray([r['uv']for r in feature['rows']])
        centre=q.mean(0);axes=np.linalg.svd(q-centre,full_matrices=False)[2];s=(q-centre)@axes[0];normal=axes[1]
        line=np.polynomial.Polynomial.fit(s,(q-centre)@normal,1);station=(uv-centre)@axes[0];distance=abs((uv-centre)@normal-line(station))
        half_width=float(np.median([np.linalg.norm(np.diff(r['source_evidence']['actual_source_sides_uv'],axis=0))*.5 for r in reference]))
        inside=(station>=s.min())&(station<=s.max())&(distance<=half_width)
        if inside.mean()<.75:continue
        evidence=dict(pts=pt,incorrect_label=name,actual_material_family='ice.C:segment0',
            reason='same_current_RGB_C_paint_strip_with_incompatible_trapezoid_identity',
            source_strip_membership_fraction=float(inside.mean()),actual_C_half_width_px=half_width,median_C_normal_distance_px=float(np.median(distance)),
            C_source_row_ids=[r['source_row_id']for r in reference],rejected_source_row_ids=[r['source_row_id']for r in feature['rows']],
            current_native_projection_domain_conflict_is_corroboration_not_sole_evidence=True,
            existing_C_readouts_already_cover_strip=True,no_extra_C_fit_weight_created=True,original_UV_preserved=True,human_acceptance=False)
        packet['rejected_source_candidates'][name]=dict(evidence=evidence,feature=packet['features'].pop(name))
        fit.write(file,packet);records.append(evidence)
    fit.write(OUT/'material_identity_conflicts.json',dict(records=records,groups=len(records),original_input_unchanged=True));fit.load()
    print(json.dumps(dict(resolved_competing_material_groups=len(records),pending_other_projection_groups=len(groups)-len(records))),flush=True)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf8')as f:
        for r in f:yield json.loads(r)

def prepare():
    OUT.mkdir(exist_ok=True)
    snapshot=OUT/'navigation_camera.npz'
    if not snapshot.exists():snapshot.write_bytes((fit.OUT/'navigation/accepted_camera_latest.npz').read_bytes())
    fit.load()
    return snapshot

def audit():
    prepare();z=dict(np.load(old.OUT/'points/point_observations.npz',allow_pickle=False));m=read(old.OUT/'points/mapping.json')
    ent={r['native_path_id']:r for r in read(old.OUT/'evidence/native_entities.json')['entities']}
    graph={r['pts']:r for r in rows(old.OUT/'review/connected_frame_index_no_projection.jsonl.gz')}
    report=[];totals=collections.Counter();all_frames={}
    for pt in range(old.FIRST,old.LAST+1,256):
        index=np.flatnonzero(z['pts']==pt);counts=collections.Counter()
        connections={r['path_id']:r for r in graph[pt]['entities']}
        for k in np.unique(z['path_indices'][index]):
            ids=index[z['path_indices'][index]==k];uv=z['uv'][ids];pid=m['path_ids'][k];e=ent[pid]
            family='ad' if '::'in pid else 'board' if pid.startswith('board/')else 'ice';counts[family]+=len(ids)
            r=dict(pts=pt,source_seconds=pt/12800,path_id=pid,label=e['label'],kind=e['kind'],rows=ids.tolist(),points=len(ids),flags=[])
            edges=np.array(connections[pid]['connection_rows'],int).reshape(-1,2)
            if len(edges):
                lengths=np.linalg.norm(z['uv'][edges[:,1]]-z['uv'][edges[:,0]],axis=1)
                r['longest_display_link_px']=float(lengths.max());r['display_links_over_300px']=int((lengths>300).sum())
                if r['display_links_over_300px']:r['flags'].append('long_connection_requires_source_span_check')
            if e['kind']=='ice_segment' and len(uv)>=3:
                centre=uv.mean(0);v=np.linalg.svd(uv-centre,full_matrices=False)[2][0];t=(uv-centre)@v;d=(uv-centre)@np.array([-v[1],v[0]])
                r['source_straight_residual_max_px']=float(abs(d).max());r['source_straight_span_px']=float(np.ptp(t))
                if len(edges):
                    steps=(z['uv'][edges[:,1]]-z['uv'][edges[:,0]])@v
                    r['order_direction_reversals']=int(min(np.count_nonzero(steps>1),np.count_nonzero(steps< -1)))
                    if r['order_direction_reversals']:r['flags'].append('straight_entity_display_backtracking')
                if abs(d).max()>4:r['flags'].append('line_readout_whole_material_shape_conflict')
            if e['kind']in ('ice_circle','ice_ellipse') and len(uv)>=5:
                try:
                    ellipse=cv2.fitEllipse(uv.astype(np.float32));q=cv2.getClosestEllipsePoints(ellipse,uv.astype(np.float32)).reshape(-1,2)
                    r['diagnostic_ellipse_residual_max_px']=float(np.linalg.norm(uv-q,axis=1).max())
                except cv2.error:r['flags'].append('ellipse_fit_degenerate_not_a_deletion_rule')
            for flag in r['flags']:totals[flag]+=1
            report.append(r)
        all_frames[str(pt)]=dict(counts)
    fit.write(OUT/'shape_audit.json',dict(source_points=fit.file_ref(old.OUT/'points/point_observations.npz'),frames=567,frame_coverage=all_frames,groups=report,diagnostic_counts=dict(totals),diagnostics_do_not_delete_or_relabel_points=True))
    bad=sorted([r for r in report if r.get('source_straight_residual_max_px',0)>4],key=lambda r:-r['source_straight_residual_max_px'])
    print(json.dumps(dict(frames=567,all_points=len(z['pts']),groups=len(report),diagnostic_counts=dict(totals),worst_straight_groups=bad[:12]),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','reread','nonice','entry_repair','circle_continuity','blue_contour','recover_blue_sources','blue_source_recovery','blue_source_navigation','temporal_gap_reread','resolve_competing_ice_labels']);p.add_argument('--pts',type=int,nargs='+');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stage=='reread':reread(a.pts,a.resume)
    elif a.stage=='nonice':nonice(a.pts)
    else:globals()[a.stage]()
