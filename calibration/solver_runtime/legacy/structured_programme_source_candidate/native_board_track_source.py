"""Same static board texture at two native exposures on the current course.

Reuses the authenticated Brown inverse/forward camera and current physical
course. Anchor intersections are recomputed, never supplied as measured XYZ.
No extra camera, print, depth or world parameters are introduced.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np
from .native_camera_local import native_camera_local
from scipy.sparse import csr_matrix, vstack
from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
from .persistent_ice_long_baseline_source import current_native_rays
from .native_ice_curve_source import SOURCE_SHA, _sha
from .paired_programme_chart import CORE_COUNT, WORLD_INDICES
from .fixed_pivot_positive_domain_fd import positive_pivot_height_columns, positive_pivot_coordinate_derivative


def current_far_board_points(camera, uv, border):
    """Intersect current source rays with the far finite vertical board course.

    Source qualification requires actual interior-face board texture. The
    furthest positive XY intersection is its far facing convex-course branch;
    a possible near-side backside crossing is not the observed surface.
    Z is computed from the ray, never clamped to ice or the board top.
    """
    c=np.asarray(camera,float);rays=current_native_rays(c,uv)
    a=np.asarray(border,float)[:,:2]
    if np.array_equal(a[0],a[-1]):a=a[:-1]
    edge=np.roll(a,-1,axis=0)-a
    d=rays[:,:2];ac=a-c[:2]
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    denominator=cross(d[:,None,:],edge[None,:,:])
    valid=np.abs(denominator)>1e-14
    safe=np.where(valid,denominator,1.)
    t=cross(ac,edge)[None,:]/safe
    u=cross(ac[None,:,:],d[:,None,:])/safe
    valid&=(t>0)&(u>=-1e-9)&(u<=1+1e-9)
    distance=np.max(np.where(valid,t,-np.inf),axis=1)
    if not np.isfinite(distance).all():
        raise ValueError('Reviewed board texture ray has no forward current course intersection')
    return c[:3]+distance[:,None]*rays


def board_material_coordinates(points,border):
    """Current developed-board arc length and Z; no Euclidean shape copying."""
    a=np.asarray(border,float)[:,:2]
    if np.array_equal(a[0],a[-1]):a=a[:-1]
    edge=np.roll(a,-1,axis=0)-a;length=np.linalg.norm(edge,axis=1)
    offset=np.asarray(points)[:,:2]
    t=np.einsum('nsi,si->ns',offset[:,None,:]-a[None,:,:],edge)/length[None,:]**2
    t=np.clip(t,0.,1.);nearest=a[None,:,:]+t[:,:,None]*edge[None,:,:]
    index=np.argmin(np.linalg.norm(offset[:,None,:]-nearest,axis=2),axis=1)
    station=np.r_[0.,np.cumsum(length)][index]+t[np.arange(len(index)),index]*length[index]
    return np.c_[station,np.asarray(points)[:,2]],float(length.sum())


def points_from_board_material(material,border):
    a=np.asarray(border,float)[:,:2]
    if np.array_equal(a[0],a[-1]):a=a[:-1]
    edge=np.roll(a,-1,axis=0)-a;length=np.linalg.norm(edge,axis=1);cum=np.r_[0.,np.cumsum(length)]
    station=np.asarray(material)[:,0]%cum[-1]
    index=np.minimum(np.searchsorted(cum,station,side='right')-1,len(length)-1)
    xy=a[index]+((station-cum[index])/length[index])[:,None]*edge[index]
    return np.c_[xy,np.asarray(material)[:,1]]


class NativeBoardTrackFactor:
    def __init__(self,state,packet_path,parameter_count):
        self.state=state;self.geometry=state.geometry.current
        self.core_count=int(state.structured_programme_coordinate_chart.core_count)
        self.world_indices=np.asarray(state.geometry_parameter_indices,int)
        self.parameter_count=int(parameter_count);self.packet_path=Path(packet_path).resolve()
        packet=json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema')!='rinkspace.native-static-board-tracks.v1'
            or packet.get('source_sha256')!=SOURCE_SHA or packet.get('time_base')!='1/15360'
            or packet.get('model_projection_used_to_generate_or_select_uv') is not False):
            raise ValueError('Explicit original native static-board source required')
        self.rows=[];self.held=[];ids=set();frames={}
        self.frozen_print_templates={str(r['id']):deepcopy(r)for r in packet.get('frozen_source_print_templates',[])}
        if self.frozen_print_templates:
            if set(self.frozen_print_templates)!=set(getattr(self.geometry,'source_print_templates',{})):
                raise ValueError('Admitted source and actual geometry must share the exact frozen template owner')
            for identifier,record in self.frozen_print_templates.items():
                if record['initialization']!=self.geometry.source_print_templates[identifier]['initialization']:
                    raise ValueError('Frozen source shape initializer differs from material producer')
        self.shared_print_templates={str(r['id']):deepcopy(r)for r in packet.get('shared_print_templates',[])}
        for item in self.shared_print_templates.values():
            authority=Path(item['shared_dimensions_authority']['path'])
            if _sha(authority)!=item['shared_dimensions_authority']['sha256']:
                raise ValueError('Shared print dimensions authority changed')
            facts=json.loads(authority.read_text(encoding='utf8'))['constraint']
            if not all(facts.get(k)is True for k in
                       ('same_printed_template','shared_printed_dimensions','physical_locations_independent')):
                raise ValueError('Explicit shared printed dimensions authority required')
            canonical=int(item['canonical_advertisement_id']);counterpart=int(item['counterpart_advertisement_id'])
            if not (1<=canonical<=26 and 1<=counterpart<=26 and abs(canonical-counterpart)==13):
                raise ValueError('Shared printed template must use the actual user-confirmed pair')
            if item.get('developed_station_orientation',1)!=1:
                raise ValueError('A print may not be reflected or rescaled on developed board material')
            for prefix in ('canonical','counterpart'):
                pts=int(item[prefix+'_anchor_pts'])
                if pts not in state.original.aliases:raise ValueError('Unknown print-template anchor exposure')
                frame=item[prefix+'_anchor_frame']
                if _sha(frame['path'])!=frame['sha256']:raise ValueError('Shared template source changed')
                uv=np.asarray(item[prefix+'_fiducial_uv'],float)
                if uv.shape!=(2,)or not np.isfinite(uv).all()or np.any(uv<0)or np.any(uv>=[1920,1080]):
                    raise ValueError('One real same-name template fiducial at each instance required')
        for record in packet['rows']:
            row=deepcopy(record)
            if row['id'] in ids or row.get('surface_id')!='boards_inner_surface':
                raise ValueError('Unique static inner-board track groups required')
            ids.add(row['id'])
            pair=self.shared_print_templates.get(str(row.get('shared_print_pair_id')))
            if pair is None and row.get('same_static_material_confirmed') is not True:
                raise ValueError('Real same-material source qualification required')
            if pair is not None and row.get('same_print_template_material_confirmed') is not True:
                raise ValueError('Different instances require confirmed same-template material correspondence')
            if pair is not None and (row.get('advertisement_id')!=pair['canonical_advertisement_id']
                    or row.get('counterpart_advertisement_id')!=pair['counterpart_advertisement_id']):
                raise ValueError('Real cross-instance material identities must match the shared template')
            for end in ('source','target'):
                native=int(row[end+'_pts'])
                if native not in state.original.aliases:raise ValueError('Unknown native exposure')
                actual=int(state.original.aliases[native])
                if actual not in state.bypts:raise ValueError('Exposure outside original programme')
                row[end+'_actual_pts']=actual
                ref=row[end+'_frame'];path=Path(ref['path']).resolve()
                if path not in frames:
                    if _sha(path)!=ref['sha256']:raise ValueError('Native source evidence changed')
                    frames[path]=ref['sha256']
                elif frames[path]!=ref['sha256']:raise ValueError('Conflicting source identity')
            ba=state.bypts[row['source_actual_pts']];bb=state.bypts[row['target_actual_pts']]
            if ba is not bb and pair is None:raise ValueError('Board texture correspondence crosses a programme cut')
            if pair is not None and row['source_pts']!=pair['canonical_anchor_pts']:
                raise ValueError('Shared-template material must use the canonical native anchor')
            row['uv0']=np.asarray(row['uv0'],float);row['uv1']=np.asarray(row['uv1'],float)
            if (row['uv0'].shape!=row['uv1'].shape or row['uv0'].ndim!=2
                or row['uv0'].shape[1]!=2 or len(row['uv0'])<2
                or not all(np.isfinite(row[k]).all() and np.all(row[k]>=0)
                           and np.all(row[k]<[1920,1080]) for k in ('uv0','uv1'))):
                raise ValueError('Finite native source-only paired UV required')
            row['sigma']=np.broadcast_to(np.asarray(row['sigma_px'],float),(len(row['uv0']),2)).copy()
            if not np.isfinite(row['sigma']).all() or np.any(row['sigma']<=0):raise ValueError('Positive declared sigma required')
            if row['split']=='fit':self.rows.append(row)
            elif row['split']=='holdout':self.held.append(row)
            else:raise ValueError('Explicit fit/holdout split required')
            if row.get('initialization'):
                self._validate_frozen_material_row(row,len(row['uv0']))
                number=int(row.get('counterpart_advertisement_id',row['advertisement_id']))
                cols=list(self.world_indices[self.geometry.material_parameter_indices(number)])
            else:cols=list(self.world_indices[44:50])
            steps={int(k):1e-6 for k in cols}
            additional=[]if pair is None else [state.original.aliases[int(pair['counterpart_anchor_pts'])]]
            for pts in (row['source_actual_pts'],row['target_actual_pts'],*additional):
                b=state.bypts[pts];view=b.start+4*b.bundle.index[pts]
                local=np.r_[b.xyz_slots,b.optics_chart.indices,b.optics_chart.mount_indices,view+np.arange(4)].astype(int)
                cols.extend(local);steps.update({int(k):1e-6 for k in local})
                steps.update({int(k):1e-4 for k in b.optics_chart.indices})
                steps.update({int(k):1e-7 for k in range(view,view+3)})
            row['_columns']=np.unique(cols).astype(int);row['_steps']=steps
        self.anchor_rows=[];self.anchor_held=[]
        for literal in packet.get('source_anchor_observations',[]):
            row=deepcopy(literal)
            if row['id']in ids or row['time_base']!='1/15360':raise ValueError('Unique authentic actual source anchor required')
            ids.add(row['id']);native=int(row['native_pts']);actual=int(state.original.aliases[native])
            if actual!=int(row['actual_pts'])or actual not in state.bypts:raise ValueError('Source anchor exposure clock changed')
            if _sha(row['source_raw_frame']['path'])!=row['source_raw_frame']['sha256']:raise ValueError('Source anchor RGB changed')
            row['uv']=np.asarray(row['uv'],float);self._validate_frozen_material_row(row,len(row['uv']))
            if row['uv'].ndim!=2 or row['uv'].shape[1]!=2 or not np.isfinite(row['uv']).all()or np.any(row['uv']<0)or np.any(row['uv']>=[1920,1080]):raise ValueError('Finite actual native UV anchor required')
            row['sigma']=np.broadcast_to(np.asarray(row['sigma_px'],float),row['uv'].shape).copy()
            if not np.isfinite(row['sigma']).all()or np.any(row['sigma']<=0):raise ValueError('Positive inherited anchor sigma required')
            row['_anchor']=True;number=int(row['advertisement_number']);b=state.bypts[actual];view=b.start+4*b.bundle.index[actual]
            cols=np.r_[self.world_indices[self.geometry.material_parameter_indices(number)],b.xyz_slots,b.optics_chart.indices,b.optics_chart.mount_indices,view+np.arange(4)].astype(int)
            row['_columns']=np.unique(cols);row['_steps']={int(k):1e-6 for k in cols}
            row['_steps'].update({int(k):1e-4 for k in b.optics_chart.indices});row['_steps'].update({int(k):1e-7 for k in range(view,view+3)})
            if row['split']=='fit':self.anchor_rows.append(row)
            elif row['split']=='holdout':self.anchor_held.append(row)
            else:raise ValueError('Source anchor FIT/HELD split must remain explicit')
        if not self.rows or not self.held:raise ValueError('Fit and independent held tracks required')
        self.height_columns=positive_pivot_height_columns(state.bypts.values())
        self._cache=OrderedDict();self._borders=OrderedDict()
        self.fit_evaluation_rows=self.rows+self.anchor_rows
        if 'fit_row_order'in packet:
            order=packet['fit_row_order'];lookup={r['id']:r for r in self.fit_evaluation_rows}
            if (not isinstance(order,list)or len(order)!=len(lookup)or len(set(order))!=len(order)
                    or set(order)!=set(lookup)):
                raise ValueError('Explicit FIT order must contain each actual admitted FIT row exactly once')
            self.fit_evaluation_rows=[lookup[identifier]for identifier in order]
        self.scales=np.concatenate([r['sigma'].ravel() for r in self.fit_evaluation_rows])
        self.quadratic=np.ones(len(self.scales),bool);self.row_count=len(self.scales)
        self.contract=dict(schema='rinkspace.native-current-board-track-factor.v1',
            packet_path=str(self.packet_path),packet_sha256=_sha(self.packet_path),
            source_operator='current full-Brown anchor ray -> current finite far board course -> current target full-Brown projection',
            new_camera_or_world_or_depth_parameters=0,held_used_in_fit=False,
            anchor_z_not_clamped=True,observed_world_xyz_supplied=False,
            correlated_texture_points_not_independent_measurements=True,
            sigma_is_engineering_weight_not_calibrated_noise=True,
            all_original_source_and_parameter_prefix_preserved=True,
            fit_groups=len(self.rows),held_groups=len(self.held),scalar_rows=self.row_count,
            accuracy_accepted=False)
        if self.frozen_print_templates:self.contract.update(
            source_operator='frozen q0 inferred normalized print material -> current shared dimensions/independent placement/current course -> actual camera full-Brown projection',
            frozen_template_ids=list(self.frozen_print_templates),additional_source_anchor_fit_groups=len(self.anchor_rows),
            additional_source_anchor_fit_scalar_rows=sum(r['uv'].size for r in self.anchor_rows),
            source_anchor_development_HELD_groups=len(self.anchor_held),
            source_UV_no_longer_redefines_world_shape_at_each_q=True,
            material_producer_same_as_actual_review_and_render=True,
            physical_parameter_count=self.geometry.parameter_count, new_camera_parameters=0,
            source_anchor_dedup_contract=deepcopy(packet.get('source_anchor_contract')))
        self.contract.update(
            source_operators_by_actual_row_family=dict(
                generic_static_board_pair='current full-Brown source ray -> current finite far board -> target camera',
                frozen_print_row_or_anchor='frozen normalized source template -> current material placement/dimensions/course -> actual camera'),
            generic_current_ray_transfer_FIT_groups=sum(not r.get('initialization')for r in self.rows),
            frozen_print_FIT_groups=sum(bool(r.get('initialization'))for r in self.rows)+len(self.anchor_rows),
            explicit_fit_row_order='fit_row_order'in packet,
            fit_row_ids_in_evaluation_order=[r['id']for r in self.fit_evaluation_rows])
        self.print_contours=deepcopy(packet.get('source_conditioned_print_contours',[]))
        print_ids=set()
        for item in self.print_contours:
            number=int(item['advertisement_id']);pts=int(item['anchor_pts'])
            if number in print_ids or not 1<=number<=26 or pts not in state.original.aliases:
                raise ValueError('Unique authenticated printed-instance anchor required')
            print_ids.add(number)
            if not any(r.get('advertisement_id')==number and r['source_pts']==pts for r in self.rows):
                raise ValueError('Print contour anchor requires actual fitted same-print material correspondence')
            if not any(r.get('advertisement_id')==number for r in self.held):
                raise ValueError('A new print requires independent held material evidence')
            anchor=Path(item['anchor_frame']['path'])
            if _sha(anchor)!=item['anchor_frame']['sha256']:
                raise ValueError('Print contour native source changed')
            if item.get('filled_rectangle_circle_or_symmetry_shape_assumption') is not False:
                raise ValueError('Source print outlines may not invent a primitive or counterpart')
            for contour in item['external_contours']:
                uv=np.asarray(contour['uv'],float)
                if (uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<3 or not np.isfinite(uv).all()
                    or np.any(uv<0) or np.any(uv>=[1920,1080])
                    or contour['role']!='actual_printed_component_external_contour'):
                    raise ValueError('Finite native external printed-component contour required')
        self.contract['source_conditioned_print_ids']=sorted(print_ids)

    def _validate_frozen_material_row(self,row,point_count):
        init=row.get('initialization',{});identifier=str(init.get('template_id'))
        record=self.frozen_print_templates.get(identifier)
        if record is None or init.get('q0_sha256')!=record['initialization']['q0_sha256']:
            raise ValueError('Explicit same frozen q0 template binding required')
        material=np.asarray(init['material_coordinates_normalized'],float)
        if material.shape!=(point_count,2)or not np.isfinite(material).all():raise ValueError('Finite normalized material initializer required')
        number=int(row.get('advertisement_number',row.get('advertisement_id')))
        target=int(row.get('counterpart_advertisement_id',number))
        allowed={int(record['canonical_advertisement_number']),int(record['counterpart_advertisement_number'])}
        if number not in allowed or target not in allowed:raise ValueError('Actual print identity differs from frozen template')

    def _border(self,q):
        world=np.asarray(q)[self.world_indices];key=world.tobytes()
        if key not in self._borders:
            course,cq,_=self.geometry.print_course_binding(world)
            self._borders[key]=np.asarray(course.sample_perimeter(cq)[0],float)
            while len(self._borders)>3:self._borders.popitem(last=False)
        return self._borders[key]

    def _evaluate(self,q,row):
        if row.get('_anchor')or row.get('initialization'):
            number=int(row['advertisement_number'])if row.get('_anchor')else int(row.get('counterpart_advertisement_id',row['advertisement_id']))
            uv=np.asarray(row['initialization']['material_coordinates_normalized'],float)
            points=self.geometry.source_template_material(number,*uv.T,np.asarray(q)[self.world_indices])[0]
            pts=int(row['actual_pts'])if row.get('_anchor')else row['target_actual_pts']
            block=self.state.bypts[pts];camera=block.bundle.camera(native_camera_local(block,np.asarray(q)[:self.core_count]),pts)
            projected,_,valid=project_central_valid_full_brown(camera,points)
            if not np.all(valid)or not np.isfinite(projected).all():raise ValueError('Actual print material leaves forward optical domain')
            observed=row['uv']if row.get('_anchor')else row['uv1']
            return ((projected-observed)/row['sigma']).ravel(),projected,points
        cameras=[]
        for pts in (row['source_actual_pts'],row['target_actual_pts']):
            b=self.state.bypts[pts];cameras.append(b.bundle.camera(native_camera_local(b,np.asarray(q)[:self.core_count]),pts))
        points=current_far_board_points(cameras[0],row['uv0'],self._border(q))
        if row.get('enforce_current_finite_board_height')is True:
            world=np.asarray(q)[self.world_indices]
            _course,cq,_chain=self.geometry.print_course_binding(world)
            height=float(_course._scene_q(cq)[5]);roundoff=64*np.finfo(float).eps*max(1.,abs(height))
            if np.any(points[:,2]<-roundoff)or np.any(points[:,2]>height+roundoff):
                raise ValueError('Actual static-print source ray leaves the current finite board height: '+str(row['id']))
        pair=self.shared_print_templates.get(str(row.get('shared_print_pair_id')))
        if pair is not None:points=self._place_counterpart(q,points,pair)
        projected,_,valid=project_central_valid_full_brown(cameras[1],points)
        if not np.all(valid) or not np.isfinite(projected).all():raise ValueError('Board track leaves forward optical domain')
        return ((projected-row['uv1'])/row['sigma']).ravel(),projected,points

    def _place_counterpart(self,q,points,pair):
        border=self._border(q);anchors=[]
        for prefix in ('canonical','counterpart'):
            actual=int(self.state.original.aliases[int(pair[prefix+'_anchor_pts'])]);block=self.state.bypts[actual]
            camera=block.bundle.camera(native_camera_local(block,np.asarray(q)[:self.core_count]),actual)
            point=current_far_board_points(camera,[pair[prefix+'_fiducial_uv']],border)
            anchors.append(board_material_coordinates(point,border)[0][0])
        material,perimeter=board_material_coordinates(points,border)
        delta=material-anchors[0]
        delta[:,0]=(delta[:,0]+perimeter/2)%perimeter-perimeter/2
        return points_from_board_material(anchors[1]+delta,border)

    def guard(self,q):
        if np.asarray(q).shape!=(self.parameter_count,):raise ValueError('Complete original state required')
        for row in self.fit_evaluation_rows:self._evaluate(q,row)

    def block(self,q,jacobian=True,*,original_start=0):
        key=np.asarray(q).tobytes();cached=self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):residual,matrix=cached
        else:
            values=[];matrices=[]
            for row in self.fit_evaluation_rows:
                r=self._evaluate(q,row)[0];values.append(r)
                if jacobian:
                    cols=row['_columns'];J=np.empty((len(r),len(cols)))
                    for i,col in enumerate(cols):
                        J[:,i]=positive_pivot_coordinate_derivative(
                            lambda v:self._evaluate(v,row)[0],q,int(col),row['_steps'][int(col)],
                            self.height_columns,base=r)
                    rr,cc=np.nonzero(J)
                    matrices.append(csr_matrix((J[rr,cc],(rr,cols[cc])),shape=(len(r),self.parameter_count)))
            residual=np.concatenate(values);matrix=vstack(matrices,format='csr') if jacobian else None
            self._cache[key]=(residual,matrix)
            while len(self._cache)>3:self._cache.popitem(last=False)
        return SourceRowBlock(residual,matrix,self.quadratic,self.scales,
            dict(family='additional_native_static_board_tracks',original_start=int(original_start),
                 original_stop=int(original_start)+self.row_count,packet_sha256=self.contract['packet_sha256']))

    def review(self,q):
        result=[]
        for row in self.rows+self.held+self.anchor_rows+self.anchor_held:
            _,projected,world=self._evaluate(q,row);observed=row['uv']if row.get('_anchor')else row['uv1'];errors=projected-observed
            result.append(dict(id=row['id'],split=row['split'],source_pts=row.get('source_pts',row.get('native_pts')),target_pts=row.get('target_pts',row.get('native_pts')),
                actual_source_anchor=bool(row.get('_anchor')),observed_UV=observed.tolist(),projected_UV=projected.tolist(),
                points=len(errors),median_native_px=float(np.median(np.linalg.norm(errors,axis=1))),
                maximum_native_px=float(np.max(np.linalg.norm(errors,axis=1))),
                signed_median_xy_px=np.median(errors,axis=0).tolist(),
                anchor_height_scene_units_minmax=[float(world[:,2].min()),float(world[:,2].max())]))
        return dict(contract=deepcopy(self.contract),rows=result,accuracy_accepted=False)

    def export_print_scene(self,scene,q):
        """Export the same source-conditioned course lift used by fitting.

        XYZ is a conditional result of this exact state, never an input
        measurement. Missing/cropped shapes and internal holes are not filled.
        """
        if not self.print_contours:return scene
        out=dict(scene);border=self._border(q);patterns=[]
        for item in self.print_contours:
            number=int(item['advertisement_id'])
            if number in getattr(self.geometry,'source_template_instance',{}):
                template_id=self.geometry.source_template_instance[number];record=self.frozen_print_templates[template_id]
                paths=[]
                for contour in record['external_contours']:
                    material=np.asarray(contour['initialization']['material_coordinates_normalized'],float)
                    if contour.get('closed')and not np.array_equal(material[0],material[-1]):material=np.vstack((material,material[0]))
                    xyz=self.geometry.source_template_material(number,*material.T,np.asarray(q)[self.world_indices])[0]
                    paths.append(dict(id=contour['id'],closed=bool(contour.get('closed')),xyz_scene_units=xyz.tolist(),
                        source_native_uv=deepcopy(contour['uv']),role=contour['role'],
                        frozen_normalized_material_coordinates=material.tolist(),
                        conditional_height_scene_units=[float(xyz[:,2].min()),float(xyz[:,2].max())]))
                patterns.append(dict(advertisement_id=number,anchor_pts=int(item['anchor_pts']),
                    anchor_actual_pts=int(self.state.original.aliases[int(item['anchor_pts'])]),
                    anchor_frame=deepcopy(item['anchor_frame']),external_contours=paths,shared_template_id=template_id,
                    actual_material_producer='PairedLocationsSharedSizesGeometry.source_template_material',
                    initializer_q_sha256=record['initialization']['q0_sha256'],
                    frozen_source_shape_inferred_not_surveyed=True,
                    independent_location_and_common_dimensions_estimated=True,
                    source_camera_does_not_rebuild_material_each_state=True,
                    internal_holes_source_only=deepcopy(item.get('internal_holes_recorded_separately',[]))))
                continue
            pair=next((r for r in self.shared_print_templates.values()if r['counterpart_advertisement_id']==int(item['advertisement_id'])),None)
            original=item if pair is None else next(r for r in self.print_contours if r['advertisement_id']==pair['canonical_advertisement_id'])
            at=int(self.state.original.aliases[int(original['anchor_pts'])])
            block=self.state.bypts[at]
            camera=block.bundle.camera(native_camera_local(block,np.asarray(q)[:self.core_count]),at)
            paths=[]
            for contour in original['external_contours']:
                uv=np.asarray(contour['uv'],float)
                if contour.get('closed') and not np.array_equal(uv[0],uv[-1]):uv=np.vstack((uv,uv[0]))
                xyz=current_far_board_points(camera,uv,border)
                if pair is not None:xyz=self._place_counterpart(q,xyz,pair)
                paths.append(dict(id=contour['id'],closed=bool(contour.get('closed')),xyz_scene_units=xyz.tolist(),
                    source_native_uv=uv.tolist(),role=contour['role'],
                    conditional_height_scene_units=[float(xyz[:,2].min()),float(xyz[:,2].max())]))
            patterns.append(dict(advertisement_id=int(item['advertisement_id']),anchor_pts=int(item['anchor_pts']),
                anchor_actual_pts=int(self.state.original.aliases[int(item['anchor_pts'])]),
                anchor_frame=deepcopy(item['anchor_frame']),external_contours=paths,
                contour_source_anchor_pts=int(original['anchor_pts']),contour_source_anchor_actual_pts=at,
                contour_source_anchor_frame=deepcopy(original['anchor_frame']),
                shared_template_id=None if pair is None else pair['id'],
                internal_holes_source_only=deepcopy(item.get('internal_holes_recorded_separately',[]))))
        out['source_conditioned_board_prints']=dict(schema='rinkspace.current-source-conditioned-board-print-contours.v1',
            q_sha256=hashlib.sha256(np.asarray(q,dtype=np.float64).tobytes()).hexdigest(),
            source_packet_path=str(self.packet_path),source_packet_sha256=self.contract['packet_sha256'],
            same_actual_course_and_camera_as_source_factor=True,conditional_XYZ_is_not_observed_world=True,
            anchor_height_not_clamped=True,whole_video_accuracy_accepted=False,patterns=patterns)
        out['source_conditioned_board_prints']['shared_templates']=deepcopy(list(self.shared_print_templates.values()))
        if self.frozen_print_templates:out['source_conditioned_board_prints'].update(
            actual_material_producer='frozen source normalized shape with current estimated width/height/station/Z',
            source_camera_reconstructs_world_shape_at_each_state=False,
            physical_geometry_parameter_count=self.geometry.parameter_count,
            normalized_shape_initializer_q_sha256s=sorted({r['initialization']['q0_sha256']for r in self.frozen_print_templates.values()}))
        return out


class SourceWithNativeBoardTracks:
    def __init__(self,base,factor):
        self.base=base;self.native_board_track_factor=factor;self.count=base.count;self.updater=base.updater
        self.quadratic=np.r_[base.quadratic,factor.quadratic];self.scales=np.r_[base.scales,factor.scales]
        self.contract=dict(schema='rinkspace.complete-source-plus-native-board-tracks.v1',
            original_source_contract=deepcopy(base.contract),board_track_contract=deepcopy(factor.contract),
            original_source_rows=len(base.scales),total_rows=len(self.scales),parameter_count=self.count)
    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)
    def guard(self,q):self.base.guard(q);self.native_board_track_factor.guard(q)
    def export_native_board_print_scene(self,scene,q):return self.native_board_track_factor.export_print_scene(scene,q)
    def blocks(self,q,jacobian=True):
        yield from self.base.blocks(q,jacobian)
        yield self.native_board_track_factor.block(q,jacobian,original_start=len(self.base.scales))


def wrap_native_board_tracks(base,state,packet_path):
    return SourceWithNativeBoardTracks(base,NativeBoardTrackFactor(state,packet_path,base.count))
