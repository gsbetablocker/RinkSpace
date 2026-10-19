"""Ad11/24 real rim sources on the continuous current board material.

The existing complete state is an unchanged prefix. Inverse board intersections
are used only to initialize three materials per badge, never as observations.
Every residual uses current native UV, current camera and current circle world.
An explicit source revision may share the radius of the two same-size prints.
The historical six-slot suffix remains readable; its last slot is then inert.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from .native_camera_local import native_camera_local
from .native_ice_curve_source import _source_optical_support, _native_error_with_optical_support
from scipy.sparse import csr_matrix
from z190_streaming_source_normal import SourceRowBlock
from wide_anchor_candidate.full_brown_projection import (
    curve_vector_error_full_brown, project_central_valid_full_brown)
from .native_ice_curve_source import (SOURCE_SHA,_sha,_native_frame_reference,
    _current_source_parameter_layout)
from .fixed_pivot_positive_domain_fd import (
    positive_pivot_height_columns, positive_pivot_coordinate_derivative,
)
from .paired_programme_chart import CORE_COUNT,WORLD_INDICES

PARAMETER_NAMES=('Ad11_station_from_lower_straight_tangent_m','Ad11_center_z_m',
                 'Ad11_outer_print_radius_m','Ad24_station_from_lower_straight_tangent_m',
                 'Ad24_center_z_m','Ad24_outer_print_radius_m')
SCENE_SCHEMA='rinkspace.additional-independent-native-board-circles.v1'
SHARED_RADIUS_BINDING='same_print_shared_radius_source_v1'
SHARED_PARAMETER_NAMES=PARAMETER_NAMES[:2]+('Ad11_Ad24_outer_print_radius_m',)+PARAMETER_NAMES[3:5]+('legacy_Ad24_radius_slot_unused_m',)


def _reviewed_source_uv_revision(packet):
    """Keep an explicit RGB-reviewed UV revision in the installed contract.

    Source review establishes the chosen printed material, not automatic
    localization precision. Existing packets without this revision are exact.
    """
    reference=packet.get('reviewed_source_uv_revision')
    if reference is None:return None
    path=Path(reference.get('path',''))
    if not path.is_file() or _sha(path)!=reference.get('sha256'):
        raise ValueError('Circle UV revision needs its unchanged actual RGB review')
    review=json.loads(path.read_text(encoding='utf8'))
    if (review.get('schema')!='rinkspace.native-circle-reviewed-outer-interface-uv-revision.v1'
            or review.get('source_video_sha256')!=SOURCE_SHA or review.get('time_base')!='1/15360'
            or review.get('physical_role')!='outermost_visible_circular_print_stroke'
            or review.get('authority',{}).get('kind')!='assistant_and_root_actual_native_RGB_outer_interface_review'
            or review.get('model_or_interpolated_UV_used') is not False
            or review.get('automatic_localization_precision_certified') is not False
            or review.get('public_reader_flags_and_wider_material_competition_preserved') is not True):
        raise ValueError('Explicit outer-interface source role review required for circle UV revision')
    parent_ref=review['original_packet'];parent_path=Path(parent_ref['path'])
    if not parent_path.is_file() or _sha(parent_path)!=parent_ref['sha256']:
        raise ValueError('Circle UV revision original packet changed')
    parent=json.loads(parent_path.read_text(encoding='utf8'))
    if parent['geometry_contract']!=packet['geometry_contract']:
        raise ValueError('Circle UV revision changed the existing circle world geometry contract')
    index=int(review['original_packet_row']);old=parent['rows'][index];new=packet['rows'][index]
    if len(packet['rows'])!=len(parent['rows']):raise ValueError('Circle UV revision changed row count')
    for k,(before,after)in enumerate(zip(parent['rows'],packet['rows'])):
        if k!=index and before!=after:raise ValueError('Circle UV revision changed another source row')
    for key in ('id','pts','actual_pts','curve','sigma_px','split','source_raw_frame','source_file_sha256'):
        if old.get(key)!=new.get(key):raise ValueError('Circle UV revision changed identity, clock, weight or source')
    if (old['id']!=review['row_id']
            or np.asarray(old['uv']).shape!=np.asarray(new['uv']).shape
            or not np.array_equal(np.asarray(old['uv'],float),np.asarray(review['original_uv'],float))
            or not np.array_equal(np.asarray(new['uv'],float),np.asarray(review['corrected_uv'],float))
            or old['sigma_px']!=review['original_sigma_px']
            or old.get('source_optical_normal_support_authorized')!=new.get('source_optical_normal_support_authorized')
            or old.get('source_optical_normal_support_authorized') is True):
        raise ValueError('Circle source-only UV revision cannot change weights or an active support operator')
    for evidence in (review['public_reader_evidence'],review['actual_RAW_outer_interface_review_image']):
        if _sha(Path(evidence['path']))!=evidence['sha256']:
            raise ValueError('Circle source UV review evidence changed')
    return dict(**reference,row_id=old['id'],original_packet=parent_ref,
        physical_role='outermost_visible_circular_print_stroke',
        assistant_and_root_source_RGB_role_review=True,automatic_localization_precision_certified=False,
        public_reader_flags_and_wider_material_competition_preserved=True,
        other_rows_and_source_sigma_clock_world_geometry_unchanged=True)


def _shared_radius_evidence(packet):
    contract=packet.get('geometry_contract',{})
    binding=contract.get('radius_binding')
    if binding is None:return None
    if binding!=SHARED_RADIUS_BINDING:
        raise ValueError('Unknown explicit Ad11/24 radius binding')
    reference=contract.get('same_print_source_evidence',{})
    path=Path(reference.get('path',''));expected=reference.get('sha256')
    if not path.is_file() or _sha(path)!=expected:
        raise ValueError('Shared radius needs unchanged actual same-print source evidence')
    evidence=json.loads(path.read_text(encoding='utf8'))
    if (evidence.get('schema')!='rinkspace.Ad11-Ad24-same-print-radius-source-evidence.v1'
            or evidence.get('advertisement_numbers')!=[11,24]
            or evidence.get('source_video_sha256')!=SOURCE_SHA
            or evidence.get('time_base')!='1/15360'
            or evidence.get('same_print_style_visually_reviewed') is not True
            or contract.get('shared_size_authority')!='latest_user_exact_duplicate_print_requirement'
            or evidence.get('centers_or_positions_forced_shared') is not False):
        raise ValueError('User exact duplicate-size authority and actual source print identity required')
    identities={(int(r['curve']['advertisement_id']),int(r['pts']),r['source_file_sha256']) for r in packet['rows']}
    certified={(int(r['advertisement_number']),int(r['pts']),r['source_file_sha256']) for r in evidence['source_frames']}
    if not identities<=certified:
        raise ValueError('Shared radius evidence does not cover these actual source exposures')
    return reference


def _legacy_circle_station_frame(course,course6,scene_unit_m,ad_id):
    bottom,_,parts=course.sample_perimeter(course6,with_jacobian=False)
    lengths=np.linalg.norm(np.roll(bottom,-1,axis=0)[:,:2]-bottom[:,:2],axis=1)*scene_unit_m
    cumulative=np.r_[0.,np.cumsum(lengths)]
    part=next(p for p in parts if p['id']==('left_end'if ad_id==11 else 'right_end'))
    first=int(part['first_vertex']);stop=first+int(part['vertex_count'])
    return float(cumulative[stop]if ad_id==11 else cumulative[first]),(-1. if ad_id==11 else 1.)


def straight_board_circle_world(course6,material3,scene_unit_m,ad_id,*,samples=361,course=None):
    """An intrinsic station/Z circle on the complete current board course.

    The legacy station origin/direction is retained at the lower tangent.
    Its value may cross either tangent or any computational course segment.
    The historical function name does not impose a straight-face domain.
    """
    q=np.asarray(course6,float);m=np.asarray(material3,float);unit=float(scene_unit_m)
    if (q.shape!=(6,) or m.shape!=(3,) or not np.isfinite(q).all()
            or not np.isfinite(m).all() or not np.isfinite(unit) or unit<=0
            or type(ad_id) is not int or ad_id not in (11,24)
            or type(samples) is not int or samples<17):
        raise ValueError('Finite current course6, circle material3 and explicit Ad11/24 required')
    a,b,rx,ry,exponent,h=q;s,z,r=m
    if not (a>rx>0 and b>ry>0 and exponent>1 and h>0):
        raise ValueError('Invalid current finite board course')
    if not (r>0 and r<=z<=h-r):
        raise ValueError(f'Ad{ad_id}: complete printed circle outside actual board height')
    if course is None:raise ValueError('Existing complete continuous course geometry required for printed circle material')
    origin,sign=_legacy_circle_station_frame(course,q,unit,ad_id)
    theta=np.linspace(0.,2*np.pi,samples);cos=np.cos(theta);sin=np.sin(theta)
    # Force exact mathematical closure, avoiding a nonzero sin(2*pi) endpoint.
    cos[-1]=cos[0];sin[-1]=sin[0]
    return course.station(origin+sign*(s+r*cos),q,z_m=z+r*sin,with_jacobian=False)


def _inverse_board_initializer(camera,uv,course6,unit,ad_id,*,course):
    """Actual source rays -> actual board plane -> circle seed, not a factor."""
    c=np.asarray(camera,float);pixels=np.asarray(uv,float)
    if c.shape!=(25,) or pixels.ndim!=2 or pixels.shape[1]!=2 or len(pixels)<3:
        raise ValueError('At least three real rim UV and current camera25 required')
    fx,fy=np.exp(c[6:8]);deskew=pixels.copy()
    deskew[:,0]-=c[10]*(pixels[:,1]-c[9])/fy
    K=np.array([[fx,0.,c[8]],[0.,fy,c[9]],[0.,0.,1.]])
    normalized=cv2.undistortPointsIter(deskew.reshape(-1,1,2),K,c[11:25],None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
    rays=np.column_stack((normalized,np.ones(len(normalized))))@cv2.Rodrigues(c[3:6])[0]
    check,depth,valid=project_central_valid_full_brown(c,c[:3]+100*rays)
    error=np.linalg.norm(check-pixels,axis=1)
    if not np.all(valid) or not np.isfinite(rays).all() or np.max(error)>1e-6:
        raise ValueError('Initializer inverse does not close the actual full-Brown native ray')
    a,b,rx,ry,p,h=np.asarray(course6,float);plane=(-1 if ad_id==11 else 1)*a/unit
    if np.any(abs(rays[:,0])<1e-12):raise ValueError('Initializer ray parallel to board')
    distance=(plane-c[0])/rays[:,0]
    if np.any(distance<=0):raise ValueError('Initializer board intersection lies behind current camera')
    points=(c[:3]+distance[:,None]*rays)*unit
    sz=np.column_stack((points[:,1]+b-ry,points[:,2]))
    design=np.column_stack((2*sz,np.ones(len(sz))))
    solution,_,rank,singular=np.linalg.lstsq(design,np.einsum('ni,ni->n',sz,sz),rcond=None)
    if rank!=3:raise ValueError('Real rim support cannot initialize three independent circle materials')
    radius_squared=solution[2]+np.dot(solution[:2],solution[:2])
    if not np.isfinite(radius_squared) or radius_squared<=0:raise ValueError('Invalid inverse circle initializer')
    unconstrained=np.r_[solution[:2],np.sqrt(radius_squared)]
    eps=32*np.finfo(float).eps*max(a,b,h,1.)
    r=min(unconstrained[2],h/2-eps)
    if r<=eps:raise ValueError('No positive finite circle initializer')
    seed=np.array([unconstrained[0],np.clip(unconstrained[1],r+eps,h-r-eps),r])
    straight_board_circle_world(course6,seed,unit,ad_id,course=course)
    return seed,dict(advertisement_number=ad_id,method='full-Brown source-ray/provisional tangent-plane algebraic circle seed only; continuous current board used by every factor',
        inverse_forward_closure_max_px=float(np.max(error)),native_uv_count=len(pixels),
        algebraic_rank=int(rank),algebraic_singular_values=singular.tolist(),
        unconstrained_material3_working_m=unconstrained.tolist(),finite_domain_seed3_working_m=seed.tolist(),
        initializer_clipped_only_to_actual_board_Z_domain=not np.array_equal(seed,unconstrained),
        station_or_face_domain_clipping=False,
        inverse_world_is_not_source_observation=True,shape_or_size_prior_added=False)


def freeze_admitted_circle_packet(candidate_paths,output_path,*,authority):
    """Freeze explicitly reviewed circle rows, preserving their source fields."""
    if not isinstance(authority,str) or not authority.strip():raise ValueError('Explicit source-review/admission authority required')
    rows=[];parents=[]
    for source in candidate_paths:
        p=Path(source).resolve();packet=json.loads(p.read_text(encoding='utf8'))
        if packet.get('source_video_sha256')!=SOURCE_SHA or packet.get('time_base')!='1/15360':
            raise ValueError('Same-event actual native source packet required')
        parents.append(dict(path=str(p),sha256=_sha(p)))
        for record in packet['rows']:
            if record.get('curve',{}).get('kind')!='board_circle':continue
            row=deepcopy(record);number=row['curve'].get('advertisement_id')
            if number not in (11,24) or row['split']!='candidate_not_admitted':
                raise ValueError('Only explicitly reviewed new Ad11/24 candidates can be admitted')
            row.update(split='fit',solver_admitted=True,source_review_authority=authority)
            rows.append(row)
    if {r['curve']['advertisement_id'] for r in rows}!={11,24}:
        raise ValueError('Both explicitly reviewed physical badge instances required')
    packet=dict(schema='rinkspace.additional-native-board-circles.v1',status='explicit_new_fit_target_authorized',
        source_video_sha256=SOURCE_SHA,time_base='1/15360',image_size_wh=[1920,1080],rows=rows,
        admission=dict(authority=authority,parent_candidate_packets=parents,original_candidates_not_overwritten=True,
            original_FIT_HELD_pending_roles_unchanged=True,source_UV_sigma_and_support_preserved=True),
        geometry_contract=dict(parameter_names=list(PARAMETER_NAMES),parameter_count=6,
            exact_halfturn_or_shared_radius_constraint=False,unknown_size_and_center_estimated=True),
        fit_uv_count=sum(len(r['uv']) for r in rows),fit_scalar_count=2*sum(len(r['uv']) for r in rows),accuracy_accepted=False)
    target=Path(output_path).resolve();serialized=json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if target.exists():
        if target.read_text(encoding='utf8')!=serialized:raise ValueError('Immutable admitted circle packet differs')
    else:target.write_text(serialized,encoding='utf8')
    return dict(path=str(target),sha256=_sha(target),fit_uv_count=packet['fit_uv_count'])


class NativeBoardCircleFactor:
    def __init__(self,state,packet_path,prefix_count):
        self.state=state;self.prefix_count=int(prefix_count);self.parameter_count=self.prefix_count+6
        self.circle_indices=np.arange(self.prefix_count,self.parameter_count)
        self.packet_path=Path(packet_path).resolve();packet=json.loads(self.packet_path.read_text(encoding='utf8'))
        if (packet.get('schema')!='rinkspace.additional-native-board-circles.v1'
                or packet.get('status')!='explicit_new_fit_target_authorized'
                or packet.get('source_video_sha256')!=SOURCE_SHA or packet.get('time_base')!='1/15360'):
            raise ValueError('Explicit admitted same-source Ad11/24 circle packet required')
        reviewed_source_uv_revision=_reviewed_source_uv_revision(packet)
        self.shared_radius_evidence=_shared_radius_evidence(packet)
        self.radius_shared=self.shared_radius_evidence is not None
        self.parameter_names=SHARED_PARAMETER_NAMES if self.radius_shared else PARAMETER_NAMES
        self.active_circle_indices=self.circle_indices[:5] if self.radius_shared else self.circle_indices
        self.geometry=state.geometry.current;self.unit=float(self.geometry.scene_unit_m)
        self.core_count,self.world_indices=_current_source_parameter_layout(state,self.parameter_count)
        self.world_parameter_count=len(self.world_indices);self.rows=[];self.held=[];ids=set()
        for record in packet['rows']:
            row=deepcopy(record);number=row['curve'].get('advertisement_id')
            if (row['id'] in ids or type(number) is not int or number not in (11,24)
                    or row['curve'].get('kind')!='board_circle'
                    or row['curve'].get('edge')!='outermost_visible_circular_print_stroke'):
                raise ValueError('Unique source-qualified physical Ad11/24 outer rim required')
            ids.add(row['id']);native=int(row['pts']);actual=int(state.original.aliases[native])
            if int(row.get('actual_pts',actual))!=actual or actual not in state.bypts:
                raise ValueError('Circle source exposure alias differs from authenticated camera')
            path,expected=_native_frame_reference(row)
            if _sha(path)!=expected:raise ValueError('Circle raw source bytes changed')
            uv=np.asarray(row['uv'],float);sigma=np.broadcast_to(np.asarray(row['sigma_px'],float),(len(uv),)).copy()
            if (uv.ndim!=2 or uv.shape[1:]!=(2,) or len(uv)<3 or sigma.shape!=(len(uv),)
                    or not np.isfinite(uv).all() or not np.isfinite(sigma).all()
                    or np.any(sigma<=0) or np.any(uv<0) or np.any(uv>=[1920.,1080.])):
                raise ValueError('At least three finite original native circle UV and positive raw sigma required')
            row.update(actual_pts=actual,uv=uv,sigma=sigma,advertisement_number=number)
            row['_source_optical_support'] = _source_optical_support(row)
            if row['split']=='fit':self.rows.append(row)
            elif row['split']=='holdout':self.held.append(row)
            else:raise ValueError('Explicit admitted FIT or HELD circle role required')
        if {r['advertisement_number'] for r in self.rows}!={11,24}:
            raise ValueError('Both six-material badge instances need actual FIT source')
        self.blocks={r['actual_pts']:state.bypts[r['actual_pts']] for r in self.rows+self.held}
        _,_,chain=self.geometry.print_course_binding(self.geometry.initial)
        chain=np.asarray(chain,float)
        if chain.shape!=(6,len(self.world_indices)):raise ValueError('Actual named66 current course chain required')
        self.active_world=self.world_indices[np.flatnonzero(np.any(chain!=0,axis=0))]
        self.camera_steps={};columns=[]
        for actual,block in self.blocks.items():
            view=block.start+4*block.bundle.index[actual]
            cols=np.r_[block.xyz_slots,block.optics_chart.indices,block.optics_chart.mount_indices,view+np.arange(4)].astype(int)
            columns.extend(cols)
            for col in cols:self.camera_steps[int(col)]=1e-6
            for col in block.optics_chart.indices:self.camera_steps[int(col)]=1e-4
            for col in range(view,view+3):self.camera_steps[col]=1e-7
        self.camera_columns=np.unique(columns);self.active=np.unique(np.r_[self.camera_columns,self.active_world,self.active_circle_indices]).astype(int)
        if np.intersect1d(self.camera_columns,np.r_[self.world_indices,self.circle_indices]).size:
            raise ValueError('Circle/camera/material column overlap')
        self.scales=np.concatenate([np.repeat(r['sigma'],2) for r in self.rows]);self.quadratic=np.ones(len(self.scales),bool)
        self.row_count=len(self.scales);self._cache=OrderedDict();self.initializer_report=None
        self.contract=dict(schema='rinkspace.additional-real-native-board-circle-factor.v1',packet_path=str(self.packet_path),
            material_operator_revision='continuous-complete-course-station-Z-v2',
            packet_sha256=_sha(self.packet_path),parameter_names=list(PARAMETER_NAMES),circle_indices=self.circle_indices.tolist(),
            prefix_parameter_count=self.prefix_count,parameter_count=self.parameter_count,new_parameter_count=6,
            evaluation_core_parameter_count=self.core_count,
            logical_world_parameter_count=self.world_parameter_count,
            logical_world_parameter_indices=self.world_indices.tolist(),
            scalar_rows=self.row_count,fit_ids=[r['id'] for r in self.rows],held_ids=[r['id'] for r in self.held],
        material='independent intrinsic station/Z circle on complete continuous current board',
            material_station_origin='legacy lower tangent, continuous perimeter station; no tangent or face bound',
            working_unit_scene_scale=self.unit,display_uniform_metric_scale_separate=True,
            centre_halfturn_or_shared_radius_constraint=False,finite_complete_circle_guard=True,
            raw_sigma_used_without_sqrt_N=True,held_rows_used_in_fit=False,
            operator='current full-Brown native vector distance to full finite circle, closest reassociated every q',
            old_source_camera_world66_q9_reference18_profiles_roles_and_clock_unchanged=True,accuracy_accepted=False)
        if self.radius_shared:
            self.contract.update(schema='rinkspace.additional-real-native-board-circle-shared-radius-factor.v1',
                parameter_names=list(self.parameter_names),radius_binding=SHARED_RADIUS_BINDING,
                material='same-size intrinsic station/Z circles on complete continuous current board; independent centers',
                center_halfturn_constraint=False,radius_shared=True,active_material_parameter_count=5,
                active_circle_parameter_indices=self.active_circle_indices.tolist(),
                shared_radius_parameter_index=int(self.circle_indices[2]),
                inactive_legacy_radius_parameter_index=int(self.circle_indices[5]),
                inactive_legacy_slot_has_no_world_residual_or_J_effect=True,
                same_print_source_evidence=deepcopy(self.shared_radius_evidence),
                centre_halfturn_or_shared_radius_constraint=True,equality_observation_or_sigma_added=False)
        if reviewed_source_uv_revision is not None:
            self.contract['reviewed_source_uv_revision']=reviewed_source_uv_revision

    def _cameras(self,q):
        core=np.asarray(q,float)[:self.core_count]
        return {actual:block.bundle.camera(native_camera_local(block,core),actual) for actual,block in self.blocks.items()}

    def _course(self,q):
        course,cq,chain=self.geometry.print_course_binding(np.asarray(q,float)[self.world_indices])
        cq=course.validate_parameters(cq)
        active=self.world_indices[np.flatnonzero(np.any(np.asarray(chain)!=0,axis=0))]
        if not np.array_equal(active,self.active_world):raise ValueError('Current circle course changed its certified active named66 columns')
        return cq

    def _material(self,q,ad_id):
        q=np.asarray(q,float);slot=0 if ad_id==11 else 3 if ad_id==24 else None
        if slot is None:raise ValueError('Explicit Ad11/24 required')
        material=q[self.circle_indices[slot:slot+3]].copy()
        if self.radius_shared:material[2]=q[self.circle_indices[2]]
        return material

    def world(self,q,ad_id,*,samples=361):
        course,_,_=self.geometry.print_course_binding(np.asarray(q,float)[self.world_indices])
        return straight_board_circle_world(self._course(q),self._material(q,ad_id),self.unit,ad_id,samples=samples,course=course)

    def guard(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,) or not np.isfinite(q).all():raise ValueError('Complete unchanged prefix plus new circle6 state required')
        for number in (11,24):self.world(q,number,samples=17)

    def initialize_suffix(self,prefix):
        q=np.asarray(prefix,float)
        if q.shape!=(self.prefix_count,) or not np.isfinite(q).all():raise ValueError('Actual complete prefix state required for circle initializer')
        cameras=self._cameras(q);cq=self._course(q);values=[];reports=[]
        for number in (11,24):
            group=[r for r in self.rows if r['advertisement_number']==number]
            if len({r['actual_pts'] for r in group})!=1:
                # Multiple frames are valid observations; choose the largest
                # true source support for this initializer, not a fake target.
                group=[max(group,key=lambda r:len(r['uv']))]
            uv=np.concatenate([r['uv'] for r in group]);actual=group[0]['actual_pts']
            course,_,_=self.geometry.print_course_binding(q[self.world_indices])
            material,report=_inverse_board_initializer(cameras[actual],uv,cq,self.unit,number,course=course)
            report.update(actual_pts=actual,native_pts=[r['pts'] for r in group]);values.extend(material);reports.append(report)
        values=np.asarray(values,float)
        if self.radius_shared:
            radius=float((values[2]+values[5])/2)
            radius=min(radius,cq[5]/2)
            values[[2,5]]=radius
            for slot in (0,3):
                values[slot+1]=np.clip(values[slot+1],radius,cq[5]-radius)
            reports.append(dict(radius_binding=SHARED_RADIUS_BINDING,shared_radius_initializer_working_m=radius,
                method='mean of the two source-ray radius seeds, with finite-board center guard; initializer only',
                equality_observation_or_sigma_added=False,inactive_legacy_radius_slot=int(self.circle_indices[5])))
        self.initializer_report=reports;return values

    def _residual(self,q,rows,*,cameras=None):
        cameras=self._cameras(q) if cameras is None else cameras;worlds={n:self.world(q,n) for n in (11,24)};values=[]
        for row in rows:
            value=curve_vector_error_full_brown(cameras[row['actual_pts']],worlds[row['advertisement_number']],row['uv'],with_jacobian=False)
            if isinstance(value,tuple):value=value[0]
            values.append(_native_error_with_optical_support(row,value)/np.repeat(row['sigma'],2))
        result=np.concatenate(values) if values else np.empty(0)
        if not np.isfinite(result).all():raise ValueError('Circle source leaves actual current optical domain')
        return result

    def block(self,q,jacobian=True,*,original_start=0):
        q=np.asarray(q,float);self.guard(q);key=q[self.active].tobytes();cached=self._cache.get(key)
        if cached is not None and (not jacobian or cached[1] is not None):residual,matrix=cached
        else:
            cameras=self._cameras(q);residual=self._residual(q,self.rows,cameras=cameras);matrix=None
            if jacobian:
                compact=np.empty((len(residual),len(self.active)))
                height_columns=positive_pivot_height_columns(self.blocks.values())
                try:
                    from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
                    compact=native_factor_dependency_jacobian(self,q,self.rows,self.active,
                        np.r_[self.world_indices,self.circle_indices],residual,
                        lambda value,groups:self._residual(value,[self.rows[i]for i in groups],
                            cameras=cameras if np.array_equal(value[self.camera_columns],q[self.camera_columns])else None),height_columns)
                finally:self._cameras(q)
                rr,cc=np.nonzero(compact);matrix=csr_matrix((compact[rr,cc],(rr,self.active[cc])),shape=(len(residual),self.parameter_count))
            self._cache[key]=(residual.copy(),matrix)
            while len(self._cache)>4:self._cache.popitem(last=False)
        ledger=dict(family='additional_real_native_board_circles',original_start=int(original_start),original_stop=int(original_start)+self.row_count,
            packet_sha256=self.contract['packet_sha256'],fit_ids=self.contract['fit_ids'],held_ids=self.contract['held_ids'],
            native_pts=[r['pts'] for r in self.rows],actual_pts=[r['actual_pts'] for r in self.rows],new_parameter_indices=self.circle_indices.tolist())
        if self.radius_shared:ledger.update(active_circle_parameter_indices=self.active_circle_indices.tolist(),
            shared_radius_parameter_index=int(self.circle_indices[2]),inactive_legacy_radius_parameter_index=int(self.circle_indices[5]))
        return SourceRowBlock(residual,matrix,self.quadratic,self.scales,ledger)

    def descriptor(self,q):
        self.guard(q);q=np.asarray(q,float);cq=self._course(q);circles={}
        for number,slot in ((11,0),(24,3)):
            material=self._material(q,number);world=self.world(q,number)
            course,_,_=self.geometry.print_course_binding(q[self.world_indices])
            origin,sign=_legacy_circle_station_frame(course,cq,self.unit,number)
            center=course.station(np.asarray([origin+sign*material[0]]),cq,z_m=material[1],with_jacobian=False)[0]
            circles[str(number)]=dict(circle_outer_stroke_xyz_scene_units=world.tolist(),center_xyz_scene_units=center.tolist(),
                radius_scene_units=float(material[2]/self.unit),material3_working_m=material.tolist(),
                physical_primitive='complete_continuous_board_course',station_direction_hint='left_end'if number==11 else'right_end',actual_curve_samples=len(world),
                extent_scope='visible outermost circular print stroke',size_or_center_independently_measured=False)
        result=dict(schema=SCENE_SCHEMA,circles=circles,parameter_names=list(PARAMETER_NAMES),parameter_indices=self.circle_indices.tolist(),
            current_circle6=q[self.circle_indices].tolist(),current_course6_working_m=cq.tolist(),circle_or_center_shared=False,
            source_packet_sha256=self.contract['packet_sha256'],initializer_report=self.initializer_report,accuracy_accepted=False)
        if self.radius_shared:
            result.update(parameter_names=list(self.parameter_names),circle_or_center_shared=True,radius_shared=True,centers_shared=False,
                radius_binding=SHARED_RADIUS_BINDING,shared_radius_working_m=float(q[self.circle_indices[2]]),
                shared_radius_parameter_index=int(self.circle_indices[2]),active_circle_parameter_indices=self.active_circle_indices.tolist(),
                inactive_legacy_radius_parameter_index=int(self.circle_indices[5]),
                inactive_legacy_slot_has_no_world_residual_or_J_effect=True,
                same_print_source_evidence=deepcopy(self.shared_radius_evidence))
        return result

    def export_scene(self,scene,q):
        out=deepcopy(scene);out['additional_board_circle_patterns']=self.descriptor(q);return out

    def review(self,q):
        self.guard(q);cameras=self._cameras(q);rows=[]
        for row in self.rows+self.held:
            residual=self._residual(q,[row],cameras=cameras);xy=(residual*np.repeat(row['sigma'],2)).reshape(-1,2);distance=np.linalg.norm(xy,axis=1)
            rows.append(dict(id=row['id'],pts=row['pts'],actual_pts=row['actual_pts'],split=row['split'],advertisement_number=row['advertisement_number'],
                raw_native_distance_px=distance.tolist(),mean_raw_native_distance_px=float(distance.mean()),max_raw_native_distance_px=float(distance.max()),
                native_source_UV=row['uv'].tolist(),sigma_px=row['sigma'].tolist(),source_file_sha256=row['source_file_sha256']))
        return dict(contract=self.contract,rows=rows,geometry=self.descriptor(q),accuracy_accepted=False)


class PrefixProfileUpdate:
    def __init__(self,base,prefix_count):self.base=base;self.prefix_count=int(prefix_count)
    def __call__(self,q):return self.base(np.asarray(q,float)[:self.prefix_count])
    def __getattr__(self,name):return getattr(self.base,name)


class SourceWithAdditionalNativeBoardCircles:
    def __init__(self,base,factor,initial_suffix=None):
        self.base=base;self.circle_factor=factor;self.prefix_count=int(base.count);self.count=self.prefix_count+6
        if factor.prefix_count!=self.prefix_count:raise ValueError('Circle factor prefix differs from complete original source')
        self.circle_indices=factor.circle_indices.copy();self.initial_suffix=None if initial_suffix is None else np.asarray(initial_suffix,float).copy()
        if self.initial_suffix is not None and (self.initial_suffix.shape!=(6,) or not np.isfinite(self.initial_suffix).all()):raise ValueError('Finite circle6 initializer required')
        self.updater=PrefixProfileUpdate(base.updater,self.prefix_count)
        self.quadratic=np.r_[base.quadratic,factor.quadratic];self.scales=np.r_[base.scales,factor.scales]
        self.contract=dict(schema='rinkspace.complete-source-plus-six-independent-real-native-board-circle-materials.v1',
            original_source_contract=deepcopy(base.contract),additional_circle_contract=deepcopy(factor.contract),
            original_parameter_count=self.prefix_count,parameter_count=self.count,original_source_rows=len(base.scales),
            total_rows=len(self.scales),new_circle_parameter_indices=self.circle_indices.tolist(),
            old_parameter_prefix_and_camera_indices_unchanged=True,old_J_has_six_exact_zero_extra_columns=True,
            original_source_UV_sigma_roles_profile_references_and_clock_preserved=True,accuracy_accepted=False)
        if factor.radius_shared:self.contract.update(schema='rinkspace.complete-source-plus-six-slot-five-active-shared-radius-board-circle-materials.v1',
            active_circle_parameter_indices=factor.active_circle_indices.tolist(),
            inactive_legacy_radius_parameter_index=int(factor.circle_indices[5]),old_six_slot_checkpoint_layout_retained=True)
    def __getattr__(self,name):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(name)
        return getattr(base,name)
    def extend_initial(self,prefix,suffix=None):
        q=np.asarray(prefix,float)
        if q.shape==(self.count,) and suffix is None:
            self.circle_factor.guard(q);return q.copy()
        if q.shape!=(self.prefix_count,) or not np.isfinite(q).all():raise ValueError('Actual previous complete prefix or complete circle state required')
        material=self.initial_suffix if suffix is None else np.asarray(suffix,float)
        if material is None:material=self.circle_factor.initialize_suffix(q)
        if material.shape!=(6,) or not np.isfinite(material).all():raise ValueError('Finite complete circle6 suffix required')
        result=np.r_[q,material];self.circle_factor.guard(result);return result
    def initial(self,*args,**kwargs):return self.extend_initial(self.base.initial(*args,**kwargs))
    def unpack(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.count,) or not np.isfinite(q).all():raise ValueError('Complete source prefix plus circle6 required')
        return self.base.unpack(q[:self.prefix_count])
    def guard(self,q):
        self.unpack(q);self.base.guard(np.asarray(q,float)[:self.prefix_count]);self.circle_factor.guard(q)
    def blocks(self,q,jacobian=True):
        self.unpack(q)
        for block in self.base.blocks(np.asarray(q,float)[:self.prefix_count],jacobian):
            matrix=None
            if jacobian:
                old=block.J.tocsr(copy=False)
                if old.shape[1]!=self.prefix_count:raise ValueError('Original block column prefix changed')
                matrix=csr_matrix((old.data,old.indices,old.indptr),shape=(old.shape[0],self.count),copy=False)
            yield SourceRowBlock(block.r,matrix,block.quadratic,block.scales,block.row_ledger,getattr(block,'loss_weights',None))
        yield self.circle_factor.block(q,jacobian,original_start=len(self.base.scales))
    def export_additional_circle_scene(self,scene,q):return self.circle_factor.export_scene(scene,q)


def wrap_additional_native_board_circles(base,state,packet_path,*,initial_suffix=None):
    return SourceWithAdditionalNativeBoardCircles(base,NativeBoardCircleFactor(state,packet_path,base.count),initial_suffix)


def additional_board_circle_paths(scene):
    descriptor=scene.get('additional_board_circle_patterns')
    if descriptor is None:return []
    if descriptor.get('schema')!=SCENE_SCHEMA:raise ValueError('Actual independent board-circle scene descriptor required')
    paths=[]
    for number,item in descriptor['circles'].items():
        n=int(number);xyz=np.asarray(item['circle_outer_stroke_xyz_scene_units'],float)
        if n not in (11,24) or xyz.ndim!=2 or xyz.shape[1:]!=(3,) or len(xyz)<17 or not np.isfinite(xyz).all() or not np.array_equal(xyz[0],xyz[-1]):
            raise ValueError('Actual closed finite Ad11/24 print world required')
        paths.append(dict(number=n,role='circle_outer_stroke_xyz',extent='printed_circle_outer_stroke',xyz=xyz.copy(),
            primitive='polyline',pattern_kind='shared_radius_circle_outer_stroke' if descriptor.get('radius_shared') else 'independent_circle_outer_stroke'))
    return paths


def install_additional_circle_rendering(render_module,*dependent_modules):
    """Install on this run's private renderer; both source and plan use it."""
    old=render_module.advertisement_review_paths
    if getattr(old,'_additional_native_board_circles',False):
        for module in dependent_modules:module.advertisement_review_paths=old
        return
    def complete_paths(scene):return old(scene)+additional_board_circle_paths(scene)
    complete_paths._additional_native_board_circles=True
    render_module.advertisement_review_paths=complete_paths
    # PlanView imports this function by value. Its private run module must be
    # passed explicitly so source and plan see precisely the same ad inventory.
    for module in dependent_modules:
        if getattr(module,'advertisement_review_paths',None) is not old:
            raise ValueError('Dependent renderer does not reference this same original review-path producer')
        module.advertisement_review_paths=complete_paths
