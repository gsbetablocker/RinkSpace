"""Ad7-only grouped art in true unfolded board axes, with centered parent slots.

The logo and each text line retain their own SVG proportions. The logo/bottom
centres and logo/common-text scales exist in the chart. Parameters without
assigned current source support remain fixed candidate initializations; footer
spacing is not measured.
The text outlines remain an official-font candidate, not a human font measurement.
No H8/shear/projective freedom remains on this model's Ad7 ink. Other ads retain
all original layout, source, geometry, and physical guard implementations.
"""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import MethodType

import numpy as np
from scipy.sparse import csr_matrix, vstack

from .first_shot_source_transition import reference, read
from .first_shot_native_time_stage import _sha, _same
from .first_shot_physical_state import GeometryDomainError


class Ad7ShapeDomainError(GeometryDomainError):
    """Only explicit grouped-art scale/containment/order failures."""


SCHEMA = 'rinkspace.ad7-grouped-shape-chart.v1'
PARAMETERS = ('logo_u', 'logo_v', 'log_logo_scene_scale', 'top_u', 'top_v',
              'bottom_u', 'bottom_v', 'log_text_scene_scale')
GROUPS = ('logo', 'top', 'bottom')
_GLOBAL = None
_ORIGINAL = {}


def _group(key):
    if key.startswith('logo_'):return 0
    if key.startswith('top_'):return 1
    if key.startswith('bottom_'):return 2
    raise ValueError('Ad7 path has no explicit logo/top/bottom group: '+key)


def canvas_metrics(state, geometry):
    """Existing exact width/height and derivatives, in the scene's common units."""
    q = state._geometry(geometry)
    width,height,dw,dh = state.provider.current._size(7,q[:100])
    unit = float(state.scene_unit_working_m)
    values = np.asarray([width,height],float)/unit
    D = np.zeros((2,len(q)))
    D[:,:100] = np.asarray([dw,dh],float)/unit
    if not np.isfinite(values).all() or np.any(values<=0) or not np.isfinite(D).all():
        raise Ad7ShapeDomainError('Positive finite actual Ad7 unfolded canvas required')
    return values,D


class Ad7GroupedShapeChart:
    def __init__(self, configuration):
        if configuration.get('schema') != SCHEMA:
            raise ValueError('Explicit Ad7 grouped shape configuration required')
        self.configuration=deepcopy(configuration)
        self.anchor=np.asarray(configuration['parent_slot_anchor'],np.float64)
        self.seed=np.asarray(configuration['physical_seed8'],np.float64)
        self.units=np.asarray(configuration['svg_axis_units'],np.float64)
        self.material=read(configuration['material'])
        if (self.anchor.shape!=(8,) or self.seed.shape!=(8,) or self.units.shape!=(2,)
                or not np.isfinite(np.r_[self.anchor,self.seed,self.units]).all()
                or np.any(self.units<=0) or self.material.get('advertisement_number')!=7):
            raise ValueError('Complete finite Ad7 shape parent/seed/material required')
        self.local={};self.keys=[];self.closed={};self.bounds=[]
        all_points=[[],[],[]]
        for key,record in self.material['contours'].items():
            points=np.asarray(record['canonical_print_uv'],float)*self.units
            if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
                raise ValueError('Finite canonical Ad7 SVG points required')
            all_points[_group(key)].append(points)
            self.keys.append(key);self.closed[key]=record['closed']
        centres=[]
        for items in all_points:
            if not items:raise ValueError('All three explicit Ad7 groups required')
            points=np.concatenate(items);centre=points.mean(axis=0)
            centres.append(centre)
            self.bounds.append(np.asarray([points.min(axis=0)-centre,points.max(axis=0)-centre]))
        self.centres=np.asarray(centres)
        for key in self.keys:
            self.local[key]=np.asarray(self.material['contours'][key]['canonical_print_uv'],float)*self.units-self.centres[_group(key)]

    def parameters(self, slots):
        slots=np.asarray(slots,float)
        if slots.shape!=(8,) or not np.isfinite(slots).all():
            raise ValueError('Finite complete eight centered Ad7 chart slots required')
        return self.seed+(slots-self.anchor)

    def _group_map(self,group,params,metrics,metric_D,points):
        width,height=metrics;ng=metric_D.shape[1]
        ui,vi,si=((0,1,2),(3,4,7),(5,6,7))[group]
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            scale=float(np.exp(params[si]))
        if not np.isfinite(scale) or scale<=0:
            raise Ad7ShapeDomainError('Ad7 grouped scale must remain finite and strictly positive')
        extent=scale*points/metrics
        uv=extent+params[[ui,vi]]
        D=np.zeros((len(points),2,ng))
        D-=extent[:,:,None]*metric_D[None,:,:]/metrics[None,:,None]
        # Physical state world tail, not global canonical column numbers.
        part=slice(ng-8,ng)
        D[:,0,part.start+ui]+=1.;D[:,1,part.start+vi]+=1.
        D[:,:,part.start+si]+=extent
        if not np.isfinite(uv).all() or not np.isfinite(D).all():
            raise Ad7ShapeDomainError('Ad7 grouped material leaves finite domain')
        return uv,D

    def paths(self,slots,metrics,metric_D):
        params=self.parameters(slots);values={};derivatives={}
        for key in self.keys:
            values[key],derivatives[key]=self._group_map(_group(key),params,metrics,metric_D,self.local[key])
        return values,derivatives

    def margins(self,slots,metrics,metric_D):
        params=self.parameters(slots);boxes=[];Dboxes=[]
        for group in range(3):
            value,D=self._group_map(group,params,metrics,metric_D,self.bounds[group])
            boxes.append(value);Dboxes.append(D)
        margin=[];derivative=[];labels=[]
        for group,(box,D) in enumerate(zip(boxes,Dboxes)):
            for axis in range(2):
                margin.extend((box[0,axis],1-box[1,axis]))
                derivative.extend((D[0,axis],-D[1,axis]))
                labels.extend((GROUPS[group]+':'+str(axis)+':lower',GROUPS[group]+':'+str(axis)+':upper'))
        for right in (1,2):
            margin.append(boxes[right][0,0]-boxes[0][1,0])
            derivative.append(Dboxes[right][0,0]-Dboxes[0][1,0]);labels.append('logo_left_of_'+GROUPS[right])
        margin.append(boxes[2][0,1]-boxes[1][1,1])
        derivative.append(Dboxes[2][0,1]-Dboxes[1][1,1]);labels.append('top_above_bottom')
        return np.asarray(margin),np.asarray(derivative),labels

    def guard(self,slots,metrics,metric_D):
        margin,D,labels=self.margins(slots,metrics,metric_D)
        bad=np.flatnonzero((margin[:12]<0).tolist()+ (margin[12:]<=0).tolist())
        if len(bad):raise Ad7ShapeDomainError('Ad7 grouped material domain failed: '+labels[int(bad[0])])
        return dict(passed=True,outer_canvas_containment=True,logo_left_and_two_rows_strict=True,
            nonzero_minimum_gap_or_arbitrary_scale_cap_added=False,minimum_margin=float(margin.min()),
            physical_axes='unfolded signed board station and vertical; no shear or projective',
            canonical_logo_proportions_retained=True,text_font_is_explicit_official_candidate=True)




def _unsupported_slots(counts):
    slots=[]
    if not counts['logo']:slots.extend((0,1,2))
    if not counts['top']:slots.extend((3,4))
    if not counts['bottom']:slots.extend((5,6))
    if not (counts['top'] or counts['bottom']):slots.append(7)
    return np.asarray(slots,int)


def _build_configuration(parent,indices,metrics,metric_D,*,material_reference,source_support_reference):
    """Common initializer; width/height may come from the saved real scene."""
    from .first_shot_print_layout import warp_normalized_material_uv
    if parent.get('identity',{}).get('ad7_grouped_shape_chart') is not None:
        raise ValueError('Restore the existing grouped chart for a grouped-model parent; do not reinterpret its slots as H8')
    material=read(material_reference)
    support=read(source_support_reference)
    if support.get('passed') is not True or support.get('model')!=material_reference or support.get('residual_selection') is not False:
        raise ValueError('Authenticated complete assigned-source group support required')
    if reference(support['inventory']['path'])!=support['inventory']:
        raise ValueError('Complete parent source support inventory changed')
    counts=support['assigned_print_points_by_group']
    if set(counts)!=set(GROUPS) or any(type(counts[k]) is not int or counts[k]<0 for k in GROUPS):
        raise ValueError('Complete nonnegative assigned-source counts required')
    fixed=_unsupported_slots(counts)
    physical=np.asarray(parent['physical_q'],float)
    indices=np.asarray(indices,int)
    if not _same(np.asarray(parent['q'])[indices],physical[indices]):
        raise ValueError('Ad7 parent slots must be identity under the existing derived chart')
    config=dict(schema=SCHEMA,producer=reference(__file__),parent_checkpoint=deepcopy(parent['checkpoint']),
        parent_q_sha256=_sha(parent['q']),parent_aux_count=len(parent['aux']),
        parent_expanded_q_sha256=_sha(parent['expanded_q']),
        seed_canvas_width_height_scene=np.asarray(metrics,float).tolist(),material=deepcopy(material_reference),
        parent_slot_anchor=physical[indices].tolist(),parent_global_slots=indices.tolist(),
        physical_seed8=[0.]*8,svg_axis_units=material['source_svg']['viewBox'][2:],
        parameter_names=list(PARAMETERS),groups=list(GROUPS),
        fixed_initialization_parameters=[PARAMETERS[j]for j in fixed],fixed_initialization_global_slots=indices[fixed].tolist(),
        source_support_reference=deepcopy(source_support_reference),assigned_source_points_by_group=deepcopy(counts),
        unsupported_group_positions_and_scales_are_candidates_not_measurements=True,
        all_parent_canonical_and_aux_bytes_retained=True,old_H8_not_a_physical_art_transform=True,
        official_full_footer_relative_size_or_spacing_measured=False,
        logo_scale_independent_of_common_text_scale=True,
        free_group_centres=[name for i,name in enumerate(GROUPS)if counts[name]>0],
        common_text_scale_is_an_explicit_official_font_candidate=True,
        common_unfolded_axes_and_no_rotation=True,new_material_model_predictions_not_parent_equivalent=True,
        seed_source='axis-aligned group least squares to inherited model only; not new source measurements')
    chart=Ad7GroupedShapeChart(config);original=[];local=[];seed=np.zeros(8)
    for group in range(3):
        keys=[key for key in chart.keys if _group(key)==group]
        proto=np.concatenate([np.asarray(material['contours'][key]['material_uv'],float)for key in keys])
        warped=warp_normalized_material_uv(proto,physical[indices])[0]
        centre=warped.mean(axis=0);i=(0,3,5)[group];seed[i:i+2]=centre
        points=np.concatenate([chart.local[key]for key in keys]);target=(warped-centre)*metrics
        original.append(target);local.append(points)
    logo=float(np.sum(local[0]*original[0])/np.sum(local[0]**2))
    text=float(sum(np.sum(x*y)for x,y in zip(local[1:],original[1:]))/sum(np.sum(x*x)for x in local[1:]))
    if min(logo,text)<=0:raise ValueError('Inherited model cannot seed a positive upright grouped art scale')
    seed[[2,7]]=np.log([logo,text]);config['physical_seed8']=seed.tolist()
    chart=Ad7GroupedShapeChart(config);chart.guard(physical[indices],metrics,metric_D)
    return config


def build_configuration(parent,physical_state,*,material_reference,source_support_reference):
    """Reuse an already constructed actual state; never read the source bundle."""
    physical=np.asarray(parent['physical_q'],float)
    geo=physical[physical_state.world_parameter_indices]
    metrics,D=canvas_metrics(physical_state,geo)
    config=_build_configuration(parent,physical_state.print_layout_global_indices_by_ad[7],
        metrics,D,material_reference=material_reference,source_support_reference=source_support_reference)
    config['seed_canvas_source']='existing actual public geometry size and its original units'
    return config


def build_configuration_from_scene(parent,scene,*,material_reference,source_support_reference,scene_reference=None):
    """Offline prepare from a same-parent accepted scene, with no new full state."""
    from .ad_landmarks import AdGeometry
    geometry=AdGeometry(scene,7)
    width=float(geometry._nominal_width)
    height=float(geometry.corner(geometry.initial,'left_top')[2]-geometry.corner(geometry.initial,'left_bottom')[2])
    if not np.isfinite([width,height]).all() or min(width,height)<=0:
        raise ValueError('Saved parent scene has no positive Ad7 unfolded canvas')
    domain=parent['identity'].get('print_H8_QP_domain',{}).get('domain')
    if domain is None:
        domain=json.loads(str(parent['aux']['print_H8_QP_domain_identity_json']))['domain']
    slots={tuple(item['full_state_columns'])for item in domain['full_original_vertex_ledger']
        if item['advertisement_id']==7}
    if len(slots)!=1 or len(next(iter(slots)))!=8:
        raise ValueError('Parent print domain must name one complete eight-slot Ad7 layout')
    config=_build_configuration(parent,np.asarray(next(iter(slots)),int),np.asarray([width,height]),
        np.zeros((2,8)),material_reference=material_reference,source_support_reference=source_support_reference)
    config['seed_canvas_source']='saved same-model accepted scene public AdGeometry station and height'
    if scene_reference is not None:
        if reference(scene_reference['path'])!=scene_reference:raise ValueError('Parent seed scene reference changed')
        config['seed_scene_reference']=deepcopy(scene_reference)
    return config


def verify_configuration_against_state(configuration_reference,parent,physical_state):
    """Strict floating agreement of offline-scene and current actual-state seeds."""
    saved=read(configuration_reference)
    expected=build_configuration(parent,physical_state,material_reference=saved['material'],source_support_reference=saved['source_support_reference'])
    for key in ('schema','producer','parent_checkpoint','parent_q_sha256','parent_aux_count',
                'parent_expanded_q_sha256','parent_global_slots','parent_slot_anchor','material','parameter_names','svg_axis_units',
                'fixed_initialization_parameters','fixed_initialization_global_slots','source_support_reference','assigned_source_points_by_group',
                'unsupported_group_positions_and_scales_are_candidates_not_measurements'):
        if saved.get(key)!=expected.get(key):raise ValueError('Saved grouped shape seed identity differs: '+key)
    differences={}
    for key in ('seed_canvas_width_height_scene','physical_seed8'):
        a,b=np.asarray(saved[key],float),np.asarray(expected[key],float)
        bound=128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(a),abs(b)))
        if a.shape!=b.shape or not np.isfinite(a).all() or np.any(abs(a-b)>bound):
            raise ValueError('Saved scene and actual-owner group seed differ beyond floating arithmetic: '+key)
        differences[key]=float(np.max(abs(a-b)))
    return dict(passed=True,parent_checkpoint=deepcopy(parent['checkpoint']),parent_aux_count=len(parent['aux']),
        seed_and_canvas_max_absolute_difference=differences,
        agreement='128 float64 epsilon times component magnitude; arithmetic comparison only, not a physical-domain relaxation')


def _chart(state):
    ref=getattr(state,'ad7_grouped_shape_reference',None)
    if ref is None:return None
    if _GLOBAL is None or _GLOBAL['reference']!=ref:raise ValueError('Install this saved Ad7 chart before model evaluation')
    return _GLOBAL['chart']


def _neutral(state,q):
    value=state._geometry(q).copy();value[state.print_layout_world_slices[7]]=0.
    return value


def templates_at(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['templates_at'](self,geometry)
    q=self._geometry(geometry);result=_ORIGINAL['templates_at'](self,_neutral(self,q))
    result=deepcopy(result);material=result['advertisements']['7']
    metric,D=canvas_metrics(self,q);values,_=chart.paths(q[self.print_layout_world_slices[7]],metric,D)
    for key,uv in values.items():material['contours'][key]['material_uv']=uv.tolist()
    material['grouped_shape_chart']=dict(configuration=deepcopy(self.ad7_grouped_shape_reference),
        parameters=chart.parameters(q[self.print_layout_world_slices[7]]).tolist(),
        no_shared_H8_applied_to_this_advertisement=True)
    return result


def scene(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['scene'](self,geometry)
    q=self._geometry(geometry);value=deepcopy(_ORIGINAL['scene'](self,_neutral(self,q)))
    value['ad7_grouped_shape_chart']=dict(configuration=deepcopy(self.ad7_grouped_shape_reference),
        producer=reference(__file__),parameter_names=list(PARAMETERS),
        physical_parameters=chart.parameters(q[self.print_layout_world_slices[7]]).tolist(),
        source_and_review_use_same_templates_at=True,other_advertisements_unchanged=True)
    return value


def guard_geometry(self,geometry):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['guard_geometry'](self,geometry)
    q=self._geometry(geometry);result=_ORIGINAL['guard_geometry'](self,_neutral(self,q))
    metric,D=canvas_metrics(self,q);proof=chart.guard(q[self.print_layout_world_slices[7]],metric,D)
    domains=deepcopy(result['shared_print_layout_domain']);domains['7']=dict(proof,
        configuration=deepcopy(self.ad7_grouped_shape_reference),old_H8_not_evaluated_as_Ad7_ink=True)
    return dict(result,shared_print_layout_domain=domains,ad7_grouped_shape_domain=proof)


def _ad7(target):
    from .first_shot_disjoint_print import KIND
    return target.get('advertisement_id')==7 and target.get('kind') in ('ad_print',KIND)


def _keys(target):
    from .first_shot_disjoint_print import KIND,PathUnion
    if target.get('kind')==KIND:return PathUnion.from_target(target).ids
    return (target['contour_key'],)


def print_layout_world_jacobians(self,geometry,descriptors):
    chart=_chart(self)
    if chart is None:return _ORIGINAL['print_layout_world_jacobians'](self,geometry,descriptors)
    q=self._geometry(geometry);paths=self.world_paths(q,descriptors)
    result=[np.zeros(path.shape+(len(q)-self.geometry_count,))for path in paths]
    other=[i for i,target in enumerate(descriptors)if not _ad7(target)]
    if other:
        old=_ORIGINAL['print_layout_world_jacobians'](self,_neutral(self,q),[descriptors[i]for i in other])
        for i,value in zip(other,old):result[i]=value
    metric,Dmetric=canvas_metrics(self,q)
    values,D=chart.paths(q[self.print_layout_world_slices[7]],metric,Dmetric)
    h=float(self.provider.fd_step)
    for i,target in enumerate(descriptors):
        if not _ad7(target):continue
        parts=[]
        for key in _keys(target):
            uv=values[key];lift=np.empty((len(uv),3,2))
            for axis in range(2):
                low=uv.copy();high=uv.copy();low[:,axis]-=h;high[:,axis]+=h
                a=self.provider.current.material(7,low[:,0],1-low[:,1],q[:100])[0]
                b=self.provider.current.material(7,high[:,0],1-high[:,1],q[:100])[0]
                lift[:,:,axis]=(b-a)/(2*h)
            value=np.einsum('nci,nip->ncp',lift,D[key][...,-8:])
            if chart.closed[key] and not np.array_equal(uv[0],uv[-1]):value=np.concatenate((value,value[:1]))
            parts.append(value)
        value=np.concatenate(parts)
        if value.shape!=paths[i].shape+(8,):raise ValueError('Ad7 group world/J path inventory differs')
        result[i][...,self.print_layout_tail_slices[7]]=value
    return tuple(result)


def install_execution(configuration_reference):
    global _GLOBAL
    if _GLOBAL is not None:
        if _GLOBAL['reference']!=configuration_reference:raise ValueError('Another Ad7 grouped model is installed')
        return deepcopy(_GLOBAL['receipt'])
    config=read(configuration_reference)
    if config.get('producer')!=reference(__file__):raise ValueError('Ad7 grouped chart producer changed')
    checkpoint=config['parent_checkpoint']
    if reference(checkpoint['path'])!=checkpoint:raise ValueError('Grouped shape parent checkpoint changed')
    with np.load(checkpoint['path'],allow_pickle=False)as saved:
        q=saved['q'];indices=np.asarray(config['parent_global_slots'],int)
        if _sha(q)!=config['parent_q_sha256'] or not _same(q[indices],np.asarray(config['parent_slot_anchor'],float)):
            raise ValueError('Complete canonical group-chart parent differs')
    from . import first_shot_disjoint_print as disjoint
    if disjoint._CONFIG is None or disjoint._CONFIG['manifest']['material_override']!=config['material']:
        raise ValueError('Install the same explicit disjoint material model before its grouped shape chart')
    chart=Ad7GroupedShapeChart(config)
    from .first_shot_physical_state import FirstShotPhysicalState
    for name in ('templates_at','scene','guard_geometry','print_layout_world_jacobians'):
        _ORIGINAL[name]=getattr(FirstShotPhysicalState,name)
        setattr(FirstShotPhysicalState,name,globals()[name])
    receipt=dict(schema=SCHEMA,producer=reference(__file__),configuration=deepcopy(configuration_reference),
        material=deepcopy(config['material']),parent_checkpoint=deepcopy(checkpoint),
        all_parent_canonical_and_aux_retained=True,new_physical_material_model=True,
        old_source_or_world_J_compatible=False,source_UV_ID_W_PTS_and_rowtime_unchanged=True,
        official_footer_spacing_not_claimed_measured=True,per_glyph_or_per_frame_parameters_added=False,
        fixed_initialization_parameters=deepcopy(config['fixed_initialization_parameters']),
        fixed_initialization_global_slots=deepcopy(config['fixed_initialization_global_slots']),
        source_support_reference=deepcopy(config['source_support_reference']),
        assigned_source_points_by_group=deepcopy(config['assigned_source_points_by_group']),
        unsupported_group_positions_and_scales_are_candidates_not_measurements=True)
    _GLOBAL=dict(reference=deepcopy(configuration_reference),chart=chart,receipt=receipt)
    return deepcopy(receipt)


def install_state(state):
    from .first_shot_disjoint_print import _physical_states
    changed=[]
    for actual in _physical_states(state):
        if getattr(actual,'ad7_grouped_shape_reference',None)==_GLOBAL['reference']:continue
        for key in _GLOBAL['chart'].keys:
            expected=_GLOBAL['chart'].material['contours'][key]['canonical_print_uv']
            if not _same(np.asarray(actual.templates['advertisements']['7']['contours'][key]['canonical_print_uv']),np.asarray(expected)):
                raise ValueError('Actual Ad7 bank differs from the grouped chart material')
        actual.ad7_grouped_shape_reference=deepcopy(_GLOBAL['reference'])
        actual._scene_cache.clear();actual._world_cache.clear();changed.append(type(actual).__name__)
    if not changed and not list(_physical_states(state)):raise ValueError('No actual physical state for Ad7 chart')
    return changed


def source_worker_init(*args,configuration_reference,original_initializer):
    original_initializer(*args)
    install_execution(configuration_reference)


def geometry_worker_init(*args,configuration_reference,original_initializer):
    original_initializer(*args)
    install_execution(configuration_reference)
    from . import first_shot_fit as fit
    install_state(fit._geometry_worker[0])


def install_successor(owner,*,parent,configuration_reference,entry_reference):
    """After disjoint/source restoration and H8-QP installation, before pools."""
    programme=owner.programme
    if programme.pool is not None or programme.bundle._geometry_pool is not None or programme.bundle._source_pool is not None:
        raise ValueError('Install grouped chart before all pools open')
    if getattr(programme,'initial_J_manifest',None) is not None or getattr(programme,'_initial_complete_J_reuse',False):
        raise ValueError('Old source/J replay cannot represent the grouped material successor')
    before=owner.start.copy();aux={key:np.asarray(value).copy()for key,value in owner.checkpoint_arrays().items()}
    startup=verify_configuration_against_state(configuration_reference,parent,actual_physical_state(programme.state))
    receipt=install_execution(configuration_reference)
    receipt.update(configuration_startup_check=startup,assigned_source_support=audit_source_support(programme.bundle.frames),entry=deepcopy(entry_reference),changed_physical_states=install_state(programme.state)+install_state(programme.bundle.state))
    if not hasattr(owner,'_print_H8_domain'):raise ValueError('Explicit original H8 QP domain must precede its Ad7-only replacement')
    domain=owner._print_H8_domain
    old_constraints=owner.linear_constraints;old_identity=owner.checkpoint_identity;old_arrays=owner.checkpoint_arrays
    original_ad7_rows=[]
    for item in domain.ledger:
        if item['advertisement_id']==7:original_ad7_rows.extend(range(*item['row_slice']))
    if not original_ad7_rows:raise ValueError('Original QP domain does not declare Ad7 rows')
    def constraints(this,q,selected):
        old=old_constraints(q,selected);report=old['report'];first,stop=report['appended_print_H8_row_slice']
        if stop-first!=domain.matrix.shape[0]:raise ValueError('Original QP row slice changed')
        keep=np.ones(len(old['lower']),bool);keep[first+np.asarray(original_ad7_rows,int)]=False
        physical,chain=this.effective_q_and_chain(q)
        state=actual_physical_state(programme.state)
        geo=physical[state.world_parameter_indices];metric,Dmetric=canvas_metrics(state,geo)
        margin,D,labels=_GLOBAL['chart'].margins(geo[state.print_layout_world_slices[7]],metric,Dmetric)
        pulled=csr_matrix(D)@csr_matrix(chain)[state.world_parameter_indices]
        pulled.eliminate_zeros()
        outside=np.setdiff1d(np.arange(this.parameter_count),selected)
        if pulled[:,outside].nnz:raise ValueError('Grouped print domain has an undeclared nonshared dependency')
        matrix=vstack((csr_matrix(old['matrix'])[keep],pulled[:,selected]),format='csr')
        return dict(old,matrix=matrix,lower=np.r_[old['lower'][keep],-margin],report=dict(report,
            ad7_old_H8_rows_replaced=original_ad7_rows,ad7_grouped_domain_rows=labels,
            other_original_halfspaces_preserved=True,closed_QP_relaxation_actual_strict_guard_still_required=True))
    archive='ad7_groups_'+configuration_reference['sha256'][:16]+'_json'
    metadata=np.asarray(json.dumps(receipt,sort_keys=True))
    def arrays(this):
        result=old_arrays()
        if archive in result and not _same(result[archive],metadata):raise ValueError('Historical grouped chart aux conflict')
        return dict(result,**{archive:metadata.copy()})
    def identity(this):return dict(old_identity(),ad7_grouped_shape_chart=deepcopy(receipt),ad7_grouped_aux_key=archive)
    owner.linear_constraints=MethodType(constraints,owner);owner.checkpoint_arrays=MethodType(arrays,owner);owner.checkpoint_identity=MethodType(identity,owner)
    refs=[reference(__file__),deepcopy(configuration_reference),deepcopy(entry_reference),
        deepcopy(_GLOBAL['chart'].configuration['material']),deepcopy(_GLOBAL['chart'].configuration['source_support_reference']),
        deepcopy(read(_GLOBAL['chart'].configuration['source_support_reference'])['inventory'])]
    for ref in refs:
        if reference(ref['path'])!=ref:raise ValueError('Grouped chart execution reference changed')
        for obj,name in ((owner,'source_producers'),(owner,'_references'),(programme,'source_producers')):
            values=getattr(obj,name)
            if ref not in values:values.append(deepcopy(ref))
    import native_transverse_bundle as runtime
    from . import first_shot_transverse_execution as context
    runtime.init_worker=partial(source_worker_init,configuration_reference=deepcopy(configuration_reference),original_initializer=runtime.init_worker)
    context._init_context_worker=partial(geometry_worker_init,configuration_reference=deepcopy(configuration_reference),original_initializer=context._init_context_worker)
    from .first_shot_disjoint_print import _clear_caches
    _clear_caches(programme)
    if not _same(owner.start,before):raise ValueError('Grouped material successor changed canonical parent')
    for key,value in aux.items():
        if not _same(owner.checkpoint_arrays()[key],value):raise ValueError('Grouped material successor changed parent aux: '+key)
    return deepcopy(receipt)


def actual_physical_state(state):
    """Resolve the first existing full physical state behind native wrappers."""
    from .first_shot_disjoint_print import _physical_states
    values=list(_physical_states(state))
    if not values:raise ValueError('No full physical state behind the native owner')
    return values[0]


def verify_parent_continuation(owner,parent):
    """Check a final full parent before any source solve under the new model."""
    if not _same(owner.start,parent['q']):raise ValueError('Complete successor canonical parent changed')
    aux=owner.checkpoint_arrays()
    for key,value in parent['aux'].items():
        if key not in aux or not _same(aux[key],value):raise ValueError('Successor parent auxiliary changed: '+key)
    expanded,_=owner.expanded_q_and_chain(owner.start)
    physical,_=owner.effective_q_and_chain(owner.start)
    if not _same(expanded,parent['expanded_q']) or not _same(physical,parent['physical_q']):
        raise ValueError('Complete inherited time/derived physical parameter mapping changed')
    kwargs=owner.solver_kwargs()
    if kwargs['initial_damping']!=parent['damping'] or kwargs['initial_line_search_fraction']!=parent['fraction']:
        raise ValueError('Accepted parent numerical search state changed')
    from .first_shot_row_camera import prepare_frame_row_camera
    state=owner.programme.state;before=np.asarray(parent['physical_q'])[:owner.programme_count]
    after=physical[:owner.programme_count];rows=np.linspace(0.,1079.,5)
    for index in range(1,1249):
        a=prepare_frame_row_camera(state,before,index).camera25_at_rows(rows)
        b=prepare_frame_row_camera(state,after,index).camera25_at_rows(rows)
        if not _same(a,b):raise ValueError('Inherited native row-time camera changed at frame '+str(index))
    return dict(passed=True,canonical_q_sha256=_sha(owner.start),parent_aux_count=len(parent['aux']),
        all_complete_canonical_aux_expanded_physical_arrays_byte_exact=True,
        native_exposures=1248,rows_per_exposure=5,camera_parameters_per_row=25,
        all_row_time_cameras_byte_exact=True,initial_damping=parent['damping'],initial_fraction=parent['fraction'],
        art_model_predictions_equivalent_to_parent=False,old_source_or_world_J_reused=False)


def audit_source_support(frames):
    """Assigned target support only; neither rank nor shape accuracy is inferred."""
    rows=[0,0,0];indices=[set(),set(),set()]
    for frame in frames:
        for row in frame['rows']:
            if not _ad7(row['target']):continue
            for group in set(map(_group,_keys(row['target']))):
                rows[group]+=1;indices[group].add(int(frame['index']))
    return dict(groups={name:dict(assigned_target_rows=rows[i],frame_indices=sorted(indices[i]))
        for i,name in enumerate(GROUPS)},support_is_not_rank_or_accuracy=True,
        free_fixed_policy_is_explicit_in_group_configuration=True)


def finalize_stage(stage,*,configuration_reference):
    """After frozen time-detail construction, freeze source-unsupported slots.

    The original uncompressed constructor contract stays 7700/172/3784. This
    explicit successor then removes declared candidate slots from the solve;
    all old constraint rows are pulled back before their fixed columns vanish.
    """
    config=read(configuration_reference)
    if _GLOBAL is None or _GLOBAL['reference']!=configuration_reference:
        raise ValueError('Install the matching grouped chart before finalizing its stage')
    fixed=_unsupported_slots(config['assigned_source_points_by_group'])
    if config['fixed_initialization_parameters']!=[PARAMETERS[j]for j in fixed]:
        raise ValueError('Fixed parameter declaration differs from complete assigned-source support')
    columns=np.asarray(config['fixed_initialization_global_slots'],int)
    expected=np.asarray(config['parent_global_slots'],int)[fixed]
    if not np.array_equal(columns,expected) or len(np.unique(columns))!=len(columns):
        raise ValueError('Grouped initialization-only column declaration differs')
    if getattr(stage,'ad7_initialization_fixed_receipt',None) is not None:
        return deepcopy(stage.ad7_initialization_fixed_receipt)
    before=stage.start.copy();old_shared=stage.shared.copy();old_fixed=stage.fixed.copy()
    if not np.isin(columns,old_shared).all() or np.isin(columns,old_fixed).any():
        raise ValueError('Initialization-only slots must be existing shared and previously unfixed coordinates')
    if not _same(before[columns],np.asarray(config['parent_slot_anchor'],float)[fixed]):
        raise ValueError('Initialization-only slots changed before freezing')
    support=audit_source_support(stage.programme.bundle.frames)
    for name in GROUPS:
        if bool(support['groups'][name]['assigned_target_rows'])!=bool(config['assigned_source_points_by_group'][name]):
            raise ValueError('Live assigned-source group support differs from sealed complete inventory: '+name)
    original_identity=stage.checkpoint_identity
    stage.fixed=np.r_[old_fixed,columns]
    stage.shared=old_shared[~np.isin(old_shared,columns)]
    stage._fixed_bits=stage.start[stage.fixed].view(np.uint64).copy()
    receipt=dict(configuration=deepcopy(configuration_reference),parameters=deepcopy(config['fixed_initialization_parameters']),
        global_slots=columns.tolist(),fixed_canonical_values=before[columns].tolist(),
        physical_initialization_parameters=_GLOBAL['chart'].seed[fixed].tolist(),
        initialization_is_candidate_not_measured=True,not_fitted_by_loss_or_QP=True,
        assigned_source_support=support,
        all_parent_q_and_aux_bytes_retained=True)
    def constraints(this,q,selected):
        if not np.array_equal(selected,this.shared):raise ValueError('Exact grouped successor shared order required')
        expanded,chain=this.expanded_q_and_chain(q)
        old=this.base.linear_constraints(expanded,old_shared.copy())
        matrix=csr_matrix(old['matrix'])
        if matrix.shape!=(len(old['lower']),len(old_shared)):
            raise ValueError('Complete original shared constraint dimensions changed')
        pulled=(matrix@chain[old_shared]).tocsr();pulled.eliminate_zeros()
        allowed=np.r_[this.shared,columns]
        outside=np.setdiff1d(np.arange(this.parameter_count),allowed)
        if pulled[:,outside].nnz:raise ValueError('Fixed-centre reduction has an undeclared constraint dependency')
        return dict(old,matrix=pulled[:,this.shared].tocsr(),report=dict(old.get('report',{}),
            grouped_initialization_only_fixed_columns=columns.tolist(),
            original_lower_and_all_halfspace_rows_retained=True))
    def identity(this):
        return dict(original_identity(),ad7_initialization_only_fixed=deepcopy(receipt))
    stage.linear_constraints=MethodType(constraints,stage)
    stage.checkpoint_identity=MethodType(identity,stage)
    stage.ad7_initialization_fixed_receipt=deepcopy(receipt)
    # Existing time-detail checkpoint arrays hold every inherited parent value;
    # this separate short new key records the changed free/fixed model contract.
    archive='ad7_initfixed_'+configuration_reference['sha256'][:16]+'_json'
    if archive in stage._aux:raise ValueError('Grouped fixed-centre auxiliary archive already exists')
    stage._aux[archive]=np.asarray(json.dumps(receipt,sort_keys=True))
    if not _same(stage.start,before):raise ValueError('Fixing initialization-only centres changed canonical q')
    return deepcopy(receipt)
