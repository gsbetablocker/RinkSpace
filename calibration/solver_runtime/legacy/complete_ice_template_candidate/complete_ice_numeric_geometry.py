"""Coupled missing ice markings on the actual current world43/46.

Existing world/source geometry is delegated unchanged. New marks are explicit
engineering/paired hypotheses, not observations. No graph is installed and
no pixel, sigma, role or scene scale is invented. Seven *shared* candidate
coordinates describe two trapezoid spans and five curling relations; no
independent per-circle/per-sheet drift coordinates are introduced.
"""
from dataclasses import dataclass,asdict
from copy import deepcopy
import numpy as np

SCHEMA='rinkspace.uninstalled-complete-coupled-ice-markings-geometry.v1'
EXTRA_NAMES=(
    'goal_trapezoid_goal_outer_span_delta_m','goal_trapezoid_board_outer_span_delta_m',
    'curling_hog_inside_offset_delta_m','curling_back_house_tangency_delta_m',
    'curling_sheet_half_width_delta_m','curling_sheet_end_outside_tee_delta_m',
    'curling_centerline_extension_outside_tee_delta_m')


@dataclass(frozen=True)
class EngineeringTemplateConfig:
    trapezoid_goal_outer_span_m:float=6.80
    trapezoid_board_outer_span_m:float=8.60
    hog_inside_from_tee_m:float=6.401
    sheet_width_reference_m:float=4.750
    sheet_end_outside_tee_m:float=(45.720-34.750)/2.
    centerline_extension_outside_tee_m:float=3.658
    hack_line_length_reference_m:float=.457
    reference_version:str='IIHF2026/27 + WorldCurling2026; actual venue version not measured'
    # Explicit caller input only, not source sigma or automatically installed
    # soft-reference residuals. None means tolerance has not been assigned.
    engineering_tolerance_m:float|None=None


def _unit(x):
    x=np.asarray(x,float);n=float(np.linalg.norm(x))
    if not np.isfinite(x).all()or n<=0.:raise ValueError('Finite nonzero physical direction required')
    return x/n


def _frame(features,name):
    points=np.asarray(features[name]['segments_xyz'],float)
    if points.shape!=(1,2,3):raise ValueError('Actual single finite goal segment required')
    centre=points[0].mean(axis=0);v=_unit(points[0,1]-points[0,0])
    if v[1]<0.:v=-v
    inward=np.array([-v[1],v[0],0.])
    if inward@(np.asarray(features['C0']['center_xyz'])-centre)<0.:inward=-inward
    return centre,inward,v


def _boundary_station(start,direction,polygon):
    p=np.asarray(polygon,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<3:raise ValueError('Actual current closed ice boundary required')
    if np.array_equal(p[0],p[-1]):p=p[:-1]
    a=p[:,:2];edge=np.roll(a,-1,axis=0)-a;u=np.asarray(direction)[:2]
    rhs=a-np.asarray(start)[:2];den=u[0]*edge[:,1]-u[1]*edge[:,0]
    good=abs(den)>1e-14;t=np.full(len(p),np.inf);v=np.full(len(p),np.inf)
    t[good]=(rhs[good,0]*edge[good,1]-rhs[good,1]*edge[good,0])/den[good]
    v[good]=(rhs[good,0]*u[1]-rhs[good,1]*u[0])/den[good]
    possible=np.flatnonzero(good&(t>=0.)&(v>=0.)&(v<=1.))
    if not len(possible):raise ValueError('Current goal outward axis does not reach original boundary')
    k=int(possible[np.argmin(t[possible])]);length=np.linalg.norm(edge,axis=1)
    if np.any(length<=0.):raise ValueError('Nondegenerate current closed course required')
    cumulative=np.r_[0.,np.cumsum(length)]
    station=float(cumulative[k]+v[k]*length[k])
    def at(s):
        s=float(s%cumulative[-1]);i=min(int(np.searchsorted(cumulative,s,side='right')-1),len(p)-1)
        return np.r_[a[i]+((s-cumulative[i])/length[i])*edge[i],0.]
    return station,at


def _boundary_hit(start,direction,polygon):
    station,at=_boundary_station(start,direction,polygon)
    return at(station)


class CompleteIceTemplateGeometry:
    def __init__(self,original,*,config=None):
        if getattr(original,'parameter_count',None)not in(43,46):
            raise ValueError('Explicit current hockey43/46 geometry required; no legacy shape truncation')
        self.original=original;self.config=config or EngineeringTemplateConfig()
        if not isinstance(self.config,EngineeringTemplateConfig):raise TypeError('Explicit engineering configuration required')
        self.base_count=original.parameter_count;self.parameter_count=self.base_count+7
        self.scene_unit_m=float(original.scene_unit_m);self.base=original.base
        self.parameter_names=tuple(original.parameter_names)+EXTRA_NAMES
        self.parameter_units=tuple(original.parameter_units)+('m',)*7
        self.initial=np.r_[original.initial,np.zeros(7)]

    def __getattr__(self,name):
        old=self.__dict__.get('original')
        if old is None:raise AttributeError(name)
        return getattr(old,name)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():
            raise ValueError('Complete explicitly declared current+7 template geometry required')
        return q

    def validate_parameters(self,q):
        q=self._q(q);self.original.validate_parameters(q[:self.base_count])
        c=self.config;d=q[self.base_count:]
        values=np.array([c.trapezoid_goal_outer_span_m+d[0],c.trapezoid_board_outer_span_m+d[1],
            c.hog_inside_from_tee_m+d[2],c.sheet_width_reference_m/2+d[4],
            c.sheet_end_outside_tee_m+d[5],c.centerline_extension_outside_tee_m+d[6]])
        if np.any(values<=0.)or values[1]<=values[0]:raise ValueError('Positive coupled physical template dimensions and outward trapezoid required')
        if c.engineering_tolerance_m is not None and (not np.isfinite(c.engineering_tolerance_m)or c.engineering_tolerance_m<=0.):
            raise ValueError('Caller engineering tolerance must be positive; not inferred from native source sigma')
        return q

    def _missing(self,scene,d):
        unit=self.scene_unit_m;c=self.config;fs={f['id']:f for f in scene['ice_features']}
        out=[]
        def metadata(fid,kind,relation,parents,edge=None):
            return dict(id=fid,label=fid,surface_id='ice_surface',kind=kind,
                plane_normal_xyz=[0.,0.,1.],geometry_status='unobserved_coupled_engineering_template_candidate',
                geometry_relation=relation,parent_feature_ids=list(parents),
                source=dict(new_observation=False,UV_count=0,measured_sigma_px=None,venue_standard_version_verified=False),
                engineering_reference=dict(version=c.reference_version,tolerance_m=c.engineering_tolerance_m,
                    tolerance_is_not_measured_source_sigma=True,edge_definition=edge),
                style=dict(fill_srgb=None,line_srgb='AE508C'if fid.startswith('curling_')else'C00000',line_width_emu=19050,dash='solid'))
        def line(fid,p0,p1,relation,parents,edge=None):
            f=metadata(fid,'polyline',relation,parents,edge);f['segments_xyz']=[[np.asarray(p0).tolist(),np.asarray(p1).tolist()]];out.append(f)
        def circle(fid,centre,radius,relation,parents):
            f=metadata(fid,'circle',relation,parents,'paired inherited circle material, not new pixel evidence')
            f.update(center_xyz=np.asarray(centre).tolist(),radius=float(radius));out.append(f)
        # Left circles share the actual observed right pair's goal-local
        # station/stagger/midpoint/separation/radius, with an explicit half-turn.
        right,ri,rv=_frame(fs,'goal_line_right');left,li,lv=_frame(fs,'goal_line_left')
        for new,name in (('A3','A2'),('A4','A1')):
            delta=np.asarray(fs[name]['center_xyz'])-right
            centre=left+(delta@ri)*li-(delta@rv)*lv
            circle(new,centre,fs[name]['radius'],'half-turn goal-local paired counterpart; no independent left-circle coordinates',('goal_line_left','goal_line_right',name))
        # Spans are transverse distances in the current goal frame, not board
        # arclengths. Both board endpoints lie on the unchanged current closed
        # boundary. No nominal +/-30m endboard or arena resize is introduced.
        for side in ('left','right'):
            goal,inward,v=_frame(fs,'goal_line_'+side)
            for sign,label in ((-1.,'lower'),(1.,'upper')):
                a=goal+sign*v*(c.trapezoid_goal_outer_span_m+d[0])/(2*unit)
                board_seed=goal+sign*v*(c.trapezoid_board_outer_span_m+d[1])/(2*unit)
                b=_boundary_hit(board_seed,-inward,scene['ice_boundary']['points_xyz'])
                line('goal_trapezoid_'+side+'_'+label+'_outer',a,b,
                    'outside marking transverse spans in current goal frame; board endpoint on unchanged actual boundary',
                    ('goal_line_'+side,'ice_boundary'),'outside marking edge; IIHF rule1.8 6.80m/8.60m nominal')
        # Complete curling carriers use each shared F/R tee pair. The old Lane
        # three-sided paint rectangles are preserved, never renamed sheet edges.
        for row in range(1,5):
            f=fs['F'+str(row)];r=fs['R'+str(row)]
            fc=np.asarray(f['center_xyz']);rc=np.asarray(r['center_xyz']);axis=_unit(rc-fc)
            across=np.array([-axis[1],axis[0],0.]);hw=(c.sheet_width_reference_m/2+d[4])/unit
            parents=('F'+str(row),'R'+str(row));prefix='curling_sheet'+str(row)+'_'
            for name,tee,inward,house in (('F',fc,axis,f),('R',rc,-axis,r)):
                def transverse(suffix,centre,relation,edge):
                    line(prefix+suffix+'_'+name,centre-hw*across,centre+hw*across,relation,parents,edge)
                transverse('tee_center',tee,'tee line crosses shared actual house centre','tee line centre; maximum width13mm, actual paint width not measured')
                transverse('hog_inside',tee+inward*(c.hog_inside_from_tee_m+d[2])/unit,
                    'one common inward hog distance for all eight ends','inside edge of102mm-wide hog line; not centre or outside edge')
                back=float(house['radius'])+d[3]/unit
                if back<=0.:raise ValueError('Positive outside backline/house relation required')
                transverse('back_outside',tee-inward*back,
                    'common house outer radius plus shared backline tangency deviation','outside back-line edge, nominal tee distance1.829m')
                transverse('sheet_end_reference',tee-inward*(c.sheet_end_outside_tee_m+d[5])/unit,
                    'shared engineered end extension; actual backboard not source-qualified','sheet backboard reference, not Lane right_short or unmeasured left polygon vertex')
                hack=tee-inward*(c.centerline_extension_outside_tee_m+d[6])/unit
                halfhack=c.hack_line_length_reference_m/(2*unit)
                line(prefix+'hack_reference_'+name,hack-halfhack*across,hack+halfhack*across,
                    'hack line at extended centreline end; common nominal457mm carrier',parents,
                    'engineering hack-line length reference, not a measured venue feature')
            extension=(c.centerline_extension_outside_tee_m+d[6])/unit
            line(prefix+'centerline_reference',fc-axis*extension,rc+axis*extension,
                'one straight axis through both actual tee centres, extended equally',parents,
                'centreline centre; row3 existing source-seen finite carrier remains separate and unchanged')
            end=(c.sheet_end_outside_tee_m+d[5])/unit
            for sign,label in ((-1.,'lower'),(1.,'upper')):
                line(prefix+'sideline_reference_'+label,fc-axis*end+sign*across*hw,rc+axis*end+sign*across*hw,
                    'common engineered half-width about the F/R row axis; not current Lane paint',parents,
                    'inside sideline reference; maximum4.750m is not a measured equality')
        return out

    def scene(self,q):
        q=self.validate_parameters(q);scene=self.original.scene(q[:self.base_count])
        new=self._missing(scene,q[self.base_count:]);old_ids={f['id']for f in scene['ice_features']}
        if any(f['id']in old_ids for f in new):raise ValueError('New reference geometry must never overwrite actual source geometry')
        scene['ice_features'].extend(new);scene['complete_ice_template_candidate']=self.manifest(q)
        return scene

    source_scene=scene

    def feature_state(self,q):
        s=self.scene(q)
        return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in s['ice_features']},
            ice_boundary=s['ice_boundary'],parameters=s['parameters'],world_geometry_manifest=self.manifest(q))

    def new_material_world(self,feature_id,q,material_t):
        t=np.asarray(material_t,float).reshape(-1)
        if not np.isfinite(t).all()or np.any(t<0.)or np.any(t>1.):raise ValueError('Declared finite material t in[0,1] required; never inferred from source UV')
        q=self.validate_parameters(q);scene=self.original.scene(q[:self.base_count])
        new={f['id']:f for f in self._missing(scene,q[self.base_count:])}
        if feature_id not in new:raise ValueError('Only explicitly generated new template feature supported')
        f=new[feature_id]
        if f['kind']=='circle':
            return np.asarray(f['center_xyz'])+f['radius']*np.c_[np.cos(2*np.pi*t),np.sin(2*np.pi*t),np.zeros(len(t))]
        a,b=np.asarray(f['segments_xyz'])[0]
        return a[None,:]+t[:,None]*(b-a)[None,:]

    def new_material_world_jacobian(self,feature_id,q,material_t,*,steps=None):
        q=self.validate_parameters(q);world=self.new_material_world(feature_id,q,material_t)
        widths=np.asarray(steps if steps is not None else [1e-6]*self.parameter_count,float)
        if widths.shape!=(self.parameter_count,)or np.any(widths<=0.):raise ValueError('Explicit positive complete parameter widths required')
        j=np.empty((*world.shape,self.parameter_count))
        for col,h in enumerate(widths):
            p=q.copy();m=q.copy();p[col]+=h;m[col]-=h
            j[:,:,col]=(self.new_material_world(feature_id,p,material_t)-self.new_material_world(feature_id,m,material_t))/(2*h)
        return world,j

    def manifest(self,q):
        q=self._q(q)
        return dict(schema=SCHEMA,installed_in_graph=False,old_parameter_count=self.base_count,
            parameter_count=self.parameter_count,additional_shared_parameter_names=list(EXTRA_NAMES),
            additional_shared_parameter_values_m=q[self.base_count:].tolist(),engineering_configuration=asdict(self.config),
            old_source_world_feature_names_and_native_observations_unchanged=True,
            new_FIT_pixels=0,new_measured_sigma_or_hard_nominal_prior=False,
            rink_dimensions_60x30_are_only_reference_current_boundary_unchanged=True,
            no_lane_side_relabel_as_hog_back_or_sheet_end=True,
            observed_right_A_pair_drives_unobserved_left_A_pair=True,
            no_independent_new_circle_or_per_sheet_drift_parameters=True,
            metric_scale_error_is_shared_correlated_not_independent_per_marker=True,
            modern_goal_crease_shape_not_installed_old61knot_templates_retained=True,
            neutral_faceoff_spots_hashmarks_and_venue_specific_paint_widths_not_installed=True,
            whole_graph_alias_profile_guard_source_binding_and_fresh_gate_still_required=True)

    def graph_descriptor(self,*,original_physical_count=34245):
        if self.base_count!=43 or original_physical_count!=34245:raise ValueError('This proposed chart only describes existing actual K20 core34245')
        count=original_physical_count+7
        return dict(schema='rinkspace.uninstalled-complete-ice-marking-chart-proposal.v1',
            old_core_count=original_physical_count,candidate_core_count=count,
            candidate_new_geometry_indices=list(range(original_physical_count,count)),
            candidate_bias_indices=list(range(count,count+3)),candidate_high25_indices=list(range(count+3,count+28)),
            candidate_full_count=count+28,new_world_parameter_count=self.parameter_count,
            additional_pixel_source_rows=0,installed=False,
            unobserved_new_free_columns_need_explicit_reference_or_qualified_source_before_a_solvable_graph=True,
            cannot_resume_fullq_by_unvalidated_truncation=True)


def current_trapezoid_side_segments(original,current_world_q,*,
        goal_carrier_span_m,board_carrier_span_m):
    """Four current coupled carrier segments, for caller's actual source replay.

    The two spans are explicit physical inputs, not fixed measured IIHF
    numbers. This function does not equate the user's exact visible pixel
    endpoints to the physical goal/board endpoints, assign source material t,
    or infer which signed world side is upper/lower in the image. Official
    outer-paint spans cannot identify a centre/outer edge of a different source
    without separate qualification. Goal station/yaw follow current world43.
    """
    geometry=CompleteIceTemplateGeometry(original)
    qw=np.asarray(current_world_q,float)
    if qw.shape!=(geometry.base_count,):raise ValueError('Actual declared complete current hockey world43/46 required')
    if not np.isfinite([goal_carrier_span_m,board_carrier_span_m]).all():
        raise ValueError('Finite explicit caller carrier spans required')
    q=np.r_[qw,np.zeros(7)]
    q[geometry.base_count]=float(goal_carrier_span_m)-geometry.config.trapezoid_goal_outer_span_m
    q[geometry.base_count+1]=float(board_carrier_span_m)-geometry.config.trapezoid_board_outer_span_m
    geometry.validate_parameters(q)
    scene=original.scene(qw)
    features={f['id']:f for f in geometry._missing(scene,q[geometry.base_count:])}
    result={}
    for side in ('left','right'):
        for sign,label in ((-1,'lower'),(1,'upper')):
            fid='goal_trapezoid_'+side+'_'+label+'_outer'
            result['goal_trapezoid_'+side+'_side_'+('minus'if sign<0 else'plus')]=dict(
                world_scene=np.asarray(features[fid]['segments_xyz'],float)[0].copy(),
                scene_unit_m=geometry.scene_unit_m,corresponding_template_id=fid,
                goal_frame_transverse_sign=sign,
                physical_t0='current goal-line carrier',physical_t1='current endboard ice boundary',
                edge_definition='caller-qualified source carrier; official outside-edge reference not silently imposed',
                image_upper_lower_identity_not_inferred=True,
                visible_source_endpoints_not_assigned_material_t=True,
                goal_carrier_span_m=float(goal_carrier_span_m),board_carrier_span_m=float(board_carrier_span_m))
    return result
