"""Actual all45/26 FIT, one 3D camera, currentworld47 and USER finite bands.

This small experiment is deliberately separate from all programme solvers.
No HELD is loaded into the objective. Every accepted trial must satisfy all
268 original authoritative pen supports on current finite world trapezoids.
"""
from calibration.solver_runtime.config import SOURCE_ROOT as _runtime_source_root, PROGRAM_SOURCE_ROOT as _runtime_program_source_root
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import sys,os,time,json,hashlib,argparse
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
B=_runtime_data_root();sys.path.insert(0,str(_runtime_source_root));sys.dont_write_bytecode=True
from simple_wide_perspective_reset_candidate.unified_physical_native_factor import *
from simple_wide_perspective_reset_candidate.unified_world_trapezoid_support import WorldTrapezoidRasterSupport
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import CorrectedSharedRinkNumericGeometryLanesAdPairs
from shared_ice_hockey_geometry_candidate.shared_ice_hockey_numeric_geometry import SharedIceHockeyNumericGeometry
from complete_ice_template_candidate.current_shared_geometry_with_free_boundary import CurrentSharedGeometryWithFreeBoundary
from complete_ice_template_candidate.complete_ice_numeric_geometry import CompleteIceTemplateGeometry,_frame,_boundary_hit
from low_independent_held_component_audit.audit_b691_to_v3_held19 import read_version,FOLDERS
from shared_nominal_geometry_reference_candidate.soft_nominal_geometry_source import SoftNominalGeometryFactor,EngineeringReferenceConfig
from simple_wide_perspective_reset_candidate.audit_simple_wide_perspective import draw_chain,all_ice_curves
from scipy.optimize import least_squares

OUT=B/'simple_wide_perspective_reset_candidate'
SEED=B/'simple_wide_physical_reset_candidate/exact_user_joint_plane_physical_family_26ads/physical_family_actual26_state_UNACCEPTED.npz'


def scene_reference_values(scene,unit):
    fs={f['id']:f for f in scene['ice_features']}
    def circle(name):return np.asarray(fs[name]['center_xyz'])[:2]*unit,float(fs[name]['radius'])*unit
    def line(name):
        p=np.asarray(fs[name]['segments_xyz'])[0,:,:2]*unit;v=p[1]-p[0];n=np.array([-v[1],v[0]])/np.linalg.norm(v)
        if n[0]<0:n=-n
        return p.mean(axis=0),n,float(np.arctan2(v[1],v[0]))
    c,r=circle('C0');a1,r1=circle('A1');a2,r2=circle('A2');am=.5*(a1+a2)
    names=('C','B1','B2','goal_line_left','goal_line_right');lines={name:line(name)for name in names};bc,bn,ba=lines['C'];rg,rn,ra=lines['goal_line_right']
    bounds=np.asarray(scene['ice_boundary']['points_xyz'])[:,:2]*unit;lo=bounds.min(axis=0);hi=bounds.max(axis=0)
    vals=[r-.5*(r1+r2),r,(c-bc)@bn,c[1],am[1]-c[1],.5*(a2[1]-a1[1]),(rg-am)@rn,(a2-a1)@rn]
    for name in names[1:]:
        angle=lines[name][2]-ba;vals.append(.5*np.arctan2(np.sin(2*angle),np.cos(2*angle)))
    vals.extend([(.5*(lines['B1'][0]+lines['B2'][0])-bc)@bn,(lines['B2'][0]-lines['B1'][0])@bn,lines['goal_line_left'][0][0]-lo[0],hi[0]-rg[0]])
    fc,fr=circle('F1');rc,rr=circle('R1');vals.extend([.5*(fr+rr),np.linalg.norm(rc-fc)])
    return np.asarray(vals)


class UnifiedExperiment:
    def __init__(self,g,g43,seed_cam,qw,budget,*,tied=False):
        self.g=g;self.factor=UnifiedPhysicalNativeFactor(g,budget=budget);self.tied=tied;self.nc=12 if tied else 14;self.nw=int(g.parameter_count)
        self.n=self.nc+self.nw+2;self.world_slice=slice(self.nc,self.nc+self.nw);self.span_slice=slice(self.nc+self.nw,self.n)
        self.hard=WorldTrapezoidRasterSupport(g,seed_cam,qw,source_scene_provider=self.factor.source_scene)
        original_ref=SoftNominalGeometryFactor(g43,np.arange(43),43,config=EngineeringReferenceConfig(distance_sigma_m=.15,include_curling_tee_and_outer_radius=True))
        self.ref_rows=list(original_ref.rows);self.ref_scales=original_ref.scales.copy();self.ref_nominal=np.array([r['nominal']for r in self.ref_rows]);self.ref_nominal[3]=.5*(np.asarray(g.base['ice_boundary']['points_xyz'])[:,1].min()+np.asarray(g.base['ice_boundary']['points_xyz'])[:,1].max())*g.scene_unit_m
        self.reference_contract=dict(original_ref.contract,current_free_boundary_end_distances_recomputed=True,
            actual_boundary_not_forced_to_60x30=True,reference_rows_are_explicit_engineering_not_measurements=True)
        self._refkey=None;self._refJ=None;self.hard_last=None
        self.quadratic=np.r_[np.zeros(4340,bool),np.ones(18,bool)]
        self.scales=np.r_[self.factor.scales,self.ref_scales]
        self.cost_weights=np.r_[self.factor.weights,np.ones(18)]

    def expand_cam(self,q):
        if not self.tied:return q[:14].copy(),np.eye(14)
        c=np.r_[q[:10],q[8:10],q[10:12]];chain=np.zeros((14,12));chain[:10,:10]=np.eye(10);chain[10:12,8:10]=np.eye(2);chain[12:14,10:12]=np.eye(2)
        return c,chain

    def initial(self,cam,qw,spans):return np.r_[cam[:10],cam[12:14],qw,spans]if self.tied else np.r_[cam,qw,spans]

    def reference(self,qw,jac):
        scene=self.factor.source_scene(qw);raw=scene_reference_values(scene,self.g.scene_unit_m)-self.ref_nominal
        J=None
        if jac:
            key=qw.tobytes()
            if self._refkey==key:J=self._refJ
            else:
                J=np.empty((18,self.nw));h=1e-6
                for k in range(self.nw):
                    qp,qm=qw.copy(),qw.copy();qp[k]+=h;qm[k]-=h
                    J[:,k]=(scene_reference_values(self.factor.source_scene(qp),self.g.scene_unit_m)-scene_reference_values(self.factor.source_scene(qm),self.g.scene_unit_m))/(2*h)
                J/=self.ref_scales[:,None];self._refkey=key;self._refJ=J
        return raw/self.ref_scales,J

    def evaluate(self,q,jac=True,*,enforce_hard=True):
        cam,chain=self.expand_cam(q);qw=q[self.world_slice];spans=q[self.span_slice]
        hr,_=self.hard.evaluate(cam,qw,spans,report=enforce_hard)
        if enforce_hard and not np.all(hr==0):raise ValueError('Actual finite world trapezoid fails original authoritative raster supports')
        if enforce_hard and hasattr(self,'goal_endpoint_support'):self.goal_endpoint_support.guard(q)
        r,j,groups,scene,native=self.factor.evaluate(np.r_[cam,qw],jac)
        ref,jref=self.reference(qw,jac);value=np.r_[r,ref]
        if jac:
            J=np.zeros((len(value),self.n));J[:4340,:self.nc]=j[:,:14]@chain;J[:4340,self.world_slice]=j[:,14:];J[4340:,self.world_slice]=jref
        else:J=None
        return value,J,groups,scene,native,hr

    def constraints(self,q,*,jacobian=True,selection=None,finite_mode='interval_intersection'):
        cam,chain=self.expand_cam(q);full=np.r_[cam,q[self.world_slice],q[self.span_slice]]
        c,J,report=self.hard.constraints(full,jacobian=jacobian,selection=selection,finite_mode=finite_mode)
        if jacobian:
            out=np.zeros((len(c),self.n));out[:,:self.nc]=J[:,:14]@chain
            out[:,self.nc:]=J[:,14:];J=out
            if self.tied:
                jp=np.asarray(report['endpoint_jacobian']);newjp=np.zeros((*jp.shape[:-1],self.n));newjp[...,:self.nc]=jp[...,:14]@chain;newjp[...,self.nc:]=jp[...,14:];report['endpoint_jacobian']=newjp.tolist()
                ji=np.asarray(report['inverse4_jacobian']);newji=np.zeros((4,self.n));newji[:,:self.nc]=ji[:,:14]@chain;newji[:,self.nc:]=ji[:,14:];report['inverse4_jacobian']=newji.tolist()
        return c,J,report

    def trial_guard(self,q,*,report=False):
        cam,_=self.expand_cam(q);r,proof=self.hard.evaluate(cam,q[self.world_slice],q[self.span_slice],report=True)
        if not np.all(r==0):raise ValueError('Not every original USER pen/finite-world support proved')
        self.factor.evaluate(np.r_[cam,q[self.world_slice]],False)
        if hasattr(self,'goal_endpoint_support'):proof['USER_qualified_left_goal_endpoint_support']=self.goal_endpoint_support.guard(q)
        return proof if report else None

    def cost(self,r):return self.factor.cost(r[:4340])+.5*float(r[4340:]@r[4340:])

    def feasible_initialization(self,q):
        # Original explicit world goal station/yaw and sharedwidths provide
        # a four-dimensional physical initialization. They remain active
        # jointly with every other parameter in the main native FIT solve.
        names=list(self.g.parameter_names);ids=[self.nc+names.index('goal_line_right_station_dx_m'),self.nc+names.index('goal_line_right_yaw_delta_rad'),self.n-2,self.n-1]
        start=q.copy();count=0
        def residual(v):
            nonlocal count;count+=1;t=start.copy();t[ids]=v
            try:return self.hard.evaluate(self.expand_cam(t)[0],t[self.world_slice],t[self.span_slice])[0]
            except ValueError:return np.full(268,1e5)
        fit=least_squares(residual,start[ids],jac='3-point',x_scale=[1,.01,1,1],max_nfev=80,ftol=1e-12,xtol=1e-12,gtol=1e-10)
        q[ids]=fit.x;r=residual(fit.x)
        if not np.all(r==0):
            # Physical coupled symmetry can conflict with the planar camera
            # initialization. Permit the entire declared C/R/f/PP/lens here;
            # do not disguise a free image line as a world marking.
            start=q.copy();ids=list(range(self.nc))+ids
            scales=np.r_[np.array([.01,.01,.01,.03,.03,.03,.1,.1,.1,.1]),[.1,.1]if self.tied else[.1,.1,.1,.1],[1,.01,1,1]]
            fit=least_squares(residual,start[ids],jac='3-point',x_scale=scales,max_nfev=150,ftol=1e-12,xtol=1e-12,gtol=1e-10)
            q[ids]=fit.x;r=residual(fit.x)
        return q,dict(initialization_activity_indices=ids,initialization_parameter_names=[self.g.parameter_names[i-self.nc]if i<self.n-2 else['goal_carrier_span_m','board_carrier_span_m'][i-(self.n-2)]for i in ids],
            all268_original_supports_proved=bool(np.all(r==0)),maximum_equation_violation_px=float(abs(r).max()),feasibility_merit=float(.5*r@r),evaluations=count,
            no_FIT_or_HELD_source_role_or_sigma_changed=True,initializer_not_locked_in_joint_fit=True)

    def fit(self,q,iterations):
        damping=1e-3;history=[];status='explicit small-probe iteration scope';last=time.perf_counter()
        for iteration in range(iterations):
            r,J,*_=self.evaluate(q);current=self.cost(r);w=np.r_[self.factor.irls(r[:4340]),np.ones(18)];sw=np.sqrt(w)
            scale=np.maximum(np.linalg.norm(J*sw[:,None],axis=0),1e-10);A=J*sw[:,None]/scale;H=A.T@A;gg=A.T@(r*sw);accepted=False;invalid=0
            for attempt in range(18):
                step=-np.linalg.solve(H+damping*np.eye(self.n),gg)/scale
                # FIT H has exactly zero shared span2 columns. Their changes
                # compensate the already proposed pose/world direction to
                # preserve the user's actual finite carrier supports; no
                # image-line nuisance or hidden UV target substitutes world.
                candidate=q+step;cam,_=self.expand_cam(candidate)
                try:
                    spans,spanreport=self.hard.span_initialization(cam,candidate[self.world_slice],q[self.span_slice])
                    candidate[self.span_slice]=spans;trial=self.cost(self.evaluate(candidate,False)[0])
                except ValueError:invalid+=1;trial=float('inf')
                if trial<current:q=candidate;damping=max(1e-9,damping/3);accepted=True;break
                damping*=5
            history.append(dict(iteration=iteration,cost=float(current),accepted=accepted,invalid_original_hard_or_physical_domain_trials=invalid,damping=float(damping)))
            print(json.dumps(dict(model='tied_PP12'if self.tied else'general_pixel_map14',budget=self.factor.budget,**history[-1])),flush=True)
            if not accepted:status='no accepted original hard-support feasible native FIT LM trial';break
            if current-trial<1e-8*max(1.,current):status='small relative native FIT cost change; no constrained optimality certificate';break
        return q,history,status


def full_overlay(image,scene,g43,qworld,spans,cam,groups,hard,name,path):
    from complete_ice_template_candidate.complete_ice_numeric_geometry import CompleteIceTemplateGeometry
    image=image.copy();template=CompleteIceTemplateGeometry(g43)
    expected=template._missing(scene,np.r_[spans-np.array([6.8,8.6]),np.zeros(5)])
    actual_trapezoids=hard._segments(qworld,spans)
    actual_template_ids={name.replace('_side_minus','_lower_outer').replace('_side_plus','_upper_outer'):value
        for name,value in actual_trapezoids.items()}
    for feature in expected:
        if feature['id']in actual_template_ids:
            feature['segments_xyz']=[actual_template_ids[feature['id']]['world_scene'].tolist()]
            feature['render_binding']='same actual world primitive used by finite USER source'
    present_ids={f['id']for f in scene['ice_features']}
    augmented=dict(scene);augmented['ice_features']=list(scene['ice_features'])+[f for f in expected if f['id']not in present_ids]
    def draw(w,color,thickness=1):
        try:p,_,v,_=project_native(cam,np.asarray(w,float));p[~v]=np.nan;draw_chain(image,p,color,thickness)
        except ValueError:pass
    original_ids={f['id']for f in scene['ice_features']}
    for f in augmented['ice_features']:
        color=(0,210,255)if f['id']in original_ids else(200,140,190)
        if f['kind']=='circle':
            draw(world_curve(augmented,dict(kind='circle_outline',feature_id=f['id'])),color)
            for ring in f.get('rings',[]):draw(world_curve(augmented,dict(kind='circle_outline',feature_id=f['id'],ring_role=ring['role'])),color)
        elif f['kind']=='polyline':
            for k in range(len(f['segments_xyz'])):draw(world_curve(augmented,dict(kind='line_segment',feature_id=f['id'],segment=k)),color)
        elif f['kind']=='polygon':draw(world_curve(augmented,dict(kind='polygon_outline',feature_id=f['id'])),color)
    bottom=np.asarray(scene['board_surface']['bottom_edge_xyz']);top=np.asarray(scene['board_surface']['top_edge_xyz'])
    draw(bottom,(255,200,0),3)
    draw(top,(0,100,255),3)
    for height in (.25,.5,.75):draw(bottom*(1-height)+top*height,(60,190,220))
    lengths=np.linalg.norm(np.diff(bottom,axis=0),axis=1)*SCENE_UNIT_M;stations=np.r_[0,np.cumsum(lengths)]
    for station in np.arange(0,stations[-1],1):
        k=min(int(np.searchsorted(stations,station,side='right')-1),len(lengths)-1);t=(station-stations[k])/lengths[k]
        a=bottom[k]*(1-t)+bottom[k+1]*t;b=top[k]*(1-t)+top[k+1]*t;draw(np.array([a,b]),(60,190,220))
    # Global one-metre grid in the very same physical ice plane.
    polygon=np.asarray(scene['ice_boundary']['points_xyz']);lo=polygon[:,:2].min(axis=0)*SCENE_UNIT_M;hi=polygon[:,:2].max(axis=0)*SCENE_UNIT_M
    for axis in (0,1):
        for value in np.arange(np.ceil(lo[axis]),hi[axis],1):
            ts=np.linspace(lo[1-axis],hi[1-axis],400);pts=np.zeros((len(ts),3));pts[:,axis]=value/SCENE_UNIT_M;pts[:,1-axis]=ts/SCENE_UNIT_M
            inside=np.array([cv2.pointPolygonTest(polygon[:,:2].astype(np.float32),tuple(p[:2]),False)>=0 for p in pts]);pts[~inside]=np.nan
            # Invalid outside portions are disconnected; no bridging.
            for part in np.split(np.arange(len(pts)),np.flatnonzero(np.diff(inside.astype(int))!=0)+1):
                if len(part)>1 and inside[part[0]]:draw(pts[part],(90,120,120))
    for ad in scene['advertisements']:
        for curve in ad['fitted_boundary_curves_xyz'].values():draw(curve,(230,120,230),2)
        points=np.array(list(ad['corners_xyz_scene_units'].values()));p,_,v,_=project_native(cam,points)
        if v.all():
            center=np.round(p.mean(axis=0)).astype(int)
            if 0<=center[0]<1920 and 0<=center[1]<1080:cv2.putText(image,'Ad'+str(ad['user_number']),tuple(center),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,120,230),1,cv2.LINE_AA)
    for group in hard.shape.groups:
        uv=np.asarray(group['native_source_UV']);draw_chain(image,uv,(0,0,255),2)
    # These are the actual physical source carriers, not a separate template
    # projection or the observed stroke. A white centre stays visible inside
    # the red authoritative raster footprint where they agree.
    for actual_name in hard.names:
        actual_world=np.asarray(actual_trapezoids[actual_name]['world_scene'])
        draw(actual_world,(255,255,255),1)
        endpoint,_,valid,_=project_native(cam,actual_world[:1])
        if valid[0]:cv2.circle(image,tuple(np.round(endpoint[0]).astype(int)),4,(0,255,255),1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(1920,60),(0,0,0),-1);cv2.putText(image,name+' / ONE physical pose and inverse map / UNACCEPTED',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,'all ice/grid/board/ads ON; magenta thin expected template; red original USER ink; no old Brown overlay',(12,46),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1,cv2.LINE_AA)
    cv2.imwrite(str(path),image)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',choices=['literal_reference','physical_feature_normalized'],default='literal_reference');ap.add_argument('--tied-pp',action='store_true');ap.add_argument('--iterations',type=int,default=18);args=ap.parse_args()
    started=time.perf_counter();v=read_version(FOLDERS[1]);base=json.loads(v['aux']['aux__shared_scene_json'].item())
    g43=SharedIceHockeyNumericGeometry(CorrectedSharedRinkNumericGeometryLanesAdPairs(base,SCENE_UNIT_M));g=CurrentSharedGeometryWithFreeBoundary(g43)
    with np.load(SEED,allow_pickle=False)as z:qw=np.r_[z['world43'],np.zeros(4)];cam=camera14_from_family(z['K'],z['R'],z['C'],z['image_only_inverse_radial'])
    probe=UnifiedExperiment(g,g43,cam,qw,args.budget,tied=args.tied_pp);q0=probe.initial(cam,qw,np.array([6.8,8.6]));initial_binding=probe.factor.audit_named_ad_scene_binding(qw)
    initial_unconstrained=probe.evaluate(q0,False,enforce_hard=False);q0,feasibility=probe.feasible_initialization(q0.copy());feasible=feasibility['all268_original_supports_proved']
    print(json.dumps(dict(initial_hard_world_feasibility=feasibility)),flush=True)
    name=('physical_tied_PP12'if args.tied_pp else'physical_pixel_map14')+'_'+args.budget
    if feasible:
        r0,J0,*_=probe.evaluate(q0);directions=[];rng=np.random.default_rng(577)
        for k in range(2):
            d=rng.normal(size=probe.n);d[:probe.nc]*=.01;d/=np.linalg.norm(d);h=1e-7
            rp=probe.evaluate(q0+h*d,False,enforce_hard=False)[0];rm=probe.evaluate(q0-h*d,False,enforce_hard=False)[0];fd=(rp-rm)/(2*h);a=J0@d
            directions.append(dict(direction=k,width=h,max_absolute=float(abs(fd-a).max()),relative=float(np.linalg.norm(fd-a)/max(1.,np.linalg.norm(fd),np.linalg.norm(a)))))
        q,history,status=probe.fit(q0.copy(),args.iterations)
    else:q=q0;history=[];status='physical finite-world hard-support initialization not feasible; not fitted';directions=[]
    r,J,groups,scene,native,hr=probe.evaluate(q,False,enforce_hard=False);finalcam,_=probe.expand_cam(q)
    proof=None
    if np.all(hr==0):proof=probe.hard.evaluate(finalcam,q[probe.world_slice],q[probe.span_slice],report=True)[1]
    output=OUT/(name+'_UNACCEPTED');output.mkdir(exist_ok=True)
    actual_scene=g.scene(q[probe.world_slice]);picture=output/'all_same_3D_camera_ice_board_prints_user_UNACCEPTED.png'
    full_overlay(cv2.imread(probe.factor.packet.frame['image_path']),actual_scene,g43,q[probe.world_slice],q[probe.span_slice],finalcam,groups,probe.hard,name,picture)
    npz=output/'actual_single_physical_source_state_UNACCEPTED.npz';np.savez_compressed(npz,local_state=q,camera14=finalcam,world47=q[probe.world_slice],trapezoid_shared_spans_m=q[probe.span_slice],initial_local_state=q0,original_unconstrained_state=probe.initial(cam,qw,np.array([6.8,8.6])))
    token=hashlib.sha256(npz.read_bytes()).hexdigest()
    report=dict(schema='actual-one-physical-camera-native45-and26-USER-world-trapezoid-diagnostic.v1',source_seed=str(SEED),source_seed_SHA=hashlib.sha256(SEED.read_bytes()).hexdigest(),state_file=str(npz),state_file_SHA=token,source_event_q_profile=v['event'],world47_active_names=list(g.parameter_names),camera_count=probe.nc,private_radial_center_tied_to_camera_PP=args.tied_pp,
        general_pixel_post_correction_not_manufacturer_calibration=not args.tied_pp,all45_original_groups_and26_manual_corners=groups,source_contract=probe.factor.contract,source_ledger=probe.factor.ledger,
        original_sigma_UV_roles_kept=True,hard_user_source_sigma0_not_divided=True,original_auto_dark_bands_not_FIT=True,
        initial_unconstrained_totalcost=probe.cost(initial_unconstrained[0]),initial_feasibility=feasibility,actual_small_direction_FD=directions,
        final_total_cost=probe.cost(r),original_native_Huber_cost=probe.factor.cost(r[:4340]),explicit18engineering_cost=float(.5*r[4340:]@r[4340:]),
        all2170_original_points_native_RMS_px=float(np.sqrt(np.mean(np.sum(native.reshape(-1,2)**2,axis=1)))),all268_hard_finite_world_support_proved=bool(np.all(hr==0)),
        user_pixel_proof=proof,maximum_hard_native_equation_violation_px=float(abs(hr).max()),trapezoid_identity_ledger=probe.hard.identity_ledger,
        image=str(picture),one_actual_camera_and_inverse_for_every_drawn_item=True,full_display_never_hides_bad_source=True,
        current_boundary_and_height=q[probe.world_slice][43:].tolist(),trapezoid_shared_spans_m=q[probe.span_slice].tolist(),
        nominal_60x30_only_reference_not_hard_target=True,reference_contract=probe.reference_contract,reference_rows=probe.ref_rows,reference_raw=(r[4340:]*probe.ref_scales).tolist(),
        optimizer_history=history,termination=status,localonly_no_programme_graph_no_HELD_fit=True,production_changed=False,accuracy_accepted=False,seconds=time.perf_counter()-started)
    (output/'actual_all45_26_USER_unified_3D_report_UNACCEPTED.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    (output/'actual_same_source_scene_UNACCEPTED.json').write_text(json.dumps(actual_scene,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(path=str(output),cost=report['final_total_cost'],RMS=report['all2170_original_points_native_RMS_px'],feasible=report['all268_hard_finite_world_support_proved'],status=status,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
