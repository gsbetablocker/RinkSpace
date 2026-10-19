"""Direct coupled hockey/curling latent scene, not old world49 deltas.

The initial projection onto this structural family changes old source worlds
explicitly. Source ID/UV/sigma/role and actual print material authority remain
external, unchanged. No expected mark becomes an observation or hidden prior.
"""
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import cv2
from wide_anchor_candidate.native_wide_source import world_curve
from .complete_ice_numeric_geometry import _boundary_hit,_frame
from .shared_hockey_frame_free_boundary import SharedHockeyFrameFreeBoundaryGeometry

SCHEMA='rinkspace.direct-coupled-physical-hockey-curling-template-latent48.v1'
HOCKEY_NAMES=('hockey_centre_x_m','hockey_centre_y_m','hockey_yaw_rad','all_five_hockey_circle_radius_m',
 'faceoff_goal_inward_distance_m','faceoff_half_transverse_separation_m','blue_carrier_gap_m',
 'left_goal_to_current_axis_wall_m','right_goal_to_current_axis_wall_m','common_crease_template_radius_m',
 'shared_trapezoid_goal_carrier_span_m','shared_trapezoid_board_carrier_span_m')
CURLING_NAMES=('curling_tee_midpoint_x_m','curling_row_midpoint_y_m','curling_common_yaw_rad',
 'curling_tee_column_separation_m','curling_four_row_pitch_m','curling_eight_house_outer_radius_m',
 'curling_middle_outer_ratio','curling_inner_outer_ratio','curling_button_outer_ratio',
 'four_Lane_paint_common_width_m','four_Lane_paint_right_end_from_Rtee_m',
 'expected_sheet_end_beyond_tee_m','expected_common_hog_inside_from_tee_m',
 'expected_back_outside_minus_house_radius_m','expected_centreline_extension_beyond_tee_m','expected_sheet_common_width_m')
NOMINAL_REFERENCES={'all_five_hockey_circle_radius_m':4.5,'faceoff_goal_inward_distance_m':6.,
 'faceoff_half_transverse_separation_m':7.,'blue_carrier_gap_m':15.,
 'left_goal_to_current_axis_wall_m':4.,'right_goal_to_current_axis_wall_m':4.,
 'shared_trapezoid_goal_carrier_span_m':6.8,'shared_trapezoid_board_carrier_span_m':8.6,
 'curling_tee_column_separation_m':34.75,'curling_eight_house_outer_radius_m':1.829,
 'expected_common_hog_inside_from_tee_m':6.401,'expected_back_outside_minus_house_radius_m':0.,
 'expected_centreline_extension_beyond_tee_m':3.658}


def axes(theta):
    c,s=np.cos(theta),np.sin(theta);return np.array([c,s,0.]),np.array([-s,c,0.])


class StructuredSharedIceTemplateGeometry:
    """28 structured marking values, actual16 print values, boundary4.

    p48 is a new absolute/mixed named chart and is NOT old world47/world49.
    Parent whole-q must be authenticated by the caller before construction.
    """
    parameter_count=48

    def __init__(self,parent,parent_world49,*,engineering_config=None):
        if type(parent)is not SharedHockeyFrameFreeBoundaryGeometry:raise TypeError('Explicit current owned world49 parent required')
        self.parent=deepcopy(parent);self.parent_q=self.parent.validate_parameters(parent_world49)
        self.base=self.parent.base;self.scene_unit_m=float(parent.scene_unit_m)
        self.boundary_owner=self.parent.original
        self.parent_scene=self.parent.source_scene(self.parent_q)
        self.config=dict(sheet_end_beyond_tee_m=(45.72-34.75)/2.,hog_inside_from_tee_m=6.401,
            back_outside_minus_house_radius_m=0.,centreline_extension_beyond_tee_m=3.658,
            goal_carrier_span_m=6.8,board_carrier_span_m=8.6,hack_carrier_length_reference_m=.457,sheet_width_reference_m=4.750)
        if engineering_config is not None:
            if not isinstance(engineering_config,dict)or set(engineering_config)-set(self.config):raise ValueError('Explicit known engineering initializer keys only')
            self.config.update(engineering_config)
        self.parameter_names=(*HOCKEY_NAMES,*CURLING_NAMES,*self.boundary_owner.original.parameter_names[7:23],*self.boundary_owner.parameter_names[43:47])
        assert len(self.parameter_names)==48
        self.parameter_units=tuple('rad'if n.endswith('_rad')else'dimensionless'if n.endswith('_ratio')else'm'for n in self.parameter_names)
        self.index={name:i for i,name in enumerate(self.parameter_names)}
        self.features={f['id']:f for f in self.parent_scene['ice_features']}
        self._cache=OrderedDict();self._lane_proxy_local={};self._crease_unit={}
        self.initial=self._initial_from_parent()
        self.validate_parameters(self.initial)

    def _initial_from_parent(self):
        fs=self.features;unit=self.scene_unit_m;c=np.asarray(fs['C0']['center_xyz'])*unit
        carrier=[]
        for name in ('C','B1','B2','goal_line_left','goal_line_right'):
            v=np.diff(np.asarray(fs[name]['segments_xyz'])[0],axis=0)[0];v=v/np.linalg.norm(v)
            if v[1]<0:v=-v
            carrier.append(float(np.arctan2(-v[0],v[1])))
        theta=float(np.arctan2(np.sin(carrier).mean(),np.cos(carrier).mean()));a,v=axes(theta)
        poly=np.asarray(self.parent_scene['ice_boundary']['points_xyz']);centre=c/unit
        lh=_boundary_hit(centre,-a,poly)*unit;rh=_boundary_hit(centre,a,poly)*unit
        goals={side:np.asarray(fs['goal_line_'+side]['segments_xyz'])[0].mean(axis=0)*unit for side in ('left','right')}
        lstation=(goals['left']-c)@a;rstation=(goals['right']-c)@a
        pair=np.array([fs[n]['center_xyz']for n in ('A1','A2')])*unit
        inward=float(rstation-((pair.mean(axis=0)-c)@a));halfY=float(abs((pair[1]-pair[0])@v)/2)
        b=[np.asarray(fs[n]['segments_xyz'])[0].mean(axis=0)*unit for n in ('B1','B2')]
        gap=float(abs((b[1]-b[0])@a))
        crease_radii=[]
        for side in ('left','right'):
            pts=np.asarray(fs['goal_crease_'+side]['polygon_xyz']);anchor=(pts[0]+pts[-1])/2
            rad=float(np.linalg.norm(pts[-1]-pts[0])/2);crease_radii.append(rad*unit)
            _,inside,across=_frame(fs,'goal_line_'+side)
            self._crease_unit[side]=np.c_[(pts-anchor)@inside/rad,(pts-anchor)@across/rad,np.zeros(len(pts))]
        hockey=np.array([c[0],c[1],theta,np.mean([fs[n]['radius']*unit for n in ('C0','A1','A2')]),
            inward,halfY,gap,float(((goals['left']-lh)@a)),float(((rh-goals['right'])@a)),
            np.mean(crease_radii),self.config['goal_carrier_span_m'],self.config['board_carrier_span_m']])
        F=np.asarray([fs['F'+str(i)]['center_xyz']for i in range(1,5)])*unit
        R=np.asarray([fs['R'+str(i)]['center_xyz']for i in range(1,5)])*unit
        delta=R.mean(axis=0)-F.mean(axis=0);yaw=float(np.arctan2(delta[1],delta[0]));ca,cv=axes(yaw)
        mid=np.mean([F.mean(axis=0),R.mean(axis=0)],axis=0);sep=float(delta@ca)
        pitch=float(np.mean(np.diff(((F+R)/2)@cv)))
        house=float(np.mean([fs[k+str(i)]['radius']*unit for k in ('F','R')for i in range(1,5)]))
        ratios={r['role']:float(r['radius'])/float(fs['F1']['radius'])for r in fs['F1']['rings']}
        widths=[];paint_ends=[]
        for row in range(1,5):
            p=np.asarray(fs['lane'+str(row)]['polygon_xyz'])*unit
            if len(p)==5 and np.array_equal(p[0],p[-1]):p=p[:-1]
            if p.shape!=(4,3)or fs['lane'+str(row)].get('open_edges')!=[3]:raise ValueError('Original four-vertex Lane three-painted-edge authority required')
            widths.append(float(((p[0]+p[1]-p[2]-p[3])/2)@cv))
            paint_ends.append(float((((p[1]+p[2])/2)-R[row-1])@ca))
            self._lane_proxy_local[row]=((p[[0,3]]-mid)@ca).copy()
        curl=np.array([mid[0],mid[1],yaw,sep,pitch,house,ratios['middle'],ratios['inner'],ratios['center'],
            np.mean(widths),np.mean(paint_ends),self.config['sheet_end_beyond_tee_m'],self.config['hog_inside_from_tee_m'],
            self.config['back_outside_minus_house_radius_m'],self.config['centreline_extension_beyond_tee_m'],self.config['sheet_width_reference_m']])
        self.initialization_ledger=dict(hockey_common_yaw_from_parent_carrier_angles_rad=carrier,
            parent_C0_A_circle_radii_m=[fs[n]['radius']*unit for n in ('C0','A1','A2')],
            common_radius_initializer_is_arithmetic_geometry_projection_not_observed_radius=True,
            historical_Lane_end_offsets_m=paint_ends,historical_Lane_widths_m=widths,
            actual_geometry_initializers_not_fixed_measurements_or_priors=True,
            no_new_60x30_target=True,engineering_expected_only_initializers=dict(self.config))
        return np.r_[hockey,curl,self.parent_q[7:23],self.parent_q[43:47]]

    def _q(self,p):
        p=np.asarray(p,float)
        if p.shape!=(48,)or not np.isfinite(p).all():raise ValueError('Explicit named structured latent48 required; never truncate old whole q')
        return p

    def _boundary_q(self,p):
        q=self.parent_q[:47].copy();q[7:23]=p[28:44];q[43:47]=p[44:48];return q

    def validate_parameters(self,p):
        p=self._q(p);cq=self.boundary_owner.current_course6(self._boundary_q(p))
        if np.any(p[[3,4,5,6,7,8,9,10,11,15,16,17,21,23,24,26,27]]<=0):raise ValueError('Structured physical radii/distances/widths must be positive')
        if not 1>p[18]>p[19]>p[20]>0:raise ValueError('Common house rings require strict concentric order')
        if p[11]<=p[10]or p[17]+p[25]<=0:raise ValueError('Outward trapezoid and positive outside back distance required')
        if p[15]/2+p[22]<=max(np.max(x)for x in self._lane_proxy_local.values()):raise ValueError('Lane painted right end must follow original unmeasured left proxies')
        owner=self.boundary_owner.transporter;q29=np.r_[self._boundary_q(p)[:23],cq]
        with owner._current_old_domain(q29):
            owner.original.panel.validate_parameters(p[28:32]);owner.original.pair_panels.validate_parameters(p[32:44])
        return p.copy()

    def _physical_boundary_and_prints(self,p,rebuild_grid):
        owner=self.boundary_owner;q47=self._boundary_q(p);cq=owner.current_course6(q47)
        scene=deepcopy(self.parent_scene);q29=np.r_[q47[:23],cq]
        with owner.transporter._current_old_domain(q29):
            scene=owner.transporter.original.panel.scene(p[28:32],scene=scene)
            scene=owner.transporter.original.pair_panels.scene(p[32:44],scene=scene)
        payload=owner.course.scene(cq,rebuild_grid=rebuild_grid)
        for k in ('parameters','ice_boundary','board_surface','meshes','boundary_validity_margins','shared_geometry_checks'):scene[k]=payload[k]
        if rebuild_grid:scene['ice_grid']=payload['ice_grid']
        if np.any(p[44:48]!=0.):owner._transport_ads(scene,q47)
        return scene

    def _feature_scene(self,scene,p):
        if getattr(self,'_derive_hockey_center_from_course',False):
            # Even a direct feature-view caller follows the same confirmed
            # structure. The actual current scene boundary is the datum.
            boundary=np.asarray(scene['ice_boundary']['points_xyz'],float)
            p=np.asarray(p,float).copy()
            p[:2]=.5*(boundary[:,:2].min(axis=0)+boundary[:,:2].max(axis=0))*self.scene_unit_m
        unit=self.scene_unit_m;fs={f['id']:f for f in scene['ice_features']};expected=[]
        def new(fid,kind):
            if fid in fs:raise ValueError('Expected template must never overwrite an existing qualified feature: '+fid)
            f=dict(id=fid,label=fid,kind=kind,plane_normal_xyz=[0.,0.,1.],surface_id='ice_surface',
                geometry_status='unobserved_coupled_expected_template',source=dict(UV_count=0,new_observation=False),
                style=dict(fill_srgb=None,line_srgb='AE508C'if fid.startswith('curling_')else'C00000',line_width_emu=19050,dash='solid'))
            expected.append(f);return f
        def line(fid,ends,*,observed=False):
            f=fs[fid]if observed else new(fid,'polyline');f['segments_xyz']=[(np.asarray(ends)/unit).tolist()]
            f['structured_material_relation']=SCHEMA;return f
        def circle(fid,centre,radius,*,observed=False):
            f=fs[fid]if observed else new(fid,'circle');f['center_xyz']=(centre/unit).tolist();f['radius']=float(radius/unit)
            f['structured_material_relation']=SCHEMA;return f
        a,v=axes(p[2]);centre=np.r_[p[:2],0.];polygon=np.asarray(scene['ice_boundary']['points_xyz'])
        wall={side:_boundary_hit(centre/unit,direction,polygon)*unit for side,direction in [('left',-a),('right',a)]}
        goal={side:wall[side]+(p[7]if side=='left'else-p[8])*a for side in ('left','right')}
        stations={side:float((point-centre)@a)for side,point in goal.items()}
        if not stations['left']<-p[6]/2<0<p[6]/2<stations['right']:raise ValueError('Actual current wall/goal/blue carrier station ordering lost')
        def clipped(anchor,name):
            points=np.array([_boundary_hit(anchor/unit,-v,polygon),_boundary_hit(anchor/unit,v,polygon)])*unit
            original=np.asarray(self.features[name]['segments_xyz'])[0]
            # Same original finite material order, rather than silently
            # reversing source line knots while making carriers parallel.
            if(np.diff(original,axis=0)[0]@v)<0:points=points[::-1]
            return points
        circle('C0',centre,p[3],observed=True)
        for name,station in [('C',0.),('B1',-p[6]/2),('B2',p[6]/2)]:line(name,clipped(centre+station*a,name),observed=True)
        for side in ('left','right'):
            inside=a if side=='left'else-a;anchor=goal[side]
            f=line('goal_line_'+side,clipped(anchor,'goal_line_'+side),observed=True)
            f['shared_physical_goal_frame']=dict(centre_xyz=(anchor/unit).tolist(),inward_xyz=inside.tolist(),across_xyz=v.tolist(),chart_anchor_not_surveyed=True)
            if side=='right':identities=[('A1',-1,True),('A2',1,True)]
            else:identities=[('A3',1,False),('A4',-1,False)]
            for fid,sign,observed in identities:circle(fid,anchor+inside*p[4]+sign*v*p[5],p[3],observed=observed)
            template=self._crease_unit[side]
            if getattr(self,'_strict_half_circle_crease',False):
                # Explicit current user structure, not the ellipse or slight
                # frame rotation inferred by an old failed world initializer.
                # The same 61 knots, current goal frame and estimable p9 radius
                # remain on every source and rendering route.
                theta=np.linspace(0.,np.pi,len(template))
                template=np.c_[np.sin(theta),np.cos(theta),np.zeros(len(theta))]
                template[[0,-1],0]=0.
            points=anchor+p[9]*(template[:,0,None]*inside+template[:,1,None]*v)
            fs['goal_crease_'+side]['polygon_xyz']=(points/unit).tolist()
            if getattr(self,'_strict_half_circle_crease',False):
                fs['goal_crease_'+side]['current_crease_structure']=dict(kind='user_confirmed_half_circle',
                    radius_estimated=True,current_goal_frame_estimated=True,legacy_inferred_ellipse_not_a_fixed_physical_fact=True)
            for sign,label in [(-1,'lower'),(1,'upper')]:
                start=anchor+sign*v*p[10]/2;seed=anchor+sign*v*p[11]/2
                end=_boundary_hit(seed/unit,-inside,polygon)*unit
                tf=line('goal_trapezoid_'+side+'_'+label+'_outer',[start,end]);tf['outside_or_centre_source_edge_not_inferred']=True
        ca,cv=axes(p[14]);mid=np.r_[p[12:14],0.]
        for row in range(1,5):
            cy=(row-2.5)*p[16];rowcentre=mid+cv*cy
            tees={'F':rowcentre-ca*p[15]/2,'R':rowcentre+ca*p[15]/2}
            for end in ('F','R'):
                fid=end+str(row);c=tees[end];f=circle(fid,c,p[17],observed=True)
                ratios={'outer':1.,'middle':p[18],'inner':p[19],'center':p[20]}
                for ring in f['rings']:ring['center_xyz']=(c/unit).tolist();ring['radius']=float(p[17]*ratios[ring['role']]/unit)
            proxy=self._lane_proxy_local[row];right=p[15]/2+p[22];half=p[21]/2
            points=np.array([rowcentre+ca*proxy[0]+cv*half,rowcentre+ca*right+cv*half,
                rowcentre+ca*right-cv*half,rowcentre+ca*proxy[1]-cv*half])
            lane=fs['lane'+str(row)];lane['polygon_xyz']=(points/unit).tolist();lane['centerline_xyz']=np.array([rowcentre+ca*max(proxy),rowcentre+ca*right]).tolist()
            lane['centerline_xyz']=(np.asarray(lane['centerline_xyz'])/unit).tolist()
            lane['paint_segments_xyz']=[(points[[0,1]]/unit).tolist(),(points[[1,2]]/unit).tolist(),(points[[2,3]]/unit).tolist()]
            lane['structured_paint_end_estimated_not_fixed_historical_6m']=float(p[22]);lane['new_sheet_end_identity_not_assigned_to_known_Lane']=True
            prefix='curling_sheet'+str(row)+'_';expected_half=p[27]/2
            for end,inside in [('F',ca),('R',-ca)]:
                tee=tees[end]
                for label,c in [('tee_center',tee),('hog_inside',tee+inside*p[24]),('back_outside',tee-inside*(p[17]+p[25])),('sheet_end_reference',tee-inside*p[23])]:
                    line(prefix+label+'_'+end,[c-cv*expected_half,c+cv*expected_half])
                hack=tee-inside*p[26];hw=self.config['hack_carrier_length_reference_m']/2
                line(prefix+'hack_reference_'+end,[hack-cv*hw,hack+cv*hw])
            line(prefix+'centerline_reference',[tees['F']-ca*p[26],tees['R']+ca*p[26]])
            for sign,label in [(-1,'lower'),(1,'upper')]:line(prefix+'sideline_reference_'+label,[tees['F']-ca*p[23]+sign*cv*expected_half,tees['R']+ca*p[23]+sign*cv*expected_half])
            if row==3 and'curling_centerline_row3'in fs:
                # Preserve the original carrier's unmeasured longitudinal
                # coverage offsets, not newly assert tee/end observations.
                ends=np.asarray(self.features['curling_centerline_row3']['segments_xyz'])[0]*unit
                olda,_=axes(self.initial[14]);oldmid=np.r_[self.initial[12:14],0.];ts=(ends-oldmid)@olda
                line('curling_centerline_row3',np.array([rowcentre+ca*t for t in ts]),observed=True)
        scene['ice_features'].extend(expected)
        boundary=np.ascontiguousarray(polygon[:,:2],np.float32)
        for f in scene['ice_features']:
            if f['kind']=='circle'and f['id']in('C0','A1','A2','A3','A4',*[k+str(i)for k in('F','R')for i in range(1,5)]):
                margin=cv2.pointPolygonTest(boundary,tuple(map(float,f['center_xyz'][:2])),True)
                if margin+1e-5<float(f['radius']):raise ValueError('Complete current structured circle leaves actual ice footprint: '+f['id'])
            if f['id']in('lane1','lane2','lane3','lane4'):
                # The painted right end is physical; the open-left vertices
                # are unmeasured coverage proxies and cannot become bounds.
                for point in np.asarray(f['polygon_xyz'])[1:3]:
                    if cv2.pointPolygonTest(boundary,tuple(map(float,point[:2])),True)<-2e-5:
                        raise ValueError('Qualified Lane painted end leaves actual ice footprint: '+f['id'])
        return scene

    def _scene(self,p,rebuild_grid):
        p=self.validate_parameters(p);key=(p.tobytes(),bool(rebuild_grid))
        if key in self._cache:self._cache.move_to_end(key);return deepcopy(self._cache[key])
        scene=self._feature_scene(self._physical_boundary_and_prints(p,rebuild_grid),p)
        scene[SCHEMA]=self.manifest(p);self._cache[key]=deepcopy(scene)
        while len(self._cache)>4:self._cache.popitem(last=False)
        return scene

    def source_scene(self,p):return self._scene(p,False)
    def scene(self,p):return self._scene(p,True)

    def semantic_world(self,target,p,*,samples=256):
        scene=self.source_scene(p)
        if target.get('kind')!='polygon_supporting_line':return world_curve(scene,target,samples=samples,scene_unit_m=self.scene_unit_m)
        feature=next(f for f in scene['ice_features']if f['id']==target['feature_id']);k=int(target['index'])
        if feature['id']not in('lane1','lane2','lane3','lane4')or k not in(0,1,2)or k in feature.get('open_edges',[]):raise ValueError('Qualified original three-sided Lane target required')
        poly=np.asarray(feature['polygon_xyz']);a,b=poly[[k,(k+1)%4]];v=b-a
        if np.sum(abs(v[:2])>1e-9)==1:return world_curve(scene,target,samples=samples,scene_unit_m=self.scene_unit_m)
        xmin,ymin,xmax,ymax=map(float,scene['parameters']['bounds']);hits=[]
        for axis,low,high in((0,xmin,xmax),(1,ymin,ymax)):
            if v[axis]==0.:continue
            for bound in(low,high):
                t=(bound-a[axis])/v[axis];point=a+t*v;other=1-axis
                if (ymin if other==1 else xmin)-1e-10<=point[other]<=(ymax if other==1 else xmax)+1e-10:hits.append(point)
        if len(hits)<2:raise ValueError('Current Lane carrier does not cross original current rectangular support bounds')
        axis=int(np.argmax(abs(v[:2])));hits=sorted(hits,key=lambda x:x[axis])
        return np.linspace(hits[0],hits[-1],samples)

    def rebind(self,row,p):
        value,_=self._named_ad_world_and_jacobian(row,p);out=dict(row);out['world']=value;return out

    def world_jacobian(self,row,p):return self._named_ad_world_and_jacobian(row,p)[1]

    def _named_ad_world_and_jacobian(self,row,p):
        p=self.validate_parameters(p);n=row.get('physical_ad_id');roles=row.get('physical_world_corner_roles')
        if not isinstance(n,(int,np.integer))or not isinstance(roles,(list,tuple)):raise ValueError('This release only binds original wide named print corners; programme material rebind not installed')
        from ad_landmarks import AdGeometry
        source=AdGeometry(self.base,int(n));expected=np.array([source.corner(source.initial,r)for r in roles])
        if not np.array_equal(np.asarray(row['world'],float),expected):raise ValueError('Actual original print material authority changed')
        owner=self.boundary_owner;q47=self._boundary_q(p);q29=np.r_[q47[:23],owner.current_course6(q47)]
        with owner.transporter._current_old_domain(q29):
            world,j23=owner.transporter.original._world_and_jacobian(row,q47[:23])
            value,j29=owner.transporter._transport_current_ad(world,j23,int(n),q29)
        J=np.zeros((*value.shape,48));J[:,:,28:44]=j29[:,:,7:23];J[:,:,44:48]=np.einsum('nik,kj->nij',j29[:,:,23:],owner.course_chain)
        return value,J

    def bind_world_jacobian(self,row,p,*,columns=None,step=1e-6,samples=256):
        p=self._q(p);target=row.get('target')
        if not isinstance(target,dict):raise ValueError('Literal original admitted semantic target required')
        ids=np.arange(48)if columns is None else np.asarray(columns,int)
        if ids.ndim!=1 or np.any(ids<0)or np.any(ids>=48)or len(np.unique(ids))!=len(ids)or not np.isfinite(step)or step<=0.:raise ValueError('Explicit unique latent columns/positive FD step required')
        world=self.semantic_world(target,p,samples=samples);J=np.empty((*world.shape,len(ids)))
        for j,k in enumerate(ids):
            a=p.copy();b=p.copy();a[k]+=step;b[k]-=step
            J[:,:,j]=(self.semantic_world(target,a,samples=samples)-self.semantic_world(target,b,samples=samples))/(2*step)
        return world,J

    def feature_state(self,p):
        s=self.source_scene(p);return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in s['ice_features']},ice_boundary=s['ice_boundary'],parameters=s['parameters'],world_geometry_manifest=self.manifest(p))

    def trapezoid_side_segments(self,p,*,goal_carrier_span_m=None,board_carrier_span_m=None):
        p=self._q(p)
        if goal_carrier_span_m is not None or board_carrier_span_m is not None:
            if goal_carrier_span_m!=p[10]or board_carrier_span_m!=p[11]:raise ValueError('Trapezoid widths are active shared latent values, never a second fixed caller chart')
        s=self.source_scene(p);fs={f['id']:f for f in s['ice_features']}
        return{'goal_trapezoid_'+side+'_side_'+label:dict(world_scene=np.asarray(fs['goal_trapezoid_'+side+'_'+edge+'_outer']['segments_xyz'])[0],scene_unit_m=self.scene_unit_m,physical_t0='current structured goal carrier',physical_t1='actual current finite course',original_visible_endpoints_not_physical_measurements=True)for side in('left','right')for label,edge in(('minus','lower'),('plus','upper'))}

    def engineering_reference_rows(self,p,*,sigma_by_name):
        p=self._q(p)
        if not isinstance(sigma_by_name,dict):raise TypeError('Caller explicit engineering tolerances required; no default source sigma')
        r=[];J=[];metadata=[]
        for name,sigma in sigma_by_name.items():
            if name not in NOMINAL_REFERENCES or not np.isfinite(sigma)or sigma<=0.:raise ValueError('Known nominal relation and explicit positive engineering tolerance required')
            k=self.index[name];r.append((p[k]-NOMINAL_REFERENCES[name])/sigma);row=np.zeros(48);row[k]=1/sigma;J.append(row)
            metadata.append(dict(name=name,reference=NOMINAL_REFERENCES[name],engineering_sigma=float(sigma),not_measured_source_uncertainty=True))
        return np.asarray(r),np.asarray(J).reshape(-1,48),dict(role='explicit_engineering_reference_only',quadratic=True,relations=metadata,venue_version_unmeasured=True,metric_scale_uncertainty_shared_correlated_not_independent_marker_measurements=True)

    def manifest(self,p):
        p=self._q(p)
        return dict(schema=SCHEMA,named_chart_not_original_world47_or49=True,parameter_count=48,
            parameter_names=list(self.parameter_names),parameter_values=p.tolist(),initialization=self.initialization_ledger,
            source_parent_world49=self.parent_q.tolist(),all_existing_target_ID_UV_sigma_roles_preserved=True,
            source_worlds_changed_by_explicit_structural_projection_not_claimed_zero_exact=True,
            five_hockey_circles_common_radius_no_per_feature_drift=True,all_hockey_transverse_carriers_parallel=True,
            four_faceoff_pairs_share_goal_inward_and_halfY=True,common_curling_axis_tee_separation_rowpitch_eight_concentric_houses=True,
            Lane_three_painted_edge_identity_retained_not_relabelled_sheet_end_hog=True,shared_Lane_right_end_offset_active_not_historical_fixed_6m=True,
            expected_sheet_width_separate_unmeasured_variable_not_known_Lane_paint_width=True,
            original_left_Lane_vertices_still_unmeasured_display_proxies=True,all_58_expected_marks_UV_count_zero_not_FIT=True,
            original61knot_crease_shape_reoriented_scaled_not_replaced_by_modern_venue_assumption=True,
            current_boundary4_and_26print_materials_same_course_no_anisotropic_circle_scaling=True,
            current_template_finite_carrier_wall_intersections_not_legacy_endpoint_offsets=True,
            references_engineering_explicit_input_only_no_hard60x30_or_pixel_sigma0_division=True,
            whole_programme_source_qualification_actors_profile_reference205_guard_renderer_installation_pending=True)


def bind_structured_native_wide_source_rows(packet,geometry,p,*,samples=256):
    """Original45 qualified targets/UV/sigma, including rotated Lane support.

    The old native binder only accepted world-X/Y-aligned supporting lines.
    This own binder keeps the same rectangle-bounds support and named edge;
    it never selects a generic nearby mark or admits any expected template.
    """
    from wide_anchor_candidate.native_wide_source import NativeWideSourcePacket
    if not isinstance(packet,NativeWideSourcePacket)or type(geometry)is not StructuredSharedIceTemplateGeometry:
        raise TypeError('Authenticated original45 packet and explicit structured geometry required')
    result=[]
    for original in packet.observations:
        row=deepcopy(original);row['world']=geometry.semantic_world(row['target'],p,samples=samples)
        row['uv']=np.asarray(row['pixels_uv'],float).copy()
        row['source_pts']=int(packet.contract['source_pts']);row['source_representation']='v5'
        row['world_source_identity']='declared structured latent scene; original semantic target/finite support'
        result.append(row)
    return result


def bind_structured_native_source_bundle(packet,ad_packet,geometry,p,*,samples=256):
    """One actual source world bundle for one caller-owned 3D camera.

    No projection, loss, sigma replacement or implicit objective budget is
    inserted here. Camera14 must be applied to every curve and print corner
    using the same project_native invocation policy. Trapezoid spans already
    live at p[10:12]; USER268 remains a hard original-raster support guard.
    """
    from wide_anchor_candidate.wide_user_ad_and_secondary_source import bind_wide_ad_world
    curves=bind_structured_native_wide_source_rows(packet,geometry,p,samples=samples)
    ads=bind_wide_ad_world(ad_packet,geometry,p)
    if len(curves)!=45 or sum(len(r['uv'])for r in curves)!=2144 or len(ads['world'])!=26:
        raise ValueError('Original45/2144 plus26 unique source inventory changed')
    return dict(curves=curves,advertisements=ads,original_ad_source_rows=deepcopy(ad_packet['rows']),
        contract=dict(schema='rinkspace.one-camera-structured-native45-user26-source-bundle.v1',
            geometry_chart_schema=SCHEMA,geometry_parameter_count=48,high_camera_parameter_count=14,local_parameter_count=62,
            original_unique_curve_groups=45,original_unique_curve_UV=2144,original_unique_ad_corners=26,original_residual_component_rows=4340,
            curve_target_IDs=[r['id']for r in curves],ad_target_IDs=[r['id']for r in ad_packet['rows']],
            curves_keep_original_native_sigma_and_literal_source_budget=True,ads_keep_original_null_sigma_and_original_group_budget=True,
            ad_numerical_sigma_only_explicit_caller_engineering_policy=True,
            USER268_sigma0_not_variance_or_new_soft_source_rows=True,
            shared_trapezoid_width_indices=[10,11],no_additional_duplicate_span2=True,
            actual_one_3D_camera_projection_required_for_every_curve_top_ad_USER_support=True,
            Ad3_4_5_worlds_still_conditional_old_print_estimates_not_surveyed_ground_truth=True,
            no_camera_fit_or_fullsource_admission_or_wholegraph_installation=True))
