"""User original raster supports tested against current finite world carriers.

The world signs are inferred once at the explicit seed and recorded, never
changed during trial evaluations. Visible pixel trace ends do not become
physical material endpoints. No sigma or engineering pixel margin exists.
"""
from simple_wide_perspective_reset_candidate.unified_physical_native_factor import *
from simple_wide_perspective_reset_candidate.exact_user_raster_shape import ExactUserRasterShape,CENTER,corrected
from scipy.optimize import least_squares


class WorldTrapezoidRasterSupport:
    def __init__(self,geometry,seed_camera14,seed_world,seed_spans=(6.8,8.6),*,source_scene_provider=None):
        self.g=geometry;self._scene_provider=source_scene_provider;self.shape=ExactUserRasterShape(support_mode='original_pen_footprint_at_same_original_x_cell')
        names=['goal_trapezoid_right_side_minus','goal_trapezoid_right_side_plus']
        segments=self.g.trapezoid_side_segments(seed_world,goal_carrier_span_m=seed_spans[0],board_carrier_span_m=seed_spans[1])
        costs=np.empty((2,2))
        for gi,group in enumerate(self.shape.groups):
            uv=np.asarray(group['native_source_UV'],float)
            for si,name in enumerate(names):
                w=segments[name]['world_scene'];knots=w[0]+np.linspace(0,1,257)[:,None]*(w[1]-w[0])
                p,_,v,_=project_native(seed_camera14,knots)
                raw=nearest_finite(uv,p,None,v[:-1]&v[1:])[0];costs[gi,si]=np.sum(raw*raw)
        assignment=[0,1]if costs[0,0]+costs[1,1]<=costs[0,1]+costs[1,0]else[1,0]
        self.names=[names[k]for k in assignment]
        self.identity_ledger=[dict(user_id=g['id'],current_world_primitive=name,
            mapping_method='minimum complete original trace squared distance at declared seed; inferred signedY only, fixed throughout',
            mapping_not_physical_endpoint_qualification=True,visible_trace_ends_not_assigned_material_t=True,
            alternative_seed_native_costs=costs[gi].tolist())for gi,(g,name)in enumerate(zip(self.shape.groups,self.names))]

    def _segments(self,qworld,spans):
        if self._scene_provider is None:return self.g.trapezoid_side_segments(qworld,goal_carrier_span_m=spans[0],board_carrier_span_m=spans[1])
        from complete_ice_template_candidate.complete_ice_numeric_geometry import _frame,_boundary_hit
        spans=np.asarray(spans,float)
        if not np.isfinite(spans).all()or spans[0]<=0 or spans[1]<=spans[0]:raise ValueError('Two positive shared outward trapezoid spans required')
        scene=self._scene_provider(qworld);features={f['id']:f for f in scene['ice_features']};out={}
        for side in ('left','right'):
            frame=features['goal_line_'+side].get('shared_physical_goal_frame')
            if frame is None:centre,inside,across=_frame(features,'goal_line_'+side)
            else:centre,inside,across=(np.asarray(frame[k],float)for k in ('centre_xyz','inward_xyz','across_xyz'))
            for sign,label in ((-1,'minus'),(1,'plus')):
                a=centre+sign*across*(spans[0]/(2*self.g.scene_unit_m));seed=centre+sign*across*(spans[1]/(2*self.g.scene_unit_m))
                b=_boundary_hit(seed,-inside,scene['ice_boundary']['points_xyz'])
                out['goal_trapezoid_'+side+'_side_'+label]=dict(world_scene=np.array([a,b]))
        return out

    def line_state(self,camera14,qworld,spans):
        spans=np.asarray(spans,float)
        if spans.shape!=(2,):raise ValueError('Two shared goal and board carrier spans required')
        segments=self._segments(qworld,spans)
        q=list(camera14[10:]);undends=[]
        c=pinhole_camera25(camera14)
        for name in self.names:
            ends=segments[name]['world_scene'];p,depth,valid=project_central_valid(c,ends)
            if not valid.all():raise ValueError('True finite world trapezoid endpoint leaves positive camera depth')
            delta=p[1]-p[0];length=np.linalg.norm(delta)
            if length<=0:raise ValueError('True finite projected trapezoid degenerate')
            n=np.array([-delta[1],delta[0]])/length
            if n[1]<0:n=-n
            q.extend([np.arctan2(n[1],n[0]),n@((p[0]-CENTER)/1000)])
            undends.append(p)
            segments[name]['projected_endpoint_depth_scene']=depth
        return np.asarray(q),undends,segments

    def evaluate(self,camera14,qworld,spans,*,report=False):
        state,undends,segments=self.line_state(camera14,qworld,spans)
        r,_,_=self.shape.evaluate(state,False,False)
        index=0;finite=[];proof_groups=[];margin=radial_disk_guard(camera14[10:],self.shape.all_native)
        for gi,(group,p)in enumerate(zip(self.shape.groups,undends)):
            delta=p[1]-p[0];length=np.linalg.norm(delta);alpha,offset=state[4+2*gi:6+2*gi];normal=np.array([np.cos(alpha),np.sin(alpha)])
            cs=self.shape.corner_arrays[gi];und=corrected(cs,camera14[10:]);F=np.einsum('...i,i->...',und-CENTER,normal)-1000*offset
            S=np.einsum('...i,i->...',und-p[0],delta)/(length*length)
            nativeends=inverse_map_forward(p,camera14[10:])[0];depths=segments[self.names[gi]]['projected_endpoint_depth_scene']
            def certify(native,cell,ri,kind,extra=None):
                if kind=='physical_endpoint':
                    ei=extra['physical_endpoint_index'];uu=p[ei];tp=float(ei)
                else:uu=corrected(native[None,:],camera14[10:])[0];tp=float((uu-p[0])@delta/(length*length))
                if not 0<=tp<=1:return None
                forward=inverse_map_forward(uu[None,:],camera14[10:])[0][0]
                if not np.all((forward>=cell[0]-1e-9)&(forward<=cell[1]+1e-9)):return None
                tw=float(tp*depths[0]/(depths[1]+(depths[0]-depths[1])*tp))
                annotated=(forward+np.array([0.,65.])+.5)/np.array([2580/1500,1160/675])-.5
                result=dict(selected_original_region_index=int(ri),exact_native_cell=cell.tolist(),native_forward_intersection=forward.tolist(),
                    annotation_forward_intersection=annotated.tolist(),equation=float((uu-CENTER)@normal/1000-offset),
                    projected_finite_fraction=tp,actual_world_material_fraction=tw,finite_aware_proof_kind=kind,covered=True)
                if extra:result.update(extra)
                return result
            points=[];fractions=[];worldfractions=[]
            for i in range(len(cs)):
                result=None;regions=range(int(self.shape.region_counts[gi][i]))
                for ri in regions:
                    cell=self.shape.cells[gi][i,ri]
                    # A true finite endpoint inside an original pen rectangle
                    # is already an exact curve/rectangle intersection.
                    for ei,native in enumerate(nativeends):
                        if np.all((native>=cell[0])&(native<=cell[1])):
                            result=certify(native,cell,ri,'physical_endpoint',dict(physical_endpoint_index=ei))
                            if result is not None:break
                    if result is not None:break
                    values=F[i,ri]
                    if values.min()>0 or values.max()<0:continue
                    # Try the original corner diagonal, then each original
                    # rectangle edge. Another union region/root may be inside
                    # the finite domain even when the first covered root is not.
                    pairs=[(int(values.argmin()),int(values.argmax()))]+[(k,(k+1)%4)for k in range(4)]
                    for ai,bi in pairs:
                        if values[ai]*values[bi]>0:continue
                        if values[ai]<=0:lo,hi=cs[i,ri,ai].copy(),cs[i,ri,bi].copy()
                        else:lo,hi=cs[i,ri,bi].copy(),cs[i,ri,ai].copy()
                        for _ in range(48):
                            mid=.5*(lo+hi);f=float((corrected(mid[None,:],camera14[10:])[0]-CENTER)@normal-1000*offset)
                            if f<=0:lo=mid
                            else:hi=mid
                        native=.5*(lo+hi);result=certify(native,cell,ri,'original_cell_edge'if (ai-bi)%4 in(1,3)else'original_corner_diagonal',dict(original_corner_pair=[ai,bi],native_root_bracket=[cs[i,ri,ai].tolist(),cs[i,ri,bi].tolist()]))
                        if result is not None:break
                    if result is not None:break
                if result is None:
                    # Unproved is not a mathematical global infeasibility
                    # certificate. It is never accepted as source coverage.
                    if r[index+i]==0:
                        smin=float(S[i].min());smax=float(S[i].max());r[index+i]=max(1e-12,max(-smax,smin-1)*length)
                    result=dict(covered=False,proof_status='no certified finite intersection in any original pen rectangle')
                else:
                    r[index+i]=0.;fractions.append(result['projected_finite_fraction']);worldfractions.append(result['actual_world_material_fraction'])
                result.update(original_point_index=i,original_native_UV=group['native_source_UV'][i],original_region_count=int(self.shape.region_counts[gi][i]))
                points.append(result)
            count=sum(point['covered']for point in points)
            proof_groups.append(dict(id=group['id'],support_mode=self.shape.support_mode,original_points=len(points),proved_original_support_intersections=count,
                all_original_supports_covered=count==len(points),points=points))
            finite.append(dict(id=group['id'],current_world_primitive=self.names[gi],actual_world_endpoints_scene=segments[self.names[gi]]['world_scene'].tolist(),
                undistorted_endpoints_px=p.tolist(),endpoint_depth_scene=depths.tolist(),finite_outside_or_unproved_points=len(points)-count,actual_certified_projection_fractions=fractions,actual_certified_world_material_fractions=worldfractions,all_finite=count==len(points)))
            index+=len(group['native_source_UV'])
        proof=dict(radial_disk_minimum=margin,groups=proof_groups,original_pen_union_regions_enumerated_without_filling_gaps=True,finite_aware_region_and_root_selection=True)
        return r,dict(original_pixel_support_proof=proof,finite_world_support=finite,identity_ledger=self.identity_ledger,
            span_parameters_are_shared_physical_lengths_not_measured_outside_paint_width=True,
            original268_ids_all_preserved=True,source_sigma0_not_divided_or_replaced=True,
            unknown_image_line4_not_an_independent_camera_or_source_parameter=True)if report else None

    def span_initialization(self,camera14,qworld,spans=(6.8,8.6)):
        def residual(q):
            if q[0]<=0 or q[1]<=q[0]:return np.full(268,1e5)
            try:return self.evaluate(camera14,qworld,q)[0]
            except ValueError:return np.full(268,1e5)
        fit=least_squares(residual,np.asarray(spans,float),jac='3-point',max_nfev=120,ftol=1e-12,xtol=1e-12,gtol=1e-10,x_scale=[1,1])
        r=residual(fit.x)
        return fit.x,dict(initial_shared_spans_m=list(spans),fitted_shared_spans_m=fit.x.tolist(),hard_feasibility_merit=float(.5*r@r),maximum_native_equation_violation=float(abs(r).max()),
            all268_hard_support_proved=bool(np.all(r==0)),initializer_only=True)

    def constraints(self,q,*,jacobian=True,selection=None,finite_mode='interval_intersection'):
        """Actual c>=0/Jc for camera14+completeworldN+sharedspans2.

        For each literal original x-cell choose one *existing* pen rectangle:
        [-minF,maxF,minS,length-maxS]>=0. F is the native inverse-polynomial
        line equation in pixels, S is undistorted longitudinal distance.
        Full-rectangle longitudinal containment is a conservative sufficient
        finite-carrier certificate, not a new uncertainty width. The final
        guard additionally checks actual original root/finite fractions.
        No union gap is filled; a caller can fix selections for one QP's
        linearization, then reevaluate the original union on nonlinear trial.
        """
        q=np.asarray(q,float);nw=int(self.g.parameter_count);n=14+nw+2
        if q.shape!=(n,):raise ValueError('camera14+complete declared world+sharedspans2 required')
        if finite_mode not in ('interval_intersection','full_rectangle'):raise ValueError('Explicit finite interval certificate mode required')
        camera=q[:14];qw=q[14:14+nw];spans=q[-2:]
        radial_disk_guard(camera[10:],self.shape.all_native)
        segments=self._segments(qw,spans)
        worlds=np.concatenate([segments[name]['world_scene']for name in self.names])
        c=pinhole_camera25(camera)
        if jacobian:
            p,depth,j11,valid=project_central_valid(c,worlds,with_jacobian=True)
            jw=np.empty((4,3,nw+2));h=1e-6
            for k in range(nw+2):
                z=np.r_[qw,spans];zp,zm=z.copy(),z.copy();zp[k]+=h;zm[k]-=h
                def ends(value):
                    ss=self._segments(value[:nw],value[-2:])
                    return np.concatenate([ss[name]['world_scene']for name in self.names])
                jw[:,:,k]=(ends(zp)-ends(zm))/(2*h)
            j11=j11.reshape(4,2,11);jp=np.zeros((4,2,n));jp[:,:,:10]=j11[:,:,:10]
            jp[:,:,14:]=np.einsum('nij,njk->nik',-j11[:,:,:3]/100,jw)
        else:p,depth,valid=project_central_valid(c,worlds);jp=None
        if not valid.all():raise ValueError('True finite world trapezoid endpoint leaves positive physical camera domain')
        values=[];derivatives=[];records=[];chosen=[];pointoffset=0
        for gi,cs in enumerate(self.shape.corner_arrays):
            p0,p1=p[2*gi:2*gi+2];delta=p1-p0;length=np.linalg.norm(delta)
            if length<=0:raise ValueError('Finite actual projected carrier is degenerate')
            tangent=delta/length;normal=np.array([-tangent[1],tangent[0]])
            if normal[1]<0:normal=-normal
            und,ju=corrected(cs,camera[10:],True);off=und-p0
            F=np.einsum('...i,i->...',off,normal);S=np.einsum('...i,i->...',off,tangent)
            minF=F.argmin(axis=2);maxF=F.argmax(axis=2);minS=S.argmin(axis=2);maxS=S.argmax(axis=2)
            fmin=np.take_along_axis(F,minF[:,:,None],axis=2)[:,:,0];fmax=np.take_along_axis(F,maxF[:,:,None],axis=2)[:,:,0]
            smin=np.take_along_axis(S,minS[:,:,None],axis=2)[:,:,0];smax=np.take_along_axis(S,maxS[:,:,None],axis=2)[:,:,0]
            full=finite_mode=='full_rectangle'
            allc=np.stack([-fmin,fmax,smin if full else smax,length-(smax if full else smin)],axis=2)
            # Duplicated padding contains the same literal first rectangle;
            # therefore it cannot create a new feasible pixel region.
            region=np.argmax(allc.min(axis=2),axis=1)if selection is None else np.asarray(selection[pointoffset:pointoffset+len(cs)],int)
            ii=np.arange(len(cs));selected=allc[ii,region];values.append(selected.reshape(-1));chosen.extend(region.tolist())
            if jacobian:
                j0,j1=jp[2*gi:2*gi+2];jd=j1-j0;jlen=tangent@jd
                jt=(np.eye(2)-np.outer(tangent,tangent))@jd/length
                rotation=np.array([[0.,-1.],[1.,0.]])
                jn=rotation@jt
                if (rotation@tangent)@normal<0:jn=-jn
                juall=np.zeros((*und.shape,n));juall[...,10:14]=ju
                joff=juall-j0
                jf=np.einsum('...ik,i->...k',joff,normal)+np.einsum('...i,ik->...k',off,jn)
                js=np.einsum('...ik,i->...k',joff,tangent)+np.einsum('...i,ik->...k',off,jt)
                jselected=np.stack([-jf[ii,region,minF[ii,region]],jf[ii,region,maxF[ii,region]],
                    js[ii,region,(minS if full else maxS)[ii,region]],jlen-js[ii,region,(maxS if full else minS)[ii,region]]],axis=1)
                derivatives.append(jselected.reshape(-1,n))
            records.append(dict(user_id=self.shape.groups[gi]['id'],original_points=len(cs),current_world_primitive=self.names[gi],selected_original_pen_region=region.tolist(),
                finite_mode=finite_mode,conservative_finite_rectangle_certificate=full,finite_interval_only_not_an_actual_root_certificate=not full,negative_constraints=int((selected<0).sum()),minimum_constraint=float(selected.min())))
            pointoffset+=len(cs)
        return np.concatenate(values),np.vstack(derivatives)if jacobian else None,dict(selection=chosen,groups=records,
            finite_mode=finite_mode,constraint_order_per_original_point=['negative_min_corner_line_equation','positive_max_corner_line_equation','finite_interval_lower_overlap','finite_interval_upper_overlap'],
            constraints_units='native line-equation px and corrected-carrier longitudinal px; hard zero with no sigma',
            undistorted_endpoints_px=p.tolist(),endpoint_jacobian=None if jp is None else jp.tolist(),inverse4_jacobian=np.eye(n)[10:14].tolist(),
            original268_ids_regions_all_preserved=True,actual_original_union_final_guard_still_required=True)
