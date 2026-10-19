"""Independent print12 on the caller's actual finite board course.

No pixel observation, prior, half-turn relation, or metre calibration is added.
Ad5 remains the historically qualified internal graphic, not the whole print.
The original parent chart is a prefix, whatever its explicit parameter count.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import numpy as np
from ad_landmarks import AdGeometry, ROLES

SCHEMA = 'rinkspace.independent-unmeasured-Ad3-Ad4-Ad5-finite-course-print12.v1'
NUMBERS = (3, 4, 5)
FIELDS = ('station_center_m', 'arc_width_m', 'zmin_m', 'zmax_m')
CURVE_ROLES = ('left_vertical_xyz','right_vertical_xyz','top_edge_xyz','bottom_edge_xyz')
EDGE_NAMES = dict(left='left_vertical_xyz',right='right_vertical_xyz',top='top_edge_xyz',bottom='bottom_edge_xyz')
ID = re.compile(r'ad(\d+):(left_top|left_bottom|right_top|right_bottom|left_vertical_xyz|right_vertical_xyz|top_edge_xyz|bottom_edge_xyz|left|right|top|bottom)\Z', re.I)


def current_course_binding(parent, q):
    """Return actual course, q6, and its explicit 6-by-parent tangent.

    New full-boundary providers should expose print_course_binding(q). Older
    released world49/structured48 providers are supported by their exact
    course4 chain. No aspect, corner shape, or unknown dimension is guessed.
    """
    if callable(getattr(parent, 'print_course_binding', None)):
        course, current, chain = parent.print_course_binding(q)
    elif callable(getattr(parent, 'course_jacobian', None)) and hasattr(parent, 'course'):
        course, current, chain = parent.course, parent.current_course6(q), parent.course_jacobian(q)
    elif hasattr(parent, 'boundary_owner') and callable(getattr(parent, '_boundary_q', None)):
        owner = parent.boundary_owner
        course, current = owner.course, owner.current_course6(parent._boundary_q(q))
        chain = np.zeros((6, parent.parameter_count)); chain[:,44:48] = owner.course_chain
    elif hasattr(parent, 'original') and hasattr(parent.original, 'course_chain') and hasattr(parent.original, 'course'):
        course, current = parent.original.course, parent.current_course6(q)
        chain = np.zeros((6, parent.parameter_count)); chain[:,43:47] = parent.original.course_chain
    else:
        raise TypeError('Current provider must expose actual print_course_binding(q), not an inferred or frozen course')
    current, chain = np.asarray(current,float), np.asarray(chain,float)
    if current.shape != (6,) or chain.shape != (6,parent.parameter_count) or not np.isfinite(chain).all():
        raise ValueError('Actual course6 and its complete parent chart tangent required')
    return course, current, chain


class SharedUnmeasuredAdvertisementGeometry:
    """Parent prefix plus independent absolute s/width/Zmin/Zmax for 3 ads."""

    world_batch_protocol_schema='rinkspace.current-scene-semantic-world-and-finite-trapezoid-bundle.v1'

    def __init__(self, parent, parent_initial_q):
        self.parent = deepcopy(parent)
        self.parent_parameter_count = int(parent.parameter_count)
        self.parent_initial_q = self.parent.validate_parameters(parent_initial_q)
        self.base = self.parent.base
        self.scene_unit_m = float(self.parent.scene_unit_m)
        self.parameter_count = self.parent_parameter_count + 12
        self.parameter_names = tuple(parent.parameter_names) + tuple(f'Ad{n}_independent_{field}' for n in NUMBERS for field in FIELDS)
        self.parameter_units = tuple(parent.parameter_units) + ('m',)*12
        self.slot = {n:i for i,n in enumerate(NUMBERS)}
        self.old = {n:AdGeometry(self.base,n) for n in NUMBERS}
        self._materials = {}
        values = []
        self.initial_materials = {}
        for n in NUMBERS:
            old = self.old[n]
            if set(old.roles) != set(ROLES): raise ValueError('Four existing material corners are required; do not fabricate an edge')
            values.extend(((old._base_s['left']+old._width_sign*old._nominal_width/2)*self.scene_unit_m,
                old._nominal_width*self.scene_unit_m,
                old._base_height['bottom']*self.scene_unit_m,old._base_height['top']*self.scene_unit_m))
            self.initial_materials[n] = dict(width_sign=int(old._width_sign), original_extent_scope=self._ad(n).get('extent_scope','unspecified'))
            for role, world in self._ad(n).get('fitted_boundary_curves_xyz',{}).items():
                if role in CURVE_ROLES: self._materials[n,role] = self._material_coordinates(np.asarray(world,float),n,role)
        self.initial_print_parameters = np.asarray(values,float)
        self.initial = np.r_[self.parent_initial_q,self.initial_print_parameters]
        self.base_scene_sha256 = hashlib.sha256(json.dumps(self.base,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        current_course_binding(self.parent,self.parent_initial_q)
        self.validate_parameters(self.initial)

    def __reduce__(self):
        return (type(self),(self.parent,self.parent_initial_q))

    def _ad(self,n):
        found = [a for a in self.base['advertisements'] if int(a['user_number']) == int(n)]
        if len(found)!=1: raise ValueError('One existing modeled print material required')
        return found[0]

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,) or not np.isfinite(q).all(): raise ValueError('Complete explicit parent-plus-print12 chart required')
        return q

    def validate_parameters(self,q):
        q=self._q(q); self.parent.validate_parameters(q[:self.parent_parameter_count])
        course,cq,_=current_course_binding(self.parent,q[:self.parent_parameter_count])
        bottom,top,*_=course.sample_perimeter(cq,with_jacobian=True)
        perimeter=float(np.linalg.norm(np.roll(bottom,-1,axis=0)[:,:2]-bottom[:,:2],axis=1).sum())*self.scene_unit_m
        low=float(np.min(bottom[:,2]))*self.scene_unit_m; high=float(np.max(top[:,2]))*self.scene_unit_m
        for i,n in enumerate(NUMBERS):
            _,width,zmin,zmax=q[self.parent_parameter_count+4*i:self.parent_parameter_count+4*i+4]
            if width<=0 or width>=perimeter/2 or zmax<=zmin: raise ValueError(f'Ad{n} needs positive minor arc width and physical height')
            if zmin<low-1e-8 or zmax>high+1e-8: raise ValueError(f'Ad{n} print exceeds actual current board Z domain')
        return q.copy()

    def identities(self,row):
        roles=row.get('physical_world_corner_roles')
        if roles is not None:
            n=row.get('physical_ad_id',row.get('ad_id'))
            if n is None or any(r not in ROLES for r in roles): raise ValueError('Explicit existing print number and named corner roles required')
            return [(int(n),r) for r in roles]
        vals=row.get('physical_ids')
        if vals:
            parsed=[ID.fullmatch(str(v)) for v in vals]
            if all(m is not None for m in parsed): return [(int(m[1]),EDGE_NAMES.get(m[2].lower(),m[2].lower())) for m in parsed]
            if any(m is not None and int(m[1]) in self.slot for m in parsed): raise ValueError('Mixed unsupported identities on an active print')
        m=ID.fullmatch(str(row.get('id','')))
        if m is not None: return [(int(m[1]),EDGE_NAMES.get(m[2].lower(),m[2].lower()))]
        n=row.get('physical_ad_id',row.get('ad_id')); role=row.get('role')
        if n is not None and role in (*ROLES,*CURVE_ROLES,*EDGE_NAMES): return [(int(n),EDGE_NAMES.get(role,role))]
        if n is not None and int(n) in self.slot: raise ValueError('Active print lacks original material role')
        return None

    def _material_coordinates(self,world,n,role):
        world=np.asarray(world,float); old=self.old[n]
        if world.ndim!=2 or world.shape[1]!=3 or not len(world) or not np.isfinite(world).all(): raise ValueError('Original finite material sampling required')
        stations=np.array([old._nearest_arc(p[:2]) for p in world])
        snapped=np.array([old._point_at_arc(s) for s in stations])
        if np.max(np.linalg.norm(snapped-world[:,:2],axis=1))>1e-7: raise ValueError('Source is off actual original finite board; no snapping is allowed')
        ds=(stations-old._base_s['left']+old._perimeter/2)%old._perimeter-old._perimeter/2
        u=ds/(old._width_sign*old._nominal_width)
        v=(world[:,2]-old._base_height['bottom'])/(old._base_height['top']-old._base_height['bottom'])
        if np.any((u < -1e-7)|(u > 1+1e-7)|(v < -1e-7)|(v > 1+1e-7)): raise ValueError('Source exceeds original finite print material')
        if role in CURVE_ROLES:
            values=u if role.startswith(('left_','right_')) else v
            expected=float(role.startswith(('right_','top_')))
            if np.max(abs(values-expected))>1e-7: raise ValueError('Original source is not its named print boundary')
        return u,v

    def _material(self,n,u,v,q):
        q=self._q(q); k=self.parent_parameter_count+4*self.slot[n]
        s,width,zmin,zmax=q[k:k+4]; old=self.old[n]
        u,v=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float)); u,v=u.reshape(-1),v.reshape(-1)
        stations=s+old._width_sign*(u-.5)*width; z=(1-v)*zmin+v*zmax
        course,cq,chain=current_course_binding(self.parent,q[:self.parent_parameter_count])
        world,J6,Js,report=course.station(stations,cq,z_m=z,with_jacobian=True)
        J=np.zeros((*world.shape,self.parameter_count)); J[:,:,:self.parent_parameter_count]=np.einsum('nik,kj->nij',J6,chain)
        J[:,:,k]=Js; J[:,:,k+1]=Js*(old._width_sign*(u-.5))[:,None]
        J[:,2,k+2]=(1-v)/self.scene_unit_m; J[:,2,k+3]=v/self.scene_unit_m
        return world,J,report

    def _world_and_jacobian(self,row,q):
        q=self._q(q); prefix=q[:self.parent_parameter_count]
        ids=self.identities(row)
        if ids is None or all(n not in self.slot for n,_ in ids):
            rebound=self.parent.rebind(row,prefix)
            value=np.asarray(rebound['world'],float); J=self.parent.world_jacobian(row,prefix)
            return value,np.concatenate((J,np.zeros((*J.shape[:-1],12))),axis=-1)
        original=np.asarray(row['world'],float)
        # Parent explicitly owns the unchanged source-world and old tangent.
        rebound=self.parent.rebind(row,prefix); value=np.asarray(rebound['world'],float).copy()
        oldJ=self.parent.world_jacobian(row,prefix)
        J=np.concatenate((oldJ,np.zeros((*oldJ.shape[:-1],12))),axis=-1)
        if all(role in ROLES for _,role in ids):
            if len(ids)!=len(original): raise ValueError('Named corner identities and original world count differ')
            for i,(n,role) in enumerate(ids):
                if n not in self.slot: continue
                expected=np.asarray(self._ad(n)['corners_xyz_scene_units'][role],float)
                if not np.array_equal(original[i],expected): raise ValueError('Original named print world authority changed')
                side,level=role.split('_'); w,j,_=self._material(n,float(side=='right'),float(level=='top'),q)
                k=self.parent_parameter_count+4*self.slot[n]
                if np.array_equal(q[k:k+4],self.initial_print_parameters[4*self.slot[n]:4*self.slot[n]+4]):
                    J[i,:,self.parent_parameter_count:]=j[0,:,self.parent_parameter_count:]
                else: value[i],J[i]=w[0],j[0]
        elif len(ids)==1 and ids[0][0] in self.slot and ids[0][1] in CURVE_ROLES:
            n,role=ids[0];u,v=self._material_coordinates(original,n,role);w,j,_=self._material(n,u,v,q)
            k=self.parent_parameter_count+4*self.slot[n]
            if np.array_equal(q[k:k+4],self.initial_print_parameters[4*self.slot[n]:4*self.slot[n]+4]): J[:,:,self.parent_parameter_count:]=j[:,:,self.parent_parameter_count:]
            else: value,J=w,j
        else: raise ValueError('Active print source needs named corners or one original finite edge')
        return value,J

    def rebind(self,row,q):
        value,_=self._world_and_jacobian(row,q); out=dict(row)
        out['world']=value if isinstance(row['world'],np.ndarray) else value.tolist()
        for key in ('world_xyz','world_xyz_scene_units'):
            if key in row:
                old=np.asarray(row[key])
                if old.shape==(3,) and len(value)==1: out[key]=value[0].tolist()
                elif old.shape==value.shape: out[key]=value.tolist()
                else: raise ValueError('Companion world shape does not match the same source material')
        return out

    def world_jacobian(self,row,q): return self._world_and_jacobian(row,q)[1]

    def corner(self,n,role,q,*,with_jacobian=False):
        if n not in self.slot or role not in ROLES: raise ValueError('One of the three original selected print corners required')
        q=self._q(q);k=self.parent_parameter_count+4*self.slot[n]
        if not np.array_equal(q[k:k+4],self.initial_print_parameters[4*self.slot[n]:4*self.slot[n]+4]):
            side,level=role.split('_');w,J,_=self._material(n,float(side=='right'),float(level=='top'),q)
            return (w[0],J[0]) if with_jacobian else w[0]
        source=dict(id=f'ad{n}:{role}',physical_ad_id=n,physical_world_corner_roles=[role],world=[self._ad(n)['corners_xyz_scene_units'][role]])
        w,J=self._world_and_jacobian(source,q)
        return (w[0],J[0]) if with_jacobian else w[0]

    def current_course6(self,q): return current_course_binding(self.parent,self._q(q)[:self.parent_parameter_count])[1]

    def print_course_binding(self,q):
        course,cq,J=current_course_binding(self.parent,self._q(q)[:self.parent_parameter_count])
        return course,cq,np.column_stack((J,np.zeros((6,12))))

    def _scene(self,q,rebuild_grid):
        q=self.validate_parameters(q); prefix=q[:self.parent_parameter_count]
        out=self.parent.scene(prefix) if rebuild_grid else self.parent.source_scene(prefix)
        ads={int(a['user_number']):a for a in out['advertisements']}
        for n,i in self.slot.items():
            x=q[self.parent_parameter_count+4*i:self.parent_parameter_count+4*i+4]
            if np.array_equal(x,self.initial_print_parameters[4*i:4*i+4]): continue
            ad=ads[n]
            ad['corners_xyz_scene_units']={r:self.corner(n,r,q).tolist() for r in ROLES}
            ad['fitted_boundary_curves_xyz']={r:self._material(n,*uv,q)[0].tolist() for (nn,r),uv in self._materials.items() if nn==n}
            ad['width_m']=float(x[1]);ad['height_m']=float(x[3]-x[2])
            ad['bottom_z_scene_units']=float(x[2]/self.scene_unit_m);ad['top_z_scene_units']=float(x[3]/self.scene_unit_m)
            ad['geometry_status']='independent_conditional_print_geometry_candidate_not_measured'
            ad[SCHEMA]=dict(parameters_metres=x.tolist(),source_extent_scope=self.initial_materials[n]['original_extent_scope'],paired_partner_not_hard_constrained=True)
        for row in out.get('advertisement_3d_corners',[]):
            n=int(row['user_number']);role=row['role']
            if n in self.slot and not np.array_equal(q[self.parent_parameter_count+4*self.slot[n]:self.parent_parameter_count+4*self.slot[n]+4],self.initial_print_parameters[4*self.slot[n]:4*self.slot[n]+4]):
                row['world_xyz_scene_units']=self.corner(n,role,q).tolist()
        out[SCHEMA]=self.manifest(q)
        return out

    def source_scene(self,q): return self._scene(q,False)
    def scene(self,q): return self._scene(q,True)
    def semantic_world(self,target,q,*,samples=256): return self.parent.semantic_world(target,self._q(q)[:self.parent_parameter_count],samples=samples)
    def one_scene_source_world_bundle(self,q,scene,rows):
        q=self._q(q)
        reader=getattr(self.parent,'one_scene_source_world_bundle',None)
        if not callable(reader):
            from complete_ice_template_candidate.full_free_boundary_geometry import semantic_world_from_current_scene
            return [semantic_world_from_current_scene(scene,row['target'],scene_unit_m=self.scene_unit_m) for row in rows]
        return reader(q[:self.parent_parameter_count],scene,rows)
    def one_scene_trapezoid_ends(self,q,scene,spans=None):
        q=self._q(q); reader=getattr(self.parent,'one_scene_trapezoid_ends',None)
        if callable(reader):return reader(q[:self.parent_parameter_count],scene,spans)
        from complete_ice_template_candidate.full_free_boundary_geometry import finite_trapezoid_ends_from_current_scene
        return finite_trapezoid_ends_from_current_scene(scene,self.scene_unit_m,spans)
    def feature_state(self,q):
        s=self.source_scene(q); return dict(scene_unit_m=self.scene_unit_m,ice_features={f['id']:f for f in s['ice_features']},ice_boundary=s['ice_boundary'],parameters=s['parameters'],world_geometry_manifest=self.manifest(q))
    def trapezoid_side_segments(self,q,**kwargs): return self.parent.trapezoid_side_segments(self._q(q)[:self.parent_parameter_count],**kwargs)
    def finite_board_records(self,records,q,**kwargs): return self.parent.finite_board_records(records,self._q(q)[:self.parent_parameter_count],**kwargs)
    def engineering_reference_rows(self,q,**kwargs):
        r,J,meta=self.parent.engineering_reference_rows(self._q(q)[:self.parent_parameter_count],**kwargs)
        return r,np.column_stack((J,np.zeros((len(J),12)))),meta
    def bind_world_jacobian(self,row,q,*,columns=None,step=1e-6,samples=256):
        q=self._q(q);columns=np.arange(self.parameter_count) if columns is None else np.asarray(columns,int)
        target=row['target'];world=self.semantic_world(target,q,samples=samples);J=np.zeros((*world.shape,len(columns)))
        for i,k in enumerate(columns):
            if k>=self.parent_parameter_count: continue
            p=q.copy();m=q.copy();p[k]+=step;m[k]-=step
            J[:,:,i]=(self.semantic_world(target,p,samples=samples)-self.semantic_world(target,m,samples=samples))/(2*step)
        return world,J

    def manifest(self,q):
        q=self._q(q)
        return dict(schema=SCHEMA,parent_world_parameter_count=self.parent_parameter_count,world_parameter_count=self.parameter_count,
            active_independent_advertisements=list(NUMBERS),print_parameter_indices=list(range(self.parent_parameter_count,self.parameter_count)),
            print_parameter_names=list(self.parameter_names[self.parent_parameter_count:]),print_parameters_conditional_metres=q[self.parent_parameter_count:].tolist(),
            initial_conditional_metres=self.initial_print_parameters.tolist(),scene_unit_metres=self.scene_unit_m,base_scene_sha256=self.base_scene_sha256,
            historical_ray_board_intersection_worlds_are_initializers_not_surveyed_measurements=True,
            Ad5_internal_graphic_original_scope_retained=True,hard180_partner_geometry_constraint=False,
            counterparts_16_17_18_preserve_parent_values_not_new_measurements=True,print_prior_rows_added=0,
            original_UV_sigma_roles_qualification_and_finite_material_sampling_unchanged=True,
            finite_actual_course_station_not_straight_chord_across_corner=True,zero_print_old_world_and_old_J_literal=True,
            boundary_height_and_print_domain_common_current_course=True,scene_scale_conditional_not_field_surveyed=True,
            graph_installed=False,solver_ran=False,accuracy_accepted=False)
