"""Continuous independent print material on the actual current finite course.

V1 is retained byte-for-byte. V2 initializes absolute print s/width/Z from
the actual parent scene and has no literal-tail value/J special branch.
Original source material u/v and native observations remain untouched.
"""
from copy import deepcopy
import hashlib,json
import numpy as np
from ad_landmarks import ROLES
from .shared_unmeasured_advertisement_geometry import (
    SharedUnmeasuredAdvertisementGeometry as _V1,NUMBERS,CURVE_ROLES,
    current_course_binding,SCHEMA as V1_SCHEMA)

SCHEMA="rinkspace.continuous-independent-unmeasured-Ad3-Ad4-Ad5-current-course-print12.v2"

class SharedUnmeasuredAdvertisementGeometryV2(_V1):
    def __init__(self,parent,parent_initial_q):
        super().__init__(parent,parent_initial_q)
        source=self.parent.source_scene(self.parent_initial_q)
        ads={int(a["user_number"]):a for a in source["advertisements"]}
        course,cq,_=current_course_binding(self.parent,self.parent_initial_q)
        bottom,top,*_=course.sample_perimeter(cq,with_jacobian=True)
        xy=np.asarray(bottom,float)[:,:2]
        edges=np.roll(xy,-1,axis=0)-xy
        lens=np.linalg.norm(edges,axis=1)
        if np.any(lens<=0):raise ValueError("Actual finite current course has degenerate segment")
        length2=lens*lens;cum=np.r_[0.,np.cumsum(lens)];perimeter=float(cum[-1])
        def station(point):
            p=np.asarray(point,float)
            t=np.clip(np.einsum("ij,ij->i",p[:2]-xy,edges)/length2,0.,1.)
            hits=xy+t[:,None]*edges;distance2=np.sum((hits-p[:2])**2,axis=1)
            k=int(np.argmin(distance2))
            if float(distance2[k])>1e-14:
                raise ValueError("Parent current print is off actual finite course; no pixel-derived snapping")
            return float((cum[k]+t[k]*lens[k])%perimeter)
        params=[];cert=[]
        for n in NUMBERS:
            corners={r:np.asarray(w,float)for r,w in ads[n]["corners_xyz_scene_units"].items()}
            if set(corners)!=set(ROLES):raise ValueError("Actual parent print must retain exactly4 corners")
            for side in ("left","right"):
                if np.max(abs(corners[side+"_top"][:2]-corners[side+"_bottom"][:2]))>1e-10:
                    raise ValueError("Original current print sides must be vertical on finite course")
            zmin=float(corners["left_bottom"][2]*self.scene_unit_m)
            zmax=float(corners["left_top"][2]*self.scene_unit_m)
            if max(abs(corners["right_bottom"][2]*self.scene_unit_m-zmin),
                   abs(corners["right_top"][2]*self.scene_unit_m-zmax))>1e-10:
                raise ValueError("Original current print top/bottom height differs acrosscorners")
            left,right=station(corners["left_top"]),station(corners["right_top"])
            sign=self.old[n]._width_sign
            signed=(right-left+perimeter/2)%perimeter-perimeter/2
            width=float(sign*signed*self.scene_unit_m)
            centre=float((left+sign*width/self.scene_unit_m/2)%perimeter)*self.scene_unit_m
            params.extend((centre,width,zmin,zmax))
            cert.append(dict(ad=n,parent_current_left_station_scene=left,
                parent_current_right_station_scene=right,width_sign=int(sign),
                current_course_perimeter_scene=perimeter,
                parameter_units="conditional metre",world_origin="actual parent current source_scene, not original base ad station"))
        self.initial_print_parameters=np.asarray(params,float)
        self.initial=np.r_[self.parent_initial_q,self.initial_print_parameters]
        self.initialization_contract=dict(
            schema="rinkspace.current-parent-print12-material-initialization.v2",
            actual_current_course_parameters=cq.tolist(),materials=cert,
            original_source_UV_sigma_role_or_pixel_qualification_modified=False,
            independent_station_is_a_new_physical_parameterization=True,
            initial_value_only_must_match_parent_current_scene=True,
            parent_old_J_not_claimed_equivalent_under_new_independent_station=True,
            no_equality_branch_in_value_or_J=True)
        self.validate_parameters(self.initial)
        current=self.source_scene(self.initial)
        by={int(a["user_number"]):a for a in current["advertisements"]}
        error=max(float(np.max(abs(np.asarray(by[n]["corners_xyz_scene_units"][r])-ads[n]["corners_xyz_scene_units"][r])))
                  for n in NUMBERS for r in ROLES)
        if error>2e-10:raise ValueError("Actual constructor print corner equivalence failed:"+str(error))
        self.initialization_contract["maximum_initial_parent_corner_error_scene_units"]=error

    def _world_and_jacobian(self,row,q):
        q=self._q(q);ids=self.identities(row);prefix=q[:self.parent_parameter_count]
        if ids is None or all(n not in self.slot for n,_ in ids):
            value=np.asarray(self.parent.rebind(row,prefix)["world"],float)
            J=self.parent.world_jacobian(row,prefix)
            return value,np.concatenate((J,np.zeros((*J.shape[:-1],12))),axis=-1)
        original=np.asarray(row["world"],float)
        if all(role in ROLES for _,role in ids):
            if len(ids)!=len(original):raise ValueError("Original namedcorner identity count differs")
            if all(n in self.slot for n,_ in ids):
                value=original.copy();J=np.zeros((*original.shape,self.parameter_count))
            else:
                value=np.asarray(self.parent.rebind(row,prefix)["world"],float).copy()
                oldJ=self.parent.world_jacobian(row,prefix)
                J=np.concatenate((oldJ,np.zeros((*oldJ.shape[:-1],12))),axis=-1)
            for i,(n,role)in enumerate(ids):
                if n not in self.slot:continue
                if not np.array_equal(original[i],np.asarray(self._ad(n)["corners_xyz_scene_units"][role],float)):
                    raise ValueError("Original named print material authority changed")
                side,level=role.split("_");w,j,_=self._material(n,float(side=="right"),float(level=="top"),q)
                value[i],J[i]=w[0],j[0]
            return value,J
        if len(ids)==1 and ids[0][0]in self.slot and ids[0][1]in CURVE_ROLES:
            n,role=ids[0];u,v=self._material_coordinates(original,n,role)
            world,J,_=self._material(n,u,v,q)
            return world,J
        raise ValueError("Active print requires untouched original namedcorner or finite material edge")

    def corner(self,n,role,q,*,with_jacobian=False):
        if n not in self.slot or role not in ROLES:raise ValueError("Selected original corner role required")
        side,level=role.split("_")
        world,J,_=self._material(n,float(side=="right"),float(level=="top"),q)
        return (world[0],J[0])if with_jacobian else world[0]

    def _scene(self,q,rebuild_grid):
        q=self.validate_parameters(q);prefix=q[:self.parent_parameter_count]
        out=self.parent.scene(prefix)if rebuild_grid else self.parent.source_scene(prefix)
        ads={int(a["user_number"]):a for a in out["advertisements"]}
        for n,i in self.slot.items():
            x=q[self.parent_parameter_count+4*i:self.parent_parameter_count+4*i+4];ad=ads[n]
            ad["corners_xyz_scene_units"]={r:self.corner(n,r,q).tolist()for r in ROLES}
            ad["fitted_boundary_curves_xyz"]={r:self._material(n,*uv,q)[0].tolist()
                for (nn,r),uv in self._materials.items()if nn==n}
            ad.update(width_m=float(x[1]),height_m=float(x[3]-x[2]),
                bottom_z_scene_units=float(x[2]/self.scene_unit_m),top_z_scene_units=float(x[3]/self.scene_unit_m),
                geometry_status="continuous_independent_conditional_print_geometry_candidate_not_measured")
            ad[SCHEMA]=dict(parameters_metres=x.tolist(),
                source_extent_scope=self.initial_materials[n]["original_extent_scope"],paired_partner_not_hard_constrained=True)
        for row in out.get("advertisement_3d_corners",[]):
            n=int(row["user_number"])
            if n in self.slot:row["world_xyz_scene_units"]=self.corner(n,row["role"],q).tolist()
        out.pop(V1_SCHEMA,None);out[SCHEMA]=self.manifest(q)
        return out

    def manifest(self,q):
        value=super().manifest(q);value["schema"]=SCHEMA
        value.update(actual_parent_current_material_initialization=deepcopy(self.initialization_contract),
            no_exact_initial_tail_special_branch=True,
            prior_v1_runtime_kept_for_failure_evidence=True,
            original_pixel_source_authority_not_changed=True,
            mathematical_parameterization_changed_from_v1=True,
            old_prefix_J_bitexact_not_claimed_at_changed_course=True,
            graph_or_solver_installed=False,accuracy_accepted=False)
        return value

