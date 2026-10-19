"""One shared, symmetric smooth corner profile for all four rink corners.

User authorized relaxing circular arcs on 2026-09-24. Exponent 2 is exactly the
old circular model. A free common exponent changes the real scene geometry,
not an image-space warp or an independent corner in each camera.
"""
import math
import copy
import numpy as np
import cv2
from rinkspace.calibration_companion.joint_scene import SharedGeometry

CORNERS=('upper_right','upper_left','lower_left','lower_right')

class SharedCornerGeometry(SharedGeometry):
    def __init__(self,base,exponent=2.,aspect=1.):
        self.__dict__=base.__dict__.copy()
        self.parameters=copy.deepcopy(base.parameters)
        self.features=copy.deepcopy(base.features)
        self.exponent=float(exponent)
        self.aspect=float(aspect)
        self.rx=float(self.corner_radii['upper_left']);self.ry=self.rx*self.aspect
        if not 1.<self.exponent<8.:raise ValueError('corner exponent must give a convex tangent profile')
        self.parameters['corner_curve']={'kind':'four_congruent_symmetric_superellipse_quarters','exponent':self.exponent,'tangent_extent_scene_units':self.rx,'tangent_extent_xy_scene_units':[self.rx,self.ry],'aspect_y_over_x':self.aspect,'circular_special_case':{'exponent':2.,'aspect':1.},'shared_across_all_frames':True,'source':'user authorized noncircular congruent symmetric corners 2026-09-24'}
        self.parameters['legacy_corner_radius_field_meaning']='x tangent extent; circular curvature only when exponent=2 and aspect=1'
        # Reclip true transverse/goal markings to the new physical boundary.
        for fid in ('C','B1','B2','goal_line_left','goal_line_right'):
            f=self.features[fid];x=float(f['segments_xyz'][0][0][0]);lo,hi=self.vertical_interval(x)
            f['segments_xyz']=[[[x,hi,0.],[x,lo,0.]]]

    def corner_points(self,side,fractions,z=0.):
        x0,y0,x1,y1=self.parameters['bounds'];r=np.array([self.rx,self.ry])
        center=np.array([x1-self.rx if 'right' in side else x0+self.rx,y1-self.ry if 'upper' in side else y0+self.ry])
        start={'upper_right':0.,'upper_left':90.,'lower_left':180.,'lower_right':270.}[side]
        t=np.deg2rad(start+90*np.asarray(fractions));cs=np.column_stack([np.cos(t),np.sin(t)])
        # Exact axis endpoints avoid fractional-power amplification of sin(pi).
        cs[np.abs(cs)<1e-14]=0.
        xy=center+r*np.sign(cs)*np.abs(cs)**(2/self.exponent)
        return np.column_stack([xy,np.full(len(xy),z)])

    def board_side(self,target,samples):
        side=target['side']
        z=self.parameters['board_height'] if target.get('edge')=='top' else 0.
        if side in CORNERS:return self.corner_points(side,np.linspace(0,1,samples),z)
        x0,y0,x1,y1=self.parameters['bounds']
        ends={'upper_long_side':[[x0+self.rx,y1],[x1-self.rx,y1]],'lower_long_side':[[x0+self.rx,y0],[x1-self.rx,y0]],'left_end':[[x0,y0+self.ry],[x0,y1-self.ry]],'right_end':[[x1,y0+self.ry],[x1,y1-self.ry]]}
        xy=np.linspace(*ends[side],samples);pts=np.column_stack([xy,np.full(samples,z)])
        corner=target.get('continue_corner')
        if corner:
            arc=self.corner_points(corner,np.linspace(0,1,samples),z)
            ds=np.array([[np.linalg.norm(a-b) for b in (arc[0],arc[-1])] for a in (pts[0],pts[-1])]);a,b=np.unravel_index(np.argmin(ds),ds.shape)
            if a==0:pts=pts[::-1]
            if b==1:arc=arc[::-1]
            pts=np.vstack([pts,arc[1:]])
        return pts

    def vertical_interval(self,x):
        x0,y0,x1,y1=self.parameters['bounds'];r=self.rx
        dx=max(x0+r-x,0.,x-(x1-r));ratio=min(1.,max(0.,dx/r))
        inset=self.ry-self.ry*max(0.,1-ratio**self.exponent)**(1/self.exponent)
        return y0+inset,y1-inset

    def board_position(self,side,fraction,z=0.):
        """Continuous perimeter position, including a panel crossing a join.

        Fractions inside [0,1] retain their original segment meaning. Outside
        it, continue along the actual adjacent boundary by physical arc length.
        """
        if 0.<=fraction<=1.:
            if side in CORNERS:
                return self.corner_points(side,[fraction],z)[0]
            endpoints=self.board_side({'side':side},2)
            result=endpoints[0]*(1.-fraction)+endpoints[-1]*fraction
            result[2]=z
            return result
        if not hasattr(self,'_perimeter_stations'):
            order=[('upper_right',False),('upper_long_side',True),('upper_left',False),('left_end',True),('lower_left',False),('lower_long_side',False),('lower_right',False),('right_end',False)]
            points=[];stations=[];meta={};offset=0.
            for name,reverse in order:
                pp=self.board_side({'side':name},1025 if name in CORNERS else 2)[:,:2]
                if reverse:pp=pp[::-1]
                ss=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pp,axis=0),axis=1))]
                meta[name]=(offset,ss,reverse)
                points.extend(pp if not points else pp[1:]);stations.extend(offset+ss if not stations else offset+ss[1:]);offset+=ss[-1]
            self._perimeter_stations=(np.array(stations),np.array(points),meta,offset)
        ss,pp,meta,total=self._perimeter_stations
        start,local,reverse=meta[side];f=1.-float(fraction) if reverse else float(fraction)
        along=np.interp(f,np.linspace(0,1,len(local)),local) if 0<=f<=1 else f*local[-1]
        at=(start+along)%total
        for next_side,(offset,distance,reversed_side) in meta.items():
            if at<=offset+distance[-1]+1e-12:
                next_fraction=float(np.interp(at-offset,distance,np.linspace(0,1,len(distance))))
                if reversed_side:next_fraction=1.-next_fraction
                return self.board_position(next_side,min(1.,max(0.,next_fraction)),z)
        raise ValueError('perimeter station outside closed boundary')

    def boundary_validity_margins(self):
        old=super().boundary_validity_margins()
        x0,y0,x1,y1=self.parameters['bounds']
        old['left_straight_length']=old['right_straight_length']=y1-y0-2*self.ry
        return old

    def boundary_clearance(self,point_xy):
        if not hasattr(self,'_outline_for_distance'):
            parts=[]
            # Counterclockwise physical outline; straight stretches arise
            # between consecutive arc endpoints.
            for c in CORNERS:parts.append(self.corner_points(c,np.linspace(0,1,160))[:,:2])
            self._outline_for_distance=np.ascontiguousarray(np.vstack(parts),dtype=np.float32)
        return float(cv2.pointPolygonTest(self._outline_for_distance,tuple(map(float,point_xy)),True))

    def hard_geometry_checks(self):
        result=super().hard_geometry_checks()
        if result is not None:
            result['corner_profile']={'four_identical':True,'reflection_symmetric':True,'tangent_to_straights':True,'exponent':self.exponent,'aspect_y_over_x':self.aspect,'circularity_required':False}
        return result
