"""Six absolute conditional-metric parameters of the existing shared course.

This is a world primitive, not a solver/source dispatcher. Original finite
knots, their order and their material identity are retained. An uncertified
off-course chord is never inferred to be a new curved material observation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

VERSION = 'shared_board_course_numeric_geometry_v1'
PARAMETER_NAMES = ('half_length_m', 'half_width_m', 'corner_rx_m',
                   'corner_ry_m', 'corner_exponent', 'board_height_m')
PARTS = (('upper_right', False, 'corner'),
         ('upper_long_side', True, 'straight'),
         ('upper_left', False, 'corner'), ('left_end', True, 'straight'),
         ('lower_left', False, 'corner'),
         ('lower_long_side', False, 'straight'),
         ('lower_right', False, 'corner'), ('right_end', False, 'straight'))
CORNER_START = dict(upper_right=0., upper_left=90., lower_left=180., lower_right=270.)


@dataclass
class BoundCourseRow:
    """Explicit authenticated old world-material coordinates, never image UV."""
    geometry: object
    old_world: np.ndarray
    kind: str
    coordinates: dict
    z_policy: str
    z_values: np.ndarray | None
    authority: str

    def evaluate(self, q=None, *, with_jacobian=False):
        g = self.geometry
        q = g.initial if q is None else g.validate_parameters(q)
        if self.kind == 'primitive':
            world, jac = g.primitive(self.coordinates['side'], self.coordinates['fractions'], q,
                                    z_policy=self.z_policy, z_values=self.z_values, with_jacobian=True)
        else:
            bottom, _, jbottom, _, _ = g.sample_perimeter(q, with_jacobian=True)
            indices = self.coordinates['indices']
            if self.kind == 'vertices':
                world, jac = bottom[indices].copy(), jbottom[indices].copy()
            elif self.kind == 'finite_segments':
                nxt = (indices + 1) % len(bottom)
                t = self.coordinates['fractions'][:, None]
                world = bottom[indices] * (1-t) + bottom[nxt] * t
                jac = jbottom[indices] * (1-t[..., None]) + jbottom[nxt] * t[..., None]
            else:
                raise ValueError('Unrecognized authenticated course binding')
            g._z(world, jac, q, self.z_policy, self.z_values)
        if np.array_equal(q, g.initial):
            world = self.old_world.copy()
        return (world, jac) if with_jacobian else world


class SharedBoardCourseNumericGeometry:
    """Reflection-symmetric superellipse corners tangent to original straights.

    New coordinates are absolute metres except exponent. The fixed conversion
    factor is conditional, not an independently measured physical scale.
    Original SharedCornerGeometry axis snapping and finite sampling are used.
    """
    parameter_count = 6
    parameter_names = PARAMETER_NAMES

    def __init__(self, base, scene_unit_m):
        self.base = deepcopy(base)
        self.scene_unit_m = float(scene_unit_m)
        if not np.isfinite(self.scene_unit_m) or self.scene_unit_m <= 0:
            raise ValueError('Positive fixed conditional scene scale required')
        par = base['parameters']
        x0, y0, x1, y1 = map(float, par['bounds'])
        if x0 != -x1 or y0 != -y1:
            raise ValueError('Original centre-origin symmetric bounds required')
        curve = par['corner_curve']
        if curve['kind'] != 'four_congruent_symmetric_superellipse_quarters':
            raise ValueError('Original authorized superellipse primitive required')
        rx, ry = map(float, curve['tangent_extent_xy_scene_units'])
        self.initial = np.array([x1*self.scene_unit_m, y1*self.scene_unit_m,
                                 rx*self.scene_unit_m, ry*self.scene_unit_m,
                                 float(curve['exponent']), float(par['board_height'])*self.scene_unit_m])
        self._initial_scene = np.array([x1, y1, rx, ry, float(curve['exponent']), float(par['board_height'])])
        self._old_bottom = np.asarray(base['ice_boundary']['points_xyz'], float).copy()
        self._old_top = np.asarray(base['board_surface']['top_edge_xyz'], float).copy()
        self.parts = deepcopy(base['ice_boundary']['parts_counterclockwise'])
        if self._old_bottom.shape != (480, 3) or self._old_top.shape != (480, 3):
            raise ValueError('Existing 480 paired finite knots required')
        self.validate_parameters(self.initial)
        b, t, _, _, ranges = self._perimeter(self.initial)
        if ranges != self.parts or not np.allclose(b, self._old_bottom, rtol=0, atol=2e-12) or not np.allclose(t, self._old_top, rtol=0, atol=2e-12):
            raise ValueError('Existing finite world does not match authorized primitive/knot order')
        if not np.array_equal(self._old_bottom[:, :2], self._old_top[:, :2]) or np.any(self._old_bottom[:, 2] != 0):
            raise ValueError('Exact paired vertical board faces required')
        if not np.array_equal(np.asarray(base['board_surface']['bottom_edge_xyz']), self._old_bottom):
            raise ValueError('Ice and board bottom must have the same original finite world')
        self.base_sha256 = hashlib.sha256(json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.contract = {'version': VERSION, 'parameter_names': list(self.parameter_names),
            'initial': self.initial.tolist(), 'fixed_conditional_scene_unit_m': self.scene_unit_m,
            'metric_scale_verified': False, 'survey_measurement': False,
            'original_numeric_solution_user_accepted': False,
            'original_numeric_authority': 'unaccepted three-frame conditional image fit',
            'structural_authority': deepcopy(curve), 'original_source': deepcopy(base.get('source')),
            'base_content_sha256': self.base_sha256, 'finite_knot_count': 480,
            'source_UV_inferred': False, 'full_source_graph_installed': False,
            'new_prior_or_source_factor': False}

    def validate_parameters(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Finite full six-dimensional absolute course chart required')
        a, b, rx, ry, p, h = q
        if not (a > rx > 0 and b > ry > 0 and p > 1 and h > 0):
            raise ValueError('Positive tangency straights/radii/height and convex p>1 required')
        return q.copy()

    def structural_margins(self, q=None):
        q = self.initial if q is None else self.validate_parameters(q)
        a,b,rx,ry,_,_ = self._scene_q(q)
        return dict(length=2*a,width=2*b,
            upper_right_radius=rx,upper_left_radius=rx,lower_left_radius=rx,lower_right_radius=rx,
            upper_straight_length=2*(a-rx),lower_straight_length=2*(a-rx),
            left_straight_length=2*(b-ry),right_straight_length=2*(b-ry))

    def _scene_q(self, q):
        if np.array_equal(q, self.initial):
            return self._initial_scene.copy()
        v = q.copy()
        v[[0,1,2,3,5]] /= self.scene_unit_m
        return v

    def _z(self, world, jac, q, policy, values):
        if policy == 'contact':
            world[:, 2] = 0.; jac[:, 2] = 0.
        elif policy == 'top':
            world[:, 2] = self._scene_q(q)[5]; jac[:, 2] = 0.; jac[:, 2, 5] = 1/self.scene_unit_m
        elif policy == 'height_fraction':
            v = np.broadcast_to(np.asarray(values, float), (len(world),))
            if not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1):
                raise ValueError('Authenticated original vertical material fraction must be in [0,1]')
            world[:, 2] = v*self._scene_q(q)[5]; jac[:, 2] = 0.; jac[:, 2, 5] = v/self.scene_unit_m
        elif policy == 'absolute_z_m':
            v = np.broadcast_to(np.asarray(values, float), (len(world),))
            if not np.isfinite(v).all() or np.any(v < 0) or np.any(v > q[5]):
                raise ValueError('Absolute material Z exceeds current positive board-height domain')
            world[:, 2] = v/self.scene_unit_m; jac[:, 2] = 0.
        else:
            raise ValueError('Explicit contact/top/height_fraction/absolute_z_m policy required')

    def primitive(self, side, fractions, q=None, *, z_policy='contact', z_values=None, with_jacobian=False):
        q = self.initial if q is None else self.validate_parameters(q)
        u = np.asarray(fractions, float).reshape(-1)
        if not np.isfinite(u).all() or np.any(u < 0) or np.any(u > 1):
            raise ValueError('Existing finite primitive fractions in [0,1] required')
        a, b, rx, ry, p, _ = self._scene_q(q)
        scale = 1/self.scene_unit_m
        jac = np.zeros((len(u), 3, 6))
        if side in CORNER_START:
            theta = np.deg2rad(CORNER_START[side] + 90*u)
            cs = np.column_stack((np.cos(theta), np.sin(theta)))
            cs[np.abs(cs) < 1e-14] = 0.
            power = np.sign(cs)*np.abs(cs)**(2/p)
            sx = 1. if 'right' in side else -1.
            sy = 1. if 'upper' in side else -1.
            center = np.array([sx*(a-rx), sy*(b-ry)])
            world = np.column_stack((center + np.array([rx, ry])*power, np.zeros(len(u))))
            jac[:,0,0] = sx*scale; jac[:,1,1] = sy*scale
            jac[:,0,2] = (power[:,0]-sx)*scale
            jac[:,1,3] = (power[:,1]-sy)*scale
            logcs = np.zeros_like(cs)
            np.log(np.abs(cs), out=logcs, where=cs != 0)
            jac[:,:2,4] = np.array([rx,ry])*power*(-2/p**2)*logcs
        else:
            if side == 'upper_long_side':
                start, end = [-a+rx,b], [a-rx,b]
                jstart = [[-1,0,1,0,0,0],[0,1,0,0,0,0]]
                jend = [[1,0,-1,0,0,0],[0,1,0,0,0,0]]
            elif side == 'lower_long_side':
                start, end = [-a+rx,-b], [a-rx,-b]
                jstart = [[-1,0,1,0,0,0],[0,-1,0,0,0,0]]
                jend = [[1,0,-1,0,0,0],[0,-1,0,0,0,0]]
            elif side in ('left_end', 'right_end'):
                sx = -1 if side == 'left_end' else 1
                start, end = [sx*a,-b+ry], [sx*a,b-ry]
                jstart = [[sx,0,0,0,0,0],[0,-1,0,1,0,0]]
                jend = [[sx,0,0,0,0,0],[0,1,0,-1,0,0]]
            else:
                raise ValueError('Original physical primitive identity required')
            start, end = np.asarray(start,float), np.asarray(end,float)
            # Original straight np.linspace exactly when material knots are uniform.
            if len(u) >= 2 and np.array_equal(u, np.linspace(0.,1.,len(u))):
                xy = np.linspace(start,end,len(u))
            else:
                xy = start*(1-u[:,None])+end*u[:,None]
            world = np.column_stack((xy, np.zeros(len(u))))
            jac[:,:2] = ((1-u[:,None,None])*np.asarray(jstart)+u[:,None,None]*np.asarray(jend))*scale
        self._z(world, jac, q, z_policy, z_values)
        return (world, jac) if with_jacobian else world

    def _perimeter(self, q):
        bs, ts, jbs, jts, ranges = [], [], [], [], []
        start = 0
        for side, reverse, kind in PARTS:
            n = 97 if kind == 'corner' else 25
            u = np.linspace(0.,1.,n)
            b, jb = self.primitive(side,u,q,with_jacobian=True)
            t, jt = self.primitive(side,u,q,z_policy='top',with_jacobian=True)
            if reverse: b,t,jb,jt = b[::-1],t[::-1],jb[::-1],jt[::-1]
            bs.append(b[:-1]); ts.append(t[:-1]); jbs.append(jb[:-1]); jts.append(jt[:-1])
            ranges.append(dict(id=side,kind=kind,first_vertex=start,vertex_count=n-1)); start += n-1
        return np.vstack(bs),np.vstack(ts),np.vstack(jbs),np.vstack(jts),ranges

    def sample_perimeter(self, q=None, *, with_jacobian=False):
        q = self.initial if q is None else self.validate_parameters(q)
        b,t,jb,jt,ranges = self._perimeter(q)
        if np.array_equal(q,self.initial): b,t = self._old_bottom.copy(),self._old_top.copy()
        return (b,t,jb,jt,ranges) if with_jacobian else (b,t,ranges)

    def _bound(self, old_world, kind, coordinates, z_policy, z_values, authority):
        old = np.asarray(old_world,float)
        if old.ndim != 2 or old.shape[1] != 3 or not np.isfinite(old).all() or not authority:
            raise ValueError('Finite original world and explicit material identity authority required')
        values = None if z_values is None else np.asarray(z_values,float).copy()
        row = BoundCourseRow(self,old.copy(),kind,coordinates,z_policy,values,str(authority))
        # Test mathematical material reconstruction, before byte-exact replay override.
        candidate_q = self.initial.copy(); candidate_q[5] = np.nextafter(candidate_q[5],np.inf)
        expected = row.evaluate(candidate_q)
        if z_policy in ('top','height_fraction'):
            expected[:,2] *= self.initial[5]/candidate_q[5]
        if not np.allclose(expected,old,rtol=0,atol=2e-11):
            raise ValueError('Original world does not satisfy its explicit finite material certificate')
        return row

    def bind_primitive(self, old_world, side, fractions, *, z_policy='contact', z_values=None, authority):
        u = np.asarray(fractions,float).reshape(-1).copy()
        if len(u) != len(old_world): raise ValueError('One original primitive material fraction per vertex required')
        return self._bound(old_world,'primitive',dict(side=side,fractions=u),z_policy,z_values,authority)

    def bind_vertices(self, old_world, vertex_indices, *, z_policy='contact', z_values=None, authority):
        i = np.asarray(vertex_indices)
        if i.ndim != 1 or not np.issubdtype(i.dtype,np.integer) or len(i) != len(old_world) or np.any(i<0) or np.any(i>=480):
            raise ValueError('Explicit original finite boundary vertex indices required')
        return self._bound(old_world,'vertices',dict(indices=i.copy()),z_policy,z_values,authority)

    def bind_finite_segments(self, old_world, segment_indices, fractions, *, z_policy='contact', z_values=None, authority):
        i = np.asarray(segment_indices)
        t = np.asarray(fractions,float).reshape(-1)
        if i.ndim != 1 or not np.issubdtype(i.dtype,np.integer) or len(i)!=len(t) or len(i)!=len(old_world) or np.any(i<0) or np.any(i>=480) or not np.isfinite(t).all() or np.any(t<0) or np.any(t>1):
            raise ValueError('Explicit original finite segment plus bounded material t required')
        return self._bound(old_world,'finite_segments',dict(indices=i.copy(),fractions=t.copy()),z_policy,z_values,authority)

    def station(self, station_m, q=None, *, z_m=0., with_jacobian=False):
        """Current finite-course station, including cumulative arc-length derivatives.

        This preserves finite polyline domain, rather than replacing it with an
        independently resampled curve. Return branch metadata at each point.
        Station and absolute Z are independent material metre coordinates.
        """
        q = self.initial if q is None else self.validate_parameters(q)
        s_m = np.asarray(station_m,float).reshape(-1)
        if not np.isfinite(s_m).all(): raise ValueError('Finite station metres required')
        b,_,jb,_,_ = self.sample_perimeter(q,with_jacobian=True)
        edge = np.roll(b,-1,axis=0)-b
        je = np.roll(jb,-1,axis=0)-jb
        lengths = np.linalg.norm(edge[:,:2],axis=1)
        jl = np.einsum('ni,nik->nk',edge[:,:2],je[:,:2])/lengths[:,None]
        cum = np.r_[0.,np.cumsum(lengths)]; jcum = np.vstack((np.zeros((1,6)),np.cumsum(jl,axis=0)))
        raw = s_m/self.scene_unit_m; wrap = np.floor(raw/cum[-1])
        s = raw % cum[-1]
        i = np.minimum(np.searchsorted(cum,s,side='right')-1,479)
        t = (s-cum[i])/lengths[i]
        ds = -wrap[:,None]*jcum[-1]
        jt = (ds-jcum[i]-t[:,None]*jl[i])/lengths[i,None]
        world = b[i]+t[:,None]*edge[i]
        jac = jb[i]+t[:,None,None]*je[i]+edge[i,:,None]*jt[:,None,:]
        self._z(world,jac,q,'absolute_z_m',z_m)
        station_jac = edge[i]/lengths[i,None]/self.scene_unit_m
        station_jac[:,2] = 0.
        report = dict(segment_index=i,segment_t=t,wrapped_turns=wrap.astype(np.int64),
            exactly_on_station_knot=(s==cum[i]),finite_polyline_domain=True,
            derivative_is_selected_station_branch=True)
        return (world,jac,station_jac,report) if with_jacobian else world

    def scene(self, q=None, *, rebuild_grid=True):
        """Boundary/mesh scene only; downstream source and ad integration pending.

        FR/Lane/markers and original source/rule metadata remain unchanged.
        At initial q every original field is byte-for-byte equivalent JSON.
        Noninitial C/B/goal endpoints and all ad material transports must be
        installed separately before declaring a complete graph/render scene.
        """
        q = self.initial if q is None else self.validate_parameters(q)
        result = deepcopy(self.base)
        if np.array_equal(q,self.initial): return result
        a,b,rx,ry,p,h = self._scene_q(q)
        par = result['parameters']
        par['bounds'] = [-a,-b,a,b]; par['corner_radius'] = rx; par['board_height'] = h
        if 'corner_radii' in par: par['corner_radii'] = {k:rx for k in par['corner_radii']}
        curve = par['corner_curve']
        curve.update(exponent=p,tangent_extent_scene_units=rx,tangent_extent_xy_scene_units=[rx,ry],aspect_y_over_x=ry/rx)
        result['boundary_validity_margins'] = self.structural_margins(q)
        checks = result.get('shared_geometry_checks',{})
        checks['corner_profile'] = dict(four_identical=True,reflection_symmetric=True,
            tangent_to_straights=True,exponent=p,aspect_y_over_x=ry/rx,circularity_required=False)
        if 'checks' in checks:checks['checks']['four_corners_keep_one_symmetric_profile']=True
        checks.get('unknown_check_reasons',{}).pop('four_corners_keep_one_symmetric_profile',None)
        checks['validation_complete']=False
        bottom,top,ranges = self.sample_perimeter(q)
        result['ice_boundary']['points_xyz'] = bottom.tolist()
        result['ice_boundary']['parts_counterclockwise'] = ranges
        surf = result['board_surface']
        surf['bottom_edge_xyz'],surf['top_edge_xyz'] = bottom.tolist(),top.tolist()
        cum = 0.
        for i,panel in enumerate(surf['panels']):
            v = bottom[(i+1)%480,:2]-bottom[i,:2]
            length = float(np.linalg.norm(v)); v /= length
            panel['s_range'] = [cum,cum+length]
            panel['tangent_xyz'] = [float(v[0]),float(v[1]),0.]
            panel['normal_into_rink_xyz'] = [float(-v[1]),float(v[0]),0.]
            panel['z_range'] = [0.,h]; cum += length
        surf['perimeter_length'] = cum
        for mesh in result['meshes']:
            if mesh['name'] == 'ice_surface': mesh['positions'] = np.vstack((bottom,[[0.,0.,0.]])).tolist()
            elif mesh['name'] == 'rink_boards': mesh['positions'] = np.vstack((bottom,top)).tolist()
        if rebuild_grid: result['ice_grid'] = _ice_grid(par,bottom)
        result[VERSION] = dict(self.contract,current_parameters=q.tolist(),
            integrated_scope='boundary paired meshes panels and grid only',
            pending_dependencies=['C/B1/B2/goal-line finite boundary endpoint clipping',
                'all advertisements actual-course station/material transport',
                'all typed absolute source world and finite board plane transport',
                'motion shared world/depth derivative and guard'],
            ice_grid_rebuilt=bool(rebuild_grid),complete_source_graph=False,
            camera_or_source_pixel_updated=False)
        return result


def _clip_halfplane(poly, axis, bound, greater):
    if not poly: return []
    def inside(point): return point[axis]>=bound-1e-9 if greater else point[axis]<=bound+1e-9
    result=[]; last=poly[-1]; oldinside=inside(last)
    for current in poly:
        newinside=inside(current)
        if oldinside!=newinside:
            d=current[axis]-last[axis]
            if abs(d)>1e-15:
                f=(bound-last[axis])/d
                result.append([last[0]+f*(current[0]-last[0]),last[1]+f*(current[1]-last[1])])
        if newinside:result.append(current)
        last,oldinside=current,newinside
    clean=[]
    for point in result:
        if not clean or math.dist(point,clean[-1])>1e-8:clean.append(point)
    if len(clean)>1 and math.dist(clean[0],clean[-1])<=1e-8:clean.pop()
    return clean


def _ice_grid(parameters,bottom):
    """Original export_final_scene finite-boundary rectangle clipping rule."""
    x0,y0,x1,y1=parameters['bounds']; spacing=float(parameters['grid_spacing'])
    if spacing<=0:raise ValueError('Original positive grid spacing required')
    columns=math.ceil((x1-x0)/spacing);rows=math.ceil((y1-y0)/spacing)
    outline=bottom[:,:2].tolist();cells=[]
    for col in range(columns):
        left,right=x0+col*spacing,min(x0+(col+1)*spacing,x1)
        for row in range(rows):
            low,high=y0+row*spacing,min(y0+(row+1)*spacing,y1);poly=outline
            for axis,bound,greater in ((0,left,True),(0,right,False),(1,low,True),(1,high,False)):
                poly=_clip_halfplane(poly,axis,bound,greater)
                if len(poly)<3:break
            if len(poly)<3:continue
            area=.5*sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1]))
            if area<=1e-8:continue
            cells.append(dict(id=f'grid_{col:03d}_{row:03d}',column=col,row=row,
                polygon_xyz=[[float(x),float(y),0.]for x,y in poly],
                area_scene_units_squared=area,clipped_at_rink_boundary=area<(right-left)*(high-low)-1e-6))
    return dict(type='rectangular_cells_clipped_to_fitted_noncircular_boundary',
        origin_xyz=[x0,y0,0.],column_axis_xyz=[1.,0.,0.],row_axis_xyz=[0.,1.,0.],
        spacing_xy=[spacing,spacing],columns=columns,rows=rows,cells=cells,
        unit_status='conditional scene units; metres per unit unverified')
