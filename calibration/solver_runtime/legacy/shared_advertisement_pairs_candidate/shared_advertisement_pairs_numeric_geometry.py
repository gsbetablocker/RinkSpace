"""Source-preserving station/Z charts for specified physical ad pairs.

Only world geometry is provided here. This module adds no factor, Gaussian
prior, camera freedom, source observation, or solver/renderer installation.
Original finite polyline samples retain their original material coordinates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import numpy as np

from ad_landmarks import AdGeometry, ROLES

VERSION = 'shared_advertisement_pairs_numeric_geometry_v1'
DEFAULT_PAIRS = ((2,15),(6,19),(7,20))
CURVE_ROLES = ('left_vertical_xyz','right_vertical_xyz','top_edge_xyz','bottom_edge_xyz')
LEGACY_CURVE_ROLES = dict(left='left_vertical_xyz',right='right_vertical_xyz',top='top_edge_xyz',bottom='bottom_edge_xyz')
_ID = re.compile(r'ad([1-9]|1[0-9]|2[0-6]):(left_top|left_bottom|right_top|right_bottom|left_vertical_xyz|right_vertical_xyz|top_edge_xyz|bottom_edge_xyz|left|right|top|bottom)\Z',re.I)


class SharedAdvertisementPairsNumericGeometry:
    """Four absolute metres per pair, on the unchanged closed board course.

    Coordinate order is station centre, minor-arc width, Z centre, height.
    The x+13 partner is the exact XY half-turn with the same material roles.
    At the initial chart, validated original source worlds are returned byte
    exactly; analytic world derivatives still describe the selected branch.
    """

    def __init__(self,base,scene_unit_m=None,*,pairs=DEFAULT_PAIRS):
        if scene_unit_m is None:
            from z190_continuous_source_bundle import SCENE_UNIT_M
            scene_unit_m=SCENE_UNIT_M
        self.scene_unit_m=float(scene_unit_m)
        if not np.isfinite(self.scene_unit_m)or self.scene_unit_m<=0:
            raise ValueError('Positive fixed metric scene scale required')
        self.base=deepcopy(base)
        self.pairs=tuple((int(a),int(b))for a,b in pairs)
        if not self.pairs or any(a<1 or a>13 or b!=a+13 for a,b in self.pairs):
            raise ValueError('Each explicit physical pair must be Adx/Ad(x+13)')
        numbers=[n for pair in self.pairs for n in pair]
        if len(numbers)!=len(set(numbers)):
            raise ValueError('Physical pair identities must be disjoint')
        if base.get('coordinate_system',{}).get('origin')!='user_PPT_ice_outline_center':
            raise ValueError('Explicit rink-centre origin required')
        boundary=np.asarray(base['ice_boundary']['points_xyz'],float)
        if boundary.ndim!=2 or boundary.shape[1]!=3 or not np.isfinite(boundary).all():
            raise ValueError('Finite existing closed boundary required')
        if np.max(abs((boundary[:,:2].min(0)+boundary[:,:2].max(0))/2))>1e-7:
            raise ValueError('Existing boundary must have the declared origin')
        authority=base.get('shared_centrosymmetric_advertisement_authority',{})
        self.authority=deepcopy(authority)
        materialized={tuple(map(int,p))for p in authority.get('materialized_pairs',[])}
        if any(pair not in materialized for pair in self.pairs):
            raise ValueError('Existing explicit materialized physical-pair authority required')
        self.old={n:AdGeometry(base,n)for n in numbers}
        self.slot={n:p for p,pair in enumerate(self.pairs)for n in pair}
        self.arc={p:self.old[low]for p,(low,_)in enumerate(self.pairs)}
        values=[]
        for p,(low,high)in enumerate(self.pairs):
            if any(set(self.old[n].roles)!=set(ROLES)for n in(low,high)):
                raise ValueError('All four original physical print corners required')
            arc=self.arc[p]
            for role in ROLES:
                one=np.asarray(self._base_corner(low,role))
                other=np.asarray(self._base_corner(high,role))
                if np.max(abs(other-one*np.array([-1.,-1.,1.])))>1e-7:
                    raise ValueError('Original named print-role pair violates half-turn')
            values.extend(((arc._base_s['left']+arc._width_sign*arc._nominal_width/2)*self.scene_unit_m,
                arc._nominal_width*self.scene_unit_m,
                (arc._base_height['bottom']+arc._base_height['top'])*self.scene_unit_m/2,
                (arc._base_height['top']-arc._base_height['bottom'])*self.scene_unit_m))
        self.initial=np.asarray(values,float)
        self.parameter_count=len(self.initial)
        self.parameter_names=tuple(f'Ad{lo}_{hi}_{field}'for lo,hi in self.pairs
            for field in ('station_center_m','arc_width_m','z_center_m','height_m'))
        self.base_sha256=hashlib.sha256(json.dumps(base,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self._scene_curve_material={}
        for n in numbers:
            ad=self._base_ad(n)
            for role,points in ad.get('fitted_boundary_curves_xyz',{}).items():
                if role not in CURVE_ROLES:continue
                self._scene_curve_material[n,role]=self._old_curve_coordinates(np.asarray(points,float),n,role)
        self.validate_parameters(self.initial)

    def _base_ad(self,number):
        ads=[a for a in self.base['advertisements']if int(a['user_number'])==int(number)]
        if len(ads)!=1:raise ValueError('One original physical advertisement required')
        return ads[0]

    def _base_corner(self,number,role):
        return np.asarray(self._base_ad(number)['corners_xyz_scene_units'][role],float)

    def _q(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():
            raise ValueError('Finite full advertisement-pair chart required')
        for p in range(len(self.pairs)):
            _,width,_,height=q[4*p:4*p+4]
            if width<=0 or width>=self.arc[p]._perimeter*self.scene_unit_m/2 or height<=0:
                raise ValueError('Positive height and positive minor-arc width required')
        return q

    def validate_parameters(self,q):
        q=self._q(q)
        for p in range(len(self.pairs)):
            _,_,z,height=q[4*p:4*p+4]
            arc=self.arc[p]
            if z-height/2<arc._ice_z*self.scene_unit_m-1e-8 or z+height/2>arc._board_top*self.scene_unit_m+1e-8:
                raise ValueError('Panel exceeds unchanged engineering board-height domain')
        return q.copy()

    def _arc_point_jacobian(self,station_m,p):
        arc=self.arc[p]
        s=np.asarray(station_m,float).reshape(-1)/self.scene_unit_m%arc._perimeter
        i=np.minimum(np.searchsorted(arc._cum,s,side='right')-1,len(arc._starts)-1)
        tangent=arc._edges[i]/arc._lengths[i,None]
        point=arc._starts[i]+(s-arc._cum[i])[:,None]*tangent
        return point,tangent/self.scene_unit_m

    def _material(self,q,u,v,number):
        q=self._q(q)
        p=self.slot[int(number)];lo,hi=self.pairs[p];arc=self.arc[p]
        x=q[4*p:4*p+4]
        u,v=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float))
        u,v=u.reshape(-1),v.reshape(-1)
        station=x[0]+arc._width_sign*(u-.5)*x[1]
        xy,tangent=self._arc_point_jacobian(station,p)
        world=np.column_stack((xy,(x[2]+(v-.5)*x[3])/self.scene_unit_m))
        jac=np.zeros((len(world),3,self.parameter_count))
        jac[:,:2,4*p]=tangent
        jac[:,:2,4*p+1]=tangent*(arc._width_sign*(u-.5))[:,None]
        jac[:,2,4*p+2]=1/self.scene_unit_m
        jac[:,2,4*p+3]=(v-.5)/self.scene_unit_m
        if int(number)==hi:
            world[:,:2]*=-1;jac[:,:2]*=-1
        elif int(number)!=lo:raise ValueError('Physical number outside selected pair')
        return world,jac

    def corner(self,number,role,q=None,*,with_jacobian=False):
        if role not in ROLES or int(number)not in self.slot:
            raise ValueError('Selected physical pair and original print corner required')
        q=self.initial if q is None else self._q(q)
        side,level=role.split('_')
        world,jac=self._material(q,float(side=='right'),float(level=='top'),int(number))
        p=self.slot[int(number)]
        if np.array_equal(q[4*p:4*p+4],self.initial[4*p:4*p+4]):world[0]=self._base_corner(number,role)
        return (world[0],jac[0])if with_jacobian else world[0]

    def identities(self,row):
        roles=row.get('physical_world_corner_roles')
        if roles is not None:
            number=row.get('physical_ad_id',row.get('ad_id'))
            if number is None or any(r not in ROLES for r in roles):
                raise ValueError('Explicit physical print number and original corner roles required')
            return [(int(number),r)for r in roles]
        ids=row.get('physical_ids')
        if ids:
            parsed=[_ID.fullmatch(str(v))for v in ids]
            if all(m is not None for m in parsed):return [(int(m[1]),LEGACY_CURVE_ROLES.get(m[2].lower(),m[2].lower()))for m in parsed]
            if any(m is not None and int(m[1])in self.slot for m in parsed):
                raise ValueError('Mixed unsupported source identities for selected panel')
            return None
        parsed=_ID.fullmatch(str(row.get('id','')))
        if parsed is not None:return [(int(parsed[1]),LEGACY_CURVE_ROLES.get(parsed[2].lower(),parsed[2].lower()))]
        number=row.get('physical_ad_id',row.get('ad_id'))
        if number is not None and row.get('role')in(*ROLES,*CURVE_ROLES,*LEGACY_CURVE_ROLES):
            return [(int(number),LEGACY_CURVE_ROLES.get(row['role'],row['role']))]
        if number is not None and int(number)in self.slot:
            raise ValueError('Selected physical panel lacks original print role')
        return None

    def correct_certified_legacy_held_ad2_chord(self,row,actual_pts,aliases):
        """Change only the exact four original packet-qualified HELD chords.

        The old generator interpolated print endpoints with 121 t samples.
        Preserve that material t on the actual course, after certifying the
        full source-only authority. General off-course curves still fail.
        """
        from pathlib import Path
        source_sha='08e8539463d5cbdffc436cdf8f6129f1d25675f3ad0e18ca473c177ff20c4703'
        original_pts=229535232;target_actual=229534976
        roles={'left':('left_top','left_bottom'),'right':('right_top','right_bottom'),
               'top':('left_top','right_top'),'bottom':('left_bottom','right_bottom')}
        match=re.fullmatch(r'ad2:(left|right|top|bottom)',str(row.get('id','')))
        authority=row.get('review_source_packet',{})
        if (match is None or int(actual_pts)!=target_actual or int(row.get('pts',-1))!=original_pts or
            aliases.get(original_pts)!=target_actual or row.get('split')!='held' or row.get('source_role')!='held' or
            row.get('review_only_never_admitted') is not True or row.get('point_correspondence') is not False or
            row.get('diagnostic_unit_sigma_not_fit_weight') is not True or authority.get('sha256')!=source_sha):
            raise ValueError('Only the four original packet-qualified Ad2 HELD chord representations may change')
        path=Path(authority['path']);payload=path.read_bytes()
        if path.name!='ad2_ad3_source_whole_visible_boundaries_qualified.json' or hashlib.sha256(payload).hexdigest()!=source_sha:
            raise ValueError('Original source-only whole Ad2/Ad3 boundary packet hash changed')
        packet=json.loads(payload);records=[r for r in packet['records']if r['physical_id']==row['id']]
        if packet.get('source_pts')!=original_pts or packet.get('source_read_uses_camera') is not False or len(records)!=1:
            raise ValueError('Original source-only HELD packet identity changed')
        record=records[0];points=record.get('source_qualified_rows',record['rows'])
        if record['source_role']!='held' or not points:raise ValueError('Original HELD qualification changed')
        checks={'uv':np.array([p['source_uv']for p in points],float),
            'normal':np.array([p.get('source_normal_uv',[1.,0.])for p in points],float),
            'normal_interval':np.array([p['normal_interval_px']for p in points],float),
            'sigma':np.ones(len(points))}
        if any(not np.array_equal(row[key],expected)for key,expected in checks.items()):
            raise ValueError('Original HELD source UV/normal interval/unit sigma changed')
        ends=np.asarray(record['world_endpoints'],float);world=np.asarray(row['world'],float)
        t=np.linspace(0.,1.,121)
        chord=(1-t[:,None])*ends[0]+t[:,None]*ends[1]
        if ends.shape!=(2,3) or not np.array_equal(world,chord):
            raise ValueError('Only the exact original endpoints and linear121 generator are certified')
        edge=match[1];end_roles=roles[edge]
        if not np.array_equal(ends,np.asarray([self._base_corner(2,r)for r in end_roles])):
            raise ValueError('Qualified endpoints differ from their named initial physical Ad2 print corners')
        u=t if edge in('top','bottom') else np.full(121,float(edge=='right'))
        v=np.full(121,float(edge=='top')) if edge in('top','bottom') else 1.-t
        if edge in('left','right'):current=world.copy()
        else:
            current,_=self._material(self.initial,u,v,2)
            current[0],current[-1]=ends[0],ends[-1]
        self._old_curve_coordinates(current,2,LEGACY_CURVE_ROLES[edge])
        generator=Path(__file__).resolve().parents[1]/'snapshot/three_frame_refit/program_tracking/z190_secondary_qualified_core_source.py'
        ledger=dict(schema='rinkspace.certified-original-held-ad2-chord-to-physical-course-representation.v1',
            id=row['id'],original_source_pts=original_pts,actual_pts=target_actual,split='held',
            source_role='held',review_only_never_admitted=True,source_packet=dict(path=str(path),sha256=source_sha),
            original_world_generator=dict(path=str(generator),sha256=hashlib.sha256(generator.read_bytes()).hexdigest(),
                source_lines=[219,238],formula='t=linspace(0,1,121); world=(1-t)*endpoints[0]+t*endpoints[1]'),
            original_world_endpoints=ends.tolist(),original_world_chord=world.tolist(),
            original_material_t=t.tolist(),physical_material_u=u.tolist(),physical_material_v=v.tolist(),
            physical_world_curve=current.tolist(),world_sample_count=121,original_source_point_count=len(points),
            world_changed=not np.array_equal(world,current),maximum_world_change_metres=float(np.max(np.linalg.norm(current-world,axis=1)))*self.scene_unit_m,
            same_original_source_pixels_intervals_sigma_and_roles=True,source_point_correspondence=False,
            original_chord_is_not_a_measured_world_curve=True,HELD_not_admitted_to_fit=True,
            ordinary_course_domain_threshold_unchanged=1e-7,general_curve_snapping=False,
            zero_HELD_world_bitexact_claim=False)
        out=dict(row);out['world']=current if isinstance(row['world'],np.ndarray)else current.tolist()
        out['qualified_legacy_held_world_representation']=ledger
        return out,ledger

    def _old_curve_coordinates(self,world,number,role):
        p=self.slot[int(number)];lo,hi=self.pairs[p];arc=self.arc[p]
        if world.ndim!=2 or world.shape[1]!=3 or not len(world)or not np.isfinite(world).all():
            raise ValueError('Finite original curve N by 3 required')
        original=world*np.array([-1.,-1.,1.])if int(number)==hi else world
        stations=np.array([arc._nearest_arc(w[:2])for w in original])
        snapped=np.array([arc._point_at_arc(s)for s in stations])
        if np.max(np.linalg.norm(snapped-original[:,:2],axis=1))>1e-7:
            raise ValueError('Original finite source curve is off its actual board course')
        delta=(stations-arc._base_s['left']+arc._perimeter/2)%arc._perimeter-arc._perimeter/2
        u=delta/(arc._width_sign*arc._nominal_width)
        v=(original[:,2]-arc._base_height['bottom'])/(arc._base_height['top']-arc._base_height['bottom'])
        if np.any((u< -1e-7)|(u>1+1e-7)|(v< -1e-7)|(v>1+1e-7)):
            raise ValueError('Source curve exceeds its original finite material domain')
        if role.startswith(('left_','right_')):
            if np.max(abs(u-float(role.startswith('right_'))))>1e-7:
                raise ValueError('Original source is not its named vertical print edge')
        elif role.startswith(('top_','bottom_')):
            if np.max(abs(v-float(role.startswith('top_'))))>1e-7:
                raise ValueError('Original source is not its named horizontal print edge')
            if len(u)>1 and not(np.all(np.diff(u)>=-1e-8)or np.all(np.diff(u)<=1e-8)):
                raise ValueError('Original finite edge material support must be monotonic')
        else:raise ValueError('Original physical print curve role required')
        return u,v

    def _world_and_jacobian(self,row,q):
        q=self._q(q)
        ids=self.identities(row)
        original=np.asarray(row['world'],float)
        if original.ndim!=2 or original.shape[1]!=3 or not len(original)or not np.isfinite(original).all():
            raise ValueError('Finite original source world N by 3 required')
        if ids is None or all(n not in self.slot for n,_ in ids):
            return original.copy(),np.zeros((len(original),3,self.parameter_count))
        if all(role in ROLES for _,role in ids):
            if len(ids)!=len(original):raise ValueError('Corner identities/world count differ')
            world=original.copy();jac=np.zeros((len(original),3,self.parameter_count))
            for i,(n,role)in enumerate(ids):
                if n not in self.slot:continue
                if np.max(abs(original[i]-self._base_corner(n,role)))>1e-7:
                    raise ValueError('Original world differs from its named physical print corner')
                world[i],jac[i]=self.corner(n,role,q,with_jacobian=True)
        elif len(ids)==1 and ids[0][0]in self.slot and ids[0][1]in CURVE_ROLES:
            n,role=ids[0]
            u,v=self._old_curve_coordinates(original,n,role)
            world,jac=self._material(q,u,v,n)
            p=self.slot[n]
            if np.array_equal(q[4*p:4*p+4],self.initial[4*p:4*p+4]):world=original.copy()
        else:raise ValueError('Finite print-edge source requires one selected physical edge')
        if np.array_equal(q,self.initial):world=original.copy()
        if world.shape!=original.shape:raise AssertionError('Finite source sampling was changed')
        return world,jac

    def rebind(self,row,q):
        ids=self.identities(row)
        if ids is None or all(n not in self.slot for n,_ in ids):return row
        world,_=self._world_and_jacobian(row,q)
        out=dict(row);out['world']=world if isinstance(row['world'],np.ndarray)else world.tolist()
        for key in ('world_xyz','world_xyz_scene_units'):
            if key not in row:continue
            value=np.asarray(row[key])
            if value.shape==(3,)and len(world)==1:out[key]=world[0].tolist()
            elif value.shape==np.asarray(row['world']).shape:out[key]=world.tolist()
            else:raise ValueError('Source companion world shape differs')
        return out

    def world_jacobian(self,row,q):
        return self._world_and_jacobian(row,q)[1]

    def boundary(self,number,role,q=None):
        if (int(number),role)not in self._scene_curve_material:
            raise ValueError('No original finite modeled print edge; do not manufacture one')
        q=self.initial if q is None else self._q(q)
        p=self.slot[int(number)]
        if np.array_equal(q[4*p:4*p+4],self.initial[4*p:4*p+4]):
            return np.asarray(self._base_ad(number)['fitted_boundary_curves_xyz'][role],float).copy()
        u,v=self._scene_curve_material[int(number),role]
        return self._material(q,u,v,int(number))[0]

    def render_paths(self,q=None):
        """Full original modeled finite edges, using this chart's sampling.

        Callers must not replace these with resampled AdGeometry paths when
        they want the exact finite-polyline representation used by sources.
        """
        q=self.initial if q is None else self._q(q)
        return [dict(number=n,role=role,extent=self._base_ad(n).get('extent_scope','unspecified'),
            xyz=self.boundary(n,role,q))for n,role in self._scene_curve_material]

    def manifest(self,q=None):
        q=self.initial if q is None else self._q(q)
        return dict(version=VERSION,physical_pairs=[list(p)for p in self.pairs],
            parameter_names=list(self.parameter_names),parameters_metres=q.tolist(),initial_metres=self.initial.tolist(),
            fixed_scene_unit_m=self.scene_unit_m,base_scene_sha256=self.base_sha256,
            numeric_world_authority='Existing original print geometry and minor boundary arc; historical fit estimates, not measurements',
            pair_authority=deepcopy(self.authority),physical_pair_rule='Identical print roles/Z/size; XY rink-centre half-turn',
            source_uv_sigma_roles_actual_aliases_and_row_order_unchanged=True,
            original_finite_polyline_samples_and_material_subdomain_preserved=True,
            new_boundary_vertices_added=False,unmodeled_edges_created=False,
            camera_or_pixel_inferred_source_identity=False,panel_numeric_priors=None,
            physical_guard='Positive minor-arc width/height and existing engineering board Z domain; no measured print prior',
            selected_world_tangent_at_exact_boundary_vertex='Right-side existing boundary segment; derivative is branch-specific, not unique at a corner',
            other_pair_geometry_unchanged=True,compatible_disjoint_Ad1_14_chart_order='Either; shared source rows compose by explicit physical identities',
            thickness_parameter_added=False,graph_or_solver_installed=False,accuracy_accepted=False)

    def scene(self,q,*,scene=None):
        q=self.validate_parameters(q)
        out=deepcopy(self.base if scene is None else scene)
        ads={int(a['user_number']):a for a in out['advertisements']}
        for n in self.slot:
            ad=ads[n];p=self.slot[n]
            ad['corners_xyz_scene_units']={role:self.corner(n,role,q).tolist()for role in ROLES}
            ad['fitted_boundary_curves_xyz']={role:self.boundary(n,role,q).tolist()for nn,role in self._scene_curve_material if nn==n}
            if not np.array_equal(q,self.initial):
                ad['bottom_z_scene_units']=(q[4*p+2]-q[4*p+3]/2)/self.scene_unit_m
                ad['top_z_scene_units']=(q[4*p+2]+q[4*p+3]/2)/self.scene_unit_m
                ad['width_m']=float(q[4*p+1]);ad['height_m']=float(q[4*p+3])
                ad['along_fractions_status']='Historical only; current station and finite material samples define geometry'
                ad['geometry_status']='shared_paired_print_geometry_candidate_unaccepted'
            ad[VERSION]=self.manifest(q)
        for row in out.get('advertisement_3d_corners',[]):
            n,role=int(row['user_number']),row['role']
            if n in self.slot and role in ROLES:row['world_xyz_scene_units']=self.corner(n,role,q).tolist()
        out[VERSION]=self.manifest(q)
        return out
