"""Independent static venue material rows on the current shared scene.

No optimizer, programme source, measurement admission or camera calibration is
created here. An authenticated packet and explicit admitted IDs are required.
The public finite-curve loss/domain and streaming SourceRowBlock are reused.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from calibration.solver_runtime.source_loss import SourceLossWeights
from .first_shot_physical_state import GeometryDomainError
from .z190_source_camera_diagnostic import absolute_value_jac15
from wide_anchor_candidate.native_wide_source import world_curve
from wide_anchor_candidate.full_brown_projection import curve_vector_error_full_brown
from z190_streaming_source_normal import SourceRowBlock


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json_sha(value):
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


def current_material_curve(scene,target,*,samples=256):
    """Literal current entity; yawed Lane carriers meet the finite course."""
    if target['kind']!='polygon_supporting_line':
        return world_curve(scene,target,samples=samples)
    feature=next(f for f in scene['ice_features']if f['id']==target['feature_id'])
    if feature['kind']!='polygon':raise ValueError('A physical polygon edge is required')
    polygon=np.asarray(feature['polygon_xyz'],float);index=int(target['index'])
    if index in feature.get('open_edges',[])or not 0<=index<len(polygon):
        raise ValueError('Original carrier must name a known material edge')
    a,b=polygon[[index,(index+1)%len(polygon)]];direction=b[:2]-a[:2]
    if not np.linalg.norm(direction)>0:raise ValueError('Nondegenerate literal carrier required')
    border=np.asarray(scene['ice_boundary']['points_xyz'],float)[:,:2]
    if np.array_equal(border[0],border[-1]):border=border[:-1]
    cross=lambda x,y:x[0]*y[1]-x[1]*y[0]
    hits=[]
    for c,edge in zip(border,np.roll(border,-1,axis=0)-border):
        denominator=cross(direction,edge)
        if abs(denominator)<1e-13*np.linalg.norm(direction)*np.linalg.norm(edge):continue
        t=cross(c-a[:2],edge)/denominator;u=cross(c-a[:2],direction)/denominator
        if -1e-10<=u<=1+1e-10:hits.append(t)
    if len(hits)<2 or max(hits)<=min(hits):
        raise GeometryDomainError('Literal material carrier does not cross current finite course')
    return np.linspace(a+min(hits)*(b-a),a+max(hits)*(b-a),samples)


class AdmittedStaticPacket:
    """One original representation, exact source bytes, explicit material IDs.

    This does not infer admission from residuals or historical `use=fit` flags.
    A/goal curves and advertisements cannot enter this control-only adapter.
    The historical alternative representation and outside entities are unused.
    """
    def __init__(self,path,*,expected_sha256,admitted_control_ids):
        path=Path(path);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Static source packet bytes changed')
        packet=json.loads(raw.decode('utf-8-sig'));incoming=packet['original_v5_observations']
        all_ids=[row['id']for row in incoming];requested=tuple(admitted_control_ids)
        if len(set(all_ids))!=len(all_ids)or len(set(requested))!=len(requested)or not requested:
            raise ValueError('Unique explicit source IDs required')
        if not set(requested)<=set(all_ids):raise ValueError('Admitted ID missing from original representation')
        rows=[deepcopy(row)for row in incoming if row['id']in set(requested)]
        allowed={'board_side','circle_outline','line_segment','polygon_supporting_line'}
        for row in rows:
            target=row['target'];feature=target.get('feature_id','')
            if target['kind']not in allowed or feature in ('A1','A2','A3','A4')or feature.startswith('goal_'):
                raise ValueError('Only explicitly admitted non-A ice/board material controls are supported')
            if target['kind']=='board_side':
                if target['edge']not in ('top','contact'):raise ValueError('Literal board material edge required')
            elif not feature:raise ValueError('Explicit physical ice entity required')
            uv=np.asarray(row['pixels_uv'],float);sigma=np.asarray(row['sigma_px'],float)
            if uv.ndim!=2 or uv.shape[1]!=2 or not len(uv)or not np.isfinite(uv).all():raise ValueError('Finite original native UV required')
            if not np.isfinite(sigma).all()or np.any(sigma<=0):raise ValueError('Positive original sigma required')
            weight=float(row.get('weight',1.))
            if not np.isfinite(weight)or weight<=0:raise ValueError('Positive literal source weight required')
            if row.get('synthetic')is not False:raise ValueError('Original real material source required')
        self._rows_json=tuple(json.dumps(row,ensure_ascii=False,allow_nan=False)for row in rows)
        self.reference=dict(path=str(path.resolve()),sha256=expected_sha256)
        self.contract=deepcopy(packet['contract']);self.frame=deepcopy(packet['frame'])
        self.holdout_ids=tuple(identity for identity in all_ids if identity not in requested)
        self.advertisement_holdout_count=len(packet['original_user_advertisement_corner_observations'])
        scales=np.concatenate([np.repeat(np.broadcast_to(np.asarray(r['sigma_px'],float),(len(r['pixels_uv']),)),2)for r in rows])
        self.scales=np.frombuffer(scales.tobytes(),dtype=np.float64)
        self.quadratic=np.frombuffer(np.zeros(len(self.scales),bool).tobytes(),dtype=bool)
        weights=np.concatenate([np.full(2*len(r['pixels_uv']),float(r.get('weight',1.)))for r in rows])
        self.loss_weights=SourceLossWeights(weights)
        self.ledger=dict(family='independent-static-venue-material-control',source_packet=self.reference,
            source_frame=self.frame,source_contract=self.contract,original_representation='original_v5_observations',
            ordered_rows=[dict(id=r['id'],target=r['target'],points=len(r['pixels_uv']),UV_float64_sha256=_sha(np.asarray(r['pixels_uv'],float)),
                sigma_px=r['sigma_px'],literal_Huber2_outer_weight=r.get('weight',1.))for r in rows],
            held_curve_ids=self.holdout_ids,advertisement_holdout_count=self.advertisement_holdout_count,
            outside_entity_or_historical_resampling_used=False,source_clock_rebound=False)
        self.identity_sha256=_json_sha(self.ledger)
    @property
    def rows(self):
        return tuple(json.loads(row)for row in self._rows_json)


class StaticCameraPolicy:
    """Explicit conditional image family, never measured hardware facts."""
    def __init__(self,kind,template_camera25):
        template=np.asarray(template_camera25,float)
        if template.shape!=(25,)or not np.isfinite(template).all():raise ValueError('Finite camera25 required')
        if kind not in ('brown13','fixed_optics7'):raise ValueError('Only conservative camera13 or fixed-optics7 supported')
        if kind=='brown13'and(abs(template[6]-template[7])>1e-13 or template[10]!=0 or np.any(template[15:]!=0)):
            raise ValueError('camera13 seed must explicitly have common focal, zero skew and only Brown4')
        self.kind=kind;self.template=template.copy();self.count=13 if kind=='brown13'else 7
        self.identity=dict(kind=kind,free_count=self.count,template_camera25=template.tolist(),
            template_sha256=_sha(template),optical_conditions_not_measured=True,
            camera15_extreme_skew_diagnostic_not_adopted=True)
    def encode(self,camera25=None):
        c=self.template if camera25 is None else np.asarray(camera25,float)
        if self.kind=='brown13':return np.r_[c[:3]/100,c[3:6],c[6],c[8:10]/1000,c[11:15]]
        return np.r_[c[:3]/100,c[3:6],c[6]]
    def decode(self,x):
        x=np.asarray(x,float)
        if x.shape!=(self.count,)or not np.isfinite(x).all():raise ValueError('Static camera coordinate count changed')
        if not np.isfinite(np.exp(x[6]))or np.exp(x[6])<=0:
            raise ValueError('Static focal leaves finite positive floating domain')
        c=self.template.copy();c[:3]=100*x[:3];c[3:6]=x[3:6];c[6]=x[6]
        if self.kind=='brown13':
            c[7]=x[6];c[8:10]=1000*x[7:9];c[10]=0.;c[11:15]=x[9:13];c[15:]=0.
        else:c[7]=x[6]+self.template[7]-self.template[6]
        return c


class StaticVenueFactor:
    """Append static-camera coordinates and continuously re-evaluate geometry.

    scene_at/geometry_guard use only the programme prefix. geometry_columns
    are the actual original world-column inventory; no small-J selection.
    geometry_fd_step must be the existing producer's declared working step.
    """
    def __init__(self,packet,camera_policy,*,programme_count,geometry_columns,
                 scene_at,geometry_guard,geometry_fd_step,samples=256):
        self.packet=packet;self.camera_policy=camera_policy;self.programme_count=int(programme_count)
        self.camera_columns=np.arange(self.programme_count,self.programme_count+camera_policy.count)
        self.parameter_count=self.programme_count+camera_policy.count
        self.geometry_columns=np.asarray(geometry_columns,int).copy()
        if len(set(self.geometry_columns))!=len(self.geometry_columns)or np.any(self.geometry_columns<0)or np.any(self.geometry_columns>=self.programme_count):
            raise ValueError('Exact distinct programme world-column inventory required')
        self.scene_at=scene_at;self.geometry_guard=geometry_guard;self.fd_step=float(geometry_fd_step);self.samples=int(samples)
        if not self.fd_step>0 or not np.isfinite(self.fd_step):raise ValueError('Positive original geometry FD step required')
        self.scales=packet.scales;self.quadratic=packet.quadratic;self.last_geometry_stencils=[]
        self.identity=dict(schema='rinkspace.static-venue-shared-world-factor.v1',packet_sha256=packet.identity_sha256,
            camera=camera_policy.identity,programme_count=self.programme_count,parameter_count=self.parameter_count,
            geometry_columns=self.geometry_columns.tolist(),camera_columns=self.camera_columns.tolist(),geometry_fd_step=self.fd_step,
            samples=self.samples,source_sigma_and_Huber2_outer_weights_unchanged=True,
            no_camera15_or_new_lens_dof=True,source_frame=packet.frame)
    def initial_state(self,programme_q):
        q=np.asarray(programme_q,float)
        if q.shape!=(self.programme_count,):raise ValueError('Exact programme prefix required')
        return np.r_[q,self.camera_policy.encode()]
    def split(self,q):
        q=np.asarray(q,float)
        if q.shape!=(self.parameter_count,)or not np.isfinite(q).all():raise ValueError('Complete finite programme+static-camera state required')
        return q[:self.programme_count],self.camera_policy.decode(q[self.camera_columns])
    def _evaluate_scene(self,scene,camera,jacobian):
        rows=[]
        for original in self.packet.rows:
            row=dict(world=current_material_curve(scene,original['target'],samples=self.samples),
                uv=np.asarray(original['pixels_uv'],float),sigma=np.broadcast_to(np.asarray(original['sigma_px'],float),(len(original['pixels_uv']),)))
            rows.append(row)
        if self.camera_policy.kind=='brown13':
            result=absolute_value_jac15(camera,rows,jacobian)
            if not jacobian:return result
            r,j=result;return r,np.c_[j[:,:6],j[:,6]+j[:,7],j[:,8:10],j[:,11:15]]
        values=[];blocks=[]
        for row in rows:
            scale=np.repeat(row['sigma'],2)
            result=curve_vector_error_full_brown(camera,row['world'],row['uv'],jacobian)
            if jacobian:
                r,j=result;values.append(r/scale);blocks.append(np.c_[j[:,:6],j[:,6]+j[:,7]]/scale[:,None])
            else:values.append(result/scale)
        return(np.concatenate(values),np.vstack(blocks))if jacobian else np.concatenate(values)
    def block(self,q,jacobian):
        programme,camera=self.split(q);self.geometry_guard(programme)
        scene=self.scene_at(programme);base=self._evaluate_scene(scene,camera,bool(jacobian))
        if not jacobian:return SourceRowBlock(base,None,self.quadratic,self.scales,self.packet.ledger,self.packet.loss_weights)
        residual,camera_jac=base;data=[];indices=[];self.last_geometry_stencils=[]
        for column in self.geometry_columns:
            candidates=[];failures=[]
            for sign in(-1.,1.):
                sample=programme.copy();sample[column]+=sign*self.fd_step
                try:self.geometry_guard(sample)
                except GeometryDomainError as error:candidates.append(None);failures.append(str(error))
                else:
                    candidates.append(self._evaluate_scene(self.scene_at(sample),camera,False));failures.append(None)
            minus,plus=candidates
            if minus is None and plus is None:raise GeometryDomainError('Both original static-venue world FD sides invalid: '+str(column))
            if minus is not None and plus is not None:derivative=(plus-minus)/(2*self.fd_step);method='central_original_step'
            elif plus is not None:derivative=(plus-residual)/self.fd_step;method='forward_original_step'
            else:derivative=(residual-minus)/self.fd_step;method='backward_original_step'
            if derivative.shape!=residual.shape:raise ValueError('Static venue geometry FD changed source rows')
            data.append(derivative);indices.append(int(column))
            self.last_geometry_stencils.append(dict(column=int(column),method=method,step=self.fd_step,minus_geometry_domain_failure=failures[0],plus_geometry_domain_failure=failures[1]))
        matrix=np.column_stack([*data,*camera_jac.T])
        columns=np.r_[indices,self.camera_columns];row_ids=np.repeat(np.arange(len(residual)),len(columns));col_ids=np.tile(columns,len(residual))
        jac=csr_matrix((matrix.ravel(),(row_ids,col_ids)),shape=(len(residual),self.parameter_count));jac.eliminate_zeros()
        return SourceRowBlock(residual,jac,self.quadratic,self.scales,self.packet.ledger,self.packet.loss_weights)
    def camera_guard(self,q):
        """Same trial source/domain, no extra calibration or parameter bounds."""
        block=self.block(q,False)
        if not np.isfinite(block.r).all():raise ValueError('Static source leaves finite public projection domain')
        return True
    def append_source_blocks(self,programme_factory,q,jacobian):
        """Replay all original blocks, zero-pad J, then append the static rows."""
        programme,_=self.split(q)
        for block in programme_factory(programme,jacobian):
            jac=None if not jacobian else hstack((csr_matrix(block.J),csr_matrix((len(block.r),self.camera_policy.count))),format='csr')
            if jacobian and jac.shape[1]!=self.parameter_count:raise ValueError('Programme source Jacobian column inventory changed')
            yield SourceRowBlock(block.r,jac,block.quadratic,block.scales,block.row_ledger,block.loss_weights)
        yield self.block(q,jacobian)
