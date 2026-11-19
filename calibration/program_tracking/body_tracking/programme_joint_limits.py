"""Authored rest coordinates and fixed-branch finite structure factors."""
import numpy as np
from scipy.sparse import csr_matrix
from .anatomical_chart import JointChart, JointEnvelope, JointAngles
from .fullbody_state import matrices
from .skate_contact import trapezoid_time_weights
from .weighted_blocks import block,full_columns,signature

JOINTS=np.array([4,5,18,19,1,2,7,8,20,21])
NAMES=['left_knee','right_knee','left_elbow','right_elbow','left_hip','right_hip','left_ankle','right_ankle','left_wrist','right_wrist']


def authored_bases(J):
    """Shaped pelvis lateral/up define anterior, independent of toe yaw."""
    lateral=J[1]-J[2];up=J[12]-J[0]
    anterior=np.cross(lateral,up);anterior/=np.linalg.norm(anterior)
    result=[]
    for joint in JOINTS:
        if joint in (4,5):distal=J[joint+3]-J[joint];toward=-anterior
        elif joint in (18,19):distal=J[joint+2]-J[joint];toward=anterior
        elif joint in (1,2):distal=J[joint+3]-J[joint];toward=anterior
        elif joint in (7,8):distal=J[joint+3]-J[joint];toward=up
        else:distal=J[joint]-J[joint-2];toward=anterior
        a=np.cross(distal,toward);a/=np.linalg.norm(a)
        b=distal-a*np.dot(a,distal);b/=np.linalg.norm(b)
        c=np.cross(a,b);result.append(np.stack((a,b,c),axis=1))
    return np.array(result)


def angles_joint(a,j):
    return JointAngles(a.twist[:,j],a.swing_vector[:,j],a.swing[:,j],a.singular[:,j],a.outside_branch[:,j])


class ProgrammeJointLimits:
    def __init__(self,problem,configuration,warm,*,anchor=None):
        self.problem=problem;self.ref=problem.reference;self.state=problem.body.state
        self.times=self.ref.times;self.base=self.ref.at(self.times);self.joints=JOINTS.copy()
        basis=authored_bases(self.ref.body.J);rest=np.broadcast_to(np.eye(3),basis.shape).copy()
        d=self.state.increments(warm,self.times)
        local=self.base['local_R'][:,self.joints-1]@matrices(np.asarray(d[:,6:]).reshape(-1,21,3)[:,self.joints-1])
        if anchor is None:self.chart=JointChart.from_reference(local,rest,basis,initial_twist=np.zeros(len(self.joints)))
        else:
            np.testing.assert_array_equal(anchor['source_pts'],self.ref.p['pts'])
            np.testing.assert_allclose(anchor['anatomical_basis'],basis,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(anchor['rest_R'],rest)
            self.chart=JointChart(rest,basis,anchor['reference_twist'])
        self.configuration=configuration;self.envelopes=[]
        for name in NAMES:
            cfg=configuration['envelopes'][name.rsplit('_',1)[-1]]
            self.envelopes.append(JointEnvelope(np.deg2rad(cfg['flex_degrees'][0]),np.deg2rad(cfg['flex_degrees'][1]),
                np.deg2rad(cfg['flex_grid_degrees']),np.deg2rad(cfg['swing_axes_degrees']),
                np.deg2rad(cfg['scale_degrees'][0]),np.deg2rad(cfg['scale_degrees'][1]),configuration['provenance']))
        self.weights=np.sqrt(trapezoid_time_weights(self.times))
        self.association=signature(self.chart.rest,self.chart.basis,self.chart.reference_twist)

    def values(self,d,rows=None,*,signed=False):
        rows=np.arange(len(self.times)) if rows is None else np.asarray(rows)
        rotation=self.base['local_R'][rows][:,self.joints-1]@matrices(np.asarray(d[:,6:]).reshape(-1,21,3)[:,self.joints-1])
        angles=self.chart.evaluate(rotation,sample_indices=rows)
        values=np.stack([(e.signed_residual if signed else e.residual)(angles_joint(angles,j))
                         for j,e in enumerate(self.envelopes)],axis=1)
        return values,angles

    def evaluate(self,x,jacobian=False):
        d=self.state.increments(x,self.times);values,angles=self.values(d)
        self.last_values=values;self.last_parameters=np.array(x,copy=True)
        residual=values*self.weights[:,None,None];J=None
        if jacobian:
            samples,joints=np.nonzero(np.any(residual>0,axis=-1));n=len(samples)
            if n:
                local=np.zeros((n,3,69));centre=np.asarray(d)[samples].copy()
                for axis in range(3):
                    columns=6+3*(self.joints[joints]-1)+axis
                    plus=centre.copy();minus=centre.copy()
                    plus[np.arange(n),columns]+=1e-6;minus[np.arange(n),columns]-=1e-6
                    # Differentiate the smooth signed boundary, then apply
                    # the hinge's active side at the current state. A local
                    # finite-difference step must not average active and
                    # inactive sides merely because the fit is near a limit.
                    vp=self.values(plus,samples,signed=True)[0];vm=self.values(minus,samples,signed=True)[0]
                    derivative=(vp-vm)[np.arange(n),joints]/(2e-6)
                    derivative*=values[samples,joints]>0
                    local[np.arange(n),:,columns]=self.weights[samples,None]*derivative
                rows=(3*(len(self.joints)*samples+joints))[:,None]+np.arange(3)
                scatter=csr_matrix((np.ones(3*n),(rows.ravel(),np.arange(3*n))),shape=(residual.size,3*n))
                J=scatter@d.chain(local,sample_indices=samples)
            else:J=csr_matrix((residual.size,self.state.size))
        b=full_columns(block('anatomical_joint_domain',residual.ravel(),J,
            dict(unit='rad',source_pts=self.ref.p['pts'],joint_names=NAMES,limits=self.configuration,
                branch='fixed lifted full-sequence anchor',rest='shaped model local identity; not clinical zero'),self.association))
        stats=[]
        for j,name in enumerate(NAMES):
            v=values[:,j];peak=int(np.argmax(np.linalg.norm(v,axis=1)))
            stats.append(dict(name=name,positive_samples=int(np.any(v>0,axis=1).sum()),
                twist_range_deg=np.degrees([angles.twist[:,j].min(),angles.twist[:,j].max()]).tolist(),
                off_axis_p95_deg=float(np.degrees(np.percentile(angles.swing[:,j],95))),
                peak_dimensionless_violation=float(v.max()),worst_pts=int(self.ref.p['pts'][peak])))
        return dict(residual=b.residual,jac=b.jacobian,parts={b.name:float(b.residual@b.residual/2)},weighted_blocks=[b],
            metrics=dict(anatomical_joint_domain=dict(joints=stats,association=self.association,clinical_angle_claim=False)))

    def proposal_linearization(self,x,rows=None):
        """Signed forms of the same guard boundaries, also inside their limits.

        This only supplies a proposal domain; the original positive-part factor
        and nonlinear peak guard still decide the actual candidate.
        """
        from scipy.sparse import vstack
        samples=np.arange(len(self.times)) if rows is None else np.asarray(rows,int)
        d=self.state.increments(x,self.times[samples]);delta=np.asarray(d)
        values=np.empty((len(samples),len(self.joints),3));pieces=[];order=[]
        for j,(joint,envelope) in enumerate(zip(self.joints,self.envelopes)):
            joint_indices=np.full(len(samples),j,int)
            columns=6+3*(joint-1)+np.arange(3)
            centre=delta[:,columns]
            def evaluate(v):
                rotation=self.base['local_R'][samples,joint-1]@matrices(v)
                return envelope.signed_residual(self.chart.evaluate(rotation,sample_indices=samples,joint_indices=joint_indices))
            values[:,j]=evaluate(centre)
            local=np.zeros((len(samples),3,69))
            for axis,column in enumerate(columns):
                plus=centre.copy();minus=centre.copy();plus[:,axis]+=1e-6;minus[:,axis]-=1e-6
                local[:,:,column]=(evaluate(plus)-evaluate(minus))/(2e-6)
            pieces.append(d.chain(local))
            order.extend((3*(len(self.joints)*np.arange(len(samples))+j))[:,None]+np.arange(3))
        order=np.asarray(order).ravel()
        return values.ravel(),vstack(pieces,format='csr')[np.argsort(order)]

    def export(self,x,path):
        from .skate_physics import fk_state
        d=self.state.increments(x,self.times);s=fk_state(self.ref,self.base,d);values,angles=self.values(d)
        np.savez_compressed(path,source_pts=self.ref.p['pts'],times=self.times,local_R=s['local_R'],global_root_R=s['Q'],
            FK_frames=s['frames'],rest_joint_centres=self.ref.body.J,parents=self.ref.body.parents,
            joint_names=np.array(NAMES),joint_ids=self.joints,anatomical_basis=self.chart.basis,rest_R=self.chart.rest,
            reference_twist=self.chart.reference_twist,twist=angles.twist,swing_vector=angles.swing_vector,
            swing=angles.swing,normalized_violation=values,body_scale=self.ref.body.scale,length_ratio=self.ref.binding.ratio)
