"""Per-episode non-degradation of the saved model's confirmed-flight physics.

These are relative engineering guards on the same mass proxy and source clock,
not calibrated accuracy limits or independent world measurements.
"""
import numpy as np

FAMILIES=frozenset(('flight_integrated_com','flight_specific_momentum'))
PREFIX='__flight__/'


def flight_blocks(value,families=FAMILIES):
    selected={b.name:b for b in value['weighted_blocks'] if b.name.rsplit('/',1)[-1] in families}
    if not selected:raise ValueError('Flight guards require declared confirmed-flight blocks')
    episodes={name.rsplit('/',1)[0] for name in selected}
    if any({name.rsplit('/',1)[-1] for name in selected if name.rsplit('/',1)[0]==episode}!=families for episode in episodes):
        raise ValueError('Each guarded flight requires both COM and specific momentum' if families==FAMILIES else 'Each guarded flight requires every declared physics family')
    for name,b in selected.items():
        if not len(b.residual) or not np.isfinite(b.residual).all():raise ValueError('Invalid flight residual: '+name)
    return selected


class ProgrammeFlightGuard:
    families=FAMILIES
    prefix=PREFIX
    schema='rinkspace.flight-nondegradation.v1'
    def __init__(self,value,previous=None,*,definition_identity=None):
        if definition_identity is not None and (not isinstance(definition_identity,str) or not definition_identity):
            raise ValueError('Nonempty flight definition identity required')
        self.definition_identity=definition_identity
        blocks=flight_blocks(value,self.families)
        self.rows={name:len(b.residual) for name,b in blocks.items()}
        self.caps={name:float(np.linalg.norm(b.residual)) for name,b in blocks.items()}
        self.jacobians={}
        if previous is not None:
            if previous.get('definition_identity')!=definition_identity:
                raise ValueError('Flight physical definition identity changed')
            if previous.get('schema')!=self.schema or previous['rows']!=self.rows or set(previous['caps'])!=set(self.caps):
                raise ValueError('Flight guard population changed')
            for name,cap in previous['caps'].items():
                cap=float(cap)
                if not np.isfinite(cap) or cap<0:raise ValueError('Invalid previous flight norm cap')
                if self.caps[name]>cap+self.tolerance(cap):raise ValueError('Saved state violates its preceding flight guard')
                self.caps[name]=min(self.caps[name],cap)

    @staticmethod
    def tolerance(cap):
        # Numerical evaluation allowance, not a physical acceptance threshold.
        return 1e-7*max(1.,float(cap))

    def _blocks(self,value):
        blocks=flight_blocks(value,self.families)
        if {name:len(b.residual) for name,b in blocks.items()}!=self.rows:
            raise ValueError('Flight guard population changed during a candidate trial')
        return blocks

    def checks(self,value):
        result={}
        for name,b in self._blocks(value).items():
            norm=float(np.linalg.norm(b.residual));cap=self.caps[name]
            result[self.prefix+name]=dict(rejected=norm>cap+self.tolerance(cap),before=cap,after=norm,
                numerical_norm_allowance=self.tolerance(cap),unit='weighted flight residual norm',
                original_radius_px=None,model_foot=None,
                meaning='Same confirmed-flight COM/H residual cannot worsen while other objective blocks improve; relative model-proxy guard, not physical accuracy acceptance')
        return result

    def domains(self,value):
        result=[]
        for name,b in self._blocks(value).items():
            if b.jacobian is None or b.jacobian.shape[0]!=self.rows[name]:raise ValueError('Actual flight Jacobian required')
            key=self.prefix+name;matrix=b.jacobian.tocsr()
            if not np.isfinite(matrix.data).all():raise ValueError('Nonfinite flight Jacobian: '+name)
            self.jacobians[key]=matrix
            result.append(dict(source_id=key,model_foot=None,unit='weighted flight residual norm',
                guard_margin=0.,jacobian=matrix,offset=np.asarray(b.residual).reshape(1,-1).copy(),
                radius=np.array([self.caps[name]]),labels=[key]))
        return result

    def dependency(self,key):
        matrix=self.jacobians.get(key)
        return None if matrix is None or not matrix.nnz else dict(jacobian=matrix,model_foot=None)

    def record(self):
        return dict(schema=self.schema,rows=self.rows,caps=self.caps,definition_identity=self.definition_identity,
            numerical_allowance='1e-7 * max(1, cap) in the actual weighted residual norm',
            same_mass_source_clock_and_residual_definition_required=True,
            acceptance_calibrated=False,proxy_is_not_measured_com=True)


class ProgrammeDenseFlightGuard(ProgrammeFlightGuard):
    """The same no-regression contract for newly covered inter-exposure gravity."""
    families=frozenset(('flight_dense_com_acceleration',))
    prefix='__flight_dense__/'
    schema='rinkspace.dense-flight-nondegradation.v1'
