"""Proposal linearizations for the existing scoped nonlinear motion limits."""
import numpy as np
from scipy.sparse import csr_matrix, diags, eye, hstack, kron, vstack

from .programme_joint_limits import NAMES
from .programme_repair_factors import component_ice
from .programme_source_step import SourceMargins
from .skate_physics import NORMALIZATION, body_constraint_jacobian, fk_state


class ScopedSurfaceDomains:
    """Signed constraints from refreshed witness pairs on the identical chart."""

    def __init__(self, problem, parent, rejected, sample_caps=(), *, selected=None):
        from .programme_repair_factors import sampled_windows
        from .programme_skate_replay import WITNESS, common_witnesses
        from .programme_surface import ProgrammeSurfaceFactor

        check=problem.scoped_surface_trial
        refreshed=[check.scan_trial(x) for x in rejected]
        geometry=check.surface.geometry
        common=common_witnesses([check.current_scan,*[scan for scan,_ in refreshed]],geometry)
        mask=sampled_windows(problem.reference,problem.body.state,common['times'],check.specification['focus_pts'])
        if selected is not None:
            knots=np.unique((selected-17)//69)
            mask &= np.asarray(problem.body.state.basis(common['times'])[:,knots].getnnz(axis=1)>0)
        keep=mask[common['sample']]
        select=set(WITNESS)|{'witness_share'}
        packet={k:(v[keep] if k in select else v) for k,v in common.items()}
        self.factor=ProgrammeSurfaceFactor(problem,check.document,packet,scale_m=check.surface.scale_m)
        self.factor.evaluate(parent[17:])
        before=self.factor.last_gap
        a,b=(geometry.parts[packet[k]] for k in ('face_a','face_b'))
        pairs=np.sort(np.c_[a,b],axis=1)
        self.cap=np.empty(len(a))
        for left,right in np.unique(pairs,axis=0):
            match=(pairs[:,0]==left)&(pairs[:,1]==right)
            self.cap[match]=max(0.,float(-before[match].min()))+check.allowance-3.e-6
        stamps=packet['times'][packet['sample']]+geometry.origin
        measured=None
        if any(limit.get('drive_current') for limit in sample_caps):
            keys=('sample','vertex','face_b')
            current=set(zip(*(check.current_scan[k] for k in keys)))
            measured=np.array([triple in current for triple in zip(*(packet[k] for k in keys))])
        for limit in sample_caps:
            if limit.get('exact_only'):
                continue
            left,right=sorted(geometry.part_names.index(k) for k in limit['parts'])
            rows=(pairs[:,0]==left)&(pairs[:,1]==right)&np.isclose(stamps,limit['source_seconds'],rtol=0.,atol=1.e-7)
            # A witness from a previous trial can sit behind the target face
            # on the parent while its triangles are disjoint. The signed
            # plane is only a proposal. Exact sampled crossing uses the
            # declared cap below.
            proposal_cap=np.maximum(float(limit['max_penetration_m']),-before[rows])
            if limit.get('drive_current'):
                proposal_cap[measured[rows]]=float(limit['max_penetration_m'])
            self.cap[rows]=np.minimum(self.cap[rows],proposal_cap)
        self.labels=[f'fresh_surface/{t:.10f}/{geometry.part_names[l]}/{geometry.part_names[r]}/v{v}/f{f}'
                     for t,(l,r),v,f in zip(stamps,pairs,packet['vertex'],packet['face_b'],strict=True)]
        self.freshness=dict(scans=[coverage for _,coverage in refreshed],local_common_witnesses=len(self.cap))

    def blocks(self,x):
        out=self.factor.evaluate(x[17:],jacobian=True,signed_proposal=True)
        matrix=diags(1./self.factor.weight)@out['jac']
        slack=self.cap+self.factor.last_gap
        return SourceMargins(matrix.tocsr(),slack,self.labels,[])


def local_surface_measurements(scan, geometry, sample_caps):
    """Measure explicitly reviewed native/midpoint pairs on the fresh mesh."""
    a,b=(geometry.parts[scan[k]] for k in ('face_a','face_b'))
    pairs=np.sort(np.c_[a,b],axis=1)
    stamps=scan['source_seconds'][scan['sample']]
    result=[]
    for limit in sample_caps:
        left,right=sorted(geometry.part_names.index(k) for k in limit['parts'])
        samples=np.isclose(stamps,limit['source_seconds'],rtol=0.,atol=1.e-7)
        selected=samples&(pairs[:,0]==left)&(pairs[:,1]==right)
        actual=float(-scan['initial_gap'][selected].min()) if selected.any() else 0.
        result.append(dict(**limit,measured_maximum_m=actual))
    return result


def local_surface_regressions(scan, geometry, sample_caps):
    return [row for row in local_surface_measurements(scan,geometry,sample_caps)
            if row['measured_maximum_m']>row['max_penetration_m']+1.e-8]


class ScopedMotionDomains:
    def __init__(self, guard, active_columns):
        self.guard = guard
        self.state = guard.state
        self.active_knots = np.unique((active_columns-17)//69)
        local = (active_columns-17)%69
        self.active_tracks = np.unique((local[local>=3]-3)//3)
        self.points = []
        source_start = float(guard.ref.p['pts'][0])
        tick = float(guard.ref.p['time_base'][1])/float(guard.ref.p['time_base'][0])
        self.pts = lambda times: np.round(source_start+times*tick, 7)
        for window, native, physical, derivative in guard.scope:
            root = np.flatnonzero(physical & self.affected(guard.physical_times))
            joint = np.flatnonzero(native & self.affected(guard.joints.times))
            com_affected = self.affected(guard.ref.times)
            com = np.flatnonzero(derivative & (com_affected[:-2]|com_affected[1:-1]|com_affected[2:]))
            steps = native[1:]|native[:-1]
            active_native = self.affected(guard.ref.times)
            end_index = np.flatnonzero(steps & (active_native[1:]|active_native[:-1])) + 1
            if not (len(root) or len(joint) or len(com) or len(end_index)):
                continue
            cfg, old = guard.policy, guard.anchor

            def cap(values):
                peak = float(values.max())
                return peak+max(2.,peak*(cfg['movement_budget_ratio']-1.))

            self.points.append(dict(prefix=f'local/{window[0]}-{window[1]}/', root=root, joint=joint, com=com,
                end_index=end_index, root_limit=cap(old['root'][physical])-2.e-5, com_limit=cap(old['com'][derivative]),
                ice_floor=-guard.radii-
                    np.maximum(-old['ice'][physical]-guard.radii,0.).max(0)-cfg['component_height_numerical_m']+3.e-6,
                joint_limit=old['anatomy'][native].max(0)+cfg['anatomy_peak_budget'],
                anatomy_increment_cap=int(native.sum())*(cfg['anatomy_new_component_budget']-1.e-5),
                rotation_limit=np.deg2rad(old['angle'][steps].max(0)+cfg['world_angular_step_budget_deg'])))

    def affected(self, times):
        return np.asarray(self.state.basis(times)[:,self.active_knots].getnnz(axis=1)>0)

    def blocks(self, x, rotation_linearization):
        """Return signed joint rows and exact local vectors for the radius checks."""
        g, state = self.guard,self.state
        matrix, slack, labels = [], [], []
        balls, averages = [], []

        def norm(name, j, value, caps, row_labels):
            balls.append(dict(source_id=name,jacobian=j,offset=value,
                              radius=np.broadcast_to(caps,(len(value),)),labels=row_labels))

        for info in self.points:
            prefix = info['prefix']
            root, joint, com, end = (info[k] for k in ('root','joint','com','end_index'))
            knot_curve = state.current_curve(x[17:])
            root_clock = g.physical_times[root]
            root_pts = self.pts(root_clock)
            ratio = g.ref.binding.ratio
            D = kron(state.basis(root_clock,2),eye(69,format='csr')[:3],format='csr')*ratio
            J = hstack((csr_matrix((D.shape[0],17)), D, csr_matrix((D.shape[0],state.shoe_size))),format='csr')
            norm(prefix+'root_acceleration',J,ratio*knot_curve.position_derivative(root_clock,2),
                 info['root_limit'],[prefix+'root_acceleration/'+str(stamp) for stamp in root_pts])

            # Signed distances retain the derivative for a toe pick or other
            # shoe component that is clear at R but may enter ice during a
            # trial. Use the same support samples and exact witness geometry.
            d=state.increments(x[17:],root_clock)
            pose=fk_state(g.ref,g.ref.at(root_clock),d)
            parts,height,local_points,upper=component_ice(g.ref.binding,pose,geometry=True)
            floor=info['ice_floor']
            sample,feet,part=np.nonzero(height < floor+.12)
            entries=list(zip(sample,feet,local_points[sample,feet,part],np.ones(len(sample)),strict=True))
            if entries:
                points=dict(rows=entries,times=root_clock,
                            row_upper_vertex=np.where(part==len(parts)-1,upper[sample,feet],-1))
                J=-NORMALIZATION*body_constraint_jacobian(g.ref,pose,d,state.active,state.basis(root_clock),points)
                matrix.append(hstack((csr_matrix((J.shape[0],17)),J),format='csr'))
                slack.append(height[sample,feet,part]-floor[feet,part])
                labels.extend(prefix+f'component_ice/{root_pts[i]}/foot_{f}/{parts[k]}'
                              for i,f,k in zip(sample,feet,part,strict=True))

            # Use the same three-sample derivative operator and mass geometry
            # as the programme guard, including points outside flight that
            # share spline support with the edited pose.
            c = g.com
            if len(com):
                acceleration,J = c.acceleration_rows(x[17:],com,native=True,jacobian=True)
                norm(prefix+'com_acceleration',J,acceleration,info['com_limit'],
                     [prefix+'com_acceleration/'+str(stamp) for stamp in self.pts(g.ref.times[1:-1][com])])

            track = np.tile(self.active_tracks,len(end))
            e = np.repeat(end,len(self.active_tracks))
            pair = g.ref.times[e[:,None]+[-1,0]]
            rot, J = rotation_linearization(state,x,pair,track,jacobian=True)
            row_labels = [prefix+f'local_rotation_step/{g.pts[i]}/track_{j}' for i,j in zip(e,track,strict=True)]
            norm(prefix+'local_rotation_step',J,rot,info['rotation_limit'][track],row_labels)

            signed, J = g.joints.proposal_linearization(x[17:],rows=joint)
            J = hstack((csr_matrix((len(signed),17)),J),format='csr')
            threshold = np.tile(info['joint_limit'].ravel(),len(joint))
            labels.extend(prefix+f'joint_component_peak/{g.pts[i]}/{name}/{k}'
                          for i in joint for name in NAMES for k in range(3))
            matrix.append(J); slack.append(threshold-signed)
            averages.append((prefix,signed.reshape(len(joint),len(NAMES),3),J,
                             g.anchor['anatomy'][joint],info['anatomy_increment_cap'],g.pts[joint]))
        return SourceMargins(vstack(matrix,format='csr'),np.concatenate(slack),labels,balls),averages

    def cuts(self,x,rotation_linearization):
        """Derive constraints at exact trial geometry, with source-row names."""
        value, averages = self.blocks(x,rotation_linearization)
        j,b,names = value
        rows, excess, labels = [], [], []
        for k in np.flatnonzero(b < -5.e-7):
            rows.append(j[k]);excess.append(-b[k]);labels.append(names[k])
        for ball in value.image_balls:
            offset = ball['offset']
            distance = np.linalg.norm(offset,axis=1)
            for i in np.flatnonzero(distance-ball['radius']>5.e-7):
                rows.append(csr_matrix(offset[i:i+1]/distance[i])@ball['jacobian'][3*i:3*i+3])
                excess.append(distance[i]-ball['radius'][i]);labels.append(ball['labels'][i])
        for prefix,signed,j,old,cap,pts in averages:
            change = np.maximum(signed,0.)-old
            increase = np.maximum(change,0).sum(axis=0)
            for index,component in zip(*np.nonzero(increase-cap>5.e-7),strict=True):
                take = np.flatnonzero(change[:,index,component]>0.)
                row = (take*len(NAMES)+index)*3+component
                rows.append(csr_matrix(j[row].sum(axis=0)))
                excess.append(increase[index,component]-cap)
                worst = take[np.argmax(change[take,index,component])]
                labels.append(prefix+f'joint_component_neighbourhood/{pts[worst]}/{NAMES[index]}/{component}')
        return (vstack(rows,format='csr') if rows else csr_matrix((0,len(x))),np.asarray(excess),labels)
