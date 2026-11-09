"""Ordered native image factors on the existing ankle-attached skate meshes.

Source occlusion/range is authored in the ledger. The moving model visible
edge and the inner point/segment association are frozen together. All
derivatives use the same five-node, implicit-row native Brown projection.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from .absolute_motion_state import chain_exposure
from .fullbody_fit import skew
from .programme_skate_evidence import exposure_times
from .rebuild_exposure_body import moving_projection
from .sequence_skate_state import perturb_relative
from .skate_physics import fk_state
from .source_contour_contract import polyline_measure_weights
from .source_visible_model import camera_centres, unoccluded
from .weighted_blocks import block, signature


SEMANTICS = ('boot_shell','heel_shell','toe_shell','outsole','rear_mount','front_mount',
             'runner_lower_edge','runner_upper_edge')
TASK_SCALE = np.array([.10,.10,.10,.30,.30,.30])  # metres, radians


def monotone(cost):
    """One declared direction; all samples in the trace share an ordered path."""
    v=np.asarray(cost,float)
    values=v[0].copy()
    parents=[]
    for row in v[1:]:
        prefix=np.minimum.accumulate(values)
        indices=np.maximum.accumulate(np.where(np.r_[True,np.diff(prefix)<0],np.arange(len(prefix)),0))
        parents.append(indices)
        values=row+prefix
    k=int(np.argmin(values)); path=[k]
    for q in reversed(parents):
        k=int(q[k]);path.append(k)
    return tuple(path[::-1])


def distances(points, curve):
    A=curve[:-1]
    V=np.diff(curve,axis=0)
    denom=np.maximum(np.sum(V*V,axis=1),1e-18)
    u=np.clip(np.einsum('nki,ki->nk',points[:,None]-A,V)/denom,0.,1.)
    Q=A+u[...,None]*V
    return np.sum((Q-points[:,None])**2,axis=-1)


def band(e, radius):
    size=np.linalg.norm(e)
    if size<=radius:return np.zeros(2), np.zeros((2,2))
    u=e/size
    return (size-radius)*u,(1-radius/size)*np.eye(2)+(radius/size)*np.outer(u,u)


def segment_point(target,a,b):
    """Projection and both moving-endpoint derivatives, including d(alpha)."""
    v=b-a; w=target-a
    l2=max(float(v@v),1e-18)
    alpha=float(np.clip(w@v/l2,0.,1.))
    A=(1-alpha)*np.eye(2); B=alpha*np.eye(2)
    if 0.<alpha<1.:
        A+=np.outer(v,((2*alpha-1)*v-w)/l2)
        B+=np.outer(v,(w-2*alpha*v)/l2)
    return a+alpha*v,A,B


@dataclass(frozen=True)
class EdgeMatch:
    vertices: tuple
    forward: tuple
    reverse: tuple
    model_weights: tuple


def edge_terms(model, source, match, radius, weight, complete, endpoints, *, chord_fraction=0.):
    """Finite region and raw geometry Jacobians on a small projected polyline."""
    raw, r, J, Jraw=[],[],[],[]
    span=(.80 if endpoints else 1.)*(1.-chord_fraction)*weight/(2 if complete else 1)
    end_terms=[]
    sw=polyline_measure_weights(source,closed=False)
    def add(e, w, derivatives, radius_px):
        b,D=band(e,radius_px)
        m=np.zeros((2,model.size))
        for k,A in derivatives:
            m[:,2*k:2*k+2] += A
        raw.append(e);r.append(b*w);J.append(w*D@m);Jraw.append(m)
    for i,k in enumerate(match.forward):
        q,A,B=segment_point(source[i],model[k],model[k+1])
        error=q-source[i]
        derivatives=[(k,A),(k+1,B)]
        add(error,np.sqrt(span*sw[i])/radius,derivatives,radius)
        if chord_fraction and i in (0,len(source)-1):
            end_terms.append((error,derivatives))
    if complete:
        for i,k in enumerate(match.reverse):
            q,A,B=segment_point(model[i],source[k],source[k+1])
            add(model[i]-q,np.sqrt(span*match.model_weights[i])/radius,
                [(i,A+B)],radius)
    for end in endpoints:
        if end['model_end'] not in ('start','end'):
            raise ValueError('Explicit endpoint must identify a semantic boundary end')
        i=0 if end['model_end']=='start' else len(model)-1
        er=float(end['radius_px'])
        add(model[i]-np.asarray(end['uv']),np.sqrt(.20*weight/len(endpoints))/er,
            [(i,np.eye(2))],er)
    if chord_fraction:
        first,last=end_terms
        # Same observed arc and total weight. Only the span between the
        # observed samples contributes, with both endpoint bands combined.
        chord_radius=2*radius
        add(last[0]-first[0],np.sqrt(chord_fraction*weight)/chord_radius,
            last[1]+[(k,-A) for k,A in first[1]],chord_radius)
    return np.concatenate(r),np.vstack(J),np.asarray(raw),np.vstack(Jraw)


def measured_chord_fraction(row, source):
    if (row['event_id'].startswith('temporal_skate_') and row['semantic'].startswith('runner_')
            and np.linalg.norm(source[-1]-source[0])>=16.):
        return .22
    return 0.


def edge_chains(edges):
    """Split a visible exterior edge set at gaps and branches; never bridge occlusion."""
    neighbors=defaultdict(set)
    for i,j in edges:
        neighbors[i].add(j); neighbors[j].add(i)
    unused={tuple(sorted(edge)) for edge in edges}
    def walk(start,nxt):
        path=[start,nxt]
        unused.remove(tuple(sorted((start,nxt))))
        while len(neighbors[path[-1]])==2:
            k=next(v for v in neighbors[path[-1]] if v!=path[-2])
            key=tuple(sorted((path[-1],k)))
            if key not in unused:break
            unused.remove(key)
            path.append(k)
        return path
    for k,nb in sorted(neighbors.items()):
        if len(nb)==2:continue
        for j in sorted(nb):
            if tuple(sorted((k,j))) in unused:yield walk(k,j)
    while unused:
        i,j=min(unused)
        yield walk(i,j)


class SemanticSkateMesh:
    def __init__(self, binding):
        self.binding=binding
        p=binding.packet()
        t=p['template'];base=np.asarray(t['vertices'])
        self.upper_offset=len(base)
        self.upper=np.asarray(p['upper'])
        self.base=np.r_[base,self.upper]
        self.upper_end=len(self.base)
        self.faces=np.r_[t['faces'], self.upper_offset+binding.faces]
        self.parts=np.r_[t['face_parts'],np.full(len(binding.faces),'upper')]
        self.length=p['parameters'].sole_length_m
        self.runner={}
        for k in ('runner_lower_edge','runner_upper_edge'):
            ids=[]
            for edge in p['edges']:
                # Keep exact original template runner samples. Decimate only
                # for this pixel objective, never for physics or rendering.
                path=edge[np.r_[np.arange(0,len(edge)-1,4),len(edge)-1]].copy()
                if k=='runner_upper_edge':path[:,2]+=p['parameters'].blade_web_height_m
                ids.append(np.arange(len(self.base),len(self.base)+len(path)))
                self.base=np.r_[self.base,path]
            self.runner[k]=ids
        catalog=defaultdict(list)
        for i,face in enumerate(self.faces):
            for a,b in zip(face,np.roll(face,-1)):
                catalog[tuple(sorted((int(a),int(b))))].append(i)
        families={k:[] for k in SEMANTICS if k not in self.runner}
        for edge,ff in catalog.items():
            parts=set(self.parts[ff]);q=self.base[list(edge)]
            if 'upper' in parts:
                families['boot_shell'].append((edge,ff))
                if q[:,0].max()<=.39*self.length:families['heel_shell'].append((edge,ff))
                if q[:,0].min()>=.59*self.length:families['toe_shell'].append((edge,ff))
            for k,allowed in [('outsole',{'outsole','heel_block'}),
                              ('rear_mount',{'rear_post','rear_mount_plate'}),
                              ('front_mount',{'front_post','front_nose_post','front_mount_plate'})]:
                if parts&allowed:families[k].append((edge,ff))
        self.families=families

    def local(self, posed, feet, ids=None):
        n=len(feet)
        if ids is None: ids=np.broadcast_to(np.arange(len(self.base)),(n,len(self.base)))
        q=self.base[ids].copy()
        upper=(ids>=self.upper_offset)&(ids<self.upper_end)
        if upper.any():
            alpha=self.binding.deformation(posed['native'],posed['frames'],
                                           skate_relative_R=posed['skate_relative_R'])
            u=self.binding.deform(self.upper,alpha)[np.arange(n),feet]
            rows,cols=np.nonzero(upper)
            q[rows,cols]=u[rows,ids[rows,cols]-self.upper_offset]
        return q

    def paths(self,q,camera,k):
        triangles=q[self.faces]
        if k in self.runner:
            path=self.runner[k][int(camera[1]>=0)]
            visible=unoccluded(camera,(q[path[:-1]]+q[path[1:]])*.5,triangles)
            return list(edge_chains([(int(a),int(b)) for a,b,take in zip(path[:-1],path[1:],visible) if take]))
        normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        front=np.einsum('ij,ij->i',normal,camera-triangles[:,0])>0
        edges=[]
        for edge,ff in self.families[k]:
            is_front=front[ff]
            # A same-facing interior triangulation is not a source silhouette.
            if is_front.any() and (len(ff)==1 or not is_front.all()):
                edges.append(edge)
        if not edges:return []
        midpoint=q[np.array(edges)].mean(axis=1)
        seen=unoccluded(camera,midpoint,triangles)
        return list(edge_chains([edge for edge,vis in zip(edges,seen) if vis]))


class OrderedNativeSkateFactors:
    def __init__(self, problem, rows, *, name='ordered_native_skate', mesh=None):
        self.problem=problem;self.rows=tuple(rows);self.name=name;self.matches=None
        self.mesh=SemanticSkateMesh(problem.binding) if mesh is None else mesh
        if any(r['semantic'] not in SEMANTICS for r in self.rows):
            raise ValueError('Unknown skate semantic')
        native=problem.reference.p['pts']
        self.frames=np.searchsorted(native,[r['pts'] for r in self.rows])
        np.testing.assert_array_equal(native[self.frames],[r['pts'] for r in self.rows])
        self.feet=np.array([r['foot'] for r in self.rows],int)
        self.times=exposure_times(problem,self.frames)
        # Bound dense five-node scratch space for programme-wide ledgers.
        self.batches=[]
        if len(self.rows)>64:
            for start in range(0,len(self.rows),64):
                end=min(start+64,len(self.rows))
                child=OrderedNativeSkateFactors(problem,self.rows[start:end],name=name,mesh=self.mesh)
                self.batches.append((start,end,child))
        self.base=problem.reference.at(self.times) if self.rows and not self.batches else None
        # These are discrete image observations. Never integrate over gaps.
        populations=defaultdict(int)
        for r in rows:populations[r['pts'],r['foot']]+=1
        self.weights=[.02*float(r.get('relative_weight',1.))/populations[r['pts'],r['foot']]
                      for r in rows]
        for start,end,child in self.batches:
            child.weights=self.weights[start:end]

    @staticmethod
    def _paths_in_source_order(paths, row, local, projected, first_uv):
        for path in paths:
            if row['source_order']=='image_trace':
                # Image order has no declared anatomical direction.
                for ordered in (path,path[::-1]):
                    if ordered[0]==ordered[-1]:
                        ring=np.array(ordered[:-1],int)
                        start=int(np.argmin(np.linalg.norm(projected[ring]-first_uv,axis=1)))
                        ring=np.roll(ring,2-start)
                        yield np.r_[ring,ring[0]]
                    else:
                        yield ordered
                continue
            # Anatomical traces keep their order when the skater turns.
            axis=0 if row['source_order']=='rear_to_front' else 2
            sign=-1 if row['source_order']=='boot_to_steel' else 1
            if sign*(local[path[-1],axis]-local[path[0],axis])<0:
                path=path[::-1]
            yield path

    def association_record(self):
        if self.matches is None:raise ValueError('Ordered source match was not initialized')
        return [dict(id=r['raw_group_id'],match=None if a is None else dict(
            vertices=list(a.vertices),forward=list(a.forward),reverse=list(a.reverse),model_weights=list(a.model_weights)))
            for r,a in zip(self.rows,self.matches,strict=True)]

    def restore(self,record):
        if [r['raw_group_id'] for r in self.rows]!=[r['id'] for r in record]:
            raise ValueError('Ordered skate association changed source row identities')
        matches=[]
        for row,entry in zip(self.rows,record,strict=True):
            m=entry['match']
            if m is None:matches.append(None);continue
            vertices,forward,reverse,weights=(tuple(m[k]) for k in ('vertices','forward','reverse','model_weights'))
            count=len(row['polyline_uv'])
            is_index=lambda v,n:type(v)is int and 0<=v<n
            if (len(vertices)<2 or not all(is_index(v,len(self.mesh.base)) for v in vertices)
                    or len(forward)!=count or len(reverse)!=len(vertices) or len(weights)!=len(vertices)
                    or not all(is_index(v,len(vertices)-1) for v in forward)
                    or not all(is_index(v,count-1) for v in reverse)
                    or list(forward)!=sorted(forward) or list(reverse)!=sorted(reverse)
                    or not np.isfinite(weights).all() or np.any(np.asarray(weights)<=0) or not np.isclose(sum(weights),1.)):
                raise ValueError('Invalid frozen ordered skate association: '+row['raw_group_id'])
            matches.append(EdgeMatch(vertices,forward,reverse,weights))
        self.matches=tuple(matches)

    def _world(self, posed, ids=None):
        f=np.repeat(self.feet,5)
        ix=np.arange(len(f))
        R,O=posed['R'][ix,f],posed['O'][ix,f]
        local=self.mesh.local(posed,f,None if ids is None else np.repeat(ids,5,axis=0))
        return (O[:,None]+np.einsum('nij,nkj->nki',R,local)).reshape(len(self.rows),5,-1,3)

    def associate(self,x):
        if self.batches:
            for _,_,child in self.batches:
                child.associate(x)
            self.matches=tuple(match for _,_,child in self.batches for match in child.matches)
            return
        p=self.problem
        if not self.rows:
            self.matches=()
            return
        d=p.body.state.increments(x,self.times);s=fk_state(p.reference,self.base,d)
        f=np.repeat(self.feet,5);i=np.arange(len(f))
        R=s['R'][i,f][2::5];O=s['O'][i,f][2::5]
        q=self.mesh.local(s,f)[2::5]
        cameras=camera_centres(p.camera,self.frames,[np.median(np.asarray(r['polyline_uv'])[:,1]) for r in self.rows])
        centres=np.einsum('nji,nj->ni',R,cameras-O)
        uv=moving_projection(p.camera,self._world(s),self.frames)['uv']
        matches=[]
        for n,(r,V) in enumerate(zip(self.rows,uv)):
            source=np.asarray(r['polyline_uv'],float)
            paths=self.mesh.paths(q[n],centres[n],r['semantic'])
            best=None
            for path in self._paths_in_source_order(paths,r,q[n],V,source[0]):
                model=V[path]
                keep=np.r_[True,np.linalg.norm(np.diff(model,axis=0),axis=1)>.05]
                path=np.array(path)[keep]; model=model[keep]
                if len(model)<2:continue
                # Avoid a two-point fragment winning for a complete source.
                F=distances(source,model)
                idx=monotone(F); rev=monotone(distances(model,source))
                loss=np.average(F[np.arange(len(source)),idx],
                                weights=polyline_measure_weights(source,closed=False))
                chord=measured_chord_fraction(r,source)
                if chord:
                    endpoints=[segment_point(source[i],model[idx[i]],model[idx[i]+1])[0]
                               for i in (0,len(source)-1)]
                    e=endpoints[1]-endpoints[0]-(source[-1]-source[0])
                    loss=(1.-chord)*loss+chord*(e@e)/4.
                if r['complete_visible']:
                    loss+=np.average(distances(model,source)[np.arange(len(model)),rev],
                                     weights=polyline_measure_weights(model,closed=False))
                    # A complete runner observation cannot match a one-post
                    # fragment; actual source and model visibility both matter.
                    if r['semantic'] in self.mesh.runner and np.ptp(q[n,path,0])<.50*self.mesh.length:
                        continue
                match=EdgeMatch(tuple(map(int,path)),idx,rev,tuple(polyline_measure_weights(model,closed=False)))
                if best is None or loss<best[0]:best=loss,match
            matches.append(None if best is None else best[1])
        self.matches=tuple(matches)

    def associate_changed(self, x, parent, parent_matches):
        """Keep the exact model paths where the attached foot did not move."""
        state=self.problem.body.state
        delta=np.asarray(x)!=np.asarray(parent)
        body=delta[:state.body_size].reshape(-1,69)
        shoe=delta[state.body_size:].reshape(-1,2,3)
        clocks=self.times.reshape(-1,5)
        legs=[np.r_[6+3*f:9+3*f,15+3*f:18+3*f,24+3*f:27+3*f,33+3*f:36+3*f]
              for f in (0,1)]
        common=np.setdiff1d(np.arange(69),np.concatenate(legs))
        def affected(knots,controls):
            span=np.clip(np.searchsorted(knots,clocks,side='right')-4,0,len(knots)-8)
            return np.isin(span[...,None]+np.arange(4),controls).any(axis=(1,2))
        changed=affected(state.knots,np.flatnonzero(body[:,common].any(axis=1)))
        for foot,axes in enumerate(legs):
            body_controls=np.flatnonzero(body[:,axes].any(axis=1))
            shoe_controls=np.flatnonzero(shoe[:,foot].any(axis=1))
            changed|=(self.feet==foot)&(affected(state.knots,body_controls)|
                                      affected(state.shoe_curve.knots,shoe_controls))
        touched=np.flatnonzero(changed)
        matches=list(parent_matches)
        if len(touched):
            subset=OrderedNativeSkateFactors(self.problem,[self.rows[i] for i in touched],mesh=self.mesh)
            subset.associate(x)
            for i,match in zip(touched,subset.matches,strict=True):
                matches[i]=match
        self.matches=tuple(matches)
        return len(touched)

    def evaluate(self,x,jacobian=False):
        if self.matches is None:self.associate(x)
        if self.batches:
            values=[]
            for start,end,child in self.batches:
                child.matches=self.matches[start:end]
                values.append(child.evaluate(x,jacobian))
            if jacobian:
                self.task_jacobians=[item for _,_,child in self.batches for item in child.task_jacobians]
                self.raw_task_jacobians=[item for _,_,child in self.batches for item in child.raw_task_jacobians]
            residual=np.concatenate([v['residual'] for v in values])
            J=vstack([v['jac'] for v in values],format='csr') if jacobian else None
            b=block(self.name,residual,J,dict(unit='px',clock='native five node',
                source_pts=[r['pts'] for r in self.rows],raw_group_ids=[r['raw_group_id'] for r in self.rows],
                independent_exposure_measure_s=.02),signature(repr(self.matches)))
            return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
                        metrics={self.name:[m for v in values for m in v['metrics'][self.name]]},weighted_blocks=[b])
        p=self.problem
        key=signature(repr(self.matches))
        if not self.rows:
            b=block(self.name,[],csr_matrix((0,p.size)) if jacobian else None,association=key)
            return dict(residual=b.residual,jac=b.jacobian,parts={self.name:0.},weighted_blocks=[b],
                        metrics={self.name:[]})
        widths=[len(a.vertices) if a is not None else 2 for a in self.matches]
        ids=np.array([np.pad(np.asarray((0,1) if a is None else a.vertices),(0,max(widths)-w),mode='edge')
                      for a,w in zip(self.matches,widths)],int)
        d=p.body.state.increments(x,self.times);s=fk_state(p.reference,self.base,d)
        nodes=self._world(s,ids)
        pr=moving_projection(p.camera,nodes,self.frames,jacobian=jacobian)
        if jacobian:
            native=pr['pixel_jac_nodes']
            B=np.zeros((*native.shape[:-1],69))
            B[...,:3]=native
            S=np.zeros((*native.shape[:-1],6))
            def diff(plus,minus):
                v=(self._world(fk_state(p.reference,self.base,plus),ids)-
                   self._world(fk_state(p.reference,self.base,minus),ids))/(2.e-6)
                return np.einsum('nkpac,nakc->nkpa',native,v)
            for axis in [*range(3,12),*range(15,21),*range(24,30)]:
                plus=d.copy();minus=d.copy();plus[:,axis]+=1.e-6;minus[:,axis]-=1.e-6
                B[...,axis]=diff(plus,minus)
            for f in (0,1):
                for a in range(3):
                    S[...,3*f+a]=diff(perturb_relative(d,f,a,1.e-6),perturb_relative(d,f,a,-1.e-6))
            # Diagnostic ONLY: independent rigid motion of each observed shoe,
            # in world axes about its current origin. Nothing writes it to x.
            f=np.repeat(self.feet,5)
            O=s['O'][np.arange(len(f)),f].reshape(len(self.rows),5,1,3)
            task=np.einsum('nkpac,nakcd->nkpad',native,-skew(nodes-O)).sum(axis=3)
            task=np.concatenate((native.sum(axis=3)/p.binding.ratio,task),axis=-1)
            self.task_jacobians=[]
            self.raw_task_jacobians=[]
        residuals=[]; Jbody=[]; Jshoe=[]; owners=[]; metrics=[]
        for n,(r,match,width,weight) in enumerate(zip(self.rows,self.matches,widths,self.weights)):
            m=dict(id=r['raw_group_id'],pts=r['pts'],foot=r['foot'],role=r['role'],semantic=r['semantic'],
                   event_id=r['event_id'],phase=r['phase'],radius_px=r['radius_px'],source_uv=r['polyline_uv'],
                   visibility=r['visibility'],complete_visible=r['complete_visible'])
            if match is None:
                m.update(status='no_visible_same_semantic_model_path')
            else:
                model=pr['uv'][n,:width]
                source=np.asarray(r['polyline_uv'])
                chord=measured_chord_fraction(r,source)
                rv,D,raw,D0=edge_terms(model,source,match,r['radius_px'],weight,
                                       r['complete_visible'],r.get('structural_endpoints',[]),chord_fraction=chord)
                residuals.append(rv)
                if chord:
                    m['chord_error_px']=float(np.linalg.norm(raw[-1]))
                actual=raw[:-1] if chord else raw
                m.update(status='evaluated',distance_px=np.linalg.norm(actual,axis=1).tolist(),
                         matched_source_uv=(source+actual[:len(source)]).tolist(),
                         forward_samples=len(match.forward),reverse_samples=len(match.reverse) if r['complete_visible'] else 0,
                         model_uv=model.tolist(),model_vertex_ids=list(match.vertices))
                if jacobian:
                    Jbody.extend(D@B[n,:width].reshape(width*2,5*69))
                    Jshoe.extend(D@S[n,:width].reshape(width*2,5*6))
                    owners.extend([n]*len(rv))
                    T=task[n,:width].reshape(-1,6)
                    self.task_jacobians.append((r,D@T,rv))
                    self.raw_task_jacobians.append((r,D0@T))
            metrics.append(m)
        residual=np.concatenate(residuals) if residuals else np.empty(0)
        J=None
        if jacobian:
            sample=np.array(owners,int)[:,None]*5+np.arange(5)
            BJ=chain_exposure(np.asarray(Jbody).reshape(-1,1,5,69),d,sample_indices=sample)
            SJ=d.owner.shoe_exposure_chain(np.asarray(Jshoe).reshape(-1,1,5,6),d.parameters,self.times,sample_indices=sample)
            J=hstack((csr_matrix((len(residual),17)),BJ+SJ),format='csr')
        b=block(self.name,residual,J,dict(unit='px',clock='native five node',
            source_pts=[r['pts'] for r in self.rows],raw_group_ids=[r['raw_group_id'] for r in self.rows],
            independent_exposure_measure_s=.02),key)
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
                    metrics={self.name:metrics},weighted_blocks=[b])

    def pose_task_probe(self,x,*,family_weight=1.):
        """Diagnostic free rigid shoe twists, solved per native exposure/foot.

        Does not modify body, shoe controls, associations or candidate state.
        """
        self.evaluate(x,True)
        groups=defaultdict(list)
        raw=defaultdict(list)
        for r,J,res in self.task_jacobians:
            groups[r['pts'],r['foot']].append((J,res))
        for r,J in self.raw_task_jacobians:
            raw[r['pts'],r['foot']].append(J)
        out=[]
        for (pts,foot),parts in groups.items():
            A=np.vstack([J for J,_ in parts])*TASK_SCALE
            b=np.concatenate([v for _,v in parts])
            rawA=np.vstack(raw[pts,foot])*TASK_SCALE
            _,ss,vh=np.linalg.svd(A,full_matrices=False)
            z=np.linalg.lstsq(np.vstack((A,.03*np.eye(6))),np.r_[-b,np.zeros(6)],rcond=None)[0]
            z/=max(1.,np.linalg.norm(z))
            out.append(dict(pts=pts,foot=foot,task_scales=TASK_SCALE.tolist(),
                singular_values=ss.tolist(),weakest_scaled_direction=vh[-1].tolist(),
                raw_geometry_singular_values=np.linalg.svd(rawA,compute_uv=False).tolist(),
                weighted_rank=int((ss>max(ss[0]*1.e-5,1.e-12)).sum()),
                source_cost=float(.5*family_weight*(b@b)),
                task_step_xyz_m_and_rot_rad=(z*TASK_SCALE).tolist(),
                diagnostic_linear_source_cost=float(.5*family_weight*np.linalg.norm(b+A@z)**2)))
        return out
