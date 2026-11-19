"""Actual posed-surface contacts on the existing body/skate state.

The broad phase encloses every retained triangle; skin-label seams are never
cut away. Signed witnesses use the oriented face that an edge really crosses.
This is sampled narrow phase and untangling, not certified nonlinear CCD.
"""
from dataclasses import dataclass
from functools import cached_property
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, vstack
from .programme_anatomy import skin_assignment
from .skate_physics import fk_state
from .memory_bounded import slice_reference, slice_delta
from .sequence_skate_state import perturb_relative
from .selected_surface_linearization import SelectedSurfaceLinearization
from .skate_contact import trapezoid_time_weights
from .surface_broad_phase import SurfaceBroadPhase, triangle_spheres
from .weighted_blocks import block, full_columns, signature


def triangle_pairs(vertices, faces, *, topology=None):
    """Conservative sphere query followed by exact current triangle AABBs."""
    triangles=vertices[faces]
    centres,radii,buckets=triangle_spheres(triangles)
    # A few large triangles must not set the radius for every fine skin face.
    # Radius buckets remain conservative: r_i+r_j <= bucket maxima sum.
    low=triangles.min(1);high=triangles.max(1)
    groups=[np.flatnonzero(buckets==key) for key in np.unique(buckets)]
    trees=[cKDTree(centres[g]) for g in groups]
    result=[]
    for u,ga in enumerate(groups):
        for v in range(u,len(groups)):
            gb=groups[v];radius=radii[ga].max()+radii[gb].max()
            coo=trees[u].sparse_distance_matrix(trees[v],radius,output_type='coo_matrix')
            a,b=ga[coo.row],gb[coo.col]
            keep=(a<b) if u==v else np.ones(len(a),bool)
            a,b=a[keep],b[keep]
            keep=np.all((low[a]<=high[b])&(low[b]<=high[a]),axis=1)
            a,b=a[keep],b[keep]
            if len(a):result.append(np.c_[a,b])
    pairs=np.concatenate(result) if result else np.empty((0,2),int)
    if topology is not None and len(pairs):
        a,b=pairs.T
        fa,fb=faces[a],faces[b]
        near=np.zeros(len(pairs),bool)
        # Neighbourhoods are indexed by the original mesh, never image distance.
        for i in range(3):
            for j in range(3):near|=np.asarray(topology[fa[:,i],fb[:,j]]).reshape(-1).astype(bool)
        pairs=pairs[~near]
    return pairs


def edge_triangle(a,b,tri,eps=1e-10):
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];ray=b-a
    h=np.cross(ray,e2);den=np.einsum('ni,ni->n',e1,h)
    valid=abs(den)>eps*np.maximum(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(ray,axis=1),1e-12)
    inv=np.divide(1.,den,out=np.zeros_like(den),where=valid)
    q0=a-tri[:,0];u=np.einsum('ni,ni->n',q0,h)*inv
    q=np.cross(q0,e1);v=np.einsum('ni,ni->n',ray,q)*inv;t=np.einsum('ni,ni->n',e2,q)*inv
    hit=valid&(u>=-eps)&(v>=-eps)&(u+v<=1+eps)&(t>eps)&(t<1-eps)
    return hit,t,np.c_[1-u-v,u,v]


FROZEN_NORMAL_MODE='frozen_root_local_v1'
MOVING_NORMAL_MODE='current_oriented_target_face_v1'


@dataclass
class OrientedFaceWitness:
    """Signed distance to the current directed face, with its normal derivative.

    This is a fixed vertex/face untangling witness, not a penetration-depth
    measurement. Moving both surfaces together must not create a cost decrease.
    """
    gap: np.ndarray
    normal: np.ndarray
    offset: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray
    area: np.ndarray
    bary: np.ndarray
    signs: np.ndarray

    @classmethod
    def from_geometry(cls,point,triangle,bary,signs):
        point=np.asarray(point,float);triangle=np.asarray(triangle,float)
        bary=np.asarray(bary,float);signs=np.asarray(signs,float)
        n=len(point)
        if (point.shape!=(n,3) or triangle.shape!=(n,3,3) or bary.shape!=(n,3)
                or signs.shape!=(n,) or not np.isfinite(point).all()
                or not np.isfinite(triangle).all() or not np.isfinite(bary).all()
                or not np.isin(signs,[-1.,1.]).all()
                or not np.allclose(bary.sum(axis=1),1.,rtol=0,atol=1e-12)):
            raise ValueError('Invalid directed surface witness geometry')
        edge1=triangle[:,1]-triangle[:,0];edge2=triangle[:,2]-triangle[:,0]
        cross=np.cross(edge1,edge2);area=np.linalg.norm(cross,axis=1)
        scale=np.linalg.norm(edge1,axis=1)*np.linalg.norm(edge2,axis=1)
        if np.any(area<=64*np.finfo(float).eps*scale) or np.any(scale==0):
            raise ValueError('Surface witness target face is degenerate at the trial state')
        normal=signs[:,None]*cross/area[:,None]
        offset=point-np.einsum('ni,nij->nj',bary,triangle)
        gap=np.einsum('ni,ni->n',normal,offset)
        return cls(gap,normal,offset,edge1,edge2,area,bary,signs)

    def vertex_gradients(self):
        """Gradient of the directed gap at the fixed current triangle."""
        tangent=(self.offset-self.normal*self.gap[:,None])*(self.signs/self.area)[:,None]
        g1=np.cross(self.edge2,tangent)
        g2=np.cross(tangent,self.edge1)
        face=-self.bary[:,:,None]*self.normal[:,None]
        face[:,0]-=g1+g2
        face[:,1]+=g1
        face[:,2]+=g2
        return self.normal,face

    def derivative(self,point_delta,triangle_delta):
        point_delta=np.asarray(point_delta,float);triangle_delta=np.asarray(triangle_delta,float)
        dcross=(np.cross(triangle_delta[:,1]-triangle_delta[:,0],self.edge2)
                +np.cross(self.edge1,triangle_delta[:,2]-triangle_delta[:,0]))
        signed=self.signs[:,None]*dcross
        dnormal=(signed-self.normal*np.einsum('ni,ni->n',self.normal,signed)[:,None])/self.area[:,None]
        doffset=point_delta-np.einsum('ni,nij->nj',self.bary,triangle_delta)
        return np.einsum('ni,ni->n',self.normal,doffset)+np.einsum('ni,ni->n',dnormal,self.offset)


LEGACY_RESPONSE_POLICY='legacy_external_v1'
CAVITY_RESPONSE_POLICY='own_collar_cavity_side_v1'


def validate_response_policy(packet,policy):
    """A refreshed active set must use the declared physical contact response."""
    if policy not in (LEGACY_RESPONSE_POLICY,CAVITY_RESPONSE_POLICY):
        raise ValueError('Unknown surface response policy: '+str(policy))
    saved=np.asarray(packet.get('surface_response_policy',LEGACY_RESPONSE_POLICY))
    if saved.shape!=():raise ValueError('Surface response policy must be a scalar')
    stored=str(saved.item())
    if stored!=policy:
        raise ValueError('Surface response policy mismatch; refresh actual witnesses before solving')


def crossing_witnesses(vertices,faces,pairs,*,response_signs=None):
    """Vertex/face separating rows for non-coplanar edge/face crossings.

    The negative edge endpoint is selected using the oriented target surface.
    Its opposing point is the actual edge/face intersection barycentric point.
    A row stores both chains; global translation/rotation cannot improve it.
    """
    if response_signs is not None:
        response_signs=np.asarray(response_signs,float)
        if response_signs.shape!=(len(pairs),2) or not np.isin(response_signs,[-1.,1.]).all():
            raise ValueError('Expected one +/-1 response sign for each directed triangle pair')
    rows=[];crossed=np.zeros(len(pairs),bool)
    for swap in (False,True):
        a,b=(pairs[:,1],pairs[:,0]) if swap else (pairs[:,0],pairs[:,1])
        A=vertices[faces[a]];B=vertices[faces[b]]
        normal=np.cross(B[:,1]-B[:,0],B[:,2]-B[:,0]);norm=np.linalg.norm(normal,axis=1)
        normal=np.divide(normal,norm[:,None],out=np.zeros_like(normal),where=norm[:,None]>1e-15)
        if response_signs is not None:normal=normal*response_signs[:,int(swap),None]
        for edge in range(3):
            hit,t,bary=edge_triangle(A[:,edge],A[:,(edge+1)%3],B)
            crossed|=hit
            ids=np.flatnonzero(hit)
            if not len(ids):continue
            plane=np.einsum('ni,ni->n',A[ids,edge]-B[ids,0],normal[ids])
            endpoint=np.where(plane<0,edge,(edge+1)%3)
            va=faces[a[ids],endpoint];vb=faces[b[ids]]
            gap=np.einsum('ni,ni->n',vertices[va]-np.einsum('ni,nij->nj',bary[ids],vertices[vb]),normal[ids])
            for k,i in enumerate(ids):
                if gap[k] < -1e-10:
                    rows.append((int(va[k]),vb[k].copy(),bary[i].copy(),normal[i].copy(),float(gap[k]),int(a[i]),int(b[i])))
    # Same vertex/target face can be discovered by either of its incident edges.
    unique={}
    for row in rows:
        key=(row[0],row[6]);previous=unique.get(key)
        if previous is None or row[4]<previous[4]:unique[key]=row
    return list(unique.values()),pairs[crossed]


class CollisionSurface:
    def __init__(self,reference,document):
        self.ref=reference;self.surface=reference.binding.surface;self.ratio=reference.binding.ratio
        # Rigid shoe poses and deformed uppers only consume the knees, ankles
        # and their ancestors. Perturbing another local joint cannot move them.
        self.gear_pose_joints = set()
        for ankle in (7, 8):
            joint = ankle
            while joint > 0:
                self.gear_pose_joints.add(joint)
                joint = int(reference.body.parents[joint])
        self.skin_count=len(self.surface.shaped);self.skin_faces=self.surface.faces.copy()
        labels=skin_assignment(self.surface)
        groups={0:'torso',3:'torso',6:'torso',9:'torso',12:'torso',13:'torso',14:'torso',15:'head'}
        for side,ids in [('left',(1,4,7,10,16,18,20)),('right',(2,5,8,11,17,19,21))]:
            for j,name in zip(ids,['thigh','shank','foot','foot','upper_arm','forearm','palm']):groups[j]=side+'_'+name
        names=list(dict.fromkeys(groups.values()));self.part_names=names.copy()
        codes=np.array([names.index(groups[int(j)]) for j in labels])
        votes=np.eye(len(names),dtype=int)[codes[self.skin_faces]].sum(1)
        # Assign every original face exactly once, including mixed-label seams.
        self.skin_parts=votes.argmax(1)
        packet=reference.binding.packet();template=packet['template'];self.template_vertices=np.asarray(template['vertices'])
        self.template_faces=np.asarray(template['faces']);self.upper_vertices=np.asarray(packet['upper']);self.upper_faces=np.asarray(reference.binding.faces)
        self.gear_count=len(self.template_vertices)+len(self.upper_vertices)
        faces=[self.skin_faces];parts=[self.skin_parts]
        types=np.asarray(template['face_parts']);shell=np.array(['outsole' in str(p) or 'heel_block' in str(p) for p in types])
        tc=self.template_vertices[self.template_faces].mean(1)[:,0]
        cut=.5*(self.template_vertices[:,0].min()+self.template_vertices[:,0].max())
        for foot,side in enumerate(('left','right')):
            base=self.skin_count+foot*self.gear_count
            faces.extend([base+self.template_faces,base+len(self.template_vertices)+self.upper_faces])
            ids=[]
            for name in ('boot','blade_rear','blade_front'):
                ids.append(len(self.part_names));self.part_names.append(side+'_'+name)
            parts.extend([np.where(shell,ids[0],np.where(tc<cut,ids[1],ids[2])),np.full(len(self.upper_faces),ids[0])])
        self.faces=np.concatenate(faces);self.parts=np.concatenate(parts)
        self.response_policy=document.get('programme_structure',{}).get('surface_response_policy',LEGACY_RESPONSE_POLICY)
        validate_response_policy({'surface_response_policy':self.response_policy},self.response_policy)
        self.face_shank_foot=np.full(len(self.faces),-1,int)
        self.collar_wall_foot=np.full(len(self.faces),-1,int)
        # Only the actual open-collar sidewalls have an inside-cavity relation
        # to the wearer's shank. Tongue, closed instep, sole and opposite shoe
        # retain their existing surface relations and all crossing triangles.
        L=float(packet['parameters'].sole_length_m)
        collar_wall=np.isin(reference.binding.region_ids,[0,1])&(
            self.upper_vertices[self.upper_faces,0].max(axis=1)<=.46*L+1e-12*max(1.,L))
        gear_faces=len(self.template_faces)+len(self.upper_faces)
        for foot,side in enumerate(('left','right')):
            self.face_shank_foot[:len(self.skin_faces)][self.skin_parts==names.index(side+'_shank')]=foot
            start=len(self.skin_faces)+foot*gear_faces+len(self.template_faces)
            self.collar_wall_foot[start+np.flatnonzero(collar_wall)]=foot
        self.vertex_count=self.skin_count+2*self.gear_count
        edges=np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]])
        r,c=edges.T
        adj=csr_matrix((np.ones(2*len(r),np.int8),(np.r_[r,c],np.r_[c,r])),shape=(self.vertex_count,self.vertex_count))
        adj.data[:]=1
        topo=adj+adj@adj
        topo.setdiag(1);topo.data[:]=1;self.topology=topo.astype(bool)
        from pathlib import Path
        import json
        policy_path=document.get('programme_structure',{}).get('policy')
        cfg=json.loads(Path(policy_path).read_text(encoding='utf8')) if policy_path else {}
        self.grips=cfg.get('source_grasp_review',[])
        self.contact_allowance=float(cfg.get('geometry',{}).get('contact_allowance_m',0.))
        self.skin_vertex_parts=codes
        from .body_clock import source_seconds
        self.origin=float(source_seconds(reference.p)[0]-reference.times[0])

    def vertices(self,state,*,skin=None):
        if skin is None:skin=self.surface.sample_vertices(state['native'],state['frames'],hand_local_R=state.get('hand_local_R'),double_precision=True)
        alpha=self.ref.binding.deformation(state['native'],state['frames'],skate_relative_R=state['skate_relative_R'])
        upper=self.ref.binding.deform(self.upper_vertices,alpha)
        # Root-local coordinates make rigid-body cancellation exact by construction.
        axes=state['Q'].swapaxes(-1,-2)[:,None]@state['R']
        origins=np.einsum('nji,nfj->nfi',state['Q'],state['O']-state['P'][:,None])
        out=[skin*self.ratio]
        for f in (0,1):
            local=np.concatenate([np.broadcast_to(self.template_vertices,(len(skin),*self.template_vertices.shape)),upper[:,f]],axis=1)
            out.append((np.einsum('nij,nvj->nvi',axes[:,f],local)+origins[:,f,None])*self.ratio)
        return np.concatenate(out,axis=1)

    def allowed(self,pairs,time):
        a,b=self.parts[pairs[:,0]],self.parts[pairs[:,1]]
        # Template component intersections and own-foot cavity are intentional.
        A=np.asarray(self.part_names)[a];B=np.asarray(self.part_names)[b]
        gear_a=pairs[:,0]>=len(self.skin_faces);gear_b=pairs[:,1]>=len(self.skin_faces)
        allowed=np.zeros(len(pairs),bool);uncertain=np.zeros(len(pairs),bool)
        for side in ('left','right'):
            ag=np.char.startswith(A,side+'_')&gear_a;bg=np.char.startswith(B,side+'_')&gear_b
            allowed|=(ag&bg)|((A==side+'_foot')&bg)|((B==side+'_foot')&ag)
        # Mouth/eyeball seams are a different topology model; retain diagnostics.
        uncertain|=(A=='head')&(B=='head')
        # Broad hand/shoe candidates are not a whole-arm exception. Only their
        # crossing faces are tagged, and unknown hand/material remains unresolved.
        grip=np.zeros(len(pairs),bool)
        seconds=time+self.origin
        for episode in self.grips:
            if episode['source_seconds'][0]<=seconds<=episode['source_seconds'][1]:
                side=('left','right')[episode['foot']]
                target_a=gear_a&np.char.startswith(A,side+'_');target_b=gear_b&np.char.startswith(B,side+'_')
                palm_a=A==episode['hand']+'_palm';palm_b=B==episode['hand']+'_palm'
                grip|=(target_a&palm_b)|(target_b&palm_a)
        return allowed,uncertain,grip

    def response_signs(self,pairs):
        """Inside the own collar, resolve breached skin back into the cavity."""
        if self.response_policy not in (LEGACY_RESPONSE_POLICY,CAVITY_RESPONSE_POLICY):
            raise ValueError('Unknown surface response policy: '+str(self.response_policy))
        signs=np.ones((len(pairs),2))
        if self.response_policy==CAVITY_RESPONSE_POLICY:
            for swap in (False,True):
                a,b=(pairs[:,1],pairs[:,0]) if swap else (pairs[:,0],pairs[:,1])
                target=self.collar_wall_foot[b]
                signs[(target>=0)&(self.face_shank_foot[a]==target),int(swap)]=-1.
        return signs

    @cached_property
    def broad_phase(self):
        binding=getattr(self.ref.binding,'metric_binding',{})
        return SurfaceBroadPhase.optional(self.faces,self.topology,native_library=binding.get('native_library'))

    def scan_sample(self,vertices,time):
        broad_phase=self.broad_phase
        pairs=None if broad_phase is None else broad_phase(vertices)
        if pairs is None:
            pairs=triangle_pairs(vertices,self.faces,topology=self.topology)
        allowed,uncertain,grip=self.allowed(pairs,time)
        ordinary=pairs[~allowed&~uncertain&~grip]
        witness,crossed=crossing_witnesses(vertices,self.faces,ordinary,response_signs=self.response_signs(ordinary))
        grip_rows,grip_cross=crossing_witnesses(vertices,self.faces,pairs[grip&~allowed])
        # Only a shallow intersecting palm face at a source-supported time can
        # be treated as conditional contact. Deep palm/held-shoe penetration
        # remains in the objective; arms and the other hand receive no waiver.
        deep=[w for w in grip_rows if w[4]<-self.contact_allowance]
        witness.extend(deep)
        if deep:
            crossed=np.unique(np.vstack([crossed,np.array([(w[5],w[6]) for w in deep],int)]),axis=0)
        return witness,dict(triangle_candidates=len(pairs),crossing_pairs=len(crossed),
            source_grip_crossing_pairs=len(grip_cross),topology_uncertain_candidates=int(uncertain.sum()),
            intentional_attachment_candidates=int(allowed.sum()),
            pairs=crossed,grip_pairs=grip_cross)


def witnesses_packet(records):
    if not records:return dict(sample=np.empty(0,int),vertex=np.empty(0,int),face_vertices=np.empty((0,3),int),bary=np.empty((0,3)),normal=np.empty((0,3)),initial_gap=np.empty(0),face_a=np.empty(0,int),face_b=np.empty(0,int))
    return dict(sample=np.array([r[0] for r in records]),vertex=np.array([r[1][0] for r in records]),face_vertices=np.array([r[1][1] for r in records]),bary=np.array([r[1][2] for r in records]),normal=np.array([r[1][3] for r in records]),initial_gap=np.array([r[1][4] for r in records]),face_a=np.array([r[1][5] for r in records]),face_b=np.array([r[1][6] for r in records]))


def scan_surface(reference,state,x,document,output,*,workers=8,emit=None):
    """Full native+midpoint sampled review, retaining the deepest pair witnesses."""
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import json,time
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    geometry=CollisionSurface(reference,document)
    native=reference.times
    times=np.sort(np.r_[native,.5*(native[1:]+native[:-1])])
    counts=[];records=[];pair_counts=[];started=time.monotonic()
    def one(item):
        index,vertices=item
        witness,diagnostic=geometry.scan_sample(vertices,times[index])
        # One deepest actual face witness per semantic pair is the current
        # active set; every detected triangle crossing remains in the ledger.
        selected={}
        for w in witness:
            pair=tuple(sorted((int(geometry.parts[w[5]]),int(geometry.parts[w[6]]))))
            if pair not in selected or w[4]<selected[pair][4]:selected[pair]=w
        pairs=diagnostic.pop('pairs');grips=diagnostic.pop('grip_pairs')
        groups={}
        for a,b in pairs:
            key=tuple(sorted((int(geometry.parts[a]),int(geometry.parts[b]))));groups[key]=groups.get(key,0)+1
        return index,diagnostic,list(selected.values()),groups
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,len(times),32):
            ts=times[first:first+32];base=reference.at(ts);d=state.increments(x,ts)
            posed=fk_state(reference,base,d);vertices=geometry.vertices(posed)
            for i,diagnostic,witness,groups in pool.map(one,[(first+j,v) for j,v in enumerate(vertices)]):
                counts.append(diagnostic);records.extend((i,w) for w in witness)
                pair_counts.extend((i,a,b,n) for (a,b),n in groups.items())
            if emit is not None and (first%256==0 or first+32>=len(times)):
                emit('surface_scan',samples=min(first+32,len(times)),total=len(times),
                    selected_witnesses=len(records),seconds=time.monotonic()-started)
    packet=witnesses_packet(records);packet.update(times=times,source_seconds=times+geometry.origin,
        source_pts=reference.p['pts'],surface_response_policy=np.array(geometry.response_policy),
        parts=np.array(geometry.part_names),pair_counts=np.array(pair_counts,int).reshape(-1,4),
        crossing_counts=np.array([r['crossing_pairs'] for r in counts]),
        grip_crossing_counts=np.array([r['source_grip_crossing_pairs'] for r in counts]),
        topology_uncertain_candidates=np.array([r['topology_uncertain_candidates'] for r in counts]))
    np.savez_compressed(output/'surface_scan.npz',**packet)
    report=dict(schema='rinkspace.actual-surface-scan.v1',native_exposures=len(native),samples=len(times),
        whole_skin_faces=len(geometry.skin_faces),retained_skin_faces=len(geometry.skin_faces),
        skin_label_seams_removed=0,faces=len(geometry.faces),vertices=geometry.vertex_count,
        selected_witnesses=len(records),crossing_sample_count=sum(r['crossing_pairs']>0 for r in counts),
        total_crossing_pairs=sum(r['crossing_pairs'] for r in counts),
        source_grip_crossing_pairs=sum(r['source_grip_crossing_pairs'] for r in counts),
        topology_filter='original mesh within two edges; own template interfaces and foot cavity explicit',
        signed_branch='oriented crossed target face with declared contact-side response; negative crossing-edge endpoint; root-local metres',
        surface_response_policy=geometry.response_policy,
        temporal_scope='all native exposures and their midpoints; sampled, not certified CCD',
        limitations=['Closed-solid containment and coplanar overlap require separate certification; no collision-free claim',
                    'Head intrapart faces retained as topology-uncertain diagnostics',
                    'Source-grip crossings are conditional local hand/shoe candidates, not proven legal penetration'],
        elapsed_seconds=time.monotonic()-started)
    (output/'surface_scan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return packet,report


class ProgrammeSurfaceFactor:
    """A versioned actual vertex/face active set in the same global spline."""
    def __init__(self,problem,document,packet,*,scale_m=.03):
        self.problem=problem;self.ref=problem.reference;self.state=problem.body.state
        self.geometry=CollisionSurface(self.ref,document)
        self.packet={k:np.asarray(v).copy() for k,v in packet.items()}
        validate_response_policy(self.packet,self.geometry.response_policy)
        self.times=self.packet['times'];self.scale_m=float(scale_m)
        self.weight=np.sqrt(trapezoid_time_weights(self.times))[self.packet['sample']]/self.scale_m
        if 'witness_share' in self.packet:
            share=self.packet['witness_share']
            if share.shape!=self.weight.shape or np.any(~np.isfinite(share) | (share<=0) | (share>1)):
                raise ValueError('Invalid shared semantic witness budget')
            self.weight*=np.sqrt(share)
        self.association=signature(self.packet['sample'],self.packet['vertex'],self.packet['face_vertices'],self.packet['bary'],self.packet['normal'],self.packet.get('witness_share'))
        self.normal_mode=document.get('programme_structure',{}).get('surface_normal_mode',FROZEN_NORMAL_MODE)
        if self.normal_mode not in (FROZEN_NORMAL_MODE,MOVING_NORMAL_MODE):
            raise ValueError('Unknown surface normal mode: '+str(self.normal_mode))
        if self.normal_mode==MOVING_NORMAL_MODE:
            np.testing.assert_array_equal(self.packet['face_vertices'],self.geometry.faces[self.packet['face_b']])
            self.response_signs=self.geometry.response_signs(np.c_[self.packet['face_a'],self.packet['face_b']])[:,0]
            self.association=signature(self.association,self.normal_mode,self.response_signs)


    def _selected(self,posed,ids,skin=None):
        geometry=self.geometry;ratio=geometry.ratio
        skin_ids=ids[ids<geometry.skin_count];out=np.zeros((len(posed['P']),len(ids),3))
        mask=ids<geometry.skin_count
        if len(skin_ids):
            if skin is None:skin=geometry.surface.sample_vertices(posed['native'],posed['frames'],skin_ids,hand_local_R=posed.get('hand_local_R'),double_precision=True)
            out[:,mask]=skin*ratio
        if np.any(~mask):
            template_count=len(geometry.template_vertices)
            local_gear=(ids[~mask]-geometry.skin_count)%geometry.gear_count
            upper_ids=np.unique(local_gear[local_gear>=template_count]-template_count)
            upper=None
            if len(upper_ids):
                alpha=self.ref.binding.deformation(posed['native'],posed['frames'],skate_relative_R=posed['skate_relative_R'])
                upper=self.ref.binding.deform(geometry.upper_vertices[upper_ids],alpha)
            axes=posed['Q'].swapaxes(-1,-2)[:,None]@posed['R']
            origins=np.einsum('nji,nfj->nfi',posed['Q'],posed['O']-posed['P'][:,None])
            for foot in (0,1):
                begin=geometry.skin_count+foot*geometry.gear_count
                take=(ids>=begin)&(ids<begin+geometry.gear_count)
                if not np.any(take):continue
                localids=ids[take]-begin
                rigid=localids<template_count
                local=np.empty((len(out),len(localids),3))
                local[:,rigid]=geometry.template_vertices[localids[rigid]]
                if not rigid.all():
                    local[:,~rigid]=upper[:,foot,np.searchsorted(upper_ids,localids[~rigid]-template_count)]
                out[:,take]=(np.einsum('nij,nvj->nvi',axes[:,foot],local)+origins[:,foot,None])*ratio
        return out

    def evaluate(self,x,jacobian=False,*,signed_proposal=False):
        from .fullbody_fit import right_jacobian
        packet=self.packet;n=len(packet['sample']);residual=np.zeros(n);jac=[]
        self.last_parameters=np.array(x,copy=True);self.last_gap=np.zeros(n)
        if not n:
            b=full_columns(block('surface_intersection',residual,csr_matrix((0,self.state.size)) if jacobian else None,
                dict(unit='m',scale=self.scale_m,method='actual oriented crossed surfaces'),self.association))
            return dict(residual=residual,jac=b.jacobian,parts={b.name:0.},weighted_blocks=[b],metrics={})
        samples=np.unique(packet['sample']);gap=np.zeros(n)
        for first in range(0,len(samples),24):
            chosen=samples[first:first+24]
            rows=np.flatnonzero(np.isin(packet['sample'],chosen));lookup=np.searchsorted(chosen,packet['sample'][rows])
            ts=self.times[chosen];base=self.ref.at(ts);d=self.state.increments(x,ts)
            posed=fk_state(self.ref,base,d)
            ids=np.unique(np.r_[packet['vertex'][rows],packet['face_vertices'][rows].ravel()])
            va=np.searchsorted(ids,packet['vertex'][rows]);vb=np.searchsorted(ids,packet['face_vertices'][rows])
            bary=packet['bary'][rows];normal=packet['normal'][rows]
            skin_ids=ids[ids<self.geometry.skin_count];skin_mask=ids<self.geometry.skin_count
            linear=None
            if jacobian and len(skin_ids):
                linear=SelectedSurfaceLinearization(self.geometry.surface,posed['native'],posed['frames'],skin_ids,hand_local_R=posed.get('hand_local_R'))
            vertices=self._selected(posed,ids,None if linear is None else linear.samples)
            if getattr(self,'normal_mode',FROZEN_NORMAL_MODE)==MOVING_NORMAL_MODE:
                witness=OrientedFaceWitness.from_geometry(vertices[lookup,va],vertices[lookup[:,None],vb],bary,self.response_signs[rows])
                gap[rows]=witness.gap
                if jacobian:
                    point_gradient,face_gradient=witness.vertex_gradients()
                    def contract(v):
                        return (np.einsum('ni,ni->n',point_gradient,v[lookup,va])
                                +np.einsum('nki,nki->n',face_gradient,v[lookup[:,None],vb]))
            else:
                def contract(v):return np.einsum('ni,ni->n',normal,v[lookup,va]-np.einsum('ni,nij->nj',bary,v[lookup[:,None],vb]))
                gap[rows]=contract(vertices)
            residual[rows]=(-gap[rows] if signed_proposal else np.maximum(-gap[rows],0.))*self.weight[rows]
            if not jacobian:continue
            active=np.ones(len(rows),bool) if signed_proposal else gap[rows]<0
            if not np.any(active):
                jac.append((rows,csr_matrix((len(rows),self.state.size))))
                continue
            local=np.zeros((len(rows),1,69));shoe=np.zeros((len(rows),1,6))
            gain=-self.weight[rows]*active
            gear_rows=~skin_mask[va] | np.any(~skin_mask[vb],axis=1)
            has_gear=np.any(active & gear_rows)
            for joint in range(1,22):
                columns=6+3*(joint-1)+np.arange(3)
                axes=right_jacobian(np.asarray(d[:,columns]))
                skin_derivatives=None if linear is None else linear.derivatives(joint,axes.swapaxes(1,2))*self.geometry.ratio
                for axis,column in enumerate(columns):
                    dv=np.zeros_like(vertices)
                    if skin_derivatives is not None:dv[:,skin_mask]=skin_derivatives[:,axis]
                    if has_gear and joint in self.geometry.gear_pose_joints:
                        plus,minus=d.copy(),d.copy();plus[:,column]+=1e-6;minus[:,column]-=1e-6
                        zp=self._selected(fk_state(self.ref,base,plus),ids[~skin_mask])
                        zm=self._selected(fk_state(self.ref,base,minus),ids[~skin_mask])
                        dv[:,~skin_mask]=(zp-zm)/(2e-6)
                    local[:,0,column]=gain*contract(dv)
            if has_gear:
                # A relative rotation changes the shoe attachment and upper,
                # with the identical posed native skeleton and hands.
                def relative_vertices(rotation):
                    R,O=self.ref.binding.transforms(posed['P'],posed['Q'],posed['native'],posed['frames'],
                        skate_relative_R=rotation)
                    gear_pose=dict(posed,R=R,O=O,skate_relative_R=rotation)
                    return self._selected(gear_pose,ids[~skin_mask])
                for foot in (0,1):
                    for axis in range(3):
                        plus=perturb_relative(d,foot,axis,1e-6).skate_relative_R
                        minus=perturb_relative(d,foot,axis,-1e-6).skate_relative_R
                        dv=np.zeros_like(vertices)
                        dv[:,~skin_mask]=(relative_vertices(plus)-relative_vertices(minus))/(2e-6)
                        shoe[:,0,3*foot+axis]=gain*contract(dv)
            # Both skin supports, gear skin deformation and relative shoe chain
            # are scattered. Root P and Q remain identically zero locally.
            J=d.chain(local,sample_indices=lookup)
            if has_gear:J+=d.owner.shoe_chain(shoe,d.parameters,d.sample_times,sample_indices=lookup)
            jac.append((rows,J))
        J=None
        if jacobian:
            order=np.concatenate([rows for rows,_ in jac]);stack=vstack([J for _,J in jac],format='csr')
            J=stack[np.argsort(order)]
        self.last_gap=gap
        b=full_columns(block('surface_intersection',residual,J,
            dict(unit='m',scale=self.scale_m,source_pts=self.ref.p['pts'],sample_seconds=self.times[packet['sample']]+self.geometry.origin,
                 surface='actual posed full SMPL-X, current hand channels, deformed boot and template steel',
                 signed_branch=('current oriented target face and normal derivative'if getattr(self,'normal_mode',FROZEN_NORMAL_MODE)==MOVING_NORMAL_MODE else'frozen oriented crossed face in root-local frame'),
                  normal_mode=getattr(self,'normal_mode',FROZEN_NORMAL_MODE),root_rigid_columns_cancel=True),self.association))
        return dict(residual=residual,jac=b.jacobian,parts={b.name:float(residual@residual/2)},weighted_blocks=[b],
            metrics=dict(surface_intersection=dict(witnesses=n,active=int((gap<0).sum()),worst_signed_gap_m=float(gap.min()),
                p95_violation_m=float(np.percentile(np.maximum(-gap,0),95)),association=self.association,
                source_seconds_of_worst=float(self.times[packet['sample'][np.argmin(gap)]]+self.geometry.origin),
                proxy_depth_is_not_surface_depth=True,normal_mode=getattr(self,'normal_mode',FROZEN_NORMAL_MODE))))
