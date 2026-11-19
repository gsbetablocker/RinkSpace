"""Visible ambiguous shoe arcs constrain either real boot, without naming a foot.

Qualified automatic boundaries or reviewed open source arcs enter this factor.
A merged mask or a mask whose anatomical label flipped must not drag the hidden
boot onto the visible one. Reviewed arcs retain boot/blade material and one
latent foot per arc. Physical identity is retained by articulated legs and
reviewed source tracks, not supplied by an uncertain segmentation label.
"""
from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import vstack

from .programme_whole_skate import ambiguous_shoe_frames,native_point_derivative,chain
from .programme_skate_evidence import exposure_times
from .rebuild_exposure_body import moving_projection
from .rebuild_prepare import digest,load
from .skate_physics import fk_state
from .weighted_blocks import block,signature


def reviewed_arc_rows(document, source_pts, frames):
    """Keep observed open arcs, with one finite measure per correlated arc.

    These are source-image regions, not labelled anatomical feet or fabricated
    closed masks. Unknown transitions never become a fitting target.
    """
    selected=set(map(int,frames));rows={};clock=np.asarray(source_pts)
    for frame in document['frames']:
        stamp=int(frame['source_pts']);index=int(np.searchsorted(clock,stamp))
        if index>=len(clock) or clock[index]!=stamp:
            raise ValueError('Reviewed arc is not on the native source clock')
        if index not in selected:continue
        for arc in frame['arcs']:
            if arc['type'] not in ('boot_outer_edge','blade_outer_edge','heel_block_edge') or not arc.get('use_for_fitting',True):continue
            points=np.asarray(arc['points_xy'],float)
            radius=float(arc['uncertainty_radius_px']);weight=float(arc.get('weight',9.))
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
                raise ValueError('An observed open arc needs finite original pixels')
            if not np.isfinite(radius) or radius<=0 or not np.isfinite(weight) or weight<=0:
                raise ValueError('Reviewed arc needs finite positive uncertainty and measure')
            group=arc.get('correlation_group',arc['id'])
            material={'boot_outer_edge':0,'blade_outer_edge':1,'heel_block_edge':2}[arc['type']]
            rows.setdefault(index,[]).append((points,radius,weight,material,group))
    result=[]
    for index,arcs in sorted(rows.items()):
        # No resampling, closing, or extension through hidden shoe surfaces.
        target=np.concatenate([a[0] for a in arcs])
        radius=np.concatenate([np.full(len(a[0]),a[1]) for a in arcs])
        scale=np.concatenate([np.full(len(a[0]),np.sqrt(a[2]/len(a[0]))) for a in arcs])
        material=np.concatenate([np.full(len(a[0]),a[3],int) for a in arcs])
        names={name:i for i,name in enumerate(dict.fromkeys(a[4] for a in arcs))}
        groups=np.concatenate([np.full(len(a[0]),names[a[4]],int) for a in arcs])
        result.append((index,target,radius,scale,material,groups))
    return result


def nearest_visible_edges(assembly,pixels,target,allowed_edges=None,boundary_faces=None):
    pairs=assembly.edges
    if allowed_edges is not None:pairs=pairs[allowed_edges]
    if boundary_faces is None:
        boundary=np.concatenate(assembly.raster(pixels))
        keep=cKDTree(boundary).query(pixels[pairs].mean(1))[0]<1.55
        # A wrong pose may hide all of the blade behind the boot. The observed
        # blade still constrains actual steel edges, never a nearby white upper.
        if keep.any() or allowed_edges is None:pairs=pairs[keep]
    # For exact material observations retain the complete set of that material's
    # projected edges. Raster-midpoint visibility switches discontinuously as a
    # shoe turns, which can stop refinement at a false boundary. The minimum
    # distance over the fixed real edge set is continuous. Source foreground
    # exclusions separately penalise material protruding beyond visible sides.
    if not len(pairs):raise ValueError('No real boot boundary edges')
    start=pixels[pairs[:,0]];edge=pixels[pairs[:,1]]-start
    fraction=np.clip(np.einsum('qec,ec->qe',target[:,None]-start,edge)
        /np.maximum(np.sum(edge*edge,axis=1),1.e-12),0.,1.)
    projected=start[None]+fraction[...,None]*edge[None]
    error=np.sum((target[:,None]-projected)**2,axis=-1);selected=error.argmin(1)
    return pairs[selected],fraction[np.arange(len(target)),selected],error[np.arange(len(target)),selected]


class VisibleSkateUnion:
    name='visible_ambiguous_skate_source'

    def __init__(self,problem,specification,frames):
        from experiments.skate_video_v1.body_skate_fit import Assembly
        self.problem=problem
        exact=bool(specification.get('exact_template',False))
        self.assembly=Assembly(problem.binding,exact_template=exact)
        self.feet=tuple(specification.get('feet',(0,1)))
        if not self.feet or len(set(self.feet))!=len(self.feet) or any(f not in (0,1) for f in self.feet):
            raise ValueError('Source arcs need distinct allowed physical feet')
        self.material_faces=None
        self.material_batches=None
        path=Path(specification['path'])
        if digest(path)!=specification['sha256']:raise ValueError('Source arcs changed')
        if specification.get('reviewed_open_arcs',False):
            document=json.loads(path.read_text(encoding='utf-8'))
            rows=reviewed_arc_rows(document,problem.reference.p['pts'],frames)
            self.batches=[];self.material_batches=[]
            template=problem.binding.packet()['template'];stop=template['part_ranges']['steel_runner'][1]
            lower_start=self.assembly.upper_count
            other_start=len(self.assembly.vertices)-(len(template['vertices'])-stop)
            blade=np.zeros(len(self.assembly.vertices),bool);blade[lower_start:other_start]=True
            for name,(lo,hi) in template['part_ranges'].items():
                if name not in ('steel_runner','outsole','heel_block'):
                    blade[lo-stop+other_start:hi-stop+other_start]=True
            self.material_edges=[(~blade[self.assembly.edges]).all(1),blade[self.assembly.edges].all(1)]
            if exact:
                upper=np.arange(len(self.assembly.vertices))<lower_start
                heel=np.zeros_like(upper);blade=np.zeros_like(upper)
                for name,(lo,hi) in template['part_ranges'].items():
                    if name=='heel_block':heel[lower_start+lo:lower_start+hi]=True
                    elif name!='outsole':blade[lower_start+lo:lower_start+hi]=True
                # Material boundaries remain meaningful inside the whole-shoe
                # silhouette: a visible gray heel cannot match a white upper,
                # and an observed blade cannot disappear behind that upper.
                material=(upper,blade,heel)
                self.material_edges=[m[self.assembly.edges].all(1) for m in material]
                self.material_faces=[self.assembly.faces[m[self.assembly.faces].all(1)] for m in material]
            elif any(np.any(r[4]==2) for r in rows):
                raise ValueError('Heel material arcs require the exact rendered template')
            for first in range(0,len(rows),32):
                group=rows[first:first+32];ids=np.array([r[0] for r in group],int)
                width=max(len(r[1]) for r in group)
                target=np.zeros((len(group),width,2));radius=np.ones((len(group),width));scale=np.zeros_like(radius)
                material=np.zeros((len(group),width),int);association=np.full((len(group),width),-1,int)
                for i,(_,a,b,c,m,g) in enumerate(group):
                    target[i,:len(a)]=a;radius[i,:len(a)]=b;scale[i,:len(a)]=c
                    material[i,:len(a)]=m;association[i,:len(a)]=g
                times=exposure_times(problem,ids)
                self.batches.append((ids,times,problem.reference.at(times),target,radius,scale))
                self.material_batches.append((material,association))
            self.name='reviewed_visible_skate_arcs'
            self.matches=None;self.associated_parameters=None;self.refresh_each_state=True
            self.provenance=dict(source_path=str(path),source_sha256=specification['sha256'],
                native_exposures=len(rows),source_labels_are_anatomically_unassigned=True,
                matching='observed open source arcs to either actual boot; unknown arcs omitted',
                points_on_one_arc_are_not_independent=True,observed_boot_and_blade_materials_retained=True,
                exact_rendered_template=exact,allowed_physical_feet=list(self.feet),
                exact_material_matching='nearest projected real material edge without raster visibility switching' if exact else None,
                complete_self_occlusion_claim=False,human_confirmed=False)
            return
        self.packet=load(path);q=self.packet
        np.testing.assert_array_equal(q['source_pts'],problem.reference.p['pts'])
        possible=ambiguous_shoe_frames(q)&((q['role']==1)&q['visible']).any(1)
        selected=np.asarray(frames);selected=selected[possible[selected]];self.batches=[]
        for first in range(0,len(selected),32):
            ids=selected[first:first+32];targets=np.zeros((len(ids),80,2));radius=np.ones((len(ids),80));scale=np.zeros_like(radius)
            for i,frame in enumerate(ids):
                for source_foot in (0,1):
                    if q['role'][frame,source_foot]!=1 or not q['visible'][frame,source_foot]:continue
                    valid=np.flatnonzero(q['boundary_valid'][frame,source_foot])
                    if len(valid)<8:continue
                    pick=valid[np.rint(np.linspace(0,len(valid)-1,40)).astype(int)];sl=slice(source_foot*40,(source_foot+1)*40)
                    targets[i,sl]=q['region'][frame,source_foot,pick]
                    radius[i,sl]=q['sigma_px'][frame,source_foot]
                    scale[i,sl]=np.sqrt(q['measure_s'][frame,source_foot]/40.)
            times=exposure_times(problem,ids)
            self.batches.append((ids,times,problem.reference.at(times),targets,radius,scale))
        self.matches=None;self.associated_parameters=None;self.refresh_each_state=True
        self.provenance=dict(source_path=str(path),source_sha256=specification['sha256'],
            native_exposures=len(selected),source_labels_are_anatomically_unassigned=True,
            matching='qualified visible source arcs to either actual boot; no hidden outline completion')

    def associate(self,x):
        p=self.problem;matches=[]
        for batch_index,(frames,times,base,target,radius,scale) in enumerate(self.batches):
            d=p.body.state.increments(x,times);s=fk_state(p.reference,base,d);both=[]
            for foot in (0,1):
                nodes=self.assembly.points(s,foot).reshape(len(frames),5,-1,3)
                uv=moving_projection(p.camera,nodes,frames)['uv']
                if self.material_batches is None:
                    rows=[nearest_visible_edges(self.assembly,a,b) for a,b in zip(uv,target)]
                else:
                    material,groups=self.material_batches[batch_index];rows=[]
                    for a,b,kinds in zip(uv,target,material):
                        pairs=np.empty((len(b),2),int);fraction=np.empty(len(b));error=np.empty(len(b))
                        for kind in np.unique(kinds):
                            selected=kinds==kind
                            pairs[selected],fraction[selected],error[selected]=nearest_visible_edges(
                                self.assembly,a,b[selected],self.material_edges[kind],
                                None if self.material_faces is None else self.material_faces[kind])
                        rows.append((pairs,fraction,error))
                both.append(tuple(np.stack(v) for v in zip(*rows)))
            chosen=both[1][2]<both[0][2]
            if self.material_batches is not None:
                # An observed arc belongs to one boot even when its endpoints
                # are individually closer to different overlapping feet.
                for i,labels in enumerate(groups):
                    for group in np.unique(labels[labels>=0]):
                        take=labels==group;weight=(scale[i,take]/radius[i,take])**2
                        chosen[i,take]=np.sum(both[1][2][i,take]*weight)<np.sum(both[0][2][i,take]*weight)
            if len(self.feet)==1:chosen[:]=bool(self.feet[0])
            matches.append((both[0][:2],both[1][:2],chosen))
        self.matches=matches;self.associated_parameters=np.array(x,copy=True)
        self.association_signature=signature(*(v for m in matches for v in (*m[0],*m[1],m[2])))

    def evaluate(self,x,jacobian=False):
        if self.matches is None or (self.refresh_each_state and not np.array_equal(x,self.associated_parameters)):
            self.associate(x)
        p=self.problem;values=[];matrices=[];distances=[]
        for batch,match in zip(self.batches,self.matches):
            frames,times,base,target,radius,scale=batch;d=p.body.state.increments(x,times);s=fk_state(p.reference,base,d)
            residual=np.zeros((*radius.shape,2));matrix=None;distance=np.zeros_like(radius)
            for foot in (0,1):
                selector=match[foot];active=(match[2]==bool(foot))&(scale>0)
                def points(state):
                    return self.assembly.points(state,foot,selector,(state['R'],state['O'])).reshape(len(frames),5,-1,3)
                pr=moving_projection(p.camera,points(s),frames,jacobian=jacobian)
                error=pr['uv']-target;z=error/radius[...,None]
                u=np.sqrt(1.+np.sum(z*z,axis=-1)/4.);k=np.sqrt(2./(u+1.));weight=scale*active
                residual+=weight[...,None]*k[...,None]*z;distance+=np.linalg.norm(error,axis=-1)*active
                if jacobian:
                    B,S=native_point_derivative(p,base,d,points,foot,pr['pixel_jac_nodes'])
                    dk=-k/(16.*u*(u+1.));D=k[...,None,None]*np.eye(2)+2.*dk[...,None,None]*z[...,:,None]*z[...,None,:]
                    D*=(weight/radius)[...,None,None]
                    body=np.einsum('nkqp,nkpad->nkqad',D,B);shoe=np.einsum('nkqp,nkpad->nkqad',D,S)
                    J=chain(p,d,times,body.reshape(len(frames),-1,5,69),shoe.reshape(len(frames),-1,5,6))
                    matrix=J if matrix is None else matrix+J
            values.append(residual.ravel());distances.extend(distance[scale>0].tolist())
            if jacobian:matrices.append(matrix)
        residual=np.concatenate(values);J=vstack(matrices,format='csr') if jacobian else None
        return dict(residual=residual,jac=J,parts={self.name:float(residual@residual/2)},
            metrics={self.name:dict(visible_source_samples=len(distances),median_px=float(np.median(distances)),p95_px=float(np.percentile(distances,95)))},
            weighted_blocks=[block(self.name,residual,J,self.provenance,self.association_signature)])
