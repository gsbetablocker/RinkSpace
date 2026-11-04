"""Exact selected-skin replay for one local rotation perturbation.

The fitting surface remains the stored FP64 promotion of FullSurface's shape,
pose directions and all 55 skinning weights.  One pose linearization prepares
its complete pose blend and skinning blends once.  A +/- local column
then changes only nine pose-feature entries and one rigid FK subtree.  This
is a finite rotation identity, not a first-order approximation; nonsmooth foot
and collar consumers still evaluate their complete original residual twice.

Only O(N * selected_vertices) base data are retained.  There is no dense skin
Jacobian, and no per-joint N-by-vertices-by-9 cache.  Per-pair scratch tensors
are bounded by chunk_size; outputs necessarily contain both selected skins.
"""
from __future__ import annotations

from numbers import Integral
import sys

import numpy as np

from .rebuild_body_geometry import CompactBody
from .hand_state import validate_hand


def supports_surface(surface):
    """Only the audited FullSurface sampling method uses this replay.

    Resolve an already imported viewer module rather than importing media
    dependencies merely to ask a capability question.  A wrapper can change
    numeric semantics even if functools.wraps preserves its name.  Custom or
    wrapped sampling and different model classes keep their original path.
    """
    module=sys.modules.get('rinkspace.viewer3d.prepare')
    cls=getattr(module,'FullSurface',None) if module is not None else None
    if cls is None or type(surface) is not cls or type(getattr(surface,'body',None)) is not CompactBody:
        return False
    method=getattr(surface,'sample_vertices',None)
    function=getattr(method,'__func__',None)
    original=cls.sample_vertices
    return (function is original and not hasattr(original,'__wrapped__')
            and getattr(original,'__module__','')=='rinkspace.viewer3d.prepare'
            and getattr(original,'__qualname__','')=='FullSurface.sample_vertices')


def _readonly(value):
    value=np.array(value,dtype=np.float64,copy=True,order='C')
    value.setflags(write=False)
    return value


class SelectedSurfaceLinearization:
    """Immutable selected surface at one exact native22/frame22 state.

    ``pair(column, plus_state, minus_state)`` accepts the existing 69-state
    column number (6..68) and two dictionaries with ``native`` and ``frames``.
    They must be exact single-joint +/- FK replays of this baseline using the
    same skeleton, shape and length scale.  World-to-local roundoff and the
    repeated reconstruction of unchanged local rotations can differ from the
    full replay at floating-point precision; no FD or model term is omitted.

    Construct a new object for every changed body state/model/selection.  Its
    arrays are read-only, and each pair call owns all output/scratch arrays,
    so independent column workers can share this object safely.
    """
    def __init__(self,surface,native,frames,vertex_ids,*,hand_local_R=None,chunk_size=128):
        if not isinstance(chunk_size,Integral) or isinstance(chunk_size,bool) or chunk_size<1:
            raise ValueError('Surface chunk_size must be a positive integer')
        body=surface.body
        self.scale=float(body.scale)
        if not np.isfinite(self.scale) or self.scale<=0:
            raise ValueError('Expected a positive finite body scale')
        native=np.asarray(native,dtype=np.float64);frames=np.asarray(frames,dtype=np.float64)
        if native.ndim!=3 or native.shape[1:]!=(22,3) or frames.shape!=(len(native),22,3,3):
            raise ValueError('Expected native22 positions and frames at the same samples')
        if not np.isfinite(native).all() or not np.isfinite(frames).all():
            raise ValueError('Nonfinite selected-surface pose')
        parents=np.asarray(body.parents,dtype=int)
        if parents.shape!=(55,) or any(parents[j]<0 or parents[j]>=j for j in range(1,55)):
            raise ValueError('Expected the topologically ordered 55-joint model')
        shaped=np.asarray(surface.shaped);posedirs=np.asarray(surface.posedirs);weights=np.asarray(surface.weights)
        if shaped.ndim!=2 or shaped.shape[1]!=3 or posedirs.shape!=(486,shaped.size) or weights.shape!=(len(shaped),55):
            raise ValueError('Unexpected FullSurface model tensor layout')
        ids=None if vertex_ids is None else np.asarray(vertex_ids)
        if ids is not None and (ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(shaped))):
            raise ValueError('Invalid selected surface vertex ids')
        self.n=len(native);self.chunk_size=int(chunk_size)
        self.hand_local_R=None if hand_local_R is None else _readonly(validate_hand(hand_local_R,self.n))
        self._parents=parents.copy();self._J=_readonly(body.J);self._default=_readonly(body.default)
        if self._J.shape!=(55,3) or self._default.shape!=(33,3,3):
            raise ValueError('Unexpected fixed model joint/default-rotation layout')
        self._frames=_readonly(frames)
        position=np.empty((self.n,55,3));rotation=np.empty((self.n,55,3,3))
        position[:,:22]=native/self.scale+self._J[0];rotation[:,:22]=frames
        local=np.empty((self.n,21,3,3))
        for j in range(1,22):local[:,j-1]=frames[:,parents[j]].swapaxes(-1,-2)@frames[:,j]
        tail=np.broadcast_to(self._default,(self.n,33,3,3)).copy()
        if self.hand_local_R is not None:tail[:,3:]=self.hand_local_R.reshape(self.n,30,3,3)
        for j in range(22,55):
            parent=int(parents[j]);rotation[:,j]=rotation[:,parent]@tail[:,j-22]
            position[:,j]=position[:,parent]+np.einsum('nij,j->ni',rotation[:,parent],self._J[j]-self._J[parent])
        self._local=_readonly(local)
        self._position=_readonly(position[:,:22])
        self._rotation=_readonly(rotation)
        self._translation=_readonly(position-np.einsum('njab,jb->nja',rotation,self._J))
        descendants=np.eye(55,dtype=bool)
        for child in range(1,55):descendants[:,child]|=descendants[:,parents[child]]
        self._indices=tuple(np.flatnonzero(descendants[j]) for j in range(22))

        # Reuse the completed immutable FullSurface selection cache, when its
        # exact original key matches.  Otherwise prepare the same stored
        # tensors locally; this helper never changes that surface's cache.
        ids_key=None if ids is None else (ids.dtype.str,ids.tobytes())
        key=(id(body),body.scale,id(surface.shaped),id(surface.posedirs),id(surface.weights),ids_key)
        entry=getattr(surface,'_sample_double_cache',None)
        if entry is not None and entry[0]==key:
            shaped,all_posedirs,weights_T=entry[1]
        else:
            selected=slice(None) if ids is None else ids
            shaped=np.asarray(shaped[selected],dtype=np.float64)
            all_posedirs=np.asarray(posedirs.reshape(486,-1,3)[:,selected],dtype=np.float64).reshape(486,-1)
            weights_T=np.ascontiguousarray(weights[selected].astype(np.float64).T)
            all_posedirs=np.ascontiguousarray(all_posedirs)
            for value in (shaped,all_posedirs,weights_T):value.setflags(write=False)
        # FullSurface caches raw shape and all 486 directions, with no fixed
        # hand contribution folded into the shape.  The cache is independent
        # of both default rotations and this state's sampled actual hands.
        self._posedirs=all_posedirs;self._weights=weights_T
        body_posedirs=all_posedirs[:189]
        if self.hand_local_R is None:
            tail_feature=(self._default-np.eye(3)).reshape(297)
            shaped=shaped+(tail_feature@all_posedirs[189:]).reshape(-1,3)
        self.vertices=len(shaped)
        # Zero skinning weights are exact model sparsity, not a threshold on
        # pose-blend influence.  Non-leg pose directions still update every
        # selected vertex through the full baseline blend_R below.
        self._skin_indices=[];self._skin_weights=[];self._weight_sums=[]
        for ids in self._indices:
            selected=ids[np.any(weights_T[ids]!=0,axis=1)]
            weight=np.array(weights_T[selected],copy=True,order='C')
            total=weight.sum(axis=0)
            for value in (selected,weight,total):value.setflags(write=False)
            self._skin_indices.append(selected);self._skin_weights.append(weight);self._weight_sums.append(total)
        self._skin_indices=tuple(self._skin_indices);self._skin_weights=tuple(self._skin_weights)
        self._weight_sums=tuple(self._weight_sums)
        self._posed=np.empty((self.n,self.vertices,3))
        self._blend_R=np.empty((self.n,self.vertices,3,3))
        self.samples=np.empty((self.n,self.vertices,3))
        for start in range(0,self.n,self.chunk_size):
            stop=min(start+self.chunk_size,self.n);n=stop-start
            if self.hand_local_R is None:
                feature=(local[start:stop]-np.eye(3)).reshape(n,189)
                posed=shaped+(feature@body_posedirs).reshape(n,self.vertices,3)
            else:
                feature=np.concatenate((local[start:stop],tail[start:stop]),axis=1)
                feature=(feature-np.eye(3)).reshape(n,486)
                posed=shaped+(feature@all_posedirs).reshape(n,self.vertices,3)
            blend_R=(rotation[start:stop].reshape(n,55,9).transpose(0,2,1)@weights_T).transpose(0,2,1).reshape(n,self.vertices,3,3)
            blend_t=(self._translation[start:stop].transpose(0,2,1)@weights_T).transpose(0,2,1)
            self._posed[start:stop]=posed;self._blend_R[start:stop]=blend_R
            self.samples[start:stop]=(np.einsum('nvij,nvj->nvi',blend_R,posed)+blend_t-self._J[0])*self.scale
        for value in (self._posed,self._blend_R,self.samples,self._parents):value.setflags(write=False)

    def record(self):
        """Allocation estimates, not a measured peak-process memory claim."""
        base=sum(value.nbytes for value in (self._frames,self._local,self._position,self._rotation,
                 self._translation,self._posed,self._blend_R,self.samples))
        return dict(samples=self.n,selected_vertices=self.vertices,chunk_size=self.chunk_size,
                    base_pose_bytes=base,selection_model_bytes=self._posedirs.nbytes+self._weights.nbytes+
                        sum(a.nbytes for a in self._skin_weights+self._weight_sums),
                    pair_output_bytes=2*self.n*self.vertices*3*8,
                    note='Per-pair temporary contractions are chunked; concurrent workers own separate outputs.')

    def pair(self,column,plus_state,minus_state):
        if not isinstance(column,Integral) or isinstance(column,bool) or not 6<=column<69:
            raise ValueError('Expected a local body column in 6..68')
        joint=(int(column)-6)//3+1;parent=int(self._parents[joint]);ids=self._skin_indices[joint]
        weights=self._skin_weights[joint];total_weight=self._weight_sums[joint]
        posedirs=self._posedirs[(joint-1)*9:joint*9]
        states=(plus_state,minus_state);changes=[];turns=[]
        for state in states:
            native=np.asarray(state['native']);frames=np.asarray(state['frames'])
            if native.shape!=(self.n,22,3) or frames.shape!=(self.n,22,3,3):
                raise ValueError('Surface pair sample/kinematic layout changed')
            if not np.isfinite(native).all() or not np.isfinite(frames).all():
                raise ValueError('Nonfinite surface perturbation')
            local=frames[:,parent].swapaxes(-1,-2)@frames[:,joint]
            changes.append(local-self._local[:,joint-1])
            turns.append(frames[:,joint]@self._frames[:,joint].swapaxes(-1,-2)-np.eye(3))
        output=(np.empty_like(self.samples),np.empty_like(self.samples))
        for start in range(0,self.n,self.chunk_size):
            stop=min(start+self.chunk_size,self.n);n=stop-start
            if len(ids):
                selected=self._rotation[start:stop,ids]
                subtree_R=(selected.reshape(n,len(ids),9).transpose(0,2,1)@weights).transpose(0,2,1).reshape(n,self.vertices,3,3)
                subtree_t=(self._translation[start:stop,ids].transpose(0,2,1)@weights).transpose(0,2,1)
                relative=(np.einsum('nvij,nvj->nvi',subtree_R,self._posed[start:stop])+subtree_t
                          -total_weight[None,:,None]*self._position[start:stop,joint,None])
            for k in range(2):
                dposed=(changes[k][start:stop].reshape(n,9)@posedirs).reshape(n,self.vertices,3)
                pose_effect=np.einsum('nvij,nvj->nvi',self._blend_R[start:stop],dposed)
                if len(ids):
                    shifted=relative+np.einsum('nvij,nvj->nvi',subtree_R,dposed)
                    moved=np.einsum('nij,nvj->nvi',turns[k][start:stop],shifted)
                    pose_effect+=moved
                output[k][start:stop]=self.samples[start:stop]+pose_effect*self.scale
        return output
