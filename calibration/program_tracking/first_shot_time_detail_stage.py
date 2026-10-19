"""Centered, nested native-time details before the unchanged physical base.

The coarse B8 coefficients and every saved canonical coordinate survive the
transition. Fine coefficients are differences from saved historical values.
A caller explicitly selects details; representation rank is not source evidence.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr
from scipy.sparse import csr_matrix, eye

from .first_shot_derived_venue_stage import DerivedVenueState, _reference
from .first_shot_native_time_stage import NativeTimeVenueStage, _same, _sha

PLAN_SCHEMA = 'rinkspace.centered-nested-native-time-detail-plan.v1'
STAGE_SCHEMA = 'rinkspace.centered-nested-native-time-detail-stage.v1'
PREFIX = 'time_detail_'
NATIVE_CAMERA_PARTIAL_BASIS = 'physical_native_camera'


def _equal(a, b, label):
    if not _same(np.asarray(a), np.asarray(b)):
        raise ValueError('Native-time detail identity differs: ' + label)


def _verified(ref):
    expected = {k:ref[k] for k in ('path', 'sha256')}
    if _reference(ref['path']) != expected:
        raise ValueError('Frozen native-time detail reference changed: ' + ref['path'])
    return expected


def _integer_indices(values, limit, label):
    result = np.asarray(values)
    if result.ndim == 1 and result.size == 0:
        result = result.astype(np.int64)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or len(np.unique(result)) != len(result)
            or np.any(result < 0) or np.any(result >= limit)):
        raise ValueError('Distinct in-range integer ' + label + ' required')
    return result.astype(np.int64)


def _merge_live_base_aux(aux, base, allowed_rebuilt_aux, live_model_reference):
    """Preserve all parent bytes; explicitly archive one rebuilt live model ID."""
    archive_names = {'print_H8_QP_domain_identity_json': 'live_print_H8_QP_'}
    if isinstance(allowed_rebuilt_aux, str):
        raise ValueError('Explicit rebuilt auxiliary key sequence required')
    allowed = tuple(allowed_rebuilt_aux)
    if len(set(allowed)) != len(allowed) or any(key not in archive_names for key in allowed):
        raise ValueError('Only the declared print H8 domain identity may be rebuilt')
    arrays = {key:np.asarray(value).copy() for key,value in base.checkpoint_arrays().items()}
    references = []
    live = None
    if allowed:
        if live_model_reference is None:
            raise ValueError('Rebuilt print auxiliary requires the explicit live print configuration')
        config_ref = _verified(live_model_reference)
        config = json.loads(Path(config_ref['path']).read_text(encoding='utf-8-sig'))
        if config.get('schema') != 'rinkspace.explicit-disjoint-print-successor.v1':
            raise ValueError('Explicit disjoint print configuration required for rebuilt auxiliary')
        runtime = _verified(config['runtime_producer'])
        if runtime != _reference(Path(__file__).with_name('first_shot_disjoint_print.py')):
            raise ValueError('Rebuilt print auxiliary must use the public disjoint print producer')
        material = _verified(config['material_override'])
        installed = getattr(base.programme, 'disjoint_print_receipt', {})
        if (installed.get('configuration') != config_ref or installed.get('material_override') != material
                or installed.get('producer') != runtime):
            raise ValueError('Installed live print model does not match the rebuilt auxiliary configuration')
        references = [config_ref, runtime, material]
        for ref in config.get('base_producers', []):
            references.append(_verified(ref))
        domain = json.loads(str(arrays.get('print_H8_QP_domain_identity_json', 'null')))
        current_identity = base.checkpoint_identity().get('print_H8_QP_domain')
        current_domain = getattr(base, '_print_H8_domain', None)
        if (not isinstance(domain, dict) or domain != current_identity or current_domain is None
                or domain.get('domain') != current_domain.identity):
            raise ValueError('Rebuilt print auxiliary is not the current live print H8 domain')
        live = dict(schema='rinkspace.explicit-live-base-model-aux-archive.v1',
            model_configuration=config_ref, runtime_producer=runtime, material_override=material,
            allowed_rebuilt_aux=list(allowed), parent_aux_values_preserved_byte_exact=True,
            live_domain_callbacks_remain_on_original_base=True, entries={})
    elif live_model_reference is not None:
        raise ValueError('Live print configuration requires an explicit rebuilt auxiliary declaration')
    for key,value in arrays.items():
        if key in allowed:
            if key not in aux:
                raise ValueError('Rebuilt auxiliary must preserve an existing parent key: '+key)
            if value.dtype.hasobject:
                raise ValueError('Safe rebuilt live auxiliary cannot contain objects')
            digest = _sha(value)
            archive_key = archive_names[key]+digest[:16]
            if archive_key in aux:
                _equal(aux[archive_key],value,'restored current live model auxiliary '+key)
            else:
                aux[archive_key] = value.copy()
            live['entries'][key] = dict(live_aux_key=archive_key, live_sha256=digest,
                live_dtype=value.dtype.str, live_shape=list(value.shape),
                preserved_parent_sha256=_sha(aux[key]), preserved_parent_dtype=aux[key].dtype.str,
                preserved_parent_shape=list(aux[key].shape))
        elif key in aux:
            _equal(aux[key],value,'reconstructed native-base aux '+key)
        else:
            aux[key] = value.copy()
    if live is not None:
        if set(live['entries']) != set(allowed):
            raise ValueError('All declared rebuilt auxiliaries must exist on the live base')
        metadata = np.asarray(json.dumps(live,sort_keys=True))
        metadata_key = 'live_base_aux_'+_sha(metadata)[:16]+'_json'
        if metadata_key in aux:
            _equal(aux[metadata_key],metadata,'restored live model auxiliary archive')
        else:
            aux[metadata_key] = metadata
        live['metadata_aux_key'] = metadata_key
    return live,references


class CenteredNativeTimeDetailPlan:
    """A stored affine refinement, independent of the final parent q value."""
    @classmethod
    def from_parent(cls, parent):
        aux = parent['aux']
        if PREFIX + 'plan_json' in aux:
            return cls._restore(aux)
        identity = json.loads(str(aux['native_time_identity_json']))
        recipe_ref = _verified(identity['recipe'])
        recipe = json.loads(Path(recipe_ref['path']).read_text(encoding='utf-8-sig'))
        return cls.build(parent['q'], aux, recipe['knots_pts'])

    @classmethod
    def _coarse(cls, aux):
        value = cls()
        value.B8 = csr_matrix((aux['native_time_basis_data'], aux['native_time_basis_indices'],
            aux['native_time_basis_indptr']), shape=tuple(aux['native_time_basis_shape']))
        value.pts = np.asarray(aux['native_time_pts']).copy()
        value.native = np.asarray(aux['native_time_native_columns']).copy()
        value.slots8 = np.asarray(aux['native_time_control_slots']).copy()
        if (value.B8.shape != (1248, 159) or value.B8.dtype != np.float64
                or value.pts.dtype != np.int64 or value.pts.shape != (1248,)
                or np.any(np.diff(value.pts) <= 0) or value.native.shape != (1248, 3)
                or value.slots8.shape != (3, 159) or not np.isfinite(value.B8.data).all()):
            raise ValueError('Complete original B8/native-PTS/typed-slot inventory required')
        if len(np.unique(value.native)) != 3744:
            raise ValueError('Native PTZ columns overlap')
        for j in range(3):
            if not np.isin(value.slots8[j], value.native[:, j]).all():
                raise ValueError('Coarse time-control type changed')
        return value

    @classmethod
    def build(cls, q, aux, knots8):
        value = cls._coarse(aux)
        q = np.asarray(q)
        if q.dtype != np.float64 or q.shape != (7700,) or not np.isfinite(q).all():
            raise ValueError('Complete finite canonical parent required')
        value.knots8 = np.asarray(knots8, np.float64).copy()
        coarse = csr_matrix(BSpline.design_matrix(value.pts.astype(float), value.knots8, 3))
        if coarse.shape != value.B8.shape or np.max(abs((coarse-value.B8).data), initial=0.) > 1e-14:
            raise ValueError('Literal original coarse knots and saved B8 differ')
        insertion = np.setdiff1d(value.pts[np.arange(4, 1247, 4)], value.knots8)
        spline = BSpline(value.knots8, np.eye(159), 3)
        for stamp in insertion:
            spline = spline.insert_knot(float(stamp))
        value.knots4, refinement = spline.t.copy(), spline.c.copy()
        if refinement.shape != (315, 159):
            raise ValueError('Expected strict B8 to B4 knot insertion')
        fine = csr_matrix(BSpline.design_matrix(value.pts.astype(float), value.knots4, 3))
        error = float(np.max(abs(fine@refinement-value.B8.toarray())))
        if error > 1e-14:
            raise ValueError('Fine basis is not an exact nested refinement')
        # Select a compact coefficient complement, not a data-identifiable set.
        _, _, pivot = qr(refinement.T, pivoting=True, mode='economic')
        value.K = np.setdiff1d(np.arange(315), pivot[:159]).astype(np.int64)
        value.D = fine[:, value.K].tocsr()
        greville = np.asarray([np.mean(value.knots4[k+1:k+4]) for k in value.K])
        slots = []
        for j in range(3):
            available = set(map(int, value.native[:, j])) - set(map(int, value.slots8[j]))
            clock = {int(col):float(stamp) for col,stamp in zip(value.native[:, j], value.pts, strict=True)}
            chosen = []
            for stamp in greville:
                column = min(available, key=lambda col:(abs(clock[col]-stamp), col))
                chosen.append(column); available.remove(column)
            slots.append(chosen)
        value.slots = np.asarray(slots, np.int64)
        value.centres = q[value.slots].copy()
        value.metadata = dict(schema=PLAN_SCHEMA, parent_anchor_q_sha256=_sha(q),
            coarse_shape=[1248,159], fine_shape=[1248,315], candidate_details=468,
            refinement_max_abs_error=error, native_pts_sha256=_sha(value.pts),
            basis_is_evaluated_at_native_PTS_only=True,
            representation_complement_is_not_source_identifiability=True)
        value._validate()
        return value

    @classmethod
    def _restore(cls, aux):
        value = cls._coarse(aux)
        value.metadata = json.loads(str(aux[PREFIX+'plan_json']))
        if value.metadata.get('schema') != PLAN_SCHEMA:
            raise ValueError('Explicit centered time-detail restore metadata required')
        for name in ('knots8', 'knots4', 'K', 'slots', 'centres'):
            setattr(value, name, np.asarray(aux[PREFIX+name]).copy())
        value.D = csr_matrix((aux[PREFIX+'basis_data'], aux[PREFIX+'basis_indices'],
            aux[PREFIX+'basis_indptr']), shape=tuple(aux[PREFIX+'basis_shape']))
        value._validate()
        return value

    def _validate(self):
        if (self.K.shape != (156,) or self.slots.shape != (3,156) or self.centres.shape != (3,156)
                or self.D.shape != (1248,156) or not np.isfinite(self.centres).all()
                or not np.isfinite(self.D.data).all() or len(np.unique(self.slots)) != 468
                or np.isin(self.slots, self.slots8).any()
                or self.metadata['native_pts_sha256'] != _sha(self.pts)):
            raise ValueError('Complete finite disjoint centered detail plan required')
        for j in range(3):
            if not np.isin(self.slots[j], self.native[:,j]).all():
                raise ValueError('New detail slot does not retain its native PTZ type')
        fine = csr_matrix(BSpline.design_matrix(self.pts.astype(float), self.knots4, 3))[:,self.K]
        difference = fine-self.D
        if difference.nnz and np.max(abs(difference.data)) > 1e-14:
            raise ValueError('Restored detail basis differs from its saved compact spline')

    def arrays(self):
        result = {PREFIX+name:np.asarray(getattr(self,name)).copy()
                  for name in ('knots8','knots4','K','slots','centres')}
        result.update({PREFIX+'basis_data':self.D.data.copy(),
            PREFIX+'basis_indices':self.D.indices.astype(np.int64),
            PREFIX+'basis_indptr':self.D.indptr.astype(np.int64),
            PREFIX+'basis_shape':np.asarray(self.D.shape,np.int64),
            PREFIX+'plan_json':np.asarray(json.dumps(self.metadata,sort_keys=True))})
        return result

    def expand(self, q):
        value = np.asarray(q).copy()
        for j in range(3):
            coarse = self.B8 @ value[self.slots8[j]]
            delta = value[self.slots[j]]-self.centres[j]
            # Preserve the original multiplication, including final bits, at zero.
            value[self.native[:,j]] = coarse if not np.any(delta) else coarse+self.D@delta
        return value

    def chain(self, active_details):
        selected = np.zeros(468,bool); selected[active_details] = True
        tangent = eye(7700,format='lil',dtype=np.float64)
        for j in range(3):
            for i,column in enumerate(self.native[:,j]):
                a,b = self.B8.indptr[i:i+2]; c,d = self.D.indptr[i:i+2]
                ids = self.D.indices[c:d]; keep = selected[j*156+ids]
                tangent.rows[column] = np.r_[self.slots8[j,self.B8.indices[a:b]],self.slots[j,ids[keep]]].tolist()
                tangent.data[column] = np.r_[self.B8.data[a:b],self.D.data[c:d][keep]].tolist()
        return tangent.tocsr()

    def camera_detail_chain(self, parameter_count, selected=None):
        selected = np.arange(468,dtype=np.int64) if selected is None else selected
        rows=[]; cols=[]; data=[]
        for local,detail in enumerate(selected):
            j,k = divmod(int(detail),156)
            column = self.D[:,[k]].tocoo()
            rows.extend(self.native[column.row,j].tolist())
            cols.extend([local]*len(column.data)); data.extend(column.data.tolist())
        return csr_matrix((data,(rows,cols)),shape=(parameter_count,len(selected)))


class NativeTimeDetailVenueStage(NativeTimeVenueStage):
    """Use the original native base, never a Jacobian compressed through B8.

    ``parent`` is the complete load_parent-style q/aux/physical_q/expanded_q/
    identity/checkpoint/damping/fraction record. ``active_details=None`` restores
    the saved selection, or selects no new details on a B8 parent. A caller must
    supply any newly enabled detail indices explicitly after its source audit.
    A rebuilt print domain requires ``allowed_rebuilt_aux`` and its authenticated
    ``live_model_reference``; historical same-name auxiliary bytes stay intact.
    """
    def __init__(self,native_base_owner,*,parent,entry_reference,transition_reference,
                 inherited_aux=None,plan=None,active_details=None,
                 allowed_rebuilt_aux=(),live_model_reference=None):
        base=native_base_owner
        if (base.parameter_count!=7700 or base.programme_count!=7687
                or len(base.shared)!=172 or len(base.fixed)!=3784
                or getattr(base,'native_time_receipt',None) is not None):
            raise ValueError('Original complete uncompressed native base before the B8 chart required')
        self.base=base; self.parameter_count=7700; self.programme_count=7687
        self.programme,self.physical_stage,self.factor=base.programme,base.physical_stage,base.factor
        self.start=np.asarray(parent['q']).copy()
        if self.start.dtype!=np.float64 or self.start.shape!=(7700,) or not np.isfinite(self.start).all():
            raise ValueError('Complete finite accepted detail parent required')
        self._parent=parent
        _verified(parent['checkpoint'])
        if parent.get('q_sha256',_sha(self.start))!=_sha(self.start):
            raise ValueError('Accepted parent q hash differs from supplied canonical state')
        self.plan=CenteredNativeTimeDetailPlan.from_parent(parent) if plan is None else plan
        if not isinstance(self.plan,CenteredNativeTimeDetailPlan):
            raise TypeError('Authenticated centered native-time detail plan required')
        self.plan._validate()
        aux={key:np.asarray(value).copy() for key,value in parent['aux'].items()}
        for key,value in (inherited_aux or {}).items():
            if key in aux:_equal(aux[key],value,'inherited parent aux '+key)
            else:aux[key]=np.asarray(value).copy()
        self.live_base_model_aux,live_refs=_merge_live_base_aux(
            aux,base,allowed_rebuilt_aux,live_model_reference)
        restored=aux.get(PREFIX+'active_indices')
        selected=np.asarray(restored,np.int64) if active_details is None and restored is not None else (
            np.empty(0,np.int64) if active_details is None else active_details)
        self.active_details=_integer_indices(selected,468,'explicit active detail indices')
        if restored is not None:_equal(restored,self.active_details,'restored active detail selection')
        for key,value in self.plan.arrays().items():
            if key in aux:_equal(aux[key],value,'restored detail plan '+key)
            else:aux[key]=value
        aux.setdefault(PREFIX+'active_indices',self.active_details.copy())
        if any(value.dtype.hasobject for value in aux.values()):
            raise ValueError('Safe complete detail auxiliary arrays cannot contain objects')
        self._aux=aux
        _equal(np.asarray(base.programme.state.pts,np.int64),self.plan.pts,'actual original native PTS')
        actual_native=np.asarray([base.programme.state.view_columns(i+1)[[0,1,3]] for i in range(1248)],np.int64)
        _equal(actual_native,self.plan.native,'actual original native column map')
        self.native_columns=self.plan.native.copy(); self.control_slots={kind:self.plan.slots8[j].copy()
            for j,kind in enumerate(('pan','tilt','logf'))}
        self.shared=np.asarray(base.shared,int).copy()
        controls=np.r_[self.plan.slots8.ravel(),self.plan.slots.ravel()[self.active_details]]
        self.fixed_native=self.native_columns.ravel()[~np.isin(self.native_columns.ravel(),controls)]
        self.fixed=np.r_[np.asarray(base.fixed,int),self.fixed_native]
        self._fixed_bits=self.start[self.fixed].view(np.uint64).copy()
        self._native_mask=np.zeros(7700,bool);self._native_mask[self.native_columns]=True
        self._time_chain=self.plan.chain(self.active_details)
        self.scales,self.quadratic=base.scales,base.quadratic
        self.source_producers=deepcopy(base.source_producers)
        self.recovery_references=deepcopy(base.recovery_references)
        self._references=deepcopy(getattr(base,'_references',[]))
        refs=[_reference(__file__),deepcopy(entry_reference),deepcopy(transition_reference)]+live_refs
        for ref in refs:
            _verified(ref)
            for target in (self._references,self.source_producers,self.recovery_references):
                if ref not in target:target.append(deepcopy(ref))
        self.native_time_receipt=deepcopy(parent['identity']['native_time_stage'])
        self.detail_receipt=dict(schema=STAGE_SCHEMA,producers=refs,parent_checkpoint=deepcopy(parent['checkpoint']),
            parent_q_sha256=_sha(self.start),plan=deepcopy(self.plan.metadata),
            active_detail_indices=self.active_details.tolist(),active_detail_count=len(self.active_details),
            candidate_detail_count=468,all_parent_aux_preserved=True,parent_aux_count=len(parent['aux']),
            all_parent_numerical_search_state_preserved=True,initial_damping=parent['damping'],initial_fraction=parent['fraction'],
            coarse_B8_retained_as_history_and_active_subspace=True,derived_chart_evaluated_once_after_centered_time=True,
            old_B8_compressed_J_not_inverted=True,source_J_cache_invalidated=True,
            source_clock_and_actual_linear_native_row_interpolator_unchanged=True,
            detail_selection_is_explicit_not_a_data_identifiability_claim=True,
            live_base_model_aux=deepcopy(self.live_base_model_aux))
        self.state=DerivedVenueState(self);self._initial_complete_J_reuse=False
        self._check_producers()
        expanded,_=self.expanded_q_and_chain(self.start)
        physical,_=self.effective_q_and_chain(self.start)
        _equal(expanded,parent['expanded_q'],'complete parent expansion at detail transition')
        _equal(physical,parent['physical_q'],'complete parent physical state at detail transition')
        if not np.isfinite(parent['damping']) or parent['damping']<=0 or not 0<parent['fraction']<=1:
            raise ValueError('Complete finite accepted numerical search state required')

    def expanded_q_and_chain(self,q):
        self._check_producers();q=self._canonical_q(q)
        return self.plan.expand(q),self._time_chain.copy()

    def chart_identity(self):
        return dict(schema=STAGE_SCHEMA,centered_native_time=deepcopy(self.detail_receipt),
            existing_base_chart=self.base.chart_identity(),order='centered_native_detail_then_derived_exactly_once')

    def checkpoint_arrays(self):
        self._check_producers()
        return {key:value.copy() for key,value in self._aux.items()}

    def checkpoint_identity(self):
        value=deepcopy(self._parent['identity']);value.update(deepcopy(self.base.checkpoint_identity()))
        value.update(schema=STAGE_SCHEMA,native_time_stage=deepcopy(self.native_time_receipt),
            native_time_detail_stage=deepcopy(self.detail_receipt),source_producers=deepcopy(self.source_producers),
            fixed_parameter_indices=self.fixed.tolist(),global_shared_indices=self.shared.tolist(),
            shared_dense_count=len(self.shared),initial_exact_complete_current_state_full_J_reuse=False,
            live_base_model_aux=deepcopy(self.live_base_model_aux))
        return value

    def source_cache_policy(self):
        return dict(old_compressed_B8_J_invalid=True,new_columns_require_native_camera_partials=True,
            initial_complete_J_reuse=False,full_fresh_J_default=True,
            incremental_old_columns_only_after_exact_same_physical_q_and_unchanged_complete_source_shadow=True,
            topology_or_source_change_invalidates_all_old_source_J=True)

    def pullback_native_camera_partials(self,J,*,input_basis,q=None,selected=None):
        """Whitened physical camera partials only; B8-canonical input is rejected."""
        if input_basis!=NATIVE_CAMERA_PARTIAL_BASIS:
            raise ValueError('Original physical native camera partials required; compressed B8 J cannot be inverted')
        J=csr_matrix(J)
        if J.shape[1] not in (self.programme_count,self.parameter_count):
            raise ValueError('Full physical native-column order required')
        indices=np.arange(468,dtype=np.int64) if selected is None else _integer_indices(selected,468,'detail columns')
        canonical=self.start if q is None else self._canonical_q(q)
        expanded,_=self.expanded_q_and_chain(canonical)
        _,base_chain=self.base.effective_q_and_chain(expanded)
        physical_detail=(csr_matrix(base_chain)@self.plan.camera_detail_chain(self.parameter_count,indices)).tocsr()
        physical_detail.eliminate_zeros()
        world=np.asarray(self.programme.state.world_parameter_indices,int)
        if physical_detail[world].nnz or physical_detail[self.programme_count:].nnz:
            raise ValueError('Camera-only details affect world geometry or the independent wide camera; full source derivatives required')
        result=(J@physical_detail[:J.shape[1]]).tocsr()
        result.eliminate_zeros()
        return result

    def solver_kwargs(self):
        value=super().solver_kwargs()
        value.update(initial_damping=float(self._parent['damping']),
            initial_line_search_fraction=float(self._parent['fraction']),reuse_source_jacobian=False)
        return value
