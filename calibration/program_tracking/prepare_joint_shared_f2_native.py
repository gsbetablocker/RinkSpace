"""Prepare one native-aware main(source-pose clock)+low shared-F2 graph.

Call ``prepare_joint_f2_graph(main_directory, low_directory)`` from a dedicated
process. The returned object owns a complete, unoptimized graph and must be
closed after export/solve. Its main model uses the corrected source-image pose
bindings; original official exposure aliases remain in the input camera file.
No source pixel is selected here: the original preflight drivers capture their
existing factor specifications, then native managers build with rekeyed models
in one global factor namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np

import fit_continuous_source_identity_repaired as main_entry
import fit_low_end_temporal_principal as low_entry
from continuous_camera_solver import values_for
from fit_continuous_rig_source_calibrated import release_native_resources
from reconstruct_source_pose_clock import reconstruct as reconstruct_source_pose
from prepare_shared_f2_composite import (
    TMP, CANDIDATE, ORIGINAL, ORIGINAL_STATES, packet, low_models, sha)
from capture_complete_shared_f2_graphs import _main_args, _low_args
from native_aware_shared_f2_rebuild import (
    capture_main_builder_context, capture_low_builder_context,
    rebuild_main_source_pose_complete, rebuild_low_complete,
    append_main_f2_source_references, append_builders_to_one_graph,
    NativeManagerMux, new_joint_graph)
from shared_f2_curve_preparation import (
    install_joint_shared_f2_curve_preparation,
    prepare_joint_shared_f2_curves)
from adaptive_curve_parallel import close_adaptive_curve_workers
from shared_scene_calibration import SharedRingScene
from continuous_exposure_clock import LEGACY_POLICY
from production_exposure_clock import derive_saved_pose_clock


@dataclass
class PreparedJointF2Graph:
    graph: object
    states: dict
    prior: dict
    main_preflight_states: dict
    main: object
    low: object
    native_manager: NativeManagerMux
    added_main_f2: dict
    provenance: dict
    low_original_factor_costs: np.ndarray | None
    _temporary: object
    _temp_root: Path
    _resources: tuple[dict, dict]

    @property
    def values(self):
        return values_for(self.graph,self.states)

    def curve_preparation(self):
        """Install scene-aware preparation around the unchanged joint solver."""
        return install_joint_shared_f2_curve_preparation(self.graph)

    def prepare_curves(self):
        return prepare_joint_shared_f2_curves(self.graph,self.states)

    def close(self):
        if self.graph is None:
            return
        close_adaptive_curve_workers(self.graph)
        self.graph._pending.clear()
        self.graph._callbacks.clear()
        self.graph._smoother=None
        for resources in self._resources:
            resources.pop('graph',None)
            resources.pop('native_edge_manager',None)
        gc.collect()
        self.native_manager.close_after_release()
        _checked_cleanup(self._temporary,self._temp_root)
        self.graph=None

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()


def _scene_inputs(candidate_scene_path, original_scene_path, old_scene_states):
    candidate=json.loads(Path(candidate_scene_path).read_text(encoding='utf8'))
    original=json.loads(Path(original_scene_path).read_text(encoding='utf8'))
    saved=packet(old_scene_states)[2001]
    released=SharedRingScene(original,{'features':['C0','F2','R3','R4']})
    f2=released.features['F2']
    feature=next(x for x in candidate['ice_features'] if x['id']=='F2')
    ring=next(x for x in feature['rings'] if x['role']=='outer')
    center=np.asarray(ring['center_xyz'][:2],np.float64)
    radius=float(ring['radius'])
    if (np.max(np.abs(np.asarray(f2['center'])+100.*saved[:2]-center))>1e-10
            or abs(f2['outer_radius']+100.*saved[2]-radius)>1e-10):
        raise ValueError('Candidate F2 differs from released scene plus saved q')
    return candidate, f2, saved


def _checked_cleanup(temporary, temp_root):
    folder=Path(temporary.name).resolve()
    if (folder.parent!=Path(temp_root).resolve()
            or not folder.name.startswith('shared-f2-joint-prep-')):
        raise ValueError('Refuse TMP cleanup outside owned joint-prep folder')
    temporary.cleanup()


def _freeze_inputs(source, target, names):
    target.mkdir()
    digests={}
    for name in names:
        payload=(source/name).read_bytes()
        digests[name]=hashlib.sha256(payload).hexdigest()
        (target/name).write_bytes(payload)
    return digests


def prepare_joint_f2_graph(main_directory, low_directory, *,
                           main_state_name='states.npz',
                           capture_baseline_factor_costs=False,
                           candidate_scene_path=CANDIDATE,
                           original_scene_path=ORIGINAL,
                           old_scene_states=ORIGINAL_STATES,
                           temp_root=TMP,
                           exposure_clock_policy=LEGACY_POLICY):
    """Return complete graph/states/prior/models/refs/native mux; no solve.

    Main input must be the corrected v14 source-pose directory. The default
    ``states.npz`` is its completed state; explicitly pass
    ``main_state_name='accepted_states.npz'`` for an intermediate checkpoint.
    All reconstruction input bytes are copied once to an owned TMP snapshot.
    Low input is the complete v11 time-principal directory.
    Caller may inspect factor errors, solve with ``curve_preparation()``, and
    export ``main.programme``/``low.programme``. The 14 added reference model
    is ``main.new_source_reference_model``; ``main.key_mapping`` and
    ``low.key_mapping`` map original graph IDs to global graph IDs.
    """
    main_directory=Path(main_directory).resolve()
    low_directory=Path(low_directory).resolve()
    candidate_scene_path=Path(candidate_scene_path).resolve()
    temp_root=Path(temp_root).resolve()
    if main_state_name not in ('states.npz','accepted_states.npz'):
        raise ValueError('Main state packet must be completed or intermediate')
    if not main_directory.is_dir() or not low_directory.is_dir():
        raise ValueError('Both saved rig directories must exist')
    candidate,f2,q_saved=_scene_inputs(candidate_scene_path,
                                       original_scene_path,old_scene_states)
    temporary=tempfile.TemporaryDirectory(prefix='shared-f2-joint-prep-',
                                           dir=temp_root)
    folder=Path(temporary.name)
    main_context,low_context={},{}
    old_main,old_low={},{}
    resources_main,resources_low={},{}
    graph=None
    mux=None
    low_original_factor_costs=None
    try:
        main_snapshot=folder/'snapshot_main'
        low_snapshot=folder/'snapshot_low'
        main_digests=_freeze_inputs(main_directory,main_snapshot,(
            'phase_00.json','initial_cameras.npz',
            'prior_initializer_states.npz','preflight_initial_states.npz',
            main_state_name))
        low_digests=_freeze_inputs(low_directory,low_snapshot,(
            'phase_00.json','states.npz','prior_initializer_states.npz',
            'shot_cameras.npz','reference_cameras.npz'))
        main_phase,main_model,main_refs,main_states,_=reconstruct_source_pose(
            main_snapshot,state_name=main_state_name,require_quadratic=True)
        main_prior=packet(main_snapshot/'prior_initializer_states.npz')
        main_initial=packet(main_snapshot/'preflight_initial_states.npz')
        low_phase,low_model,low_refs,low_states,low_prior=low_models(low_snapshot)
        if (main_phase['model_sha256']!=sha(candidate_scene_path)
                or low_phase['model_sha256']!=sha(candidate_scene_path)
                or main_phase['source_sha256']!=low_phase['source_sha256']):
            raise ValueError('Input rigs do not share exact source/candidate scene')
        with capture_main_builder_context(main_context):
            main_entry.run(_main_args(folder/'capture_main'),old_main)
        release_native_resources(old_main)
        main_context.pop('graph',None)
        with capture_low_builder_context(low_context):
            low_entry.run(_low_args(folder/'capture_low',low_snapshot),old_low)
        if capture_baseline_factor_costs:
            old_values=values_for(old_low['graph'],low_states)
            low_original_factor_costs=np.asarray([
                float(f.error(old_values)) for f in old_low['graph']._pending],
                np.float64)
        release_native_resources(old_low)
        low_context.pop('graph',None)
        # Historical captures and their frozen SHA gates above reproduce the
        # parent objective. The candidate model is installed only now, before
        # the NEW managers snapshot C/principal basis weights into C++ specs.
        main_context,main_phase,main_model,main_clock_binding=derive_saved_pose_clock(
            main_context,main_phase,main_model,policy=exposure_clock_policy,
            parent_phase_sha256=main_digests['phase_00.json'])
        low_context,low_phase,low_model,low_clock_binding=derive_saved_pose_clock(
            low_context,low_phase,low_model,policy=exposure_clock_policy,
            parent_phase_sha256=low_digests['phase_00.json'])
        # v14 installed this opt-in before constructing its corrected graph.
        from continuous_board_seam_jacobian_optin import install_board_seam_optin
        install_board_seam_optin()
        total_nodes=len(main_states)+len(low_states)+29
        graph=new_joint_graph(total_nodes)
        resources_main['graph']=graph
        resources_low['graph']=graph
        with append_builders_to_one_graph(graph):
            main=rebuild_main_source_pose_complete(main_context,main_phase,
                main_model,main_refs,main_states,main_prior,main_initial,
                q_saved,_main_args(folder/'build_main'),
                resources=resources_main)
            added=append_main_f2_source_references(main,candidate,q_saved,f2)
            low=rebuild_low_complete(low_context,low_phase,low_model,low_refs,
                low_states,low_prior,q_saved,
                _low_args(folder/'build_low',low_snapshot),
                resources=resources_low)
        states={**main.states,**low.states}
        prior={**main.prior,**low.prior}
        # The new scene prior is centred at -q_saved in candidate-delta
        # coordinates. Save its true factor target, separate from the zero
        # initial scene state used to reproduce the candidate world exactly.
        prior[2001]=np.asarray(added['released_prior_target'],np.float64)
        main_preflight_states={
            main.key_mapping.get(int(k),int(k)):v.copy()
            for k,v in main_initial.items()}
        main_preflight_states[2001]=np.zeros(11)
        if (len(states)!=len(main.states)+len(low.states)-1
                or len(prior)!=len(states)
                or len(graph._pending)!=(main.original_factors+
                                         added['added_factor_count']+
                                         low.original_factors)):
            raise ValueError('Joint node/factor inventory is inconsistent')
        if any(not np.array_equal(main.states[k],low.states[k])
               for k in set(main.states)&set(low.states)):
            raise ValueError('Shared F2 scene node has inconsistent initial state')
        mux=NativeManagerMux(main.manager,low.manager)
        provenance=dict(schema='rinkspace.prepared-native-joint-f2.v1',
            main_directory=str(main_directory),low_directory=str(low_directory),
            main_state_field=main_state_name,
            main_state_sha256=main_digests[main_state_name],
            frozen_main_file_sha256=main_digests,
            frozen_low_file_sha256=low_digests,
            low_states_sha256=low_digests['states.npz'],
            candidate_scene_sha256=sha(candidate_scene_path),
            original_scene_sha256=sha(original_scene_path),
            saved_original_f2_graph_state=q_saved.tolist(),
            new_scene_graph_key=2001,
            scene_state_role='delta from candidate_scene F2',
            scene_prior_target_role='-saved original F2 q in candidate-delta coordinates',
            new_reference_prior_packet_role='initial seeds for 28 new nodes, not historical camera priors',
            source_pose_clock_retained=True,
            original_official_aliases_are_not_pose_aliases=True,
            main_original_factor_count=main.original_factors,
            low_original_factor_count=low.original_factors,
            added_main_f2_factor_count=added['added_factor_count'],
            total_factor_count=len(graph._pending),total_node_count=len(states),
            solve_started=False)
        if main_clock_binding is not None:
            provenance.update(exposure_clock_policy=exposure_clock_policy,
                exposure_clock_derivations=dict(main_long_side=main_clock_binding,
                                               low_end=low_clock_binding),
                objective_changed_from_parent=True, original_prior_targets_preserved=True)
        prepared=PreparedJointF2Graph(graph,states,prior,main_preflight_states,
                                    main,low,mux,added,provenance,
                                    low_original_factor_costs,temporary,temp_root,
                                    (resources_main,resources_low))
        prepared.effective_programme_phases=dict(main_long_side=main_phase,low_end=low_phase)
        return prepared
    except BaseException:
        managers=[resources.get('native_edge_manager')
                  for resources in (resources_main,resources_low)]
        if graph is not None:
            close_adaptive_curve_workers(graph)
            graph._pending.clear();graph._callbacks.clear();graph._smoother=None
        for resources in (resources_main,resources_low):
            resources.pop('graph',None);resources.pop('native_edge_manager',None)
        gc.collect()
        if mux is not None:
            mux.close_after_release()
        else:
            for manager in managers:
                if manager is not None:
                    manager.close_after_release()
        release_native_resources(old_main)
        release_native_resources(old_low)
        _checked_cleanup(temporary,temp_root)
        raise
