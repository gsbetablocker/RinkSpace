"""Continue a complete accepted native state after an ordinary RGB identity edit."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys

import numpy as np

import fit_native_family as native
from calibration.program_tracking import first_shot_post_native_identity as delta
from calibration.program_tracking import first_shot_resume_cache as cache
from calibration.program_tracking import first_shot_source_transition as transition

reuse = native.reuse
manifest_path = None
initializer_path = None
fixed_scene = False
frame_export_directory = None


class IdentityBundle(native.FamilyBundle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        delta.attach_post_native_identity(self)


class PreviousInventory:
    def __init__(self, owner, manifest):
        self._owner = owner
        self.bundle = SimpleNamespace(frames=owner.bundle._post_identity_previous_frames,
            state=owner.bundle._post_identity_previous_state)
        self.scales = owner.bundle._post_identity_previous_scales
        self.quadratic = owner.bundle._post_identity_previous_mask
        self.association_reference = manifest['base_association']

    def __getattr__(self, key):
        return getattr(self._owner, key)

    def checkpoint_arrays(self):
        return self._owner.checkpoint_arrays.__func__(self)

    def _ids_and_pts(self):
        inventory, _ = cache.source_inventory(self.bundle)
        return inventory['source_point_ids'], inventory['native_pts']

    def complete_state_payload(self, q):
        return self._owner.complete_state_payload.__func__(self, q)


class IdentityReplay(native.FamilyReplay):
    def __init__(self, adapter, **kwargs):
        ref, manifest, *_ = delta.read_transition(manifest_path)
        view = PreviousInventory(adapter, manifest)
        initializer = None
        if initializer_path is not None:
            from calibration.program_tracking.first_shot_parameter_initializer import read_parameter_initializer
            _, recipe, arrays, parent_q, initializer = read_parameter_initializer(
                initializer_path, source_transition=ref)
            if (transition.reference(kwargs['checkpoint']) != recipe['initializer']
                    or transition.reference(kwargs['prior_report']) != recipe['accepted_parent_report']
                    or not np.array_equal(adapter.start.view(np.uint64), arrays['q'].view(np.uint64))):
                raise ValueError('Actual new-source start differs from explicit parameter initializer')
            for name, value in adapter.checkpoint_arrays().items():
                if not np.array_equal(arrays['aux__' + name], value):
                    raise ValueError('Parameter initializer changed the actual auxiliary chart: ' + name)
            # Authenticate the genuinely accepted parent on its original source,
            # then separately admit the exact intervening parameter edit.
            view.start = parent_q
            accepted_kwargs = dict(kwargs, checkpoint=recipe['accepted_parent_checkpoint']['path'])
        else:
            accepted_kwargs = kwargs
        strict = transition.checkpoint_reference_verifier(kwargs['pool_transition'])
        original = cache.CompleteSourceReplay
        class RichReferenceVerifier(original):
            def verify_reference(self, value):
                strict.verified, strict.transitions = self.verified, self.transitions
                return strict.verify_reference(value)
        cache.CompleteSourceReplay = RichReferenceVerifier
        try:
            native.BaseFreshCurrentReplay.__init__(self, view, **accepted_kwargs)
        finally:
            cache.CompleteSourceReplay = original
        prior = transition.read(transition.reference(kwargs['prior_report']))
        root = Path(prior['snapshot']['path']).parent.parent
        identity = transition.read(transition.reference(root / 'run_identity.json'))
        if (identity['binding'] != manifest['base_binding']
                or identity['association'] != manifest['base_association']):
            raise ValueError('Previous immutable inventory belongs to another source basis')
        inventory, _ = cache.source_inventory(view.bundle)
        saved_ref = identity['source_inventory']
        if transition.reference(saved_ref['path']) != saved_ref:
            raise ValueError('Previous immutable source inventory bytes changed')
        with np.load(saved_ref['path'], allow_pickle=False) as saved:
            if set(saved.files) != set(inventory):
                raise ValueError('Previous source inventory fields changed')
            for key, value in inventory.items():
                if not np.array_equal(saved[key], value):
                    raise ValueError('Previous complete source inventory differs: ' + key)
        if (adapter.bundle.state.binding_reference != manifest['revised_binding']
                or adapter.association_reference != manifest['revised_association']):
            raise ValueError('New source identity differs from the admitted transition')
        self.adapter = adapter
        self._new_source_ref = ref
        self.receipt['old_source_rows'] = self.receipt.pop('source_rows')
        self.receipt['old_source_points'] = self.receipt.pop('source_points')
        self.receipt.update(transition_manifest=ref, old_binding=manifest['base_binding'],
            new_binding=manifest['revised_binding'], source_rows=len(adapter.scales),
            source_points=sum(f['source_point_count'] for f in adapter.bundle.frames),
            same_source_basis_as_checkpoint=False, same_source_objective_as_checkpoint=False,
            old_source_basis_prior_cost=prior['source_objective_cost'],
            old_new_costs_are_different_objectives=True, cross_basis_cost_reduction_not_claimed=True,
            old_complete_UV_ID_target_PTS_inventory_exact=True,
            source_inventory_UV_ID_sigma_whitening_PTS_order_and_chart_exact=False,
            old_checkpoint_source_inventory_exact_new_inventory_separately_verified=True,
            complete_source_transition_comparison=deepcopy(adapter.bundle.post_native_identity_receipt['source_scope']),
            latest_accepted_q_aux_damping_and_fraction_preserved=True,
            new_source_full_r_J_world_required=True, identity_continuation_entry=transition.reference(__file__))
        if initializer is not None:
            self.receipt.update(explicit_parameter_initializer=initializer,
                qaug_sha256=initializer['parameter_q_sha256'],
                accepted_old_source_resume_claimed=False,
                latest_accepted_q_aux_damping_and_fraction_preserved=False,
                complete_rescue_q_and_all_accepted_aux_damping_fraction_preserved=True)
            if (adapter.seed_receipt['initial_damping'] != self.damping
                    or adapter.seed_receipt['initial_line_search_fraction'] != self.fraction):
                raise ValueError('Explicit initializer altered accepted damping/fraction')
        for name in ('frames', 'state', 'scales', 'mask'):
            delattr(adapter.bundle, '_post_identity_previous_' + name)

    def install(self):
        result = super().install()
        # The authenticated source edit requires its own current J. Its new
        # direction starts at a full numerical trial without rewriting the
        # accepted q, auxiliary values, damping or saved canonical fraction.
        from calibration.program_tracking.first_shot_prediction_refresh import install_prediction_refresh
        ref = transition.reference(__file__)
        refresh = install_prediction_refresh(self.adapter, driver=reuse.mount.driver,
            entry_reference=ref, minimum_prediction_ratio=.25,
            reset_initial_fresh_source_fraction=True)
        self.receipt.update(source_J_execution_policy=deepcopy(self.adapter.source_jacobian_reuse_policy),
            prediction_quality_source_J_refresh=refresh)
        if ref not in self.adapter.source_producers:
            self.adapter.source_producers.append(ref)
        if initializer_path is not None:
            from calibration.program_tracking import first_shot_parameter_initializer as initial
            for producer in (transition.reference(initial.__file__), transition.reference(initializer_path)):
                if producer not in self.adapter.source_producers:
                    self.adapter.source_producers.append(producer)
        if fixed_scene:
            from calibration.program_tracking.first_shot_fixed_geometry import (
                scene_camera_parameter_groups, install_fixed_geometry)
            groups = scene_camera_parameter_groups(self.adapter.state)
            indices = np.r_[groups['base_scene_geometry'], groups['added_geometry_deviations']]
            if (len(groups['base_scene_geometry']) != 118
                    or len(groups['added_geometry_deviations']) != 5 or len(groups['print_layout']) != 48):
                raise ValueError('Explicit base118/deviation5/print48 chart changed')
            self.receipt['fixed_scene'] = install_fixed_geometry(self.adapter,
                fixed_parameter_indices=indices, fixed_parameter_values=self.adapter.start[indices].copy(),
                native_pp_values=None, entry_reference=ref,
                value_reference=transition.reference(initializer_path))
        if frame_export_directory is not None:
            from calibration.program_tracking.first_shot_safe_frame_export import export_source_frames
            self.adapter.guard(self.adapter.start)
            self.receipt['complete_actual_initializer_guard_passed'] = True
            self.receipt['safe_selected_source_export'] = export_source_frames(self.adapter,
                output_directory=frame_export_directory, frame_indices=(201, 385, 589, 842, 1239, 1248))
        return result


def main():
    global manifest_path, initializer_path, fixed_scene, frame_export_directory
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--identity-revision-manifest', required='--help' not in sys.argv and '-h' not in sys.argv)
    parser.add_argument('--from-current')
    parser.add_argument('--parameter-initializer-manifest')
    parser.add_argument('--fix-base-scene-and-deviations', action='store_true')
    parser.add_argument('--source-frame-export')
    parser.add_argument('--recover-pivot-roundoff', action='store_true')
    parser.add_argument('--live-review-export', action='store_true')
    flags, remaining = parser.parse_known_args()
    initializer_path = flags.parameter_initializer_manifest
    fixed_scene = flags.fix_base_scene_and_deviations
    frame_export_directory = flags.source_frame_export
    if ((flags.from_current and initializer_path) or (fixed_scene and not initializer_path)):
        raise ValueError('Fixed rescue geometry requires the explicit initializer, exclusive with --from-current')
    if flags.from_current:
        if '--seed' in remaining or '--prior-report' in remaining:
            raise ValueError('--from-current supplies its exact matching canonical and report')
        current = transition.read(transition.reference(flags.from_current))
        canonical = current['canonical_checkpoint']
        if transition.reference(canonical['path']) != {k: canonical[k] for k in ('path', 'sha256')}:
            raise ValueError('Latest canonical bytes changed')
        remaining.extend(('--seed', canonical['path'], '--prior-report', flags.from_current))
    sys.argv = [sys.argv[0], *remaining]
    native.pivot_roundoff, native.live_review = flags.recover_pivot_roundoff, flags.live_review_export
    if '--help' not in sys.argv and '-h' not in sys.argv:
        manifest_path = Path(flags.identity_revision_manifest)
        _, manifest, *_ = delta.read_transition(manifest_path)
        required = argparse.ArgumentParser(add_help=False)
        for key in ('binding', 'association', 'seed', 'prior-report', 'output'):
            required.add_argument('--' + key, required=True)
        required.add_argument('--geometry-context', action='store_true', required=True)
        required.add_argument('--recover-finite-extent', action='store_true', required=True)
        required.add_argument('--pool-transition', default=str(reuse.mount.driver.Z / 'geometry_pool_reuse'))
        args, _ = required.parse_known_args()
        if (transition.reference(args.binding) != manifest['revised_binding']
                or transition.reference(args.association) != manifest['revised_association']):
            raise ValueError('CLI source differs from the admitted ordinary identity delta')
        report = transition.read(transition.reference(args.prior_report))
        verifier = transition.checkpoint_reference_verifier(args.pool_transition)
        verifier.verify_references(report)
        if initializer_path is not None:
            from calibration.program_tracking.first_shot_parameter_initializer import read_parameter_initializer
            _, recipe, arrays, _, _ = read_parameter_initializer(initializer_path,
                source_transition=transition.reference(manifest_path))
            if (transition.reference(args.seed) != recipe['initializer']
                    or transition.reference(args.prior_report) != recipe['accepted_parent_report']
                    or recipe['original_source_binding'] != manifest['base_binding']
                    or recipe['revised_source_binding'] != manifest['revised_binding']
                    or arrays['q'].shape != (7674,)):
                raise ValueError('CLI must use the exact explicit initializer and its accepted parent report')
        else:
            with np.load(args.seed, allow_pickle=False) as saved:
                identity = json.loads(str(saved['identity_json']))
                verifier.verify_references(identity)
                if (saved['q'].shape != (7674,) or hashlib.sha256(saved['q'].tobytes()).hexdigest() != report['qaug_sha256']
                        or report['binding'] != manifest['base_binding'] or identity['binding'] != manifest['base_binding']
                        or identity['association'] != manifest['base_association']):
                    raise ValueError('Continuation must start at the exact accepted previous source basis')
        native.revised.parent_installation = native.revised.fused.value.install()
        native.revised.validation_references = [transition.reference(reuse.mount.driver.Z / 'value_real_state_QA/report.json'),
            transition.reference(reuse.mount.driver.Z / 'pivot_step_QA/report.json')]
        delta.install_post_native_identity(manifest_path)
    previous = reuse.annotate_current_report
    def annotate(report, owner):
        previous(report, owner)
        if isinstance(report, dict) and 'fresh_current_source_J_every_new_state' in report:
            report.update(source_basis='RGB-verified ordinary identities after authenticated native readouts',
                same_source_basis_as_checkpoint=False, same_source_objective_as_checkpoint=False,
                old_new_source_costs_are_different_objectives=True, cross_basis_cost_reduction_not_claimed=True,
                post_native_ordinary_identity_revision=deepcopy(owner.bundle.post_native_identity_receipt),
                same_state_original_source_value_replay=dict(owner.source_value_replay_disabled_receipt))
            if hasattr(owner, 'fixed_geometry_receipt'):
                held = owner.fixed_geometry_receipt
                report.pop('only_four_extra_PP_gauge_coordinates_fixed', None)
                report.update(all_5172_old_parameters_native_PP_axis_a_and_shared_geometry5_joint=False,
                    initial_exact_complete_current_state_full_J_reuse=False,
                    fixed_scene_geometry=deepcopy(held),
                    fixed_parameter_indices=np.r_[owner.fixed, held['fixed_parameter_indices']].tolist(),
                    existing_PP_gauge_fixed_parameter_indices=owner.fixed.tolist(),
                    base_scene118_and_deviation5_fixed_at_explicit_initializer=True,
                    print_layout48_and_existing_camera_parameters_optimized=True,
                    native_PP2496_preserved_as_existing_model_coordinates=True,
                    effective_shared_dense_parameters=len(owner.shared)-len(held['fixed_geometry_indices']))
            if (frame_export_directory is not None and report.get('kind') == 'accepted'
                    and 'canonical_checkpoint' in report):
                from calibration.program_tracking.first_shot_safe_frame_export import export_source_frames
                candidate_ref = report['canonical_checkpoint']
                if transition.reference(candidate_ref['path']) != candidate_ref:
                    raise ValueError('Accepted source export canonical identity changed')
                with np.load(candidate_ref['path'], allow_pickle=False) as saved:
                    q = saved['q'].copy()
                folder = (Path(frame_export_directory).parent / 'candidate_frames' /
                    ('%04d_' % report['accepted_index'] + report['qaug_sha256'][:12]))
                report['safe_selected_source_export'] = export_source_frames(owner,
                    output_directory=folder, frame_indices=(201, 385, 589, 842, 1239, 1248), q=q)
        return report
    reuse.annotate_current_report = annotate
    reuse.ExecutionBundle, reuse.FreshCurrentReplay = IdentityBundle, IdentityReplay
    if '--fresh-current-state' not in sys.argv:
        sys.argv.append('--fresh-current-state')
    reuse.main()


if __name__ == '__main__':
    main()
