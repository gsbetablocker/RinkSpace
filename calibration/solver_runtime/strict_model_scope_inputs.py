"""Public consumer and startup gate for exact-source ice/board input branches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.strict_absolute_source import StrictAbsoluteSourcePack
from calibration.program_tracking.strict_model_scope import file_sha, load_motion_partition


def load_strict_programme_motion_inputs(path, *, expected_sha256=None):
    """Load only physically compact approved arrays, retaining all source clocks."""
    path = Path(path).resolve()
    if expected_sha256 is not None and file_sha(path) != expected_sha256:
        raise ValueError('Strict programme source binding changed')
    binding = json.loads(path.read_text(encoding='utf8'))
    if (binding.get('schema') != 'rinkspace.strict-model-scope-original-motion-inputs.v1'
            or binding.get('unknown_or_unreviewed_default') != 'excluded'):
        raise ValueError('Strict model-scope programme input branch required')
    source = binding['source']
    blocks = {}
    for row in binding['blocks']:
        name = row['block']
        if name in blocks:
            raise ValueError('Duplicate strict programme source block')
        values, metadata = load_motion_partition(row['path'], expected_sha256=row['sha256'],
            source_sha256=source['video_sha256'], time_base=source['time_base'])
        if metadata['block'] != name or len(values['a']) != row['after_points']:
            raise ValueError('Actual strict programme source arrays differ')
        blocks[name] = values
    return blocks, binding


class StrictProgrammeSourceInputs:
    """Actual compact consumer shared by review and the future graph adapter.

    Existing absolute observations are replaced by this pack's source-measured
    FIT rows. HELD rows remain a separate member; unknown legacy families never
    reappear through aliases, missing descriptors, or calibration-only UVs.
    """
    def __init__(self, *, motion_inputs, absolute_inputs, current_pixel_inventory):
        self.motion_blocks,self.motion_binding=load_strict_programme_motion_inputs(
            motion_inputs['path'],expected_sha256=motion_inputs['sha256'])
        self.absolute=StrictAbsoluteSourcePack(absolute_inputs['path'],expected_sha256=absolute_inputs['sha256'])
        if self.absolute.metadata['source'] != self.motion_binding['source']:
            raise ValueError('Strict absolute and motion original source identities differ')
        inventory_path=Path(current_pixel_inventory['path']).resolve()
        if file_sha(inventory_path)!=current_pixel_inventory['sha256']:
            raise ValueError('Original all-family pixel inventory changed')
        with np.load(inventory_path,allow_pickle=False) as inventory:
            metadata=json.loads(str(inventory['metadata_json'].item()))
        self.absolute.require_complete_legacy_family_gate(metadata['family_receipts'])
        exposures=np.unique(np.concatenate([block['stamps'] for block in self.motion_blocks.values()]))
        if not np.array_equal(exposures,self.absolute.metadata['expected_actual_pts']):
            raise ValueError('Strict absolute pack omitted declared actual programme exposures')
        self.fit_absolute_rows=self.absolute.fit_rows
        self.held_absolute_rows=self.absolute.held_rows
        self.current_family_receipts=metadata['family_receipts']

    def consume_absolute_inputs(self, legacy_entries):
        validated,retired=self.absolute.compact_legacy_absolute_inputs(legacy_entries)
        # Validated legacy rows are provenance matches only. Emitting them in
        # addition to remeasured pack rows would duplicate the same source UV.
        return dict(fit_rows=self.fit_absolute_rows,held_rows=self.held_absolute_rows,
            retired_legacy_points=retired,validated_legacy_provenance=validated,
            unchecked_legacy_rows_returned=False)


def load_strict_programme_source_inputs(config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf8'))
    policy=config['strict_model_scope_policy']
    return StrictProgrammeSourceInputs(motion_inputs=policy['motion_inputs'],
        absolute_inputs=policy['absolute_inputs'],current_pixel_inventory=policy['current_pixel_inventory'])


def require_solver_scope_policy(config_path, *, require_policy=False):
    """Fail before an old FIT route can consume a pending or unconnected branch.

    The review consumer can read pending compact arrays. Running the original
    solver additionally requires its complete source graph to use the same
    branch for every pixel family; a sidecar alone does not establish that.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf8'))
    policy = config.get('strict_model_scope_policy')
    if policy is None:
        if require_policy:
            raise ValueError('当前 named66 入口必须配置严格冰面/围挡准入策略；禁止无策略回退旧输入。')
        return None
    if policy.get('schema') == 'rinkspace.strict-finite-dataset-startup-policy.v2':
        from calibration.program_tracking.strict_model_scope import authenticate_source_reference
        from calibration.program_tracking.strict_source_fit_dataset import SCHEMA
        if (policy.get('unknown_or_unreviewed_default') != 'excluded'
                or policy.get('allowed_static_surfaces') != ['ice', 'board']):
            raise ValueError('Strict finite dataset scope policy is invalid')
        dataset_ref, motion_ref = policy['fit_dataset'], policy['motion_inputs']
        authenticate_source_reference(dataset_ref)
        authenticate_source_reference(motion_ref)
        dataset = json.loads(Path(dataset_ref['path']).read_text(encoding='utf8'))
        motion = json.loads(Path(motion_ref['path']).read_text(encoding='utf8'))
        if dataset.get('schema') != SCHEMA or dataset['source'] != motion['source']:
            raise ValueError('Strict dataset and motion source identities differ')
        authenticate_source_reference(dataset['frozen_model_reference'])
        options = config['source_options']
        for option, reference in [('--strict-source-fit-dataset',dataset_ref),('--strict-motion-inputs',motion_ref)]:
            if Path(options.get(option,'' )).resolve() != Path(reference['path']).resolve():
                raise ValueError('Required strict graph route is not explicitly configured: '+option)
        from .strict_source_dataset_adapter import StrictDatasetFactor
        from .strict_reviewed_programme_source import StrictReviewedProgrammeSource
        if not callable(StrictDatasetFactor.iter_blocks) or not callable(StrictReviewedProgrammeSource.blocks):
            raise ValueError('Strict graph implementation unavailable')
        return dict(schema=policy['schema'],fit_dataset=dataset_ref,motion_inputs=motion_ref,
                    runtime_full_inventory_validation_required=True)
    if (policy.get('schema') != 'rinkspace.strict-model-scope-startup-policy.v1'
            or policy.get('unknown_or_unreviewed_default') != 'excluded'
            or policy.get('allowed_static_surfaces') != ['ice', 'board']):
        raise ValueError('Strict ice/board startup policy is invalid')
    reference = policy.get('motion_inputs')
    if not isinstance(reference, dict) or file_sha(reference['path']) != reference['sha256']:
        raise ValueError('Strict motion input branch is missing or changed')
    if (policy.get('all_pixel_families_routed_to_strict_input') is not True
            or policy.get('source_review_ready_for_fit') is not True):
        raise ValueError('当前严格冰面/围挡输入仍在逐帧核查，完整源图尚未接入全部像素家族；禁止旧输入继续拟合。')
    inputs=load_strict_programme_source_inputs(config_path)
    if inputs.absolute.metadata['all_exposures_processed'] is not True:
        raise ValueError('严格源输入存在未处理原始曝光；禁止旧输入继续拟合。')
    if (inputs.absolute.metadata['source_uncertainty_ready_for_fit'] is not True
            or inputs.absolute.metadata['all_exposures_exact_native_BGR_bound'] is not True):
        raise ValueError('严格源测量尚缺原片曝光绑定或测量不确定性；禁止把缺失权重补为拟合输入。')
    # Flags in an input file alone cannot certify an implementation route.
    raise ValueError('严格源输入可读取；完整求解消费适配器尚未安装，禁止回退到旧观测输入。')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate the strict model-scope startup policy without solving')
    parser.add_argument('--config', type=Path, required=True)
    options = parser.parse_args()
    try:
        require_solver_scope_policy(options.config)
    except ValueError as error:
        raise SystemExit(str(error))


if __name__ == '__main__':
    main()
