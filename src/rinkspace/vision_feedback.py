"""Connect the existing source-feedback runtime to the trained source reader."""
from __future__ import annotations
from .feedback_runtime import SourceFeedbackRead


def begin_read(binding, guides):
    ids = {"board.top", "board.contact"}
    for guide in guides:
        identity = str(guide['id'])
        ids.update((identity, identity.removeprefix('ice.'), identity.split(':')[0],
                    identity.removeprefix('ice.').split(':')[0]))
    return SourceFeedbackRead(binding, entity_ids=ids, tags=('source_reader', 'ice', 'board', 'temporal'))


def check_guide_operators(ticket, guides):
    conflicts = []
    for guide in guides:
        if guide.get('kind') != 'ice_paint':
            continue
        identity = str(guide['id'])
        operator = 'paint_center' if guide['physical_role'] == 'painted_stroke_center_axis' else 'material_interface'
        supported = ticket.require_operator((identity, identity.removeprefix('ice.'),
            identity.split(':')[0], identity.removeprefix('ice.').split(':')[0]), operator)
        if not supported:
            # Knowledge is consulted, but it cannot replace the current native definition.
            if not guide.get('native_geometry'):
                raise ValueError('Source role conflicts with authenticated feedback: '+identity)
            conflicts.append(dict(id=identity, operator=operator, resolution='current_native_geometry_authoritative'))
    return conflicts


def finish_read(ticket, rgb, guided, *, operator_conflicts=(), geometry=None):
    readings = guided['readings']
    checks = []
    for reading in readings:
        method = reading.get('support', {}).get('method')
        if method in ('ordered_blue_rail_upper_material_entry','ordered_yellow_foot_lower_material_exit'):
            checks.append(dict(id=reading['expected_id'],check=method,
                actual_measured_count=len(reading['actual_source_uv']), generic_edge_replaced=True))
        elif method in ('current_native_RGB_with_per_query_precision_evidence','current_native_print_RGB_half_transition'):
            checks.append(dict(id=reading['expected_id'],check=method,
                native_path_id=reading.get('native_path_id'),
                actual_measured_count=len(reading['actual_source_uv']),
                precise_source_count=len(reading['support']['actual_readout_evidence']['precise_source_indices']),
                retained_current_RGB_count=len(reading['support']['actual_readout_evidence']['retained_current_RGB_source_indices']),
                actual_source_precision_and_temporal_identity_consumed=True))
        elif method == 'opposite_source_colour_transitions_midpoint':
            checks.append(dict(id=reading['expected_id'], check='current_RGB_opposite_material_flanks',
                actual_measured_count=len(reading['actual_source_uv']),
                coherent_measured_count=len(reading.get('coherent_source_uv', reading['actual_source_uv']))))
        elif method == 'single_source_RGB_material_transition':
            checks.append(dict(id=reading['expected_id'], check='current_RGB_material_transition',
                actual_measured_count=len(reading['actual_source_uv']),
                coherent_measured_count=len(reading.get('coherent_source_uv', reading['actual_source_uv']))))
    for operator,recipe in (('painted_stroke_center_axis','single_actual_stroke_two_flanks_topology'),
                            ('filled_disc_boundary','declared_target_material_interface')):
        executed=[r for r in readings if r.get('kind')=='ice_paint' and r.get('physical_role')==operator
            and r.get('support',{}).get('actual_readout_evidence',{}).get('precise_source_indices')]
        if executed:
            ticket.use_current_reader_recipe(recipe,postcondition=dict(actual_current_RGB_read_executed=True,
                source_UV_or_identity_transferred=False,pixels_from_model=False,
                current_native_entity_ids=[r['expected_id'] for r in executed],
                precise_read_count=sum(len(r['support']['actual_readout_evidence']['precise_source_indices']) for r in executed),
                native_geometry=geometry))
    # Different reader implementations must not claim consumption of a more
    # specific recipe merely because it was retrieved. Executed operator checks
    # and their actual current-source outcomes are the durable feedback evidence.
    return ticket.finish(rgb, dict(current_source_checks=checks,
        unresolved_case_recipes_are_retrieval_only=True, operator_conflicts=list(operator_conflicts),
        native_geometry=geometry, original_source_UV_transferred=False,
        camera_or_world_optimized=False), tool='SourceRinkReader')
