"""Construct only the already admitted circle/material/placement source owners."""
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np

def existing_LOW_source_owners(state,q,directory,options,scene):
    from structured_programme_source_candidate.native_board_circle_source import NativeBoardCircleFactor
    from structured_programme_source_candidate.native_board_material_point_source import NativeBoardMaterialPointFactor,SourceWithAdditionalNativeBoardMaterialPoints
    from structured_programme_source_candidate.course_patterns_independent_placements import SourceWithIndependentPatternPlacements
    from structured_programme_source_candidate.current_full_source import CurrentPatterns
    from z190_shared_board_pattern_source import SharedPerimeterBoardPatternGeometry,SharedBoardPatternSourceFactor
    with np.load(Path(directory)/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        pattern_start=int(z['aux__shared_pattern_start'])
        placement_count=len(z['aux__current_independent_pattern_placement_parameter_indices'])
    assert placement_count==4
    prefix=len(q)-placement_count
    circle=NativeBoardCircleFactor(state,options['--additional-board-circle-source'],prefix-6)
    provider=state.geometry
    geometry=SharedPerimeterBoardPatternGeometry(provider.current.base,provider.scene_unit_m,
        facet_authority=scene['shared_board_patterns']['facet_authority'])
    original=SharedBoardPatternSourceFactor(geometry)
    owner=SimpleNamespace(start=pattern_start,state=state,geometry=geometry,factor=original,external_factor=original,
        _original_course_pattern_geometry=geometry,_original_course_pattern_factor=original,
        _original_course_external_pattern_factor=original,course_world_geometry=provider,fit=[],held=[],bind_current_geometry=None)
    owner.bind_current_geometry=MethodType(CurrentPatterns.bind_current_geometry,owner)
    state.patterns=owner
    material=NativeBoardMaterialPointFactor(state,options['--additional-board-material-point-source'],prefix)
    empty=SimpleNamespace(count=prefix,quadratic=np.empty(0,bool),scales=np.empty(0),contract={},updater=None,
        blocks=lambda q,jacobian:iter(()),guard=lambda q:None)
    placements=SourceWithIndependentPatternPlacements(SourceWithAdditionalNativeBoardMaterialPoints(empty,material),
        state,Path(options['--independent-board-pattern-placement-source']))
    assert np.array_equal(placements.extend_initial(q[:prefix],suffix=q[prefix:]),q)
    return circle,material,placements
