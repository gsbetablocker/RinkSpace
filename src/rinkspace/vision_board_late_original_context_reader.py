"""Separate late-cut original-source reader with existing typed operators.

Only a named internal printed graphic uses its existing narrow interface
operator. Existing three-role material points retain typed two-image clearance;
all other interfaces use the frozen actual-pair source reader.
"""
from .vision_board_actual_pair_recovery_reader import ActualPairRecoveryBoardSourceReader
from .vision_board_typed_material_clearance import TypedMaterialBoardBoxClearance
from .vision_board_narrow_graphic_profiles import read_narrow_internal_graphic_interface


class LateOriginalContextBoardSourceReader(ActualPairRecoveryBoardSourceReader):
    def _read_edge(self, image, exclusion, bands, track, *, known_source_board_material=None):
        if track['template']['anchor']['curve'].get('extent_scope') == 'internal_graphic_only':
            return read_narrow_internal_graphic_interface(image, exclusion, bands, track,
                known_source_board_material=known_source_board_material)
        return super()._read_edge(image, exclusion, bands, track,
            known_source_board_material=known_source_board_material)

    def read(self, image, *args, **kwargs):
        if kwargs.get('person_components') is not None and not isinstance(
                getattr(self, 'box_clearance', None), TypedMaterialBoardBoxClearance):
            self.box_clearance = TypedMaterialBoardBoxClearance(self.actual_successor)
        result = super().read(image, *args, **kwargs)
        for row in result['advertisement_rows']:
            if row['curve'].get('extent_scope') == 'internal_graphic_only':
                row['source_readout_uncertainty']['measurement_method'] = 'actual_internal_graphic_single_interface_q25_q50_q75'
                row['source_profile_extent_is_navigation_only'] = True
                row['opposite_graphic_border_used_as_output'] = False
        return result
