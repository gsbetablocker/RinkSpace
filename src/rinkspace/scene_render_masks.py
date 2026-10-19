"""Carry the saved image classification into the existing polygon renderer.

These display polygons do not participate in camera fitting. Unobserved ice
classification remains excluded; this is separate from the model's extent.
"""
import cv2
import numpy as np


def apply_saved_display_masks(row, observation, arrays):
    exports = observation.get('actual_mask_exports')
    if not exports:
        return False
    decoded = {}
    for name in ('ice_plane_roi', 'foreground_exclusion'):
        info = exports[name]
        if info['coordinate_space'] != 'sample_image_px' or info['sample_size'] != row['sample_size']:
            raise ValueError('Display masks and camera sample domains differ')
        shape = tuple(info['shape_hw'])
        mask = np.unpackbits(arrays[info['npz_key']], bitorder=info['bitorder'])[:np.prod(shape)].reshape(shape)
        decoded[name] = mask.astype(np.uint8)
    # External ice outlines alone would fill excluded holes. Carry every hole
    # as an explicit foreground polygon too, to retain conservative clipping.
    contours, hierarchy = cv2.findContours(decoded['ice_plane_roi'], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ice, holes = [], []
    if hierarchy is not None:
        for contour, entry in zip(contours, hierarchy[0]):
            points = contour.reshape(-1, 2).tolist()
            if len(points) >= 3:
                (ice if entry[3] == -1 else holes).append(points)
    foreground, _ = cv2.findContours(decoded['foreground_exclusion'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row['ice_polygons_sample'] = ice
    row['foreground_polygons_sample'] = holes + [c.reshape(-1, 2).tolist() for c in foreground if len(c) >= 3]
    row['display_mask_provenance'] = dict(source='saved_actual_typed_scene_masks', source_pts=observation['pts'],
        mask_exports=exports, purpose='display_clipping_only_not_geometric_validation',
        model_extent_unchanged=True, holes_explicitly_excluded=True)
    return True
