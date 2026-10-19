"""Explicit reader for the distinct official-exposure and camera-pose keys.

The historical quadratic reader remains unchanged on disk. This serial helper
uses the new pose key only while reconstructing the model; the original NPZ's
official exposure identities are retained as a separate array and never edited.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import audit_quadratic_result as original


def reconstruct(directory, *, state_name, require_quadratic=True):
    directory = Path(directory).resolve()
    phase = json.loads((directory/'phase_00.json').read_text(encoding='utf8'))
    if (not require_quadratic or not phase.get('source_pose_groups')
            or not phase.get('source_camera_rebinding')):
        raise ValueError('Expected explicit source-image-pose quadratic graph')
    bindings = phase['source_camera_rebinding']['source_binding_metadata']['frame_key_by_source_pts']
    previous = original._camera_packet
    def camera_packet(path):
        packet = previous(path)
        if Path(path).resolve().parent != directory:
            raise ValueError('Unexpected camera template during pose-clock reconstruction')
        official = packet['official_exposure_representative_pts']
        poses = packet['source_pose_representative_pts']
        expected = np.asarray([bindings[str(int(p))] for p in packet['pts']], np.int64)
        if (not np.array_equal(packet['exposure_representative_pts'], official)
                or not np.array_equal(poses, expected)):
            raise ValueError('Official identity and model pose binding differ from saved metadata')
        packet['exposure_representative_pts'] = poses.copy()
        return packet
    original._camera_packet = camera_packet
    try:
        result = original.reconstruct(directory, state_name=state_name,
                                      require_quadratic=True)
    finally:
        original._camera_packet = previous
    result[-1]['provenance']['reconstruction_pose_key_field'] = 'source_pose_representative_pts'
    result[-1]['provenance']['official_exposure_identity_unchanged_on_disk'] = True
    return result
