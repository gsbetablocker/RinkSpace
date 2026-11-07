"""Fixed saved-candidate identity checks; never infer identity from neighbours."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .rebuild_prepare import digest
from .skate_binding import TEMPLATE_VERSION, BINDING_VERSION


IDENTITY_FIELDS = (
    'base_skate_parameters_json', 'R_FB_initial', 't_FB_initial_solver',
    'template_version', 'binding_version', 'parent_hand_parameters_sha256',
    'parent_timing_parameters_sha256', 'camera_binding_json',
)
# Locations and request/override notes are provenance, not model identity.
CAMERA_LOCATION_FIELDS = {
    'record', 'state_path', 'native_library', 'requested_native_library',
    'recorded_native_library', 'native_library_override', 'prepared',
}


def checkpoint_identity(binding, project, camera_info):
    programme = Path(project)/'events/ankara_2026/programmes/okada_mei_sp'
    dependency=getattr(binding.reference,'identity_dependencies',None)
    return dict(
        base_skate_parameters_json=np.array(json.dumps(asdict(binding.base_parameters))),
        R_FB_initial=np.asarray(binding.R0).copy(),
        t_FB_initial_solver=np.asarray(binding.t0).copy(),
        template_version=np.array(TEMPLATE_VERSION), binding_version=np.array(BINDING_VERSION),
        parent_hand_parameters_sha256=np.array(digest(programme/'opening_hand_1103.npz')if dependency is None else dependency['hand_sha256']),
        parent_timing_parameters_sha256=np.array(digest(programme/'opening_timing_1103.npz')if dependency is None else dependency['timing_sha256']),
        camera_binding_json=np.array(json.dumps(camera_info, sort_keys=True)),
    )


def _json(value):
    scalar = np.asarray(value)
    if scalar.shape != ():
        raise ValueError('Expected scalar JSON identity')
    return json.loads(str(scalar))


def validate_checkpoint_identity(saved, expected=None):
    """Require recorded fixed fields; compare content exactly, ignoring paths.

    With no expected identity this is only an early completeness check. A bare
    legacy packet is rejected: directory adjacency, a kernel digest or the
    current model alone cannot authenticate its missing fixed inputs.
    """
    missing = [key for key in IDENTITY_FIELDS if key not in saved]
    if missing:
        raise ValueError('Checkpoint fixed identity missing fields: '+', '.join(missing)+
            '; no checksum-bound fixed identity supplied; neighbouring sidecars are not proof')
    if expected is None:
        return
    mismatched = []
    for key in IDENTITY_FIELDS:
        a, b = saved[key], expected[key]
        try:
            if key.endswith('_json'):
                left, right = _json(a), _json(b)
                if key == 'camera_binding_json':
                    left = {k: v for k, v in left.items() if k not in CAMERA_LOCATION_FIELDS}
                    right = {k: v for k, v in right.items() if k not in CAMERA_LOCATION_FIELDS}
                equal = left == right
            elif key in ('R_FB_initial', 't_FB_initial_solver'):
                left, right = np.asarray(a), np.asarray(b)
                equal = (left.dtype == right.dtype and left.shape == right.shape and
                         np.isfinite(left).all() and left.tobytes() == right.tobytes())
            else:
                equal = np.asarray(a).shape == () and str(np.asarray(a)) == str(np.asarray(b))
        except (TypeError, ValueError, AttributeError):
            equal = False
        if not equal:
            mismatched.append(key)
    if mismatched:
        raise ValueError('Checkpoint fixed identity mismatch: '+', '.join(mismatched))
