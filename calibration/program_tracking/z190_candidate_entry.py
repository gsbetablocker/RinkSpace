"""Load this task's standing-tripod candidate without switching production.

Sparse source keys only. Missing poses require actual source-motion recovery;
they are never filled by interpolation or by assuming 60 exposures per second.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

try:
    from .z190_physical_camera import PhysicalCameraModel, UnknownCalibrationError
except ImportError:
    from z190_physical_camera import PhysicalCameraModel, UnknownCalibrationError


DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'configs/camera_hardware/sony_z190_standing_tripod_candidate_v1.json'


@dataclass(frozen=True)
class StandingCameraCandidate:
    model: PhysicalCameraModel
    source_pts: object
    exposure_aliases: dict
    canonical_views: dict
    optimized_shared_columns: tuple
    accuracy_accepted: bool = False

    def canonical(self, source_pts):
        if source_pts not in self.exposure_aliases:
            raise UnknownCalibrationError(f'No fitted source camera for PTS {source_pts}; recover it from real source evidence')
        return self.canonical_views[self.exposure_aliases[source_pts]]

    def camera25(self, source_pts):
        """Existing renderer/projector camera25, preserving original scene units."""
        return self.model.camera25(self.canonical(source_pts))

    def project_xyz(self, source_pts, xyz, *, with_jacobian=False, require_valid=True):
        """Mesh-fitting factor: shared five columns and exposure R3/logf four.

        XYZ must belong to the declared scene and typed physical surface.
        Source exposure identity is explicit; alias inputs share one state.
        """
        canonical = self.canonical(source_pts)
        calibration = self.model.calibrations[canonical.installation_id]
        result = calibration.project_xyz(canonical.view, xyz, with_jacobian=with_jacobian,
                                         require_valid=require_valid)
        if not with_jacobian:
            return result
        uv, depth, shared, view, valid = result
        return uv, depth, shared[:, self.optimized_shared_columns], view, valid


def load_standing_camera_candidate(config_path=DEFAULT_CONFIG):
    """Read the concrete three-installation package; no files are modified."""
    record = json.loads(Path(config_path).read_text(encoding='utf8'))
    if hashlib.sha256(Path(record['source_scene_path']).read_bytes()).hexdigest() != record['source_scene_sha256']:
        raise ValueError('Standing-camera candidate scene hash changed')
    path = Path(record['candidate_npz_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['candidate_npz_sha256']:
        raise ValueError('Standing-camera numeric package hash changed')
    model, pts, aliases, views = PhysicalCameraModel.load_npz(path)
    expected = PhysicalCameraModel.load_calibration_json(config_path)
    if ({k: v.manifest() for k, v in model.calibrations.items()} !=
        {k: v.manifest() for k, v in expected.calibrations.items()}):
        raise ValueError('Standing-camera entry and numeric calibrations disagree')
    if record['optimized_shared_state_columns'] != [0, 1, 8, 9, 10]:
        raise ValueError('Unexpected optimizer variables for fixed standing candidate')
    return StandingCameraCandidate(model, pts, aliases, views, (0, 1, 8, 9, 10))
