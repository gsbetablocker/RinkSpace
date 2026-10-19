"""Import compatibility for immutable historical calibration source packets.

The maintained implementation lives only in project/calibration. This package
provides the former import namespace without duplicating its code or data.
"""
from pathlib import Path

_project = Path(__file__).resolve().parents[2]
__path__ = [str(path) for path in (_project / "calibration", _project / "three_frame_refit")
            if path.is_dir()]
