"""Construct the pinned BodyNet+HandMR / FootMR-off camera prior."""
from pathlib import Path
import sys

# Support the same repository-root invocation as the other body entry points.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.program_tracking.body_tracking.exmr_hand_only import main

if __name__ == "__main__":
    main()
