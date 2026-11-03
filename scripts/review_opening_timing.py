"""Review saved opening phase timing, optionally from an accepted checkpoint."""
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project/'src'), str(project)]
from calibration.program_tracking.body_tracking.opening_timing_review import main

if __name__ == '__main__':
    main()