"""Bind, verify and cache an already fitted opening candidate."""
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(project/'src'), str(project)]
from rinkspace.core.native import NativeCore
core = NativeCore('D:/CodexData/RinkSpace_Data/build/solver_continue_1102/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.opening_motion_delivery import main

if __name__ == '__main__':
    main()
