"""Use run_body_performance.py to select the verified execution profile."""
from pathlib import Path
import os,sys
project=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(project),str(project/'src')]
from rinkspace.core.native import NativeCore
NativeCore(os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY',
    'D:/CodexData/RinkSpace_Data/build/solver_continue_1102/rinkspace_core.dll'))
from calibration.program_tracking.body_tracking.skate_integration import main
if __name__=='__main__':main()
