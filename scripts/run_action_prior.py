"""Run the shared real GVHMR clip adapter in the existing body environment."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root)]
from calibration.program_tracking.body_tracking.action_prior import main
if __name__=='__main__':main()
