"""Run a complete source event through the common A/B body/skate adapter."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root)]
from calibration.program_tracking.body_tracking.action_integration import main
if __name__=='__main__':main()
