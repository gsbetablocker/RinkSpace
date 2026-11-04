from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'src')]
from calibration.program_tracking.body_tracking.support_refinement import main
if __name__=='__main__':main()
