from pathlib import Path
import json,sys
project=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(project/'src'),str(project)]
if __name__=='__main__':
    from rinkspace.core.native import NativeCore
    record=json.loads((project/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.json').read_text(encoding='utf8'))
    held_core=NativeCore(record['native_library']['path'])
    from calibration.program_tracking.body_tracking.boot_integration import main
    main()
