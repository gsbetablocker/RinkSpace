"""Load the selected native runtime first, then reproduce the cached S01 round."""
from pathlib import Path
import argparse,json,os,sys
from datetime import datetime,timezone

project=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(project/'src'),str(project)]


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    bootstrap=argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--project',default=str(project));bootstrap.add_argument('--output')
    known,_=bootstrap.parse_known_args(argv)
    if '-h' in argv or '--help' in argv:
        from calibration.program_tracking.body_tracking.fullbody_round import main as round_main
        return round_main(argv)
    output=Path(known.output).resolve() if known.output else None
    log_path=None
    if output is not None:
        from rinkspace.path_policy import guard_project_write
        log_path=guard_project_write(output/'run_progress.jsonl')
        guard_project_write(output/'full/w5/input_signature.json.pending')
        output.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(main_log=str(log_path),fit_log=str(output/'fit_progress.jsonl'))),flush=True)
        with log_path.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(event='launcher_start',utc=datetime.now(timezone.utc).isoformat(),argv=argv))+'\n')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
    try:
        from rinkspace.core.native import NativeCore
        record_path=Path(known.project)/'events/ankara_2026/programmes/okada_mei_sp/fit_v11/camera_solver_refined_1102.json'
        record=json.loads(record_path.read_text(encoding='utf8'))
        held_core=NativeCore(record['native_library']['path'])
        from calibration.program_tracking.body_tracking.fullbody_round import main as round_main
        return round_main(argv,log_announced=log_path is not None)
    except BaseException as error:
        if log_path is not None:
            with log_path.open('a',encoding='utf8') as stream:
                stream.write(json.dumps(dict(event='launcher_failed',utc=datetime.now(timezone.utc).isoformat(),error=repr(error)))+'\n')
        raise


if __name__=='__main__':main()
