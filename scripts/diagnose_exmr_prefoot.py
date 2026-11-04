from pathlib import Path
import argparse
from calibration.program_tracking.body_tracking.exmr_integration import make_case,save_packet
from calibration.program_tracking.body_tracking.exmr_prior import write_json
from calibration.program_tracking.body_tracking.support_refinement import measure

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();project=Path(__file__).resolve().parents[1];base=Path(a.base_root);output=Path(a.output)
    case=output/'common/exmr_prefoot';case.mkdir(parents=True,exist_ok=True)
    with (case/'diagnostic_progress.jsonl').open('a',encoding='utf8',buffering=1) as log:
        problem,review,x,info,supports,document=make_case(project,base,output,'exmr_prefoot',log)
        value=problem.evaluate(x,False)
        packet=save_packet(problem,info,supports,document,case/'initial',status='common_initial')
        write_json(case/'diagnostic.json',dict(mode='read_only_no_optimization',accepted_steps=0,
            gate=value['gate'],metrics=measure(problem,review,x,value),physical_acceptance=False,
            result=str(case/'initial/fitted/result.npz')))

if __name__=='__main__':main()