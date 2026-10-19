"""Explicit source-audit target initializer; historical compute.json is unchanged."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from rinkfit_start.target_prepare import run as prepare

def run(checkpoint,context,output,stop_file=None,policy=None):
    path=Path(policy)if policy else Path(__file__).with_name('compute_source1018.json')
    return prepare(checkpoint,context,output,stop_file,policy=json.loads(path.read_text(encoding='utf8')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--context',required=True);p.add_argument('--output',required=True);p.add_argument('--stop-file');p.add_argument('--policy');a=p.parse_args();run(a.checkpoint,a.context,a.output,a.stop_file,a.policy)
