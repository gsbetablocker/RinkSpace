"""Stable third-stage CLI: strict L-infinity repair then same-delta W audit.
The returned report includes true final-direction W coefficients. This
wrapper never runs a physical trial or imports an old LP dual face.
"""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--qp-report',required=True);p.add_argument('--core-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;out=Path(a.output).resolve();core=Path(a.core_directory).resolve();qp=Path(a.qp_report).resolve()
first=subprocess.run([sys.executable,str(here/'repair_fixed_entity_W_infinity.py'),'--qp-report',str(qp),'--core-directory',str(core),'--output',str(out)])
if first.returncode:raise SystemExit(first.returncode)
raw=out/'feasibility_report.json';(out/'report.json').rename(raw)
second=subprocess.run([sys.executable,str(here/'finalize_fixed_entity_candidate.py'),'--repair-report',str(raw),'--core-directory',str(core)])
if second.returncode:raise SystemExit(second.returncode)
data=(out/'final_report.json').read_bytes();(out/'report.json').write_bytes(data)
print(json.dumps(dict(report=dict(path=str(out/'report.json'),sha256=hashlib.sha256(data).hexdigest()),
 same_delta_feasibility_report=str(raw),final_W_coefficients_verified=True,no_physical_trial=True)),flush=True)
