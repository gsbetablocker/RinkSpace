from pathlib import Path
import sys,json,time,io
import numpy as np
from scipy.sparse import diags
project=Path(r'D:/OneDrive/codex_projs/RinkSpace');sys.path[:0]=[str(project),str(project/'src')]
from calibration.program_tracking.body_tracking.action_integration import initialize_clip
from calibration.program_tracking.body_tracking.skate_integration import JointContactProblem,source_evidence,NORMALIZATION
from calibration.program_tracking.body_tracking.action_live import GlideActionFactors
from calibration.program_tracking.body_tracking.ice_soft import ice_soft_rows
from calibration.program_tracking.body_tracking.fullbody_round import write_json
base=Path(r'D:/CodexData/RinkSpace_Data/runs/body_s01_20261029');root=base/'action_integration_1103';clip='jump_s02';out=root/clip
e=json.loads((project/'events/ankara_2026/programmes/okada_mei_sp/action_evidence_1103.json').read_text(encoding='utf-8'))['clips'][clip]
r,c,info=initialize_clip(project,base,out,root/'source_cache/jump_s02.native50.npz',out/'prior/camera_prior.npz','s02',heldout_pts=e['heldout_pts'])
sup=source_evidence(out,shot_id='s02',source_pts=r.p['pts'],records=e['supports'])
seconds=(r.p['pts'][0]/12800.-.01,r.p['pts'][-1]/12800.+.01)
p=JointContactProblem(r,c,out/'observations.npz',seconds,sup,out/'derivative_check',io.StringIO(),'continuous_shot',optimize_shared=False)
p.action_factors=GlideActionFactors(p,e,sigma_mps=e['glide_sigma_mps'],enabled=True)
x=np.r_[r.binding.x,np.zeros(p.body.size)];rng=np.random.default_rng(1103);v=rng.normal(size=p.body.size);v/=np.linalg.norm(v);h=1e-5
start=time.perf_counter();a=p.action_factors.evaluate(x[17:],True);pair=[p.action_factors.evaluate(x[17:]+sign*h*v,False)['residual']for sign in (1,-1)]
fd=(pair[0]-pair[1])/(2*h);pred=a['jac'][:,17:]@v
err=dict(action_relative=float(np.linalg.norm(fd-pred)/max(np.linalg.norm(fd),1e-12)),action_max_abs=float(abs(fd-pred).max()),action_metrics=a['metrics'])
raw=p.contact(x,True);ice=ice_soft_rows(raw,p.ice_policy);J=diags(ice['derivative_per_margin_m']*NORMALIZATION)@raw['body_jac'][ice['indices']]
pair=[]
for sign in (1,-1):
 z=x.copy();z[17:]+=sign*h*v;pair.append(ice_soft_rows(p.contact(z),p.ice_policy)['residual'])
fd=(pair[0]-pair[1])/(2*h);pred=J@v
err.update(ice_relative=float(np.linalg.norm(fd-pred)/max(np.linalg.norm(fd),1e-12)),ice_max_abs=float(abs(fd-pred).max()))
err.update(seconds=time.perf_counter()-start)
write_json(root/'live_derivative_check.json',err);print(json.dumps(err),flush=True)
assert err['action_relative']<.01 and err['ice_relative']<.001
