from calibration.program_tracking.extract_first_shot_ice import *
from scipy.optimize import least_squares
ex=Extractor(DATA/'repair_20261014');i=230;s=ex.rows[i]['source'];b=cv2.imread(str(DATA/f'frames/{i:06d}.png'));m=cv2.imread(s['mask_path'],0);bc,_=current_broadcast_exclusion(b);dom,_=current_ice_visible_domain(i);native=NativeProfiles(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),s,(m>0)|bc|dom);r=ex.house_reader;g=r.groups['R3'];rows,d=r.measurements(g,i,native,np.zeros(6));curves={role:r.projection(g,i,role,np.linspace(0,2*np.pi,513),None)[0] for role in g['radii']};conics={role:r.conic(c) for role,c in curves.items()};obs={role:np.array([z['uv'] for z in rows if z['role']==role]) for role in g['radii']};bound=np.array([40.,30.,.06,.05,.025,.08])
def residual(z):
 v=[]
 for role,oo in obs.items():
  if len(oo):v.extend(r.transformed_conic_distance(conics[role],oo,z))
 return np.r_[v,z/np.array([60,40,.12,.1,.045,.15])*.25]
for ds in [None,1e-4,.01]:
 kw={} if ds is None else dict(diff_step=ds)
 f=least_squares(residual,np.zeros(6),bounds=(-bound,bound),loss='soft_l1',f_scale=1.5,max_nfev=100,x_scale=bound,**kw)
 print('ds',ds,'fitted',f.x,'cost',f.cost,'initial',np.median(abs(residual(np.zeros(6)))),'final',np.median(abs(residual(f.x))),'nfev',f.nfev,'message',f.message)
for role in obs:
 print(role,len(obs[role]),'signedinitial',r.transformed_conic_distance(conics[role],obs[role],np.zeros(6)).round(1))

