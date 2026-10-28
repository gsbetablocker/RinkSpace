"""Deterministic synthetic data with known truth, only for executable tests."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import project_rays, K


def make_graph(n=7, points=100, noise=0., pure_zoom=False, radial=0., aspect=1.18):
    rng = np.random.default_rng(101)
    w, h = 1280, 720
    angles = np.array([[0,0,0],[.09,.07,.01],[-.07,.12,-.02],[.12,-.09,.03],
                       [-.12,-.07,-.02],[.04,.17,.01],[-.08,-.15,.03]])[:n]
    if pure_zoom: angles[:] = 0
    R = Rotation.from_rotvec(angles).as_matrix()
    optics = np.zeros((n,8)); f = np.array([1300,1450,1200,1550,1350,1250,1400])[:n]
    optics[:,0]=f*aspect;optics[:,1]=f;optics[:,2]=602;optics[:,3]=378
    optics[:,4]=radial*(f/w)**2
    rays=np.column_stack((rng.uniform(-.18,.18,points),rng.uniform(-.085,.085,points),np.ones(points)))
    uv = np.array([project_rays(rays@r.T,o)[0] for r,o in zip(R,optics)])
    uv += rng.normal(0,noise,uv.shape)
    arrays={k:[] for k in ('i','j','a','b','test','screened','track','edge')};edges=[]
    for i in range(n):
        for j in range(i+1,min(n,i+4)):
            e=len(edges);H=K(optics[j])@R[j]@R[i].T@np.linalg.inv(K(optics[i]))
            edges.append(dict(i=i,j=j,H=H))
            for k,v in [('i',np.full(points,i)),('j',np.full(points,j)),('a',uv[i]),('b',uv[j]),
                        ('test',np.arange(points)%4==0),('screened',np.ones(points,bool)),
                        ('track',np.arange(points)),('edge',np.full(points,e))]:arrays[k].append(v)
    d={k:np.concatenate(v) for k,v in arrays.items()}
    meta=dict(width=w,height=h,frames=[dict(pts=i*256,time_base='1/12800') for i in range(n)],edges=edges,
        candidate_frame_count=n,training_tracks=int((np.arange(points)%4!=0).sum()),
        holdout_tracks=int((np.arange(points)%4==0).sum()))
    return dict(data=d,meta=meta),optics,R
