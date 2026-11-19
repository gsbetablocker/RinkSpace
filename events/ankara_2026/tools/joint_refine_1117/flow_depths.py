"""Conditional flow reference depths, rebound to current native source cameras.

These are inverse-projection initializations, never measured 3D landmarks. The
native rolling-row forward must reproduce each unchanged left source pixel.
"""
import numpy as np


def rays(uv,camera):
    target=(uv-np.c_[camera['cx'],camera['cy']])/np.c_[camera['fx'],camera['fy']]
    point=target.copy()
    k1,k2,p,q=(camera[k] for k in ('k1','k2','p1','p2'))
    valid=camera['status']==0
    for _ in range(30):
        x,y=point.T;r2=x*x+y*y;rad=1+r2*(k1+r2*k2);d=k1+2*k2*r2
        prediction=np.c_[x*rad+2*p*x*y+q*(r2+2*x*x),y*rad+p*(r2+2*y*y)+2*q*x*y]
        aa=rad+2*x*x*d+2*p*y+6*q*x;ab=2*x*y*d+2*p*x+2*q*y;bb=rad+2*y*y*d+6*p*y+2*q*x
        det=aa*bb-ab*ab;good=(aa>0)&(bb>0)&(det>1.e-12)&np.isfinite(det)
        valid&=good;den=np.where(good,det,1.)
        dx,dy=(target-prediction).T
        update=np.c_[(bb*dx-ab*dy)/den,(aa*dy-ab*dx)/den]
        update=np.where(valid[:,None],np.clip(update,-.18,.18),0.)
        point+=update
        if np.max(abs(update),initial=0)<1.e-11:break
    ray=np.einsum('nji,nj->ni',camera['rotation'],np.c_[point,np.ones(len(point))])
    ray[~valid]=np.nan
    return ray


def board_hit(centre,ray,boundary,height):
    origin=boundary[:-1,:2];edge=np.diff(boundary[:,:2],axis=0)
    answer=np.full_like(centre,np.nan)
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for lo in range(0,len(centre),128):
        c=centre[lo:lo+128];r=ray[lo:lo+128];offset=origin[None]-c[:,None,:2]
        den=cross(r[:,None,:2],edge[None])
        with np.errstate(invalid='ignore',divide='ignore'):
            distance=cross(offset,edge[None])/den;station=cross(offset,r[:,None,:2])/den
            z=c[:,None,2]+distance*r[:,None,2]
        good=(abs(den)>1.e-14)&(distance>0)&(station>=0)&(station<=1)&(z>0)&(z<height)
        closest=np.min(np.where(good,distance,np.inf),axis=1);ok=np.isfinite(closest)
        answer[lo+np.flatnonzero(ok)]=c[ok]+closest[ok,None]*r[ok]
    return answer


def bind(native,shot,rigs,controls,times,index,uv,family):
    board=native.scene['board'];core=native.core
    stations=np.linspace(0,core.board_perimeter(board),2049)
    foot=np.asarray(core.board_points(board,stations,np.zeros(len(stations))))
    world=np.full((len(uv),3),np.nan);error=np.full(len(uv),np.inf)
    unique,inv=np.unique(rigs[index],axis=0,return_inverse=True)
    for i,rig in enumerate(unique):
        ix=np.flatnonzero(inv==i);t=times[index[ix]]
        row=t+rig[37]*(uv[ix,1]/1079.-.5)
        cam=core.evaluate_motion_camera(rig,controls,shot['first'],shot['dt'],row)
        ray=rays(uv[ix],cam);isboard=family[ix]==0
        value=np.full((len(ix),3),np.nan)
        value[isboard]=board_hit(cam['centre'][isboard],ray[isboard],foot,float(board[5]))
        with np.errstate(invalid='ignore',divide='ignore'):
            depth=-cam['centre'][~isboard,2]/ray[~isboard,2]
            value[~isboard]=cam['centre'][~isboard]+depth[:,None]*ray[~isboard]
        valid=np.isfinite(value).all(1)&(cam['status']==0)
        ice=np.flatnonzero(~isboard)
        valid[ice]&=(depth>0)&(abs(value[ice,0])<board[0])&(abs(value[ice,1])<board[1])
        check=np.flatnonzero(valid)
        if len(check):
            z=core.project_motion(rig,controls,shot['first'],shot['dt'],1080,t[check],value[check])
            e=np.linalg.norm(z['uv']-uv[ix[check]],axis=1)
            accepted=(z['status']==0)&(e<.05)
            world[ix[check[accepted]]]=value[check[accepted]];error[ix[check[accepted]]]=e[accepted]
    return world,error
