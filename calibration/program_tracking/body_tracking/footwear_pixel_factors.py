"""Native-pixel body and footwear factors with local image Jacobians.

Scalar contour distances keep the historical squared objective. Blade side is
one declared finite-edge candidate per curve, never a per-point side switch.
"""
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from .footwear_fit import sample_polyline


def nearest(points,line,closed=False):
    p=np.asarray(points,float);a=np.asarray(line,float);b=np.roll(a,-1,axis=0) if closed else a[1:]
    if not closed:a=a[:-1]
    v=b-a;den=np.maximum(np.sum(v*v,1),1e-16);alpha=np.clip(np.einsum('mkc,kc->mk',p[:,None]-a[None],v)/den,0,1)
    delta=a[None]+alpha[:,:,None]*v[None]-p[:,None];squared=np.sum(delta*delta,2);which=np.argmin(squared,1);rows=np.arange(len(p));norm=np.sqrt(squared[rows,which]);unit=np.divide(delta[rows,which],norm[:,None],out=np.zeros((len(p),2)),where=norm[:,None]>1e-14)
    return norm,which,alpha[rows,which],unit


def pixel_factors(uv,problem,body=True,blade_side_candidates=True):
    """uv=(N,113,2); return residual, dres/d(flat uv), parts, choices."""
    residual=[];rr=[];cc=[];vv=[];parts={};offset=0;choices=[]
    def add(values,name,columns,derivative):
        nonlocal offset
        value=np.asarray(values,float).ravel();residual.append(value);parts[name]=parts.get(name,0.)+float(value@value/2)
        columns=np.asarray(columns,int);derivative=np.asarray(derivative,float)
        rr.extend(np.repeat(np.arange(offset,offset+len(value)),columns.shape[1]));cc.extend(columns.ravel());vv.extend(derivative.ravel());offset+=len(value)
    def curve(observed,fi,ids,scale,name,closed=False):
        distance,k,a,u=nearest(observed,uv[fi,ids],closed);end=(k+1)%len(ids);pairs=np.stack((ids[k],ids[end]),1);columns=((fi*113+pairs[:,:,None])*2+np.arange(2)[None,None]).reshape(len(k),4)
        derivative=np.concatenate(((1-a[:,None])*u,a[:,None]*u),axis=1)/scale
        add(distance/scale,name,columns,derivative)
    for row in problem.visible_rows:
        if row.get('reverse_boundary',True):raise ValueError('Joint experiment requires explicitly partial source-only contours')
        fi=row['frame'];base=row['foot']*48;ids=np.arange(14)+base;h=cv2.convexHull(uv[fi,ids].astype(np.float32),returnPoints=False).ravel()
        curve(row['observed'],fi,ids[h],row['sigma_px']*np.sqrt(48),'visible_boot_edges',True)
    for row in problem.blade_rows:
        fi,foot=row['frame'],row['foot'];source=sample_polyline(row['uv'],24);candidates=[np.arange(14,31)+foot*48]
        if blade_side_candidates:candidates.append(np.arange(31,48)+foot*48)
        best=min(range(len(candidates)),key=lambda i:float(np.sum(nearest(source,uv[fi,candidates[i]])[0]**2)));ids=candidates[best];choices.append(dict(pts=row['pts'],foot=foot,selected_candidate='blade_left' if best==0 else 'blade_right',source_side_confirmed=False))
        curve(source,fi,ids,row['sigma_px']*np.sqrt(24),'visible_blade_curves')
        if not row['partial']:
            # Reverse curve coverage is not admitted for the current controlled
            # joint target; current complete curves are evaluated as source
            # lower edges with finite endpoints, with this change recorded.
            raise ValueError('Declare source-only blade segments explicitly; physical endpoint coverage is not assumed')
    for row in problem.landmark_rows:
        fi,foot=row['frame'],row['foot']
        for name,pair in [('heel',[0,7]),('toe',[1,8])]:
            ids=np.asarray(pair)+foot*48;a,b=uv[fi,ids];v=b-a;point=np.asarray(row[name]);den=max(float(v@v),1e-16);raw=float((point-a)@v/den);alpha=np.clip(raw,0,1);q=a+alpha*v;Ja=(1-alpha)*np.eye(2);Jb=alpha*np.eye(2)
            if 0<raw<1:
                da=(-(point-a)+(2*alpha-1)*v)/den;db=((point-a)-2*alpha*v)/den;Ja+=np.outer(v,da);Jb+=np.outer(v,db)
            columns=np.tile(((fi*113+ids[:,None])*2+np.arange(2)).ravel(),(2,1));derivative=np.c_[Ja,Jb]/row['sigma_px'];add((q-point)/row['sigma_px'],'boot_direction_endpoints',columns,derivative)
    if body:
        ids=problem.body_problem.indices;frames=problem.frames;qualified=problem.p['source_qualified'][frames][:,ids]&(~problem.body_problem.hold[frames,None])
        fi,ki=np.nonzero(qualified);point_ids=ids[ki]+96;error=(uv[fi,point_ids]-problem.p['source_uv'][frames[fi],ids[ki]])/problem.body_problem.sig[ki,None];columns=((fi*113+point_ids)[:,None]*2+np.arange(2)).reshape(-1,1);weight=np.repeat(1/problem.body_problem.sig[ki],2)[:,None];add(error,'body_pixels',columns,weight)
    vector=np.concatenate(residual);A=coo_matrix((vv,(rr,cc)),shape=(len(vector),uv.size)).tocsr();return vector,A,parts,choices
