"""A rigid boot shell and separate curved blade in each ankle frame.

Shape and fixed ankle alignment are shared through the sequence. Anatomical
joint corrections move the equipment; neither boot shape nor a free boot XYZ
is independently optimized per frame. All metric values are conditional.
"""
import numpy as np
from scipy.spatial.transform import Rotation

NAMES=('shoe_rear','shoe_front','shoe_width','ankle_to_sole','upper_height','blade_drop','blade_rear_extension','blade_front_extension','rocker_rise','toe_rise','boot_align_x','boot_align_y','boot_align_z','blade_pitch')
INITIAL=np.r_[np.log([.065,.165,.082,.082,.045,.040,.012,.015,.012,.012]),np.zeros(4)]
# Positive geometry and moderate angular charts are numerical/model domains,
# not measured skate specifications. Report any active boundary.
LOWER=np.r_[np.log([.025,.08,.04,.025,.015,.01,.002,.002,.001,.001]),np.deg2rad([-35,-35,-35,-15])]
UPPER=np.r_[np.log([.12,.26,.14,.16,.12,.09,.055,.055,.035,.05]),np.deg2rad([35,35,35,15])]
LEG_DOFS=((1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(4,0),(5,0),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2))


def decode(parameters):
    p=np.asarray(parameters,float).copy();p[:10]=np.exp(p[:10]);return dict(zip(NAMES,p.tolist()))


def local_vertices(parameters):
    p=decode(parameters);back,front,width,H,up,drop,be,fe,rocker,toerise=[p[n] for n in NAMES[:10]]
    # Convex side profile with toe, instep, mouth and heel, extruded to width.
    yz=np.array([[-H,-back],[-H+toerise,front],[-H+toerise+.035,front],[-H*.35,front*.30],[up,.015],[up,-back*.8],[up*.4,-back]])
    fractions=np.array([.43,.44,.44,.50,.43,.43,.43]);shell=[]
    for side in [-1,1]:shell.extend(np.c_[side*width*fractions,yz])
    shell=np.asarray(shell);stations=np.linspace(0,1,17);z=-back-be+(back+front+be+fe)*stations
    y=-H-drop+rocker*(2*stations-1)**2+(z-(front-back)/2)*np.tan(p['blade_pitch'])
    blade=np.concatenate([np.c_[np.full(17,side*.002),y,z] for side in [-1,1]],axis=0)
    return np.concatenate((shell,blade)),dict(shell=np.arange(14),blade=np.arange(14,48),blade_left=np.arange(14,31),blade_right=np.arange(31,48))


def boot_alignment(parameters,foot):
    a=np.asarray(parameters[10:13],float)*([1,1,1] if foot==0 else [1,-1,-1])
    return Rotation.from_rotvec(a).as_matrix()


def extra_rotations(values):
    values=np.asarray(values);out={j:np.zeros((len(values),3)) for j,_ in LEG_DOFS}
    for d,(j,axis) in enumerate(LEG_DOFS):out[j][:,axis]=values[:,d]
    return out


def equipment_world(P,Q,joints,joint_frames,parameters):
    vertices,parts=local_vertices(parameters);n=len(P);world=np.empty((n,2,len(vertices),3));rotation=np.empty((n,2,3,3));origin=np.empty((n,2,3))
    for foot,joint in enumerate([7,8]):
        rotation[:,foot]=Q@joint_frames[:,joint]@boot_alignment(parameters,foot)
        origin[:,foot]=P+np.einsum('nij,nj->ni',Q,joints[:,joint])
        world[:,foot]=origin[:,foot,None]+np.einsum('nij,kj->nki',rotation[:,foot],vertices)
    return world,parts,rotation,origin


def lowest_blade_world(rotation,origin,parameters):
    """Exact minimum of two parabolic lower edges; neither end is forced flat."""
    p=decode(parameters);back,front,H,drop,be,fe,rocker=[p[n] for n in ['shoe_rear','shoe_front','ankle_to_sole','blade_drop','blade_rear_extension','blade_front_extension','rocker_rise']]
    length=back+front+be+fe;z0=-back-be;zc=(front-back)/2;slope=np.tan(p['blade_pitch']);r=np.asarray(rotation)
    A=r[...,2,1]*4*rocker;B=r[...,2,1]*(-4*rocker+length*slope)+r[...,2,2]*length
    stationary=np.divide(-B,2*A,out=np.zeros_like(B),where=abs(A)>1e-12)
    stations=np.stack((np.zeros_like(B),np.ones_like(B),np.clip(stationary,0,1)),axis=-1)
    candidates=[]
    for side in [-1,1]:
        z=z0+length*stations;y=-H-drop+rocker*(2*stations-1)**2+(z-zc)*slope
        local=np.stack((np.full_like(z,side*.002),y,z),axis=-1)
        candidates.append(origin[...,None,:]+np.einsum('...ij,...kj->...ki',r,local))
    candidates=np.concatenate(candidates,axis=-2);best=np.argmin(candidates[...,2],axis=-1)
    return np.take_along_axis(candidates,best[...,None,None],axis=-2)[...,0,:]
