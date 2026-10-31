"""Explicit fixed delivery modes and shared scalar native camera coordinates.

Sharing is imposed in the native retained directions, not by penalty residuals.
The input numeric state must already lie on the declared shared manifold.
"""
from copy import deepcopy
from pathlib import Path
import json
from numbers import Integral
import numpy as np
from .camera_experiment import CAMERA_NAMES, _SCALAR_STORAGE_INDEX


def read_camera_coupling(path):
    config=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if config.get('schema')!='rinkspace.camera_coupling.v1' or not config.get('id'):
        raise ValueError('Explicit camera coupling schema/id required')
    if config.get('hardware_calibrated') is not False or not config.get('basis'):
        raise ValueError('Coupling needs its engineering basis and calibration status')
    if not isinstance(config.get('groups'),list):raise ValueError('Coupling groups required')
    used=set()
    for g in config['groups']:
        cameras=g['camera_indices'];names=g['shared_names'];mags=g['delivery_magnifications']
        if len(cameras)<2 or len(set(cameras))!=len(cameras) or any(type(i) is not int or i<0 for i in cameras):
            raise ValueError('Distinct nonnegative physical camera indices required')
        if len(mags)!=len(cameras) or not np.isfinite(mags).all() or min(mags)<=0:
            raise ValueError('One fixed positive delivery magnification per camera required')
        if not names or len(set(names))!=len(names) or set(names)-_SCALAR_STORAGE_INDEX.keys():
            raise ValueError('Only named independent scalar native coordinates may be shared')
        if not np.isfinite(g['reference_focal_base_pixels']) or g['reference_focal_base_pixels']<=0 or not np.isfinite(g['wide_boundary_base_log_f']):
            raise ValueError('Fixed optical reference conventions required')
        for i in cameras:
            for name in names:
                if (i,name) in used:raise ValueError('Camera coordinate belongs to two groups')
                used.add((i,name))
    return config


def build_multicamera_directions(rigs,indices,scales,coupling=None):
    """Return camera-only stacked rd and unique-column metadata.

    Native chart rows remain camera-local. Shared scalar rows point to exactly
    the same retained column and step, preserving equality on every retraction.
    """
    rigs=np.asarray(rigs,float);scales=np.asarray(scales,float)
    if not np.isfinite(rigs).all() or rigs.ndim!=2 or rigs.shape[1]!=43 or len(indices)!=len(rigs) or scales.shape!=(31,) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Expected finite rigs[N,43], chart selections and 31 positive scales')
    groups=(coupling or {}).get('groups',[]);owners={};shared_records=[]
    for j,g in enumerate(groups):
        cameras=g['camera_indices']
        if max(cameras)>=len(rigs):raise ValueError('Shared camera index outside current input')
        for i,mag in zip(cameras,g['delivery_magnifications'],strict=True):
            if not np.isclose(rigs[i,24],g['reference_focal_base_pixels']*mag,rtol=2e-13,atol=0):
                raise ValueError('Shared camera focal reference differs from delivery mode')
            if not np.isclose(rigs[i,38],g['wide_boundary_base_log_f']+np.log(mag),rtol=0,atol=2e-13):
                raise ValueError('Shared camera wide basis differs from delivery mode')
        for name in g['shared_names']:
            chart=CAMERA_NAMES.index(name);storage=_SCALAR_STORAGE_INDEX[name]
            active=[chart in indices[i] for i in cameras]
            if any(active) and not all(active):raise ValueError('Select all linked cameras for a shared coordinate')
            if not all(rigs[i,storage]==rigs[cameras[0],storage] for i in cameras):
                raise ValueError('Initial state is outside shared camera manifold: '+name)
            for i in cameras:owners[i,chart]=('shared',j,name)
            shared_records.append(dict(group=g['id'],parameter=name,cameras=cameras,active=all(active)))
    keys={};records=[];placements=[]
    for camera,selected in enumerate(indices):
        if len(set(selected))!=len(selected) or any(isinstance(k,(bool,np.bool_)) or not isinstance(k,Integral) or not 0<=k<31 for k in selected):
            raise ValueError('Invalid selected native camera chart')
        for chart in selected:
            key=owners.get((camera,chart),('camera',camera,chart))
            if key not in keys:
                keys[key]=len(keys);records.append(dict(column=keys[key],kind=key[0],
                    parameter=CAMERA_NAMES[chart],camera_indices=[]))
            column=keys[key];records[column]['camera_indices'].append(camera)
            placements.append((camera*31+chart,column,scales[chart]))
    rd=np.zeros((len(rigs)*31,len(keys)))
    for row,column,value in placements:rd[row,column]=value
    return rd,dict(unique_camera_columns=len(keys),active_camera_chart_counts=list(map(len,indices)),
        shared_coordinates=shared_records,column_records=records,
        coupling=deepcopy(coupling),independent_camera_parameters=not bool(groups))


def change_zoom_reference(rig,new_fref):
    """Preserve the physical pupil/centre function under z=log(f/fref)."""
    result=np.array(rig,float,copy=True)
    if result.shape!=(43,) or not np.isfinite(new_fref) or new_fref<=0:
        raise ValueError('Finite 43-value rig and positive reference focal required')
    delta=np.log(new_fref/result[24])
    result[20]+=result[22]*delta+result[23]*delta*delta
    result[22]+=2*result[23]*delta
    # Existing optical functions are re-expressed exactly at the new reference.
    result[26]+=result[29]*delta;result[27]+=result[30]*delta
    result[31]+=result[32]*delta;result[33]+=result[34]*delta
    result[24]=new_fref
    return result
