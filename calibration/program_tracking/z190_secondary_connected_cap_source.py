"""Opt-in finite cap domains without an invented source world station.

The old confirmed lower scans retain their original straight-side operator.
New source-only upper scans may select the existing connected lower straight,
lower-left corner or left short side. No opposite perimeter edge is admitted.
"""
import numpy as np
from z190_secondary_confirmed_nearboard_source import NEAR_KIND
from z190_nearboard_analytic_source import analytic_nearboard_absolute_value_jac15


def connected_lower_left_cap(scene,max_world_step=.25):
    top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    x0,y0,x1,y1=map(float,scene['parameters']['bounds'])
    rx,ry=map(float,scene['parameters']['corner_curve']['tangent_extent_xy_scene_units'])
    tol=1e-7
    keep=(np.abs(top[:,0]-x0)<=tol)|((top[:,0]<=x0+rx+tol)&(top[:,1]<=y0+ry+tol))|(np.abs(top[:,1]-y0)<=tol)
    indices=np.flatnonzero(keep)
    if len(indices)<5 or not np.all(np.diff(indices)==1):
        raise ValueError('Connected physical leftshort/corner/lowerstraight cap is not a single current boundary chain')
    line=top[indices];world=[line[0]]
    for a,b in zip(line[:-1],line[1:]):
        n=int(np.ceil(np.linalg.norm(b-a)/max_world_step))
        world.extend(a+(b-a)*i/n for i in range(1,n+1))
    world=np.asarray(world,float)
    if not np.allclose(world[0,:2],[x0,y1-ry],atol=tol,rtol=0)or not np.allclose(world[-1,:2],[x1-rx,y0],atol=tol,rtol=0):
        raise ValueError('Connected cap starts/ends outside the declared physical chain')
    return world,dict(physical_facets=['left_short_side','lower_left_corner','lower_long_straight'],
        original_current_board_top_vertex_indices=[int(indices[0]),int(indices[-1])],
        original_vertices=len(line),subdivided_vertices=len(world),maximum_world_step_scene_units=max_world_step,
        source_station_or_tangent_not_assigned=True,opposite_perimeter_edges_excluded=True)


def connected_cap_absolute_value_jac15(camera,rows,jacobian=True):
    values=[];blocks=[]
    for row in rows:
        if row.get('curve_observation_kind')!=NEAR_KIND or 'upper_world'not in row:
            result=analytic_nearboard_absolute_value_jac15(camera,[row],jacobian)
            if jacobian:value,matrix=result;values.append(value);blocks.append(matrix)
            else:values.append(result)
            continue
        count=len(row['uv']);value=np.zeros((count,2));matrix=np.zeros((count,2,15))if jacobian else None
        upper=np.array(['upper_source_packet'in m for m in row['source_horizontal_intervals']],bool)
        for mask,world in ((~upper,row['world']),(upper,row['upper_world'])):
            index=np.flatnonzero(mask)
            if not len(index):continue
            part=dict(row,world=world,uv=np.asarray(row['uv'])[index],sigma=np.asarray(row['sigma'])[index],
                source_horizontal_intervals=[row['source_horizontal_intervals'][i]for i in index])
            result=analytic_nearboard_absolute_value_jac15(camera,[part],jacobian)
            if jacobian:
                v,j=result;value[index]=v.reshape(-1,2);matrix[index]=j.reshape(-1,2,15)
            else:value[index]=result.reshape(-1,2)
        values.append(value.ravel())
        if jacobian:blocks.append(matrix.reshape(-1,15))
    result=np.concatenate(values)if values else np.empty(0)
    if not jacobian:return result
    return result,np.vstack(blocks)if blocks else np.empty((0,15))
