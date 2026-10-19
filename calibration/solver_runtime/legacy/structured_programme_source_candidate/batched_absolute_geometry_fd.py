"""Batch exact-input projections inside the existing absolute geometry FD.

The original material rebinder, source operator and FD width remain in use.
Projection reuse is limited to this camera/group/FD evaluation; missing exact
coordinates fall back to the original projector. No source graph or q is owned.
"""
from contextlib import nullcontext
from collections import OrderedDict

import numpy as np

from projection import project_central_valid
from projection_reuse import reuse_verified_projection


def _feasible_world_stencil(geometry, world, column, initial_step):
    """Choose a physical-domain stencil; never evaluate outside a true guard."""
    h=float(initial_step);trials=[]
    # _q can be only a coordinate/effective-world converter on a raw current
    # geometry. Domain decisions require the public complete physical guard.
    validate=geometry.validate_parameters
    validate(world)
    while True:
        points={};errors={}
        for sign in (1,-1):
            value=world.copy();value[column]+=sign*h
            if value[column]==world[column]:raise ValueError('No representable feasible absolute geometry FD step')
            try:validate(value);points[sign]=value
            except ValueError as error:errors[sign]=str(error)
        trials.append(dict(step=h,positive_legal=1 in points,negative_legal=-1 in points,errors=errors))
        if len(points)==2:
            return (points[1],points[-1]),np.array([1.,-1.])/(2*h),dict(
                column=int(column),method='central',initial_step=float(initial_step),step=h,physical_domain_trials=trials)
        for sign,value in points.items():
            farther=world.copy();farther[column]+=2*sign*h
            try:validate(farther)
            except ValueError as error:
                trials[-1].setdefault('twice_step_errors',{})[sign]=str(error);continue
            # The derivative at the nominal state is a second-order one-sided
            # physical-domain limit. The nominal source value is unchanged.
            return (world.copy(),value,farther),np.array([-3.,4.,-1.])/(2*sign*h),dict(
                column=int(column),method='one_sided_second_order',side=int(sign),initial_step=float(initial_step),
                step=h,physical_domain_trials=trials)
        h*=.5


def evaluate_geometry_group(geometry, world66, camera25, rows, absolute, step,
                            certificates=None, batched_projection=True):
    world = np.asarray(world66, dtype=float)
    camera = np.asarray(camera25, dtype=float)
    h = float(step)
    ng = int(geometry.parameter_count)
    if (ng < 1 or world.shape != (ng,) or camera.shape != (25,) or
            not np.isfinite(world).all() or not np.isfinite(camera).all() or
            not np.isfinite(h) or h <= 0.):
        raise ValueError('Finite current material world parameters/camera25/positive FD width required')
    checked = set() if certificates is None else certificates
    geometry.validate_parameters(world)
    cache=geometry.__dict__.setdefault('_absolute_geometry_fd_stencil_cache',OrderedDict())
    token=(world.tobytes(),h)
    if token not in cache:cache[token]={}
    stencils=cache[token];cache.move_to_end(token)
    while len(cache)>2:cache.popitem(last=False)
    domain_checks=[]
    def stencil(column):
        if column not in stencils:
            points,coefficients,report=_feasible_world_stencil(geometry,world,int(column),h)
            stencils[column]=(points,coefficients,report)
        points,coefficients,report=stencils[column]
        if (report['method']!='central' or report['step']!=h) and report not in domain_checks:domain_checks.append(report)
        return points,coefficients,report['step']
    count = sum(2 * len(row['uv']) for row in rows)
    columns = np.zeros((count, ng))
    requests = []
    cursor = 0
    for row in rows:
        size = 2 * len(row['uv'])
        active = np.asarray(geometry.source_parameter_activity(row, world))
        if active.dtype != np.dtype(bool) or active.shape != (ng,):
            raise ValueError('Every current material world activity column required')
        fields = tuple(geometry.source_parameter_fields(row))
        if not fields or fields[0] != 'world' or any(field not in row for field in fields):
            raise ValueError('Original companion material fields required')
        key = geometry._key(row)
        if key not in checked:
            for k in np.flatnonzero(~active):
                points,_,_=stencil(k);rebound=[geometry.rebind(row,value)for value in points]
                if any(not np.array_equal(rebound[0][field],other[field])for other in rebound[1:]for field in fields):
                    raise ValueError('Inactive original material field changed')
            checked.add(key)
        for k in np.flatnonzero(active):
            points,coefficients,used_step=stencil(k)
            requests.append((cursor,size,int(k),[geometry.rebind(row,value)for value in points],coefficients,used_step))
        cursor += size
    context = nullcontext()
    if requests and batched_projection:
        pieces = []
        for _, _, _, rebounds, _, _ in requests:
            for rebound in rebounds:
                xyz = np.asarray(rebound['world'], dtype=float).reshape(-1, 3)
                pieces.append(xyz)
                # The original nearest-curve operator also checks each segment
                # midpoint. Preserve that precise arithmetic and all knots.
                if len(xyz) > 1:
                    pieces.append(.5 * (xyz[:-1] + xyz[1:]))
        projection_world = np.concatenate(pieces)
        uv, depth, central = project_central_valid(camera, projection_world)
        context = reuse_verified_projection(camera, projection_world, uv, depth, central)
    with context:
        for offset, size, k, rebounds, coefficients, used_step in requests:
            values=[absolute(camera,[rebound],False)for rebound in rebounds]
            if any(np.shape(value)!=(size,)for value in values):
                raise ValueError('Original absolute source row inventory changed')
            if len(values)==2:
                # Preserve exact historical central-difference arithmetic.
                columns[offset:offset + size, k]=(values[0]-values[1])/(2*used_step)
            else:
                # Subtract the nominal value first to avoid cancellation of
                # three large residuals at a physical boundary.
                columns[offset:offset + size,k]=coefficients[1]*(values[1]-values[0])+coefficients[2]*(values[2]-values[0])
    if not np.isfinite(columns).all():
        raise ValueError('Nonfinite original absolute geometry FD')
    geometry._last_absolute_geometry_fd_domain_checks=domain_checks
    return columns
