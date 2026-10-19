"""Actual saved-source location values/derivatives; stdout only, never fitting."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

import joint_absolute_source_context_20261011 as ctx
from calibration.solver_runtime.original_measured_transition_locations import (
    validate_packet, install_original_measured_transition_locations)
from calibration.solver_runtime.original_measured_location_mode import MODE_FIELD, AUTHORITY_FIELD
from calibration.solver_runtime.source_loss import weighted_source_cost
from structured_programme_source_candidate.batched_absolute_geometry_fd import evaluate_geometry_group
from joint_absolute_source_derivatives_20261011 import camera_map

np = ctx.np


def install_on_source_context(c, path):
    """Exercise the real installer on original, un-compacted absolute groups."""
    packet, original, contract = validate_packet(path)
    grouped = defaultdict(dict)
    for index, item in enumerate(original):
        if item['collection_role'] != 'FIT':
            continue
        key = (item['block'], int(item['group_order']))
        grouped[key][int(item['row_order'])] = (index, deepcopy(item['source_row']))
    for block in c.state.blocks:
        orders = [order for name, order in grouped if name == block.name]
        groups = [(0, []) for _ in range(max(orders)+1)]
        for order in orders:
            rows = grouped[block.name, order]
            if sorted(rows) != list(range(max(rows)+1)):
                raise ValueError('Original absolute group row positions are incomplete')
            indices = [rows[k][0] for k in sorted(rows)]
            actual = {int(original[i]['group_actual_pts']) for i in indices}
            if len(actual) != 1:
                raise ValueError('Original absolute group exposure changed')
            groups[order] = (actual.pop(), [rows[k][1] for k in sorted(rows)])
        block.groups = groups
        block.bundle.abs_groups = groups
    installed = install_original_measured_transition_locations(c.state, path)
    assert installed == contract
    blocks = {block.name: block for block in c.state.blocks}
    revised = {}
    for record in packet['records']:
        index = record['original_packet_row_index']; item = original[index]
        row = blocks[item['block']].groups[item['group_order']][1][item['row_order']]
        revised[index] = row
    return packet, revised


def current_revised(c, index, revised):
    row, keep = c.current_original(index)
    if index not in revised:
        return row, keep
    row[MODE_FIELD] = revised[index][MODE_FIELD]
    row[AUTHORITY_FIELD] = revised[index][AUTHORITY_FIELD]
    count = len(c.originals[index]['source_row']['uv'])
    if not np.array_equal(keep, np.arange(count)):
        row['active_original_absolute_input_compaction'] = dict(
            original_point_count=count, retained_original_point_indices=keep.tolist())
    row[AUTHORITY_FIELD].validate_runtime_row(row)
    return row, keep


def typed_raw(c, block, camera, row, jacobian):
    """Remove only normal support; preserve every original typed branch."""
    raw = dict(row)
    raw.pop(MODE_FIELD, None); raw.pop(AUTHORITY_FIELD, None)
    raw.pop('normal_interval', None)
    if block != 'secondary' or not row.get('original_outer_source_qualification_operator'):
        return c.calls[block](camera, [raw], jacobian)
    # Original strict secondary raw: connected-cap scanline inside its extent,
    # its SAME source branch endpoint XY outside. No generic nearest fallback.
    import z190_secondary_original_ring_source_qualification as strict
    result = strict.connected_cap_absolute_value_jac15(camera, [raw], jacobian)
    value, matrix = result if jacobian else (result, None)
    world = np.asarray(row['world'], float)
    projected, depth, valid = strict.project_central_valid(camera, world)
    if not np.all(valid & (depth > 0.) & np.isfinite(projected).all(axis=1)):
        raise ValueError('Original strict source curve left its central domain')
    x = np.asarray(row['uv'])[:, 0]
    low, high = np.min(projected[:, 0]), np.max(projected[:, 0])
    outside = (x < low) | (x > high)
    if outside.any():
        ends = strict._source_branch_extent_indices(projected, row['scanline_branch'])
        selected = np.where(x[outside] < low, ends[0], ends[1])
        value = value.copy()
        value.reshape(-1, 2)[outside] = (projected[selected]-np.asarray(row['uv'])[outside])/np.asarray(row['sigma'])[outside, None]
        if jacobian:
            _, _, j11, _ = strict.project_central_valid(camera, world, with_jacobian=True)
            allj = np.c_[j11, strict.brown4_world_jac(camera, world).reshape(-1, 4)].reshape(-1, 2, 15)
            matrix = matrix.copy()
            matrix.reshape(-1, 2, 15)[outside] = allj[selected]/np.asarray(row['sigma'])[outside, None, None]
    return (value, matrix) if jacobian else value


def bind(c, index, row, value):
    item = c.originals[index]; at = int(item['group_actual_pts'])
    c.provider.register_row(item['source_row'], actual_pts=at)
    return ctx.camera(c, value, at), c.provider.rebind(row, value[c.world_indices])


def representatives(c, packet):
    first = {}
    for record in packet['records']:
        index = record['original_packet_row_index']; row,keep = c.current_original(index)
        if not len(keep):continue
        item = c.originals[index]
        key = (tuple(ref['sha256'] for ref in record.get('producer_references', [])),
            record['estimator_kind'], c.calls[item['block']].__module__,
            row.get('curve_observation_kind'), row.get('scanline_fixed_axis'),
            row.get('scanline_branch'), row.get('original_outer_source_qualification_operator'),
            bool(row.get('point_correspondence')))
        first.setdefault(key,index)
    return list(first.values())


def verify_derivatives(c, value, revised, indices):
    for index in indices:
        item = c.originals[index]; row,_ = current_revised(c,index,revised)
        cam,bound = bind(c,index,row,value); call = c.calls[item['block']]
        r,camJ = call(cam,[bound],True)
        world = value[c.world_indices]
        G = evaluate_geometry_group(c.provider,world,cam,[row],call,1e-6)
        rawcall = lambda camera, rows, jac: typed_raw(c,item['block'],camera,rows[0],jac)
        rawG = evaluate_geometry_group(c.provider,world,cam,[row],rawcall,1e-6)
        assert np.array_equal(G,rawG)
        cols,mapping = camera_map(c,int(item['group_actual_pts']),value=value)
        physicalJ = np.column_stack((G,np.asarray(camJ)@mapping))
        allcols = np.r_[c.world_indices,cols]
        result = {}
        for kind,select in [('world',np.arange(len(c.world_indices))),
            ('camera',np.arange(len(c.world_indices),len(allcols))),('combined',np.arange(len(allcols)))]:
            d = np.zeros(len(value)); amplitudes=np.full(len(select),1e-6)
            selected_cols=allcols[select]
            block=c.state.bypts[int(item['group_actual_pts'])]
            begin=block.start+4*block.bundle.index[int(item['group_actual_pts'])]
            amplitudes[np.isin(selected_cols,begin+np.arange(3))]=1e-7
            amplitudes[np.isin(selected_cols,block.optics_chart.indices)]=1e-4
            d[selected_cols]=np.sin(np.arange(len(select))+1)*amplitudes
            def actual(q):
                c.provider.validate_parameters(q[c.world_indices])
                cc,rr=bind(c,index,row,q)
                return np.asarray(call(cc,[rr],False))
            errors=[]
            for scale in (1.,.5):
                fd=(actual(value+scale*d)-actual(value-scale*d))/(2*scale)
                errors.append(float(np.max(abs(fd-physicalJ@d[allcols]))))
            result[kind]=errors
        print(json.dumps(dict(stage='actual_original_location_representative_J',index=index,
            block=item['block'],id=row['id'],typed_raw_world_J_bitexact=True,
            actual_physical_direction_errors_at_h_h2=result)),flush=True)


def run(path, state_dir):
    start=time.perf_counter();c=ctx.load_context();print('context_seconds',time.perf_counter()-start,flush=True)
    packet,revised=install_on_source_context(c,path)
    state_dir=Path(state_dir);identity=ctx.a.read(state_dir/'state_identity.json')
    state_path=state_dir/'complete_state_UNACCEPTED.npz';assert ctx.a.sha(state_path)==identity['state_npz_sha256']
    with np.load(state_path,allow_pickle=False) as z:after=c.state.evaluation_q(z['q'].copy())
    for label,value in [('q8',c.q),('q091',after)]:
        costs=defaultdict(lambda:dict(old=0.,new=0.,qualified_old=0.,qualified_new=0.,points=0))
        pixels=defaultdict(list);count=qualified=0
        for index,item in enumerate(c.originals):
            if item['collection_role']!='FIT':continue
            old,keep=c.current_original(index)
            if not len(keep):continue
            new,_=current_revised(c,index,revised)
            camera,oldbound=bind(c,index,old,value);newbound=c.provider.rebind(new,value[c.world_indices])
            call=c.calls[item['block']];before=np.asarray(call(camera,[oldbound],False))
            if index in revised:
                current,J=call(camera,[newbound],True)
                raw,rawJ=typed_raw(c,item['block'],camera,oldbound,True)
                assert np.array_equal(current,raw),(index,'typed value')
                assert np.array_equal(J,rawJ),(index,'typed camera J')
                qualified+=1
            else:
                current=np.asarray(call(camera,[newbound],False))
                assert np.array_equal(current,before),(index,'unqualified value')
            count+=1;quad=ctx.original_quadratic(c,index);loss=ctx.original_loss(c,index)
            assert np.array_equal(old['sigma'],new['sigma'])
            for key in ('uv','normal','normal_interval'):
                if key in old:assert np.array_equal(old[key],new[key])
            a=weighted_source_cost(before,quad,loss);b=weighted_source_cost(current,quad,loss)
            record=costs[item['block']];record['old']+=a;record['new']+=b
            if index in revised:
                record['qualified_old']+=a;record['qualified_new']+=b;record['points']+=len(keep)
                pixels[item['block']].append(np.asarray(raw).reshape(-1,2)*np.asarray(old['sigma'])[:,None])
        for block,arrays in pixels.items():
            xy=np.concatenate(arrays);norm=np.linalg.norm(xy,axis=1)
            costs[block]['qualified_raw_pixel_mean_xy']=xy.mean(axis=0).tolist()
            costs[block]['qualified_raw_pixel_norm_p50_p95_max']=np.r_[np.quantile(norm,[.5,.95]),norm.max()].tolist()
        print(json.dumps(dict(stage='all_original_absolute_actual_location_values',state=label,
            active_original_FIT_operator_rows=count,qualified_rows=qualified,
            all_other_active_values_bitexact=True,qualified_all_typed_raw_values_and_camera_J_bitexact=True,
            UV_sigma_quadratic_loss_integration_roles_and_clock_unchanged=True,costs=costs)),flush=True)
    verify_derivatives(c,after,revised,representatives(c,packet))
    print('SUCCESS_no_fitting_no_complete_source_J_no_source_image_no_report',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--state-dir',type=Path,default=ctx.a.R/'okada_review86_q8_all_saved_location_estimators_fix_20261011/accepted_states/091f7278c5c601208c168fdbf9b560f227503d2d2dfd6fd0f3a10a7568fb16c8')
    opt=parser.parse_args();run(opt.packet,opt.state_dir)
