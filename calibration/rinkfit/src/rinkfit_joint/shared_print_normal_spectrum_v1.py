"""Conditional shape information from the actual weighted observed-row target.

Per-row curve station is eliminated in pixel space. This is a conditional
spectrum with the current camera/scene held, not a certificate for the coupled
camera/scene/shape system and not a calibrated uncertainty calculation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.sparse import coo_matrix
from contextlib import nullcontext
from .shared_print_shape_mapping_v1 import normalized_points, file_ref
from .joint_model import as_event_view, native_camera_from_joint
from rinkfit_start.native_scene import NativeScene


def rank_from_gram(gram, rows):
    """Report only machine-precision rank; no fit/physics cutoff is invented."""
    h = (np.asarray(gram, float) + np.asarray(gram, float).T) * .5
    eigenvalues = np.linalg.eigvalsh(h)
    largest = max(float(eigenvalues[-1]), 0.) if len(eigenvalues) else 0.
    # A conservative normal-equation rounding floor, explicitly reported. Modes
    # near this floor remain numerically unresolved; they are never called true.
    floor = np.finfo(float).eps * max(h.shape[0], rows, 1) * largest
    rank = int(np.count_nonzero(eigenvalues > floor))
    return dict(ambient_coordinates=len(eigenvalues), conditional_normal_rank=rank,
                numerical_nullity=len(eigenvalues)-rank,
                gram_rounding_floor=floor, largest_gram_eigenvalue=largest,
                minimum_gram_eigenvalue=float(eigenvalues[0]) if len(eigenvalues) else 0.,
                positive_gram_eigenvalues=eigenvalues[eigenvalues > floor].tolist(),
                source_covariance_or_noise_precision_claimed=False,
                coupled_camera_scene_observability_proven=False)


def projected_normal_rows(pixel_coordinate_jacobian, tangent, scale):
    j = np.asarray(pixel_coordinate_jacobian, float)
    js = np.einsum('nij,nj->ni', j, tangent)
    length = np.linalg.norm(js, axis=1)
    valid = np.isfinite(j).all(axis=(1, 2)) & np.isfinite(length) & (length > 0)
    normal = np.zeros_like(js)
    normal[valid] = np.stack((-js[valid, 1], js[valid, 0]), axis=1) / length[valid, None]
    return np.einsum('ni,nij->nj', normal, j) * np.asarray(scale)[:, None], valid


def run(model_file, mapping_file, request_file, weights_file, aux_file, output):
    model = json.loads(Path(model_file).read_text(encoding='utf-8'))
    mapping = json.loads(Path(mapping_file).read_text(encoding='utf-8'))
    request = json.loads(Path(request_file).read_text(encoding='utf-8'))
    weights = json.loads(Path(weights_file).read_text(encoding='utf-8'))
    aux = json.loads(Path(aux_file).read_text(encoding='utf-8'))
    selection = json.loads(Path(request['manifest']['file']).read_text(encoding='utf-8'))
    refs = dict(model=file_ref(model_file), mapping=file_ref(mapping_file),
                request=file_ref(request_file), weights=file_ref(weights_file), aux=file_ref(aux_file))
    if refs['model']['sha256'] != mapping['model']['sha256']:
        raise ValueError('normal spectrum model differs from actual shape mapping')
    def array(descriptor):
        path = descriptor.get('file', descriptor.get('path'))
        if file_ref(path)['sha256'] != descriptor['sha256']:
            raise ValueError('normal spectrum array SHA mismatch: '+path)
        return np.memmap(path, dtype=descriptor['dtype'], mode='r', shape=tuple(descriptor['shape']))
    raw = request['raw_rows']
    mi = array(request['arrays']['material_index'])
    pts = array(request['arrays']['native_pts'])
    shot = array(request['arrays']['shot_index'])
    uv = array(request['arrays']['uv'])
    packet = json.loads(Path(request['source_packet']['file']).read_text(encoding='utf-8'))
    E = array(packet['arrays']['E'])
    fit = array(selection['fit_rows']).astype(np.int64)
    per_fit_weight = array(weights['fit_weights'])
    if len(fit) != len(per_fit_weight):
        raise ValueError('normal spectrum actual weight/fit denominator')
    aux_root = Path(aux_file).parent
    def accepted(name, dtype):
        ref = aux['files'][name]
        path = aux_root/name
        if file_ref(path)['sha256'] != ref['sha256']:
            raise ValueError('normal spectrum accepted auxiliary SHA mismatch')
        return np.memmap(path, dtype=dtype, mode='r')
    actual_path = accepted('path.u32', '<u4')
    station = accepted('station.f64', '<f8')
    if len(actual_path) != len(station) or len(actual_path) > raw:
        raise ValueError('normal spectrum accepted auxiliary domain')
    # Appended source rows do not create a print shape. They are outside this
    # conditional audit; old-domain path/station come from the exact accepted CP.
    row_to_group = {}
    groups = []
    for group in mapping['shared_shapes']:
        if not group['fit_rows']:
            continue
        group = dict(group, fit_positions=[])
        gi = len(groups)
        groups.append(group)
        for key in group['members']:
            row_to_group[tuple(key)] = gi
    for pos, rid in enumerate(fit):
        if rid >= len(actual_path):
            continue
        gi = row_to_group.get((int(mi[rid]), int(actual_path[rid])))
        if gi is not None:
            groups[gi]['fit_positions'].append(pos)
    scene = NativeScene(as_event_view(model, next(iter(model['shots']))))
    cameras = {sid: native_camera_from_joint(model, sid) for sid in request['shot_ids']}
    spectra = []
    progress_file = Path(output).with_suffix('.progress.json')
    with nullcontext():
        for gi, group in enumerate(groups):
            positions = np.asarray(group.pop('fit_positions'), dtype=np.int64)
            rid = fit[positions]
            cmi, cpi = group['canonical_path']
            shape_path = model['materials'][cmi]['paths'][cpi]
            q = normalized_points(model, shape_path)
            repeated = np.array_equal(q[0], q[-1])
            vertices = len(q)-int(repeated)
            edges = len(q)-1
            z = np.asarray(station[rid])*edges
            if not np.isfinite(z).all() or np.any(z < 0) or np.any(z > edges):
                raise ValueError('normal spectrum illegal actual accepted station')
            cell = np.minimum(np.floor(z).astype(int), edges-1)
            alpha = z-cell
            local = q[cell]*(1-alpha[:, None])+q[cell+1]*alpha[:, None]
            tangent = q[cell+1]-q[cell]
            jac = np.full((len(rid), 2, 2), np.nan)
            valid_projection = np.ones(len(rid), dtype=bool)
            for sid_index, sid in enumerate(request['shot_ids']):
                use = np.flatnonzero(np.asarray(shot[rid]) == sid_index)
                if not len(use):
                    continue
                # Each alias retains its physical installation, while the same
                # template derivative is shared across all actual exposures.
                for material, branch in set(zip(map(int, mi[rid[use]]), map(int, actual_path[rid[use]]))):
                    sub = use[(mi[rid[use]] == material) & (actual_path[rid[use]] == branch)]
                    path = model['materials'][material]['paths'][branch]
                    ad = np.asarray(model['blocks'][path['advertisement_block']]['values'], float)
                    off = int(path.get('advertisement_offset', 0));ad = ad[off:off+4]
                    x = ad[0]+ad[1]*local[sub, 0]
                    y = ad[2]+(ad[3]-ad[2])*local[sub, 1]
                    t = (np.asarray(pts[rid[sub]], float)-model['shots'][sid]['time_origin_pts'])/12800.
                    camera = cameras[sid]
                    t += camera.tau*(np.asarray(uv[rid[sub], 1])/1079.-.5)
                    delta = 1e-6
                    for axis in (0, 1):
                        if axis == 0:
                            plus = scene.board(x+ad[1]*delta, y);minus = scene.board(x-ad[1]*delta, y)
                        else:
                            plus = scene.board(x, y+(ad[3]-ad[2])*delta);minus = scene.board(x, y-(ad[3]-ad[2])*delta)
                        a = camera.at(plus, t);b = camera.at(minus, t)
                        jac[sub, :, axis] = (a['uv']-b['uv'])/(2*delta)
                        valid_projection[sub] &= a['valid'] & b['valid']
            scale = np.sqrt(np.asarray(per_fit_weight[positions]))/np.asarray(E[rid])
            k, finite_normal = projected_normal_rows(jac, tangent, scale)
            # At a polygon knot the one-sided derivatives can differ. Retain
            # those fit rows in the solver, but exclude them from this smooth
            # conditional rank claim rather than certify the wrong derivative.
            knot = np.abs(z-np.rint(z)) <= 8*np.finfo(float).eps*max(edges, 1)
            valid = valid_projection & finite_normal & ~knot
            lhs = cell % vertices;rhs = (cell+1) % vertices
            rr = np.repeat(np.arange(len(rid))[valid], 4)
            cc = np.stack((2*lhs, 2*lhs+1, 2*rhs, 2*rhs+1), axis=1)[valid].ravel()
            vv = np.stack(((1-alpha)*k[:, 0],(1-alpha)*k[:, 1],alpha*k[:, 0],alpha*k[:, 1]),axis=1)[valid].ravel()
            matrix = coo_matrix((vv,(rr,cc)), shape=(len(rid),2*vertices)).tocsr()
            gram = (matrix.T@matrix).toarray()
            spectrum = rank_from_gram(gram, int(np.count_nonzero(valid)))
            held = valid_projection & np.isfinite(jac).all(axis=(1,2)) & knot
            held_rows=np.flatnonzero(held)
            if len(held_rows):
                cols=np.stack((2*lhs,2*lhs+1,2*rhs,2*rhs+1),axis=1)[held_rows]
                chunks=[]
                for pixel in (0,1):
                    values=np.stack(((1-alpha)*jac[:,pixel,0],(1-alpha)*jac[:,pixel,1],alpha*jac[:,pixel,0],alpha*jac[:,pixel,1]),axis=1)[held_rows]*scale[held_rows,None]
                    chunks.append(coo_matrix((values.ravel(),(np.repeat(np.arange(len(held_rows))*2+pixel,4),cols.ravel())),shape=(2*len(held_rows),2*vertices)).tocsr())
                held_matrix=chunks[0]+chunks[1]
                upper_gram=gram+(held_matrix.T@held_matrix).toarray()
            else:upper_gram=gram
            upper_spectrum=rank_from_gram(upper_gram,int(np.count_nonzero(valid))+2*len(held_rows))
            spectra.append(dict(canonical_path=group['canonical_path'], members=group['members'],
                label=model['materials'][cmi]['label'], fit_rows=len(rid),
                smooth_valid_rows=int(np.count_nonzero(valid)), knot_rows=int(np.count_nonzero(knot)),
                invalid_current_projection_rows=int(np.count_nonzero(~valid_projection)),
                zero_or_nonfinite_station_derivative_rows=int(np.count_nonzero(~finite_normal)),
                knot_station_held_conditional_upper=upper_spectrum,
                **spectrum))
            if gi % 20 == 0:
                progress_file.write_text(json.dumps(dict(completed=gi+1,total=len(groups),
                    conditional_rank=sum(x['conditional_normal_rank'] for x in spectra),
                    ambient_coordinates=sum(x['ambient_coordinates'] for x in spectra))),encoding='utf-8')
    report = dict(schema='rinkfit.shared_print_conditional_normal_spectrum.v1',dependencies=refs,
        actual_fit_rows=len(fit), spectra=spectra,
        summary=dict(shape_groups=len(spectra),fit_print_rows=sum(x['fit_rows'] for x in spectra),
            ambient_coordinates=sum(x['ambient_coordinates'] for x in spectra),
            conditional_normal_rank=sum(x['conditional_normal_rank'] for x in spectra),
            numerical_nullity=sum(x['numerical_nullity'] for x in spectra),
            knot_held_conditional_rank_upper=sum(x['knot_station_held_conditional_upper']['conditional_normal_rank'] for x in spectra),
            invalid_current_projection_rows=sum(x['invalid_current_projection_rows'] for x in spectra),
            knot_rows=sum(x['knot_rows'] for x in spectra)),
        stage='sourceUV_observed_row_initialization', original_source_W_or_sigma_replaced=False,
        weights_are_actual_engineering_budget_not_calibrated_covariance=True,
        station_elimination='orthogonal pixel normal; exact endpoints/knots conservatively omitted from smooth rank',
        numerical_rank_rule='normal Gram eigenvalue > eps*max(rows,ambient)*largest eigenvalue',
        coupled_camera_scene_observability_proven=False, physical_shape_accuracy_proven=False,
        numerical_seed_nullspace_is_not_physical_truth=True, production_model_modified=False)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report['summary']


def main():
    parser = argparse.ArgumentParser()
    for name in ('model','mapping','request','weights','aux','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.model,args.mapping,args.request,args.weights,args.aux,args.output)))


if __name__=='__main__':main()
