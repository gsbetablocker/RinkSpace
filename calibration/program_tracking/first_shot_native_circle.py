"""Opt-in prototype for original finite-profile queries on physical circles.

No installer or FIT admission is provided here.  A caller must authenticate a
whole original family and independently review its complete native inventory.
The model remains a world circle projected by the original Brown4 and row-time
camera.  No image ellipse, source nearest foot, or residual-ranked branch is
used by the forward factor.

Fixed-camera roots enumerate the complete rational Brown4 circle polynomial.
The row-time prototype follows the existing native-line implicit-root scheme:
all central-camera polynomial seeds are retained and every unresolved seed
fails.  This is not a proof against extra rolling-row roots born between those
seeds; production installation requires a root-completeness certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.polynomial import Polynomial as Poly


class NativeCircleQueryError(ValueError):
    """A saved finite query lacks one certified forward physical branch."""


@dataclass(frozen=True)
class CircleQueries:
    native_pts: int
    house_id: str
    roles: tuple[str, ...]
    query_ids: tuple[str, ...]
    observed: np.ndarray
    origins: np.ndarray
    normals: np.ndarray
    support: np.ndarray
    sigma: np.ndarray
    native_readouts_sha256: str

    def select(self, role):
        indices = np.flatnonzero(np.asarray(self.roles) == role)
        if not len(indices):
            raise ValueError("No native queries for physical circle role")
        return CircleQueries(self.native_pts, self.house_id,
            tuple(self.roles[i] for i in indices), tuple(self.query_ids[i] for i in indices),
            *(np.array(getattr(self, key)[indices], copy=True) for key in
              ("observed", "origins", "normals", "support", "sigma")),
            self.native_readouts_sha256)


def compile_paint_circle_queries(native_packet, *, house_id='C0'):
    """Preserve actual narrow-paint midpoint reads; diagnostic admission only.

    The saved inventory was selected by a historical navigation estimator.
    Keeping it does not recover rejected queries or calibrate their noise.
    """
    if house_id != 'C0':
        raise ValueError('Only the explicitly defined C0 narrow-paint center is supported')
    source = native_packet['source']; pts = int(source['native_pts'])
    if int(source['actual_pts']) != pts or source['time_base'] != '1/12800':
        raise ValueError('Original native paint-circle clock changed')
    features = [f for f in native_packet['features'] if f['id'] == 'ice.C0:outer']
    if len(features) != 1 or features[0].get('measurement_operator') != 'paint_center':
        raise ValueError('One original C0 narrow-paint center estimator required')
    feature = features[0]; fit = feature['source_circle_fit']; direct = fit['direct_readouts']
    if fit.get('current_frame_only') is not True or fit.get('actual_native_readout_count') != len(direct):
        raise ValueError('The complete recorded native C0 inventory is required')
    roles=[]; ids=[]; observed=[]; origins=[]; normals=[]; support=[]; sigma=[]
    for row in direct:
        e = row['source_evidence']; q=np.asarray(e['query_uv'],float); n=np.asarray(e['profile_normal'],float)
        uv=np.asarray(row['uv'],float); interval=np.asarray(e['complete_band_native_interval_px'],float)
        sides=np.asarray(e['actual_source_sides_uv'],float); shift=float(e['native_profile_shift_px']); scale=float(row['sigma_px'])
        if (row.get('role') != 'outer' or e.get('physical_layer') != 'outer'
                or e.get('declared_inside_outside_materials') != ['light_ice','narrow_circular_paint','light_ice']
                or e.get('measurement_origin') != 'current_native_narrow_circle_two_local_ice_side_shoulders_and_monotone_half_level_crossings'
                or e.get('actual_source_support_outside_person_and_broadcast_masks') is not True
                or q.shape != (2,) or n.shape != (2,) or uv.shape != (2,) or interval.shape != (2,) or sides.shape != (2,2)
                or not np.isfinite(np.r_[q,n,uv,interval,sides.ravel(),shift,scale]).all()
                or not np.isclose(np.linalg.norm(n),1.,rtol=0,atol=2e-12)
                or not interval[0] < shift < interval[1] or scale <= 0.
                or np.max(abs(q+n*shift-uv)) > 5.000001e-5
                or np.max(abs(sides.mean(0)-uv)) > 5.000001e-5
                or scale != float(e['transition_sigma_px'])
                or int(row['query_index']) != int(e['query_index'])):
            raise ValueError('Saved native C0 midpoint, support, material or source scalar changed')
        if any(np.any(x < [0.,0.]) or np.any(x >= [1920.,1080.]) for x in (q,uv)):
            raise ValueError('Original native C0 query leaves its recorded sensor')
        roles.append('outer');ids.append(f"{pts}/ice.C0:outer/query/{int(row['query_index'])}")
        observed.append(uv);origins.append(q);normals.append(n);support.append(interval);sigma.append(scale)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Unique nonempty original native C0 query inventory required')
    digest=hashlib.sha256(json.dumps(direct,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    arrays=[np.asarray(x,float)for x in (observed,origins,normals,support,sigma)]
    for a in arrays:a.setflags(write=False)
    return CircleQueries(pts,house_id,tuple(roles),tuple(ids),*arrays,digest)


def compile_circle_queries(native_packet, *, house_id="F2"):
    """Authenticate saved query scalars without adding material admission.

    The original IQR support sigma is retained verbatim.  This function does
    not claim empirical noise calibration or infer an independent factor from
    a compressed W.  A measured raw L, when supplied later, takes precedence.
    """
    if house_id == "C0":
        return compile_paint_circle_queries(native_packet, house_id=house_id)
    source = native_packet["source"]
    pts = int(source["native_pts"])
    if int(source["actual_pts"]) != pts or source["time_base"] != "1/12800":
        raise ValueError("Original native source clock changed")
    features = [f for f in native_packet["features"]
        if f.get("joint_group_navigation", {}).get("house_id") == house_id
        and f.get("source_circle_fit")]
    roles, ids, observed, origins, normals, support, sigma, archived = ([] for _ in range(8))
    material = {"outer": ["dark_outer_house_fill", "light_ice"],
        "middle": ["light_middle_house_ring", "dark_outer_house_fill"],
        "inner": ["dark_inner_house_fill", "light_middle_house_ring"]}
    if {f["id"] for f in features} != {f"ice.{house_id}:{role}" for role in material}:
        raise ValueError("An entire declared three-ring family is required")
    for feature in features:
        role = feature["id"].split(":")[-1]
        saved = feature["source_circle_fit"]
        direct = saved["direct_readouts"]
        if (feature.get("measurement_operator") != "material_interface"
                or saved.get("current_frame_only") is not True
                or saved.get("actual_native_readout_count") != len(direct)):
            raise ValueError("One current native material-interface support is required")
        for row in direct:
            evidence = row["source_evidence"]
            q = np.asarray(evidence["query_uv"], float)
            n = np.asarray(evidence["profile_normal"], float)
            uv = np.asarray(row["uv"], float)
            interval = np.asarray(evidence["whole_transition_native_interval_px"], float)
            shift = float(evidence["native_profile_shift_px"])
            scale = float(row["sigma_px"])
            if (row.get("role") != role or evidence.get("physical_layer") != role
                    or evidence.get("declared_inside_outside_materials") != material[role]
                    or evidence.get("actual_source_support_outside_person_and_broadcast_masks") is not True
                    or q.shape != (2,) or n.shape != (2,) or uv.shape != (2,)
                    or interval.shape != (2,) or not np.isfinite(np.r_[q,n,uv,interval,shift,scale]).all()
                    or not np.isclose(np.linalg.norm(n), 1., rtol=0, atol=2e-12)
                    or not interval[0] < shift < interval[1] or scale <= 0.
                    or np.max(abs(q+n*shift-uv)) > 5.000001e-5):
                raise ValueError("Native circle query support, UV, role or scalar coordinate changed")
            if any(np.any(x < [0.,0.]) or np.any(x >= [1920.,1080.]) for x in (q,uv)):
                raise ValueError("Original native observations and query origins must be in-sensor")
            if scale != float(evidence["transition_sigma_px"]):
                raise ValueError("Recorded source localization support sigma changed")
            query_index = int(evidence["query_index"])
            if query_index != int(row["query_index"]):
                raise ValueError("Saved native query index changed")
            roles.append(role); ids.append(f"{pts}/{feature['id']}/query/{query_index}")
            observed.append(uv); origins.append(q); normals.append(n)
            support.append(interval); sigma.append(scale); archived.append(row)
    if len(set(ids)) != len(ids):
        raise ValueError("A real native query cannot be admitted twice")
    digest = hashlib.sha256(json.dumps(archived,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    arrays = [np.asarray(x, float) for x in (observed,origins,normals,support,sigma)]
    for array in arrays:
        array.setflags(write=False)
    return CircleQueries(pts,house_id,tuple(roles),tuple(ids),*arrays,digest)


def native_circle_operator(queries, *, raw_noise_factor=None):
    """Use the full original raw L, or its authenticated original diagonal.

    All latent columns are retained.  This is a new likelihood basis and row
    inventory; its Huber cost is not comparable with compressed-family cost.
    """
    from .source_evidence_whitening import latent_whitening_operator
    count = len(queries.query_ids)
    L = np.diag(queries.sigma) if raw_noise_factor is None else np.asarray(raw_noise_factor,float)
    if L.ndim != 2 or L.shape[0] != count or not np.isfinite(L).all() or np.linalg.matrix_rank(L) != count:
        raise ValueError("Original complete native localization factor must have full measured row rank")
    basis = np.zeros((count,2*count))
    for i,n in enumerate(queries.normals):
        basis[i,2*i:2*i+2] = n
    return latent_whitening_operator(queries.query_ids,np.eye(count),
        explicit_noise_factor=L,physical_basis=basis,
        group_id=f"{queries.native_pts}/{queries.house_id}/native_circle_family",
        covariance_origin="authenticated_original_circle_native_scalar_factor_without_template_compression")


def _circle(coefficients, coefficient_jacobian):
    c = np.asarray(coefficients,float)
    if c.shape != (3,3) or not np.isfinite(c).all():
        raise ValueError("Explicit current world center/cosine/sine vectors required")
    radius = np.linalg.norm(c[1]); other = np.linalg.norm(c[2])
    tolerance = 128*np.finfo(float).eps*max(1.,radius,other,np.max(abs(c)))
    if radius <= tolerance or abs(radius-other) > tolerance or abs(c[1]@c[2]) > tolerance*radius:
        raise ValueError("The supplied current physical target must remain one circle")
    J = np.empty((3,3,0)) if coefficient_jacobian is None else np.asarray(coefficient_jacobian,float)
    if J.ndim != 3 or J.shape[:2] != (3,3) or not np.isfinite(J).all():
        raise ValueError("Circle coefficient J must retain the complete original world chart")
    return c,J


def _sample(camera, coefficients, derivative, theta, *, jacobian):
    from .first_shot_source_kernel import _interfaces
    from .first_shot_native_line_kernel import _camera
    camera = _camera(camera)
    basis = np.array([1.,np.cos(theta),np.sin(theta)])
    tangent = -coefficients[1]*np.sin(theta)+coefficients[2]*np.cos(theta)
    point = basis @ coefficients
    project,brown,_,_ = _interfaces()
    if jacobian:
        uv,depth,j11,valid = project(camera,point[None],with_jacobian=True)
        jcam = np.c_[np.asarray(j11).reshape(2,11),brown(camera,point[None]).reshape(2,4)]
        world_projection = -jcam[:,:3]/100.
        first = world_projection@tangent
        jworld = world_projection@np.einsum("i,ijk->jk",basis,derivative)
    else:
        uv,depth,valid = project(camera,point[None],with_jacobian=False)
        # The implicit geometric root still needs its tangent.
        _,_,j11,good = project(camera,point[None],with_jacobian=True)
        valid = valid & good
        first = (-np.asarray(j11).reshape(2,11)[:,:3]/100.)@tangent
        jcam = jworld = None
    if not bool(valid[0]) or depth[0] <= .05 or not np.isfinite(uv).all():
        raise NativeCircleQueryError("Circle query left the actual forward central Brown branch")
    return uv[0],first,jcam,jworld


def circle_brown_roots(camera, coefficients, query, normal):
    """Enumerate the full fixed-camera Brown4 rational circle, not an ellipse."""
    import cv2
    from .first_shot_native_line_kernel import _camera
    camera = _camera(camera)
    coefficients,_ = _circle(coefficients,None)
    R = cv2.Rodrigues(camera[3:6])[0]
    center = R@(coefficients[0]-camera[:3]);a=R@coefficients[1];b=R@coefficients[2]
    X,Y,Z = [Poly([center[k]+a[k],2*b[k],center[k]-a[k]]) for k in range(3)]
    rr = X*X+Y*Y; k1,k2,p1,p2 = camera[11:15]
    U = X*Z**4+k1*X*rr*Z**2+k2*X*rr**2+2*p1*X*Y*Z**3+p2*(3*X*X+Y*Y)*Z**3
    V = Y*Z**4+k1*Y*rr*Z**2+k2*Y*rr**2+p1*(X*X+3*Y*Y)*Z**3+2*p2*X*Y*Z**3
    transverse = np.array([normal[1],-normal[0]],float)
    polynomial = transverse[0]*(np.exp(camera[6])*U+camera[10]*V+(camera[8]-query[0])*Z**5)
    polynomial += transverse[1]*(np.exp(camera[7])*V+(camera[9]-query[1])*Z**5)
    scale = np.max(abs(polynomial.coef))
    if not np.isfinite(scale) or scale == 0.:
        raise NativeCircleQueryError("Circle and query have an unresolved polynomial intersection")
    polynomial = (polynomial/scale).trim(0.)
    roots = polynomial.roots()
    real = roots.real[abs(roots.imag) <= 8*np.sqrt(np.finfo(float).eps)*np.maximum(1.,abs(roots.real))]
    angles = [float(2*np.arctan(t)%(2*np.pi)) for t in real]
    angles.append(np.pi)  # tan(theta/2) misses its projective infinity.
    accepted=[]
    for theta in angles:
        try:
            uv,first,_,_ = _sample(camera,coefficients,np.empty((3,3,0)),theta,jacobian=False)
        except NativeCircleQueryError:
            continue
        if abs(transverse@(uv-query)) > 1e-6:
            continue
        if any(abs(np.arctan2(np.sin(theta-old),np.cos(theta-old))) <= 1e-7 for old in accepted):
            continue
        accepted.append(theta)
    return accepted


def evaluate_circle_queries(camera_or_trajectory,queries,coefficients,
                            coefficient_jacobian=None,*,jacobian=True):
    """Finite profile intersection-minus-native UV and original chart J.

    A caller must pass queries for one ring.  Returned raw rows remain 2D so
    the existing full original L/physical-normal operator can consume them.
    Rolling-row output is prototype-only until its global root certificate is
    added; explicit receipts prevent accidental production installation.
    """
    if len(set(queries.roles)) != 1:
        raise ValueError("One physical circle target is evaluated per call")
    if not jacobian and coefficient_jacobian is not None:
        raise ValueError("Value-only native circle evaluation cannot consume world derivatives")
    c,D = _circle(coefficients,coefficient_jacobian)
    trajectory = camera_or_trajectory if hasattr(camera_or_trajectory,"camera15_mapping") else None
    if trajectory is None:
        central = np.asarray(camera_or_trajectory,float)
        columns = np.arange(15)
    else:
        central = trajectory.camera25_at_row(.5*(trajectory.image_height-1.))
        columns = np.asarray(trajectory.columns,int)
    values=[]; cameras=[]; geometries=[]; evidence=[]
    for source,normal,observed,support,identity in zip(queries.origins,queries.normals,
            queries.observed,queries.support,queries.query_ids,strict=True):
        transverse = np.array([normal[1],-normal[0]])
        seeds = circle_brown_roots(central,c,source,normal)
        candidates=[]; unresolved=[]; physical_roots=[]
        for theta in seeds:
            if trajectory is None:
                predicted,first,jcam,jworld = _sample(central,c,D,theta,jacobian=True)
                denominator = transverse@first
                # A degree-ten polynomial's almost multiple roots are not
                # resolved to machine epsilon.  Reject the root when its
                # first derivative is below the polynomial root resolution;
                # this is numerical domain certification, not material slack.
                if abs(denominator) <= 8*np.sqrt(np.finfo(float).eps)*max(1.,np.linalg.norm(first)):
                    raise NativeCircleQueryError("Native circle profile is tangent: "+identity)
                direct = np.c_[jworld,jcam]
                total = direct-first[:,None]*(transverse@direct/denominator)[None]
                sensor_y=float(predicted[1])
            else:
                sensor_y=float(_sample(central,c,D,theta,jacobian=False)[0][1])
                converged=False
                for iteration in range(24):
                    try:
                        camera,local_columns,chain=trajectory.camera15_mapping(sensor_y,
                            jacobian=True,geometry_navigation=True)
                        if not np.array_equal(local_columns,columns):
                            raise ValueError("Circle query changed the complete neighbouring camera columns")
                        predicted,first,jcam,jworld=_sample(camera,c,D,theta,jacobian=True)
                        velocity=jcam@trajectory.camera15_row_derivative(sensor_y,geometry_navigation=True)
                    except ValueError as error:
                        unresolved.append(str(error));break
                    system=np.array([[transverse@first,transverse@velocity],[first[1],velocity[1]-1.]])
                    singular=np.linalg.svd(system,compute_uv=False)
                    if singular[-1] <= 8*np.sqrt(np.finfo(float).eps)*max(1.,singular[0]):
                        raise NativeCircleQueryError("Native circle row/profile is tangent or singular: "+identity)
                    equation=np.array([transverse@(predicted-source),predicted[1]-sensor_y])
                    if np.max(abs(equation)) <= 1e-8:
                        converged=True;break
                    step=np.linalg.solve(system,-equation)
                    theta+=step[0];sensor_y+=step[1]
                    if not np.isfinite([theta,sensor_y]).all():
                        unresolved.append("Nonfinite circle row/profile iterate");break
                if not converged:
                    if not unresolved:unresolved.append("Circle row/profile root did not converge")
                    continue
                direct=np.c_[jworld,jcam@chain]
                implicit=-np.linalg.solve(system,np.vstack((transverse@direct,direct[1])))
                total=direct+first[:,None]*implicit[0]+velocity[:,None]*implicit[1]
            shift=float(normal@(predicted-source))
            physical_roots.append(dict(theta=float(theta%(2*np.pi)),
                predicted_uv=predicted.tolist(),profile_station_px=shift,
                within_source_support=bool(support[0] <= shift <= support[1])))
            if not support[0] <= shift <= support[1]:
                continue
            if abs(transverse@(predicted-source)) > 1e-6:
                raise NativeCircleQueryError("Physical circle root does not reproduce the saved profile")
            if any(np.linalg.norm(predicted-item[0]) <= 1e-7 for item in candidates):
                continue
            candidates.append((predicted,total,theta,sensor_y,shift))
        if unresolved:
            raise NativeCircleQueryError(identity+": unresolved physical branches: "+"; ".join(unresolved))
        if len(candidates) != 1:
            error=NativeCircleQueryError(f"{identity}: finite actual profile requires one branch; found {len(candidates)}")
            error.report=dict(query_id=identity,query_uv=source.tolist(),
                profile_normal=normal.tolist(),actual_source_support=support.tolist(),
                actual_physical_roots=physical_roots,
                observation_used_to_rank_branches=False)
            raise error
        predicted,total,theta,sensor_y,shift=candidates[0]
        values.append(predicted-observed)
        if jacobian:
            geometries.append(total[:,:D.shape[2]]);cameras.append(total[:,D.shape[2]:])
        evidence.append(dict(query_id=identity,whole_circle_seed_count=len(seeds),
            accepted_finite_support_roots=1,profile_station_px=shift,source_support=support.tolist(),
            theta=float(theta%(2*np.pi)),predicted_sensor_row=sensor_y))
    count=2*len(values)
    return (np.asarray(values).reshape(-1),
        np.vstack(cameras) if jacobian else np.empty((count,0)),
        np.vstack(geometries) if jacobian else np.empty((count,0)),columns,
        dict(prototype_only=True,installed_in_FIT=False,queries=evidence,
            image_ellipse_used=False,observation_used_to_rank_branches=False,
            root_completeness="fixed_camera_full_Brown4_polynomial" if trajectory is None else
                "all_central_Brown4_seeds_implicit_RS_continuation_requires_global_certificate"))
