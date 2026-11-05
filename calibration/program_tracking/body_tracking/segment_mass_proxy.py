"""Shape-derived rigid-segment volume fractions, not measured human masses."""
import numpy as np
from scipy.spatial import ConvexHull, QhullError


# Native SMPL-X 0:22 order, matching body_prior.NATIVE22_NAMES without importing
# its video/neural preparation dependencies into this NumPy/SciPy helper.
NATIVE22_NAMES = (
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')


class _DegenerateSegment(ValueError):
    pass


def tetraVolumeProperties(points):
    """Return convex volume, COM and own-COM unit-mass inertia in input units.

    Length is the input unit, volume its cube, and inertia its square. The
    inertia axes are the unchanged input axes, translated to this own COM.

    Triangulate the closed convex hull and join every face to the mean of all
    input points, which lies inside the hull. No Qhull joggling or invented
    thickness is applied to degenerate input.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('Expected finite N x 3 segment shape points')
    if len(points) < 4:
        raise _DegenerateSegment('fewer than four shape points')
    interior = points.mean(axis=0)
    if np.linalg.matrix_rank(points - interior) < 3:
        raise _DegenerateSegment('coplanar or lower-rank shape points')
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise _DegenerateSegment('ConvexHull failed without artificial point perturbation') from error
    triangles = points[hull.simplices] - interior
    volumes = np.abs(np.einsum('ni,ni->n', triangles[:, 0],
                              np.cross(triangles[:, 1], triangles[:, 2]))) / 6.
    volume = float(volumes.sum())
    if not np.isfinite(volume) or volume <= 0 or not np.isfinite(hull.volume) or hull.volume <= 0:
        raise _DegenerateSegment('nonpositive convex segment volume')
    if not np.isclose(volume, hull.volume, rtol=1e-9, atol=0.):
        raise ValueError('Interior-face tetrahedra do not reproduce ConvexHull volume')
    centroid = interior + np.sum(volumes[:, None] * triangles.sum(axis=1) / 4., axis=0) / volume
    if not np.isfinite(centroid).all():
        raise ValueError('Nonfinite segment volume centroid')
    # Uniform tetrahedron second moment: (sum(v v.T) + sum(v) sum(v).T)/20.
    # Work around the interior point to avoid a distant origin cancellation.
    tetra = np.concatenate((np.zeros((len(triangles), 1, 3)), triangles), axis=1)
    summed = tetra.sum(axis=1)
    centre_local = np.sum(volumes[:, None] * summed / 4., axis=0) / volume
    second = (np.einsum('nvi,nvj->nij', tetra, tetra)
              + np.einsum('ni,nj->nij', summed, summed)) / 20.
    covariance = np.einsum('n,nij->ij', volumes / volume, second) - np.outer(centre_local, centre_local)
    inertia = np.trace(covariance) * np.eye(3) - covariance
    return volume, centroid, (inertia + inertia.T) / 2.


def tetraVolumeCentroid(points):
    """Compatibility interface returning only volume and COM in input units."""
    volume, centre, _ = tetraVolumeProperties(points)
    return volume, centre


def prepare_segment_mass_proxy(surface):
    """Partition the same shaped 10475-vertex SMPL-X mesh into native22 parts.

    Dominant aggregate skin weight defines each vertex group. Face weights are
    assigned to head; finger weights to their wrist. A nonempty degenerate
    group is merged explicitly into its native parent, then that parent is
    recomputed after all descendants. Output centres use unscaled shaped-model
    coordinates. The surface and its render/shader data are never modified.
    """
    points = np.asarray(surface.shaped, float)
    weights = np.asarray(surface.weights, float)
    parents = np.asarray(surface.body.parents, int)[:22]
    if (points.shape != (10475, 3) or weights.shape != (10475, 55)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(weights.sum(axis=1) <= 0)):
        raise ValueError('Expected the finite shaped10475 mesh and nonnegative weights55')
    if parents.shape != (22,) or any(not 0 <= parents[j] < j for j in range(1, 22)):
        raise ValueError('Expected ordered native SMPL-X parents for joints 0:22')
    aggregate = weights[:, :22].copy()
    aggregate[:, 15] += weights[:, 22:25].sum(axis=1)
    aggregate[:, 20] += weights[:, 25:40].sum(axis=1)
    aggregate[:, 21] += weights[:, 40:55].sum(axis=1)
    assignment = np.argmax(aggregate, axis=1)
    groups = {j: np.flatnonzero(assignment == j) for j in range(22)}
    initial_counts = [len(groups[j]) for j in range(22)]
    merged, empty, parts = [], [], {}
    # Native parents precede children. Descending order completes every merged
    # child before its parent is measured, including chains of degenerate parts.
    for joint in reversed(range(22)):
        indices = groups[joint]
        if not len(indices):
            empty.append(joint)
            continue
        try:
            parts[joint] = (*tetraVolumeProperties(points[indices]), len(indices))
        except _DegenerateSegment as error:
            if joint == 0:
                raise ValueError('Root shape group is degenerate and has no parent to merge into') from error
            parent = int(parents[joint])
            merged.append(dict(native_joint_id=joint, parent_native_joint_id=parent,
                               vertices=len(indices), reason=str(error)))
            groups[parent] = np.r_[groups[parent], indices]
    if len(parts) < 2:
        raise ValueError('Shape segment mass proxy needs at least two nondegenerate parts')
    ids = np.array(sorted(parts), dtype=int)
    volumes = np.array([parts[j][0] for j in ids])
    centres = np.stack([parts[j][1] for j in ids])
    inertia = np.stack([parts[j][2] for j in ids])
    fractions = volumes / volumes.sum()
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError('Invalid normalized positive shape-volume fractions')
    # Trace each originally assigned group through the unchanged merge chain.
    merged_parent = {row['native_joint_id']: row['parent_native_joint_id'] for row in merged}
    retained = np.full(22, -1, dtype=int)
    for joint in range(22):
        if initial_counts[joint]:
            target = joint
            while target in merged_parent:
                target = merged_parent[target]
            retained[joint] = target
    return dict(
        id='shape-segment-convex-volume.uniform-density.v1',
        native_joint_ids=ids, centres_rest=centres, mass_fractions=fractions,
        names=[NATIVE22_NAMES[j] for j in ids], volumes_rest=volumes,
        vertex_counts=np.array([parts[j][3] for j in ids], int),
        unit_mass_inertia_rest=inertia,
        inertia_units='square of the unscaled model coordinate unit of surface.shaped',
        inertia_reference='about each segment own COM, in unchanged shaped-model rest axes',
        metric_scaling='COM offsets multiply metres_per_model_unit; unit-mass inertia multiplies its square once',
        initial_native_joint_to_retained_native_joint=retained,
        initial_vertex_counts=initial_counts, merged_groups=merged,
        empty_native_joint_ids=sorted(empty),
        assumptions=['uniform-density', 'rigid-segment', 'convex-cap'],
        mass_status='shape-volume fractions; not measured body mass or athlete weight',
        centre_units='same unscaled model coordinates as surface.shaped',
        assignment='argmax of aggregate native22 skin weights; ties use first native joint',
        face_weights_merged_into_head=15,
        left_finger_weights_merged_into_wrist=20,
        right_finger_weights_merged_into_wrist=21,
        surface_render_mesh_unchanged=True)


def prepare_body_skate_mass_proxy(body_proxy, joint_centres_rest, skate_template_points, *,
                                  metres_per_model_unit, metres_per_skate_unit,
                                  relative_mass_per_skate, mass_variant_id):
    """Attach two rigid template skate proxies to an existing body mass model.

    ``joint_centres_rest`` is (22,3), in the SAME unscaled shaped-model axes as
    body_proxy centres_rest. Output body COM offsets are relative to those
    native joints; the caller supplies their current world positions and R.
    Body local axes remain the shaped-model rest axes (not a bone-aligned frame).
    ``skate_template_points`` is either one shared (V,3) template or a pair of
    left/right templates, in the local frame mapped by each current shoe R/O.
    For SkateBinding.packet() these can be template vertices + upper, with
    metres_per_skate_unit=binding.ratio. No FK or shoe deformation is done here.

    Each skate mass is relative to total BODY mass=1, never to already combined
    body+equipment mass. Body fractions are validated, not re-normalized. The
    single final division includes the equipment once. The hull combines shell,
    holder and steel into one unmeasured rigid uniform-volume proxy; it is not
    the flexible boot-cuff model or a measured material density distribution.
    """
    body_scale = float(metres_per_model_unit)
    skate_scale = float(metres_per_skate_unit)
    per_skate = np.asarray(relative_mass_per_skate, float)
    if per_skate.ndim == 0:
        per_skate = np.repeat(per_skate, 2)
    if (not np.isfinite([body_scale, skate_scale]).all() or min(body_scale, skate_scale) <= 0
            or per_skate.shape != (2,) or not np.isfinite(per_skate).all() or np.any(per_skate <= 0)):
        raise ValueError('Require positive finite metric scales and two positive assumed skate masses')
    if not isinstance(mass_variant_id, str) or not mass_variant_id.strip():
        raise ValueError('Declare a nonempty mass_variant_id for the mass/inertia assumption')
    joints = np.asarray(joint_centres_rest, float)
    ids = np.asarray(body_proxy['native_joint_ids'], int)
    centres = np.asarray(body_proxy['centres_rest'], float)
    inertia = np.asarray(body_proxy['unit_mass_inertia_rest'], float)
    masses = np.asarray(body_proxy['mass_fractions'], float)
    S = len(ids)
    if (joints.shape != (22, 3) or ids.ndim != 1 or np.any((ids < 0) | (ids >= 22))
            or centres.shape != (S, 3) or inertia.shape != (S, 3, 3)
            or masses.shape != (S,) or not all(np.isfinite(a).all() for a in (joints, centres, inertia, masses))
            or np.any(masses <= 0) or not np.isclose(masses.sum(), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('Require the normalized prepared body proxy and matching unscaled rest joints')
    try:
        shared = np.asarray(skate_template_points, float)
    except ValueError:
        shared = None
    templates = ([shared, shared] if shared is not None and shared.ndim == 2
                 else list(skate_template_points))
    if len(templates) != 2:
        raise ValueError('Supply one shared or two local skate templates')
    skate = [tetraVolumeProperties(points) for points in templates]
    relative_masses = np.r_[masses, per_skate]
    total = float(relative_masses.sum())
    if not np.isfinite(total):
        raise ValueError('Combined relative masses must have a finite positive sum')
    fractions = relative_masses / total
    return dict(
        id=body_proxy['id']+';rigid-attached-skate-convex-volume.v1',
        mass_variant_id=mass_variant_id,
        segment_names=list(body_proxy['names'])+['left_skate_proxy', 'right_skate_proxy'],
        segment_native_joint_ids=np.r_[ids, -1, -1],
        segment_mass_fractions=fractions,
        segment_com_rest_local_m=np.r_[(centres-joints[ids])*body_scale,
                                       np.stack([part[1] for part in skate])*skate_scale],
        segment_unit_mass_inertia_local_m2=np.r_[inertia*body_scale**2,
                                                np.stack([part[2] for part in skate])*skate_scale**2],
        body_segment_count=S,
        relative_mass_per_skate=per_skate.copy(),
        metres_per_model_unit=body_scale, metres_per_skate_unit=skate_scale,
        body_mass_proxy_id=body_proxy['id'],
        merged_groups=[dict(row) for row in body_proxy['merged_groups']],
        initial_native_joint_to_retained_native_joint=np.asarray(
            body_proxy['initial_native_joint_to_retained_native_joint']).copy(),
        rest_frame='body: shaped-model axes at native joint; skate: supplied template axes at shoe origin',
        inertia_reference='each segment own COM, per unit segment mass; never multiply segment mass twice',
        skate_mass_model='one rigid uniform convex proxy combining shell/holder/steel; unmeasured relative mass',
        absolute_mass_inferred=False, physical_accuracy_certified=False)
