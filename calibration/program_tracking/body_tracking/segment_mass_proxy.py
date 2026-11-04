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


def tetraVolumeCentroid(points):
    """Return convex volume and volume centroid in the input coordinate units.

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
    return volume, centroid


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
            parts[joint] = (*tetraVolumeCentroid(points[indices]), len(indices))
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
    fractions = volumes / volumes.sum()
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError('Invalid normalized positive shape-volume fractions')
    return dict(
        id='shape-segment-convex-volume.uniform-density.v1',
        native_joint_ids=ids, centres_rest=centres, mass_fractions=fractions,
        names=[NATIVE22_NAMES[j] for j in ids], volumes_rest=volumes,
        vertex_counts=np.array([parts[j][2] for j in ids], int),
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
