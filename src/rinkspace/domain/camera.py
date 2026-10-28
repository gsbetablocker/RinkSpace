"""Canonical physical camera serialization for every native consumer.

Keep the persisted camera definition intact. Its packed native state contains
all currently effective optics, including the saved wide Brown extension.
"""

def complete_camera_rig(camera):
    """Serialize the selected native camera, including its existing C1 wide hinge.

    The knot is a model-basis setting; all four coefficients have solver chart
    directions. An old proof that the hinge was inactive is provenance, not a
    permission to omit it after freely fitted zoom enters that region.
    """
    import numpy as np
    rig=np.asarray(camera['rig'],np.float64)
    if rig.shape not in ((38,), (43,)) or not np.isfinite(rig).all():
        raise ValueError('Unsupported or nonfinite native physical camera pack')
    if rig.shape==(43,):return rig.copy()
    extension=camera.get('brown_wide_extension')
    if extension is None:
        proofs=camera.get('inactive_extension_proofs',[])
        if proofs:
            if len(proofs)!=1 or proofs[0].get('kind')!='wide_brown_hinge':raise ValueError('Unsupported native optics extension')
            extension=proofs[0]['original_definition']
    if extension is None:return rig.copy()
    if extension.get('schema')!='rinkfit.source_supported_wide_brown_hinge.v1' or extension.get('power')!=2:raise ValueError('Unsupported native wide optics formula')
    wide=np.asarray([extension['activation_log_f_native'],*extension['coefficients']],np.float64)
    if wide.shape!=(5,) or not np.isfinite(wide).all():raise ValueError('Invalid current native wide optics values')
    return np.r_[rig,wide]


def validate_camera_source(camera, controls, shot, source):
    """Validate one actual camera against its selected source and exposure span.

    Registry ``camera_domain`` labels cannot establish that a stored camera has
    the right pixel grid, native clock or coefficient support. This checks the
    already current files, without opening media or historical provenance.
    """
    from fractions import Fraction
    import math

    identity = shot['id']
    if camera.get('source_id') != source['id'] or shot['source_id'] != source['id']:
        raise ValueError('Camera source identity differs from its selected shot: '+identity)
    if camera.get('shot_id') is not None and camera['shot_id'] != identity:
        raise ValueError('Camera shot identity differs from selected shot: '+identity)
    if shot.get('rig_id') is not None and camera.get('rig_id') != shot['rig_id']:
        raise ValueError('Camera rig identity differs from selected shot: '+identity)
    if camera.get('source_sha256') is not None and camera['source_sha256'] != source['sha256']:
        raise ValueError('Camera source SHA256 differs from selected source: '+identity)
    names = ('time_base_num', 'time_base_den')
    for document in (camera, source):
        if any(type(document.get(name)) is not int or document[name] <= 0 for name in names):
            raise ValueError('Camera/source time base requires positive native integers: '+identity)
    if Fraction(*(camera[n] for n in names)) != Fraction(*(source[n] for n in names)):
        raise ValueError('Camera time base differs from selected source: '+identity)
    for name in ('width', 'height'):
        value = camera.get('image_'+name)
        if type(value) is not int or value <= 0 or value != source[name]:
            raise ValueError('Camera image dimensions differ from selected source: '+identity)
    if any(type(camera.get(name)) is not int
           for name in ('reference_pts', 'first_pts', 'stop_pts_exclusive')):
        raise ValueError('Camera exposure identity requires native integer PTS: '+identity)
    span = shot['processing']
    if not (camera['first_pts'] <= span['first_pts'] <= span['last_pts']
            < span['stop_pts_exclusive'] <= camera['stop_pts_exclusive']):
        raise ValueError('Actual camera does not cover selected processing domain: '+identity)
    rig = complete_camera_rig(camera)
    first, dt = camera['first_control_time'], camera['control_dt']
    times = [(int(span[name])-camera['reference_pts'])*camera['time_base_num']/camera['time_base_den']
             for name in ('first_pts', 'last_pts')]
    start, stop = times[0]-abs(rig[37])/2, times[1]+abs(rig[37])/2
    # Native MotionSpline has the same closed coefficient support and 1e-12
    # endpoint roundoff allowance. The readout halo is physical, not a new frame.
    if (not all(math.isfinite(t) for t in (first, dt, start, stop)) or dt <= 0
            or start < first+dt-1e-12 or stop > first+(len(controls)-2)*dt+1e-12):
        raise ValueError('Actual camera spline does not cover full selected readout domain: '+identity)
