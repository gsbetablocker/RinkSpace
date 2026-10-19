"""Recover the proven one-ULP Ad17 trial violation in its existing domain.

This changes only a trial's independent Ad17 placement coordinate. It neither
changes the accepted parent nor the physical print guard/tolerance. The caller
must still run its complete nonlinear guard and original true source cost.
"""
import numpy as np

def recover_ad17_print_roundoff(geometry,world):
    original=np.asarray(world,float)
    try:geometry.validate_parameters(original)
    except ValueError as error:
        if str(error)!='Ad17: current qualified print exceeds current board Z domain':return original,None
    else:return original,None
    if 17 in getattr(geometry,'source_template_instance',{}):return original,None
    indices=getattr(geometry,'location_indices',{}).get(17)
    if indices is None:return original,None
    column=int(indices[1])
    course,cq,_=geometry.print_course_binding(original)
    _,top,*_=course.sample_perimeter(cq,with_jacobian=False)
    high=float(np.max(np.asarray(top)[:,2]))*geometry.scene_unit_m
    _,height,_,jh=geometry._size(17,original)
    zmin,dz=geometry._instance_zmin(17,original);_,vhigh=geometry._material_vertical_extent(17)
    if dz[column]!=1. or np.count_nonzero(dz)!=1 or jh[column]!=0.:
        raise ValueError('Ad17 numerical trial repair needs its explicit independent placement only')
    lhs=float(zmin+vhigh*height);rhs=float(high+1e-8);excess=lhs-rhs
    ulp=float(np.spacing(max(abs(lhs),abs(rhs))))
    if not(0.<excess<=ulp and np.isfinite(ulp)and ulp>0.):return original,None
    candidate=original.copy();initial=float(original[column])
    candidate[column]=np.nextafter(initial-excess,-np.inf)
    correction_bound=2*ulp+abs(float(np.spacing(initial)))
    checks=0
    while True:
        if abs(float(candidate[column])-initial)>correction_bound:
            raise ValueError('Ad17 trial repair exceeded calculated floating-point correction bound')
        try:geometry.validate_parameters(candidate)
        except ValueError as error:
            if str(error)!='Ad17: current qualified print exceeds current board Z domain':raise
            value=float(candidate[column]);candidate[column]=np.nextafter(value,-np.inf)
            if candidate[column]==value:raise ValueError('Ad17 independent placement has no representable inward trial value')
        else:break
        checks+=1
    changed=np.flatnonzero(candidate!=original)
    if not np.array_equal(changed,np.asarray([column])):raise ValueError('Ad17 trial correction changed another physical coordinate')
    after_zmin,_=geometry._instance_zmin(17,candidate)
    return candidate,dict(schema='rinkspace.Ad17-independent-placement-one-ULP-trial-recovery.v1',
        applied=True,logical_world_column=column,advertisement_id=17,source_guard_tolerance_unchanged=1e-8,
        before_print_top=lhs,unchanged_board_top_plus_guard_tolerance=rhs,before_guard_excess=excess,
        representable_guard_operation_ulp=ulp,maximum_coordinate_correction_from_float_arithmetic=correction_bound,
        independent_z_offset_before=initial,independent_z_offset_after=float(candidate[column]),
        actual_coordinate_correction=float(candidate[column])-initial,
        after_print_top=float(after_zmin+vhigh*height),additional_nextafter_checks=checks,
        all_other_world_coordinates_bitexact=True,accepted_parent_changed=False,physical_domain_or_source_tolerance_changed=False,
        corrected_complete_nonlinear_guard_and_true_source_cost_still_required=True,
        numerical_roundoff_is_not_a_measured_placement=True)
