"""Capture the original complete source on an explicit expanded world chart.

The frozen parent driver is read, never edited. Exact literal transitions
below change chart validation/metadata and its external CSR insertion; source
rows, original operator calls, priors, profiles and camera paths are retained.
"""
import inspect
from pathlib import Path
from shared_advertisement_pairs_candidate import z190_joint_intercamera_fixed_xyz_paired_mount_source as parent
from .hockey_graph import append_hockey_chart_optics_csr
from .hockey_source_chart import HockeyCoordinateChart


def capture_hockey_driver(args, state):
    chart = HockeyCoordinateChart(state.geometry.new_parameter_count)
    chart.core_q(state.q)
    if state.geometry.parameter_count != 23 + chart.hockey_parameter_count:
        raise ValueError('Complete declared world23 plus hockey coordinates required')
    source = inspect.getsource(parent.run)
    marker = "    write(state.output/'input_manifest.json',manifest);guard(state.q);source_review(state.q,dict(stage='initial_unaccepted'))"
    if source.count(marker) != 1:
        raise RuntimeError('Authenticated original complete source capture boundary changed')
    source = source[:source.index(marker)] + '    return locals()\n'
    transitions = {
        "len(state.q)!=34225": f"len(state.q)!={chart.core_count}",
        "len(state.geometry_parameter_indices)!=23": f"len(state.geometry_parameter_indices)!={23 + chart.hockey_parameter_count}",
        "len(set(shared_schur_indices.tolist()))!=59": f"len(set(shared_schur_indices.tolist()))!={59 + chart.hockey_parameter_count}",
        "physical_parameter_count=34225,physical_shared_parameter_count=59":
            f"physical_parameter_count={chart.core_count},physical_shared_parameter_count={59 + chart.hockey_parameter_count}",
    }
    for old, new in transitions.items():
        if source.count(old) != 1:
            raise RuntimeError('Original explicit chart assertion changed: ' + old)
        source = source.replace(old, new, 1)
    source = source.replace('def run(args):', 'def isolated_hockey_capture(args):', 1)
    old_units = 'shared_geometry_metres=q[state.geometry_parameter_indices].tolist()'
    if source.count(old_units) != 1:
        raise RuntimeError('Original geometry diagnostics field changed')
    source = source.replace(old_units,
        'shared_geometry_parameters=q[state.geometry_parameter_indices].tolist(),'
        'shared_geometry_parameter_units=list(state.geometry.parameter_units)', 1)
    namespace = dict(parent.__dict__)
    namespace.update(prepare=lambda unused: state,
                     append_installation_optics_csr=append_hockey_chart_optics_csr)
    exec(compile(source, __file__, 'exec'), namespace)
    captured = namespace['isolated_hockey_capture'](args)
    if len(captured['shared_schur_indices']) != 59 + chart.hockey_parameter_count:
        raise ValueError('Actual complete source omitted shared hockey parameters')
    captured['manifest'].update(
        shared_ice_hockey_geometry=state.shared_ice_hockey_contract,
        whole_hockey_chart_capture=chart.manifest(),
        hockey_nominal_reference_rows_installed=False,
        rink_60x30_boundary_reference_or_transport_installed=False,
        original_source_operators_pixels_sigma_roles_profiles_and_nonroll_priors_preserved=True)
    return captured
