"""Explicit free-metric event geometry, initialized from reviewed real scenes."""
from copy import deepcopy
import numpy as np
from shared import (EVENT, FIELD, LIBRARY, OUT, NativeCore, NativeSceneGPU,
    ParameterState, SceneParameterModel, arrays, numeric_sha, original_native,
    read, save, write)


def create():
    original = original_native('jin')
    chen = original_native('chen')
    doc = read(original.scene_path)
    doc['revision'] = 'four_programme_free_metric_1117'
    doc['status'] = 'joint_fit_candidate_pending_source_validation'
    doc['native_family'] = 'shared_geometry_v8_free_ice_metric'
    doc['shared_geometry_options'] = 255
    doc.pop('ice_tolerance', None)
    doc['ice_metric_freedom'] = dict(schema='rinkspace.free_ice_metric.v1', native_option=128,
        coordinate_reference='Frozen nominal values; independent actual positions, radii, spacings and slopes',
        removed=['percentage boxes','equal circle radii','equal row spacing','F/R alignment',
                 'parallel carriers','parallel or equal-width lanes','mirror end dimensions'],
        retained=['positive dimensions','nested rings','finite line/circle/ellipse types',
                  'shared Lane endpoints','goal/board trapezoid attachments','continuous board',
                  'concentric layers within each house','crease centre on goal midpoint'],
        physical_scale='rink length L=1; no verified metre scale',
        observations_changed=False, user_visual_acceptance=False)
    q = np.array(original.scene['shared_layout'])
    report = {}
    # Static overrides were real, shared event hypotheses, not nuisance points.
    # Reattach the trapezoids to the actual carriers and continuous board.
    for path in doc['paths']:
        key = path['id']
        if 'goal_trapezoid' in key:
            branch = (0 if 'left' in key else 2) + (1 if 'upper' in key else 0)
            side = branch//2
            end = np.asarray(original.paths[key]['parameters']).reshape(2,3)
            line_id = 'demo_initial/goal_line_' + ('left:0' if side==0 else 'right:0')
            carrier = np.asarray(original.paths[line_id]['parameters']).reshape(2,3)
            midpoint = carrier.mean(axis=0)
            tangent = carrier[0]-carrier[1]
            tangent *= (1. if tangent[1]>0 else -1.)/np.linalg.norm(tangent)
            sign = 1. if branch%2 else -1.
            q[39+99+branch] = sign*np.dot(end[0]-midpoint,tangent)/q[33]-1.
            q[39+103+branch] = abs(end[1,1])/q[34]-1.
            report[key] = dict(previous=end.tolist(), carrier_attachment_error=float(
                np.linalg.norm(end[0]-midpoint-np.dot(end[0]-midpoint,tangent)*tangent)))
            for k in ('native_kind','parameters','vertices','knots'):
                path.pop(k,None)
            path['generated_code'] = 420+branch
    # Seed the explicitly reviewed Chen circles while restoring common ownership.
    for key, path in chen.paths.items():
        if key not in original.paths or path['layer']!='ice':
            continue
        source = next(p for p in read(chen.scene_path)['paths'] if p['id']==key)
        if source.get('native_kind') != 1 or 'generated_code' in source:
            continue
        target = next(p for p in doc['paths'] if p['id']==key)
        code = target.get('generated_code')
        if code is None:
            continue
        values = np.asarray(path['parameters'])
        if 100<=code<132:
            house, ring = divmod(code-100,4)
            row = house%4
            q[39+4+house] = (values[0]-q[0 if house<4 else 1])/q[3]
            baseline_f = np.asarray(original.paths['demo_initial/F'+str(row+1)+':center']['parameters'])
            assert house>=4, 'Current explicit circle overrides are R-side only'
            q[39+12+row] = (values[1]-baseline_f[1])/q[3]
            ratio = q[5+ring] if ring<3 else 1.
            q[39+16+4*house+ring] = values[3]/(q[4]*ratio)-1.
        elif 406<=code<=409:
            i=code-406
            sx=(1,1,-1,-1)[i];sy=(-1,1,1,-1)[i]
            q[39+82+i]=(values[0]-sx*q[25])/q[20]
            q[39+86+i]=(values[1]-sy*q[27])/q[20]
            q[39+91+i]=values[3]/q[20]-1.
        report[key] = dict(chen_reviewed_circle=values.tolist(), generated_code=code)
    doc['shared_layout'] = q.tolist()
    doc['initialization_provenance'] = dict(shared=str(original.scene_path),
        circle_hypotheses=str(chen.scene_path), method='exact parameter migration; no source measurement generated')
    write(FIELD, doc)
    model = SceneParameterModel(doc, FIELD, optimize_scene=True)
    values = original.parameter_state.values.copy()
    values[6:152] = q
    state = ParameterState(model.initial_state.layout, values)
    native = NativeSceneGPU(FIELD, NativeCore(LIBRARY), parameter_state=state)
    assert len(native.paths)==922 and not native.unlocalized
    write(OUT/'scene_migration.json',dict(paths=report,scene=str(FIELD),values_sha256=numeric_sha(values),
        note='Initial circle transfer changes connected R-side Lane geometry. This is a candidate, not an equivalence claim.'))
    save(OUT/'initial_scene.npz', scene_values=values, layout_revision=state.layout['layout_revision'],
         scene_values_sha256=numeric_sha(values))
    print('free metric scene:',len(native.paths),'paths;',len(values),'stored values',flush=True)


if __name__=='__main__':
    create()
