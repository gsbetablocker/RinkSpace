"""One historical native29 direction: fixed-foot value replay, no new J."""
from pathlib import Path
import copy
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_repaired_native_station_values_20261011 as values
common,np = values.common,values.np
from calibration.solver_runtime import finite_rectangle_station_step_constraints as station
from calibration.solver_runtime import finite_rectangle_projection_fd as projection
from calibration.solver_runtime.finite_rectangle_station_faces import finite_rectangle_station_halfspaces
from calibration.solver_runtime.native_truth_nominal_chart import capture_native_nominal_query_chart,native_nominal_query_carriers

SEALED = Path(str(common.RUN)+'.bootstrap')/'native29_saved_direction_causal_audit.json'
OUT = values.OUT.with_name('historical_a454_native29_fixed_foot_directional_value_regression_20261011.json')


def main():
    began = time.perf_counter()
    human,unused_q,unused_step,accepted,caps,refs = values.load_saved()
    original_arrays_sha = values.native_source_array_digest(human)
    sealed = common.read(SEALED)
    record = sealed['directions'][0]
    direction = record['saved_direction']
    assert common.sha(direction['path']) == direction['sha256']
    with np.load(direction['path'],allow_pickle=False) as archive:
        q,step = archive['q'].copy(),archive['step'].copy()
    assert common.asha(q) == record['parent_q_sha256']
    assert common.asha(step) == record['step_sha256']
    eq,ed = human.state.evaluation_q(q),human.state.evaluation_q(step)
    local = copy.copy(human)
    local.records=[human.records[29]]
    local.full_stroke_records=[human.full_stroke_records[29]]
    actual=local.records[0]['row']['actual_pts']
    local.blocks={actual:human.blocks[actual]}
    base_query=local.queries(eq,full_stroke=True)[0]['query']
    full_query=local.queries(eq+ed,full_stroke=True)[0]['query']
    np.testing.assert_allclose(common.carrier(base_query),record['native29_nominal_carrier'],rtol=0.,atol=1e-8)
    np.testing.assert_allclose(common.carrier(full_query),record['samples'][0]['carrier'],rtol=0.,atol=1e-8)
    chart=capture_native_nominal_query_chart(local,eq)
    frozen=native_nominal_query_carriers(local,eq+ed,chart)
    np.testing.assert_allclose(frozen,record['samples'][0]['nominal_branch_nonlinear_carrier'],rtol=0.,atol=1e-8)
    contexts=station.nominal_station_contexts(local,eq,full_queries=[dict(query=base_query)])
    projections=projection.projected_station_contexts(local,eq,contexts,np.empty(0,int))
    assert projections['report']['real_FD_evaluations']==0
    count=len(contexts)
    uv=np.stack([c['record']['uv'][c['source_vertex']]for c in contexts])
    normals=np.stack([c['record']['normals'][c['source_vertex']]for c in contexts])
    nb=np.stack([c['record']['bounds'][c['source_vertex']]for c in contexts])
    tb=np.stack([c['record']['tangent_bounds'][c['source_vertex']]for c in contexts])
    cone=np.tile([-np.inf,np.inf],(count,1))
    for i,c in enumerate(contexts):
        if c['nominal_physical_endpoint_side']=='lower':cone[i,0]=0.
        elif c['nominal_physical_endpoint_side']=='upper':cone[i,1]=0.
    model=finite_rectangle_station_halfspaces(projections['projected_foot_xy'],np.zeros((count,2,1)),
        projections['station_screen_tangent'],projections['nominal_physical_station'],np.tile([0.,1.],(count,1)),
        uv,normals,nb,tb,np.zeros(count),station_tangent_cone=cone)
    faces=[p for p in model['face_provenance']if p['kind']!='nonnegative_epigraph']

    def chart_face_values(value):
        feet=projection.projected_fixed_foot_carriers(local,value,contexts)
        tangent=np.c_[-normals[:,1],normals[:,0]]
        carrier=np.c_[np.einsum('ni,ni->n',feet-uv,normals),np.einsum('ni,ni->n',feet-uv,tangent)]
        result=[]
        for p in faces:
            point=p['source_point_index'];v=0.
            for weight,parent in zip(p.get('source_face_weights',[1.]),p['original_source_faces']):
                axis=0 if parent['axis']=='normal'else 1;sign=-1. if parent['side']=='lower'else 1.
                bound=(nb if axis==0 else tb)[point,0 if sign<0 else 1]
                v+=weight*sign*(carrier[point,axis]-bound)
            result.append(v)
        return np.asarray(result)

    nominal=chart_face_values(eq)
    assert np.max(nominal)<=1e-8
    directional=[]
    for h in (1e-3,1e-4):
        directional.append((chart_face_values(eq+h*ed)-chart_face_values(eq-h*ed))/(2*h))
    samples=[]
    for alpha in (1.,.5,.25):
        trial=chart_face_values(eq+alpha*ed)
        original=local.queries(eq+alpha*ed,full_stroke=True)[0]['query']
        curvature=trial-nominal-alpha*directional[-1]
        worst=int(np.argmax(trial))
        samples.append(dict(alpha=alpha,actual_original_dynamic_guard_maximum_px=float(abs(original['residual']).max()),
            maximum_fixed_foot_local_face_value_px=float(np.maximum(trial,0.).max()),
            maximum_same_chart_directional_curvature_px=float(abs(curvature).max()),
            worst_face=faces[worst],worst_face_same_chart_curvature_px=float(curvature[worst]),
            original_dynamic_guard_accepts_zero_cap=bool(abs(original['residual']).max()<=1e-8)))
    assert values.native_source_array_digest(human)==original_arrays_sha
    implementation=values.verify_implementation_unchanged()
    old_linear=np.asarray(record['native29_carrier_directional_first_order_px'])
    old_curvature=frozen-np.asarray(record['native29_nominal_carrier'])-old_linear
    report=dict(schema='rinkspace.one-historical-native29-fixed-foot-value-regression.v1',
        script=common.ref(__file__),sealed_previous_audit=common.ref(SEALED),saved_direction=common.ref(direction['path']),
        saved_parent_q_sha256=record['parent_q_sha256'],loaded_native_source_reference=refs,
        implementation=implementation,implementation_captured_before_import=values.IMPLEMENTATION_BEFORE_IMPORT,
        implementation_import_to_report_SHA_unchanged=True,
        original_nominal_dynamic_full_and_frozen_carriers_match_sealed_old_audit=True,
        original_source_arrays_before_after_sha256=original_arrays_sha,original_source_arrays_before_after_bitexact=True,
        new_fit=False,source_or_truth_J_constructed=False,small_directional_value_differences_only=True,
        source_images_written=0,source_frames_decoded=0,
        original_source_rectangle_and_zero_cap_unchanged=True,
        historical_old_frozen_tangent_chart_curvature_maximum_px=float(abs(old_curvature).max()),
        historical_dynamic_balanced_guard_full_px=float(abs(full_query['residual']).max()),
        old_frozen_tangent_vs_dynamic_guard_are_different_coordinates=True,
        fixed_foot_local_chart_h1e3_vs_h1e4_directional_FD_maximum_px=float(abs(directional[0]-directional[1]).max()),
        contexts=count,local_faces=len(faces),samples=samples,
        limits=['Only the first sealed historical a454 direction, not the other five, was replayed.',
            'Directional value stability proves consistent fixed-foot response coordinates; this report generates no Jacobian and does not alone validate arbitrary future directions.',
            'A local face value is distinct from the original dynamic guard, and the original guard remains the acceptance authority.'],seconds=time.perf_counter()-began)
    OUT.write_text(common.json.dumps(report,default=common.plain,ensure_ascii=False,indent=2),encoding='utf8')
    print(common.json.dumps(dict(path=str(OUT),sha256=common.sha(OUT),bytes=OUT.stat().st_size,seconds=report['seconds'])))


if __name__=='__main__':main()
