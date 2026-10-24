"""Current full-S01 parameter registration for the original d78 solver.

This adapts current complete native model data. It changes no parameter value,
physical equation, bound, scale, gauge or source geometry and starts no solve.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from rinkfit_start.ad_chart import PrintOnlyAdChart
from rinkfit_start.camera_spline_chart import time_chart_from_model


def register_full_model(native_model,unobserved_artwork_ids=(),unobserved_geometry_blocks=None):
    model=deepcopy(native_model)
    unobserved=set(map(int,unobserved_artwork_ids))
    if not unobserved.issubset(set(range(8,13))):raise ValueError('Only unsupported whole-print objects may leave this affine registration')
    print_ad_ids=tuple(ad for ad in range(8,13)if ad not in unobserved)
    unobserved_geometry=set(native_model.get('current_input_scope',{}).get('unobserved_geometry_blocks',()))if unobserved_geometry_blocks is None else set(unobserved_geometry_blocks)
    if any(not name.startswith('ice/')for name in unobserved_geometry):raise ValueError('Unsupported geometry registration must name explicit ice blocks')
    names=[b['name']for b in model['blocks']]
    if len(set(names))!=len(names):raise ValueError('Duplicate current native parameter block')
    begin=int(model['camera']['control_block_begin']);nc=int(model['camera']['n_controls'])
    if len(model['blocks'][0]['values'])!=32 or any(not b['name'].startswith('control/')for b in model['blocks'][begin:begin+nc]):
        raise ValueError('Complete original global/head and native spline storage required')
    physical_counts={key:0 for key in ('shared_rink_and_camera_head','camera_controls','ice_geometry','observed_ad_installations','whole_print_objects','shared_material_features')}
    replaced=[]
    for b in model['blocks']:
        name=b['name'];free=True
        if name=='global32':role='shared_rink_and_camera_head'
        elif name.startswith('control/'):role='camera_controls'
        elif name.startswith('ice/'):role='ice_geometry'
        elif name.startswith('ad/'):
            ad=int(name.split('/')[1]);free=2<=ad<=7;role='observed_ad_installations'
        elif name.startswith('print/Ad') and name.endswith('_current_whole'):
            ad=int(name.split('/Ad')[1].split('_')[0]);free=8<=ad<=12 and ad not in unobserved;role='whole_print_objects'
        elif name.startswith('print/'):
            free=False;role=None;replaced.append(name)
        elif name.startswith('feature/ice.intersection.C.') and name.endswith('/shared_C_station'):
            if len(b['values'])!=1 or b['lower']!=[0.] or b['upper']!=[1.]:
                raise ValueError('A shared material feature needs its delivered one-dimensional station domain')
            role='shared_material_features'
        else:raise ValueError('Current full-model parameter owner needs explicit registration: '+name)
        if name in unobserved_geometry:free=False
        b['constant']=not free
        if free:physical_counts[role]+=len(b['values'])-len(b.get('fixed_indices',()))
    for ad in range(2,13):
        if f'ad/{ad}'not in names or f'print/Ad{ad}_current_whole'not in names:
            raise ValueError('The complete delivered static artwork object is missing for Ad'+str(ad))
    scope=model.setdefault('current_input_scope',{})
    scope.update(schema='rinkfit.current_advertisement_input.v1',scope_id='S01_full',full_physical_solver=True,
                 physical_scope='original camera, rink/ice/board and every supported current static advertisement object',print_only_ad_ids=list(print_ad_ids),unobserved_geometry_blocks=sorted(unobserved_geometry))
    model['print_only_ad_ids']=list(print_ad_ids)
    chart=PrintOnlyAdChart(model,ad_ids=print_ad_ids)
    time_chart=time_chart_from_model(model)
    if time_chart is None:raise ValueError('Current d78 parent must retain its declared original time family')
    C,scales,coordinate_names,time_info=time_chart.compose_physical(chart,model)
    if not np.isfinite(C.data).all() or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Invalid full original physical chart')
    if any(a['values']!=b['values'] or a['lower']!=b['lower'] or a['upper']!=b['upper'] or a['scales']!=b['scales'] or a.get('fixed_indices',[])!=b.get('fixed_indices',[])for a,b in zip(native_model['blocks'],model['blocks'])):
        raise AssertionError('A current physical value/domain/scale/gauge changed')
    if model['camera']!=native_model['camera'] or model['materials']!=native_model['materials']:
        raise AssertionError('A physical definition or static source shape changed')
    report=dict(schema='rinkfit.current_full_S01_parameter_registration.v1',physical_free_families=physical_counts,
                full_fine_storage_free_physical_scalars=len(chart.free),original_time_chart_solver_columns=C.shape[1],
                original_time_family=time_chart.version,original_time_chart=time_info,
                all_camera_controls_stored=nc,current_advertisement_object_DOF=physical_counts['observed_ad_installations']+physical_counts['whole_print_objects'],
                unobserved_artwork_objects_fixed=sorted(unobserved),unobserved_artwork_values_and_geometry_preserved=True,
                unobserved_geometry_blocks_fixed=sorted(unobserved_geometry),unobserved_geometry_values_preserved=True,
                original_unused_per_contour_print_blocks_fixed=replaced,
                replaced_legacy_print_columns_are_not_unreferenced_solver_variables=True,
                one_static_source_object_per_advertisement=True,physical_values_bounds_scales_gauges_and_geometry_unchanged=True,
                complete_model_required_not_a_single_advertisement_release=True,no_Solve=True)
    model.setdefault('provenance',{})['full_S01_parameter_registration']={k:v for k,v in report.items()if k not in ('original_time_chart','original_unused_per_contour_print_blocks_fixed')}
    return model,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=json.loads(Path(a.model).read_text(encoding='utf8'));model,report=register_full_model(source)
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'registration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items()if k not in ('original_time_chart','original_unused_per_contour_print_blocks_fixed')},ensure_ascii=False))
if __name__=='__main__':main()
