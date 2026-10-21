"""Exact frozen-ice comparison with numeric JSON angle canonicalization.

C++ serialization changes start_angle 0.0 to 0. Both describe the identical
finite circular primitive. Normalize only that continuous scalar's JSON type;
all consumed values, scene, path IDs, domains and geometry guards remain exact.
No point correspondences, source observations or physical parameters are added.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
from .io_utils import file_sha256

BASE_SHA256 = '6b28fb443983befc0b1b780886ded5a8dcd1d63dddf46a4c77a9d79853cfe3aa'

def _canonical_model(model):
    materials=[]
    for material in model['materials']:
        paths=[]
        for path in material['paths']:
            q=dict(path)
            if path['kind']==1 and 'start_angle' in path:
                v=path['start_angle']
                if type(v) not in (int,float) or not math.isfinite(v):
                    raise ValueError('Finite numeric circular start_angle required')
                q['start_angle']=float(v)
            paths.append(q)
        materials.append({**material,'paths':paths})
    return {**model,'materials':materials}

def _load_base():
    path=Path(__file__).with_name('source_quality_compare_v4.py')
    if file_sha256(path)!=BASE_SHA256:
        raise ValueError('Frozen geometry comparison implementation changed')
    spec=importlib.util.spec_from_file_location('rinkfit_joint._numeric_geometry_comparison_v5_base',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    strict=m._semantic_geometry
    def normalized(model,labels,frozen_path_ids=None):
        return strict(_canonical_model(model),labels,frozen_path_ids)
    m._semantic_geometry=normalized
    return m,strict

_BASE,_STRICT = _load_base()

class ScopedSourceQualityComparison(_BASE.ScopedSourceQualityComparison):
    def compare(self,model_path,output_dir):
        report=super().compare(model_path,output_dir)
        model=_BASE._read(model_path)
        report.update(schema='rinkfit.scoped_source_quality_comparison.v5',
            numeric_JSON_representation_normalization=dict(field='circle.start_angle',operation='finite int/float to exact binary64; no tolerance',
                raw_reference_geometry_digest=_STRICT(self.reference,self.ice_labels,self.frozen_path_ids),
                raw_candidate_geometry_digest=_STRICT(model,self.ice_labels,self.frozen_path_ids),
                geometry_guard_disabled=False,actual_geometric_parameter_change_allowed=False),
            unchanged_numerical_comparison=_BASE._ref(_BASE.__file__),producer=_BASE._ref(__file__))
        (Path(output_dir)/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        return report

def main():
    p=argparse.ArgumentParser()
    for key in ('binding','model','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--shots',nargs='+',default=['s01','s02','s03','s04'])
    a=p.parse_args();r=ScopedSourceQualityComparison(a.binding,a.shots).compare(a.model,a.out)
    print(json.dumps(dict(output=a.out,native_frames=r['native_frames'],grid_status=r['grid']['status'],grid=r['grid'].get('all')),ensure_ascii=False))

if __name__=='__main__':main()
