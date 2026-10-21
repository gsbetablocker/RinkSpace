"""Strict diagnostic transport for the existing real dense source reader.

The source reader and its estimators are reused byte for byte. Nonfinite reader
outputs remain explicit unsupported diagnostic values; they never become zero
residuals, accepted source measurements, or qualification changes.
"""
from pathlib import Path
import json, math, types
import numpy as np
from . import dense_current_source_review as legacy
from .dense_fit_feedback import reference
from .io_utils import read_json


def finite_diagnostic_document(value):
    nonfinite=[]
    def visit(item,path):
        if isinstance(item,np.ndarray):item=item.tolist()
        if isinstance(item,np.generic):item=item.item()
        if isinstance(item,float) and not math.isfinite(item):
            nonfinite.append(dict(json_path=path,value='nan' if math.isnan(item) else ('positive_infinity' if item>0 else 'negative_infinity')))
            return None
        if isinstance(item,dict):return {k:visit(v,path+'/'+str(k).replace('~','~0').replace('/','~1')) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [visit(v,path+'/'+str(i))for i,v in enumerate(item)]
        return item
    result=visit(value,'')
    if nonfinite:
        result['nonfinite_reader_diagnostics']=dict(values=nonfinite,encoded_as_null=True,finite_values_replaced_with_zero=False,qualification_or_candidate_issued=False)
    return result


def _write_strict_new(path,value):
    path=Path(path).resolve()
    if len(str(path).encode('utf-16-le'))//2>=260 or any(len(p.encode('utf-16-le'))//2>255 for p in path.parts):raise ValueError('Source review transport path exceeds project limits')
    value=dict(value,diagnostic_transport=reference(__file__),source_reader_implementation=reference(legacy.__file__))
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8')as stream:stream.write(encoded)
    return reference(path)


def review_dense_findings(**kwargs):
    cfg=read_json(kwargs['config'])if not isinstance(kwargs['config'],dict)else dict(kwargs['config'])
    cfg=dict(cfg,original_reader_policy_sources=list(cfg.get('original_reader_policy_sources',[]))+[reference(__file__)])
    kwargs=dict(kwargs,config=cfg)
    # A private function namespace avoids mutating the frozen reader module or
    # other in-flight consumers. Its actual source remains pinned in the reader
    # dependencies; this transport is additionally pinned above.
    namespace=dict(legacy.__dict__,_plain=finite_diagnostic_document,_new_json=_write_strict_new)
    function=types.FunctionType(legacy.review_dense_findings.__code__,namespace,legacy.review_dense_findings.__name__,legacy.review_dense_findings.__defaults__,legacy.review_dense_findings.__closure__)
    function.__kwdefaults__=legacy.review_dense_findings.__kwdefaults__
    return function(**kwargs)
