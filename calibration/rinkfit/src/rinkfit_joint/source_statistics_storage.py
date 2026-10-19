"""Compact immutable MC artifacts; retain exact normal covariance provenance."""
import hashlib
import numpy as np


def compact_normal_replay(arrays):
    result=dict(arrays);result.pop('full_uv_covariance_px2',None)
    samples=np.asarray(result.pop('normal_shift_delta_px'),dtype='<f8')
    C=np.asarray(result['normal_covariance_px2'],float)
    if samples.ndim!=2 or C.shape!=(samples.shape[1],samples.shape[1]):raise ValueError('invalid exact MC sample/covariance layout')
    result['MC_trial_count']=np.asarray(samples.shape[0],np.int64)
    result['MC_normal_shift_mean_px']=samples.mean(0)
    result['MC_normal_shift_minmax_px']=np.stack((samples.min(0),samples.max(0)))
    result['MC_normal_shift_deltas_sha256']=np.asarray(hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest())
    boundary=np.asarray(result.get('synthetic_window_boundary_counts',np.zeros(samples.shape[1],int)))
    flat=np.asarray(result.get('synthetic_flat_peak_counts',np.zeros(samples.shape[1],int)))
    bad=np.flatnonzero((boundary>0)|(flat>0)|(np.diag(C)<=0))
    result['diagnostic_nonregular_source_indices']=bad
    if len(bad):result['diagnostic_nonregular_normal_shift_delta_px']=samples[:,bad]
    return result


def encode_rng_state(state):
    import copy
    result=copy.deepcopy(state)
    for name in ('state','inc'):
        if name in result.get('state',{}):result['state'][name]=hex(int(result['state'][name]))
    return result


def restore_rng_state(encoded):
    import copy
    state=copy.deepcopy(encoded)
    for name in ('state','inc'):
        if name in state.get('state',{})and isinstance(state['state'][name],str):state['state'][name]=int(state['state'][name],0)
    constructor=getattr(np.random,state['bit_generator'])
    generator=np.random.Generator(constructor());generator.bit_generator.state=state
    return generator
