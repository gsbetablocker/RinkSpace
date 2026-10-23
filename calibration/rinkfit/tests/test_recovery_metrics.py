import numpy as np
from rinkfit_joint.recovery_metrics import _coverage,_temporal


def test_coverage_keeps_missing_native_exposures_and_invalid_readouts():
    exposures=np.arange(256256,256256+8*256,256,dtype=np.int64)
    pts=exposures[[0,1,5,7]];rows=np.arange(4);valid=np.array([True,False,True,True])
    result=_coverage(exposures,pts,rows,valid)
    assert result['native_exposures']==8
    assert result['source_visible_exposures']==4 and result['valid_source_exposures']==3
    assert result['unsupported_exposures']==4 and result['longest_source_gap_exposures']==3
    assert result['longest_source_gap_PTS']==[int(exposures[2]),int(exposures[4])]


def test_temporal_uses_same_physical_point_and_never_free_station_or_cut():
    uv=np.array([[1.,1.],[2.,1.],[3.,1.],[4.,1.],[5.,1.],[6.,1.]])
    pred=uv.copy();pred[1,0]+=1.
    valid=np.ones(6,bool);active=valid.copy();kind=np.array([1,1,1,1,0,0]);sid=np.array(['s01','s01','s01','s02','s02','s02']);mat=np.array([0,0,0,0,1,1]);pts=256256+np.arange(6)*256
    model=dict(shots={'s01':{},'s02':{}},materials=[dict(label='Ad2.left_top',paths=[dict(physical_point=True)]),dict(label='board.contact',paths=[dict(physical_point=False)])])
    report=_temporal(pred,uv,valid,active,kind,sid,mat,model,pts,np.zeros(6,bool),['r'+str(i) for i in range(6)])
    assert report['by_shot']['s01']['second_difference_error_px']['n']==1
    assert report['by_shot']['s01']['second_difference_error_px']['max']==2.
    assert report['by_shot']['s02']['second_difference_error_px']['n']==0
    assert report['free_station_or_Query_tracks_fabricated'] is False


def test_same_vertex_ID_across_real_native_times_is_legal(tmp_path):
    import json,hashlib
    from rinkfit_joint.recovery_metrics import compare
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    n=3;data={}
    for name,value in dict(uv=np.array([[10.,20.],[11.,20.],[12.,20.]],dtype='<f8'),native_pts=np.array([256256,256512,256768],dtype='<i8'),material_index=np.zeros(n,dtype='<u4'),kind=np.ones(n,dtype='u1'),shot_index=np.zeros(n,dtype='<u2'),admission=np.ones(n,dtype='u1'),fit_role=np.ones(n,dtype='u1'),E=np.ones(n,dtype='<f8')*2).items():
        path=tmp_path/(name+'.bin');value.tofile(path);data[name]=dict(file=str(path),dtype=value.dtype.str,shape=list(value.shape),sha256=sha(path))
    packet=dict(row_count=n,arrays=data,shot_ids=['s%02d'%i for i in range(1,10)],source_row_ids=['same_physical_vertex']*3,source_dataset_ids=['real_immutable_generation']*3,queries=[])
    pp=tmp_path/'packet.json';pp.write_text(json.dumps(packet),encoding='utf8');state=tmp_path/'state';state.mkdir();shotdefs={};first=256256
    for i,length in enumerate([3]+[1]*7+[8424],1):
        shotdefs['s%02d'%i]=dict(scope=dict(first_pts=first,last_pts=first+256*(length-1)));first+=256*length
    model=dict(shots=shotdefs,materials=[dict(label='Ad2.left_top',paths=[dict(physical_point=True)])]);mp=state/'model.json';mp.write_text(json.dumps(model),encoding='utf8');ledger={}
    for prefix in ('','raw_station_RS_'):
        for name,value in dict(predicted_uv=np.array([[11.,20.],[12.,20.],[13.,20.]],dtype='<f8'),error_native_uv=np.array([[1.,0.]]*3,dtype='<f8'),valid=np.array([1,0,1],dtype='u1')).items():
            file=prefix+name+('.u8' if name=='valid' else '.f64');path=state/file;value.tofile(path);ledger[file]=dict(sha256=sha(path),size_bytes=path.stat().st_size)
    from rinkfit_joint.report_identity import make_replay_binding
    value=dict(model_sha256=sha(mp),files=ledger)
    value['source_objective_binding']=make_replay_binding(pp,packet,value,[])
    (state/'value_report.json').write_text(json.dumps(value),encoding='utf8')
    explicit=tmp_path/'CP_explicit_metrics';explicit.mkdir();cpfiles={}
    for name,value in dict(explicit_predicted_uv=np.array([[11.,20.],[13.,20.],[1012.,20.]],dtype='<f8'),explicit_error_native_uv=np.array([[1.,0.],[2.,0.],[1000.,0.]],dtype='<f8'),explicit_RS_equality=np.zeros(n,dtype='<f8'),explicit_query_equality=np.full((n,2),np.nan,dtype='<f8'),explicit_valid=np.array([1,1,0],dtype='u1'),explicit_arithmetic_valid=np.ones(n,dtype='u1'),explicit_ray_domain_valid=np.array([1,1,0],dtype='u1'),explicit_station_domain_valid=np.ones(n,dtype='u1'),explicit_visibility_valid=np.ones(n,dtype='u1'),explicit_width_domain_valid=np.ones(n,dtype='u1')).items():
        file=name+('.u8' if name.endswith('valid') else '.f64');path=explicit/file;value.tofile(path);cpfiles[file]=dict(sha256=sha(path),size_bytes=path.stat().st_size)
    (explicit/'manifest.json').write_text(json.dumps(dict(packet_sha256=sha(pp),raw_rows=n,files=cpfiles,valid_semantics='test native domain validity',state_token='tiny_complete_CP')),encoding='utf8')
    report=compare(pp,[dict(label='candidate',state=str(state),explicit=str(explicit))],tmp_path/'report')
    result=report['states']['candidate']['modes']['fixed_station_independent_RS']
    assert report['states']['candidate']['native_exposures']==8434
    assert report['common_denominator_rows']==3
    assert result['overall']['active_rows']==3 and result['overall']['invalid_rows']==1
    assert result['overall']['rms_px']==1.

    # Existing valid-only aliases remain compatible; the full-N score must
    # retain the undefined native root rather than silently reduce N to two.
    full=result['overall']['same_denominator_statistics']
    assert full['N']==3 and full['undefined_error_rows']==1
    assert full['rms_px']=='+Infinity' and full['rms_status'].startswith('not_finite')
    assert full['median_px']==1. and full['p95_px']=='+Infinity'
    import gzip
    ledger=result['source_failure_evidence']['complete_ledger']
    with gzip.open(ledger['file'],'rt',encoding='utf8') as stream:failed=[json.loads(line) for line in stream]
    assert len(failed)==1 and failed[0]['native_pts']==256512
    assert failed[0]['source_uv']==[11.,20.] and failed[0]['same_denominator_error_px']=='+Infinity'
    assert failed[0]['available_arithmetic_operator_error_px']==1.
    cp=report['states']['candidate']['modes']['CP_explicit']
    assert cp['overall']['rms_px']==np.sqrt(2.5)  # legacy conditional alias
    full=cp['overall']['same_denominator_statistics']
    assert full['N']==3 and full['undefined_error_rows']==0 and full['max_px']==1000.
    assert np.isclose(full['rms_px'],np.sqrt((1+4+1000000)/3))
    assert full['arithmetic_finite_but_mode_invalid_rows']==1
    worst=cp['worst_invalid_arithmetic_finite_source_rows'][0]
    assert worst['native_pts']==256768 and worst['source_uv']==[12.,20.]
    assert worst['same_denominator_error_px']==1000. and 'ray_domain' in worst['failure_flags']
    with gzip.open(cp['source_failure_evidence']['complete_ledger']['file'],'rt',encoding='utf8') as stream:cpfailed=[json.loads(line) for line in stream]
    assert len(cpfailed)==1 and cpfailed[0]['same_denominator_error_px']==1000.
    json.dumps(report,allow_nan=False)  # undefined values use strings/null, never invalid JSON floats


def test_same_population_CP_statistics_do_not_change_with_domain_valid_mask():
    from rinkfit_joint.recovery_metrics import _population_summary
    d=np.array([1.,3.,9.]);ids=np.arange(3);E=np.full(3,2.)
    a=_population_summary(d,ids,np.array([True,True,False]),E,'CP_explicit',{'arithmetic':np.ones(3,bool)})
    b=_population_summary(d,ids,np.array([True,False,True]),E,'CP_explicit',{'arithmetic':np.ones(3,bool)})
    assert a['rms_px']!=b['rms_px']  # legacy valid-only values have different denominators/members
    assert a['same_denominator_statistics']['N']==b['same_denominator_statistics']['N']==3
    assert a['same_denominator_statistics']['rms_px']==b['same_denominator_statistics']['rms_px']
    assert np.isclose(a['same_denominator_statistics']['rms_px'],np.sqrt(91/3))
    assert a['same_denominator_statistics']['max_px']==9.
    assert a['legacy_fields_are_conditional_valid_only'] is True


def test_full_N_nearest_rank_preserves_RS_infinity_and_undefined_arithmetic():
    import json
    from rinkfit_joint.recovery_metrics import _same_denominator_statistics
    d=np.ones(20);ids=np.arange(20);E=np.full(20,2.);valid=np.ones(20,bool);valid[-1]=False
    one=_same_denominator_statistics(d,ids,valid,E,'fixed_station_independent_RS')
    assert one['N']==20 and one['defined_error_rows']==19 and one['undefined_error_rows']==1
    assert one['median_px']==one['p95_px']==1.  # ceil(.95*20)=19, no interpolation toward infinity
    assert one['max_px']==one['rms_px']=='+Infinity'
    valid[-2]=False;two=_same_denominator_statistics(d,ids,valid,E,'fixed_station_independent_RS')
    assert two['N']==20 and two['p95_px']=='+Infinity' and two['rms_status'].startswith('not_finite')
    d[0]=np.nan;cp=_same_denominator_statistics(d,ids,np.ones(20,bool),E,'CP_explicit',{'arithmetic':np.isfinite(d)})
    assert cp['N']==20 and cp['undefined_error_rows']==1 and cp['rms_px']=='+Infinity'
    assert cp['rms_status'].startswith('not_finite')
    json.dumps([one,two,cp],allow_nan=False)


def test_failure_evidence_serializes_nonfinite_values_as_null_or_explicit_string():
    import json
    from rinkfit_joint.recovery_metrics import _source_evidence_record
    data=dict(RS=np.array([np.inf]),Query=np.array([[np.nan,-np.inf]]))
    record=_source_evidence_record(0,'fixed_station_independent_RS',data,np.array([np.nan]),np.array([False]),np.array([np.inf]),
        ['fixed_vertex'],['qualified_source'],np.array(['s01']),np.array(['Ad2.left_top']),np.array([256256]),
        np.array([[10.,20.]]),np.array([[np.inf,-np.inf]]))
    assert record['predicted_uv']==[None,None] and record['source_uv']==[10.,20.]
    assert record['CP_RS_equality_px'] is None and record['CP_Query_equality_px']==[None,None]
    assert record['E_px'] is None and record['available_arithmetic_operator_error_px'] is None
    assert record['same_denominator_error_px']=='+Infinity'
    encoded=json.dumps(record,allow_nan=False)
    assert json.loads(encoded)['same_denominator_error_px']=='+Infinity'
