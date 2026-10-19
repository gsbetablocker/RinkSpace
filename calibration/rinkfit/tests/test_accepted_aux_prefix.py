import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pytest
from rinkfit_joint.accepted_aux_prefix import rebind_engineering_nuisance, _cpp_hex, _quoted
from rinkfit_joint.io_utils import file_sha256


def fixture(tmp_path):
    old=tmp_path/'old'; new=tmp_path/'new'; state=tmp_path/'run/candidate'
    for p in (old,new,state/'explicit_nuisance'): p.mkdir(parents=True)
    def put(folder,name,a):
        a=np.asarray(a); p=folder/(name+'.bin');a.tofile(p)
        return dict(file=str(p),sha256=file_sha256(p),dtype=a.dtype.str,shape=list(a.shape))
    values=dict(native_pts=np.array([256256,256512,256768],'<i8'),shot_index=np.array([0,0,0],'<u2'),
      uv=np.array([[100.,500.],[200.,600.],[300.,700.]],'<f8'),kind=np.array([0,1,0],'u1'),
      material_index=np.array([0,1,0],'<u4'),E=np.array([2.,2.,2.],'<f8'),
      admission=np.array([1,1,1],'u1'),fit_role=np.array([1,1,1],'u1'),
      path=np.array([0,0,0],'<u4'),station=np.array([.5,.5,.5],'<f8'),
      row=np.array([500.,600.,700.],'<f8'),alpha=np.array([0.,0.,.7],'<f8'))
    query=dict(row_index=0,alpha=[-1.,1.],native_uv=[[99.,500.],[101.,500.]],normal=[1.,0.])
    groups=[dict(id='g'+str(i),rows=[i],rank=0,W=[],statistic_kind='constraint_only_unknown_noise',unknown_noise_source_refs=[dict(file='fixed-source',sha256='a'*64)]) for i in range(3)]
    docs=[]
    for folder,n in ((old,2),(new,3)):
        doc=dict(schema='rinkfit.joint_packet.v1',row_count=n,shot_ids=['s01'],
          source_dataset_ids=['source']*n,source_row_ids=['r'+str(i)for i in range(n)],
          arrays={k:put(folder,k,a[:n])for k,a in values.items()},queries=[query],groups=groups[:n])
        p=folder/'packet.json';p.write_text(json.dumps(doc),encoding='utf8');docs.append(doc)
    parent=old/'packet.json';src=new/'packet.json';source_sha=file_sha256(parent)
    model=dict(schema='tiny',blocks=[dict(name='scene/x',values=[1.25],constant=False,fixed_indices=[])])
    (state/'model.json').write_text(json.dumps(model),encoding='utf8')
    aux=dict(station=np.array([.25,.75],'<f8'),row=np.array([501.,601.],'<f8'),alpha=np.array([.1,0.],'<f8'),path=np.array([0,0],'<u4'))
    files={}
    for name,a in aux.items():
        f=name+('.u32'if name=='path'else'.f64');p=state/'explicit_nuisance'/f;a.tofile(p);files[f]=dict(sha256=file_sha256(p),size_bytes=p.stat().st_size)
    manifest=dict(schema='rinkfit.joint_explicit_PHR_nuisance_state.v1',packet_sha256=source_sha,raw_rows=2,files=files)
    (state/'explicit_nuisance/manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    identity=[source_sha,'model-topology','budget','statistics','scales','engineering_recovery_native_pixel_v1']
    params=[('scene/x',[1.25]),('station/0',[.25]),('row/0',[501.]),('row/1',[601.]),('alpha/0',[.1]),('budget_sq',[0.])]
    discrete={'finite_material_paths_RLE_u32_v1':'2:0;','joint_packet_sha256':source_sha}
    canonical=' '.join(_quoted(v)for v in identity)+'\n'+str(len(params))+'\n'
    canonical+=''.join(_quoted(k)+' 0 '+str(len(v))+''.join(' '+_cpp_hex(x)for x in v)+'\n'for k,v in params)
    canonical+=str(len(discrete))+'\n'+''.join(_quoted(k)+' '+_quoted(v)+'\n'for k,v in sorted(discrete.items()))+'0\n'
    token=14695981039346656037
    for b in canonical.encode():token=((token^b)*1099511628211)&((1<<64)-1)
    token=f'{token:016x}'
    cp='RINKFIT_AL_CHECKPOINT_V1\n"config"\n2 0 0 0 0 inf 0\n0\n0\n1\n'+_quoted(token)+'\n'+' '.join(_quoted(v)for v in identity)+'\n'+str(len(params))+'\n'
    cp+=''.join(_quoted(k)+' 0\n'+str(len(v))+''.join(' '+repr(x)for x in v)+'\n'for k,v in params)
    cp+=str(len(discrete))+'\n'+''.join(_quoted(k)+' '+_quoted(v)+'\n'for k,v in sorted(discrete.items()))+'0\n0\nEND_RINKFIT_AL_CHECKPOINT\n'
    (state/'engineering_checkpoint.json').write_text(cp,encoding='utf8')
    receipt=dict(schema='rinkfit.joint_engineering_recovery_run.v1',objective_id=identity[-1],configuration_sha256='config',candidate_state_token=token,
      source_packet=str(parent),source_packet_sha256=source_sha,model_sha256=file_sha256(state/'model.json'),source_recalled=False,
      formal_Phase_II_KKT_passed=False,common_worst_budget_consumed=False,original_W_likelihood_consumed=False,callback_error='')
    (state.parent/'engineering_receipt.json').write_text(json.dumps(receipt),encoding='utf8')
    return src,state,tmp_path/'bound/packet.json',docs,aux


def test_engineering_append_keeps_complete_candidate_prefix_and_only_new_fresh(tmp_path):
    src,state,out,_,aux=fixture(tmp_path)
    result=rebind_engineering_nuisance(src,state,out)
    doc=json.loads(out.read_text());binding=doc['accepted_explicit_state_binding']
    assert result['preserved_raw_rows']==2 and result['fresh_appended_rows']==1
    assert binding['state_kind']=='engineering_candidate' and binding['old_AL_duals_resumed'] is False
    for name,a in aux.items():
        d=doc['arrays'][name];v=np.fromfile(d['file'],dtype=d['dtype'])
        assert np.array_equal(v[:2],a)
    assert np.fromfile(doc['arrays']['station']['file'],'<f8')[2]==.5
    assert np.fromfile(doc['arrays']['row']['file'],'<f8')[2]==700.
    assert np.fromfile(doc['arrays']['alpha']['file'],'<f8')[2]==.7
    assert binding['native_replay_pass_claimed'] is False and binding['formal_acceptance_claimed'] is False


@pytest.mark.parametrize('mutation',['UV','Query','W','ID','E','normal'])
def test_changed_original_source_prefix_is_rejected(tmp_path,mutation):
    src,state,out,_,_=fixture(tmp_path);doc=json.loads(src.read_text())
    if mutation in ('UV','E'):
        name='uv'if mutation=='UV'else'E';d=doc['arrays'][name];a=np.fromfile(d['file'],d['dtype']).reshape(d['shape']);a.flat[0]+=1;a.tofile(d['file']);d['sha256']=file_sha256(d['file'])
    elif mutation=='Query':doc['queries'][0]['native_uv'][0][0]+=1
    elif mutation=='normal':doc['queries'][0]['normal']=[0.,1.]
    elif mutation=='W':doc['groups'][0]['W']=[1.]
    else:doc['source_row_ids'][0]='other ID'
    src.write_text(json.dumps(doc),encoding='utf8')
    with pytest.raises(ValueError,match='prefix'):rebind_engineering_nuisance(src,state,out)


@pytest.mark.parametrize('mutation',['token','objective','duals','physical','aux'])
def test_wrong_engineering_state_or_old_duals_are_rejected(tmp_path,mutation):
    src,state,out,_,_=fixture(tmp_path);cp=state/'engineering_checkpoint.json';text=cp.read_text()
    if mutation=='token':text=text.replace('"scene/x" 0\n1 1.25','"scene/x" 0\n1 1.5')
    elif mutation=='objective':text=text.replace('engineering_recovery_native_pixel_v1','old_PHR_objective')
    elif mutation=='duals':text=text.replace('0\n0\n1\n','1 2\n0\n1\n',1)
    elif mutation=='physical':
        p=state/'model.json';m=json.loads(p.read_text());m['blocks'][0]['values']=[2.];p.write_text(json.dumps(m));r=state.parent/'engineering_receipt.json';j=json.loads(r.read_text());j['model_sha256']=file_sha256(p);r.write_text(json.dumps(j))
    else:text=text.replace('"row/0" 0\n1 501.0','"row/0" 0\n1 502.0')
    cp.write_text(text,encoding='utf8')
    with pytest.raises(ValueError):rebind_engineering_nuisance(src,state,out)


def test_independent_reprofile_is_not_candidate_state(tmp_path):
    src,state,out,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='not independent reprofile'):rebind_engineering_nuisance(src,state.parent/'independent_RS_state',out)


def test_exact_model_repack_restores_only_nonconsumed_mirror():
    from rinkfit_joint.accepted_aux_prefix import _exact_model_from_CP
    old=dict(blocks=[dict(name='scene/x',values=[.5]),dict(name='shot/s01/control/0',values=[1.,2.,3.]),dict(name='motion/master',values=[1.,2.,3.])],
             shots=dict(s01=dict(control_blocks=[1],control_coordinates=[dict(block=2,offset=0)])),rigs=dict(main=dict(reference_C=[1.,2.,3.])))
    exact,changes=_exact_model_from_CP(old,[[.5],[9.,8.,7.],[1.,2.,3.]])
    assert exact['blocks'][1]['values']==[9.,8.,7.] and len(changes)==1
    assert exact['rigs']==old['rigs'] and exact['shots']==old['shots'] and exact['blocks'][2]==old['blocks'][2]
    assert old['blocks'][1]['values']==[1.,2.,3.]


@pytest.mark.parametrize('mutation',['consumed_mirror','changed_master','other_physical'])
def test_exact_repack_cannot_change_an_actual_projection_value(mutation):
    from rinkfit_joint.accepted_aux_prefix import _exact_model_from_CP
    old=dict(blocks=[dict(name='scene/x',values=[.5]),dict(name='shot/s01/control/0',values=[1.,2.,3.]),dict(name='motion/master',values=[1.,2.,3.])],
             shots=dict(s01=dict(control_blocks=[1],control_coordinates=[dict(block=2,offset=0)])))
    cp=[[.5],[9.,8.,7.],[1.,2.,3.]]
    if mutation=='consumed_mirror':old['shots']['s02']=dict(control_blocks=[1])
    elif mutation=='changed_master':cp[2]=[1.,2.,4.]
    else:cp[0]=[.75]
    with pytest.raises(ValueError):_exact_model_from_CP(old,cp)


def test_native_MSVC_hexfloat_token_encoding_includes_zero_fraction_digits():
    assert _cpp_hex(0.)=='0x0.0000000000000p+0'
    assert _cpp_hex(-0.)=='-0x0.0000000000000p+0'
    assert _cpp_hex(1.25)=='0x1.4000000000000p+0'


def test_many_W_groups_hash_and_map_one_immutable_dependency_once(tmp_path,monkeypatch):
    import rinkfit_joint.accepted_aux_prefix as module
    p=tmp_path/'shared_W.f64';np.arange(12,dtype='<f8').tofile(p);sha=file_sha256(p)
    groups=[dict(id='g'+str(i),rows=[i],rank=2,statistic_kind='original',W=dict(file=str(p),sha256=sha,dtype='<f8',shape=[2,2],offset_bytes=32*i))for i in range(3)]
    calls=dict(hash=0,mapping=0);real_hash=module.file_sha256;real_map=module.np.memmap
    def hashed(path):calls['hash']+=1;return real_hash(path)
    def mapped(*args,**kwargs):calls['mapping']+=1;return real_map(*args,**kwargs)
    monkeypatch.setattr(module,'file_sha256',hashed);monkeypatch.setattr(module.np,'memmap',mapped)
    digest,count=module._runtime_group_digest(dict(row_count=3,groups=groups),tmp_path,set())
    assert count==3 and len(digest)==64 and calls==dict(hash=1,mapping=1)
