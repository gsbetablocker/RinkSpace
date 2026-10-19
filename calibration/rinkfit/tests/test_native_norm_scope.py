from copy import deepcopy
import hashlib,json
import numpy as np
import pytest
from rinkfit_start.native_norms import validate_semantic_scope


def fixture(tmp_path,mask=None):
    mask=np.array([1,1,1,0,1]if mask is None else mask,dtype=np.uint8);kind=np.array([0,1,2,0,2],dtype=np.uint8)
    def write(name,a):
        p=tmp_path/name;p.write_bytes(np.ascontiguousarray(a).tobytes());return dict(file=str(p),dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    def jwrite(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj),encoding='utf8');return dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    md=write('mask.u8',mask);kd=write('kind.u8',kind);na=int(mask.sum());profiles=int(np.count_nonzero(mask&(kind==2)));entities=int(np.count_nonzero(mask&(kind==1)))
    contract=dict(raw_rows=5,packet_sha256='a'*64,source_identity='b'*64,original_formal_rank=8,original_groups_asset_sha256='c'*64,original_weights_array_sha256='d'*64)
    stats=dict(active_source_mask_sha256=md['sha256'],active_observations=na,formal_residual_rows=2*na-profiles,original_groups_asset_sha256='c'*64,original_weights_array_sha256='d'*64,independent_statistics_unchanged=True)
    sem=dict(observations=5,active_observations=na,active_source_mask=md,arrays=dict(kind=kd),active_statistics=jwrite('stats.json',stats),immutable_source_arrays={})
    pk=dict(observations=5,immutable_source_hashes={'observations.npz':'b'*64},arrays={})
    for s,p,shape in [('source_uv','native_uv',[5,2]),('native_pts','native_pts',[5]),('frame_indices','frame_index',[5])]:
        h=hashlib.sha256(s.encode()).hexdigest();pk['arrays'][p]=dict(sha256=h,shape=shape);sem['immutable_source_arrays'][s]=dict(sha256_c_contiguous_array_bytes=h,shape=shape)
    t=dict(raw_rows=5,active_rows=na,entity_count=entities,profile_count=profiles,excluded_rows=np.flatnonzero(mask==0).tolist(),full_W_rank=stats['formal_residual_rows'],statistical_partition_exactly_once=True,raw_arrays_preserved=True,source_order_is_raw_array_order=True,groups=[{}for _ in range(na)],original_groups_including_independent=na,packet=dict(sha256='a'*64),semantic=dict(sha256='e'*64))
    return t,sem,pk,contract,stats,write,jwrite


@pytest.mark.parametrize('mask,active,profiles,rank',[(None,4,2,6),([1,1,0,0,1],3,1,5)])
def test_selected_mask_changes_only_active_counts_and_formal_rank(tmp_path,mask,active,profiles,rank):
    t,s,p,c,_,_,_=fixture(tmp_path,mask);scope,m,k=validate_semantic_scope(t,s,p,c,tmp_path)
    assert scope['raw_rows']==5 and scope['active_rows']==active
    assert scope['entity_count']==1 and scope['profile_count']==profiles
    assert scope['full_W_rank']==rank
    assert scope['excluded_source_rows']==np.flatnonzero(~m).tolist()


@pytest.mark.parametrize('field',['active_rows','entity_count','profile_count','full_W_rank','excluded_rows'])
def test_stale_transport_scope_is_rejected(tmp_path,field):
    t,s,p,c,_,_,_=fixture(tmp_path)
    t[field]=[]if field=='excluded_rows'else t[field]+1
    with pytest.raises(ValueError):validate_semantic_scope(t,s,p,c,tmp_path)


def test_mask_changes_require_bound_statistics_not_only_matching_count(tmp_path):
    t,s,p,c,_,write,_=fixture(tmp_path);s['active_source_mask']=write('new_mask.u8',np.array([0,1,1,1,1],dtype=np.uint8));t['excluded_rows']=[0]
    with pytest.raises(ValueError,match='statistics are bound'):validate_semantic_scope(t,s,p,c,tmp_path)


@pytest.mark.parametrize('which',['packet','source','UV','PTS','frame','weights','group_asset'])
def test_raw_source_and_original_statistics_cannot_be_swapped(tmp_path,which):
    t,s,p,c,stats,_,jwrite=fixture(tmp_path)
    if which=='packet':t['packet']['sha256']='f'*64
    elif which=='source':p['immutable_source_hashes']['observations.npz']='f'*64
    elif which in ['UV','PTS','frame']:
        key={'UV':'source_uv','PTS':'native_pts','frame':'frame_indices'}[which];s['immutable_source_arrays'][key]['sha256_c_contiguous_array_bytes']='f'*64
    else:
        key='original_weights_array_sha256'if which=='weights'else'original_groups_asset_sha256';stats[key]='f'*64;s['active_statistics']=jwrite('changed_stats.json',stats)
    with pytest.raises(ValueError):validate_semantic_scope(t,s,p,c,tmp_path)


def test_cached_gpu_partials_must_match_selected_mask_and_semantic(tmp_path):
    t,s,p,c,_,_,_=fixture(tmp_path)
    identity=dict(packet=t['packet']['sha256'],semantic='f'*64,source=c['source_identity'],active_mask=s['active_source_mask']['sha256'],statistics=s['active_statistics']['sha256'])
    t['gpu_rows']=dict(raw_rows=5,active_rows=t['active_rows'],excluded_rows=t['excluded_rows'],packet_identity=identity)
    with pytest.raises(ValueError,match='GPU partials'):validate_semantic_scope(t,s,p,c,tmp_path)


def test_non_boolean_mask_and_changed_raw_count_are_rejected(tmp_path):
    t,s,p,c,_,write,_=fixture(tmp_path);s['active_source_mask']=write('float_mask.f64',np.array([1.,1.,1.,0.,1.]))
    with pytest.raises(ValueError,match='Boolean bytes'):validate_semantic_scope(t,s,p,c,tmp_path)
    t,s,p,c,_,_,_=fixture(tmp_path);t['raw_rows']=4
    with pytest.raises(ValueError,match='raw source count'):validate_semantic_scope(t,s,p,c,tmp_path)


def derived_fixture(tmp_path):
    t,s,p,c,_,write,jwrite=fixture(tmp_path)
    for source,key,shape in [('source_uv','native_uv',(5,2)),('native_pts','native_pts',(5,)),('frame_indices','frame_index',(5,))]:
        desc=write(key+'.f64',np.arange(np.prod(shape),dtype=np.float64).reshape(shape))
        p['arrays'][key]=desc;s['immutable_source_arrays'][source]=dict(sha256_c_contiguous_array_bytes=desc['sha256'],shape=desc['shape'])
    strings=tmp_path/'strings.json';strings.write_text('{}',encoding='utf8');p['string_tables']=dict(file=str(strings),sha256=hashlib.sha256(strings.read_bytes()).hexdigest())
    asset=tmp_path/'seeds.json';asset.write_text('{}',encoding='utf8');p['assets']={'seeds.json':dict(sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),role='seed')}
    p['clock']={'rate':50,'source':'original'}
    parent=jwrite('original_packet.json',p);c['packet_sha256']=parent['sha256']
    derived=deepcopy(p);derived['source_scope_packet_resolution']={'original_raw_packet':parent}
    derived['semantic_packet']=deepcopy(t['semantic']);derived['assets']['seeds.json']['file']=str(asset)
    t['packet']=jwrite('derived_packet.json',derived)
    return t,s,derived,c,jwrite


def test_controlled_derived_packet_resolves_paths_without_changing_source(tmp_path):
    t,s,p,c,_=derived_fixture(tmp_path)
    scope,_,_=validate_semantic_scope(t,s,p,c,tmp_path)
    assert scope['original_packet_sha256']==c['packet_sha256']
    assert scope['packet_sha256']==t['packet']['sha256']
    assert scope['packet_sha256']!=scope['original_packet_sha256']


@pytest.mark.parametrize('change',['raw_descriptor','clock','labels','asset_metadata','asset_payload','missing_asset','parent','semantic'])
def test_derived_packet_cannot_change_original_source_or_asset_payload(tmp_path,change):
    t,s,p,c,jwrite=derived_fixture(tmp_path)
    if change=='raw_descriptor':p['arrays']['native_uv']['shape']=[10]
    elif change=='clock':p['clock']['rate']=60
    elif change=='labels':p['string_tables']['sha256']='f'*64
    elif change=='asset_metadata':p['assets']['seeds.json']['role']='changed'
    elif change=='asset_payload':(tmp_path/'seeds.json').write_text('{"changed":true}',encoding='utf8')
    elif change=='missing_asset':p['assets'].clear()
    elif change=='parent':p['source_scope_packet_resolution']['original_raw_packet']['sha256']='f'*64
    elif change=='semantic':p['semantic_packet']['sha256']='f'*64
    with pytest.raises(ValueError):validate_semantic_scope(t,s,p,c,tmp_path)


@pytest.mark.parametrize('corrupt',[None,'revision_identity','new_excluded','stats_revision','stats_exclusion'])
def test_selected_audit_revision_keeps_historical_exclusion_and_binds_new_authority(tmp_path,corrupt):
    t,s,p,c,stats,_,jwrite=fixture(tmp_path)
    s['source_exclusion']=jwrite('historical_exclusion.json',{'excluded':[3]})
    revision=dict(source_identity=c['source_identity'],raw_observations=5,active_observations=4,historical_human_authorized_exclusions=[3],new_audit_exclusions=[],raw_rows_UV_ID_labels_W_E_remain_immutable=True)
    if corrupt=='revision_identity':revision['source_identity']='f'*64
    if corrupt=='new_excluded':revision['new_audit_exclusions']=[0]
    s['source_audit_revision']=jwrite('revision.json',revision)
    stats['source_audit_revision_sha256']=s['source_audit_revision']['sha256'];stats['source_exclusion_sha256']=s['source_audit_revision']['sha256']
    if corrupt=='stats_revision':stats['source_audit_revision_sha256']='f'*64
    if corrupt=='stats_exclusion':stats['source_exclusion_sha256']=s['source_exclusion']['sha256']
    s['active_statistics']=jwrite('stats.json',stats)
    if corrupt is None:validate_semantic_scope(t,s,p,c,tmp_path)
    else:
        with pytest.raises(ValueError):validate_semantic_scope(t,s,p,c,tmp_path)
