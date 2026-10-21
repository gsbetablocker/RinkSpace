"""Strict unchanged-prefix inheritance for reviewed static ice point appends.

The inherited board certificate is verified in full. This wrapper grants no
board authority to the suffix and binds each new UV to a reviewed native track.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .io_utils import file_sha256, write_json
from .board_source_qualification_guard import _load_ref, FAMILY_PREFIXES, BOUND_ARRAYS

SCHEMA = 'rinkfit.board_material_fit_qualification.v4_static_ice_prefix'
BUNDLE_SCHEMA = 'rinkfit.native_static_ice_tracklet_qualified_bundle.v1'
SOURCE_SHA = 'e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'


def _ref(path):
    path=Path(path).resolve()
    return dict(file=str(path),sha256=file_sha256(path))


EXECUTED_MODULE_REF=_ref(Path(__file__))


def _array(doc,name,base,seen):
    d=doc['arrays'][name];p=_load_ref(d,base,seen)
    return np.memmap(p,dtype=d['dtype'],mode='r',offset=int(d.get('offset_bytes',0)),shape=tuple(d['shape']))


def _prefix_arrays(parent,packet,oldbase,newbase,n,m,seen):
    if set(parent['arrays'])!=set(packet['arrays']):raise ValueError('Static ice append changes inherited array fields')
    arrays={}
    for name,old in parent['arrays'].items():
        new=packet['arrays'][name]
        if (old['dtype']!=new['dtype'] or old['shape'][0]!=n or new['shape'][0]!=m or old['shape'][1:]!=new['shape'][1:]):raise ValueError('Static ice append changes array ABI: '+name)
        a=_load_ref(old,oldbase,seen);b=_load_ref(new,newbase,seen)
        size=n*np.dtype(old['dtype']).itemsize*int(np.prod(old['shape'][1:] or [1]))
        with a.open('rb')as x,b.open('rb')as y:
            x.seek(int(old.get('offset_bytes',0)));y.seek(int(new.get('offset_bytes',0)))
            while size:
                count=min(size,1024*1024);left=x.read(count);right=y.read(count)
                if len(left)!=count or left!=right:raise ValueError('Static ice append changes inherited raw prefix: '+name)
                size-=count
        arrays[name]=_array(packet,name,newbase,seen)
    return arrays


def _check_suffix(packet,parent,packet_path,parentpath,bundle_path,seen,issued_arrays):
    bundle=json.loads(bundle_path.read_text(encoding='utf8'))
    if bundle.get('schema')!=BUNDLE_SCHEMA or bundle.get('source_ledger_preserved')is not True:raise ValueError('Static ice append lacks the qualified source bundle')
    refs=('model','source_generation','static_surface_qualification','source_rows_ref','qualification_receipts')
    resolved={k:_load_ref(bundle[k],bundle_path.parent,seen)for k in refs}
    if not bundle.get('reader_module_refs'):raise ValueError('Static ice bundle lacks exact executed reader modules')
    for ref in bundle['reader_module_refs']:_load_ref(ref,bundle_path.parent,seen)
    model=json.loads(resolved['model'].read_text(encoding='utf8'))
    generation=json.loads(resolved['source_generation'].read_text(encoding='utf8'))
    qualification=json.loads(resolved['static_surface_qualification'].read_text(encoding='utf8'))
    if qualification.get('schema')!='rinkfit.static_ice_texture_qualification.v1' or qualification.get('source_video_sha256')!=SOURCE_SHA:raise ValueError('Static ice source qualification identity differs')
    tracksref=generation['tracks_file'];_load_ref(tracksref,resolved['source_generation'].parent,seen)
    if tracksref['sha256']!=qualification['tracks_file_sha256'] or generation['source_video']['sha256']!=SOURCE_SHA or generation['source_video']['time_base']!='1/12800' or generation.get('no_cross_cut_or_scope_gap')is not True:raise ValueError('Static ice acquisition native clock/track ledger differs')
    for ref in generation.get('actor_ledgers',[]):_load_ref(ref,resolved['source_generation'].parent,seen)
    _load_ref(generation['scope'],resolved['source_generation'].parent,seen)
    evidence={};accepted=set()
    for track in qualification['tracks']:
        if track['status']!='accepted_static_ice':continue
        tid=track['track_id'];accepted.add(tid)
        if track['diagnostic']['two_direction_rows']!=len(track['evidence']):raise ValueError('Static track contains one-direction or unreviewed rows')
        for row in track['evidence']:
            key=(tid,int(row['native_pts']))
            if key in evidence:raise ValueError('Repeated static ice qualification exposure')
            _load_ref(row['review_file'],resolved['static_surface_qualification'].parent,seen)
            evidence[key]=row
    p=resolved['qualification_receipts']
    if p.suffix=='.json':
        value=json.loads(p.read_text(encoding='utf8'));rows=value['rows']if isinstance(value,dict)else value
    else:
        with p.open('r',encoding='utf8')as stream:rows=[json.loads(line)for line in stream if line.strip()]
    with resolved['source_rows_ref'].open('r',encoding='utf8')as stream:source_rows=[json.loads(line)for line in stream if line.strip()]
    n=int(parent['row_count']);m=int(packet['row_count'])
    if len(rows)!=m-n or len(source_rows)!=len(rows) or len(evidence)!=len(rows) or len(accepted)!=qualification['accepted_tracks'] or len(rows)!=qualification['accepted_rows']:raise ValueError('Static ice suffix does not cover the exact reviewed source scope')
    issued=dict(packet,arrays=issued_arrays)
    _prefix_arrays(parent,issued,Path(parentpath).parent,Path(packet_path).parent,n,m,seen)
    arrays={k:_array(packet,k,Path(packet_path).parent,seen)for k in packet['arrays']}
    seenrows=set();track_material={};material_track={}
    for j,row in enumerate(rows):
        i=n+j;tid=row.get('track_id',row.get('physical_track_id'));pts=int(row['native_pts']);key=(tid,pts)
        r=evidence.get(key)
        if r is None or key in seenrows:raise ValueError('Static ice suffix has an unknown/repeated reviewed exposure')
        seenrows.add(key)
        uv=row.get('source_uv',row.get('current_uv'));mi=int(row['joint_material_index']);sid=row['shot_id'];rid='static_ice_texture/'+tid+'/'+str(pts)
        original=source_rows[j]
        if (any(original.get(k)!=row.get(k)for k in ('track_id','native_pts','shot_id','source_uv','joint_material_index','current_rgb_sha256','target_id'))
            or original.get('source_row_id')!=rid or original.get('current_uv')!=uv or original.get('source_video_sha256')!=SOURCE_SHA
            or original.get('time_base')!=[1,12800] or original.get('surface_class')!='ice' or original.get('surface_instance_id')!='ice'
            or original.get('static_material_qualified')is not True or original.get('pixels_from_model')is not False
            or original.get('point_correspondence')is not True or original.get('world_position_is_shared_track_latent')is not True
            or original.get('frame_role')!='development_candidate' or original.get('numeric_statistics_ready')is not False
            or original.get('statistical_objective_eligible')is not False):raise ValueError('Static ice suffix redirects the actual native source journal')
        if row['qualification_ref']!=bundle['static_surface_qualification']:raise ValueError('Static ice receipt qualification dependency differs')
        if row.get('status')!='accepted_static_ice' or row['current_rgb_sha256']!=r['current_rgb_sha256'] or uv!=r['source_uv']:raise ValueError('Static ice receipt differs from actual reviewed source pixels')
        if packet['source_row_ids'][i]!=rid or str(packet['source_row_ids'][i]).startswith(FAMILY_PREFIXES):raise ValueError('Static ice source ID redirects to another family')
        if sid!=tid.split('/')[0] or packet['shot_ids'][int(arrays['shot_index'][i])]!=sid or int(arrays['native_pts'][i])!=pts or (pts-256256)%256:raise ValueError('Static ice suffix changes shot/native PTS')
        if not np.array_equal(arrays['uv'][i],uv) or int(arrays['material_index'][i])!=mi or arrays['kind'][i]!=1 or arrays['station'][i]!=0 or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1:raise ValueError('Static ice packet does not consume the exact fixed-point UV/clock')
        material=model['materials'][mi];pathindex=int(arrays['path'][i]);path=material['paths'][pathindex]
        target='ice.tracklet.'+tid.replace('/','.')
        if mi<len(parent['material_binding']['materials']) or material['label']!=target or row['target_id']!=target or not path.get('physical_point') or path['kind']!=0 or pathindex!=0:raise ValueError('Static ice suffix lacks the shared physical ice point material')
        if track_material.setdefault(tid,mi)!=mi or material_track.setdefault(mi,tid)!=tid:raise ValueError('Static ice track does not share one distinct physical point')
    if seenrows!=set(evidence):raise ValueError('Static ice source coverage incomplete')
    return bundle,arrays


def assert_static_ice_prefix_qualification(packet,packet_path,*,verified_dependency_cache=None,parent_document=None,certificate=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache
    packet_path=Path(packet_path).resolve();certpath=_load_ref(packet['board_material_source_qualification'],packet_path.parent,seen)if certificate is None else Path(certificate[0])
    cert=json.loads(certpath.read_text(encoding='utf8'))if certificate is None else certificate[1]
    if cert.get('schema')!=SCHEMA:raise ValueError('Unsupported static ice prefix certificate')
    parentpath=_load_ref(cert['parent_packet'],certpath.parent,seen);parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    from .board_source_qualification_guard import assert_board_material_source_qualification
    assert_board_material_source_qualification(parent,parentpath,verified_dependency_cache=seen)
    for root in (parentpath.parent,certpath.parent):
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            if(root/name).exists():raise ValueError('Static ice append inherited a recalled source certificate')
    n=int(parent['row_count']);m=int(packet['row_count'])
    if n!=cert['original_raw_rows'] or m!=cert['raw_rows'] or m<=n or m-n!=cert['appended_raw_rows'] or packet['shot_ids']!=parent['shot_ids']:raise ValueError('Static ice append raw prefix/scope differs')
    for key in ('source_row_ids','source_dataset_ids','source_family_ids'):
        if len(packet[key])!=m or packet[key][:n]!=parent[key]:raise ValueError('Static ice append redirects inherited source registry')
    for key in ('queries','independent_rows','independent_statistic_kind','stroke_pairs','source_station_domains'):
        if packet.get(key)!=parent.get(key):raise ValueError('Static ice append changes inherited physical/query/statistic scope: '+key)
    if packet['groups'][:len(parent['groups'])]!=parent['groups']:raise ValueError('Static ice append changes inherited group/W definitions')
    suffix=[]
    for g in packet['groups'][len(parent['groups']):]:
        if g.get('statistic_kind')!='constraint_only_unknown_noise' or g['rank']!=0 or g['W']!=[] or g.get('unknown_noise_source_has_statistical_precision')is not False:raise ValueError('Static ice suffix invents statistical precision')
        suffix.extend(g['rows'])
    if sorted(suffix)!=list(range(n,m)):raise ValueError('Static ice geometry-only group partition differs')
    if packet['joint_statistics_rank']!=parent['joint_statistics_rank']:raise ValueError('Static ice append changes inherited statistical rank')
    for ref in cert['reader_module_refs']:_load_ref(ref,certpath.parent,seen)
    if cert['reader_module_refs']!=[EXECUTED_MODULE_REF]:raise ValueError('Static ice prefix consumer version changed')
    for name in BOUND_ARRAYS:
        d=packet['arrays'][name]
        if cert['membership_array_sha256'].get(name)!=d['sha256']:raise ValueError('Static ice append source array binding differs')
    changed_aux=any(packet['arrays'][k]['sha256']!=v['sha256']for k,v in cert['issued_arrays'].items()if k not in BOUND_ARRAYS)
    if changed_aux:
        from .joint_solver import _source_membership_fingerprint
        fingerprint=_source_membership_fingerprint(packet);explicit=False
        for ref in packet.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')!=fingerprint:continue
            wrapperpath=_load_ref(ref,packet_path.parent,seen);wrapper=json.loads(wrapperpath.read_text(encoding='utf8'))
            if wrapper.get('board_material_source_qualification')==packet.get('board_material_source_qualification') and _source_membership_fingerprint(wrapper)==fingerprint:
                explicit=True;break
        if not explicit:raise ValueError('Static ice numerical auxiliaries require an explicit verified same-source wrapper')
    bundlepath=_load_ref(cert['actual_append_source_bundle'],certpath.parent,seen)
    _check_suffix(packet,parent,packet_path,parentpath,bundlepath,seen,cert['issued_arrays'])


def assert_board_source_qualification_current(packet,packet_path):
    """Dispatch without changing the archived v3 verifier bound by old proofs."""
    ref=packet.get('board_material_source_qualification')
    if ref:
        path=_load_ref(ref,Path(packet_path).resolve().parent,set())
        if json.loads(path.read_text(encoding='utf8')).get('schema')==SCHEMA:
            return assert_static_ice_prefix_qualification(packet,packet_path)
    from .board_source_qualification_guard import assert_board_material_source_qualification
    return assert_board_material_source_qualification(packet,packet_path)


def finalize_static_ice_tracklet_prefix_certificate(parent_packet_path,appended_packet_document,bundle_path,output_certificate_path,*,parent_document=None):
    parentpath=Path(parent_packet_path).resolve();output=Path(output_certificate_path).resolve();bundlepath=Path(bundle_path).resolve()
    if output.exists():raise FileExistsError('Static ice prefix certificate is immutable')
    packet=appended_packet_document;parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    if _ref(Path(__file__))!=EXECUTED_MODULE_REF:raise ValueError('Static ice finalizer code changed during this execution')
    cert=dict(schema=SCHEMA,parent_packet=_ref(parentpath),original_raw_rows=int(parent['row_count']),raw_rows=int(packet['row_count']),appended_raw_rows=int(packet['row_count'])-int(parent['row_count']),actual_append_source_bundle=_ref(bundlepath),membership_array_sha256={k:packet['arrays'][k]['sha256']for k in BOUND_ARRAYS},issued_arrays=packet['arrays'],reader_module_refs=[EXECUTED_MODULE_REF],inherited_board_qualification_rescanned=True,new_board_qualification_issued=False,new_statistical_precision_invented=False)
    assert_static_ice_prefix_qualification(packet,output.parent/'packet.json',parent_document=parent,certificate=(output,cert))
    write_json(output,cert);packet['board_material_source_qualification']=_ref(output)
    packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
