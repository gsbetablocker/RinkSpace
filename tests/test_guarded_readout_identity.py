"""Guard propagation/export preserve exact raw IDs and source discontinuities."""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np
import pytest

PROG=Path(__file__).resolve().parents[1]/'events/ankara_2026/programmes/okada_mei_sp'
if str(PROG) not in sys.path:sys.path.insert(0,str(PROG))
from readout_exclusions import ReadoutExclusions, retained_runs, digest
from propagate_ice_readout_identities import guarded_candidate_links, verified_line_groups, propagate_shot, read, write
from export_identity_source_rows import spaced_actual_keys


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n',encoding='utf8')


@pytest.fixture
def exclusion_files(tmp_path):
    capture=tmp_path/'capture';put(capture/'binding.json',dict(source_sha256='a'*64))
    folder=tmp_path/'exclusions';folder.mkdir();records=[]
    for pts in (256,512):
        raw=capture/'s07'/f'{pts}.npz';raw.parent.mkdir(exist_ok=True)
        np.savez(raw,candidate_uv=np.c_[np.arange(6)*28.,np.zeros(6)])
        arrays=folder/f'{pts}.npz';np.savez(arrays,affected_candidate_ids=np.array([2],np.int32),trigger_bits=np.array([64],np.uint8))
        records.append(dict(capture='full_direct',shot='s07',pts=pts,source_BGR_sha256='b'*64,source_RGB_sha256='c'*64,
            raw_measurement=dict(path=str(raw),sha256=digest(raw)),arrays=dict(path=str(arrays),sha256=digest(arrays)),
            applied_to_all_original_candidate_ids=True,unresolved_support=True,added_source_exclusion_pixels=1,
            evaluated_original_candidate_count=6,affected_candidate_count=1))
    index=folder/'frames_index.jsonl';coverage=folder/'coverage.json'
    report=dict(schema='rinkspace.all_raw_candidates_added_person_guard.v1',status='complete',all_raw_candidates_considered=True,
        source_candidate_id_spaces_separate=True,source_exposures_per_capture=2,output_frames=2,
        capture_bindings=dict(full_direct=dict(path=str(capture/'binding.json'),sha256=digest(capture/'binding.json'))),
        by_capture_shot={'full_direct/s07':dict(frames=2)})
    def save():
        index.write_text(''.join(json.dumps(r)+'\n'for r in records),encoding='utf8')
        report['index']=dict(path=str(index),sha256=digest(index));put(coverage,report)
    save()
    return dict(capture=capture,coverage=coverage,index=index,records=records,report=report,save=save)


def consume(data,helper=None):
    helper=helper or ReadoutExclusions(data['coverage']);helper.bind_capture('full_direct',data['capture'])
    row=data['records'][0]
    return helper,helper.for_frame('full_direct','s07',256,row['raw_measurement']['path'],row['raw_measurement']['sha256'],'b'*64,'c'*64,6)


def test_complete_index_and_original_bound_ids(exclusion_files):
    helper,ids=consume(exclusion_files)
    assert ids==frozenset({2})and helper.receipt()['verified_consumed_exposures']==1


def test_index_hash_and_duplicate_exposure_fail(exclusion_files):
    data=exclusion_files;data['index'].write_text(data['index'].read_text()+'\n')
    with pytest.raises(ValueError,match='index SHA changed'):ReadoutExclusions(data['coverage'])
    data['records'].append(data['records'][0]);data['save']()
    with pytest.raises(ValueError,match='Duplicate exclusion'):ReadoutExclusions(data['coverage'])


def test_full_exposure_denominator_and_capture_binding_required(exclusion_files,tmp_path):
    data=exclusion_files;data['report']['source_exposures_per_capture']=3;data['save']()
    with pytest.raises(ValueError,match='denominator'):ReadoutExclusions(data['coverage'])
    data['report']['source_exposures_per_capture']=2;data['save']()
    helper=ReadoutExclusions(data['coverage']);other=tmp_path/'other';put(other/'binding.json',{})
    with pytest.raises(ValueError,match='different capture'):helper.bind_capture('full_direct',other)


def test_RGB_and_changed_original_NPZ_fail(exclusion_files):
    data=exclusion_files;helper=ReadoutExclusions(data['coverage']);helper.bind_capture('full_direct',data['capture']);row=data['records'][0]
    with pytest.raises(ValueError,match='RGB binding differs'):
        helper.for_frame('full_direct','s07',256,row['raw_measurement']['path'],row['raw_measurement']['sha256'],'d'*64,'c'*64,6)
    Path(row['raw_measurement']['path']).write_bytes(b'changed original')
    with pytest.raises(ValueError,match='NPZ SHA changed'):consume(data,helper)


@pytest.mark.parametrize('ids',[[6],[2,1],[-1]])
def test_original_candidate_bounds_and_order_required(exclusion_files,ids):
    data=exclusion_files;row=data['records'][0];path=Path(row['arrays']['path'])
    np.savez(path,affected_candidate_ids=np.array(ids,np.int32),trigger_bits=np.full(len(ids),64,np.uint8))
    row['arrays']['sha256']=digest(path);row['affected_candidate_count']=len(ids);data['save']()
    with pytest.raises(ValueError,match='IDs/bits/bounds'):consume(data)


def test_cached_exclusion_NPZ_change_cannot_be_reused(exclusion_files):
    helper,_=consume(exclusion_files);path=Path(exclusion_files['records'][0]['arrays']['path'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='NPZ SHA changed'):consume(exclusion_files,helper)


def edge(pairs,a=0,b=0):
    return dict(a=a,b=b,pairs=pairs,source_only=True,identity_assumed_from_curve_name=False,
        pairs_index_space='previous/current_original_candidate_npz_candidate_id',unambiguous=True,
        correlation_median=.99,normal_median_px=.2,backward_normal_median_px=.3,support=999)


def test_candidate_gate_is_target_span_only_and_old_quality_is_not_new_measurement():
    previous=np.c_[np.arange(3),np.zeros(3)];current=np.c_[np.arange(3)*28.,np.zeros(3)]
    source=edge([[0,0],[1,1],[2,2],[2,2]])
    kept,insufficient,receipt=guarded_candidate_links(dict(links=[source]),previous,current,set(),set())
    assert not insufficient and len(kept)==1 and kept[0]['previous_support']==kept[0]['support']==3
    assert kept[0]['actual_previous_UV_span_px']==2 and kept[0]['unambiguous']
    assert 'correlation_median' not in kept[0]and kept[0]['original_quality_summary']['correlation_median']==.99
    assert kept[0]['old_quality_summary_recomputed_for_subset']is False
    assert source['pairs']==[[0,0],[1,1],[2,2],[2,2]]


def test_filter_rechecks_unique_support_and_current_span():
    uv=np.c_[np.arange(4)*28.,np.zeros(4)];record=dict(links=[edge([[i,i]for i in range(4)])])
    assert guarded_candidate_links(record,uv,uv,{0,1},set())[0]==[]
    current=uv.copy();current[:,0]=[0.,20.,40.,55.9]
    assert guarded_candidate_links(record,uv,current,set(),set())[0]==[]


def test_original_ambiguous_alternative_is_requalified_after_guard_removal():
    uv=np.c_[np.arange(9)*28.,np.zeros(9)]
    main=edge([[i,i]for i in range(5)]);rival=edge([[i,i+5]for i in range(4)],b=1)
    main['unambiguous']=False;rival['unambiguous']=False
    all_edges=guarded_candidate_links(dict(links=[main,rival]),uv,uv,set(),set())[0]
    assert not any(e['unambiguous']for e in all_edges)
    kept,insufficient,receipt=guarded_candidate_links(dict(links=[main,rival]),uv,uv,set(),{5,6,7,8})
    assert len(kept)==1 and kept[0]['unambiguous']and len(insufficient)==1
    assert {tuple(p)for p in kept[0]['pairs']}<={tuple(p)for p in main['pairs']}
    assert receipt['occluded_pair_occurrences_removed']==4


def line_record(pts=256):
    report=dict(curve=0,source_line_support_qualified=True,paired_source_boundaries_required=True,
        boundary_reports=[dict(source_line_support_qualified=True)]*2,selected_candidate_ids=list(range(6)))
    return dict(pts=pts,line_fragments=[dict(curve=0,candidate_ids=list(range(6)))],
        line_qualification=dict(whole_line_reports=[report]))


def test_verified_line_groups_split_at_each_excluded_actual_readout():
    inventory=[dict(operator='paint_center',closed=False,curve_id='C:segment0',native_path_id='path')]
    groups=list(verified_line_groups(line_record(),inventory,0,6,Counter(),{2,4}))
    assert [group['candidate_ids']for group in groups]==[[0,1],[3],[5]]
    assert all(group['original_qualification_not_recomputed']for group in groups)


def test_export_spacing_does_not_bridge_denied_branch_hole():
    pool={i:dict(uv=[i*24.,0.])for i in range(7)}
    selected=[spaced_actual_keys(run,pool)for run in retained_runs(range(7),{3})]
    spans=[[pool[run[0]]['uv'],pool[run[-1]]['uv']]for run in selected]
    assert spans==[[[0.,0.],[48.,0.]],[[96.,0.],[144.,0.]]]


def test_candidate_graph_never_labels_guarded_anchors_or_joins_across_line_hole(tmp_path):
    output=tmp_path/'output';output.mkdir();frames=[];uv=np.c_[np.arange(6)*28.,np.zeros(6)].astype(np.float32)
    inventory=[dict(operator='paint_center',closed=False,curve_id='C:segment0',native_path_id='path')]
    for pts in (256,512):
        raw=tmp_path/f'{pts}.npz';np.savez(raw,candidate_uv=uv)
        record=line_record(pts);record.update(source_BGR_sha256='b'*64,
            raw_candidates=dict(path=str(raw),sha256=digest(raw)),summary=dict(actual_RGB_candidates=6,by_path={}))
        source=tmp_path/f'{pts}.source.json';put(source,record)
        metadata=tmp_path/f'{pts}.link.json';put(metadata,dict(source_frame=dict(path=str(source),sha256=digest(source)),
            source_candidates=record['raw_candidates'],original_candidate_count=6,source_BGR_sha256='b'*64,source_RGB_sha256='c'*64))
        frames.append(dict(pts=pts,metadata_path=str(metadata),source_candidates=record['raw_candidates'],source_RGB_sha256='c'*64))
    pair=tmp_path/'pair.json';put(pair,dict(previous_source_RGB_sha256='c'*64,current_source_RGB_sha256='c'*64,links=[edge([[i,i]for i in range(6)])]))
    anchors=[]
    for ai,(entity,cid)in enumerate([('left',0),('right',5),('occluded',2)]):
        anchors.append((ai,dict(shot='s07',pts=256,entity=entity,candidate_ids=[cid],source_BGR_sha256='b'*64,measurement_sha256=frames[0]['source_candidates']['sha256'])))
    class Guard:
        def for_frame(self,*args):return frozenset({2})
        def receipt(self):return dict(synthetic=True)
    summary=propagate_shot('full_direct','s07',{},frames,[dict(previous_pts=256,current_pts=512,metadata_path=str(pair))],
        dict(native_inventory=inventory),anchors,['left','occluded','right'],output,Guard())
    report=read(output/'full_direct/s07.json.gz')
    with np.load(output/'full_direct/s07.npz')as z:labels=z['label_index'].reshape(2,6)
    np.testing.assert_array_equal(labels,[[0,0,-1,2,2,2],[0,0,-1,2,2,2]])
    assert summary['skipped_occluded_anchor_candidates']==1 and not summary['conflicting_candidates']
    assert report['skipped_occluded_anchor_readouts'][0]['candidate_ids']==[2]
    assert all(2 not in group['candidate_ids']for group in report['qualified_groups'])
