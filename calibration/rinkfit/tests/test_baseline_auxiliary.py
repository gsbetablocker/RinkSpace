import hashlib,json
import numpy as np
import pytest
from rinkfit_joint.baseline_auxiliary import array_items,inherited_semantic_array,checked_array


def test_streaming_source_IDs_do_not_merge_same_vertex_across_exposures(tmp_path):
    p=tmp_path/'packet.json';values=['same_vertex','same_vertex','含汉字与\"引号']+['long_'+str(i) for i in range(10000)]
    p.write_text(json.dumps(dict(source_row_ids=values,queries=[dict(row_index=7,native_uv=[[1.,2.]])]),ensure_ascii=False),encoding='utf8')
    assert list(array_items(p,'source_row_ids'))==values
    assert list(array_items(p,'queries'))==[dict(row_index=7,native_uv=[[1.,2.]])]


def test_relative_array_is_resolved_only_through_same_SHA_parent(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir();array=parent/'resolved.u32';np.array([3,4],dtype='<u4').tofile(array)
    desc=dict(file='resolved.u32',dtype='<u4',shape=[2],sha256=hashlib.sha256(array.read_bytes()).hexdigest());pp=parent/'manifest.json';pp.write_text(json.dumps(dict(resolved_material_index=desc)),encoding='utf8')
    cp=child/'manifest.json';cp.write_text(json.dumps(dict(resolved_material_index=desc,source_audit_parent_manifest=dict(path=str(pp),sha256=hashlib.sha256(pp.read_bytes()).hexdigest()))),encoding='utf8')
    assert np.array_equal(inherited_semantic_array(cp,'resolved_material_index'),[3,4])
    np.array([4,3],dtype='<u4').tofile(array)
    with pytest.raises(ValueError,match='changed native array'):inherited_semantic_array(cp,'resolved_material_index')
