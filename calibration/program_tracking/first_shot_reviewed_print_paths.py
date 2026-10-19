"""Explicit source-reviewed membership in an existing physical print path.

Only pixel-to-material identity changes. The original pixels, full W, source
clock, geometry bank, free curve station and parameter chart remain intact.
"""
from collections import OrderedDict
from copy import deepcopy
from types import MethodType
import numpy as np
from .first_shot_source_transition import read,reference
SCHEMA='rinkspace.reviewed-existing-print-path-membership.v1'

def validate_manifest(mapping_reference):
    value=read(mapping_reference)
    if (value.get('schema')!=SCHEMA or value.get('authority')!='agent_rgb_review'
            or value.get('review_complete')is not True
            or value.get('model_argmin_is_identity_authority')is not False
            or value.get('source_UV_ID_sigma_W_clock_unchanged')is not True
            or value.get('user_confirmed')is True):
        raise ValueError('Explicit original-RGB print membership review required')
    if value.get('producer')!=reference(__file__):raise ValueError('Reviewed path producer changed')
    for ref in value['evidence']:
        if reference(ref['path'])!=ref:raise ValueError('Reviewed RGB evidence changed')
    lookup={}
    for item in value['records']:
        key=int(item['frame']),str(item['source_point_id'])
        if (key in lookup or not item.get('source_image_reviewed')
                or item.get('source_physical_identity')!='ISU_logo_upper_I_ink_boundary'
                or item.get('true_path_id')!='logo_part01' or item.get('advertisement_id')!=7):
            raise ValueError('Unique explicitly reviewed physical I boundary required')
        if reference(item['source_packet']['path'])!=item['source_packet']:
            raise ValueError('Reviewed original source packet changed')
        lookup[key]=item
    if not lookup:raise ValueError('An empty review cannot change print identities')
    return value,lookup

def install_successor(owner,*,mapping_reference):
    from .first_shot_source_snapshot import _frame_contract
    from .first_shot_disjoint_print import KIND,PathUnion,_clear_caches
    from .first_shot_fit import _target_key
    value,lookup=validate_manifest(mapping_reference);programme=owner.programme;bundle=programme.bundle
    if any(pool is not None for pool in (programme.pool,bundle._geometry_pool,bundle._source_pool)):
        raise ValueError('Reviewed memberships must precede all execution pools')
    state=programme.state
    if (read(state.binding_reference)['source']['sha256']!=value['source_video_sha256']
            or state.source_clock['time_base']!=value['time_base']):
        raise ValueError('Reviewed membership source identity or clock differs')
    q0=owner.start.copy();before=_frame_contract(bundle.frames)
    auxiliaries={k:v.copy()for k,v in owner.checkpoint_arrays().items()}
    output=[];used=set();changes=[]
    for old_frame in bundle.frames:
        frame=dict(old_frame);rows=[]
        for old_row in old_frame['rows']:
            keys=[(int(frame['index']),str(pid))for pid in old_row['source_point_ids']]
            selected=[key for key in keys if key in lookup]
            if not selected:rows.append(old_row);continue
            if len(keys)!=1 or len(selected)!=1 or len(old_row['uv'])!=1:
                raise ValueError('Reviewed print membership requires its original one-pixel row')
            key=selected[0];item=lookup[key];target=deepcopy(old_row['target'])
            if (frame['native_pts']!=item['native_pts'] or frame['source_packet']!=item['source_packet']
                    or not np.array_equal(np.asarray(old_row['uv'])[0],item['source_uv'])
                    or target.get('kind')!=KIND or target.get('advertisement_id')!=7
                    or target.get('contour_key')!='component000'
                    or 'source_reviewed_material_arc'in target):
                raise ValueError('Reviewed I membership differs from original source row')
            union=PathUnion.from_target(target);path=item['true_path_id']
            if path not in union.ids:raise ValueError('Review cannot invent a material path outside the existing bank')
            part=union.ids.index(path);count=union.offsets[part+1]-union.offsets[part]
            target['curve_topology']=dict(target['curve_topology'],path_ids=[path],offsets=[0,count],closed=[union.closed[part]])
            PathUnion.from_target(target)
            row=dict(old_row,target=target,_target_key=_target_key(target))
            row['source_true_path_history']=deepcopy(old_row.get('source_true_path_history',[]))+[
                dict(previous_target=deepcopy(old_row['target']),mapping=deepcopy(mapping_reference),
                     physical_identity=item['source_physical_identity'],true_path_id=path)]
            rows.append(row);used.add(key);changes.append(dict(frame=key[0],source_point_id=key[1],true_path_id=path))
        frame['rows']=rows;output.append(frame)
    if used!=set(lookup)or _frame_contract(output)!=before:
        raise ValueError('Reviewed membership must retain every original UV/ID/sigma/W/clock and match every listed point')
    bundle.frames=output;bundle.descriptors=OrderedDict()
    for frame in output:
        for row in frame['rows']:bundle.descriptors.setdefault(row['_target_key'],deepcopy(row['target']))
    bundle.keys=tuple(bundle.descriptors);bundle.target_descriptors=tuple(bundle.descriptors.values());_clear_caches(programme)
    receipt=dict(schema=SCHEMA,producer=reference(__file__),mapping=deepcopy(mapping_reference),
        changed_points=len(changes),records=changes,original_complete_source_contract=before,
        free_curve_station_preserved=True,new_material_path_or_corner_correspondence_added=False,
        source_UV_ID_sigma_W_clock_and_all_q_aux_unchanged=True,fresh_source_and_world_J_required=True)
    original_identity=owner.checkpoint_identity
    def identity(this):return dict(original_identity(),reviewed_print_path_membership=deepcopy(receipt))
    owner.checkpoint_identity=MethodType(identity,owner)
    for obj,name in ((owner,'source_producers'),(owner,'recovery_references'),(owner,'_references'),(programme,'source_producers')):
        refs=getattr(obj,name,[])
        for ref in (reference(__file__),mapping_reference):
            if ref not in refs:refs.append(deepcopy(ref))
    actual=owner.checkpoint_arrays()
    if set(actual)!=set(auxiliaries)or not np.array_equal(owner.start,q0):raise ValueError('Source identity review changed numerical state')
    for key,expected in auxiliaries.items():
        value=np.asarray(actual[key])
        if value.dtype!=expected.dtype or value.shape!=expected.shape or value.tobytes()!=expected.tobytes():
            raise ValueError('Source review changed auxiliary '+key)
    return receipt
