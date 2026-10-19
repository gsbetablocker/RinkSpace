"""Canonical provenance and compact inputs for local absolute diagnostics."""
import json
from collections import Counter
import joint_absolute_source_context_20261011 as ctx
a=ctx.a
np=ctx.np


def wide_groups(c):
    core=c.state.structured_programme_coordinate_chart.core_count
    private=np.asarray(c.state.structured_programme_coordinate_chart.high_indices,int)
    values=c.wide.evaluate(c.q[:core],c.q[private],False)[0];weights=4*c.wide.cost_weights
    groups=[];offset=0
    def append(count,identity,physical,role):
        nonlocal offset
        stop=offset+count
        groups.append(dict(family='strong_high_wide_source',source_kind='wide',actual_pts=56218880,
            id=identity,physical_group=physical,scalar_rows=count,wide_start=offset,wide_stop=stop,
            actual=values[offset:stop].copy(),raw=values[offset:stop].copy(),scales=c.wide.scales[offset:stop].copy(),
            quadratic=c.wide.quadratic[offset:stop].copy(),loss=ctx.SourceLossWeights(weights[offset:stop]),
            source_role=role,original_private_camera25_preserved=True))
        offset=stop
    for i,(observation,uv)in enumerate(zip(c.wide.observations,c.wide.uvs)):
        append(2*len(uv),str(observation.get('id','wide_original_curve_'+str(i))),
            json.dumps(observation['target'],sort_keys=True), 'original_typed_curve')
    assert offset==4288
    for i,row in enumerate(c.wide.ad_packet['rows']):
        append(2,'wide_original_corner_'+str(i),'Ad'+str(row['user_ad_number']),str(row['role']))
    assert offset==4340
    append(2,'wide_original_secondary_installation_entity','secondary_installation_pivot',
           'original_installation_entity_support_pixel_protocol')
    for row,uv in zip(c.wide.trace_packet['contours'],c.wide.trace_uv):
        append(2*len(uv),row['id'],'board_top','correlated_complete_user_top_stroke')
    assert offset==5808
    return groups


def column_mapping(c,numeric):
    columns=np.unique(np.concatenate([entry[0]for entry in numeric.values()])).astype(int)
    records={}
    def declare(indices,category,names,units,owner):
        for col,name,unit in zip(np.asarray(indices,int),names,units):
            row=dict(evaluation_index=int(col),category=category,name=str(name),unit=str(unit),owner=owner)
            if int(col)in records:
                old=records[int(col)]
                if old['category']!=category:raise ValueError('Conflicting real parameter ownership')
                continue
            records[int(col)]=row
    declare(c.world_indices,'geometry',c.provider.parameter_names,c.provider.parameter_units,'current_complete_world')
    from z190_shared_board_pattern_source import PERIMETER_PARAMETER_NAMES
    declare(np.arange(c.state.patterns.start,c.state.patterns.start+9),'geometry',
        PERIMETER_PARAMETER_NAMES,['working_m']*9,'shared_board_pattern_material')
    declare(c.circle.active_circle_indices,'geometry',[c.circle.parameter_names[i]for i in range(len(c.circle.active_circle_indices))],
        ['working_m']*len(c.circle.active_circle_indices),'actual_print_circles')
    declare(c.placement.placement_indices,'geometry',c.placement.placement_contract['parameter_names'],
        ['working_m']*len(c.placement.placement_indices),'independent_Ad25_Ad26_material_placements')
    for at,block in c.state.bypts.items():
        begin=block.start+4*block.bundle.index[int(at)]
        declare(np.arange(begin,begin+4),'camera',['Rodrigues_R'+str(i)for i in range(3)]+['native_logf'],
            ['rad']*3+['log_native_focal_px'],dict(kind='programme_exposure_RF4',actual_pts=int(at),block=block.name))
        for indices,names,units in ((block.xyz_slots,['pivot_'+v for v in 'XYZ'],['working_m']*3),
            (block.optics_chart.mount_indices,['mount_'+v for v in 'XYZ'],['working_m']*3),
            (block.optics_chart.indices,['principal_x','principal_y','k1_chart'],['native_px','native_px','k1_increment_div_0.001'])):
            declare(indices,'camera',names,units,dict(kind='shared_installation',installation_id=block.bundle.calibration.installation_id))
    chart=c.state.structured_programme_coordinate_chart
    declare(chart.high_indices,'camera',c.wide.private_camera_parameter_names,
        ['scene_units_div100']*3+['rad']*3+['log_native_focal_px']*2+['native_px_div1000']*3+
        ['literal_Brown_coefficient']*12+['sensor_tilt_rad']*2,dict(kind='independent_wide_camera25',source_pts=56218880))
    declare(chart.bias_indices,'camera',['statistical_roll_bias_'+str(i)for i in range(3)],
        ['rad']*3,'original_correlated_engineering_roll_prior')
    ext_begin=int(c.arrays['aux__external_start']);ext_stop=int(c.arrays['aux__external_stop'])
    ext_pts=c.arrays['aux__external_canonical_pts']
    assert ext_stop-ext_begin==4*len(ext_pts)+2
    for i,at in enumerate(ext_pts):
        declare(np.arange(ext_begin+4*i,ext_begin+4*i+4),'camera',['Rodrigues_R'+str(j)for j in range(3)]+['native_logf'],
            ['rad']*3+['log_native_focal_px'],dict(kind='original_external_exposure_RF4',actual_pts=int(at)))
    declare(np.arange(ext_stop-2,ext_stop),'nuisance',['external_inverse_anchor_logdepth_'+str(i)for i in range(2)],
        ['log_original_depth_chart']*2,'original_external_unknown_source_anchor_depth')
    missing=[int(col)for col in columns if int(col)not in records]
    if missing:raise ValueError('Unclassified actual affecting source coordinates: '+str(missing))
    layout=c.state.canonical_parameter_layout
    result=[]
    for col in columns:
        row=records[int(col)];row['canonical_index']=int(layout.evaluation_to_canonical[int(col)])
        row['fixed_by_original_run']=bool(int(col)in c.fixed_evaluation)
        result.append(row)
    free=np.asarray([row['evaluation_index']for row in result if not row['fixed_by_original_run']],int)
    partitions={kind:np.asarray([row['evaluation_index']for row in result if row['category']==kind and not row['fixed_by_original_run']],int)
        for kind in ('camera','geometry','nuisance')}
    return result,free,partitions


def assemble(groups,numeric,partitions,raw=False):
    sizes=np.asarray([group['scalar_rows']for group in groups],int);offsets=np.r_[0,np.cumsum(sizes)]
    values=np.concatenate([group['raw'if raw else'actual']for group in groups])
    metric=np.concatenate([ctx.source_irls_weights(group['actual'],group['quadratic'],group['loss'])for group in groups])
    matrices={kind:np.zeros((len(values),len(cols)))for kind,cols in partitions.items()}
    slots={kind:{int(col):i for i,col in enumerate(cols)}for kind,cols in partitions.items()}
    for index,group in enumerate(groups):
        key=group.get('numeric_key',index);columns,J,rawJ=numeric[key]
        chosen=rawJ if raw else J
        if 'numeric_row_slice'in group:chosen=chosen[slice(*group['numeric_row_slice'])]
        lo,hi=offsets[index:index+2]
        assert chosen.shape==(group['scalar_rows'],len(columns))
        for kind,lookup in slots.items():
            positions=[j for j,col in enumerate(columns)if int(col)in lookup]
            matrices[kind][lo:hi,[lookup[int(columns[j])]for j in positions]]=chosen[:,positions]
    return values,metric,matrices,np.repeat(np.arange(len(groups)),sizes),offsets


def scalar_roles(groups):
    """Engineering priors are never pixel observations or physical anchors."""
    roles=[]
    for group in groups:
        if group['source_kind']=='global_tail':
            for row in group['row_labels']:
                if row['role']in ('engineering_roll','engineering_bias'):role='engineering_prior'
                elif group['family']=='external_main_to_low':
                    role='finite_source_support_without_measured_centre'if row['raw_support_retained_no_measured_centre']else'unknown_material_pixel_correspondence'
                else:role='identified_material_pixel_with_support'
                roles.append(role)
        elif group.get('source_role')=='original_installation_entity_support_pixel_protocol':
            roles.extend(['finite_source_support_without_measured_centre']*group['scalar_rows'])
        else:
            row=group.get('row',{})
            supported=(row.get('_source_optical_support')is not None or 'normal_interval'in row or
                row.get('source_point_bounds_authorized')is True or 'point_support_bounds_xy_px'in row)
            role=('correlated_secondary_centre_pixel_reference'if group['source_kind']=='manual_centre'else
                'identified_material_pixel_with_support'if supported else'identified_material_typed_pixel_reference')
            roles.extend([role]*group['scalar_rows'])
    return np.asarray(roles)


def group_report(groups,offsets,actual_analysis,raw_analysis):
    result=[];weights=actual_analysis['arrays']['nominal_IRLS_row_weights']
    for i,group in enumerate(groups):
        lo,hi=offsets[i:i+2];actual=group['actual'];raw=group['raw'];scales=group['scales']
        item={key:group[key]for key in ('family','actual_pts','id','physical_group','source_kind','scalar_rows')}
        if 'source_role'in group:item['source_role']=group['source_role']
        item.update(group_index=i,scalar_start=int(lo),scalar_stop=int(hi),
            actual_original_source_cost=ctx.weighted_source_cost(actual,group['quadratic'],group['loss']),
            raw_normalized_l2_norm=float(np.linalg.norm(raw)),actual_normalized_l2_norm=float(np.linalg.norm(actual)),
            support_zero_actual_nonzero_raw_scalar_count=int(np.count_nonzero((actual==0.)&(raw!=0.))),
            raw_pixel_component_abs_p95=float(np.quantile(abs(raw*scales),.95))if group.get('report_pixel_components',True)else None,
            actual_pixel_component_abs_p95=float(np.quantile(abs(actual*scales),.95))if group.get('report_pixel_components',True)else None)
        for name,analysis in (('actual',actual_analysis),('rawtyped',raw_analysis)):
            entry=dict(weighted_original_energy=float(np.sum(analysis['arrays']['weighted_residual'][lo:hi]**2)))
            for stencil in ('first','second'):
                for component in ('camera_only_remaining','geometry_only_remaining','joint_remaining','geometry_after_camera_component'):
                    vector=analysis['arrays'][stencil+'_'+component][lo:hi]
                    entry[stencil+'_'+component+'_energy']=float(vector@vector)
                joint=analysis['arrays'][stencil+'_joint_remaining'][lo:hi]
                pixel=np.divide(joint,np.sqrt(weights[lo:hi]),out=np.zeros_like(joint),where=weights[lo:hi]>0)*scales
                entry[stencil+'_joint_local_pixel_component_abs_p95']=float(np.quantile(abs(pixel),.95))if group.get('report_pixel_components',True)else None
                entry[stencil+'_joint_local_pixel_vector_l2_max']=float(np.max(np.linalg.norm(pixel.reshape(-1,2),axis=1)))if len(pixel)%2==0 and group.get('report_pixel_components',True)else None
            entry['joint_two_stencil_difference_norm']=float(np.linalg.norm(
                analysis['arrays']['first_joint_remaining'][lo:hi]-analysis['arrays']['second_joint_remaining'][lo:hi]))
            item[name]=entry
        result.append(item)
    return result
