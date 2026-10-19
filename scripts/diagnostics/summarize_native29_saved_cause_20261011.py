"""Summarize sealed source-column coupling and already verified global shrink."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.sparse import csr_matrix
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_review86_samea454_actual947_minimal_pivot_trial_resume_memory_sha')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
    auditpath=Path(str(RUN)+'.bootstrap')/'native29_saved_direction_causal_audit.json';audit=read(auditpath)
    inventory=RUN/'strict_complete_source_replay_inventory.json';seal=read(inventory)
    block=RUN/'initial_normal_source_blocks/030.npz'
    expected=next(x['sha256']for x in seal['files']if x['relative_path']=='initial_normal_source_blocks/030.npz')
    assert sha(block)==expected
    with np.load(block,allow_pickle=False)as z:
        J=csr_matrix((z['J_data'],z['J_indices'],z['J_indptr']),shape=tuple(z['J_shape']))
        ledger=json.loads(str(z['ledger_json']));parent=str(z['q_sha256']);token=str(z['frozen_profile_token'])
    config=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_samea454_actual947_roles_SOC_sufficient_minimal_pivot_trial_20261011.json')
    authoritypath=config['source_options']['--direct-human-truth-authority'];authority=read(authoritypath)
    rowlengths={r['id']:2*len(r['uv'])for r in authority['records']}
    ids=ledger['fit_observation_ids'];offsets=np.r_[0,np.cumsum([rowlengths[x]for x in ids])]
    assert offsets[-1]==J.shape[0]==448
    name='user_native230024192_ad20_right_vertical_20261008';i=ids.index(name);rows=[int(offsets[i]),int(offsets[i+1])]
    assert rows==[274,282]
    selected=J[rows[0]:rows[1]];cols=np.unique(selected.indices)
    with np.load(audit['directions'][0]['saved_direction']['path'],allow_pickle=False)as z:direction=z['step'].copy();assert str(z['complete_q_sha256'])==parent
    groups=dict(low_shared_pivot_mount_optics=[11,12,13,34201,34202,34203,34210,34211,34212],
        one_actual_exposure_RF4=[15742,15743,15744,15745],
        shared_course_and_Ad7_20_dimensions=[34243,34245,34246,34247,34248,34250,34251],
        Ad20_independent_station_and_zmin=[34266,34267])
    assert sorted(x for g in groups.values()for x in g)==cols.tolist()
    contributions={name:(selected[:,columns]@direction[columns]).tolist()for name,columns in groups.items()}
    width=selected[:,34243].toarray().ravel();station=selected[:,34266].toarray().ravel()
    result=dict(schema='rinkspace.native29-22-columns-to-8503-global-line-search-causal-evidence.v1',
        script=ref(__file__),actual_guard_curvature_and_global_q_reconstruction=ref(auditpath),
        sealed_source_inventory=ref(inventory),authenticated_original_source_block=ref(block),
        original_source_row_ledger=ledger,parent_q_sha256=parent,frozen_profile_token=token,
        original_source_FIT_row_slice_start_stop=rows,source_authority=ref(authoritypath),
        source_row_id=name,actual_pts=230024192,canonical_22_columns=cols.tolist(),column_groups=groups,
        actual_first_source_J_direction_contributions=contributions,
        source_J_direction_total=(selected@direction).tolist(),
        FIT_units='Unchanged source sigma and correlated-trace mean normalized objective chart, not raw pixels',
        Ad20_width_station_dependence=dict(shared_width_column=34243,independent_station_column=34266,
            rowwise_width_over_station=(width/station).tolist(),
            maximum_abs_width_plus_half_station=float(abs(width+.5*station).max()),
            right_border_alone_cannot_identify_width_and_station_separately=True),
        local_to_global=[dict(iteration=d['iteration'],parent=d['parent_q_sha256'],child=d['child_q_sha256'],
            accepted_alpha=d['accepted_alpha'],pivot_retracted_columns=d['pivot_retracted_columns'],
            RF4_exposures_all_nonzero=d['changed_RF4_exposures'],
            RF4_exposures_all_match_parent_plus_alpha_direction=d['all_RF4_exposures_shrunk_by_same_alpha'])for d in audit['directions']],
        diagnosis=[
            'native29仅直接依赖本时刻RF4、LOW共用P/mount/optics及共同场地/广告尺寸和Ad20独立位置；其余8502时刻RF4不直接进入这条约束。',
            '六次接受态与q+alpha完整方向相比仅Pxy列11/12做了实际最近边界回退；全部8503组RF4完全按同一alpha缩小。局部约束通过共同line search抑制全片方向。',
            '名义slab交点normal carrier是源界的精确常量，zero-J删除在该局部分支是正确的；错配来自使用新Linf均衡foot的实际carrier去校正旧未均衡tangent Jacobian。',
            '近平行竖边的两个轴斜率使均衡到旧tangent的缩放权重约0.00293；旧SOC低估了同一物理曲率所需的tangent坐标修正。',
            '已有真实小步可行、完整KKT证书及该数值路径缺陷，不构成广告/冰线/全镜头目标相互不可行的证明。',
            '修复只统一已认证整条竖边物理station消元的J/响应；不改源证据、零cap、目标权重或真实guard，真实全源新方向和九镜头仍需验证。'],
        new_fit_or_complete_source_J=False,source_images_written=0,accuracy_accepted=False)
    output=Path(str(RUN)+'.bootstrap')/'native29_22_columns_global_shrink_causal_summary.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(output),sha256=sha(output),width_station_max_error=result['Ad20_width_station_dependence']['maximum_abs_width_plus_half_station'])))
if __name__=='__main__':main()
