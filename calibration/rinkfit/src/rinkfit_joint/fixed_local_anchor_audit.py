"""Stratify the unchanged source-point cohort by known local coordinates.

A static entity with a fitted surface coordinate is not an absolute anchor.
Invalid rows stay in each denominator with extended-real infinite error.
"""
from pathlib import Path
from collections import defaultdict
import argparse,json,hashlib
import numpy as np
from .io_utils import read_json
from .dense_joint_packet import _array
from .observed_primal_restart import reference,write_new,checked
from .recovery_metrics import _same_denominator_statistics


def audit(report_file,output):
    report=read_json(report_file);model=read_json(checked(report['model']));packet_file=checked(report['packet']);packet=read_json(packet_file)
    cache=checked(report['distances']);data=np.load(cache);rows=data['packet_rows']
    if len(rows)!=report['rows']:raise ValueError('Cohort denominator differs')
    digest=hashlib.sha256()
    for i in rows:digest.update(str(int(i)).encode()+b'\t'+packet['source_row_ids'][int(i)].encode('utf8')+b'\n')
    if digest.hexdigest()!=report['cohort_row_ID_sha256']:raise ValueError('Cohort row identities differ')
    materials=_array(packet['arrays']['material_index'],packet_file.parent)[rows]
    shots=_array(packet['arrays']['shot_index'],packet_file.parent)[rows]
    E=_array(packet['arrays']['E'],packet_file.parent).reshape(-1)[rows]
    groups=defaultdict(list)
    for local,(mi,shot)in enumerate(zip(materials,shots)):
        material=model['materials'][int(mi)];paths=material['paths']
        if len(paths)!=1 or not paths[0].get('physical_point'):raise ValueError('Point path is ambiguous')
        path=paths[0];known=path['kind']==2 and path.get('geometry_block',-1)<0
        kind='known_local_ad_points'if known else'free_or_initializer_surface_points'
        groups[kind].append(local);groups[kind+'/'+packet['shot_ids'][int(shot)]].append(local)
    distance=data['true_RS_distance'];valid=(data['status']&7)==7
    summary={k:_same_denominator_statistics(distance,np.asarray(v),valid,E,k)for k,v in groups.items()}
    comparison={}
    if 'baseline' in report:
        baseline=read_json(checked(report['baseline']));prior=np.load(checked(baseline['distances']))
        if baseline['cohort_row_ID_sha256']!=report['cohort_row_ID_sha256'] or not np.array_equal(prior['packet_rows'],rows):raise ValueError('Baseline cohort differs')
        bd=prior['true_RS_distance'];bv=(prior['status']&7)==7
        for kind,indices in groups.items():
            ix=np.asarray(indices);both=valid[ix]&bv[ix]
            comparison[kind]=dict(N=len(ix),baseline=_same_denominator_statistics(bd,ix,bv,E,kind),candidate=summary[kind],newly_invalid=int(np.sum(bv[ix]&~valid[ix])),newly_valid=int(np.sum(~bv[ix]&valid[ix])),sum_squared_error_delta_on_common_valid=float(np.sum(distance[ix][both]**2-bd[ix][both]**2)),worsened_beyond_original_E=int(np.sum((distance[ix][both]-bd[ix][both])>E[ix][both])))
        prior.close()
    data.close()
    result=dict(schema='rinkfit.actual_fixed_local_anchor_strata.v1',implementation=reference(__file__),guard_report=reference(report_file),cache=reference(cache),cohort_row_ID_sha256=report['cohort_row_ID_sha256'],
        all_original_cohort_rows_retained=True,world_coordinates_calibrated=False,heldout_independence_claimed=False,summary=summary,same_source_comparison=comparison)
    write_new(output,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(audit(a.report,a.out)))
if __name__=='__main__':main()
