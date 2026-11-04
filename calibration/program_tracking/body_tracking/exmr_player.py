"""Register current network comparison in the existing native-source 3D player."""
from pathlib import Path
import argparse,json
import numpy as np
from .exmr_prior import load,write_json,digest

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')


def register(project,base,output,*,include_old=False):
    from rinkspace.viewer3d.data import ResultCatalog
    from rinkspace.viewer3d.prepare import prepare_variant
    project,base,output=map(Path,[project,base,output]);player=output/'player';player.mkdir(exist_ok=True)
    case=output/'common/gvhmr'
    source_input=case/'input.npz'
    if not source_input.exists():raise FileNotFoundError('Materialize the common GVHMR state first')
    (player/'input.npz').write_bytes(source_input.read_bytes())
    compact=player/'compact_smplx.npz'
    if not compact.exists():compact.write_bytes((base/'compact_smplx.npz').read_bytes())
    pts=load(source_input)['pts'];variants=[]
    requested=[('gvhmr_fixed','GVHMR 固定基线（共同初值）',output/'common/gvhmr/initial/fitted/result.npz',False),
        ('gvhmr_joint1','GVHMR 联合1步（未验收）',output/'common/gvhmr/fitted/result.npz',True),
        ('exmr_prefoot','ExMR 足细化前（共同初值，未优化）',output/'common/exmr_prefoot/initial/fitted/result.npz',False),
        ('exmr_final','ExMR 足细化后（结构unknown，未优化）',output/'common/exmr_final/initial/fitted/result.npz',False)]
    for label,title,path,fitted in requested:
        if not path.exists():raise FileNotFoundError(path)
        packet=load(path);np.testing.assert_array_equal(packet['pts'],pts)
        variants.append(dict(id=label,label=title,kind='fullbody',result=str(path.resolve()),shell_topology='saved',
            body_reference_label='own_camera_'+label+'_shared_S01_shape_and_skate',
            body_refinement_intervals_seconds=[[76.,95.]] if fitted else []))
    if include_old:
        for label,title,path in [('old_A','历史原 A',base/'action_integration_1103/spin_s05/A/fitted/result.npz'),
            ('old_root','历史 root',base/'action_integration_1103/spin_s05/root_repair/A_pose/fitted/result.npz'),
            ('old_support','历史支撑修正',base/'support_spin_1104/stage1_retained/fitted/result.npz')]:
            np.testing.assert_array_equal(load(path)['pts'],pts)
            variants.append(dict(id=label,label=title,kind='fullbody',result=str(path),shell_topology='saved',
                body_reference_label='preserved_historical_result',body_refinement_intervals_seconds=[]))
    original=json.loads((project/PROGRAMME/'viewer3d_spin_support_1104.json').read_text(encoding='utf8'))
    config=dict(original,shot='spin_models_1104',source_shot='s05',source_run=str(player),
        default_variant='exmr_prefoot',comparison_variants=['gvhmr_fixed','exmr_prefoot','exmr_final'],
        initial_source_pts=round(88.12*12800),variants=variants,
        review_status='Same full 76-95 second source clock. Network/body/skate candidates require source review; none accepted as truth.',
        root_repair_status='Own network articulation; common source-ray world initialization; finite soft ice; no automatic network-static contact labels.',
        body_window_scope='Complete continuous 76.00-95.00 second spin; no internal frame-window freezing.',
        camera_comparison=str(output/'camera_comparison.json'))
    target=project/PROGRAMME/'viewer3d_spin_models_1104.json'
    if len(str(target).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    write_json(target,config)
    catalog=ResultCatalog(project=project,data_root=base.parents[1],config=target)
    caches=[]
    for variant in variants:
        print(json.dumps(dict(event='player_cache_started',variant=variant['id'],frames=len(pts))),flush=True)
        cache=prepare_variant(catalog,catalog.variant(variant['id']))
        caches.append(dict(variant=variant['id'],cache=str(cache)))
        print(json.dumps(dict(event='player_cache_ready',variant=variant['id'])),flush=True)
    write_json(output/'player_registration.json',dict(config=str(target),source_pts=[int(pts[0]),int(pts[-1])],
        source_frames=len(pts),source_time_base=[1,12800],same_pts_all_variants=True,
        caches=caches,source_panel=True,foot_views=['support_top','support_side'],no_visible_GUI_opened=True))
    return target


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    p.add_argument('--base-root',required=True);p.add_argument('--output',required=True);p.add_argument('--include-old',action='store_true')
    a=p.parse_args(argv);register(a.project,a.base_root,a.output,include_old=a.include_old)

if __name__=='__main__':main()