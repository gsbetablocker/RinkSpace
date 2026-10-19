"""Type an independently frozen source-only track packet for the existing graph."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();s=json.loads(a.input.read_text(encoding='utf8'));rows=[]
    for pair in s['pairs']:
        if pair.get('source_identity_not_selected_by_model_or_projection') is not True:raise ValueError('Source identity qualification missing')
        for role,split in [('FIT','fit'),('HELD','holdout')]:
            selected=np.array(pair['split'])==role
            if selected.sum()<2:continue
            refs={}
            for end in ('source','target'):
                raw=pair[end+'_rawref'];refs[end+'_frame']=dict(path=raw['path'],sha256=raw['png_sha256'],native_bgr_sha256=raw['native_bgr_sha256'])
            rows.append(dict(id=pair['id']+'_'+split,source_pts=pair['source_pts'],target_pts=pair['target_pts'],
                surface_id='boards_inner_surface',same_static_material_confirmed=True,split=split,
                uv0=np.asarray(pair['uv0'])[selected].tolist(),uv1=np.asarray(pair['uv1'])[selected].tolist(),
                sigma_px=6.,sigma_semantics='6 native px declared engineering floor; forward-backward error is not statistical sigma',
                frozen_source_pair_id=pair['id'],source_material_review=pair['source_material_review'],
                original_source_track_indices=np.asarray(pair['original_source_track_indices'])[selected].tolist(),**refs))
    out=dict(schema='rinkspace.native-static-board-tracks.v1',source_sha256=s['source_video_sha256'],time_base=s['time_base'],
        model_projection_used_to_generate_or_select_uv=False,
        frozen_source_packet=dict(path=str(a.input.resolve()),sha256=hashlib.sha256(a.input.read_bytes()).hexdigest()),
        source_UV_and_HELD_roles_preserved=True,original_source_observations_retained=True,rows=rows)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),rows=len(rows))))

if __name__=='__main__':main()
