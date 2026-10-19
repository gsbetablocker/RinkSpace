"""Load the source blue-cap edge identified by the user, preserving raw UVs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
EVIDENCE = ROOT/'evidence/main_nearboard_source_candidates_20260928'
PACKET = EVIDENCE/'full_main_bottom_qualified_packet.npz'
CONFIRMATION = EVIDENCE/'user_confirmed_main_lower_board_top.json'
PACKET_SHA = '06deb7a8619337fbd71fc339b8b887c9eb36abaf844f33c38f7fe10356ecef8f'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_main_nearboard_source(data, aliases, *, packet=PACKET,
                                  confirmation=CONFIRMATION):
    """One actual source measurement per confirmed camera exposure.

    All native rows remain in the immutable packet. For repeated exposures use
    the earliest source row, not a duplicated observation or an averaged pixel.
    Native frames lacking a candidate remain unknown throughout the main group.
    """
    evidence=json.loads(Path(confirmation).read_text(encoding='utf8'))
    if (data['camera_id']!='main_long_side'
            or _sha(packet)!=PACKET_SHA
            or evidence['candidate_packet_sha256']!=PACKET_SHA
            or evidence['physical_identity']!='lower_long_side_board_top'
            or evidence['source_pts']!=230545920
            or evidence['user_answer_zh']!='是，下长边围挡顶面朝冰一侧的边缘'
            or evidence['source_video_sha256']!=data['diag']['source_sha256']):
        raise ValueError('Main near-board source identity or packet changed')
    with np.load(packet,allow_pickle=False) as z:
        packet_data={k:z[k].copy() for k in z.files}
    p=packet_data
    expected=np.arange(230542336,230556672+256,256,dtype=np.int64)
    if (not np.array_equal(p['pts'],expected)
            or str(p['source_video_sha256'])!=data['diag']['source_sha256']
            or p['offsets'].shape!=(58,) or p['offsets'][0]!=0
            or p['offsets'][-1]!=len(p['observed_uv'])
            or np.any(np.diff(p['offsets'])<2)
            or len(p['observed_uv'])!=2156):
        raise ValueError('Reviewed consecutive source support is inconsistent')
    for name in ('observed_uv','outward_normal','normal_interval_px'):
        if p[name].shape!=(2156,2) or not np.isfinite(p[name]).all():
            raise ValueError('Malformed source support: '+name)
    if any(int(t) not in aliases or int(t) not in data['pts'] for t in p['pts']):
        raise ValueError('Source row is outside the complete main camera group')
    groups={}
    for i,t in enumerate(p['pts']):
        groups.setdefault(int(aliases[int(t)]),[]).append(i)
    rows=[]
    for representative,indices in sorted(groups.items()):
        i=indices[0]
        a,b=map(int,p['offsets'][i:i+2])
        rows.append(dict(representative_pts=representative,
            reviewed_native_pts=[int(p['pts'][j]) for j in indices],
            measurement_native_pts=int(p['pts'][i]),
            source_sha256=str(p['source_video_sha256']),
            source_pixel_sha256=str(p['source_pixel_sha256'][i]),
            original_packet_row_indices=list(range(a,b)),
            source_review_status='reviewed_bluecap_ice_boundary',
            physical_identity=evidence['physical_identity'],
            observed_uv=p['observed_uv'][a:b].astype(np.float64),
            outward_normal=p['outward_normal'][a:b].astype(np.float64),
            normal_interval_px=p['normal_interval_px'][a:b].astype(np.float64)))
    metadata=dict(enabled=True,packet=str(Path(packet).resolve()),packet_sha256=PACKET_SHA,
        confirmation=str(Path(confirmation).resolve()),confirmation_sha256=_sha(confirmation),
        native_frames=57,canonical_exposures=len(rows),raw_source_uv_count=2156,
        used_source_uv_count=sum(len(r['observed_uv']) for r in rows),
        source_uv_changed=False,
        duplicate_policy='earliest measured native frame per confirmed exposure; all original rows preserved',
        other_main_native_frames_unknown=len(data['pts'])-57,
        geometry_status=evidence['geometry_status'],solver_started=False)
    return rows,metadata
