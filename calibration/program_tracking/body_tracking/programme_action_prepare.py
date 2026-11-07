"""Prepare a revised action ledger on an existing immutable programme input."""
from pathlib import Path
import argparse
import json
import os

import numpy as np

from .rebuild_prepare import digest, load
from .fullbody_round import write_json
from .episode_factor_adapter import validate_unique_episode_ids
from .programme_action_factors import active_episode_rows, supported_phase_mask


def prepare(parent, ledger_path, output, stage):
    parent, ledger_path, output = (Path(p).resolve() for p in (parent, ledger_path, output))
    action = json.loads(ledger_path.read_text(encoding='utf8'))
    document = json.loads((parent/'source_evidence.json').read_text(encoding='utf8'))
    source = load(parent/'input.npz')
    pts = source['pts']
    if (action['source_id'] != document['source_id'] or
        action['time_base'] != source['time_base'].tolist() or
        action['scope_pts'] != pts[[0,-1]].tolist() or len(pts) != action['exposures']):
        raise ValueError('Action ledger belongs to a different programme source clock')
    if action['parent_evidence_sha256'] != digest(parent/'source_evidence.json'):
        raise ValueError('Action ledger binds another original source-evidence revision')
    covered = np.zeros(len(pts), int)
    for interval in action['coverage_intervals']:
        covered[active_episode_rows(pts,interval)] += 1
    if not np.all(covered==1):
        raise ValueError('Programme description intervals do not cover native exposures exactly once')
    episodes = document['source_episode_review']['episodes']
    for episode in episodes:
        episode.update(action.get('inherited_episode_details',{}).get(episode['episode_id'],{}))
    episodes.extend(action['episodes'])
    validate_unique_episode_ids(episodes)
    seen_support = np.zeros((len(pts),2),int)
    flight = np.zeros(len(pts),int)
    for episode in episodes:
        rows = active_episode_rows(pts,episode)
        if episode['kind']=='support_candidate':
            seen_support[rows,episode['foot']] += 1
        if episode['kind']=='flight_visible_candidate':
            flight[rows] += 1
    if (seen_support>1).any() or (flight>1).any():
        raise ValueError('Overlapping source episodes would duplicate physics samples')
    for foot in (0,1):
        supported,lifted=supported_phase_mask(document,pts,foot)
        if np.any(supported&lifted):
            raise ValueError('Same source exposure and foot marked as support and lifted')
    for row in action['skate_regions']:
        slot=np.searchsorted(pts,row['pts'])
        if slot==len(pts) or pts[slot]!=row['pts']:
            raise ValueError('Shoe observation has no source exposure')
    document['source_episode_review'].update(
        whole_programme_scope_pts=action['scope_pts'],
        whole_programme_coverage_intervals=action['coverage_intervals'],
        source_action_ledger=str(ledger_path),source_action_ledger_sha256=digest(ledger_path))
    document['full_source_skate_structures']['rows'].extend(action['skate_regions'])
    document['programme_action_physics']=dict(schema='rinkspace.programme-action-physics.v1',
        stage=stage,structure_enabled=stage=='physics',flight_dynamics_enabled=stage=='physics',
        propose_rotation_path=stage=='source_path',primary_controls_only=stage=='source_path',
        source_action_ledger_sha256=digest(ledger_path),same_immutable_input=True,
        ice_plane_policy='existing finite ice')
    output.mkdir(parents=True,exist_ok=False)
    for f in parent.iterdir():
        if f.is_file() and f.name!='source_evidence.json':
            os.link(f,output/f.name)
    write_json(output/'source_evidence.json',document)
    write_json(output/'action_input_identity.json',dict(parent=str(parent),
        parent_evidence_sha256=action['parent_evidence_sha256'], stage=stage,
        ledger_sha256=digest(ledger_path),evidence_sha256=digest(output/'source_evidence.json'),
        prepared_input_sha256=digest(output/'input.npz'),
        prepared_observation_sha256=digest(output/'observations.npz'),
        actual_native_exposures=len(pts), flight_exposures=int(np.count_nonzero(flight)),
        source_episode_count=len(episodes), source_skate_regions_added=len(action['skate_regions'])))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','ledger','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage',choices=['source_path','physics'],required=True)
    args=parser.parse_args()
    print(prepare(args.parent,args.ledger,args.output,args.stage),flush=True)


if __name__=='__main__':
    main()
