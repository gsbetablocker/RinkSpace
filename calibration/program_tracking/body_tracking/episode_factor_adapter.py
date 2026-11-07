"""Prevent multi-flight metric overwrites / duplicate weighted block names.
Wrap an existing factor; residual/Jacobian are unchanged. No factor discovery.
"""
from dataclasses import replace

class EpisodeScopedFactor:
    def __init__(self,factor,episode_id):
        if not isinstance(episode_id,str) or not episode_id or '/' in episode_id:raise ValueError('Unique plain episode ID required')
        self.factor=factor;self.episode_id=episode_id
    def evaluate(self,x,jacobian=False):
        v=self.factor.evaluate(x,jacobian);prefix='episode/'+self.episode_id+'/'
        out=dict(v)
        out['parts']={prefix+k:value for k,value in v.get('parts',{}).items()}
        out['metrics']={prefix+'metrics':v.get('metrics',{})}
        out['weighted_blocks']=[replace(b,name=prefix+b.name)for b in v.get('weighted_blocks',[])]
        # Existing factors here have no externally consumed block_slices. If added,
        # apply the same prefix, preserving actual offsets.
        if 'block_slices'in v:out['block_slices']={prefix+k:value for k,value in v['block_slices'].items()}
        if 'factor_row_counts' in v:
            out['factor_row_counts']=[(prefix+name,count) for name,count in v['factor_row_counts']]
        self.last=out
        return out

def validate_unique_episode_ids(episodes):
    ids=[e['episode_id']for e in episodes]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate episode IDs')
    return ids
