"""Select actual source rows for a native-frame point overlay, with the existing 500 cap."""
import numpy as np
from collections import defaultdict

def select_source_rows(rows,limit=500,previous_point_ids=()):
 if limit!=500:raise ValueError('Programme source policy requires a 500-point maximum')
 if len(rows)<=limit:return list(range(len(rows))),dict(candidates=len(rows),selected=len(rows),under_source_ceiling=True)
 xy=np.asarray([r['uv']for r in rows],dtype=np.float64)
 if not np.isfinite(xy).all():raise ValueError('Nonfinite source position')
 if ((xy<0).any(axis=1)|(xy>[1919,1079]).any(axis=1)).any():raise ValueError('Point outside actual source sensor')
 ids=[];seen=set();groups=defaultdict(list);stable=np.asarray([r.get('point_id')in previous_point_ids for r in rows],bool)
 def add(i):
  i=int(i)
  if i not in seen:seen.add(i);ids.append(i)
 for i,r in enumerate(rows):
  if r.get('source_kind')=='preserved_user_manual_native_image_click' or r.get('user_confirmed')is True:add(i)
  group=r.get('feature_id',r.get('role','unknown'));groups[group].append(i)
 for g,ii in groups.items():
  ii=np.asarray(ii,dtype=int);q=xy[ii]
  if str(g).startswith('board.'):
   spans=defaultdict(list)
   for i in ii:spans[rows[i].get('segment_id',g)].append(i)
   for pp in spans.values():
    pp=np.asarray(pp,int);add(pp[np.argmin(xy[pp,0])]);add(pp[np.argmax(xy[pp,0])])
  elif str(g).startswith('ice.') or str(g).startswith('unbound.ice.'):
   for axis in (0,1):add(ii[np.argmin(q[:,axis])]);add(ii[np.argmax(q[:,axis])])
  elif str(g).startswith('Ad'):
   center=np.median(q,axis=0);add(ii[np.argmin(np.sum((q-center)**2,axis=1))])
 if len(ids)>limit:raise ValueError('Required manual/entity/span seed coverage exceeds source point cap; must resolve aliases/visibility rather than silently drop')
 d=np.full(len(rows),np.inf)
 for i in ids:d=np.minimum(d,np.sum((xy-xy[i])**2,axis=1))
 if not ids:add(int(np.lexsort((xy[:,1],xy[:,0]))[0]));d=np.sum((xy-xy[ids[0]])**2,axis=1)
 d[list(seen)]=-1
 while len(ids)<limit:
  i=int(np.argmax(d));best=float(d[i])
  if best<=0:break
  # A prior surviving source ID wins a subpixel-equivalent coverage tie;
  # its UV is this frame's new pixel reading, never a transported position.
  if stable.any():
   tie=stable&(d>=max(0.,np.sqrt(best)-1.)**2)
   if tie.any():jj=np.flatnonzero(tie);i=int(jj[np.argmax(d[jj])])
  add(i);d=np.minimum(d,np.sum((xy-xy[i])**2,axis=1));d[list(seen)]=-1
 covered={rows[i].get('feature_id',rows[i].get('role'))for i in ids}
 return ids,dict(candidates=len(rows),selected=len(ids),under_source_ceiling=len(ids)<=limit,all_required_named_features_represented=True,represented_feature_ids=sorted(str(x)for x in covered),selection_moves_source_UV=False,selection_is_statistical_weight=False)
