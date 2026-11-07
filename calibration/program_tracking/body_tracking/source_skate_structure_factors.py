"""Complete source blade structures matched to the existing bound skate geometry.

Visible boundaries are finite pixel regions, never known material points. A
frame budget is shared across features; camera-derived support XY is correlated
with these same pixels and receives a declared share of that budget.
"""
import numpy as np
from scipy.sparse import csr_matrix,vstack
from .visible_runner_contours import VisibleRunnerContourFactors


def structure_segments(binding):
    packet=binding.packet();template=packet['template'];vertices=np.asarray(template['vertices']);faces=np.asarray(template['faces']);parts=np.asarray(template['face_parts'])
    segments=[];families={}
    def add(name,paths):
        ids=[]
        for path in paths:
            for a,b in zip(path[:-1],path[1:]):
                if np.linalg.norm(b-a)<1e-12:continue
                ids.append(len(segments));segments.append(np.stack((a,b)))
        families[name]=np.asarray(ids,int)
    def creases(names):
        paths=[]
        for name in names:
            selected=faces[parts==name];edge_faces={}
            if not len(selected):continue
            normal=np.cross(vertices[selected[:,1]]-vertices[selected[:,0]],vertices[selected[:,2]]-vertices[selected[:,0]])
            normal/=np.maximum(np.linalg.norm(normal,axis=1),1e-20)[:,None]
            for i,tri in enumerate(selected):
                for a,b in zip(tri,np.roll(tri,-1)):edge_faces.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
            for edge,ff in edge_faces.items():
                if len(ff)!=2 or abs(float(normal[ff[0]]@normal[ff[1]]))<.99999:paths.append(vertices[list(edge)])
        return paths
    lower=packet['edges'];upper=lower+[0,0,packet['parameters'].blade_web_height_m]
    add('runner_lower_edge',lower);add('runner_upper_edge',upper)
    add('toe_pick',creases(['toe_pick_head']))
    tail=[]
    for lo,hi in zip(lower,upper):
        take=lo[:,0]<=.12*packet['parameters'].sole_length_m;tail.extend([lo[take],hi[take],np.stack((lo[0],hi[0]))])
    add('rear_extension',tail)
    all_parts=np.unique(parts).tolist();add('blade_assembly_outline',creases(all_parts)+list(lower)+list(upper))
    add('boot_sole',creases([v for v in all_parts if 'outsole'in v]))
    add('heel_block',creases([v for v in all_parts if 'heel_block'in v]))
    add('front_stanchion',creases(['front_post','front_nose_post']))
    add('rear_stanchion',creases(['rear_post']))
    q=np.asarray(segments).reshape(-1,3);return dict(points=q,starts=np.arange(0,len(q),2),ends=np.arange(1,len(q),2),feature_segments=families)


class FullSkateStructureFactors(VisibleRunnerContourFactors):
    def __init__(self,problem,document):
        if document.get('association_policy',{}).get('model_visibility')=='view_visible_exterior':
            from .source_visible_model import structure_geometry
            geometry=structure_geometry(problem.binding)
        else:geometry=structure_segments(problem.binding)
        for row in document['rows']:
            if not len(geometry['feature_segments'].get(row['structure_feature'],[])):raise ValueError('No bound model geometry for '+row['structure_feature'])
        super().__init__(problem,document,segment_geometry=geometry)
    def evaluate(self,x,jacobian=False):
        out=super().evaluate(x,jacobian)
        cost=out['parts'].pop('visible_runner_contour',0.);out['parts']['full_source_skate_structure']=cost
        rows=out['metrics'].get('visible_runner_contours',[])
        for row in rows:
            row.pop('latent_runner_side',None);row.pop('latent_runner_u',None)
        out['metrics']['source_structure_rows']=len(rows)
        out['metrics']['source_structure_exposures']=len(set(r['pts']for r in rows))
        out['metrics']['same_pixel_frame_budget_not_independent_3d']=True
        return out


class UnassignedSkateSetFactors:
    """Visible source shoes with no asserted anatomical identity.

Each exposure chooses one of the two one-to-one matches. Both shoes remain in
the same continuous bound state; no artificial foot identity or XY support
truth is created at an occlusion. The winning branch has a piecewise Jacobian.
"""
    def __init__(self,problem,groups,episode=None):
        self.problem=problem
        self.groups=[(g['pts'],[FullSkateStructureFactors(problem,b)for b in g['branches']])for g in groups]
        self.last=None;self.episode=episode;self.episode_factors=[];self.episode_choice=None
        if episode:
            import copy
            raw=[row for g in groups for row in g['branches'][0]['rows']]
            for candidate in episode['candidates']:
                rows=[]
                for row in raw:
                    track=candidate['association_by_observation'][row['source_track_id']];item=copy.deepcopy(row)
                    item['foot']=candidate['fixed_body_foot_by_track'][track];item['episode_track_id']=track;rows.append(item)
                self.episode_factors.append(FullSkateStructureFactors(problem,dict(rows=rows,association_policy=episode.get('association_policy',{}))))
    def episode_evaluate(self,x,jacobian):
        if self.episode_choice is None:
            values=[f.evaluate(x,False)for f in self.episode_factors];self.episode_costs=[v['image_cost']for v in values];self.episode_choice=int(np.argmin(self.episode_costs))
        factor=self.episode_factors[self.episode_choice];out=factor.evaluate(x,jacobian);cost=out['image_cost']
        from .weighted_blocks import signature
        self.association_signature=signature(str(self.episode_choice),factor.association_signature)
        associations=dict(episode_id=self.episode.get('episode_id'),candidate_id=self.episode['candidates'][self.episode_choice]['candidate_id'],candidate_costs=self.episode_costs,
            candidate=self.episode['candidates'][self.episode_choice],fixed_whole_episode=True,physical_identity_confirmed=False,association_from='source candidate set scored on common body state at explicit outer boundary')
        for row in out['metrics'].get('visible_runner_contours',[]):row['source_anatomical_foot']=None;row['foot_is_model_association']=True
        self.last=dict(residual=out['residual'],jac=out['jac'],parts={'unassigned_source_skate_sets':cost},image_cost=cost,
            metrics=dict(source_unassigned_episode=associations,visible_unassigned_skate_contours=out['metrics'].get('visible_runner_contours',[])))
        return self.last
    def evaluate(self,x,jacobian=False):
        if self.episode:return self.episode_evaluate(x,jacobian)
        values=[];associations=[]
        for pts,branches in self.groups:
            trials=[b.evaluate(x,False)for b in branches];costs=[v['image_cost']for v in trials];choice=int(np.argmin(costs))
            out=branches[choice].evaluate(x,True)if jacobian else trials[choice]
            associations.append(dict(pts=pts,branch=choice,branch_costs=costs,source_anatomical_identity='unassigned',
                selected_model_association=[dict(source_track_id=r.get('source_track_id'),model_foot=r['foot'])for r in branches[choice].rows if r['structure_feature']=='runner_lower_edge']))
            for row in out['metrics'].get('visible_runner_contours',[]):row['source_anatomical_foot']=None;row['foot_is_model_association']=True
            values.append(out)
        residual=np.concatenate([v['residual']for v in values])if values else np.empty(0)
        cost=float(residual@residual/2)
        from .weighted_blocks import signature
        self.association_signature=signature(str([(v['pts'],v['branch'])for v in associations]),str([getattr(b,'association_signature','')for _,bs in self.groups for b in bs]))
        self.last=dict(residual=residual,jac=(vstack([v['jac']for v in values],format='csr')if values else csr_matrix((0,self.problem.size)))if jacobian else None,
            parts={'unassigned_source_skate_sets':cost},image_cost=cost,metrics=dict(source_unassigned_associations=associations,
                visible_unassigned_skate_contours=[row for v in values for row in v['metrics'].get('visible_runner_contours',[])]))
        return self.last
