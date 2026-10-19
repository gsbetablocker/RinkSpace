"""Direct two-frame line/conic registration and explicit component atlases.

Original source pixels are used in both frames.  No repeated point IDs become
correspondences, no edge is synthesized as Hj inv(Hi), and no derived factors
enter the physical likelihood.  The GS undistorted conic/line approximation is
an initializer; finite endpoint coverage belongs to the full physical model.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.optimize import least_squares
from .data import Observations,SeedCatalog
from .runtime import atomic_json,Journal
from .graph import PairwiseEdge,RegistrationGraph,GraphAnchor
from .pipeline import read_run

_CENTRE=np.array([959.5,539.5]);_SCALE=1000.
_CONTEXT=None


def primitive(xy,kind):
    q=(np.asarray(xy)-_CENTRE)/_SCALE;x,y=q.T
    if kind=="segment":
        design=np.c_[x,y,np.ones(len(x))]
        if len(q)<2:return None
        _,s,V=np.linalg.svd(design,full_matrices=True);c=V[-1]
        if np.linalg.norm(c[:2])<1e-12:return None
        return c/np.linalg.norm(c[:2])
    if kind=="circle":
        if len(q)<6:return None
        design=np.c_[x*x,x*y,y*y,x,y,np.ones(len(x))]
        _,s,V=np.linalg.svd(design,full_matrices=True);c=V[-1]
        # Only a nondegenerate observed ellipse supplies a conic hypothesis.
        C=np.array([[c[0],c[1]/2,c[3]/2],[c[1]/2,c[2],c[4]/2],[c[3]/2,c[4]/2,c[5]]])
        if c[1]*c[1]-4*c[0]*c[2]>=0 or abs(np.linalg.det(C))<1e-12:return None
        return c/np.linalg.norm(c)
    return None


def distance(xy,c):
    x,y=xy.T
    if len(c)==3:return (c[0]*x+c[1]*y+c[2])*_SCALE
    v=c[0]*x*x+c[1]*x*y+c[2]*y*y+c[3]*x+c[4]*y+c[5]
    dx=2*c[0]*x+c[1]*y+c[3];dy=c[1]*x+2*c[2]*y+c[4]
    den=np.hypot(dx,dy)
    if np.any(den<1e-12):raise ValueError("conic first-order distance singularity")
    return v/den*_SCALE


def transform(H,q):
    h=np.c_[q,np.ones(len(q))]@H.T
    if np.any(abs(h[:,2])<1e-10):raise ValueError("relative projective denominator")
    return h[:,:2]/h[:,2,None]


def _initialize(data_dir):
    global _CONTEXT
    obs=Observations.load(data_dir);seeds=SeedCatalog(data_dir);frames={};kinds={};plane={}
    for label,desc in seeds.materials.items():
        variants=desc['variants']
        if len(variants)!=1:continue
        v=seeds.variants[variants[0]];kind=v['geometry']['kind'];pl=v['plane_candidate']
        if pl is not None and kind in ('circle','segment'):kinds[label]=kind;plane[label]=pl
    for frame in range(1,1249):
        rows=obs.indices_for_frame(frame);groups={}
        for label in set(obs.source_target_labels[rows].tolist()):
            if label not in kinds:continue
            rr=rows[obs.source_target_labels[rows]==label];c=primitive(obs.source_uv[rr],kinds[label])
            if c is not None:groups[label]={'q':(obs.source_uv[rr]-_CENTRE)/_SCALE,'rows':rr,'primitive':c,'plane':plane[label]}
        frames[frame]=groups
    _CONTEXT=(obs,seeds,frames)


def _job(job):
    i,j,plane=job;obs,seeds,frames=_CONTEXT;A=frames[i];B=frames[j]
    labels=sorted(k for k in A.keys()&B.keys() if A[k]['plane']==plane)
    pairs=[(A[k],B[k]) for k in labels]
    record={'source_frame':i,'target_frame':j,'plane':plane,'materials':labels,'status':'insufficient_actual_curve_information'}
    # Every line contributes at most two independent conic/line directions;
    # every unknown-station circle contributes at most five.
    circle_families={}
    for label,(a,b) in zip(labels,pairs):
        if len(a['primitive'])==6:circle_families.setdefault(label.split(':')[0],0);circle_families[label.split(':')[0]]+=1
    # Nested rings share one physical centre; independently fitted noisy conics
    # must not invent the unobservable in-plane rotation of this pencil.
    maxrank=2*sum(len(a['primitive'])==3 for a,b in pairs)+sum(7 if count>=2 else 5 for count in circle_families.values())
    record['structural_rank_upper_bound']=min(8,maxrank)
    if maxrank<8 or not pairs:return record,None
    source=np.concatenate([a['q'] for a,b in pairs]);target=np.concatenate([b['q'] for a,b in pairs])
    if min(np.linalg.svd(source-source.mean(0),compute_uv=False)[-1],np.linalg.svd(target-target.mean(0),compute_uv=False)[-1])<1e-5:return record,None
    def residual(x):
        H=np.r_[x,1.].reshape(3,3)
        try:
            inv=np.linalg.inv(H)
            return np.concatenate([r for a,b in pairs for r in (distance(transform(H,a['q']),b['primitive']),distance(transform(inv,b['q']),a['primitive']))])
        except (ValueError,np.linalg.LinAlgError):return np.full(len(source)+len(target),1e8)
    # The only final fitting objective is the two original images. The second
    # seed can be historical without turning the estimated edge into Hj/Hi.
    T=np.array([[.001,0,-.9595],[0,.001,-.5395],[0,0,1.]])
    initial=[np.eye(3)]
    try:
        candidate=T@seeds.historical_H(plane,j)@np.linalg.inv(seeds.historical_H(plane,i))@np.linalg.inv(T)
        if abs(candidate[2,2])>1e-12:initial.append(candidate/candidate[2,2])
    except (ValueError,KeyError,np.linalg.LinAlgError):pass
    fits=[least_squares(residual,h.reshape(-1)[:8],max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-7) for h in initial]
    fit=min(fits,key=lambda f:np.dot(f.fun,f.fun));J=fit.jac
    col=np.linalg.norm(J,axis=0);col[col==0]=1;singular=np.linalg.svd(J/col,compute_uv=False)
    rank=int(np.count_nonzero(singular>singular[0]*1e-5));record.update(rank=rank,singular_values=singular.tolist(),rms_px=float(np.sqrt(np.mean(fit.fun**2))),max_normal_proxy_px=float(np.max(abs(fit.fun))),optimizer_status=int(fit.status),evaluations=int(fit.nfev))
    if rank<8:record['status']='weak_curve_directions_retained';return record,None
    H=np.r_[fit.x,1.].reshape(3,3)
    denom=np.c_[source,np.ones(len(source))]@H[2]
    orientation=np.linalg.det(H)/(denom**3)
    record['normalized_H_condition']=float(np.linalg.cond(H))
    if record['normalized_H_condition']>1e8 or np.any(orientation<=0) or not (np.all(denom>0) or np.all(denom<0)):
        record['status']='relative_chart_orientation_or_condition_limited';return record,None
    G=np.linalg.inv(T)@H@T;G/=np.linalg.norm(G)
    ri=np.concatenate([a['rows'] for a,b in pairs]);rj=np.concatenate([b['rows'] for a,b in pairs])
    edge=PairwiseEdge(i,j,G,plane,'direct fit of original two-frame same-material line/conic supports; no persistent station; GS/no-distortion initializer',obs.source_uv[ri],original_evidence=True,evidence_kind='independent_same_material_curve_registration',data_rank=rank,source_record_keys=tuple(obs.record_keys(ri)),target_record_keys=tuple(obs.record_keys(rj)))
    record.update(status='direct_original_curve_edge',edge_id=edge.edge_id,source_rows=len(ri),target_rows=len(rj))
    return record,edge.to_dict()


def build_curve_graph(run_dir,workers=8,output_name="curve_graph") :
    root,run=read_run(run_dir);out=root/output_name;out.mkdir(exist_ok=True)
    _initialize(run['data_dir']);obs,seeds,frames=_CONTEXT
    jobs=set();signature={}
    for i in range(1,1249):
        byplane={}
        for label,entry in frames[i].items():byplane.setdefault(entry['plane'],[]).append(label)
        for pl,labels in byplane.items():
            sig=(pl,tuple(sorted(labels)));signature.setdefault(sig,[]).append(i)
            if i<1248:jobs.add((i,i+1,pl))
    for (pl,labels),ids in signature.items():
        # Temporal support and material count, not a first-frame-only anchor.
        if len(ids)<2:continue
        anchors=[int(chunk[len(chunk)//2]) for chunk in np.array_split(np.array(ids),min(3,len(ids)))]
        for anchor in anchors:
            jobs.update((min(anchor,i),max(anchor,i),pl) for i in ids if i!=anchor)
    ordered=sorted(jobs);atomic_json(out/'schedule.json',{'all_frames':1248,'jobs':ordered,'source_observations':182017,'final_likelihood_factors':False})
    records=[];edges_by_plane={pl:[] for pl in seeds.planes};journal=Journal(out)
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize,initargs=(run['data_dir'],)) as pool:
        fs={pool.submit(_job,j):j for j in ordered}
        for future in as_completed(fs):
            record,e=future.result();records.append(record)
            if e:edges_by_plane[e['plane_id']].append(e)
            if len(records)%50==0:journal.emit('curve_edges_progress',completed=len(records),jobs=len(ordered),edges=sum(map(len,edges_by_plane.values())))
    atomic_json(out/'edge_ledger.json',records);summary={}
    for plane,items in edges_by_plane.items():
        original=json.loads((root/'graph'/f'{plane}.json').read_text(encoding='utf-8'))
        allitems={e['edge_id']:e for e in original['edges']}
        allitems.update({e['edge_id']:e for e in items});edges=[]
        for e in allitems.values():
            edges.append(PairwiseEdge(e['source_frame'],e['target_frame'],np.array(e['G_i_to_j']),plane,e['provenance'],e['source_witness_uv'],original_evidence=e['original_evidence'],evidence_kind=e['evidence_kind'],data_rank=e['data_rank'],weight=e['weight'],source_record_keys=e['source_record_keys'],target_record_keys=e['target_record_keys'],target_entity_witness_uv=e.get('target_entity_witness_uv'),derived_from_H=e['derived_from_H'],edge_id=e['edge_id']))
        graph=RegistrationGraph(range(1,1249),plane,edges);anchors=[]
        for a in original['anchors']:anchors.append(GraphAnchor(a['frame_id'],np.array(a['H_atlas_to_native']),a['atlas_id'],a['provenance']))
        init=graph.initialize(anchors,gauge_mode='component_projective');atomic_json(out/f'{plane}.json',init.to_dict())
        charts=[]
        for c in init.components:
            # Components lacking a conditional metric anchor retain unknown Q;
            # identity is never substituted for an unknown Euclidean upgrade.
            anchor=next((a for a in anchors if a.frame_id==c.gauge_frame),None)
            charts.append({'component_id':c.component_id,'reference_frame':c.gauge_frame,'Q_metric_to_component_atlas':None if anchor is None else anchor.H_atlas_to_native.tolist(),'metric_status':'unknown' if anchor is None else 'conditional_legacy_world_seed','physical_plane_M':seeds.embedding(plane).tolist()})
        atomic_json(out/f'{plane}_charts.json',charts)
        summary[plane]={'frames':1248,'curve_edges':len(items),'total_original_edges':len(edges),'components':len(init.components),'unknown_metric_components':sum(c['metric_status']=='unknown' for c in charts),'final_physical_verification':False}
    atomic_json(out/'summary.json',summary);journal.emit('curve_graph_finished',summary=summary)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();print(json.dumps(build_curve_graph(args.run_dir,args.workers),indent=2))



def refine_curve_graphs(run_dir,output_name="curve_graph"):
    root,run=read_run(run_dir);out=root/output_name;summary={}
    for path in sorted(out.glob('*.json')):
        if path.stem in ('schedule','edge_ledger','summary','status','refinement_summary') or path.stem.endswith(('_charts','_refined')):continue
        original=json.loads(path.read_text(encoding='utf-8'))
        if 'edges' not in original:continue
        if not any(e['evidence_kind']=='independent_same_material_curve_registration' for e in original['edges']):continue
        edges=[]
        for e in original['edges']:
            edges.append(PairwiseEdge(e['source_frame'],e['target_frame'],np.array(e['G_i_to_j']),original['plane_id'],e['provenance'],e['source_witness_uv'],original_evidence=e['original_evidence'],evidence_kind=e['evidence_kind'],data_rank=e['data_rank'],weight=e['weight'],source_record_keys=e['source_record_keys'],target_record_keys=e['target_record_keys'],target_entity_witness_uv=e.get('target_entity_witness_uv'),derived_from_H=e['derived_from_H'],edge_id=e['edge_id']))
        anchors=[GraphAnchor(a['frame_id'],np.array(a['H_atlas_to_native']),a['atlas_id'],a['provenance']) for a in original['anchors']]
        graph=RegistrationGraph(range(1,1249),original['plane_id'],edges);init=graph.initialize(anchors,gauge_mode='component_projective')
        refined=graph.refine(init,max_iterations=run['config']['graph_refinement_iterations'])
        atomic_json(out/f'{path.stem}_refined.json',refined.to_dict())
        summary[path.stem]={'status':refined.status,'variables':refined.variable_count,'initial_edge_witness_rms_px':refined.initial_native_witness_rms_px,'final_edge_witness_rms_px':refined.final_native_witness_rms_px,'source_observations':182017,'final_physical_verification':False}
    atomic_json(out/'refinement_summary.json',summary);return summary
