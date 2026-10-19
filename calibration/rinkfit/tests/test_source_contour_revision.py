from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from rinkfit_start.source_contour_revision import apply_source_contour_revision,map_retained_stations

ROOT=Path(__file__).resolve().parents[3]
REVISION=ROOT/'events/ankara_2026/programmes/okada_mei_sp/src_rev1018/ad4_v1.json'

@pytest.fixture
def revision():return json.loads(REVISION.read_text(encoding='utf8'))

@pytest.fixture
def model(revision):
 return {'blocks':[{'name':'print/w158','values':[1,0,0,0,1,0]},{'name':'ad/4','values':[.1,.2,.01,.02]}],
  'materials':[{'label':'unrelated','paths':[{'kind':0,'id':'untouched'}]},
   {'label':revision['template_path_patch']['material_label'],'paths':[{'kind':2,'closed':False,'id':'w158/path0','variant_id':'w158','geometry_block':0,'advertisement_block':1,'points':deepcopy(revision['template_path_patch']['expected_old_points'])}]}]}


def test_apply_retains_material_identity_and_shared_blocks(model,revision):
 original=deepcopy(model);out=apply_source_contour_revision(model,revision)
 assert model==original and out['blocks']==model['blocks']
 assert out['materials'][0]==model['materials'][0]
 assert [m['label']for m in out['materials']]==[m['label']for m in model['materials']]
 paths=out['materials'][1]['paths'];old=model['materials'][1]['paths'][0]
 assert [p['points']for p in paths]==[old['points'][:13],old['points'][16:20]]
 assert [len(p['points'])for p in paths]==[13,4]
 assert all(p['closed'] is False and p['geometry_block']==0 and p['advertisement_block']==1 for p in paths)
 assert out['source_contour_revisions'][0]['fresh_RS_S_Q_required']


def test_transport_preserves_every_retained_original_segment(revision):
 # Independent original-edge interpolation checks every retained edge at
 # non-knot locations, so a shortened path cannot silently rescale stations.
 points=np.array(revision['template_path_patch']['expected_old_points'])
 oldedge=np.array(list(range(12))+list(range(16,19)))
 x=(oldedge[:,None]+np.array([.001,.173,.71,.999])).ravel()
 p,t,ok=map_retained_stations(np.zeros(len(x),int),x/19,revision)
 assert ok.all()
 expected=points[np.floor(x).astype(int)]*(1-(x%1)[:,None])+points[np.floor(x).astype(int)+1]*(x%1)[:,None]
 actual=[]
 for b,s in zip(p,t):
  ix=revision['template_path_patch']['retained_paths'][b]['old_vertex_indices'];q=points[ix];z=s*(len(q)-1);i=min(int(z),len(q)-2);u=z-i;actual.append(q[i]*(1-u)+q[i+1]*u)
 np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-16)


def test_transport_gap_is_unmapped_without_endpoint_clipping(revision):
 s=np.array([0,12/19,12.001/19,14/19,15.999/19,16/19,1.])
 p,t,ok=map_retained_stations(np.zeros(len(s),int),s,revision)
 assert p.tolist()==[0,0,-1,-1,-1,1,1]
 assert ok.tolist()==[True,True,False,False,False,True,True]
 np.testing.assert_array_equal(t[ok],[0,1,0,1]);assert np.isnan(t[~ok]).all()


def test_incompatible_model_signature_is_rejected(model,revision):
 model['materials'][1]['paths'][0]['points'][8][0]+=.01
 with pytest.raises(ValueError,match='signature'):apply_source_contour_revision(model,revision)


def test_reapplying_patch_is_rejected(model,revision):
 with pytest.raises(ValueError,match='single open path'):apply_source_contour_revision(apply_source_contour_revision(model,revision),revision)

@pytest.mark.parametrize('case',['bridge','closed','missing_source'])
def test_revision_cannot_invent_connectivity_or_drop_source(model,revision,case):
 patch=revision['template_path_patch']
 if case=='bridge':patch['retained_paths'][0]['old_vertex_indices']=[0,1,16,17]
 elif case=='closed':patch['retained_paths'][0]['closed']=True
 else:patch['retained_paths'][0].pop('source_evidence')
 with pytest.raises(ValueError):apply_source_contour_revision(model,revision)

@pytest.mark.parametrize('path,s',[(1,.5),(0,-.01),(0,1.01),(.5,.5),(0,float('nan'))])
def test_transport_rejects_illegal_input(revision,path,s):
 with pytest.raises(ValueError):map_retained_stations(path,s,revision)
