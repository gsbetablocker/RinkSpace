from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'events/ankara_2026/programmes/okada_mei_sp')]
from points_s02 import SourceHouseReader,NativeGrid
from calibration.program_tracking.repair_shot2_ice import filter_grey_circle_readouts
from rinkspace.session.gpu_geometry import initialize_curve_stations


def test_open_native_half_ellipse_does_not_wrap_or_make_a_chord():
    reader=object.__new__(SourceHouseReader)
    class Core:
        def sample_path(self,board,kind,params,vertices,stations):
            return [[v,2*v,0.]for v in stations]
    reader.grid=SimpleNamespace(core=Core());reader.scene={'board':[]}
    reader.native_paths={('goal','outer'):(6,[],{})}
    reader._project=lambda pt,xyz:(xyz[:,:2],np.ones(len(xyz)))
    q,_=reader.projection({'paths':[{'feature_id':'goal'}]},0,'outer',np.array([0.,np.pi,2*np.pi]))
    np.testing.assert_allclose(q,[[0,0],[.5,1],[1,2]])


def test_red_crossing_cannot_become_grey_circle_material():
    rgb=np.full((50,50,3),100,dtype=np.uint8);rgb[19:22,19:22,0]=160
    def row(x):return {'uv':[x,20.], 'source_evidence':{'profile_normal':[0.,1.],'actual_source_sides_uv':[[x,18.],[x,22.]]}}
    good,bad=filter_grey_circle_readouts([row(20.),row(35.)],rgb)
    assert len(good)==1 and good[0]['uv'][0]==35.
    assert bad[0]['reason']=='red_straight_band_crossing_grey_circle'


def test_cpu_station_initialization_keeps_original_pixels_and_fixed_corners():
    called=[]
    class Resident:
        def __init__(self,stations):self.stations=stations
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def evaluate(self,backend):
            called.append(backend);values=np.zeros((len(self.stations),11));values[:,3]=self.stations*100
            return {'values':values,'status':np.zeros(len(values),int)}
    native=SimpleNamespace(paths={'line':{'kind':0}},resident=lambda **kw:Resident(kw['stations']))
    uv=np.array([[25.,0.],[90.,0.]]);before=uv.copy()
    stations,report=initialize_curve_stations(native,camera={},controls=[],path_ids=['line','line'],stations=np.array([.5,1.]),
        pts=np.array([575744,575744]),observed=uv,kinds=np.array([1,0]),backend='cpu')
    assert called==['cpu'];np.testing.assert_allclose(stations,[.25,1.]);np.testing.assert_array_equal(uv,before)
    assert not report['unresolved']


def test_s02_review_rebinding_does_not_recursively_replace_shared_order(monkeypatch):
    import review_points_s04 as shared
    monkeypatch.setattr(shared,'CurrentNativeGrid',NativeGrid)
    path={'uv':np.array([[0.,0.],[10.,0.],[20.,0.]]),'valid':np.ones(3,bool)}
    np.testing.assert_allclose(NativeGrid.order(np.array([[5.,2.],[17.,-1.]]),path),[.5,1.7])
