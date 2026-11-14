import ast
from pathlib import Path

import numpy as np
from rinkspace.session.print_shape_queries import KeyShapePrintPattern, key_shape_stations
from rinkspace.session.print_patterns import PrintPattern


def test_straight_shape_queries_do_not_grow_with_reference_vertex_density():
    sparse=np.array([[0.,0.],[1000.,0.]])
    dense=np.c_[np.linspace(0,1000,2001),np.zeros(2001)]
    a,_=key_shape_stations(sparse)
    b,_=key_shape_stations(dense)
    np.testing.assert_array_equal(a,[1.,999.])
    np.testing.assert_array_equal(a,b)


def test_rectangle_keeps_both_incident_sides_at_four_corners():
    p=np.array([[0.,0.],[200.,0.],[200.,100.],[0.,100.],[0.,0.]])
    t,plan=key_shape_stations(p)
    assert len(t)==8
    assert len([k for k in plan['keys']if k['kind']=='corner_sides'])==4
    assert np.all(np.diff(t)>0)
    assert t.min()>=1 and t.max()<=599


def test_smooth_loop_covers_all_compass_extrema_without_arc_filling():
    for n in (721,1441):
        angle=np.linspace(0,2*np.pi,n)
        p=100*np.c_[np.cos(angle),np.sin(angle)]
        t,plan=key_shape_stations(p)
        length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        q=np.c_[np.interp(t,length,p[:,0]),np.interp(t,length,p[:,1])]
        assert len(t)==8
        theta=np.sort(np.mod(np.arctan2(q[:,1],q[:,0]),2*np.pi))
        gaps=np.diff(np.r_[theta,theta[0]+2*np.pi])
        assert gaps.max()<.82
        assert all(k['kind']=='isolated_support_extremum'for k in plan['keys'])


def test_bending_open_path_retains_both_extreme_bends():
    x=np.linspace(0,600,1201);p=np.c_[x,60*np.sin(2*np.pi*x/600)]
    t,_=key_shape_stations(p)
    length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    q=np.c_[np.interp(t,length,p[:,0]),np.interp(t,length,p[:,1])]
    assert q[:,1].max()>59 and q[:,1].min()<-59
    assert q[:,0].min()<2 and q[:,0].max()>598
    assert len(t)<20


def test_complete_print_registration_and_semantic_gates_are_inherited():
    assert KeyShapePrintPattern.register is PrintPattern.register
    assert KeyShapePrintPattern.resolved_glyphs is PrintPattern.resolved_glyphs


def test_key_query_change_preserves_all_original_material_and_foreground_gates():
    import inspect
    import textwrap
    original=ast.parse(textwrap.dedent(inspect.getsource(PrintPattern.read_paths))).body[0]
    modified=ast.parse(textwrap.dedent(inspect.getsource(KeyShapePrintPattern.read_paths))).body[0]
    modified.args.defaults=original.args.defaults
    class RestoreQueryOnly(ast.NodeTransformer):
        def visit_Assign(self,node):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            if names==['key_plans']:return None
            if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='key_shape_stations':
                return ast.parse('t=np.arange(1,length[-1]-1,spacing)').body[0]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and isinstance(node.value.func.value,ast.Name)and node.value.func.value.id=='key_plans':return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='dict':node.keywords=[k for k in node.keywords if k.arg not in ('query_policy','key_query_plans')]
            return node
    modified=RestoreQueryOnly().visit(modified)
    assert ast.dump(modified,include_attributes=False)==ast.dump(original,include_attributes=False)
