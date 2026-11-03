from types import SimpleNamespace
import unittest

try:
    import imgui
except ImportError:
    raise unittest.SkipTest('requires results3d runtime with imgui')
from rinkspace.viewer3d.imgui_buffers import install_dynamic_buffers


class Buffer:
    def __init__(self,size):self.size=size;self.resizes=[]
    def orphan(self,size):self.size=size;self.resizes.append(size)


def test_large_index_upload_grows_storage_and_keeps_existing_buffer_objects():
    vertex=Buffer(imgui.VERTEX_SIZE*65536);index=Buffer(imgui.INDEX_SIZE*65536)
    calls=[]
    data=SimpleNamespace(commands_lists=[SimpleNamespace(vtx_buffer_size=40000,idx_buffer_size=308472)])
    def upstream(draw):
        for row in draw.commands_lists:
            assert row.vtx_buffer_size*imgui.VERTEX_SIZE<=vertex.size
            assert row.idx_buffer_size*imgui.INDEX_SIZE<=index.size
        calls.append(draw)
    renderer=SimpleNamespace(_vertex_buffer=vertex,_index_buffer=index,render=upstream)
    install_dynamic_buffers(renderer);install_dynamic_buffers(renderer)
    renderer.render(data)
    assert renderer._vertex_buffer is vertex and renderer._index_buffer is index
    assert not vertex.resizes and len(index.resizes)==1
    assert calls==[data]
    renderer.render(data)
    assert len(index.resizes)==1


def test_capacity_uses_largest_list_in_bytes_for_both_uploads():
    vertex=Buffer(1);index=Buffer(1)
    renderer=SimpleNamespace(_vertex_buffer=vertex,_index_buffer=index,render=lambda _:None)
    install_dynamic_buffers(renderer)
    renderer.render(SimpleNamespace(commands_lists=[SimpleNamespace(vtx_buffer_size=7,idx_buffer_size=20),
        SimpleNamespace(vtx_buffer_size=11,idx_buffer_size=12)]))
    assert vertex.size==11*imgui.VERTEX_SIZE
    assert index.size==20*imgui.INDEX_SIZE
    renderer.render(SimpleNamespace(commands_lists=[]))
    assert len(vertex.resizes)==len(index.resizes)==1
