"""Grow AITViewer's fixed ImGui buffers before its unchanged draw routine."""
import imgui


def install_dynamic_buffers(renderer):
    """Keep the existing GL buffers/VAO; orphan storage only when it is small.

    The upstream renderer reuses each buffer for one command list at a time.
    Capacity therefore follows the largest list's byte count, not the total
    number of plot samples or a hard-coded scene limit.
    """
    if getattr(renderer,'rinkspace_dynamic_buffers',False):return
    original=renderer.render
    renderer.rinkspace_dynamic_buffers=True
    renderer.rinkspace_buffer_usage={}

    def render(draw_data):
        lists=draw_data.commands_lists
        usage={}
        for name,field,stride in (('vertex','vtx_buffer_size',imgui.VERTEX_SIZE),
                                  ('index','idx_buffer_size',imgui.INDEX_SIZE)):
            required=max((getattr(commands,field)*stride for commands in lists),default=0)
            buffer=getattr(renderer,'_'+name+'_buffer')
            if required>buffer.size:buffer.orphan(max(required,buffer.size*2))
            usage[name]=dict(required_bytes=required,capacity_bytes=buffer.size)
        renderer.rinkspace_buffer_usage=usage
        return original(draw_data)
    renderer.render=render
