"""Camera-only access for native factors; programme motion keeps its full sync."""

def native_camera_local(block,core):
    method=getattr(block,'camera_local',None)
    # Existing lightweight camera-evidence adapters expose only local().
    return block.local(core) if method is None else method(core)
