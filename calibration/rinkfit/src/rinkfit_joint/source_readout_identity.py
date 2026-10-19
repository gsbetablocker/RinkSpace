"""Exact aliases of one native physical readout operator, never residual pruning."""
from .io_utils import object_sha256

def actual_readout_key(row):
 p=row['source_measurement_audit']['profile']
 return object_sha256(dict(native_pts=row['native_pts'],material_role=row['material_role'],normal=row['normal'],navigation_seed_uv=row['navigation_seed_uv'],channel=p['channel'],polarity=p['polarity'],profile_configuration=p.get('configuration'),interpolation=p.get('interpolation'),current_uv=row['current_uv']))
