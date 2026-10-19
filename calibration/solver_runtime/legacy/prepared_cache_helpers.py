"""Reconstruct immutable camera mappings when caching isolated source inputs."""
import copyreg
from types import MappingProxyType

def restore_mappingproxy(data):
    return MappingProxyType(data)

def reduce_mappingproxy(value):
    return restore_mappingproxy,(dict(value),)

def install_mappingproxy_pickle():
    copyreg.pickle(MappingProxyType,reduce_mappingproxy)

def restore_dict_state(instance,state):
    instance.__dict__.update(state)

def install_geometry_unpickle():
    from shared_rink_numeric_geometry_lanes import SharedRinkNumericGeometryLanes
    from shared_rink_numeric_geometry_lanes_ads import SharedRinkNumericGeometryLanesAds
    for cls in (SharedRinkNumericGeometryLanes,SharedRinkNumericGeometryLanesAds):
        cls.__setstate__=restore_dict_state
