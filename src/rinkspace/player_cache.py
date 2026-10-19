"""Small disposable UI caches; complete localization evidence stays read-only."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

from .io import dump_json, load_json, sha256
from .storage import AssetStore

CACHE_VERSION=8
DISPLAY_KEYS=frozenset(('ice_polygons_sample','foreground_polygons_sample','display_mask_provenance'))
HEADER_KEYS=frozenset(('schema_version','algorithm_version','model_id','model_revision','revision','source_id',
    'video_id','source_sha256','model_sha256','selected_window','status_counts','role','review_note','label',
    'anchor_ablation','diagnostic_preview_only','experimental','all_selected_frames_in_denominator',
    'reobserve_every_source_frames','created_at','updated_at','media_time_offset_s','frame_duration_s',
    'landmark_observation_schema','landmark_observation_provenance','landmark_mode',
    'reference_library','reference_library_sha256','reference_source_ids','same_programme_references_allowed'))
FRAME_KEYS=frozenset(('sample_index','source_frame_index','source_pts','source_time_base','source_time_s','pts','time_base',
    'media_time_s','source_duration_s','width','height','source_size','sample_size','image_width','image_height','source_width','source_height',
    'model_id','model_revision','model_sha256','source_id','source_sha256','shot_id','source_shot_id','camera_component_id',
    'status','reason','H_model_from_image','H_image_from_model','projection_chain','matrix_direction','image_domain',
    'support_polygon_image','ice_polygons_sample','foreground_polygons_sample','foreground_polygons_image','person_boxes_image',
    'anchor_id','reference_view_id','anchor_age_s','anchor_lineage','future_anchor_used','last_image_observation_age_s',
    'uncertainty','camera_uncertainty','independent_validation','independent_validation_passed','independent_geometry_validated',
    'physical_ids','break_applied','athlete_ground_state','athlete_continuity','scene_observations','display_mask_provenance',
    'image_scale_from_source','landmark_observations','landmark_detection_summary','observation_status',
    'camera_group_id','installation_id','anchor_source_id','image_evidence','reference_library_sha256','display_adapter'))
LANDMARK_VIEW_KEYS=frozenset(('observation_id','id','kind','surface_type','colour_class','colour_candidates',
    'source_id','source_sha256','source_frame_index','source_pts','source_time_base','source_time_s','pts','time_base',
    'source_size','sample_size','image_domain','measurement_domain','coordinate_domain','image_scale_from_source',
    'points_source_px','visible_segment_ranges','visible_segment_range_convention','visible_segments_source_px',
    'points_sample_px','visible_segments_sample_px','candidate_model_ids','selected_model_id','physical_id',
    'association_status','used_by_solver','identity_basis','manual_confirmed','user_confirmed','confirmation_actor',
    'not_automatic','role','rejection_stage','rejection_reason',
    'measurement_uncertainty_source_px','measurement_uncertainty_sample_px','group_weight',
    'residual_summary_sample_px','mask_fraction'))


def _pixel_size(value,field):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(field+' must contain exactly two native pixel dimensions')
    for dimension in value:
        if (isinstance(dimension,bool) or not isinstance(dimension,(int,float))
                or dimension<=0 or (isinstance(dimension,float)
                    and (not math.isfinite(dimension) or not dimension.is_integer()))):
            raise ValueError(field+' must contain positive finite integer pixel dimensions')
    return [int(dimension) for dimension in value]


def _adapt_projection_for_display(view):
    """Fill an omitted sample-to-source scale only from explicit dimensions.

    This operates on the disposable player payload. Camera estimates, source
    evidence and the localization file are never changed or reinterpreted.
    """
    chain=view.get('projection_chain')
    if not isinstance(chain,dict) or chain.get('H_source_image_from_sample') is not None:return
    adapter={'schema_version':'rinkspace.player-display-adapter.v1',
             'field':'projection_chain.H_source_image_from_sample',
             'inference_unchanged':True,'source_document_unchanged':True}
    try:
        candidates={}
        if view.get('width') is not None or view.get('height') is not None:
            candidates['width,height']=_pixel_size([view.get('width'),view.get('height')],'width,height')
        if view.get('source_size') is not None:
            candidates['source_size']=_pixel_size(view['source_size'],'source_size')
        if not candidates:raise ValueError('Missing explicit native width,height or source_size')
        # These aliases are not a fallback, but must agree when also supplied.
        if view.get('source_width') is not None or view.get('source_height') is not None:
            candidates['source_width,source_height']=_pixel_size(
                [view.get('source_width'),view.get('source_height')],'source_width,source_height')
        source_size=next(iter(candidates.values()))
        if any(size!=source_size for size in candidates.values()):
            raise ValueError('Conflicting explicit native source dimensions')
        sample_size=_pixel_size(chain.get('current_sample_size'),'projection_chain.current_sample_size')
        if view.get('sample_size') is not None and _pixel_size(view['sample_size'],'sample_size')!=sample_size:
            raise ValueError('Conflicting explicit current sample dimensions')
    except ValueError as error:
        adapter.update(status='rejected',reason=str(error))
    else:
        view['projection_chain']={**chain,'H_source_image_from_sample':[
            [source_size[0]/sample_size[0],0,0],
            [0,source_size[1]/sample_size[1],0],[0,0,1]]}
        adapter.update(status='applied',reason='Explicit native/source sample dimension ratios',
                       source_size=source_size,current_sample_size=sample_size,
                       source_dimension_fields=list(candidates),
                       sample_dimension_field='projection_chain.current_sample_size')
    view['display_adapter']=adapter


def _signature(path,include_display_masks=False):
    stat=path.stat()
    signature={'version':CACHE_VERSION,'mtime_ns':stat.st_mtime_ns,'bytes':stat.st_size}
    if include_display_masks:
        sidecar=path.with_name('display_masks.json')
        display_stat=sidecar.stat() if sidecar.is_file() else None
        signature['display_masks']=({'mtime_ns':display_stat.st_mtime_ns,'bytes':display_stat.st_size}
                                    if display_stat is not None else None)
    return signature


def _display_overrides(path,full):
    """Validate and index the optional view-only masks once per cache miss."""
    sidecar=path.with_name('display_masks.json')
    if not sidecar.is_file():return {}
    data=load_json(sidecar)
    if data.get('schema_version')!='rinkspace.display-masks.v1':
        raise ValueError('Unsupported display-mask sidecar schema')
    if data.get('localization_sha256')!=sha256(path):
        raise ValueError('Display-mask sidecar does not bind this localization')
    for key in ('source_sha256','model_sha256'):
        if not full.get(key) or data.get(key)!=full[key]:
            raise ValueError('Display-mask sidecar identity differs: '+key)
    source_keys={(frame['pts'],Fraction(frame['time_base'])) for frame in full.get('frames',[])}
    overrides={}
    for row in data.get('frames',[]):
        key=(row['pts'],Fraction(row['time_base']))
        if key not in source_keys or key in overrides:
            raise ValueError('Display-mask sidecar has an unknown or duplicate source frame')
        if not DISPLAY_KEYS.issubset(row):
            raise ValueError('Display-mask sidecar is missing a display field')
        overrides[key]={field:row[field] for field in DISPLAY_KEYS}
    return overrides


def _header(path):
    """Read the small top-level prefix of the project's indented JSON writer.

    Standard delivery files put identity/window metadata before the frame array.
    Nonstandard layouts fall back to JSON parsing once and are still cached.
    """
    prefix=''
    with path.open(encoding='utf8') as stream:
        while len(prefix)<2*1024*1024:
            chunk=stream.read(65536)
            if not chunk:break
            prefix+=chunk
            match=re.search(r'\n  "frames"\s*:',prefix)
            if match:
                header=json.loads(prefix[:match.start()].rstrip().removesuffix(',')+'\n}')
                window=header.get('selected_window',{})
                if window.get('frame_count') is not None and window.get('first_pts') is not None and window.get('last_pts') is not None and window.get('time_base'):
                    return header
                break
    return load_json(path)


class LocalizationCache:
    def __init__(self,root):
        self.root=Path(root)
        self.storage=AssetStore(root)
        self.directory=self.storage.output('workbench/model_player_cache')

    def _path(self,path,kind):
        name=hashlib.sha256(self.storage.asset_url(path).encode()).hexdigest()[:24]
        return self.directory/f'{name}.{kind}.json'

    def _read(self,path,kind,signature):
        cache=self._path(path,kind)
        if cache.is_file():
            try:
                stored=load_json(cache)
                if stored.get('source_stat')==signature:return stored['data']
            except (OSError,ValueError,KeyError):pass
        return None

    def _write(self,path,kind,signature,data):
        cache=self._path(path,kind)
        temp=cache.with_suffix('.tmp')
        dump_json(temp,{'source_stat':signature,'data':data})
        temp.replace(cache)

    def summary(self,path):
        path=Path(path);signature=_signature(path)
        cached=self._read(path,'summary',signature)
        if cached is not None:return cached
        header=_header(path)
        result={k:v for k,v in header.items() if k in HEADER_KEYS}
        frames=header.get('frames')
        if frames is not None:
            times=[f['source_time_s'] for f in frames if isinstance(f.get('source_time_s'),(float,int))]
            result.update(frame_count=len(frames),start_s=min(times) if times else None,end_s=max(times) if times else None)
        else:
            window=header['selected_window'];tb=Fraction(window['time_base'])
            result.update(frame_count=window['frame_count'],start_s=float(window['first_pts']*tb),end_s=float(window['last_pts']*tb))
        self._write(path,'summary',signature,result)
        return result

    def player(self,path):
        path=Path(path);signature=_signature(path,include_display_masks=True)
        cached=self._read(path,'player',signature)
        if cached is not None:return cached
        full=load_json(path)
        display_overrides=_display_overrides(path,full)
        result={k:v for k,v in full.items() if k in HEADER_KEYS}
        result['frames']=[]
        for frame in full.get('frames',[]):
            view={k:v for k,v in frame.items() if k in FRAME_KEYS}
            _adapt_projection_for_display(view)
            if isinstance(view.get('landmark_observations'),list):
                # Keep every measured point, gap and failure status. Ellipse fits,
                # brightness arrays and solver alternatives remain in the full
                # source document, fetched only when its diagnostics are opened.
                view['landmark_observations']=[
                    {k:v for k,v in row.items() if k in LANDMARK_VIEW_KEYS}
                    for row in view['landmark_observations']]
            if display_overrides:
                view.update(display_overrides.get((frame['pts'],Fraction(frame['time_base'])),{}))
            result['frames'].append(view)
        result['player_payload_scope']='display_geometry_source_identity_and_status_only; full_diagnostics_at_document_url'
        self._write(path,'player',signature,result)
        return result
