"""Fail-closed canonical source identity before native print numerical seeding.

Original coarse seed names stay in their immutable source records. Only an
original-image canonical authority can choose the packet material and path.
A fitted scene may subsequently initialize station inside that one path.
"""
import numpy as np
from .seed_source_canonical_v1 import load_seed_paths


def seed_bindings(*,source_config,config,model,eligible,path_review=None):
    count=len(config['seeds']);materials=np.full(count,-1,dtype='<i4');paths=np.full(count,-1,dtype='<i4')
    known=load_seed_paths(source_config=source_config,path_review=path_review)if path_review is not None else{}
    indices={m['label']:i for i,m in enumerate(model['materials'])};unknown={}
    for si,seed in enumerate(config['seeds']):
        if seed['label']not in eligible:
            unknown[si]='source_named_panel_not_qualified';continue
        proof=known.get(si)
        if proof is None:
            unknown[si]='original_image_canonical_material_path_unknown';continue
        label=proof['canonical_label'];path=int(proof['path'])
        if label not in indices:raise ValueError('Source-confirmed canonical material absent from model')
        mi=indices[label];material=model['materials'][mi]
        if(label.split('.')[0]!=seed['label'].split('.')[0]or not 0<=path<len(material['paths'])
                or material['paths'][path].get('physical_point',False)):
            raise ValueError('Source-confirmed canonical finite path absent from original physical panel')
        materials[si]=mi;paths[si]=path
    return materials,paths,unknown


def numerical_rows(*,seed_indices,uv,materials,paths,model):
    """Install exactly the same canonical material that the producer serializes."""
    result=[]
    for si,point in zip(seed_indices,uv):
        mi=int(materials[int(si)]);path=int(paths[int(si)])
        if mi<0 or path<0:raise ValueError('Unknown source identity cannot reach a numerical initializer')
        result.append(dict(material_index=mi,material_label=model['materials'][mi]['label'],source_uv=point.tolist(),allowed_paths=[path],path_index=path))
    if len(result)!=len(seed_indices):raise ValueError('Source seed/UV layout differs')
    return result


class CurrentExposureIdentityGateV1:
    """A named anchor and a good optical flow do not qualify its descendants.

    Only an explicit original-image review of this exact emitted source row can
    confirm its canonical material/path. Missing reviews remain candidates in
    the immutable NPZ store; no model residual or nearest projected path enters
    this decision. The review is an authority record, not a numerical claim that
    this loader independently certifies the image interpretation.
    """
    def __init__(self, *, source_config, candidate_manifest, path_review,
                 current_review, config, materials, paths, model):
        from .seed_source_path_v1 import _read, _check_ref
        from .io_utils import file_sha256
        from pathlib import Path
        self.entries = {}
        self.last_reasons = []
        self.cache = {}
        self.materials = materials
        self.paths = paths
        self.model = model
        self.config = config
        if path_review is None or current_review is None:
            raise ValueError('Native print issuance requires original seed/path and current-exposure identity reviews; optical flow is navigation only')
        review = _read(current_review)
        if (review.get('schema') != 'rinkfit.native_print_current_identity_review.v1'
                or review.get('source_original_images_reviewed') is not True
                or review.get('source_model_projection_used') is not False
                or review.get('qualification_uses_model_residual') is not False
                or review.get('source_pixels_changed') is not False
                or review.get('source_video_sha256') != config['source_video_sha256']):
            raise ValueError('Explicit source-only current-exposure print identity authority required')
        for key, actual in [('source_config', source_config),
                            ('candidate_manifest', candidate_manifest),
                            ('source_seed_path_review', path_review)]:
            declared = review[key]
            if (Path(declared['file']).resolve() != Path(actual).resolve()
                    or declared['sha256'] != file_sha256(actual)):
                raise ValueError('Current print identity review changed ' + key)
        for entry in review['rows']:
            si = int(entry['seed_index'])
            if not 0 <= si < len(config['seeds']):
                raise ValueError('Current identity seed index outside original source')
            seed = config['seeds'][si]
            key = (entry['shot_id'], int(entry['native_pts']), si)
            if key in self.entries:
                raise ValueError('Duplicate current-exposure print identity verdict')
            uv = np.asarray(entry['source_uv'], dtype=float)
            if (uv.shape != (2,) or not np.isfinite(uv).all()
                    or entry['source_label'] != seed['label']
                    or entry.get('uses_fitted_model_projection') is not False
                    or entry.get('qualification_uses_model_residual') is not False
                    or entry.get('raw_UV_PTS_ID_preserved') is not True
                    or entry.get('decision') not in {'confirmed', 'unknown', 'revoked'}
                    or not entry.get('reason')):
                raise ValueError('Current print verdict lacks an unchanged source identity')
            expected = 'native_named_print_1022/' + key[0] + '/' + str(key[1]) + '/seed' + str(si)
            if entry['source_row_id'] != expected:
                raise ValueError('Current print verdict source ID/PTS/seed mismatch')
            if entry['decision'] == 'confirmed':
                mi = int(materials[si]); path = int(paths[si])
                if (mi < 0 or path < 0
                        or entry.get('physical_entity_path_source_confirmed') is not True
                        or entry['canonical_label'] != model['materials'][mi]['label']
                        or int(entry['path']) != path
                        or not entry.get('source_identity_evidence')):
                    raise ValueError('Current verdict cannot grant an unconfirmed canonical seed/path')
                for evidence in entry['source_identity_evidence']:
                    _check_ref(evidence, self.cache)
            self.entries[key] = entry
        self.review = review
        self.review_file = current_review

    def mask(self, *, shot_id, native_pts, seed_indices, uv, frame):
        """Check the actual emitted row, never a source label or track aggregate."""
        from pathlib import Path
        if len(seed_indices) != len(uv) or ('native_pts' in frame and int(frame['native_pts']) != int(native_pts)):
            raise ValueError('Current identity gate source row/frame layout differs')
        keep = np.zeros(len(seed_indices), dtype=bool)
        reasons = []
        for i, (si, point) in enumerate(zip(seed_indices, uv)):
            si = int(si)
            entry = self.entries.get((shot_id, int(native_pts), si))
            if entry is None:
                reasons.append('current_exposure_entity_identity_unknown')
                continue
            raw = entry['source_readout']
            if (not np.array_equal(np.asarray(entry['source_uv'], float), point)
                    or entry['source_BGR_sha256'] != frame['source_BGR_sha256']
                    or Path(raw['file']).resolve() != Path(frame['file']).resolve()
                    or raw['sha256'] != frame['sha256']):
                raise ValueError('Current identity verdict differs from actual immutable UV/RGB/readout')
            keep[i] = entry['decision'] == 'confirmed'
            reasons.append('source_confirmed' if keep[i] else 'current_exposure_entity_identity_' + entry['decision'])
        self.last_reasons = reasons
        return keep


def canonical_allowed_paths(*, seed_indices, materials, paths):
    """Retain exact canonical permissions in the existing per-material ABI.

    This compact registry has one permission set per material, so it cannot
    represent distinct source-confirmed paths for different rows of that same
    material. Refuse such an issue rather than silently union their identities.
    """
    result = {}
    for si in np.unique(seed_indices):
        mi = int(materials[int(si)]); path = int(paths[int(si)])
        if mi < 0 or path < 0:
            raise ValueError('Unconfirmed seed cannot issue a finite path permission')
        key = str(mi)
        if key in result and result[key] != [path]:
            raise ValueError('Distinct per-row canonical paths require an explicit row-path registry; the compact per-material ABI cannot union identities')
        result[key] = [path]
    return result
