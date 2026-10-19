"""Source-identity admission for complete current-native circle estimators.

A good local profile or a small distance to a fitted circle does not establish
physical instance identity. Reviewed rejections bind the original video, native
PTS and complete target estimator, so re-running or changing sparse stations
cannot silently undo a rejection. This module never reads solver residuals,
changes material measurements, slices a correlated operator, or assigns colour
rules to other events. It authenticates the programme's explicit source facts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

POLICY_SCHEMA = 'rinkspace.native-circle-source-admission.v1'
FACT_SCHEMA = 'rinkspace.reviewed-complete-circle-estimator-quarantine.v1'
REVISION_SCHEMA = 'rinkspace.complete-native-estimator-source-revision.v1'
SOURCE_FIELDS = ('event_id', 'programme_id', 'source_video_sha256', 'time_base')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


class ReviewedCircleAdmission:
    """Explicit negative source facts, independent of output directory/model.

    The caller supplies an authenticated policy reference and a path resolver.
    Historical aliases can therefore use the project's existing AssetStore.
    Source facts and their exact original source packets are reauthenticated
    before every admission; changes fail closed instead of retaining old facts.
    This gate only enforces the specified negative facts. Unlisted material is
    not certified as correct merely because it is not rejected here.
    """
    def __init__(self, policy_reference, *, resolve=None):
        self.resolve = resolve or (lambda path: Path(path).resolve())
        self.dependencies = {}
        self.policy_reference = deepcopy(policy_reference)
        self.policy = self._read(policy_reference)
        if self.policy.get('schema') != POLICY_SCHEMA:
            raise ValueError('Explicit native-circle admission policy required')
        self.source_identity = tuple(self.policy['source'][k] for k in SOURCE_FIELDS)
        if not re.fullmatch('[0-9a-f]{64}', self.source_identity[2]):
            raise ValueError('Exact original video SHA256 required')
        self.denials = {}
        for fact_reference in self.policy['reviewed_quarantine_manifests']:
            facts = self._read(fact_reference)
            if (facts.get('schema') != FACT_SCHEMA
                    or facts.get('review_complete') is not True
                    or facts.get('authority') != 'agent_rgb_review'
                    or facts.get('user_confirmed') is not False):
                raise ValueError('Completed source RGB review required; no inferred human confirmation')
            for field in ('source_identity_review', 'actual_original_group_export'):
                self._bytes(facts[field])
            for item in facts['specifications']:
                self._add(item, fact_reference)
        if not self.denials:
            raise ValueError('Empty source admission policy')

    def _bytes(self, ref):
        if not ref.get('path') or not re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')):
            raise ValueError('Authenticated source evidence reference required')
        path = Path(self.resolve(ref['path']))
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != ref['sha256']:
            raise ValueError('Source admission evidence changed: ' + str(path))
        self.dependencies[str(path)] = ref['sha256']
        return data

    def _read(self, ref):
        return json.loads(self._bytes(ref).decode('utf-8-sig'))

    def _identity(self, source):
        return tuple(source[k] for k in SOURCE_FIELDS)

    def _add(self, item, fact_reference):
        ids = item.get('source_point_ids', [])
        targets = {str(p).split('/segment/')[0] for p in ids}
        if (not ids or len(set(ids)) != len(ids) or len(targets) != 1
                or item.get('material_identity_confirmed') is not False
                or not item.get('reason') or not item.get('evidence_references')
                or not re.fullmatch('[0-9a-f]{64}', item.get('operator_matrix_sha256', ''))
                or len(item.get('operator_matrix_shape', [])) != 2
                or item['operator_matrix_shape'][0] <= 0
                or item['operator_matrix_shape'][1] != 2*len(ids)):
            raise ValueError('Review must name one complete uncertain target estimator')
        target = next(iter(targets))
        packets = []
        for ref in item['evidence_references']:
            raw = self._bytes(ref)
            if Path(ref['path']).suffix.lower() != '.json':
                continue
            packet = json.loads(raw.decode('utf-8-sig'))
            source = packet.get('source')
            if not isinstance(source, dict) or 'source_video_sha256' not in source:
                continue
            if (self._identity(source) != self.source_identity
                    or int(source['native_pts']) != int(item['native_pts'])
                    or int(source['programme_frame_index_1based']) != int(item['frame'])):
                raise ValueError('Reviewed estimator and original native source provenance disagree')
            if 'features' in packet:
                features = packet['features']
            elif 'features_by_curve' in packet:
                features = list(packet['features_by_curve'].values())
            else:
                continue
            matches = [f for f in features if f.get('target_id') == target]
            if len(matches) != 1 or not matches[0].get('source_circle_fit'):
                raise ValueError('Reviewed original packet lacks the complete circle estimator')
            packets.append(ref)
        if not packets:
            raise ValueError('Native source packet is required to bind a reviewed denial')
        key = self.source_identity + (int(item['native_pts']), target)
        if key in self.denials:
            raise ValueError('Duplicate complete native estimator denial')
        self.denials[key] = dict(review=deepcopy(item), fact_reference=deepcopy(fact_reference),
            original_packet_references=deepcopy(packets))

    def assert_current(self):
        for path, expected in self.dependencies.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Source admission evidence changed: ' + path)

    def _denial(self, source, feature):
        identity = self._identity(source)
        if identity != self.source_identity:
            return None
        denial = self.denials.get(identity + (int(source['native_pts']), feature['target_id']))
        if denial is not None and int(source['programme_frame_index_1based']) != denial['review']['frame']:
            raise ValueError('Native PTS and programme index disagree in source admission')
        return denial

    def admit_features(self, source, features):
        """Return a derivative and whole archives; do not mutate the input."""
        self.assert_current()
        output = deepcopy(features)
        values = list(output.values()) if isinstance(output, dict) else output
        rejected_groups = {f.get('source_evidence_group_id') for f in values
            if self._denial(source, f) is not None and f.get('source_evidence_group_id')}
        if any(f.get('source_evidence_group_id') in rejected_groups
                and self._denial(source, f) is None for f in values):
            raise ValueError('Source denial would split a shared estimator; complete review required')
        archives = []
        for feature in values:
            denial = self._denial(source, feature)
            if denial is None:
                continue
            # The key deliberately excludes sparse point IDs, UV and fitted W:
            # regenerating the same unverified physical estimator cannot clear
            # a source-identity denial by changing its numerical representation.
            if not feature.get('point_count') and not feature.get('source_circle_fit'):
                continue
            archive = dict(source=deepcopy(source), target_id=feature['target_id'],
                complete_original_feature=deepcopy(feature),
                original_feature_canonical_sha256=_digest(feature), **deepcopy(denial),
                excluded_from_FIT=True, original_source_evidence_modified=False,
                complete_operator_is_retained_by_original_group_export=True,
                no_correlated_operator_rows_or_columns_sliced=True)
            archives.append(archive)
            feature['segments'] = []
            feature['point_count'] = 0
            for field in ('source_circle_fit', 'joint_group_navigation',
                    'geometric_interpretation', 'source_evidence_group_id'):
                feature.pop(field, None)
            feature['status'] = 'complete_estimator_source_identity_quarantined'
            feature['source_admission_revision'] = deepcopy(self.policy_reference)
            feature.setdefault('extraction_diagnostic', {}).update(
                physical_identity_rejected=True, source_identity_confirmed=False,
                missing_reason=denial['review']['reason'])
        return output, archives

    def revise_packet(self, packet, *, additional_archives=()):
        """Make an explicitly revised packet, preserving whole failed estimators."""
        field = 'features_by_curve' if 'features_by_curve' in packet else 'features'
        features, archives = self.admit_features(packet['source'], packet[field])
        result = deepcopy(packet)
        result[field] = features
        existing = list(result.get('inactive_source_estimators', []))
        for archive in [*additional_archives, *archives]:
            # Admission must not merge an archive from another source clock.
            if (self._identity(archive['source']) != self._identity(packet['source'])
                    or archive['source']['native_pts'] != packet['source']['native_pts']):
                raise ValueError('Source archive belongs to another native clock')
            if archive not in existing:
                existing.append(deepcopy(archive))
        result['inactive_source_estimators'] = existing
        active = list(features.values()) if isinstance(features, dict) else features
        if field == 'features':
            result[field] = [f for f in active if f.get('point_count')]
        result['point_count'] = sum(f.get('point_count', 0) for f in active)
        active_ids = {f.get('source_evidence_group_id') for f in active if f.get('point_count')}
        if 'source_correlated_groups' in result:
            result['source_correlated_groups'] = [g for g in result['source_correlated_groups']
                if g['group_id'] in active_ids]
        previous = result.get('source_admission_revision')
        result['source_admission_revision'] = dict(schema=REVISION_SCHEMA,
            policy=deepcopy(self.policy_reference), admission_producer=reference(__file__),
            source_identity_fields=list(SOURCE_FIELDS),
            source_identity=list(self._identity(packet['source'])),
            native_pts=int(packet['source']['native_pts']),
            complete_estimator_denials_applied=len(existing),
            input_packet_canonical_sha256=(previous or {}).get('input_packet_canonical_sha256', _digest(packet)),
            historical_packet_bytes_modified=False, original_failed_estimators_fully_archived=True,
            fitted_circle_residual_is_not_material_identity=True,
            unlisted_material_identity_certified=False)
        return result


def install_circle_source_admission(extractor, admission):
    """Wrap the shared return point used by both cached and newly read circles.

    The caller must use revise_packet at the write boundary to carry archives
    into the final packet. No historical producer file is edited or replaced.
    """
    original = extractor.nested_house_features
    extractor.source_admission_archives = {}

    def nested_house_features(index, native):
        features, diagnostics = original(index, native)
        source = extractor.rows[index]['source']
        revised, archives = admission.admit_features(source, features)
        extractor.source_admission_archives[index] = archives
        return revised, diagnostics

    extractor.nested_house_features = nested_house_features
    return extractor
