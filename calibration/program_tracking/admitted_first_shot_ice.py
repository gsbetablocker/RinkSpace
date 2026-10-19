"""Current extraction adapter with explicit, authenticated source admission.

The legacy extractor and its source caches remain historical producers. This
adapter reuses them in a separate process and redirects every write to a new
output root, adding complete archives and an explicit source revision before
writing an ice packet. Both cached and newly measured circles use one gate.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from . import extract_first_shot_ice as legacy
from .native_circle_source_admission import install_circle_source_admission, reference


class AdmittedFirstShotExtractor:
    def __init__(self, admission, *, source_root, output_root, model_root=None):
        self.source_root = Path(source_root).resolve()
        self.output_root = Path(output_root).resolve()
        if self.output_root.is_relative_to(self.source_root):
            raise ValueError('Current extraction must not overwrite the historical source root')
        self.admission = admission
        self.extractor = legacy.Extractor(self.source_root, model_root)
        install_circle_source_admission(self.extractor, admission)

    def _revise(self, packet, index):
        revised = self.admission.revise_packet(packet,
            additional_archives=self.extractor.source_admission_archives.get(index, ()))
        report = revised.get('rejected_material_candidate_report_path')
        if report and Path(report).resolve().is_relative_to(self.source_root):
            revised['rejected_material_candidate_report_path'] = str(
                self.output_root / Path(report).resolve().relative_to(self.source_root))
        revised['admitted_extraction_producers'] = [reference(path) for path in (
            __file__, legacy.__file__,
            legacy.ROOT / 'calibration/program_tracking/native_ice_material_repair.py',
            legacy.ROOT / 'calibration/program_tracking/native_circle_identity_review.py')]
        return revised

    def circle_packet(self, index, *, fresh=False):
        """Read a real cache or real current-native source, without writing it."""
        start = time.perf_counter()
        ex = self.extractor
        ex.use_circle_cache = not fresh
        source = deepcopy(ex.rows[index]['source'])
        bgr = legacy.cv2.imread(str(legacy.DATA / f'frames/{index:06d}.png'))
        person = legacy.cv2.imread(source['mask_path'], 0)
        if bgr is None or person is None:
            raise ValueError('Missing exact native source image or current foreground mask')
        broadcast, bcast = legacy.current_broadcast_exclusion(bgr)
        outside_ice, domain = legacy.current_ice_visible_domain(index)
        native = legacy.NativeProfiles(legacy.cv2.cvtColor(bgr, legacy.cv2.COLOR_BGR2RGB),
            source, (person > 0) | broadcast | outside_ice)
        features, diagnostics = ex.nested_house_features(index, native)
        source.update(mask_sha256=legacy.sha(source['mask_path']),
            broadcast_timer_exclusion=bcast, ice_visible_domain=domain)
        packet = dict(schema='rinkspace.repaired.current-native-circle-cache.v1',
            programme_frame_index_1based=index, source=source,
            algorithm_sha256=legacy.sha(legacy.ROOT / 'calibration/program_tracking/native_ice_material_repair.py'),
            model_scene_sha256=legacy.sha(ex.model_root / 'scene.json'),
            circle_geometry_sha256=ex.circle_geometry_sha256,
            features_by_curve=features, diagnostics=diagnostics,
            current_native_only=True, model_and_temporal_pixels_not_observations=True,
            elapsed_seconds=time.perf_counter()-start)
        return self._revise(packet, index)

    def extract(self, index, *, fresh=False):
        """Reuse the complete legacy extraction with a guarded packet writer.

        Legacy write lookup is redirected only for this synchronous call and
        restored in finally. Run this adapter in its own process, as the CLI
        does; it is not a concurrent in-process extraction API.
        """
        self.admission.assert_current()
        original_write = legacy.write
        self.extractor.use_circle_cache = not fresh
        written = {}

        def write(path, value):
            path = Path(path).resolve()
            if not path.is_relative_to(self.source_root):
                raise ValueError('Unexpected legacy extraction write outside source root')
            destination = self.output_root / path.relative_to(self.source_root)
            if isinstance(value, dict) and ('features' in value or 'features_by_curve' in value):
                value = self._revise(value, index)
                written[str(path)] = value
            if destination.exists():
                raise FileExistsError('Preserve prior extraction evidence: ' + str(destination))
            original_write(destination, value)

        legacy.write = write
        try:
            packet = self.extractor.extract(index)
        finally:
            legacy.write = original_write
        key = str((self.source_root / f'ice_candidates/{index:06d}.json').resolve())
        revised = written[key] if key in written else self._revise(packet, index)
        for archive in revised['inactive_source_estimators']:
            self.extractor.previous.pop(archive['target_id'].removeprefix('ice.'), None)
        return revised

    def write_circle(self, packet):
        self.admission.assert_current()
        index = int(packet['programme_frame_index_1based'])
        destination = self.output_root / f'circle_candidates/{index:06d}.json'
        if destination.exists():
            raise FileExistsError('Preserve prior extraction evidence: ' + str(destination))
        legacy.write(destination, self._revise(packet, index))
        return destination
