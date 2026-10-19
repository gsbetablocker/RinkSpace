"""Recover after a missing profile from two independently reread source rows."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from .vision_board_confirmed_context_continuity_reader import ConfirmedContextContinuityBoardSourceReader

class ActualPairRecoveryBoardSourceReader(ConfirmedContextContinuityBoardSourceReader):
    def _actual_confirmed_boundary_readout(self,row):
        if (row.get('curve',{}).get('kind')!='advertisement_boundary'
                or row.get('source_identity_confirmed') is not True
                or row.get('source_authority_current_shot_confirmed') is not True
                or row.get('actual_source_RGB_measured') is not True
                or row.get('model_projection_or_fit_residual_used_for_identity') is not False):return False
        uv=np.asarray(row['uv'],float);blur=np.asarray(row['source_readout_uncertainty']['actual_readout_blur_interval_px'],float)
        if (len(uv)<2 or blur.shape!=(len(uv),2) or not np.isfinite(blur).all()
                or (blur[:,0]>0).any() or (blur[:,1]<0).any()):return False
        track=self.tracks.get(row['physical_material_id'])
        return bool(track is not None and track.get('current_instance_identity_confirmed') is True
            and self._reviewed_full_entity(track['template']['anchor']))

    def read(self,*args,**kwargs):
        result=super().read(*args,**kwargs);binding=result['source'];prior=getattr(self,'last_actual_boundary_readouts',{})
        observed={}
        for row in result['advertisement_rows']:
            if not self._actual_confirmed_boundary_readout(row):continue
            name=row['physical_material_id'];old=prior.get(name);temporal=row['temporal_evidence']
            if (row.get('allowed_for_fit') is not True and old is not None
                    and temporal['kind']=='same_shot_actual_RGB_temporal_identity'
                    and temporal.get('previous_frame_RGB_measured') is False
                    and temporal.get('current_frame_RGB_measured') is True
                    and old['shot_id']==row['shot_id']
                    and old['actual_pts']==temporal['previous_actual_pts']
                    and self.actual_successor.get(old['actual_pts'])==row['actual_pts']
                    and old['decoded_native_BGR_sha256']==temporal['previous_decoded_BGR_sha256']):
                # Both UV/blur arrays are verbatim actual profile outputs.
                # Their scope/instance confirmations came from authenticated
                # original material context and hard-mask actual source LK,
                # independently of whether an earlier edge profile existed.
                proof=dict(kind='two_actual_original_RGB_boundary_readouts_after_missing_profile',
                    physical_material_id=name,source_sha256=self.source_sha256,time_base=self.time_base,
                    shot_id=row['shot_id'],previous_actual_pts=old['actual_pts'],previous_native_pts=old['native_pts'],
                    previous_decoded_native_BGR_sha256=old['decoded_native_BGR_sha256'],
                    previous_actual_RGB_interface_uv=old['uv'],previous_actual_readout_blur_interval_px=old['blur'],
                    previous_source_identity_authority=old['source_identity_authority'],
                    current_actual_pts=row['actual_pts'],current_native_pts=row['native_pts'],
                    current_decoded_native_BGR_sha256=binding['decoded_native_BGR_sha256'],
                    current_actual_RGB_interface_uv=np.asarray(row['uv'],float).copy(),
                    current_actual_readout_blur_interval_px=np.asarray(row['source_readout_uncertainty']['actual_readout_blur_interval_px'],float).copy(),
                    current_source_identity_authority=row['source_identity_authority'],
                    both_actual_source_instance_contexts_independently_confirmed=True,
                    both_readouts_measured_by_actual_RGB_profile_operator=True,
                    model_projection_or_fit_residual_used_for_identity=False,
                    previous_unmeasured_source_frames_not_completed=True)
                temporal.update(previous_frame_RGB_measured=True,same_physical_entity_confirmed=True,
                    previous_actual_profile_was_measured_before_admission=True)
                row.update(source_actual_boundary_pair_recovery_proof=proof,
                    admission_status='admitted_source_advertisement_temporal_identity',status='measured_visible_source_boundary',
                    static_material_confirmed=True,pixel_location_confirmed=True,allowed_for_fit=True)
                self.tracks[name]['previous_frame_RGB_measured']=True
            observed[name]=dict(actual_pts=row['actual_pts'],native_pts=row['native_pts'],shot_id=row['shot_id'],
                decoded_native_BGR_sha256=binding['decoded_native_BGR_sha256'],
                uv=np.asarray(row['uv'],float).copy(),
                blur=np.asarray(row['source_readout_uncertainty']['actual_readout_blur_interval_px'],float).copy(),
                source_identity_authority=deepcopy(row['source_identity_authority']))
        self.last_actual_boundary_readouts=observed
        return result
