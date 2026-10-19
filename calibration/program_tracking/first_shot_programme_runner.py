"""Retain the public runner while reporting the actual programme-only scope."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import numpy as np
from scipy.sparse import save_npz

from .first_shot_source_transition import reference, read
from .first_shot_joint_stage_runtime import write, sha


def programme_runner(parent_class):
    """Compose with the unchanged physical factory's owning runner."""
    class ProgrammeRunner(parent_class):
        def _scope(self):
            return dict(programme_source_objective=self.owner.receipt,
                programme_runner=reference(__file__),
                wide_source_role='historical/reference only; excluded from objective',
                objective_is_complete_official50_programme=True)

        def prepare(self, **extra):
            result = super().prepare(**extra)
            if result['source_rows'] != len(self.owner.scales):
                raise ValueError('Actual complete programme row count changed')
            result.update(self._scope())
            result.update(schema='rinkspace.complete-programme-source-stage-QA.v1',
                programme_cost=result['combined_cost'], wide_cost=0., wide_rows=0,
                retired_wide_reference_rows=len(self.owner.factor.scales),
                combined_cost_is_programme_only=True)
            write(self.output/'append_QA.json', result)
            return result

        def source_blocks(self, q, jacobian):
            if not jacobian:
                yield from self.owner.source_blocks(q, False)
                return
            from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
            bank=self.cache/('generation_%06d'%self.generation)
            bank.mkdir(parents=True,exist_ok=False)
            values=[];blocks=[];losses={};cursor=0
            hook=getattr(self.owner,'source_J_block_reference',None)
            for index,block in enumerate(self.owner.source_blocks(q,True)):
                borrowed=hook(q,index,block)if callable(hook)else None
                if borrowed is None:
                    path=bank/('block_%02d.npz'%index)
                    save_npz(path,block.J,compressed=False)
                    matrix_ref=reference(path)
                else:
                    if reference(borrowed['path'])!=borrowed:
                        raise ValueError('Verified external programme J reference changed')
                    matrix_ref=borrowed
                values.append(block.r.copy())
                losses.update({key+'__block_'+str(index):value for key,value in
                    source_loss_checkpoint_arrays(block.loss_weights,len(block.r)).items()})
                blocks.append(dict(block=index,row_start=cursor,row_count=len(block.r),J=matrix_ref,
                    J_shape=list(block.J.shape),J_nonzeros=block.J.nnz,row_ledger=block.row_ledger,
                    verified_existing_CSR_reference_reused=borrowed is not None))
                cursor+=len(block.r)
                yield block
            if cursor!=len(self.owner.scales):
                raise ValueError('Current programme source writer omitted active rows')
            np.savez(bank/'linearization_state.npz',**self.owner.complete_state_payload(q),
                r=np.concatenate(values),scales=self.owner.scales,quadratic=self.owner.quadratic)
            np.savez(bank/'block_loss_weights.npz',**losses)
            result=dict(schema='rinkspace.complete-canonical-effective-joint-source-J.v1',generation=self.generation,
                qaug_sha256=sha(q),J_shape=[cursor,len(q)],J_nonzeros=sum(b['J_nonzeros']for b in blocks),
                complete_current_state_r_scales_quadratic=reference(bank/'linearization_state.npz'),blocks=blocks,
                block_loss_weights=reference(bank/'block_loss_weights.npz'),
                source_J_before_Huber_weights=True,source_producers=self.owner.source_producers,
                source_identity=self.owner.checkpoint_identity(),**self._scope(),
                full_static_rows_always_present=False,
                programme_rows=len(self.owner.scales), wide_rows=0,
                all_active_objective_rows_present=True,
                fresh_current_r_loss_masks_scales_and_payload_written=True,
                original_CSR_bytes_not_copied_when_verified_reference_hook_available=True)
            write(bank/'manifest.json', result)
            write(self.cache/'latest.json', result)
            self.generation+=1
            print(json.dumps(dict(stage='complete_programme_source_J_exported',generation=self.generation-1,
                q_sha256=sha(q),rows=cursor,columns=len(q),
                verified_CSR_references_reused=sum(b['verified_existing_CSR_reference_reused']for b in blocks))),flush=True)

        def checkpoint(self, q, event):
            # The parent retains complete state export and source verification.
            # Suppress its outdated final-block-as-wide console label.
            captured = StringIO()
            with redirect_stdout(captured):
                result = super().checkpoint(q, event)
            if result is None:
                return None
            report_path = Path(result['current_report']['path'])
            report = read(result['current_report'])
            report.update(self._scope())
            report.update(schema='rinkspace.accepted-programme-source-complete-state.v1',
                programme_source_cost=report['source_objective_cost'], wide_source_cost=0.,
                wide_group_metrics_role='historical fixed-camera diagnostic; not fitted or accepted')
            write(report_path, report)
            write(self.output/'current.json', report)
            updated = reference(report_path)
            if not self.records or self.records[-1]['q_sha256'] != sha(q):
                raise ValueError('Actual accepted programme record changed')
            self.records[-1]['report'] = updated
            write(self.output/'accepted_state_index.json', dict(records=self.records))
            result = dict(result, current_report=updated)
            # Other main-process diagnostics remain available; replace only the
            # old structured acceptance summary that falsely names a wide cost.
            for line in captured.getvalue().splitlines():
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    print(line, flush=True)
                    continue
                if record.get('stage') != 'accepted_canonical_effective_stage_state':
                    print(line, flush=True)
            print(json.dumps(dict(stage='accepted_complete_programme_source_state',
                kind=report['kind'], accepted_index=report['accepted_index'],
                q_sha256=sha(q), programme_cost=report['programme_source_cost'],
                wide_camera_and_pixel_fit=False)), flush=True)
            return result

    return ProgrammeRunner
