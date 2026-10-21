"""Fit real source texture tracks with one fixed unknown ice point per track.

Image translation is the two-coordinate subgroup of the declared SL(3) screen
correction. Its temporal knots are actual texture births and support edges,
never an unconditional per-frame homography. This recovery front end does not
certify a physical camera or a covariance model for RGB texture measurements.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import argparse, json
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from .io_utils import read_json, read_jsonl, write_json
from .screen_recovery_state import ScreenRecoveryState, reference, verified


def _stats(error, unit='px'):
    e = np.asarray(error, float)
    if not len(e): return dict(count=0, unit=unit, median=None, p95=None, maximum=None, rms=None)
    return dict(count=len(e), unit=unit, median=float(np.median(e)), p95=float(np.quantile(e,.95)),
        maximum=float(e.max()), rms=float(np.sqrt(np.mean(e*e))))


def _intervals(pairs):
    result = []
    for pts in sorted(pairs):
        if result and result[-1][1] == pts: result[-1][1] += 256
        else: result.append([pts, pts+256])
    return result


class TrackletProblem:
    """One continuous, source-supported shot interval with shared latent XY.

The first translation control is zero to choose the screen coordinate gauge;
all latent world positions and their first observed residuals remain free.
There is no source UV fixed to an H-inverted pseudo-world target.
"""
    def __init__(self, state, sid, groups, first, last):
        self.state=state; self.sid=sid; self.first=first; self.last=last
        self.groups=groups; self.ids=sorted(groups)
        self.knots=np.unique([first,last]+[min(r['native_pts'] for r in groups[k]) for k in self.ids])
        self.K=len(self.knots); self.T=len(self.ids); self.C=2*(self.K-1)
        self.latent0=[]; self.pts=[]; self.track=[]; self.uv=[]; self.H=[]; self.front=[]; self.rows=[]
        self.lookup={k:i for i,k in enumerate(self.ids)}
        for k in self.ids:
            rows=sorted(groups[k],key=lambda r:r['native_pts'])
            uv,valid=state.inverse_ice(rows[0]['native_pts'],sid,[rows[0]['source_uv']])
            if not valid.all(): raise ValueError('Track initialization has no inherited ice inverse')
            self.latent0.append(uv[0])
            for row in rows:
                pts=row['native_pts']; index=state.seed.frame_index(pts,sid)
                self.pts.append(pts);self.track.append(self.lookup[k]);self.uv.append(row['source_uv'])
                self.H.append(state.matrix(pts,sid));self.front.append(state.seed.a['front_sign'][index]);self.rows.append(row)
        self.latent0=np.asarray(self.latent0,float); self.pts=np.asarray(self.pts,np.int64)
        self.track=np.asarray(self.track,int); self.uv=np.asarray(self.uv,float);self.H=np.asarray(self.H,float)
        self.front=np.asarray(self.front,float); self.N=len(self.pts)
        i=np.searchsorted(self.knots,self.pts,side='right')-1;i=np.clip(i,0,self.K-2)
        alpha=(self.pts-self.knots[i])/(self.knots[i+1]-self.knots[i])
        self.control=np.c_[i,i+1]; self.weights=np.c_[1-alpha,alpha]
        self.x0=np.r_[np.zeros(self.C),self.latent0.ravel()]
        # Check branch and all source rows before any trial optimization.
        self.project(self.x0)

    def unpack(self,x):
        correction=np.vstack((np.zeros((1,2)),np.asarray(x[:self.C]).reshape(-1,2)))
        latent=np.asarray(x[self.C:]).reshape(-1,2)
        return correction,latent

    def project(self,x):
        correction,latent=self.unpack(x);X=latent[self.track]
        q=np.einsum('nij,nj->ni',self.H,np.c_[X,np.ones(self.N)])
        if np.any(self.front*q[:,2]<=0) or not np.isfinite(q).all():
            raise ValueError('Trial leaves static source track projective branch; no rows may disappear')
        uv=q[:,:2]/q[:,2:3]
        shift=(correction[self.control]*self.weights[:,:,None]).sum(axis=1)
        return uv+shift,q

    def residual(self,x):
        pred,_=self.project(x)
        return (pred-self.uv).ravel()

    def jacobian(self,x):
        _,q=self.project(x); rows=[];cols=[];values=[]
        # d(HX / H_3X) / dX for each track's shared 2D latent point.
        d=(self.H[:,:2,:2]*q[:,2,None,None]-q[:,:2,None]*self.H[:,2,None,:2])/q[:,2,None,None]**2
        index=np.arange(self.N)
        for a in range(2):
            for b in range(2):
                rows.extend((2*index+a).tolist());cols.extend((self.C+2*self.track+b).tolist());values.extend(d[:,a,b].tolist())
        for side in range(2):
            valid=self.control[:,side]>0
            for axis in range(2):
                rows.extend((2*index[valid]+axis).tolist())
                cols.extend((2*(self.control[valid,side]-1)+axis).tolist())
                values.extend(self.weights[valid,side].tolist())
        return coo_matrix((values,(rows,cols)),shape=(2*self.N,len(x))).tocsr()

    def motion_errors(self,x):
        pred,_=self.project(x);result=[]
        for i in range(1,self.N):
            if self.track[i]==self.track[i-1] and self.pts[i]-self.pts[i-1]==256:
                result.append(np.linalg.norm((pred[i]-pred[i-1])-(self.uv[i]-self.uv[i-1])))
        return np.asarray(result)

    def segment(self,x,trackref):
        correction,_=self.unpack(x);delta=np.zeros((self.K,8))
        delta[:,2]=correction[:,0]*2/1920.;delta[:,5]=correction[:,1]*2/1080.
        return dict(basis='linear_within_source_supported_interval', native_pts=self.knots.tolist(),
            delta=delta.tolist(), source_tracklets=[trackref], subgroup='2D_image_translation',
            gauge='first_correction_zero_all_world_latents_free',
            source_supported_adjacent_pairs=(self.last-self.first)//256,
            actual_source_tracks=self.T, current_native_source_observations=self.N)


def fit_tracklets(state_file, generation_file, output, qualification_file=None):
    state=ScreenRecoveryState(state_file);generation=read_json(generation_file)
    if state.corrections: raise ValueError('Rebuild this correction from the inherited screen state and complete new source generation')
    if generation['source_video']['sha256']!=state.manifest['source_video_sha256']:
        raise ValueError('Tracklets belong to a different source video')
    trackfile=verified(generation['tracks_file'],Path(generation_file).parent)
    trackref=reference(trackfile)
    if qualification_file is None:
        bound=generation.get('static_surface_qualification')
        if bound is None:
            raise ValueError('Texture colour/nearby qualified curve is insufficient static material evidence; exact-track source review is required')
        qualification_file=verified(bound,Path(generation_file).parent)
    qualification_file=Path(qualification_file).resolve();qualification=read_json(qualification_file)
    if qualification.get('tracks_file_sha256')!=trackref['sha256']:
        raise ValueError('Static material qualification belongs to different actual tracklet pixels')
    reviews={}
    for review in qualification['tracks']:
        if review['track_id'] in reviews:raise ValueError('Duplicate static material review')
        if review['status'] not in ('accepted_static_ice','unknown','rejected'):
            raise ValueError('Invalid static material source status')
        if not review.get('reason'):raise ValueError('Source material review must retain the reason')
        if review['status']=='accepted_static_ice' and not review.get('evidence'):
            raise ValueError('Accepted static material track requires actual source-image evidence')
        for evidence in review.get('evidence',[]):
            if not isinstance(evidence.get('native_pts'),int) or not evidence.get('current_rgb_sha256'):
                raise ValueError('Static material evidence needs exact native RGB identity')
            if evidence.get('review_file'):verified(evidence['review_file'],qualification_file.parent)
        reviews[review['track_id']]=review
    qualref=reference(qualification_file)
    groups=defaultdict(list);reasons=Counter();all_rows=0;source_identities={}

    for row in read_jsonl(trackfile):
        all_rows+=1;sid=row['shot_id'];pts=row['native_pts']
        if (row.get('source_video_sha256')!=state.manifest['source_video_sha256'] or
            row.get('time_base')!=[1,12800] or row.get('pixels_from_model') is not False or
            row.get('fit_role')!='development' or row.get('surface_class')!='ice' or
            row.get('surface_instance_id')!='ice' or not row.get('current_rgb_sha256') or
            not row.get('stationary_material_evidence')):
            raise ValueError('Tracklet row lacks actual static-ice/source/development evidence')
        if state.shot_id(pts)!=sid: raise ValueError('Tracklet cut/clock mismatch')
        if sid=='s01': reasons['preserved_s01_complete_dependencies']+=1;continue
        if not state.seed.a['navigation_role_allowed'][state.seed.frame_index(pts,sid)]:
            raise ValueError('Protected source row reached screen optimization')
        if not np.isfinite(row['source_uv']).all(): raise ValueError('Nonfinite source UV')
        source_identities[(row['track_id'],pts)]=row['current_rgb_sha256']
        review=reviews.get(row['track_id'])
        if review is None or review['status']!='accepted_static_ice':
            reasons['static_surface_'+('unknown' if review is None else review['status'])]+=1
            continue
        groups[(sid,row['track_id'])].append(row)
    for k,review in reviews.items():
        for evidence in review.get('evidence',[]):
            if source_identities.get((k,evidence['native_pts']))!=evidence['current_rgb_sha256']:
                raise ValueError('Static material review evidence is not the exact source track exposure')

    eligible=defaultdict(dict);source_pairs=defaultdict(set);track_rejections=[]
    for (sid,k),rows in groups.items():
        rows.sort(key=lambda r:r['native_pts']);p=np.asarray([r['native_pts'] for r in rows])
        if len(rows)<2:
            reasons['single_exposure_no_motion']+=len(rows);continue
        if np.any(np.diff(p)!=256):
            raise ValueError('Track has a source/role gap; split it at acquisition instead of interpolating source UV')
        eligible[sid][k]=rows
        source_pairs[sid].update(p[:-1].tolist())
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('New immutable screen-tracklet candidate required')
    out.mkdir(parents=True)
    reports=[];corrections={};latent_rows=[];consumed=0;accepted=0
    for sid in state.seed.meta['per_shot']:
        segments=[]
        for first,last in _intervals(source_pairs[sid]):
            local={k:rows for k,rows in eligible[sid].items() if first<=rows[0]['native_pts'] and rows[-1]['native_pts']<=last}
            interval=dict(shot_id=sid,first_pts=first,last_pts=last,source_rows=sum(len(x) for x in local.values()),source_tracks=len(local))
            try:
                problem=TrackletProblem(state,sid,local,first,last)
                inverse_initial=problem.residual(problem.x0)
                def baseline_residual(x):return problem.residual(np.r_[np.zeros(problem.C),x])
                def baseline_jacobian(x):return problem.jacobian(np.r_[np.zeros(problem.C),x])[:,problem.C:]
                baseline=least_squares(baseline_residual,problem.x0[problem.C:],jac=baseline_jacobian,
                    method='trf',tr_solver='lsmr',x_scale='jac')
                if not baseline.success:raise ValueError('Parent-state shared latent baseline did not converge')
                base_x=np.r_[np.zeros(problem.C),baseline.x]
                initial=problem.residual(base_x); initial_motion=problem.motion_errors(base_x)
                result=least_squares(problem.residual,base_x,jac=problem.jacobian,method='trf',tr_solver='lsmr',x_scale='jac')
                final=problem.residual(result.x);final_motion=problem.motion_errors(result.x)
                if (not result.success or np.dot(final,final)>=np.dot(initial,initial) or
                    np.dot(final_motion,final_motion)>np.dot(initial_motion,initial_motion)):
                    interval.update(status='retained_parent_no_joint_source_and_motion_improvement',solver_message=result.message,
                        baseline_shared_latent_source_error=_stats(np.linalg.norm(initial.reshape(-1,2),axis=1)),
                        trial_source_error=_stats(np.linalg.norm(final.reshape(-1,2),axis=1)),
                        baseline_motion_error=_stats(initial_motion),trial_motion_error=_stats(final_motion))
                    reports.append(interval);reasons[interval['status']]+=problem.N;continue
                segment=problem.segment(result.x,trackref);segment['static_surface_qualification']=qualref;segment['source_tracklets'].append(qualref);segments.append(segment);consumed+=problem.N;accepted+=1
                correction,latent=problem.unpack(result.x)
                deltaXY=np.linalg.norm(latent-problem.latent0,axis=1)
                interval.update(status='source_supported_translation_candidate',
                    inverse_navigation_initial_source_error=_stats(np.linalg.norm(inverse_initial.reshape(-1,2),axis=1)),
                    initial_source_error=_stats(np.linalg.norm(initial.reshape(-1,2),axis=1)),
                    baseline_world_latents_fitted_to_same_all_source_rows=True,
                    final_source_error=_stats(np.linalg.norm(final.reshape(-1,2),axis=1)),
                    initial_source_motion_error=_stats(initial_motion), final_source_motion_error=_stats(final_motion),
                    all_initial_source_rows_in_final_residual=len(final)==len(initial),
                    initial_residual_state_sha256=state.sha256, controls=len(problem.knots),
                    first_observation_world_XY_fixed=False, shared_world_latents=problem.T,
                    latent_change_L=_stats(deltaXY,unit='L'), solver_evaluations=result.nfev,
                    solver_optimality=float(result.optimality), terminal_message=result.message,
                    support_edge_correction_jump_px=float(np.linalg.norm(correction[-1])),
                    engineering_objective_without_calibrated_covariance=True)
                for k,X,X0 in zip(problem.ids,latent,problem.latent0):
                    latent_rows.append(dict(track_id=k,shot_id=sid,reference_xy=X.tolist(),initial_reference_xy=X0.tolist(),
                        world_chart_sha256=state.world_chart_sha256,world_z=0.,
                        status='optimized_shared_unknown_ice_point',first_inverse_is_initialization_only=True))
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                interval.update(status='candidate_refused_all_interval_rows_remain_unresolved',reason=str(exc))
                reasons[interval['status']]+=interval['source_rows']
            reports.append(interval)
            print(json.dumps(interval,separators=(',',':')),flush=True)
        if segments:corrections[sid]=dict(segments=segments,outside_support='inherit_seed_and_report_unknown')
    manifest=dict(state.manifest);manifest['parent_screen_state']=reference(state_file)
    manifest['corrections']=corrections;manifest['source_generations']=[reference(generation_file),qualref]
    manifest['initial_state']='exact_parent_screen_state_zero_source_correction'
    manifest['implementation']=dict(manifest.get('implementation',{}),
        screen_recovery_state=reference(Path(__file__).with_name('screen_recovery_state.py')),
        screen_tracklet_registration=reference(__file__))
    write_json(out/'state.json',manifest);write_json(out/'latent_ice_points.json',latent_rows)
    candidate=ScreenRecoveryState(out/'state.json')
    # Exact serialization agreement on every actual source row in accepted
    # intervals, using the fitted shared latent X rather than a virtual grid.
    saved_max=0.;inverse={r['track_id']:r for r in latent_rows}
    for sid,control in corrections.items():
        by_frame=defaultdict(list)
        for segment in control['segments']:
            first,last=segment['native_pts'][0],segment['native_pts'][-1]
            for k,rows in eligible[sid].items():
                if k not in inverse or not(first<=rows[0]['native_pts'] and rows[-1]['native_pts']<=last):continue
                for row in rows:by_frame[row['native_pts']].append(inverse[k]['reference_xy'])
        for pts,points in by_frame.items():
            X=np.asarray(points,float)
            base,_=state.project_ice(pts,sid,X,world_chart_sha256=state.world_chart_sha256)
            actual,_=candidate.project_ice(pts,sid,X,world_chart_sha256=state.world_chart_sha256)
            delta=candidate.delta(pts,sid);expected=base+[delta[2]*1920/2,delta[5]*1080/2]
            saved_max=max(saved_max,float(np.max(np.abs(actual-expected))))
    if saved_max>1e-8:raise ValueError('Saved provider does not reproduce optimized translation state')
    coverage={sid:dict(possible_pairs=state.seed.meta['per_shot'][sid]-1,
        measured_source_pairs=len(source_pairs[sid]),
        corrected_source_pairs=sum(s['source_supported_adjacent_pairs'] for s in corrections.get(sid,{}).get('segments',[])),
        unknown_motion_pairs=state.seed.meta['per_shot'][sid]-1-sum(s['source_supported_adjacent_pairs'] for s in corrections.get(sid,{}).get('segments',[])))
        for sid in state.seed.meta['per_shot']}
    report=dict(schema='rinkfit.source_tracklet_screen_registration.v1',parent_state=reference(state_file),
        candidate_state=reference(out/'state.json'),source_generation=reference(generation_file),
        source_tracklets=trackref,static_surface_qualification=qualref,only_qualified_static_ice_tracks_fitted=True,all_source_rows=all_rows,actually_optimized_rows=consumed,
        accepted_source_intervals=accepted,by_shot=coverage,intervals=reports,not_optimized_reasons=dict(reasons),
        serialized_provider_translation_max_abs_difference_px=saved_max,
        source_UV_changed=False,world_latents_shared_across_all_track_exposures=True,
        source_inverse_used_only_as_initialization=True,physical_checkpoint_initialized=False,
        source_accuracy_or_stability_pass_claimed=False,
        limitations=['Only the translation subgroup was source-optimized; rotation/zoom residuals remain visible.',
            'Unsupported/protected intervals retain the original seed and unknown source-motion status.',
            'Support-edge correction jumps are explicitly reported; this is not a whole-programme stable physical candidate.',
            'This is an engineering source-texture objective without calibrated pixel covariance or independent holdout.'])
    write_json(out/'registration.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',required=True)
    p.add_argument('--generation',required=True);p.add_argument('--out',required=True);p.add_argument('--qualification');a=p.parse_args()
    print(json.dumps(fit_tracklets(a.state,a.generation,a.out,a.qualification),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
