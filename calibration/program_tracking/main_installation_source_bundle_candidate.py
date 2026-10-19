"""One physical centre for all four classified main-camera cuts.

Every relative factor uses observed source pixels within one cut. Reparameterized
old cameras initialize the numerical branch only; their projections never become
source observations. The candidate keeps square pixels, centred principal point,
zero skew and zero Brown as explicit, unverified lens assumptions.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from main7_source_bundle_candidate import Bundle,solve
from main7_source_camera_candidate import B,TMP,arrays,sha,q7,source_rows,errors
from scene_source_geometry_binding import rebind_scene_geometry


def prepare(base,previous,reviewed_bridge_directory=None,candidate_scene_path=None,additional_fit_sources=None,key_seed_file=None):
    original_scene=json.loads((B/'candidate_scene.json').read_text(encoding='utf8'))
    scene=original_scene if candidate_scene_path is None else json.loads(candidate_scene_path.read_text(encoding='utf8'))
    seven=base/'render_markings/main7_geometry';four=base/'pelvis/main4_geometry'
    seven_seed=previous/'candidate_cameras.npz'
    if not seven_seed.exists():seven_seed=previous/'main7_candidate_cameras.npz'
    streams={7:arrays(seven_seed),4:arrays(previous/'main4_candidate_cameras.npz')}
    for n in (1,9):streams[n]=arrays(base/f'main{n}_geometry'/f'main{n}_B_reparameterized_numerical_seed.npz')
    canonical={};initial={};cut_of={};bindings=[];rows=[];pairs=[]
    for n,s in streams.items():
        if np.any(s['camera25'][:,11:]!=0) or np.any(s['camera25'][:,6]!=s['camera25'][:,7]):
            raise ValueError(f'Main {n} seed differs from the declared pinhole candidate')
        aliases={int(p):int(r) for p,r in zip(s['pts'],s['exposure_representative_pts'])}
        canonical.update(aliases);cut_of.update({p:n for p in aliases.values()})
        initial.update({aliases[int(p)]:c for p,c in zip(s['pts'],s['camera25'])})
        folder=seven if n==7 else four if n==4 else base/f'main{n}_geometry'
        motion=folder/('source_motion_matches.npz' if n==7 else f'main{n}_source_motion_matches.npz')
        if reviewed_bridge_directory is not None and n in (1,4,9):motion=reviewed_bridge_directory/f'main{n}_source_motion_matches_reviewed.npz'
        m=arrays(motion);bindings.append(dict(cut=n,path=str(motion),sha256=sha(motion),role='within_cut_observed_source_ray_pairs'))
        for i,(pa,pb) in enumerate(m['pair_pts']):
            pa,pb=int(pa),int(pb)
            if pa not in aliases or pb not in aliases or aliases[pa]!=pa or aliases[pb]!=pb or pa==pb:
                raise ValueError(f'Main {n} source pair is not two actual canonical exposures in its cut')
            lo,hi=map(int,m['offsets'][i:i+2]);pairs.append((pa,pb,m['source_uv'][lo:hi],m['target_uv'][lo:hi]))
        if n==7:
            added,binding=source_rows(scene);bindings.extend(binding);rows.extend(added)
        else:
            # Reviewed main 1 / 9 master packets replace their intentionally
            # retained first-pass packets. Tail Lane provenance is already here.
            name=f'main{n}_actual_source_rows_reviewed.json' if n in (1,9) else 'main4_bundle_absolute_source_rows.json'
            file=folder/name;added=json.loads(file.read_text(encoding='utf8'))
            for row in added:
                if row['pts'] not in aliases:raise ValueError(f'Main {n} absolute row crosses its classified cut')
                for key in ('uv','world','sigma'):row[key]=np.asarray(row[key],float)
            if candidate_scene_path is not None:added=[rebind_scene_geometry(r,original_scene,scene)for r in added]
            rows.extend(added);bindings.append(dict(cut=n,path=str(file),sha256=sha(file),role='actual_absolute_source_pixels'))
    # Keep successful source-resected tail keys as numerical seeds, and identify
    # aliases before replacing them; there are no extra duplicate key variables.
    keyfile=previous/'shared_main_key_cameras.npz'
    if keyfile.exists():
        key=arrays(keyfile);initial.update({canonical[int(p)]:c for p,c in zip(key['pts'],key['camera25']) if int(p) in canonical})
    # Warm every full-cut pose from the preceding actual-source bundle. Old B
    # reparameterizations are numerical seeds only for the first all-cut run.
    allfile=previous/'all_main_candidate_cameras.npz'
    if allfile.exists():
        latest=arrays(allfile)
        if set(map(int,latest['pts']))!=set(canonical):raise ValueError('All-main warm seed does not match the native classified cuts')
        initial.update({canonical[int(p)]:c for p,c in zip(latest['pts'],latest['camera25'])})
    external_centres=[]
    if key_seed_file is not None:
        seed=arrays(key_seed_file);used=[];outside=[]
        for p,c in zip(seed['pts'],seed['camera25']):
            p=int(p)
            if p not in canonical:
                outside.append(p);continue
            if np.any(c[11:]!=0) or c[6]!=c[7] or not np.array_equal(c[8:11],[960.,540.,0.]):
                raise ValueError('Main external key seed changes declared lens assumptions')
            initial[canonical[p]]=c;external_centres.append(c[:3]);used.append(p)
        if not used:raise ValueError('External key seed has no native main-cut exposures')
        bindings.append(dict(path=str(key_seed_file),sha256=sha(key_seed_file),
            role='shared_all_installation_geometry_key_numeric_seed',used_main_pts=used,
            outside_classified_main_cuts_not_used=outside,not_source_observations=True))
    stamps=sorted(set(canonical.values()));observations={p:[] for p in stamps};seen=set();suppressed=[]
    for row in sorted(rows,key=lambda r:(r['pts'],r.get('source_native_pts',r['pts']))):
        if row['split']!='fit':continue
        p=canonical[int(row['pts'])];identity=(p,row['id'])
        if identity in seen:
            suppressed.append(dict(pts=row['pts'],canonical_pts=p,id=row['id'],origin=row['origin']));continue
        seen.add(identity);observations[p].append(row)
    # Newly reviewed actual ice rows were explicitly promoted by the parent.
    # Preserve original splits and source identities, including additional
    # contour portions of an already sampled curve/exposure; do not call these
    # independent holdouts after fitting them.
    for file in additional_fit_sources or []:
        added=json.loads(file.read_text(encoding='utf8'))
        for row in added:
            for key in ('uv','world','sigma','normal','normal_interval'):
                if key in row:row[key]=np.asarray(row[key],float)
            row['original_source_split']=row.get('original_source_split',row['split']);row['split']='fit'
            row['fit_authority']='parent explicitly promoted actual ice/Lane/ad development source checks to fitting'
            if candidate_scene_path is not None:row=rebind_scene_geometry(row,original_scene,scene)
            p=canonical[row['pts']];row['same_exposure_curve_extension']=(p,row['id'])in seen
            observations[p].append(row);rows.append(row);seen.add((p,row['id']))
        bindings.append(dict(path=str(file),sha256=sha(file),role='promoted_actual_source_review_no_longer_holdout'))
    if candidate_scene_path is not None:bindings.append(dict(path=str(candidate_scene_path),sha256=sha(candidate_scene_path),role='versioned_shared_geometry_candidate'))
    # The main 7 source packet samples absolute keys. Its original actual board
    # detector evidence supplies every remaining exposure, with the same finite
    # closed physical perimeter that fixed the short-end branch in v2.
    edgefile=TMP/'programme_native_evidence_v13/driver_inputs/source_edges/shot07.npz';edge=arrays(edgefile)
    far=np.asarray(scene['ice_boundary']['points_xyz'],float);top=np.asarray(scene['board_surface']['top_edge_xyz'],float)
    if not np.array_equal(far[0],far[-1]):far=np.vstack([far,far[0]])
    if not np.array_equal(top[0],top[-1]):top=np.vstack([top,top[0]])
    dense=[]
    for i,native in enumerate(edge['pts']):
        p=int(native)
        if canonical[p]!=p:continue
        lo,hi=map(int,edge['offsets'][i:i+2]);uv=edge['observed_uv'][lo:hi];kind=edge['edge_id'][lo:hi]
        for k,name,world in ((0,'far_top',top),(1,'far_contact',far)):
            if (p,name) in seen or (p,'board_top' if k==0 else 'ice_board_contact') in seen:continue
            selected=uv[(kind==k)&(uv[:,1]<650)&(uv[:,0]>40)&(uv[:,0]<1880)][::6].copy()
            if len(selected)<3:continue
            row=dict(pts=p,id=name,split='fit',uv=selected,world=world,sigma=np.full(len(selected),4.),origin=str(edgefile))
            observations[p].append(row);dense.append(row);seen.add((p,name))
    bindings.append(dict(cut=7,path=str(edgefile),sha256=sha(edgefile),role='actual_full_perimeter_native_board_pixels'))
    centre=np.median(external_centres,axis=0) if external_centres else streams[7]['camera25'][0,:3]
    q=np.r_[centre/100,np.concatenate([q7(initial[p])[3:] for p in stamps])]
    return scene,streams,canonical,cut_of,rows,dense,bindings,suppressed,Bundle(stamps,observations,pairs),q


def run(base,previous,output,reviewed_bridge_directory=None,candidate_scene_path=None,additional_fit_sources=None,key_seed_file=None,prepare_only=False,fixed_centre=False):
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    scene,streams,canonical,cuts,rows,dense,bindings,suppressed,bundle,q0=prepare(base,previous,reviewed_bridge_directory,candidate_scene_path,additional_fit_sources,key_seed_file)
    if fixed_centre:
        if key_seed_file is None:raise ValueError('Fixed joint-calibration centre requires the explicit shared key seed')
        original_evaluate=bundle.evaluate
        def fixed_evaluate(q,jacobian=True):
            result=original_evaluate(q,jacobian)
            if jacobian:
                # Keep the fixed CSR pattern and C slots for the shared solver
                # contract. These calibration variables are not re-estimated
                # by the much denser per-programme wall sample distribution.
                result[1].data[result[1].indices<3]=0.
            return result
        bundle.evaluate=fixed_evaluate
    manifest=dict(schema='rinkspace.main-installation-source-bundle-input.v1',candidate_accepted=False,
        cut_counts={str(n):dict(native=len(s['pts']),actual_exposures=len(set(s['exposure_representative_pts']))) for n,s in streams.items()},
        canonical_exposures=len(bundle.stamps),matches=len(bundle.a),residual_rows=bundle.residual_rows,unknowns=len(q0),
        source_bindings=bindings,suppressed_same_physical_curve_exposure=suppressed,
        no_cross_cut_source_pairs=True,no_model_projected_pixels_as_observations=True,no_additional_tail_lane_duplicate=True,
        sparse_pattern='fixed sorted CSR; cached per-exposure rotations; 65536-match Jacobian chunks',
        lens_assumptions=['square_pixels','principal960_540','zero_skew','zero_Brown'],
        installation_centre_mode='fixed_from_shared_all_installation_source_calibration' if fixed_centre else 'estimated_in_programme_bundle',
        installation_centre_initial=(q0[:3]*100.).tolist())
    (output/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    if candidate_scene_path is not None:(output/'candidate_scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(stage='prepared',canonical_exposures=len(bundle.stamps),matches=len(bundle.a),seconds=time.perf_counter()-started)),flush=True)
    if prepare_only:
        (output/'prepared_without_solve.json').write_text(json.dumps(dict(
            source_observations_unchanged=True,shared_C_initial=(q0[:3]*100.).tolist(),
            prepare_only=True,accuracy_accepted=False),indent=2),encoding='utf8')
        return manifest
    q,history,seconds=solve(bundle,q0,output);cameras={p:bundle.camera(q,p) for p in bundle.stamps}
    if fixed_centre and not np.array_equal(q[:3],q0[:3]):raise ValueError('Frozen shared installation centre changed')
    for n,s in streams.items():
        np.savez_compressed(output/f'main{n}_candidate_cameras.npz',pts=s['pts'],
            camera25=np.array([cameras[canonical[int(p)]] for p in s['pts']]),exposure_representative_pts=s['exposure_representative_pts'],
            status=np.full(len(s['pts']),'global_actual_source_candidate_unvalidated'))
    allpts=np.sort(np.concatenate([s['pts'] for s in streams.values()]));np.savez_compressed(output/'all_main_candidate_cameras.npz',
        pts=allpts,camera25=np.array([cameras[canonical[int(p)]] for p in allpts]),exposure_representative_pts=np.array([canonical[int(p)] for p in allpts]))
    report=dict(schema='rinkspace.main-installation-source-bundle.v1',accuracy_accepted=False,shared_C=(q[:3]*100).tolist(),
        conditional_scene_units=True,seconds=seconds,preparation_seconds=seconds and time.perf_counter()-started-seconds,
        input_manifest=manifest,history=history,source_absolute_errors=[dict(native_pts=r['pts'],cut=cuts[canonical[r['pts']]],
        id=r['id'],split=r['split'],origin=r['origin'],original_source_split=r.get('original_source_split'),
        errors=errors(cameras[canonical[r['pts']]],[r])) for r in rows])
    # Independently sourced main 4 Lane rows remain outside all fitting factors.
    lane=base/'pelvis/main4_geometry/main4_native_lane_holdout_rows.json'
    if lane.exists():
        added=json.loads(lane.read_text(encoding='utf8'))
        for r in added:
            for k in ('world','uv','sigma'):r[k]=np.asarray(r[k],float)
        key='main4_promoted_source_fit_review'if lane in(additional_fit_sources or[])else 'independent_main4_lane_holdout'
        if candidate_scene_path is not None:added=[rebind_scene_geometry(r,json.loads((B/'candidate_scene.json').read_text(encoding='utf8')),scene)for r in added]
        report[key]=dict(path=str(lane),sha256=sha(lane),not_independent=key!='independent_main4_lane_holdout',
            errors=[dict(pts=r['pts'],id=r['id'],errors=errors(cameras[canonical[r['pts']]],[r])) for r in added])
    file=output/'report.json';file.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'main_camera_centre.json').write_text(json.dumps(dict(C=report['shared_C'],conditional_scene_units=True,accuracy_accepted=False,
        scene_path=str(candidate_scene_path or B/'candidate_scene.json'),scene_sha256=sha(candidate_scene_path or B/'candidate_scene.json'),report=str(file),report_sha256=sha(file)),indent=2),encoding='utf8')
    print(json.dumps(dict(stage='complete',shared_C=report['shared_C'],seconds=seconds,accuracy_accepted=False)),flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reviewed-bridge-directory',type=Path)
    ap.add_argument('--candidate-scene',type=Path);ap.add_argument('--additional-fit-source',type=Path,action='append')
    ap.add_argument('--key-seed',type=Path);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--fixed-centre',action='store_true')
    a=ap.parse_args();run(a.base,a.previous,a.output,a.reviewed_bridge_directory,a.candidate_scene,a.additional_fit_source,a.key_seed,a.prepare_only,a.fixed_centre)
