"""Source-scoped standing/quiet/hand evidence on the complete current state."""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,csr_matrix,diags,hstack,vstack
from .opening_motion import surface_contact_vectors
from .skate_physics import fk_state,batch_features,body_constraint_jacobian,NORMALIZATION
from .spin_grasp_factors import tolerant_support_response
from .weighted_blocks import block


def standing_contact_rows(features,quiet,mode='anonymous_minimum'):
    """Select actual eligible steel minima without substituting one foot for the other."""
    quiet=np.asarray(quiet,int);eligible=np.flatnonzero(features['eligible'])
    if not len(eligible):raise ValueError('Standing has no eligible physical steel features')
    heights=features['height'][quiet][:,:,eligible]
    chosen=eligible[np.argmin(heights,axis=2)]
    each=features['height'][quiet[:,None],np.arange(2)[None,:],chosen]
    if mode=='anonymous_minimum':
        rows=quiet;feet=np.argmin(each,axis=1);selected=chosen[np.arange(len(quiet)),feet]
    elif mode=='both_feet':
        rows=np.repeat(quiet,2);feet=np.tile(np.arange(2),len(quiet));selected=chosen.ravel()
    else:raise ValueError('Unknown source standing contact mode')
    return dict(rows=rows,feet=feet,features=selected,
        heights=features['height'][rows,feet,selected],points=features['local_point'][rows,feet,selected],
        height_by_foot=each,minimum_height=each.min(axis=1))


class ProgrammeOpeningFactors:
    def __init__(self,problem,config):
        self.p=problem;self.ref=problem.reference;self.cfg=config
        pts=self.ref.p['pts'];self.indices=np.flatnonzero((pts>=config['start_pts'])&(pts<=config['hand_release_end_pts']))
        if not len(self.indices):raise ValueError('Opening evidence has no source-clock overlap')
        self.times=self.ref.times[self.indices];self.base=self.ref.at(self.times)
        self.quiet=np.flatnonzero(pts[self.indices]<=config['quiet_end_pts'])
        self.joints=np.asarray(config['quiet_native_joints'],int)
        self.width=6+3*len(self.joints)+9
        self.q_seed=self.base['Q'][0];self.local_seed=self.base['local_R'][0,self.joints-1]
        self.stance=None;reference=config.get('conditional_stance_reference')
        if reference is not None:
            from .rebuild_prepare import digest
            if digest(reference['path'])!=reference['sha256']:raise ValueError('Conditional opening pose reference changed')
            with np.load(reference['path'],allow_pickle=False)as packet:
                clock=packet['source_pts']if 'source_pts'in packet else packet['pts']
                found=np.flatnonzero(clock==reference['source_pts'])
                if len(found)!=1:raise ValueError('Conditional stance reference has no unique source PTS')
                local=packet['local_R_mid'][found[0]]
            if local.shape!=(21,3,3):raise ValueError('Conditional stance must use native22 local rotations excluding root')
            np.testing.assert_allclose(local.transpose(0,2,1)@local,np.broadcast_to(np.eye(3),local.shape),atol=1e-6,rtol=0)
            self.stance_joints=np.asarray(reference['native_joints'],int)
            self.stance=local[self.stance_joints-1].copy()
            release_end=int(reference['release_end_pts'])
            if release_end<=config['quiet_end_pts']:raise ValueError('Stance prior needs a finite source release interval')
            u=np.clip((pts[self.indices]-config['quiet_end_pts'])/(release_end-config['quiet_end_pts']),0,1)
            weight=(1-u*u*(3-2*u))*np.sqrt(config['source_measure_s']*reference['weight'])/np.deg2rad(reference['scale_deg'])
            self.stance_row_scale=np.repeat(weight,3*len(self.stance_joints))
        surface=self.ref.binding.surface;rest=(surface.shaped-surface.body.J[0])*surface.body.scale*self.ref.binding.ratio
        torso=surface.weights[:,[0,1,2,3,6,9]].sum(axis=1)>.8
        self.waist=np.flatnonzero(torso&(rest[:,0]<-.05)&(rest[:,1]>-.12)&(rest[:,1]<.28))
        hand=np.flatnonzero(surface.weights[:,[21,*range(40,55)]].sum(axis=1)>.8)
        centres=.45*surface.body.J[21]+.55*surface.body.J[[40,43,49]]
        _,near=cKDTree(surface.shaped[hand]).query(centres,k=8);self.palms=hand[near]
        if not len(self.waist):raise ValueError('No right waist material patch')
        rr=[];cc=[];vv=[];row=0;self.rows={};measure=float(config['source_measure_s'])
        scales=np.r_[np.full(3,config['quiet_position_scale_m']),np.full(self.width-12,np.deg2rad(config['quiet_angle_scale_deg']))]
        for i in self.quiet[1:]:
            for j,sigma in enumerate(scales):
                rr.extend((row,row));cc.extend((i*self.width+j,self.quiet[0]*self.width+j));vv.extend((1/sigma,-1/sigma));row+=1
        self.rows['opening_source_quiet_body']=slice(0,row);first=row
        u=np.clip((pts[self.indices]-config['hand_release_start_pts'])/(config['hand_release_end_pts']-config['hand_release_start_pts']),0,1)
        weights=1-u*u*(3-2*u)
        for i,w in enumerate(weights):
            if w<=0:continue
            for j in range(9):
                rr.append(row);cc.append(i*self.width+self.width-9+j);vv.append(w/(config['hand_relation_scale_m']*np.sqrt(3)));row+=1
        self.rows['opening_source_hand_waist']=slice(first,row)
        self.operator=coo_matrix((np.asarray(vv)*np.sqrt(measure/2),(rr,cc)),shape=(row,len(self.times)*self.width)).tocsr()
    def features(self,s):
        q=Rotation.from_matrix(self.q_seed.T@s['Q']).as_rotvec()
        loc=s['local_R'][:,self.joints-1]
        a=Rotation.from_matrix((self.local_seed.transpose(0,2,1)[None]@loc).reshape(-1,3,3)).as_rotvec().reshape(len(q),-1)
        ids=np.r_[self.waist,self.palms.ravel()];kw={}if 'hand_local_R'not in s else dict(hand_local_R=s['hand_local_R'])
        skin=self.ref.binding.surface.sample_vertices(s['native'],s['frames'],ids,double_precision=True,**kw)*self.ref.binding.ratio
        hand=surface_contact_vectors(skin[:,len(self.waist):].reshape(len(q),3,8,3).mean(2),skin[:,:len(self.waist)]).reshape(len(q),9)
        return np.c_[s['P']*self.ref.binding.ratio,q,a,hand]
    def stance_vectors(self,s):
        local=s['local_R'][:,self.stance_joints-1]
        return Rotation.from_matrix((self.stance.transpose(0,2,1)[None]@local).reshape(-1,3,3)).as_rotvec().reshape(len(local),-1)
    def evaluate(self,x,jacobian=False):
        state=self.p.body.state;d=state.increments(x,self.times);s=fk_state(self.ref,self.base,d);features=self.features(s)
        r=np.asarray(self.operator@features.ravel());J=None
        if jacobian:
            derivatives=[];stance_derivatives=[];eps=1e-5
            for a in state.active:
                plus=d.copy();minus=d.copy();plus[:,a]+=eps;minus[:,a]-=eps
                sp=fk_state(self.ref,self.base,plus);sm=fk_state(self.ref,self.base,minus)
                derivatives.append((self.features(sp)-self.features(sm))/(2*eps))
                if self.stance is not None:stance_derivatives.append((self.stance_vectors(sp)-self.stance_vectors(sm))/(2*eps))
            J=self.operator@d.chain(np.stack(derivatives,-1))
            J=hstack((csr_matrix((len(r),17)),J),format='csr')
        named=[block(k,r[sl],J[sl]if jacobian else None,dict(source_pts=self.ref.p['pts'][self.indices],evidence=self.cfg['evidence']))for k,sl in self.rows.items()]
        if self.stance is not None:
            sr=self.stance_vectors(s).ravel()*self.stance_row_scale;SJ=None
            if jacobian:
                SJ=diags(self.stance_row_scale)@d.chain(np.stack(stance_derivatives,-1))
                SJ=hstack((csr_matrix((len(sr),17)),SJ),format='csr')
            named.append(block('opening_source_conditional_stance',sr,SJ,dict(reference=self.cfg['conditional_stance_reference'],
                old_world_P_Q_not_used=True,finite_pose_reference_not_calibration_truth=True)))
        if not self.ref.ice_plane_known:raise ValueError('Standing evidence requires known finite ice model')
        f=batch_features(self.ref.binding,s['R'],s['O'],s)
        mode=self.cfg.get('standing_contact_mode','anonymous_minimum')
        contact=standing_contact_rows(f,self.quiet,mode)
        ii,foot,height,point=(contact[key]for key in ('rows','feet','heights','points'))
        response,slope=tolerant_support_response(height,self.cfg['standing_curvature_m'])
        weight=np.sqrt(self.cfg['source_measure_s']*self.cfg['standing_weight'])/self.cfg['standing_scale_m'];rs=response*weight;Js=None
        if jacobian:
            e=dict(rows=[(int(i),int(fo),q,1.)for i,fo,q in zip(ii,foot,point)],row_upper_vertex=[-1]*len(ii),times=self.times)
            D=NORMALIZATION*body_constraint_jacobian(self.ref,s,d,state.active,state.basis(self.times),e)
            Js=hstack((csr_matrix((len(rs),17)),diags(weight*slope)@D),format='csr')
        name='opening_source_both_feet_standing'if mode=='both_feet'else'opening_source_anonymous_standing'
        named.append(block(name,rs,Js,dict(source_pts=self.ref.p['pts'][self.indices[ii]],unit='m',
            standing_contact_mode=mode,model_feet=foot.tolist(),foot_identity_unknown=(mode=='anonymous_minimum'),contact_material_points_unknown=True,
            user_confirmation=self.cfg.get('user_confirmation'),evidence=self.cfg['evidence'])))
        parts={b.name:float(b.residual@b.residual/2)for b in named}
        hand_distance=np.linalg.norm(features[:,-9:].reshape(-1,3,3),axis=-1)
        held=self.ref.p['pts'][self.indices]<=self.cfg['hand_release_start_pts']
        return dict(residual=np.concatenate([b.residual for b in named]),jac=vstack([b.jacobian for b in named],format='csr')if jacobian else None,parts=parts,weighted_blocks=named,
            metrics=dict(opening_source_standing=dict(source_pts=self.ref.p['pts'][self.indices[self.quiet]].tolist(),minimum_steel_height_m=contact['minimum_height'].tolist(),
                steel_height_by_foot_m=contact['height_by_foot'].tolist(),standing_contact_mode=mode,
                height_reference='conditional model ice plane; combined scene/camera/body/skate error is uncalibrated',
                calibrated_height_accuracy_m=None,height_residual_is_physical_contact_verdict=False,
                active_envelope_foot_not_load_truth=np.argmin(contact['height_by_foot'],axis=1).tolist(),
                quiet_P_range_m=float(np.linalg.norm(features[self.quiet,:3]-features[self.quiet[0],:3],axis=1).max()),
                hand_waist_max_distance_m=float(hand_distance.max()),
                hand_waist_source_pts=self.ref.p['pts'][self.indices].tolist(),hand_waist_patch_distances_m=hand_distance.tolist(),
                held_hand_waist_max_distance_m=float(hand_distance[held].max())if held.any()else None,
                release_hand_waist_max_distance_m=float(hand_distance[~held].max())if (~held).any()else None,
                conditional_stance_reference_deviation_deg=(None if self.stance is None else dict(
                    native_joints=self.stance_joints.tolist(),
                    quiet_max_by_joint=np.degrees(np.linalg.norm(self.stance_vectors(s)[self.quiet].reshape(len(self.quiet),-1,3),axis=-1).max(axis=0)).tolist(),
                    old_world_P_Q_not_used=True)),
                old_source_pose_is_conditional=True)))
