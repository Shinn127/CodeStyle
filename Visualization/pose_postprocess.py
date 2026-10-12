"""Causal, render-only pose filtering and ground-aware two-bone foot IK."""
import numpy as np
from Utils import quat


def nlerp(a,b,weight):
    b=np.where((a*b).sum(-1,keepdims=True)<0,-b,b)
    return quat.normalize(a+(b-a)*np.asarray(weight)[...,None]).astype(np.float32)


def rotation_between(a,b):
    a=a/max(float(np.linalg.norm(a)),1e-8);b=b/max(float(np.linalg.norm(b)),1e-8)
    dot=float(np.clip(a@b,-1,1))
    if dot<-.99999:
        axis=np.cross(a,np.eye(3,dtype=np.float32)[np.argmin(np.abs(a))])
        return np.concatenate(([0.],axis/max(float(np.linalg.norm(axis)),1e-8))).astype(np.float32)
    return quat.normalize(np.concatenate(([1+dot],np.cross(a,b)))).astype(np.float32)


def two_bone_target(hip,knee,ankle,target,*,pole=None):
    """Keep both lengths and the current knee plane; clamp unreachable targets."""
    upper=knee-hip;lower=ankle-knee
    a=float(np.linalg.norm(upper));b=float(np.linalg.norm(lower))
    if min(a,b)<1e-6:raise ValueError('Degenerate leg lengths')
    direction=target-hip;distance=float(np.linalg.norm(direction))
    direction=direction/max(distance,1e-8) if distance>1e-8 else upper/a
    distance=float(np.clip(distance,abs(a-b)+1e-5,a+b-1e-5))
    bend=upper-direction*float(upper@direction)
    if np.linalg.norm(bend)<1e-5:
        axis=np.array([0,0,1],np.float32) if pole is None else np.asarray(pole,np.float32)
        bend=axis-direction*float(axis@direction)
        if np.linalg.norm(bend)<1e-5:
            axis=np.eye(3,dtype=np.float32)[np.argmin(np.abs(direction))]
            bend=axis-direction*float(axis@direction)
    bend/=max(float(np.linalg.norm(bend)),1e-8)
    along=(a*a-b*b+distance*distance)/(2*distance)
    height=np.sqrt(max(a*a-along*along,0.))
    return hip+direction*along+bend*height,hip+direction*distance


class PosePostprocessor:
    def __init__(self,names,parents,rest_offsets,*,cutoff=6.,foot_ik=True):
        self.names=list(names);self.parents=np.asarray(parents,np.int64)
        self.offsets=np.asarray(rest_offsets,np.float32).copy()
        if self.parents[0]!=-1 or self.offsets.shape!=(len(names),3):raise ValueError('Invalid render skeleton')
        if not np.isfinite(cutoff) or cutoff<0:raise ValueError('Invalid smoothing cutoff')
        depth=np.zeros(len(names),np.int64)
        for i in range(1,len(names)):
            if not 0<=self.parents[i]<i:raise ValueError('Skeleton must be in parent-first order')
            depth[i]=depth[self.parents[i]]+1
        self.levels=[np.flatnonzero(depth==d) for d in range(1,int(depth.max())+1)]
        self.cutoff=float(cutoff);self.foot_ik=bool(foot_ik)
        self.legs=[]
        for side in ('Left','Right'):
            ids=[self.names.index(side+part) for part in ('UpLeg','Leg','Foot','ToeBase')]
            h,k,f,t=ids
            if self.parents[k]!=h or self.parents[f]!=k or self.parents[t]!=f:raise ValueError('Expected a direct hip/knee/ankle/toe chain')
            self.legs.append(ids)
        self.reset()

    def reset(self):
        self.local=None;self.root=None;self.previous_feet=None;self.pelvis=0.
        self.raw_local=None;self.raw_root=None;self.angular_velocity=None;self.root_velocity=None
        self.foot_velocity=None
        self.contacts=[dict(active=False,anchor=None,kind=0,weight=0.,timer=0.,reason='none') for _ in self.legs]
        self.diagnostics=dict(contacts=0,fully_locked=0,candidates=0,pelvis_offset=0.,release_reasons=[])

    def fk(self,local,root):
        positions=np.empty((len(self.names),3),np.float32);rotations=np.empty((len(self.names),4),np.float32)
        positions[0]=root;rotations[0]=local[0]
        for ids in self.levels:
            parents=self.parents[ids]
            positions[ids]=positions[parents]+quat.mul_vec(rotations[parents],self.offsets[ids])
            rotations[ids]=quat.mul(rotations[parents],local[ids])
        return positions,quat.normalize(rotations)

    @staticmethod
    def floor_height(p):
        # Ankle / toe joint centers are above the sole, in meters.
        return min(float(p[0,1])-.08,float(p[1,1])-.02)

    def process(self,positions,rotations,*,dt=1./60):
        positions=np.asarray(positions,np.float32);rotations=np.asarray(rotations,np.float32)
        if positions.shape!=(len(self.names),3) or rotations.shape!=(len(self.names),4):raise ValueError('Invalid global pose shape')
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('Nonfinite pose or timestep')
        rotations=quat.normalize(rotations)
        local=np.empty_like(rotations);local[0]=rotations[0]
        local[1:]=quat.inv_mul(rotations[self.parents[1:]],rotations[1:])
        if self.local is None or self.cutoff==0:
            self.local=local.copy();self.root=positions[0].copy()
            self.angular_velocity=np.zeros((len(local),3),np.float32);self.root_velocity=np.zeros(3,np.float32)
        else:
            delta=quat.mul_inv(local,self.raw_local)
            delta=np.where(delta[:,:1]<0,-delta,delta)
            angular_velocity=quat.to_scaled_angle_axis(delta)/dt
            derivative_alpha=1-np.exp(-2*np.pi*2*dt)
            self.angular_velocity+=derivative_alpha*(angular_velocity-self.angular_velocity)
            # Filter signed derivatives before their norm, so alternating quantization
            # noise cannot keep opening the adaptive cutoff as if it were fast motion.
            alpha=1-np.exp(-2*np.pi*(self.cutoff+.35*np.linalg.norm(self.angular_velocity,axis=-1))*dt)
            self.local=nlerp(self.local,local,alpha)
            self.root_velocity+=derivative_alpha*((positions[0]-self.raw_root)/dt-self.root_velocity)
            speed=float(np.linalg.norm(self.root_velocity))
            alpha_root=1-np.exp(-2*np.pi*(self.cutoff+2*speed)*dt)
            self.root+=alpha_root*(positions[0]-self.root)
        self.raw_local=local.copy();self.raw_root=positions[0].copy()
        local=self.local.copy()
        pose,rot=self.fk(local,self.root)
        feet=np.array([pose[[f,t]] for _,_,f,t in self.legs])
        if not self.foot_ik:
            self.previous_feet=feet
            self.diagnostics=dict(contacts=0,fully_locked=0,candidates=0,pelvis_offset=0.,release_reasons=[])
            return pose,rot
        velocity=np.zeros_like(feet) if self.previous_feet is None else (feet-self.previous_feet)/dt
        raw_velocity=velocity
        if self.foot_velocity is None:self.foot_velocity=velocity.copy()
        else:self.foot_velocity+=(1-np.exp(-2*np.pi*4*dt))*(velocity-self.foot_velocity)
        velocity=self.foot_velocity
        self.previous_feet=feet.copy();targets=[]
        candidate_count=0
        for i,(hip,knee,foot,toe) in enumerate(self.legs):
            state=self.contacts[i];points=feet[i]
            heights=points[:,1]-np.array([.08,.02],np.float32)
            kind=int(np.argmin(heights)) if not state['active'] else state['kind']
            point=points[kind];v=velocity[i,kind]
            height=float(heights[kind]);horizontal=float(np.linalg.norm(v[[0,2]]))
            if state['active']:
                pull=float(np.linalg.norm((point-state['anchor'])[[0,2]]))
                if height>.10 or float(v[1])>.70 or pull>.20:
                    state['active']=False;state['timer']=0.
                    state['reason']='height' if height>.10 else 'lift' if float(v[1])>.70 else 'reach'
            else:
                # The signed low-pass rejects alternating jitter, while the raw
                # guard prevents acquiring a lock during its startup lag on fast gliding.
                eligible=(height<.06 and abs(float(v[1]))<.30 and horizontal<.70
                          and float(np.linalg.norm(raw_velocity[i,kind,[0,2]]))<1.40)
                candidate_count+=int(eligible)
                state['timer']=state['timer']+dt if eligible else 0.
                if state['timer']>=2./60-1e-8:
                    state.update(active=True,kind=kind,anchor=point.copy())
                    state['reason']='none'
                    state['anchor'][1]=.08 if kind==0 else .02
            state['weight']=float(np.clip(state['weight']+dt*(12 if state['active'] else -16),0,1))
            target=pose[foot].copy()
            if state['anchor'] is not None and state['weight']>0:
                planted=state['anchor'].copy()
                if state['kind']==1:planted-=quat.mul_vec(rot[foot],self.offsets[toe])
                # Preserve foot roll; lift the ankle enough to keep either sole end above the floor.
                planted[1]=max(planted[1],.08,.02-float(quat.mul_vec(rot[foot],self.offsets[toe])[1]))
                delta=planted-target
                delta*=min(1.,.18/max(float(np.linalg.norm(delta)),1e-8))
                target+=state['weight']*delta
            toe_delta=quat.mul_vec(rot[foot],self.offsets[toe])
            target[1]+=max(0.,.08-float(target[1]),.02-float(target[1]+toe_delta[1]))
            targets.append(target)
        wanted=min(.08,max(0.,-min(self.floor_height(p) for p in feet)))
        for (hip,knee,foot,toe),target,state in zip(self.legs,targets,self.contacts):
            if state['weight']<=0:continue
            length=float(np.linalg.norm(self.offsets[knee])+np.linalg.norm(self.offsets[foot]))-.001
            horizontal=float(np.linalg.norm((target-pose[hip])[[0,2]]))
            bound=float(target[1]+np.sqrt(max(length*length-horizontal*horizontal,0.))-pose[hip,1])
            if bound<0:wanted=min(wanted,bound)
        wanted=float(np.clip(wanted,-.08,.08))
        self.pelvis+=(1-np.exp(-2*np.pi*10*dt))*(wanted-self.pelvis)
        root=self.root.copy();root[1]+=self.pelvis
        pose,rot=self.fk(local,root)
        for (hip,knee,foot,toe),target in zip(self.legs,targets):
            if np.linalg.norm(target-pose[foot])<1e-6:continue
            new_knee,new_foot=two_bone_target(pose[hip],pose[knee],pose[foot],target,
                pole=quat.mul_vec(rot[foot],self.offsets[toe]))
            hip_rotation=quat.normalize(quat.mul(rotation_between(pose[knee]-pose[hip],new_knee-pose[hip]),rot[hip]))
            knee_rotation=quat.normalize(quat.mul(rotation_between(pose[foot]-pose[knee],new_foot-new_knee),rot[knee]))
            local[hip]=quat.inv_mul(rot[self.parents[hip]],hip_rotation)
            local[knee]=quat.inv_mul(hip_rotation,knee_rotation)
            local[foot]=quat.inv_mul(knee_rotation,rot[foot])
        pose,rot=self.fk(local,root)
        self.diagnostics=dict(contacts=sum(s['active'] for s in self.contacts),
            fully_locked=sum(s['active'] and s['weight']>=.99 for s in self.contacts),candidates=candidate_count,
            pelvis_offset=float(self.pelvis),release_reasons=[s['reason'] for s in self.contacts])
        # IK is never fed into the temporal filter, avoiding self-reinforcing locks.
        return pose,rot
