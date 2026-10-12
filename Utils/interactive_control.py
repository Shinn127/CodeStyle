"""Online 60 Hz trajectory planning and autoregressive label-model state."""
from collections import deque
import numpy as np
import torch

from Utils import quat
from Utils.motion375 import decode_selected_state


def wrap_angle(value):
    return (value+np.pi)%(2*np.pi)-np.pi


def planar_rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,s],[-s,c]],dtype=np.float32)


class InteractiveTrajectory:
    """Past = generated root history; future = smoothly projected keyboard intent."""
    def __init__(self,*,fps=60,window=60,stride=10,turn_speed=90.,smooth_commands=False):
        if fps!=60 or window<1 or stride<1 or len(range(-window,window,stride))!=12:
            raise ValueError('Expected 60fps and twelve trajectory context samples')
        if not np.isfinite(turn_speed) or turn_speed<=0:raise ValueError('Invalid turn speed')
        self.dt=1./fps;self.window=window
        self.offsets=np.arange(-window,window,stride)
        self.turn_speed=np.radians(turn_speed)
        self.smooth_commands=bool(smooth_commands)
        self.reset()

    def reset(self):
        self.position=np.zeros(2,np.float32);self.yaw=0.;self.desired_yaw=0.
        self.velocity=np.zeros(2,np.float32)
        self.intent_velocity=np.zeros(2,np.float32)
        self.intent_turn=0.
        self.history=deque(maxlen=self.window+1)
        self.history.append((self.position.copy(),self.yaw,self.velocity.copy()))

    def advance(self,root_motion):
        root=np.asarray(root_motion,dtype=np.float32)
        if root.shape!=(3,) or not np.isfinite(root).all():raise ValueError('Nonfinite root feedback')
        # Same previous-frame velocity / degree-per-second yaw integration as motion375.
        self.velocity=planar_rotation(self.yaw)@root[[0,2]]
        self.position=self.position+self.velocity*self.dt
        self.yaw=wrap_angle(self.yaw+np.radians(float(root[1]))*self.dt)
        self.history.append((self.position.copy(),self.yaw,self.velocity.copy()))

    def condition(self,forward=0.,side=0.,turn=0.,*,speed=1.2):
        if not np.isfinite([forward,side,turn,speed]).all() or not 0<=speed<=3:
            raise ValueError('Invalid keyboard control')
        if self.smooth_commands:
            self.intent_turn+=(float(np.clip(turn,-1,1))-self.intent_turn)*min(1.,8*self.dt)
            turn=self.intent_turn
        self.desired_yaw=wrap_angle(self.desired_yaw+float(np.clip(turn,-1,1))*self.turn_speed*self.dt)
        command=np.array([side,forward],np.float32)
        command*=speed/max(float(np.linalg.norm(command)),1.)
        states=list(self.history)
        goal_velocity=planar_rotation(self.desired_yaw)@command
        self.intent_velocity+=(goal_velocity-self.intent_velocity)*min(1.,8*self.dt)
        future_command=(planar_rotation(self.desired_yaw).T@self.intent_velocity
                        if self.smooth_commands and np.linalg.norm(command)>0 else command)
        # The trajectory is a desired plan. Using observed zero velocity at its
        # center forever would tell a stationary model to keep standing still.
        position=self.position.copy();yaw=self.yaw;velocity=self.intent_velocity.copy()
        future={0:(position.copy(),yaw,velocity.copy())}
        for frame in range(1,int(max(self.offsets))+1):
            goal_yaw=self.desired_yaw+float(np.clip(turn,-1,1))*self.turn_speed*frame*self.dt
            delta=np.clip(wrap_angle(goal_yaw-yaw),-self.turn_speed*self.dt,self.turn_speed*self.dt)
            yaw=wrap_angle(yaw+delta)
            goal_velocity=planar_rotation(yaw)@future_command
            velocity=velocity+(goal_velocity-velocity)*min(1.,8*self.dt)
            position=position+velocity*self.dt
            future[frame]=(position.copy(),yaw,velocity.copy())
        samples=[states[max(0,len(states)-1+int(k))] if k<0 else future[int(k)] for k in self.offsets]
        positions=np.stack([s[0] for s in samples])
        directions=np.stack([planar_rotation(s[1])@np.array([0.,1.],np.float32) for s in samples])
        velocities=np.stack([s[2] for s in samples])
        inverse=planar_rotation(self.yaw).T
        p=(positions-self.position)@inverse.T;d=directions@inverse.T;v=velocities@inverse.T
        trajectory=np.concatenate((p[:,0],p[:,1],d[:,0],d[:,1],v[:,0],v[:,1],np.linalg.norm(v,axis=-1))).astype(np.float32)
        debug=dict(positions=positions,directions=directions)
        return trajectory,debug


class LabelControlSession:
    def __init__(self,model,initial_frame,mean,std,style_names,*,style=0,seed=42,
                 steps=8,noise_std=1.,style_blend=.25,window=60,stride=10,turn_speed=90.,requantize=True,smooth_plan=False):
        if getattr(model,'conditioning_mode','label')!='label':raise ValueError('Interactive entry requires a label controller')
        if len(style_names)!=model.num_styles:raise ValueError('Style names differ from model labels')
        if steps<1 or noise_std<=0 or not np.isfinite([noise_std,style_blend]).all() or style_blend<0:
            raise ValueError('Invalid sampling or transition settings')
        if not isinstance(requantize,bool):raise ValueError('requantize must be a bool')
        self.model=model.eval().requires_grad_(False);self.device=next(model.parameters()).device
        self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32)
        if self.mean.shape!=(375,) or self.std.shape!=(375,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError('Invalid frozen normalization')
        self.names=list(style_names);self.seed=seed;self.steps=steps;self.noise_std=noise_std
        self.requantize=requantize;self.style_blend=style_blend;self._features={}
        self.planner=InteractiveTrajectory(window=window,stride=stride,turn_speed=turn_speed,smooth_commands=smooth_plan)
        self.mean_t=torch.as_tensor(self.mean,device=self.device)
        self.std_t=torch.as_tensor(self.std,device=self.device)
        initial=np.asarray(initial_frame,np.float32)
        if initial.shape!=(375,) or not np.isfinite(initial).all():raise ValueError('Invalid initial motion375 frame')
        self.initial=torch.as_tensor((initial-self.mean)/self.std,device=self.device)[None]
        self.style=int(style);self._check_style(self.style)
        self.reset()

    def _check_style(self,style):
        if not 0<=style<len(self.names):raise ValueError('Style label out of range')

    @torch.no_grad()
    def feature(self,style):
        self._check_style(style)
        if style not in self._features:
            labels=torch.tensor([style],device=self.device)
            self._features[style]=self.model.encode_conditions(x_curr=self.initial,style_label=labels)[2].detach()
        return self._features[style]

    @torch.no_grad()
    def set_style(self,style):
        style=int(style);self._check_style(style)
        if style==self.style:return
        self.start_feature=self.current_feature.clone();self.style=style;self.elapsed=0.
        if self.style_blend==0:self.current_feature=self.feature(style)

    @torch.no_grad()
    def reset(self):
        self.generator=torch.Generator(device=self.device).manual_seed(self.seed)
        self.planner.reset();self.frames=0;self.elapsed=self.style_blend
        self.latent=self.model.frame_latent(self.initial).detach()
        self.history_latent=self.latent.clone()
        self.motion=self.model.vqvae.decode(self.latent).detach()
        self.raw_motion=(self.motion*self.std_t[84:]+self.mean_t[84:])[0].cpu().numpy()
        self.current_feature=self.feature(self.style);self.start_feature=self.current_feature
        self.trajectory,self.debug=self.planner.condition(speed=0.)

    @torch.no_grad()
    def step(self,forward=0.,side=0.,turn=0.,*,speed=1.2):
        self.planner.advance(self.raw_motion[:3])
        self.trajectory,self.debug=self.planner.condition(forward,side,turn,speed=speed)
        self.elapsed+=self.planner.dt
        alpha=1. if self.style_blend==0 else min(1.,self.elapsed/self.style_blend)
        alpha=alpha*alpha*(3-2*alpha)
        self.current_feature=torch.lerp(self.start_feature,self.feature(self.style),alpha)
        traj=torch.as_tensor((self.trajectory-self.mean[:84])/self.std[:84],device=self.device)[None]
        self.input=torch.cat((traj,self.motion),-1)
        noise=torch.randn((1,self.model.latent_dim),device=self.device,generator=self.generator)
        z0=self.model.latent_offset+self.model.latent_scale*self.model.latent_transport_std*self.noise_std*noise
        continuous=self.model.sample_latent(x_curr=self.input,style_label=None,style_feature=self.current_feature,
            prev_latent=self.latent,z0=z0,num_steps=self.steps,solver='euler',
            **({'history_latent':self.history_latent} if getattr(self.model,'history_conditioning',False) else {}))
        self.history_latent=self.latent.detach()
        self.motion,self.latent=self.model.feedback_state(continuous,requantize=self.requantize,latent_clip=True)
        self.raw_motion=(self.motion*self.std_t[84:]+self.mean_t[84:])[0].cpu().numpy()
        if not np.isfinite(self.raw_motion).all():raise ValueError('Nonfinite generated motion; reset the session')
        self.frames+=1
        return self.raw_motion

    def selected_pose(self):
        frame=np.concatenate((self.trajectory,self.raw_motion))
        _,position,rotation,_=decode_selected_state(frame)
        yaw=quat.from_angle_axis(np.array([self.planner.yaw],np.float32),np.array([[0,1,0]],np.float32))[:,None]
        origin=np.array([self.planner.position[0],0.,self.planner.position[1]],np.float32)
        return quat.mul_vec(yaw,position)+origin,quat.mul(yaw,rotation)
