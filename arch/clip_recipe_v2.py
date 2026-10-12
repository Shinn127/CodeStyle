"""Temporal references, a single probability interface, and repeated style conditioning."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .flowmatch_vqvae_controller import FlowMatchVQVAEController, _timestep_embedding


class TemporalStyleEncoder(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, tokens=60):
        super().__init__()
        self.tokens = tokens
        self.input = nn.Linear(input_dim, hidden)
        self.position = nn.Parameter(torch.randn(1, tokens, hidden) * .02)
        self.time = nn.Linear(2, hidden)
        block = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, sequence, times, valid):
        # Empty padded exemplars never enter attention, preventing all-masked NaNs.
        if sequence.ndim != 4 or sequence.shape[:-1] != valid.shape or times.shape != valid.shape:
            raise ValueError('References must be [B,R,T,D] with matching times/mask')
        b,r,t,d = sequence.shape
        if t != self.tokens or not valid.any(-1).any(-1).all():
            raise ValueError('Reference token count differs or a sample has no exemplar')
        flat = valid.flatten(0,1); active = flat.any(-1)
        x = sequence.flatten(0,1)[active]; stamp = times.flatten(0,1)[active]
        mask = flat[active]
        dt = torch.cat((stamp[:,:1] * 0, stamp[:,1:] - stamp[:,:-1]), -1)
        h = self.input(x) + self.position + self.time(torch.stack((stamp,dt),-1))
        if h.is_cuda and torch.is_grad_enabled():
            # FP32 math attention also covers input gradients through the frozen Φ.
            with sdpa_kernel(SDPBackend.MATH):h=self.encoder(h,src_key_padding_mask=~mask)
        else:h=self.encoder(h,src_key_padding_mask=~mask)
        h = self.norm(h)
        pooled = (h * mask[...,None]).sum(1) / mask.sum(1,keepdim=True)
        output = x.new_zeros((b*r,pooled.shape[-1]))
        output[active] = pooled
        output = output.reshape(b,r,-1)
        return output.sum(1) / active.reshape(b,r).sum(1,keepdim=True)

    def dense(self,sequence,times):
        """Trusted fixed-length clips, with the same all-false attention mask and reduction."""
        dt=torch.cat((times[:,:1]*0,times[:,1:]-times[:,:-1]),-1)
        h=self.input(sequence)+self.position+self.time(torch.stack((times,dt),-1))
        padding=torch.zeros(sequence.shape[:2],device=sequence.device,dtype=torch.bool)
        if h.is_cuda and torch.is_grad_enabled():
            with sdpa_kernel(SDPBackend.MATH):h=self.encoder(h,src_key_padding_mask=padding)
        else:h=self.encoder(h,src_key_padding_mask=padding)
        return self.norm(h).sum(1)/self.tokens


class RecipeConditionEncoder(nn.Module):
    def __init__(self, code_dim, slots, *, slot_dim=128, global_dim=256):
        super().__init__()
        self.slots, self.slot_dim, self.global_dim = slots,slot_dim,global_dim
        self.code = nn.Sequential(nn.Linear(code_dim,64),nn.GELU(),nn.Linear(64,64))
        self.slot_id = nn.Parameter(torch.randn(slots,8) * .02)
        self.to_slot = nn.Sequential(nn.Linear(code_dim+64+1+8,slot_dim),nn.GELU(),nn.LayerNorm(slot_dim))
        self.to_global = nn.Sequential(nn.Linear(slots*slot_dim,global_dim),nn.GELU(),nn.Linear(global_dim,global_dim))
        self.width = global_dim + slots*slot_dim

    def forward(self, probs, codebook):
        book = codebook.detach()
        mean = probs @ book
        geometry=(book-book.mean(0))/book.std(0,unbiased=False).clamp_min(1e-6)
        features = probs @ self.code(geometry)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1,keepdim=True)
        slot = self.slot_id[None].expand(probs.shape[0],-1,-1)
        tokens = self.to_slot(torch.cat((mean,features,entropy,slot),-1))
        return torch.cat((self.to_global(tokens.flatten(1)),tokens.flatten(1)),-1)


def modulate(x, shift, scale):
    return x * (1+scale[:,None]) + shift[:,None]


class StyleBlock(nn.Module):
    def __init__(self, hidden, heads, ffn):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden,elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden,elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden,3*hidden)
        self.attn_out = nn.Linear(hidden,hidden)
        self.query = nn.Linear(hidden,hidden)
        self.key_value = nn.Linear(hidden,2*hidden)
        self.cross_out = nn.Linear(hidden,hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden,ffn),nn.GELU(),nn.Linear(ffn,hidden))
        self.modulation = nn.Sequential(nn.SiLU(),nn.Linear(hidden,7*hidden))
        nn.init.zeros_(self.modulation[-1].weight); nn.init.zeros_(self.modulation[-1].bias)

    def split_heads(self, x):
        b,t,d = x.shape
        return x.reshape(b,t,self.heads,d//self.heads).transpose(1,2)

    def prepare_slots(self, slots):
        k,v = self.key_value(slots).chunk(2,-1)
        return self.split_heads(k),self.split_heads(v)

    def forward(self, x, coefficients, kv):
        shift,scale,gate,shift2,scale2,gate2,cross_gate = coefficients.chunk(7,-1)
        q,k,v = self.qkv(modulate(self.norm1(x),shift,scale)).chunk(3,-1)
        attended = F.scaled_dot_product_attention(self.split_heads(q),self.split_heads(k),self.split_heads(v))
        attended = attended.transpose(1,2).flatten(2)
        x = x + gate[:,None] * self.attn_out(attended)
        q = self.split_heads(self.query(self.norm_cross(x[:,:1])))
        attended = F.scaled_dot_product_attention(q,*kv).transpose(1,2).flatten(2)
        main = x[:,:1] + cross_gate[:,None] * self.cross_out(attended)
        x = torch.cat((main,x[:,1:]),1)
        return x + gate2[:,None] * self.mlp(modulate(self.norm2(x),shift2,scale2))


class StyleInjectedField(nn.Module):
    def __init__(self, *, latent_dim, motion_dim, traj_dim, style_width, slots,
                 slot_dim=128, global_dim=256, hidden=256, heads=4, ffn=1024,
                 layers=4, time_dim=64):
        super().__init__()
        self.fp32_math_attention = True
        self.time_dim,self.global_dim,self.slots,self.slot_dim = time_dim,global_dim,slots,slot_dim
        self.latent_proj = nn.Linear(latent_dim,hidden)
        self.motion_proj = nn.Linear(motion_dim,hidden)
        self.traj_proj = nn.Linear(traj_dim,hidden)
        self.style_proj = nn.Linear(style_width,hidden)
        self.time_proj = nn.Linear(time_dim,hidden)
        self.slot_proj = nn.Linear(slot_dim,hidden)
        self.global_proj = nn.Linear(global_dim,hidden)
        self.position_embeddings = nn.Parameter(torch.randn(1,6,hidden)*.02)
        self.blocks = nn.ModuleList([StyleBlock(hidden,heads,ffn) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(hidden,elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(),nn.Linear(hidden,2*hidden))
        nn.init.zeros_(self.output_modulation[-1].weight); nn.init.zeros_(self.output_modulation[-1].bias)
        self.out = nn.Linear(hidden,latent_dim)

    def coefficients(self, global_style, t):
        time = self.time_proj(_timestep_embedding(t, self.time_dim))
        c = global_style + time
        return time,[block.modulation(c) for block in self.blocks],self.output_modulation(c)

    def prepare_style(self,style):
        slots = self.slot_proj(style[:,self.global_dim:].reshape(-1,self.slots,self.slot_dim))
        return dict(style_token=self.style_proj(style),
            global_style=self.global_proj(style[:,:self.global_dim]),
            kv=[block.prepare_slots(slots) for block in self.blocks])

    def prepare_conditions(self,z_prev,content,traj,style,style_cache=None):
        fixed=self.prepare_style(style) if style_cache is None else style_cache
        if self.training and getattr(self,'_graph_raw_conditions',False):
            return dict(raw=(z_prev,content,traj),style_token=fixed['style_token'],global_style=fixed['global_style'],kv=fixed['kv'])
        return dict(tokens=torch.stack((self.latent_proj(z_prev),self.motion_proj(content),
            self.traj_proj(traj),fixed['style_token']),1),global_style=fixed['global_style'],kv=fixed['kv'])

    def forward(self, *, z_t,z_prev,t,content,traj,style,condition_tokens=None):
        cache = condition_tokens if condition_tokens is not None else self.prepare_conditions(z_prev,content,traj,style)
        return self.projected(z_t,t,cache)

    def projected(self,z_t,t,cache):
        time,coefficients,final = cache['coefficients'] if 'coefficients' in cache else self.coefficients(cache['global_style'],t)
        x = torch.cat((self.latent_proj(z_t)[:,None],cache['tokens'],time[:,None]),1) + self.position_embeddings
        def apply():
            h=x
            for block,c,kv in zip(self.blocks,coefficients,cache['kv']):h=block(h,c,kv)
            shift,scale=final.chunk(2,-1)
            return self.out(modulate(self.output_norm(h[:,:1]),shift,scale)[:,0])
        if self.fp32_math_attention and self.training and x.is_cuda:
            with sdpa_kernel(SDPBackend.MATH):return apply()
        return apply()


class ClipRecipeV2Controller(FlowMatchVQVAEController):
    def __init__(self, *, reference_tokens=60, reference_frames=240, clip_hidden=128,
                 clip_layers=2, slot_dim=128, global_dim=256, style_guidance_scale=1.5, **kwargs):
        slots = kwargs.get('selected_style_slots')
        slots = tuple(slots) if slots is not None else tuple(range(min(8,kwargs['vqvae'].num_latents)))
        kwargs.update(conditioning_mode='clip_recipe',use_style_code_selector=True,
            style_dim=global_dim+len(slots)*slot_dim, selected_style_slots=slots)
        super().__init__(**kwargs)
        self.conditioning_version=2
        self.reference_tokens,self.reference_frames = reference_tokens,reference_frames
        self.clip_encoder=TemporalStyleEncoder(self.latent_dim,clip_hidden,clip_layers,reference_tokens)
        self.clip_selector=nn.Linear(clip_hidden,len(slots)*self.vqvae.quantizer.num_embeddings)
        nn.init.normal_(self.clip_selector.weight,std=.01);nn.init.zeros_(self.clip_selector.bias)
        self.condition_encoder=RecipeConditionEncoder(self.vqvae.code_dim,len(slots),slot_dim=slot_dim,global_dim=global_dim)
        self.histogram_selector=None
        self.style_selector.requires_grad_(False)  # Legacy projections are never condition inputs.
        self.style_selector.style_embedding=None
        self.style_selector.to_logits=None
        self.style_selector.proto_to_style=None
        self.style_selector.proto_norm=None
        self.null_style=nn.Parameter(torch.zeros(self.style_dim))
        self.field=StyleInjectedField(latent_dim=self.latent_dim,motion_dim=self.motion_dim,traj_dim=self.traj_dim,
            style_width=self.style_dim,slots=len(slots),slot_dim=slot_dim,global_dim=global_dim,
            hidden=kwargs.get('flow_hidden_dim',256),heads=kwargs.get('flow_heads',4),
            ffn=kwargs.get('flow_ffn_dim',1024),layers=kwargs.get('flow_layers',4),time_dim=kwargs.get('time_embed_dim',64))
        if not math.isfinite(style_guidance_scale) or style_guidance_scale<0:
            raise ValueError('Invalid style guidance scale')
        self.style_guidance_scale=style_guidance_scale
        self._sampling_style_cache=None

    def train(self,mode=True):
        if mode:self._sampling_style_cache=None
        return super().train(mode)

    @torch.no_grad()
    def requantize_latent(self,latents):
        # Avoid a global bincount/perplexity and host diagnostic scalar per feedback.
        if hasattr(self.vqvae.quantizer,'quantize_only'):return self.vqvae.quantizer.quantize_only(latents.float())
        return super().requantize_latent(latents)

    def probabilities_from_references(self, latents,times,valid):
        h=self.clip_encoder(self.normalize_latent(latents),times,valid)
        logits=self.clip_selector(h).reshape(len(h),len(self.selected_style_slots),-1)
        dense=(logits/self.style_selector.temperature).softmax(-1)
        ids=torch.argsort(dense,descending=True,stable=True)[...,:self.style_selector.topk]
        probs=torch.zeros_like(dense).scatter(-1,ids,dense.gather(-1,ids))
        return probs/probs.sum(-1,keepdim=True),logits

    def condition_from_probabilities(self, probs, *, validate=True):
        if validate:
            probs,mask=self.style_selector._normalize_manual_probs(probs,probs.shape[0] if probs.ndim==3 else 1)
            if not mask.all():raise ValueError('All recipe slots are required')
        return self.condition_encoder(probs,self.vqvae.quantizer.embedding_weight)

    def recipe_from_histogram(self,*args,**kwargs):
        raise ValueError('Temporal controller requires reference sequences; unordered histograms cannot substitute')

    @torch.no_grad()
    def recipe_from_indices(self,indices,*,clip_lengths=None):
        data=np.asarray(indices)
        if data.ndim!=2 or data.shape[1]!=self.vqvae.num_latents or not len(data):
            raise ValueError('Reference indices must be nonempty [frames,VQ slots]')
        lengths=list(clip_lengths or [len(data)])
        if min(lengths)<2 or sum(lengths)!=len(data):raise ValueError('Reference clip boundaries differ from frame count')
        codes=np.zeros((1,len(lengths),self.reference_tokens,data.shape[1]),dtype=np.int64)
        valid=np.zeros(codes.shape[:-1],bool);times=np.zeros(codes.shape[:-1],np.float32);begin=0
        for i,length in enumerate(lengths):
            count=min(length,self.reference_tokens)
            positions=np.linspace(0,length-1,count).astype(np.int64)
            codes[0,i,:count]=data[begin+positions];valid[0,i,:count]=True
            times[0,i,:count]=positions/60.;begin+=length
        device=self.latent_offset.device
        latents=self.vqvae.quantizer.lookup(torch.from_numpy(codes).to(device).reshape(-1,codes.shape[-1]))
        latents=latents.reshape(*codes.shape[:-1],self.latent_dim)
        probs,_=self.probabilities_from_references(latents,torch.from_numpy(times).to(device),torch.from_numpy(valid).to(device))
        return probs[0].cpu().numpy()

    def encode_conditions(self, *, x_curr,style_label,manual_style_probs=None,
                          manual_style_strength=1.,return_style_details=False,**kwargs):
        if style_label is not None or manual_style_probs is None or manual_style_strength!=1.:
            raise ValueError('V2 requires a pure complete recipe, or an explicitly encoded temporal reference')
        x_curr=self._flatten_frames(x_curr,self.input_dim)
        probs,mask=self.style_selector._normalize_manual_probs(manual_style_probs,x_curr.shape[0])
        if not mask.all():raise ValueError('All recipe slots are required')
        traj,motion=self.split_traj_motion(x_curr)
        result=(motion,traj,self.condition_from_probabilities(probs,validate=False))
        if return_style_details:
            return result+(dict(style_probs=probs,style_logits=probs.clamp_min(1e-8).log(),style_logits_are_dense=False),)
        return result

    def _integrate(self, *, z_prev,e_c,e_traj,e_s,num_steps,noise_std,z0=None):
        if num_steps<1:raise ValueError('Euler requires positive steps')
        z=self.sample_transport_noise_like(z_prev,noise_std=noise_std) if z0 is None else z0
        n=len(z);w=self.style_guidance_scale
        guided=w!=1
        style=torch.cat((e_s,self.null_style[None].expand(n,-1)),0) if guided else e_s
        history=torch.cat((z_prev,z_prev)) if guided else z_prev
        content=torch.cat((e_c,e_c)) if guided else e_c
        traj=torch.cat((e_traj,e_traj)) if guided else e_traj
        can_cache=not self.training and not e_s.requires_grad and not e_s.is_inference()
        key=(id(e_s),e_s._version,num_steps,w,tuple(p._version for p in self.field.parameters()),
             self.null_style._version) if can_cache else None
        saved=self._sampling_style_cache if can_cache else None
        if saved is not None and saved['key']==key and saved['input'] is e_s:
            fixed,coeff=saved['fixed'],saved['coefficients']
        else:
            fixed=self.field.prepare_style(style)
            coeff=[self.field.coefficients(fixed['global_style'],z.new_full((len(style),1),k/num_steps)) for k in range(num_steps)]
            if can_cache:self._sampling_style_cache=dict(key=key,input=e_s,fixed=fixed,coefficients=coeff)
        cache=self.field.prepare_conditions(history,content,traj,style,style_cache=fixed)
        for k in range(num_steps):
            t=z.new_full((len(style),1),k/num_steps)
            state=torch.cat((z,z)) if guided else z
            cache_step={**cache,'coefficients':coeff[k]}
            out=self.field(z_t=state,z_prev=history,t=t,content=content,traj=traj,style=style,condition_tokens=cache_step)
            target=self.transport_denormalize_latent(out)
            if guided:
                conditional,null=target.chunk(2);target=null+w*(conditional-null)
            z=z+(target-z)/max(1-k/num_steps,self.target_time_eps)/num_steps
        return z

    @torch.no_grad()
    def predict_next_motion(self,**kwargs):
        return self.feedback_state(self.sample_latent(**kwargs))


class MotionStyleMetric(nn.Module):
    def __init__(self,hidden=128,layers=2,tokens=64):
        super().__init__()
        self.encoder=TemporalStyleEncoder(432,hidden,layers,tokens)

    def forward(self,motion,times,valid=None,*,dense=False):
        # Root-frame positions/rotations and their differences; no root3 or joint
        # velocities, which contain explicit root translation/yaw contributions.
        if motion.ndim!=3 or motion.shape[-1]!=291:raise ValueError('Style metric requires [B,T,291]')
        if valid is None:valid=torch.ones(motion.shape[:2],dtype=torch.bool,device=motion.device)
        body=motion[...,3:219]
        dt=(times[:,1:]-times[:,:-1])*60
        if not dense and (dt<=0).any():raise ValueError('Style metric times must increase')
        difference=torch.cat((torch.zeros_like(body[:,:1]),(body[:,1:]-body[:,:-1])/dt[...,None]),1)
        data=torch.cat((body,difference),-1)
        feature=self.encoder.dense(data,times) if dense else self.encoder(data[:,None],times[:,None],valid[:,None])
        return F.normalize(feature,dim=-1)


def build_clip_recipe_v2(config,*,vqvae,num_styles):
    slots=config.get('style_slot_indices')
    return ClipRecipeV2Controller(vqvae=vqvae,num_styles=num_styles,selected_style_slots=slots,
        style_selector_topk=int(config.get('style_selector_topk',4)),
        style_selector_temperature=float(config.get('style_selector_temperature',1)),
        style_proto_normalization='none',dropout=0.,prediction_mode='target',flow_loss_space='normalized',
        inference_requantize=True,inference_latent_clip=True,
        reference_frames=int(config.get('reference_frames',240)),reference_tokens=int(config.get('reference_tokens',60)),
        clip_hidden=int(config.get('clip_hidden',128)),clip_layers=int(config.get('clip_layers',2)),
        slot_dim=int(config.get('slot_dim',128)),global_dim=int(config.get('global_dim',256)),
        flow_hidden_dim=int(config.get('flow_hidden_dim',256)),flow_layers=int(config.get('flow_layers',4)),
        flow_heads=int(config.get('flow_heads',4)),flow_ffn_dim=int(config.get('flow_ffn_dim',1024)),
        style_guidance_scale=float(config.get('style_guidance_scale',1.5)))
