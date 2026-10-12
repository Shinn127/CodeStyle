"""Window-wise reference preparation and single-evaluation controller objectives."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .style_code_prior import TrainStyleRecipePool, style_prior_divergence, style_code_entropy


class TemporalReferencePool(TrainStyleRecipePool):
    def sample_windows(self, labels, *, exclusions):
        if labels.device.type!='cpu' or len(exclusions)!=len(labels):
            raise ValueError('Reference selection uses original CPU window labels/exclusions')
        b=len(labels);r=max(self.exemplar_choices);heads=self.cache.indices.shape[1]
        resident=getattr(self,'resident',None)
        codes=np.zeros((b,r,self.frames,heads),dtype=np.int64) if resident is None else None
        starts=np.zeros((b,r),np.int64)
        valid=np.zeros((b,r),bool);records=[];keys=[]
        for row,label in enumerate(labels.tolist()):
            blocked=set(exclusions[row]) & self.source_ids[label]
            candidates=[s for s in self.by_style[label] if s['source_entry_id'] not in blocked]
            count=self.exemplar_choices[int(torch.randint(len(self.exemplar_choices),()))]
            if len(candidates)<count:raise ValueError('Not enough independent train references after source exclusion')
            variant=int(torch.randint(max(self.bank_size,1),()))
            key=(label,frozenset(blocked),count,variant)
            plan=self.bank.get(key)
            if plan is None:
                identity=json.dumps([self.seed,label,count,variant,sorted(str(x) for x in blocked)])
                seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')
                rng=np.random.default_rng(seed);plan=[]
                for pick in rng.permutation(len(candidates))[:count]:
                    s=candidates[int(pick)]
                    start=s['frame_start']+int(rng.integers(s['num_frames']-self.frames+1))
                    plan.append((s['source_entry_id'],start))
                self.bank[key]=plan
                if len(self.bank)>self.bank_max_entries:self.bank.popitem(last=False)
            else:self.bank.move_to_end(key)
            for i,(_,start) in enumerate(plan):
                starts[row,i]=start
                if codes is not None:codes[row,i]=self.cache.indices[start:start+self.frames]
                valid[row,i]=True
            records.append(plan);keys.append(key)
        valid=torch.from_numpy(valid)
        values=torch.from_numpy(codes) if codes is not None else resident.reference_codes(starts,valid,self.frames)
        return values,valid,records,keys


def reference_latents(model,codes,valid):
    points=np.linspace(0,codes.shape[2]-1,model.reference_tokens).astype(np.int64)
    selected=codes[:,:,points]
    latents=model.vqvae.quantizer.lookup(selected.reshape(-1,selected.shape[-1])).reshape(*selected.shape[:-1],model.latent_dim)
    times=torch.as_tensor(points/60.,dtype=torch.float32,device=codes.device).expand(selected.shape[:-1])
    mask=valid[...,None].expand(selected.shape[:-1])
    return latents,times,mask


class ReferenceMetricCache:
    def __init__(self,metric,*,max_entries=262144,device_cache=True):
        self.metric=metric.eval().requires_grad_(False);self.max_entries=max_entries;self.values=OrderedDict()
        self.device_cache=device_cache
        self.graphs={}
        self.time_grids={}

    @torch.no_grad()
    def frozen_embedding(self,model,selected,points):
        count=len(selected);device=selected.device
        grid_key=(device,tuple(points.tolist()))
        base=self.time_grids.get(grid_key) if self.device_cache else None
        if base is None:
            base=torch.as_tensor(points/60.,dtype=torch.float32,device=device)
            if self.device_cache:self.time_grids[grid_key]=base
        times=base.expand(count,len(points))
        def evaluate(codes):
            z=model.vqvae.quantizer.lookup(codes.reshape(-1,codes.shape[-1])).reshape(*codes.shape[:-1],model.latent_dim)
            motion=model.vqvae.decode(z)
            return F.normalize(self.metric(motion,times,dense=self.device_cache).mean(0),dim=0)
        if not self.device_cache or device.type!='cuda':return evaluate(selected).detach()
        key=(count,tuple(points.tolist()),tuple(p._version for p in self.metric.parameters()),
            tuple(p._version for p in model.vqvae.parameters()))
        record=self.graphs.get(key)
        if record is None:
            static=torch.zeros_like(selected);stream=torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):evaluate(static)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=evaluate(static)
            torch.cuda.current_stream(device).wait_stream(stream)
            record=(static,graph,output,times);self.graphs[key]=record
        static,graph,output,_=record
        static.copy_(selected);graph.replay()
        return output.clone()

    @torch.no_grad()
    def embeddings(self,model,codes,valid,keys,points):
        result=[];device=codes.device
        for row,key in enumerate(keys):
            value=self.values.get(key)
            if value is None:
                count=key[2] if isinstance(key,tuple) and len(key)==4 else None
                selected=(codes[row,:count] if count is not None else codes[row,valid[row]])[:,points]
                # Fixed batch shape per key keeps cache hit/miss and resume outputs identical.
                value=self.frozen_embedding(model,selected,points)
                if not self.device_cache:value=value.cpu()
                self.values[key]=value
                if len(self.values)>self.max_entries:self.values.popitem(last=False)
            else:self.values.move_to_end(key)
            result.append(value)
        return torch.stack(result).to(device)


def window_objective(model,window,latents,labels,reference_codes,reference_valid,*,
        metric_cache,reference_keys,unpaired,epoch,microbatch=512,
        style_weight=.1,root_weight=.1,prior_weight=.02,entropy_weight=.003,
        style_dropout=.1,codec_probability=.2,codec_noise=.15,aux_fraction=.25,
        backward=True):
    """Single-step FM; auxiliary sequence supervision never executes an ODE solve."""
    if window.shape[1]!=240 or not 0<=style_dropout<1:raise ValueError('V2 uses 240-frame windows')
    b=len(window);anchors=torch.linspace(0,238,64,device=window.device).long();points=anchors+1
    times=points.to(torch.float32)/60.
    references,stamp,mask=reference_latents(model,reference_codes,reference_valid)
    probs,logits=model.probabilities_from_references(references,stamp,mask)
    style=model.condition_from_probabilities(probs,validate=False)
    dropped=(torch.rand(b,device=window.device)<style_dropout)&~unpaired
    condition=torch.where(dropped[:,None],model.null_style[None],style)
    fixed=model.field.prepare_style(condition)
    expanded=dict(style_token=fixed['style_token'].repeat_interleave(64,0),
        global_style=fixed['global_style'].repeat_interleave(64,0),
        kv=[(k.repeat_interleave(64,0),v.repeat_interleave(64,0)) for k,v in fixed['kv']])
    prior=model.style_prior_probs[labels]
    ce=style_prior_divergence(prior,probs,logits=logits,temperature=model.style_selector.temperature)
    entropy=style_code_entropy(probs)
    regularizer=prior_weight*ce+entropy_weight*entropy
    with torch.no_grad():
        previous=latents[:,anchors].clone()
        use=(torch.rand((b,64,1),device=window.device)<codec_probability*min(epoch/5,1.))
        strength=torch.rand((b,64,1),device=window.device)*codec_noise
        normalized=model.normalize_latent(previous)
        noisy=model.denormalize_latent(normalized+use*strength*torch.randn_like(normalized))
        observed,previous=model.feedback_state(noisy.reshape(-1,model.latent_dim))
        previous=previous.reshape(b,64,-1)
        observed=observed.reshape(b,64,291)
        codec_changed=(previous.reshape(b,64,model.vqvae.num_latents,-1)!=
                       latents[:,anchors].reshape(b,64,model.vqvae.num_latents,-1)).any(-1).float().mean()
    current=torch.cat((window[:,anchors,:84],observed),-1).reshape(-1,375)
    previous=model.normalize_latent(previous).reshape(-1,model.latent_dim)
    target=model.normalize_latent(latents[:,points]).reshape(-1,model.latent_dim)
    styles=condition.repeat_interleave(64,0)
    paired=(~unpaired).repeat_interleave(64)
    # Always supervise unpaired windows; fill the remaining auxiliary budget with paired windows.
    eligible=torch.nonzero(~dropped,as_tuple=False).flatten()
    auxiliary=torch.nonzero(unpaired,as_tuple=False).flatten()
    count=max(math_ceil(b*aux_fraction),len(auxiliary))
    other=eligible[~unpaired[eligible]]
    auxiliary=torch.cat((auxiliary,other[torch.randperm(len(other),device=window.device)[:max(count-len(auxiliary),0)]]))
    auxiliary_mask=torch.zeros(b,dtype=torch.bool,device=window.device);auxiliary_mask[auxiliary]=True
    outputs=[];flow_sum=target.new_zeros(());flow_terms=[];z0=torch.randn_like(target)*model.latent_transport_std
    t=torch.rand((len(target),1),device=window.device);t[~paired]=0
    bridge=torch.where(paired[:,None],(1-t)*z0+t*target,z0)
    for begin in range(0,len(target),microbatch):
        end=min(begin+microbatch,len(target))
        shared=dict(style_token=expanded['style_token'][begin:end],global_style=expanded['global_style'][begin:end],
            kv=[(k[begin:end],v[begin:end]) for k,v in expanded['kv']])
        tokens=model.field.prepare_conditions(previous[begin:end],current[begin:end,84:],current[begin:end,:84],
            styles[begin:end],style_cache=shared)
        # Graphed callables reuse saved forward buffers: accumulation must use eager
        # calls until all microbatch forwards have received their own backward.
        tokens['allow_graph']=len(target)<=microbatch
        _,_,prediction=model._predict_velocity(z_t=bridge[begin:end],z_prev=previous[begin:end],t=t[begin:end],
            e_c=current[begin:end,84:],e_traj=current[begin:end,:84],e_s=styles[begin:end],condition_tokens=tokens)
        per=(prediction-target[begin:end]).square().mean(-1)
        flow=per.masked_fill(~paired[begin:end],0).sum()/paired.sum().clamp_min(1)
        # Split mode is retained only for legacy execution and performance comparison.
        if backward and getattr(model,'backward_mode','combined')=='split':flow.backward(retain_graph=True)
        else:flow_terms.append(flow)
        flow_sum+=flow.detach()
        indices=torch.arange(begin,end,device=window.device)//64
        selected=auxiliary_mask[indices]
        outputs.append((begin,selected,prediction[selected]))
    style_loss=target.new_zeros(());root_loss=target.new_zeros(());changed=target.new_zeros(())
    if len(auxiliary):
        collected=torch.cat([p for _,_,p in outputs],0)
        # Microbatch order is window-major, matching sorted auxiliary window IDs.
        sorted_ids=auxiliary.sort().values
        raw=model.denormalize_latent(collected)
        clipped=model.clip_latent_to_data_bounds(raw)
        with torch.no_grad():quantized=model.requantize_latent(clipped)
        st=clipped+(quantized-clipped).detach()
        decoded=model.vqvae.decode(st).reshape(len(sorted_ids),64,291)
        root_loss=F.l1_loss(decoded[...,:3],window[sorted_ids][:,points,84:87])
        if style_weight>0:
            ref_keys=[reference_keys[i] for i in sorted_ids.cpu().tolist()]
            reference=metric_cache.embeddings(model,reference_codes[sorted_ids],reference_valid[sorted_ids],ref_keys,points.cpu().numpy())
            generated=metric_cache.metric(decoded,times.expand(len(sorted_ids),-1),dense=metric_cache.device_cache)
            positive=(generated*reference).sum(-1)
            # Reference contrast is label-free; identical target styles are not false negatives.
            similarity=generated@reference.T/.1
            negatives=labels[sorted_ids,None]!=labels[sorted_ids][None]
            similarity=similarity.masked_fill(~negatives & ~torch.eye(len(sorted_ids),device=window.device,dtype=torch.bool),-float('inf'))
            style_loss=(1-positive).mean()+.1*F.cross_entropy(similarity,torch.arange(len(sorted_ids),device=window.device))
        changed=(quantized.reshape(-1,model.vqvae.num_latents,model.vqvae.code_dim)!=
                 clipped.reshape(-1,model.vqvae.num_latents,model.vqvae.code_dim)).any(-1).float().mean()
    auxiliary_loss=style_weight*style_loss+root_weight*root_loss
    if backward:
        total=regularizer+auxiliary_loss
        if flow_terms:total=total+sum(flow_terms)
        total.backward()
    return dict(loss=flow_sum+regularizer.detach()+auxiliary_loss.detach(),flow_loss=flow_sum,
        style_loss=style_loss.detach(),root_loss=root_loss.detach(),prior_ce=ce.detach(),
        prior_kl=(ce-model._prior_entropy_by_style(kl=True)[labels].mean()).detach(),entropy=entropy.detach(),
        style_dropout_fraction=dropped.float().mean(),unpaired_fraction=unpaired.float().mean(),
        codec_noise_fraction=use.float().mean(),auxiliary_windows=target.new_tensor(len(auxiliary)),
        codec_code_change_fraction=codec_changed,
        quantized_endpoint_fraction=changed,num_anchors=target.new_tensor(len(target)))


def math_ceil(x):
    return int(np.ceil(x))
