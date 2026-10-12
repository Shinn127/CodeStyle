"""Static CUDA execution of a single, dropout-free flow-field call per backward."""
import torch
from torch import nn


def graph_training_field(model,batch_size):
    field=model.field
    if model.latent_offset.device.type!='cuda':raise ValueError('Field graphs require CUDA')
    if getattr(model,'backward_mode','combined')!='combined':raise ValueError('CUDA field graphs require one combined backward')
    original=field.projected;device=model.latent_offset.device;hidden=field.out.in_features
    layers=len(field.blocks);heads=field.blocks[0].heads
    class Projected(nn.Module):
        def __init__(self):
            super().__init__()
            excluded=('style_proj.','slot_proj.','global_proj.')
            self.weights=nn.ParameterList([p for name,p in field.named_parameters()
                if not name.startswith(excluded) and '.key_value.' not in name])
        def forward(self,z,previous,content,traj,t,style_token,global_style,*kv):
            tokens=torch.stack((field.latent_proj(previous),field.motion_proj(content),field.traj_proj(traj),style_token),1)
            cache=dict(tokens=tokens,global_style=global_style,kv=list(zip(kv[::2],kv[1::2])))
            return original(z,t,cache)
    samples=(torch.zeros(batch_size,model.latent_dim,device=device),torch.zeros(batch_size,model.latent_dim,device=device),
        torch.zeros(batch_size,model.motion_dim,device=device),torch.zeros(batch_size,model.traj_dim,device=device),
        torch.zeros(batch_size,1,device=device),torch.zeros(batch_size,hidden,device=device,requires_grad=True),
        torch.zeros(batch_size,hidden,device=device,requires_grad=True))
    samples+=tuple(torch.zeros(batch_size,heads,field.slots,hidden//heads,device=device,requires_grad=True) for _ in range(2*layers))
    wrapper=Projected().train()
    captured=torch.cuda.make_graphed_callables(wrapper,samples,num_warmup_iters=3,allow_unused_input=True)
    def projected(z,t,cache):
        if field.training and len(z)==batch_size and cache.get('allow_graph',True) and 'coefficients' not in cache:
            tensors=(z,*cache['raw'],t,cache['style_token'],cache['global_style'])+tuple(x for pair in cache['kv'] for x in pair)
            return captured(*tensors)
        if 'raw' in cache:
            previous,content,traj=cache['raw']
            cache={**cache,'tokens':torch.stack((field.latent_proj(previous),field.motion_proj(content),field.traj_proj(traj),cache['style_token']),1)}
        return original(z,t,cache)
    field._graph_raw_conditions=True
    field.projected=projected
    model.zero_grad(set_to_none=True)
    return captured
