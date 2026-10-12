"""Train-only, fixed-window donor correspondence. No generated feedback or VQ updates."""
from pathlib import Path
import json

import numpy as np
import torch


def phase_descriptor(motion):
    """Amplitude-normalized left/right leg direction, velocity and relative contact cues."""
    feet = motion[..., 3:75].reshape(*motion.shape[:-1], 24, 3)[..., [18, 19, 22, 23], :]
    velocity = motion[..., 219:291].reshape(*motion.shape[:-1], 24, 3)[..., [18, 19, 22, 23], :]
    delta = feet[..., :2, :].mean(-2) - feet[..., 2:, :].mean(-2)
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(.05)
    dv = velocity[..., :2, :].mean(-2) - velocity[..., 2:, :].mean(-2)
    dv = dv / dv.norm(dim=-1, keepdim=True).clamp_min(.1)
    height = feet[..., 1]
    floor = height.amin(dim=(-1, -2), keepdim=True)
    contact = torch.exp(-(height - floor) / .03)
    return torch.cat((delta, dv, .5 * contact), -1).flatten(1)


def retarget_motion(source_root, donor_motion):
    result = donor_motion.clone()
    delta = source_root - donor_motion[..., :3]
    translation = delta.clone()
    translation[..., 1] = 0
    omega = torch.zeros_like(delta)
    omega[..., 1] = torch.deg2rad(delta[..., 1])
    positions = result[..., 3:75].reshape(*result.shape[:-1], 24, 3)
    correction = translation.unsqueeze(-2) + torch.linalg.cross(omega.unsqueeze(-2).expand_as(positions), positions, dim=-1)
    result[..., 219:291] += correction.flatten(-2)
    result[..., :3] = source_root
    return result


class StyleDonorBank:
    def __init__(self, db, *, window=16, windows_per_clip=16, device='cpu', phase_aware=False, max_steps=None,
                 match_cache_gb=0.):
        if window < 2 or windows_per_clip < 1:
            raise ValueError('Donor bank requires window>=2 and windows_per_clip>=1')
        db = Path(db)
        meta = json.loads((db / 'meta.json').read_text())
        self.segments = json.loads((db / 'segments.json').read_text())
        self.source = np.memmap(db / 'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
        self.window, self.device = window, torch.device(device)
        self.phase_aware = bool(phase_aware)
        if not np.isfinite(match_cache_gb) or match_cache_gb<0:raise ValueError('Invalid match cache budget')
        self.max_steps = window//2+1 if max_steps is None else int(max_steps)
        if not 1<=self.max_steps<=window: raise ValueError('Invalid donor rollout depth')
        self.anchor_stride = window-self.max_steps+1
        self.suffix_map = {s: i for i, s in enumerate(sorted({str(x.get('suffix', '')) for x in self.segments}))}
        if '' in self.suffix_map:
            raise ValueError('Donor bank requires action suffix metadata')
        self.id_map = {s: i for i, s in enumerate(dict.fromkeys(x['source_entry_id'] for x in self.segments))}
        rows = []
        for segment in self.segments:
            if segment['split'] != 'train' or segment['num_frames'] < window + 1:
                continue
            offsets = np.unique(np.linspace(0, segment['num_frames'] - window - 1,
                                            windows_per_clip).astype(np.int64))
            rows.extend((segment, int(segment['frame_start']) + int(o)) for o in offsets)
        if not rows:
            raise ValueError('No train clips long enough for donor windows')
        self.starts = np.array([start for _, start in rows])
        self.entries = [s['source_entry_id'] for s, _ in rows]
        self.labels = torch.tensor([s['label'] for s, _ in rows], device=device)
        self.entry_codes = torch.tensor([self.id_map[s] for s in self.entries], device=device)
        suffix = np.array([self.suffix_map[s['suffix']] for s, _ in rows])
        self.groups = {int(s): torch.from_numpy(np.flatnonzero(suffix == s)).to(device) for s in np.unique(suffix)}
        roots, phases, cached_motion = [], [], []
        for begin in range(0, len(rows), 512):
            ids = self.starts[begin:begin + 512, None] + np.arange(window + 1)
            motion = torch.from_numpy(np.array(self.source[ids, 84:], copy=True))
            roots.append(motion[:, 1:, :3])
            phases.append(phase_descriptor(motion))
            if self.phase_aware: cached_motion.append(motion.to(device))
        self.roots = torch.cat(roots).to(device)
        self.phases = torch.cat(phases).to(device)
        self.num_styles = int(max(s['label'] for s in self.segments)) + 1
        self._group_versions = None
        self._group_data = {}
        self.match_picks=None; self.match_counts=None; self.match_cache_bytes=0
        if self.phase_aware:
            self.motions = torch.cat(cached_motion)
            # Every possible aligned source anchor, plus the clamped last anchor.
            # Profiles are immutable, train-only and FP32; no motion reads during updates.
            anchors = []
            for segment in self.segments:
                if segment['split'] != 'train' or segment['num_frames'] < window + 1: continue
                last = segment['num_frames'] - window - 1
                anchors.extend(segment['frame_start'] + np.unique(np.r_[np.arange(0, last+1, self.anchor_stride), last]))
            self.query_starts = np.unique(anchors).astype(np.int64)
            roots, phases = [], []
            for begin in range(0, len(self.query_starts), 512):
                ids = self.query_starts[begin:begin+512, None] + np.arange(window+1)
                motion = torch.from_numpy(np.array(self.source[ids, 84:], copy=True)).to(device)
                # Clone this small slice: retaining the view keeps all 291 motion
                # channels alive for every query while constructing the bank.
                roots.append(motion[:, 1:, :3].clone()); phases.append(phase_descriptor(motion))
            self.query_roots = torch.cat(roots); self.query_phases = torch.cat(phases)
            # A source anchor's correspondence is invariant to model updates. Cache
            # every observation offset at once; overlapping training windows reuse it.
            counts_dtype=torch.int16 if np.bincount([s['label'] for s,_ in rows]).max()<=32767 else torch.int32
            count_bytes=2 if counts_dtype==torch.int16 else 4
            size=len(self.query_starts)*window*self.num_styles*(4+count_bytes)
            capacity=int(match_cache_gb*2**30)
            if self.device.type=='cuda':
                reusable=torch.cuda.memory_reserved(self.device)-torch.cuda.memory_allocated(self.device)
                capacity=min(capacity,max(0,torch.cuda.mem_get_info(self.device)[0]+reusable-int(2*2**30)))
            if size<=capacity:
                try:
                    self.match_picks=torch.empty((len(self.query_starts),window,self.num_styles),device=device,dtype=torch.int32)
                    self.match_counts=torch.empty_like(self.match_picks,dtype=counts_dtype)
                    self.match_ready=np.zeros(len(self.query_starts),bool)
                    self.match_cache_bytes=size
                except torch.OutOfMemoryError:
                    self.match_picks=None;self.match_counts=None

    def _matching_groups(self):
        # Frozen bank tensors: do not re-gather/reduce them for every optimizer step.
        # Rebuild after in-place edits (also useful for diagnostic fixtures).
        versions = tuple(t._version for t in (self.roots, self.phases, self.labels, self.entry_codes))
        if self.phase_aware:versions += (self.query_roots._version,self.query_phases._version)
        if versions != self._group_versions:
            if self.match_picks is not None:self.match_ready.fill(False)
            self._group_data = {action: (bank, self.roots[bank], self.phases[bank],
                self.phases[bank].square().mean(-1), self.labels[bank], self.entry_codes[bank])
                for action, bank in self.groups.items()}
            self._group_versions = versions
        return self._group_data

    @torch.no_grad()
    def targets(self, *, x_window, labels, latents, starts, segment_ids, norm_mean, norm_std, frame_latent_fn, probability,
                return_entries=True):
        if self.phase_aware:
            return self._phase_targets(x_window=x_window, labels=labels, latents=latents, starts=starts,
                segment_ids=segment_ids, norm_mean=norm_mean, norm_std=norm_std,
                frame_latent_fn=frame_latent_fn, probability=probability, return_entries=return_entries)
        if x_window.size(1) != 2 or not 0 <= probability <= 1:
            raise ValueError('Window donor bank supports single-step TF and probability in [0,1]')
        n = len(labels)
        starts = np.asarray(starts.cpu(), dtype=np.int64)
        segments = [self.segments[i] for i in segment_ids.cpu().tolist()]
        entries = [s['source_entry_id'] for s in segments]
        local = starts - np.array([s['frame_start'] for s in segments])
        lengths = np.array([s['num_frames'] for s in segments])
        eligible = lengths >= self.window + 1
        anchor_local = np.minimum((local // self.window) * self.window, np.maximum(0, lengths - self.window - 1))
        offsets = local - anchor_local
        anchors = starts - offsets
        # Short source clips safely fall back, without reading across segment boundaries.
        requested = torch.rand(n, device=self.device) < probability
        requested_cpu = requested.cpu().numpy()
        query_ids = np.flatnonzero(eligible & requested_cpu)
        ids = anchors[query_ids, None] + np.arange(self.window + 1)[None]
        motion = torch.from_numpy(np.array(self.source[ids, 84:], copy=True)).to(self.device)
        query_roots = motion[:, 1:, :3]
        query_phase = phase_descriptor(motion) if len(query_ids) else motion.new_empty((0, (self.window + 1) * 10))
        query_rows = np.full(n, -1, dtype=np.int64)
        query_rows[query_ids] = np.arange(len(query_ids))
        suffix = np.array([self.suffix_map[s['suffix']] for s in segments])
        entry_codes = torch.tensor([self.id_map[e] for e in entries], device=self.device)
        chosen = torch.full((n,), -1, device=self.device, dtype=torch.long)
        candidates = torch.zeros(n, device=self.device)
        match_cost = torch.zeros(n, device=self.device)
        group_data = self._matching_groups()
        for action in np.unique(suffix[eligible]):
            if int(action) not in self.groups:
                continue
            all_src = np.flatnonzero((suffix == action) & eligible)
            # Preserve exactly the original number/order of RNG draws, including clean
            # samples. Their unused matching work can then be skipped safely.
            uniform_rank = torch.rand(len(all_src), device=self.device)
            keep = requested_cpu[all_src]
            src_cpu = all_src[keep]
            if not len(src_cpu):
                continue
            src = torch.from_numpy(src_cpu).to(self.device)
            query = torch.from_numpy(query_rows[src_cpu]).to(self.device)
            bank, bank_roots, p, p_norm, bank_labels, bank_entries = group_data[int(action)]
            root = query_roots[query]
            dv = (root[:, None, :, [0, 2]] - bank_roots[None, :, :, [0, 2]]).norm(dim=-1)
            dy = (root[:, None, :, 1] - bank_roots[None, :, :, 1]).abs()
            limits = torch.maximum(root[..., [0, 2]].norm(dim=-1) * .2, root.new_tensor(.25))
            valid = ((dv <= limits[:, None]).all(-1) & (dy <= 30).all(-1)
                     & (labels[src, None] != bank_labels[None])
                     & (entry_codes[src, None] != bank_entries[None]))
            q = query_phase[query]
            phase_cost = (q.square().mean(-1)[:, None] + p_norm[None] - 2 * q @ p.t() / p.size(1)).clamp_min(0)
            cost = phase_cost + (dv / limits[:, None]).square().mean(-1) + (dy / 30).square().mean(-1)
            style_cost = cost.new_full((len(src), self.num_styles), float('inf'))
            style_cost.scatter_reduce_(1, bank_labels[None].expand(len(src), -1),
                cost.masked_fill(~valid, float('inf')), reduce='amin')
            available = torch.isfinite(style_cost)
            rank = (uniform_rank[torch.from_numpy(np.flatnonzero(keep)).to(self.device)] * available.sum(1)).long()
            style = ((available.cumsum(1) - 1 == rank[:, None]) & available).long().argmax(1)
            valid_style = valid & (bank_labels[None] == style[:, None])
            nearest = cost.masked_fill(~valid_style, float('inf')).argmin(1)
            use = requested[src] & available.any(1)
            chosen[src] = torch.where(use, bank[nearest], -1)
            candidates[src] = torch.where(use, valid_style.sum(1), 0).to(candidates.dtype)
            match_cost[src] = torch.where(use, phase_cost.gather(1, nearest[:, None]).squeeze(1), 0)
        changed = chosen >= 0
        # One device-to-host transfer supplies both row IDs and donor IDs needed by
        # the memmap. Avoid nonzero() followed by two separate synchronous copies.
        chosen_cpu = chosen.cpu().numpy()
        selected_cpu = np.flatnonzero(chosen_cpu >= 0)
        selected = torch.from_numpy(selected_cpu).to(self.device)
        target_labels, targets = labels.clone(), latents[:, 1:].clone()
        recon = x_window[:, 1:, 84:].clone()
        mean = torch.as_tensor(norm_mean, device=self.device, dtype=torch.float32)
        std = torch.as_tensor(norm_std, device=self.device, dtype=torch.float32)
        if len(selected):
            bank_cpu = chosen_cpu[selected_cpu]
            donor_ids = self.starts[bank_cpu] + offsets[selected_cpu] + 1
            donor = torch.from_numpy(np.array(self.source[donor_ids, 84:], copy=True)).to(self.device)
            source = x_window[selected, 1] * std + mean
            transplanted = retarget_motion(source[:, 84:87], donor)
            normalized = (torch.cat((source[:, :84], transplanted), -1) - mean) / std
            targets[selected, 0] = frame_latent_fn(normalized)
            recon[selected, 0] = normalized[:, 84:]
            target_labels[selected] = self.labels[chosen[selected]]
            for i, b in zip(selected_cpu, bank_cpu):
                entries[int(i)] = self.entries[int(b)]
        count = max(len(selected_cpu), 1)
        return target_labels, targets, recon, changed, entries, {
            '_changed_count': len(selected_cpu),
            'donor_candidates_mean': candidates.sum() / count,
            'donor_phase_cost': match_cost.sum() / count,
        }

    @torch.no_grad()
    def _phase_targets(self, *, x_window, labels, latents, starts, segment_ids, norm_mean, norm_std,
                       frame_latent_fn, probability, return_entries):
        n, frames, _ = x_window.shape
        steps = frames-1
        if not 1 <= steps <= self.max_steps or not 0 <= probability <= 1:
            raise ValueError('Phase donor window must cover every target step')
        starts = np.asarray(starts.cpu(), dtype=np.int64)
        segments = [self.segments[i] for i in segment_ids.cpu().tolist()]
        if any(s['split']!='train' for s in segments):raise ValueError('Donor queries must be train-only')
        entries = [s['source_entry_id'] for s in segments]
        lengths = np.array([s['num_frames'] for s in segments])
        local = starts-np.array([s['frame_start'] for s in segments])
        if np.any(local<0) or np.any(local+frames>lengths):
            raise ValueError('Source window crosses a clip boundary')
        eligible = lengths >= self.window+1
        offset = local-np.minimum((local//self.anchor_stride)*self.anchor_stride, np.maximum(0,lengths-self.window-1))
        # CPU RNG is checkpointed along with CUDA RNG. Only flags/indices cross to
        # the GPU; do not synchronize requested CUDA masks or reread source motion.
        requested = (torch.rand(n)<probability).numpy() & eligible
        suffix = np.array([self.suffix_map[s['suffix']] for s in segments])
        chosen = torch.full((n,),-1,device=self.device,dtype=torch.long)
        candidates = torch.zeros(n,device=self.device); costs = torch.zeros_like(candidates)
        opposite_count = candidates.new_zeros(())
        group_data=self._matching_groups()
        for action in np.unique(suffix[requested]):
            src_cpu = np.flatnonzero(requested & (suffix==action))
            src = torch.from_numpy(src_cpu).to(self.device)
            anchors = starts[src_cpu]-offset[src_cpu]
            ids = np.searchsorted(self.query_starts,anchors)
            if np.any(ids>=len(self.query_starts)) or not np.array_equal(self.query_starts[ids],anchors):
                raise ValueError('Source anchor missing from train-only phase cache')
            ids_cpu=ids
            ids = torch.from_numpy(ids).to(self.device)
            offsets = torch.from_numpy(offset[src_cpu]).to(self.device)
            bank, root_bank, phase_bank, phase_norm, bank_labels, bank_entries = group_data[int(action)]
            missing_cpu=np.arange(len(src_cpu)) if self.match_picks is None else np.flatnonzero(~self.match_ready[ids_cpu])
            if len(missing_cpu):
                missing=torch.from_numpy(missing_cpu).to(self.device)
                work_ids=ids[missing]
                root = self.query_roots[work_ids]; phase = self.query_phases[work_ids]
                dv = (root[:,None,:,[0,2]]-root_bank[None,:,:,[0,2]]).norm(dim=-1)
                dy = (root[:,None,:,1]-root_bank[None,:,:,1]).abs()
                limits = torch.maximum(root[...,[0,2]].norm(dim=-1)*.2,root.new_tensor(.25))
                entry = torch.tensor([self.id_map[entries[i]] for i in src_cpu[missing_cpu]],device=self.device,dtype=torch.long)
                valid = ((dv<=limits[:,None]).all(-1) & (dy<=30).all(-1)
                         & (labels[src[missing],None]!=bank_labels[None]) & (entry[:,None]!=bank_entries[None]))
                # A warm cache changes the number of missing queries. Keep GEMM
                # shape fixed so near-tied donors do not change with cache warmth
                # (or when a resumed run reconstructs its cache).
                products=[]
                for begin in range(0,len(phase),32):
                    rows=phase[begin:begin+32]
                    padded=torch.nn.functional.pad(rows,(0,0,0,32-len(rows)))
                    products.append((padded@phase_bank.t())[:len(rows)])
                dot=torch.cat(products)
                phase_cost = (phase.square().mean(-1)[:,None]+phase_norm[None]
                              -2*dot/phase.size(1)).clamp_min(0)
                cost = phase_cost+(dv/limits[:,None]).square().mean(-1)+(dy/30).square().mean(-1)
                picks,counts=self._candidate_tables(cost,valid,phase,phase_bank,bank,bank_labels,
                    offsets[missing] if self.match_picks is None else None)
                if self.match_picks is not None:
                    self.match_picks[work_ids]=picks.to(torch.int32)
                    self.match_counts[work_ids]=counts.to(self.match_counts.dtype)
                    self.match_ready[ids_cpu[missing_cpu]]=True
            if self.match_picks is not None:
                picks=self.match_picks[ids,offsets].long();counts=self.match_counts[ids,offsets]
            else:picks=picks[:,0].long();counts=counts[:,0]
            available = picks>=0
            rank = (torch.rand(len(src),device=self.device)*available.sum(1)).long()
            style = ((available.cumsum(1)-1==rank[:,None]) & available).long().argmax(1)
            use = available.any(1)
            pick=picks.gather(1,style[:,None]).squeeze(1)
            chosen[src] = torch.where(use,pick,-1)
            candidates[src] = torch.where(use,counts.gather(1,style[:,None]).squeeze(1),0).to(candidates.dtype)
            q=self.query_phases[ids];p=self.phases[pick.clamp_min(0)]
            rows=torch.arange(len(src),device=self.device)
            qv=q.reshape(-1,self.window+1,10)[rows,offsets,3:6]
            pv=p.reshape(-1,self.window+1,10)[rows,offsets,3:6]
            opposite_count+=((qv.norm(dim=-1)>.95)&(pv.norm(dim=-1)>.95)&((qv*pv).sum(-1)<0)&use).sum()
            selected_phase=(q.square().mean(-1)+p.square().mean(-1)-2*(q*p).mean(-1)).clamp_min(0)
            costs[src] = torch.where(use,selected_phase,0)
        changed = chosen>=0; count = changed.sum()
        work_cpu = np.flatnonzero(requested); work = torch.from_numpy(work_cpu).to(self.device)
        targets = latents[:,1:].clone(); recon = x_window[:,1:,84:].clone(); out_labels = labels.clone()
        if len(work_cpu):
            safe = chosen[work].clamp_min(0)
            offsets = torch.from_numpy(offset[work_cpu]).to(self.device)
            rows = offsets[:,None]+torch.arange(1,steps+1,device=self.device)
            donor = self.motions[safe[:,None],rows]
            mean = torch.as_tensor(norm_mean,device=self.device,dtype=torch.float32)
            std = torch.as_tensor(norm_std,device=self.device,dtype=torch.float32)
            source_root = x_window[work,1:,84:87]*std[84:87]+mean[84:87]
            transplanted = (retarget_motion(source_root,donor)-mean[84:])/std[84:]
            synthetic = torch.cat((x_window[work,1:,:84],transplanted),-1)
            encoded = frame_latent_fn(synthetic.flatten(0,1)).reshape(len(work),steps,-1)
            mask = changed[work,None,None]
            targets[work] = torch.where(mask,encoded,targets[work])
            recon[work] = torch.where(mask,transplanted,recon[work])
            out_labels[work] = torch.where(changed[work],self.labels[safe],labels[work])
        if return_entries:
            choices = chosen.cpu().numpy()
            for i in np.flatnonzero(choices>=0): entries[i] = self.entries[choices[i]]
        denom = count.clamp_min(1)
        std = torch.as_tensor(norm_std,device=self.device,dtype=torch.float32)
        first_delta=(recon[:,0,3:75]-x_window[:,0,87:159])*std[87:159]
        jump=first_delta.reshape(n,24,3).norm(dim=-1).mean(-1)*100
        style_counts=torch.zeros(self.num_styles,device=self.device)
        style_counts.scatter_add_(0,out_labels,changed.float())
        return out_labels, targets, recon, changed, entries, {
            '_changed_count':count,'donor_candidates_mean':candidates.sum()/denom,'donor_phase_cost':costs.sum()/denom,
            'donor_opposite_fraction':opposite_count/denom,'donor_first_pose_step_cm':(jump*changed).sum()/denom,
            '_style_counts':style_counts}

    def _candidate_tables(self,cost,valid,phase,phase_bank,bank,bank_labels,offsets=None):
        qv=phase.reshape(len(phase),self.window+1,10)[:,:self.window,3:6]
        pv=phase_bank.reshape(len(bank),self.window+1,10)[:,:self.window,3:6]
        if offsets is not None:
            qv=qv[torch.arange(len(qv),device=self.device),offsets][:,None]
            pv=pv[:,offsets].permute(1,0,2)[:,None]
        else:pv=pv.permute(1,0,2)[None]
        confident=(qv.norm(dim=-1)[...,None]>.95) & (pv.norm(dim=-1)>.95)
        allowed=valid[:,None] & (~confident | ((qv[:,:,None]*pv).sum(-1)>=0))
        index=bank_labels[None,None].expand_as(allowed)
        minimum=cost.new_full((*allowed.shape[:2],self.num_styles),float('inf'))
        minimum.scatter_reduce_(2,index,cost[:,None].expand_as(allowed).masked_fill(~allowed,float('inf')),reduce='amin')
        winners=allowed & (cost[:,None]==minimum.gather(2,index))
        sentinel=len(self.starts)
        picks=torch.full_like(minimum,sentinel,dtype=torch.int32)
        picks.scatter_reduce_(2,index,bank[None,None].expand_as(allowed).to(torch.int32).masked_fill(~winners,sentinel),reduce='amin')
        picks.masked_fill_(picks==sentinel,-1)
        counts=torch.zeros_like(picks);counts.scatter_add_(2,index,allowed.to(torch.int32))
        return picks,counts
