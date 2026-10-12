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
    def __init__(self, db, *, window=16, windows_per_clip=16, device='cpu'):
        if window < 2 or windows_per_clip < 1:
            raise ValueError('Donor bank requires window>=2 and windows_per_clip>=1')
        db = Path(db)
        meta = json.loads((db / 'meta.json').read_text())
        self.segments = json.loads((db / 'segments.json').read_text())
        self.source = np.memmap(db / 'X.f32', mode='r', dtype=np.float32, shape=(meta['num_frames'], 375))
        self.window, self.device = window, torch.device(device)
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
        roots, phases, history_roots, history_phases = [], [], [], []
        for begin in range(0, len(rows), 512):
            ids = self.starts[begin:begin + 512, None] + np.arange(window + 1)
            motion = torch.from_numpy(np.array(self.source[ids, 84:], copy=True))
            roots.append(motion[:, 1:, :3])
            phases.append(phase_descriptor(motion))
            history_roots.append(motion[:, 0, :3])
            history_phases.append(phase_descriptor(motion[:, :1]))
        self.roots = torch.cat(roots).to(device)
        self.phases = torch.cat(phases).to(device)
        self.history_roots = torch.cat(history_roots).to(device)
        self.history_phases = torch.cat(history_phases).to(device)
        self.num_styles = int(max(s['label'] for s in self.segments)) + 1
        self._group_versions = None
        self._group_data = {}

    def _matching_groups(self):
        # Frozen bank tensors: do not re-gather/reduce them for every optimizer step.
        # Rebuild after in-place edits (also useful for diagnostic fixtures).
        versions = tuple(t._version for t in (self.roots, self.phases, self.labels, self.entry_codes))
        if versions != self._group_versions:
            self._group_data = {action: (bank, self.roots[bank], self.phases[bank],
                self.phases[bank].square().mean(-1), self.labels[bank], self.entry_codes[bank])
                for action, bank in self.groups.items()}
            self._group_versions = versions
        return self._group_data

    @torch.no_grad()
    def perturb_history(self, *, current, previous_latent, labels, segment_ids, selected,
                        norm_mean, norm_std, frame_latent_fn):
        """Corrupt observations only. The caller's real next-frame target stays intact.

        Another train style, same action, approximately aligned single-frame phase.
        Root differences are a soft retrieval cost, not a target-validity gate: this
        is explicitly denoising augmentation, not a physically paired transition.
        """
        if current.ndim != 2 or current.size(1) != 375 or selected.shape != labels.shape:
            raise ValueError('History augmentation requires [B,375] and a [B] mask')
        mean = torch.as_tensor(norm_mean, device=self.device, dtype=torch.float32)
        std = torch.as_tensor(norm_std, device=self.device, dtype=torch.float32)
        raw = current * std + mean
        segments = [self.segments[i] for i in segment_ids.cpu().tolist()]
        entries = [s['source_entry_id'] for s in segments]
        suffix = np.array([self.suffix_map[s['suffix']] for s in segments])
        requested = selected.bool().cpu().numpy()
        chosen = torch.full_like(labels, -1)
        for action in np.unique(suffix[requested]):
            if int(action) not in self.groups:
                continue
            src = torch.from_numpy(np.flatnonzero(requested & (suffix == action))).to(self.device)
            bank = self.groups[int(action)]
            q = phase_descriptor(raw[src, None, 84:])
            p = self.history_phases[bank]
            phase = (q.square().mean(-1)[:, None] + p.square().mean(-1)[None]
                     - 2 * q @ p.t() / p.size(1)).clamp_min(0)
            root = raw[src, 84:87]
            delta = root[:, None] - self.history_roots[bank][None]
            cost = phase + .1 * ((delta[..., [0, 2]] / .25).square().sum(-1)
                                + (delta[..., 1] / 30).square())
            valid = labels[src, None] != self.labels[bank][None]
            style_cost = cost.new_full((len(src), self.num_styles), float('inf'))
            style_cost.scatter_reduce_(1, self.labels[bank][None].expand(len(src), -1),
                cost.masked_fill(~valid, float('inf')), reduce='amin')
            available = torch.isfinite(style_cost)
            rank = (torch.rand(len(src), device=self.device) * available.sum(1)).long()
            style = ((available.cumsum(1) - 1 == rank[:, None]) & available).long().argmax(1)
            nearest = cost.masked_fill(~(valid & (self.labels[bank][None] == style[:, None])),
                                      float('inf')).argmin(1)
            chosen[src] = torch.where(available.any(1), bank[nearest], -1)
        chosen_cpu = chosen.cpu().numpy()
        ids = np.flatnonzero(chosen_cpu >= 0)
        changed = chosen >= 0
        result, latent, history_labels = current.clone(), previous_latent.clone(), labels.clone()
        if len(ids):
            src = torch.from_numpy(ids).to(self.device)
            donor_rows = chosen_cpu[ids]
            donor = torch.from_numpy(np.array(self.source[self.starts[donor_rows], 84:], copy=True)).to(self.device)
            motion = retarget_motion(raw[src, 84:87], donor)
            result[src, 84:] = (motion - mean[84:]) / std[84:]
            result[src, 84:87] = current[src, 84:87]
            latent[src] = frame_latent_fn(result[src])
            history_labels[src] = self.labels[chosen[src]]
            for i, b in zip(ids, donor_rows):
                entries[int(i)] = self.entries[int(b)]
        return result, latent, history_labels, changed, entries

    @torch.no_grad()
    def targets(self, *, x_window, labels, latents, starts, segment_ids, norm_mean, norm_std, frame_latent_fn, probability):
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
